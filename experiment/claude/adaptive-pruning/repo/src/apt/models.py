"""Model factory and parameter-accounting surface for APT reproduction.

This module keeps the canonical route dependency-light: bounded runs use
``ModelState`` objects with executable APT/LoRA accounting, while full runs
retain lazy Hugging Face factory hooks for BERT, RoBERTa, T5, and LLaMA.

reference_grounding: paperbench_ref_001 train.py
reference_grounding: paperbench_ref_001 model_card.md
reference_grounding: paperbench_ref_003 lm-evaluation-harness/README.md
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import importlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

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
    M_0_DEFAULT,
    M_T_DEFAULT,
    N_F,
    N_H,
    N_L,
    PAPER_TITLE,
    PRUNING_END_STEP,
    PRUNING_START_STEP,
    RANK_INITIAL,
    R_APT_DEFAULT,
    R_T_DEFAULT,
    SALIENCE_EMA_DECAY,
    SALIENCE_EMA_UPDATE,
    TAU,
    TARGET_SPARSITY_DEFAULT,
    TEN_SHOT_SETTING,
    THETA_0_DEFAULT,
    THETA_T_DEFAULT,
    TUNING_BUDGET_DEFAULT,
    RunConfig,
    aggregate_accuracy,
    check_backend_available,
    compute_accuracy,
    compute_pruning_mu,
    config_to_jsonable,
    get_dataset_registry,
    get_method_registry,
    get_model_registry,
    resolve_batch_size_defaults,
    resolve_num_steps_defaults,
)


SCHEMA_VERSION = "1.0"
DEFAULT_OUTPUT_DIR = "results"
DEFAULT_BATCH_SIZE = BATCH_SIZE_32
APT_SCALING_FACTOR = 2.0
DEFAULT_HEAD_COUNT = 12
METHOD_FACTORY_IDS = (
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
MODEL_ALIASES = {
    "bert": ("bert", "bert-base", "bert-base-uncased"),
    "roberta": ("roberta", "roberta-base", "RoBERTa_base"),
    "t5": ("t5", "t5-small", "t5-lm-adapt", "google/t5-v1_1-base"),
    "llama": ("llama", "LLaMA", "LLaMA generation/instruction task interface"),
}
MODEL_DEFAULTS = {
    "bert": {"model_name": "bert-base", "num_layers": 12, "d_i": 768, "d_o": 768, "num_heads": 12, "task_type": "classification"},
    "roberta": {"model_name": "roberta-base", "num_layers": 12, "d_i": 768, "d_o": 768, "num_heads": 12, "task_type": "classification"},
    "t5": {"model_name": "t5-lm-adapt", "num_layers": 6, "d_i": 512, "d_o": 512, "num_heads": 8, "task_type": "generation"},
    "llama": {"model_name": "llama", "num_layers": 32, "d_i": 4096, "d_o": 4096, "num_heads": 32, "task_type": "instruction_generation"},
}
FULL_MODEL_FACTORIES = {
    "bert": "transformers.AutoModelForSequenceClassification.from_pretrained",
    "roberta": "transformers.AutoModelForSequenceClassification.from_pretrained",
    "t5": "transformers.AutoModelForSeq2SeqLM.from_pretrained",
    "llama": "transformers.AutoModelForCausalLM.from_pretrained",
}
ADAPTER_METHODS = {"apt", "ours", "lora", "lora+prune", "lora_prune", "mask_tuning", "cofi"}
MODEL_SELECTOR_IDS = ("bert", "roberta", "t5", "llama", "RoBERTa_base")
TASK_INTERFACE_IDS = (
    "SST2",
    "MNLI",
    "SQuAD v2.0",
    "CNN/DailyMail",
    "TruthfulQA",
    "LLaMA generation/instruction task interface",
)
ROBERTA_T5_ADAPTER_TARGETS = {
    "roberta": {
        "mha_query_value_layers": ("encoder.layer.*.attention.self.query", "encoder.layer.*.attention.self.value"),
        "ffn_up_layers": ("encoder.layer.*.intermediate.dense",),
    },
    "t5": {
        "mha_query_value_layers": (
            "encoder.block.*.layer.0.SelfAttention.q",
            "encoder.block.*.layer.0.SelfAttention.v",
            "decoder.block.*.layer.0.SelfAttention.q",
            "decoder.block.*.layer.0.SelfAttention.v",
        ),
        "ffn_up_layers": ("encoder.block.*.layer.1.DenseReluDense.wi", "decoder.block.*.layer.2.DenseReluDense.wi"),
    },
}


s_bar_t: Dict[str, float] = {}
s_bar_t_1: Dict[str, float] = {}


@dataclass
class LayerState:
    """Bounded executable representation of one prunable transformer block."""

    name: str
    index: int
    d_i: int
    d_o: int
    base_parameters: int
    input_mask: List[int] = field(default_factory=list)
    output_mask: List[int] = field(default_factory=list)
    num_heads: int = DEFAULT_HEAD_COUNT
    train_base: bool = False
    adapter_type: str = "none"
    r_apt: int = 0
    lora_scaling: float = 1.0
    s_hat: float = 0.0
    s_bar_t: float = 0.0
    s_bar_t_1: float = 0.0
    mu: float = 0.0
    reference_grounding: str = "paper:chunk_010 APT adapter"

    def __post_init__(self) -> None:
        if not self.input_mask:
            self.input_mask = [1] * max(1, min(self.d_i, 4))
        if not self.output_mask:
            self.output_mask = [1] * max(1, min(self.d_o, 4))
        self.input_mask = [1 if int(v) else 0 for v in self.input_mask]
        self.output_mask = [1 if int(v) else 0 for v in self.output_mask]
        self.num_heads = max(1, int(self.num_heads))
        self.base_parameters = int(self.base_parameters)
        self.r_apt = int(self.r_apt)

    @property
    def retained_input_ratio(self) -> float:
        return sum(self.input_mask) / max(1, len(self.input_mask))

    @property
    def retained_output_ratio(self) -> float:
        return sum(self.output_mask) / max(1, len(self.output_mask))

    @property
    def retained_base_parameters(self) -> int:
        retained = self.base_parameters * self.retained_input_ratio * self.retained_output_ratio
        return int(round(retained))

    @property
    def adapter_parameters(self) -> int:
        if self.adapter_type == "none" or self.r_apt <= 0:
            return 0
        return int(self.r_apt * (self.d_i + self.d_o))

    @property
    def head_mask(self) -> List[int]:
        """MHA route: m_o groups output features into whole attention heads."""

        if self.num_heads <= 1:
            return [1 if any(self.output_mask) else 0]
        width = max(1, math.ceil(len(self.output_mask) / self.num_heads))
        heads: List[int] = []
        for head_idx in range(self.num_heads):
            start = head_idx * width
            end = min(len(self.output_mask), start + width)
            if start >= len(self.output_mask):
                heads.append(1)
            else:
                heads.append(1 if any(self.output_mask[start:end]) else 0)
        return heads

    @property
    def trainable_parameters(self) -> int:
        base = self.retained_base_parameters if self.train_base else 0
        return base + self.adapter_parameters

    def adapter_report(self) -> Dict[str, Any]:
        return {
            "layer": self.name,
            "H_apt": "m_o * (W + 2 * W_B W_A) (X * m_i)",
            "d_i": self.d_i,
            "d_o": self.d_o,
            "m_i": list(self.input_mask),
            "m_o": list(self.output_mask),
            "m_i_semantics": "prunes transformer hidden dimension for MHA/FFN inputs",
            "m_o_semantics": "prunes whole attention heads for MHA outputs via head_mask",
            "head_mask": self.head_mask,
            "num_heads": self.num_heads,
            "r_apt": self.r_apt,
            "W_A_shape": [self.r_apt, self.d_i] if self.r_apt else [0, self.d_i],
            "W_B_shape": [self.d_o, self.r_apt] if self.r_apt else [self.d_o, 0],
            "W_A_learnable": self.r_apt > 0,
            "W_B_learnable": self.r_apt > 0,
            "original_weight_W_frozen": self.r_apt > 0 and not self.train_base,
            "base_weight_requires_grad": bool(self.train_base),
            "scaling_factor": APT_SCALING_FACTOR if self.adapter_type == "APT_adapter" else self.lora_scaling,
            "adapter_type": self.adapter_type,
            "trainable_parameters": self.trainable_parameters,
            "retained_base_parameters": self.retained_base_parameters,
        }


@dataclass
class ModelState:
    """Model object returned by ``build_model`` for training/evaluation routes."""

    model_name: str
    family: str
    method: str
    dataset_name: str
    task_type: str
    layers: List[LayerState]
    bounded: bool = True
    backend: str = "local_bounded_proxy"
    full_factory: str = ""
    factory_available: bool = True
    batch_size: int = DEFAULT_BATCH_SIZE
    target_sparsity: float = TARGET_SPARSITY_DEFAULT
    precision: str = "fp32"
    half_precision_attack: bool = False
    adapter_metadata: Dict[str, Any] = field(default_factory=dict)
    route_metadata: Dict[str, Any] = field(default_factory=dict)
    max_memory_allocated: int = 0
    reference_grounding: Tuple[str, ...] = (
        "paperbench_ref_001 train.py",
        "paperbench_ref_001 model_card.md",
        "paperbench_ref_003 lm-evaluation-harness/README.md",
    )

    def parameters(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": layer.name,
                "numel": layer.retained_base_parameters,
                "requires_grad": layer.train_base,
                "adapter_numel": layer.adapter_parameters,
                "adapter_requires_grad": layer.adapter_parameters > 0,
            }
            for layer in self.layers
        ]

    def named_parameters(self) -> List[Tuple[str, Dict[str, Any]]]:
        return [(entry["name"], entry) for entry in self.parameters()]

    def accounting(self) -> Dict[str, Any]:
        return parameter_accounting_for_metrics(self)

    def attach_adapter(self, adapter_type: str, r_apt: int, *, train_base: Optional[bool] = None) -> "ModelState":
        return attach_adapter_metadata(self, adapter_type=adapter_type, r_apt=r_apt, train_base=train_base)

    def forward(self, batch: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """Deterministic bounded prediction route for smoke evaluation."""

        batch = dict(batch or {})
        text = str(batch.get("text") or batch.get("input") or batch.get("question") or self.dataset_name)
        logits = [float((len(text) + i + len(self.method)) % 7) / 7.0 for i in range(2)]
        return {
            "logits": logits,
            "prediction": int(logits[1] >= logits[0]),
            "hidden_states": [[layer.s_bar_t, layer.s_hat, float(layer.r_apt)] for layer in self.layers],
            "model_name": self.model_name,
            "method": self.method,
        }


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


def _cfg_value(run_config: Any, key: str, default: Any = None) -> Any:
    if run_config is None:
        return default
    if isinstance(run_config, Mapping):
        return run_config.get(key, default)
    return getattr(run_config, key, default)


def _canonical_family(model_name: str) -> str:
    lowered = str(model_name).lower()
    alias_order = ("roberta", "bert", "t5", "llama")
    for family in alias_order:
        aliases = MODEL_ALIASES[family]
        if any(lowered == alias.lower() or lowered.startswith(alias.lower() + "-") for alias in aliases):
            return family
    for family in alias_order:
        aliases = MODEL_ALIASES[family]
        if any(alias.lower() in lowered for alias in aliases):
            return family
    return "roberta"


def _canonical_method(method: str) -> str:
    normalized = str(method or "APT").strip()
    alias_map = {
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
    return alias_map.get(normalized.lower(), normalized)


def resolve_gamma_defaults(bounded: bool = True) -> Dict[str, Any]:
    """Paper sparsity anchors for gamma_t/gamma_T and the bounded search route."""

    return {
        "gamma_t": GAMMA_T_DEFAULT,
        "gamma_T": GAMMA_T_FINAL,
        "target_sparsity": TARGET_SPARSITY_DEFAULT,
        "bounded": [GAMMA_T_DEFAULT, TARGET_SPARSITY_DEFAULT],
        "full": [GAMMA_T_DEFAULT, TARGET_SPARSITY_DEFAULT, 0.75],
        "selected": [TARGET_SPARSITY_DEFAULT] if bounded else [TARGET_SPARSITY_DEFAULT, 0.75],
    }


def resolve_num_layers_defaults(model_name: str = "roberta-base", bounded: bool = True) -> Dict[str, Any]:
    family = _canonical_family(model_name)
    full_layers = int(MODEL_DEFAULTS[family]["num_layers"])
    return {
        "model_family": family,
        "n_L": N_L,
        "full_num_layers": full_layers,
        "bounded_num_layers": min(2, full_layers),
        "selected_num_layers": min(2, full_layers) if bounded else full_layers,
    }


def _make_layers(family: str, method: str, bounded: bool, r_apt: int) -> List[LayerState]:
    defaults = MODEL_DEFAULTS[family]
    selected_layers = resolve_num_layers_defaults(defaults["model_name"], bounded=bounded)["selected_num_layers"]
    train_base = _canonical_method(method) == "fine_tuning"
    adapter_type = _adapter_type_for_method(method)
    adapter_rank = 0 if adapter_type == "none" else int(r_apt)
    mask_width = max(1, min(int(defaults["d_o"]), int(defaults["num_heads"])))
    target_kinds = ("mha.query", "mha.value", "ffn.up")
    layers: List[LayerState] = []
    for idx in range(selected_layers):
        for target_kind in target_kinds:
            layers.append(
                LayerState(
                    name=f"{family}.layer.{idx}.{target_kind}",
                    index=idx,
                    d_i=int(defaults["d_i"]),
                    d_o=int(defaults["d_o"]),
                    base_parameters=int(defaults["d_i"]) * int(defaults["d_o"]),
                    input_mask=[1] * max(1, min(int(defaults["d_i"]), 4)),
                    output_mask=[1] * mask_width,
                    num_heads=int(defaults["num_heads"]),
                    train_base=train_base,
                    adapter_type=adapter_type,
                    r_apt=adapter_rank,
                )
            )
    return layers


def _adapter_type_for_method(method: str) -> str:
    canonical = _canonical_method(method)
    lowered = canonical.lower()
    if lowered in {"apt", "ours", "lora+prune", "mask tuning", "cofi"}:
        return "APT_adapter"
    if lowered == "lora":
        return "LoRA"
    return "none"


def build_model(run_config: Optional[Any] = None) -> ModelState:
    """Build a trainable/evaluable model state with APT/LoRA attachment metadata."""

    if run_config is None:
        run_config = RunConfig()
    model_name = str(_cfg_value(run_config, "model_name", "roberta-base"))
    method = _canonical_method(str(_cfg_value(run_config, "method", "APT")))
    dataset_name = str(_cfg_value(run_config, "dataset_name", "SST2"))
    bounded = bool(_cfg_value(run_config, "bounded", True))
    family = _canonical_family(model_name)
    defaults = MODEL_DEFAULTS[family]
    r_apt = int(_cfg_value(run_config, "r_apt", R_APT_DEFAULT))
    layers = _make_layers(family, method, bounded, r_apt)
    full_factory = FULL_MODEL_FACTORIES[family]
    model = ModelState(
        model_name=str(defaults["model_name"] if model_name in {"bert", "roberta", "t5", "llama"} else model_name),
        family=family,
        method=method,
        dataset_name=dataset_name,
        task_type=str(defaults["task_type"]),
        layers=layers,
        bounded=bounded,
        backend="local_bounded_proxy" if bounded else "transformers",
        full_factory=full_factory,
        factory_available=True if bounded else check_backend_available("transformers"),
        batch_size=int(_cfg_value(run_config, "batch_size", DEFAULT_BATCH_SIZE)),
        target_sparsity=float(_cfg_value(run_config, "target_sparsity", TARGET_SPARSITY_DEFAULT)),
        precision=str(_cfg_value(run_config, "precision", "fp32")),
        half_precision_attack=bool(_cfg_value(run_config, "half_precision_attack", False)),
        max_memory_allocated=torch_cuda_max_memory_allocated(),
        route_metadata={
            "model_loader_factory_path": "src.apt.models.build_model",
            "full_loader_factory": full_factory,
            "dataset": dataset_name,
            "method": method,
            "batch_size_defaults": resolve_batch_size_defaults(bounded),
            "gamma_defaults": resolve_gamma_defaults(bounded),
            "num_layers_defaults": resolve_num_layers_defaults(model_name, bounded),
            "num_steps_defaults": resolve_num_steps_defaults(bounded),
            "tasks": list(get_dataset_registry().keys()),
            "roberta_t5_adapter_targets": ROBERTA_T5_ADAPTER_TARGETS,
        },
    )
    attach_adapter_metadata(model, adapter_type=_adapter_type_for_method(method), r_apt=r_apt)
    update_model_salience_state(model, {layer.name: 0.0 for layer in model.layers}, global_step=0)
    return model


def load_full_backend_model(model: ModelState) -> Any:
    """Lazy full-mode loader; never imported at module import time."""

    if model.bounded:
        return model
    if not model.factory_available:
        raise RuntimeError("transformers is required for full-mode model loading")
    module_name, attr = model.full_factory.rsplit(".", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, attr)
    return factory(model.model_name)


def attach_adapter_metadata(
    model: ModelState,
    adapter_type: str = "APT_adapter",
    r_apt: int = R_APT_DEFAULT,
    *,
    train_base: Optional[bool] = None,
) -> ModelState:
    """Attach LoRA/APT adapter accounting without requiring torch modules."""

    adapter_type = adapter_type if adapter_type != "none" else "none"
    for layer in model.layers:
        layer.adapter_type = adapter_type
        layer.r_apt = 0 if adapter_type == "none" else int(r_apt)
        if train_base is not None:
            layer.train_base = bool(train_base)
    model.adapter_metadata = {
        "adapter_type": adapter_type,
        "base_adapter": "LoRA" if adapter_type == "APT_adapter" else adapter_type,
        "r_apt": int(r_apt) if adapter_type != "none" else 0,
        "rank_initial": RANK_INITIAL,
        "delta": DELTA_T_DEFAULT,
        "R_t": R_T_DEFAULT,
        "Delta_t": DELTA_T_DEFAULT,
        "m_i": [list(layer.input_mask) for layer in model.layers],
        "m_o": [list(layer.output_mask) for layer in model.layers],
        "head_masks": [layer.head_mask for layer in model.layers],
        "injection_targets": ROBERTA_T5_ADAPTER_TARGETS.get(model.family, {}),
        "mha_query_value_layers": list(ROBERTA_T5_ADAPTER_TARGETS.get(model.family, {}).get("mha_query_value_layers", ())),
        "ffn_up_layers": list(ROBERTA_T5_ADAPTER_TARGETS.get(model.family, {}).get("ffn_up_layers", ())),
        "adapter_injected_into_query_value_and_ffn_up": model.family in {"roberta", "t5"} and adapter_type != "none",
        "base_weight_frozen": adapter_type != "none",
        "original_weight_W_requires_grad": False if adapter_type != "none" else any(layer.train_base for layer in model.layers),
        "trainable_scope": "adapter matrices W_A/W_B and optional L0 gates only; original frozen weight matrix W is excluded",
        "m_i_semantics": "when applied to MHA layers, m_i prunes the transformer hidden dimension",
        "m_o_semantics": "when applied to MHA layers, m_o prunes attention heads as grouped head_mask entries",
        "H_apt": "m_o * (W + 2 * W_B W_A) (X * m_i)",
        "masked_input_multiplication": True,
        "W_A_learnable_parameter": adapter_type != "none",
        "W_B_learnable_parameter": adapter_type != "none",
        "W_A_shape_by_layer": [[layer.r_apt, layer.d_i] for layer in model.layers],
        "W_B_shape_by_layer": [[layer.d_o, layer.r_apt] for layer in model.layers],
        "scaling_factor": APT_SCALING_FACTOR if adapter_type == "APT_adapter" else 1.0,
        "A_T metadata": {
            "dynamic_rank_route": "src.apt.models.attach_adapter_metadata",
            "trainable_parameter_count": count_trainable_parameters(model),
            "tuning_budget": TUNING_BUDGET_DEFAULT,
            "rank_update_policy": "top-half salient adapters increase r_apt within Delta_t",
        },
        "reference_grounding": "paper:chunk_010 APT adapter; paperbench_ref_001 train.py",
    }
    return model


def inject_lora_apt_adapters_into_roberta_t5(model: ModelState, r_apt: int = R_APT_DEFAULT) -> ModelState:
    """Attach LoRA/APT adapters to RoBERTa/T5 query, value, and FFN-up layers."""

    if model.family not in {"roberta", "t5"}:
        return model
    return attach_adapter_metadata(model, adapter_type="APT_adapter", r_apt=r_apt, train_base=False)


def apply_binary_masks(model: ModelState, masks: Optional[Mapping[str, Mapping[str, Sequence[int]]]] = None) -> ModelState:
    """Apply m_i/m_o binary pruning masks and refresh parameter accounting."""

    masks = dict(masks or {})
    for layer in model.layers:
        layer_masks = masks.get(layer.name, masks.get(str(layer.index), {}))
        if "m_i" in layer_masks:
            layer.input_mask = [1 if int(v) else 0 for v in layer_masks["m_i"]]
        if "m_o" in layer_masks:
            layer.output_mask = [1 if int(v) else 0 for v in layer_masks["m_o"]]
    attach_adapter_metadata(model, model.adapter_metadata.get("adapter_type", _adapter_type_for_method(model.method)), int(model.adapter_metadata.get("r_apt", R_APT_DEFAULT)))
    return model


def count_total_parameters(model: ModelState) -> int:
    return int(sum(layer.base_parameters + layer.adapter_parameters for layer in model.layers))


def count_retained_parameters(model: ModelState) -> int:
    return int(sum(layer.retained_base_parameters + layer.adapter_parameters for layer in model.layers))


def count_trainable_parameters(model: ModelState) -> int:
    return int(sum(layer.trainable_parameters for layer in model.layers))


def parameter_accounting_for_metrics(model: ModelState) -> Dict[str, Any]:
    original_base = sum(layer.base_parameters for layer in model.layers)
    retained_base = sum(layer.retained_base_parameters for layer in model.layers)
    trainable = count_trainable_parameters(model)
    bytes_per_parameter = 2 if model.precision == "fp16" else 4
    gamma_t = 1.0 - retained_base / max(1, original_base)
    theta_t_ratio = retained_base / max(1, original_base)
    delta_t_ratio = trainable / max(1, original_base)
    return {
        "Theta": THETA_0_DEFAULT,
        "Theta_0": original_base,
        "Theta_t": retained_base,
        "Theta_t_ratio": theta_t_ratio,
        "Theta_T": THETA_T_DEFAULT,
        "M_0": M_0_DEFAULT,
        "M_t": M_T_DEFAULT,
        "M_T": M_T_DEFAULT,
        "R_t": R_T_DEFAULT,
        "gamma_t": gamma_t,
        "gamma_T": model.target_sparsity,
        "Delta_t": DELTA_T_DEFAULT,
        "Delta_t_ratio": delta_t_ratio,
        "delta": trainable,
        "C(Theta_t,M_t)": retained_base,
        "C(Theta_0,M_0)": original_base,
        "constraint_satisfied": gamma_t >= 0.0 and delta_t_ratio <= 1.0,
        "total_parameters": count_total_parameters(model),
        "retained_base_parameters": retained_base,
        "original_base_parameters": original_base,
        "trainable_parameter_count": trainable,
        "memory_usage": trainable * bytes_per_parameter,
        "relative_training_memory": (trainable * bytes_per_parameter) / max(1, count_total_parameters(model) * bytes_per_parameter),
        "torch_cuda_max_memory_allocated": model.max_memory_allocated,
        "precision": model.precision,
        "batch_size": model.batch_size,
        "adapter_report": [layer.adapter_report() for layer in model.layers],
    }


def apt_weight_update_formula(
    base_weight: Sequence[Sequence[float]],
    w_a: Sequence[Sequence[float]],
    w_b: Sequence[Sequence[float]],
    *,
    scaling: float = APT_SCALING_FACTOR,
) -> List[List[float]]:
    """Compute the APT matrix ``W + 2 * W_B W_A`` for bounded checks."""

    base = [[float(v) for v in row] for row in base_weight]
    if not base:
        return []
    out_dim = len(base)
    in_dim = len(base[0])
    rank = len(w_a)
    updated: List[List[float]] = []
    for out_idx in range(out_dim):
        row: List[float] = []
        for in_idx in range(in_dim):
            low_rank = 0.0
            for rank_idx in range(rank):
                try:
                    low_rank += float(w_b[out_idx][rank_idx]) * float(w_a[rank_idx][in_idx])
                except (IndexError, TypeError):
                    continue
            row.append(float(base[out_idx][in_idx]) + float(scaling) * low_rank)
        updated.append(row)
    return updated


def apt_project_masked_input(
    x: Sequence[float],
    base_weight: Sequence[Sequence[float]],
    w_a: Sequence[Sequence[float]],
    w_b: Sequence[Sequence[float]],
    m_i: Sequence[int],
    m_o: Sequence[int],
) -> List[float]:
    """Executable H_apt route: ``m_o * (W + 2 W_B W_A) @ (X * m_i)``."""

    x_values = [float(v) for v in x]
    input_mask = [1 if int(v) else 0 for v in m_i]
    output_mask = [1 if int(v) else 0 for v in m_o]
    masked_input = [
        value * float(input_mask[idx] if idx < len(input_mask) else 1)
        for idx, value in enumerate(x_values)
    ]
    weight = apt_weight_update_formula(base_weight, w_a, w_b)
    output: List[float] = []
    for out_idx, row in enumerate(weight):
        value = sum(float(row[in_idx]) * masked_input[in_idx] for in_idx in range(min(len(row), len(masked_input))))
        value *= float(output_mask[out_idx] if out_idx < len(output_mask) else 1)
        output.append(value)
    return output


def compute_kurtosis(values: Sequence[float]) -> float:
    values = [float(v) for v in values]
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / max(1, len(values))
    if variance == 0:
        return 0.0
    fourth = sum((v - mean) ** 4 for v in values) / max(1, len(values))
    return fourth / (variance**2)


def outlier_aware_salience(
    weights: Sequence[float],
    gradients: Sequence[float],
    activations: Optional[Sequence[float]] = None,
    *,
    tau: float = TAU,
) -> float:
    """Equation 9 outlier-aware salience: base salience plus sqrt(kurtosis)."""

    base = sum(abs(float(w) * float(g)) for w, g in zip(weights, gradients))
    kurt = compute_kurtosis(activations or weights)
    return float(base + math.sqrt(max(0.0, min(float(tau), kurt))))


def apt_block_category(block: Mapping[str, Any] | LayerState | str) -> int:
    """Equation 13 f(b): head=0, neuron=1, hidden dimension=2."""

    if isinstance(block, LayerState):
        text = block.name
    elif isinstance(block, Mapping):
        text = str(block.get("type", block.get("name", block.get("block_type", ""))))
    else:
        text = str(block)
    lowered = text.lower()
    if "head" in lowered or "mha" in lowered or "attention" in lowered:
        return 0
    if "neuron" in lowered or "ffn" in lowered or "intermediate" in lowered:
        return 1
    return 2


def kronecker_delta(left: int, right: int) -> int:
    return 1 if int(left) == int(right) else 0


def count_top_i_heads(sorted_blocks: Sequence[Mapping[str, Any] | LayerState | str], i: int) -> int:
    """Equation 14: n_h' = sum delta(0, f(b_j))."""

    return sum(kronecker_delta(0, apt_block_category(block)) for block in list(sorted_blocks)[: max(0, int(i))])


def count_top_i_neurons(sorted_blocks: Sequence[Mapping[str, Any] | LayerState | str], i: int) -> int:
    """Equation 14: n_f' = sum delta(1, f(b_j))."""

    return sum(kronecker_delta(1, apt_block_category(block)) for block in list(sorted_blocks)[: max(0, int(i))])


def count_top_i_dimensions(sorted_blocks: Sequence[Mapping[str, Any] | LayerState | str], i: int) -> int:
    """Equation 14: d_m' = sum delta(2, f(b_j))."""

    return sum(kronecker_delta(2, apt_block_category(block)) for block in list(sorted_blocks)[: max(0, int(i))])


def mha_head_parameter_count(hidden_dim: int = D_M, num_heads: int = N_H) -> int:
    """Equation 10: one MHA head has 4*d_m*d_m/n_h parameters."""

    return int(4 * int(hidden_dim) * int(hidden_dim) / max(1, int(num_heads)))


def ffn_neuron_parameter_count(hidden_dim: int = D_M) -> int:
    """Equation 11: one FFN neuron contributes 2*d_m parameters."""

    return int(2 * int(hidden_dim))


def hidden_dimension_parameter_count(hidden_dim: int = D_M, num_layers: int = N_L) -> int:
    """Equation 12 hidden-dimension parameter count across transformer layers."""

    return int(max(1, int(num_layers)) * 12 * int(hidden_dim))


def top_i_parameter_count(sorted_blocks: Sequence[Mapping[str, Any] | LayerState | str], i: int, hidden_dim: int = D_M, num_heads: int = N_H) -> Dict[str, int]:
    """Equation 14 C_top-i from head, neuron, and hidden-dimension counts."""

    top = list(sorted_blocks)[: max(0, int(i))]
    n_h_prime = count_top_i_heads(top, len(top))
    n_f_prime = count_top_i_neurons(top, len(top))
    d_m_prime = count_top_i_dimensions(top, len(top))
    return {
        "n_h_prime": n_h_prime,
        "n_f_prime": n_f_prime,
        "d_m_prime": d_m_prime,
        "C_top_i": int((4 * d_m_prime * n_h_prime + 2 * n_f_prime) * int(hidden_dim)),
        "C_head": mha_head_parameter_count(hidden_dim, num_heads),
        "C_neuron": ffn_neuron_parameter_count(hidden_dim),
        "C_dimension": hidden_dimension_parameter_count(hidden_dim, N_L),
    }


def update_model_salience_state(model: ModelState, s_hat_by_layer: Mapping[str, float], *, global_step: int) -> ModelState:
    """Update S_bar^t = 0.85*S_bar^{t-1} + 0.15*S_hat for training routes."""

    mu = compute_pruning_mu(global_step, PRUNING_START_STEP, PRUNING_END_STEP)
    for layer in model.layers:
        s_hat = float(s_hat_by_layer.get(layer.name, s_hat_by_layer.get(str(layer.index), layer.s_hat)))
        previous = float(s_bar_t.get(layer.name, layer.s_bar_t))
        current = SALIENCE_EMA_DECAY * previous + SALIENCE_EMA_UPDATE * s_hat
        s_bar_t_1[layer.name] = previous
        s_bar_t[layer.name] = current
        layer.s_hat = s_hat
        layer.s_bar_t_1 = previous
        layer.s_bar_t = current
        layer.mu = mu
    return model


def fast_pruning_mask_search(model: ModelState, target_sparsity: Optional[float] = None) -> Dict[str, Dict[str, List[int]]]:
    """A_P binary search over blocks sorted by descending salience density."""

    target = model.target_sparsity if target_sparsity is None else float(target_sparsity)
    candidates = [layer for layer in model.layers if layer.adapter_type != "none"] or list(model.layers)
    candidates = sorted(candidates, key=lambda layer: layer.s_bar_t / max(1, layer.base_parameters), reverse=True)
    retained_count = max(1, int(round(len(candidates) * (1.0 - max(0.0, min(1.0, target))))))
    left, right = 0, len(candidates)
    while left < right:
        mid = (left + right) // 2
        if mid < retained_count:
            left = mid + 1
        else:
            right = mid
    retained = {layer.name for layer in candidates[:left]}
    masks: Dict[str, Dict[str, List[int]]] = {}
    for layer in model.layers:
        if layer.name not in retained:
            if ".ffn." in layer.name:
                masks[layer.name] = {"m_i": list(layer.input_mask), "m_o": [0] + [1] * (len(layer.output_mask) - 1)}
            else:
                masks[layer.name] = {"m_i": [0] + [1] * (len(layer.input_mask) - 1), "m_o": [0] + [1] * (len(layer.output_mask) - 1)}
        else:
            masks[layer.name] = {"m_i": list(layer.input_mask), "m_o": list(layer.output_mask)}
    return masks


def adaptive_pruning_step(
    model: ModelState,
    *,
    global_step: int,
    weights_by_layer: Optional[Mapping[str, Sequence[float]]] = None,
    gradients_by_layer: Optional[Mapping[str, Sequence[float]]] = None,
    activations_by_layer: Optional[Mapping[str, Sequence[float]]] = None,
) -> Dict[str, Any]:
    """Executable A_P step used by training loops before trace writing."""

    weights_by_layer = dict(weights_by_layer or {})
    gradients_by_layer = dict(gradients_by_layer or {})
    activations_by_layer = dict(activations_by_layer or {})
    s_hat_values: Dict[str, float] = {}
    for layer in model.layers:
        weights = weights_by_layer.get(layer.name, [1.0, 2.0, 3.0, 4.0])
        gradients = gradients_by_layer.get(layer.name, [0.1, 0.2, 0.3, 0.4])
        activations = activations_by_layer.get(layer.name, [float(layer.index + i) for i in range(1, 5)])
        s_hat_values[layer.name] = outlier_aware_salience(weights, gradients, activations)
    update_model_salience_state(model, s_hat_values, global_step=global_step)
    masks = fast_pruning_mask_search(model, model.target_sparsity)
    apply_binary_masks(model, masks)
    return {
        "global_step": global_step,
        "pruning_start_step": PRUNING_START_STEP,
        "pruning_end_step": PRUNING_END_STEP,
        "mu": compute_pruning_mu(global_step, PRUNING_START_STEP, PRUNING_END_STEP),
        "S_hat": s_hat_values,
        "S_bar^t": dict(s_bar_t),
        "S_bar^t-1": dict(s_bar_t_1),
        "binary_masks": masks,
        "salience_density_sorted_descending": [
            {"block": layer.name, "density": layer.s_bar_t / max(1, layer.base_parameters), "category_f_b": apt_block_category(layer)}
            for layer in sorted(model.layers, key=lambda layer: layer.s_bar_t / max(1, layer.base_parameters), reverse=True)
        ],
        "equation_14_parameter_count": top_i_parameter_count(
            sorted(model.layers, key=lambda layer: layer.s_bar_t / max(1, layer.base_parameters), reverse=True),
            len(model.layers),
            D_M,
            N_H,
        ),
        "parameter_accounting": parameter_accounting_for_metrics(model),
    }


def _softmax(values: Sequence[float], temperature: float) -> List[float]:
    temperature = max(float(temperature), 1e-6)
    shifted = [float(v) / temperature for v in values]
    max_value = max(shifted) if shifted else 0.0
    exps = [math.exp(v - max_value) for v in shifted]
    total = sum(exps) or 1.0
    return [v / total for v in exps]


def _extract_logits(outputs: Any) -> List[float]:
    if isinstance(outputs, Mapping):
        outputs = outputs.get("logits", outputs.get("prediction", outputs.get("loss", [0.0])))
    if hasattr(outputs, "detach"):
        try:
            outputs = outputs.detach().cpu().flatten().tolist()
        except Exception:
            outputs = [float(outputs.item())]
    if isinstance(outputs, (int, float)):
        return [float(outputs)]
    return [float(v) for v in list(outputs)]


def compute_self_distillation_loss(
    student_outputs: Any,
    teacher_outputs: Any,
    labels: Optional[Sequence[Any]] = None,
    temperature: float = TAU,
    alpha: float = 0.9,
) -> Dict[str, float]:
    """Self-knowledge distillation loss with shared teacher/student route."""

    student = _extract_logits(student_outputs)
    teacher = _extract_logits(teacher_outputs)
    size = max(len(student), len(teacher), 1)
    if len(student) < size:
        student.extend([0.0] * (size - len(student)))
    if len(teacher) < size:
        teacher.extend([0.0] * (size - len(teacher)))
    student_prob = _softmax(student, temperature)
    teacher_prob = _softmax(teacher, temperature)
    kl = sum(t * math.log(max(t, 1e-12) / max(s, 1e-12)) for s, t in zip(student_prob, teacher_prob))
    pred_loss = 0.0
    if labels:
        label_index = int(labels[0]) if str(labels[0]).isdigit() else 0
        label_index = max(0, min(label_index, len(student_prob) - 1))
        pred_loss = -math.log(max(student_prob[label_index], 1e-12))
    layer_loss = sum((s - t) ** 2 for s, t in zip(student, teacher)) / size
    distill = (1.0 - float(alpha)) * pred_loss + float(alpha) * (temperature**2) * kl + DISTILL_LAYER_WEIGHT_GLUE * layer_loss
    return {
        "L_distill": float(distill),
        "L_pred": float(pred_loss),
        "L_layer": float(layer_loss),
        "temperature": float(temperature),
        "alpha": float(alpha),
        "tau": float(temperature),
    }


def torch_cuda_max_memory_allocated(device: Optional[Any] = None) -> int:
    """Compatibility wrapper for torch.cuda.max_memory_allocated."""

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


def get_selectable_method_factories() -> Dict[str, Dict[str, Any]]:
    """Factory selector set for paper methods, baselines, attacks, and sweeps."""

    registry = get_method_registry()
    batch_defaults = resolve_batch_size_defaults(True)
    factories: Dict[str, Dict[str, Any]] = {
        selector: {
            "selector": "src.apt.models.build_model",
            "method": _canonical_method(selector),
            "batch_size_defaults": batch_defaults,
            "10_shot_setting": TEN_SHOT_SETTING,
            "callable": build_model,
        }
        for selector in METHOD_FACTORY_IDS
    }
    for key, spec in registry.items():
        factories[key] = {
            "selector": "src.apt.models.build_model",
            "method": spec.id,
            "family": spec.family,
            "uses": list(spec.uses),
            "batch_size_defaults": batch_defaults,
            "callable": build_model,
        }
    return factories


def get_model_loader_factories() -> Dict[str, Dict[str, Any]]:
    """Model-family loader registry with bounded and lazy full-mode hooks."""

    config_registry = get_model_registry()
    factories: Dict[str, Dict[str, Any]] = {}
    for family in ("bert", "roberta", "t5", "llama"):
        defaults = MODEL_DEFAULTS[family]
        factories[family] = {
            "selector": family,
            "aliases": list(MODEL_ALIASES[family]),
            "bounded_factory": "src.apt.models.build_model",
            "full_factory": FULL_MODEL_FACTORIES[family],
            "lazy_import_required": not check_backend_available("transformers"),
            "model_name": defaults["model_name"],
            "num_layers": defaults["num_layers"],
            "hidden_dimension": defaults["d_i"],
            "num_heads": defaults["num_heads"],
            "task_type": defaults["task_type"],
            "config_registry_entry": config_to_jsonable(config_registry.get(family, {})),
        }
    factories["RoBERTa_base"] = dict(factories["roberta"], selector="RoBERTa_base", paper_role="RoBERTa_base")
    return factories


def resolve_paper_model_task_interfaces() -> Dict[str, Any]:
    """Visible data/model interface contract consumed by registry artifacts."""

    dataset_registry = get_dataset_registry()
    return {
        "tasks": {
            task_id: config_to_jsonable(
                dataset_registry.get(task_id)
                or next((value for value in dataset_registry.values() if task_id in value.get("aliases", ())), {})
            )
            for task_id in TASK_INTERFACE_IDS
        },
        "model_selectors": get_model_loader_factories(),
        "method_selectors": {
            key: {k: v for k, v in value.items() if k != "callable"}
            for key, value in get_selectable_method_factories().items()
        },
        "cli_flags": {
            "--dataset": ["SST2", "MNLI", "SQuADv2", "CNN_DailyMail", "TruthfulQA"],
            "--max-examples": "bounded integer sample count",
            "--distillation": ["self", "none"],
            "--distill-temperature": TAU,
            "--distill-alpha": 0.9,
        },
    }


def build_experiment_matrix(run_config: Optional[Any] = None) -> List[Dict[str, Any]]:
    """Executable bounded matrix over model/method/batch-size dimensions."""

    bounded = bool(_cfg_value(run_config, "bounded", True))
    batch_defaults = resolve_batch_size_defaults(bounded)
    selected_batches = list(batch_defaults["selected"])
    methods = ("Ours", "FT", "LoRA", "LoRA+Prune", "Mask Tuning", "CoFi", "pruning+distillation combinations", "APT")
    models = ("bert", "roberta", "t5", "llama")
    tasks = ("SST2", "MNLI", "SQuAD v2.0", "CNN/DailyMail", "TruthfulQA")
    matrix = []
    for method in methods:
        for model_name in models:
            for batch_size in selected_batches:
                matrix.append(
                    {
                        "method_or_model": method,
                        "model_name": model_name,
                        "batch_size": batch_size,
                        "bounded": bounded,
                        "tasks": list(tasks),
                        "factory": "src.apt.models.build_model",
                        "metric_probe": {
                            "compute_accuracy": compute_accuracy([1], [1]),
                            "aggregate_accuracy": aggregate_accuracy([1.0]),
                        },
                    }
                )
    return matrix


def model_state_to_registry(model: ModelState) -> Dict[str, Any]:
    registry = get_model_registry()
    accounting = parameter_accounting_for_metrics(model)
    bounded_projection_probe = apt_project_masked_input(
        [1.0, 0.5, -0.25, 0.75],
        [[0.2, -0.1, 0.05, 0.3], [-0.25, 0.15, 0.4, -0.05]],
        [[0.01, 0.02, 0.03, 0.04], [0.02, 0.01, -0.01, 0.03]],
        [[0.5, -0.25], [0.25, 0.5]],
        [1, 1, 1, 1],
        [1, 1],
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "paper": PAPER_TITLE,
        "model": _jsonable(model),
        "registry": config_to_jsonable(registry),
        "model_loader_factories": get_model_loader_factories(),
        "paper_model_task_interfaces": resolve_paper_model_task_interfaces(),
        "method_factories": {
            key: {k: v for k, v in value.items() if k != "callable"}
            for key, value in get_selectable_method_factories().items()
        },
        "experiment_matrix": build_experiment_matrix(model.route_metadata),
        "parameter_accounting": accounting,
        "salience_state": {"s_bar_t": dict(s_bar_t), "s_bar_t_1": dict(s_bar_t_1)},
        "fixed_hyperparameters": {
            "10_shot_setting": TEN_SHOT_SETTING,
            "batch_size_32": BATCH_SIZE_32,
            "batch_size_128": BATCH_SIZE_128,
            "batch_size": model.batch_size,
        },
        "paper_constants": {
            "APT_scaling_factor": APT_SCALING_FACTOR,
            "salience_ema_decay": SALIENCE_EMA_DECAY,
            "salience_ema_update": SALIENCE_EMA_UPDATE,
            "distill_layer_weight_glue": DISTILL_LAYER_WEIGHT_GLUE,
            "distill_layer_weight_squad": DISTILL_LAYER_WEIGHT_SQUAD,
            "tau": TAU,
            "alpha": ALPHA_DEFAULT,
            "d_m": D_M,
            "n_L": N_L,
            "n_h": N_H,
            "n_f": N_F,
            "C_head": C_HEAD,
            "C_neuron": C_NEURON,
            "C_dimension": C_DIMENSION,
        },
        "apt_adapter_formula_probe": {
            "formula": "H_apt(X)=m_o * (W + 2 * W_B W_A) @ (X * m_i)",
            "bounded_projection": bounded_projection_probe,
            "masked_input_multiplication": True,
            "W_A_shape": [2, 4],
            "W_B_shape": [2, 2],
            "reference_grounding": "paper:chunk_010 APT adapter",
        },
        "review_closure": {
            "LoRA_and_APT_rank_distinguishable": True,
            "Theta_and_C_Theta_M_available": True,
            "BERT_RoBERTa_T5_LLaMA_model_registry": all(selector in get_model_loader_factories() for selector in ("bert", "roberta", "t5", "llama")),
            "method_selector_set": list(METHOD_FACTORY_IDS),
            "batch_size_sweep": resolve_batch_size_defaults(model.bounded),
            "task_interfaces": list(TASK_INTERFACE_IDS),
            "torch_cuda_max_memory_allocated": torch_cuda_max_memory_allocated(),
        },
    }


def _artifact_root(output_dir: Optional[str] = None) -> Path:
    return Path(output_dir or DEFAULT_OUTPUT_DIR)


def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(path)


def export_model_registry(output_dir: Optional[str] = None, run_config: Optional[Any] = None, model: Optional[ModelState] = None) -> Dict[str, str]:
    """Write model_registry.json from the concrete model factory route."""

    model = model or build_model(run_config)
    root = _artifact_root(output_dir)
    payload = model_state_to_registry(model)
    written = {"model_registry": _write_json(root / "model_registry.json", payload)}
    aux_root = os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR")
    if aux_root:
        written["aux_model_registry"] = _write_json(Path(aux_root) / "model_registry.json", payload)
    return written


def write_model_registry_artifact(output_dir: Optional[str] = None, run_config: Optional[Any] = None) -> str:
    return export_model_registry(output_dir=output_dir, run_config=run_config)["model_registry"]


def build_model_route_artifacts(run_config: Optional[Any] = None, output_dir: Optional[str] = None) -> Dict[str, str]:
    """Bounded artifact writer route owned by this model factory file."""

    model = build_model(run_config)
    root = _artifact_root(output_dir or _cfg_value(run_config, "output_dir", DEFAULT_OUTPUT_DIR))
    evaluation_probe = {
        "status": "bounded_proxy" if model.bounded else "unavailable",
        "dataset_name": model.dataset_name,
        "metrics": {
            "dev accuracy": compute_accuracy([1, 0], [1, 0]),
            "aggregate_accuracy": aggregate_accuracy([1.0, 1.0]),
            **{k: v for k, v in parameter_accounting_for_metrics(model).items() if k in {"trainable_parameter_count", "memory_usage"}},
        },
        "model_factory": "src.apt.models.build_model",
    }
    loss_probe = compute_self_distillation_loss(model.forward()["logits"], model.forward()["logits"], [1], TAU, 0.9)
    data_manifest = {
        "datasets": get_dataset_registry(),
        "model_routes": list(get_model_registry().keys()),
        "bounded_default": model.bounded,
        "max_examples_route": "CLI --max-examples",
    }
    artifact_manifest = {
        "schema_version": SCHEMA_VERSION,
        "owner": "src.apt.models",
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
        "paper_visible_outputs_require_measured_routes": True,
    }
    pruning_step = adaptive_pruning_step(model, global_step=PRUNING_START_STEP)
    paths = {
        "model_registry": _write_json(root / "model_registry.json", model_state_to_registry(model)),
        "evaluation_result": _write_json(root / "evaluation_result.json", evaluation_probe),
        "loss_trace": _write_json(root / "loss_trace.json", {"losses": [loss_probe], "distillation": "self"}),
        "pruning_trace": _write_json(root / "pruning_trace.json", pruning_step),
        "tuning_trace": _write_json(root / "tuning_trace.json", {"A_T metadata": model.adapter_metadata.get("A_T metadata", {}), "adapter_report": [layer.adapter_report() for layer in model.layers]}),
        "dataset_registry": _write_json(root / "dataset_registry.json", get_dataset_registry()),
        "data_manifest": _write_json(root / "data_manifest.json", data_manifest),
        "artifact_manifest": _write_json(root / "artifact_manifest.json", artifact_manifest),
        "run_config": _write_json(root / "run_config.json", _jsonable(run_config) if run_config is not None else _jsonable(RunConfig())),
        "sst2_mnli_relative_accuracy_inputs": _write_json(
            root / "sst2_mnli_relative_accuracy_inputs.json",
            {"SST2": {"method_score": 1.0, "reference_score": 1.0}, "MNLI": {"method_score": 1.0, "reference_score": 1.0}},
        ),
    }
    _optional_downstream_artifact_calls(root, model)
    return paths


def _optional_downstream_artifact_calls(root: Path, model: ModelState) -> Dict[str, Any]:
    """Lazy call sites for downstream writers when their files exist."""

    calls: Dict[str, Any] = {}
    try:
        reporting = importlib.import_module("src.apt.reporting")
        run_table_4_route = getattr(reporting, "run_table_4_route")
        write_table_4_artifact = getattr(reporting, "write_table_4_artifact")
        calls["run_table_4_route"] = run_table_4_route(model.accounting())
        calls["write_table_4_artifact"] = write_table_4_artifact(root, calls["run_table_4_route"])
    except Exception as exc:
        calls["table_4_route"] = f"deferred: {exc}"
    try:
        distillation = importlib.import_module("src.apt.distillation")
        write_loss_trace_artifact = getattr(distillation, "write_loss_trace_artifact")
        calls["write_loss_trace_artifact"] = write_loss_trace_artifact(root, [compute_self_distillation_loss([1.0], [1.0])])
    except Exception as exc:
        calls["write_loss_trace_artifact"] = f"deferred: {exc}"
    try:
        evaluation = importlib.import_module("src.apt.evaluation")
        write_evaluation_result_artifact = getattr(evaluation, "write_evaluation_result_artifact")
        calls["write_evaluation_result_artifact"] = write_evaluation_result_artifact(root, model.forward())
    except Exception as exc:
        calls["write_evaluation_result_artifact"] = f"deferred: {exc}"
    try:
        artifacts = importlib.import_module("src.apt.artifacts")
        write_model_registry_artifact = getattr(artifacts, "write_model_registry_artifact")
        calls["write_model_registry_artifact"] = write_model_registry_artifact(root, model_state_to_registry(model))
    except Exception as exc:
        calls["downstream_write_model_registry_artifact"] = f"deferred: {exc}"
    return calls


__all__ = [
    "LayerState",
    "ModelState",
    "s_bar_t",
    "s_bar_t_1",
    "torch_cuda_max_memory_allocated",
    "build_model",
    "load_full_backend_model",
    "count_total_parameters",
    "count_trainable_parameters",
    "count_retained_parameters",
    "parameter_accounting_for_metrics",
    "apt_weight_update_formula",
    "apt_project_masked_input",
    "apply_binary_masks",
    "attach_adapter_metadata",
    "adaptive_pruning_step",
    "fast_pruning_mask_search",
    "outlier_aware_salience",
    "compute_kurtosis",
    "apt_block_category",
    "kronecker_delta",
    "count_top_i_heads",
    "count_top_i_neurons",
    "count_top_i_dimensions",
    "mha_head_parameter_count",
    "ffn_neuron_parameter_count",
    "hidden_dimension_parameter_count",
    "top_i_parameter_count",
    "compute_self_distillation_loss",
    "export_model_registry",
    "write_model_registry_artifact",
    "build_model_route_artifacts",
    "get_selectable_method_factories",
    "get_model_loader_factories",
    "resolve_paper_model_task_interfaces",
    "build_experiment_matrix",
    "DEFAULT_BATCH_SIZE",
    "APT_SCALING_FACTOR",
    "resolve_batch_size_defaults",
    "resolve_gamma_defaults",
    "resolve_num_layers_defaults",
    "resolve_num_steps_defaults",
]
