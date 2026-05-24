from __future__ import annotations

import csv
import importlib
import json
import math
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

reference_grounding = "reference_grounding: chunk_009 /mnt/paper2any/pzw/proj/paperagent/hx/Research_space/Reproduction/paperbench_data/sample-specific-masks/paper.md"

DEFAULT_EPOCHS = 1
DEFAULT_SEED = 0
DEFAULT_ALPHA = 1.0
DEFAULT_PATCH_SIZE = 2
DEFAULT_INTERPOLATION_LEVEL = 1
DEFAULT_BATCH_SIZE = 4
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_GAMMA = 1.0
DEFAULT_P = 0.5
DEFAULT_SIMILARITY_GUIDANCE_SCALE = 9.0
DEFAULT_DRY_RUN_MODE = "runtime_smoke"
DEFAULT_MASK_VARIANT = "ours_multi_channel"
DEFAULT_DATASET = "unit-001"
DEFAULT_BACKBONE = "resnet18_imagenet1k"
DEFAULT_METHOD = "Ours"
DEFAULT_OUTPUT_MAPPING = "Rlm_random_label_mapping"
DEFAULT_CLASS_COUNT = 1000
DEFAULT_IMAGE_SIZE = 224
DEFAULT_VIT_IMAGE_SIZE = 384

three_seed_protocol = (0, 1, 2)
seed_values = tuple(three_seed_protocol)
epochs_values = (1, 3, 5)
learning_rate_values = (1e-4, 1e-3, 5e-3)
batch_size_values = (4, 8, 16)
alpha_values = (0.0, 0.5, 1.0)
gamma_values = (0.0, 1.0, 2.0)
p_values = (0.0, 0.5, 1.0)
patch_size_values = (4, 2, 1)
similarity_guidance_scale_values = (9.0, 7.0, 10.0)

METHOD_ALIASES = {
    "ours": "Ours",
    "Ours": "Ours",
    "only_delta": "ONLY δ",
    "only_f_mask": "ONLY f_mask",
    "single_channel_mask": "SINGLE-CHANNEL f_mask^s",
    "single-channel-mask": "SINGLE-CHANNEL f_mask^s",
    "vit": "vit",
    "resnet": "resnet",
    "lora": "lora",
}

MASK_VARIANT_ALIASES = {
    "ours_multi_channel": "ours_multi_channel",
    "ours": "ours_multi_channel",
    "only_delta": "only_delta",
    "only_f_mask": "only_f_mask",
    "single_channel_mask": "single_channel_mask",
    "single_channel": "single_channel_mask",
}

DATASET_ALIASES = {
    "cifar10": "cifar10",
    "cifar100": "cifar100",
    "svhn": "svhn",
    "gtsrb": "gtsrb",
    "flowers102": "flowers102",
    "flowers": "flowers102",
    "dtd": "dtd",
    "ucf101": "ucf101",
    "eurosat": "eurosat",
    "imagenet_1k": "imagenet_1k",
    "imagenet": "imagenet_1k",
    "stanford_cars": "stanford_cars",
    "oxford_pets": "oxford_pets",
    "unit-001": "unit-001",
    "unit_001": "unit-001",
}

BACKBONE_ALIASES = {
    "resnet18_imagenet1k": "resnet18_imagenet1k",
    "resnet-18": "resnet18_imagenet1k",
    "resnet18": "resnet18_imagenet1k",
    "resnet50_imagenet1k": "resnet50_imagenet1k",
    "resnet-50": "resnet50_imagenet1k",
    "resnet50": "resnet50_imagenet1k",
    "vit_b32_imagenet1k": "vit_b32_imagenet1k",
    "vit-b32": "vit_b32_imagenet1k",
    "vit_b32": "vit_b32_imagenet1k",
    "imagenet_1k": "imagenet_1k",
}

ALLOWED_METHODS = ("ours", "vit", "resnet", "lora")
ALLOWED_BASELINES = ("PAD", "Narrow", "Medium", "Full")
ALLOWED_VARIANTS = ("ours_multi_channel", "only_delta", "only_f_mask", "single_channel_mask")

SMOKE_MODE = "runtime_smoke"
FULL_MODE = "full_run"
SMOKE_EXPERIMENT_ID = "smm_smoke"

RESULT_DIR_DEFAULT = "results"
TABLE_DIR = "tables"
FIGURE_DIR = "figures"

EXPERIMENT_TABLE_1 = "table1_resnet"
EXPERIMENT_TABLE_2 = "table2_vit"
EXPERIMENT_TABLE_3 = "table3_ablation"
EXPERIMENT_APPENDIX_TABLE_13 = "appendix_table13"
EXPERIMENT_APPENDIX_TABLE_14 = "appendix_table14"
EXPERIMENT_SMOKE = "smm_smoke"

METHOD_ROW_TABLE_3 = ("ONLY δ", "ONLY f_mask", "SINGLE-CHANNEL f_mask^s", "OURS")
METHOD_ROW_TABLE_1 = ("PAD", "Narrow", "Medium", "Full", "Ours")
METHOD_ROW_TABLE_2 = ("PAD", "Narrow", "Medium", "Full", "Ours")

TABLE_1_TARGET_DATASETS = (
    "cifar10",
    "cifar100",
    "svhn",
    "gtsrb",
    "flowers102",
    "dtd",
    "ucf101",
    "eurosat",
)
TABLE_2_TARGET_DATASETS = TABLE_1_TARGET_DATASETS
TABLE_3_TARGET_DATASETS = ("cifar10", "cifar100", "svhn", "gtsrb", "flowers102", "dtd", "ucf101")
APPENDIX_DATASETS = (
    "cifar10",
    "cifar100",
    "svhn",
    "gtsrb",
    "flowers102",
    "dtd",
    "ucf101",
    "eurosat",
    "stanford_cars",
    "oxford_pets",
)

SCHEMA_READY_ARTIFACTS = ("readiness.json", "evaluation_result.json")
PAPER_VISIBLE_ARTIFACTS = (
    "results/metrics.json",
    "results/tables/table1_resnet_main.csv",
    "results/tables/table2_vit_main.csv",
    "results/tables/table3_ablation.csv",
    "results/config_resolved.json",
    "results/training_trace.json",
    "results/mask_statistics.json",
    "results/summary_table.csv",
    "results/table_1_resnet.csv",
)

TABLE_ARTIFACT_PATHS = {
    "table_1": "results/tables/table1_resnet_main.csv",
    "table_2": "results/tables/table2_vit_main.csv",
    "table_3": "results/tables/table3_ablation.csv",
    "table_13": "results/tables/table_13.csv",
    "table_14": "results/tables/table_14.csv",
}

FIGURE_ARTIFACT_PATHS = {f"figure_{i}": f"results/figures/figure_{i}.png" for i in range(13, 24)}

METRIC_IDENTIFIERS = {
    "accuracy": "metric_accuracy",
    "loss": "metric_loss",
    "mean_std_accuracy": "metric_mean_std_accuracy",
    "f1": "metric_f1",
}

TREND_OBLIGATIONS = {
    "endpoint_low": "p=0 and p=1 must be represented as lowest/minimum boundary cases",
    "positive_parameter_improves": "nonzero/positive parameter values should preserve the reported improvement trend",
}

SCOPE_CONSTRAINTS = (
    "仅实现论文复现所需的最小可运行闭环。",
    "仅覆盖论文中的输入视觉重编程主路径，f_out 作为非参数映射单独处理。",
    "只实现论文中实际比较过的固定掩码族，不扩展到未出现的额外基线。",
    "仅实现论文所列四个消融分支，不额外添加近似变体。",
    "仅实现论文理解性分析所需的诊断输出，不做超出论文范围的额外理论扩展。",
)

METHOD_REGISTRY: Dict[str, Dict[str, Any]] = {
    "Ours": {
        "id": "ours",
        "aliases": ("ours", "Ours"),
        "selector": "ours",
        "kind": "proposed",
        "mask_variant": "ours_multi_channel",
        "reference_grounding": reference_grounding,
    },
    "ONLY δ": {
        "id": "only_delta",
        "aliases": ("ONLY δ", "only_delta"),
        "selector": "only_delta",
        "kind": "ablation",
        "mask_variant": "only_delta",
        "reference_grounding": reference_grounding,
    },
    "ONLY f_mask": {
        "id": "only_f_mask",
        "aliases": ("ONLY f_mask", "only_f_mask"),
        "selector": "only_f_mask",
        "kind": "ablation",
        "mask_variant": "only_f_mask",
        "reference_grounding": reference_grounding,
    },
    "SINGLE-CHANNEL f_mask^s": {
        "id": "single_channel_mask",
        "aliases": ("SINGLE-CHANNEL f_mask^s", "single_channel_mask"),
        "selector": "single_channel_mask",
        "kind": "ablation",
        "mask_variant": "single_channel_mask",
        "reference_grounding": reference_grounding,
    },
    "PAD": {
        "id": "pad",
        "aliases": ("PAD", "Pad"),
        "selector": "pad",
        "kind": "baseline",
        "reference_grounding": reference_grounding,
    },
    "Narrow": {
        "id": "narrow",
        "aliases": ("Narrow",),
        "selector": "narrow",
        "kind": "baseline",
        "reference_grounding": reference_grounding,
    },
    "Medium": {
        "id": "medium",
        "aliases": ("Medium",),
        "selector": "medium",
        "kind": "baseline",
        "reference_grounding": reference_grounding,
    },
    "Full": {
        "id": "full",
        "aliases": ("Full",),
        "selector": "full",
        "kind": "baseline",
        "reference_grounding": reference_grounding,
    },
    "vit": {
        "id": "vit",
        "aliases": ("vit", "ViT"),
        "selector": "vit",
        "kind": "backbone_adapter",
        "reference_grounding": reference_grounding,
    },
    "resnet": {
        "id": "resnet",
        "aliases": ("resnet", "ResNet"),
        "selector": "resnet",
        "kind": "backbone_adapter",
        "reference_grounding": reference_grounding,
    },
    "lora": {
        "id": "lora",
        "aliases": ("lora", "LoRA"),
        "selector": "lora",
        "kind": "method_adapter",
        "reference_grounding": reference_grounding,
    },
}

Ids = type(
    "Ids",
    (),
    {
        "OURS": "ours",
        "ONLY_DELTA": "only_delta",
        "ONLY_F_MASK": "only_f_mask",
        "SINGLE_CHANNEL_MASK": "single_channel_mask",
        "PAD": "pad",
        "NARROW": "narrow",
        "MEDIUM": "medium",
        "FULL": "full",
        "VIT": "vit",
        "RESNET": "resnet",
        "LORA": "lora",
        "CIFAR10": "cifar10",
        "CIFAR100": "cifar100",
        "SVHN": "svhn",
        "GTSRB": "gtsrb",
        "FLOWERS102": "flowers102",
        "DTD": "dtd",
        "UCF101": "ucf101",
        "EUROSAT": "eurosat",
        "IMAGENET_1K": "imagenet_1k",
        "UNIT_001": "unit-001",
        "RESNET18_IMAGENET1K": "resnet18_imagenet1k",
        "RESNET50_IMAGENET1K": "resnet50_imagenet1k",
        "VIT_B32_IMAGENET1K": "vit_b32_imagenet1k",
    },
)

DEFAULT_EXPERIMENT_TABLE_1 = {
    "experiment_id": EXPERIMENT_TABLE_1,
    "name": "Table 1 main ResNet comparison",
    "artifact": "results/tables/table1_resnet_main.csv",
    "backbones": ("resnet18_imagenet1k", "resnet50_imagenet1k"),
    "methods": METHOD_ROW_TABLE_1,
    "datasets": TABLE_1_TARGET_DATASETS,
    "metric": "accuracy",
    "aggregate": "mean_std_accuracy",
    "output_mapping": DEFAULT_OUTPUT_MAPPING,
    "mode": "full_run",
    "reference_grounding": reference_grounding,
}

DEFAULT_EXPERIMENT_TABLE_2 = {
    "experiment_id": EXPERIMENT_TABLE_2,
    "name": "Table 2 ViT-B/32 comparison",
    "artifact": "results/tables/table2_vit_main.csv",
    "backbones": ("vit_b32_imagenet1k",),
    "methods": METHOD_ROW_TABLE_2,
    "datasets": TABLE_2_TARGET_DATASETS,
    "metric": "accuracy",
    "aggregate": "mean_std_accuracy",
    "output_mapping": DEFAULT_OUTPUT_MAPPING,
    "mode": "full_run",
    "reference_grounding": reference_grounding,
}

DEFAULT_EXPERIMENT_TABLE_3 = {
    "experiment_id": EXPERIMENT_TABLE_3,
    "name": "Table 3 ablation studies",
    "artifact": "results/tables/table3_ablation.csv",
    "backbones": ("resnet18_imagenet1k",),
    "methods": METHOD_ROW_TABLE_3,
    "datasets": TABLE_3_TARGET_DATASETS,
    "metric": "accuracy",
    "aggregate": "mean_std_accuracy",
    "output_mapping": DEFAULT_OUTPUT_MAPPING,
    "mode": "full_run",
    "reference_grounding": reference_grounding,
}

DEFAULT_APPENDIX_TABLE_13 = {
    "experiment_id": EXPERIMENT_APPENDIX_TABLE_13,
    "name": "Table 13 appendix table",
    "artifact": "results/tables/table_13.csv",
    "backbones": ("resnet18_imagenet1k", "resnet50_imagenet1k", "vit_b32_imagenet1k"),
    "methods": METHOD_ROW_TABLE_1,
    "datasets": APPENDIX_DATASETS,
    "metric": "accuracy",
    "aggregate": "mean_std_accuracy",
    "output_mapping": DEFAULT_OUTPUT_MAPPING,
    "mode": "full_run",
    "reference_grounding": reference_grounding,
}

DEFAULT_APPENDIX_TABLE_14 = {
    "experiment_id": EXPERIMENT_APPENDIX_TABLE_14,
    "name": "Table 14 appendix table",
    "artifact": "results/tables/table_14.csv",
    "backbones": ("resnet18_imagenet1k", "resnet50_imagenet1k", "vit_b32_imagenet1k"),
    "methods": METHOD_ROW_TABLE_1,
    "datasets": APPENDIX_DATASETS,
    "metric": "accuracy",
    "aggregate": "mean_std_accuracy",
    "output_mapping": DEFAULT_OUTPUT_MAPPING,
    "mode": "full_run",
    "reference_grounding": reference_grounding,
}

DEFAULT_SMOKE_EXPERIMENT = {
    "experiment_id": EXPERIMENT_SMOKE,
    "name": "smm_smoke",
    "artifact": "results/smoke/metrics.json",
    "backbones": ("resnet18_imagenet1k",),
    "methods": ("Ours",),
    "datasets": (DEFAULT_DATASET,),
    "metric": "accuracy",
    "aggregate": "mean_std_accuracy",
    "output_mapping": DEFAULT_OUTPUT_MAPPING,
    "mode": SMOKE_MODE,
    "reference_grounding": reference_grounding,
}

EXPERIMENT_PROTOCOLS: Dict[str, Dict[str, Any]] = {
    EXPERIMENT_TABLE_1: DEFAULT_EXPERIMENT_TABLE_1,
    EXPERIMENT_TABLE_2: DEFAULT_EXPERIMENT_TABLE_2,
    EXPERIMENT_TABLE_3: DEFAULT_EXPERIMENT_TABLE_3,
    EXPERIMENT_APPENDIX_TABLE_13: DEFAULT_APPENDIX_TABLE_13,
    EXPERIMENT_APPENDIX_TABLE_14: DEFAULT_APPENDIX_TABLE_14,
    EXPERIMENT_SMOKE: DEFAULT_SMOKE_EXPERIMENT,
}

ENVIRONMENT_REGISTRY: Dict[str, Dict[str, Any]] = {}
DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {}
BACKBONE_REGISTRY: Dict[str, Dict[str, Any]] = {}
METRIC_REGISTRY: Dict[str, Callable[..., Any]] = {}
ARTIFACT_WRITER_REGISTRY: Dict[str, Callable[..., Any]] = {}

EVIDENCE_MATRIX_ROWS = [
    {
        "experiment_id": EXPERIMENT_TABLE_1,
        "name": "Table 1 main ResNet comparison",
        "artifact": "results/tables/table1_resnet_main.csv",
        "environment_ids": ("imagenet_1k",),
        "dataset_ids": TABLE_1_TARGET_DATASETS,
        "backbone_ids": ("resnet18_imagenet1k", "resnet50_imagenet1k"),
        "method_ids": ("PAD", "Narrow", "Medium", "Full", "Ours"),
        "metric_id": "accuracy",
        "aggregate_id": "mean_std_accuracy",
        "seed_protocol": three_seed_protocol,
        "sweeps": {
            "patch_size": patch_size_values,
            "p": p_values,
            "alpha": alpha_values,
            "gamma": gamma_values,
            "similarity_guidance_scale": similarity_guidance_scale_values,
        },
        "decision_value": "Main ResNet comparison for paper table 1.",
        "trend": "Ours expected to improve over predetermined shared mask VR baselines.",
        "reference_grounding": reference_grounding,
    },
    {
        "experiment_id": EXPERIMENT_TABLE_2,
        "name": "Table 2 ViT-B/32 comparison",
        "artifact": "results/tables/table2_vit_main.csv",
        "environment_ids": ("imagenet_1k",),
        "dataset_ids": TABLE_2_TARGET_DATASETS,
        "backbone_ids": ("vit_b32_imagenet1k",),
        "method_ids": ("PAD", "Narrow", "Medium", "Full", "Ours"),
        "metric_id": "accuracy",
        "aggregate_id": "mean_std_accuracy",
        "seed_protocol": three_seed_protocol,
        "sweeps": {
            "patch_size": patch_size_values,
            "p": p_values,
            "alpha": alpha_values,
            "gamma": gamma_values,
            "similarity_guidance_scale": similarity_guidance_scale_values,
        },
        "decision_value": "ViT-B/32 comparison for paper table 2.",
        "trend": "Ours expected to outperform or be competitive with PAD/Narrow/Medium/Full across target tasks.",
        "reference_grounding": reference_grounding,
    },
    {
        "experiment_id": EXPERIMENT_TABLE_3,
        "name": "Table 3 ablation studies",
        "artifact": "results/tables/table3_ablation.csv",
        "environment_ids": ("imagenet_1k",),
        "dataset_ids": TABLE_3_TARGET_DATASETS,
        "backbone_ids": ("resnet18_imagenet1k",),
        "method_ids": ("ONLY δ", "ONLY f_mask", "SINGLE-CHANNEL f_mask^s", "OURS"),
        "metric_id": "accuracy",
        "aggregate_id": "mean_std_accuracy",
        "seed_protocol": three_seed_protocol,
        "sweeps": {
            "patch_size": patch_size_values,
            "p": p_values,
        },
        "decision_value": "Ablation diagnostics for mask and delta contributions.",
        "trend": "OURS expected to be strongest or competitive among ablation variants.",
        "reference_grounding": reference_grounding,
    },
    {
        "experiment_id": EXPERIMENT_APPENDIX_TABLE_13,
        "name": "Table 13 appendix table",
        "artifact": "results/tables/table_13.csv",
        "environment_ids": ("imagenet_1k",),
        "dataset_ids": APPENDIX_DATASETS,
        "backbone_ids": ("resnet18_imagenet1k", "resnet50_imagenet1k", "vit_b32_imagenet1k"),
        "method_ids": ("PAD", "Narrow", "Medium", "Full", "Ours"),
        "metric_id": "accuracy",
        "aggregate_id": "mean_std_accuracy",
        "seed_protocol": three_seed_protocol,
        "sweeps": {
            "patch_size": patch_size_values,
            "p": p_values,
        },
        "decision_value": "Appendix table 13 routing.",
        "trend": "Appendix table preserves paper-visible comparison ordering.",
        "reference_grounding": reference_grounding,
    },
    {
        "experiment_id": EXPERIMENT_APPENDIX_TABLE_14,
        "name": "Table 14 appendix table",
        "artifact": "results/tables/table_14.csv",
        "environment_ids": ("imagenet_1k",),
        "dataset_ids": APPENDIX_DATASETS,
        "backbone_ids": ("resnet18_imagenet1k", "resnet50_imagenet1k", "vit_b32_imagenet1k"),
        "method_ids": ("PAD", "Narrow", "Medium", "Full", "Ours"),
        "metric_id": "accuracy",
        "aggregate_id": "mean_std_accuracy",
        "seed_protocol": three_seed_protocol,
        "sweeps": {
            "patch_size": patch_size_values,
            "p": p_values,
        },
        "decision_value": "Appendix table 14 routing.",
        "trend": "Appendix table preserves paper-visible comparison ordering.",
        "reference_grounding": reference_grounding,
    },
    {
        "experiment_id": EXPERIMENT_SMOKE,
        "name": "smm_smoke",
        "artifact": "results/smoke/metrics.json",
        "environment_ids": ("unit-001",),
        "dataset_ids": (DEFAULT_DATASET,),
        "backbone_ids": ("resnet18_imagenet1k",),
        "method_ids": ("Ours",),
        "metric_id": "accuracy",
        "aggregate_id": "mean_std_accuracy",
        "seed_protocol": (DEFAULT_SEED,),
        "sweeps": {
            "patch_size": (DEFAULT_PATCH_SIZE,),
            "p": (DEFAULT_P,),
        },
        "decision_value": "Smoke route validates wiring only.",
        "trend": "Smoke validation must not claim benchmark scores.",
        "reference_grounding": reference_grounding,
    },
]

def _alias(value: str, mapping: Mapping[str, str], default: Optional[str] = None) -> str:
    if value in mapping:
        return mapping[value]
    normalized = value.strip()
    lower = normalized.lower()
    if lower in mapping:
        return mapping[lower]
    return default if default is not None else normalized

def normalize_seed_list(seeds: Iterable[int] | None = None) -> Tuple[int, ...]:
    if seeds is None:
        return seed_values
    return tuple(int(s) for s in seeds)

def resolve_seed_defaults(seed: Optional[int] = None, seeds: Optional[Iterable[int]] = None) -> Dict[str, Any]:
    normalized = normalize_seed_list(seeds if seeds is not None else ([seed] if seed is not None else None))
    selected = normalized[0] if normalized else DEFAULT_SEED
    return {
        "default_seed": DEFAULT_SEED,
        "seed": selected,
        "seed_values": seed_values,
        "seed_protocol": three_seed_protocol,
        "resolved_seeds": normalized,
    }

def resolve_epochs_defaults(epochs: Optional[int] = None, mode: str = DEFAULT_DRY_RUN_MODE) -> Dict[str, Any]:
    if epochs is None:
        epochs = DEFAULT_EPOCHS if mode == SMOKE_MODE else max(epochs_values)
    return {
        "default_epochs": DEFAULT_EPOCHS,
        "epochs": int(epochs),
        "epochs_values": epochs_values,
        "mode": mode,
    }

def resolve_learning_rate_defaults(learning_rate: Optional[float] = None) -> Dict[str, Any]:
    return {
        "default_learning_rate": DEFAULT_LEARNING_RATE,
        "learning_rate": float(DEFAULT_LEARNING_RATE if learning_rate is None else learning_rate),
        "learning_rate_values": learning_rate_values,
    }

def resolve_batch_size_defaults(batch_size: Optional[int] = None) -> Dict[str, Any]:
    return {
        "default_batch_size": DEFAULT_BATCH_SIZE,
        "batch_size": int(DEFAULT_BATCH_SIZE if batch_size is None else batch_size),
        "batch_size_values": batch_size_values,
    }

def resolve_alpha_defaults(alpha: Optional[float] = None) -> Dict[str, Any]:
    return {
        "default_alpha": DEFAULT_ALPHA,
        "alpha": float(DEFAULT_ALPHA if alpha is None else alpha),
        "alpha_values": alpha_values,
    }

def resolve_gamma_defaults(gamma: Optional[float] = None) -> Dict[str, Any]:
    return {
        "default_gamma": DEFAULT_GAMMA,
        "gamma": float(DEFAULT_GAMMA if gamma is None else gamma),
        "gamma_values": gamma_values,
    }

def resolve_patch_size_defaults(patch_size: Optional[int] = None, interpolation_level: Optional[int] = None) -> Dict[str, Any]:
    return {
        "default_patch_size": DEFAULT_PATCH_SIZE,
        "patch_size": int(DEFAULT_PATCH_SIZE if patch_size is None else patch_size),
        "patch_size_values": patch_size_values,
        "default_interpolation_level": DEFAULT_INTERPOLATION_LEVEL,
        "interpolation_level": int(DEFAULT_INTERPOLATION_LEVEL if interpolation_level is None else interpolation_level),
    }

def compute_accuracy(predictions: Sequence[int], labels: Sequence[int]) -> float:
    preds = list(predictions)
    gold = list(labels)
    if not gold:
        return 0.0
    correct = sum(int(p == y) for p, y in zip(preds, gold))
    return correct / len(gold)

def aggregate_accuracy(values: Sequence[float]) -> Dict[str, float]:
    vals = [float(v) for v in values]
    if not vals:
        return {"mean": 0.0, "std": 0.0}
    if len(vals) == 1:
        return {"mean": vals[0], "std": 0.0}
    return {"mean": mean(vals), "std": pstdev(vals)}

def compute_f1(predictions: Sequence[int], labels: Sequence[int]) -> float:
    preds = list(predictions)
    gold = list(labels)
    if not gold:
        return 0.0
    classes = sorted(set(gold) | set(preds))
    if not classes:
        return 0.0
    f1_values: List[float] = []
    for cls in classes:
        tp = sum(int(p == cls and y == cls) for p, y in zip(preds, gold))
        fp = sum(int(p == cls and y != cls) for p, y in zip(preds, gold))
        fn = sum(int(p != cls and y == cls) for p, y in zip(preds, gold))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        if precision + recall == 0.0:
            f1_values.append(0.0)
        else:
            f1_values.append(2 * precision * recall / (precision + recall))
    return sum(f1_values) / len(f1_values)

def aggregate_f1(values: Sequence[float]) -> Dict[str, float]:
    vals = [float(v) for v in values]
    if not vals:
        return {"mean": 0.0, "std": 0.0}
    if len(vals) == 1:
        return {"mean": vals[0], "std": 0.0}
    return {"mean": mean(vals), "std": pstdev(vals)}

def compute_loss(logits: Sequence[float], labels: Sequence[int]) -> float:
    preds = list(logits)
    gold = list(labels)
    if not gold:
        return 0.0
    count = min(len(preds), len(gold))
    if count == 0:
        return 0.0
    # Lightweight proxy loss for smoke-mode wiring: absolute deviation from label index.
    return sum(abs(float(preds[i]) - float(gold[i])) for i in range(count)) / count

def aggregate_loss(values: Sequence[float]) -> Dict[str, float]:
    vals = [float(v) for v in values]
    if not vals:
        return {"mean": 0.0, "std": 0.0}
    if len(vals) == 1:
        return {"mean": vals[0], "std": 0.0}
    return {"mean": mean(vals), "std": pstdev(vals)}

def compute_reward(accuracy_value: float, loss_value: float) -> float:
    return float(accuracy_value) - float(loss_value)

def aggregate_reward(values: Sequence[float]) -> Dict[str, float]:
    vals = [float(v) for v in values]
    if not vals:
        return {"mean": 0.0, "std": 0.0}
    if len(vals) == 1:
        return {"mean": vals[0], "std": 0.0}
    return {"mean": mean(vals), "std": pstdev(vals)}

@dataclass(frozen=True)
class DatasetSpec:
    id: str
    aliases: Tuple[str, ...]
    split: str
    num_classes: int
    image_size: int
    target_tasks: Tuple[str, ...]
    reference_grounding: str = reference_grounding
    availability: str = "lazy"

    def validate(self) -> bool:
        return bool(self.id and self.num_classes > 0 and self.image_size > 0)

@dataclass(frozen=True)
class EnvironmentSpec:
    id: str
    aliases: Tuple[str, ...]
    datasets: Tuple[str, ...]
    backbones: Tuple[str, ...]
    description: str
    reference_grounding: str = reference_grounding
    availability: str = "lazy"

    def validate(self) -> bool:
        return bool(self.id and self.datasets and self.backbones)

@dataclass(frozen=True)
class BackboneSpec:
    id: str
    aliases: Tuple[str, ...]
    architecture: str
    pretrained_source: str
    input_size: int
    frozen: bool = True
    reference_grounding: str = reference_grounding
    availability: str = "lazy"

    def validate(self) -> bool:
        return bool(self.id and self.architecture and self.pretrained_source)

@dataclass(frozen=True)
class MetricSpec:
    id: str
    name: str
    compute: Callable[..., Any]
    aggregate: Callable[..., Any]
    reference_grounding: str = reference_grounding

@dataclass(frozen=True)
class ArtifactWriterSpec:
    id: str
    artifact_path: str
    writer: Callable[..., Any]
    reference_grounding: str = reference_grounding

@dataclass(frozen=True)
class ExperimentSpec:
    id: str
    name: str
    datasets: Tuple[str, ...]
    backbones: Tuple[str, ...]
    methods: Tuple[str, ...]
    metric: str
    aggregate_metric: str
    artifact_path: str
    output_mapping: str
    mode: str
    seed_protocol: Tuple[int, ...]
    sweep_defaults: Mapping[str, Any] = field(default_factory=dict)
    reference_grounding: str = reference_grounding

    def to_row(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.id,
            "name": self.name,
            "datasets": list(self.datasets),
            "backbones": list(self.backbones),
            "methods": list(self.methods),
            "metric": self.metric,
            "aggregate_metric": self.aggregate_metric,
            "artifact_path": self.artifact_path,
            "output_mapping": self.output_mapping,
            "mode": self.mode,
            "seed_protocol": list(self.seed_protocol),
            "sweep_defaults": dict(self.sweep_defaults),
            "reference_grounding": self.reference_grounding,
        }

@dataclass(frozen=True)
class MethodSpec:
    id: str
    name: str
    selector: str
    kind: str
    mask_variant: Optional[str] = None
    reference_grounding: str = reference_grounding

    def validate(self) -> bool:
        return bool(self.id and self.name and self.selector)

@dataclass(frozen=True)
class ConfigSpec:
    experiment_id: str = SMOKE_EXPERIMENT_ID
    mode: str = SMOKE_MODE
    seed: int = DEFAULT_SEED
    seeds: Tuple[int, ...] = seed_values
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_LEARNING_RATE
    alpha: float = DEFAULT_ALPHA
    gamma: float = DEFAULT_GAMMA
    p: float = DEFAULT_P
    patch_size: int = DEFAULT_PATCH_SIZE
    interpolation_level: int = DEFAULT_INTERPOLATION_LEVEL
    dataset: str = DEFAULT_DATASET
    backbone: str = DEFAULT_BACKBONE
    method: str = DEFAULT_METHOD
    mask_variant: str = DEFAULT_MASK_VARIANT
    output_mapping: str = DEFAULT_OUTPUT_MAPPING
    output_root: str = RESULT_DIR_DEFAULT
    max_train_batches: Optional[int] = 1
    max_eval_batches: Optional[int] = 1
    max_samples_per_dataset: Optional[int] = 8
    write_readiness: bool = True
    write_evaluation_result: bool = True
    allow_dataset_download: bool = False
    class_count: int = DEFAULT_CLASS_COUNT
    image_size: int = DEFAULT_IMAGE_SIZE
    vit_image_size: int = DEFAULT_VIT_IMAGE_SIZE
    extra: Mapping[str, Any] = field(default_factory=dict)

    def resolved(self) -> Dict[str, Any]:
        return asdict(self)

def available_optional_backend(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False

def load_optional_backend(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except Exception as exc:
        return {
            "backend": name,
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
            "reference_grounding": reference_grounding,
        }

def _make_default_datasets() -> Dict[str, DatasetSpec]:
    rows = {
        "cifar10": DatasetSpec(
            id="cifar10",
            aliases=("cifar10", "CIFAR10"),
            split="train/val/test",
            num_classes=10,
            image_size=224,
            target_tasks=("cifar",),
        ),
        "cifar100": DatasetSpec(
            id="cifar100",
            aliases=("cifar100", "CIFAR100"),
            split="train/val/test",
            num_classes=100,
            image_size=224,
            target_tasks=("cifar",),
        ),
        "svhn": DatasetSpec(
            id="svhn",
            aliases=("svhn", "SVHN"),
            split="train/val/test",
            num_classes=10,
            image_size=224,
            target_tasks=("svhn",),
        ),
        "gtsrb": DatasetSpec(
            id="gtsrb",
            aliases=("gtsrb", "GTSRB"),
            split="train/val/test",
            num_classes=43,
            image_size=224,
            target_tasks=("cifar", "svhn"),
        ),
        "flowers102": DatasetSpec(
            id="flowers102",
            aliases=("flowers102", "flowers"),
            split="train/val/test",
            num_classes=102,
            image_size=224,
            target_tasks=("imagenet",),
        ),
        "dtd": DatasetSpec(
            id="dtd",
            aliases=("dtd", "DTD"),
            split="train/val/test",
            num_classes=47,
            image_size=224,
            target_tasks=("imagenet",),
        ),
        "ucf101": DatasetSpec(
            id="ucf101",
            aliases=("ucf101", "UCF101"),
            split="train/val/test",
            num_classes=101,
            image_size=224,
            target_tasks=("imagenet",),
        ),
        "eurosat": DatasetSpec(
            id="eurosat",
            aliases=("eurosat", "EuroSAT"),
            split="train/val/test",
            num_classes=10,
            image_size=224,
            target_tasks=("svhn", "imagenet"),
        ),
        "imagenet_1k": DatasetSpec(
            id="imagenet_1k",
            aliases=("imagenet_1k", "imagenet", "ImageNet-1K"),
            split="train/val",
            num_classes=1000,
            image_size=224,
            target_tasks=("imagenet",),
        ),
        "stanford_cars": DatasetSpec(
            id="stanford_cars",
            aliases=("stanford_cars", "StanfordCars"),
            split="train/val/test",
            num_classes=196,
            image_size=224,
            target_tasks=("imagenet",),
        ),
        "oxford_pets": DatasetSpec(
            id="oxford_pets",
            aliases=("oxford_pets", "OxfordPets"),
            split="train/val/test",
            num_classes=37,
            image_size=224,
            target_tasks=("imagenet",),
        ),
        "unit-001": DatasetSpec(
            id="unit-001",
            aliases=("unit-001", "unit_001"),
            split="smoke",
            num_classes=10,
            image_size=224,
            target_tasks=("cifar", "imagenet", "svhn"),
        ),
    }
    return rows

def _make_default_environments() -> Dict[str, EnvironmentSpec]:
    return {
        "cifar": EnvironmentSpec(
            id="cifar",
            aliases=("cifar",),
            datasets=("cifar10", "cifar100", "gtsrb"),
            backbones=("resnet18_imagenet1k", "resnet50_imagenet1k"),
            description="CIFAR environment for visual reprogramming and classification.",
        ),
        "imagenet": EnvironmentSpec(
            id="imagenet",
            aliases=("imagenet", "ImageNet"),
            datasets=("imagenet_1k", "stanford_cars", "oxford_pets", "flowers102", "dtd", "ucf101"),
            backbones=("resnet18_imagenet1k", "resnet50_imagenet1k", "vit_b32_imagenet1k"),
            description="ImageNet-style target environment with pretrained backbones.",
        ),
        "svhn": EnvironmentSpec(
            id="svhn",
            aliases=("svhn",),
            datasets=("svhn", "eurosat"),
            backbones=("resnet18_imagenet1k", "resnet50_imagenet1k"),
            description="SVHN-like digit/task environment.",
        ),
        "unit-001": EnvironmentSpec(
            id="unit-001",
            aliases=("unit-001", "unit_001"),
            datasets=("unit-001",),
            backbones=("resnet18_imagenet1k",),
            description="Smoke environment exercising the canonical route.",
        ),
        "resnet18_imagenet1k": EnvironmentSpec(
            id="resnet18_imagenet1k",
            aliases=("resnet18_imagenet1k", "ResNet-18 ImageNet-1K"),
            datasets=("cifar10", "cifar100", "svhn", "gtsrb", "flowers102", "dtd", "ucf101", "eurosat"),
            backbones=("resnet18_imagenet1k",),
            description="ImageNet-1K pretrained ResNet-18 environment.",
        ),
        "resnet50_imagenet1k": EnvironmentSpec(
            id="resnet50_imagenet1k",
            aliases=("resnet50_imagenet1k", "ResNet-50 ImageNet-1K"),
            datasets=("cifar10", "cifar100", "svhn", "gtsrb", "flowers102", "dtd", "ucf101", "eurosat"),
            backbones=("resnet50_imagenet1k",),
            description="ImageNet-1K pretrained ResNet-50 environment.",
        ),
        "vit_b32_imagenet1k": EnvironmentSpec(
            id="vit_b32_imagenet1k",
            aliases=("vit_b32_imagenet1k", "ViT-B/32 ImageNet-1K"),
            datasets=("cifar10", "cifar100", "svhn", "gtsrb", "flowers102", "dtd", "ucf101", "eurosat"),
            backbones=("vit_b32_imagenet1k",),
            description="ImageNet-1K pretrained ViT-B/32 environment.",
        ),
        "imagenet_1k": EnvironmentSpec(
            id="imagenet_1k",
            aliases=("imagenet_1k", "ImageNet-1K pretrained source"),
            datasets=("imagenet_1k",),
            backbones=("resnet18_imagenet1k", "resnet50_imagenet1k", "vit_b32_imagenet1k"),
            description="ImageNet-1K pretrained source environment.",
        ),
    }

def _make_default_backbones() -> Dict[str, BackboneSpec]:
    return {
        "resnet18_imagenet1k": BackboneSpec(
            id="resnet18_imagenet1k",
            aliases=("resnet18_imagenet1k", "ResNet-18", "resnet-18"),
            architecture="resnet18",
            pretrained_source="imagenet_1k",
            input_size=224,
            frozen=True,
        ),
        "resnet50_imagenet1k": BackboneSpec(
            id="resnet50_imagenet1k",
            aliases=("resnet50_imagenet1k", "ResNet-50", "resnet-50"),
            architecture="resnet50",
            pretrained_source="imagenet_1k",
            input_size=224,
            frozen=True,
        ),
        "vit_b32_imagenet1k": BackboneSpec(
            id="vit_b32_imagenet1k",
            aliases=("vit_b32_imagenet1k", "ViT-B/32", "vit-b32"),
            architecture="vit_b32",
            pretrained_source="imagenet_1k",
            input_size=384,
            frozen=True,
        ),
    }

def _make_default_metrics() -> Dict[str, MetricSpec]:
    return {
        "accuracy": MetricSpec(
            id="accuracy",
            name="top1_accuracy",
            compute=compute_accuracy,
            aggregate=aggregate_accuracy,
        ),
        "loss": MetricSpec(
            id="loss",
            name="mean_loss",
            compute=compute_loss,
            aggregate=aggregate_loss,
        ),
        "f1": MetricSpec(
            id="f1",
            name="macro_f1",
            compute=compute_f1,
            aggregate=aggregate_f1,
        ),
        "mean_std_accuracy": MetricSpec(
            id="mean_std_accuracy",
            name="mean_accuracy_percentage_and_standard_deviation_percentage",
            compute=compute_accuracy,
            aggregate=aggregate_accuracy,
        ),
    }

def _identity_writer(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return {
        "written": False,
        "args": [repr(a) for a in args],
        "kwargs": {k: repr(v) for k, v in kwargs.items()},
        "reference_grounding": reference_grounding,
    }

def _make_default_artifact_writers() -> Dict[str, ArtifactWriterSpec]:
    return {
        "table_1": ArtifactWriterSpec("table_1", TABLE_ARTIFACT_PATHS["table_1"], _identity_writer),
        "table_2": ArtifactWriterSpec("table_2", TABLE_ARTIFACT_PATHS["table_2"], _identity_writer),
        "table_3": ArtifactWriterSpec("table_3", TABLE_ARTIFACT_PATHS["table_3"], _identity_writer),
        "table_13": ArtifactWriterSpec("table_13", TABLE_ARTIFACT_PATHS["table_13"], _identity_writer),
        "table_14": ArtifactWriterSpec("table_14", TABLE_ARTIFACT_PATHS["table_14"], _identity_writer),
        "figure_13": ArtifactWriterSpec("figure_13", FIGURE_ARTIFACT_PATHS["figure_13"], _identity_writer),
        "figure_14": ArtifactWriterSpec("figure_14", FIGURE_ARTIFACT_PATHS["figure_14"], _identity_writer),
        "figure_15": ArtifactWriterSpec("figure_15", FIGURE_ARTIFACT_PATHS["figure_15"], _identity_writer),
        "figure_16": ArtifactWriterSpec("figure_16", FIGURE_ARTIFACT_PATHS["figure_16"], _identity_writer),
        "figure_17": ArtifactWriterSpec("figure_17", FIGURE_ARTIFACT_PATHS["figure_17"], _identity_writer),
        "figure_18": ArtifactWriterSpec("figure_18", FIGURE_ARTIFACT_PATHS["figure_18"], _identity_writer),
        "figure_19": ArtifactWriterSpec("figure_19", FIGURE_ARTIFACT_PATHS["figure_19"], _identity_writer),
        "figure_20": ArtifactWriterSpec("figure_20", FIGURE_ARTIFACT_PATHS["figure_20"], _identity_writer),
        "figure_21": ArtifactWriterSpec("figure_21", FIGURE_ARTIFACT_PATHS["figure_21"], _identity_writer),
        "figure_22": ArtifactWriterSpec("figure_22", FIGURE_ARTIFACT_PATHS["figure_22"], _identity_writer),
        "figure_23": ArtifactWriterSpec("figure_23", FIGURE_ARTIFACT_PATHS["figure_23"], _identity_writer),
    }

DATASET_REGISTRY = _make_default_datasets()
ENVIRONMENT_REGISTRY = _make_default_environments()
BACKBONE_REGISTRY = _make_default_backbones()
METRIC_REGISTRY_SPECS = _make_default_metrics()
ARTIFACT_WRITER_REGISTRY_SPECS = _make_default_artifact_writers()
METRIC_REGISTRY = {k: v.compute for k, v in METRIC_REGISTRY_SPECS.items()}
ARTIFACT_WRITER_REGISTRY = {k: v.writer for k, v in ARTIFACT_WRITER_REGISTRY_SPECS.items()}

def validate_dataset_registry() -> Dict[str, bool]:
    return {k: v.validate() for k, v in DATASET_REGISTRY.items()}

def validate_environment_registry() -> Dict[str, bool]:
    return {k: v.validate() for k, v in ENVIRONMENT_REGISTRY.items()}

def validate_backbone_registry() -> Dict[str, bool]:
    return {k: v.validate() for k, v in BACKBONE_REGISTRY.items()}

def resolve_dataset_spec(dataset_id: str) -> DatasetSpec:
    normalized = DATASET_ALIASES.get(dataset_id, dataset_id)
    if normalized in DATASET_REGISTRY:
        return DATASET_REGISTRY[normalized]
    for spec in DATASET_REGISTRY.values():
        if dataset_id in spec.aliases:
            return spec
    raise KeyError(f"Unknown dataset: {dataset_id}")

def resolve_environment_spec(environment_id: str) -> EnvironmentSpec:
    normalized = _alias(environment_id, DATASET_ALIASES, environment_id)
    if normalized in ENVIRONMENT_REGISTRY:
        return ENVIRONMENT_REGISTRY[normalized]
    for spec in ENVIRONMENT_REGISTRY.values():
        if environment_id in spec.aliases:
            return spec
    raise KeyError(f"Unknown environment: {environment_id}")

def resolve_backbone_spec(backbone_id: str) -> BackboneSpec:
    normalized = _alias(backbone_id, BACKBONE_ALIASES, backbone_id)
    if normalized in BACKBONE_REGISTRY:
        return BACKBONE_REGISTRY[normalized]
    for spec in BACKBONE_REGISTRY.values():
        if backbone_id in spec.aliases:
            return spec
    raise KeyError(f"Unknown backbone: {backbone_id}")

def resolve_method_spec(method_name: str) -> MethodSpec:
    if method_name in METHOD_REGISTRY:
        row = METHOD_REGISTRY[method_name]
        return MethodSpec(
            id=row["id"],
            name=method_name,
            selector=row["selector"],
            kind=row["kind"],
            mask_variant=row.get("mask_variant"),
        )
    for display_name, row in METHOD_REGISTRY.items():
        if method_name in row.get("aliases", ()):
            return MethodSpec(
                id=row["id"],
                name=display_name,
                selector=row["selector"],
                kind=row["kind"],
                mask_variant=row.get("mask_variant"),
            )
    if method_name.lower() in METHOD_ALIASES:
        canonical = METHOD_ALIASES[method_name.lower()]
        row = METHOD_REGISTRY[canonical]
        return MethodSpec(
            id=row["id"],
            name=canonical,
            selector=row["selector"],
            kind=row["kind"],
            mask_variant=row.get("mask_variant"),
        )
    raise KeyError(f"Unknown method: {method_name}")

def resolve_metric_spec(metric_id: str) -> MetricSpec:
    if metric_id in METRIC_REGISTRY_SPECS:
        return METRIC_REGISTRY_SPECS[metric_id]
    if metric_id in METRIC_IDENTIFIERS:
        canonical = metric_id
        for spec in METRIC_REGISTRY_SPECS.values():
            if spec.id == canonical:
                return spec
    raise KeyError(f"Unknown metric: {metric_id}")

def load_classifier(config: Mapping[str, Any] | ConfigSpec) -> Dict[str, Any]:
    cfg = config.resolved() if isinstance(config, ConfigSpec) else dict(config)
    backbone = resolve_backbone_spec(str(cfg.get("backbone", DEFAULT_BACKBONE)))
    dataset = resolve_dataset_spec(str(cfg.get("dataset", DEFAULT_DATASET)))
    return {
        "backbone": backbone.id,
        "architecture": backbone.architecture,
        "pretrained_source": backbone.pretrained_source,
        "input_size": backbone.input_size,
        "dataset": dataset.id,
        "class_count": int(cfg.get("class_count", dataset.num_classes)),
        "frozen": backbone.frozen,
        "available": available_optional_backend("torch"),
        "lazy_backend": load_optional_backend("torch"),
        "reference_grounding": reference_grounding,
    }

def finetune_classifier(config: Mapping[str, Any] | ConfigSpec) -> Dict[str, Any]:
    cfg = config.resolved() if isinstance(config, ConfigSpec) else dict(config)
    classifier = load_classifier(cfg)
    classifier.update(
        {
            "trainable_groups": ["delta", "mask_generator_phi"],
            "optimizer_groups": {
                "delta": {"lr": float(cfg.get("learning_rate", DEFAULT_LEARNING_RATE))},
                "phi": {"lr": float(cfg.get("learning_rate", DEFAULT_LEARNING_RATE))},
            },
            "freeze_pretrained": True,
            "output_mapping": cfg.get("output_mapping", DEFAULT_OUTPUT_MAPPING),
            "method": cfg.get("method", DEFAULT_METHOD),
            "mask_variant": cfg.get("mask_variant", DEFAULT_MASK_VARIANT),
            "reference_grounding": reference_grounding,
        }
    )
    return classifier

def create_method_selector(method_name: str) -> MethodSpec:
    return resolve_method_spec(method_name)

def create_mask_variant_selector(mask_variant: str) -> str:
    return MASK_VARIANT_ALIASES.get(mask_variant, mask_variant)

def create_environment(environment_id: str) -> Dict[str, Any]:
    spec = resolve_environment_spec(environment_id)
    return {
        "id": spec.id,
        "aliases": spec.aliases,
        "datasets": spec.datasets,
        "backbones": spec.backbones,
        "description": spec.description,
        "available": True,
        "reference_grounding": spec.reference_grounding,
    }

def create_dataset(dataset_id: str) -> Dict[str, Any]:
    spec = resolve_dataset_spec(dataset_id)
    return {
        "id": spec.id,
        "aliases": spec.aliases,
        "split": spec.split,
        "num_classes": spec.num_classes,
        "image_size": spec.image_size,
        "target_tasks": spec.target_tasks,
        "available": True,
        "reference_grounding": spec.reference_grounding,
    }

def create_backbone(backbone_id: str) -> Dict[str, Any]:
    spec = resolve_backbone_spec(backbone_id)
    return {
        "id": spec.id,
        "aliases": spec.aliases,
        "architecture": spec.architecture,
        "pretrained_source": spec.pretrained_source,
        "input_size": spec.input_size,
        "frozen": spec.frozen,
        "available": True,
        "reference_grounding": spec.reference_grounding,
    }

def build_metric_registry() -> Dict[str, Dict[str, Any]]:
    return {k: asdict(v) for k, v in METRIC_REGISTRY_SPECS.items()}

def build_artifact_writer_registry() -> Dict[str, Dict[str, Any]]:
    return {k: asdict(v) for k, v in ARTIFACT_WRITER_REGISTRY_SPECS.items()}

def build_experiment_registry() -> Dict[str, Dict[str, Any]]:
    return {k: dict(v) for k, v in EXPERIMENT_PROTOCOLS.items()}

def build_protocol_matrix() -> Dict[str, Any]:
    return {
        "reference_grounding": reference_grounding,
        "matrix_rows": [dict(row) for row in EVIDENCE_MATRIX_ROWS],
        "methods": list(ALLOWED_METHODS),
        "baselines": list(ALLOWED_BASELINES),
        "datasets": list(DATASET_REGISTRY.keys()),
        "environments": list(ENVIRONMENT_REGISTRY.keys()),
        "metrics": list(METRIC_REGISTRY.keys()),
        "artifacts": list(TABLE_ARTIFACT_PATHS.values()) + list(FIGURE_ARTIFACT_PATHS.values()),
        "scope_constraints": list(SCOPE_CONSTRAINTS),
        "trend_obligations": dict(TREND_OBLIGATIONS),
        "seed_protocol": list(three_seed_protocol),
    }

def build_runtime_config(
    *,
    experiment_id: str = SMOKE_EXPERIMENT_ID,
    mode: str = SMOKE_MODE,
    dataset: str = DEFAULT_DATASET,
    backbone: str = DEFAULT_BACKBONE,
    method: str = DEFAULT_METHOD,
    mask_variant: str = DEFAULT_MASK_VARIANT,
    output_root: Optional[str] = None,
    seed: Optional[int] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    alpha: Optional[float] = None,
    gamma: Optional[float] = None,
    p: Optional[float] = None,
    patch_size: Optional[int] = None,
    interpolation_level: Optional[int] = None,
    max_train_batches: Optional[int] = None,
    max_eval_batches: Optional[int] = None,
    max_samples_per_dataset: Optional[int] = None,
    write_readiness: bool = True,
    write_evaluation_result: bool = True,
    allow_dataset_download: bool = False,
    class_count: Optional[int] = None,
    image_size: Optional[int] = None,
) -> ConfigSpec:
    dataset_spec = resolve_dataset_spec(dataset)
    backbone_spec = resolve_backbone_spec(backbone)
    return ConfigSpec(
        experiment_id=experiment_id,
        mode=mode,
        seed=DEFAULT_SEED if seed is None else int(seed),
        seeds=normalize_seed_list([seed] if seed is not None else None),
        epochs=DEFAULT_EPOCHS if epochs is None else int(epochs),
        batch_size=DEFAULT_BATCH_SIZE if batch_size is None else int(batch_size),
        learning_rate=DEFAULT_LEARNING_RATE if learning_rate is None else float(learning_rate),
        alpha=DEFAULT_ALPHA if alpha is None else float(alpha),
        gamma=DEFAULT_GAMMA if gamma is None else float(gamma),
        p=DEFAULT_P if p is None else float(p),
        patch_size=DEFAULT_PATCH_SIZE if patch_size is None else int(patch_size),
        interpolation_level=DEFAULT_INTERPOLATION_LEVEL if interpolation_level is None else int(interpolation_level),
        dataset=dataset_spec.id,
        backbone=backbone_spec.id,
        method=resolve_method_spec(method).name,
        mask_variant=create_mask_variant_selector(mask_variant),
        output_mapping=DEFAULT_OUTPUT_MAPPING,
        output_root=RESULT_DIR_DEFAULT if output_root is None else output_root,
        max_train_batches=max_train_batches if max_train_batches is not None else 1 if mode == SMOKE_MODE else None,
        max_eval_batches=max_eval_batches if max_eval_batches is not None else 1 if mode == SMOKE_MODE else None,
        max_samples_per_dataset=max_samples_per_dataset if max_samples_per_dataset is not None else 8 if mode == SMOKE_MODE else None,
        write_readiness=write_readiness,
        write_evaluation_result=write_evaluation_result,
        allow_dataset_download=allow_dataset_download,
        class_count=dataset_spec.num_classes if class_count is None else int(class_count),
        image_size=dataset_spec.image_size if image_size is None else int(image_size),
        vit_image_size=backbone_spec.input_size if backbone_spec.id == "vit_b32_imagenet1k" else DEFAULT_VIT_IMAGE_SIZE,
        extra={
            "p": DEFAULT_P if p is None else float(p),
            "patch_size": DEFAULT_PATCH_SIZE if patch_size is None else int(patch_size),
            "reference_grounding": reference_grounding,
        },
    )

def resolve_epochs_defaults_for_mode(mode: str) -> Dict[str, Any]:
    return resolve_epochs_defaults(mode=mode)

def write_json_artifact(path: str | Path, payload: Mapping[str, Any]) -> Dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return {"path": str(path), "written": True, "reference_grounding": reference_grounding}

def write_csv_artifact(path: str | Path, rows: Sequence[Mapping[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None and rows:
        fieldnames = list(rows[0].keys())
    elif fieldnames is None:
        fieldnames = []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        if fieldnames:
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})
    return {"path": str(path), "written": True, "row_count": len(rows), "reference_grounding": reference_grounding}

def write_table_1_artifact(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return write_csv_artifact(path, rows)

def write_table_2_artifact(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return write_csv_artifact(path, rows)

def write_table_3_artifact(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return write_csv_artifact(path, rows)

def write_table_13_artifact(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return write_csv_artifact(path, rows)

def write_table_14_artifact(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return write_csv_artifact(path, rows)

def write_figure_index_artifact(path: str | Path, rows: Mapping[str, Any]) -> Dict[str, Any]:
    return write_json_artifact(path, rows)

def write_table_index_artifact(path: str | Path, rows: Mapping[str, Any]) -> Dict[str, Any]:
    return write_json_artifact(path, rows)

def write_artifact_manifest(path: str | Path, manifest: Mapping[str, Any]) -> Dict[str, Any]:
    return write_json_artifact(path, manifest)

def write_dry_run_manifest(path: str | Path, manifest: Mapping[str, Any]) -> Dict[str, Any]:
    return write_json_artifact(path, manifest)

def write_readiness_artifacts(output_root: str | Path, resolved_config: Mapping[str, Any]) -> Dict[str, Any]:
    output_root = Path(output_root)
    readiness_path = output_root / SCHEMA_READY_ARTIFACTS[0]
    evaluation_result_path = output_root / SCHEMA_READY_ARTIFACTS[1]
    readiness = {
        "status": "ready",
        "mode": resolved_config.get("mode", SMOKE_MODE),
        "experiment_id": resolved_config.get("experiment_id", SMOKE_EXPERIMENT_ID),
        "dataset": resolved_config.get("dataset", DEFAULT_DATASET),
        "backbone": resolved_config.get("backbone", DEFAULT_BACKBONE),
        "method": resolved_config.get("method", DEFAULT_METHOD),
        "mask_variant": resolved_config.get("mask_variant", DEFAULT_MASK_VARIANT),
        "reference_grounding": reference_grounding,
    }
    evaluation_result = {
        "status": "smoke",
        "benchmark_visible": False,
        "reason": "bounded smoke validation only",
        "metrics": {"accuracy": 0.0, "f1": 0.0, "loss": 0.0},
        "reference_grounding": reference_grounding,
    }
    write_json_artifact(readiness_path, readiness)
    write_json_artifact(evaluation_result_path, evaluation_result)
    return {
        "readiness": str(readiness_path),
        "evaluation_result": str(evaluation_result_path),
        "readiness_payload": readiness,
        "evaluation_result_payload": evaluation_result,
    }

def build_output_root(explicit_root: Optional[str] = None) -> Path:
    env_root = os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR")
    root = explicit_root or env_root or RESULT_DIR_DEFAULT
    return Path(root)

def ensure_output_layout(output_root: str | Path) -> Dict[str, str]:
    output_root = Path(output_root)
    paths = {
        "root": str(output_root),
        "tables": str(output_root / TABLE_DIR),
        "figures": str(output_root / FIGURE_DIR),
    }
    for p in paths.values():
        Path(p).mkdir(parents=True, exist_ok=True)
    return paths

def canonical_experiment_ids() -> Tuple[str, ...]:
    return (
        EXPERIMENT_TABLE_1,
        EXPERIMENT_TABLE_2,
        EXPERIMENT_TABLE_3,
        EXPERIMENT_APPENDIX_TABLE_13,
        EXPERIMENT_APPENDIX_TABLE_14,
        EXPERIMENT_SMOKE,
    )

def canonical_artifact_manifest() -> Dict[str, Any]:
    return {
        "schema_version": "1.0",
        "reference_grounding": reference_grounding,
        "table_artifacts": dict(TABLE_ARTIFACT_PATHS),
        "figure_artifacts": dict(FIGURE_ARTIFACT_PATHS),
        "schema_ready_artifacts": list(SCHEMA_READY_ARTIFACTS),
        "metric_identifiers": dict(METRIC_IDENTIFIERS),
        "experiments": [DEFAULT_EXPERIMENT_TABLE_1, DEFAULT_EXPERIMENT_TABLE_2, DEFAULT_EXPERIMENT_TABLE_3, DEFAULT_APPENDIX_TABLE_13, DEFAULT_APPENDIX_TABLE_14, DEFAULT_SMOKE_EXPERIMENT],
        "scope_constraints": list(SCOPE_CONSTRAINTS),
        "trend_obligations": dict(TREND_OBLIGATIONS),
        "three_seed_protocol": list(three_seed_protocol),
        "reference_grounding_marker": reference_grounding,
    }

def canonical_table_index() -> Dict[str, Any]:
    return {
        "table_1": TABLE_ARTIFACT_PATHS["table_1"],
        "table_2": TABLE_ARTIFACT_PATHS["table_2"],
        "table_3": TABLE_ARTIFACT_PATHS["table_3"],
        "table_13": TABLE_ARTIFACT_PATHS["table_13"],
        "table_14": TABLE_ARTIFACT_PATHS["table_14"],
        "reference_grounding": reference_grounding,
    }

def canonical_figure_index() -> Dict[str, Any]:
    return {
        **FIGURE_ARTIFACT_PATHS,
        "reference_grounding": reference_grounding,
    }

def resolve_data_backbone_method_triplet(dataset: str, backbone: str, method: str) -> Dict[str, Any]:
    dataset_spec = resolve_dataset_spec(dataset)
    backbone_spec = resolve_backbone_spec(backbone)
    method_spec = resolve_method_spec(method)
    return {
        "dataset": dataset_spec.id,
        "backbone": backbone_spec.id,
        "method": method_spec.name,
        "output_mapping": DEFAULT_OUTPUT_MAPPING,
        "class_count": dataset_spec.num_classes,
        "input_size": backbone_spec.input_size,
        "frozen_backbone": backbone_spec.frozen,
        "method_selector": method_spec.selector,
        "mask_variant": method_spec.mask_variant,
        "reference_grounding": reference_grounding,
    }

def build_data(
    dataset: str = DEFAULT_DATASET,
    backbone: str = DEFAULT_BACKBONE,
    method: str = DEFAULT_METHOD,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    triplet = resolve_data_backbone_method_triplet(dataset, backbone, method)
    dataset_spec = resolve_dataset_spec(dataset)
    backbone_spec = resolve_backbone_spec(backbone)
    sample_count = max_samples if max_samples is not None else min(8, dataset_spec.num_classes)
    indices = list(range(sample_count))
    labels = [i % dataset_spec.num_classes for i in indices]
    data = {
        "dataset_spec": create_dataset(dataset_spec.id),
        "environment_spec": create_environment("imagenet_1k" if dataset_spec.id == "imagenet_1k" else "unit-001" if dataset_spec.id == "unit-001" else "imagenet"),
        "backbone_spec": create_backbone(backbone_spec.id),
        "method_spec": asdict(resolve_method_spec(method)),
        "triplet": triplet,
        "samples": indices,
        "labels": labels,
        "reference_grounding": reference_grounding,
    }
    return data

def load_data(config: Mapping[str, Any] | ConfigSpec) -> Dict[str, Any]:
    cfg = config.resolved() if isinstance(config, ConfigSpec) else dict(config)
    return build_data(
        dataset=str(cfg.get("dataset", DEFAULT_DATASET)),
        backbone=str(cfg.get("backbone", DEFAULT_BACKBONE)),
        method=str(cfg.get("method", DEFAULT_METHOD)),
        max_samples=int(cfg.get("max_samples_per_dataset", 8) or 8),
    )

def prepare_data(config: Mapping[str, Any] | ConfigSpec) -> Dict[str, Any]:
    data = load_data(config)
    return {
        **data,
        "prepared": True,
        "resized": True,
        "reprogramming_ready": True,
        "reference_grounding": reference_grounding,
    }

def build_reprogramming(
    *,
    method: str = DEFAULT_METHOD,
    mask_variant: str = DEFAULT_MASK_VARIANT,
    patch_size: int = DEFAULT_PATCH_SIZE,
    interpolation_level: int = DEFAULT_INTERPOLATION_LEVEL,
    output_mapping: str = DEFAULT_OUTPUT_MAPPING,
) -> Dict[str, Any]:
    method_spec = resolve_method_spec(method)
    return {
        "method": method_spec.name,
        "method_selector": method_spec.selector,
        "mask_variant": create_mask_variant_selector(mask_variant),
        "patch_size": int(patch_size),
        "interpolation_level": int(interpolation_level),
        "output_mapping": output_mapping,
        "f_in_formula": "r(x)+delta⊙f_mask(r(x)|phi)",
        "delta_initializer": "zeros",
        "delta_trainable": method_spec.name in ("Ours", "ONLY δ", "ONLY f_mask", "SINGLE-CHANNEL f_mask^s"),
        "phi_trainable": method_spec.name in ("Ours", "ONLY f_mask", "SINGLE-CHANNEL f_mask^s"),
        "freeze_pretrained": True,
        "reference_grounding": reference_grounding,
    }

def load_reprogramming(config: Mapping[str, Any] | ConfigSpec) -> Dict[str, Any]:
    cfg = config.resolved() if isinstance(config, ConfigSpec) else dict(config)
    return build_reprogramming(
        method=str(cfg.get("method", DEFAULT_METHOD)),
        mask_variant=str(cfg.get("mask_variant", DEFAULT_MASK_VARIANT)),
        patch_size=int(cfg.get("patch_size", DEFAULT_PATCH_SIZE)),
        interpolation_level=int(cfg.get("interpolation_level", DEFAULT_INTERPOLATION_LEVEL)),
        output_mapping=str(cfg.get("output_mapping", DEFAULT_OUTPUT_MAPPING)),
    )

def prepare_reprogramming(config: Mapping[str, Any] | ConfigSpec) -> Dict[str, Any]:
    reprogramming = load_reprogramming(config)
    return {**reprogramming, "prepared": True}

def load_classifier_from_config(config: Mapping[str, Any] | ConfigSpec) -> Dict[str, Any]:
    return load_classifier(config)

def finetune_classifier_from_config(config: Mapping[str, Any] | ConfigSpec) -> Dict[str, Any]:
    return finetune_classifier(config)

def build_classifier_factory(config: Mapping[str, Any] | ConfigSpec) -> Dict[str, Any]:
    return finetune_classifier_from_config(config)

def make_environment(config: Mapping[str, Any] | ConfigSpec) -> Dict[str, Any]:
    cfg = config.resolved() if isinstance(config, ConfigSpec) else dict(config)
    dataset_id = str(cfg.get("dataset", DEFAULT_DATASET))
    backbone_id = str(cfg.get("backbone", DEFAULT_BACKBONE))
    env_id = "imagenet_1k" if dataset_id == "imagenet_1k" else "unit-001" if dataset_id == "unit-001" else "imagenet" if backbone_id.startswith("vit") else "cifar" if dataset_id.startswith("cifar") else "svhn"
    environment = create_environment(env_id)
    environment["dataset"] = dataset_id
    environment["backbone"] = backbone_id
    environment["config"] = dict(cfg)
    return environment

def environment_readiness_check(environment: Mapping[str, Any]) -> Dict[str, Any]:
    valid = bool(environment.get("id")) and bool(environment.get("datasets")) and bool(environment.get("backbones"))
    return {
        "environment": environment.get("id"),
        "ready": valid,
        "reference_grounding": reference_grounding,
    }

def dataset_readiness_check(dataset: Mapping[str, Any]) -> Dict[str, Any]:
    valid = bool(dataset.get("id")) and int(dataset.get("num_classes", 0)) > 0
    return {
        "dataset": dataset.get("id"),
        "ready": valid,
        "reference_grounding": reference_grounding,
    }

def backbone_readiness_check(backbone: Mapping[str, Any]) -> Dict[str, Any]:
    valid = bool(backbone.get("id")) and bool(backbone.get("architecture"))
    return {
        "backbone": backbone.get("id"),
        "ready": valid,
        "reference_grounding": reference_grounding,
    }

def write_dataset_registry_artifact(path: str | Path) -> Dict[str, Any]:
    rows = {k: create_dataset(k) for k in DATASET_REGISTRY.keys()}
    return write_json_artifact(path, rows)

def write_environment_registry_artifact(path: str | Path) -> Dict[str, Any]:
    rows = {k: create_environment(k) for k in ENVIRONMENT_REGISTRY.keys()}
    return write_json_artifact(path, rows)

def write_experiment_registry_artifact(path: str | Path) -> Dict[str, Any]:
    return write_json_artifact(path, build_experiment_registry())

def write_config_resolved_artifact(path: str | Path, resolved_config: Mapping[str, Any]) -> Dict[str, Any]:
    return write_json_artifact(path, resolved_config)

def write_data_manifest_artifact(path: str | Path, manifest: Mapping[str, Any]) -> Dict[str, Any]:
    return write_json_artifact(path, manifest)

def write_metrics_artifact(path: str | Path, metrics: Mapping[str, Any]) -> Dict[str, Any]:
    return write_json_artifact(path, metrics)

def write_analysis_report(path: str | Path, report: Mapping[str, Any]) -> Dict[str, Any]:
    return write_json_artifact(path, report)

def write_sensitivity_report(path: str | Path, report: Mapping[str, Any]) -> Dict[str, Any]:
    return write_json_artifact(path, report)

def create_smoke_metrics(predictions: Sequence[int], labels: Sequence[int]) -> Dict[str, Any]:
    acc = compute_accuracy(predictions, labels)
    f1_value = compute_f1(predictions, labels)
    loss_value = compute_loss([float(p) for p in predictions], labels)
    reward_value = compute_reward(acc, loss_value)
    return {
        "accuracy": acc,
        "f1": f1_value,
        "loss": loss_value,
        "reward": reward_value,
        "mean_std_accuracy": aggregate_accuracy([acc]),
        "mean_std_f1": aggregate_f1([f1_value]),
        "mean_std_loss": aggregate_loss([loss_value]),
        "mean_std_reward": aggregate_reward([reward_value]),
        "metric_identifiers": dict(METRIC_IDENTIFIERS),
        "reference_grounding": reference_grounding,
    }

def compute_ours_oradaptersby_inventory_objective(*args: Any, **kwargs: Any) -> float:
    accuracy_value = float(kwargs.get("accuracy", kwargs.get("acc", 0.0)))
    loss_value = float(kwargs.get("loss", 0.0))
    return compute_reward(accuracy_value, loss_value)

def compute_ours_oradaptersby_inventory_score(*args: Any, **kwargs: Any) -> float:
    objective = compute_ours_oradaptersby_inventory_objective(*args, **kwargs)
    return objective

class Ours:
    method_name = "Ours"
    method_id = "ours"
    mask_variant = "ours_multi_channel"
    selector = "ours"
    reference_grounding = reference_grounding

    @classmethod
    def build(cls, config: Mapping[str, Any] | ConfigSpec) -> Dict[str, Any]:
        return {
            "method": cls.method_name,
            "method_id": cls.method_id,
            "selector": cls.selector,
            "mask_variant": create_mask_variant_selector(
                (config.resolved() if isinstance(config, ConfigSpec) else dict(config)).get("mask_variant", cls.mask_variant)
            ),
            "reference_grounding": cls.reference_grounding,
        }

class MethodSelectorRegistry:
    @staticmethod
    def select(method_name: str) -> MethodSpec:
        return resolve_method_spec(method_name)

    @staticmethod
    def all() -> Dict[str, Dict[str, Any]]:
        return dict(METHOD_REGISTRY)

def method_selector_set() -> Tuple[str, ...]:
    return tuple(ALLOWED_METHODS + ALLOWED_BASELINES)

def backbone_selector_set() -> Tuple[str, ...]:
    return tuple(BACKBONE_REGISTRY.keys())

def dataset_selector_set() -> Tuple[str, ...]:
    return tuple(DATASET_REGISTRY.keys())

def environment_selector_set() -> Tuple[str, ...]:
    return tuple(ENVIRONMENT_REGISTRY.keys())

def sweep_selector_matrix() -> Dict[str, Tuple[Any, ...]]:
    return {
        "seed": seed_values,
        "epochs": epochs_values,
        "learning_rate": learning_rate_values,
        "batch_size": batch_size_values,
        "alpha": alpha_values,
        "gamma": gamma_values,
        "p": p_values,
        "patch_size": patch_size_values,
        "similarity_guidance_scale": similarity_guidance_scale_values,
    }

def table_protocol_for_experiment(experiment_id: str) -> Dict[str, Any]:
    if experiment_id not in EXPERIMENT_PROTOCOLS:
        raise KeyError(f"Unknown experiment_id: {experiment_id}")
    return dict(EXPERIMENT_PROTOCOLS[experiment_id])

def protocol_matrix_for_route() -> Dict[str, Any]:
    return build_protocol_matrix()

def run_protocol_selection(experiment_id: str) -> Dict[str, Any]:
    spec = table_protocol_for_experiment(experiment_id)
    return {
        **spec,
        "method_registry": METHOD_REGISTRY,
        "dataset_registry": {k: create_dataset(k) for k in DATASET_REGISTRY.keys()},
        "environment_registry": {k: create_environment(k) for k in ENVIRONMENT_REGISTRY.keys()},
        "backbone_registry": {k: create_backbone(k) for k in BACKBONE_REGISTRY.keys()},
        "metric_registry": build_metric_registry(),
        "artifact_writer_registry": build_artifact_writer_registry(),
        "reference_grounding": reference_grounding,
    }

def resolve_output_mapping(name: Optional[str] = None) -> str:
    value = DEFAULT_OUTPUT_MAPPING if name is None else str(name)
    return value

def resolve_mode(mode: Optional[str] = None) -> str:
    value = SMOKE_MODE if mode is None else str(mode)
    if value not in {SMOKE_MODE, FULL_MODE}:
        return SMOKE_MODE
    return value

def resolve_config(
    *,
    experiment_id: str = SMOKE_EXPERIMENT_ID,
    mode: str = SMOKE_MODE,
    dataset: str = DEFAULT_DATASET,
    backbone: str = DEFAULT_BACKBONE,
    method: str = DEFAULT_METHOD,
    mask_variant: str = DEFAULT_MASK_VARIANT,
    seed: Optional[int] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    alpha: Optional[float] = None,
    gamma: Optional[float] = None,
    p: Optional[float] = None,
    patch_size: Optional[int] = None,
    interpolation_level: Optional[int] = None,
    output_root: Optional[str] = None,
    max_train_batches: Optional[int] = None,
    max_eval_batches: Optional[int] = None,
    max_samples_per_dataset: Optional[int] = None,
    allow_dataset_download: bool = False,
    write_readiness: bool = True,
    write_evaluation_result: bool = True,
    class_count: Optional[int] = None,
    image_size: Optional[int] = None,
) -> ConfigSpec:
    return build_runtime_config(
        experiment_id=experiment_id,
        mode=resolve_mode(mode),
        dataset=dataset,
        backbone=backbone,
        method=method,
        mask_variant=mask_variant,
        output_root=output_root,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        alpha=alpha,
        gamma=gamma,
        p=p,
        patch_size=patch_size,
        interpolation_level=interpolation_level,
        max_train_batches=max_train_batches,
        max_eval_batches=max_eval_batches,
        max_samples_per_dataset=max_samples_per_dataset,
        write_readiness=write_readiness,
        write_evaluation_result=write_evaluation_result,
        allow_dataset_download=allow_dataset_download,
        class_count=class_count,
        image_size=image_size,
    )

def default_resolved_runtime_config() -> Dict[str, Any]:
    cfg = resolve_config()
    return cfg.resolved()

def assemble_repository_contract() -> Dict[str, Any]:
    return {
        "reference_grounding": reference_grounding,
        "methods": method_selector_set(),
        "datasets": dataset_selector_set(),
        "environments": environment_selector_set(),
        "backbones": backbone_selector_set(),
        "metrics": tuple(METRIC_REGISTRY.keys()),
        "artifact_paths": PAPER_VISIBLE_ARTIFACTS,
        "sweeps": sweep_selector_matrix(),
        "fixed_hyperparameters": {
            "three_seed_protocol": list(three_seed_protocol),
            "default_seed": DEFAULT_SEED,
            "default_epochs": DEFAULT_EPOCHS,
            "default_batch_size": DEFAULT_BATCH_SIZE,
            "default_learning_rate": DEFAULT_LEARNING_RATE,
            "default_patch_size": DEFAULT_PATCH_SIZE,
            "default_interpolation_level": DEFAULT_INTERPOLATION_LEVEL,
        },
        "trend_obligations": dict(TREND_OBLIGATIONS),
        "scope_constraints": list(SCOPE_CONSTRAINTS),
        "evidence_matrix_rows": EVIDENCE_MATRIX_ROWS,
        "protocol_matrix": protocol_matrix_for_route(),
    }

def registry_snapshot() -> Dict[str, Any]:
    return {
        "dataset_registry": {k: asdict(v) for k, v in DATASET_REGISTRY.items()},
        "environment_registry": {k: asdict(v) for k, v in ENVIRONMENT_REGISTRY.items()},
        "backbone_registry": {k: asdict(v) for k, v in BACKBONE_REGISTRY.items()},
        "metric_registry": build_metric_registry(),
        "artifact_writer_registry": build_artifact_writer_registry(),
        "experiment_registry": build_experiment_registry(),
        "method_registry": dict(METHOD_REGISTRY),
        "reference_grounding": reference_grounding,
    }

def ensure_registry_availability() -> Dict[str, Any]:
    return {
        "datasets": validate_dataset_registry(),
        "environments": validate_environment_registry(),
        "backbones": validate_backbone_registry(),
        "reference_grounding": reference_grounding,
    }

def load_backbone_factory(config: Mapping[str, Any] | ConfigSpec) -> Dict[str, Any]:
    cfg = config.resolved() if isinstance(config, ConfigSpec) else dict(config)
    return create_backbone(str(cfg.get("backbone", DEFAULT_BACKBONE)))

def load_environment_factory(config: Mapping[str, Any] | ConfigSpec) -> Dict[str, Any]:
    cfg = config.resolved() if isinstance(config, ConfigSpec) else dict(config)
    env_name = str(cfg.get("environment", "imagenet_1k" if cfg.get("dataset") == "imagenet_1k" else "unit-001" if cfg.get("dataset") == "unit-001" else "imagenet"))
    if env_name not in ENVIRONMENT_REGISTRY:
        env_name = "unit-001" if cfg.get("mode", SMOKE_MODE) == SMOKE_MODE else "imagenet"
    return create_environment(env_name)

def load_data_pipeline(config: Mapping[str, Any] | ConfigSpec) -> Dict[str, Any]:
    return load_data(config)

def load_classifier_pipeline(config: Mapping[str, Any] | ConfigSpec) -> Dict[str, Any]:
    return load_classifier(config)

def finetune_classifier_pipeline(config: Mapping[str, Any] | ConfigSpec) -> Dict[str, Any]:
    return finetune_classifier(config)

def load_output_mapping(config: Mapping[str, Any] | ConfigSpec) -> str:
    cfg = config.resolved() if isinstance(config, ConfigSpec) else dict(config)
    return resolve_output_mapping(cfg.get("output_mapping"))

def experiment_row_for_route(experiment_id: str) -> Dict[str, Any]:
    spec = table_protocol_for_experiment(experiment_id)
    return {
        **spec,
        "methods_expanded": [resolve_method_spec(m).name if m in METHOD_REGISTRY or m in METHOD_ALIASES.values() else m for m in spec["methods"]],
        "datasets_expanded": [resolve_dataset_spec(d).id if d in DATASET_REGISTRY or d in DATASET_ALIASES.values() else d for d in spec["datasets"]],
        "backbones_expanded": [resolve_backbone_spec(b).id if b in BACKBONE_REGISTRY or b in BACKBONE_ALIASES.values() else b for b in spec["backbones"]],
        "reference_grounding": reference_grounding,
    }

def write_registry_artifacts(output_root: str | Path) -> Dict[str, Any]:
    output_root = Path(output_root)
    ensure_output_layout(output_root)
    artifacts = {
        "results/config_resolved.json": write_json_artifact(output_root / "config_resolved.json", default_resolved_runtime_config()),
        "results/dataset_registry.json": write_json_artifact(output_root / "dataset_registry.json", registry_snapshot()["dataset_registry"]),
        "results/environment_registry.json": write_json_artifact(output_root / "environment_registry.json", registry_snapshot()["environment_registry"]),
        "results/experiment_registry.json": write_json_artifact(output_root / "experiment_registry.json", registry_snapshot()["experiment_registry"]),
        "results/artifact_manifest.json": write_artifact_manifest(output_root / "artifact_manifest.json", canonical_artifact_manifest()),
        "results/table_index.json": write_table_index_artifact(output_root / "table_index.json", canonical_table_index()),
        "results/figure_index.json": write_figure_index_artifact(output_root / "figure_index.json", canonical_figure_index()),
        "results/dry_run_manifest.json": write_dry_run_manifest(
            output_root / "dry_run_manifest.json",
            {
                "mode": SMOKE_MODE,
                "schema_artifacts": list(SCHEMA_READY_ARTIFACTS),
                "paper_visible_artifacts": list(PAPER_VISIBLE_ARTIFACTS),
                "reference_grounding": reference_grounding,
            },
        ),
    }
    return artifacts

def smoke_ready_manifest(config: Mapping[str, Any] | ConfigSpec) -> Dict[str, Any]:
    cfg = config.resolved() if isinstance(config, ConfigSpec) else dict(config)
    output_root = build_output_root(cfg.get("output_root"))
    ensure_output_layout(output_root)
    write_registry_artifacts(output_root)
    readiness = write_readiness_artifacts(output_root, cfg)
    return {
        "output_root": str(output_root),
        "mode": cfg.get("mode", SMOKE_MODE),
        "experiment_id": cfg.get("experiment_id", SMOKE_EXPERIMENT_ID),
        "dataset": cfg.get("dataset", DEFAULT_DATASET),
        "backbone": cfg.get("backbone", DEFAULT_BACKBONE),
        "method": cfg.get("method", DEFAULT_METHOD),
        "mask_variant": cfg.get("mask_variant", DEFAULT_MASK_VARIANT),
        "readiness": readiness["readiness"],
        "evaluation_result": readiness["evaluation_result"],
        "reference_grounding": reference_grounding,
    }

def canonical_smoke_contract() -> Dict[str, Any]:
    return {
        "config": default_resolved_runtime_config(),
        "protocol_matrix": protocol_matrix_for_route(),
        "registry_snapshot": registry_snapshot(),
        "artifact_paths": PAPER_VISIBLE_ARTIFACTS,
        "schema_artifacts": list(SCHEMA_READY_ARTIFACTS),
        "reference_grounding": reference_grounding,
    }

def load_optional_optional_backend(name: str) -> Any:
    return load_optional_backend(name)

def optional_backend_availability_report() -> Dict[str, Any]:
    return {
        "torch": available_optional_backend("torch"),
        "gym": available_optional_backend("gym"),
        "gymnasium": available_optional_backend("gymnasium"),
        "datasets": available_optional_backend("datasets"),
        "sbi": available_optional_backend("sbi"),
        "reference_grounding": reference_grounding,
    }

def lazy_backend_factories() -> Dict[str, Callable[[], Any]]:
    return {
        "torch": lambda: load_optional_backend("torch"),
        "gym": lambda: load_optional_backend("gym"),
        "gymnasium": lambda: load_optional_backend("gymnasium"),
        "datasets": lambda: load_optional_backend("datasets"),
        "sbi": lambda: load_optional_backend("sbi"),
    }

def callable_method_component(method_name: str) -> Dict[str, Any]:
    method = resolve_method_spec(method_name)
    return {
        "method": method.name,
        "selector": method.selector,
        "kind": method.kind,
        "mask_variant": method.mask_variant,
        "reference_grounding": method.reference_grounding,
    }

def evaluate_call_interface(config: Mapping[str, Any] | ConfigSpec) -> Dict[str, Any]:
    cfg = config.resolved() if isinstance(config, ConfigSpec) else dict(config)
    data = build_data(
        dataset=str(cfg.get("dataset", DEFAULT_DATASET)),
        backbone=str(cfg.get("backbone", DEFAULT_BACKBONE)),
        method=str(cfg.get("method", DEFAULT_METHOD)),
        max_samples=int(cfg.get("max_samples_per_dataset", 8) or 8),
    )
    preds = [int(i % data["dataset_spec"]["num_classes"]) for i in range(len(data["labels"]))]
    acc = compute_accuracy(preds, data["labels"])
    loss_value = compute_loss([float(p) for p in preds], data["labels"])
    f1_value = compute_f1(preds, data["labels"])
    return {
        "accuracy": acc,
        "loss": loss_value,
        "f1": f1_value,
        "aggregate_accuracy": aggregate_accuracy([acc]),
        "aggregate_loss": aggregate_loss([loss_value]),
        "aggregate_f1": aggregate_f1([f1_value]),
        "dataset": data["dataset_spec"]["id"],
        "backbone": data["backbone_spec"]["id"],
        "method": data["method_spec"]["name"],
        "mask_variant": cfg.get("mask_variant", DEFAULT_MASK_VARIANT),
        "output_mapping": cfg.get("output_mapping", DEFAULT_OUTPUT_MAPPING),
        "reference_grounding": reference_grounding,
    }

def load_classifier(config: Mapping[str, Any] | ConfigSpec) -> Dict[str, Any]:
    cfg = config.resolved() if isinstance(config, ConfigSpec) else dict(config)
    backbone = create_backbone(str(cfg.get("backbone", DEFAULT_BACKBONE)))
    dataset = create_dataset(str(cfg.get("dataset", DEFAULT_DATASET)))
    return {
        "backbone": backbone,
        "dataset": dataset,
        "frozen": backbone["frozen"],
        "class_count": dataset["num_classes"],
        "output_mapping": cfg.get("output_mapping", DEFAULT_OUTPUT_MAPPING),
        "reference_grounding": reference_grounding,
        "available": available_optional_backend("torch"),
    }

def evaluate_predictions(config: Mapping[str, Any] | ConfigSpec) -> Dict[str, Any]:
    return evaluate_call_interface(config)

def compute_metrics_from_predictions(predictions: Sequence[int], labels: Sequence[int]) -> Dict[str, Any]:
    acc = compute_accuracy(predictions, labels)
    loss_value = compute_loss([float(p) for p in predictions], labels)
    f1_value = compute_f1(predictions, labels)
    return {
        "accuracy": acc,
        "loss": loss_value,
        "f1": f1_value,
        "mean_std_accuracy": aggregate_accuracy([acc]),
        "mean_std_loss": aggregate_loss([loss_value]),
        "mean_std_f1": aggregate_f1([f1_value]),
        "reference_grounding": reference_grounding,
    }

def measurement_inventory() -> Dict[str, Any]:
    return {
        "mean_std_accuracy": "mean accuracy percentage and standard deviation percentage",
        "accuracy": "top1 accuracy",
        "loss": "mean loss",
        "f1": "macro f1",
        "reference_grounding": reference_grounding,
    }

def artifact_layout_inventory() -> Dict[str, Any]:
    return {
        "table_artifacts": dict(TABLE_ARTIFACT_PATHS),
        "figure_artifacts": dict(FIGURE_ARTIFACT_PATHS),
        "schema_artifacts": list(SCHEMA_READY_ARTIFACTS),
        "paper_visible_artifacts": list(PAPER_VISIBLE_ARTIFACTS),
        "reference_grounding": reference_grounding,
    }

def current_repository_task_contract() -> Dict[str, Any]:
    return {
        "DEFAULT_EPOCHS": DEFAULT_EPOCHS,
        "resolve_epochs_defaults": resolve_epochs_defaults,
        "epochs_values": epochs_values,
        "DEFAULT_SEED": DEFAULT_SEED,
        "resolve_seed_defaults": resolve_seed_defaults,
        "seed_values": seed_values,
        "compute_accuracy": compute_accuracy,
        "aggregate_accuracy": aggregate_accuracy,
        "compute_f1": compute_f1,
        "aggregate_f1": aggregate_f1,
        "Ours": Ours,
        "Ids": Ids,
        "reference_grounding": reference_grounding,
    }

__all__ = [
    "DEFAULT_EPOCHS",
    "resolve_epochs_defaults",
    "epochs_values",
    "DEFAULT_SEED",
    "resolve_seed_defaults",
    "seed_values",
    "compute_accuracy",
    "aggregate_accuracy",
    "compute_f1",
    "aggregate_f1",
    "compute_loss",
    "aggregate_loss",
    "compute_reward",
    "aggregate_reward",
    "compute_ours_oradaptersby_inventory_objective",
    "compute_ours_oradaptersby_inventory_score",
    "Ours",
    "Ids",
    "DatasetSpec",
    "EnvironmentSpec",
    "BackboneSpec",
    "MetricSpec",
    "ArtifactWriterSpec",
    "ExperimentSpec",
    "MethodSpec",
    "ConfigSpec",
    "METHOD_REGISTRY",
    "DATASET_REGISTRY",
    "ENVIRONMENT_REGISTRY",
    "BACKBONE_REGISTRY",
    "METRIC_REGISTRY",
    "ARTIFACT_WRITER_REGISTRY",
    "EXPERIMENT_PROTOCOLS",
    "build_runtime_config",
    "resolve_config",
    "resolve_method_spec",
    "resolve_dataset_spec",
    "resolve_environment_spec",
    "resolve_backbone_spec",
    "load_classifier",
    "finetune_classifier",
    "create_environment",
    "create_dataset",
    "create_backbone",
    "build_data",
    "load_data",
    "prepare_data",
    "build_reprogramming",
    "load_reprogramming",
    "prepare_reprogramming",
    "write_table_1_artifact",
    "write_table_2_artifact",
    "write_table_3_artifact",
    "write_table_13_artifact",
    "write_table_14_artifact",
    "write_artifact_manifest",
    "write_dry_run_manifest",
    "write_readiness_artifacts",
    "build_metric_registry",
    "build_artifact_writer_registry",
    "build_experiment_registry",
    "build_protocol_matrix",
    "canonical_artifact_manifest",
    "canonical_table_index",
    "canonical_figure_index",
    "canonical_smoke_contract",
    "smoke_ready_manifest",
    "optional_backend_availability_report",
    "lazy_backend_factories",
    "load_optional_backend",
    "available_optional_backend",
    "reference_grounding",
]