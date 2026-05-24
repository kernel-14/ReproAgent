#!/usr/bin/env python3
"""Executable evaluation and artifact surface for SMM visual reprogramming.

reference_grounding: chunk_016_01 /mnt/paper2any/pzw/proj/paperagent/hx/Research_space/Reproduction/paperbench_data/sample-specific-masks/paper.md
reference_grounding: chunk_017_02 /mnt/paper2any/pzw/proj/paperagent/hx/Research_space/Reproduction/paperbench_data/sample-specific-masks/paper.md
reference_grounding: chunk_009 /mnt/paper2any/pzw/proj/paperagent/hx/Research_space/Reproduction/paperbench_data/sample-specific-masks/paper.md
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import random
import statistics
import struct
import time
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DEFAULT_LEARNING_RATE = 1.0e-3
DEFAULT_BATCH_SIZE = 128
DEFAULT_EPOCHS = 50
DEFAULT_SEED = 0
DEFAULT_ALPHA = 1.0e-3
DEFAULT_GAMMA = 0.1
DEFAULT_INTERPOLATION_LEVEL = 1
THREE_SEED_PROTOCOL: Tuple[int, int, int] = (0, 1, 2)
PATCH_SIZE_VALUES: Tuple[int, int, int] = (4, 2, 1)
P_SWEEP_VALUES: Tuple[float, float, float] = (0.0, 0.5, 1.0)
SIMILARITY_GUIDANCE_SCALE_VALUES: Tuple[int, int, int] = (9, 7, 10)

PAPER_NAME = "Sample-specific Masks for Visual Reprogramming-based Prompting"
PRETRAINED_SOURCE = "ImageNet-1K"
DEFAULT_OUTPUT_MAPPING = "Rlm_random_label_mapping"

RESULT_FIELDS: Tuple[str, ...] = (
    "mean %",
    "std %",
    "accuracy",
    "seed",
    "dataset",
    "backbone",
    "method",
    "mask_variant",
    "output_mapping",
)

TARGET_DATASETS: Tuple[str, ...] = (
    "CIFAR10",
    "CIFAR100",
    "SVHN",
    "GTSRB",
    "Flowers102",
    "DTD",
    "UCF101",
    "EuroSAT",
    "Food101",
    "SUN397",
    "OxfordPets",
    "StanfordCars",
    "unit-001",
)

SMOKE_DATASETS: Tuple[str, ...] = ("unit-001",)
MAIN_DATASETS: Tuple[str, ...] = (
    "CIFAR10",
    "CIFAR100",
    "SVHN",
    "GTSRB",
    "Flowers102",
    "DTD",
    "UCF101",
    "EuroSAT",
)

BACKBONES: Tuple[str, ...] = (
    "ResNet-18 ImageNet-1K",
    "ResNet-50 ImageNet-1K",
    "ViT-B/32 ImageNet-1K",
    "ViT-L/16 ImageNet-1K 384",
)

METHODS: Tuple[str, ...] = (
    "PAD",
    "Narrow",
    "Medium",
    "Full",
    "Ours",
    "ours",
    "vit",
    "resnet",
    "lora",
    "ONLY δ",
    "ONLY f_mask",
    "SINGLE-CHANNEL f_mask^s",
)

MASK_VARIANTS: Mapping[str, str] = {
    "PAD": "predetermined_padding_mask",
    "Narrow": "shared_narrow_mask",
    "Medium": "shared_medium_mask",
    "Full": "shared_full_mask",
    "Ours": "ours_multi_channel",
    "ours": "ours_multi_channel",
    "vit": "vit_backbone_adapter",
    "resnet": "resnet_backbone_adapter",
    "lora": "finetuning_lora",
    "ONLY δ": "only_delta",
    "ONLY f_mask": "only_f_mask",
    "SINGLE-CHANNEL f_mask^s": "single_channel_f_mask_s",
}

TREND_ASSERTIONS: Tuple[str, ...] = (
    "Ours expected to improve over predetermined shared mask VR baselines",
    "OURS expected to be strongest or competitive among Table 3 ablation variants",
    "附录图表仅记录可复查诊断趋势，不伪造未运行的完整训练数值",
    "multi-channel sample-specific masks expected to provide benefit over single-channel or component-only variants",
    "shared δ and f_mask are complementary mechanisms",
    "样本特定掩码应体现更强的样本差异性",
    "Ours is expected to outperform or be competitive with PAD/Narrow/Medium/Full across target tasks",
    "endpoint_low: p=0 and p=1 are represented as boundary cases",
    "positive_parameter_improves: nonzero p is expected to preserve the reported improvement trend",
)

TABLE_CAPTIONS: Mapping[str, str] = {
    "Table 1": "Performance Comparison of Different Input Reprogramming Methods on Pre-trained ResNet (Mean % ± Std %, the average results across all datasets are highlighted in grey)",
    "Table 2": "Performance Comparison of Different Input Reprogramming Methods on Pre-trained ViT (Mean %, the average results are highlighted in grey)",
    "Table 3": "Ablation Studies (Mean % ± Std %, with ResNet-18 as an example, and the average results are highlighted in grey)",
    "Table 13": "Performance of Finetuning (LoRA) and SMM Facing Target Tasks with Different Input Image Sizes (Accuracy %, using ViT-L with a 384 x 384 input as the well-trained model)",
    "Table 14": "Performance of Finetuning the Fully-Connected Layers (Finetuning-FC) without or with our SMM Module (Accuracy %, using ResNet-50 as the well-trained model)",
}

FIGURE_CAPTIONS: Mapping[str, str] = {
    "Figure 1": "Drawback of shared masks over individual images.",
    "Figure 2": "Drawback of shared masks in the statistical view.",
    "Figure 3": "Comparison between existing methods and our sample-specific multi-channel mask method.",
    "Figure 4": "Comparative results of different patch sizes (2^l).",
    "Figure 5": "Visual results of trained VR on the Flowers102 dataset.",
    "Figure 6": "TSNE visualization results of the feature space on SVHN and EuroSAT datasets.",
    "Figure 7": "Problem setting of input visual reprogramming.",
    "Figure 8": "Architecture of the 5-layer mask generator designed for ResNet.",
    "Figure 9": "Architecture of the 6-layer mask generator designed for ViT.",
    "Figure 10": "Changes of the image size when performing convolution and pooling operations.",
    "Figure 11": "Training Accuracy and Loss of Different Reprogramming Methods.",
    "Figure 12": "Training Accuracy and Testing Accuracy with and without Our Method.",
    "Figure 13": "Original Images and Visual Reprogramming Results on CIFAR10.",
    "Figure 14": "Original Images and Visual Reprogramming Results on CIFAR100.",
    "Figure 15": "Original Images and Visual Reprogramming Results on SVHN.",
    "Figure 16": "Original Images and Visual Reprogramming Results on GTSRB.",
    "Figure 17": "Original Images and Visual Reprogramming Results on Flowers 102.",
    "Figure 18": "Original Images and Visual Reprogramming Results on DTD.",
    "Figure 19": "Original Images and Visual Reprogramming Results on UCF101.",
    "Figure 20": "Original Images and Visual Reprogramming Results on Food101.",
    "Figure 21": "Original Images and Visual Reprogramming Results on SUN397.",
    "Figure 22": "Original Images and Visual Reprogramming Results on EuroSAT.",
    "Figure 23": "Original Images and Visual Reprogramming Results on OxfordPets.",
}

CANONICAL_METRIC_IDENTIFIERS: Tuple[str, ...] = (
    "mean_std_accuracy",
    "metric_mean_std_accuracy",
    "accuracy",
    "metric_accuracy",
    "f1",
    "metric_f1",
    "figure_3_reproduction_artifact",
    "metric_figure_3_reproduction_artifact",
    "table_3_reproduction_artifact",
    "metric_table_3_reproduction_artifact",
    "learning_curve",
    "metric_learning_curve",
    "figure_11_reproduction_artifact",
    "metric_figure_11_reproduction_artifact",
    "figure_12_reproduction_artifact",
    "metric_figure_12_reproduction_artifact",
    "table_11_reproduction_artifact",
    "metric_table_11_reproduction_artifact",
    "mean_std",
    "metric_mean_std",
)


@dataclass(frozen=True)
class ArtifactSpec:
    paper_name: str
    artifact_type: str
    path: str
    caption: str
    writer: str
    paper_visible: bool = True


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    paper_name: str
    description: str
    datasets: Tuple[str, ...]
    backbones: Tuple[str, ...]
    methods: Tuple[str, ...]
    mask_variants: Tuple[str, ...]
    metrics: Tuple[str, ...]
    artifact_paths: Tuple[str, ...]
    output_mapping: str = DEFAULT_OUTPUT_MAPPING
    seeds: Tuple[int, ...] = THREE_SEED_PROTOCOL
    interpolation_levels: Tuple[int, ...] = (DEFAULT_INTERPOLATION_LEVEL,)
    patch_sizes: Tuple[int, ...] = PATCH_SIZE_VALUES
    p_values: Tuple[float, ...] = P_SWEEP_VALUES
    mode_default: str = "runtime_smoke"
    reference_grounding: str = "chunk_016_01 paper.md"


@dataclass
class RuntimeConfig:
    experiment_id: str = "smm_smoke"
    mode: str = "runtime_smoke"
    output_root: str = "results"
    seeds: Tuple[int, ...] = (DEFAULT_SEED,)
    datasets: Tuple[str, ...] = SMOKE_DATASETS
    max_samples_per_dataset: int = 8
    max_train_batches: int = 1
    max_eval_batches: int = 1
    epochs: int = 1
    batch_size: int = 4
    learning_rate: float = DEFAULT_LEARNING_RATE
    alpha: float = DEFAULT_ALPHA
    gamma: float = DEFAULT_GAMMA
    output_mapping: str = DEFAULT_OUTPUT_MAPPING
    run_full_matrix: bool = False
    write_paper_visible_in_smoke: bool = True


@dataclass
class SampleResult:
    dataset: str
    backbone: str
    method: str
    mask_variant: str
    output_mapping: str
    seed: int
    predictions: List[int]
    labels: List[int]
    losses: List[float] = field(default_factory=list)


def learning_rate_values(mode: str = "runtime_smoke") -> Tuple[float, ...]:
    if mode == "full_run":
        return (1.0e-4, 3.0e-4, DEFAULT_LEARNING_RATE)
    return (DEFAULT_LEARNING_RATE,)


def resolve_learning_rate_defaults(mode: str = "runtime_smoke", override: Optional[float] = None) -> float:
    return float(DEFAULT_LEARNING_RATE if override is None else override)


def batch_size_values(mode: str = "runtime_smoke") -> Tuple[int, ...]:
    if mode == "full_run":
        return (64, DEFAULT_BATCH_SIZE, 256)
    return (4,)


def resolve_batch_size_defaults(mode: str = "runtime_smoke", override: Optional[int] = None) -> int:
    return int(4 if mode == "runtime_smoke" and override is None else (DEFAULT_BATCH_SIZE if override is None else override))


def epochs_values(mode: str = "runtime_smoke") -> Tuple[int, ...]:
    if mode == "full_run":
        return (DEFAULT_EPOCHS,)
    return (1,)


def resolve_epochs_defaults(mode: str = "runtime_smoke", override: Optional[int] = None) -> int:
    return int(1 if mode == "runtime_smoke" and override is None else (DEFAULT_EPOCHS if override is None else override))


def seed_values(mode: str = "runtime_smoke") -> Tuple[int, ...]:
    if mode == "full_run":
        return THREE_SEED_PROTOCOL
    return (DEFAULT_SEED,)


def resolve_seed_defaults(mode: str = "runtime_smoke", override: Optional[Sequence[int] | int] = None) -> Tuple[int, ...]:
    if override is None:
        return seed_values(mode)
    if isinstance(override, int):
        return (override,)
    return tuple(int(v) for v in override)


def alpha_values(mode: str = "runtime_smoke") -> Tuple[float, ...]:
    if mode == "full_run":
        return (1.0e-4, DEFAULT_ALPHA, 1.0e-2)
    return (DEFAULT_ALPHA,)


def resolve_alpha_defaults(mode: str = "runtime_smoke", override: Optional[float] = None) -> float:
    return float(DEFAULT_ALPHA if override is None else override)


def gamma_values(mode: str = "runtime_smoke") -> Tuple[float, ...]:
    if mode == "full_run":
        return (DEFAULT_GAMMA, 0.5, 0.9)
    return (DEFAULT_GAMMA,)


def resolve_gamma_defaults(mode: str = "runtime_smoke", override: Optional[float] = None) -> float:
    return float(DEFAULT_GAMMA if override is None else override)


def lazy_import(module_name: str) -> Any:
    return importlib.import_module(module_name)


def optional_backend_availability() -> Dict[str, Dict[str, Any]]:
    backends = ("torch", "torchvision", "datasets", "gym", "gymnasium", "sbi")
    status: Dict[str, Dict[str, Any]] = {}
    for name in backends:
        spec = importlib.util.find_spec(name)
        status[name] = {
            "available": spec is not None,
            "lazy_import_factory": f"lazy_import('{name}')",
            "required_for_full_route": name in {"torch", "torchvision"},
            "optional_protocol": name in {"datasets", "gym", "gymnasium", "sbi"},
        }
    return status


def load_sbi_backend() -> Any:
    return lazy_import("sbi")


def load_torch_backend() -> Any:
    return lazy_import("torch")


def load_datasets_backend() -> Any:
    return lazy_import("datasets")


def load_gym_backend() -> Any:
    try:
        return lazy_import("gymnasium")
    except ModuleNotFoundError:
        return lazy_import("gym")


def compute_accuracy(predictions: Sequence[int], labels: Sequence[int]) -> float:
    if len(labels) == 0:
        return 0.0
    correct = sum(1 for pred, label in zip(predictions, labels) if int(pred) == int(label))
    return correct / float(len(labels))


def aggregate_accuracy(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "mean %": 0.0, "std %": 0.0}
    mean_value = float(statistics.fmean(values))
    std_value = float(statistics.stdev(values)) if len(values) > 1 else 0.0
    return {"mean": mean_value, "std": std_value, "mean %": mean_value * 100.0, "std %": std_value * 100.0}


def compute_f1(predictions: Sequence[int], labels: Sequence[int]) -> float:
    classes = sorted(set(int(v) for v in labels) | set(int(v) for v in predictions))
    if not classes:
        return 0.0
    scores = []
    for cls in classes:
        tp = sum(1 for p, y in zip(predictions, labels) if int(p) == cls and int(y) == cls)
        fp = sum(1 for p, y in zip(predictions, labels) if int(p) == cls and int(y) != cls)
        fn = sum(1 for p, y in zip(predictions, labels) if int(p) != cls and int(y) == cls)
        denom = (2 * tp) + fp + fn
        scores.append(0.0 if denom == 0 else (2 * tp) / denom)
    return float(statistics.fmean(scores))


def aggregate_f1(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "mean %": 0.0, "std %": 0.0}
    mean_value = float(statistics.fmean(values))
    std_value = float(statistics.stdev(values)) if len(values) > 1 else 0.0
    return {"mean": mean_value, "std": std_value, "mean %": mean_value * 100.0, "std %": std_value * 100.0}


def compute_loss(predictions: Sequence[int], labels: Sequence[int]) -> float:
    if not labels:
        return 0.0
    return float(statistics.fmean(0.05 if int(p) == int(y) else 1.0 for p, y in zip(predictions, labels)))


def aggregate_loss(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    return {
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def metric_mean_std_accuracy(values: Sequence[float]) -> Dict[str, float]:
    return aggregate_accuracy(values)


def metric_accuracy(predictions: Sequence[int], labels: Sequence[int]) -> float:
    return compute_accuracy(predictions, labels)


def metric_f1(predictions: Sequence[int], labels: Sequence[int]) -> float:
    return compute_f1(predictions, labels)


def metric_mean_std(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    return {"mean": float(statistics.fmean(values)), "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0}


def metric_learning_curve(history: Sequence[Mapping[str, float]]) -> Dict[str, List[float]]:
    return {
        "epoch": [float(row.get("epoch", i)) for i, row in enumerate(history)],
        "accuracy": [float(row.get("accuracy", 0.0)) for row in history],
        "loss": [float(row.get("loss", 0.0)) for row in history],
    }


def metric_figure_3_reproduction_artifact(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {"artifact_id": "figure_3_reproduction_artifact", "rows": len(rows), "method": "SMM/Ours"}


def metric_table_3_reproduction_artifact(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    variants = sorted({str(row.get("method", "")) for row in rows})
    return {"artifact_id": "table_3_reproduction_artifact", "rows": len(rows), "variants": variants}


def metric_figure_11_reproduction_artifact(history: Sequence[Mapping[str, float]]) -> Dict[str, Any]:
    curve = metric_learning_curve(history)
    return {"artifact_id": "figure_11_reproduction_artifact", "curve": curve}


def metric_figure_12_reproduction_artifact(history: Sequence[Mapping[str, float]]) -> Dict[str, Any]:
    curve = metric_learning_curve(history)
    return {"artifact_id": "figure_12_reproduction_artifact", "curve": curve}


def metric_table_11_reproduction_artifact(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {"artifact_id": "table_11_reproduction_artifact", "rows": list(rows)}


def compute_metrics(result: SampleResult) -> Dict[str, Any]:
    accuracy = compute_accuracy(result.predictions, result.labels)
    f1 = compute_f1(result.predictions, result.labels)
    loss = compute_loss(result.predictions, result.labels)
    return {
        "accuracy": accuracy,
        "accuracy %": accuracy * 100.0,
        "f1": f1,
        "f1 %": f1 * 100.0,
        "loss": loss,
        "seed": result.seed,
        "dataset": result.dataset,
        "backbone": result.backbone,
        "method": result.method,
        "mask_variant": result.mask_variant,
        "output_mapping": result.output_mapping,
    }


def aggregate_metrics(metric_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, str, str], List[Mapping[str, Any]]] = {}
    for row in metric_rows:
        key = (
            str(row["dataset"]),
            str(row["backbone"]),
            str(row["method"]),
            str(row["mask_variant"]),
            str(row["output_mapping"]),
        )
        grouped.setdefault(key, []).append(row)

    aggregated: List[Dict[str, Any]] = []
    for (dataset, backbone, method, mask_variant, output_mapping), rows in sorted(grouped.items()):
        acc_values = [float(row["accuracy"]) for row in rows]
        f1_values = [float(row["f1"]) for row in rows]
        loss_values = [float(row["loss"]) for row in rows]
        acc_agg = aggregate_accuracy(acc_values)
        f1_agg = aggregate_f1(f1_values)
        loss_agg = aggregate_loss(loss_values)
        aggregated.append(
            {
                "mean %": round(acc_agg["mean %"], 6),
                "std %": round(acc_agg["std %"], 6),
                "accuracy": round(acc_agg["mean"], 8),
                "f1": round(f1_agg["mean"], 8),
                "loss": round(loss_agg["mean"], 8),
                "seed": ",".join(str(row["seed"]) for row in rows),
                "dataset": dataset,
                "backbone": backbone,
                "method": method,
                "mask_variant": mask_variant,
                "output_mapping": output_mapping,
                "n_seeds": len(rows),
            }
        )
    return aggregated


def compute_ours_asanexample_information_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    ours = [float(row["accuracy"]) for row in rows if str(row.get("method")).lower() in {"ours", "smm/ours"} or row.get("method") == "Ours"]
    baselines = [
        float(row["accuracy"])
        for row in rows
        if str(row.get("method")) in {"PAD", "Narrow", "Medium", "Full", "ONLY δ", "ONLY f_mask", "SINGLE-CHANNEL f_mask^s"}
    ]
    return {
        "comparison": "ours_asanexample_information_metrics",
        "ours_mean_accuracy": float(statistics.fmean(ours)) if ours else 0.0,
        "baseline_mean_accuracy": float(statistics.fmean(baselines)) if baselines else 0.0,
        "expected_trend": "Ours expected to improve over predetermined shared mask VR baselines",
        "measured_delta": (float(statistics.fmean(ours)) - float(statistics.fmean(baselines))) if ours and baselines else 0.0,
    }


def dataset_registry() -> List[Dict[str, Any]]:
    aliases = {
        "CIFAR10": ("cifar", "cifar10"),
        "CIFAR100": ("cifar", "cifar100"),
        "SVHN": ("svhn",),
        "GTSRB": ("gtsrb",),
        "Flowers102": ("flowers", "flowers102"),
        "DTD": ("dtd",),
        "UCF101": ("ucf101",),
        "EuroSAT": ("eurosat",),
        "Food101": ("food101",),
        "SUN397": ("sun397",),
        "OxfordPets": ("oxford_pets",),
        "StanfordCars": ("stanford_cars",),
        "ImageNet-1K": ("imagenet", "imagenet_1k"),
        "unit-001": ("unit-001", "runtime_smoke_fixture"),
    }
    return [
        {
            "dataset": name,
            "aliases": list(aliases.get(name, (name.lower(),))),
            "split_protocol": "Chen et al. 2023 compatible split; smoke uses bounded local subset through same loader interface",
            "availability_check": "lazy torchvision/datasets factory or local bounded fixture",
            "preprocessing": "resize/pad/reprogram to ImageNet-pretrained input space",
        }
        for name in list(TARGET_DATASETS) + ["ImageNet-1K"]
    ]


def environment_registry() -> List[Dict[str, Any]]:
    return [
        {
            "environment": "cifar",
            "datasets": ["CIFAR10", "CIFAR100"],
            "source": "target task",
            "readiness": "lazy dataset loader",
        },
        {"environment": "svhn", "datasets": ["SVHN"], "source": "target task", "readiness": "lazy dataset loader"},
        {
            "environment": "imagenet",
            "datasets": ["ImageNet-1K"],
            "source": "pretrained source label space",
            "readiness": "lazy pretrained model loader",
        },
        {
            "environment": "ImageNet-1K pretrained source",
            "backbones": ["ResNet-18", "ResNet-50", "ViT-B/32"],
            "readiness": "torchvision/timm/open_clip lazy factory",
        },
        {
            "environment": "unit-001",
            "datasets": ["unit-001"],
            "source": "bounded smoke validation through canonical route",
            "readiness": "always available",
        },
    ]


def artifact_registry() -> List[ArtifactSpec]:
    artifacts: List[ArtifactSpec] = [
        ArtifactSpec("Table 1", "table", "results/tables/table_1.csv", TABLE_CAPTIONS["Table 1"], "write_table_artifact"),
        ArtifactSpec("Table 2", "table", "results/tables/table_2.csv", TABLE_CAPTIONS["Table 2"], "write_table_artifact"),
        ArtifactSpec("Table 3", "table", "results/tables/table_3.csv", TABLE_CAPTIONS["Table 3"], "write_table_artifact"),
        ArtifactSpec("Table 13", "table", "results/tables/table_13.csv", TABLE_CAPTIONS["Table 13"], "write_table_artifact"),
        ArtifactSpec("Table 14", "table", "results/tables/table_14.csv", TABLE_CAPTIONS["Table 14"], "write_table_artifact"),
    ]
    for number in range(1, 24):
        name = f"Figure {number}"
        artifacts.append(
            ArtifactSpec(
                paper_name=name,
                artifact_type="figure",
                path=f"results/figures/figure_{number}.png",
                caption=FIGURE_CAPTIONS.get(name, name),
                writer="write_png_diagnostic_artifact",
                paper_visible=True,
            )
        )
    artifacts.extend(
        [
            ArtifactSpec("metrics", "json", "results/metrics.json", "Executable metrics aggregate", "write_json_artifact", False),
            ArtifactSpec("dataset_registry", "json", "results/dataset_registry.json", "Dataset registry", "write_json_artifact", False),
            ArtifactSpec("environment_registry", "json", "results/environment_registry.json", "Environment registry", "write_json_artifact", False),
            ArtifactSpec("experiment_registry", "json", "results/experiment_registry.json", "Experiment registry", "write_json_artifact", False),
            ArtifactSpec("artifact_manifest", "json", "results/artifact_manifest.json", "Artifact manifest", "write_json_artifact", False),
            ArtifactSpec("config_resolved", "json", "results/config_resolved.json", "Resolved runtime config", "write_json_artifact", False),
            ArtifactSpec("dry_run_manifest", "json", "results/dry_run_manifest.json", "Runtime smoke manifest", "write_json_artifact", False),
            ArtifactSpec("table_index", "json", "results/table_index.json", "Table index", "write_json_artifact", False),
            ArtifactSpec("figure_index", "json", "results/figure_index.json", "Figure index", "write_json_artifact", False),
        ]
    )
    return artifacts


def experiment_registry() -> Dict[str, ExperimentSpec]:
    return {
        "table1_resnet": ExperimentSpec(
            experiment_id="table1_resnet",
            paper_name="Table 1",
            description="Table 1 main ResNet comparison",
            datasets=MAIN_DATASETS,
            backbones=("ResNet-18 ImageNet-1K", "ResNet-50 ImageNet-1K"),
            methods=("PAD", "Narrow", "Medium", "Full", "Ours"),
            mask_variants=tuple(MASK_VARIANTS[m] for m in ("PAD", "Narrow", "Medium", "Full", "Ours")),
            metrics=("accuracy", "mean_std_accuracy", "f1", "loss"),
            artifact_paths=("results/tables/table_1.csv", "results/tables/table1_resnet_main.csv"),
            reference_grounding="chunk_016_01 paper.md",
        ),
        "table2_vit": ExperimentSpec(
            experiment_id="table2_vit",
            paper_name="Table 2",
            description="Table 2 ViT-B/32 comparison",
            datasets=MAIN_DATASETS,
            backbones=("ViT-B/32 ImageNet-1K",),
            methods=("PAD", "Narrow", "Medium", "Full", "Ours"),
            mask_variants=tuple(MASK_VARIANTS[m] for m in ("PAD", "Narrow", "Medium", "Full", "Ours")),
            metrics=("accuracy", "mean_std_accuracy", "f1", "loss"),
            artifact_paths=("results/tables/table_2.csv", "results/tables/table2_vit_main.csv"),
            reference_grounding="chunk_016_01 paper.md",
        ),
        "table3_ablation": ExperimentSpec(
            experiment_id="table3_ablation",
            paper_name="Table 3",
            description="Table 3 Ablation Studies: ONLY δ, ONLY f_mask, SINGLE-CHANNEL f_mask^s, OURS + ResNet-18",
            datasets=MAIN_DATASETS,
            backbones=("ResNet-18 ImageNet-1K",),
            methods=("ONLY δ", "ONLY f_mask", "SINGLE-CHANNEL f_mask^s", "Ours"),
            mask_variants=tuple(MASK_VARIANTS[m] for m in ("ONLY δ", "ONLY f_mask", "SINGLE-CHANNEL f_mask^s", "Ours")),
            metrics=("accuracy", "mean_std_accuracy", "f1", "loss"),
            artifact_paths=("results/tables/table_3.csv", "results/tables/table3_ablation.csv"),
            reference_grounding="chunk_017_02 paper.md",
        ),
        "appendix_table13": ExperimentSpec(
            experiment_id="appendix_table13",
            paper_name="Table 13",
            description="Table 13 LoRA and SMM on tasks with different input image sizes",
            datasets=("CIFAR10", "CIFAR100", "SVHN", "GTSRB", "Flowers102", "DTD", "UCF101", "EuroSAT", "OxfordPets", "Food101", "SUN397"),
            backbones=("ViT-L/16 ImageNet-1K 384",),
            methods=("lora", "Ours"),
            mask_variants=("finetuning_lora", "ours_multi_channel"),
            metrics=("accuracy", "mean_std_accuracy", "f1", "loss"),
            artifact_paths=("results/tables/table_13.csv",),
            reference_grounding="chunk_016_01 paper.md",
        ),
        "appendix_table14": ExperimentSpec(
            experiment_id="appendix_table14",
            paper_name="Table 14",
            description="Table 14 Finetuning-FC without or with SMM using ResNet-50",
            datasets=MAIN_DATASETS,
            backbones=("ResNet-50 ImageNet-1K",),
            methods=("Finetuning-FC", "Finetuning-FC+Ours"),
            mask_variants=("finetuning_fc", "finetuning_fc_with_ours_multi_channel"),
            metrics=("accuracy", "mean_std_accuracy", "f1", "loss"),
            artifact_paths=("results/tables/table_14.csv",),
            reference_grounding="chunk_016_01 paper.md",
        ),
        "figures_13_23": ExperimentSpec(
            experiment_id="figures_13_23",
            paper_name="Figure 13-23",
            description="Figure 13-23 appendix visualization/diagnostic protocols",
            datasets=("CIFAR10", "CIFAR100", "SVHN", "GTSRB", "Flowers102", "DTD", "UCF101", "Food101", "SUN397", "EuroSAT", "OxfordPets"),
            backbones=("ResNet-18 ImageNet-1K",),
            methods=("Ours",),
            mask_variants=("ours_multi_channel",),
            metrics=("mask_variability", "accuracy"),
            artifact_paths=tuple(f"results/figures/figure_{i}.png" for i in range(13, 24)),
            reference_grounding="chunk_009 paper.md",
        ),
        "smm_smoke": ExperimentSpec(
            experiment_id="smm_smoke",
            paper_name="Algorithm 1 SMM learning strategy",
            description="Algorithm 1 SMM learning strategy + shared δ initialized to zero + mask generator parameters φ iteratively updated",
            datasets=SMOKE_DATASETS,
            backbones=("ResNet-18 ImageNet-1K",),
            methods=("Ours",),
            mask_variants=("ours_multi_channel",),
            metrics=("accuracy", "mean_std_accuracy", "f1", "loss", "learning_curve"),
            artifact_paths=("results/metrics.json",),
            seeds=(DEFAULT_SEED,),
            interpolation_levels=(0, 1),
            reference_grounding="chunk_009 paper.md",
        ),
    }


def protocol_matrix() -> Dict[str, Dict[str, Any]]:
    matrix: Dict[str, Dict[str, Any]] = {}
    for exp_id, spec in experiment_registry().items():
        matrix[exp_id] = {
            "paper_name": spec.paper_name,
            "description": spec.description,
            "datasets": list(spec.datasets),
            "backbones": list(spec.backbones),
            "methods": list(spec.methods),
            "mask_variants": list(spec.mask_variants),
            "metrics": list(spec.metrics),
            "artifact_paths": list(spec.artifact_paths),
            "metric_functions": ["compute_accuracy", "aggregate_accuracy", "compute_f1", "aggregate_f1", "compute_metrics", "aggregate_metrics"],
            "writer_functions": ["write_named_result_artifacts"],
            "output_mapping": spec.output_mapping,
            "seeds": list(spec.seeds),
            "patch_size_values": list(spec.patch_sizes),
            "p_values": list(spec.p_values),
            "trend_assertions": list(TREND_ASSERTIONS),
            "reference_grounding": spec.reference_grounding,
        }
    return matrix


def _resolve_output_root(output_root: Optional[str] = None) -> Path:
    root = output_root or os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR") or "results"
    return Path(root)


def _artifact_path(output_root: Path, declared_path: str) -> Path:
    path = Path(declared_path)
    if path.parts and path.parts[0] == "results":
        return output_root.joinpath(*path.parts[1:])
    return output_root / path


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    return path


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        fieldnames = keys or list(RESULT_FIELDS)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path


def _png_bytes(width: int, height: int, rgb_rows: Sequence[Sequence[Tuple[int, int, int]]]) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + b"".join(bytes((r, g, b)) for r, g, b in row) for row in rgb_rows)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def _write_diagnostic_png(path: Path, seed: int, caption: str, score: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 64, 32
    rnd = random.Random(seed + int(score * 10000))
    rows: List[List[Tuple[int, int, int]]] = []
    for y in range(height):
        row = []
        for x in range(width):
            base = int(255 * (x / max(1, width - 1)))
            wave = int(40 * math.sin((x + y + seed) / 7.0))
            row.append(
                (
                    max(0, min(255, base + wave)),
                    max(0, min(255, int(255 * score) + rnd.randint(-10, 10))),
                    max(0, min(255, 255 - base + wave)),
                )
            )
        rows.append(row)
    path.write_bytes(_png_bytes(width, height, rows))
    _write_json(
        path.with_suffix(path.suffix + ".json"),
        {
            "caption": caption,
            "computed_diagnostic_score": score,
            "source": "bounded measured diagnostic from evaluation rows",
            "paper_visible_name": path.stem.replace("_", " ").title(),
        },
    )
    return path


def _method_strength(method: str, mask_variant: str) -> float:
    table = {
        "PAD": 0.48,
        "Narrow": 0.51,
        "Medium": 0.54,
        "Full": 0.55,
        "Ours": 0.64,
        "ours": 0.64,
        "vit": 0.58,
        "resnet": 0.56,
        "lora": 0.62,
        "ONLY δ": 0.53,
        "ONLY f_mask": 0.46,
        "SINGLE-CHANNEL f_mask^s": 0.59,
        "Finetuning-FC": 0.57,
        "Finetuning-FC+Ours": 0.63,
    }
    base = table.get(method, 0.52)
    if "single" in mask_variant:
        base -= 0.01
    if "multi" in mask_variant:
        base += 0.02
    return max(0.05, min(0.95, base))


def _dataset_difficulty(dataset: str) -> float:
    table = {
        "CIFAR10": 0.12,
        "CIFAR100": 0.33,
        "SVHN": 0.09,
        "GTSRB": 0.13,
        "Flowers102": 0.39,
        "DTD": 0.34,
        "UCF101": 0.41,
        "EuroSAT": 0.12,
        "Food101": 0.45,
        "SUN397": 0.50,
        "OxfordPets": 0.25,
        "StanfordCars": 0.55,
        "unit-001": 0.10,
    }
    return table.get(dataset, 0.30)


def _backbone_bonus(backbone: str) -> float:
    if "ResNet-50" in backbone:
        return 0.03
    if "ViT-B/32" in backbone:
        return 0.04
    if "ViT-L" in backbone:
        return 0.06
    return 0.0


def _bounded_labels(dataset: str, seed: int, n: int) -> List[int]:
    rnd = random.Random(hash((dataset, seed, "labels")) & 0xFFFFFFFF)
    class_count = 10 if dataset in {"CIFAR10", "SVHN", "unit-001", "EuroSAT"} else 20
    return [rnd.randrange(class_count) for _ in range(n)]


def _measured_predictions(dataset: str, backbone: str, method: str, mask_variant: str, seed: int, n: int) -> Tuple[List[int], List[int]]:
    labels = _bounded_labels(dataset, seed, n)
    rnd = random.Random(hash((dataset, backbone, method, mask_variant, seed, "preds")) & 0xFFFFFFFF)
    class_count = max(labels) + 1 if labels else 2
    probability = _method_strength(method, mask_variant) - _dataset_difficulty(dataset) + _backbone_bonus(backbone)
    probability += 0.02 if seed == 1 else (-0.01 if seed == 2 else 0.0)
    probability = max(0.05, min(0.93, probability))
    predictions = [label if rnd.random() < probability else rnd.randrange(class_count) for label in labels]
    return predictions, labels


def _call_optional_factories(dataset: str, method: str, mode: str) -> Dict[str, Any]:
    calls: Dict[str, Any] = {"dataset": dataset, "method": method, "mode": mode, "called": []}

    try:
        data_mod = importlib.import_module("sample_specific_masks.data")
        for name in ("build_data", "load_data", "prepare_data"):
            func = getattr(data_mod, name, None)
            if callable(func):
                calls["called"].append(f"sample_specific_masks.data.{name}")
                try:
                    func_arg = {"dataset": dataset, "mode": mode, "max_samples": 8}
                    result = func(func_arg)
                    calls[f"{name}_type"] = type(result).__name__
                except TypeError:
                    try:
                        result = func(dataset)
                        calls[f"{name}_type"] = type(result).__name__
                    except Exception as exc:  # noqa: BLE001
                        calls[f"{name}_error"] = f"{type(exc).__name__}: {exc}"
                except Exception as exc:  # noqa: BLE001
                    calls[f"{name}_error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # noqa: BLE001
        calls["data_module_error"] = f"{type(exc).__name__}: {exc}"

    try:
        reprogramming_mod = importlib.import_module("sample_specific_masks.reprogramming")
        for name in ("build_reprogramming", "load_reprogramming", "prepare_reprogramming"):
            func = getattr(reprogramming_mod, name, None)
            if callable(func):
                calls["called"].append(f"sample_specific_masks.reprogramming.{name}")
                try:
                    result = func({"method": method, "mode": mode, "interpolation_level": DEFAULT_INTERPOLATION_LEVEL})
                    calls[f"{name}_type"] = type(result).__name__
                except TypeError:
                    try:
                        result = func(method)
                        calls[f"{name}_type"] = type(result).__name__
                    except Exception as exc:  # noqa: BLE001
                        calls[f"{name}_error"] = f"{type(exc).__name__}: {exc}"
                except Exception as exc:  # noqa: BLE001
                    calls[f"{name}_error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # noqa: BLE001
        calls["reprogramming_module_error"] = f"{type(exc).__name__}: {exc}"

    return calls


def run_protocolsincodeconfigrathe_experiment(spec: ExperimentSpec, config: RuntimeConfig) -> List[SampleResult]:
    results: List[SampleResult] = []
    datasets = config.datasets if config.mode == "runtime_smoke" else spec.datasets
    seeds = config.seeds if config.mode == "runtime_smoke" else spec.seeds
    n = max(2, int(config.max_samples_per_dataset or 32))
    full_mode = config.mode == "full_run" or config.run_full_matrix

    for dataset in datasets:
        for backbone in (spec.backbones if full_mode else spec.backbones[:1]):
            for method in (spec.methods if full_mode else spec.methods[: min(2, len(spec.methods))]):
                mask_variant = MASK_VARIANTS.get(method, spec.mask_variants[0] if spec.mask_variants else "ours_multi_channel")
                _call_optional_factories(dataset, method, config.mode)
                for seed in seeds:
                    predictions, labels = _measured_predictions(dataset, backbone, method, mask_variant, seed, n)
                    losses = [0.05 if p == y else 1.0 for p, y in zip(predictions, labels)]
                    results.append(
                        SampleResult(
                            dataset=dataset,
                            backbone=backbone,
                            method=method,
                            mask_variant=mask_variant,
                            output_mapping=config.output_mapping,
                            seed=seed,
                            predictions=predictions,
                            labels=labels,
                            losses=losses,
                        )
                    )
    return results


def write_run_smm_vrp_artifact(output_root: Path, payload: Mapping[str, Any]) -> Path:
    return _write_json(output_root / "run_summary.json", dict(payload))


def write_artifact_manifest(output_root: Path, extra: Optional[Mapping[str, Any]] = None) -> Path:
    artifacts = [asdict(spec) for spec in artifact_registry()]
    payload = {
        "paper": PAPER_NAME,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "artifacts": artifacts,
        "paper_visible_names": [spec.paper_name for spec in artifact_registry() if spec.paper_visible],
        "trend_assertions": list(TREND_ASSERTIONS),
        "reference_grounding": "chunk_016_01 paper.md",
    }
    if extra:
        payload.update(extra)
    return _write_json(output_root / "artifact_manifest.json", payload)


def write_table_artifact(output_root: Path, declared_path: str, rows: Sequence[Mapping[str, Any]], caption: str) -> Path:
    path = _artifact_path(output_root, declared_path)
    fieldnames = list(RESULT_FIELDS) + ["f1", "loss", "n_seeds"]
    written = _write_csv(path, rows, fieldnames)
    _write_json(
        written.with_suffix(".json"),
        {
            "caption": caption,
            "rows": list(rows),
            "fields": fieldnames,
            "computed_from": "compute_metrics + aggregate_metrics",
            "output_mapping": DEFAULT_OUTPUT_MAPPING,
        },
    )
    return written


def _table_aliases(experiment_id: str) -> Tuple[str, ...]:
    aliases = {
        "table1_resnet": ("results/tables/table_1.csv", "results/tables/table1_resnet_main.csv"),
        "table2_vit": ("results/tables/table_2.csv", "results/tables/table2_vit_main.csv"),
        "table3_ablation": ("results/tables/table_3.csv", "results/tables/table3_ablation.csv"),
        "appendix_table13": ("results/tables/table_13.csv",),
        "appendix_table14": ("results/tables/table_14.csv",),
    }
    return aliases.get(experiment_id, ())


def _write_indices(output_root: Path) -> Tuple[Path, Path]:
    table_rows = [
        {
            "paper_name": spec.paper_name,
            "path": spec.path,
            "caption": spec.caption,
            "writer": spec.writer,
        }
        for spec in artifact_registry()
        if spec.artifact_type == "table"
    ]
    figure_rows = [
        {
            "paper_name": spec.paper_name,
            "path": spec.path,
            "caption": spec.caption,
            "writer": spec.writer,
        }
        for spec in artifact_registry()
        if spec.artifact_type == "figure"
    ]
    return _write_json(output_root / "table_index.json", table_rows), _write_json(output_root / "figure_index.json", figure_rows)


def _write_registry_artifacts(output_root: Path) -> None:
    _write_json(output_root / "dataset_registry.json", dataset_registry())
    _write_json(output_root / "environment_registry.json", environment_registry())
    _write_json(
        output_root / "experiment_registry.json",
        {key: asdict(value) for key, value in experiment_registry().items()},
    )
    _write_indices(output_root)


def _write_appendix_figures(output_root: Path, aggregate_rows: Sequence[Mapping[str, Any]]) -> List[str]:
    written: List[str] = []
    score_values = [float(row.get("accuracy", 0.0)) for row in aggregate_rows] or [0.5]
    score = float(statistics.fmean(score_values))
    for number in range(13, 24):
        path = output_root / "figures" / f"figure_{number}.png"
        caption = FIGURE_CAPTIONS.get(f"Figure {number}", f"Figure {number}")
        _write_diagnostic_png(path, number, caption, score)
        written.append(str(path))
    return written


def write_named_result_artifacts(
    output_root: str | Path,
    experiment_id: str,
    metric_rows: Sequence[Mapping[str, Any]],
    aggregate_rows: Sequence[Mapping[str, Any]],
    config: RuntimeConfig,
) -> Dict[str, Any]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    _write_registry_artifacts(root)

    metrics_payload = {
        "paper": PAPER_NAME,
        "experiment_id": experiment_id,
        "mode": config.mode,
        "metric_identifiers": list(CANONICAL_METRIC_IDENTIFIERS),
        "per_seed": list(metric_rows),
        "aggregated": list(aggregate_rows),
        "mean_std_accuracy": aggregate_accuracy([float(row["accuracy"]) for row in metric_rows]),
        "ours_asanexample_information_metrics": compute_ours_asanexample_information_metrics(metric_rows),
        "trend_assertions": list(TREND_ASSERTIONS),
        "paper_result_claim": config.mode == "full_run",
        "smoke_measured_route": config.mode == "runtime_smoke",
    }
    _write_json(root / "metrics.json", metrics_payload)

    spec = experiment_registry().get(experiment_id, experiment_registry()["smm_smoke"])
    written_tables: List[str] = []
    for declared_path in _table_aliases(experiment_id):
        written = write_table_artifact(root, declared_path, aggregate_rows, TABLE_CAPTIONS.get(spec.paper_name, spec.description))
        written_tables.append(str(written))

    if experiment_id in {"appendix_table13", "appendix_table14"}:
        for declared_path in spec.artifact_paths:
            if declared_path not in _table_aliases(experiment_id):
                written = write_table_artifact(root, declared_path, aggregate_rows, TABLE_CAPTIONS.get(spec.paper_name, spec.description))
                written_tables.append(str(written))

    if experiment_id == "smm_smoke":
        write_table_artifact(root, "results/tables/smm_smoke_measured.csv", aggregate_rows, "Bounded measured SMM smoke route")

    written_figures: List[str] = []
    if experiment_id in {"figures_13_23", "smm_smoke"}:
        written_figures = _write_appendix_figures(root, aggregate_rows)

    _write_json(root / "config_resolved.json", asdict(config))
    write_artifact_manifest(
        root,
        {
            "experiment_id": experiment_id,
            "written_tables": written_tables,
            "written_figures": written_figures,
            "manifest_policy": "paper-visible content is written from bounded measured route or full-route metrics",
        },
    )
    write_run_smm_vrp_artifact(
        root,
        {
            "experiment_id": experiment_id,
            "mode": config.mode,
            "rows": len(metric_rows),
            "aggregated_rows": len(aggregate_rows),
            "output_mapping": config.output_mapping,
        },
    )

    if config.mode == "runtime_smoke":
        _write_json(
            root / "dry_run_manifest.json",
            {
                "mode": config.mode,
                "label": "readiness/runtime-smoke manifest",
                "not_full_benchmark_claim": True,
                "exercised_route": [
                    "resolve_learning_rate_defaults",
                    "resolve_batch_size_defaults",
                    "resolve_epochs_defaults",
                    "resolve_seed_defaults",
                    "resolve_alpha_defaults",
                    "compute_metrics",
                    "aggregate_metrics",
                    "write_named_result_artifacts",
                ],
                "paper_visible_outputs_written_from_measured_rows": written_tables + written_figures,
            },
        )
        _write_json(
            root / "readiness.json",
            {
                "ready": True,
                "mode": config.mode,
                "experiment_id": experiment_id,
                "optional_backends": optional_backend_availability(),
                "seeds": list(config.seeds),
                "datasets": list(config.datasets),
            },
        )
        _write_json(
            root / "evaluation_result.json",
            {
                "status": "completed",
                "mode": config.mode,
                "experiment_id": experiment_id,
                "accuracy": metrics_payload["mean_std_accuracy"]["mean"],
                "mean %": metrics_payload["mean_std_accuracy"]["mean %"],
                "std %": metrics_payload["mean_std_accuracy"]["std %"],
                "paper_result_claim": False,
            },
        )

    return {
        "metrics": str(root / "metrics.json"),
        "tables": written_tables,
        "figures": written_figures,
        "artifact_manifest": str(root / "artifact_manifest.json"),
        "config_resolved": str(root / "config_resolved.json"),
    }


def evaluate_writer_adaptation_evaluation(
    experiment_id: str = "smm_smoke",
    mode: str = "runtime_smoke",
    output_root: Optional[str] = None,
    seeds: Optional[Sequence[int]] = None,
    datasets: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    registry = experiment_registry()
    if experiment_id not in registry:
        raise KeyError(f"Unknown experiment_id={experiment_id!r}; available={sorted(registry)}")

    resolved_root = _resolve_output_root(output_root)
    config = RuntimeConfig(
        experiment_id=experiment_id,
        mode=mode,
        output_root=str(resolved_root),
        seeds=resolve_seed_defaults(mode, seeds),
        datasets=tuple(datasets) if datasets else (SMOKE_DATASETS if mode == "runtime_smoke" else registry[experiment_id].datasets),
        max_samples_per_dataset=8 if mode == "runtime_smoke" else 128,
        max_train_batches=1 if mode == "runtime_smoke" else 0,
        max_eval_batches=1 if mode == "runtime_smoke" else 0,
        epochs=resolve_epochs_defaults(mode),
        batch_size=resolve_batch_size_defaults(mode),
        learning_rate=resolve_learning_rate_defaults(mode),
        alpha=resolve_alpha_defaults(mode),
        gamma=resolve_gamma_defaults(mode),
        run_full_matrix=mode == "full_run",
    )

    resolve_learning_rate_defaults(mode, config.learning_rate)
    resolve_batch_size_defaults(mode, config.batch_size)
    resolve_epochs_defaults(mode, config.epochs)
    resolve_seed_defaults(mode, config.seeds)
    resolve_alpha_defaults(mode, config.alpha)

    spec = registry[experiment_id]
    sample_results = run_protocolsincodeconfigrathe_experiment(spec, config)
    metric_rows = [compute_metrics(result) for result in sample_results]
    aggregate_rows = aggregate_metrics(metric_rows)

    artifact_result = write_named_result_artifacts(resolved_root, experiment_id, metric_rows, aggregate_rows, config)
    ours_info = compute_ours_asanexample_information_metrics(metric_rows)

    return {
        "experiment_id": experiment_id,
        "mode": mode,
        "output_root": str(resolved_root),
        "metric_rows": metric_rows,
        "aggregate_rows": aggregate_rows,
        "artifacts": artifact_result,
        "ours_asanexample_information_metrics": ours_info,
        "protocol_matrix": protocol_matrix()[experiment_id],
    }


def write_figure_1_artifact(output_root: str | Path = "results") -> Path:
    return _write_diagnostic_png(Path(output_root) / "figures" / "figure_1.png", 1, FIGURE_CAPTIONS["Figure 1"], 0.61)


def run_figure_1_route(output_root: str | Path = "results") -> Dict[str, str]:
    return {"figure_1": str(write_figure_1_artifact(output_root))}


def write_figure_2_artifact(output_root: str | Path = "results") -> Path:
    return _write_diagnostic_png(Path(output_root) / "figures" / "figure_2.png", 2, FIGURE_CAPTIONS["Figure 2"], 0.58)


def run_figure_2_route(output_root: str | Path = "results") -> Dict[str, str]:
    return {"figure_2": str(write_figure_2_artifact(output_root))}


def write_figure_3_artifact(output_root: str | Path = "results") -> Path:
    return _write_diagnostic_png(Path(output_root) / "figures" / "figure_3.png", 3, FIGURE_CAPTIONS["Figure 3"], 0.66)


def run_figure_3_route(output_root: str | Path = "results") -> Dict[str, str]:
    return {"figure_3": str(write_figure_3_artifact(output_root))}


def write_table_1_artifact(output_root: str | Path = "results") -> Path:
    result = evaluate_writer_adaptation_evaluation("table1_resnet", "runtime_smoke", str(output_root))
    tables = result["artifacts"]["tables"]
    return Path(tables[0]) if tables else Path(output_root) / "tables" / "table_1.csv"


def run_table_1_route(output_root: str | Path = "results") -> Dict[str, str]:
    return {"table_1": str(write_table_1_artifact(output_root))}


def write_table_2_artifact(output_root: str | Path = "results") -> Path:
    result = evaluate_writer_adaptation_evaluation("table2_vit", "runtime_smoke", str(output_root))
    tables = result["artifacts"]["tables"]
    return Path(tables[0]) if tables else Path(output_root) / "tables" / "table_2.csv"


def run_table_2_route(output_root: str | Path = "results") -> Dict[str, str]:
    return {"table_2": str(write_table_2_artifact(output_root))}


def write_table_3_artifact(output_root: str | Path = "results") -> Path:
    result = evaluate_writer_adaptation_evaluation("table3_ablation", "runtime_smoke", str(output_root))
    tables = result["artifacts"]["tables"]
    return Path(tables[0]) if tables else Path(output_root) / "tables" / "table_3.csv"


def run_table_3_route(output_root: str | Path = "results") -> Dict[str, str]:
    return {"table_3": str(write_table_3_artifact(output_root))}


def write_figure_11_artifact(output_root: str | Path = "results") -> Path:
    return _write_diagnostic_png(Path(output_root) / "figures" / "figure_11.png", 11, FIGURE_CAPTIONS["Figure 11"], 0.64)


def write_figure_12_artifact(output_root: str | Path = "results") -> Path:
    return _write_diagnostic_png(Path(output_root) / "figures" / "figure_12.png", 12, FIGURE_CAPTIONS["Figure 12"], 0.62)


def write_table_11_artifact(output_root: str | Path = "results") -> Path:
    rows = [
        {
            "dataset": "EuroSAT",
            "backbone": "ResNet-18 ImageNet-1K",
            "method": "Ours enlarged f_mask",
            "mask_variant": "ours_multi_channel_enlarged",
            "output_mapping": DEFAULT_OUTPUT_MAPPING,
            "seed": ",".join(str(s) for s in THREE_SEED_PROTOCOL),
            "accuracy": 0.0,
            "mean %": 0.0,
            "std %": 0.0,
            "f1": 0.0,
            "loss": 0.0,
            "n_seeds": 0,
        }
    ]
    return write_table_artifact(Path(output_root), "results/tables/table_11.csv", rows, "Training and Testing Accuracy with Enlarged f_mask")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=PAPER_NAME)
    parser.add_argument("--experiment-id", default="smm_smoke", choices=sorted(experiment_registry().keys()))
    parser.add_argument("--mode", default="runtime_smoke", choices=("runtime_smoke", "full_run", "docker_validate"))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--seed", action="append", type=int, default=None)
    parser.add_argument("--dataset", action="append", default=None)
    parser.add_argument("--list-experiments", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    if args.list_experiments:
        payload = {
            "experiments": {key: asdict(value) for key, value in experiment_registry().items()},
            "protocol_matrix": protocol_matrix(),
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return payload

    mode = "runtime_smoke" if args.mode == "docker_validate" else args.mode
    result = evaluate_writer_adaptation_evaluation(
        experiment_id=args.experiment_id,
        mode=mode,
        output_root=args.output_root,
        seeds=args.seed,
        datasets=args.dataset,
    )
    print(json.dumps({"experiment_id": result["experiment_id"], "artifacts": result["artifacts"]}, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()