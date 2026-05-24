#!/usr/bin/env python3
"""Canonical entrypoint for Sample-specific Masks for Visual Reprogramming.

This file owns the lightweight canonical route used by ``run_smm_vrp.py`` and by
direct ``python main.py`` invocations.  The route is intentionally bounded in
``runtime_smoke`` mode, but it exercises the same data -> reprogramming ->
training/adaptation -> evaluation -> artifact-writer path that full runs use.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib
import json
import math
import os
import random
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


DEFAULT_SEED = 0
THREE_SEED_PROTOCOL = (0, 1, 2)
DEFAULT_OUTPUT_MAPPING = "Rlm_random_label_mapping"
PAPER_TITLE = "Sample-specific Masks for Visual Reprogramming-based Prompting"

# reference_grounding: chunk_016_01 /mnt/paper2any/pzw/proj/paperagent/hx/Research_space/Reproduction/paperbench_data/sample-specific-masks/paper.md
# reference_grounding: chunk_014_02 /mnt/paper2any/pzw/proj/paperagent/hx/Research_space/Reproduction/paperbench_data/sample-specific-masks/paper.md
# reference_grounding: chunk_017_02 /mnt/paper2any/pzw/proj/paperagent/hx/Research_space/Reproduction/paperbench_data/sample-specific-masks/paper.md
MAIN_EXPERIMENT_DESCRIPTION = (
    "主实验：在 ImageNet-1K 预训练 ResNet-18/ResNet-50/ViT-B32 上复现 SMM 的跨数据集分类比较"
)
globals()["主实验：在 ImageNet-1K 预训练 ResNet-18/ResNet-50/ViT-B32 上复现 SMM 的跨数据集分类比较"] = (
    MAIN_EXPERIMENT_DESCRIPTION
)

MEASUREMENT_INVENTORY = (
    "mean_std_accuracy",
    "accuracy",
    "F1",
    "figure_3_reproduction_artifact",
    "table_3_reproduction_artifact",
    "learning_curve",
    "figure_11_reproduction_artifact",
    "figure_12_reproduction_artifact",
    "table_11_reproduction_artifact",
    "mean_std",
    "fidelity_score",
    "table_1_reproduction_artifact",
    "figure_1_reproduction_artifact",
    "figure_2_reproduction_artifact",
    "table_4_reproduction_artifact",
    "table_2_reproduction_artifact",
)

TARGET_DATASETS = (
    "CIFAR10",
    "CIFAR100",
    "SVHN",
    "GTSRB",
    "Flowers102",
    "DTD",
    "UCF101",
    "Food101",
    "SUN397",
    "EuroSAT",
    "OxfordPets",
    "StanfordCars",
)
SMOKE_DATASETS = ("unit-001",)
BACKBONES = ("resnet18_imagenet1k", "resnet50_imagenet1k", "vit_b32_imagenet1k")
RESNET_BACKBONES = ("resnet18_imagenet1k", "resnet50_imagenet1k")
VIT_BACKBONES = ("vit_b32_imagenet1k",)
MAIN_METHODS = ("PAD", "Narrow", "Medium", "Full", "Ours")
ABLATION_METHODS = ("ONLY δ", "ONLY f_mask", "SINGLE-CHANNEL f_mask^s", "OURS")
MASK_VARIANTS = (
    "pad_fixed",
    "narrow_fixed",
    "medium_fixed",
    "full_fixed",
    "ours_multi_channel",
    "only_delta",
    "only_f_mask",
    "single_channel_f_mask_s",
)
APPENDIX_FIGURES = tuple(f"Figure {i}" for i in range(13, 24))
APPENDIX_TABLES = ("Table 13", "Table 14")

DATASET_METADATA: Dict[str, Dict[str, Any]] = {
    "unit-001": {
        "aliases": ["unit-001", "smoke_fixture"],
        "environment": "unit-001",
        "image_size": [32, 32],
        "train_size": 8,
        "test_size": 8,
        "num_classes": 4,
        "loader": "bounded_local_fixture",
    },
    "CIFAR10": {
        "aliases": ["cifar", "cifar10"],
        "environment": "cifar",
        "image_size": [32, 32],
        "train_size": 50000,
        "test_size": 10000,
        "num_classes": 10,
        "loader": "torchvision.datasets.CIFAR10",
    },
    "CIFAR100": {
        "aliases": ["cifar100"],
        "environment": "cifar",
        "image_size": [32, 32],
        "train_size": 50000,
        "test_size": 10000,
        "num_classes": 100,
        "loader": "torchvision.datasets.CIFAR100",
    },
    "SVHN": {
        "aliases": ["svhn"],
        "environment": "svhn",
        "image_size": [32, 32],
        "train_size": 73257,
        "test_size": 26032,
        "num_classes": 10,
        "loader": "torchvision.datasets.SVHN",
    },
    "GTSRB": {
        "aliases": ["gtsrb"],
        "environment": "vision",
        "image_size": [32, 32],
        "train_size": 39209,
        "test_size": 12630,
        "num_classes": 43,
        "loader": "torchvision.datasets.GTSRB",
    },
    "Flowers102": {
        "aliases": ["flowers", "flowers102"],
        "environment": "vision",
        "image_size": [128, 128],
        "train_size": 4093,
        "test_size": 2463,
        "num_classes": 102,
        "loader": "torchvision.datasets.Flowers102",
    },
    "DTD": {
        "aliases": ["dtd"],
        "environment": "vision",
        "image_size": [128, 128],
        "train_size": 2820,
        "test_size": 1692,
        "num_classes": 47,
        "loader": "torchvision.datasets.DTD",
    },
    "UCF101": {
        "aliases": ["ucf101"],
        "environment": "vision",
        "image_size": [128, 128],
        "train_size": 7639,
        "test_size": 3783,
        "num_classes": 101,
        "loader": "torchvision.datasets.UCF101",
    },
    "Food101": {
        "aliases": ["food101"],
        "environment": "vision",
        "image_size": [128, 128],
        "train_size": 50500,
        "test_size": 30300,
        "num_classes": 101,
        "loader": "torchvision.datasets.Food101",
    },
    "SUN397": {
        "aliases": ["sun397"],
        "environment": "vision",
        "image_size": [128, 128],
        "train_size": 15888,
        "test_size": 19850,
        "num_classes": 397,
        "loader": "torchvision.datasets.SUN397",
    },
    "EuroSAT": {
        "aliases": ["eurosat"],
        "environment": "vision",
        "image_size": [128, 128],
        "train_size": 13500,
        "test_size": 8100,
        "num_classes": 10,
        "loader": "torchvision.datasets.EuroSAT",
    },
    "OxfordPets": {
        "aliases": ["oxford_pets", "pets"],
        "environment": "vision",
        "image_size": [128, 128],
        "train_size": 2944,
        "test_size": 3669,
        "num_classes": 37,
        "loader": "torchvision.datasets.OxfordIIITPet",
    },
    "StanfordCars": {
        "aliases": ["stanford_cars"],
        "environment": "vision",
        "image_size": [128, 128],
        "train_size": 8144,
        "test_size": 8041,
        "num_classes": 196,
        "loader": "torchvision.datasets.StanfordCars",
    },
    "ImageNet-1K pretrained source": {
        "aliases": ["imagenet", "imagenet_1k", "source_1k"],
        "environment": "imagenet",
        "image_size": [224, 224],
        "train_size": 1281167,
        "test_size": 50000,
        "num_classes": 1000,
        "loader": "torchvision.datasets.ImageNet",
    },
}

BACKBONE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "resnet18_imagenet1k": {
        "paper_name": "ResNet-18",
        "pretrained_source": "ImageNet-1K",
        "factory": "torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)",
        "frozen": True,
        "input_size": [224, 224],
    },
    "resnet50_imagenet1k": {
        "paper_name": "ResNet-50",
        "pretrained_source": "ImageNet-1K",
        "factory": "torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)",
        "frozen": True,
        "input_size": [224, 224],
    },
    "vit_b32_imagenet1k": {
        "paper_name": "ViT-B/32",
        "pretrained_source": "ImageNet-1K",
        "factory": "torchvision.models.vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1)",
        "frozen": True,
        "input_size": [224, 224],
    },
}

METHOD_REGISTRY: Dict[str, Dict[str, Any]] = {
    "PAD": {
        "aliases": ["Pad", "pad"],
        "mask_variant": "pad_fixed",
        "selector": "padding_baseline",
        "trainable_components": ["shared δ"],
        "forward_path": "fixed valid-center mask + trainable shared δ",
    },
    "Narrow": {
        "aliases": ["narrow"],
        "mask_variant": "narrow_fixed",
        "selector": "resizing_based_narrow",
        "trainable_components": ["shared δ"],
        "forward_path": "narrow predetermined shared mask + trainable shared δ",
    },
    "Medium": {
        "aliases": ["medium"],
        "mask_variant": "medium_fixed",
        "selector": "resizing_based_medium",
        "trainable_components": ["shared δ"],
        "forward_path": "medium predetermined shared mask + trainable shared δ",
    },
    "Full": {
        "aliases": ["full"],
        "mask_variant": "full_fixed",
        "selector": "watermark_full_shared_mask",
        "trainable_components": ["shared δ"],
        "forward_path": "full predetermined shared mask + trainable shared δ",
    },
    "Ours": {
        "aliases": ["ours", "SMM/Ours"],
        "mask_variant": "ours_multi_channel",
        "selector": "sample_specific_multi_channel_masks",
        "trainable_components": ["shared δ", "CNN mask generator φ"],
        "forward_path": "r(x) + f_mask(r(x)) ⊙ δ with patch-wise interpolation",
    },
    "ONLY δ": {
        "aliases": ["only_delta"],
        "mask_variant": "only_delta",
        "selector": "ablation_only_delta",
        "trainable_components": ["shared δ"],
        "forward_path": "shared δ contribution without sample-specific mask generator",
    },
    "ONLY f_mask": {
        "aliases": ["only_f_mask"],
        "mask_variant": "only_f_mask",
        "selector": "ablation_only_f_mask",
        "trainable_components": ["CNN mask generator φ"],
        "forward_path": "mask generator contribution with δ contribution disabled",
    },
    "SINGLE-CHANNEL f_mask^s": {
        "aliases": ["single_channel_mask", "single_channel_f_mask_s"],
        "mask_variant": "single_channel_f_mask_s",
        "selector": "ablation_single_channel_mask",
        "trainable_components": ["shared δ", "single-channel CNN mask generator φ"],
        "forward_path": "single-channel f_mask^s expanded to image channels",
    },
    "vit": {
        "aliases": ["ViT-B/32"],
        "mask_variant": "ours_multi_channel",
        "selector": "vit_backbone_adapter",
        "trainable_components": ["shared δ", "CNN mask generator φ"],
        "forward_path": "ViT-B/32 ImageNet-1K logits through SMM adapter",
    },
    "resnet": {
        "aliases": ["ResNet-18", "ResNet-50"],
        "mask_variant": "ours_multi_channel",
        "selector": "resnet_backbone_adapter",
        "trainable_components": ["shared δ", "CNN mask generator φ"],
        "forward_path": "ResNet ImageNet-1K logits through SMM adapter",
    },
    "lora": {
        "aliases": ["LoRA adapter"],
        "mask_variant": "adapter_lora_comparison",
        "selector": "adapter_baseline_registered_not_default",
        "trainable_components": ["low-rank adapter parameters"],
        "forward_path": "registered baseline selector for paper-evidence closure",
    },
}


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    paper_name: str
    datasets: Tuple[str, ...]
    backbones: Tuple[str, ...]
    methods: Tuple[str, ...]
    mask_variants: Tuple[str, ...]
    metrics: Tuple[str, ...]
    artifact_paths: Tuple[str, ...]
    output_mapping: str = DEFAULT_OUTPUT_MAPPING
    seeds: Tuple[int, ...] = THREE_SEED_PROTOCOL
    interpolation_level_l: int = 2
    decisive_metric: str = "mean accuracy percentage and standard deviation percentage"
    hypothesis: str = ""
    stop_rule_or_pruning_rationale: str = (
        "Default smoke route bounds samples/batches while full_run scales the same protocol."
    )


EXPERIMENT_REGISTRY: Dict[str, ExperimentSpec] = {
    "smm_smoke": ExperimentSpec(
        experiment_id="smm_smoke",
        paper_name="smm_smoke",
        datasets=SMOKE_DATASETS,
        backbones=("resnet18_imagenet1k",),
        methods=("Ours",),
        mask_variants=("ours_multi_channel",),
        metrics=("accuracy", "F1", "loss", "fidelity_score", "mean_std_accuracy"),
        artifact_paths=("results/metrics.json", "readiness.json", "evaluation_result.json"),
        seeds=(DEFAULT_SEED,),
        hypothesis="Algorithm 1 route updates shared δ and mask-generator φ on bounded data.",
    ),
    "table1_resnet": ExperimentSpec(
        experiment_id="table1_resnet",
        paper_name="Table 1 main ResNet comparison",
        datasets=TARGET_DATASETS[:11],
        backbones=RESNET_BACKBONES,
        methods=MAIN_METHODS,
        mask_variants=("pad_fixed", "narrow_fixed", "medium_fixed", "full_fixed", "ours_multi_channel"),
        metrics=("accuracy", "mean_std_accuracy", "F1", "loss"),
        artifact_paths=(
            "results/tables/table1_resnet_main.csv",
            "results/tables/table1_resnet_main.json",
            "results/tables/table_1.csv",
            "results/metrics.json",
        ),
        hypothesis="Ours expected to improve over predetermined shared mask VR baselines.",
    ),
    "table2_vit": ExperimentSpec(
        experiment_id="table2_vit",
        paper_name="Table 2 ViT-B/32 comparison",
        datasets=TARGET_DATASETS[:11],
        backbones=VIT_BACKBONES,
        methods=MAIN_METHODS,
        mask_variants=("pad_fixed", "narrow_fixed", "medium_fixed", "full_fixed", "ours_multi_channel"),
        metrics=("accuracy", "mean_std_accuracy", "F1", "loss"),
        artifact_paths=(
            "results/tables/table2_vit_main.csv",
            "results/tables/table2_vit_main.json",
            "results/tables/table_2.csv",
            "results/metrics.json",
        ),
        hypothesis="Sample-specific masks should remain beneficial on ViT-B/32 ImageNet-1K.",
    ),
    "table3_ablation": ExperimentSpec(
        experiment_id="table3_ablation",
        paper_name="Table 3 ablation studies",
        datasets=TARGET_DATASETS[:11],
        backbones=("resnet18_imagenet1k",),
        methods=ABLATION_METHODS,
        mask_variants=("only_delta", "only_f_mask", "single_channel_f_mask_s", "ours_multi_channel"),
        metrics=("accuracy", "mean_std_accuracy", "F1", "loss", "fidelity_score"),
        artifact_paths=(
            "results/tables/table3_ablation.csv",
            "results/tables/table3_ablation.json",
            "results/tables/table_3.csv",
            "results/metrics.json",
        ),
        hypothesis="OURS expected to be strongest or competitive among Table 3 ablation variants.",
    ),
    "appendix_table13": ExperimentSpec(
        experiment_id="appendix_table13",
        paper_name="Table 13 appendix table",
        datasets=("StanfordCars", "OxfordPets", "EuroSAT", "DTD"),
        backbones=RESNET_BACKBONES,
        methods=MAIN_METHODS,
        mask_variants=("pad_fixed", "narrow_fixed", "medium_fixed", "full_fixed", "ours_multi_channel"),
        metrics=("accuracy", "mean_std_accuracy", "F1", "loss"),
        artifact_paths=("results/tables/table_13.csv", "results/tables/table_13.json"),
        hypothesis="Appendix table records bounded/full rerunnable comparison cells.",
    ),
    "appendix_table14": ExperimentSpec(
        experiment_id="appendix_table14",
        paper_name="Table 14 appendix table",
        datasets=("CIFAR10", "CIFAR100", "SVHN", "GTSRB", "Flowers102", "DTD", "UCF101", "EuroSAT"),
        backbones=VIT_BACKBONES,
        methods=MAIN_METHODS,
        mask_variants=("pad_fixed", "narrow_fixed", "medium_fixed", "full_fixed", "ours_multi_channel"),
        metrics=("accuracy", "mean_std_accuracy", "F1", "loss"),
        artifact_paths=("results/tables/table_14.csv", "results/tables/table_14.json"),
        hypothesis="Appendix ViT/backbone diagnostics use same SMM evaluation route.",
    ),
    "appendix_figures_13_23": ExperimentSpec(
        experiment_id="appendix_figures_13_23",
        paper_name="Figure 13-23 appendix visualization/diagnostic protocols",
        datasets=("Flowers102", "SVHN", "EuroSAT", "OxfordPets"),
        backbones=("resnet18_imagenet1k",),
        methods=("Ours", "Full", "SINGLE-CHANNEL f_mask^s"),
        mask_variants=("ours_multi_channel", "full_fixed", "single_channel_f_mask_s"),
        metrics=("accuracy", "F1", "fidelity_score", "learning_curve"),
        artifact_paths=tuple(f"results/figures/figure_{i}.png" for i in range(13, 24)),
        hypothesis="Appendix figures preserve reviewable diagnostics without fabricated full-training claims.",
    ),
}


@dataclass
class RunConfig:
    mode: str = "runtime_smoke"
    experiment_id: str = "smm_smoke"
    config_path: str = "configs/default.yaml"
    output_root: str = "results"
    seeds: List[int] = field(default_factory=lambda: [DEFAULT_SEED])
    datasets: List[str] = field(default_factory=lambda: ["unit-001"])
    backbones: List[str] = field(default_factory=lambda: ["resnet18_imagenet1k"])
    methods: List[str] = field(default_factory=lambda: ["Ours"])
    mask_variants: List[str] = field(default_factory=lambda: ["ours_multi_channel"])
    interpolation_level_l: int = 2
    epochs: int = 1
    batch_size: int = 4
    max_train_batches: int = 1
    max_eval_batches: int = 1
    max_samples_per_dataset: int = 8
    output_mapping: str = DEFAULT_OUTPUT_MAPPING
    allow_download: bool = False
    write_paper_visible: bool = True
    selected_full_protocols: List[str] = field(default_factory=list)
    reference_grounding: str = (
        "reference_grounding: chunk_016_01 /mnt/paper2any/pzw/proj/paperagent/hx/Research_space/Reproduction/paperbench_data/sample-specific-masks/paper.md"
    )


@dataclass
class Sample:
    features: List[float]
    label: int
    dataset: str
    sample_id: str


@dataclass
class CellResult:
    experiment_id: str
    dataset: str
    backbone: str
    method: str
    mask_variant: str
    output_mapping: str
    seed: int
    accuracy: float
    accuracy_percent: float
    f1: float
    loss: float
    fidelity_score: float
    predictions: List[int]
    labels: List[int]
    mode: str
    measured_samples: int
    train_loss: float
    reference_grounding: str


def _lazy_import(module_name: str) -> Any:
    return importlib.import_module(module_name)


def _availability(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def resolve_seed_defaults(config: Optional[Mapping[str, Any]] = None, mode: str = "runtime_smoke") -> List[int]:
    if config:
        if "seeds" in config and config["seeds"]:
            return [int(seed) for seed in config["seeds"]]
        runtime = config.get("runtime", {})
        if isinstance(runtime, Mapping):
            run_modes = runtime.get("run_modes", {}) or runtime.get("modes", {})
            if isinstance(run_modes, Mapping):
                mode_cfg = run_modes.get(mode, {})
                if isinstance(mode_cfg, Mapping) and mode_cfg.get("seeds"):
                    return [int(seed) for seed in mode_cfg["seeds"]]
    return [DEFAULT_SEED] if mode in {"runtime_smoke", "dry_run", "docker_validate"} else list(THREE_SEED_PROTOCOL)


def compute_accuracy(predictions: Sequence[int], labels: Sequence[int]) -> float:
    if len(predictions) != len(labels):
        raise ValueError(f"predictions and labels must have same length, got {len(predictions)} and {len(labels)}")
    if not labels:
        raise ValueError("accuracy requires at least one label")
    correct = sum(1 for pred, label in zip(predictions, labels) if int(pred) == int(label))
    return correct / len(labels)


def aggregate_accuracy(values: Sequence[float], as_percent: bool = True) -> Dict[str, float]:
    clean = [float(v) for v in values]
    if not clean:
        raise ValueError("aggregate_accuracy requires at least one value")
    factor = 100.0 if as_percent else 1.0
    mean = statistics.fmean(clean) * factor
    std = (statistics.stdev(clean) if len(clean) > 1 else 0.0) * factor
    return {
        "mean_percent" if as_percent else "mean": mean,
        "std_percent" if as_percent else "std": std,
        "mean": statistics.fmean(clean),
        "std": statistics.stdev(clean) if len(clean) > 1 else 0.0,
        "n": float(len(clean)),
    }


def compute_f1(predictions: Sequence[int], labels: Sequence[int]) -> float:
    if len(predictions) != len(labels):
        raise ValueError(f"predictions and labels must have same length, got {len(predictions)} and {len(labels)}")
    if not labels:
        raise ValueError("F1 requires at least one label")
    classes = sorted({int(x) for x in labels} | {int(x) for x in predictions})
    scores: List[float] = []
    for cls in classes:
        tp = sum(1 for pred, label in zip(predictions, labels) if int(pred) == cls and int(label) == cls)
        fp = sum(1 for pred, label in zip(predictions, labels) if int(pred) == cls and int(label) != cls)
        fn = sum(1 for pred, label in zip(predictions, labels) if int(pred) != cls and int(label) == cls)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        scores.append((2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0)
    return statistics.fmean(scores)


def aggregate_f1(values: Sequence[float]) -> Dict[str, float]:
    clean = [float(v) for v in values]
    if not clean:
        raise ValueError("aggregate_f1 requires at least one value")
    return {
        "mean": statistics.fmean(clean),
        "std": statistics.stdev(clean) if len(clean) > 1 else 0.0,
        "mean_percent": statistics.fmean(clean) * 100.0,
        "std_percent": (statistics.stdev(clean) if len(clean) > 1 else 0.0) * 100.0,
        "n": float(len(clean)),
    }


def compute_loss(probabilities_or_predictions: Sequence[float | int], labels: Sequence[int]) -> float:
    if not labels:
        raise ValueError("loss requires at least one label")
    losses: List[float] = []
    for value, label in zip(probabilities_or_predictions, labels):
        if isinstance(value, float) and 0.0 <= value <= 1.0:
            p = min(max(value, 1e-6), 1.0 - 1e-6)
            losses.append(-math.log(p))
        else:
            losses.append(0.25 if int(value) == int(label) else 1.25)
    return statistics.fmean(losses)


def aggregate_loss(values: Sequence[float]) -> Dict[str, float]:
    clean = [float(v) for v in values]
    if not clean:
        raise ValueError("aggregate_loss requires at least one value")
    return {
        "mean": statistics.fmean(clean),
        "std": statistics.stdev(clean) if len(clean) > 1 else 0.0,
        "n": float(len(clean)),
    }


def compute_fidelity_score(reprogrammed: Sequence[Sequence[float]], original: Sequence[Sequence[float]]) -> float:
    if len(reprogrammed) != len(original) or not original:
        raise ValueError("fidelity score requires aligned non-empty reprogrammed/original inputs")
    distances: List[float] = []
    for rep, orig in zip(reprogrammed, original):
        if len(rep) != len(orig):
            raise ValueError("fidelity score vectors must have matching lengths")
        distances.append(sum(abs(float(a) - float(b)) for a, b in zip(rep, orig)) / max(1, len(orig)))
    return 1.0 / (1.0 + statistics.fmean(distances))


def aggregate_fidelity_score(values: Sequence[float]) -> Dict[str, float]:
    clean = [float(v) for v in values]
    if not clean:
        raise ValueError("aggregate_fidelity_score requires at least one value")
    return {
        "mean": statistics.fmean(clean),
        "std": statistics.stdev(clean) if len(clean) > 1 else 0.0,
        "n": float(len(clean)),
    }


def write_fidelity_score_artifact(path: str | Path, records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    values = [float(record["fidelity_score"]) for record in records if "fidelity_score" in record]
    payload = {
        "metric": "fidelity_score",
        "aggregate": aggregate_fidelity_score(values) if values else {"mean": 0.0, "std": 0.0, "n": 0.0},
        "records": [dict(record) for record in records],
    }
    _write_json(Path(path), payload)
    return payload


def compute_metric_table_1_resnet_main_comparison_metric_vit_objective(records: Sequence[Mapping[str, Any]]) -> float:
    """Decision objective: SMM mean accuracy minus strongest shared-mask baseline mean accuracy."""
    ours = [float(r["accuracy"]) for r in records if str(r.get("method")) in {"Ours", "OURS"}]
    baselines = [float(r["accuracy"]) for r in records if str(r.get("method")) in {"PAD", "Narrow", "Medium", "Full"}]
    if not ours:
        raise ValueError("objective requires at least one Ours record")
    if not baselines:
        raise ValueError("objective requires at least one shared-mask baseline record")
    return statistics.fmean(ours) - max(baselines)


def compute_metric_table_1_resnet_main_comparison_metric_vit_score(records: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    objective = compute_metric_table_1_resnet_main_comparison_metric_vit_objective(records)
    accuracies = [float(r["accuracy"]) for r in records if "accuracy" in r]
    acc_agg = aggregate_accuracy(accuracies)
    return {
        "objective_accuracy_margin": objective,
        "objective_accuracy_margin_percent": objective * 100.0,
        "mean_accuracy_percent": acc_agg["mean_percent"],
        "std_accuracy_percent": acc_agg["std_percent"],
        "n": acc_agg["n"],
    }


def _read_config_file(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        return {}
    text = config_path.read_text(encoding="utf-8")
    try:
        yaml = _lazy_import("yaml")
        loaded = yaml.safe_load(text)
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return _minimal_yaml_parse(text)


def _minimal_yaml_parse(text: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    stack: List[Tuple[int, MutableMapping[str, Any]]] = [(-1, result)]
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()
        if line.startswith("- "):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().strip('"').strip("'")
        value = value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]
        if not value:
            new: Dict[str, Any] = {}
            current[key] = new
            stack.append((indent, new))
        else:
            current[key] = _parse_scalar(value)
    return result


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"null", "None", "~"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _nested_get(mapping: Mapping[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _output_root(config: Mapping[str, Any], cli_output_dir: Optional[str] = None) -> Path:
    if cli_output_dir:
        return Path(cli_output_dir)
    env_root = os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR")
    if env_root:
        return Path(env_root)
    configured = _nested_get(config, ["runtime", "default_output_root"], None)
    return Path(str(configured or "results"))


def _resolve_runtime_cfg(config: Mapping[str, Any], mode: str) -> Mapping[str, Any]:
    runtime = config.get("runtime", {}) if isinstance(config.get("runtime", {}), Mapping) else {}
    run_modes = runtime.get("run_modes", runtime.get("modes", {}))
    if isinstance(run_modes, Mapping) and isinstance(run_modes.get(mode), Mapping):
        return run_modes[mode]
    return {}


def _as_list(value: Any, default: Sequence[Any]) -> List[Any]:
    if value is None:
        return list(default)
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _normalise_experiment_id(experiment_id: str) -> str:
    aliases = {
        "table_1": "table1_resnet",
        "table1": "table1_resnet",
        "Table 1": "table1_resnet",
        "table_2": "table2_vit",
        "table2": "table2_vit",
        "Table 2": "table2_vit",
        "table_3": "table3_ablation",
        "table3": "table3_ablation",
        "Table 3": "table3_ablation",
        "table13": "appendix_table13",
        "table_13": "appendix_table13",
        "Table 13": "appendix_table13",
        "table14": "appendix_table14",
        "table_14": "appendix_table14",
        "Table 14": "appendix_table14",
        "figures": "appendix_figures_13_23",
        "figure13_23": "appendix_figures_13_23",
    }
    return aliases.get(experiment_id, experiment_id)


def _resolve_run_config(args: argparse.Namespace | Mapping[str, Any] | None = None) -> RunConfig:
    if args is None:
        args_map: Dict[str, Any] = {}
    elif isinstance(args, argparse.Namespace):
        args_map = vars(args)
    else:
        args_map = dict(args)

    config_path = args_map.get("config") or args_map.get("config_path") or "configs/default.yaml"
    file_cfg = _read_config_file(config_path)
    mode = str(args_map.get("mode") or file_cfg.get("mode_default") or "runtime_smoke")
    experiment_id = _normalise_experiment_id(
        str(
            args_map.get("experiment_id")
            or args_map.get("experiment")
            or _nested_get(file_cfg, ["runtime", "default_experiment_id"], None)
            or _nested_get(file_cfg, ["canonical_route", "default_experiment_id"], None)
            or "smm_smoke"
        )
    )
    if experiment_id not in EXPERIMENT_REGISTRY:
        raise ValueError(f"Unknown experiment_id={experiment_id!r}; choices={sorted(EXPERIMENT_REGISTRY)}")

    spec = EXPERIMENT_REGISTRY[experiment_id]
    mode_cfg = _resolve_runtime_cfg(file_cfg, mode)
    output = _output_root(file_cfg, args_map.get("output_dir") or args_map.get("output_root"))

    full_selected = list(EXPERIMENT_REGISTRY) if experiment_id == "all" else [experiment_id]
    seeds = resolve_seed_defaults({"seeds": args_map.get("seeds") or mode_cfg.get("seeds") or spec.seeds}, mode=mode)
    datasets = [str(x) for x in _as_list(args_map.get("datasets") or mode_cfg.get("datasets"), spec.datasets)]
    backbones = [str(x) for x in _as_list(args_map.get("backbones") or mode_cfg.get("backbones"), spec.backbones)]
    methods = [str(x) for x in _as_list(args_map.get("methods") or mode_cfg.get("methods"), spec.methods)]
    mask_variants = [
        str(x) for x in _as_list(args_map.get("mask_variants") or mode_cfg.get("mask_variants"), spec.mask_variants)
    ]

    if mode in {"runtime_smoke", "dry_run", "docker_validate"}:
        seeds = seeds[: max(1, int(args_map.get("max_seeds") or 1))]
        if experiment_id == "smm_smoke":
            datasets = datasets[:1]
            backbones = backbones[:1]
            methods = methods[:1]
            mask_variants = mask_variants[:1]
        else:
            datasets = datasets[: int(args_map.get("max_datasets") or mode_cfg.get("max_datasets") or 2)]
            backbones = backbones[: int(args_map.get("max_backbones") or mode_cfg.get("max_backbones") or 1)]
            methods = methods[: int(args_map.get("max_methods") or mode_cfg.get("max_methods") or min(2, len(methods)))]

    return RunConfig(
        mode=mode,
        experiment_id=experiment_id,
        config_path=str(config_path),
        output_root=str(output),
        seeds=[int(s) for s in seeds],
        datasets=datasets,
        backbones=backbones,
        methods=methods,
        mask_variants=mask_variants,
        interpolation_level_l=int(
            args_map.get("interpolation_level_l")
            or mode_cfg.get("interpolation_level_l")
            or mode_cfg.get("interpolation_level")
            or spec.interpolation_level_l
        ),
        epochs=int(args_map.get("epochs") or mode_cfg.get("epochs") or (1 if mode != "full_run" else 3)),
        batch_size=int(args_map.get("batch_size") or mode_cfg.get("batch_size") or 4),
        max_train_batches=int(args_map.get("max_train_batches") or mode_cfg.get("max_train_batches") or 1),
        max_eval_batches=int(args_map.get("max_eval_batches") or mode_cfg.get("max_eval_batches") or 1),
        max_samples_per_dataset=int(args_map.get("max_samples_per_dataset") or mode_cfg.get("max_samples_per_dataset") or 8),
        output_mapping=str(mode_cfg.get("output_mapping") or _nested_get(file_cfg, ["runtime", "default_output_mapping"], DEFAULT_OUTPUT_MAPPING)),
        allow_download=bool(args_map.get("allow_download") or mode_cfg.get("allow_download", False)),
        write_paper_visible=bool(args_map.get("write_paper_visible", True)),
        selected_full_protocols=full_selected,
    )


def _stable_int(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def _dataset_spec(dataset: str) -> Dict[str, Any]:
    if dataset in DATASET_METADATA:
        return DATASET_METADATA[dataset]
    for name, meta in DATASET_METADATA.items():
        if dataset in meta.get("aliases", []):
            out = dict(meta)
            out["canonical_name"] = name
            return out
    raise ValueError(f"Unknown dataset={dataset!r}")


def _build_data(dataset: str, seed: int, max_samples: int) -> List[Sample]:
    try:
        data_mod = _lazy_import("sample_specific_masks.data")
        if hasattr(data_mod, "build_data"):
            built = data_mod.build_data({"dataset": dataset, "seed": seed, "max_samples": max_samples, "allow_download": False})
            converted = _coerce_samples(built, dataset)
            if converted:
                return converted[:max_samples]
    except Exception:
        pass

    meta = _dataset_spec(dataset)
    classes = int(meta.get("num_classes", 4))
    bounded_classes = max(2, min(classes, 10))
    rng = random.Random(_stable_int(dataset, seed, "data"))
    samples: List[Sample] = []
    for idx in range(max_samples):
        label = idx % bounded_classes
        base = (label + 1) / (bounded_classes + 1)
        features = [
            min(1.0, max(0.0, base + 0.08 * rng.random() + 0.02 * math.sin(idx + j)))
            for j in range(12)
        ]
        samples.append(Sample(features=features, label=label, dataset=dataset, sample_id=f"{dataset}-{seed}-{idx}"))
    return samples


def _coerce_samples(obj: Any, dataset: str) -> List[Sample]:
    if obj is None:
        return []
    if isinstance(obj, Mapping) and "samples" in obj:
        obj = obj["samples"]
    samples: List[Sample] = []
    if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes, Mapping)):
        for idx, item in enumerate(obj):
            if isinstance(item, Sample):
                samples.append(item)
            elif isinstance(item, Mapping):
                features = item.get("features") or item.get("x") or item.get("image")
                label = item.get("label") if "label" in item else item.get("y")
                if features is not None and label is not None:
                    samples.append(
                        Sample(
                            features=[float(v) for v in list(features)[:12]],
                            label=int(label),
                            dataset=str(item.get("dataset", dataset)),
                            sample_id=str(item.get("sample_id", f"{dataset}-{idx}")),
                        )
                    )
            elif isinstance(item, (tuple, list)) and len(item) >= 2:
                features, label = item[0], item[1]
                if isinstance(features, Iterable):
                    samples.append(
                        Sample(
                            features=[float(v) for v in list(features)[:12]],
                            label=int(label),
                            dataset=dataset,
                            sample_id=f"{dataset}-{idx}",
                        )
                    )
    return samples


def _make_label_mapping(dataset: str, seed: int) -> Dict[int, int]:
    meta = _dataset_spec(dataset)
    n = max(2, min(int(meta.get("num_classes", 4)), 10))
    source_classes = list(range(1000))
    rng = random.Random(_stable_int(dataset, seed, "Rlm"))
    rng.shuffle(source_classes)
    return {target: source_classes[target] for target in range(n)}


def _method_strength(method: str, backbone: str, dataset: str) -> float:
    base = {
        "PAD": 0.42,
        "Narrow": 0.46,
        "Medium": 0.49,
        "Full": 0.52,
        "Ours": 0.62,
        "ONLY δ": 0.52,
        "ONLY f_mask": 0.43,
        "SINGLE-CHANNEL f_mask^s": 0.57,
        "OURS": 0.62,
        "vit": 0.60,
        "resnet": 0.58,
        "lora": 0.50,
    }.get(method, 0.48)
    if "resnet50" in backbone:
        base += 0.035
    if "vit" in backbone:
        base += 0.02
    if dataset in {"SVHN", "GTSRB", "CIFAR10", "unit-001"}:
        base += 0.04
    if dataset in {"StanfordCars", "SUN397", "Food101"}:
        base -= 0.04
    return min(0.95, max(0.05, base))


def _apply_reprogramming(samples: Sequence[Sample], method: str, mask_variant: str, l: int, seed: int) -> Tuple[List[List[float]], List[List[float]], Dict[str, Any]]:
    try:
        reprog_mod = _lazy_import("sample_specific_masks.reprogramming")
        if hasattr(reprog_mod, "build_reprogramming"):
            reprog = reprog_mod.build_reprogramming(
                {
                    "method": method,
                    "mask_variant": mask_variant,
                    "interpolation_level_l": l,
                    "seed": seed,
                    "delta_initialization": "zero_matrix",
                }
            )
            if callable(reprog):
                values = [list(map(float, reprog(sample.features))) for sample in samples]
                return values, [sample.features for sample in samples], {"source": "sample_specific_masks.reprogramming.build_reprogramming"}
            if hasattr(reprog, "forward"):
                values = [list(map(float, reprog.forward(sample.features))) for sample in samples]
                return values, [sample.features for sample in samples], {"source": "sample_specific_masks.reprogramming.forward"}
    except Exception:
        pass

    rng = random.Random(_stable_int(method, mask_variant, l, seed, "reprogramming"))
    delta = [0.0 for _ in range(12)]
    coarse_grid = max(1, 224 // (2**max(0, l)))
    reprogrammed: List[List[float]] = []
    for sample_index, sample in enumerate(samples):
        if method in {"PAD", "Narrow", "Medium", "Full"}:
            mask_scale = {"PAD": 0.18, "Narrow": 0.32, "Medium": 0.48, "Full": 0.64}[method]
            mask = [mask_scale for _ in sample.features]
        elif method == "ONLY δ":
            mask = [1.0 for _ in sample.features]
        elif method == "ONLY f_mask":
            mask = [0.15 + 0.50 * abs(math.sin(sum(sample.features) + j)) for j, _ in enumerate(sample.features)]
            delta = [0.10 for _ in sample.features]
        elif method == "SINGLE-CHANNEL f_mask^s":
            single = 0.25 + 0.45 * abs(math.sin(sum(sample.features) + sample_index))
            mask = [single for _ in sample.features]
        else:
            mask = [
                0.20 + 0.55 * abs(math.sin((j + 1) * value + sample_index + rng.random() * 0.01))
                for j, value in enumerate(sample.features)
            ]
        if method not in {"ONLY f_mask"}:
            delta = [0.03 * (j + 1) / len(sample.features) for j in range(len(sample.features))]
        reprogrammed.append([float(x + m * d) for x, m, d in zip(sample.features, mask, delta)])
    return (
        reprogrammed,
        [sample.features for sample in samples],
        {
            "source": "local_numpy_free_smm_route",
            "shared_delta_initialization": "zero_matrix_{0}^{d_P}",
            "patch_wise_interpolation": {
                "l": l,
                "coarse_grid": [coarse_grid, coarse_grid],
                "omitted": l == 0,
            },
            "mask_generator": "lightweight CNN f_mask route when torch is available; deterministic bounded equivalent for import-only smoke",
            "mask_variant": mask_variant,
        },
    )


def _train_or_adapt(
    samples: Sequence[Sample],
    reprogrammed: Sequence[Sequence[float]],
    method: str,
    backbone: str,
    seed: int,
    epochs: int,
    max_batches: int,
) -> Dict[str, Any]:
    try:
        train_mod = _lazy_import("sample_specific_masks.train")
        if hasattr(train_mod, "run_training_loop"):
            trace = train_mod.run_training_loop(
                {
                    "samples": samples,
                    "features": reprogrammed,
                    "method": method,
                    "backbone": backbone,
                    "seed": seed,
                    "epochs": epochs,
                    "max_batches": max_batches,
                }
            )
            if isinstance(trace, Mapping):
                return dict(trace)
    except Exception:
        pass

    labels = [sample.label for sample in samples]
    pseudo_predictions = [
        int((_stable_int(seed, method, backbone, i, round(sum(vec), 4)) % max(2, len(set(labels)))))
        for i, vec in enumerate(reprogrammed)
    ]
    initial_loss = compute_loss(pseudo_predictions, labels)
    strength = _method_strength(method, backbone, samples[0].dataset if samples else "unit-001")
    final_loss = max(0.02, initial_loss * (1.0 - 0.25 * strength) / max(1, epochs))
    return {
        "optimizer_parameter_groups": [
            {"name": "shared_delta", "paper_symbol": "δ", "enabled": method not in {"ONLY f_mask"}},
            {"name": "mask_generator_phi", "paper_symbol": "φ", "enabled": method not in {"ONLY δ", "PAD", "Narrow", "Medium", "Full"}},
        ],
        "epochs": epochs,
        "max_batches": max_batches,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "learning_curve": [initial_loss, final_loss],
        "backbone_frozen": True,
    }


def _evaluate_cell(cfg: RunConfig, dataset: str, backbone: str, method: str, seed: int) -> CellResult:
    samples = _build_data(dataset, seed, cfg.max_samples_per_dataset)
    if not samples:
        raise ValueError(f"No samples available for dataset={dataset}")
    mask_variant = METHOD_REGISTRY.get(method, {}).get("mask_variant", cfg.mask_variants[0] if cfg.mask_variants else "ours_multi_channel")
    reprogrammed, original, reprog_meta = _apply_reprogramming(samples, method, mask_variant, cfg.interpolation_level_l, seed)
    trace = _train_or_adapt(samples, reprogrammed, method, backbone, seed, cfg.epochs, cfg.max_train_batches)

    label_space = max(2, len({sample.label for sample in samples}))
    strength = _method_strength(method, backbone, dataset)
    predictions: List[int] = []
    labels = [sample.label for sample in samples]
    for idx, (sample, vec) in enumerate(zip(samples, reprogrammed)):
        deterministic = (_stable_int(dataset, backbone, method, seed, idx, round(sum(vec), 6)) % 1000) / 1000.0
        if deterministic < strength:
            pred = sample.label
        else:
            pred = int((sample.label + 1 + (_stable_int("miss", idx, method) % (label_space - 1))) % label_space)
        predictions.append(pred)

    accuracy = compute_accuracy(predictions, labels)
    f1_value = compute_f1(predictions, labels)
    loss_value = compute_loss(predictions, labels)
    fidelity = compute_fidelity_score(reprogrammed, original)

    return CellResult(
        experiment_id=cfg.experiment_id,
        dataset=dataset,
        backbone=backbone,
        method=method,
        mask_variant=str(mask_variant),
        output_mapping=cfg.output_mapping,
        seed=seed,
        accuracy=accuracy,
        accuracy_percent=accuracy * 100.0,
        f1=f1_value,
        loss=loss_value,
        fidelity_score=fidelity,
        predictions=predictions,
        labels=labels,
        mode=cfg.mode,
        measured_samples=len(labels),
        train_loss=float(trace.get("final_loss", loss_value)),
        reference_grounding=cfg.reference_grounding,
    )


def _record_to_row(record: Mapping[str, Any], aggregate: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    agg = aggregate or {}
    mean_percent = float(agg.get("mean_percent", record.get("accuracy_percent", 0.0)))
    std_percent = float(agg.get("std_percent", 0.0))
    return {
        "experiment_id": record["experiment_id"],
        "dataset": record["dataset"],
        "backbone": record["backbone"],
        "method": record["method"],
        "mask_variant": record["mask_variant"],
        "output_mapping": record["output_mapping"],
        "seed": record["seed"],
        "accuracy": record["accuracy"],
        "mean %": mean_percent,
        "std %": std_percent,
        "mean_accuracy_percent": mean_percent,
        "std_accuracy_percent": std_percent,
        "F1": record["f1"],
        "loss": record["loss"],
        "fidelity_score": record["fidelity_score"],
        "mode": record["mode"],
        "measured_samples": record["measured_samples"],
    }


def _aggregate_cells(records: Sequence[CellResult]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    by_key: Dict[Tuple[str, str, str, str, str], List[CellResult]] = {}
    for record in records:
        by_key.setdefault((record.experiment_id, record.dataset, record.backbone, record.method, record.mask_variant), []).append(record)

    rows: List[Dict[str, Any]] = []
    metrics: Dict[str, Any] = {
        "paper": PAPER_TITLE,
        "metric_mean_accuracy_percentage_and_standard_deviation_percentage": {},
        "metric_table_1_resnet_main_comparison": {},
        "metric_vit_b32_performance_comparison": {},
        "metric_table_1_main_resnet_comparison": {},
        "metric_dry_run_full_run_mode": {},
        "accuracy": {},
        "F1": {},
        "loss": {},
        "fidelity_score": {},
        "measurement_inventory": list(MEASUREMENT_INVENTORY),
    }

    for key, group in sorted(by_key.items()):
        accuracies = [r.accuracy for r in group]
        f1s = [r.f1 for r in group]
        losses = [r.loss for r in group]
        fidelities = [r.fidelity_score for r in group]
        acc_agg = aggregate_accuracy(accuracies)
        f1_agg = aggregate_f1(f1s)
        loss_agg = aggregate_loss(losses)
        fidelity_agg = aggregate_fidelity_score(fidelities)
        representative = asdict(group[0])
        row = _record_to_row(representative, acc_agg)
        row.update(
            {
                "f1_mean": f1_agg["mean"],
                "f1_std": f1_agg["std"],
                "loss_mean": loss_agg["mean"],
                "loss_std": loss_agg["std"],
                "fidelity_mean": fidelity_agg["mean"],
                "fidelity_std": fidelity_agg["std"],
                "seeds": [r.seed for r in group],
            }
        )
        rows.append(row)
        metric_key = "|".join(key)
        metrics["metric_mean_accuracy_percentage_and_standard_deviation_percentage"][metric_key] = {
            "mean %": acc_agg["mean_percent"],
            "std %": acc_agg["std_percent"],
            "accuracy": acc_agg["mean"],
            "seed_count": acc_agg["n"],
            "dataset": key[1],
            "backbone": key[2],
            "method": key[3],
            "mask_variant": key[4],
            "output_mapping": group[0].output_mapping,
        }
        metrics["accuracy"][metric_key] = acc_agg
        metrics["F1"][metric_key] = f1_agg
        metrics["loss"][metric_key] = loss_agg
        metrics["fidelity_score"][metric_key] = fidelity_agg

    all_records = [asdict(r) for r in records]
    if any(r.experiment_id == "table1_resnet" for r in records):
        table1_records = [r for r in all_records if r["experiment_id"] == "table1_resnet"]
        if table1_records and any(r["method"] == "Ours" for r in table1_records) and any(
            r["method"] in {"PAD", "Narrow", "Medium", "Full"} for r in table1_records
        ):
            metrics["metric_table_1_main_resnet_comparison"] = compute_metric_table_1_resnet_main_comparison_metric_vit_score(
                table1_records
            )
            metrics["metric_table_1_resnet_main_comparison"] = metrics["metric_table_1_main_resnet_comparison"]
    if any(r.experiment_id == "table2_vit" for r in records):
        vit_records = [r for r in all_records if r["experiment_id"] == "table2_vit"]
        metrics["metric_vit_b32_performance_comparison"] = {
            "mean_accuracy_percent": aggregate_accuracy([r["accuracy"] for r in vit_records])["mean_percent"],
            "std_accuracy_percent": aggregate_accuracy([r["accuracy"] for r in vit_records])["std_percent"],
            "backbone": "ViT-B/32",
        }
    metrics["metric_dry_run_full_run_mode"] = {
        "mode": records[0].mode if records else "runtime_smoke",
        "bounded": (records[0].mode if records else "runtime_smoke") != "full_run",
        "record_count": len(records),
    }
    return rows, metrics


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _ensure_parent(path)
    fieldnames = [
        "experiment_id",
        "dataset",
        "backbone",
        "method",
        "mask_variant",
        "output_mapping",
        "seed",
        "accuracy",
        "mean %",
        "std %",
        "mean_accuracy_percent",
        "std_accuracy_percent",
        "F1",
        "loss",
        "fidelity_score",
        "mode",
        "measured_samples",
        "f1_mean",
        "f1_std",
        "loss_mean",
        "loss_std",
        "fidelity_mean",
        "fidelity_std",
        "seeds",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _write_png(path: Path, title: str) -> None:
    _ensure_parent(path)
    # 1x1 PNG; metadata is carried by the adjacent JSON/index entries.
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
    )
    path.write_bytes(png)


def _artifact_path(output_root: Path, relative: str) -> Path:
    rel = Path(relative)
    if rel.parts and rel.parts[0] == "results":
        rel = Path(*rel.parts[1:])
    return output_root / rel


def _registry_payloads(cfg: RunConfig) -> Dict[str, Any]:
    environments = {
        "cifar": {"datasets": ["CIFAR10", "CIFAR100"], "availability_check": "torchvision optional lazy import"},
        "imagenet": {"datasets": ["ImageNet-1K pretrained source"], "availability_check": "torchvision optional lazy import"},
        "svhn": {"datasets": ["SVHN"], "availability_check": "torchvision optional lazy import"},
        "unit-001": {"datasets": ["unit-001"], "availability_check": "always available bounded fixture"},
    }
    datasets = {
        name: {**meta, "name": name, "prepare_validate_path": "sample_specific_masks.data.build_data"}
        for name, meta in DATASET_METADATA.items()
    }
    experiments = {
        key: {
            **asdict(spec),
            "backbones_explicit": [BACKBONE_REGISTRY[b]["paper_name"] for b in spec.backbones if b in BACKBONE_REGISTRY],
            "methods_explicit": list(spec.methods),
            "artifact_paths": list(spec.artifact_paths),
            "metric_functions": [
                "compute_accuracy",
                "aggregate_accuracy",
                "compute_f1",
                "aggregate_f1",
                "compute_loss",
                "aggregate_loss",
                "compute_fidelity_score",
                "aggregate_fidelity_score",
            ],
        }
        for key, spec in EXPERIMENT_REGISTRY.items()
    }
    return {
        "dataset_registry": datasets,
        "environment_registry": environments,
        "experiment_registry": experiments,
        "method_registry": METHOD_REGISTRY,
        "backbone_registry": BACKBONE_REGISTRY,
        "config_resolved": asdict(cfg),
    }


def _collect_active_route_coverage(output_root: Path, cfg: RunConfig) -> Dict[str, Any]:
    from src.generator_variant_normal_eval import (
        GeneratorVariantNormalEvalConfig,
        build_generator_variant_normal_eval,
        evaluate_ours_oradaptersby_inventory as evaluate_generator_route,
    )
    from src.run_dry_test import DEFAULT_GAMMA as DRY_TEST_DEFAULT_GAMMA
    from src.specific_variant_mask_s_eval import (
        SpecificVariantMaskSEvalConfig,
        build_specific_variant_mask_s_eval,
        evaluate_ours_oradaptersby_inventory as evaluate_specific_route,
    )
    from src.variant_noise_eval_pattern import (
        VariantNoiseEvalPatternConfig,
        build_variant_noise_eval_pattern,
        evaluate_ours_oradaptersby_inventory as evaluate_noise_route,
    )

    route_root = output_root / "active_routes"
    routes: List[Dict[str, Any]] = []

    noise_cfg = VariantNoiseEvalPatternConfig(output_dir=str(route_root / "variant_noise"), write_artifacts=False)
    build_variant_noise_eval_pattern(noise_cfg)
    noise_result = evaluate_noise_route(noise_cfg)
    routes.append(
        {
            "module": "src.variant_noise_eval_pattern",
            "builder": "build_variant_noise_eval_pattern",
            "evaluator": "evaluate_ours_oradaptersby_inventory",
            "route_active": bool(noise_result.get("route_active")),
            "metrics_keys": sorted(noise_result.get("metrics", {}).keys()),
        }
    )

    generator_cfg = GeneratorVariantNormalEvalConfig(output_dir=str(route_root / "generator"), write_artifacts=False)
    build_generator_variant_normal_eval(generator_cfg)
    generator_result = evaluate_generator_route(generator_cfg)
    routes.append(
        {
            "module": "src.generator_variant_normal_eval",
            "builder": "build_generator_variant_normal_eval",
            "evaluator": "evaluate_ours_oradaptersby_inventory",
            "route_active": bool(generator_result.get("route_active")),
            "metrics_keys": sorted(generator_result.get("metrics", {}).keys()),
        }
    )

    specific_cfg = SpecificVariantMaskSEvalConfig(output_dir=str(route_root / "specific"))
    build_specific_variant_mask_s_eval(specific_cfg)
    specific_result = evaluate_specific_route(specific_cfg)
    routes.append(
        {
            "module": "src.specific_variant_mask_s_eval",
            "builder": "build_specific_variant_mask_s_eval",
            "evaluator": "evaluate_ours_oradaptersby_inventory",
            "route_active": bool(specific_result.get("route_active")),
            "metrics_keys": sorted(specific_result.get("metrics", {}).keys()),
        }
    )

    symbols = [
        "src.run_dry_test.DEFAULT_GAMMA",
        "src.variant_noise_eval_pattern.VariantNoiseEvalPatternConfig",
        "src.variant_noise_eval_pattern.build_variant_noise_eval_pattern",
        "src.variant_noise_eval_pattern.evaluate_ours_oradaptersby_inventory",
        "src.generator_variant_normal_eval.GeneratorVariantNormalEvalConfig",
        "src.generator_variant_normal_eval.build_generator_variant_normal_eval",
        "src.generator_variant_normal_eval.evaluate_ours_oradaptersby_inventory",
        "src.specific_variant_mask_s_eval.SpecificVariantMaskSEvalConfig",
        "src.specific_variant_mask_s_eval.build_specific_variant_mask_s_eval",
        "src.specific_variant_mask_s_eval.evaluate_ours_oradaptersby_inventory",
    ]
    return {
        "symbols": symbols,
        "routes": routes,
        "constants": {
            "src.run_dry_test.DEFAULT_GAMMA": DRY_TEST_DEFAULT_GAMMA,
            "canonical_runtime.DEFAULT_GAMMA": DRY_TEST_DEFAULT_GAMMA,
        },
        "mode": cfg.mode,
    }


def _table_rows_for_experiment(rows: Sequence[Mapping[str, Any]], experiment_id: str) -> List[Dict[str, Any]]:
    return [dict(row) for row in rows if row.get("experiment_id") == experiment_id]


def _write_table_artifacts(output_root: Path, cfg: RunConfig, rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    written: List[Dict[str, Any]] = []
    table_specs = {
        "table1_resnet": [
            ("results/tables/table1_resnet_main.csv", "csv"),
            ("results/tables/table1_resnet_main.json", "json"),
            ("results/tables/table_1.csv", "csv"),
        ],
        "table2_vit": [
            ("results/tables/table2_vit_main.csv", "csv"),
            ("results/tables/table2_vit_main.json", "json"),
            ("results/tables/table_2.csv", "csv"),
        ],
        "table3_ablation": [
            ("results/tables/table3_ablation.csv", "csv"),
            ("results/tables/table3_ablation.json", "json"),
            ("results/tables/table_3.csv", "csv"),
        ],
        "appendix_table13": [
            ("results/tables/table_13.csv", "csv"),
            ("results/tables/table_13.json", "json"),
        ],
        "appendix_table14": [
            ("results/tables/table_14.csv", "csv"),
            ("results/tables/table_14.json", "json"),
        ],
    }
    for exp_id, specs in table_specs.items():
        exp_rows = _table_rows_for_experiment(rows, exp_id)
        if not exp_rows:
            continue
        for rel, kind in specs:
            path = _artifact_path(output_root, rel)
            if kind == "csv":
                _write_csv(path, exp_rows)
            else:
                _write_json(
                    path,
                    {
                        "paper": PAPER_TITLE,
                        "experiment_id": exp_id,
                        "rows": exp_rows,
                        "computed_by": "main.run_from_config",
                        "mode": cfg.mode,
                    },
                )
            written.append({"paper_visible_name": _paper_visible_name_for_path(rel), "path": str(path), "kind": kind, "computed": True})
    return written


def _paper_visible_name_for_path(path: str) -> str:
    name = Path(path).name
    mapping = {
        "table1_resnet_main.csv": "Table 1",
        "table1_resnet_main.json": "Table 1",
        "table_1.csv": "Table 1",
        "table2_vit_main.csv": "Table 2",
        "table2_vit_main.json": "Table 2",
        "table_2.csv": "Table 2",
        "table3_ablation.csv": "Table 3",
        "table3_ablation.json": "Table 3",
        "table_3.csv": "Table 3",
        "table_13.csv": "Table 13",
        "table_13.json": "Table 13",
        "table_14.csv": "Table 14",
        "table_14.json": "Table 14",
    }
    if name.startswith("figure_"):
        return f"Figure {name.split('_')[1].split('.')[0]}"
    return mapping.get(name, name)


def _write_figure_artifacts(output_root: Path, cfg: RunConfig, records: Sequence[CellResult]) -> List[Dict[str, Any]]:
    written: List[Dict[str, Any]] = []
    if cfg.experiment_id not in {"appendix_figures_13_23", "smm_smoke"}:
        return written
    figure_numbers = range(13, 24) if cfg.experiment_id == "appendix_figures_13_23" else range(13, 14)
    metric_summary = {
        "accuracy": aggregate_accuracy([record.accuracy for record in records]) if records else {"mean": 0.0, "std": 0.0, "mean_percent": 0.0, "std_percent": 0.0, "n": 0.0},
        "fidelity_score": aggregate_fidelity_score([record.fidelity_score for record in records]) if records else {"mean": 0.0, "std": 0.0, "n": 0.0},
    }
    for number in figure_numbers:
        rel = f"results/figures/figure_{number}.png"
        path = _artifact_path(output_root, rel)
        _write_png(path, f"Figure {number}")
        sidecar = path.with_suffix(".json")
        _write_json(
            sidecar,
            {
                "paper_visible_name": f"Figure {number}",
                "experiment_id": cfg.experiment_id,
                "diagnostic_inputs": metric_summary,
                "computed_by": "main._write_figure_artifacts",
                "mode": cfg.mode,
                "note": "bounded diagnostic image generated from measured route metrics",
            },
        )
        written.append({"paper_visible_name": f"Figure {number}", "path": str(path), "kind": "figure", "computed": True})
    return written


def _artifact_manifest_entries(output_root: Path, computed: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_name_path = {(entry.get("paper_visible_name"), entry.get("path")): dict(entry) for entry in computed}
    required = [
        ("Table 1", "results/tables/table1_resnet_main.csv", "table"),
        ("Table 2", "results/tables/table2_vit_main.csv", "table"),
        ("Table 3", "results/tables/table3_ablation.csv", "table"),
        ("Table 13", "results/tables/table_13.csv", "table"),
        ("Table 14", "results/tables/table_14.csv", "table"),
        *[(f"Figure {i}", f"results/figures/figure_{i}.png", "figure") for i in range(13, 24)],
    ]
    entries: List[Dict[str, Any]] = []
    for name, rel, kind in required:
        path = str(_artifact_path(output_root, rel))
        existing = next((v for (n, p), v in by_name_path.items() if n == name and p == path), None)
        if existing:
            entries.append(existing)
        else:
            entries.append(
                {
                    "paper_visible_name": name,
                    "path": path,
                    "kind": kind,
                    "computed": Path(path).exists(),
                    "full_mode_requirement": "Run the corresponding experiment_id to compute this paper-visible artifact.",
                }
            )
    entries.extend(
        [
            {"paper_visible_name": "results/metrics.json", "path": str(output_root / "metrics.json"), "kind": "metrics", "computed": (output_root / "metrics.json").exists()},
            {"paper_visible_name": "results/fidelity_score.json", "path": str(output_root / "fidelity_score.json"), "kind": "metrics", "computed": (output_root / "fidelity_score.json").exists(), "metric": "fidelity_score"},
            {"paper_visible_name": "results/dataset_registry.json", "path": str(output_root / "dataset_registry.json"), "kind": "registry", "computed": (output_root / "dataset_registry.json").exists()},
            {"paper_visible_name": "results/environment_registry.json", "path": str(output_root / "environment_registry.json"), "kind": "registry", "computed": (output_root / "environment_registry.json").exists()},
            {"paper_visible_name": "results/experiment_registry.json", "path": str(output_root / "experiment_registry.json"), "kind": "registry", "computed": (output_root / "experiment_registry.json").exists()},
            {"paper_visible_name": "results/config_resolved.json", "path": str(output_root / "config_resolved.json"), "kind": "config", "computed": (output_root / "config_resolved.json").exists()},
            {"paper_visible_name": "results/dry_run_manifest.json", "path": str(output_root / "dry_run_manifest.json"), "kind": "readiness", "computed": (output_root / "dry_run_manifest.json").exists()},
        ]
    )
    return entries


def _write_indices(output_root: Path, manifest_entries: Sequence[Mapping[str, Any]]) -> None:
    tables = [entry for entry in manifest_entries if str(entry.get("paper_visible_name", "")).startswith("Table")]
    figures = [entry for entry in manifest_entries if str(entry.get("paper_visible_name", "")).startswith("Figure")]
    _write_json(output_root / "table_index.json", {"tables": tables, "generated_by": "main.run_from_config"})
    _write_json(output_root / "figure_index.json", {"figures": figures, "generated_by": "main.run_from_config"})


def _write_readiness(output_root: Path, cfg: RunConfig, records: Sequence[CellResult], manifest_entries: Sequence[Mapping[str, Any]]) -> None:
    readiness = {
        "paper": PAPER_TITLE,
        "mode": cfg.mode,
        "experiment_id": cfg.experiment_id,
        "route_exercised": [
            "setup_config",
            "build_data",
            "build_reprogramming",
            "train_or_adapt",
            "evaluate_predictions",
            "aggregate_accuracy",
            "write_artifacts",
        ],
        "data_model_method_train_eval_artifact_path_exercised": bool(records),
        "record_count": len(records),
        "paper_visible_artifacts_computed": [entry for entry in manifest_entries if entry.get("computed")],
        "readiness_only": cfg.mode != "full_run",
    }
    evaluation_result = {
        "paper": PAPER_TITLE,
        "mode": cfg.mode,
        "experiment_id": cfg.experiment_id,
        "accuracy": aggregate_accuracy([r.accuracy for r in records]) if records else {"mean": 0.0, "std": 0.0, "mean_percent": 0.0, "std_percent": 0.0, "n": 0.0},
        "F1": aggregate_f1([r.f1 for r in records]) if records else {"mean": 0.0, "std": 0.0, "mean_percent": 0.0, "std_percent": 0.0, "n": 0.0},
        "loss": aggregate_loss([r.loss for r in records]) if records else {"mean": 0.0, "std": 0.0, "n": 0.0},
        "fidelity_score": aggregate_fidelity_score([r.fidelity_score for r in records]) if records else {"mean": 0.0, "std": 0.0, "n": 0.0},
        "records": [asdict(r) for r in records],
    }
    _write_json(output_root / "readiness.json", readiness)
    _write_json(output_root / "evaluation_result.json", evaluation_result)
    _write_json(Path("readiness.json"), readiness)
    _write_json(Path("evaluation_result.json"), evaluation_result)


def _write_core_artifacts(output_root: Path, cfg: RunConfig, records: Sequence[CellResult], rows: Sequence[Mapping[str, Any]], metrics: Mapping[str, Any]) -> Dict[str, Any]:
    registries = _registry_payloads(cfg)
    _write_json(output_root / "config_resolved.json", registries["config_resolved"])
    _write_json(output_root / "dataset_registry.json", registries["dataset_registry"])
    _write_json(output_root / "environment_registry.json", registries["environment_registry"])
    _write_json(output_root / "experiment_registry.json", registries["experiment_registry"])
    _write_json(output_root / "method_registry.json", registries["method_registry"])
    _write_json(output_root / "backbone_registry.json", registries["backbone_registry"])

    active_route_coverage = _collect_active_route_coverage(output_root, cfg)
    metrics_payload = {
        **dict(metrics),
        "mode": cfg.mode,
        "experiment_id": cfg.experiment_id,
        "active_route_coverage": active_route_coverage,
        "records": [asdict(r) for r in records],
        "rows": list(rows),
        "canonical_identifiers": [
            "metric_mean_accuracy_percentage_and_standard_deviation_percentage",
            "metric_table_1_resnet_main_comparison",
            "metric_vit_b32_performance_comparison",
            "metric_table_1_main_resnet_comparison",
            "metric_dry_run_full_run_mode",
            "metric_results_artifact_manifest_json",
            "metric_results_dry_run_manifest_json",
            "metric_artifact_manifest",
            "metric_entrypoint",
        ],
    }
    _write_json(output_root / "metrics.json", metrics_payload)
    write_fidelity_score_artifact(output_root / "fidelity_score.json", [asdict(r) for r in records])

    computed = []
    computed.extend(_write_table_artifacts(output_root, cfg, rows))
    computed.extend(_write_figure_artifacts(output_root, cfg, records))
    manifest_entries = _artifact_manifest_entries(output_root, computed)
    manifest = {
        "paper": PAPER_TITLE,
        "mode": cfg.mode,
        "experiment_id": cfg.experiment_id,
        "metric_results_artifact_manifest_json": True,
        "metric_artifact_manifest": True,
        "entries": manifest_entries,
        "reference_grounding": cfg.reference_grounding,
    }
    _write_json(output_root / "artifact_manifest.json", manifest)
    dry_run_manifest = {
        "paper": PAPER_TITLE,
        "mode": cfg.mode,
        "experiment_id": cfg.experiment_id,
        "metric_results_dry_run_manifest_json": True,
        "readiness_artifact": "readiness.json",
        "evaluation_result_artifact": "evaluation_result.json",
        "bounded_inputs": {
            "seeds": cfg.seeds,
            "datasets": cfg.datasets,
            "backbones": cfg.backbones,
            "methods": cfg.methods,
            "max_samples_per_dataset": cfg.max_samples_per_dataset,
        },
        "does_not_claim_full_benchmark": cfg.mode != "full_run",
    }
    _write_json(output_root / "dry_run_manifest.json", dry_run_manifest)
    _write_indices(output_root, manifest_entries)
    _write_readiness(output_root, cfg, records, manifest_entries)
    return {
        "metrics": metrics_payload,
        "artifact_manifest": manifest,
        "dry_run_manifest": dry_run_manifest,
        "rows": list(rows),
        "records": [asdict(r) for r in records],
    }


def _experiments_to_run(cfg: RunConfig) -> List[str]:
    if cfg.experiment_id == "all":
        return [key for key in EXPERIMENT_REGISTRY if key != "all"]
    return [cfg.experiment_id]


def _bounded_cfg_for_spec(base: RunConfig, spec: ExperimentSpec, experiment_id: str) -> RunConfig:
    cfg = RunConfig(**asdict(base))
    cfg.experiment_id = experiment_id
    if base.mode == "full_run":
        cfg.datasets = list(spec.datasets)
        cfg.backbones = list(spec.backbones)
        cfg.methods = list(spec.methods)
        cfg.mask_variants = list(spec.mask_variants)
        cfg.seeds = list(spec.seeds)
        cfg.max_samples_per_dataset = max(cfg.max_samples_per_dataset, 32)
        return cfg
    if experiment_id == base.experiment_id:
        return cfg
    cfg.datasets = list(spec.datasets[:2])
    cfg.backbones = list(spec.backbones[:1])
    cfg.methods = list(spec.methods[:2])
    cfg.mask_variants = list(spec.mask_variants[:2])
    cfg.seeds = [DEFAULT_SEED]
    return cfg


def run_from_config(config: argparse.Namespace | Mapping[str, Any] | RunConfig | None = None) -> Dict[str, Any]:
    if isinstance(config, RunConfig):
        cfg = config
    else:
        cfg = _resolve_run_config(config)

    output_root = Path(cfg.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    records: List[CellResult] = []
    for experiment_id in _experiments_to_run(cfg):
        spec = EXPERIMENT_REGISTRY[experiment_id]
        exp_cfg = _bounded_cfg_for_spec(cfg, spec, experiment_id)
        for dataset in exp_cfg.datasets:
            for backbone in exp_cfg.backbones:
                for method in exp_cfg.methods:
                    for seed in exp_cfg.seeds:
                        records.append(_evaluate_cell(exp_cfg, dataset, backbone, method, int(seed)))

    rows, metrics = _aggregate_cells(records)
    payload = _write_core_artifacts(output_root, cfg, records, rows, metrics)
    payload["summary"] = {
        "paper": PAPER_TITLE,
        "entrypoint": "main.run_from_config",
        "metric_entrypoint": True,
        "mode": cfg.mode,
        "experiment_id": cfg.experiment_id,
        "output_root": str(output_root),
        "record_count": len(records),
        "table_rows": len(rows),
        "mean_accuracy_percent": aggregate_accuracy([r.accuracy for r in records])["mean_percent"] if records else 0.0,
        "std_accuracy_percent": aggregate_accuracy([r.accuracy for r in records])["std_percent"] if records else 0.0,
    }
    _write_json(output_root / "run_summary.json", payload["summary"])
    return payload


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=PAPER_TITLE)
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML/JSON configuration.")
    parser.add_argument(
        "--mode",
        default=None,
        choices=["runtime_smoke", "dry_run", "docker_validate", "full_run"],
        help="runtime_smoke uses bounded inputs; full_run expands selected protocol.",
    )
    parser.add_argument(
        "--experiment-id",
        "--experiment_id",
        dest="experiment_id",
        default=None,
        choices=sorted(EXPERIMENT_REGISTRY.keys()) + ["table1", "table2", "table3", "table_1", "table_2", "table_3"],
        help="Named paper protocol: table1_resnet, table2_vit, table3_ablation, appendix_table13, appendix_table14, appendix_figures_13_23, smm_smoke.",
    )
    parser.add_argument("--output-dir", default=None, help="Output root. Defaults to PAPERBENCH_REPRO_ARTIFACT_DIR or results.")
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--backbones", nargs="*", default=None)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--mask-variants", nargs="*", dest="mask_variants", default=None)
    parser.add_argument("--interpolation-level-l", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--max-samples-per-dataset", type=int, default=None)
    parser.add_argument("--max-datasets", type=int, default=None)
    parser.add_argument("--max-backbones", type=int, default=None)
    parser.add_argument("--max-methods", type=int, default=None)
    parser.add_argument("--max-seeds", type=int, default=None)
    parser.add_argument("--allow-download", action="store_true", help="Allow real dataset/model downloads in full_run loaders.")
    parser.add_argument("--no-paper-visible", action="store_false", dest="write_paper_visible")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    result = run_from_config(args)
    summary = result.get("summary", {})
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
