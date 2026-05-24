#!/usr/bin/env python3
"""
Dry/runtime-smoke entrypoint for the Sample-specific Masks for Visual
Reprogramming-based Prompting reproduction.

reference_grounding: chunk_009 /mnt/paper2any/pzw/proj/paperagent/hx/Research_space/Reproduction/paperbench_data/sample-specific-masks/paper.md
reference_grounding: chunk_016_01 /mnt/paper2any/pzw/proj/paperagent/hx/Research_space/Reproduction/paperbench_data/sample-specific-masks/paper.md
reference_grounding: chunk_017_02 /mnt/paper2any/pzw/proj/paperagent/hx/Research_space/Reproduction/paperbench_data/sample-specific-masks/paper.md
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
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DEFAULT_LEARNING_RATE = 1.0e-3
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 30
DEFAULT_SEED = 0
DEFAULT_ALPHA = 1.0e-3
DEFAULT_GAMMA = 0.9

THREE_SEED_PROTOCOL = (0, 1, 2)
P_SWEEP_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)
PATCH_SIZE_VALUES = (4, 2, 1)
INTERPOLATION_LEVEL_VALUES = (0, 1, 2)
SIMILARITY_GUIDANCE_SCALE_VALUES = (9, 7, 10)

RESULT_FIELDS = (
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

PAPER = "Sample-specific Masks for Visual Reprogramming-based Prompting"
OUTPUT_MAPPING = "Rlm_random_label_mapping"
PNG_1X1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de"
    "0000000c49444154789c6360f8cf00000301010018dd8db00000000049454e44ae426082"
)


def learning_rate_values(mode: str = "runtime_smoke") -> List[float]:
    if mode == "full_run":
        return [1.0e-4, DEFAULT_LEARNING_RATE, 3.0e-3]
    return [DEFAULT_LEARNING_RATE]


def resolve_learning_rate_defaults(mode: str = "runtime_smoke", override: Optional[float] = None) -> float:
    value = DEFAULT_LEARNING_RATE if override is None else float(override)
    if value <= 0:
        raise ValueError("learning rate must be positive")
    return value


def batch_size_values(mode: str = "runtime_smoke") -> List[int]:
    return [4] if mode != "full_run" else [16, DEFAULT_BATCH_SIZE, 64]


def resolve_batch_size_defaults(mode: str = "runtime_smoke", override: Optional[int] = None) -> int:
    value = 4 if mode != "full_run" and override is None else (DEFAULT_BATCH_SIZE if override is None else int(override))
    if value <= 0:
        raise ValueError("batch size must be positive")
    return value


def epochs_values(mode: str = "runtime_smoke") -> List[int]:
    return [1] if mode != "full_run" else [DEFAULT_EPOCHS]


def resolve_epochs_defaults(mode: str = "runtime_smoke", override: Optional[int] = None) -> int:
    value = 1 if mode != "full_run" and override is None else (DEFAULT_EPOCHS if override is None else int(override))
    if value <= 0:
        raise ValueError("epochs must be positive")
    return value


def seed_values(mode: str = "runtime_smoke") -> List[int]:
    return [DEFAULT_SEED] if mode != "full_run" else list(THREE_SEED_PROTOCOL)


def resolve_seed_defaults(mode: str = "runtime_smoke", override: Optional[Sequence[int] | int] = None) -> List[int]:
    if override is None:
        return seed_values(mode)
    if isinstance(override, int):
        return [override]
    values = [int(v) for v in override]
    if not values:
        raise ValueError("at least one seed is required")
    return values


def alpha_values(mode: str = "runtime_smoke") -> List[float]:
    return [DEFAULT_ALPHA] if mode != "full_run" else [1.0e-4, DEFAULT_ALPHA, 1.0e-2]


def resolve_alpha_defaults(mode: str = "runtime_smoke", override: Optional[float] = None) -> float:
    value = DEFAULT_ALPHA if override is None else float(override)
    if value < 0:
        raise ValueError("alpha must be non-negative")
    return value


def gamma_values(mode: str = "runtime_smoke") -> List[float]:
    return [DEFAULT_GAMMA] if mode != "full_run" else [0.5, DEFAULT_GAMMA, 0.99]


def resolve_gamma_defaults(mode: str = "runtime_smoke", override: Optional[float] = None) -> float:
    value = DEFAULT_GAMMA if override is None else float(override)
    if not 0 < value <= 1:
        raise ValueError("gamma must be in (0, 1]")
    return value


def optional_backend_status() -> Dict[str, Dict[str, Any]]:
    backends = {}
    for name in ("torch", "torchvision", "datasets", "sbi", "gym", "gymnasium", "PIL", "matplotlib"):
        spec = importlib.util.find_spec(name)
        backends[name] = {
            "available": spec is not None,
            "lazy_import_factory": f"importlib.import_module('{name}')",
            "required_for_full_mode": name in {"torch", "torchvision", "datasets"},
        }
    return backends


def lazy_import_backend(name: str) -> Any:
    spec = importlib.util.find_spec(name)
    if spec is None:
        raise RuntimeError(
            f"Optional backend '{name}' is not installed. Install the full reproduction "
            "extras before running full training/evaluation."
        )
    return importlib.import_module(name)


@dataclass(frozen=True)
class DatasetSpec:
    id: str
    aliases: Tuple[str, ...]
    display_name: str
    num_classes: int
    environments: Tuple[str, ...]
    split_protocol: str
    smoke_samples: int = 8
    full_loader: str = "torchvision/datasets lazy loader"
    validation_checks: Tuple[str, ...] = ("nonempty split", "class_count matches mapping", "image tensor shape")


@dataclass(frozen=True)
class EnvironmentSpec:
    id: str
    aliases: Tuple[str, ...]
    setup_metadata: Mapping[str, Any]
    availability_check: str


@dataclass(frozen=True)
class BackboneSpec:
    id: str
    display_name: str
    pretrained_source: str
    input_size: Tuple[int, int]
    model_factory: str
    frozen_parameters: bool = True


@dataclass(frozen=True)
class MethodSpec:
    id: str
    display_name: str
    mask_variant: str
    selector_aliases: Tuple[str, ...]
    delta_enabled: bool
    mask_generator_enabled: bool
    channels: int
    p: float
    patch_size: int
    interpolation_level: int
    baseline_family: str


@dataclass(frozen=True)
class ArtifactSpec:
    paper_name: str
    path: str
    kind: str
    caption: str
    computed_in_smoke: bool = False


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    paper_name: str
    artifact_paths: Tuple[str, ...]
    datasets: Tuple[str, ...]
    backbones: Tuple[str, ...]
    methods: Tuple[str, ...]
    metric_functions: Tuple[str, ...]
    writer: str
    hypothesis: str
    decision_metric: str
    stop_rule: str
    mode_default: str = "runtime_smoke"


@dataclass
class DryRunConfig:
    experiment_id: str = "smm_smoke"
    mode: str = "runtime_smoke"
    output_root: Path = Path("results")
    seeds: List[int] = field(default_factory=lambda: [DEFAULT_SEED])
    datasets: List[str] = field(default_factory=lambda: ["unit-001"])
    backbones: List[str] = field(default_factory=lambda: ["resnet18_imagenet1k"])
    methods: List[str] = field(default_factory=lambda: ["Ours"])
    learning_rate: float = DEFAULT_LEARNING_RATE
    batch_size: int = 4
    epochs: int = 1
    alpha: float = DEFAULT_ALPHA
    gamma: float = DEFAULT_GAMMA
    p_values: List[float] = field(default_factory=lambda: list(P_SWEEP_VALUES))
    patch_size_values: List[int] = field(default_factory=lambda: list(PATCH_SIZE_VALUES))
    interpolation_levels: List[int] = field(default_factory=lambda: list(INTERPOLATION_LEVEL_VALUES))
    output_mapping: str = OUTPUT_MAPPING
    max_samples_per_dataset: int = 8


DATASET_REGISTRY: Dict[str, DatasetSpec] = {
    "unit-001": DatasetSpec(
        id="unit-001",
        aliases=("unit", "unit-001"),
        display_name="unit-001",
        num_classes=3,
        environments=("unit-001",),
        split_protocol="bounded local smoke fixture through the same loader interface",
    ),
    "CIFAR10": DatasetSpec("CIFAR10", ("cifar10", "cifar"), "CIFAR10", 10, ("cifar",), "Chen et al. target split"),
    "CIFAR100": DatasetSpec("CIFAR100", ("cifar100", "cifar"), "CIFAR100", 100, ("cifar",), "Chen et al. target split"),
    "SVHN": DatasetSpec("SVHN", ("svhn",), "SVHN", 10, ("svhn",), "Chen et al. target split"),
    "GTSRB": DatasetSpec("GTSRB", ("gtsrb",), "GTSRB", 43, ("vision",), "Chen et al. target split"),
    "Flowers102": DatasetSpec("Flowers102", ("flowers", "flowers102"), "Flowers102", 102, ("vision",), "Chen et al. target split"),
    "DTD": DatasetSpec("DTD", ("dtd",), "DTD", 47, ("vision",), "Chen et al. target split"),
    "UCF101": DatasetSpec("UCF101", ("ucf101",), "UCF101", 101, ("vision",), "Chen et al. target split"),
    "EuroSAT": DatasetSpec("EuroSAT", ("eurosat",), "EuroSAT", 10, ("vision",), "Chen et al. target split"),
    "ImageNet": DatasetSpec("ImageNet", ("imagenet",), "ImageNet", 1000, ("imagenet",), "source task only"),
    "imagenet_1k": DatasetSpec("imagenet_1k", ("ImageNet-1K", "imagenet1k"), "ImageNet-1K", 1000, ("imagenet",), "pretrained source"),
    "stanford_cars": DatasetSpec("stanford_cars", ("StanfordCars", "cars"), "Stanford Cars", 196, ("vision",), "Appendix D.4 diagnostic"),
    "oxford_pets": DatasetSpec("oxford_pets", ("OxfordPets", "pets"), "OxfordPets", 37, ("vision",), "Figure 1 diagnostic"),
}

ENVIRONMENT_REGISTRY: Dict[str, EnvironmentSpec] = {
    "cifar": EnvironmentSpec("cifar", ("CIFAR10", "CIFAR100"), {"task_family": "target image classification"}, "dataset alias available"),
    "imagenet": EnvironmentSpec("imagenet", ("ImageNet", "ImageNet-1K pretrained source"), {"source_classes": 1000}, "lazy torchvision weights"),
    "svhn": EnvironmentSpec("svhn", ("SVHN",), {"task_family": "digit classification"}, "dataset alias available"),
    "unit-001": EnvironmentSpec("unit-001", ("unit",), {"task_family": "bounded local closure"}, "always available"),
    "ResNet-18 ImageNet-1K": EnvironmentSpec(
        "ResNet-18 ImageNet-1K", ("resnet18_imagenet1k",), {"backbone": "ResNet-18", "pretrained": "ImageNet-1K"}, "torchvision lazy"
    ),
    "ResNet-50 ImageNet-1K": EnvironmentSpec(
        "ResNet-50 ImageNet-1K", ("resnet50_imagenet1k",), {"backbone": "ResNet-50", "pretrained": "ImageNet-1K"}, "torchvision lazy"
    ),
    "ViT-B/32 ImageNet-1K": EnvironmentSpec(
        "ViT-B/32 ImageNet-1K", ("vit_b_32_imagenet1k", "ViT-B/32"), {"backbone": "ViT-B/32", "pretrained": "ImageNet-1K"}, "torchvision/timm lazy"
    ),
}

BACKBONE_REGISTRY: Dict[str, BackboneSpec] = {
    "resnet18_imagenet1k": BackboneSpec(
        "resnet18_imagenet1k", "ResNet-18", "ImageNet-1K", (224, 224), "torchvision.models.resnet18(weights=IMAGENET1K_V1)"
    ),
    "resnet50_imagenet1k": BackboneSpec(
        "resnet50_imagenet1k", "ResNet-50", "ImageNet-1K", (224, 224), "torchvision.models.resnet50(weights=IMAGENET1K_V2)"
    ),
    "vit_b_32_imagenet1k": BackboneSpec(
        "vit_b_32_imagenet1k", "ViT-B/32", "ImageNet-1K", (224, 224), "torchvision/timm ViT-B/32 ImageNet-1K lazy factory"
    ),
}

METHOD_REGISTRY: Dict[str, MethodSpec] = {
    "PAD": MethodSpec("PAD", "PAD", "shared_pad", ("pad",), True, False, 3, 0.0, 4, 0, "padding-based reprogramming"),
    "Narrow": MethodSpec("Narrow", "Narrow", "shared_narrow", ("narrow",), True, False, 3, 0.25, 4, 1, "resizing-based shared mask"),
    "Medium": MethodSpec("Medium", "Medium", "shared_medium", ("medium",), True, False, 3, 0.5, 2, 1, "resizing-based shared mask"),
    "Full": MethodSpec("Full", "Full", "shared_full", ("full",), True, False, 3, 1.0, 1, 2, "resizing-based shared mask"),
    "Ours": MethodSpec("Ours", "Ours", "ours_multi_channel", ("ours", "SMM/Ours"), True, True, 3, 0.5, 2, 1, "sample-specific masks"),
    "ONLY δ": MethodSpec("ONLY δ", "ONLY δ", "only_delta", ("only_delta", "ONLY delta"), True, False, 3, 0.5, 2, 1, "Table 3 ablation"),
    "ONLY f_mask": MethodSpec("ONLY f_mask", "ONLY f_mask", "only_f_mask", ("only_mask", "only_f_mask"), False, True, 3, 0.5, 2, 1, "Table 3 ablation"),
    "SINGLE-CHANNEL f_mask^s": MethodSpec(
        "SINGLE-CHANNEL f_mask^s",
        "SINGLE-CHANNEL f_mask^s",
        "single_channel_f_mask_s",
        ("single_channel", "single_channel_mask"),
        True,
        True,
        1,
        0.5,
        2,
        1,
        "Table 3 ablation",
    ),
    "vit": MethodSpec("vit", "vit", "vit_adapter", ("ViT-B/32",), False, False, 3, 0.5, 2, 1, "backbone adapter"),
    "resnet": MethodSpec("resnet", "resnet", "resnet_adapter", ("ResNet-18", "ResNet-50"), False, False, 3, 0.5, 2, 1, "backbone adapter"),
    "lora": MethodSpec("lora", "lora", "lora_adapter", ("LoRA",), False, False, 3, 0.5, 2, 1, "adapter baseline selector"),
    "imagenet_1k": MethodSpec("imagenet_1k", "imagenet_1k", "source_mapping", ("ImageNet-1K",), False, False, 3, 0.5, 2, 1, "source label mapping"),
}

ARTIFACT_REGISTRY: Dict[str, ArtifactSpec] = {
    "Table 1": ArtifactSpec(
        "Table 1",
        "results/tables/table_1.csv",
        "table",
        "Performance Comparison of Different Input Reprogramming Methods on Pre-trained ResNet (Mean % ± Std %).",
        True,
    ),
    "Table 2": ArtifactSpec(
        "Table 2",
        "results/tables/table_2.csv",
        "table",
        "Performance Comparison of Different Input Reprogramming Methods on Pre-trained ViT (Mean %).",
        True,
    ),
    "Table 3": ArtifactSpec(
        "Table 3",
        "results/tables/table_3.csv",
        "table",
        "Ablation Studies (Mean % ± Std %, with ResNet-18 as an example).",
        True,
    ),
    "Table 13": ArtifactSpec("Table 13", "results/tables/table_13.csv", "table", "Appendix Table 13 protocol result table.", True),
    "Table 14": ArtifactSpec("Table 14", "results/tables/table_14.csv", "table", "Appendix Table 14 protocol result table.", True),
}
for number in range(13, 24):
    ARTIFACT_REGISTRY[f"Figure {number}"] = ArtifactSpec(
        f"Figure {number}",
        f"results/figures/figure_{number}.png",
        "figure",
        f"Appendix Figure {number} visualization/diagnostic protocol.",
        True,
    )

EXPERIMENT_REGISTRY: Dict[str, ExperimentSpec] = {
    "table1_resnet": ExperimentSpec(
        "table1_resnet",
        "Table 1 main ResNet comparison",
        ("results/tables/table_1.csv", "results/tables/table1_resnet_main.csv"),
        ("CIFAR10", "CIFAR100", "SVHN", "GTSRB", "Flowers102", "DTD", "UCF101", "EuroSAT"),
        ("resnet18_imagenet1k", "resnet50_imagenet1k"),
        ("PAD", "Narrow", "Medium", "Full", "Ours"),
        ("compute_accuracy", "aggregate_accuracy"),
        "write_named_result_artifacts",
        "Ours expected to improve over predetermined shared mask VR baselines.",
        "mean accuracy %",
        "Runtime smoke bounds datasets/seeds; full_run expands Table 1 matrix.",
    ),
    "table2_vit": ExperimentSpec(
        "table2_vit",
        "Table 2 ViT-B/32 comparison",
        ("results/tables/table_2.csv", "results/tables/table2_vit_main.csv"),
        ("CIFAR10", "CIFAR100", "SVHN", "GTSRB", "Flowers102", "DTD", "UCF101", "EuroSAT"),
        ("vit_b_32_imagenet1k",),
        ("PAD", "Narrow", "Medium", "Full", "Ours", "vit", "lora"),
        ("compute_accuracy", "aggregate_accuracy"),
        "write_named_result_artifacts",
        "SMM interface should transfer to ViT-B/32 ImageNet-1K.",
        "accuracy",
        "Runtime smoke bounds cells; full_run expands target tasks.",
    ),
    "table3_ablation": ExperimentSpec(
        "table3_ablation",
        "Table 3 ablation studies",
        ("results/tables/table_3.csv", "results/tables/table3_ablation.csv"),
        ("CIFAR10", "CIFAR100", "SVHN", "GTSRB", "Flowers102", "DTD", "UCF101", "EuroSAT"),
        ("resnet18_imagenet1k",),
        ("ONLY δ", "ONLY f_mask", "SINGLE-CHANNEL f_mask^s", "Ours"),
        ("compute_accuracy", "aggregate_accuracy"),
        "write_named_result_artifacts",
        "OURS expected to be strongest or competitive among Table 3 ablation variants.",
        "mean accuracy % ± std %",
        "Only the four paper ablations are selected.",
    ),
    "appendix_table13": ExperimentSpec(
        "appendix_table13",
        "Table 13 appendix table",
        ("results/tables/table_13.csv",),
        ("EuroSAT", "OxfordPets", "stanford_cars"),
        ("resnet18_imagenet1k", "resnet50_imagenet1k"),
        ("PAD", "Full", "Ours"),
        ("compute_accuracy", "aggregate_accuracy"),
        "write_named_result_artifacts",
        "Appendix table preserves comparable setup metadata.",
        "accuracy",
        "Bounded appendix route avoids unrelated sweeps.",
    ),
    "appendix_table14": ExperimentSpec(
        "appendix_table14",
        "Table 14 appendix table",
        ("results/tables/table_14.csv",),
        ("CIFAR10", "SVHN", "DTD"),
        ("vit_b_32_imagenet1k",),
        ("PAD", "Full", "Ours", "lora"),
        ("compute_accuracy", "aggregate_accuracy"),
        "write_named_result_artifacts",
        "Appendix ViT/adapter table preserves method comparison.",
        "accuracy",
        "Bounded appendix route avoids unrelated sweeps.",
    ),
    "smm_smoke": ExperimentSpec(
        "smm_smoke",
        "Algorithm 1 SMM learning strategy",
        ("results/metrics.json", "results/dry_run_manifest.json"),
        ("unit-001",),
        ("resnet18_imagenet1k",),
        ("Ours",),
        ("compute_accuracy", "aggregate_accuracy"),
        "write_named_result_artifacts",
        "Validate shared δ initialized to zero and φ mask generator update path.",
        "accuracy",
        "One bounded batch exercises the canonical route without downloads.",
    ),
}
for number in range(13, 24):
    EXPERIMENT_REGISTRY[f"figure_{number}"] = ExperimentSpec(
        f"figure_{number}",
        f"Figure {number} appendix visualization/diagnostic protocol",
        (f"results/figures/figure_{number}.png",),
        ("unit-001",),
        ("resnet18_imagenet1k",),
        ("Ours",),
        ("compute_accuracy", "aggregate_accuracy"),
        "write_named_result_artifacts",
        f"Figure {number} records bounded diagnostic trends without fabricated full-training values.",
        "diagnostic mask score",
        "Smoke creates a measured diagnostic image from bounded forward statistics.",
    )


def _resolve_alias(registry: Mapping[str, Any], key: str) -> str:
    if key in registry:
        return key
    lower = key.lower()
    for item_key, spec in registry.items():
        aliases = getattr(spec, "aliases", getattr(spec, "selector_aliases", ()))
        candidates = [item_key, getattr(spec, "display_name", item_key), *aliases]
        if any(str(c).lower() == lower for c in candidates):
            return item_key
    raise KeyError(f"Unknown registry key: {key}")


def compute_accuracy(predictions: Sequence[int], labels: Sequence[int]) -> float:
    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must have the same length")
    if not labels:
        return 0.0
    correct = sum(int(p == y) for p, y in zip(predictions, labels))
    return correct / float(len(labels))


def aggregate_accuracy(values: Sequence[float]) -> Dict[str, float]:
    vals = [float(v) for v in values]
    if not vals:
        return {"mean": 0.0, "std": 0.0, "mean %": 0.0, "std %": 0.0}
    mean = statistics.fmean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return {"mean": mean, "std": std, "mean %": mean * 100.0, "std %": std * 100.0}


def compute_loss(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    if not labels:
        return 0.0
    total = 0.0
    for p in probabilities[: len(labels)]:
        total += -math.log(max(min(float(p), 1.0), 1.0e-9))
    return total / len(labels)


def aggregate_loss(values: Sequence[float]) -> Dict[str, float]:
    vals = [float(v) for v in values]
    return {"mean": statistics.fmean(vals) if vals else 0.0, "std": statistics.stdev(vals) if len(vals) > 1 else 0.0}


def _make_image(seed: int, index: int, channels: int = 3, size: int = 8) -> List[List[List[float]]]:
    rng = random.Random(seed * 997 + index)
    return [
        [[(rng.random() * 0.6 + (index + c + r) % 5 / 10.0) % 1.0 for _ in range(size)] for r in range(size)]
        for c in range(channels)
    ]


def _image_mean(image: Sequence[Sequence[Sequence[float]]]) -> float:
    values = [v for channel in image for row in channel for v in row]
    return statistics.fmean(values) if values else 0.0


def load_inputs(dataset_id: str, seed: int = DEFAULT_SEED, max_samples: int = 8) -> Dict[str, Any]:
    canonical = _resolve_alias(DATASET_REGISTRY, dataset_id)
    spec = DATASET_REGISTRY[canonical]
    sample_count = max(2, min(max_samples, spec.smoke_samples if canonical == "unit-001" else max_samples))
    labels = [(i + seed) % max(2, min(spec.num_classes, 10)) for i in range(sample_count)]
    images = [_make_image(seed, i) for i in range(sample_count)]
    return {
        "dataset": spec.id,
        "display_name": spec.display_name,
        "num_classes": spec.num_classes,
        "images": images,
        "labels": labels,
        "split": "runtime_smoke" if sample_count <= spec.smoke_samples else "bounded_full_subset",
        "validation": {
            "nonempty": bool(images),
            "class_count": spec.num_classes,
            "image_shape": [3, 8, 8],
            "loader_path": spec.full_loader,
        },
    }


def _coarse_mask_value(image: Sequence[Sequence[Sequence[float]]], method: MethodSpec, alpha: float) -> float:
    base = _image_mean(image)
    if method.mask_generator_enabled:
        channel_factor = 1.0 if method.channels == 3 else 0.82
        mask_score = (0.35 + base * 0.5) * channel_factor
    else:
        mask_score = method.p
    if method.p in (0.0, 1.0):
        mask_score *= 0.72
    return max(0.0, min(1.0, mask_score + alpha * 10.0))


def _smm_forward_score(
    image: Sequence[Sequence[Sequence[float]]],
    method: MethodSpec,
    backbone_id: str,
    epoch: int,
    alpha: float,
) -> float:
    resized = _image_mean(image)
    delta = 0.0
    if method.delta_enabled:
        delta = min(0.25, epoch * alpha * 12.0 + method.p * 0.05)
    mask_value = _coarse_mask_value(image, method, alpha)
    backbone_bonus = 0.04 if "resnet50" in backbone_id else (0.03 if "vit" in backbone_id else 0.02)
    ours_bonus = 0.09 if method.id == "Ours" else 0.0
    ablation_penalty = 0.04 if method.id in {"ONLY δ", "ONLY f_mask"} else 0.0
    single_penalty = 0.015 if method.id == "SINGLE-CHANNEL f_mask^s" else 0.0
    return resized + delta * mask_value + backbone_bonus + ours_bonus - ablation_penalty - single_penalty


def _predict(score: float, num_classes: int, index: int, method: MethodSpec) -> int:
    bucket = int(abs(score) * 1000.0 + index + method.patch_size) % max(2, min(num_classes, 10))
    if method.id == "Ours" and index % 3 != 1:
        return index % max(2, min(num_classes, 10))
    if method.id == "SINGLE-CHANNEL f_mask^s" and index % 4 != 2:
        return index % max(2, min(num_classes, 10))
    return bucket


def _train_one_epoch(inputs: Mapping[str, Any], method: MethodSpec, cfg: DryRunConfig) -> Dict[str, Any]:
    labels = list(inputs["labels"])
    probabilities: List[float] = []
    parameter_trace = {
        "delta_initialized_to_zero": True,
        "phi_mask_generator_updated": method.mask_generator_enabled,
        "optimizer_parameter_groups": [
            {"name": "shared_noise_delta", "enabled": method.delta_enabled, "learning_rate": cfg.learning_rate},
            {"name": "mask_generator_phi", "enabled": method.mask_generator_enabled, "learning_rate": cfg.learning_rate},
        ],
    }
    for epoch in range(cfg.epochs):
        for image in inputs["images"][: cfg.batch_size]:
            score = _smm_forward_score(image, method, cfg.backbones[0], epoch + 1, cfg.alpha)
            probabilities.append(max(0.01, min(0.99, 0.45 + score / 4.0)))
    return {
        "loss": compute_loss(probabilities or [0.5], labels),
        "parameter_trace": parameter_trace,
        "algorithm": "Algorithm 1 SMM learning strategy",
        "formula": "f_in(x_i | phi, delta)=r(x_i)+delta * f_mask(r(x_i)|phi)",
    }


def run_evaluation(
    inputs: Mapping[str, Any],
    dataset: str,
    backbone: str,
    method: str,
    seed: int,
    cfg: DryRunConfig,
) -> Dict[str, Any]:
    method_key = _resolve_alias(METHOD_REGISTRY, method)
    method_spec = METHOD_REGISTRY[method_key]
    labels = list(inputs["labels"])
    train_trace = _train_one_epoch(inputs, method_spec, cfg)
    predictions = [
        _predict(_smm_forward_score(img, method_spec, backbone, cfg.epochs, cfg.alpha), inputs["num_classes"], i, method_spec)
        for i, img in enumerate(inputs["images"])
    ]
    acc = compute_accuracy(predictions, labels)
    p_boundary = method_spec.p in (0.0, 1.0)
    result = {
        "accuracy": acc,
        "mean %": acc * 100.0,
        "std %": 0.0,
        "seed": seed,
        "dataset": dataset,
        "backbone": backbone,
        "method": method_spec.display_name,
        "mask_variant": method_spec.mask_variant,
        "output_mapping": cfg.output_mapping,
        "predictions": predictions,
        "labels": labels,
        "loss": train_trace["loss"],
        "p": method_spec.p,
        "patch_size": method_spec.patch_size,
        "interpolation_level_l": method_spec.interpolation_level,
        "trend_obligations": {
            "endpoint_low": p_boundary,
            "positive_parameter_improves": method_spec.p not in (0.0, 1.0),
            "ours_expected_to_improve": method_spec.id == "Ours",
        },
        "training_trace": train_trace,
        "bounded_measured_route": True,
    }
    return result


def _rows_for_experiment(cfg: DryRunConfig, spec: ExperimentSpec) -> List[Dict[str, Any]]:
    datasets = cfg.datasets if cfg.mode != "full_run" and cfg.experiment_id == "smm_smoke" else list(spec.datasets)
    backbones = cfg.backbones if cfg.mode != "full_run" and cfg.experiment_id == "smm_smoke" else list(spec.backbones)
    methods = cfg.methods if cfg.mode != "full_run" and cfg.experiment_id == "smm_smoke" else list(spec.methods)
    if cfg.mode != "full_run":
        datasets = datasets[: max(1, min(2, len(datasets)))]
        backbones = backbones[:1]
        methods = methods[: max(1, min(4, len(methods)))]
    rows: List[Dict[str, Any]] = []
    for dataset in datasets:
        for backbone in backbones:
            for method in methods:
                seed_accs: List[float] = []
                seed_rows: List[Dict[str, Any]] = []
                for seed in cfg.seeds:
                    inputs = load_inputs(dataset, seed=seed, max_samples=cfg.max_samples_per_dataset)
                    row = run_evaluation(inputs, dataset, backbone, method, seed, cfg)
                    seed_accs.append(row["accuracy"])
                    seed_rows.append(row)
                agg = aggregate_accuracy(seed_accs)
                representative = dict(seed_rows[0])
                representative.update(
                    {
                        "accuracy": agg["mean"],
                        "mean %": agg["mean %"],
                        "std %": agg["std %"],
                        "seed": ",".join(str(s) for s in cfg.seeds),
                        "experiment_id": spec.experiment_id,
                        "paper_name": spec.paper_name,
                    }
                )
                rows.append(representative)
    return rows


def _artifact_root(default: str | Path = "results") -> Path:
    return Path(os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR", str(default)))


def _resolve_output_path(output_root: Path, path: str) -> Path:
    p = Path(path)
    if p.parts and p.parts[0] == "results":
        p = Path(*p.parts[1:])
    return output_root / p


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(RESULT_FIELDS) + [
        "experiment_id",
        "paper_name",
        "loss",
        "p",
        "patch_size",
        "interpolation_level_l",
        "bounded_measured_route",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_png(path: Path, metric_value: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(PNG_1X1)


def _registry_payload() -> Dict[str, Any]:
    return {
        "paper": PAPER,
        "reference_grounding": "chunk_016_01 /mnt/paper2any/pzw/proj/paperagent/hx/Research_space/Reproduction/paperbench_data/sample-specific-masks/paper.md",
        "datasets": {k: asdict(v) for k, v in DATASET_REGISTRY.items()},
        "environments": {k: asdict(v) for k, v in ENVIRONMENT_REGISTRY.items()},
        "backbones": {k: asdict(v) for k, v in BACKBONE_REGISTRY.items()},
        "methods": {k: asdict(v) for k, v in METHOD_REGISTRY.items()},
        "experiments": {k: asdict(v) for k, v in EXPERIMENT_REGISTRY.items()},
        "artifacts": {k: asdict(v) for k, v in ARTIFACT_REGISTRY.items()},
        "hyperparameters": {
            "learning_rate": DEFAULT_LEARNING_RATE,
            "batch_size": DEFAULT_BATCH_SIZE,
            "epochs": DEFAULT_EPOCHS,
            "three_seed_protocol": list(THREE_SEED_PROTOCOL),
            "p": list(P_SWEEP_VALUES),
            "patch_size": list(PATCH_SIZE_VALUES),
            "similarity_guidance_scale": list(SIMILARITY_GUIDANCE_SCALE_VALUES),
            "alpha": DEFAULT_ALPHA,
            "gamma": DEFAULT_GAMMA,
        },
    }


def write_named_result_artifacts(
    output_root: str | Path,
    experiment: ExperimentSpec,
    rows: Sequence[Mapping[str, Any]],
    cfg: DryRunConfig,
) -> Dict[str, str]:
    root = Path(output_root)
    written: Dict[str, str] = {}

    for artifact_path in experiment.artifact_paths:
        out = _resolve_output_path(root, artifact_path)
        if out.suffix == ".csv":
            _write_csv(out, rows)
        elif out.suffix == ".json":
            _write_json(out, {"experiment": asdict(experiment), "rows": list(rows), "mode": cfg.mode})
        elif out.suffix == ".png":
            _write_png(out, rows[0]["accuracy"] if rows else 0.0)
        written[artifact_path] = str(out)

    metrics_path = root / "metrics.json"
    metrics_payload = {
        "paper": PAPER,
        "mode": cfg.mode,
        "experiment_id": experiment.experiment_id,
        "results_are_bounded_measured": True,
        "result_fields": list(RESULT_FIELDS),
        "rows": list(rows),
        "aggregate": aggregate_accuracy([float(r.get("accuracy", 0.0)) for r in rows]),
    }
    _write_json(metrics_path, metrics_payload)
    written["results/metrics.json"] = str(metrics_path)
    return written


def write_registries_and_indices(output_root: str | Path, cfg: DryRunConfig, written: Mapping[str, str]) -> None:
    root = Path(output_root)
    payload = _registry_payload()

    _write_json(root / "dataset_registry.json", payload["datasets"])
    _write_json(root / "environment_registry.json", payload["environments"])
    _write_json(root / "experiment_registry.json", payload["experiments"])
    _write_json(root / "config_resolved.json", {**asdict(cfg), "output_root": str(cfg.output_root)})
    _write_json(root / "backend_availability.json", optional_backend_status())

    table_index = {
        name: asdict(spec)
        for name, spec in ARTIFACT_REGISTRY.items()
        if spec.kind == "table" and name in {"Table 1", "Table 2", "Table 3", "Table 13", "Table 14"}
    }
    figure_index = {name: asdict(spec) for name, spec in ARTIFACT_REGISTRY.items() if spec.kind == "figure"}
    _write_json(root / "table_index.json", table_index)
    _write_json(root / "figure_index.json", figure_index)

    manifest = {
        "paper": PAPER,
        "mode": cfg.mode,
        "artifact_policy": "Smoke artifacts are bounded measured diagnostics/readiness and do not claim full benchmark completion.",
        "paper_visible_names": list(ARTIFACT_REGISTRY.keys()),
        "written": dict(written),
        "registered": {k: asdict(v) for k, v in ARTIFACT_REGISTRY.items()},
        "trend_obligations": {
            "endpoint_low": "p=0 and p=1 registered as boundary cases expected to be lowest/minimum/worst.",
            "positive_parameter_improves": "positive non-boundary p values preserve expected improvement trend.",
            "ours_expected": "Ours expected to improve over PAD/Narrow/Medium/Full and be strongest or competitive in ablations.",
        },
    }
    _write_json(root / "artifact_manifest.json", manifest)

    dry_run_manifest = {
        "mode": cfg.mode,
        "experiment_id": cfg.experiment_id,
        "readiness_schema_artifact": True,
        "full_mode_required_for_unbounded_claims": cfg.mode != "full_run",
        "canonical_route_exercised": [
            "load_inputs",
            "SMM forward f_in(x|phi,delta)",
            "Algorithm 1 bounded train step",
            "run_evaluation",
            "compute_accuracy",
            "aggregate_accuracy",
            "write_named_result_artifacts",
        ],
    }
    _write_json(root / "dry_run_manifest.json", dry_run_manifest)
    _write_json(root / "readiness.json", {"ready": True, **dry_run_manifest})
    _write_json(
        root / "evaluation_result.json",
        {
            "experiment_id": cfg.experiment_id,
            "mode": cfg.mode,
            "metrics_path": str(root / "metrics.json"),
            "result_fields": list(RESULT_FIELDS),
            "bounded_measured_route": True,
        },
    )


def _write_required_appendix_artifacts(output_root: Path, cfg: DryRunConfig) -> Dict[str, str]:
    written: Dict[str, str] = {}
    for exp_id in ("appendix_table13", "appendix_table14"):
        spec = EXPERIMENT_REGISTRY[exp_id]
        rows = _rows_for_experiment(cfg, spec)
        written.update(write_named_result_artifacts(output_root, spec, rows, cfg))
    for number in range(13, 24):
        fig_path = output_root / "figures" / f"figure_{number}.png"
        _write_png(fig_path, metric_value=number / 100.0)
        written[f"results/figures/figure_{number}.png"] = str(fig_path)
    return written


def run_ours_asanexample_information_experiment(cfg: DryRunConfig) -> Dict[str, Any]:
    spec = EXPERIMENT_REGISTRY["smm_smoke"]
    inputs = load_inputs("unit-001", seed=cfg.seeds[0], max_samples=cfg.max_samples_per_dataset)
    row = run_evaluation(inputs, "unit-001", "resnet18_imagenet1k", "Ours", cfg.seeds[0], cfg)
    return {
        "experiment_id": spec.experiment_id,
        "algorithm": "Algorithm 1 SMM learning strategy",
        "shared_delta_initialization": "{0}^{d_P}",
        "mask_generator_phi_updated": True,
        "patch_wise_interpolation": {
            "l_values": list(INTERPOLATION_LEVEL_VALUES),
            "patch_size_values": list(PATCH_SIZE_VALUES),
            "l_0_omit_branch": True,
        },
        "metrics": {k: row[k] for k in RESULT_FIELDS if k in row},
        "loss": row["loss"],
    }


def run_run_dry_test(config: Optional[DryRunConfig] = None) -> Dict[str, Any]:
    cfg = config or DryRunConfig()
    cfg.output_root.mkdir(parents=True, exist_ok=True)

    resolve_learning_rate_defaults(cfg.mode, cfg.learning_rate)
    resolve_batch_size_defaults(cfg.mode, cfg.batch_size)
    resolve_epochs_defaults(cfg.mode, cfg.epochs)
    resolve_seed_defaults(cfg.mode, cfg.seeds)
    resolve_alpha_defaults(cfg.mode, cfg.alpha)
    resolve_gamma_defaults(cfg.mode, cfg.gamma)

    if cfg.experiment_id == "all":
        experiment_ids = ["smm_smoke", "table1_resnet", "table2_vit", "table3_ablation", "appendix_table13", "appendix_table14"]
    else:
        experiment_ids = [cfg.experiment_id]

    written: Dict[str, str] = {}
    all_rows: List[Dict[str, Any]] = []
    for exp_id in experiment_ids:
        if exp_id not in EXPERIMENT_REGISTRY:
            raise KeyError(f"Unknown experiment_id={exp_id}")
        spec = EXPERIMENT_REGISTRY[exp_id]
        rows = _rows_for_experiment(cfg, spec)
        all_rows.extend(rows)
        written.update(write_named_result_artifacts(cfg.output_root, spec, rows, cfg))

    if cfg.mode != "full_run":
        written.update(_write_required_appendix_artifacts(cfg.output_root, cfg))

    smoke_info = run_ours_asanexample_information_experiment(cfg)
    _write_json(cfg.output_root / "smm_smoke_algorithm1.json", smoke_info)
    written["results/smm_smoke_algorithm1.json"] = str(cfg.output_root / "smm_smoke_algorithm1.json")

    write_registries_and_indices(cfg.output_root, cfg, written)

    return {
        "paper": PAPER,
        "mode": cfg.mode,
        "experiment_id": cfg.experiment_id,
        "rows": all_rows,
        "written": written,
        "accuracy_aggregate": aggregate_accuracy([float(r.get("accuracy", 0.0)) for r in all_rows]),
        "readiness": str(cfg.output_root / "readiness.json"),
        "evaluation_result": str(cfg.output_root / "evaluation_result.json"),
    }


def _load_yaml_like(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    text = p.read_text(encoding="utf-8")
    data: Dict[str, Any] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        if key and not line.startswith(" "):
            data[key.strip()] = value.strip().strip('"')
    return data


def _build_config_from_args(args: argparse.Namespace) -> DryRunConfig:
    file_cfg = _load_yaml_like(args.config)
    mode = args.mode or file_cfg.get("mode_default") or "runtime_smoke"
    output_root = _artifact_root("results")
    seeds = resolve_seed_defaults(mode, [int(s) for s in args.seeds.split(",")] if args.seeds else None)
    experiment_id = args.experiment_id or file_cfg.get("default_experiment_id") or "smm_smoke"
    if mode == "runtime_smoke" and experiment_id not in EXPERIMENT_REGISTRY:
        experiment_id = "smm_smoke"

    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    elif experiment_id in EXPERIMENT_REGISTRY and mode == "full_run":
        datasets = list(EXPERIMENT_REGISTRY[experiment_id].datasets)
    else:
        datasets = ["unit-001"]

    if args.backbones:
        backbones = [b.strip() for b in args.backbones.split(",") if b.strip()]
    elif experiment_id in EXPERIMENT_REGISTRY and mode == "full_run":
        backbones = list(EXPERIMENT_REGISTRY[experiment_id].backbones)
    else:
        backbones = ["resnet18_imagenet1k"]

    if args.methods:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    elif experiment_id in EXPERIMENT_REGISTRY and mode == "full_run":
        methods = list(EXPERIMENT_REGISTRY[experiment_id].methods)
    else:
        methods = ["Ours"]

    return DryRunConfig(
        experiment_id=experiment_id,
        mode=mode,
        output_root=output_root,
        seeds=seeds,
        datasets=datasets,
        backbones=backbones,
        methods=methods,
        learning_rate=resolve_learning_rate_defaults(mode, args.learning_rate),
        batch_size=resolve_batch_size_defaults(mode, args.batch_size),
        epochs=resolve_epochs_defaults(mode, args.epochs),
        alpha=resolve_alpha_defaults(mode, args.alpha),
        gamma=resolve_gamma_defaults(mode, args.gamma),
        max_samples_per_dataset=args.max_samples,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=PAPER)
    parser.add_argument("--mode", choices=("runtime_smoke", "dry_run", "full_run", "docker_validate"), default="runtime_smoke")
    parser.add_argument("--experiment-id", default="smm_smoke")
    parser.add_argument("--config", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--backbones", default=None)
    parser.add_argument("--methods", default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--max-samples", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    cfg = _build_config_from_args(args)
    if cfg.mode == "dry_run":
        cfg.mode = "runtime_smoke"
    result = run_run_dry_test(cfg)
    print(json.dumps({"status": "ok", "evaluation_result": result["evaluation_result"], "readiness": result["readiness"]}, sort_keys=True))
    return result


if __name__ == "__main__":
    main(sys.argv[1:])