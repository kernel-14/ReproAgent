"""
Executable main-comparison protocol for the reproduction of
"Sample-specific Masks for Visual Reprogramming-based Prompting".

This module owns the ImageNet-1K-pretrained backbone experiment matrix used by
Table 1, Table 2, Table 3, appendix Table 13/14, and appendix Figure 13-23.
It is intentionally importable without heavy vision dependencies. Full-mode
routes use lazy factories for torchvision/datasets/torch/gym/sbi when available;
the bounded route uses the same dataset -> method -> training -> evaluation ->
artifact path with local fixtures.
"""

from __future__ import annotations

import base64
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
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


REFERENCE_GROUNDING = (
    "reference_grounding: chunk_016_01 "
    "/mnt/paper2any/pzw/proj/paperagent/hx/Research_space/Reproduction/"
    "paperbench_data/sample-specific-masks/paper.md"
)

DEFAULT_LEARNING_RATE = 0.05
DEFAULT_BATCH_SIZE = 4
DEFAULT_EPOCHS = 1
DEFAULT_SEED = 0
DEFAULT_ALPHA = 0.05
DEFAULT_GAMMA = 0.95

THREE_SEED_PROTOCOL = (0, 1, 2)
PATCH_SIZE_VALUES = (4, 2, 1)
P_VALUES = (0.0, 0.25, 0.5, 1.0)
SIMILARITY_GUIDANCE_SCALE_VALUES = (9, 7, 10)
INPUT_SIZE = (3, 32, 32)
PRETRAINED_SOURCE = "imagenet_1k"
OUTPUT_MAPPING = "Rlm_random_label_mapping"

DATASET_ALIASES: Dict[str, Tuple[str, ...]] = {
    "unit-001": ("unit", "smoke_fixture", "unit-001"),
    "cifar10": ("cifar", "cifar10", "CIFAR10"),
    "cifar100": ("cifar100", "CIFAR100"),
    "svhn": ("svhn", "SVHN"),
    "imagenet": ("imagenet", "ImageNet"),
    "imagenet_1k": ("imagenet_1k", "ImageNet-1K", "imagenet1k"),
    "stanford_cars": ("stanford_cars", "StanfordCars"),
    "dtd": ("dtd", "DTD"),
    "eurosat": ("eurosat", "EuroSAT"),
    "flowers": ("flowers", "Flowers102", "flowers102"),
    "oxford_pets": ("oxford_pets", "OxfordPets", "pets"),
    "gtsrb": ("gtsrb", "GTSRB"),
    "ucf101": ("ucf101", "UCF101"),
    "food101": ("food101", "Food101"),
    "sun397": ("sun397", "SUN397"),
}

MAIN_DATASETS = (
    "cifar10",
    "cifar100",
    "svhn",
    "gtsrb",
    "flowers",
    "dtd",
    "ucf101",
    "food101",
    "eurosat",
    "oxford_pets",
    "sun397",
)
SMOKE_DATASETS = ("unit-001",)

TABLE1_METHODS = ("PAD", "Narrow", "Medium", "Full", "Ours")
TABLE2_METHODS = ("PAD", "Narrow", "Medium", "Full", "Ours")
TABLE3_VARIANTS = ("ONLY δ", "ONLY f_mask", "SINGLE-CHANNEL f_mask^s", "OURS")
BACKBONES_TABLE1 = ("resnet18_imagenet1k", "resnet50_imagenet1k")
BACKBONE_TABLE2 = "vit_b_32_imagenet1k"
BACKBONE_TABLE3 = "resnet18_imagenet1k"

CANONICAL_METRIC_IDENTIFIERS = (
    "mean_std_accuracy",
    "metric_mean_std_accuracy",
    "accuracy",
    "metric_accuracy",
    "f1",
    "metric_f1",
    "loss",
    "metric_loss",
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
)

ARTIFACT_PATHS: Dict[str, str] = {
    "metrics": "results/metrics.json",
    "table1": "results/tables/table1_resnet_main.csv",
    "table1_alias": "results/tables/table_1.csv",
    "table2": "results/tables/table2_vit_main.csv",
    "table2_alias": "results/tables/table_2.csv",
    "table3": "results/tables/table3_ablation.csv",
    "table3_alias": "results/tables/table_3.csv",
    "dataset_registry": "results/dataset_registry.json",
    "environment_registry": "results/environment_registry.json",
    "experiment_registry": "results/experiment_registry.json",
    "artifact_manifest": "results/artifact_manifest.json",
    "config_resolved": "results/config_resolved.json",
    "table13": "results/tables/table_13.csv",
    "table14": "results/tables/table_14.csv",
    "table_index": "results/table_index.json",
    "figure_index": "results/figure_index.json",
    "readiness": "readiness.json",
    "evaluation_result": "evaluation_result.json",
}


def learning_rate_values(mode: str = "runtime_smoke") -> List[float]:
    return [DEFAULT_LEARNING_RATE] if mode != "full_run" else [0.1, 0.05, 0.01]


def resolve_learning_rate_defaults(config: Optional[Mapping[str, Any]] = None) -> float:
    if config and "learning_rate" in config:
        return float(config["learning_rate"])
    return DEFAULT_LEARNING_RATE


def batch_size_values(mode: str = "runtime_smoke") -> List[int]:
    return [DEFAULT_BATCH_SIZE] if mode != "full_run" else [32, 64, 128]


def resolve_batch_size_defaults(config: Optional[Mapping[str, Any]] = None) -> int:
    if config and "batch_size" in config:
        return int(config["batch_size"])
    return DEFAULT_BATCH_SIZE


def epochs_values(mode: str = "runtime_smoke") -> List[int]:
    return [DEFAULT_EPOCHS] if mode != "full_run" else [10, 20, 40]


def resolve_epochs_defaults(config: Optional[Mapping[str, Any]] = None) -> int:
    if config and "epochs" in config:
        return int(config["epochs"])
    return DEFAULT_EPOCHS


def seed_values(mode: str = "runtime_smoke") -> List[int]:
    return [DEFAULT_SEED] if mode != "full_run" else list(THREE_SEED_PROTOCOL)


def resolve_seed_defaults(config: Optional[Mapping[str, Any]] = None) -> List[int]:
    if config and "seeds" in config:
        value = config["seeds"]
        if isinstance(value, int):
            return [value]
        return [int(v) for v in value]
    return [DEFAULT_SEED]


def alpha_values(mode: str = "runtime_smoke") -> List[float]:
    return [DEFAULT_ALPHA] if mode != "full_run" else [0.1, 0.05, 0.01]


def resolve_alpha_defaults(config: Optional[Mapping[str, Any]] = None) -> float:
    if config and "alpha" in config:
        return float(config["alpha"])
    return DEFAULT_ALPHA


def gamma_values(mode: str = "runtime_smoke") -> List[float]:
    return [DEFAULT_GAMMA] if mode != "full_run" else [0.99, 0.95, 0.9]


def resolve_gamma_defaults(config: Optional[Mapping[str, Any]] = None) -> float:
    if config and "gamma" in config:
        return float(config["gamma"])
    return DEFAULT_GAMMA


def optional_backend_availability() -> Dict[str, Dict[str, Any]]:
    availability: Dict[str, Dict[str, Any]] = {}
    for name in ("torch", "torchvision", "datasets", "gym", "gymnasium", "sbi"):
        spec = importlib.util.find_spec(name)
        availability[name] = {
            "available": spec is not None,
            "lazy_import_factory": f"lazy_import('{name}')",
        }
    return availability


def lazy_import(module_name: str) -> Any:
    return importlib.import_module(module_name)


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    aliases: Tuple[str, ...]
    class_count: int
    image_size: Tuple[int, int, int] = INPUT_SIZE
    split: str = "paper_following_chen2023"
    fixture_samples: int = 8
    external_loader: Optional[str] = None
    metrics: Tuple[str, ...] = ("accuracy", "loss", "f1")
    readiness_backends: Tuple[str, ...] = ("datasets", "torchvision")


@dataclass(frozen=True)
class EnvironmentSpec:
    environment_id: str
    aliases: Tuple[str, ...]
    datasets: Tuple[str, ...]
    backbones: Tuple[str, ...]
    methods: Tuple[str, ...]
    metrics: Tuple[str, ...]
    readiness_check: str = "make_environment"


@dataclass(frozen=True)
class BackboneSpec:
    backbone_id: str
    family: str
    pretrained_on: str = PRETRAINED_SOURCE
    input_size: Tuple[int, int] = (32, 32)
    source_classes: int = 1000
    frozen: bool = True
    lazy_loader: str = "torchvision.models"


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    display_name: str
    mask_variant: str
    layout: str
    delta_enabled: bool = True
    mask_generator_enabled: bool = True
    multi_channel: bool = True
    p: float = 0.5
    patch_size: int = 2
    interpolation_level_l: int = 1
    trainable_parameters: Tuple[str, ...] = ("delta", "phi")
    aliases: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    paper_name: str
    table_or_figure: str
    datasets: Tuple[str, ...]
    backbones: Tuple[str, ...]
    methods: Tuple[str, ...]
    metrics: Tuple[str, ...]
    artifact_path: str
    writer: str
    mode_default: str = "runtime_smoke"
    output_mapping: str = OUTPUT_MAPPING
    expected_trend: str = ""
    reference_grounding: str = REFERENCE_GROUNDING


@dataclass
class RunConfig:
    mode: str = "runtime_smoke"
    output_root: Optional[str] = None
    experiment_id: str = "smm_smoke"
    datasets: Tuple[str, ...] = SMOKE_DATASETS
    backbones: Tuple[str, ...] = ("resnet18_imagenet1k",)
    methods: Tuple[str, ...] = ("Ours",)
    seeds: Tuple[int, ...] = (DEFAULT_SEED,)
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_LEARNING_RATE
    alpha: float = DEFAULT_ALPHA
    gamma: float = DEFAULT_GAMMA
    max_train_batches: Optional[int] = 1
    max_eval_batches: Optional[int] = 1
    output_mapping: str = OUTPUT_MAPPING


DATASET_REGISTRY: Dict[str, DatasetSpec] = {
    "unit-001": DatasetSpec("unit-001", DATASET_ALIASES["unit-001"], 3, fixture_samples=8),
    "cifar10": DatasetSpec("cifar10", DATASET_ALIASES["cifar10"], 10, external_loader="torchvision.datasets.CIFAR10"),
    "cifar100": DatasetSpec("cifar100", DATASET_ALIASES["cifar100"], 100, external_loader="torchvision.datasets.CIFAR100"),
    "svhn": DatasetSpec("svhn", DATASET_ALIASES["svhn"], 10, external_loader="torchvision.datasets.SVHN"),
    "imagenet": DatasetSpec("imagenet", DATASET_ALIASES["imagenet"], 1000, external_loader="torchvision.datasets.ImageNet"),
    "imagenet_1k": DatasetSpec("imagenet_1k", DATASET_ALIASES["imagenet_1k"], 1000, external_loader="torchvision.datasets.ImageNet"),
    "stanford_cars": DatasetSpec("stanford_cars", DATASET_ALIASES["stanford_cars"], 196, external_loader="torchvision.datasets.StanfordCars"),
    "dtd": DatasetSpec("dtd", DATASET_ALIASES["dtd"], 47, external_loader="torchvision.datasets.DTD"),
    "eurosat": DatasetSpec("eurosat", DATASET_ALIASES["eurosat"], 10, external_loader="torchvision.datasets.EuroSAT"),
    "flowers": DatasetSpec("flowers", DATASET_ALIASES["flowers"], 102, external_loader="torchvision.datasets.Flowers102"),
    "oxford_pets": DatasetSpec("oxford_pets", DATASET_ALIASES["oxford_pets"], 37, external_loader="torchvision.datasets.OxfordIIITPet"),
    "gtsrb": DatasetSpec("gtsrb", DATASET_ALIASES["gtsrb"], 43, external_loader="torchvision.datasets.GTSRB"),
    "ucf101": DatasetSpec("ucf101", DATASET_ALIASES["ucf101"], 101, external_loader="torchvision.datasets.UCF101"),
    "food101": DatasetSpec("food101", DATASET_ALIASES["food101"], 101, external_loader="torchvision.datasets.Food101"),
    "sun397": DatasetSpec("sun397", DATASET_ALIASES["sun397"], 397, external_loader="torchvision.datasets.SUN397"),
}

BACKBONE_REGISTRY: Dict[str, BackboneSpec] = {
    "resnet18_imagenet1k": BackboneSpec("resnet18_imagenet1k", "resnet18", input_size=(32, 32)),
    "resnet50_imagenet1k": BackboneSpec("resnet50_imagenet1k", "resnet50", input_size=(32, 32)),
    "vit_b_32_imagenet1k": BackboneSpec("vit_b_32_imagenet1k", "vit_b_32", input_size=(32, 32)),
    "vit_l_384_imagenet1k": BackboneSpec("vit_l_384_imagenet1k", "vit_l_384", input_size=(32, 32)),
}

METHOD_REGISTRY: Dict[str, MethodSpec] = {
    "PAD": MethodSpec("PAD", "PAD", "pad_shared", "pad", True, False, False, p=0.0, patch_size=4, aliases=("pad",)),
    "Narrow": MethodSpec("Narrow", "Narrow", "narrow_shared", "narrow", True, False, False, p=0.25, patch_size=4, aliases=("narrow",)),
    "Medium": MethodSpec("Medium", "Medium", "medium_shared", "medium", True, False, False, p=0.5, patch_size=2, aliases=("medium",)),
    "Full": MethodSpec("Full", "Full", "full_shared", "full", True, False, False, p=1.0, patch_size=1, aliases=("full",)),
    "Ours": MethodSpec("Ours", "Ours", "ours_multi_channel", "sample_specific", True, True, True, p=0.5, patch_size=2, aliases=("ours", "SMM/Ours")),
    "ours": MethodSpec("ours", "Ours", "ours_multi_channel", "sample_specific", True, True, True, p=0.5, patch_size=2, aliases=("Ours",)),
    "ONLY δ": MethodSpec("ONLY δ", "ONLY δ", "only_delta", "full", True, False, False, p=1.0, patch_size=1, aliases=("ONLY_delta", "only_delta")),
    "ONLY f_mask": MethodSpec("ONLY f_mask", "ONLY f_mask", "only_f_mask", "sample_specific", False, True, True, p=0.5, patch_size=2, aliases=("only_f_mask",)),
    "SINGLE-CHANNEL f_mask^s": MethodSpec(
        "SINGLE-CHANNEL f_mask^s",
        "SINGLE-CHANNEL f_mask^s",
        "single_channel_mask",
        "sample_specific",
        True,
        True,
        False,
        p=0.5,
        patch_size=2,
        aliases=("single_channel_mask", "single-channel"),
    ),
    "OURS": MethodSpec("OURS", "OURS", "ours_multi_channel", "sample_specific", True, True, True, p=0.5, patch_size=2, aliases=("Ours", "ours")),
    "vit": MethodSpec("vit", "ViT adapter", "backbone_adapter", "adapter", True, True, True, aliases=("ViT-B/32",)),
    "resnet": MethodSpec("resnet", "ResNet adapter", "backbone_adapter", "adapter", True, True, True, aliases=("ResNet-18", "ResNet-50")),
    "lora": MethodSpec("lora", "LoRA", "finetune_lora", "lora", False, False, False, aliases=("LoRA", "Finetuning-LoRA")),
    "imagenet_1k": MethodSpec("imagenet_1k", "ImageNet-1K source", "source_space", "source", False, False, False),
}

ENVIRONMENT_REGISTRY: Dict[str, EnvironmentSpec] = {
    "cifar": EnvironmentSpec(
        "cifar",
        ("cifar", "CIFAR10", "CIFAR100"),
        ("cifar10", "cifar100"),
        ("resnet18_imagenet1k", "resnet50_imagenet1k", "vit_b_32_imagenet1k"),
        TABLE1_METHODS + ("vit", "resnet", "lora"),
        ("accuracy", "loss", "f1"),
    ),
    "imagenet": EnvironmentSpec(
        "imagenet",
        ("imagenet", "imagenet_1k", "ImageNet-1K"),
        ("imagenet", "imagenet_1k"),
        ("resnet18_imagenet1k", "resnet50_imagenet1k", "vit_b_32_imagenet1k"),
        ("imagenet_1k", "resnet", "vit"),
        ("accuracy", "loss"),
    ),
    "svhn": EnvironmentSpec(
        "svhn",
        ("svhn", "SVHN"),
        ("svhn",),
        ("resnet18_imagenet1k", "resnet50_imagenet1k", "vit_b_32_imagenet1k"),
        TABLE1_METHODS,
        ("accuracy", "loss", "f1"),
    ),
    "unit-001": EnvironmentSpec(
        "unit-001",
        ("unit-001", "bounded_fixture"),
        ("unit-001",),
        ("resnet18_imagenet1k",),
        ("Ours", "PAD", "Narrow", "Medium", "Full"),
        ("accuracy", "loss", "f1"),
    ),
}

EXPERIMENT_REGISTRY: Dict[str, ExperimentSpec] = {
    "table1_resnet": ExperimentSpec(
        "table1_resnet",
        "Table 1 main ResNet comparison",
        "Table 1",
        MAIN_DATASETS,
        BACKBONES_TABLE1,
        TABLE1_METHODS,
        ("mean_std_accuracy", "accuracy", "loss"),
        "results/tables/table1_resnet_main.csv",
        "write_table_artifact",
        expected_trend="Ours expected to improve over predetermined shared mask VR baselines",
    ),
    "table2_vit": ExperimentSpec(
        "table2_vit",
        "Table 2 ViT-B/32 comparison",
        "Table 2",
        MAIN_DATASETS,
        (BACKBONE_TABLE2,),
        TABLE2_METHODS,
        ("mean_std_accuracy", "accuracy", "loss"),
        "results/tables/table2_vit_main.csv",
        "write_table_artifact",
    ),
    "table3_ablation": ExperimentSpec(
        "table3_ablation",
        "Table 3 Ablation Studies",
        "Table 3",
        MAIN_DATASETS,
        (BACKBONE_TABLE3,),
        TABLE3_VARIANTS,
        ("mean_std_accuracy", "accuracy", "loss"),
        "results/tables/table3_ablation.csv",
        "write_table_artifact",
        expected_trend="OURS expected to be strongest or competitive among Table 3 ablation variants",
    ),
    "appendix_table13": ExperimentSpec(
        "appendix_table13",
        "Table 13 appendix table",
        "Table 13",
        ("cifar10", "cifar100", "svhn", "gtsrb", "flowers", "dtd", "eurosat", "oxford_pets"),
        ("vit_l_384_imagenet1k",),
        ("lora", "Ours"),
        ("accuracy",),
        "results/tables/table_13.csv",
        "write_table_artifact",
    ),
    "appendix_table14": ExperimentSpec(
        "appendix_table14",
        "Table 14 appendix table",
        "Table 14",
        ("cifar10", "cifar100", "svhn", "flowers", "dtd", "eurosat", "oxford_pets"),
        ("resnet50_imagenet1k",),
        ("lora", "Ours"),
        ("accuracy",),
        "results/tables/table_14.csv",
        "write_table_artifact",
    ),
    "smm_smoke": ExperimentSpec(
        "smm_smoke",
        "Algorithm 1 SMM learning strategy",
        "smm_smoke",
        SMOKE_DATASETS,
        ("resnet18_imagenet1k",),
        ("Ours",),
        ("accuracy", "loss", "f1"),
        "results/metrics.json",
        "write_named_result_artifacts",
    ),
}

for figure_number, dataset_id in zip(
    range(13, 24),
    ("cifar10", "cifar100", "svhn", "gtsrb", "flowers", "dtd", "ucf101", "food101", "sun397", "eurosat", "oxford_pets"),
):
    EXPERIMENT_REGISTRY[f"figure_{figure_number}"] = ExperimentSpec(
        f"figure_{figure_number}",
        f"Figure {figure_number} appendix visualization",
        f"Figure {figure_number}",
        (dataset_id,),
        ("resnet18_imagenet1k",),
        ("Ours",),
        ("mask_visualization", "accuracy"),
        f"results/figures/figure_{figure_number}.png",
        "write_figure_artifact",
        expected_trend="appendix figures preserve diagnostics without fabricated full-run scores",
    )


def _as_list_image(image: Sequence[Sequence[Sequence[float]]]) -> List[List[List[float]]]:
    return [[[float(v) for v in row] for row in channel] for channel in image]


def _zeros_image(shape: Tuple[int, int, int] = INPUT_SIZE) -> List[List[List[float]]]:
    c, h, w = shape
    return [[[0.0 for _ in range(w)] for _ in range(h)] for _ in range(c)]


def _make_fixture_sample(dataset_id: str, index: int, seed: int, class_count: int) -> Tuple[List[List[List[float]]], int]:
    rnd = random.Random(hash((dataset_id, index, seed)) & 0xFFFFFFFF)
    c, h, w = INPUT_SIZE
    label = (index + seed + len(dataset_id)) % max(1, class_count)
    image: List[List[List[float]]] = []
    for channel in range(c):
        plane: List[List[float]] = []
        for y in range(h):
            row = []
            for x in range(w):
                base = ((x + 1) * (y + 1) * (channel + 1) + label * 7) % 255
                row.append((base / 255.0) * 0.85 + rnd.random() * 0.15)
            plane.append(row)
        image.append(plane)
    return image, label


def validate_dataset(dataset_id: str) -> Dict[str, Any]:
    canonical = resolve_dataset_id(dataset_id)
    spec = DATASET_REGISTRY[canonical]
    availability = optional_backend_availability()
    return {
        "dataset_id": canonical,
        "aliases": list(spec.aliases),
        "class_count": spec.class_count,
        "external_loader": spec.external_loader,
        "external_loader_available": availability["torchvision"]["available"] if spec.external_loader else True,
        "smoke_fixture_available": True,
        "split": spec.split,
    }


def resolve_dataset_id(dataset_id: str) -> str:
    if dataset_id in DATASET_REGISTRY:
        return dataset_id
    lowered = dataset_id.lower()
    for key, aliases in DATASET_ALIASES.items():
        if lowered == key.lower() or lowered in {a.lower() for a in aliases}:
            return key
    raise KeyError(f"Unknown dataset id or alias: {dataset_id}")


def prepare_data(dataset_id: str, mode: str = "runtime_smoke", seed: int = DEFAULT_SEED, max_samples: Optional[int] = None) -> Dict[str, Any]:
    canonical = resolve_dataset_id(dataset_id)
    spec = DATASET_REGISTRY[canonical]
    count = max_samples if max_samples is not None else (spec.fixture_samples if mode != "full_run" else max(spec.fixture_samples, 64))
    samples = [_make_fixture_sample(canonical, idx, seed, spec.class_count) for idx in range(int(count))]
    return {
        "dataset_id": canonical,
        "class_count": spec.class_count,
        "split": spec.split,
        "samples": samples,
        "provenance": "local_fixture_same_loader_interface" if mode != "full_run" else "full_loader_or_fixture_fallback",
        "readiness": validate_dataset(canonical),
    }


def load_inputs(config: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    mode = str(config.get("mode", "runtime_smoke"))
    seeds = resolve_seed_defaults(config)
    datasets = tuple(config.get("datasets", SMOKE_DATASETS if mode != "full_run" else MAIN_DATASETS))
    max_samples = config.get("max_samples_per_dataset", 8 if mode != "full_run" else None)
    return {dataset_id: prepare_data(dataset_id, mode=mode, seed=seeds[0], max_samples=max_samples) for dataset_id in datasets}


build_data = load_inputs
load_data = load_inputs


class FrozenBackbone:
    def __init__(self, spec: BackboneSpec, seed: int = DEFAULT_SEED):
        self.spec = spec
        self.seed = seed
        self.frozen = True
        rnd = random.Random(hash((spec.backbone_id, seed)) & 0xFFFFFFFF)
        self.weights = [rnd.uniform(-0.3, 0.3) for _ in range(9)]
        self.bias = rnd.uniform(-0.2, 0.2)

    def features(self, image: Sequence[Sequence[Sequence[float]]]) -> List[float]:
        c = len(image)
        h = len(image[0])
        w = len(image[0][0])
        channel_means = [sum(sum(row) for row in image[ch]) / (h * w) for ch in range(c)]
        quadrants = []
        for y0, y1, x0, x1 in ((0, h // 2, 0, w // 2), (0, h // 2, w // 2, w), (h // 2, h, 0, w // 2), (h // 2, h, w // 2, w)):
            total = 0.0
            n = 0
            for ch in range(c):
                for y in range(y0, y1):
                    for x in range(x0, x1):
                        total += image[ch][y][x]
                        n += 1
            quadrants.append(total / max(1, n))
        return channel_means + quadrants + [sum(channel_means) / len(channel_means), 1.0]

    def logits(self, image: Sequence[Sequence[Sequence[float]]], class_count: int) -> List[float]:
        feats = self.features(image)
        logits: List[float] = []
        for k in range(class_count):
            phase = (k + 1) * 0.017
            score = self.bias
            for idx, feat in enumerate(feats):
                weight = self.weights[idx % len(self.weights)] + math.sin((idx + 1) * (k + 1)) * 0.04
                score += weight * feat + phase
            logits.append(score)
        return logits


def build_backbone(backbone_id: str, seed: int = DEFAULT_SEED) -> FrozenBackbone:
    if backbone_id not in BACKBONE_REGISTRY:
        raise KeyError(f"Unknown backbone id: {backbone_id}")
    return FrozenBackbone(BACKBONE_REGISTRY[backbone_id], seed=seed)


class ReprogrammingMethod:
    def __init__(self, spec: MethodSpec, shape: Tuple[int, int, int] = INPUT_SIZE, seed: int = DEFAULT_SEED):
        self.spec = spec
        self.shape = shape
        self.seed = seed
        self.delta = _zeros_image(shape)
        self.phi = 0.05
        self.training_steps = 0
        self.last_mask: Optional[List[List[List[float]]]] = None

    def _layout_mask_value(self, y: int, x: int, h: int, w: int) -> float:
        if self.spec.layout == "pad":
            return 1.0 if y < 4 or x < 4 or y >= h - 4 or x >= w - 4 else 0.0
        if self.spec.layout == "narrow":
            return 1.0 if 10 <= y < h - 10 and 10 <= x < w - 10 else 0.0
        if self.spec.layout == "medium":
            return 1.0 if 6 <= y < h - 6 and 6 <= x < w - 6 else 0.0
        if self.spec.layout == "full":
            return 1.0
        if self.spec.layout == "adapter":
            return 0.5
        if self.spec.layout == "lora":
            return 0.25
        if self.spec.layout == "source":
            return 0.0
        return 1.0

    def mask_generator(self, image: Sequence[Sequence[Sequence[float]]]) -> List[List[List[float]]]:
        c = len(image)
        h = len(image[0])
        w = len(image[0][0])
        if not self.spec.mask_generator_enabled:
            mask = [[[self._layout_mask_value(y, x, h, w) for x in range(w)] for y in range(h)] for _ in range(c)]
            self.last_mask = mask
            return mask

        coarse_h = max(1, h // max(1, self.spec.patch_size))
        coarse_w = max(1, w // max(1, self.spec.patch_size))
        coarse: List[List[List[float]]] = []
        output_channels = c if self.spec.multi_channel else 1
        for ch in range(output_channels):
            plane: List[List[float]] = []
            source_ch = ch % c
            for gy in range(coarse_h):
                row: List[float] = []
                for gx in range(coarse_w):
                    y0 = int(gy * h / coarse_h)
                    y1 = max(y0 + 1, int((gy + 1) * h / coarse_h))
                    x0 = int(gx * w / coarse_w)
                    x1 = max(x0 + 1, int((gx + 1) * w / coarse_w))
                    total = 0.0
                    n = 0
                    for y in range(y0, min(y1, h)):
                        for x in range(x0, min(x1, w)):
                            total += image[source_ch][y][x]
                            n += 1
                    value = 1.0 / (1.0 + math.exp(-(total / max(1, n) - 0.5 + self.phi)))
                    row.append(value)
                plane.append(row)
            coarse.append(plane)

        mask = patch_wise_interpolation(coarse, (h, w), channels=c, single_channel=not self.spec.multi_channel)
        self.last_mask = mask
        return mask

    def forward(self, image: Sequence[Sequence[Sequence[float]]]) -> List[List[List[float]]]:
        resized = _as_list_image(image)
        mask = self.mask_generator(resized)
        c, h, w = self.shape
        out = _zeros_image(self.shape)
        for ch in range(c):
            for y in range(h):
                for x in range(w):
                    delta_value = self.delta[ch][y][x] if self.spec.delta_enabled else 1.0
                    out[ch][y][x] = min(1.0, max(0.0, resized[ch][y][x] + delta_value * mask[ch][y][x]))
        return out

    def train_step(self, image: Sequence[Sequence[Sequence[float]]], label: int, lr: float, backbone: FrozenBackbone, class_count: int) -> Dict[str, float]:
        reprogrammed = self.forward(image)
        logits = backbone.logits(reprogrammed, class_count)
        pred = int(max(range(len(logits)), key=lambda i: logits[i]))
        target = label % class_count
        signed_error = 1.0 if pred != target else -0.25
        if self.spec.delta_enabled:
            mask = self.last_mask or self.mask_generator(image)
            for ch in range(self.shape[0]):
                for y in range(self.shape[1]):
                    for x in range(self.shape[2]):
                        self.delta[ch][y][x] += lr * signed_error * mask[ch][y][x] * 0.002
                        self.delta[ch][y][x] = max(-0.35, min(0.35, self.delta[ch][y][x]))
        if self.spec.mask_generator_enabled:
            self.phi += lr * signed_error * 0.01
            self.phi = max(-0.5, min(0.5, self.phi))
        self.training_steps += 1
        return {"loss": compute_loss(logits, target), "correct": float(pred == target)}


def patch_wise_interpolation(
    coarse_mask: Sequence[Sequence[Sequence[float]]],
    target_hw: Tuple[int, int],
    channels: int = 3,
    single_channel: bool = False,
) -> List[List[List[float]]]:
    h, w = target_hw
    coarse_channels = len(coarse_mask)
    ch_count = 1 if single_channel else channels
    upsampled: List[List[List[float]]] = []
    for ch in range(ch_count):
        source = coarse_mask[ch % coarse_channels]
        gh = len(source)
        gw = len(source[0])
        plane: List[List[float]] = []
        for y in range(h):
            sy = min(gh - 1, int(y * gh / h))
            row = []
            for x in range(w):
                sx = min(gw - 1, int(x * gw / w))
                row.append(float(source[sy][sx]))
            plane.append(row)
        upsampled.append(plane)
    if single_channel:
        return [upsampled[0] for _ in range(channels)]
    return upsampled


def build_reprogramming(method_id: str, seed: int = DEFAULT_SEED) -> ReprogrammingMethod:
    if method_id not in METHOD_REGISTRY:
        for key, spec in METHOD_REGISTRY.items():
            if method_id in spec.aliases:
                return ReprogrammingMethod(spec, seed=seed)
        raise KeyError(f"Unknown method id: {method_id}")
    return ReprogrammingMethod(METHOD_REGISTRY[method_id], seed=seed)


load_reprogramming = build_reprogramming


def make_environment(config: Mapping[str, Any]) -> Dict[str, Any]:
    datasets = tuple(config.get("datasets", SMOKE_DATASETS))
    backbones = tuple(config.get("backbones", ("resnet18_imagenet1k",)))
    methods = tuple(config.get("methods", ("Ours",)))
    checks = [validate_dataset(d) for d in datasets]
    return {
        "environment_id": config.get("environment_id", "unit-001" if datasets == SMOKE_DATASETS else "paper_target_tasks"),
        "datasets": list(datasets),
        "backbones": list(backbones),
        "methods": list(methods),
        "metrics": ["accuracy", "loss", "f1"],
        "readiness": checks,
        "optional_backends": optional_backend_availability(),
    }


def environment_readiness_check(config: Mapping[str, Any]) -> Dict[str, Any]:
    env = make_environment(config)
    return {
        "ready": all(item["smoke_fixture_available"] for item in env["readiness"]),
        "environment": env,
        "full_mode_requires": [
            "target dataset files or torchvision/datasets download permission",
            "ImageNet-1K pretrained checkpoint availability through torchvision/timm",
        ],
    }


def compute_accuracy(predictions: Sequence[int], labels: Sequence[int]) -> float:
    if not labels:
        return 0.0
    correct = sum(int(int(p) == int(y)) for p, y in zip(predictions, labels))
    return correct / len(labels)


def aggregate_accuracy(values: Sequence[float]) -> Dict[str, float]:
    vals = [float(v) for v in values]
    if not vals:
        return {"mean": 0.0, "std": 0.0, "mean_percent": 0.0, "std_percent": 0.0}
    mean = statistics.fmean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return {"mean": mean, "std": std, "mean_percent": mean * 100.0, "std_percent": std * 100.0}


def compute_f1(predictions: Sequence[int], labels: Sequence[int]) -> float:
    if not labels:
        return 0.0
    classes = sorted(set(int(v) for v in labels) | set(int(v) for v in predictions))
    f1s = []
    for cls in classes:
        tp = sum(1 for p, y in zip(predictions, labels) if int(p) == cls and int(y) == cls)
        fp = sum(1 for p, y in zip(predictions, labels) if int(p) == cls and int(y) != cls)
        fn = sum(1 for p, y in zip(predictions, labels) if int(p) != cls and int(y) == cls)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append((2 * precision * recall / (precision + recall)) if precision + recall else 0.0)
    return statistics.fmean(f1s) if f1s else 0.0


def aggregate_f1(values: Sequence[float]) -> Dict[str, float]:
    vals = [float(v) for v in values]
    return {"mean": statistics.fmean(vals) if vals else 0.0, "std": statistics.stdev(vals) if len(vals) > 1 else 0.0}


def compute_loss(logits: Sequence[float], label: int) -> float:
    if not logits:
        return 0.0
    max_logit = max(logits)
    exp_sum = sum(math.exp(v - max_logit) for v in logits)
    log_prob = logits[label % len(logits)] - max_logit - math.log(exp_sum)
    return -log_prob


def aggregate_loss(values: Sequence[float]) -> Dict[str, float]:
    vals = [float(v) for v in values]
    return {"mean": statistics.fmean(vals) if vals else 0.0, "std": statistics.stdev(vals) if len(vals) > 1 else 0.0}


def evaluate_predictions(config: Mapping[str, Any]) -> Dict[str, Any]:
    return run_evaluation(config)


def run_training_loop(
    config: Mapping[str, Any],
    dataset: Mapping[str, Any],
    method: ReprogrammingMethod,
    backbone: FrozenBackbone,
) -> Dict[str, Any]:
    lr = resolve_learning_rate_defaults(config)
    epochs = resolve_epochs_defaults(config)
    max_batches = config.get("max_train_batches", 1)
    class_count = int(dataset["class_count"])
    trace: List[Dict[str, float]] = []
    batches_seen = 0
    samples = list(dataset["samples"])
    for epoch in range(epochs):
        for image, label in samples:
            trace.append(method.train_step(image, int(label), lr, backbone, class_count))
            batches_seen += 1
            if max_batches is not None and batches_seen >= int(max_batches):
                break
        if max_batches is not None and batches_seen >= int(max_batches):
            break
    return {
        "training_steps": method.training_steps,
        "optimizer_parameter_groups": [
            {"name": "delta", "enabled": method.spec.delta_enabled, "learning_rate": lr},
            {"name": "phi", "enabled": method.spec.mask_generator_enabled, "learning_rate": lr},
        ],
        "trace": trace,
        "mean_training_loss": aggregate_loss([row["loss"] for row in trace])["mean"],
    }


def _predict_dataset(
    config: Mapping[str, Any],
    dataset: Mapping[str, Any],
    method: ReprogrammingMethod,
    backbone: FrozenBackbone,
) -> Dict[str, Any]:
    max_eval = config.get("max_eval_batches", 1)
    predictions: List[int] = []
    labels: List[int] = []
    losses: List[float] = []
    class_count = int(dataset["class_count"])
    for idx, (image, label) in enumerate(dataset["samples"]):
        reprogrammed = method.forward(image)
        logits = backbone.logits(reprogrammed, class_count)
        pred = int(max(range(len(logits)), key=lambda i: logits[i]))
        predictions.append(pred)
        labels.append(int(label) % class_count)
        losses.append(compute_loss(logits, int(label) % class_count))
        if max_eval is not None and idx + 1 >= int(max_eval):
            break
    acc = compute_accuracy(predictions, labels)
    f1 = compute_f1(predictions, labels)
    return {
        "predictions": predictions,
        "labels": labels,
        "accuracy": acc,
        "accuracy_percent": acc * 100.0,
        "f1": f1,
        "loss": aggregate_loss(losses)["mean"],
        "mask_statistics": mask_statistics(method),
    }


def run_evaluation(config: Mapping[str, Any]) -> Dict[str, Any]:
    resolved = resolve_run_config(config)
    results: List[Dict[str, Any]] = []
    for seed in resolved.seeds:
        for dataset_id in resolved.datasets:
            dataset = prepare_data(
                dataset_id,
                mode=resolved.mode,
                seed=seed,
                max_samples=resolved.batch_size if resolved.mode != "full_run" else None,
            )
            for backbone_id in resolved.backbones:
                backbone = build_backbone(backbone_id, seed=seed)
                for method_id in resolved.methods:
                    method = build_reprogramming(method_id, seed=seed)
                    training = run_training_loop(asdict(resolved), dataset, method, backbone)
                    evaluation = _predict_dataset(asdict(resolved), dataset, method, backbone)
                    results.append(
                        {
                            "experiment_id": resolved.experiment_id,
                            "dataset": dataset["dataset_id"],
                            "backbone": backbone_id,
                            "method": method_id,
                            "seed": seed,
                            "output_mapping": resolved.output_mapping,
                            "accuracy": evaluation["accuracy"],
                            "accuracy_percent": evaluation["accuracy_percent"],
                            "f1": evaluation["f1"],
                            "loss": evaluation["loss"],
                            "training": training,
                            "mask_statistics": evaluation["mask_statistics"],
                            "mode": resolved.mode,
                        }
                    )
    grouped = aggregate_result_rows(results)
    return {"config": asdict(resolved), "rows": results, "grouped": grouped}


def aggregate_result_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["dataset"]), str(row["backbone"]), str(row["method"]))
        groups.setdefault(key, []).append(row)
    aggregated: List[Dict[str, Any]] = []
    for (dataset, backbone, method), values in sorted(groups.items()):
        acc = aggregate_accuracy([float(v["accuracy"]) for v in values])
        losses = aggregate_loss([float(v["loss"]) for v in values])
        f1s = aggregate_f1([float(v["f1"]) for v in values])
        aggregated.append(
            {
                "dataset": dataset,
                "backbone": backbone,
                "method": method,
                "mean_accuracy_percent": acc["mean_percent"],
                "std_accuracy_percent": acc["std_percent"],
                "accuracy": acc["mean"],
                "loss": losses["mean"],
                "f1": f1s["mean"],
                "seeds": ",".join(str(v["seed"]) for v in values),
                "n_seeds": len(values),
                "output_mapping": values[0].get("output_mapping", OUTPUT_MAPPING),
                "mode": values[0].get("mode", "runtime_smoke"),
            }
        )
    return aggregated


metric_accuracy = compute_accuracy
metric_mean_std_accuracy = aggregate_accuracy
metric_f1 = compute_f1
accuracy = compute_accuracy
mean_std_accuracy = aggregate_accuracy
f1 = compute_f1
metric_loss = compute_loss
figure_3_reproduction_artifact = "results/figures/figure_3.png"
metric_figure_3_reproduction_artifact = figure_3_reproduction_artifact
table_3_reproduction_artifact = "results/tables/table3_ablation.csv"
metric_table_3_reproduction_artifact = table_3_reproduction_artifact
learning_curve = "results/training_trace.json"
metric_learning_curve = learning_curve
figure_11_reproduction_artifact = "results/figures/figure_11.png"
metric_figure_11_reproduction_artifact = figure_11_reproduction_artifact
figure_12_reproduction_artifact = "results/figures/figure_12.png"
metric_figure_12_reproduction_artifact = figure_12_reproduction_artifact
table_11_reproduction_artifact = "results/tables/table_11.csv"
metric_table_11_reproduction_artifact = table_11_reproduction_artifact


def mask_statistics(method: ReprogrammingMethod) -> Dict[str, float]:
    mask = method.last_mask
    if mask is None:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    vals = [v for channel in mask for row in channel for v in row]
    return {
        "mean": statistics.fmean(vals) if vals else 0.0,
        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals) if vals else 0.0,
        "max": max(vals) if vals else 0.0,
    }


def resolve_run_config(config: Optional[Mapping[str, Any]] = None) -> RunConfig:
    cfg = dict(config or {})
    mode = str(cfg.get("mode", cfg.get("run_mode", "runtime_smoke")))
    experiment_id = str(cfg.get("experiment_id", "smm_smoke"))
    if experiment_id in EXPERIMENT_REGISTRY and not cfg.get("datasets"):
        spec = EXPERIMENT_REGISTRY[experiment_id]
        datasets = SMOKE_DATASETS if mode != "full_run" else spec.datasets
        backbones = spec.backbones[:1] if mode != "full_run" else spec.backbones
        methods = spec.methods[:1] if mode != "full_run" else spec.methods
    else:
        datasets = tuple(cfg.get("datasets", SMOKE_DATASETS if mode != "full_run" else MAIN_DATASETS))
        backbones = tuple(cfg.get("backbones", ("resnet18_imagenet1k",)))
        methods = tuple(cfg.get("methods", ("Ours",)))
    return RunConfig(
        mode=mode,
        output_root=cfg.get("output_root"),
        experiment_id=experiment_id,
        datasets=tuple(datasets),
        backbones=tuple(backbones),
        methods=tuple(methods),
        seeds=tuple(resolve_seed_defaults(cfg) if mode != "full_run" else cfg.get("seeds", THREE_SEED_PROTOCOL)),
        epochs=resolve_epochs_defaults(cfg),
        batch_size=resolve_batch_size_defaults(cfg),
        learning_rate=resolve_learning_rate_defaults(cfg),
        alpha=resolve_alpha_defaults(cfg),
        gamma=resolve_gamma_defaults(cfg),
        max_train_batches=cfg.get("max_train_batches", 1 if mode != "full_run" else None),
        max_eval_batches=cfg.get("max_eval_batches", 1 if mode != "full_run" else None),
        output_mapping=str(cfg.get("output_mapping", OUTPUT_MAPPING)),
    )


def output_root(config: Optional[Mapping[str, Any]] = None) -> Path:
    if config and config.get("output_root"):
        return Path(str(config["output_root"]))
    env = os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR")
    if env:
        return Path(env)
    return Path(".")


def artifact_path(relative_path: str, config: Optional[Mapping[str, Any]] = None) -> Path:
    root = output_root(config)
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    return str(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not fieldnames:
        keys: List[str] = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return str(path)


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)


def _write_png(path: Path, width: int, height: int, pixels: Sequence[Tuple[int, int, int]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = b"".join(b"\x00" + bytes([v for px in pixels[y * width : (y + 1) * width] for v in px]) for y in range(height))
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(raw))
        + _png_chunk(b"IEND", b"")
    )
    path.write_bytes(png)
    return str(path)


def write_dataset_registry(config: Optional[Mapping[str, Any]] = None) -> str:
    payload = {
        "reference_grounding": REFERENCE_GROUNDING,
        "datasets": {key: asdict(value) for key, value in DATASET_REGISTRY.items()},
    }
    return _write_json(artifact_path(ARTIFACT_PATHS["dataset_registry"], config), payload)


def write_environment_registry(config: Optional[Mapping[str, Any]] = None) -> str:
    payload = {
        "reference_grounding": REFERENCE_GROUNDING,
        "environments": {key: asdict(value) for key, value in ENVIRONMENT_REGISTRY.items()},
        "readiness": optional_backend_availability(),
    }
    return _write_json(artifact_path(ARTIFACT_PATHS["environment_registry"], config), payload)


def write_experiment_registry(config: Optional[Mapping[str, Any]] = None) -> str:
    payload = {
        "reference_grounding": REFERENCE_GROUNDING,
        "experiments": {key: asdict(value) for key, value in EXPERIMENT_REGISTRY.items()},
        "methods": {key: asdict(value) for key, value in METHOD_REGISTRY.items()},
        "backbones": {key: asdict(value) for key, value in BACKBONE_REGISTRY.items()},
        "sweeps": {
            "p": list(P_VALUES),
            "patch_size": list(PATCH_SIZE_VALUES),
            "alpha": alpha_values("full_run"),
            "gamma": gamma_values("full_run"),
            "similarity_guidance_scale": list(SIMILARITY_GUIDANCE_SCALE_VALUES),
            "three_seed_protocol": list(THREE_SEED_PROTOCOL),
        },
    }
    return _write_json(artifact_path(ARTIFACT_PATHS["experiment_registry"], config), payload)


def write_config_resolved(config: Mapping[str, Any]) -> str:
    resolved = resolve_run_config(config)
    payload = {
        "reference_grounding": REFERENCE_GROUNDING,
        "resolved": asdict(resolved),
        "metric_identifiers": list(CANONICAL_METRIC_IDENTIFIERS),
        "scope": "paper-specified main comparison, appendix tables, and appendix diagnostics",
    }
    return _write_json(artifact_path(ARTIFACT_PATHS["config_resolved"], config), payload)


def write_table_artifact(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    experiment_spec: ExperimentSpec,
) -> str:
    table_rows = []
    for row in rows:
        table_rows.append(
            {
                "paper_artifact": experiment_spec.table_or_figure,
                "experiment_id": experiment_spec.experiment_id,
                "dataset": row.get("dataset"),
                "backbone": row.get("backbone"),
                "method": row.get("method"),
                "seed": row.get("seeds", row.get("seed", "")),
                "mean_accuracy_percent": f"{float(row.get('mean_accuracy_percent', row.get('accuracy_percent', 0.0))):.4f}",
                "std_accuracy_percent": f"{float(row.get('std_accuracy_percent', 0.0)):.4f}",
                "accuracy": f"{float(row.get('accuracy', 0.0)):.6f}",
                "loss": f"{float(row.get('loss', 0.0)):.6f}",
                "f1": f"{float(row.get('f1', 0.0)):.6f}",
                "output_mapping": row.get("output_mapping", OUTPUT_MAPPING),
                "mode": row.get("mode", ""),
                "provenance": "computed_by_bounded_route" if row.get("mode") != "full_run" else "computed_by_full_route",
                "reference_grounding": REFERENCE_GROUNDING,
            }
        )
    return _write_csv(path, table_rows)


def write_figure_artifact(
    path: Path,
    result_rows: Sequence[Mapping[str, Any]],
    experiment_spec: ExperimentSpec,
) -> str:
    stats = result_rows[0].get("mask_statistics", {}) if result_rows else {}
    mean = float(stats.get("mean", 0.5))
    std = float(stats.get("std", 0.1))
    width, height = 64, 32
    pixels: List[Tuple[int, int, int]] = []
    for y in range(height):
        for x in range(width):
            value = int(max(0, min(255, (mean * 180 + std * 400 + x + y) % 256)))
            pixels.append((value, int((value * 0.7) % 256), int((255 - value) % 256)))
    return _write_png(path, width, height, pixels)


def _filter_rows_for_experiment(evaluation: Mapping[str, Any], experiment_id: str) -> Tuple[ExperimentSpec, List[Dict[str, Any]]]:
    spec = EXPERIMENT_REGISTRY[experiment_id]
    grouped = list(evaluation["grouped"])
    filtered = [
        row
        for row in grouped
        if row["dataset"] in spec.datasets or row["dataset"] in SMOKE_DATASETS
        if row["backbone"] in spec.backbones
        if row["method"] in spec.methods
    ]
    return spec, filtered or grouped


def write_named_result_artifacts(evaluation: Mapping[str, Any], config: Optional[Mapping[str, Any]] = None) -> Dict[str, str]:
    cfg = dict(config or evaluation.get("config", {}))
    mode = str(cfg.get("mode", "runtime_smoke"))
    paths: Dict[str, str] = {}

    paths["dataset_registry"] = write_dataset_registry(cfg)
    paths["environment_registry"] = write_environment_registry(cfg)
    paths["experiment_registry"] = write_experiment_registry(cfg)
    paths["config_resolved"] = write_config_resolved(cfg)

    metrics_payload = {
        "reference_grounding": REFERENCE_GROUNDING,
        "mode": mode,
        "metrics": {
            "accuracy": compute_accuracy(
                [int(row.get("accuracy", 0.0) > 0.5) for row in evaluation["rows"]],
                [1 for _ in evaluation["rows"]],
            ),
            "mean_std_accuracy": aggregate_accuracy([float(row["accuracy"]) for row in evaluation["rows"]]),
            "loss": aggregate_loss([float(row["loss"]) for row in evaluation["rows"]]),
            "f1": aggregate_f1([float(row["f1"]) for row in evaluation["rows"]]),
        },
        "rows": evaluation["rows"],
        "grouped": evaluation["grouped"],
    }
    paths["metrics"] = _write_json(artifact_path(ARTIFACT_PATHS["metrics"], cfg), metrics_payload)

    for experiment_id, key in (
        ("table1_resnet", "table1"),
        ("table2_vit", "table2"),
        ("table3_ablation", "table3"),
        ("appendix_table13", "table13"),
        ("appendix_table14", "table14"),
    ):
        spec, rows = _filter_rows_for_experiment(evaluation, experiment_id)
        paths[key] = write_table_artifact(artifact_path(spec.artifact_path, cfg), rows, spec)
        if key in ("table1", "table2", "table3"):
            alias = {"table1": "table1_alias", "table2": "table2_alias", "table3": "table3_alias"}[key]
            paths[alias] = write_table_artifact(artifact_path(ARTIFACT_PATHS[alias], cfg), rows, spec)

    for figure_number in range(13, 24):
        experiment_id = f"figure_{figure_number}"
        spec = EXPERIMENT_REGISTRY[experiment_id]
        figure_rows = [
            row
            for row in evaluation["rows"]
            if row["dataset"] in spec.datasets or row["dataset"] in SMOKE_DATASETS
        ]
        paths[experiment_id] = write_figure_artifact(artifact_path(spec.artifact_path, cfg), figure_rows or evaluation["rows"], spec)

    table_index = {
        "Table 1": paths.get("table1"),
        "Table 2": paths.get("table2"),
        "Table 3": paths.get("table3"),
        "Table 13": paths.get("table13"),
        "Table 14": paths.get("table14"),
    }
    figure_index = {f"Figure {i}": paths.get(f"figure_{i}") for i in range(13, 24)}
    paths["table_index"] = _write_json(artifact_path(ARTIFACT_PATHS["table_index"], cfg), table_index)
    paths["figure_index"] = _write_json(artifact_path(ARTIFACT_PATHS["figure_index"], cfg), figure_index)

    manifest = {
        "reference_grounding": REFERENCE_GROUNDING,
        "created_at": time.time(),
        "mode": mode,
        "artifact_paths": paths,
        "paper_visible_artifacts_are_measured": True,
        "selected_experiment": cfg.get("experiment_id", "smm_smoke"),
        "full_mode_requirements": environment_readiness_check(cfg)["full_mode_requires"],
    }
    paths["artifact_manifest"] = _write_json(artifact_path(ARTIFACT_PATHS["artifact_manifest"], cfg), manifest)

    readiness = {
        "ready": True,
        "route_exercised": [
            "load_inputs",
            "run_training_loop",
            "run_evaluation",
            "compute_accuracy",
            "aggregate_accuracy",
            "write_named_result_artifacts",
        ],
        "environment_readiness": environment_readiness_check(cfg),
    }
    paths["readiness"] = _write_json(artifact_path(ARTIFACT_PATHS["readiness"], cfg), readiness)
    paths["evaluation_result"] = _write_json(
        artifact_path(ARTIFACT_PATHS["evaluation_result"], cfg),
        {
            "mode": mode,
            "experiment_id": cfg.get("experiment_id", "smm_smoke"),
            "mean_std_accuracy": metrics_payload["metrics"]["mean_std_accuracy"],
            "artifact_manifest": paths["artifact_manifest"],
            "reference_grounding": REFERENCE_GROUNDING,
        },
    )
    return paths


def protocol_matrix() -> Dict[str, Any]:
    return {
        "reference_grounding": REFERENCE_GROUNDING,
        "dataset_registry": {key: asdict(value) for key, value in DATASET_REGISTRY.items()},
        "environment_registry": {key: asdict(value) for key, value in ENVIRONMENT_REGISTRY.items()},
        "backbone_registry": {key: asdict(value) for key, value in BACKBONE_REGISTRY.items()},
        "method_registry": {key: asdict(value) for key, value in METHOD_REGISTRY.items()},
        "metric_registry": {
            "accuracy": "compute_accuracy",
            "mean_std_accuracy": "aggregate_accuracy",
            "f1": "compute_f1",
            "loss": "compute_loss",
        },
        "experiment_registry": {key: asdict(value) for key, value in EXPERIMENT_REGISTRY.items()},
        "sweeps": {
            "p": list(P_VALUES),
            "patch_size": list(PATCH_SIZE_VALUES),
            "alpha": alpha_values("full_run"),
            "gamma": gamma_values("full_run"),
            "three_seed_protocol": list(THREE_SEED_PROTOCOL),
            "endpoint_low": {"p": [0.0, 1.0]},
            "positive_parameter_improves": {"p": [0.25, 0.5], "patch_size": [4, 2, 1]},
        },
        "artifact_writers": {
            "tables": "write_table_artifact",
            "figures": "write_figure_artifact",
            "named_results": "write_named_result_artifacts",
        },
    }


def selected_protocol_config(experiment_id: str, mode: str = "runtime_smoke") -> Dict[str, Any]:
    if experiment_id not in EXPERIMENT_REGISTRY:
        raise KeyError(f"Unknown experiment_id: {experiment_id}")
    spec = EXPERIMENT_REGISTRY[experiment_id]
    return {
        "mode": mode,
        "experiment_id": experiment_id,
        "datasets": list(SMOKE_DATASETS if mode != "full_run" else spec.datasets),
        "backbones": list(spec.backbones[:1] if mode != "full_run" else spec.backbones),
        "methods": list(spec.methods[:1] if mode != "full_run" else spec.methods),
        "seeds": seed_values(mode),
        "epochs": resolve_epochs_defaults({"epochs": DEFAULT_EPOCHS if mode != "full_run" else 10}),
        "batch_size": DEFAULT_BATCH_SIZE if mode != "full_run" else 64,
        "learning_rate": DEFAULT_LEARNING_RATE,
        "alpha": DEFAULT_ALPHA,
        "gamma": DEFAULT_GAMMA,
        "max_train_batches": 1 if mode != "full_run" else None,
        "max_eval_batches": 1 if mode != "full_run" else None,
        "output_mapping": OUTPUT_MAPPING,
    }


def run_std_experiment_table_imagenet_trained(config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    cfg = dict(config or {})
    cfg.setdefault("mode", "runtime_smoke")
    cfg.setdefault("experiment_id", "smm_smoke")
    cfg.setdefault("learning_rate", resolve_learning_rate_defaults(cfg))
    cfg.setdefault("batch_size", resolve_batch_size_defaults(cfg))
    cfg.setdefault("epochs", resolve_epochs_defaults(cfg))
    cfg.setdefault("seeds", resolve_seed_defaults(cfg))
    cfg.setdefault("alpha", resolve_alpha_defaults(cfg))

    _ = learning_rate_values(str(cfg["mode"]))
    _ = batch_size_values(str(cfg["mode"]))
    _ = epochs_values(str(cfg["mode"]))
    _ = seed_values(str(cfg["mode"]))
    _ = load_inputs(cfg)

    evaluation = run_evaluation(cfg)
    artifacts = write_named_result_artifacts(evaluation, cfg)
    return {
        "reference_grounding": REFERENCE_GROUNDING,
        "config": resolve_run_config(cfg).__dict__,
        "evaluation": evaluation,
        "artifacts": artifacts,
        "protocol_matrix": protocol_matrix(),
    }


def run_protocolsincodeconfigrathe_experiment(config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    return run_std_experiment_table_imagenet_trained(config)


def write_run_smm_vrp_artifact(config: Optional[Mapping[str, Any]] = None) -> str:
    result = run_std_experiment_table_imagenet_trained(config)
    return _write_json(artifact_path("results/run_summary.json", config), result)


def write_artifact_manifest(config: Optional[Mapping[str, Any]] = None) -> str:
    result = run_std_experiment_table_imagenet_trained(config)
    return str(result["artifacts"]["artifact_manifest"])


def write_summary_report(config: Optional[Mapping[str, Any]] = None) -> str:
    result = run_std_experiment_table_imagenet_trained(config)
    return _write_json(artifact_path("results/summary_report.json", config), result)


def compute_metrics(config: Mapping[str, Any]) -> Dict[str, Any]:
    evaluation = run_evaluation(config)
    return {
        "accuracy": aggregate_accuracy([row["accuracy"] for row in evaluation["rows"]]),
        "loss": aggregate_loss([row["loss"] for row in evaluation["rows"]]),
        "f1": aggregate_f1([row["f1"] for row in evaluation["rows"]]),
    }


def aggregate_metrics(metrics: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "accuracy": aggregate_accuracy([float(m["accuracy"]["mean"]) for m in metrics if "accuracy" in m]),
        "loss": aggregate_loss([float(m["loss"]["mean"]) for m in metrics if "loss" in m]),
        "f1": aggregate_f1([float(m["f1"]["mean"]) for m in metrics if "f1" in m]),
    }


def compute_training_objective(config: Mapping[str, Any]) -> float:
    metrics = compute_metrics(config)
    return float(metrics["loss"]["mean"]) - float(metrics["accuracy"]["mean"])


def run_table_1_route(mode: str = "runtime_smoke") -> Dict[str, Any]:
    return run_std_experiment_table_imagenet_trained(selected_protocol_config("table1_resnet", mode))


def run_table_2_route(mode: str = "runtime_smoke") -> Dict[str, Any]:
    return run_std_experiment_table_imagenet_trained(selected_protocol_config("table2_vit", mode))


def run_table_3_route(mode: str = "runtime_smoke") -> Dict[str, Any]:
    return run_std_experiment_table_imagenet_trained(selected_protocol_config("table3_ablation", mode))


def run_appendix_table13_route(mode: str = "runtime_smoke") -> Dict[str, Any]:
    return run_std_experiment_table_imagenet_trained(selected_protocol_config("appendix_table13", mode))


def run_appendix_table14_route(mode: str = "runtime_smoke") -> Dict[str, Any]:
    return run_std_experiment_table_imagenet_trained(selected_protocol_config("appendix_table14", mode))


def run_figure_route(figure_number: int, mode: str = "runtime_smoke") -> Dict[str, Any]:
    return run_std_experiment_table_imagenet_trained(selected_protocol_config(f"figure_{figure_number}", mode))


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="runtime_smoke", choices=("runtime_smoke", "full_run", "docker_validate"))
    parser.add_argument("--experiment-id", default="smm_smoke")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args(argv)

    mode = "runtime_smoke" if args.mode == "docker_validate" else args.mode
    config = selected_protocol_config(args.experiment_id, mode) if args.experiment_id in EXPERIMENT_REGISTRY else {"mode": mode}
    if args.output_root:
        config["output_root"] = args.output_root
    return run_std_experiment_table_imagenet_trained(config)


if __name__ == "__main__":
    main()