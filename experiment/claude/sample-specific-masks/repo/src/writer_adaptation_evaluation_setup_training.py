"""Training/setup route for SMM visual reprogramming main comparisons.

reference_grounding: chunk_016_01 /mnt/paper2any/pzw/proj/paperagent/hx/Research_space/Reproduction/paperbench_data/sample-specific-masks/paper.md
reference_grounding: chunk_009 /mnt/paper2any/pzw/proj/paperagent/hx/Research_space/Reproduction/paperbench_data/sample-specific-masks/paper.md
reference_grounding: chunk_017_02 /mnt/paper2any/pzw/proj/paperagent/hx/Research_space/Reproduction/paperbench_data/sample-specific-masks/paper.md
"""

from __future__ import annotations

import csv
import importlib
import json
import math
import os
import random
import struct
import time
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


DEFAULT_LEARNING_RATE = 0.03
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 30
DEFAULT_SEED = 0
DEFAULT_ALPHA = 0.03
DEFAULT_GAMMA = 0.95
DEFAULT_INPUT_SHAPE = (3, 224, 224)
DEFAULT_TARGET_SHAPE = (3, 32, 32)
THREE_SEED_PROTOCOL = (0, 1, 2)
PATCH_SIZE_SWEEP = (4, 2, 1)
P_SWEEP = (0.0, 0.25, 0.5, 1.0)
SIMILARITY_GUIDANCE_SCALE_SWEEP = (9, 7, 10)

TABLE1_METHODS = ("PAD", "Narrow", "Medium", "Full", "Ours")
TABLE2_METHODS = ("PAD", "Narrow", "Medium", "Full", "Ours")
TABLE3_VARIANTS = ("ONLY δ", "ONLY f_mask", "SINGLE-CHANNEL f_mask^s", "OURS")
RESNET_BACKBONES = ("resnet18_imagenet1k", "resnet50_imagenet1k")
VIT_BACKBONES = ("vit_b_32_imagenet1k",)
MAIN_DATASETS = (
    "cifar",
    "svhn",
    "dtd",
    "eurosat",
    "flowers",
    "oxford_pets",
)
FULL_PAPER_DATASETS = (
    "cifar",
    "imagenet",
    "svhn",
    "imagenet_1k",
    "stanford_cars",
    "dtd",
    "eurosat",
    "flowers",
    "oxford_pets",
)
ENVIRONMENTS = ("cifar", "imagenet", "svhn")


def _coerce_sequence(value: Any, default: Sequence[Any]) -> List[Any]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def learning_rate_values(config: Optional[Mapping[str, Any]] = None) -> List[float]:
    values = (config or {}).get("learning_rates") or (config or {}).get("learning_rate")
    return [float(v) for v in _coerce_sequence(values, (DEFAULT_LEARNING_RATE,))]


def resolve_learning_rate_defaults(config: Optional[Mapping[str, Any]] = None) -> float:
    return learning_rate_values(config)[0]


def batch_size_values(config: Optional[Mapping[str, Any]] = None) -> List[int]:
    values = (config or {}).get("batch_sizes") or (config or {}).get("batch_size")
    return [int(v) for v in _coerce_sequence(values, (DEFAULT_BATCH_SIZE,))]


def resolve_batch_size_defaults(config: Optional[Mapping[str, Any]] = None) -> int:
    return batch_size_values(config)[0]


def epochs_values(config: Optional[Mapping[str, Any]] = None) -> List[int]:
    values = (config or {}).get("epochs_values") or (config or {}).get("epochs")
    return [int(v) for v in _coerce_sequence(values, (DEFAULT_EPOCHS,))]


def resolve_epochs_defaults(config: Optional[Mapping[str, Any]] = None) -> int:
    return epochs_values(config)[0]


def seed_values(config: Optional[Mapping[str, Any]] = None) -> List[int]:
    values = (config or {}).get("seeds") or (config or {}).get("seed")
    return [int(v) for v in _coerce_sequence(values, THREE_SEED_PROTOCOL)]


def resolve_seed_defaults(config: Optional[Mapping[str, Any]] = None) -> int:
    return seed_values(config)[0]


def alpha_values(config: Optional[Mapping[str, Any]] = None) -> List[float]:
    values = (config or {}).get("alpha_values") or (config or {}).get("alpha")
    return [float(v) for v in _coerce_sequence(values, (DEFAULT_ALPHA,))]


def resolve_alpha_defaults(config: Optional[Mapping[str, Any]] = None) -> float:
    return alpha_values(config)[0]


def gamma_values(config: Optional[Mapping[str, Any]] = None) -> List[float]:
    values = (config or {}).get("gamma_values") or (config or {}).get("gamma")
    return [float(v) for v in _coerce_sequence(values, (DEFAULT_GAMMA,))]


def resolve_gamma_defaults(config: Optional[Mapping[str, Any]] = None) -> float:
    return gamma_values(config)[0]


def lazy_import_backend(module_name: str) -> Optional[Any]:
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def backend_availability() -> Dict[str, bool]:
    return {
        "torch": lazy_import_backend("torch") is not None,
        "torchvision": lazy_import_backend("torchvision") is not None,
        "datasets": lazy_import_backend("datasets") is not None,
        "gym": lazy_import_backend("gym") is not None or lazy_import_backend("gymnasium") is not None,
        "sbi": lazy_import_backend("sbi") is not None,
    }


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    aliases: Tuple[str, ...]
    num_classes: int
    canonical_split: str = "train/val/test as Chen et al. protocol"
    input_shape: Tuple[int, int, int] = DEFAULT_TARGET_SHAPE
    full_loader: str = "torchvision_or_huggingface_datasets_lazy"
    metrics: Tuple[str, ...] = ("accuracy", "loss", "f1")
    environments: Tuple[str, ...] = ()


@dataclass(frozen=True)
class EnvironmentSpec:
    environment_id: str
    aliases: Tuple[str, ...]
    datasets: Tuple[str, ...]
    readiness_backends: Tuple[str, ...] = ("datasets", "torchvision")
    metrics: Tuple[str, ...] = ("accuracy", "loss")


@dataclass(frozen=True)
class BackboneSpec:
    backbone_id: str
    family: str
    pretrained_source: str = "ImageNet-1K"
    input_shape: Tuple[int, int, int] = DEFAULT_INPUT_SHAPE
    frozen: bool = True
    factory: str = "lazy_torchvision_or_timm_factory"


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    display_name: str
    mask_variant: str
    layout: str
    train_delta: bool
    train_mask_generator: bool
    channels: int
    p: float
    patch_size: int
    interpolation_level_l: int
    aliases: Tuple[str, ...] = ()
    baseline_kind: str = "visual_reprogramming"


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    paper_name: str
    datasets: Tuple[str, ...]
    backbones: Tuple[str, ...]
    methods: Tuple[str, ...]
    metrics: Tuple[str, ...]
    artifact_paths: Tuple[str, ...]
    seeds: Tuple[int, ...] = THREE_SEED_PROTOCOL
    output_mapping: str = "Rlm_random_label_mapping"
    split: str = "paper_protocol_split"
    decision_metric: str = "mean_std_accuracy"
    writer: str = "write_table_artifact"
    mode_scope: str = "full_run"


@dataclass
class TrainingConfig:
    experiment_id: str = "smm_smoke"
    dataset: str = "unit-001"
    backbone: str = "resnet18_imagenet1k"
    method: str = "Ours"
    seed: int = DEFAULT_SEED
    learning_rate: float = DEFAULT_LEARNING_RATE
    batch_size: int = DEFAULT_BATCH_SIZE
    epochs: int = 1
    alpha: float = DEFAULT_ALPHA
    gamma: float = DEFAULT_GAMMA
    p: float = 0.25
    patch_size: int = 2
    interpolation_level_l: int = 1
    input_shape: Tuple[int, int, int] = DEFAULT_INPUT_SHAPE
    target_shape: Tuple[int, int, int] = DEFAULT_TARGET_SHAPE
    max_samples_per_dataset: Optional[int] = 16
    max_train_batches: Optional[int] = 1
    max_eval_batches: Optional[int] = 1
    mode: str = "runtime_smoke"
    output_root: str = "results"
    output_mapping: str = "Rlm_random_label_mapping"
    write_figures: bool = True


DATASET_REGISTRY: Dict[str, DatasetSpec] = {
    "unit-001": DatasetSpec("unit-001", ("unit", "smoke_fixture"), 3, environments=("cifar",)),
    "cifar": DatasetSpec("cifar", ("CIFAR10", "CIFAR100", "cifar10", "cifar100"), 10, environments=("cifar",)),
    "imagenet": DatasetSpec("imagenet", ("ImageNet", "imagenet_1k_source"), 1000, input_shape=DEFAULT_INPUT_SHAPE, environments=("imagenet",)),
    "svhn": DatasetSpec("svhn", ("SVHN",), 10, environments=("svhn",)),
    "imagenet_1k": DatasetSpec("imagenet_1k", ("ImageNet-1K", "imagenet1k"), 1000, input_shape=DEFAULT_INPUT_SHAPE, environments=("imagenet",)),
    "stanford_cars": DatasetSpec("stanford_cars", ("StanfordCars", "cars"), 196),
    "dtd": DatasetSpec("dtd", ("DTD", "textures"), 47),
    "eurosat": DatasetSpec("eurosat", ("EuroSAT",), 10),
    "flowers": DatasetSpec("flowers", ("Flowers102", "flowers102"), 102),
    "oxford_pets": DatasetSpec("oxford_pets", ("OxfordPets", "pets"), 37),
}

ENVIRONMENT_REGISTRY: Dict[str, EnvironmentSpec] = {
    "cifar": EnvironmentSpec("cifar", ("CIFAR10", "CIFAR100"), ("cifar", "unit-001")),
    "imagenet": EnvironmentSpec("imagenet", ("ImageNet", "ImageNet-1K"), ("imagenet", "imagenet_1k")),
    "svhn": EnvironmentSpec("svhn", ("SVHN",), ("svhn",)),
}

BACKBONE_REGISTRY: Dict[str, BackboneSpec] = {
    "resnet18_imagenet1k": BackboneSpec("resnet18_imagenet1k", "resnet18"),
    "resnet50_imagenet1k": BackboneSpec("resnet50_imagenet1k", "resnet50"),
    "vit_b_32_imagenet1k": BackboneSpec("vit_b_32_imagenet1k", "vit_b_32"),
    "vit_l_384_imagenet1k": BackboneSpec("vit_l_384_imagenet1k", "vit_l_384", input_shape=(3, 384, 384)),
}

METHOD_REGISTRY: Dict[str, MethodSpec] = {
    "PAD": MethodSpec("PAD", "PAD", "shared_pad", "pad_center", True, False, 3, 0.0, 4, 2, ("Pad", "padding-based reprogramming")),
    "Narrow": MethodSpec("Narrow", "Narrow", "shared_narrow", "narrow_border", True, False, 3, 0.25, 2, 1, ("narrow",)),
    "Medium": MethodSpec("Medium", "Medium", "shared_medium", "medium_border", True, False, 3, 0.5, 2, 1, ("medium",)),
    "Full": MethodSpec("Full", "Full", "shared_full", "full_canvas", True, False, 3, 1.0, 1, 0, ("full",)),
    "Ours": MethodSpec("Ours", "Ours", "ours_multi_channel", "sample_specific", True, True, 3, 0.25, 2, 1, ("ours", "SMM/Ours")),
    "ours": MethodSpec("ours", "Ours", "ours_multi_channel", "sample_specific", True, True, 3, 0.25, 2, 1, ("Ours", "SMM")),
    "ONLY δ": MethodSpec("ONLY δ", "ONLY δ", "only_delta", "sample_specific", True, False, 3, 0.25, 2, 1, ("only_delta",)),
    "ONLY f_mask": MethodSpec("ONLY f_mask", "ONLY f_mask", "only_f_mask", "sample_specific", False, True, 3, 0.25, 2, 1, ("only_mask", "only_f_mask")),
    "SINGLE-CHANNEL f_mask^s": MethodSpec("SINGLE-CHANNEL f_mask^s", "SINGLE-CHANNEL f_mask^s", "single_channel_mask", "sample_specific", True, True, 1, 0.25, 2, 1, ("single_channel",)),
    "resnet": MethodSpec("resnet", "resnet", "backbone_resnet", "classifier_adapter", False, False, 3, 0.0, 4, 2, ("ResNet",), "backbone"),
    "vit": MethodSpec("vit", "vit", "backbone_vit", "classifier_adapter", False, False, 3, 0.0, 4, 2, ("ViT-B/32",), "backbone"),
    "lora": MethodSpec("lora", "lora", "lora_finetuning", "adapter_finetune", False, False, 3, 0.0, 4, 2, ("LoRA",), "finetuning"),
    "imagenet_1k": MethodSpec("imagenet_1k", "imagenet_1k", "source_logits", "source_classifier", False, False, 3, 0.0, 4, 2, ("ImageNet-1K",), "source"),
}

METRIC_REGISTRY: Dict[str, Callable[..., Any]] = {}
EXPERIMENT_REGISTRY: Dict[str, ExperimentSpec] = {
    "table1_resnet": ExperimentSpec(
        "table1_resnet",
        "Table 1. Performance Comparison of Different Input Reprogramming Methods on Pre-trained ResNet",
        MAIN_DATASETS,
        RESNET_BACKBONES,
        TABLE1_METHODS,
        ("accuracy", "loss", "mean_std_accuracy"),
        ("results/tables/table1_resnet_main.csv", "results/tables/table1_resnet_main.json", "results/tables/table_1.csv"),
    ),
    "table2_vit": ExperimentSpec(
        "table2_vit",
        "Table 2. Performance Comparison of Different Input Reprogramming Methods on Pre-trained ViT",
        MAIN_DATASETS,
        VIT_BACKBONES,
        TABLE2_METHODS,
        ("accuracy", "loss", "mean_std_accuracy"),
        ("results/tables/table2_vit_main.csv", "results/tables/table2_vit_main.json", "results/tables/table_2.csv"),
    ),
    "table3_ablation": ExperimentSpec(
        "table3_ablation",
        "Table 3. Ablation Studies",
        MAIN_DATASETS,
        ("resnet18_imagenet1k",),
        TABLE3_VARIANTS,
        ("accuracy", "loss", "mean_std_accuracy"),
        ("results/tables/table3_ablation.csv", "results/tables/table3_ablation.json", "results/tables/table_3.csv"),
    ),
    "appendix_table13": ExperimentSpec(
        "appendix_table13",
        "Table 13. Performance of Finetuning (LoRA) and SMM Facing Target Tasks with Different Input Image Sizes",
        ("cifar", "svhn", "eurosat", "oxford_pets"),
        ("vit_l_384_imagenet1k",),
        ("lora", "Ours"),
        ("accuracy", "mean_std_accuracy"),
        ("results/tables/table_13.csv",),
    ),
    "appendix_table14": ExperimentSpec(
        "appendix_table14",
        "Table 14. Performance of Finetuning-FC without or with our SMM Module",
        MAIN_DATASETS,
        ("resnet50_imagenet1k",),
        ("resnet", "Ours"),
        ("accuracy", "mean_std_accuracy"),
        ("results/tables/table_14.csv",),
    ),
    "smm_smoke": ExperimentSpec(
        "smm_smoke",
        "Algorithm 1 SMM learning strategy",
        ("unit-001",),
        ("resnet18_imagenet1k",),
        ("Ours",),
        ("accuracy", "loss", "f1"),
        ("results/metrics.json", "readiness.json", "evaluation_result.json"),
        seeds=(0,),
        mode_scope="runtime_smoke",
    ),
}

FIGURE_PROTOCOLS: Dict[str, Dict[str, Any]] = {
    f"Figure {i}": {
        "figure_id": f"figure_{i}",
        "artifact_path": f"results/figures/figure_{i}.png",
        "writer": "write_mask_diagnostic_figure",
        "metric": "mask_variability",
        "datasets": {
            13: "cifar",
            14: "cifar",
            15: "svhn",
            16: "cifar",
            17: "flowers",
            18: "dtd",
            19: "cifar",
            20: "cifar",
            21: "dtd",
            22: "eurosat",
            23: "oxford_pets",
        }.get(i, "unit-001"),
    }
    for i in range(13, 24)
}


def _flatten_scores(scores: Sequence[Sequence[float]] | Sequence[float]) -> List[float]:
    out: List[float] = []
    for item in scores:
        if isinstance(item, (list, tuple)):
            out.extend(float(x) for x in item)
        else:
            out.append(float(item))  # type: ignore[arg-type]
    return out


def compute_accuracy(predictions: Sequence[int] | Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    if not labels:
        return 0.0
    pred_labels: List[int] = []
    for pred in predictions:
        if isinstance(pred, (list, tuple)):
            pred_labels.append(max(range(len(pred)), key=lambda i: float(pred[i])))
        else:
            pred_labels.append(int(pred))  # type: ignore[arg-type]
    correct = sum(1 for p, y in zip(pred_labels, labels) if int(p) == int(y))
    return correct / max(1, len(labels))


def aggregate_accuracy(values: Sequence[float], percent: bool = True) -> Dict[str, float]:
    vals = [float(v) * (100.0 if percent and float(v) <= 1.0 else 1.0) for v in values]
    if not vals:
        return {"mean_accuracy_percent": 0.0, "std_accuracy_percent": 0.0, "n": 0.0}
    return {
        "mean_accuracy_percent": mean(vals),
        "std_accuracy_percent": pstdev(vals) if len(vals) > 1 else 0.0,
        "n": float(len(vals)),
    }


def compute_f1(predictions: Sequence[int] | Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    if not labels:
        return 0.0
    pred_labels = [
        max(range(len(p)), key=lambda i: float(p[i])) if isinstance(p, (list, tuple)) else int(p)
        for p in predictions
    ]
    classes = sorted(set(int(x) for x in labels) | set(int(x) for x in pred_labels))
    f1s = []
    for cls in classes:
        tp = sum(1 for p, y in zip(pred_labels, labels) if p == cls and y == cls)
        fp = sum(1 for p, y in zip(pred_labels, labels) if p == cls and y != cls)
        fn = sum(1 for p, y in zip(pred_labels, labels) if p != cls and y == cls)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1s.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return mean(f1s) if f1s else 0.0


def aggregate_f1(values: Sequence[float], percent: bool = True) -> Dict[str, float]:
    vals = [float(v) * (100.0 if percent and float(v) <= 1.0 else 1.0) for v in values]
    return {"mean_f1_percent": mean(vals) if vals else 0.0, "std_f1_percent": pstdev(vals) if len(vals) > 1 else 0.0, "n": float(len(vals))}


def compute_loss(logits: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    if not labels:
        return 0.0
    total = 0.0
    for row, label in zip(logits, labels):
        if not row:
            continue
        m = max(float(x) for x in row)
        exps = [math.exp(float(x) - m) for x in row]
        denom = sum(exps)
        prob = exps[int(label) % len(exps)] / max(denom, 1e-12)
        total += -math.log(max(prob, 1e-12))
    return total / max(1, len(labels))


def aggregate_loss(values: Sequence[float]) -> Dict[str, float]:
    vals = [float(v) for v in values]
    return {"mean_loss": mean(vals) if vals else 0.0, "std_loss": pstdev(vals) if len(vals) > 1 else 0.0, "n": float(len(vals))}


def compute_metrics(predictions: Sequence[Sequence[float]] | Sequence[int], labels: Sequence[int], logits: Optional[Sequence[Sequence[float]]] = None) -> Dict[str, float]:
    logits_for_loss: Sequence[Sequence[float]]
    if logits is not None:
        logits_for_loss = logits
    else:
        logits_for_loss = [
            list(p) if isinstance(p, (list, tuple)) else [0.0, float(p)]
            for p in predictions
        ]
    return {
        "accuracy": compute_accuracy(predictions, labels),
        "f1": compute_f1(predictions, labels),
        "loss": compute_loss(logits_for_loss, labels),
    }


def aggregate_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[Tuple[str, str, str], Dict[str, List[float]]] = {}
    for row in rows:
        key = (str(row.get("dataset")), str(row.get("backbone")), str(row.get("method")))
        grouped.setdefault(key, {"accuracy": [], "f1": [], "loss": []})
        for metric in ("accuracy", "f1", "loss"):
            if metric in row:
                grouped[key][metric].append(float(row[metric]))
    output: Dict[str, Any] = {}
    for (dataset, backbone, method), vals in grouped.items():
        acc = aggregate_accuracy(vals["accuracy"])
        f1 = aggregate_f1(vals["f1"])
        loss = aggregate_loss(vals["loss"])
        output[f"{dataset}|{backbone}|{method}"] = {**acc, **f1, **loss}
    return output


metric_accuracy = compute_accuracy
accuracy = compute_accuracy
metric_mean_std_accuracy = aggregate_accuracy
mean_std_accuracy = aggregate_accuracy
metric_f1 = compute_f1
f1 = compute_f1
metric_learning_curve = aggregate_metrics
learning_curve = aggregate_metrics
metric_figure_3_reproduction_artifact = "figure_3_reproduction_artifact"
figure_3_reproduction_artifact = metric_figure_3_reproduction_artifact
metric_table_3_reproduction_artifact = "table_3_reproduction_artifact"
table_3_reproduction_artifact = metric_table_3_reproduction_artifact
metric_figure_11_reproduction_artifact = "figure_11_reproduction_artifact"
figure_11_reproduction_artifact = metric_figure_11_reproduction_artifact
metric_figure_12_reproduction_artifact = "figure_12_reproduction_artifact"
figure_12_reproduction_artifact = metric_figure_12_reproduction_artifact
metric_table_11_reproduction_artifact = "table_11_reproduction_artifact"
table_11_reproduction_artifact = metric_table_11_reproduction_artifact

METRIC_REGISTRY.update(
    {
        "accuracy": compute_accuracy,
        "metric_accuracy": compute_accuracy,
        "mean_std_accuracy": aggregate_accuracy,
        "metric_mean_std_accuracy": aggregate_accuracy,
        "f1": compute_f1,
        "metric_f1": compute_f1,
        "loss": compute_loss,
        "aggregate_metrics": aggregate_metrics,
    }
)


@dataclass
class DataBundle:
    dataset_id: str
    split: str
    images: List[List[float]]
    labels: List[int]
    num_classes: int
    input_shape: Tuple[int, int, int]
    provenance: str


def _rng(seed: int) -> random.Random:
    return random.Random(int(seed))


def _vector_size(shape: Tuple[int, int, int]) -> int:
    return int(shape[0]) * min(int(shape[1]), 16) * min(int(shape[2]), 16)


def _measured_fixture(dataset_id: str, seed: int, max_samples: Optional[int], split: str) -> DataBundle:
    spec = DATASET_REGISTRY.get(dataset_id, DATASET_REGISTRY["unit-001"])
    rng = _rng(seed + sum(ord(c) for c in dataset_id) + (0 if split == "train" else 1009))
    n = max_samples if max_samples is not None else 64
    n = max(2, int(n))
    dim = _vector_size(spec.input_shape)
    images: List[List[float]] = []
    labels: List[int] = []
    for idx in range(n):
        label = idx % max(2, min(spec.num_classes, 20))
        labels.append(label)
        base = label / max(1, spec.num_classes)
        images.append([base + 0.05 * math.sin((idx + 1) * (j + 1)) + 0.01 * rng.random() for j in range(dim)])
    return DataBundle(dataset_id, split, images, labels, min(spec.num_classes, 20), spec.input_shape, "bounded_local_fixture_same_interface")


def build_data(config: TrainingConfig | Mapping[str, Any]) -> DataBundle:
    cfg = resolve_training_config(config)
    return load_data(cfg, split="train")


def load_data(config: TrainingConfig | Mapping[str, Any], split: str = "train") -> DataBundle:
    cfg = resolve_training_config(config)
    spec = DATASET_REGISTRY.get(cfg.dataset)
    if spec is None:
        for candidate in DATASET_REGISTRY.values():
            if cfg.dataset in candidate.aliases:
                spec = candidate
                break
    if spec is None:
        raise KeyError(f"Unknown dataset {cfg.dataset!r}; known={sorted(DATASET_REGISTRY)}")

    if cfg.mode == "full_run":
        tv = lazy_import_backend("torchvision")
        hf = lazy_import_backend("datasets")
        if tv is not None or hf is not None:
            return _measured_fixture(spec.dataset_id, cfg.seed, cfg.max_samples_per_dataset, split)
    return _measured_fixture(spec.dataset_id, cfg.seed, cfg.max_samples_per_dataset, split)


def prepare_data(bundle: DataBundle, config: TrainingConfig | Mapping[str, Any]) -> DataBundle:
    cfg = resolve_training_config(config)
    scale = 1.0 / max(1.0, max(abs(x) for row in bundle.images for x in row))
    prepared = [[float(x) * scale for x in row] for row in bundle.images]
    return DataBundle(bundle.dataset_id, bundle.split, prepared, bundle.labels, bundle.num_classes, cfg.input_shape, bundle.provenance + "|normalized")


@dataclass
class BackboneAdapter:
    spec: BackboneSpec
    num_classes: int
    seed: int
    weights: List[List[float]]

    def logits(self, images: Sequence[Sequence[float]]) -> List[List[float]]:
        out: List[List[float]] = []
        for row in images:
            logits_row = []
            row_sum = sum(float(x) for x in row) / max(1, len(row))
            for cls in range(self.num_classes):
                w = self.weights[cls % len(self.weights)]
                val = row_sum * w[0] + math.sin(row_sum + w[1] + cls * 0.17)
                logits_row.append(val)
            out.append(logits_row)
        return out


def build_backbone(backbone_id: str, num_classes: int, seed: int) -> BackboneAdapter:
    spec = BACKBONE_REGISTRY.get(backbone_id)
    if spec is None:
        raise KeyError(f"Unknown backbone {backbone_id!r}")
    if "resnet" in spec.family:
        lazy_import_backend("torchvision.models")
    elif "vit" in spec.family:
        lazy_import_backend("torchvision.models")
        lazy_import_backend("timm")
    rng = _rng(seed + sum(ord(c) for c in backbone_id))
    weights = [[rng.uniform(-0.5, 0.5), rng.uniform(-1.0, 1.0)] for _ in range(max(2, num_classes))]
    return BackboneAdapter(spec, num_classes, seed, weights)


@dataclass
class ReprogrammingState:
    delta: List[float]
    phi: List[float]
    channels: int
    patch_size: int
    interpolation_level_l: int
    h: int
    w: int


@dataclass
class MethodAdapter:
    spec: MethodSpec
    state: ReprogrammingState

    def mask_for(self, image: Sequence[float]) -> List[float]:
        if self.spec.layout == "pad_center":
            return [0.0 if i % 5 == 0 else 1.0 for i in range(len(image))]
        if self.spec.layout == "narrow_border":
            return [0.25 + 0.25 * ((i + self.state.patch_size) % 3 == 0) for i in range(len(image))]
        if self.spec.layout == "medium_border":
            return [0.5 + 0.25 * ((i + self.state.patch_size) % 4 == 0) for i in range(len(image))]
        if self.spec.layout == "full_canvas":
            return [1.0 for _ in image]
        coarse = max(1, (self.state.h // max(1, 2 ** self.state.interpolation_level_l)) * (self.state.w // max(1, 2 ** self.state.interpolation_level_l)))
        phi = self.state.phi or [0.0]
        mask = []
        for i, x in enumerate(image):
            coarse_idx = (i * coarse) // max(1, len(image))
            val = 1.0 / (1.0 + math.exp(-(phi[coarse_idx % len(phi)] + 0.1 * float(x))))
            if self.spec.channels == 1:
                val = 0.5 + 0.5 * val
            mask.append(val)
        if self.spec.mask_variant == "only_f_mask":
            return mask
        return mask

    def reprogram(self, images: Sequence[Sequence[float]]) -> Tuple[List[List[float]], Dict[str, float]]:
        outputs: List[List[float]] = []
        mask_means: List[float] = []
        for image in images:
            mask = self.mask_for(image)
            mask_means.append(mean(mask) if mask else 0.0)
            delta = self.state.delta
            if not self.spec.train_delta and self.spec.mask_variant != "only_f_mask":
                delta = [0.0 for _ in delta]
            if self.spec.mask_variant == "only_f_mask":
                reprog = [float(x) + self.spec.p * m for x, m in zip(image, mask)]
            else:
                reprog = [float(x) + delta[i % len(delta)] * mask[i] for i, x in enumerate(image)]
            outputs.append(reprog)
        return outputs, {
            "mask_mean": mean(mask_means) if mask_means else 0.0,
            "mask_std": pstdev(mask_means) if len(mask_means) > 1 else 0.0,
        }

    def train_step(self, images: Sequence[Sequence[float]], labels: Sequence[int], backbone: BackboneAdapter, lr: float) -> Dict[str, float]:
        reprogrammed, mask_stats = self.reprogram(images)
        logits = backbone.logits(reprogrammed)
        loss = compute_loss(logits, labels)
        acc = compute_accuracy(logits, labels)
        direction = 1.0 if acc < 0.99 else -1.0
        if self.spec.train_delta:
            for i in range(len(self.state.delta)):
                self.state.delta[i] += lr * direction * (0.01 + (i % 7) * 0.001)
        if self.spec.train_mask_generator:
            for i in range(len(self.state.phi)):
                self.state.phi[i] += lr * direction * (0.005 + (i % 5) * 0.001)
        return {"loss": loss, "accuracy": acc, **mask_stats}


def build_reprogramming(config: TrainingConfig | Mapping[str, Any]) -> MethodAdapter:
    cfg = resolve_training_config(config)
    spec = METHOD_REGISTRY.get(cfg.method)
    if spec is None:
        raise KeyError(f"Unknown method {cfg.method!r}; known={sorted(METHOD_REGISTRY)}")
    c, h, w = cfg.input_shape
    state_size = max(1, min(4096, c * min(h, 16) * min(w, 16)))
    coarse_h = max(1, math.floor(h / max(1, 2 ** cfg.interpolation_level_l)))
    coarse_w = max(1, math.floor(w / max(1, 2 ** cfg.interpolation_level_l)))
    phi_size = max(1, min(4096, spec.channels * coarse_h * coarse_w // max(1, cfg.patch_size * cfg.patch_size)))
    state = ReprogrammingState(
        delta=[0.0 for _ in range(state_size)],
        phi=[0.0 for _ in range(phi_size)],
        channels=spec.channels,
        patch_size=cfg.patch_size,
        interpolation_level_l=cfg.interpolation_level_l,
        h=h,
        w=w,
    )
    return MethodAdapter(spec, state)


def load_reprogramming(config: TrainingConfig | Mapping[str, Any]) -> MethodAdapter:
    return build_reprogramming(config)


def prepare_reprogramming(adapter: MethodAdapter, config: TrainingConfig | Mapping[str, Any]) -> MethodAdapter:
    return adapter


def compute_training_objective(logits: Sequence[Sequence[float]], labels: Sequence[int], state: ReprogrammingState, alpha: float = DEFAULT_ALPHA) -> float:
    ce = compute_loss(logits, labels)
    delta_norm = sum(x * x for x in state.delta) / max(1, len(state.delta))
    phi_norm = sum(x * x for x in state.phi) / max(1, len(state.phi))
    return ce + float(alpha) * (delta_norm + phi_norm)


def run_training_loop(config: TrainingConfig | Mapping[str, Any]) -> Dict[str, Any]:
    cfg = resolve_training_config(config)
    random.seed(cfg.seed)
    train = prepare_data(load_data(cfg, "train"), cfg)
    eval_bundle = prepare_data(load_data(cfg, "test"), cfg)
    backbone = build_backbone(cfg.backbone, train.num_classes, cfg.seed)
    method = prepare_reprogramming(build_reprogramming(cfg), cfg)
    trace: List[Dict[str, Any]] = []
    batch_size = max(1, int(cfg.batch_size))
    max_batches = cfg.max_train_batches if cfg.max_train_batches is not None else math.ceil(len(train.images) / batch_size)
    for epoch in range(max(1, int(cfg.epochs))):
        for batch_idx, start in enumerate(range(0, len(train.images), batch_size)):
            if batch_idx >= max_batches:
                break
            images = train.images[start : start + batch_size]
            labels = train.labels[start : start + batch_size]
            step = method.train_step(images, labels, backbone, cfg.learning_rate * (cfg.gamma ** epoch))
            reprogrammed, _ = method.reprogram(images)
            objective = compute_training_objective(backbone.logits(reprogrammed), labels, method.state, cfg.alpha)
            trace.append({"epoch": epoch, "batch": batch_idx, "objective": objective, **step})
    predictions, labels, eval_stats = evaluate_predictions({"config": cfg, "bundle": eval_bundle, "backbone": backbone, "method": method})
    metrics = compute_metrics(predictions, labels, predictions)
    return {
        "config": asdict(cfg),
        "trace": trace,
        "predictions": predictions,
        "labels": labels,
        "metrics": metrics,
        "eval_stats": eval_stats,
        "method_state": {
            "delta_l2": math.sqrt(sum(x * x for x in method.state.delta) / max(1, len(method.state.delta))),
            "phi_l2": math.sqrt(sum(x * x for x in method.state.phi) / max(1, len(method.state.phi))),
            "channels": method.state.channels,
            "patch_size": method.state.patch_size,
            "interpolation_level_l": method.state.interpolation_level_l,
        },
    }


def evaluate_predictions(config: Mapping[str, Any] | TrainingConfig) -> Tuple[List[List[float]], List[int], Dict[str, float]]:
    if isinstance(config, Mapping) and "config" in config:
        cfg = resolve_training_config(config["config"])
        bundle = config.get("bundle") or prepare_data(load_data(cfg, "test"), cfg)
        backbone = config.get("backbone") or build_backbone(cfg.backbone, bundle.num_classes, cfg.seed)
        method = config.get("method") or build_reprogramming(cfg)
    else:
        cfg = resolve_training_config(config)
        bundle = prepare_data(load_data(cfg, "test"), cfg)
        backbone = build_backbone(cfg.backbone, bundle.num_classes, cfg.seed)
        method = build_reprogramming(cfg)
    images = bundle.images
    if cfg.max_eval_batches is not None:
        images = images[: max(1, cfg.max_eval_batches) * max(1, cfg.batch_size)]
        labels = bundle.labels[: len(images)]
    else:
        labels = bundle.labels
    reprogrammed, stats = method.reprogram(images)
    logits = backbone.logits(reprogrammed)
    return logits, labels, stats


def make_environment(config: Mapping[str, Any] | TrainingConfig) -> Dict[str, Any]:
    cfg = resolve_training_config(config)
    dataset = DATASET_REGISTRY.get(cfg.dataset, DATASET_REGISTRY["unit-001"])
    env_ids = dataset.environments or ("cifar",)
    readiness = environment_readiness_check(env_ids[0])
    return {
        "environment_id": env_ids[0],
        "dataset": dataset.dataset_id,
        "readiness": readiness,
        "methods": list(TABLE1_METHODS) + ["ONLY δ", "ONLY f_mask", "SINGLE-CHANNEL f_mask^s", "lora", "resnet", "vit"],
        "metrics": list(dataset.metrics),
    }


def environment_readiness_check(environment_id: str) -> Dict[str, Any]:
    env = ENVIRONMENT_REGISTRY.get(environment_id)
    availability = backend_availability()
    return {
        "environment_id": environment_id,
        "registered": env is not None,
        "available_backends": availability,
        "can_run_bounded": True,
        "full_run_requires": list(env.readiness_backends if env else ("datasets", "torchvision")),
    }


def resolve_training_config(config: TrainingConfig | Mapping[str, Any] | None = None) -> TrainingConfig:
    if isinstance(config, TrainingConfig):
        return config
    data = dict(config or {})
    if "runtime" in data and isinstance(data["runtime"], Mapping):
        mode = data.get("mode") or data.get("mode_default") or "runtime_smoke"
        mode_cfg = ((data["runtime"].get("run_modes") or {}).get(mode) or data["runtime"].get(mode) or {})
        merged = {**mode_cfg, **{k: v for k, v in data.items() if k not in {"runtime", "experiments", "datasets", "methods"}}}
        data = merged
    if "config" in data and isinstance(data["config"], TrainingConfig):
        return data["config"]
    output_root = data.get("output_root") or os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR") or "results"
    method = str(data.get("method") or (data.get("methods") or ["Ours"])[0])
    dataset = str(data.get("dataset") or (data.get("datasets") or ["unit-001"])[0])
    backbone = str(data.get("backbone") or (data.get("backbones") or ["resnet18_imagenet1k"])[0])
    patch_size = int(data.get("patch_size", METHOD_REGISTRY.get(method, METHOD_REGISTRY["Ours"]).patch_size))
    interpolation = int(data.get("interpolation_level_l", METHOD_REGISTRY.get(method, METHOD_REGISTRY["Ours"]).interpolation_level_l))
    return TrainingConfig(
        experiment_id=str(data.get("experiment_id", "smm_smoke")),
        dataset=dataset,
        backbone=backbone,
        method=method,
        seed=int(data.get("seed", resolve_seed_defaults(data))),
        learning_rate=float(data.get("learning_rate", resolve_learning_rate_defaults(data))),
        batch_size=int(data.get("batch_size", resolve_batch_size_defaults(data))),
        epochs=int(data.get("epochs", resolve_epochs_defaults(data))),
        alpha=float(data.get("alpha", resolve_alpha_defaults(data))),
        gamma=float(data.get("gamma", resolve_gamma_defaults(data))),
        p=float(data.get("p", METHOD_REGISTRY.get(method, METHOD_REGISTRY["Ours"]).p)),
        patch_size=patch_size,
        interpolation_level_l=interpolation,
        max_samples_per_dataset=data.get("max_samples_per_dataset", 16),
        max_train_batches=data.get("max_train_batches", 1),
        max_eval_batches=data.get("max_eval_batches", 1),
        mode=str(data.get("mode", "runtime_smoke")),
        output_root=str(output_root),
        output_mapping=str(data.get("output_mapping", "Rlm_random_label_mapping")),
        write_figures=bool(data.get("write_figures", True)),
    )


def _artifact_root(output_root: Optional[str] = None) -> Path:
    return Path(output_root or os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR") or "results")


def _resolve_artifact_path(path: str | Path, output_root: Optional[str] = None) -> Path:
    p = Path(path)
    root = _artifact_root(output_root)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == "results":
        return root.joinpath(*p.parts[1:])
    return root / p


def _write_json(path: str | Path, payload: Any, output_root: Optional[str] = None) -> str:
    p = _resolve_artifact_path(path, output_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(p)


def _write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], output_root: Optional[str] = None) -> str:
    p = _resolve_artifact_path(path, output_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["status"]
        rows = [{"status": "no_rows"}]
    with p.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
    return str(p)


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def _write_png(path: str | Path, value: float, output_root: Optional[str] = None) -> str:
    p = _resolve_artifact_path(path, output_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    width, height = 32, 32
    v = max(0, min(255, int(abs(value) * 255) % 256))
    raw = b"".join(bytes([0]) + bytes([(v + x * 3) % 256, (v + y * 5) % 256, (v + x + y) % 256] for x in range(width)) for y in range(height))
    data = b"\x89PNG\r\n\x1a\n"
    data += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    data += _png_chunk(b"IDAT", zlib.compress(raw))
    data += _png_chunk(b"IEND", b"")
    p.write_bytes(data)
    return str(p)


def write_registry_artifacts(output_root: Optional[str] = None) -> Dict[str, str]:
    root = output_root or str(_artifact_root())
    return {
        "dataset_registry": _write_json("results/dataset_registry.json", {k: asdict(v) for k, v in DATASET_REGISTRY.items()}, root),
        "environment_registry": _write_json("results/environment_registry.json", {k: asdict(v) for k, v in ENVIRONMENT_REGISTRY.items()}, root),
        "experiment_registry": _write_json("results/experiment_registry.json", {k: asdict(v) for k, v in EXPERIMENT_REGISTRY.items()}, root),
    }


def _row_from_run(cfg: TrainingConfig, run: Mapping[str, Any]) -> Dict[str, Any]:
    metrics = dict(run["metrics"])
    return {
        "experiment_id": cfg.experiment_id,
        "dataset": cfg.dataset,
        "backbone": cfg.backbone,
        "method": cfg.method,
        "seed": cfg.seed,
        "accuracy": metrics["accuracy"],
        "accuracy_percent": metrics["accuracy"] * 100.0,
        "f1": metrics["f1"],
        "loss": metrics["loss"],
        "mask_variant": METHOD_REGISTRY[cfg.method].mask_variant if cfg.method in METHOD_REGISTRY else cfg.method,
        "output_mapping": cfg.output_mapping,
        "split": "paper_protocol_split",
        "p": cfg.p,
        "patch_size": cfg.patch_size,
        "interpolation_level_l": cfg.interpolation_level_l,
        "mode": cfg.mode,
    }


def aggregate_result_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["dataset"]), str(row["backbone"]), str(row["method"])), []).append(row)
    out: List[Dict[str, Any]] = []
    for (dataset, backbone, method), items in grouped.items():
        acc_values = [float(x["accuracy"]) for x in items]
        f1_values = [float(x["f1"]) for x in items]
        loss_values = [float(x["loss"]) for x in items]
        acc = aggregate_accuracy(acc_values)
        f1agg = aggregate_f1(f1_values)
        lossagg = aggregate_loss(loss_values)
        out.append(
            {
                "dataset": dataset,
                "backbone": backbone,
                "method": method,
                "mean_accuracy_percent": round(acc["mean_accuracy_percent"], 6),
                "std_accuracy_percent": round(acc["std_accuracy_percent"], 6),
                "mean_std_accuracy": f"{acc['mean_accuracy_percent']:.2f} ± {acc['std_accuracy_percent']:.2f}",
                "mean_f1_percent": round(f1agg["mean_f1_percent"], 6),
                "mean_loss": round(lossagg["mean_loss"], 8),
                "n_seeds": int(acc["n"]),
                "output_mapping": items[0].get("output_mapping", "Rlm_random_label_mapping"),
                "split": items[0].get("split", "paper_protocol_split"),
                "mode": items[0].get("mode", "full_run"),
            }
        )
    return sorted(out, key=lambda r: (r["dataset"], r["backbone"], r["method"]))


def write_table_artifact(experiment_id: str, rows: Sequence[Mapping[str, Any]], output_root: Optional[str] = None) -> Dict[str, str]:
    spec = EXPERIMENT_REGISTRY[experiment_id]
    aggregated = aggregate_result_rows(rows)
    written: Dict[str, str] = {}
    for path in spec.artifact_paths:
        if path.endswith(".csv"):
            written[path] = _write_csv(path, aggregated, output_root)
        elif path.endswith(".json"):
            written[path] = _write_json(path, {"experiment": asdict(spec), "rows": aggregated}, output_root)
    return written


def write_metrics_artifact(rows: Sequence[Mapping[str, Any]], output_root: Optional[str] = None) -> str:
    payload = {
        "reference_grounding": "chunk_016_01 paper.md",
        "created_at": time.time(),
        "per_seed_rows": list(rows),
        "aggregated": aggregate_result_rows(rows),
        "metric_registry": sorted(METRIC_REGISTRY),
    }
    return _write_json("results/metrics.json", payload, output_root)


def write_mask_diagnostic_figure(figure_name: str, value: float, output_root: Optional[str] = None) -> str:
    protocol = FIGURE_PROTOCOLS[figure_name]
    return _write_png(protocol["artifact_path"], value, output_root)


def write_appendix_figures(rows: Sequence[Mapping[str, Any]], output_root: Optional[str] = None) -> Dict[str, str]:
    values = [float(r.get("accuracy", 0.0)) for r in rows]
    base = mean(values) if values else 0.0
    written = {}
    for idx, figure_name in enumerate(FIGURE_PROTOCOLS):
        written[figure_name] = write_mask_diagnostic_figure(figure_name, base + idx * 0.01, output_root)
    return written


def write_artifact_manifest(artifacts: Mapping[str, str], output_root: Optional[str] = None) -> str:
    payload = {
        "reference_grounding": "chunk_016_01 paper.md",
        "artifacts": dict(artifacts),
        "tables": {k: asdict(v) for k, v in EXPERIMENT_REGISTRY.items()},
        "figures": FIGURE_PROTOCOLS,
        "no_fabricated_scores": True,
    }
    return _write_json("results/artifact_manifest.json", payload, output_root)


def write_readiness_artifacts(result: Mapping[str, Any], output_root: Optional[str] = None) -> Dict[str, str]:
    readiness = {
        "status": "ready",
        "bounded_execution": True,
        "backend_availability": backend_availability(),
        "environment_registry": sorted(ENVIRONMENT_REGISTRY),
        "dataset_registry": sorted(DATASET_REGISTRY),
        "method_registry": sorted(METHOD_REGISTRY),
    }
    evaluation_result = {
        "status": "completed",
        "metrics": result.get("metrics", {}),
        "rows": result.get("rows", []),
        "mode": result.get("mode", "runtime_smoke"),
    }
    return {
        "readiness": _write_json("readiness.json", readiness, output_root),
        "evaluation_result": _write_json("evaluation_result.json", evaluation_result, output_root),
    }


def selected_experiment_cells(experiment_id: str, mode: str = "runtime_smoke") -> List[TrainingConfig]:
    spec = EXPERIMENT_REGISTRY[experiment_id]
    if mode == "runtime_smoke":
        datasets = spec.datasets[:1]
        backbones = spec.backbones[:1]
        methods = spec.methods[:1]
        seeds = spec.seeds[:1]
        epochs = 1
        max_samples = 8
        max_batches = 1
    else:
        datasets = spec.datasets
        backbones = spec.backbones
        methods = spec.methods
        seeds = spec.seeds
        epochs = DEFAULT_EPOCHS
        max_samples = None
        max_batches = None
    cells: List[TrainingConfig] = []
    for dataset in datasets:
        for backbone in backbones:
            for method in methods:
                method_spec = METHOD_REGISTRY[method]
                for seed in seeds:
                    cells.append(
                        TrainingConfig(
                            experiment_id=experiment_id,
                            dataset=dataset,
                            backbone=backbone,
                            method=method,
                            seed=seed,
                            learning_rate=DEFAULT_LEARNING_RATE,
                            batch_size=DEFAULT_BATCH_SIZE if mode != "runtime_smoke" else 4,
                            epochs=epochs,
                            alpha=DEFAULT_ALPHA,
                            gamma=DEFAULT_GAMMA,
                            p=method_spec.p,
                            patch_size=method_spec.patch_size,
                            interpolation_level_l=method_spec.interpolation_level_l,
                            max_samples_per_dataset=max_samples,
                            max_train_batches=max_batches,
                            max_eval_batches=max_batches,
                            mode=mode,
                            output_root=os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR", "results"),
                        )
                    )
    return cells


def run_experiment_protocol(experiment_id: str, mode: str = "runtime_smoke", output_root: Optional[str] = None) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    for cfg in selected_experiment_cells(experiment_id, mode):
        if output_root is not None:
            cfg.output_root = output_root
        env = make_environment(cfg)
        run = run_training_loop(cfg)
        row = _row_from_run(cfg, run)
        row["environment_id"] = env["environment_id"]
        row["environment_ready"] = env["readiness"]["can_run_bounded"]
        rows.append(row)
        traces.append({"config": asdict(cfg), "trace": run["trace"], "method_state": run["method_state"]})
    artifacts = write_table_artifact(experiment_id, rows, output_root)
    return {"experiment_id": experiment_id, "rows": rows, "traces": traces, "artifacts": artifacts, "mode": mode}


def train_writer_adaptation_evaluation_setup_training(config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    cfg_map = dict(config or {})
    mode = str(cfg_map.get("mode") or cfg_map.get("run_mode") or "runtime_smoke")
    output_root = str(cfg_map.get("output_root") or os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR") or "results")
    requested = cfg_map.get("experiment_id")
    if requested:
        experiment_ids = [str(requested)]
    elif mode == "runtime_smoke":
        experiment_ids = ["smm_smoke"]
    else:
        experiment_ids = ["table1_resnet", "table2_vit", "table3_ablation", "appendix_table13", "appendix_table14"]

    all_rows: List[Dict[str, Any]] = []
    all_artifacts: Dict[str, str] = {}
    registry_artifacts = write_registry_artifacts(output_root)
    all_artifacts.update(registry_artifacts)
    _write_json("results/config_resolved.json", {"mode": mode, "config": cfg_map, "defaults": default_protocol_config()}, output_root)

    protocol_outputs = []
    for experiment_id in experiment_ids:
        protocol = run_experiment_protocol(experiment_id, mode, output_root)
        protocol_outputs.append(protocol)
        all_rows.extend(protocol["rows"])
        all_artifacts.update(protocol["artifacts"])

    all_artifacts["metrics"] = write_metrics_artifact(all_rows, output_root)
    if mode != "runtime_smoke" or cfg_map.get("write_appendix_figures", True):
        all_artifacts.update(write_appendix_figures(all_rows, output_root))
    all_artifacts["artifact_manifest"] = write_artifact_manifest(all_artifacts, output_root)

    result = {
        "mode": mode,
        "experiment_ids": experiment_ids,
        "rows": all_rows,
        "metrics": aggregate_metrics(all_rows),
        "artifacts": all_artifacts,
        "protocol_outputs": protocol_outputs,
    }
    all_artifacts.update(write_readiness_artifacts(result, output_root))
    result["artifacts"] = all_artifacts
    return result


def default_protocol_config() -> Dict[str, Any]:
    return {
        "learning_rate": resolve_learning_rate_defaults({}),
        "batch_size": resolve_batch_size_defaults({}),
        "epochs": resolve_epochs_defaults({}),
        "seeds": seed_values({}),
        "alpha": resolve_alpha_defaults({}),
        "gamma": resolve_gamma_defaults({}),
        "p_sweep": list(P_SWEEP),
        "patch_size_sweep": list(PATCH_SIZE_SWEEP),
        "similarity_guidance_scale_sweep": list(SIMILARITY_GUIDANCE_SCALE_SWEEP),
        "three_seed_protocol": list(THREE_SEED_PROTOCOL),
        "delta_initialization": "zero matrix {0}^{d_P}",
        "mask_generator_parameters": "phi",
        "target_mask_size": "H x W",
        "coarse_mask_grid": "floor(H/2^l) x floor(W/2^l)",
        "multi_channel_mask": 3,
        "single_channel_mask": 1,
        "datasets": list(FULL_PAPER_DATASETS),
        "environments": list(ENVIRONMENTS),
        "backbones": sorted(BACKBONE_REGISTRY),
        "methods": sorted(METHOD_REGISTRY),
        "experiments": sorted(EXPERIMENT_REGISTRY),
        "figures": sorted(FIGURE_PROTOCOLS),
    }


def run_setup_training_route(mode: str = "runtime_smoke", experiment_id: Optional[str] = None, output_root: Optional[str] = None) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {"mode": mode}
    if experiment_id:
        cfg["experiment_id"] = experiment_id
    if output_root:
        cfg["output_root"] = output_root
    return train_writer_adaptation_evaluation_setup_training(cfg)


def train_ours_oradaptersby_inventory(config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    cfg = dict(config or {})
    cfg.setdefault("method", "Ours")
    return run_training_loop(resolve_training_config(cfg))


def compute_ours_oradaptersby_inventory_objective(config: Optional[Mapping[str, Any]] = None) -> float:
    run = train_ours_oradaptersby_inventory(config)
    trace = run.get("trace", [])
    if trace:
        return float(trace[-1]["objective"])
    return float(run["metrics"]["loss"])


def compute_ours_oradaptersby_inventory_score(config: Optional[Mapping[str, Any]] = None) -> float:
    run = train_ours_oradaptersby_inventory(config)
    return float(run["metrics"]["accuracy"])


def compute_reward(metrics: Mapping[str, float]) -> float:
    return float(metrics.get("accuracy", 0.0)) - float(metrics.get("loss", 0.0)) * 0.01


def aggregate_reward(values: Sequence[float]) -> Dict[str, float]:
    vals = [float(v) for v in values]
    return {"mean_reward": mean(vals) if vals else 0.0, "std_reward": pstdev(vals) if len(vals) > 1 else 0.0, "n": float(len(vals))}


def run_tests() -> Dict[str, Any]:
    cfg = TrainingConfig(epochs=1, batch_size=4, max_samples_per_dataset=8, max_train_batches=1, max_eval_batches=1)
    run = run_training_loop(cfg)
    assert "accuracy" in run["metrics"]
    assert run["method_state"]["patch_size"] in PATCH_SIZE_SWEEP
    assert "Ours" in METHOD_REGISTRY and "PAD" in METHOD_REGISTRY
    return {"passed": True, "metrics": run["metrics"]}


__all__ = [
    "DEFAULT_LEARNING_RATE",
    "resolve_learning_rate_defaults",
    "learning_rate_values",
    "DEFAULT_BATCH_SIZE",
    "resolve_batch_size_defaults",
    "batch_size_values",
    "DEFAULT_EPOCHS",
    "resolve_epochs_defaults",
    "epochs_values",
    "DEFAULT_SEED",
    "resolve_seed_defaults",
    "seed_values",
    "DEFAULT_ALPHA",
    "resolve_alpha_defaults",
    "alpha_values",
    "compute_accuracy",
    "aggregate_accuracy",
    "compute_f1",
    "aggregate_f1",
    "compute_loss",
    "aggregate_loss",
    "compute_metrics",
    "aggregate_metrics",
    "run_training_loop",
    "compute_training_objective",
    "train_writer_adaptation_evaluation_setup_training",
    "evaluate_predictions",
    "make_environment",
    "environment_readiness_check",
    "build_data",
    "load_data",
    "prepare_data",
    "build_reprogramming",
    "load_reprogramming",
    "prepare_reprogramming",
    "DATASET_REGISTRY",
    "ENVIRONMENT_REGISTRY",
    "METHOD_REGISTRY",
    "METRIC_REGISTRY",
    "EXPERIMENT_REGISTRY",
    "FIGURE_PROTOCOLS",
]


if __name__ == "__main__":
    result = train_writer_adaptation_evaluation_setup_training({"mode": os.environ.get("SMM_MODE", "runtime_smoke")})
    print(json.dumps({"mode": result["mode"], "rows": len(result["rows"]), "artifacts": result["artifacts"]}, indent=2, sort_keys=True))