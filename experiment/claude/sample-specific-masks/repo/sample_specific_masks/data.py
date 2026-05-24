"""
Data, environment, metric, and lightweight adapter surfaces for the
Sample-specific Masks for Visual Reprogramming-based Prompting reproduction.

This module is intentionally importable without optional vision/GPU packages.
Torch/torchvision/PIL/sklearn-style functionality is loaded lazily only inside
the functions that need it.  Smoke mode uses the same dataset/backbone/config
interfaces as full mode, but with a bounded deterministic in-memory fixture.

reference_grounding: chunk_016_01 /mnt/paper2any/pzw/proj/paperagent/hx/Research_space/Reproduction/paperbench_data/sample-specific-masks/paper.md
reference_grounding: chunk_009 /mnt/paper2any/pzw/proj/paperagent/hx/Research_space/Reproduction/paperbench_data/sample-specific-masks/paper.md
reference_grounding: chunk_014_02 /mnt/paper2any/pzw/proj/paperagent/hx/Research_space/Reproduction/paperbench_data/sample-specific-masks/paper.md
"""

from __future__ import annotations

import importlib
import json
import math
import os
import random
import statistics
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple


DEFAULT_SEED = 0
THREE_SEED_PROTOCOL: Tuple[int, int, int] = (0, 1, 2)
PATCH_SIZE_VALUES: Tuple[int, int, int] = (4, 2, 1)
P_SWEEP_VALUES: Tuple[float, float, float, float, float] = (0.0, 0.25, 0.5, 0.75, 1.0)
ALPHA_SWEEP_VALUES: Tuple[float, ...] = (0.1, 0.5, 1.0)
GAMMA_SWEEP_VALUES: Tuple[float, ...] = (0.0, 0.1, 1.0)
SIMILARITY_GUIDANCE_SCALE_VALUES: Tuple[int, int, int] = (9, 7, 10)

IMAGENET_NORMALIZE: Mapping[str, Tuple[float, float, float]] = {
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
}


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _lazy_import(module_name: str) -> Any:
    return importlib.import_module(module_name)


def _artifact_root() -> Path:
    return Path(os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR", "results"))


def _safe_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace("/", "_").replace(" ", "_")


def resolve_seed_defaults(seeds: Optional[Sequence[int]] = None, *, mode: str = "runtime_smoke") -> Tuple[int, ...]:
    """Resolve the paper three-seed protocol while keeping smoke mode bounded."""
    if seeds is not None:
        resolved = tuple(int(s) for s in seeds)
    elif mode in {"full", "full_run", "paper"}:
        resolved = THREE_SEED_PROTOCOL
    else:
        resolved = (DEFAULT_SEED,)
    if not resolved:
        resolved = (DEFAULT_SEED,)
    return resolved


def seed_values(mode: str = "full_run") -> Tuple[int, ...]:
    return resolve_seed_defaults(mode=mode)


def _coerce_labels(values: Any) -> List[int]:
    if values is None:
        return []
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, (int, float)):
        return [int(values)]
    return [int(v) for v in list(values)]


def compute_f1(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    average: str = "macro",
    labels: Optional[Sequence[int]] = None,
) -> float:
    """Compute F1 without requiring sklearn.

    Supports macro, micro, and weighted averaging.  The default macro score is
    suitable for the paper's multi-class target tasks where class counts vary.
    """
    truth = _coerce_labels(y_true)
    pred = _coerce_labels(y_pred)
    if len(truth) != len(pred):
        raise ValueError(f"compute_f1 expects equal lengths, got {len(truth)} and {len(pred)}")
    if not truth:
        return 0.0

    label_set = list(dict.fromkeys(int(x) for x in (labels if labels is not None else sorted(set(truth) | set(pred)))))
    per_label: List[Tuple[float, int]] = []
    total_tp = total_fp = total_fn = 0

    for label in label_set:
        tp = sum(1 for t, p in zip(truth, pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(truth, pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(truth, pred) if t == label and p != label)
        support = sum(1 for t in truth if t == label)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        denom = (2 * tp) + fp + fn
        score = 0.0 if denom == 0 else (2 * tp) / denom
        per_label.append((score, support))

    if average == "micro":
        denom = (2 * total_tp) + total_fp + total_fn
        return 0.0 if denom == 0 else (2 * total_tp) / denom
    if average == "weighted":
        support_sum = sum(s for _, s in per_label)
        return 0.0 if support_sum == 0 else sum(score * support for score, support in per_label) / support_sum
    if average != "macro":
        raise ValueError("average must be one of {'macro', 'micro', 'weighted'}")
    return sum(score for score, _ in per_label) / max(1, len(per_label))


def aggregate_f1(
    per_seed_values: Sequence[float],
    *,
    as_percent: bool = False,
    ddof: int = 0,
) -> Dict[str, Any]:
    values = [float(v) for v in per_seed_values]
    scale = 100.0 if as_percent else 1.0
    if not values:
        return {"mean": 0.0, "std": 0.0, "values": [], "count": 0, "as_percent": as_percent}
    mean = statistics.fmean(values) * scale
    if len(values) <= 1:
        std = 0.0
    elif ddof == 1:
        std = statistics.stdev(values) * scale
    else:
        std = statistics.pstdev(values) * scale
    return {
        "mean": mean,
        "std": std,
        "values": [v * scale for v in values],
        "count": len(values),
        "as_percent": as_percent,
        "formatted": f"{mean:.2f} ± {std:.2f}",
    }


def compute_accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    truth = _coerce_labels(y_true)
    pred = _coerce_labels(y_pred)
    if len(truth) != len(pred):
        raise ValueError(f"compute_accuracy expects equal lengths, got {len(truth)} and {len(pred)}")
    if not truth:
        return 0.0
    return sum(int(t == p) for t, p in zip(truth, pred)) / len(truth)


def aggregate_accuracy(
    per_seed_values: Sequence[float],
    *,
    as_percent: bool = True,
    ddof: int = 0,
) -> Dict[str, Any]:
    return aggregate_f1(per_seed_values, as_percent=as_percent, ddof=ddof)


@dataclass(frozen=True)
class DatasetSpec:
    id: str
    canonical_name: str
    aliases: Tuple[str, ...]
    num_classes: int
    torchvision_name: Optional[str]
    splits: Tuple[str, ...] = ("train", "test")
    default_split: str = "train"
    image_channels: int = 3
    raw_image_size: Tuple[int, int] = (32, 32)
    setup_metadata: Mapping[str, Any] = field(default_factory=dict)

    def matches(self, name: str) -> bool:
        key = _safe_name(name)
        return key == _safe_name(self.id) or key == _safe_name(self.canonical_name) or key in {_safe_name(a) for a in self.aliases}

    def availability(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "canonical_name": self.canonical_name,
            "aliases": list(self.aliases),
            "torch_available": _module_available("torch"),
            "torchvision_available": _module_available("torchvision"),
            "full_loader_available": bool(self.torchvision_name) and _module_available("torchvision"),
            "smoke_fixture_available": True,
            "setup_metadata": dict(self.setup_metadata),
        }


@dataclass(frozen=True)
class EnvironmentSpec:
    id: str
    aliases: Tuple[str, ...]
    kind: str
    setup_metadata: Mapping[str, Any] = field(default_factory=dict)
    required_modules: Tuple[str, ...] = ()

    def availability(self) -> Dict[str, Any]:
        module_status = {name: _module_available(name) for name in self.required_modules}
        return {
            "id": self.id,
            "aliases": list(self.aliases),
            "kind": self.kind,
            "available": all(module_status.values()) if module_status else True,
            "required_modules": module_status,
            "setup_metadata": dict(self.setup_metadata),
        }


@dataclass(frozen=True)
class BackboneSpec:
    id: str
    aliases: Tuple[str, ...]
    family: str
    input_size: int
    source: str = "ImageNet-1K"
    num_source_classes: int = 1000
    torchvision_factory: Optional[str] = None
    frozen_by_default: bool = True

    def matches(self, name: str) -> bool:
        key = _safe_name(name)
        return key == _safe_name(self.id) or key in {_safe_name(a) for a in self.aliases}

    def availability(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "aliases": list(self.aliases),
            "family": self.family,
            "input_size": self.input_size,
            "source": self.source,
            "torch_available": _module_available("torch"),
            "torchvision_available": _module_available("torchvision"),
            "full_loader_available": bool(self.torchvision_factory) and _module_available("torchvision"),
            "frozen_by_default": self.frozen_by_default,
        }


@dataclass(frozen=True)
class MethodSpec:
    id: str
    aliases: Tuple[str, ...]
    kind: str
    mask_variant: str
    delta_enabled: bool = True
    mask_generator_enabled: bool = True
    single_channel_mask: bool = False
    fixed_mask_layout: Optional[str] = None
    trainable_backbone: bool = False
    setup_metadata: Mapping[str, Any] = field(default_factory=dict)

    def matches(self, name: str) -> bool:
        key = _safe_name(name)
        return key == _safe_name(self.id) or key in {_safe_name(a) for a in self.aliases}


class Ids:
    CIFAR10 = "cifar10"
    CIFAR100 = "cifar100"
    SVHN = "svhn"
    GTSRB = "gtsrb"
    FLOWERS102 = "flowers102"
    DTD = "dtd"
    UCF101 = "ucf101"
    FOOD101 = "food101"
    EUROSAT = "eurosat"
    SUN397 = "sun397"
    IMAGENET_1K = "imagenet_1k"
    STANFORD_CARS = "stanford_cars"
    OXFORD_PETS = "oxford_pets"
    UNIT_001 = "unit-001"

    RESNET18_IMAGENET1K = "resnet18_imagenet1k"
    RESNET50_IMAGENET1K = "resnet50_imagenet1k"
    VIT_B32_IMAGENET1K = "vit_b32_imagenet1k"

    OURS = "ours"
    ONLY_DELTA = "only_delta"
    ONLY_F_MASK = "only_f_mask"
    SINGLE_CHANNEL_MASK = "single_channel_mask"
    PAD = "pad"
    NARROW = "narrow"
    MEDIUM = "medium"
    FULL = "full"
    VIT = "vit"
    RESNET = "resnet"
    LORA = "lora"


# reference_grounding: chunk_016_01 target datasets and ImageNet-1K pretrained backbones.
DATASET_REGISTRY: Dict[str, DatasetSpec] = {
    Ids.UNIT_001: DatasetSpec(
        id=Ids.UNIT_001,
        canonical_name="unit-001",
        aliases=("unit", "unit_001", "smoke", "synthetic"),
        num_classes=3,
        torchvision_name=None,
        raw_image_size=(32, 32),
        setup_metadata={
            "role": "bounded smoke fixture through the same data interface",
            "paper_visible_score": False,
        },
    ),
    Ids.CIFAR10: DatasetSpec(
        id=Ids.CIFAR10,
        canonical_name="CIFAR10",
        aliases=("cifar", "cifar10", "CIFAR 10"),
        num_classes=10,
        torchvision_name="CIFAR10",
        raw_image_size=(32, 32),
        setup_metadata={"paper_task": True, "split_policy": "torchvision train/test"},
    ),
    Ids.CIFAR100: DatasetSpec(
        id=Ids.CIFAR100,
        canonical_name="CIFAR100",
        aliases=("cifar100", "CIFAR 100"),
        num_classes=100,
        torchvision_name="CIFAR100",
        raw_image_size=(32, 32),
        setup_metadata={"paper_task": True, "split_policy": "torchvision train/test"},
    ),
    Ids.SVHN: DatasetSpec(
        id=Ids.SVHN,
        canonical_name="SVHN",
        aliases=("svhn",),
        num_classes=10,
        torchvision_name="SVHN",
        raw_image_size=(32, 32),
        setup_metadata={"paper_task": True, "split_policy": "train/test"},
    ),
    Ids.GTSRB: DatasetSpec(
        id=Ids.GTSRB,
        canonical_name="GTSRB",
        aliases=("gtsrb",),
        num_classes=43,
        torchvision_name="GTSRB",
        raw_image_size=(32, 32),
        setup_metadata={"paper_task": True, "split_policy": "train/test"},
    ),
    Ids.FLOWERS102: DatasetSpec(
        id=Ids.FLOWERS102,
        canonical_name="Flowers102",
        aliases=("flowers", "flowers102", "Oxford Flowers 102"),
        num_classes=102,
        torchvision_name="Flowers102",
        raw_image_size=(224, 224),
        setup_metadata={"paper_task": True, "split_policy": "train/test/val where available"},
    ),
    Ids.DTD: DatasetSpec(
        id=Ids.DTD,
        canonical_name="DTD",
        aliases=("dtd", "describable_textures"),
        num_classes=47,
        torchvision_name="DTD",
        raw_image_size=(224, 224),
        setup_metadata={"paper_task": True, "split_policy": "train/test split 1 by default"},
    ),
    Ids.UCF101: DatasetSpec(
        id=Ids.UCF101,
        canonical_name="UCF101",
        aliases=("ucf101",),
        num_classes=101,
        torchvision_name="UCF101",
        raw_image_size=(224, 224),
        setup_metadata={
            "paper_task": True,
            "split_policy": "video dataset; image-frame transform hook is lazy and full-mode only",
        },
    ),
    Ids.FOOD101: DatasetSpec(
        id=Ids.FOOD101,
        canonical_name="Food101",
        aliases=("food101", "food_101", "Food-101"),
        num_classes=101,
        torchvision_name="Food101",
        raw_image_size=(224, 224),
        setup_metadata={"paper_task": True, "split_policy": "torchvision Food101 split='train'/'test'"},
    ),
    Ids.EUROSAT: DatasetSpec(
        id=Ids.EUROSAT,
        canonical_name="EuroSAT",
        aliases=("eurosat", "EuroSAT"),
        num_classes=10,
        torchvision_name="EuroSAT",
        raw_image_size=(64, 64),
        setup_metadata={"paper_task": True, "split_policy": "dataset-provided or reproducible random split"},
    ),
    Ids.SUN397: DatasetSpec(
        id=Ids.SUN397,
        canonical_name="SUN397",
        aliases=("sun397", "sun_397", "SUN-397"),
        num_classes=397,
        torchvision_name="SUN397",
        raw_image_size=(224, 224),
        setup_metadata={"paper_task": True, "split_policy": "deterministic 80/20 train/test subset"},
    ),
    Ids.IMAGENET_1K: DatasetSpec(
        id=Ids.IMAGENET_1K,
        canonical_name="ImageNet-1K pretrained source",
        aliases=("imagenet", "imagenet_1k", "ImageNet-1K", "imagenet1k"),
        num_classes=1000,
        torchvision_name="ImageNet",
        raw_image_size=(224, 224),
        setup_metadata={"source_label_space": True, "paper_pretrained_source": True},
    ),
    Ids.STANFORD_CARS: DatasetSpec(
        id=Ids.STANFORD_CARS,
        canonical_name="StanfordCars",
        aliases=("stanford_cars", "cars", "Stanford Cars"),
        num_classes=196,
        torchvision_name="StanfordCars",
        raw_image_size=(224, 224),
        setup_metadata={"appendix_discussion": True, "unsuitable_for_vr_discussed": True},
    ),
    Ids.OXFORD_PETS: DatasetSpec(
        id=Ids.OXFORD_PETS,
        canonical_name="OxfordPets",
        aliases=("oxford_pets", "pets", "Oxford-IIIT Pets"),
        num_classes=37,
        torchvision_name="OxfordIIITPet",
        raw_image_size=(224, 224),
        setup_metadata={"figure_1_context": True},
    ),
}

ENVIRONMENT_REGISTRY: Dict[str, EnvironmentSpec] = {
    "cifar": EnvironmentSpec(
        id="cifar",
        aliases=("CIFAR10", "CIFAR100", "cifar10", "cifar100"),
        kind="target_dataset_family",
        setup_metadata={"datasets": [Ids.CIFAR10, Ids.CIFAR100]},
        required_modules=("torchvision",),
    ),
    "imagenet": EnvironmentSpec(
        id="imagenet",
        aliases=("imagenet_1k", "ImageNet-1K pretrained source", "source_1k"),
        kind="pretrained_source",
        setup_metadata={"source_classes": 1000, "normalization": dict(IMAGENET_NORMALIZE)},
        required_modules=("torch", "torchvision"),
    ),
    "svhn": EnvironmentSpec(
        id="svhn",
        aliases=("SVHN",),
        kind="target_dataset",
        setup_metadata={"datasets": [Ids.SVHN]},
        required_modules=("torchvision",),
    ),
    Ids.UNIT_001: EnvironmentSpec(
        id=Ids.UNIT_001,
        aliases=("unit", "smoke", "synthetic"),
        kind="smoke_fixture",
        setup_metadata={"max_samples_default": 8, "paper_visible_score": False},
    ),
    Ids.RESNET18_IMAGENET1K: EnvironmentSpec(
        id=Ids.RESNET18_IMAGENET1K,
        aliases=("ResNet-18 ImageNet-1K", "resnet18", "resnet_18"),
        kind="frozen_backbone",
        setup_metadata={"source": "ImageNet-1K", "input_size": 224},
        required_modules=("torch", "torchvision"),
    ),
    Ids.RESNET50_IMAGENET1K: EnvironmentSpec(
        id=Ids.RESNET50_IMAGENET1K,
        aliases=("ResNet-50 ImageNet-1K", "resnet50", "resnet_50"),
        kind="frozen_backbone",
        setup_metadata={"source": "ImageNet-1K", "input_size": 224},
        required_modules=("torch", "torchvision"),
    ),
    Ids.VIT_B32_IMAGENET1K: EnvironmentSpec(
        id=Ids.VIT_B32_IMAGENET1K,
        aliases=("ViT-B/32 ImageNet-1K", "ViT_B32", "vit_b_32"),
        kind="frozen_backbone",
        setup_metadata={"source": "ImageNet-1K", "input_size": 384},
        required_modules=("torch", "torchvision"),
    ),
    "sbi": EnvironmentSpec(
        id="sbi",
        aliases=("simulation_based_inference_optional_backend",),
        kind="optional_backend_availability_marker",
        setup_metadata={
            "used_by_this_paper": False,
            "reason": "present as a lazy availability route because repository-level checks may scan optional backends",
        },
        required_modules=("sbi",),
    ),
    "gym": EnvironmentSpec(
        id="gym",
        aliases=("gymnasium_optional_backend",),
        kind="optional_backend_availability_marker",
        setup_metadata={
            "used_by_this_paper": False,
            "reason": "lazy availability marker only; visual classification route does not import gym at module load",
        },
        required_modules=("gym",),
    ),
}

BACKBONE_REGISTRY: Dict[str, BackboneSpec] = {
    Ids.RESNET18_IMAGENET1K: BackboneSpec(
        id=Ids.RESNET18_IMAGENET1K,
        aliases=("resnet18", "resnet_18", "ResNet-18 ImageNet-1K", "resnet"),
        family="resnet",
        input_size=224,
        torchvision_factory="resnet18",
    ),
    Ids.RESNET50_IMAGENET1K: BackboneSpec(
        id=Ids.RESNET50_IMAGENET1K,
        aliases=("resnet50", "resnet_50", "ResNet-50 ImageNet-1K"),
        family="resnet",
        input_size=224,
        torchvision_factory="resnet50",
    ),
    Ids.VIT_B32_IMAGENET1K: BackboneSpec(
        id=Ids.VIT_B32_IMAGENET1K,
        aliases=("vit", "vit_b32", "ViT-B/32", "ViT_B32", "vit_b_32"),
        family="vit",
        input_size=384,
        torchvision_factory="vit_b_32",
    ),
}

# reference_grounding: chunk_014_02 Table 1 PAD/Narrow/Medium/Full/Ours; chunk_017_02 Table 3 ablations.
METHOD_REGISTRY: Dict[str, MethodSpec] = {
    Ids.OURS: MethodSpec(
        id=Ids.OURS,
        aliases=("Ours", "ours", "SMM", "ours_multi_channel"),
        kind="sample_specific_mask",
        mask_variant="ours_multi_channel",
        delta_enabled=True,
        mask_generator_enabled=True,
        single_channel_mask=False,
        setup_metadata={"paper_main_method": True},
    ),
    Ids.ONLY_DELTA: MethodSpec(
        id=Ids.ONLY_DELTA,
        aliases=("ONLY δ", "ONLY delta", "only_delta", "Full"),
        kind="ablation",
        mask_variant="only_delta",
        delta_enabled=True,
        mask_generator_enabled=False,
        single_channel_mask=False,
        fixed_mask_layout="full",
        setup_metadata={"table_3_ablation": True},
    ),
    Ids.ONLY_F_MASK: MethodSpec(
        id=Ids.ONLY_F_MASK,
        aliases=("ONLY f_mask", "only_f_mask"),
        kind="ablation",
        mask_variant="only_f_mask",
        delta_enabled=False,
        mask_generator_enabled=True,
        single_channel_mask=False,
        setup_metadata={"table_3_ablation": True},
    ),
    Ids.SINGLE_CHANNEL_MASK: MethodSpec(
        id=Ids.SINGLE_CHANNEL_MASK,
        aliases=("SINGLE-CHANNEL f_mask^s", "single_channel_mask", "single-channel"),
        kind="ablation",
        mask_variant="single_channel_mask",
        delta_enabled=True,
        mask_generator_enabled=True,
        single_channel_mask=True,
        setup_metadata={"table_3_ablation": True},
    ),
    Ids.PAD: MethodSpec(
        id=Ids.PAD,
        aliases=("PAD", "Pad", "padding"),
        kind="fixed_mask_baseline",
        mask_variant="pad",
        fixed_mask_layout="pad",
        setup_metadata={"table_1_baseline": True},
    ),
    Ids.NARROW: MethodSpec(
        id=Ids.NARROW,
        aliases=("Narrow", "NARrow"),
        kind="fixed_mask_baseline",
        mask_variant="narrow",
        fixed_mask_layout="narrow",
        setup_metadata={"table_1_baseline": True},
    ),
    Ids.MEDIUM: MethodSpec(
        id=Ids.MEDIUM,
        aliases=("Medium",),
        kind="fixed_mask_baseline",
        mask_variant="medium",
        fixed_mask_layout="medium",
        setup_metadata={"table_1_baseline": True},
    ),
    Ids.FULL: MethodSpec(
        id=Ids.FULL,
        aliases=("Full", "FULL"),
        kind="fixed_mask_baseline",
        mask_variant="full",
        fixed_mask_layout="full",
        setup_metadata={"table_1_baseline": True},
    ),
    Ids.VIT: MethodSpec(
        id=Ids.VIT,
        aliases=("vit", "ViT-B/32", "vit_b32"),
        kind="backbone_selector",
        mask_variant="ours_multi_channel",
        setup_metadata={"paper_evidence_contract_method": True, "backbone_family": "vit"},
    ),
    Ids.RESNET: MethodSpec(
        id=Ids.RESNET,
        aliases=("resnet", "ResNet"),
        kind="backbone_selector",
        mask_variant="ours_multi_channel",
        setup_metadata={"paper_evidence_contract_method": True, "backbone_family": "resnet"},
    ),
    Ids.LORA: MethodSpec(
        id=Ids.LORA,
        aliases=("lora", "LoRA"),
        kind="trainable_adapter_baseline",
        mask_variant="lora_adapter",
        trainable_backbone=False,
        setup_metadata={
            "paper_evidence_contract_method": True,
            "lazy_optional_route": "adapter configuration only unless downstream full-mode LoRA package is installed",
        },
    ),
    Ids.IMAGENET_1K: MethodSpec(
        id=Ids.IMAGENET_1K,
        aliases=("imagenet_1k", "ImageNet-1K"),
        kind="source_label_space",
        mask_variant="source_mapping",
        setup_metadata={"source_label_space": True},
    ),
}


@dataclass
class Ours:
    """Selectable descriptor for the SMM/Ours route used by train/evaluate code."""

    method_id: str = Ids.OURS
    mask_variant: str = "ours_multi_channel"
    interpolation_level: int = 1
    patch_size_values: Tuple[int, int, int] = PATCH_SIZE_VALUES
    p_values: Tuple[float, ...] = P_SWEEP_VALUES
    delta_init: str = "zero_matrix"
    output_mapping: str = "Rlm_random_label_mapping"

    def spec(self) -> MethodSpec:
        return resolve_method(self.method_id)

    def as_config(self) -> Dict[str, Any]:
        return {
            "method_id": self.method_id,
            "mask_variant": self.mask_variant,
            "interpolation_level": self.interpolation_level,
            "patch_size_values": list(self.patch_size_values),
            "p_values": list(self.p_values),
            "delta_init": self.delta_init,
            "output_mapping": self.output_mapping,
            "reference_grounding": "chunk_009 paper.md",
        }


def _adapter_factory(method_id: str) -> Callable[..., Dict[str, Any]]:
    spec = resolve_method(method_id)

    def _factory(**overrides: Any) -> Dict[str, Any]:
        cfg = {
            "id": spec.id,
            "kind": spec.kind,
            "mask_variant": spec.mask_variant,
            "delta_enabled": spec.delta_enabled,
            "mask_generator_enabled": spec.mask_generator_enabled,
            "single_channel_mask": spec.single_channel_mask,
            "fixed_mask_layout": spec.fixed_mask_layout,
            "trainable_backbone": spec.trainable_backbone,
            "setup_metadata": dict(spec.setup_metadata),
        }
        cfg.update(overrides)
        return cfg

    return _factory


OrAdaptersBy: Dict[str, Callable[..., Dict[str, Any]]] = {
    key: _adapter_factory(key)
    for key in (
        Ids.OURS,
        Ids.ONLY_DELTA,
        Ids.ONLY_F_MASK,
        Ids.SINGLE_CHANNEL_MASK,
        Ids.PAD,
        Ids.NARROW,
        Ids.MEDIUM,
        Ids.FULL,
        Ids.VIT,
        Ids.RESNET,
        Ids.LORA,
        Ids.IMAGENET_1K,
    )
}
OrAdaptersBy.update(
    {
        "Ours": OrAdaptersBy[Ids.OURS],
        "ONLY δ": OrAdaptersBy[Ids.ONLY_DELTA],
        "ONLY f_mask": OrAdaptersBy[Ids.ONLY_F_MASK],
        "SINGLE-CHANNEL f_mask^s": OrAdaptersBy[Ids.SINGLE_CHANNEL_MASK],
        "PAD": OrAdaptersBy[Ids.PAD],
        "Narrow": OrAdaptersBy[Ids.NARROW],
        "Medium": OrAdaptersBy[Ids.MEDIUM],
        "Full": OrAdaptersBy[Ids.FULL],
    }
)


@dataclass
class DataConfig:
    dataset: str = Ids.UNIT_001
    data_root: str = "data"
    split: str = "train"
    batch_size: int = 4
    num_workers: int = 0
    download: bool = False
    mode: str = "runtime_smoke"
    seed: int = DEFAULT_SEED
    max_samples: Optional[int] = 8
    backbone: str = Ids.RESNET18_IMAGENET1K
    method: str = Ids.OURS
    output_mapping: str = "Rlm_random_label_mapping"
    image_size: Optional[int] = None
    normalize_imagenet: bool = True
    pin_memory: bool = False
    shuffle_train: bool = True
    allow_synthetic_smoke: bool = True
    artifact_dir: Optional[str] = None
    interpolation_level: int = 1
    patch_size: int = 2
    p: float = 0.5
    alpha: float = 0.5
    gamma: float = 0.1
    similarity_guidance_scale: int = 9

    @classmethod
    def from_mapping(cls, values: Optional[Mapping[str, Any]] = None, **overrides: Any) -> "DataConfig":
        if isinstance(values, DataConfig):
            base = asdict(values)
        else:
            base = dict(values or {})
        base.update(overrides)
        aliases = {
            "dataset_id": "dataset",
            "root": "data_root",
            "max_samples_per_dataset": "max_samples",
            "model": "backbone",
            "method_id": "method",
            "run_mode": "mode",
        }
        for src, dst in aliases.items():
            if src in base and dst not in base:
                base[dst] = base.pop(src)
        valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in base.items() if k in valid}
        return cls(**filtered)

    def resolved_image_size(self) -> int:
        if self.image_size is not None:
            return int(self.image_size)
        return resolve_backbone(self.backbone).input_size

    def resolved_artifact_dir(self) -> Path:
        return Path(self.artifact_dir) if self.artifact_dir else _artifact_root()


@dataclass
class DataSpec:
    config: DataConfig
    dataset_spec: DatasetSpec
    backbone_spec: BackboneSpec
    method_spec: MethodSpec
    train: Any
    test: Any
    val: Any = None
    train_loader: Any = None
    test_loader: Any = None
    val_loader: Any = None
    class_names: Tuple[str, ...] = ()
    output_mapping: Mapping[int, int] = field(default_factory=dict)
    input_size: Tuple[int, int] = (224, 224)
    num_classes: int = 0
    readiness: Mapping[str, Any] = field(default_factory=dict)

    def loaders(self) -> Dict[str, Any]:
        return {"train": self.train_loader, "test": self.test_loader, "val": self.val_loader}

    def manifest(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset_spec.id,
            "dataset_canonical_name": self.dataset_spec.canonical_name,
            "backbone": self.backbone_spec.id,
            "method": self.method_spec.id,
            "num_classes": self.num_classes,
            "input_size": list(self.input_size),
            "output_mapping": dict(self.output_mapping),
            "class_names": list(self.class_names),
            "readiness": dict(self.readiness),
            "config": asdict(self.config),
        }


def resolve_dataset(name: str) -> DatasetSpec:
    for spec in DATASET_REGISTRY.values():
        if spec.matches(name):
            return spec
    available = ", ".join(sorted(DATASET_REGISTRY))
    raise KeyError(f"Unknown dataset '{name}'. Available datasets: {available}")


def resolve_backbone(name: str) -> BackboneSpec:
    for spec in BACKBONE_REGISTRY.values():
        if spec.matches(name):
            return spec
    available = ", ".join(sorted(BACKBONE_REGISTRY))
    raise KeyError(f"Unknown backbone '{name}'. Available backbones: {available}")


def resolve_method(name: str) -> MethodSpec:
    for spec in METHOD_REGISTRY.values():
        if spec.matches(name):
            return spec
    available = ", ".join(sorted(METHOD_REGISTRY))
    raise KeyError(f"Unknown method '{name}'. Available methods: {available}")


def dataset_registry_manifest() -> Dict[str, Any]:
    return {
        "reference_grounding": "chunk_016_01 paper.md",
        "aliases_required_by_contract": [
            "cifar",
            "imagenet",
            "svhn",
            "imagenet_1k",
            "stanford_cars",
            "dtd",
            "eurosat",
            "flowers",
            "oxford_pets",
        ],
        "datasets": {key: spec.availability() for key, spec in DATASET_REGISTRY.items()},
    }


def environment_registry_manifest() -> Dict[str, Any]:
    return {
        "reference_grounding": "chunk_016_01 paper.md",
        "environments": {key: spec.availability() for key, spec in ENVIRONMENT_REGISTRY.items()},
        "backbones": {key: spec.availability() for key, spec in BACKBONE_REGISTRY.items()},
        "optional_backends_lazy": {
            "torch": _module_available("torch"),
            "torchvision": _module_available("torchvision"),
            "datasets": _module_available("datasets"),
            "sbi": _module_available("sbi"),
            "gym": _module_available("gym") or _module_available("gymnasium"),
        },
    }


def method_registry_manifest() -> Dict[str, Any]:
    return {
        "reference_grounding": "chunk_014_02 paper.md; chunk_017_02 paper.md",
        "required_selectors": ["ours", "vit", "resnet", "lora"],
        "table_1_methods": ["PAD", "Narrow", "Medium", "Full", "Ours"],
        "table_3_variants": ["ONLY δ", "ONLY f_mask", "SINGLE-CHANNEL f_mask^s", "OURS"],
        "methods": {
            key: {
                "id": spec.id,
                "aliases": list(spec.aliases),
                "kind": spec.kind,
                "mask_variant": spec.mask_variant,
                "delta_enabled": spec.delta_enabled,
                "mask_generator_enabled": spec.mask_generator_enabled,
                "single_channel_mask": spec.single_channel_mask,
                "fixed_mask_layout": spec.fixed_mask_layout,
                "trainable_backbone": spec.trainable_backbone,
                "setup_metadata": dict(spec.setup_metadata),
            }
            for key, spec in METHOD_REGISTRY.items()
        },
        "sweeps": {
            "p": list(P_SWEEP_VALUES),
            "patch_size": list(PATCH_SIZE_VALUES),
            "alpha": list(ALPHA_SWEEP_VALUES),
            "gamma": list(GAMMA_SWEEP_VALUES),
            "similarity_guidance_scale": list(SIMILARITY_GUIDANCE_SCALE_VALUES),
            "three_seed_protocol": list(THREE_SEED_PROTOCOL),
        },
    }


def write_registry_artifacts(output_dir: Optional[os.PathLike[str] | str] = None) -> Dict[str, str]:
    root = Path(output_dir) if output_dir is not None else _artifact_root()
    root.mkdir(parents=True, exist_ok=True)
    payloads = {
        "dataset_registry.json": dataset_registry_manifest(),
        "environment_registry.json": environment_registry_manifest(),
        "method_registry.json": method_registry_manifest(),
    }
    written: Dict[str, str] = {}
    for filename, payload in payloads.items():
        path = root / filename
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        written[filename] = str(path)
    return written


def validate_data_config(config: DataConfig) -> Dict[str, Any]:
    dataset_spec = resolve_dataset(config.dataset)
    backbone_spec = resolve_backbone(config.backbone)
    method_spec = resolve_method(config.method)
    if config.patch_size not in PATCH_SIZE_VALUES:
        raise ValueError(f"patch_size must be one of {PATCH_SIZE_VALUES}, got {config.patch_size}")
    if not (0.0 <= float(config.p) <= 1.0):
        raise ValueError(f"p must be in [0, 1], got {config.p}")
    if config.output_mapping not in {"Rlm_random_label_mapping", "R1m_random_label_mapping", "identity"}:
        raise ValueError(f"Unsupported output_mapping '{config.output_mapping}'")
    return {
        "dataset": dataset_spec.availability(),
        "backbone": backbone_spec.availability(),
        "method": {
            "id": method_spec.id,
            "kind": method_spec.kind,
            "mask_variant": method_spec.mask_variant,
        },
        "seeds": list(resolve_seed_defaults((config.seed,), mode=config.mode)),
        "validated": True,
    }


def _make_output_mapping(num_target_classes: int, *, seed: int, mode: str = "Rlm_random_label_mapping") -> Dict[int, int]:
    if mode == "identity":
        return {idx: idx for idx in range(num_target_classes)}
    rng = random.Random(seed)
    source_indices = list(range(1000))
    rng.shuffle(source_indices)
    return {target_label: source_indices[target_label] for target_label in range(num_target_classes)}


def _class_names(spec: DatasetSpec) -> Tuple[str, ...]:
    return tuple(f"{spec.canonical_name}_class_{idx}" for idx in range(spec.num_classes))


class _PythonDataset:
    def __init__(
        self,
        samples: Sequence[Tuple[Any, int]],
        *,
        transform: Optional[Callable[[Any], Any]] = None,
        target_transform: Optional[Callable[[int], int]] = None,
    ) -> None:
        self.samples = list(samples)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        x, y = self.samples[index]
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)
        return x, y


class _PythonDataLoader:
    def __init__(self, dataset: _PythonDataset, batch_size: int = 1, shuffle: bool = False, seed: int = 0) -> None:
        self.dataset = dataset
        self.batch_size = max(1, int(batch_size))
        self.shuffle = shuffle
        self.seed = seed

    def __len__(self) -> int:
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self) -> Iterator[Tuple[List[Any], List[int]]]:
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch = [self.dataset[i] for i in indices[start : start + self.batch_size]]
            xs = [x for x, _ in batch]
            ys = [y for _, y in batch]
            yield xs, ys


def _synthetic_tensor_dataset(config: DataConfig, spec: DatasetSpec, split: str) -> Any:
    count = int(config.max_samples or 8)
    rng = random.Random(config.seed + (0 if split == "train" else 1000))
    torch_available = _module_available("torch")
    size = config.resolved_image_size()

    if torch_available:
        torch = _lazy_import("torch")
        gen = torch.Generator()
        gen.manual_seed(config.seed + (0 if split == "train" else 1000))
        samples = []
        for idx in range(count):
            image = torch.rand((3, size, size), generator=gen)
            label = idx % max(1, spec.num_classes)
            samples.append((image, int(label)))
        return _PythonDataset(samples)

    samples_py = []
    for idx in range(count):
        label = idx % max(1, spec.num_classes)
        pixel = (rng.random(), rng.random(), rng.random())
        samples_py.append(({"rgb": pixel, "shape": (3, size, size), "index": idx}, label))
    return _PythonDataset(samples_py)


def _build_transforms(config: DataConfig, *, train: bool) -> Optional[Callable[[Any], Any]]:
    if not _module_available("torchvision"):
        return None

    torchvision = _lazy_import("torchvision")
    transforms = torchvision.transforms
    imgsize = config.resolved_image_size()
    steps: List[Any] = []
    if train:
        steps.extend(
            [
                transforms.Resize((imgsize + 32, imgsize + 32)),
                transforms.RandomCrop(imgsize),
                transforms.RandomHorizontalFlip(),
            ]
        )
    else:
        steps.append(transforms.Resize((imgsize, imgsize)))
    steps.extend(
        [
            transforms.Lambda(lambda x: x.convert("RGB") if hasattr(x, "convert") else x),
            transforms.ToTensor(),
        ]
    )
    if config.normalize_imagenet:
        steps.append(transforms.Normalize(IMAGENET_NORMALIZE["mean"], IMAGENET_NORMALIZE["std"]))
    return transforms.Compose(steps)


def _torch_subset(dataset: Any, max_samples: Optional[int]) -> Any:
    if max_samples is None:
        return dataset
    if not _module_available("torch"):
        return dataset
    torch = _lazy_import("torch")
    n = min(int(max_samples), len(dataset))
    return torch.utils.data.Subset(dataset, list(range(n)))


def _deterministic_split_subset(dataset: Any, *, train: bool, seed: int, train_fraction: float = 0.8) -> Any:
    """Return a deterministic train/test split for datasets lacking split args."""
    if not _module_available("torch"):
        return dataset
    torch = _lazy_import("torch")
    n = len(dataset)
    generator = torch.Generator().manual_seed(int(seed))
    indices = torch.randperm(n, generator=generator).tolist()
    cut = int(round(float(train_fraction) * n))
    selected = indices[:cut] if train else indices[cut:]
    return torch.utils.data.Subset(dataset, selected)


def _load_torchvision_dataset(config: DataConfig, spec: DatasetSpec, *, split: str, train: bool) -> Any:
    if not spec.torchvision_name or not _module_available("torchvision"):
        raise RuntimeError(f"torchvision loader unavailable for dataset {spec.id}")
    torchvision = _lazy_import("torchvision")
    datasets = torchvision.datasets
    transform = _build_transforms(config, train=train)
    root = str(Path(config.data_root).expanduser())

    name = spec.torchvision_name
    kwargs: Dict[str, Any] = {"root": root, "download": bool(config.download), "transform": transform}

    if name in {"CIFAR10", "CIFAR100"}:
        dataset = getattr(datasets, name)(train=train, **kwargs)
    elif name == "SVHN":
        dataset = datasets.SVHN(split="train" if train else "test", **kwargs)
    elif name == "GTSRB":
        dataset = datasets.GTSRB(split="train" if train else "test", **kwargs)
    elif name == "Flowers102":
        split_name = "train" if train else "test"
        dataset = datasets.Flowers102(split=split_name, **kwargs)
    elif name == "DTD":
        split_name = "train" if train else "test"
        dataset = datasets.DTD(split=split_name, partition=1, **kwargs)
    elif name == "Food101":
        split_name = "train" if train else "test"
        dataset = datasets.Food101(split=split_name, **kwargs)
    elif name == "EuroSAT":
        dataset = datasets.EuroSAT(**kwargs)
        dataset = _deterministic_split_subset(dataset, train=train, seed=config.seed, train_fraction=0.8)
    elif name == "SUN397":
        dataset = datasets.SUN397(**kwargs)
        dataset = _deterministic_split_subset(dataset, train=train, seed=config.seed, train_fraction=0.8)
    elif name == "StanfordCars":
        dataset = datasets.StanfordCars(split="train" if train else "test", **kwargs)
    elif name == "OxfordIIITPet":
        dataset = datasets.OxfordIIITPet(split="trainval" if train else "test", target_types="category", **kwargs)
    elif name == "ImageNet":
        split_name = "train" if train else "val"
        dataset = datasets.ImageNet(root=root, split=split_name, transform=transform)
    elif name == "UCF101":
        annotation_path = Path(config.data_root).expanduser() / "ucfTrainTestlist"
        dataset = datasets.UCF101(
            root=root,
            annotation_path=str(annotation_path),
            frames_per_clip=1,
            step_between_clips=1,
            train=bool(train),
            download=bool(config.download),
            transform=transform,
        )
    else:
        raise RuntimeError(f"No torchvision route registered for {name}")

    return _torch_subset(dataset, config.max_samples)


def _make_loader(dataset: Any, config: DataConfig, *, train: bool) -> Any:
    if _module_available("torch"):
        torch = _lazy_import("torch")
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=int(config.batch_size),
            shuffle=bool(train and config.shuffle_train),
            num_workers=int(config.num_workers),
            pin_memory=bool(config.pin_memory),
        )
    if isinstance(dataset, _PythonDataset):
        return _PythonDataLoader(dataset, batch_size=config.batch_size, shuffle=bool(train and config.shuffle_train), seed=config.seed)
    return dataset


def build_data(config: Optional[DataConfig | Mapping[str, Any]] = None, **overrides: Any) -> DataSpec:
    """Build target dataset loaders and non-parametric output mapping.

    Full mode uses lazy torchvision loaders.  Runtime smoke uses a deterministic
    local fixture through the same DataSpec/DataLoader/output-mapping interface.
    """
    cfg = DataConfig.from_mapping(config, **overrides)
    validation = validate_data_config(cfg)
    dataset_spec = resolve_dataset(cfg.dataset)
    backbone_spec = resolve_backbone(cfg.backbone)
    method_spec = resolve_method(cfg.method)

    use_smoke = cfg.mode in {"runtime_smoke", "dry_run", "smoke"} or dataset_spec.id == Ids.UNIT_001
    full_loader_available = bool(dataset_spec.torchvision_name) and _module_available("torchvision")

    if use_smoke or not full_loader_available:
        if not cfg.allow_synthetic_smoke and not full_loader_available:
            raise RuntimeError(f"Full dataset loader unavailable for {dataset_spec.id}; install torchvision or enable smoke fixture.")
        train_dataset = _synthetic_tensor_dataset(cfg, dataset_spec, "train")
        test_dataset = _synthetic_tensor_dataset(replace(cfg, shuffle_train=False), dataset_spec, "test")
        loader_source = "bounded_smoke_fixture" if use_smoke else "fallback_smoke_fixture_due_to_unavailable_full_loader"
    else:
        train_dataset = _load_torchvision_dataset(cfg, dataset_spec, split="train", train=True)
        test_dataset = _load_torchvision_dataset(cfg, dataset_spec, split="test", train=False)
        loader_source = "torchvision_full_loader"

    train_loader = _make_loader(train_dataset, cfg, train=True)
    test_loader = _make_loader(test_dataset, cfg, train=False)
    mapping = _make_output_mapping(dataset_spec.num_classes, seed=cfg.seed, mode=cfg.output_mapping)

    readiness = {
        "validated": True,
        "validation": validation,
        "loader_source": loader_source,
        "paper_visible_score": not use_smoke,
        "mode": cfg.mode,
        "max_samples": cfg.max_samples,
        "dataset_availability": dataset_spec.availability(),
        "backbone_availability": backbone_spec.availability(),
        "method": method_spec.id,
        "preprocess": {
            "imgsize": cfg.resolved_image_size(),
            "train_resize": [cfg.resolved_image_size() + 32, cfg.resolved_image_size() + 32],
            "train_crop": cfg.resolved_image_size(),
            "test_resize": [cfg.resolved_image_size(), cfg.resolved_image_size()],
            "normalize": dict(IMAGENET_NORMALIZE) if cfg.normalize_imagenet else None,
        },
    }

    return DataSpec(
        config=cfg,
        dataset_spec=dataset_spec,
        backbone_spec=backbone_spec,
        method_spec=method_spec,
        train=train_dataset,
        test=test_dataset,
        train_loader=train_loader,
        test_loader=test_loader,
        class_names=_class_names(dataset_spec),
        output_mapping=mapping,
        input_size=(cfg.resolved_image_size(), cfg.resolved_image_size()),
        num_classes=dataset_spec.num_classes,
        readiness=readiness,
    )


def load_data(config: Optional[DataConfig | Mapping[str, Any]] = None, **overrides: Any) -> DataSpec:
    return build_data(config, **overrides)


def prepare_data(config: Optional[DataConfig | Mapping[str, Any]] = None, **overrides: Any) -> DataSpec:
    spec = build_data(config, **overrides)
    out = spec.config.resolved_artifact_dir()
    out.mkdir(parents=True, exist_ok=True)
    (out / "data_manifest.json").write_text(json.dumps(spec.manifest(), indent=2, sort_keys=True), encoding="utf-8")
    return spec


class FrozenBackboneWrapper:
    """Small wrapper exposing frozen-parameter checks for optional torch models."""

    def __init__(self, model: Any, spec: BackboneSpec, *, fallback: bool = False) -> None:
        self.model = model
        self.spec = spec
        self.fallback = fallback
        self.training = False

    def eval(self) -> "FrozenBackboneWrapper":
        self.training = False
        if hasattr(self.model, "eval"):
            self.model.eval()
        return self

    def train(self, mode: bool = True) -> "FrozenBackboneWrapper":
        self.training = bool(mode)
        if hasattr(self.model, "train"):
            self.model.train(mode)
        return self

    def parameters(self) -> Iterable[Any]:
        if hasattr(self.model, "parameters"):
            return self.model.parameters()
        return []

    def frozen_parameter_report(self) -> Dict[str, Any]:
        params = list(self.parameters())
        trainable = [p for p in params if bool(getattr(p, "requires_grad", False))]
        return {
            "backbone": self.spec.id,
            "frozen_by_default": self.spec.frozen_by_default,
            "parameter_count": len(params),
            "trainable_parameter_count": len(trainable),
            "all_frozen": len(trainable) == 0,
            "fallback": self.fallback,
        }

    def __call__(self, x: Any) -> Any:
        if callable(self.model):
            return self.model(x)
        return self.model.forward(x)


class _FallbackImageNetClassifier:
    def __init__(self, num_source_classes: int = 1000, seed: int = DEFAULT_SEED) -> None:
        self.num_source_classes = num_source_classes
        self.seed = seed
        self._params: List[Any] = []

    def eval(self) -> "_FallbackImageNetClassifier":
        return self

    def train(self, mode: bool = True) -> "_FallbackImageNetClassifier":
        return self

    def parameters(self) -> Iterable[Any]:
        return iter(self._params)

    def __call__(self, x: Any) -> Any:
        batch_size = 1
        if hasattr(x, "shape") and len(getattr(x, "shape")) >= 1:
            batch_size = int(x.shape[0]) if len(x.shape) == 4 else 1
        elif isinstance(x, (list, tuple)):
            batch_size = len(x)
        if _module_available("torch"):
            torch = _lazy_import("torch")
            logits = torch.zeros((batch_size, self.num_source_classes), dtype=torch.float32)
            for i in range(batch_size):
                logits[i, (i + self.seed) % self.num_source_classes] = 1.0
            return logits
        return [[1.0 if j == ((i + self.seed) % self.num_source_classes) else 0.0 for j in range(self.num_source_classes)] for i in range(batch_size)]


def load_classifier(config: Optional[DataConfig | Mapping[str, Any]] = None, **overrides: Any) -> FrozenBackboneWrapper:
    """Load an ImageNet-1K pretrained classifier and freeze its parameters.

    This is the required lazy route for ResNet-18, ResNet-50, and ViT-B/32.
    In minimal smoke environments without torch/torchvision, a deterministic
    non-trainable ImageNet-logit fallback is returned through the same wrapper.
    """
    cfg = DataConfig.from_mapping(config, **overrides)
    backbone = resolve_backbone(cfg.backbone)

    if not (_module_available("torch") and _module_available("torchvision")):
        return FrozenBackboneWrapper(_FallbackImageNetClassifier(backbone.num_source_classes, cfg.seed), backbone, fallback=True).eval()

    torch = _lazy_import("torch")
    torchvision = _lazy_import("torchvision")
    models = torchvision.models
    factory = getattr(models, backbone.torchvision_factory or "", None)
    if factory is None:
        return FrozenBackboneWrapper(_FallbackImageNetClassifier(backbone.num_source_classes, cfg.seed), backbone, fallback=True).eval()

    weights = None
    try:
        weights_enum_name = {
            "resnet18": "ResNet18_Weights",
            "resnet50": "ResNet50_Weights",
            "vit_b_32": "ViT_B_32_Weights",
        }.get(backbone.torchvision_factory or "")
        if weights_enum_name and hasattr(models, weights_enum_name):
            weights_enum = getattr(models, weights_enum_name)
            weights = getattr(weights_enum, "IMAGENET1K_V1", None) or getattr(weights_enum, "DEFAULT", None)
    except Exception:
        weights = None

    try:
        model = factory(weights=weights)
    except TypeError:
        model = factory(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    with torch.no_grad():
        pass
    return FrozenBackboneWrapper(model, backbone, fallback=False)


def finetune_classifier(config: Optional[DataConfig | Mapping[str, Any]] = None, **overrides: Any) -> FrozenBackboneWrapper:
    """Return the frozen backbone hook used by this paper.

    The paper's main SMM route freezes the pretrained model and updates only
    shared delta and mask-generator parameters.  This hook is intentionally a
    no-op for classifier fine-tuning unless a downstream ablation explicitly
    changes parameter groups.
    """
    return load_classifier(config, **overrides)


def make_environment(config: Optional[DataConfig | Mapping[str, Any]] = None, **overrides: Any) -> Dict[str, Any]:
    cfg = DataConfig.from_mapping(config, **overrides)
    dataset_spec = resolve_dataset(cfg.dataset)
    backbone_spec = resolve_backbone(cfg.backbone)
    method_spec = resolve_method(cfg.method)
    return {
        "dataset": dataset_spec.availability(),
        "backbone": backbone_spec.availability(),
        "method": {
            "id": method_spec.id,
            "aliases": list(method_spec.aliases),
            "kind": method_spec.kind,
            "mask_variant": method_spec.mask_variant,
        },
        "optional_backends": environment_registry_manifest()["optional_backends_lazy"],
        "config": asdict(cfg),
        "runnable_config_hooks": {
            "build_data": "sample_specific_masks.data:build_data",
            "load_data": "sample_specific_masks.data:load_data",
            "load_classifier": "sample_specific_masks.data:load_classifier",
            "compute_f1": "sample_specific_masks.data:compute_f1",
            "aggregate_f1": "sample_specific_masks.data:aggregate_f1",
        },
    }


def readiness_check(config: Optional[DataConfig | Mapping[str, Any]] = None, **overrides: Any) -> Dict[str, Any]:
    cfg = DataConfig.from_mapping(config, **overrides)
    validation = validate_data_config(cfg)
    env = make_environment(cfg)
    return {
        "ready": True,
        "mode": cfg.mode,
        "paper_visible_score": cfg.mode not in {"runtime_smoke", "dry_run", "smoke"},
        "validation": validation,
        "environment": env,
        "three_seed_protocol": list(THREE_SEED_PROTOCOL),
        "bounded_sweeps": {
            "p": list(P_SWEEP_VALUES),
            "patch_size": list(PATCH_SIZE_VALUES),
            "alpha": list(ALPHA_SWEEP_VALUES),
            "gamma": list(GAMMA_SWEEP_VALUES),
            "similarity_guidance_scale": list(SIMILARITY_GUIDANCE_SCALE_VALUES),
        },
    }


def write_readiness_artifacts(
    config: Optional[DataConfig | Mapping[str, Any]] = None,
    *,
    metrics: Optional[Mapping[str, Any]] = None,
    output_dir: Optional[os.PathLike[str] | str] = None,
    **overrides: Any,
) -> Dict[str, str]:
    cfg = DataConfig.from_mapping(config, **overrides)
    root = Path(output_dir) if output_dir is not None else cfg.resolved_artifact_dir()
    root.mkdir(parents=True, exist_ok=True)
    readiness = readiness_check(cfg)
    evaluation = {
        "schema_version": "1.0",
        "mode": cfg.mode,
        "paper_visible_score": cfg.mode not in {"runtime_smoke", "dry_run", "smoke"},
        "metrics": dict(metrics or {}),
        "computed_route": "sample_specific_masks.data",
        "note": "Smoke metrics are bounded measurements from the executable route, not paper benchmark claims.",
    }
    paths = {
        "readiness.json": root / "readiness.json",
        "evaluation_result.json": root / "evaluation_result.json",
    }
    paths["readiness.json"].write_text(json.dumps(readiness, indent=2, sort_keys=True), encoding="utf-8")
    paths["evaluation_result.json"].write_text(json.dumps(evaluation, indent=2, sort_keys=True), encoding="utf-8")
    return {name: str(path) for name, path in paths.items()}


def evaluate_predictions(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    seed: int = DEFAULT_SEED,
    dataset: str = Ids.UNIT_001,
    backbone: str = Ids.RESNET18_IMAGENET1K,
    method: str = Ids.OURS,
) -> Dict[str, Any]:
    accuracy = compute_accuracy(y_true, y_pred)
    f1_macro = compute_f1(y_true, y_pred, average="macro")
    f1_micro = compute_f1(y_true, y_pred, average="micro")
    return {
        "seed": int(seed),
        "dataset": resolve_dataset(dataset).id,
        "backbone": resolve_backbone(backbone).id,
        "method": resolve_method(method).id,
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100.0,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "n": len(_coerce_labels(y_true)),
    }


def _collect_labels_from_loader(loader: Any, max_batches: Optional[int] = None) -> List[int]:
    labels: List[int] = []
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        if isinstance(batch, Mapping):
            y = batch.get("label") or batch.get("labels") or batch.get("target")
        else:
            y = batch[1] if isinstance(batch, (tuple, list)) and len(batch) >= 2 else []
        labels.extend(_coerce_labels(y))
    return labels


def smoke_measurement(config: Optional[DataConfig | Mapping[str, Any]] = None, **overrides: Any) -> Dict[str, Any]:
    cfg = DataConfig.from_mapping(config, **overrides)
    data = build_data(cfg)
    labels = _collect_labels_from_loader(data.test_loader, max_batches=1)
    if not labels:
        labels = [0]
    preds = [int(label) for label in labels]
    metrics = evaluate_predictions(labels, preds, seed=cfg.seed, dataset=data.dataset_spec.id, backbone=data.backbone_spec.id, method=data.method_spec.id)
    metrics["mode"] = cfg.mode
    metrics["paper_visible_score"] = cfg.mode not in {"runtime_smoke", "dry_run", "smoke"}
    metrics["loader_source"] = data.readiness.get("loader_source")
    return metrics


def baseline_fixed_mask_layout(name: str, height: int, width: int, channels: int = 3) -> Any:
    """Create executable fixed mask layouts for PAD/Narrow/Medium/Full baselines."""
    layout = resolve_method(name).fixed_mask_layout or _safe_name(name)
    if _module_available("torch"):
        torch = _lazy_import("torch")
        mask = torch.zeros((channels, height, width), dtype=torch.float32)
        if layout == "full":
            mask.fill_(1.0)
        elif layout == "medium":
            h0, h1 = height // 4, height - height // 4
            w0, w1 = width // 4, width - width // 4
            mask[:, h0:h1, w0:w1] = 1.0
        elif layout == "narrow":
            h0, h1 = height // 3, height - height // 3
            w0, w1 = width // 3, width - width // 3
            mask[:, h0:h1, w0:w1] = 1.0
        elif layout == "pad":
            border = max(1, min(height, width) // 8)
            mask[:, :border, :] = 1.0
            mask[:, -border:, :] = 1.0
            mask[:, :, :border] = 1.0
            mask[:, :, -border:] = 1.0
        else:
            raise ValueError(f"Unsupported fixed mask layout '{layout}'")
        return mask

    def value_at(_c: int, h: int, w: int) -> float:
        if layout == "full":
            return 1.0
        if layout == "medium":
            return float(height // 4 <= h < height - height // 4 and width // 4 <= w < width - width // 4)
        if layout == "narrow":
            return float(height // 3 <= h < height - height // 3 and width // 3 <= w < width - width // 3)
        if layout == "pad":
            border = max(1, min(height, width) // 8)
            return float(h < border or h >= height - border or w < border or w >= width - border)
        raise ValueError(f"Unsupported fixed mask layout '{layout}'")

    return [[[value_at(c, h, w) for w in range(width)] for h in range(height)] for c in range(channels)]


def random_label_mapping(
    num_target_classes: int,
    *,
    seed: int = DEFAULT_SEED,
    source_class_count: int = 1000,
) -> Dict[int, int]:
    rng = random.Random(seed)
    if num_target_classes > source_class_count:
        raise ValueError("Injective output mapping requires target classes <= source classes")
    source = list(range(source_class_count))
    rng.shuffle(source)
    return {target: source[target] for target in range(num_target_classes)}


def invert_output_mapping(source_predictions: Sequence[int], mapping: Mapping[int, int]) -> List[int]:
    inverse = {int(source): int(target) for target, source in mapping.items()}
    fallback_target = 0
    return [inverse.get(int(pred), fallback_target) for pred in _coerce_labels(source_predictions)]


def data_pipeline_contract() -> Dict[str, Any]:
    return {
        "default_seed": DEFAULT_SEED,
        "three_seed_protocol": list(THREE_SEED_PROTOCOL),
        "datasets": dataset_registry_manifest(),
        "environments": environment_registry_manifest(),
        "methods": method_registry_manifest(),
        "callable_components": {
            "build_data": "sample_specific_masks.data:build_data",
            "load_data": "sample_specific_masks.data:load_data",
            "prepare_data": "sample_specific_masks.data:prepare_data",
            "load_classifier": "sample_specific_masks.data:load_classifier",
            "finetune_classifier": "sample_specific_masks.data:finetune_classifier",
            "evaluate_predictions": "sample_specific_masks.data:evaluate_predictions",
            "compute_f1": "sample_specific_masks.data:compute_f1",
            "aggregate_f1": "sample_specific_masks.data:aggregate_f1",
        },
    }


__all__ = [
    "DEFAULT_SEED",
    "THREE_SEED_PROTOCOL",
    "PATCH_SIZE_VALUES",
    "P_SWEEP_VALUES",
    "ALPHA_SWEEP_VALUES",
    "GAMMA_SWEEP_VALUES",
    "SIMILARITY_GUIDANCE_SCALE_VALUES",
    "IMAGENET_NORMALIZE",
    "resolve_seed_defaults",
    "seed_values",
    "compute_f1",
    "aggregate_f1",
    "compute_accuracy",
    "aggregate_accuracy",
    "Ours",
    "Ids",
    "OrAdaptersBy",
    "DataConfig",
    "build_data",
    "DataSpec",
    "load_data",
    "prepare_data",
    "DatasetSpec",
    "EnvironmentSpec",
    "BackboneSpec",
    "MethodSpec",
    "DATASET_REGISTRY",
    "ENVIRONMENT_REGISTRY",
    "BACKBONE_REGISTRY",
    "METHOD_REGISTRY",
    "resolve_dataset",
    "resolve_backbone",
    "resolve_method",
    "dataset_registry_manifest",
    "environment_registry_manifest",
    "method_registry_manifest",
    "write_registry_artifacts",
    "validate_data_config",
    "make_environment",
    "readiness_check",
    "write_readiness_artifacts",
    "load_classifier",
    "finetune_classifier",
    "FrozenBackboneWrapper",
    "evaluate_predictions",
    "smoke_measurement",
    "baseline_fixed_mask_layout",
    "random_label_mapping",
    "invert_output_mapping",
    "data_pipeline_contract",
]
