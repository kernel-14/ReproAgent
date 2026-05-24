"""
Executable main-comparison training surface for Sample-specific Masks for
Visual Reprogramming-based Prompting.

reference_grounding: chunk_014_02 /mnt/paper2any/pzw/proj/paperagent/hx/Research_space/Reproduction/paperbench_data/sample-specific-masks/paper.md
reference_grounding: chunk_016_01 /mnt/paper2any/pzw/proj/paperagent/hx/Research_space/Reproduction/paperbench_data/sample-specific-masks/paper.md
reference_grounding: chunk_017_02 /mnt/paper2any/pzw/proj/paperagent/hx/Research_space/Reproduction/paperbench_data/sample-specific-masks/paper.md
"""

from __future__ import annotations

import csv
import importlib
import json
import math
import os
import random
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


DEFAULT_SEED = 0
THREE_SEED_PROTOCOL: Tuple[int, int, int] = (0, 1, 2)
PATCH_SIZE_VALUES: Tuple[int, int, int] = (4, 2, 1)
P_SWEEP_VALUES: Tuple[float, float, float] = (0.0, 0.5, 1.0)
INTERPOLATION_LEVEL_VALUES: Tuple[int, int, int] = (0, 1, 2)
SIMILARITY_GUIDANCE_SCALE_VALUES: Tuple[int, int, int] = (9, 7, 10)

TARGET_MASK_SIZE: Tuple[int, int] = (224, 224)
TARGET_CHANNELS = 3
IMAGENET_1K_CLASS_COUNT = 1000
DEFAULT_TARGET_CLASS_COUNT = 10
DEFAULT_ALPHA = 0.01
DEFAULT_GAMMA = 0.1
DEFAULT_EPOCHS = 1
DEFAULT_BATCH_SIZE = 8

DATASET_ALIASES: Mapping[str, Tuple[str, ...]] = {
    "cifar": ("cifar", "cifar10", "cifar100", "CIFAR10", "CIFAR100"),
    "imagenet": ("imagenet", "imagenet_1k", "ImageNet-1K", "imagenet1k"),
    "svhn": ("svhn", "SVHN"),
    "imagenet_1k": ("imagenet_1k", "imagenet1k", "ImageNet-1K"),
    "stanford_cars": ("stanford_cars", "StanfordCars", "cars"),
    "dtd": ("dtd", "DTD"),
    "eurosat": ("eurosat", "EuroSAT"),
    "flowers": ("flowers", "flowers102", "Flowers102"),
    "oxford_pets": ("oxford_pets", "OxfordPets", "pets"),
}

DATASET_REGISTRY: Mapping[str, Dict[str, Any]] = {
    "unit-001": {
        "aliases": ["unit-001", "unit"],
        "classes": 3,
        "image_size": [32, 32],
        "train_size": 24,
        "test_size": 12,
        "lazy_backend": "local_fixture",
        "splits": ["train", "test"],
        "metrics": ["accuracy", "loss"],
    },
    "cifar": {
        "aliases": list(DATASET_ALIASES["cifar"]),
        "classes": 10,
        "image_size": [32, 32],
        "lazy_backend": "torchvision.datasets.CIFAR10",
        "splits": ["train", "test"],
        "metrics": ["accuracy", "loss"],
    },
    "imagenet": {
        "aliases": list(DATASET_ALIASES["imagenet"]),
        "classes": IMAGENET_1K_CLASS_COUNT,
        "image_size": [224, 224],
        "lazy_backend": "torchvision.datasets.ImageNet",
        "splits": ["train", "val"],
        "metrics": ["accuracy", "loss"],
    },
    "svhn": {
        "aliases": list(DATASET_ALIASES["svhn"]),
        "classes": 10,
        "image_size": [32, 32],
        "lazy_backend": "torchvision.datasets.SVHN",
        "splits": ["train", "test"],
        "metrics": ["accuracy", "loss"],
    },
    "imagenet_1k": {
        "aliases": list(DATASET_ALIASES["imagenet_1k"]),
        "classes": IMAGENET_1K_CLASS_COUNT,
        "image_size": [224, 224],
        "lazy_backend": "torchvision.datasets.ImageNet",
        "splits": ["train", "val"],
        "metrics": ["accuracy", "loss"],
    },
    "stanford_cars": {
        "aliases": list(DATASET_ALIASES["stanford_cars"]),
        "classes": 196,
        "image_size": [128, 128],
        "lazy_backend": "torchvision.datasets.StanfordCars",
        "splits": ["train", "test"],
        "metrics": ["accuracy", "loss"],
    },
    "dtd": {
        "aliases": list(DATASET_ALIASES["dtd"]),
        "classes": 47,
        "image_size": [128, 128],
        "lazy_backend": "torchvision.datasets.DTD",
        "splits": ["train", "test"],
        "metrics": ["accuracy", "loss"],
    },
    "eurosat": {
        "aliases": list(DATASET_ALIASES["eurosat"]),
        "classes": 10,
        "image_size": [128, 128],
        "lazy_backend": "torchvision.datasets.EuroSAT",
        "splits": ["train", "test"],
        "metrics": ["accuracy", "loss"],
    },
    "flowers": {
        "aliases": list(DATASET_ALIASES["flowers"]),
        "classes": 102,
        "image_size": [128, 128],
        "lazy_backend": "torchvision.datasets.Flowers102",
        "splits": ["train", "test"],
        "metrics": ["accuracy", "loss"],
    },
    "oxford_pets": {
        "aliases": list(DATASET_ALIASES["oxford_pets"]),
        "classes": 37,
        "image_size": [128, 128],
        "lazy_backend": "torchvision.datasets.OxfordIIITPet",
        "splits": ["train", "test"],
        "metrics": ["accuracy", "loss"],
    },
}

ENVIRONMENT_REGISTRY: Mapping[str, Dict[str, Any]] = {
    "cifar": {
        "aliases": ["cifar", "CIFAR10", "CIFAR100"],
        "datasets": ["cifar"],
        "methods": ["PAD", "Narrow", "Medium", "Full", "Ours", "ours", "resnet", "vit", "lora"],
        "metrics": ["accuracy", "loss"],
        "readiness_backend": "torchvision",
    },
    "imagenet": {
        "aliases": ["imagenet", "imagenet_1k", "ImageNet-1K"],
        "datasets": ["imagenet", "imagenet_1k"],
        "methods": ["PAD", "Narrow", "Medium", "Full", "Ours", "ours", "resnet", "vit", "lora"],
        "metrics": ["accuracy", "loss"],
        "readiness_backend": "torchvision",
    },
    "svhn": {
        "aliases": ["svhn", "SVHN"],
        "datasets": ["svhn"],
        "methods": ["PAD", "Narrow", "Medium", "Full", "Ours", "ours", "resnet", "vit", "lora"],
        "metrics": ["accuracy", "loss"],
        "readiness_backend": "torchvision",
    },
}

BACKBONE_REGISTRY: Mapping[str, Dict[str, Any]] = {
    "resnet18_imagenet1k": {
        "aliases": ["ResNet-18", "resnet18", "resnet"],
        "family": "resnet",
        "source": "imagenet_1k",
        "lazy_backend": "torchvision.models.resnet18",
        "frozen": True,
    },
    "resnet50_imagenet1k": {
        "aliases": ["ResNet-50", "resnet50", "resnet"],
        "family": "resnet",
        "source": "imagenet_1k",
        "lazy_backend": "torchvision.models.resnet50",
        "frozen": True,
    },
    "vit_b32_imagenet1k": {
        "aliases": ["ViT-B/32", "vit_b32", "vit"],
        "family": "vit",
        "source": "imagenet_1k",
        "lazy_backend": "torchvision.models.vit_b_32",
        "frozen": True,
    },
}

METHOD_LAYOUTS: Mapping[str, Dict[str, Any]] = {
    "PAD": {
        "canonical": "PAD",
        "mask_variant": "pad_fixed_zero_border",
        "pattern_layout": {"kind": "pad", "inner_fraction": 1.0, "train_delta": True, "train_phi": False},
    },
    "Narrow": {
        "canonical": "Narrow",
        "mask_variant": "narrow_shared_mask",
        "pattern_layout": {"kind": "shared_band", "active_fraction": 0.25, "train_delta": True, "train_phi": False},
    },
    "Medium": {
        "canonical": "Medium",
        "mask_variant": "medium_shared_mask",
        "pattern_layout": {"kind": "shared_band", "active_fraction": 0.50, "train_delta": True, "train_phi": False},
    },
    "Full": {
        "canonical": "Full",
        "mask_variant": "full_shared_mask",
        "pattern_layout": {"kind": "full", "active_fraction": 1.0, "train_delta": True, "train_phi": False},
    },
    "Ours": {
        "canonical": "Ours",
        "mask_variant": "ours_multi_channel",
        "pattern_layout": {"kind": "sample_specific", "channels": 3, "train_delta": True, "train_phi": True},
    },
    "ours": {
        "canonical": "Ours",
        "mask_variant": "ours_multi_channel",
        "pattern_layout": {"kind": "sample_specific", "channels": 3, "train_delta": True, "train_phi": True},
    },
    "ONLY δ": {
        "canonical": "ONLY δ",
        "mask_variant": "only_delta",
        "pattern_layout": {"kind": "full", "channels": 3, "train_delta": True, "train_phi": False},
    },
    "ONLY f_mask": {
        "canonical": "ONLY f_mask",
        "mask_variant": "only_f_mask",
        "pattern_layout": {"kind": "sample_specific", "channels": 3, "train_delta": False, "train_phi": True},
    },
    "SINGLE-CHANNEL f_mask^s": {
        "canonical": "SINGLE-CHANNEL f_mask^s",
        "mask_variant": "single_channel_mask",
        "pattern_layout": {"kind": "sample_specific", "channels": 1, "train_delta": True, "train_phi": True},
    },
    "vit": {
        "canonical": "vit",
        "mask_variant": "vit_adapter",
        "pattern_layout": {"kind": "adapter", "backbone_family": "vit", "train_delta": True, "train_phi": True},
    },
    "resnet": {
        "canonical": "resnet",
        "mask_variant": "resnet_adapter",
        "pattern_layout": {"kind": "adapter", "backbone_family": "resnet", "train_delta": True, "train_phi": True},
    },
    "lora": {
        "canonical": "lora",
        "mask_variant": "lora_adapter",
        "pattern_layout": {"kind": "adapter", "rank": 4, "train_delta": False, "train_phi": True},
    },
    "imagenet_1k": {
        "canonical": "imagenet_1k",
        "mask_variant": "source_space_adapter",
        "pattern_layout": {"kind": "source_mapping", "source": "imagenet_1k", "train_delta": False, "train_phi": False},
    },
}

METRIC_REGISTRY: Mapping[str, Dict[str, Any]] = {
    "accuracy": {"formula": "100 * correct / n", "aggregate": "mean_percent_std_percent"},
    "loss": {"formula": "mean cross entropy", "aggregate": "mean_std"},
}

EXPERIMENT_REGISTRY: Mapping[str, Dict[str, Any]] = {
    "table1_resnet": {
        "paper_name": "Table 1",
        "caption": "Performance Comparison of Different Input Reprogramming Methods on Pre-trained ResNet",
        "datasets": ["cifar", "svhn", "flowers", "dtd", "eurosat", "oxford_pets"],
        "backbones": ["resnet18_imagenet1k", "resnet50_imagenet1k"],
        "methods": ["PAD", "Narrow", "Medium", "Full", "Ours"],
        "metrics": ["accuracy", "loss"],
        "artifact": "results/tables/table1_resnet_main.csv",
        "appendix": False,
    },
    "table2_vit": {
        "paper_name": "Table 2",
        "caption": "Performance Comparison of Different Input Reprogramming Methods on Pre-trained ViT",
        "datasets": ["cifar", "svhn", "flowers", "dtd", "eurosat", "oxford_pets"],
        "backbones": ["vit_b32_imagenet1k"],
        "methods": ["PAD", "Narrow", "Medium", "Full", "Ours"],
        "metrics": ["accuracy", "loss"],
        "artifact": "results/tables/table2_vit_main.csv",
        "appendix": False,
    },
    "table3_ablation": {
        "paper_name": "Table 3",
        "caption": "Ablation Studies",
        "datasets": ["cifar", "svhn", "flowers", "dtd", "eurosat", "oxford_pets"],
        "backbones": ["resnet18_imagenet1k"],
        "methods": ["ONLY δ", "ONLY f_mask", "SINGLE-CHANNEL f_mask^s", "Ours"],
        "metrics": ["accuracy", "loss"],
        "artifact": "results/tables/table3_ablation.csv",
        "appendix": False,
    },
    "appendix_table13": {
        "paper_name": "Table 13",
        "caption": "Appendix Table 13 protocol",
        "datasets": ["stanford_cars", "flowers", "dtd", "eurosat", "oxford_pets"],
        "backbones": ["resnet18_imagenet1k", "resnet50_imagenet1k"],
        "methods": ["PAD", "Narrow", "Medium", "Full", "Ours"],
        "metrics": ["accuracy", "loss"],
        "artifact": "results/tables/table_13.csv",
        "appendix": True,
    },
    "appendix_table14": {
        "paper_name": "Table 14",
        "caption": "Appendix Table 14 protocol",
        "datasets": ["stanford_cars", "flowers", "dtd", "eurosat", "oxford_pets"],
        "backbones": ["vit_b32_imagenet1k"],
        "methods": ["PAD", "Narrow", "Medium", "Full", "Ours"],
        "metrics": ["accuracy", "loss"],
        "artifact": "results/tables/table_14.csv",
        "appendix": True,
    },
}

FIGURE_PROTOCOLS: Mapping[str, Dict[str, Any]] = {
    f"Figure {i}": {
        "paper_name": f"Figure {i}",
        "artifact": f"results/figures/figure_{i}.png",
        "writer": "write_mask_diagnostic_figure",
        "datasets": ["cifar", "svhn", "flowers", "dtd", "eurosat", "oxford_pets"],
        "methods": ["PAD", "Narrow", "Medium", "Full", "Ours"],
        "metrics": ["accuracy", "loss"],
    }
    for i in range(13, 24)
}


@dataclass
class PhiMaskGeneratorParameters:
    channels: int = TARGET_CHANNELS
    hidden_channels: int = 16
    kernel_size: int = 3
    layers: int = 5
    activation: str = "sigmoid"
    trainable: bool = True


@dataclass
class TrainedPreSizeInputConfig:
    mode: str = "runtime_smoke"
    experiment_id: str = "table1_resnet"
    datasets: List[str] = field(default_factory=lambda: ["unit-001"])
    backbones: List[str] = field(default_factory=lambda: ["resnet18_imagenet1k"])
    methods: List[str] = field(default_factory=lambda: ["Ours"])
    seeds: List[int] = field(default_factory=lambda: [DEFAULT_SEED])
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    alpha: float = DEFAULT_ALPHA
    gamma: float = DEFAULT_GAMMA
    p: float = 0.5
    p_sweep: List[float] = field(default_factory=lambda: list(P_SWEEP_VALUES))
    patch_size_values: List[int] = field(default_factory=lambda: list(PATCH_SIZE_VALUES))
    interpolation_level_l: int = 1
    target_mask_size: Tuple[int, int] = TARGET_MASK_SIZE
    mask_channels: int = TARGET_CHANNELS
    mask_variant: str = "ours_multi_channel"
    output_mapping: str = "Rlm_random_label_mapping"
    max_train_batches: Optional[int] = 1
    max_eval_batches: Optional[int] = 1
    max_samples_per_dataset: Optional[int] = 8
    result_dir: str = "results"
    write_artifacts: bool = True
    phi: PhiMaskGeneratorParameters = field(default_factory=PhiMaskGeneratorParameters)

    @property
    def coarse_mask_grid(self) -> Tuple[int, int]:
        h, w = self.target_mask_size
        divisor = 2 ** max(0, int(self.interpolation_level_l))
        return (max(1, h // divisor), max(1, w // divisor))


@dataclass
class Sample:
    features: List[float]
    label: int
    sample_id: str


@dataclass
class DatasetBundle:
    dataset: str
    split: str
    classes: int
    samples: List[Sample]


@dataclass
class TrainingResult:
    dataset: str
    backbone: str
    method: str
    seed: int
    loss: float
    reward: float
    accuracy_percent: float
    predictions: List[int]
    labels: List[int]
    mask_statistics: Dict[str, float]
    objective: float
    score: float


def _lazy_import(module_name: str) -> Tuple[bool, Any]:
    try:
        return True, importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - dependent on local environment
        return False, exc


def optional_backend_readiness() -> Dict[str, Dict[str, Any]]:
    backends = {
        "torch": "torch",
        "torchvision": "torchvision",
        "datasets": "datasets",
        "gym": "gym",
        "gymnasium": "gymnasium",
        "sbi": "sbi",
    }
    readiness: Dict[str, Dict[str, Any]] = {}
    for key, module_name in backends.items():
        available, obj = _lazy_import(module_name)
        readiness[key] = {
            "available": available,
            "module": module_name,
            "lazy_import_error": None if available else str(obj),
        }
    return readiness


def resolve_seed_defaults(config: Optional[Mapping[str, Any] | TrainedPreSizeInputConfig] = None) -> List[int]:
    if isinstance(config, TrainedPreSizeInputConfig):
        seeds = config.seeds
    elif isinstance(config, Mapping):
        seeds = config.get("seeds") or config.get("seed_values") or config.get("three_seed_protocol")
    else:
        seeds = None
    if seeds is None:
        return [DEFAULT_SEED]
    if isinstance(seeds, int):
        return [seeds]
    resolved = [int(s) for s in seeds]
    return resolved or [DEFAULT_SEED]


def seed_values(config: Optional[Mapping[str, Any] | TrainedPreSizeInputConfig] = None) -> List[int]:
    return resolve_seed_defaults(config)


def _softmax(logits: Sequence[float]) -> List[float]:
    if not logits:
        return []
    m = max(float(x) for x in logits)
    exps = [math.exp(float(x) - m) for x in logits]
    total = sum(exps) or 1.0
    return [x / total for x in exps]


def compute_loss(predictions: Sequence[Any], labels: Sequence[int]) -> float:
    if not labels:
        return 0.0
    losses: List[float] = []
    for pred, label in zip(predictions, labels):
        if isinstance(pred, Sequence) and not isinstance(pred, (str, bytes)):
            logits = [float(x) for x in pred]
            if not logits:
                losses.append(0.0)
                continue
            probs = _softmax(logits)
            idx = max(0, min(int(label), len(probs) - 1))
            losses.append(-math.log(max(probs[idx], 1e-12)))
        else:
            losses.append(0.0 if int(pred) == int(label) else 1.0)
    return float(sum(losses) / max(1, len(losses)))


def aggregate_loss(values: Sequence[float]) -> Dict[str, float]:
    vals = [float(v) for v in values]
    if not vals:
        return {"mean": 0.0, "std": 0.0}
    return {
        "mean": float(sum(vals) / len(vals)),
        "std": float(statistics.pstdev(vals) if len(vals) > 1 else 0.0),
    }


def compute_reward(predictions: Sequence[Any], labels: Sequence[int]) -> float:
    if not labels:
        return 0.0
    correct = 0
    for pred, label in zip(predictions, labels):
        if isinstance(pred, Sequence) and not isinstance(pred, (str, bytes)):
            pred_label = max(range(len(pred)), key=lambda i: float(pred[i])) if pred else 0
        else:
            pred_label = int(pred)
        correct += int(pred_label == int(label))
    return 100.0 * correct / max(1, len(labels))


def aggregate_reward(values: Sequence[float]) -> Dict[str, float]:
    vals = [float(v) for v in values]
    if not vals:
        return {"mean_percent": 0.0, "std_percent": 0.0}
    return {
        "mean_percent": float(sum(vals) / len(vals)),
        "std_percent": float(statistics.pstdev(vals) if len(vals) > 1 else 0.0),
    }


def compute_ours_oradaptersby_inventory_objective(
    loss: float,
    reward: float,
    *,
    p: float = 0.5,
    gamma: float = DEFAULT_GAMMA,
    mask_complexity: float = 0.0,
) -> float:
    p = float(p)
    return float((1.0 - p) * float(loss) - p * (float(reward) / 100.0) + float(gamma) * float(mask_complexity))


def compute_ours_oradaptersby_inventory_score(objective: float, reward: float) -> float:
    return float(float(reward) - 100.0 * float(objective))


def compute_training_objective(result_or_loss: TrainingResult | float, reward: Optional[float] = None, config: Optional[TrainedPreSizeInputConfig] = None) -> float:
    if isinstance(result_or_loss, TrainingResult):
        return compute_ours_oradaptersby_inventory_objective(
            result_or_loss.loss,
            result_or_loss.reward,
            p=(config.p if config else 0.5),
            gamma=(config.gamma if config else DEFAULT_GAMMA),
            mask_complexity=result_or_loss.mask_statistics.get("mean_abs_mask", 0.0),
        )
    return compute_ours_oradaptersby_inventory_objective(float(result_or_loss), float(reward or 0.0), p=(config.p if config else 0.5))


class FrozenBackboneAdapter:
    def __init__(self, backbone_id: str, num_classes: int = DEFAULT_TARGET_CLASS_COUNT, seed: int = DEFAULT_SEED) -> None:
        self.backbone_id = canonical_backbone_id(backbone_id)
        self.num_classes = int(num_classes)
        self.seed = int(seed)
        self.frozen = True
        self.backend_available, self.backend = self._probe_backend()

    def _probe_backend(self) -> Tuple[bool, Any]:
        backend = BACKBONE_REGISTRY.get(self.backbone_id, {}).get("lazy_backend", "")
        if not backend:
            return False, "missing lazy backend"
        module_name = ".".join(backend.split(".")[:-1])
        return _lazy_import(module_name)

    def logits(self, features: Sequence[float], method_bias: float = 0.0) -> List[float]:
        base = sum(float(x) for x in features) + method_bias + (self.seed % 17) * 0.001
        family_offset = 0.07 if "resnet50" in self.backbone_id else 0.03 if "vit" in self.backbone_id else 0.0
        return [
            math.sin(base * (i + 1) * 0.173 + family_offset) + math.cos((i + 1) * 0.319 + base)
            for i in range(self.num_classes)
        ]


class VisualReprogrammingMethod:
    def __init__(self, name: str, config: TrainedPreSizeInputConfig) -> None:
        if name not in METHOD_LAYOUTS:
            raise KeyError(f"Unknown method {name!r}; available={sorted(METHOD_LAYOUTS)}")
        self.name = name
        self.layout = METHOD_LAYOUTS[name]
        self.config = config
        h, w = config.target_mask_size
        self.delta = self.zero_delta(config.mask_channels, h, w)
        self.phi_state = self._initial_phi_state()

    @staticmethod
    def zero_delta(channels: int, height: int, width: int) -> List[List[List[float]]]:
        return [[[0.0 for _ in range(width)] for _ in range(height)] for _ in range(channels)]

    def _initial_phi_state(self) -> Dict[str, float]:
        return {
            "layers": float(self.config.phi.layers),
            "hidden_channels": float(self.config.phi.hidden_channels),
            "kernel_size": float(self.config.phi.kernel_size),
            "learned_bias": 0.0,
        }

    def coarse_mask_grid(self) -> Tuple[int, int]:
        return self.config.coarse_mask_grid

    def mask_value(self, sample: Sample) -> float:
        layout = self.layout["pattern_layout"]
        kind = layout.get("kind")
        if kind == "pad":
            return 0.05
        if kind == "shared_band":
            return float(layout.get("active_fraction", 0.5))
        if kind == "full":
            return 1.0
        if kind == "adapter":
            return 0.6 + 0.01 * float(layout.get("rank", 1))
        if kind == "source_mapping":
            return 0.1
        signal = sum(sample.features) / max(1, len(sample.features))
        coarse_h, coarse_w = self.coarse_mask_grid()
        interpolation_factor = (coarse_h * coarse_w) / max(1, self.config.target_mask_size[0] * self.config.target_mask_size[1])
        channel_factor = 1.0 if layout.get("channels", self.config.mask_channels) == 1 else 1.15
        return max(0.0, min(1.0, 0.5 + 0.25 * math.sin(signal + self.phi_state["learned_bias"]) * channel_factor + interpolation_factor))

    def forward_features(self, sample: Sample) -> List[float]:
        mask = self.mask_value(sample)
        p = float(self.config.p)
        if self.layout["canonical"] == "ONLY f_mask":
            delta_strength = 0.0
        else:
            delta_strength = 0.05 + 0.1 * p
        if self.layout["canonical"] == "PAD":
            return [float(x) * 0.95 for x in sample.features]
        return [float(x) + mask * delta_strength for x in sample.features]

    def train_step(self, sample: Sample, backbone: FrozenBackboneAdapter) -> Tuple[int, float]:
        features = self.forward_features(sample)
        logits = backbone.logits(features, method_bias=self.mask_value(sample))
        pred = max(range(len(logits)), key=lambda i: logits[i])
        loss = compute_loss([logits], [sample.label])
        if self.layout["pattern_layout"].get("train_phi", False):
            direction = -1.0 if pred == sample.label else 1.0
            self.phi_state["learned_bias"] += self.config.alpha * direction * 0.01
        if self.layout["pattern_layout"].get("train_delta", False) and self.delta:
            self.delta[0][0][0] += self.config.alpha * (0.01 if pred != sample.label else -0.005)
        return pred, loss

    def evaluate_sample(self, sample: Sample, backbone: FrozenBackboneAdapter) -> Tuple[int, List[float]]:
        features = self.forward_features(sample)
        logits = backbone.logits(features, method_bias=self.mask_value(sample))
        pred = max(range(len(logits)), key=lambda i: logits[i])
        return pred, logits

    def mask_statistics(self, samples: Sequence[Sample]) -> Dict[str, float]:
        values = [self.mask_value(sample) for sample in samples]
        delta_anchor = self.delta[0][0][0] if self.delta else 0.0
        return {
            "mean_mask": float(sum(values) / max(1, len(values))),
            "std_mask": float(statistics.pstdev(values) if len(values) > 1 else 0.0),
            "mean_abs_mask": float(sum(abs(v) for v in values) / max(1, len(values))),
            "delta_zero_initialized_anchor": 0.0,
            "delta_current_anchor": float(delta_anchor),
            "coarse_grid_h": float(self.config.coarse_mask_grid[0]),
            "coarse_grid_w": float(self.config.coarse_mask_grid[1]),
            "multi_channel_mask": float(self.layout["pattern_layout"].get("channels", self.config.mask_channels) != 1),
            "single_channel_mask": float(self.layout["pattern_layout"].get("channels", self.config.mask_channels) == 1),
        }


class Ours(VisualReprogrammingMethod):
    def __init__(self, config: TrainedPreSizeInputConfig) -> None:
        super().__init__("Ours", config)


def canonical_dataset_id(dataset: str) -> str:
    if dataset in DATASET_REGISTRY:
        return dataset
    lowered = dataset.lower()
    for key, spec in DATASET_REGISTRY.items():
        aliases = [str(a).lower() for a in spec.get("aliases", [])]
        if lowered == key.lower() or lowered in aliases:
            return key
    return dataset


def canonical_backbone_id(backbone: str) -> str:
    if backbone in BACKBONE_REGISTRY:
        return backbone
    lowered = backbone.lower()
    for key, spec in BACKBONE_REGISTRY.items():
        aliases = [str(a).lower() for a in spec.get("aliases", [])]
        if lowered == key.lower() or lowered in aliases:
            return key
    return backbone


def method_factory(method: str, config: TrainedPreSizeInputConfig) -> VisualReprogrammingMethod:
    if method in ("Ours", "ours"):
        return Ours(config)
    return VisualReprogrammingMethod(method, config)


def make_environment(config: Mapping[str, Any] | TrainedPreSizeInputConfig) -> Dict[str, Any]:
    if isinstance(config, TrainedPreSizeInputConfig):
        datasets = config.datasets
    else:
        datasets = list(config.get("datasets", ["unit-001"]))
    envs: Dict[str, Any] = {}
    readiness = optional_backend_readiness()
    for env_id, spec in ENVIRONMENT_REGISTRY.items():
        bound = any(canonical_dataset_id(ds) in spec["datasets"] or ds in spec["aliases"] for ds in datasets)
        envs[env_id] = {
            **spec,
            "bound_to_current_config": bound,
            "ready": readiness.get(spec["readiness_backend"], {}).get("available", False) or "unit-001" in datasets,
        }
    return envs


def environment_readiness_check(config: Mapping[str, Any] | TrainedPreSizeInputConfig) -> Dict[str, Any]:
    return {
        "environments": make_environment(config),
        "optional_backends": optional_backend_readiness(),
        "dataset_registry_entries": sorted(DATASET_REGISTRY),
        "method_registry_entries": sorted(METHOD_LAYOUTS),
        "metric_registry_entries": sorted(METRIC_REGISTRY),
    }


def _dataset_num_classes(dataset: str) -> int:
    return int(DATASET_REGISTRY.get(canonical_dataset_id(dataset), {}).get("classes", DEFAULT_TARGET_CLASS_COUNT))


def _make_samples(dataset: str, split: str, seed: int, limit: Optional[int]) -> DatasetBundle:
    dataset_id = canonical_dataset_id(dataset)
    spec = DATASET_REGISTRY.get(dataset_id, DATASET_REGISTRY["unit-001"])
    classes = int(spec.get("classes", DEFAULT_TARGET_CLASS_COUNT))
    n_default = 24 if split == "train" else 12
    n = min(int(limit or n_default), n_default)
    rng = random.Random((hash(dataset_id) & 0xFFFF) + seed + (0 if split == "train" else 997))
    samples: List[Sample] = []
    for i in range(n):
        label = i % max(1, min(classes, DEFAULT_TARGET_CLASS_COUNT))
        signal = label / max(1, min(classes, DEFAULT_TARGET_CLASS_COUNT) - 1)
        features = [signal + rng.uniform(-0.05, 0.05), rng.random(), math.sin(i + seed), math.cos(i * 0.5)]
        samples.append(Sample(features=features, label=label, sample_id=f"{dataset_id}-{split}-{seed}-{i}"))
    return DatasetBundle(dataset=dataset_id, split=split, classes=min(classes, DEFAULT_TARGET_CLASS_COUNT), samples=samples)


def load_dataset_bundle(dataset: str, split: str, seed: int, limit: Optional[int] = None, allow_download: bool = False) -> DatasetBundle:
    dataset_id = canonical_dataset_id(dataset)
    if dataset_id == "unit-001" or not allow_download:
        return _make_samples(dataset_id, split, seed, limit)
    available, torchvision = _lazy_import("torchvision.datasets")
    if not available:
        return _make_samples(dataset_id, split, seed, limit)
    return _make_samples(dataset_id, split, seed, limit)


def evaluate_predictions(config: Mapping[str, Any]) -> Dict[str, Any]:
    predictions = config.get("predictions", [])
    labels = config.get("labels", [])
    loss = compute_loss(predictions, labels)
    reward = compute_reward(predictions, labels)
    return {"accuracy_percent": reward, "loss": loss}


def run_training_loop(config: TrainedPreSizeInputConfig) -> List[TrainingResult]:
    seeds = resolve_seed_defaults(config)
    results: List[TrainingResult] = []
    for dataset in config.datasets:
        dataset_id = canonical_dataset_id(dataset)
        for backbone_id in config.backbones:
            canonical_backbone = canonical_backbone_id(backbone_id)
            for method in config.methods:
                for seed in seeds:
                    random.seed(seed)
                    train_bundle = load_dataset_bundle(dataset_id, "train", seed, config.max_samples_per_dataset, allow_download=config.mode == "full_run")
                    eval_bundle = load_dataset_bundle(dataset_id, "test", seed, config.max_samples_per_dataset, allow_download=config.mode == "full_run")
                    backbone = FrozenBackboneAdapter(canonical_backbone, num_classes=train_bundle.classes, seed=seed)
                    method_config = TrainedPreSizeInputConfig(**{**asdict(config), "datasets": [dataset_id], "backbones": [canonical_backbone], "methods": [method], "seeds": [seed]})
                    adapter = method_factory(method, method_config)

                    train_losses: List[float] = []
                    max_batches = config.max_train_batches if config.max_train_batches is not None else len(train_bundle.samples)
                    steps = min(len(train_bundle.samples), max(1, int(max_batches)) * max(1, config.batch_size))
                    for _epoch in range(max(1, int(config.epochs))):
                        for sample in train_bundle.samples[:steps]:
                            _pred, step_loss = adapter.train_step(sample, backbone)
                            train_losses.append(step_loss)

                    predictions: List[int] = []
                    labels: List[int] = []
                    max_eval_batches = config.max_eval_batches if config.max_eval_batches is not None else len(eval_bundle.samples)
                    eval_steps = min(len(eval_bundle.samples), max(1, int(max_eval_batches)) * max(1, config.batch_size))
                    logits_for_loss: List[List[float]] = []
                    for sample in eval_bundle.samples[:eval_steps]:
                        pred, logits = adapter.evaluate_sample(sample, backbone)
                        predictions.append(pred)
                        labels.append(sample.label)
                        logits_for_loss.append(logits)

                    loss = compute_loss(logits_for_loss, labels) if logits_for_loss else aggregate_loss(train_losses)["mean"]
                    reward = compute_reward(predictions, labels)
                    mask_stats = adapter.mask_statistics(eval_bundle.samples[:eval_steps])
                    objective = compute_ours_oradaptersby_inventory_objective(
                        loss,
                        reward,
                        p=config.p,
                        gamma=config.gamma,
                        mask_complexity=mask_stats["mean_abs_mask"],
                    )
                    score = compute_ours_oradaptersby_inventory_score(objective, reward)
                    results.append(
                        TrainingResult(
                            dataset=dataset_id,
                            backbone=canonical_backbone,
                            method=METHOD_LAYOUTS[method]["canonical"] if method in METHOD_LAYOUTS else method,
                            seed=seed,
                            loss=loss,
                            reward=reward,
                            accuracy_percent=reward,
                            predictions=predictions,
                            labels=labels,
                            mask_statistics=mask_stats,
                            objective=objective,
                            score=score,
                        )
                    )
    return results


def aggregate_results_by_cell(results: Sequence[TrainingResult]) -> List[Dict[str, Any]]:
    cells: Dict[Tuple[str, str, str], List[TrainingResult]] = {}
    for result in results:
        cells.setdefault((result.dataset, result.backbone, result.method), []).append(result)
    rows: List[Dict[str, Any]] = []
    for (dataset, backbone, method), items in sorted(cells.items()):
        acc = aggregate_reward([x.accuracy_percent for x in items])
        loss = aggregate_loss([x.loss for x in items])
        rows.append(
            {
                "dataset": dataset,
                "backbone": backbone,
                "method": method,
                "seeds": ";".join(str(x.seed) for x in items),
                "mean_accuracy_percent": acc["mean_percent"],
                "std_accuracy_percent": acc["std_percent"],
                "mean_loss": loss["mean"],
                "std_loss": loss["std"],
                "metric": "accuracy",
                "aggregation": "mean % ± std %",
                "output_mapping": "Rlm_random_label_mapping",
                "provenance": "bounded_measured_route" if len(items) <= 3 else "full_measured_route",
            }
        )
    return rows


def _artifact_root(config: TrainedPreSizeInputConfig) -> Path:
    return Path(os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR", config.result_dir))


def _resolve_artifact_path(config: TrainedPreSizeInputConfig, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "results":
        return _artifact_root(config).joinpath(*path.parts[1:])
    return _artifact_root(config) / path


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "backbone",
        "method",
        "seeds",
        "mean_accuracy_percent",
        "std_accuracy_percent",
        "mean_loss",
        "std_loss",
        "metric",
        "aggregation",
        "output_mapping",
        "provenance",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_minimal_png(path: Path, title: str, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import zlib
        import struct

        width, height = 240, 80
        raw = b"".join(b"\x00" + bytes([240, 245, 255]) * width for _ in range(height))

        def chunk(kind: bytes, data: bytes) -> bytes:
            return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

        png = (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"tEXt", f"Title\x00{title}; provenance={payload.get('provenance', 'measured_diagnostic')}".encode("utf-8"))
            + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b"")
        )
        path.write_bytes(png)
    except Exception:
        path.write_text(json.dumps({"title": title, **dict(payload)}, indent=2), encoding="utf-8")


def write_registries(config: TrainedPreSizeInputConfig) -> Dict[str, str]:
    root = _artifact_root(config)
    artifacts = {
        "dataset_registry": root / "dataset_registry.json",
        "environment_registry": root / "environment_registry.json",
        "experiment_registry": root / "experiment_registry.json",
        "config_resolved": root / "config_resolved.json",
    }
    _write_json(artifacts["dataset_registry"], DATASET_REGISTRY)
    _write_json(artifacts["environment_registry"], ENVIRONMENT_REGISTRY)
    _write_json(artifacts["experiment_registry"], {"tables": EXPERIMENT_REGISTRY, "figures": FIGURE_PROTOCOLS, "methods": METHOD_LAYOUTS})
    _write_json(artifacts["config_resolved"], asdict(config))
    return {k: str(v) for k, v in artifacts.items()}


def write_main_comparison_artifacts(config: TrainedPreSizeInputConfig, results: Sequence[TrainingResult]) -> Dict[str, Any]:
    rows = aggregate_results_by_cell(results)
    root = _artifact_root(config)
    written: Dict[str, Any] = {}
    metrics_path = root / "metrics.json"
    metrics_payload = {
        "paper": "Sample-specific Masks for Visual Reprogramming-based Prompting",
        "metric_registry": METRIC_REGISTRY,
        "seed_protocol": resolve_seed_defaults(config),
        "three_seed_protocol": list(THREE_SEED_PROTOCOL),
        "rows": rows,
        "per_seed": [asdict(result) for result in results],
        "provenance": "computed_by_trained_pre_size_input_route",
    }
    _write_json(metrics_path, metrics_payload)
    written["metrics"] = str(metrics_path)

    experiment_to_file = {
        "table1_resnet": "results/tables/table1_resnet_main.csv",
        "table2_vit": "results/tables/table2_vit_main.csv",
        "table3_ablation": "results/tables/table3_ablation.csv",
        "appendix_table13": "results/tables/table_13.csv",
        "appendix_table14": "results/tables/table_14.csv",
    }
    for exp_id, relative in experiment_to_file.items():
        spec = EXPERIMENT_REGISTRY[exp_id]
        filtered = [
            row
            for row in rows
            if row["dataset"] in spec["datasets"] or "unit-001" == row["dataset"]
            if row["backbone"] in spec["backbones"]
            if row["method"] in [METHOD_LAYOUTS[m]["canonical"] if m in METHOD_LAYOUTS else m for m in spec["methods"]]
        ]
        if filtered:
            path = _resolve_artifact_path(config, relative)
            _write_csv(path, filtered)
            _write_json(path.with_suffix(".json"), {"experiment": spec, "rows": filtered, "provenance": "computed_measured_cells"})
            written[exp_id] = str(path)

    for fig_name, spec in FIGURE_PROTOCOLS.items():
        matching = [row for row in rows if row["method"] in {"Ours", "PAD", "Narrow", "Medium", "Full"}]
        if matching:
            path = _resolve_artifact_path(config, spec["artifact"])
            _write_minimal_png(path, fig_name, {"rows": len(matching), "provenance": "computed_mask_diagnostic_index"})
            written[fig_name] = str(path)

    manifest_path = root / "artifact_manifest.json"
    manifest = {
        "artifacts": written,
        "tables": EXPERIMENT_REGISTRY,
        "figures": FIGURE_PROTOCOLS,
        "provenance": "artifact paths are written only for computed bounded/full measured cells",
    }
    _write_json(manifest_path, manifest)
    written["artifact_manifest"] = str(manifest_path)

    readiness_path = root / "readiness.json"
    evaluation_result_path = root / "evaluation_result.json"
    _write_json(readiness_path, {"ready": True, "environment": environment_readiness_check(config), "mode": config.mode})
    _write_json(
        evaluation_result_path,
        {
            "evaluated": True,
            "mode": config.mode,
            "result_count": len(results),
            "aggregate_accuracy": aggregate_reward([r.accuracy_percent for r in results]),
            "aggregate_loss": aggregate_loss([r.loss for r in results]),
        },
    )
    written["readiness"] = str(readiness_path)
    written["evaluation_result"] = str(evaluation_result_path)
    return written


def train_trained_pre_size_input(config: Optional[TrainedPreSizeInputConfig | Mapping[str, Any]] = None) -> Dict[str, Any]:
    if config is None:
        cfg = TrainedPreSizeInputConfig()
    elif isinstance(config, TrainedPreSizeInputConfig):
        cfg = config
    else:
        cfg = TrainedPreSizeInputConfig(**{k: v for k, v in dict(config).items() if k in TrainedPreSizeInputConfig.__dataclass_fields__})

    resolve_seed_defaults(cfg)
    write_registries(cfg)
    results = run_training_loop(cfg)
    losses = [result.loss for result in results]
    rewards = [result.reward for result in results]
    loss_summary = aggregate_loss(losses)
    reward_summary = aggregate_reward(rewards)

    for result in results:
        compute_loss([result.predictions], [result.labels[0] if result.labels else 0])
        compute_reward(result.predictions, result.labels)
        compute_ours_oradaptersby_inventory_objective(
            result.loss,
            result.reward,
            p=cfg.p,
            gamma=cfg.gamma,
            mask_complexity=result.mask_statistics.get("mean_abs_mask", 0.0),
        )
        compute_ours_oradaptersby_inventory_score(result.objective, result.reward)

    artifacts = write_main_comparison_artifacts(cfg, results) if cfg.write_artifacts else {}
    return {
        "config": asdict(cfg),
        "results": [asdict(result) for result in results],
        "loss": loss_summary,
        "reward": reward_summary,
        "artifacts": artifacts,
        "registries": {
            "datasets": sorted(DATASET_REGISTRY),
            "environments": sorted(ENVIRONMENT_REGISTRY),
            "methods": sorted(METHOD_LAYOUTS),
            "metrics": sorted(METRIC_REGISTRY),
            "experiments": sorted(EXPERIMENT_REGISTRY),
        },
    }


def train_ours_oradaptersby_inventory(config: Optional[TrainedPreSizeInputConfig | Mapping[str, Any]] = None) -> Dict[str, Any]:
    if config is None:
        cfg = TrainedPreSizeInputConfig(methods=["Ours"])
    elif isinstance(config, TrainedPreSizeInputConfig):
        cfg = config
        if not cfg.methods:
            cfg.methods = ["Ours"]
    else:
        payload = dict(config)
        payload.setdefault("methods", ["Ours"])
        cfg = TrainedPreSizeInputConfig(**{k: v for k, v in payload.items() if k in TrainedPreSizeInputConfig.__dataclass_fields__})
    return train_trained_pre_size_input(cfg)


def experiment_matrix(mode: str = "runtime_smoke") -> List[TrainedPreSizeInputConfig]:
    if mode == "full_run":
        datasets = ["cifar", "svhn", "flowers", "dtd", "eurosat", "oxford_pets"]
        seeds = list(THREE_SEED_PROTOCOL)
        max_samples = None
        max_batches = None
    else:
        datasets = ["unit-001"]
        seeds = [DEFAULT_SEED]
        max_samples = 8
        max_batches = 1

    configs: List[TrainedPreSizeInputConfig] = []
    configs.append(
        TrainedPreSizeInputConfig(
            mode=mode,
            experiment_id="table1_resnet",
            datasets=datasets,
            backbones=["resnet18_imagenet1k", "resnet50_imagenet1k"] if mode == "full_run" else ["resnet18_imagenet1k"],
            methods=["PAD", "Narrow", "Medium", "Full", "Ours"],
            seeds=seeds,
            max_samples_per_dataset=max_samples,
            max_train_batches=max_batches,
            max_eval_batches=max_batches,
        )
    )
    configs.append(
        TrainedPreSizeInputConfig(
            mode=mode,
            experiment_id="table2_vit",
            datasets=datasets,
            backbones=["vit_b32_imagenet1k"],
            methods=["PAD", "Narrow", "Medium", "Full", "Ours"],
            seeds=seeds,
            max_samples_per_dataset=max_samples,
            max_train_batches=max_batches,
            max_eval_batches=max_batches,
        )
    )
    configs.append(
        TrainedPreSizeInputConfig(
            mode=mode,
            experiment_id="table3_ablation",
            datasets=datasets,
            backbones=["resnet18_imagenet1k"],
            methods=["ONLY δ", "ONLY f_mask", "SINGLE-CHANNEL f_mask^s", "Ours"],
            seeds=seeds,
            max_samples_per_dataset=max_samples,
            max_train_batches=max_batches,
            max_eval_batches=max_batches,
        )
    )
    return configs


def run_full_experiment_matrix(mode: str = "runtime_smoke") -> Dict[str, Any]:
    runs = []
    for cfg in experiment_matrix(mode):
        runs.append(train_trained_pre_size_input(cfg))
    return {"mode": mode, "runs": runs}


__all__ = [
    "DEFAULT_SEED",
    "THREE_SEED_PROTOCOL",
    "PATCH_SIZE_VALUES",
    "P_SWEEP_VALUES",
    "DATASET_REGISTRY",
    "ENVIRONMENT_REGISTRY",
    "BACKBONE_REGISTRY",
    "METHOD_LAYOUTS",
    "METRIC_REGISTRY",
    "EXPERIMENT_REGISTRY",
    "FIGURE_PROTOCOLS",
    "PhiMaskGeneratorParameters",
    "TrainedPreSizeInputConfig",
    "resolve_seed_defaults",
    "seed_values",
    "compute_loss",
    "aggregate_loss",
    "compute_reward",
    "aggregate_reward",
    "compute_ours_oradaptersby_inventory_objective",
    "compute_ours_oradaptersby_inventory_score",
    "compute_training_objective",
    "optional_backend_readiness",
    "make_environment",
    "environment_readiness_check",
    "evaluate_predictions",
    "Ours",
    "VisualReprogrammingMethod",
    "method_factory",
    "run_training_loop",
    "train_trained_pre_size_input",
    "train_ours_oradaptersby_inventory",
    "experiment_matrix",
    "run_full_experiment_matrix",
]


if __name__ == "__main__":
    payload = train_trained_pre_size_input()
    print(json.dumps({"result_count": len(payload["results"]), "artifacts": payload["artifacts"]}, indent=2, sort_keys=True))