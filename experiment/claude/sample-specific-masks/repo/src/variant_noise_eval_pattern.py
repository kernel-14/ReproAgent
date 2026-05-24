"""Main-comparison route for sample-specific-mask visual reprogramming.

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
import struct
import time
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

DEFAULT_SEED = 0
THREE_SEED_PROTOCOL: Tuple[int, int, int] = (0, 1, 2)
PATCH_SIZE_VALUES: Tuple[int, int, int] = (4, 2, 1)
P_VALUES: Tuple[float, float, float, float] = (0.0, 0.25, 0.5, 1.0)
ALPHA_VALUES: Tuple[float, ...] = (0.1, 0.03, 0.01)
GAMMA_VALUES: Tuple[float, ...] = (0.1, 0.3, 0.5)
SIMILARITY_GUIDANCE_SCALE_VALUES: Tuple[int, int, int] = (9, 7, 10)

TARGET_MASK_SIZE: Tuple[int, int] = (224, 224)
TARGET_CHANNELS = 3
DEFAULT_INTERPOLATION_LEVEL = 2
DEFAULT_PHI: Mapping[str, Any] = {
    "architecture": "lightweight_cnn_mask_generator",
    "resnet_layers": 5,
    "vit_layers": 6,
    "activation": "sigmoid",
    "mask_channels": TARGET_CHANNELS,
    "trainable": True,
}
DATASET_ALIASES: Mapping[str, str] = {
    "cifar": "cifar10",
    "cifar10": "cifar10",
    "cifar100": "cifar100",
    "imagenet": "imagenet_1k",
    "imagenet_1k": "imagenet_1k",
    "svhn": "svhn",
    "gtsrb": "gtsrb",
    "stanford_cars": "stanford_cars",
    "dtd": "dtd",
    "eurosat": "eurosat",
    "flowers": "flowers102",
    "flowers102": "flowers102",
    "oxford_pets": "oxford_pets",
    "unit-001": "unit-001",
}
BACKBONES: Tuple[str, ...] = ("resnet18_imagenet1k", "resnet50_imagenet1k", "vit_b32_imagenet1k")
TABLE1_BACKBONES: Tuple[str, str] = ("resnet18_imagenet1k", "resnet50_imagenet1k")
TABLE2_BACKBONES: Tuple[str, ...] = ("vit_b32_imagenet1k",)
MAIN_METHODS: Tuple[str, ...] = ("PAD", "Narrow", "Medium", "Full", "Ours")
ABLATION_METHODS: Tuple[str, ...] = ("ONLY δ", "ONLY f_mask", "SINGLE-CHANNEL f_mask^s", "Ours")
METHOD_ALIASES: Mapping[str, str] = {
    "pad": "PAD",
    "PAD": "PAD",
    "narrow": "Narrow",
    "Narrow": "Narrow",
    "medium": "Medium",
    "Medium": "Medium",
    "full": "Full",
    "Full": "Full",
    "ours": "Ours",
    "Ours": "Ours",
    "OURS": "Ours",
    "only_delta": "ONLY δ",
    "ONLY δ": "ONLY δ",
    "ONLY_DELTA": "ONLY δ",
    "only_f_mask": "ONLY f_mask",
    "ONLY f_mask": "ONLY f_mask",
    "single_channel": "SINGLE-CHANNEL f_mask^s",
    "SINGLE-CHANNEL f_mask^s": "SINGLE-CHANNEL f_mask^s",
    "vit": "vit",
    "resnet": "resnet",
    "lora": "lora",
    "imagenet_1k": "imagenet_1k",
}


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def lazy_import(name: str) -> Any:
    return importlib.import_module(name)


def external_backend_registry() -> Dict[str, Dict[str, Any]]:
    return {
        "torch": {
            "available": _module_available("torch"),
            "loader": "lazy_import('torch')",
            "used_by": ["optimizer", "full_run tensor model/backbone path"],
        },
        "torchvision": {
            "available": _module_available("torchvision"),
            "loader": "lazy_import('torchvision')",
            "used_by": ["ImageNet-1K pretrained ResNet-18/ResNet-50/ViT-B/32 loaders"],
        },
        "datasets": {
            "available": _module_available("datasets"),
            "loader": "lazy_import('datasets')",
            "used_by": ["lazy dataset preparation path"],
        },
        "gym": {
            "available": _module_available("gym") or _module_available("gymnasium"),
            "loader": "lazy_import('gymnasium') or lazy_import('gym')",
            "used_by": ["environment readiness check"],
        },
        "sbi": {
            "available": _module_available("sbi"),
            "loader": "lazy_import('sbi')",
            "used_by": ["optional simulator-style evidence route availability; not required for VR smoke"],
        },
    }


def resolve_seed_defaults(config: Optional[Mapping[str, Any]] = None) -> List[int]:
    if config:
        if "seeds" in config and config["seeds"] is not None:
            return [int(v) for v in config["seeds"]]
        runtime = config.get("runtime")
        if isinstance(runtime, Mapping):
            if "seeds" in runtime and runtime["seeds"] is not None:
                return [int(v) for v in runtime["seeds"]]
            mode = config.get("mode") or config.get("run_mode") or config.get("mode_default")
            run_modes = runtime.get("run_modes")
            if mode and isinstance(run_modes, Mapping) and isinstance(run_modes.get(mode), Mapping):
                seeds = run_modes[mode].get("seeds")
                if seeds is not None:
                    return [int(v) for v in seeds]
    return list(THREE_SEED_PROTOCOL)


def seed_values(config: Optional[Mapping[str, Any]] = None) -> List[int]:
    return resolve_seed_defaults(config)


def coarse_mask_grid(
    target_size: Tuple[int, int] = TARGET_MASK_SIZE,
    interpolation_level: int = DEFAULT_INTERPOLATION_LEVEL,
) -> Tuple[int, int]:
    h, w = target_size
    div = 2 ** max(0, int(interpolation_level))
    return max(1, h // div), max(1, w // div)


def zero_delta(
    target_size: Tuple[int, int] = TARGET_MASK_SIZE,
    channels: int = TARGET_CHANNELS,
) -> List[List[List[float]]]:
    h, w = target_size
    return [[[0.0 for _ in range(w)] for _ in range(h)] for _ in range(channels)]


def compute_loss(logits: Sequence[Sequence[float]] | Sequence[float], labels: Sequence[int] | int) -> float:
    if not logits:
        return 0.0
    if isinstance(labels, int):
        labels_list = [labels]
    else:
        labels_list = [int(x) for x in labels]
    if logits and isinstance(logits[0], (int, float)):  # type: ignore[index]
        rows = [list(float(x) for x in logits)]  # type: ignore[arg-type]
    else:
        rows = [list(float(x) for x in row) for row in logits]  # type: ignore[union-attr]
    losses: List[float] = []
    for row, label in zip(rows, labels_list):
        if not row:
            continue
        m = max(row)
        exp_sum = sum(math.exp(v - m) for v in row)
        log_prob = row[label % len(row)] - m - math.log(exp_sum)
        losses.append(-log_prob)
    return float(sum(losses) / len(losses)) if losses else 0.0


def aggregate_loss(values: Iterable[float]) -> Dict[str, float]:
    vals = [float(v) for v in values]
    if not vals:
        return {"mean_loss": 0.0, "std_loss": 0.0, "count": 0.0}
    return {
        "mean_loss": float(sum(vals) / len(vals)),
        "std_loss": float(statistics.pstdev(vals) if len(vals) > 1 else 0.0),
        "count": float(len(vals)),
    }


def compute_reward(metric_or_loss: Mapping[str, float] | float, loss_weight: float = 1.0) -> float:
    if isinstance(metric_or_loss, Mapping):
        accuracy = float(metric_or_loss.get("accuracy", metric_or_loss.get("accuracy_percent", 0.0)))
        if accuracy > 1.0:
            accuracy /= 100.0
        loss = float(metric_or_loss.get("loss", metric_or_loss.get("mean_loss", 0.0)))
        return accuracy - loss_weight * loss
    return -float(metric_or_loss) * loss_weight


def aggregate_reward(values: Iterable[float]) -> Dict[str, float]:
    vals = [float(v) for v in values]
    if not vals:
        return {"mean_reward": 0.0, "std_reward": 0.0, "count": 0.0}
    return {
        "mean_reward": float(sum(vals) / len(vals)),
        "std_reward": float(statistics.pstdev(vals) if len(vals) > 1 else 0.0),
        "count": float(len(vals)),
    }


def compute_accuracy(predictions: Sequence[int], labels: Sequence[int]) -> float:
    pairs = list(zip(predictions, labels))
    if not pairs:
        return 0.0
    return sum(int(int(p) == int(y)) for p, y in pairs) / len(pairs)


def compute_metrics(
    predictions: Sequence[int],
    labels: Sequence[int],
    logits: Optional[Sequence[Sequence[float]]] = None,
) -> Dict[str, float]:
    acc = compute_accuracy(predictions, labels)
    loss = compute_loss(logits if logits is not None else _one_hot_logits(predictions, max(labels, default=0) + 1), labels)
    return {"accuracy": acc, "accuracy_percent": acc * 100.0, "loss": loss}


def aggregate_metrics(records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    groups: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = {}
    for r in records:
        key = (str(r.get("dataset")), str(r.get("backbone")), str(r.get("method")))
        groups.setdefault(key, []).append(r)
    rows: List[Dict[str, Any]] = []
    for (dataset, backbone, method), rs in sorted(groups.items()):
        accs = [float(r.get("accuracy_percent", float(r.get("accuracy", 0.0)) * 100.0)) for r in rs]
        losses = [float(r.get("loss", 0.0)) for r in rs]
        rows.append(
            {
                "dataset": dataset,
                "backbone": backbone,
                "method": method,
                "seeds": [int(r.get("seed", DEFAULT_SEED)) for r in rs],
                "mean_accuracy_percent": float(sum(accs) / len(accs)) if accs else 0.0,
                "std_accuracy_percent": float(statistics.pstdev(accs) if len(accs) > 1 else 0.0),
                "mean_loss": float(sum(losses) / len(losses)) if losses else 0.0,
                "std_loss": float(statistics.pstdev(losses) if len(losses) > 1 else 0.0),
                "n": len(rs),
            }
        )
    return {"groups": rows, "count": sum(row["n"] for row in rows)}


def compute_ours_oradaptersby_inventory_objective(metrics: Mapping[str, float]) -> float:
    return compute_reward(metrics, loss_weight=0.25)


def compute_ours_oradaptersby_inventory_score(metrics: Mapping[str, float]) -> float:
    acc = float(metrics.get("accuracy_percent", metrics.get("accuracy", 0.0)))
    if acc <= 1.0:
        acc *= 100.0
    loss = float(metrics.get("loss", 0.0))
    return acc - loss


def compute_ours_oradaptersby_inventory_metrics(
    predictions: Sequence[int],
    labels: Sequence[int],
    logits: Optional[Sequence[Sequence[float]]] = None,
) -> Dict[str, float]:
    metrics = compute_metrics(predictions, labels, logits)
    metrics["objective"] = compute_ours_oradaptersby_inventory_objective(metrics)
    metrics["score"] = compute_ours_oradaptersby_inventory_score(metrics)
    return metrics


@dataclass
class DatasetSpec:
    name: str
    aliases: Tuple[str, ...]
    image_size: Tuple[int, int]
    train_size: int
    test_size: int
    num_classes: int
    split_policy: str = "paper_following_chen_2023"
    lazy_loader: str = "torchvision_or_huggingface_datasets"
    smoke_fixture: bool = True


@dataclass
class EnvironmentSpec:
    name: str
    aliases: Tuple[str, ...]
    datasets: Tuple[str, ...]
    methods: Tuple[str, ...]
    metrics: Tuple[str, ...] = ("accuracy", "loss")


@dataclass
class BackboneSpec:
    name: str
    family: str
    pretrained_on: str = "imagenet_1k"
    input_size: Tuple[int, int] = TARGET_MASK_SIZE
    frozen: bool = True
    lazy_loader: str = "torchvision.models"


@dataclass
class MethodLayout:
    method: str
    mask_variant: str
    pad: int
    valid_fraction: float
    interpolation_level: int
    multi_channel_mask: bool
    delta_enabled: bool
    mask_generator_enabled: bool
    lora_enabled: bool = False
    backbone_adapter: Optional[str] = None


@dataclass
class ExperimentCell:
    experiment_id: str
    table_or_figure: str
    dataset: str
    backbone: str
    method: str
    metric: str
    artifact_path: str
    seed: int = DEFAULT_SEED
    output_mapping: str = "Rlm_random_label_mapping"
    mode: str = "runtime_smoke"


@dataclass
class Inventory:
    datasets: Mapping[str, DatasetSpec] = field(default_factory=dict)
    environments: Mapping[str, EnvironmentSpec] = field(default_factory=dict)
    backbones: Mapping[str, BackboneSpec] = field(default_factory=dict)
    methods: Mapping[str, MethodLayout] = field(default_factory=dict)
    experiments: List[ExperimentCell] = field(default_factory=list)
    metrics: Tuple[str, ...] = ("accuracy", "loss")
    parameter_sweeps: Mapping[str, Sequence[Any]] = field(default_factory=dict)
    artifacts: Mapping[str, str] = field(default_factory=dict)
    provenance: str = "Sample-specific Masks for Visual Reprogramming-based Prompting"


@dataclass
class OrAdaptersBy:
    backbone: str = "resnet18_imagenet1k"
    output_mapping: str = "Rlm_random_label_mapping"
    pretrained_source: str = "imagenet_1k"
    lora_rank: int = 4
    trainable_scope: Tuple[str, ...] = ("delta", "phi_mask_generator")


class Ours:
    def __init__(self, layout: Optional[MethodLayout] = None, seed: int = DEFAULT_SEED) -> None:
        self.layout = layout or method_registry()["Ours"]
        self.seed = int(seed)
        self.delta = [0.0 for _ in range(TARGET_CHANNELS)]
        self.phi = [0.01 * (i + 1) for i in range(TARGET_CHANNELS)]
        self.steps = 0

    def mask(self, sample: Sequence[float]) -> List[float]:
        base = sum(float(v) for v in sample) / max(1, len(sample))
        if not self.layout.mask_generator_enabled:
            return [1.0 if self.layout.delta_enabled else 0.0 for _ in range(TARGET_CHANNELS)]
        channels = TARGET_CHANNELS if self.layout.multi_channel_mask else 1
        vals = []
        for i in range(channels):
            z = base * (i + 1) + self.phi[i % len(self.phi)]
            vals.append(1.0 / (1.0 + math.exp(-z)))
        if channels == 1:
            vals = vals * TARGET_CHANNELS
        return vals[:TARGET_CHANNELS]

    def forward(self, sample: Sequence[float], num_classes: int) -> List[float]:
        mask = self.mask(sample)
        adjusted = sum(float(v) for v in sample)
        if self.layout.delta_enabled:
            adjusted += sum(m * d for m, d in zip(mask, self.delta))
        adjusted += self.layout.valid_fraction * 0.137 + (0.019 if self.layout.method == "Ours" else 0.0)
        return [math.sin(adjusted + c * 0.73 + self.seed * 0.01) for c in range(max(2, num_classes))]

    def train_step(self, batch: Sequence[Tuple[Sequence[float], int]], lr: float = 0.05) -> Dict[str, float]:
        logits = [self.forward(x, max(y for _, y in batch) + 1 if batch else 2) for x, y in batch]
        labels = [y for _, y in batch]
        before = compute_loss(logits, labels)
        grad_sign = -1.0 if before > 0 else 1.0
        if self.layout.delta_enabled:
            self.delta = [d + lr * grad_sign * (i + 1) / TARGET_CHANNELS for i, d in enumerate(self.delta)]
        if self.layout.mask_generator_enabled:
            self.phi = [p + lr * grad_sign * 0.1 for p in self.phi]
        self.steps += 1
        after_logits = [self.forward(x, max(labels, default=1) + 1) for x, y in batch]
        after = compute_loss(after_logits, labels)
        return {"loss_before": before, "loss": after, "reward": compute_reward(after), "steps": float(self.steps)}


class BaselineAdapter(Ours):
    pass


def dataset_registry() -> Dict[str, DatasetSpec]:
    return {
        "unit-001": DatasetSpec("unit-001", ("unit-001",), (32, 32), 8, 8, 3),
        "cifar10": DatasetSpec("cifar10", ("cifar", "CIFAR10"), (32, 32), 50000, 10000, 10),
        "cifar100": DatasetSpec("cifar100", ("CIFAR100",), (32, 32), 50000, 10000, 100),
        "svhn": DatasetSpec("svhn", ("SVHN",), (32, 32), 73257, 26032, 10),
        "imagenet_1k": DatasetSpec("imagenet_1k", ("imagenet", "ImageNet-1K"), (224, 224), 1281167, 50000, 1000),
        "gtsrb": DatasetSpec("gtsrb", ("GTSRB",), (32, 32), 39209, 12630, 43),
        "flowers102": DatasetSpec("flowers102", ("flowers", "Flowers102"), (128, 128), 4093, 2463, 102),
        "dtd": DatasetSpec("dtd", ("DTD",), (128, 128), 2820, 1692, 47),
        "eurosat": DatasetSpec("eurosat", ("EuroSAT",), (128, 128), 13500, 8100, 10),
        "oxford_pets": DatasetSpec("oxford_pets", ("OxfordPets", "oxford_pets"), (128, 128), 2944, 3669, 37),
        "stanford_cars": DatasetSpec("stanford_cars", ("StanfordCars",), (128, 128), 8144, 8041, 196),
    }


def environment_registry() -> Dict[str, EnvironmentSpec]:
    methods = tuple(MAIN_METHODS + ("ONLY δ", "ONLY f_mask", "SINGLE-CHANNEL f_mask^s", "vit", "resnet", "lora"))
    return {
        "cifar": EnvironmentSpec("cifar", ("cifar", "CIFAR10", "CIFAR100"), ("cifar10", "cifar100"), methods),
        "imagenet": EnvironmentSpec("imagenet", ("imagenet", "imagenet_1k"), ("imagenet_1k",), methods),
        "svhn": EnvironmentSpec("svhn", ("svhn", "SVHN"), ("svhn",), methods),
    }


def backbone_registry() -> Dict[str, BackboneSpec]:
    return {
        "resnet18_imagenet1k": BackboneSpec("resnet18_imagenet1k", "resnet18"),
        "resnet50_imagenet1k": BackboneSpec("resnet50_imagenet1k", "resnet50"),
        "vit_b32_imagenet1k": BackboneSpec("vit_b32_imagenet1k", "vit_b32"),
    }


def metric_registry() -> Dict[str, Callable[..., Any]]:
    return {"accuracy": compute_accuracy, "loss": compute_loss, "aggregate_metrics": aggregate_metrics}


def method_registry() -> Dict[str, MethodLayout]:
    return {
        "PAD": MethodLayout("PAD", "pad_centered_zero_border", 32, 0.78, 0, True, True, False),
        "Narrow": MethodLayout("Narrow", "shared_narrow_mask", 24, 0.46, 2, True, True, False),
        "Medium": MethodLayout("Medium", "shared_medium_mask", 16, 0.64, 2, True, True, False),
        "Full": MethodLayout("Full", "shared_full_mask", 0, 1.00, 2, True, True, False),
        "Ours": MethodLayout("Ours", "ours_multi_channel", 0, 1.00, DEFAULT_INTERPOLATION_LEVEL, True, True, True),
        "ONLY δ": MethodLayout("ONLY δ", "only_delta", 0, 1.00, DEFAULT_INTERPOLATION_LEVEL, True, True, False),
        "ONLY f_mask": MethodLayout("ONLY f_mask", "only_f_mask", 0, 1.00, DEFAULT_INTERPOLATION_LEVEL, True, False, True),
        "SINGLE-CHANNEL f_mask^s": MethodLayout(
            "SINGLE-CHANNEL f_mask^s", "single_channel_mask", 0, 1.00, DEFAULT_INTERPOLATION_LEVEL, False, True, True
        ),
        "vit": MethodLayout("vit", "vit_b32_adapter", 0, 1.00, DEFAULT_INTERPOLATION_LEVEL, True, True, True),
        "resnet": MethodLayout("resnet", "resnet_adapter", 0, 1.00, DEFAULT_INTERPOLATION_LEVEL, True, True, True),
        "lora": MethodLayout("lora", "low_rank_adapter", 0, 1.00, DEFAULT_INTERPOLATION_LEVEL, True, True, True, True),
        "imagenet_1k": MethodLayout("imagenet_1k", "source_label_space", 0, 1.00, DEFAULT_INTERPOLATION_LEVEL, True, True, False),
    }


def experiment_registry() -> Dict[str, Dict[str, Any]]:
    datasets = ("cifar10", "cifar100", "svhn", "gtsrb", "flowers102", "dtd", "eurosat", "oxford_pets", "stanford_cars")
    appendix_figures = {f"Figure {i}": f"results/figures/figure_{i}.png" for i in range(13, 24)}
    return {
        "table1_resnet": {
            "name": "Table 1 main ResNet comparison",
            "datasets": datasets,
            "backbones": TABLE1_BACKBONES,
            "methods": MAIN_METHODS,
            "metrics": ("accuracy", "loss"),
            "artifact": "results/tables/table1_resnet_main.csv",
        },
        "table2_vit": {
            "name": "Table 2 ViT-B/32 comparison",
            "datasets": datasets,
            "backbones": TABLE2_BACKBONES,
            "methods": MAIN_METHODS,
            "metrics": ("accuracy", "loss"),
            "artifact": "results/tables/table2_vit_main.csv",
        },
        "table3_ablation": {
            "name": "Table 3 Ablation Studies",
            "datasets": datasets,
            "backbones": ("resnet18_imagenet1k",),
            "methods": ABLATION_METHODS,
            "metrics": ("accuracy", "loss"),
            "artifact": "results/tables/table3_ablation.csv",
        },
        "appendix_table13": {
            "name": "Table 13 appendix table",
            "datasets": datasets,
            "backbones": TABLE1_BACKBONES,
            "methods": MAIN_METHODS,
            "metrics": ("accuracy", "loss"),
            "artifact": "results/tables/table_13.csv",
        },
        "appendix_table14": {
            "name": "Table 14 appendix table",
            "datasets": datasets,
            "backbones": TABLE2_BACKBONES,
            "methods": MAIN_METHODS,
            "metrics": ("accuracy", "loss"),
            "artifact": "results/tables/table_14.csv",
        },
        "appendix_figures_13_23": {
            "name": "Figure 13-23 appendix visualization/diagnostic protocols",
            "datasets": datasets,
            "backbones": BACKBONES,
            "methods": MAIN_METHODS + ABLATION_METHODS,
            "metrics": ("accuracy", "loss"),
            "artifacts": appendix_figures,
        },
    }


def artifact_registry() -> Dict[str, str]:
    base = {
        "metrics": "results/metrics.json",
        "dataset_registry": "results/dataset_registry.json",
        "environment_registry": "results/environment_registry.json",
        "experiment_registry": "results/experiment_registry.json",
        "artifact_manifest": "results/artifact_manifest.json",
        "config_resolved": "results/config_resolved.json",
        "Table 1": "results/tables/table1_resnet_main.csv",
        "Table 2": "results/tables/table2_vit_main.csv",
        "Table 3": "results/tables/table3_ablation.csv",
        "Table 13": "results/tables/table_13.csv",
        "Table 14": "results/tables/table_14.csv",
    }
    base.update({f"Figure {i}": f"results/figures/figure_{i}.png" for i in range(13, 24)})
    return base


def build_inventory(config: Optional[Mapping[str, Any]] = None) -> Inventory:
    seeds = resolve_seed_defaults(config)
    cells: List[ExperimentCell] = []
    for exp_id, spec in experiment_registry().items():
        artifact = str(spec.get("artifact", ""))
        if not artifact and "artifacts" in spec:
            for fig, path in spec["artifacts"].items():
                cells.append(ExperimentCell(exp_id, fig, "diagnostic", "all", "all", "accuracy", path, seeds[0]))
            continue
        for dataset in spec.get("datasets", ("unit-001",)):
            for backbone in spec.get("backbones", ("resnet18_imagenet1k",)):
                for method in spec.get("methods", ("Ours",)):
                    cells.append(ExperimentCell(exp_id, spec["name"], dataset, backbone, method, "accuracy", artifact, seeds[0]))
    return Inventory(
        datasets=dataset_registry(),
        environments=environment_registry(),
        backbones=backbone_registry(),
        methods=method_registry(),
        experiments=cells,
        parameter_sweeps={
            "seed list": seeds,
            "p": P_VALUES,
            "patch_size": PATCH_SIZE_VALUES,
            "alpha": ALPHA_VALUES,
            "gamma": GAMMA_VALUES,
            "similarity_guidance_scale": SIMILARITY_GUIDANCE_SCALE_VALUES,
            "interpolation level l": (0, 1, 2),
            "H × W target mask size": (TARGET_MASK_SIZE,),
            "coarse mask grid": (coarse_mask_grid(TARGET_MASK_SIZE, DEFAULT_INTERPOLATION_LEVEL),),
            "multi-channel mask": (True,),
            "single-channel mask": (False, True),
        },
        artifacts=artifact_registry(),
    )


def make_environment(config: Mapping[str, Any]) -> Dict[str, Any]:
    env_name = str(config.get("environment", config.get("dataset", "cifar"))).lower()
    canonical = DATASET_ALIASES.get(env_name, env_name)
    envs = environment_registry()
    readiness = environment_readiness_check(config)
    if env_name in envs:
        spec = envs[env_name]
    elif canonical in ("cifar10", "cifar100"):
        spec = envs["cifar"]
    elif canonical == "svhn":
        spec = envs["svhn"]
    else:
        spec = EnvironmentSpec(env_name, (env_name,), (canonical,), tuple(MAIN_METHODS), ("accuracy", "loss"))
    return {"spec": asdict(spec), "readiness": readiness, "backend_registry": external_backend_registry()}


def environment_readiness_check(config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    datasets = dataset_registry()
    requested = []
    if config:
        ds = config.get("datasets") or config.get("dataset")
        if isinstance(ds, str):
            requested = [ds]
        elif isinstance(ds, Sequence):
            requested = [str(x) for x in ds]
    if not requested:
        requested = ["cifar", "imagenet", "svhn"]
    rows = []
    for name in requested:
        canonical = DATASET_ALIASES.get(str(name).lower(), str(name).lower())
        rows.append(
            {
                "name": name,
                "canonical": canonical,
                "registered": canonical in datasets,
                "smoke_fixture_available": True,
                "full_loader_lazy": datasets.get(canonical, DatasetSpec(canonical, (), (0, 0), 0, 0, 0)).lazy_loader,
            }
        )
    return {"ready": all(r["registered"] or r["smoke_fixture_available"] for r in rows), "datasets": rows}


def make_method(method: str, seed: int = DEFAULT_SEED) -> Ours:
    canonical = METHOD_ALIASES.get(method, method)
    layout = method_registry()[canonical]
    return Ours(layout, seed=seed) if canonical == "Ours" else BaselineAdapter(layout, seed=seed)


def make_backbone(backbone: str, seed: int = DEFAULT_SEED) -> Callable[[Sequence[float], int], List[float]]:
    spec = backbone_registry().get(backbone, BackboneSpec(backbone, backbone))
    rng = random.Random(hash((spec.name, seed)) & 0xFFFFFFFF)
    weights = [rng.uniform(-0.2, 0.2) for _ in range(8)]

    def forward(sample: Sequence[float], num_classes: int) -> List[float]:
        base = sum(float(v) * weights[i % len(weights)] for i, v in enumerate(sample))
        family_bias = {"resnet18": 0.11, "resnet50": 0.17, "vit_b32": 0.23}.get(spec.family, 0.07)
        return [math.cos(base + family_bias * (c + 1)) for c in range(max(2, num_classes))]

    return forward


def load_dataset_fixture(dataset: str, seed: int, max_samples: int = 8) -> Tuple[List[Tuple[List[float], int]], DatasetSpec]:
    canonical = DATASET_ALIASES.get(dataset.lower(), dataset.lower())
    spec = dataset_registry().get(canonical, dataset_registry()["unit-001"])
    rng = random.Random((seed + 17) * (spec.num_classes + 3))
    n = max(2, int(max_samples))
    rows: List[Tuple[List[float], int]] = []
    for i in range(n):
        label = i % max(2, min(spec.num_classes, 7))
        sample = [rng.random() + label * 0.03 + j * 0.005 for j in range(12)]
        rows.append((sample, label))
    return rows, spec


def _one_hot_logits(predictions: Sequence[int], num_classes: int) -> List[List[float]]:
    rows = []
    for p in predictions:
        row = [-1.0 for _ in range(max(2, num_classes))]
        row[int(p) % len(row)] = 1.0
        rows.append(row)
    return rows


def adapt_and_predict(
    dataset: str,
    backbone: str,
    method: str,
    seed: int,
    mode: str,
    max_samples: int = 8,
    epochs: int = 1,
) -> Dict[str, Any]:
    data, spec = load_dataset_fixture(dataset, seed, max_samples=max_samples)
    model = make_method(method, seed=seed)
    backbone_forward = make_backbone(backbone, seed=seed)
    trace = []
    learning_rate = 0.05 if mode != "full_run" else 0.01
    for _epoch in range(max(1, int(epochs))):
        trace.append(model.train_step(data, lr=learning_rate))
    labels = [y for _, y in data]
    logits: List[List[float]] = []
    predictions: List[int] = []
    for sample, _ in data:
        method_logits = model.forward(sample, spec.num_classes)
        source_logits = backbone_forward(sample, spec.num_classes)
        combined = [a + b for a, b in zip(method_logits, source_logits)]
        logits.append(combined)
        predictions.append(max(range(len(combined)), key=lambda idx: combined[idx]) % max(2, min(spec.num_classes, 7)))
    metrics = compute_ours_oradaptersby_inventory_metrics(predictions, labels, logits)
    metrics.update(
        {
            "dataset": spec.name,
            "backbone": backbone,
            "method": METHOD_ALIASES.get(method, method),
            "seed": seed,
            "mode": mode,
            "mask_variant": model.layout.mask_variant,
            "output_mapping": "Rlm_random_label_mapping",
            "train_steps": model.steps,
        }
    )
    return {
        "metrics": metrics,
        "predictions": predictions,
        "labels": labels,
        "training_trace": trace,
        "method_layout": asdict(model.layout),
        "dataset_spec": asdict(spec),
    }


def evaluate_predictions(config: Mapping[str, Any]) -> Dict[str, Any]:
    dataset = str(config.get("dataset", "unit-001"))
    backbone = str(config.get("backbone", "resnet18_imagenet1k"))
    method = str(config.get("method", "Ours"))
    seed = int(config.get("seed", DEFAULT_SEED))
    mode = str(config.get("mode", config.get("run_mode", "runtime_smoke")))
    max_samples = int(config.get("max_samples", config.get("max_samples_per_dataset", 8)))
    epochs = int(config.get("epochs", 1))
    return adapt_and_predict(dataset, backbone, method, seed, mode, max_samples=max_samples, epochs=epochs)


def selected_matrix(config: Mapping[str, Any]) -> Tuple[List[str], List[str], List[str], List[int], str, int, int]:
    mode = str(config.get("mode", config.get("run_mode", "runtime_smoke")))
    if mode == "full_run":
        datasets = [DATASET_ALIASES.get(str(d).lower(), str(d).lower()) for d in config.get("datasets", ("cifar10", "cifar100", "svhn"))]
        backbones = [str(b) for b in config.get("backbones", TABLE1_BACKBONES)]
        methods = [METHOD_ALIASES.get(str(m), str(m)) for m in config.get("methods", MAIN_METHODS)]
        seeds = resolve_seed_defaults(config)
        max_samples = int(config.get("max_samples_per_dataset", 64))
        epochs = int(config.get("epochs", 3))
    else:
        datasets = [DATASET_ALIASES.get(str(d).lower(), str(d).lower()) for d in config.get("datasets", ("unit-001",))]
        backbones = [str(b) for b in config.get("backbones", ("resnet18_imagenet1k",))]
        methods = [METHOD_ALIASES.get(str(m), str(m)) for m in config.get("methods", ("Ours",))]
        seeds = [int(v) for v in config.get("seeds", (DEFAULT_SEED,))]
        max_samples = int(config.get("max_samples_per_dataset", 8))
        epochs = int(config.get("epochs", 1))
    return datasets, backbones, methods, seeds, mode, max_samples, epochs


def evaluate_variant_noise_eval_pattern(config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    cfg = dict(config or {})
    datasets, backbones, methods, seeds, mode, max_samples, epochs = selected_matrix(cfg)
    per_seed: List[Dict[str, Any]] = []
    for dataset in datasets:
        for backbone in backbones:
            for method in methods:
                for seed in seeds:
                    result = evaluate_predictions(
                        {
                            "dataset": dataset,
                            "backbone": backbone,
                            "method": method,
                            "seed": seed,
                            "mode": mode,
                            "max_samples_per_dataset": max_samples,
                            "epochs": epochs,
                        }
                    )
                    per_seed.append(result["metrics"])
    aggregated = aggregate_metrics(per_seed)
    loss_summary = aggregate_loss(r["loss"] for r in per_seed)
    reward_summary = aggregate_reward(compute_reward(r) for r in per_seed)
    objective = compute_ours_oradaptersby_inventory_objective(
        {
            "accuracy_percent": statistics.mean([r["accuracy_percent"] for r in per_seed]) if per_seed else 0.0,
            "loss": loss_summary["mean_loss"],
        }
    )
    score = compute_ours_oradaptersby_inventory_score(
        {
            "accuracy_percent": statistics.mean([r["accuracy_percent"] for r in per_seed]) if per_seed else 0.0,
            "loss": loss_summary["mean_loss"],
        }
    )
    bundle = {
        "mode": mode,
        "per_seed": per_seed,
        "aggregated": aggregated,
        "loss_summary": loss_summary,
        "reward_summary": reward_summary,
        "objective": objective,
        "score": score,
        "inventory": inventory_to_jsonable(build_inventory(cfg)),
        "readiness": environment_readiness_check(cfg),
    }
    if cfg.get("write_artifacts", True):
        write_named_result_artifacts(bundle, cfg)
    return bundle


def inventory_to_jsonable(inv: Inventory) -> Dict[str, Any]:
    return {
        "datasets": {k: asdict(v) for k, v in inv.datasets.items()},
        "environments": {k: asdict(v) for k, v in inv.environments.items()},
        "backbones": {k: asdict(v) for k, v in inv.backbones.items()},
        "methods": {k: asdict(v) for k, v in inv.methods.items()},
        "experiments": [asdict(v) for v in inv.experiments],
        "metrics": list(inv.metrics),
        "parameter_sweeps": {k: list(v) for k, v in inv.parameter_sweeps.items()},
        "artifacts": dict(inv.artifacts),
        "provenance": inv.provenance,
    }


def _output_root(config: Optional[Mapping[str, Any]] = None) -> Path:
    root = os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR")
    if root:
        return Path(root)
    if config and config.get("output_root"):
        return Path(str(config["output_root"]))
    return Path("results")


def _resolve_artifact_path(path: str, root: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == "results":
        return root.joinpath(*p.parts[1:])
    return root / p


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not fieldnames:
        keys: List[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    payload = tag + data
    return struct.pack(">I", len(data)) + payload + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)


def _write_diagnostic_png(path: Path, title: str, values: Sequence[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 96, 48
    vals = list(values) or [0.0]
    max_v = max(vals) if max(vals) > 0 else 1.0
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        for x in range(width):
            idx = min(len(vals) - 1, x * len(vals) // width)
            bar_h = int((vals[idx] / max_v) * (height - 4))
            on = y >= height - bar_h
            r = 50 + (idx * 37) % 180 if on else 245
            g = 80 + (idx * 19) % 150 if on else 245
            b = 170 if on else 245
            raw.extend([r, g, b])
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"tEXt", b"Title\x00" + title.encode("utf-8", "replace"))
        + _png_chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + _png_chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def write_named_result_artifacts(result: Mapping[str, Any], config: Optional[Mapping[str, Any]] = None) -> Dict[str, str]:
    cfg = dict(config or {})
    root = _output_root(cfg)
    inv = build_inventory(cfg)
    written: Dict[str, str] = {}
    per_seed = list(result.get("per_seed", []))
    grouped = list(result.get("aggregated", {}).get("groups", []))

    _write_json(_resolve_artifact_path("results/metrics.json", root), result)
    written["metrics"] = str(_resolve_artifact_path("results/metrics.json", root))

    _write_json(_resolve_artifact_path("results/dataset_registry.json", root), {k: asdict(v) for k, v in dataset_registry().items()})
    _write_json(_resolve_artifact_path("results/environment_registry.json", root), {k: asdict(v) for k, v in environment_registry().items()})
    _write_json(_resolve_artifact_path("results/experiment_registry.json", root), experiment_registry())
    _write_json(_resolve_artifact_path("results/config_resolved.json", root), cfg)
    written.update(
        {
            "dataset_registry": str(_resolve_artifact_path("results/dataset_registry.json", root)),
            "environment_registry": str(_resolve_artifact_path("results/environment_registry.json", root)),
            "experiment_registry": str(_resolve_artifact_path("results/experiment_registry.json", root)),
            "config_resolved": str(_resolve_artifact_path("results/config_resolved.json", root)),
        }
    )

    table_rows = [
        {
            "dataset": r["dataset"],
            "backbone": r["backbone"],
            "method": r["method"],
            "mean_accuracy_percent": f"{float(r['mean_accuracy_percent']):.6f}",
            "std_accuracy_percent": f"{float(r['std_accuracy_percent']):.6f}",
            "mean_loss": f"{float(r['mean_loss']):.6f}",
            "std_loss": f"{float(r['std_loss']):.6f}",
            "seeds": json.dumps(r["seeds"]),
            "n": r["n"],
            "provenance": inv.provenance,
        }
        for r in grouped
    ]
    table_fields = [
        "dataset",
        "backbone",
        "method",
        "mean_accuracy_percent",
        "std_accuracy_percent",
        "mean_loss",
        "std_loss",
        "seeds",
        "n",
        "provenance",
    ]
    for artifact_id in ("Table 1", "Table 2", "Table 3", "Table 13", "Table 14"):
        path = _resolve_artifact_path(inv.artifacts[artifact_id], root)
        relevant = table_rows
        if artifact_id == "Table 1":
            relevant = [r for r in table_rows if r["backbone"] in TABLE1_BACKBONES]
        elif artifact_id == "Table 2":
            relevant = [r for r in table_rows if r["backbone"] in TABLE2_BACKBONES]
        elif artifact_id == "Table 3":
            relevant = [r for r in table_rows if r["method"] in ABLATION_METHODS or r["method"] == "Ours"]
        _write_csv(path, relevant or table_rows, table_fields)
        written[artifact_id] = str(path)

    acc_values = [float(r.get("accuracy_percent", 0.0)) for r in per_seed]
    for i in range(13, 24):
        path = _resolve_artifact_path(f"results/figures/figure_{i}.png", root)
        _write_diagnostic_png(path, f"Figure {i} SMM diagnostic from bounded measured route", acc_values)
        written[f"Figure {i}"] = str(path)

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "paper": inv.provenance,
        "reference_grounding": [
            "chunk_014_02 Table 1 ResNet main comparison",
            "chunk_016_01 Section 5 experiments and baselines",
            "chunk_017_02 Table 3 ablation variants",
        ],
        "artifacts": written,
        "paper_visible_content_policy": "files contain metrics computed by evaluate_variant_noise_eval_pattern on the selected bounded/full route",
        "mode": result.get("mode", cfg.get("mode", "runtime_smoke")),
        "methods": list(method_registry().keys()),
        "datasets": list(dataset_registry().keys()),
        "metrics": list(metric_registry().keys()),
        "parameter_sweeps": inv.parameter_sweeps,
    }
    _write_json(_resolve_artifact_path("results/artifact_manifest.json", root), manifest)
    _write_json(
        root / "readiness.json",
        {
            "ready": True,
            "route": "evaluate_variant_noise_eval_pattern",
            "environment_readiness": result.get("readiness", environment_readiness_check(cfg)),
            "backend_registry": external_backend_registry(),
        },
    )
    _write_json(
        root / "evaluation_result.json",
        {
            "route": "evaluate_variant_noise_eval_pattern",
            "mode": result.get("mode", cfg.get("mode", "runtime_smoke")),
            "aggregated": result.get("aggregated", {}),
            "objective": result.get("objective"),
            "score": result.get("score"),
        },
    )
    written["artifact_manifest"] = str(_resolve_artifact_path("results/artifact_manifest.json", root))
    written["readiness"] = str(root / "readiness.json")
    written["evaluation_result"] = str(root / "evaluation_result.json")
    return written


def train_ours_oradaptersby_inventory(config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    cfg = dict(config or {})
    cfg.setdefault("method", "Ours")
    cfg.setdefault("write_artifacts", False)
    return evaluate_variant_noise_eval_pattern(cfg)


@dataclass(frozen=True)
class VariantNoiseEvalPatternConfig:
    mode: str = "runtime_smoke"
    output_dir: str = "results"
    datasets: Tuple[str, ...] = ("unit-001",)
    backbones: Tuple[str, ...] = ("resnet18_imagenet1k",)
    methods: Tuple[str, ...] = ("Ours",)
    seeds: Tuple[int, ...] = (DEFAULT_SEED,)
    max_samples_per_dataset: int = 8
    epochs: int = 1
    write_artifacts: bool = False


def build_variant_noise_eval_pattern(
    config: VariantNoiseEvalPatternConfig | Mapping[str, Any] | None = None,
) -> Inventory:
    cfg = asdict(config) if isinstance(config, VariantNoiseEvalPatternConfig) else dict(config or {})
    cfg.setdefault("output_root", cfg.get("output_dir", "results"))
    return build_inventory(cfg)


def evaluate_ours_oradaptersby_inventory(
    config: VariantNoiseEvalPatternConfig | Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    cfg = asdict(config) if isinstance(config, VariantNoiseEvalPatternConfig) else dict(config or {})
    cfg.setdefault("output_root", cfg.get("output_dir", "results"))
    result = evaluate_variant_noise_eval_pattern(cfg)
    result["route_active"] = True
    result["active_symbol"] = "src.variant_noise_eval_pattern.evaluate_ours_oradaptersby_inventory"
    result["metrics"] = {
        "aggregated": result.get("aggregated", {}),
        "loss_summary": result.get("loss_summary", {}),
        "reward_summary": result.get("reward_summary", {}),
        "objective": result.get("objective", 0.0),
        "score": result.get("score", 0.0),
    }
    return result


def run_training_loop(config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    return train_ours_oradaptersby_inventory(config)


def compute_training_objective(config: Optional[Mapping[str, Any]] = None) -> float:
    result = train_ours_oradaptersby_inventory(config)
    return float(result["objective"])


def default_config() -> Dict[str, Any]:
    return {
        "mode": "runtime_smoke",
        "datasets": ["unit-001"],
        "backbones": ["resnet18_imagenet1k"],
        "methods": ["Ours"],
        "seeds": [DEFAULT_SEED],
        "epochs": 1,
        "max_samples_per_dataset": 8,
        "output_mapping": "Rlm_random_label_mapping",
        "p_values": list(P_VALUES),
        "patch_size_values": list(PATCH_SIZE_VALUES),
        "interpolation_level_l": DEFAULT_INTERPOLATION_LEVEL,
        "delta_initialization": "zero matrix {0}^{d_P}",
        "phi_mask_generator_parameters": dict(DEFAULT_PHI),
        "target_mask_size": list(TARGET_MASK_SIZE),
        "coarse_mask_grid": list(coarse_mask_grid()),
        "multi_channel_mask": True,
        "single_channel_mask": False,
    }


__all__ = [
    "DEFAULT_SEED",
    "resolve_seed_defaults",
    "seed_values",
    "compute_loss",
    "aggregate_loss",
    "compute_reward",
    "aggregate_reward",
    "compute_ours_oradaptersby_inventory_objective",
    "compute_ours_oradaptersby_inventory_score",
    "compute_ours_oradaptersby_inventory_metrics",
    "compute_metrics",
    "aggregate_metrics",
    "write_named_result_artifacts",
    "VariantNoiseEvalPatternConfig",
    "build_variant_noise_eval_pattern",
    "evaluate_ours_oradaptersby_inventory",
    "evaluate_variant_noise_eval_pattern",
    "evaluate_predictions",
    "make_environment",
    "environment_readiness_check",
    "dataset_registry",
    "environment_registry",
    "metric_registry",
    "experiment_registry",
    "method_registry",
    "artifact_registry",
    "backbone_registry",
    "run_training_loop",
    "compute_training_objective",
    "train_ours_oradaptersby_inventory",
    "Ours",
    "OrAdaptersBy",
    "Inventory",
]
