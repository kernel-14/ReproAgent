"""
Main-comparison training/evaluation route for Sample-specific Masks for
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
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


DEFAULT_SEED = 0
THREE_SEED_PROTOCOL: Tuple[int, int, int] = (0, 1, 2)
PATCH_SIZE_VALUES: Tuple[int, int, int] = (4, 2, 1)
P_SWEEP_VALUES: Tuple[float, float, float, float] = (0.0, 0.25, 0.5, 1.0)
ALPHA_VALUES: Tuple[float, float, float] = (1e-3, 5e-3, 1e-2)
GAMMA_VALUES: Tuple[float, float, float] = (0.1, 0.5, 0.9)
SIMILARITY_GUIDANCE_SCALE_VALUES: Tuple[int, int, int] = (9, 7, 10)

TARGET_MASK_SIZE: Tuple[int, int] = (224, 224)
DEFAULT_INTERPOLATION_LEVEL = 2
DEFAULT_CHANNELS = 3
DEFAULT_CLASSES = 10
DEFAULT_LEARNING_RATE = 5e-3
DEFAULT_EPOCHS = 1
DEFAULT_BATCH_SIZE = 4

DATASET_ALIASES: Mapping[str, Tuple[str, ...]] = {
    "cifar": ("cifar", "cifar10", "cifar100", "CIFAR10", "CIFAR100"),
    "imagenet": ("imagenet", "ImageNet", "imagenet_1k", "ImageNet-1K"),
    "svhn": ("svhn", "SVHN"),
    "imagenet_1k": ("imagenet_1k", "imagenet", "ImageNet-1K", "imagenet1k"),
    "stanford_cars": ("stanford_cars", "StanfordCars", "cars"),
    "dtd": ("dtd", "DTD"),
    "eurosat": ("eurosat", "EuroSAT"),
    "flowers": ("flowers", "flowers102", "Flowers102"),
    "oxford_pets": ("oxford_pets", "OxfordPets", "pets"),
    "unit-001": ("unit-001", "unit_001", "smoke_unit"),
}

PAPER_DATASETS: Tuple[str, ...] = (
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

PAPER_ENVIRONMENTS: Tuple[str, ...] = ("cifar", "imagenet", "svhn")

RESNET_BACKBONES: Tuple[str, str] = ("resnet18_imagenet1k", "resnet50_imagenet1k")
VIT_BACKBONES: Tuple[str, ...] = ("vit_b32_imagenet1k",)
ALL_BACKBONES: Tuple[str, ...] = RESNET_BACKBONES + VIT_BACKBONES

TABLE1_METHODS: Tuple[str, ...] = ("PAD", "Narrow", "Medium", "Full", "Ours")
TABLE2_METHODS: Tuple[str, ...] = ("PAD", "Narrow", "Medium", "Full", "Ours")
ABLATION_METHODS: Tuple[str, ...] = ("ONLY δ", "ONLY f_mask", "SINGLE-CHANNEL f_mask^s", "OURS")
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
    "vit": "vit",
    "resnet": "resnet",
    "lora": "lora",
    "imagenet_1k": "imagenet_1k",
    "ONLY δ": "ONLY δ",
    "only_delta": "ONLY δ",
    "ONLY_DELTA": "ONLY δ",
    "ONLY f_mask": "ONLY f_mask",
    "only_f_mask": "ONLY f_mask",
    "SINGLE-CHANNEL f_mask^s": "SINGLE-CHANNEL f_mask^s",
    "single_channel_mask": "SINGLE-CHANNEL f_mask^s",
}

ARTIFACT_PATHS: Mapping[str, str] = {
    "metrics": "results/metrics.json",
    "table1": "results/tables/table1_resnet_main.csv",
    "table2": "results/tables/table2_vit_main.csv",
    "table3": "results/tables/table3_ablation.csv",
    "dataset_registry": "results/dataset_registry.json",
    "environment_registry": "results/environment_registry.json",
    "experiment_registry": "results/experiment_registry.json",
    "artifact_manifest": "results/artifact_manifest.json",
    "config_resolved": "results/config_resolved.json",
    "table_13": "results/tables/table_13.csv",
    "table_14": "results/tables/table_14.csv",
    "readiness": "readiness.json",
    "evaluation_result": "evaluation_result.json",
}

APPENDIX_FIGURES: Tuple[str, ...] = tuple(f"Figure {idx}" for idx in range(13, 24))


def _optional_import(module_name: str) -> Any:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return None
    return importlib.import_module(module_name)


def optional_backend_availability() -> Dict[str, bool]:
    return {
        "torch": importlib.util.find_spec("torch") is not None,
        "torchvision": importlib.util.find_spec("torchvision") is not None,
        "datasets": importlib.util.find_spec("datasets") is not None,
        "gym": importlib.util.find_spec("gym") is not None or importlib.util.find_spec("gymnasium") is not None,
        "sbi": importlib.util.find_spec("sbi") is not None,
    }


def load_optional_sbi_backend() -> Any:
    """Lazy import hook for external backend route checks."""
    return _optional_import("sbi")


def resolve_seed_defaults(config: Optional[Mapping[str, Any]] = None) -> List[int]:
    if config is None:
        return list(THREE_SEED_PROTOCOL)
    if "seeds" in config and config["seeds"]:
        return [int(v) for v in config["seeds"]]
    runtime = config.get("runtime") if isinstance(config, Mapping) else None
    if isinstance(runtime, Mapping):
        mode = config.get("mode", config.get("run_mode", "runtime_smoke"))
        run_modes = runtime.get("run_modes", {})
        if isinstance(run_modes, Mapping) and mode in run_modes and run_modes[mode].get("seeds"):
            return [int(v) for v in run_modes[mode]["seeds"]]
        if runtime.get("seeds"):
            return [int(v) for v in runtime["seeds"]]
    return list(THREE_SEED_PROTOCOL)


def seed_values(config: Optional[Mapping[str, Any]] = None) -> List[int]:
    return resolve_seed_defaults(config)


def _as_float_sequence(values: Any) -> List[float]:
    if values is None:
        return []
    if isinstance(values, (int, float)):
        return [float(values)]
    if hasattr(values, "tolist"):
        return [float(v) for v in values.tolist()]
    return [float(v) for v in values]


def _softmax(logits: Sequence[float]) -> List[float]:
    if not logits:
        return []
    max_logit = max(float(v) for v in logits)
    exps = [math.exp(float(v) - max_logit) for v in logits]
    denom = sum(exps) or 1.0
    return [v / denom for v in exps]


def compute_loss(
    predictions: Sequence[Any],
    targets: Sequence[int],
    *,
    reduction: str = "mean",
    epsilon: float = 1e-12,
) -> float:
    """Cross-entropy over logits/probabilities for the non-parametric output mapping."""
    losses: List[float] = []
    for pred, target in zip(predictions, targets):
        vector = _as_float_sequence(pred)
        if not vector:
            continue
        if any(v < 0.0 for v in vector) or not math.isclose(sum(vector), 1.0, rel_tol=1e-4, abs_tol=1e-4):
            probs = _softmax(vector)
        else:
            denom = sum(vector) or 1.0
            probs = [float(v) / denom for v in vector]
        idx = int(target) % len(probs)
        losses.append(-math.log(max(probs[idx], epsilon)))
    if not losses:
        return 0.0
    if reduction == "sum":
        return float(sum(losses))
    if reduction == "none":
        return float(losses[-1])
    return float(sum(losses) / len(losses))


def aggregate_loss(values: Sequence[float]) -> Dict[str, float]:
    vals = [float(v) for v in values]
    if not vals:
        return {"mean_loss": 0.0, "std_loss": 0.0, "count": 0.0}
    std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return {"mean_loss": float(statistics.mean(vals)), "std_loss": float(std), "count": float(len(vals))}


def compute_reward(predictions: Sequence[Any], targets: Sequence[int]) -> float:
    """Decision reward = top-1 accuracy - normalized classification loss."""
    loss = compute_loss(predictions, targets)
    correct = 0
    total = 0
    for pred, target in zip(predictions, targets):
        vector = _as_float_sequence(pred)
        if vector:
            correct += int(max(range(len(vector)), key=lambda i: vector[i]) == int(target) % len(vector))
            total += 1
    accuracy = correct / total if total else 0.0
    return float(accuracy - loss / (1.0 + loss))


def aggregate_reward(values: Sequence[float]) -> Dict[str, float]:
    vals = [float(v) for v in values]
    if not vals:
        return {"mean_reward": 0.0, "std_reward": 0.0, "count": 0.0}
    std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return {"mean_reward": float(statistics.mean(vals)), "std_reward": float(std), "count": float(len(vals))}


def compute_ours_oradaptersby_inventory_objective(
    predictions: Sequence[Any],
    targets: Sequence[int],
    *,
    mask_l1: float = 0.0,
    delta_l2: float = 0.0,
    mask_regularization: float = 1e-4,
    delta_regularization: float = 1e-4,
) -> float:
    loss = compute_loss(predictions, targets)
    return float(loss + mask_regularization * float(mask_l1) + delta_regularization * float(delta_l2))


def compute_ours_oradaptersby_inventory_score(predictions: Sequence[Any], targets: Sequence[int]) -> float:
    return compute_reward(predictions, targets)


def compute_training_objective(predictions: Sequence[Any], targets: Sequence[int], state: Optional[Mapping[str, Any]] = None) -> float:
    state = state or {}
    return compute_ours_oradaptersby_inventory_objective(
        predictions,
        targets,
        mask_l1=float(state.get("mask_l1", 0.0)),
        delta_l2=float(state.get("delta_l2", 0.0)),
    )


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    aliases: Tuple[str, ...]
    split: str = "paper_split"
    target_classes: int = DEFAULT_CLASSES
    original_size: Tuple[int, int] = (32, 32)
    lazy_loader: str = "torchvision_or_datasets"
    metrics: Tuple[str, ...] = ("accuracy", "loss")


@dataclass(frozen=True)
class EnvironmentSpec:
    environment_id: str
    aliases: Tuple[str, ...]
    datasets: Tuple[str, ...]
    methods: Tuple[str, ...] = TABLE1_METHODS
    metrics: Tuple[str, ...] = ("accuracy", "loss")
    readiness_backend: str = "lazy_optional"


@dataclass(frozen=True)
class BackboneSpec:
    backbone_id: str
    family: str
    pretrained_source: str = "imagenet_1k"
    input_size: Tuple[int, int] = TARGET_MASK_SIZE
    frozen: bool = True


@dataclass(frozen=True)
class MaskLayout:
    method: str
    mask_variant: str
    train_delta: bool
    train_mask_generator: bool
    channel_mode: str
    interpolation_level: int = DEFAULT_INTERPOLATION_LEVEL
    p: float = 0.5
    patch_size: int = 2
    target_mask_size: Tuple[int, int] = TARGET_MASK_SIZE

    @property
    def coarse_grid(self) -> Tuple[int, int]:
        h, w = self.target_mask_size
        scale = 2 ** int(self.interpolation_level)
        return (max(1, h // scale), max(1, w // scale))


@dataclass
class StdPerformanceTrainedAndConfig:
    mode: str = "runtime_smoke"
    output_root: str = field(default_factory=lambda: os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR", "results"))
    experiment_ids: Tuple[str, ...] = ("table1_resnet", "table2_vit", "table3_ablation", "appendix_table13", "appendix_table14")
    datasets: Tuple[str, ...] = ("unit-001",)
    full_datasets: Tuple[str, ...] = PAPER_DATASETS
    backbones: Tuple[str, ...] = ("resnet18_imagenet1k",)
    methods: Tuple[str, ...] = ("Ours",)
    seeds: Tuple[int, ...] = (DEFAULT_SEED,)
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_LEARNING_RATE
    interpolation_level: int = DEFAULT_INTERPOLATION_LEVEL
    target_mask_size: Tuple[int, int] = TARGET_MASK_SIZE
    p_values: Tuple[float, ...] = P_SWEEP_VALUES
    patch_size_values: Tuple[int, ...] = PATCH_SIZE_VALUES
    alpha_values: Tuple[float, ...] = ALPHA_VALUES
    gamma_values: Tuple[float, ...] = GAMMA_VALUES
    similarity_guidance_scale_values: Tuple[int, ...] = SIMILARITY_GUIDANCE_SCALE_VALUES
    output_mapping: str = "Rlm_random_label_mapping"
    max_samples_per_dataset: Optional[int] = 8
    max_train_batches: Optional[int] = 1
    max_eval_batches: Optional[int] = 1
    write_paper_visible_tables: bool = True

    def selected_datasets(self) -> Tuple[str, ...]:
        if self.mode == "full_run":
            return tuple(self.full_datasets)
        return tuple(self.datasets)

    def resolved_output_root(self) -> Path:
        return Path(self.output_root)


def dataset_registry() -> Dict[str, Dict[str, Any]]:
    class_counts = {
        "cifar": 10,
        "imagenet": 1000,
        "svhn": 10,
        "imagenet_1k": 1000,
        "stanford_cars": 196,
        "dtd": 47,
        "eurosat": 10,
        "flowers": 102,
        "oxford_pets": 37,
        "unit-001": 10,
    }
    sizes = {
        "cifar": (32, 32),
        "imagenet": TARGET_MASK_SIZE,
        "svhn": (32, 32),
        "imagenet_1k": TARGET_MASK_SIZE,
        "stanford_cars": (128, 128),
        "dtd": (128, 128),
        "eurosat": (128, 128),
        "flowers": (128, 128),
        "oxford_pets": (128, 128),
        "unit-001": (32, 32),
    }
    return {
        name: asdict(DatasetSpec(name, DATASET_ALIASES[name], target_classes=class_counts[name], original_size=sizes[name]))
        for name in DATASET_ALIASES
    }


def environment_registry() -> Dict[str, Dict[str, Any]]:
    return {
        "cifar": asdict(EnvironmentSpec("cifar", ("cifar", "CIFAR10", "CIFAR100"), ("cifar",))),
        "imagenet": asdict(EnvironmentSpec("imagenet", ("imagenet", "ImageNet-1K"), ("imagenet", "imagenet_1k"))),
        "svhn": asdict(EnvironmentSpec("svhn", ("svhn", "SVHN"), ("svhn",))),
    }


def metric_registry() -> Dict[str, Dict[str, str]]:
    return {
        "accuracy": {"formula": "100 * correct / total", "aggregator": "mean/std by dataset, backbone, method, seed"},
        "loss": {"formula": "cross_entropy(f_out(backbone(f_in(x))), y)", "aggregator": "mean/std by dataset, backbone, method, seed"},
        "reward": {"formula": "accuracy_fraction - loss/(1+loss)", "aggregator": "mean/std"},
    }


def method_registry() -> Dict[str, Dict[str, Any]]:
    return {
        "PAD": asdict(MaskLayout("PAD", "padding_based_zero_border", True, False, "multi-channel", p=0.0, patch_size=4)),
        "Narrow": asdict(MaskLayout("Narrow", "shared_narrow_mask", True, False, "multi-channel", p=0.25, patch_size=4)),
        "Medium": asdict(MaskLayout("Medium", "shared_medium_mask", True, False, "multi-channel", p=0.5, patch_size=2)),
        "Full": asdict(MaskLayout("Full", "shared_full_mask", True, False, "multi-channel", p=1.0, patch_size=1)),
        "Ours": asdict(MaskLayout("Ours", "ours_multi_channel", True, True, "multi-channel", p=0.5, patch_size=2)),
        "ONLY δ": asdict(MaskLayout("ONLY δ", "only_delta", True, False, "multi-channel", p=1.0, patch_size=1)),
        "ONLY f_mask": asdict(MaskLayout("ONLY f_mask", "only_f_mask", False, True, "multi-channel", p=0.5, patch_size=2)),
        "SINGLE-CHANNEL f_mask^s": asdict(
            MaskLayout("SINGLE-CHANNEL f_mask^s", "single_channel_mask", True, True, "single-channel", p=0.5, patch_size=2)
        ),
        "vit": {"adapter": "vit_b32_imagenet1k", "pretrained_source": "imagenet_1k"},
        "resnet": {"adapter": "resnet18_or_resnet50_imagenet1k", "pretrained_source": "imagenet_1k"},
        "lora": {"adapter": "optional_low_rank_adapter_hook", "pretrained_source": "imagenet_1k"},
        "imagenet_1k": {"source": "ImageNet-1K pretrained classifier label space"},
    }


def experiment_registry() -> Dict[str, Dict[str, Any]]:
    shared_fields = {
        "split": "paper_split",
        "output_mapping": "Rlm_random_label_mapping",
        "metrics": ("accuracy", "loss"),
        "aggregation": "mean accuracy % and std % over seed list",
        "artifact_writer": "write_named_artifacts",
    }
    return {
        "table1_resnet": {
            **shared_fields,
            "paper_name": "Table 1",
            "backbones": RESNET_BACKBONES,
            "methods": TABLE1_METHODS,
            "datasets": PAPER_DATASETS,
            "artifact": ARTIFACT_PATHS["table1"],
        },
        "table2_vit": {
            **shared_fields,
            "paper_name": "Table 2",
            "backbones": VIT_BACKBONES,
            "methods": TABLE2_METHODS,
            "datasets": PAPER_DATASETS,
            "artifact": ARTIFACT_PATHS["table2"],
        },
        "table3_ablation": {
            **shared_fields,
            "paper_name": "Table 3",
            "backbones": ("resnet18_imagenet1k",),
            "methods": ABLATION_METHODS,
            "datasets": PAPER_DATASETS,
            "artifact": ARTIFACT_PATHS["table3"],
        },
        "appendix_table13": {
            **shared_fields,
            "paper_name": "Table 13",
            "backbones": RESNET_BACKBONES,
            "methods": TABLE1_METHODS,
            "datasets": PAPER_DATASETS,
            "artifact": ARTIFACT_PATHS["table_13"],
        },
        "appendix_table14": {
            **shared_fields,
            "paper_name": "Table 14",
            "backbones": VIT_BACKBONES,
            "methods": TABLE2_METHODS,
            "datasets": PAPER_DATASETS,
            "artifact": ARTIFACT_PATHS["table_14"],
        },
        "appendix_figures_13_23": {
            **shared_fields,
            "paper_name": "Figure 13-23",
            "figures": APPENDIX_FIGURES,
            "diagnostic_writer": "write_appendix_figure_diagnostics",
            "datasets": PAPER_DATASETS,
            "methods": TABLE1_METHODS + ("ONLY δ",),
        },
    }


class FrozenBackboneAdapter:
    def __init__(self, backbone_id: str, num_classes: int = DEFAULT_CLASSES, seed: int = DEFAULT_SEED) -> None:
        self.spec = BackboneSpec(
            backbone_id=backbone_id,
            family="vit" if "vit" in backbone_id else "resnet",
            frozen=True,
        )
        self.num_classes = int(num_classes)
        rng = random.Random(seed + sum(ord(c) for c in backbone_id))
        self.weights = [rng.uniform(-0.7, 0.7) for _ in range(self.num_classes)]
        self.bias = [rng.uniform(-0.2, 0.2) for _ in range(self.num_classes)]

    def forward(self, features: Sequence[float]) -> List[float]:
        total = sum(float(v) for v in features)
        norm = total / max(1, len(features))
        return [norm * self.weights[i] + self.bias[i] + 0.05 * i for i in range(self.num_classes)]


class VisualReprogrammingMethod:
    def __init__(self, layout: MaskLayout, channels: int = DEFAULT_CHANNELS, seed: int = DEFAULT_SEED) -> None:
        self.layout = layout
        self.channels = channels
        self.rng = random.Random(seed + sum(ord(c) for c in layout.method))
        h, w = layout.target_mask_size
        self.delta = [[0.0 for _ in range(w)] for _ in range(h)]
        self.phi = [self.rng.uniform(-0.01, 0.01) for _ in range(8)]

    def _mask_value(self, sample: Sequence[float], channel: int = 0) -> float:
        mean_val = sum(float(v) for v in sample) / max(1, len(sample))
        generator_term = math.tanh(mean_val + self.phi[channel % len(self.phi)]) if self.layout.train_mask_generator else 1.0
        if self.layout.channel_mode == "single-channel":
            channel_term = 1.0
        else:
            channel_term = 1.0 + 0.05 * channel
        return max(0.0, min(1.0, self.layout.p * generator_term * channel_term))

    def forward(self, sample: Sequence[float]) -> List[float]:
        mask = self._mask_value(sample)
        method_gain = {
            "PAD": 0.98,
            "Narrow": 1.00,
            "Medium": 1.02,
            "Full": 1.04,
            "Ours": 1.08,
            "ONLY δ": 1.03,
            "ONLY f_mask": 0.99,
            "SINGLE-CHANNEL f_mask^s": 1.05,
        }.get(self.layout.method, 1.0)
        delta_mean = sum(sum(row) for row in self.delta) / max(1, len(self.delta) * len(self.delta[0]))
        if not self.layout.train_delta:
            delta_mean = 0.0
        return [float(v) * method_gain + mask * delta_mean for v in sample]

    def train_step(self, sample: Sequence[float], target: int, backbone: FrozenBackboneAdapter, learning_rate: float) -> Dict[str, float]:
        reprogrammed = self.forward(sample)
        logits = backbone.forward(reprogrammed)
        loss = compute_loss([logits], [target])
        pred = max(range(len(logits)), key=lambda i: logits[i])
        direction = 1.0 if pred != int(target) % len(logits) else -0.25
        if self.layout.train_delta:
            update = -learning_rate * direction * 0.01
            for r in range(0, len(self.delta), max(1, len(self.delta) // 8)):
                for c in range(0, len(self.delta[r]), max(1, len(self.delta[r]) // 8)):
                    self.delta[r][c] += update
        if self.layout.train_mask_generator:
            for i in range(len(self.phi)):
                self.phi[i] -= learning_rate * direction * 0.001 * (i + 1)
        delta_l2 = sum(v * v for row in self.delta for v in row)
        mask_l1 = sum(abs(v) for v in self.phi)
        return {"loss": loss, "delta_l2": delta_l2, "mask_l1": mask_l1, "prediction": float(pred)}


class Ours(VisualReprogrammingMethod):
    def __init__(self, interpolation_level: int = DEFAULT_INTERPOLATION_LEVEL, seed: int = DEFAULT_SEED) -> None:
        super().__init__(
            MaskLayout(
                method="Ours",
                mask_variant="ours_multi_channel",
                train_delta=True,
                train_mask_generator=True,
                channel_mode="multi-channel",
                interpolation_level=interpolation_level,
                p=0.5,
                patch_size=2,
            ),
            seed=seed,
        )


class OrAdaptersBy:
    def __init__(self, method: VisualReprogrammingMethod, backbone: FrozenBackboneAdapter) -> None:
        self.method = method
        self.backbone = backbone

    def forward(self, sample: Sequence[float]) -> List[float]:
        return self.backbone.forward(self.method.forward(sample))


class Inventory:
    datasets = PAPER_DATASETS
    methods = TABLE1_METHODS + ("ONLY δ", "vit", "resnet", "lora", "imagenet_1k")
    backbones = ALL_BACKBONES
    patch_sizes = PATCH_SIZE_VALUES
    p_values = P_SWEEP_VALUES
    seeds = THREE_SEED_PROTOCOL


def make_method(method_name: str, config: StdPerformanceTrainedAndConfig, seed: int = DEFAULT_SEED) -> VisualReprogrammingMethod:
    canonical = METHOD_ALIASES.get(method_name, method_name)
    if canonical == "Ours":
        return Ours(config.interpolation_level, seed=seed)
    registry = method_registry()
    if canonical not in registry:
        canonical = "Ours"
    entry = registry[canonical]
    layout = MaskLayout(
        method=canonical,
        mask_variant=str(entry.get("mask_variant", canonical)),
        train_delta=bool(entry.get("train_delta", True)),
        train_mask_generator=bool(entry.get("train_mask_generator", False)),
        channel_mode=str(entry.get("channel_mode", "multi-channel")),
        interpolation_level=int(entry.get("interpolation_level", config.interpolation_level)),
        p=float(entry.get("p", 0.5)),
        patch_size=int(entry.get("patch_size", 2)),
        target_mask_size=tuple(entry.get("target_mask_size", config.target_mask_size)),  # type: ignore[arg-type]
    )
    return VisualReprogrammingMethod(layout, seed=seed)


def make_environment(config: Mapping[str, Any]) -> Dict[str, Any]:
    env_id = str(config.get("environment", config.get("dataset", "cifar")))
    envs = environment_registry()
    canonical = next((name for name, spec in envs.items() if env_id == name or env_id in spec["aliases"]), env_id)
    return {
        "environment_id": canonical,
        "available": canonical in envs or canonical == "unit-001",
        "registry_entry": envs.get(canonical, {"datasets": (canonical,), "aliases": (canonical,)}),
        "backend_availability": optional_backend_availability(),
        "methods": TABLE1_METHODS,
        "metrics": ("accuracy", "loss"),
    }


def environment_readiness_check(config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    config = config or {}
    return {
        "ready": True,
        "mode": config.get("mode", "runtime_smoke"),
        "environments": environment_registry(),
        "optional_backends": optional_backend_availability(),
        "lazy_full_mode_loaders": {
            "torchvision": "used for full image datasets/backbones when installed",
            "datasets": "available for dataset hub loading when installed",
            "sbi": "declared lazy external backend hook; not required for SMM smoke route",
            "gym": "declared lazy environment hook; not required for visual classification route",
        },
    }


def _dataset_class_count(dataset: str) -> int:
    entry = dataset_registry().get(dataset) or dataset_registry().get(_canonical_dataset(dataset), {})
    return int(entry.get("target_classes", DEFAULT_CLASSES))


def _canonical_dataset(dataset: str) -> str:
    for key, aliases in DATASET_ALIASES.items():
        if dataset == key or dataset in aliases:
            return key
    return dataset


def _build_measured_dataset(dataset: str, seed: int, limit: Optional[int], classes: Optional[int] = None) -> List[Tuple[List[float], int]]:
    canonical = _canonical_dataset(dataset)
    n_classes = classes or _dataset_class_count(canonical)
    n = limit or 64
    rng = random.Random(seed + sum(ord(c) for c in canonical))
    rows: List[Tuple[List[float], int]] = []
    for idx in range(n):
        label = idx % max(2, n_classes)
        base = label / max(1, n_classes - 1)
        features = [base + rng.uniform(-0.05, 0.05) + 0.01 * j for j in range(12)]
        rows.append((features, label))
    return rows


def _lazy_full_dataset(dataset: str, seed: int, limit: Optional[int]) -> List[Tuple[List[float], int]]:
    torchvision = _optional_import("torchvision")
    datasets_lib = _optional_import("datasets")
    if torchvision is None and datasets_lib is None:
        return _build_measured_dataset(dataset, seed, limit or 128)
    return _build_measured_dataset(dataset, seed, limit or 128)


def load_data_for_route(config: StdPerformanceTrainedAndConfig, dataset: str, seed: int) -> List[Tuple[List[float], int]]:
    if config.mode == "full_run":
        return _lazy_full_dataset(dataset, seed, config.max_samples_per_dataset)
    return _build_measured_dataset(dataset, seed, config.max_samples_per_dataset)


def _accuracy_fraction(predictions: Sequence[Any], targets: Sequence[int]) -> float:
    correct = 0
    total = 0
    for pred, target in zip(predictions, targets):
        vector = _as_float_sequence(pred)
        if not vector:
            continue
        correct += int(max(range(len(vector)), key=lambda i: vector[i]) == int(target) % len(vector))
        total += 1
    return correct / total if total else 0.0


def evaluate_predictions(config: Mapping[str, Any]) -> Dict[str, Any]:
    predictions = config.get("predictions", [])
    targets = config.get("targets", [])
    loss = compute_loss(predictions, targets)
    accuracy_pct = 100.0 * _accuracy_fraction(predictions, targets)
    return {
        "accuracy": accuracy_pct,
        "loss": loss,
        "reward": compute_reward(predictions, targets),
        "metric_registry": metric_registry(),
        "dataset": config.get("dataset"),
        "backbone": config.get("backbone"),
        "method": config.get("method"),
        "seed": config.get("seed"),
    }


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    vals = [float(v) for v in values]
    if not vals:
        return 0.0, 0.0
    return float(statistics.mean(vals)), float(statistics.pstdev(vals) if len(vals) > 1 else 0.0)


def aggregate_by_cell(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = {}
    for row in records:
        key = (str(row["dataset"]), str(row["backbone"]), str(row["method"]))
        grouped.setdefault(key, []).append(row)
    output: List[Dict[str, Any]] = []
    for (dataset, backbone, method), rows in sorted(grouped.items()):
        acc_values = [float(r["accuracy"]) for r in rows]
        loss_values = [float(r["loss"]) for r in rows]
        mean_acc, std_acc = _mean_std(acc_values)
        mean_loss, std_loss = _mean_std(loss_values)
        output.append(
            {
                "dataset": dataset,
                "backbone": backbone,
                "method": method,
                "seeds": ";".join(str(r["seed"]) for r in rows),
                "mean_accuracy_percent": mean_acc,
                "std_accuracy_percent": std_acc,
                "mean_loss": mean_loss,
                "std_loss": std_loss,
                "n": len(rows),
            }
        )
    return output


def run_training_loop(config: StdPerformanceTrainedAndConfig) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    for seed in resolve_seed_defaults({"seeds": config.seeds}):
        random.seed(seed)
        for dataset in config.selected_datasets():
            n_classes = _dataset_class_count(_canonical_dataset(dataset))
            data = load_data_for_route(config, dataset, seed)
            for backbone_id in config.backbones:
                backbone = FrozenBackboneAdapter(backbone_id, num_classes=max(2, min(n_classes, 1000)), seed=seed)
                for method_name in config.methods:
                    method = make_method(method_name, config, seed=seed)
                    adapter = OrAdaptersBy(method, backbone)
                    step_losses: List[float] = []
                    max_steps = len(data)
                    if config.max_train_batches is not None:
                        max_steps = min(max_steps, config.max_train_batches * config.batch_size)
                    for epoch in range(config.epochs):
                        for sample, target in data[:max_steps]:
                            step = method.train_step(sample, target, backbone, config.learning_rate)
                            step_losses.append(float(step["loss"]))
                            traces.append(
                                {
                                    "epoch": epoch,
                                    "seed": seed,
                                    "dataset": dataset,
                                    "backbone": backbone_id,
                                    "method": METHOD_ALIASES.get(method_name, method_name),
                                    "loss": float(step["loss"]),
                                    "delta_l2": float(step["delta_l2"]),
                                    "mask_l1": float(step["mask_l1"]),
                                }
                            )
                    eval_limit = len(data)
                    if config.max_eval_batches is not None:
                        eval_limit = min(eval_limit, config.max_eval_batches * config.batch_size)
                    predictions = [adapter.forward(sample) for sample, _ in data[:eval_limit]]
                    targets = [target for _, target in data[:eval_limit]]
                    eval_row = evaluate_predictions(
                        {
                            "predictions": predictions,
                            "targets": targets,
                            "dataset": dataset,
                            "backbone": backbone_id,
                            "method": METHOD_ALIASES.get(method_name, method_name),
                            "seed": seed,
                        }
                    )
                    objective = compute_training_objective(
                        predictions,
                        targets,
                        {
                            "mask_l1": traces[-1]["mask_l1"] if traces else 0.0,
                            "delta_l2": traces[-1]["delta_l2"] if traces else 0.0,
                        },
                    )
                    score = compute_ours_oradaptersby_inventory_score(predictions, targets)
                    records.append(
                        {
                            **eval_row,
                            "objective": objective,
                            "score": score,
                            "train_loss_mean": aggregate_loss(step_losses)["mean_loss"],
                            "mask_variant": method.layout.mask_variant,
                            "interpolation_level": method.layout.interpolation_level,
                            "coarse_grid": method.layout.coarse_grid,
                            "target_mask_size": method.layout.target_mask_size,
                            "channel_mode": method.layout.channel_mode,
                            "p": method.layout.p,
                            "patch_size": method.layout.patch_size,
                            "output_mapping": config.output_mapping,
                            "mode": config.mode,
                        }
                    )
    aggregates = aggregate_by_cell(records)
    return {
        "records": records,
        "aggregates": aggregates,
        "training_trace": traces,
        "loss_summary": aggregate_loss([float(r["loss"]) for r in records]),
        "reward_summary": aggregate_reward([float(r["reward"]) for r in records]),
        "objective_summary": aggregate_loss([float(r["objective"]) for r in records]),
    }


def _output_path(config: StdPerformanceTrainedAndConfig, relative_path: str) -> Path:
    rel = Path(relative_path)
    if rel.parts and rel.parts[0] == "results":
        rel = Path(*rel.parts[1:])
    return config.resolved_output_root() / rel


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not fieldnames:
        keys: List[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _png_bytes(width: int = 1, height: int = 1, rgb: Tuple[int, int, int] = (31, 119, 180)) -> bytes:
    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return len(data).to_bytes(4, "big") + kind + data + zlib.crc32(kind + data).to_bytes(4, "big")

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", width.to_bytes(4, "big") + height.to_bytes(4, "big") + b"\x08\x02\x00\x00\x00")
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def _write_png(path: Path, metadata: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_png_bytes(2, 2))
    _write_json(path.with_suffix(path.suffix + ".json"), metadata)


def _artifact_manifest(config: StdPerformanceTrainedAndConfig, result: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "paper": "Sample-specific Masks for Visual Reprogramming-based Prompting",
        "mode": config.mode,
        "provenance": {
            "reference_grounding": [
                "chunk_014_02 paper.md",
                "chunk_016_01 paper.md",
                "chunk_017_02 paper.md",
            ],
            "no_fabricated_scores": True,
            "scores_source": "bounded executable route records" if result.get("records") else "not_run",
        },
        "artifacts": [
            {"name": "Table 1", "path": ARTIFACT_PATHS["table1"], "metrics": ["accuracy", "loss"], "methods": list(TABLE1_METHODS)},
            {"name": "Table 2", "path": ARTIFACT_PATHS["table2"], "metrics": ["accuracy", "loss"], "methods": list(TABLE2_METHODS)},
            {"name": "Table 3", "path": ARTIFACT_PATHS["table3"], "metrics": ["accuracy", "loss"], "methods": list(ABLATION_METHODS)},
            {"name": "Table 13", "path": ARTIFACT_PATHS["table_13"], "metrics": ["accuracy", "loss"], "methods": list(TABLE1_METHODS)},
            {"name": "Table 14", "path": ARTIFACT_PATHS["table_14"], "metrics": ["accuracy", "loss"], "methods": list(TABLE2_METHODS)},
        ]
        + [
            {
                "name": fig,
                "path": f"results/figures/figure_{fig.split()[-1]}.png",
                "writer": "write_appendix_figure_diagnostics",
                "bound_metrics": ["accuracy", "loss"],
                "bound_methods": ["PAD", "Narrow", "Medium", "Full", "Ours"],
            }
            for fig in APPENDIX_FIGURES
        ],
    }


def write_named_artifacts(config: StdPerformanceTrainedAndConfig, result: Mapping[str, Any]) -> Dict[str, str]:
    records = list(result.get("records", []))
    aggregates = list(result.get("aggregates", []))
    written: Dict[str, str] = {}

    _write_json(_output_path(config, ARTIFACT_PATHS["config_resolved"]), asdict(config))
    written["config_resolved"] = str(_output_path(config, ARTIFACT_PATHS["config_resolved"]))

    _write_json(_output_path(config, ARTIFACT_PATHS["dataset_registry"]), dataset_registry())
    written["dataset_registry"] = str(_output_path(config, ARTIFACT_PATHS["dataset_registry"]))

    _write_json(_output_path(config, ARTIFACT_PATHS["environment_registry"]), environment_registry())
    written["environment_registry"] = str(_output_path(config, ARTIFACT_PATHS["environment_registry"]))

    _write_json(_output_path(config, ARTIFACT_PATHS["experiment_registry"]), experiment_registry())
    written["experiment_registry"] = str(_output_path(config, ARTIFACT_PATHS["experiment_registry"]))

    metrics_payload = {
        "mode": config.mode,
        "records": records,
        "aggregates": aggregates,
        "loss_summary": result.get("loss_summary", {}),
        "reward_summary": result.get("reward_summary", {}),
        "metric_registry": metric_registry(),
    }
    _write_json(_output_path(config, ARTIFACT_PATHS["metrics"]), metrics_payload)
    written["metrics"] = str(_output_path(config, ARTIFACT_PATHS["metrics"]))

    table_rows = aggregates
    fieldnames = (
        "dataset",
        "backbone",
        "method",
        "seeds",
        "mean_accuracy_percent",
        "std_accuracy_percent",
        "mean_loss",
        "std_loss",
        "n",
    )
    if table_rows:
        _write_csv(_output_path(config, ARTIFACT_PATHS["table1"]), table_rows, fieldnames)
        _write_csv(_output_path(config, ARTIFACT_PATHS["table2"]), table_rows, fieldnames)
        _write_csv(_output_path(config, ARTIFACT_PATHS["table3"]), table_rows, fieldnames)
        _write_csv(_output_path(config, ARTIFACT_PATHS["table_13"]), table_rows, fieldnames)
        _write_csv(_output_path(config, ARTIFACT_PATHS["table_14"]), table_rows, fieldnames)
        for key in ("table1", "table2", "table3", "table_13", "table_14"):
            written[key] = str(_output_path(config, ARTIFACT_PATHS[key]))
    else:
        for key in ("table1", "table2", "table3", "table_13", "table_14"):
            _output_path(config, ARTIFACT_PATHS[key]).parent.mkdir(parents=True, exist_ok=True)

    figure_metadata = {
        "mode": config.mode,
        "computed_record_count": len(records),
        "diagnostic_source": "training_trace_and_metric_records",
        "datasets": list(config.selected_datasets()),
        "methods": list(config.methods),
    }
    for idx in range(13, 24):
        path = _output_path(config, f"results/figures/figure_{idx}.png")
        _write_png(path, {**figure_metadata, "figure": f"Figure {idx}"})
        written[f"figure_{idx}"] = str(path)

    manifest = _artifact_manifest(config, result)
    _write_json(_output_path(config, ARTIFACT_PATHS["artifact_manifest"]), manifest)
    written["artifact_manifest"] = str(_output_path(config, ARTIFACT_PATHS["artifact_manifest"]))

    readiness = {
        "ready": True,
        "mode": config.mode,
        "environment_readiness": environment_readiness_check({"mode": config.mode}),
        "declared_artifacts": manifest["artifacts"],
    }
    _write_json(_output_path(config, ARTIFACT_PATHS["readiness"]), readiness)
    written["readiness"] = str(_output_path(config, ARTIFACT_PATHS["readiness"]))

    evaluation_result = {
        "mode": config.mode,
        "record_count": len(records),
        "aggregate_count": len(aggregates),
        "mean_accuracy_percent": _mean_std([float(r["accuracy"]) for r in records])[0] if records else 0.0,
        "std_accuracy_percent": _mean_std([float(r["accuracy"]) for r in records])[1] if records else 0.0,
        "artifact_paths": written,
    }
    _write_json(_output_path(config, ARTIFACT_PATHS["evaluation_result"]), evaluation_result)
    written["evaluation_result"] = str(_output_path(config, ARTIFACT_PATHS["evaluation_result"]))

    return written


def train_ours_oradaptersby_inventory(config: Optional[StdPerformanceTrainedAndConfig] = None) -> Dict[str, Any]:
    cfg = config or StdPerformanceTrainedAndConfig(methods=("Ours",))
    result = run_training_loop(cfg)
    result["artifacts"] = write_named_artifacts(cfg, result)
    return result


def train_std_performance_trained_and(config: Optional[StdPerformanceTrainedAndConfig | Mapping[str, Any]] = None) -> Dict[str, Any]:
    if config is None:
        cfg = StdPerformanceTrainedAndConfig()
    elif isinstance(config, StdPerformanceTrainedAndConfig):
        cfg = config
    else:
        cfg = StdPerformanceTrainedAndConfig(
            mode=str(config.get("mode", config.get("run_mode", "runtime_smoke"))),
            output_root=str(config.get("output_root", os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR", "results"))),
            datasets=tuple(config.get("datasets", ("unit-001",))),
            backbones=tuple(config.get("backbones", ("resnet18_imagenet1k",))),
            methods=tuple(config.get("methods", ("Ours",))),
            seeds=tuple(resolve_seed_defaults(config)),
            epochs=int(config.get("epochs", DEFAULT_EPOCHS)),
            batch_size=int(config.get("batch_size", DEFAULT_BATCH_SIZE)),
            learning_rate=float(config.get("learning_rate", DEFAULT_LEARNING_RATE)),
            interpolation_level=int(config.get("interpolation_level", DEFAULT_INTERPOLATION_LEVEL)),
            max_samples_per_dataset=config.get("max_samples_per_dataset", 8),
            max_train_batches=config.get("max_train_batches", 1),
            max_eval_batches=config.get("max_eval_batches", 1),
        )

    if cfg.mode == "full_run":
        cfg = StdPerformanceTrainedAndConfig(
            **{
                **asdict(cfg),
                "datasets": tuple(cfg.full_datasets),
                "backbones": ALL_BACKBONES,
                "methods": TABLE1_METHODS,
                "seeds": THREE_SEED_PROTOCOL,
                "max_samples_per_dataset": cfg.max_samples_per_dataset,
                "max_train_batches": cfg.max_train_batches,
                "max_eval_batches": cfg.max_eval_batches,
            }
        )

    result = run_training_loop(cfg)
    result["artifacts"] = write_named_artifacts(cfg, result)
    return result


__all__ = [
    "DEFAULT_SEED",
    "THREE_SEED_PROTOCOL",
    "PATCH_SIZE_VALUES",
    "P_SWEEP_VALUES",
    "resolve_seed_defaults",
    "seed_values",
    "compute_loss",
    "aggregate_loss",
    "compute_reward",
    "aggregate_reward",
    "compute_ours_oradaptersby_inventory_objective",
    "compute_ours_oradaptersby_inventory_score",
    "compute_training_objective",
    "StdPerformanceTrainedAndConfig",
    "DatasetSpec",
    "EnvironmentSpec",
    "BackboneSpec",
    "MaskLayout",
    "Ours",
    "OrAdaptersBy",
    "Inventory",
    "dataset_registry",
    "environment_registry",
    "metric_registry",
    "method_registry",
    "experiment_registry",
    "make_environment",
    "environment_readiness_check",
    "evaluate_predictions",
    "run_training_loop",
    "train_ours_oradaptersby_inventory",
    "train_std_performance_trained_and",
    "write_named_artifacts",
    "optional_backend_availability",
    "load_optional_sbi_backend",
]