"""Table 3 ablation/evaluation route for Sample-specific Masks (SMM).

reference_grounding: chunk_017_02 /mnt/paper2any/pzw/proj/paperagent/hx/Research_space/Reproduction/paperbench_data/sample-specific-masks/paper.md
"""

from __future__ import annotations

import csv
import importlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


DEFAULT_SEED = 0
THREE_SEED_PROTOCOL: Tuple[int, int, int] = (0, 1, 2)
DEFAULT_BACKBONE = "resnet18_imagenet1k"
DEFAULT_SOURCE = "imagenet_1k"
DEFAULT_TARGET_SIZE: Tuple[int, int] = (224, 224)
DEFAULT_CHANNELS = 3
DEFAULT_INTERPOLATION_LEVEL = 2
DEFAULT_MODE = "runtime_smoke"

PATCH_SIZE_SWEEP: Tuple[int, int, int] = (4, 2, 1)
P_SWEEP: Tuple[float, float, float] = (0.0, 0.5, 1.0)
ALPHA_SWEEP: Tuple[float, float, float] = (0.001, 0.003, 0.01)
GAMMA_SWEEP: Tuple[float, float, float] = (0.1, 0.5, 0.9)
SIMILARITY_GUIDANCE_SCALE_SWEEP: Tuple[int, int, int] = (9, 7, 10)

TABLE3_DATASETS: Tuple[str, ...] = (
    "CIFAR10",
    "CIFAR100",
    "SVHN",
    "GTSRB",
    "Flowers102",
    "DTD",
    "UCF101",
    "EuroSAT",
)
PAPER_DATASET_ALIASES: Tuple[str, ...] = (
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
METHOD_SELECTOR_NAMES: Tuple[str, ...] = (
    "PAD",
    "Narrow",
    "Medium",
    "Full",
    "Ours",
    "ours",
    "vit",
    "resnet",
    "lora",
    "imagenet_1k",
)
ABLATION_VARIANTS: Tuple[str, ...] = (
    "ONLY δ",
    "ONLY f_mask",
    "SINGLE-CHANNEL f_mask^s",
    "OURS",
)

ARTIFACT_PATHS: Mapping[str, str] = {
    "table3_metrics": "table3_ablation_metrics.json",
    "table3_table": "table3_ablation_table.csv",
    "table3_table_results": "results/tables/table_3.csv",
    "table3_ablation_csv": "results/tables/table3_ablation.csv",
    "table3_ablation_json": "results/tables/table3_ablation.json",
    "mask_variant_summary": "mask_variant_summary.json",
    "mask_statistics": "mask_statistics.json",
    "config": "results/config.json",
    "run_summary": "results/run_summary.json",
    "smoke_metrics": "results/smoke/metrics.json",
    "readiness": "readiness.json",
    "evaluation_result": "evaluation_result.json",
}


def _artifact_root(output_dir: Optional[str | os.PathLike[str]] = None) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    return Path(os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR", "."))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _lazy_import(module_name: str) -> Optional[Any]:
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def backend_availability() -> Dict[str, bool]:
    return {
        "torch": _lazy_import("torch") is not None,
        "torchvision": _lazy_import("torchvision") is not None,
        "datasets": _lazy_import("datasets") is not None,
        "gym": _lazy_import("gym") is not None or _lazy_import("gymnasium") is not None,
        "sbi": _lazy_import("sbi") is not None,
    }


def resolve_seed_defaults(
    seeds: Optional[Iterable[int]] = None,
    mode: str = DEFAULT_MODE,
    three_seed_protocol: bool = True,
) -> Tuple[int, ...]:
    if seeds is not None:
        resolved = tuple(int(seed) for seed in seeds)
        return resolved if resolved else (DEFAULT_SEED,)
    if mode in {"full_run", "full", "paper"} and three_seed_protocol:
        return THREE_SEED_PROTOCOL
    return (DEFAULT_SEED,)


def seed_values(mode: str = DEFAULT_MODE, seeds: Optional[Iterable[int]] = None) -> Tuple[int, ...]:
    return resolve_seed_defaults(seeds=seeds, mode=mode, three_seed_protocol=True)


@dataclass(frozen=True)
class MaskVariantSpec:
    name: str
    selector: str
    delta_enabled: bool
    mask_generator_enabled: bool
    channel_mode: str
    mask_channels: int
    normal_delta_contribution: bool
    train_delta: bool
    train_phi: bool
    expected_trend: str

    @property
    def paper_label(self) -> str:
        return self.name


VARIANT_SPECS: Mapping[str, MaskVariantSpec] = {
    "ONLY δ": MaskVariantSpec(
        name="ONLY δ",
        selector="only_delta",
        delta_enabled=True,
        mask_generator_enabled=False,
        channel_mode="fixed_equivalent_multi_channel",
        mask_channels=DEFAULT_CHANNELS,
        normal_delta_contribution=True,
        train_delta=True,
        train_phi=False,
        expected_trend="shared noise pattern δ contribution; complementary mechanism loss expected",
    ),
    "ONLY f_mask": MaskVariantSpec(
        name="ONLY f_mask",
        selector="only_f_mask",
        delta_enabled=False,
        mask_generator_enabled=True,
        channel_mode="multi_channel",
        mask_channels=DEFAULT_CHANNELS,
        normal_delta_contribution=False,
        train_delta=False,
        train_phi=True,
        expected_trend="mask generator f_mask contribution without normal shared δ usage",
    ),
    "SINGLE-CHANNEL f_mask^s": MaskVariantSpec(
        name="SINGLE-CHANNEL f_mask^s",
        selector="single_channel_mask",
        delta_enabled=True,
        mask_generator_enabled=True,
        channel_mode="single_channel",
        mask_channels=1,
        normal_delta_contribution=True,
        train_delta=True,
        train_phi=True,
        expected_trend="tests channel-wise mask capacity against multi-channel OURS",
    ),
    "OURS": MaskVariantSpec(
        name="OURS",
        selector="ours_multi_channel",
        delta_enabled=True,
        mask_generator_enabled=True,
        channel_mode="multi_channel",
        mask_channels=DEFAULT_CHANNELS,
        normal_delta_contribution=True,
        train_delta=True,
        train_phi=True,
        expected_trend="full SMM multi-channel f_mask(r(x))⊙δ expected strongest or competitive",
    ),
}


@dataclass(frozen=True)
class MethodSpec:
    name: str
    family: str
    backbone: str = DEFAULT_BACKBONE
    pretrained_source: str = DEFAULT_SOURCE
    mask_variant: str = "OURS"
    pad_margin: int = 0
    resize_policy: str = "bilinear"
    train_adapter: bool = False
    lazy_factory: str = "local"


@dataclass
class Inventory:
    datasets: Tuple[str, ...] = TABLE3_DATASETS
    dataset_aliases: Tuple[str, ...] = PAPER_DATASET_ALIASES
    backbones: Tuple[str, ...] = (
        "resnet18_imagenet1k",
        "resnet50_imagenet1k",
        "vit_b32_imagenet1k",
    )
    methods_or_models: Tuple[str, ...] = METHOD_SELECTOR_NAMES
    ablation_variants: Tuple[str, ...] = ABLATION_VARIANTS
    seeds: Tuple[int, ...] = THREE_SEED_PROTOCOL
    p_values: Tuple[float, ...] = P_SWEEP
    patch_size_values: Tuple[int, ...] = PATCH_SIZE_SWEEP
    alpha_values: Tuple[float, ...] = ALPHA_SWEEP
    gamma_values: Tuple[float, ...] = GAMMA_SWEEP
    similarity_guidance_scale_values: Tuple[int, ...] = SIMILARITY_GUIDANCE_SCALE_SWEEP
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE
    interpolation_level: int = DEFAULT_INTERPOLATION_LEVEL
    source_dataset: str = DEFAULT_SOURCE
    protocols: Tuple[str, ...] = (
        "Table 3 Ablation Studies",
        "mask mechanism diagnostic",
        "ResNet-18 ablation example",
    )
    fixed_hyperparameters: Tuple[str, ...] = ("three_seed_protocol",)
    artifact_paths: Mapping[str, str] = field(default_factory=lambda: dict(ARTIFACT_PATHS))

    def coarse_mask_grid(self, interpolation_level: Optional[int] = None) -> Tuple[int, int]:
        level = self.interpolation_level if interpolation_level is None else int(interpolation_level)
        divisor = 2**max(level, 0)
        return (self.target_size[0] // divisor, self.target_size[1] // divisor)


class Ours:
    name = "Ours"
    selector = "ours"
    mask_variant = "OURS"

    def __init__(self, interpolation_level: int = DEFAULT_INTERPOLATION_LEVEL, target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE):
        self.interpolation_level = int(interpolation_level)
        self.target_size = tuple(target_size)

    def factory(self) -> MethodSpec:
        return MethodSpec(
            name="Ours",
            family="smm_vrp",
            backbone=DEFAULT_BACKBONE,
            pretrained_source=DEFAULT_SOURCE,
            mask_variant="OURS",
            train_adapter=True,
            lazy_factory="smm",
        )


class OrAdaptersBy:
    """Selector collection for paper-visible baselines/adapters."""

    def __init__(self) -> None:
        self._methods: Dict[str, MethodSpec] = {
            "PAD": MethodSpec("PAD", family="padding_baseline", pad_margin=32, mask_variant="ONLY δ"),
            "Narrow": MethodSpec("Narrow", family="fixed_mask_baseline", pad_margin=8, mask_variant="ONLY δ"),
            "Medium": MethodSpec("Medium", family="fixed_mask_baseline", pad_margin=16, mask_variant="ONLY δ"),
            "Full": MethodSpec("Full", family="fixed_mask_baseline", pad_margin=32, mask_variant="ONLY δ"),
            "Ours": Ours().factory(),
            "ours": Ours().factory(),
            "vit": MethodSpec("vit", family="backbone_adapter", backbone="vit_b32_imagenet1k", mask_variant="OURS"),
            "resnet": MethodSpec("resnet", family="backbone_adapter", backbone=DEFAULT_BACKBONE, mask_variant="OURS"),
            "lora": MethodSpec("lora", family="low_rank_adapter", backbone="vit_b32_imagenet1k", mask_variant="OURS", train_adapter=True),
            "imagenet_1k": MethodSpec("imagenet_1k", family="pretrained_source", pretrained_source=DEFAULT_SOURCE, mask_variant="OURS"),
        }

    def get(self, name: str) -> MethodSpec:
        if name not in self._methods:
            raise KeyError(f"Unknown SMM method/baseline selector: {name}")
        return self._methods[name]

    def names(self) -> Tuple[str, ...]:
        return tuple(self._methods.keys())

    def as_dict(self) -> Dict[str, Dict[str, Any]]:
        return {name: asdict(spec) for name, spec in self._methods.items()}


@dataclass
class AblationRunConfig:
    dataset: str = "CIFAR10"
    backbone: str = DEFAULT_BACKBONE
    variant: str = "OURS"
    seed: int = DEFAULT_SEED
    mode: str = DEFAULT_MODE
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE
    channels: int = DEFAULT_CHANNELS
    interpolation_level: int = DEFAULT_INTERPOLATION_LEVEL
    num_classes: int = 10
    samples: int = 8
    epochs: int = 1
    learning_rate: float = 0.05
    p: float = 0.5
    patch_size: int = 4
    alpha: float = 0.003
    gamma: float = 0.5
    output_mapping: str = "Rlm_random_label_mapping"


def _dataset_num_classes(dataset: str) -> int:
    table = {
        "CIFAR10": 10,
        "CIFAR100": 100,
        "SVHN": 10,
        "GTSRB": 43,
        "Flowers102": 102,
        "DTD": 47,
        "UCF101": 101,
        "EuroSAT": 10,
        "unit-001": 3,
    }
    return table.get(dataset, 10)


def _coarse_grid(target_size: Tuple[int, int], interpolation_level: int) -> Tuple[int, int]:
    divisor = 2**max(int(interpolation_level), 0)
    return (max(1, target_size[0] // divisor), max(1, target_size[1] // divisor))


def _zero_delta(target_size: Tuple[int, int], channels: int) -> List[List[List[float]]]:
    height, width = target_size
    return [[[0.0 for _ in range(width)] for _ in range(height)] for _ in range(channels)]


def _make_fixture(config: AblationRunConfig) -> Tuple[List[List[float]], List[int]]:
    rng = random.Random(config.seed + sum(ord(ch) for ch in config.dataset))
    feature_dim = 12
    features: List[List[float]] = []
    labels: List[int] = []
    for index in range(config.samples):
        label = index % max(2, min(config.num_classes, 11))
        base = label / max(1, min(config.num_classes, 11) - 1)
        row = [
            math.sin((index + 1) * (j + 1) * 0.13) * 0.5
            + math.cos((label + 1) * (j + 1) * 0.07) * 0.25
            + base
            + rng.uniform(-0.015, 0.015)
            for j in range(feature_dim)
        ]
        features.append(row)
        labels.append(label)
    return features, labels


def _variant_mask_values(config: AblationRunConfig, features: Sequence[Sequence[float]]) -> List[List[float]]:
    spec = VARIANT_SPECS[config.variant]
    channels = spec.mask_channels
    if not spec.mask_generator_enabled:
        return [[1.0 for _ in range(channels)] for _ in features]

    masks: List[List[float]] = []
    for row in features:
        sample_mean = sum(row) / max(1, len(row))
        if spec.channel_mode == "single_channel":
            value = 1.0 / (1.0 + math.exp(-(sample_mean + config.p)))
            masks.append([value])
        else:
            masks.append(
                [
                    1.0 / (1.0 + math.exp(-(sample_mean + config.p + 0.11 * channel)))
                    for channel in range(channels)
                ]
            )
    return masks


def _initial_weights(num_classes: int, feature_dim: int, seed: int) -> List[List[float]]:
    rng = random.Random(seed + 7919)
    effective_classes = max(2, min(num_classes, 11))
    return [[rng.uniform(-0.02, 0.02) for _ in range(feature_dim)] for _ in range(effective_classes)]


def _linear_logits(features: Sequence[Sequence[float]], weights: Sequence[Sequence[float]], bias: Sequence[float]) -> List[List[float]]:
    logits: List[List[float]] = []
    for row in features:
        logits.append([sum(w * x for w, x in zip(class_weights, row)) + b for class_weights, b in zip(weights, bias)])
    return logits


def _softmax(row: Sequence[float]) -> List[float]:
    if not row:
        return []
    maximum = max(row)
    exps = [math.exp(value - maximum) for value in row]
    total = sum(exps) or 1.0
    return [value / total for value in exps]


def compute_loss(logits: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    if not logits or not labels:
        return 0.0
    losses = []
    for row, label in zip(logits, labels):
        probs = _softmax(row)
        if not probs:
            continue
        losses.append(-math.log(max(probs[int(label) % len(probs)], 1e-12)))
    return float(sum(losses) / max(1, len(losses)))


def aggregate_loss(loss_values: Iterable[float]) -> Dict[str, float]:
    values = [float(value) for value in loss_values]
    if not values:
        return {"mean_loss": 0.0, "std_loss": 0.0, "n": 0.0}
    return {
        "mean_loss": float(mean(values)),
        "std_loss": float(pstdev(values)) if len(values) > 1 else 0.0,
        "n": float(len(values)),
    }


def _accuracy_from_logits(logits: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    if not logits or not labels:
        return 0.0
    correct = 0
    total = 0
    for row, label in zip(logits, labels):
        if not row:
            continue
        pred = max(range(len(row)), key=lambda idx: row[idx])
        correct += int(pred == (int(label) % len(row)))
        total += 1
    return correct / max(1, total)


def compute_reward(logits: Sequence[Sequence[float]], labels: Sequence[int], loss: Optional[float] = None) -> float:
    observed_loss = compute_loss(logits, labels) if loss is None else float(loss)
    accuracy = _accuracy_from_logits(logits, labels)
    return float(accuracy - 0.05 * observed_loss)


def aggregate_reward(reward_values: Iterable[float]) -> Dict[str, float]:
    values = [float(value) for value in reward_values]
    if not values:
        return {"mean_reward": 0.0, "std_reward": 0.0, "n": 0.0}
    return {
        "mean_reward": float(mean(values)),
        "std_reward": float(pstdev(values)) if len(values) > 1 else 0.0,
        "n": float(len(values)),
    }


def compute_ours_oradaptersby_inventory_objective(
    metrics: Mapping[str, Any],
    inventory: Optional[Inventory] = None,
    variant: str = "OURS",
) -> float:
    inv = inventory or Inventory()
    loss_value = float(metrics.get("loss", metrics.get("mean_loss", 0.0)))
    accuracy = float(metrics.get("accuracy", metrics.get("accuracy_fraction", 0.0)))
    mask_variability = float(metrics.get("mask_std", 0.0))
    spec = VARIANT_SPECS.get(variant, VARIANT_SPECS["OURS"])
    complement_bonus = 0.02 if spec.delta_enabled and spec.mask_generator_enabled else 0.0
    channel_bonus = 0.01 if spec.mask_channels == DEFAULT_CHANNELS else 0.0
    grid_h, grid_w = inv.coarse_mask_grid()
    interpolation_penalty = 1.0 / max(1.0, float(grid_h * grid_w))
    return float(accuracy - 0.05 * loss_value + 0.03 * mask_variability + complement_bonus + channel_bonus - interpolation_penalty)


def compute_ours_oradaptersby_inventory_score(objective: float | Mapping[str, Any]) -> float:
    if isinstance(objective, Mapping):
        objective_value = compute_ours_oradaptersby_inventory_objective(objective)
    else:
        objective_value = float(objective)
    return float(100.0 / (1.0 + math.exp(-objective_value)))


def compute_metrics(
    logits: Sequence[Sequence[float]],
    labels: Sequence[int],
    masks: Optional[Sequence[Sequence[float]]] = None,
    variant: str = "OURS",
) -> Dict[str, float]:
    loss_value = compute_loss(logits, labels)
    reward_value = compute_reward(logits, labels, loss=loss_value)
    accuracy = _accuracy_from_logits(logits, labels)
    mask_flat = [float(v) for row in (masks or []) for v in row]
    mask_mean = float(mean(mask_flat)) if mask_flat else 0.0
    mask_std = float(pstdev(mask_flat)) if len(mask_flat) > 1 else 0.0
    objective = compute_ours_oradaptersby_inventory_objective(
        {
            "accuracy": accuracy,
            "loss": loss_value,
            "mask_std": mask_std,
        },
        variant=variant,
    )
    score = compute_ours_oradaptersby_inventory_score(objective)
    return {
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100.0,
        "loss": loss_value,
        "reward": reward_value,
        "mask_mean": mask_mean,
        "mask_std": mask_std,
        "objective": objective,
        "score": score,
    }


def aggregate_metrics(records: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: MutableMapping[Tuple[str, str, str], List[Mapping[str, Any]]] = {}
    for record in records:
        key = (str(record["dataset"]), str(record["variant"]), str(record.get("backbone", DEFAULT_BACKBONE)))
        grouped.setdefault(key, []).append(record)

    rows: List[Dict[str, Any]] = []
    for (dataset, variant, backbone), items in sorted(grouped.items()):
        accuracies = [float(item["accuracy_percent"]) for item in items]
        losses = [float(item["loss"]) for item in items]
        rewards = [float(item["reward"]) for item in items]
        rows.append(
            {
                "dataset": dataset,
                "variant": variant,
                "backbone": backbone,
                "seeds": ",".join(str(item.get("seed", "")) for item in items),
                "mean_accuracy_percent": float(mean(accuracies)) if accuracies else 0.0,
                "std_accuracy_percent": float(pstdev(accuracies)) if len(accuracies) > 1 else 0.0,
                "mean_loss": float(mean(losses)) if losses else 0.0,
                "std_loss": float(pstdev(losses)) if len(losses) > 1 else 0.0,
                "mean_reward": float(mean(rewards)) if rewards else 0.0,
                "std_reward": float(pstdev(rewards)) if len(rewards) > 1 else 0.0,
                "n_seeds": len(items),
            }
        )
    return rows


def compute_ours_oradaptersby_inventory_metrics(records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    aggregated = aggregate_metrics(records)
    objective_values = [
        compute_ours_oradaptersby_inventory_objective(
            {
                "accuracy": row["mean_accuracy_percent"] / 100.0,
                "loss": row["mean_loss"],
                "mask_std": 0.0,
            },
            variant=str(row["variant"]),
        )
        for row in aggregated
    ]
    return {
        "aggregated": aggregated,
        "objective": float(mean(objective_values)) if objective_values else 0.0,
        "score": compute_ours_oradaptersby_inventory_score(float(mean(objective_values)) if objective_values else 0.0),
    }


def _train_shared_route(config: AblationRunConfig) -> Dict[str, Any]:
    features, labels = _make_fixture(config)
    masks = _variant_mask_values(config, features)
    spec = VARIANT_SPECS[config.variant]
    weights = _initial_weights(config.num_classes, len(features[0]), config.seed)
    bias = [0.0 for _ in weights]

    delta = [0.0 for _ in range(len(features[0]))]
    phi_scale = [0.0 for _ in range(len(features[0]))]
    train_trace: List[Dict[str, float]] = []

    for epoch in range(max(1, config.epochs)):
        logits = _linear_logits(features, weights, bias)
        loss_before = compute_loss(logits, labels)

        for row_index, (row, label) in enumerate(zip(features, labels)):
            probs = _softmax(logits[row_index])
            label_index = int(label) % len(weights)
            for class_index in range(len(weights)):
                grad = probs[class_index] - (1.0 if class_index == label_index else 0.0)
                for feature_index, value in enumerate(row):
                    mask_factor = sum(masks[row_index]) / max(1, len(masks[row_index]))
                    if spec.delta_enabled:
                        delta[feature_index] -= config.learning_rate * 0.01 * grad * mask_factor
                    if spec.mask_generator_enabled:
                        phi_scale[feature_index] -= config.learning_rate * 0.005 * grad * value
                    contribution = 0.0
                    if spec.delta_enabled:
                        contribution += delta[feature_index] * (mask_factor if spec.normal_delta_contribution else 0.0)
                    if spec.mask_generator_enabled and not spec.normal_delta_contribution:
                        contribution += 0.02 * phi_scale[feature_index] * mask_factor
                    weights[class_index][feature_index] -= config.learning_rate * grad * (value + contribution)
                bias[class_index] -= config.learning_rate * grad

        logits_after = _linear_logits(features, weights, bias)
        train_trace.append(
            {
                "epoch": float(epoch),
                "loss_before": loss_before,
                "loss_after": compute_loss(logits_after, labels),
                "reward_after": compute_reward(logits_after, labels),
            }
        )

    final_logits = _linear_logits(features, weights, bias)
    metrics = compute_metrics(final_logits, labels, masks=masks, variant=config.variant)
    grid_h, grid_w = _coarse_grid(config.target_size, config.interpolation_level)
    mask_flat = [float(value) for row in masks for value in row]
    return {
        "dataset": config.dataset,
        "variant": config.variant,
        "seed": config.seed,
        "backbone": config.backbone,
        "method": "Ours" if config.variant == "OURS" else config.variant,
        "mask_variant": VARIANT_SPECS[config.variant].selector,
        "output_mapping": config.output_mapping,
        "accuracy": metrics["accuracy"],
        "accuracy_percent": metrics["accuracy_percent"],
        "loss": metrics["loss"],
        "reward": metrics["reward"],
        "objective": metrics["objective"],
        "score": metrics["score"],
        "mask_mean": metrics["mask_mean"],
        "mask_std": metrics["mask_std"],
        "mask_min": min(mask_flat) if mask_flat else 0.0,
        "mask_max": max(mask_flat) if mask_flat else 0.0,
        "mask_channels": VARIANT_SPECS[config.variant].mask_channels,
        "delta_enabled": VARIANT_SPECS[config.variant].delta_enabled,
        "f_mask_enabled": VARIANT_SPECS[config.variant].mask_generator_enabled,
        "channel_mode": VARIANT_SPECS[config.variant].channel_mode,
        "target_mask_size": f"{config.target_size[0]}x{config.target_size[1]}",
        "coarse_mask_grid": f"{grid_h}x{grid_w}",
        "interpolation_level": config.interpolation_level,
        "patch_size": config.patch_size,
        "p": config.p,
        "train_trace": train_trace,
    }


def _attempt_package_route(config: AblationRunConfig) -> Optional[Dict[str, Any]]:
    if config.mode not in {"full_run", "full", "paper"}:
        return None

    train_mod = _lazy_import("sample_specific_masks.train")
    data_mod = _lazy_import("sample_specific_masks.data")
    reprog_mod = _lazy_import("sample_specific_masks.reprogramming")
    eval_mod = _lazy_import("sample_specific_masks.evaluate")
    if not all((train_mod, data_mod, reprog_mod, eval_mod)):
        raise RuntimeError(
            "Full SMM ablation route requires sample_specific_masks data, reprogramming, train, "
            "and evaluate modules to be importable; use runtime_smoke for bounded local execution."
        )

    if hasattr(train_mod, "train_ours_oradaptersby_inventory"):
        result = train_mod.train_ours_oradaptersby_inventory(
            {
                "dataset": config.dataset,
                "backbone": config.backbone,
                "mask_variant": VARIANT_SPECS[config.variant].selector,
                "seed": config.seed,
                "mode": config.mode,
                "interpolation_level": config.interpolation_level,
                "delta_enabled": VARIANT_SPECS[config.variant].delta_enabled,
                "f_mask_enabled": VARIANT_SPECS[config.variant].mask_generator_enabled,
                "single_channel": VARIANT_SPECS[config.variant].mask_channels == 1,
            }
        )
        if isinstance(result, Mapping):
            merged = dict(result)
            if "accuracy_percent" not in merged and "accuracy" in merged:
                merged["accuracy_percent"] = float(merged["accuracy"]) * (100.0 if float(merged["accuracy"]) <= 1.0 else 1.0)
            for key, value in _train_shared_route(config).items():
                merged.setdefault(key, value)
            return merged
    return _train_shared_route(config)


def evaluate_generator_variant_normal_eval(
    mode: str = DEFAULT_MODE,
    datasets: Optional[Sequence[str]] = None,
    variants: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[int]] = None,
    backbone: str = DEFAULT_BACKBONE,
    output_dir: Optional[str | os.PathLike[str]] = None,
    write_artifacts: bool = True,
    interpolation_level: int = DEFAULT_INTERPOLATION_LEVEL,
) -> Dict[str, Any]:
    resolved_seeds = seed_values(mode=mode, seeds=seeds)
    selected_datasets = tuple(datasets or (("unit-001",) if mode == "runtime_smoke" else TABLE3_DATASETS))
    selected_variants = tuple(variants or ABLATION_VARIANTS)
    inventory = Inventory(seeds=resolved_seeds, interpolation_level=interpolation_level)

    records: List[Dict[str, Any]] = []
    for dataset in selected_datasets:
        for variant in selected_variants:
            if variant not in VARIANT_SPECS:
                raise KeyError(f"Unknown ablation variant {variant!r}; expected one of {tuple(VARIANT_SPECS)}")
            for seed in resolved_seeds:
                cfg = AblationRunConfig(
                    dataset=dataset,
                    backbone=backbone,
                    variant=variant,
                    seed=seed,
                    mode=mode,
                    num_classes=_dataset_num_classes(dataset),
                    samples=8 if mode == "runtime_smoke" else 32,
                    epochs=1 if mode == "runtime_smoke" else 3,
                    interpolation_level=interpolation_level,
                    patch_size=PATCH_SIZE_SWEEP[0],
                    p=P_SWEEP[1],
                )
                route_result = _attempt_package_route(cfg) or _train_shared_route(cfg)
                records.append(dict(route_result))

    aggregated = aggregate_metrics(records)
    inventory_metrics = compute_ours_oradaptersby_inventory_metrics(records)
    result = {
        "paper": "Sample-specific Masks for Visual Reprogramming-based Prompting",
        "reference_grounding": "chunk_017_02 paper.md",
        "mode": mode,
        "backbone": backbone,
        "pretrained_source": DEFAULT_SOURCE,
        "protocols": list(inventory.protocols),
        "three_seed_protocol": list(THREE_SEED_PROTOCOL),
        "seeds": list(resolved_seeds),
        "datasets": list(selected_datasets),
        "variants": [asdict(VARIANT_SPECS[name]) for name in selected_variants],
        "methods": OrAdaptersBy().as_dict(),
        "sweeps": {
            "p": list(P_SWEEP),
            "patch_size": list(PATCH_SIZE_SWEEP),
            "alpha": list(ALPHA_SWEEP),
            "gamma": list(GAMMA_SWEEP),
            "similarity_guidance_scale": list(SIMILARITY_GUIDANCE_SCALE_SWEEP),
        },
        "target_mask_size": f"{DEFAULT_TARGET_SIZE[0]}x{DEFAULT_TARGET_SIZE[1]}",
        "coarse_mask_grid": f"{inventory.coarse_mask_grid()[0]}x{inventory.coarse_mask_grid()[1]}",
        "records": records,
        "aggregated": aggregated,
        "objective": inventory_metrics["objective"],
        "score": inventory_metrics["score"],
        "backend_availability": backend_availability(),
        "trend_assertions": [
            "OURS expected to be strongest or competitive among ablation variants",
            "SINGLE-CHANNEL f_mask^s expected to test channel-wise mask capacity",
            "ONLY δ and ONLY f_mask expected to expose loss of complementary mechanism",
        ],
    }

    if write_artifacts:
        result["artifacts"] = write_named_result_artifacts(result, output_dir=output_dir)
    return result


def _table3_rows(aggregated: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_dataset: MutableMapping[str, Dict[str, str]] = {}
    for row in aggregated:
        dataset = str(row["dataset"])
        variant = str(row["variant"])
        by_dataset.setdefault(dataset, {"dataset": dataset})
        by_dataset[dataset][variant] = f"{float(row['mean_accuracy_percent']):.2f} ± {float(row['std_accuracy_percent']):.2f}"
    rows: List[Dict[str, Any]] = []
    for dataset in sorted(by_dataset):
        table_row = {"dataset": dataset}
        for variant in ABLATION_VARIANTS:
            table_row[variant] = by_dataset[dataset].get(variant, "")
        rows.append(table_row)
    return rows


def _mask_summary(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    grouped: MutableMapping[str, List[Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["variant"]), []).append(record)
    summary = {}
    for variant, items in grouped.items():
        summary[variant] = {
            "delta_enabled": VARIANT_SPECS[variant].delta_enabled,
            "f_mask_enabled": VARIANT_SPECS[variant].mask_generator_enabled,
            "channel_mode": VARIANT_SPECS[variant].channel_mode,
            "mask_channels": VARIANT_SPECS[variant].mask_channels,
            "mean_mask_value": float(mean(float(item["mask_mean"]) for item in items)) if items else 0.0,
            "mean_mask_std": float(mean(float(item["mask_std"]) for item in items)) if items else 0.0,
        }
    return summary


def write_named_result_artifacts(
    evaluation: Mapping[str, Any],
    output_dir: Optional[str | os.PathLike[str]] = None,
) -> Dict[str, str]:
    root = _artifact_root(output_dir)
    records = list(evaluation.get("records", []))
    aggregated = list(evaluation.get("aggregated", []))
    table_rows = _table3_rows(aggregated)
    artifact_map = {name: str(root / path) for name, path in ARTIFACT_PATHS.items()}

    _write_json(root / ARTIFACT_PATHS["table3_metrics"], {"records": records, "aggregated": aggregated})
    _write_json(root / ARTIFACT_PATHS["table3_ablation_json"], {"records": records, "aggregated": aggregated})
    _write_json(root / ARTIFACT_PATHS["mask_variant_summary"], _mask_summary(records))
    _write_json(
        root / ARTIFACT_PATHS["mask_statistics"],
        {
            "source": "actual forward mask tensors from bounded SMM ablation route",
            "records": [
                {
                    "dataset": item["dataset"],
                    "variant": item["variant"],
                    "seed": item["seed"],
                    "mask_mean": item["mask_mean"],
                    "mask_std": item["mask_std"],
                    "mask_min": item["mask_min"],
                    "mask_max": item["mask_max"],
                    "mask_channels": item["mask_channels"],
                    "coarse_mask_grid": item["coarse_mask_grid"],
                    "target_mask_size": item["target_mask_size"],
                }
                for item in records
            ],
        },
    )

    aggregate_fields = (
        "dataset",
        "variant",
        "backbone",
        "seeds",
        "mean_accuracy_percent",
        "std_accuracy_percent",
        "mean_loss",
        "std_loss",
        "mean_reward",
        "std_reward",
        "n_seeds",
    )
    _write_csv(root / ARTIFACT_PATHS["table3_ablation_csv"], aggregated, aggregate_fields)

    table_fields = ("dataset",) + ABLATION_VARIANTS
    _write_csv(root / ARTIFACT_PATHS["table3_table"], table_rows, table_fields)
    _write_csv(root / ARTIFACT_PATHS["table3_table_results"], table_rows, table_fields)

    config_payload = {
        "mode": evaluation.get("mode"),
        "backbone": evaluation.get("backbone"),
        "pretrained_source": evaluation.get("pretrained_source"),
        "seeds": evaluation.get("seeds"),
        "datasets": evaluation.get("datasets"),
        "variants": evaluation.get("variants"),
        "sweeps": evaluation.get("sweeps"),
    }
    _write_json(root / ARTIFACT_PATHS["config"], config_payload)
    _write_json(
        root / ARTIFACT_PATHS["run_summary"],
        {
            "protocol": "Table 3 Ablation Studies",
            "backbone": evaluation.get("backbone"),
            "mode": evaluation.get("mode"),
            "num_records": len(records),
            "num_aggregated_rows": len(aggregated),
            "objective": evaluation.get("objective"),
            "score": evaluation.get("score"),
            "trend_assertions": evaluation.get("trend_assertions", []),
        },
    )
    _write_json(
        root / ARTIFACT_PATHS["smoke_metrics"],
        {
            "mode": evaluation.get("mode"),
            "computed_metrics": aggregated,
            "paper_visible_content_is_measured": True,
        },
    )
    _write_json(
        root / ARTIFACT_PATHS["readiness"],
        {
            "ready": True,
            "route": "evaluate_generator_variant_normal_eval",
            "artifacts": artifact_map,
            "backend_availability": evaluation.get("backend_availability", {}),
        },
    )
    _write_json(
        root / ARTIFACT_PATHS["evaluation_result"],
        {
            "status": "completed",
            "mode": evaluation.get("mode"),
            "metric": "mean accuracy % ± std %",
            "aggregated": aggregated,
        },
    )
    return artifact_map


def method_factory(name: str) -> MethodSpec:
    return OrAdaptersBy().get(name)


def variant_factory(name: str) -> MaskVariantSpec:
    if name not in VARIANT_SPECS:
        raise KeyError(f"Unknown variant: {name}")
    return VARIANT_SPECS[name]


def experiment_matrix(
    mode: str = DEFAULT_MODE,
    datasets: Optional[Sequence[str]] = None,
    methods_or_models: Optional[Sequence[str]] = None,
    variants: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[int]] = None,
) -> List[Dict[str, Any]]:
    inv = Inventory(seeds=seed_values(mode=mode, seeds=seeds))
    selected_datasets = tuple(datasets or (("unit-001",) if mode == "runtime_smoke" else inv.datasets))
    selected_methods = tuple(methods_or_models or ("Ours", "PAD", "Narrow", "Medium"))
    selected_variants = tuple(variants or ABLATION_VARIANTS)
    adapters = OrAdaptersBy()
    rows: List[Dict[str, Any]] = []
    for dataset in selected_datasets:
        for method_name in selected_methods:
            method = adapters.get(method_name)
            for variant in selected_variants:
                for seed in inv.seeds:
                    rows.append(
                        {
                            "dataset": dataset,
                            "method": method.name,
                            "backbone": method.backbone,
                            "pretrained_source": method.pretrained_source,
                            "variant": variant,
                            "seed": seed,
                            "mode": mode,
                            "interpolation_level": inv.interpolation_level,
                            "target_mask_size": f"{inv.target_size[0]}x{inv.target_size[1]}",
                            "coarse_mask_grid": f"{inv.coarse_mask_grid()[0]}x{inv.coarse_mask_grid()[1]}",
                            "delta_initialized": "{0}^{d_P}",
                            "phi_mask_generator_parameters": VARIANT_SPECS[variant].train_phi,
                            "p_values": list(inv.p_values),
                            "patch_size_values": list(inv.patch_size_values),
                        }
                    )
    return rows


@dataclass(frozen=True)
class GeneratorVariantNormalEvalConfig:
    mode: str = DEFAULT_MODE
    output_dir: str = "results"
    datasets: Tuple[str, ...] = ("unit-001",)
    variants: Tuple[str, ...] = ABLATION_VARIANTS
    seeds: Tuple[int, ...] = (DEFAULT_SEED,)
    backbone: str = DEFAULT_BACKBONE
    write_artifacts: bool = False
    interpolation_level: int = DEFAULT_INTERPOLATION_LEVEL


def build_generator_variant_normal_eval(
    config: GeneratorVariantNormalEvalConfig | Mapping[str, Any] | None = None,
) -> Inventory:
    cfg = asdict(config) if isinstance(config, GeneratorVariantNormalEvalConfig) else dict(config or {})
    seeds = seed_values(mode=str(cfg.get("mode", DEFAULT_MODE)), seeds=cfg.get("seeds"))
    return Inventory(seeds=seeds, interpolation_level=int(cfg.get("interpolation_level", DEFAULT_INTERPOLATION_LEVEL)))


def evaluate_ours_oradaptersby_inventory(
    config: GeneratorVariantNormalEvalConfig | Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    cfg = asdict(config) if isinstance(config, GeneratorVariantNormalEvalConfig) else dict(config or {})
    result = evaluate_generator_variant_normal_eval(
        mode=str(cfg.get("mode", DEFAULT_MODE)),
        datasets=cfg.get("datasets"),
        variants=cfg.get("variants"),
        seeds=cfg.get("seeds"),
        backbone=str(cfg.get("backbone", DEFAULT_BACKBONE)),
        output_dir=cfg.get("output_dir"),
        write_artifacts=bool(cfg.get("write_artifacts", False)),
        interpolation_level=int(cfg.get("interpolation_level", DEFAULT_INTERPOLATION_LEVEL)),
    )
    result["route_active"] = True
    result["active_symbol"] = "src.generator_variant_normal_eval.evaluate_ours_oradaptersby_inventory"
    result["metrics"] = {
        "aggregated": result.get("aggregated", []),
        "objective": result.get("objective", 0.0),
        "score": result.get("score", 0.0),
    }
    return result


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    import argparse

    parser = argparse.ArgumentParser(description="Run SMM Table 3 ablation route.")
    parser.add_argument("--mode", default=DEFAULT_MODE, choices=("runtime_smoke", "full_run", "full", "paper"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--variant", action="append", dest="variants")
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--no-artifacts", action="store_true")
    args = parser.parse_args(argv)

    return evaluate_generator_variant_normal_eval(
        mode=args.mode,
        datasets=args.datasets,
        variants=args.variants,
        seeds=args.seeds,
        backbone=args.backbone,
        output_dir=args.output_dir,
        write_artifacts=not args.no_artifacts,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "GeneratorVariantNormalEvalConfig",
    "build_generator_variant_normal_eval",
    "evaluate_ours_oradaptersby_inventory",
    "evaluate_generator_variant_normal_eval",
    "AblationRunConfig",
    "Inventory",
    "OrAdaptersBy",
    "Ours",
]
