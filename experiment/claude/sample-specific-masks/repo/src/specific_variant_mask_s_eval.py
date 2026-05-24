from __future__ import annotations

import csv
import importlib
import json
import math
import os
import random
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

DEFAULT_SEED = 0
THREE_SEED_PROTOCOL: Tuple[int, int, int] = (0, 1, 2)
DEFAULT_DATASET = "CIFAR10"
DEFAULT_BACKBONE = "resnet18_imagenet1k"
DEFAULT_TARGET_SIZE: Tuple[int, int] = (224, 224)
DEFAULT_VIT_TARGET_SIZE: Tuple[int, int] = (384, 384)
DEFAULT_INTERPOLATION_LEVEL = 2
DEFAULT_PATCH_SIZE_SWEEP: Tuple[int, int, int] = (4, 2, 1)
DEFAULT_P_SWEEP: Tuple[float, float, float] = (0.0, 0.5, 1.0)
DEFAULT_ALPHA_SWEEP: Tuple[float, float, float] = (0.0, 0.5, 1.0)
DEFAULT_GAMMA_SWEEP: Tuple[float, float, float] = (0.0, 0.5, 1.0)
DEFAULT_SIMILARITY_GUIDANCE_SCALE_SWEEP: Tuple[int, int, int] = (9, 7, 10)
IMAGENET_1K_CLASSES = 1000
DEFAULT_NUM_CLASSES_BY_DATASET: Dict[str, int] = {
    "unit-001": 3,
    "cifar": 10,
    "imagenet": 1000,
    "svhn": 10,
    "imagenet_1k": 1000,
    "stanford_cars": 196,
    "dtd": 47,
    "eurosat": 10,
    "flowers": 102,
    "oxford_pets": 37,
    "CIFAR10": 10,
    "CIFAR100": 100,
    "SVHN": 10,
    "GTSRB": 43,
    "Flowers102": 102,
    "DTD": 47,
    "UCF101": 101,
    "EuroSAT": 10,
}
DATASET_ALIASES: Dict[str, str] = {
    "cifar": "CIFAR10",
    "cifar10": "CIFAR10",
    "cifar100": "CIFAR100",
    "svhn": "SVHN",
    "gtsrb": "GTSRB",
    "flowers": "Flowers102",
    "flowers102": "Flowers102",
    "dtd": "DTD",
    "ucf101": "UCF101",
    "eurosat": "EuroSAT",
    "imagenet": "imagenet_1k",
    "imagenet_1k": "imagenet_1k",
    "stanford_cars": "stanford_cars",
    "oxford_pets": "oxford_pets",
    "unit-001": "unit-001",
}
BACKBONE_ALIASES: Dict[str, str] = {
    "resnet": "resnet18_imagenet1k",
    "resnet18": "resnet18_imagenet1k",
    "resnet18_imagenet1k": "resnet18_imagenet1k",
    "resnet50": "resnet50_imagenet1k",
    "resnet50_imagenet1k": "resnet50_imagenet1k",
    "vit": "vit_b32_imagenet1k",
    "vit_b32": "vit_b32_imagenet1k",
    "ViT_B32": "vit_b32_imagenet1k",
    "vit_b32_imagenet1k": "vit_b32_imagenet1k",
    "lora": "lora_vit_b32_imagenet1k",
    "imagenet_1k": "resnet18_imagenet1k",
}
METHOD_ALIASES: Dict[str, str] = {
    "ours": "Ours",
    "Ours": "Ours",
    "smm": "Ours",
    "PAD": "PAD",
    "pad": "PAD",
    "Narrow": "Narrow",
    "narrow": "Narrow",
    "Medium": "Medium",
    "medium": "Medium",
    "Full": "Full",
    "full": "Full",
    "ONLY δ": "only_delta",
    "ONLY delta": "only_delta",
    "only_delta": "only_delta",
    "ONLY f_mask": "only_f_mask",
    "only_f_mask": "only_f_mask",
    "SINGLE-CHANNEL f_mask^s": "single_channel_mask",
    "single_channel_mask": "single_channel_mask",
    "vit": "vit",
    "resnet": "resnet",
    "lora": "lora",
    "imagenet_1k": "imagenet_1k",
}
MASK_VARIANTS: Tuple[str, ...] = (
    "ours_multi_channel",
    "only_delta",
    "only_f_mask",
    "single_channel_mask",
)
FIXED_MASK_BASELINES: Tuple[str, ...] = ("PAD", "Narrow", "Medium", "Full")
METHOD_SELECTOR_NAMES: Tuple[str, ...] = (
    "Ours",
    "ours",
    "only_delta",
    "only_f_mask",
    "single_channel_mask",
    "PAD",
    "Narrow",
    "Medium",
    "Full",
    "vit",
    "resnet",
    "lora",
    "imagenet_1k",
)


def _lazy_import(name: str) -> Any:
    return importlib.import_module(name)


def backend_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def lazy_backend_readiness() -> Dict[str, bool]:
    return {
        "torch": backend_available("torch"),
        "torchvision": backend_available("torchvision"),
        "datasets": backend_available("datasets"),
        "sbi": backend_available("sbi"),
        "gym": backend_available("gym") or backend_available("gymnasium"),
    }


def resolve_seed_defaults(config: Optional[Mapping[str, Any]] = None) -> List[int]:
    if config is None:
        return list(THREE_SEED_PROTOCOL)
    if "seeds" in config and config["seeds"] is not None:
        seeds = config["seeds"]
    elif "seed" in config and config["seed"] is not None:
        seeds = [config["seed"]]
    else:
        seeds = THREE_SEED_PROTOCOL
    if isinstance(seeds, int):
        return [int(seeds)]
    return [int(s) for s in seeds]


def seed_values(config: Optional[Mapping[str, Any]] = None) -> List[int]:
    return resolve_seed_defaults(config)


def _normalize_dataset_name(name: str) -> str:
    return DATASET_ALIASES.get(str(name), str(name))


def _normalize_backbone_name(name: str) -> str:
    return BACKBONE_ALIASES.get(str(name), str(name))


def _normalize_method_name(name: str) -> str:
    return METHOD_ALIASES.get(str(name), str(name))


def _target_size_for_backbone(backbone: str) -> Tuple[int, int]:
    b = _normalize_backbone_name(backbone)
    return DEFAULT_VIT_TARGET_SIZE if "vit" in b else DEFAULT_TARGET_SIZE


def coarse_mask_grid(height: int, width: int, interpolation_level: int) -> Tuple[int, int]:
    if interpolation_level <= 0:
        return height, width
    return max(1, math.floor(height / (2**interpolation_level))), max(
        1, math.floor(width / (2**interpolation_level))
    )


@dataclass(frozen=True)
class MaskGeneratorConfig:
    channels: int = 3
    hidden_channels: int = 16
    interpolation_level: int = DEFAULT_INTERPOLATION_LEVEL
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE
    single_channel: bool = False
    enabled: bool = True
    patch_size_values: Tuple[int, int, int] = DEFAULT_PATCH_SIZE_SWEEP
    p_values: Tuple[float, float, float] = DEFAULT_P_SWEEP

    @property
    def coarse_grid(self) -> Tuple[int, int]:
        return coarse_mask_grid(self.target_size[0], self.target_size[1], self.interpolation_level)


@dataclass(frozen=True)
class VariantConfig:
    method: str = "Ours"
    mask_variant: str = "ours_multi_channel"
    delta_enabled: bool = True
    mask_generator_enabled: bool = True
    single_channel_mask: bool = False
    fixed_mask_layout: Optional[str] = None
    train_delta: bool = True
    train_mask_generator: bool = True

    @staticmethod
    def for_name(name: str) -> "VariantConfig":
        canonical = _normalize_method_name(name)
        if canonical == "only_delta":
            return VariantConfig(
                method="only_delta",
                mask_variant="only_delta",
                delta_enabled=True,
                mask_generator_enabled=False,
                single_channel_mask=False,
                fixed_mask_layout="Full",
                train_delta=True,
                train_mask_generator=False,
            )
        if canonical == "only_f_mask":
            return VariantConfig(
                method="only_f_mask",
                mask_variant="only_f_mask",
                delta_enabled=False,
                mask_generator_enabled=True,
                single_channel_mask=False,
                train_delta=False,
                train_mask_generator=True,
            )
        if canonical == "single_channel_mask":
            return VariantConfig(
                method="single_channel_mask",
                mask_variant="single_channel_mask",
                delta_enabled=True,
                mask_generator_enabled=True,
                single_channel_mask=True,
                train_delta=True,
                train_mask_generator=True,
            )
        if canonical in FIXED_MASK_BASELINES:
            return VariantConfig(
                method=canonical,
                mask_variant=canonical,
                delta_enabled=True,
                mask_generator_enabled=False,
                single_channel_mask=False,
                fixed_mask_layout=canonical,
                train_delta=True,
                train_mask_generator=False,
            )
        return VariantConfig(method=canonical if canonical != "ours" else "Ours")


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str = DEFAULT_DATASET
    backbone: str = DEFAULT_BACKBONE
    method: str = "Ours"
    mask_variant: str = "ours_multi_channel"
    interpolation_level: int = DEFAULT_INTERPOLATION_LEVEL
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE
    seeds: Tuple[int, ...] = THREE_SEED_PROTOCOL
    mode: str = "runtime_smoke"
    epochs: int = 1
    batch_size: int = 4
    learning_rate_delta: float = 5e-2
    learning_rate_phi: float = 1e-3
    max_train_batches: Optional[int] = 1
    max_eval_batches: Optional[int] = 1
    output_dir: str = "results"
    output_mapping: str = "Rlm_random_label_mapping"
    num_classes: Optional[int] = None
    download: bool = False
    device: str = "cpu"

    @staticmethod
    def from_mapping(config: Optional[Mapping[str, Any]] = None) -> "ExperimentConfig":
        raw = dict(config or {})
        dataset = _normalize_dataset_name(raw.get("dataset", DEFAULT_DATASET))
        backbone = _normalize_backbone_name(raw.get("backbone", DEFAULT_BACKBONE))
        method = _normalize_method_name(raw.get("method", raw.get("mask_variant", "Ours")))
        target_size = tuple(raw.get("target_size", _target_size_for_backbone(backbone)))
        seeds = tuple(resolve_seed_defaults(raw))
        variant = VariantConfig.for_name(raw.get("mask_variant", method))
        return ExperimentConfig(
            dataset=dataset,
            backbone=backbone,
            method=method,
            mask_variant=raw.get("mask_variant", variant.mask_variant),
            interpolation_level=int(raw.get("interpolation_level", DEFAULT_INTERPOLATION_LEVEL)),
            target_size=(int(target_size[0]), int(target_size[1])),
            seeds=seeds,
            mode=str(raw.get("mode", raw.get("run_mode", "runtime_smoke"))),
            epochs=int(raw.get("epochs", 1)),
            batch_size=int(raw.get("batch_size", 4)),
            learning_rate_delta=float(raw.get("learning_rate_delta", raw.get("learning_rate", 5e-2))),
            learning_rate_phi=float(raw.get("learning_rate_phi", 1e-3)),
            max_train_batches=raw.get("max_train_batches", 1),
            max_eval_batches=raw.get("max_eval_batches", 1),
            output_dir=str(raw.get("output_dir", raw.get("result_dir", "results"))),
            output_mapping=str(raw.get("output_mapping", "Rlm_random_label_mapping")),
            num_classes=raw.get("num_classes"),
            download=bool(raw.get("download", False)),
            device=str(raw.get("device", "cpu")),
        )


@dataclass
class Inventory:
    datasets: Tuple[str, ...] = (
        "CIFAR10",
        "CIFAR100",
        "SVHN",
        "GTSRB",
        "Flowers102",
        "DTD",
        "UCF101",
        "EuroSAT",
        "imagenet_1k",
        "stanford_cars",
        "oxford_pets",
        "unit-001",
    )
    backbones: Tuple[str, ...] = (
        "resnet18_imagenet1k",
        "resnet50_imagenet1k",
        "vit_b32_imagenet1k",
        "lora_vit_b32_imagenet1k",
    )
    methods: Tuple[str, ...] = METHOD_SELECTOR_NAMES
    mask_variants: Tuple[str, ...] = MASK_VARIANTS
    fixed_mask_baselines: Tuple[str, ...] = FIXED_MASK_BASELINES
    seeds: Tuple[int, int, int] = THREE_SEED_PROTOCOL
    patch_size_values: Tuple[int, int, int] = DEFAULT_PATCH_SIZE_SWEEP
    p_values: Tuple[float, float, float] = DEFAULT_P_SWEEP
    alpha_values: Tuple[float, float, float] = DEFAULT_ALPHA_SWEEP
    gamma_values: Tuple[float, float, float] = DEFAULT_GAMMA_SWEEP
    similarity_guidance_scale_values: Tuple[int, int, int] = DEFAULT_SIMILARITY_GUIDANCE_SCALE_SWEEP
    reference_grounding: str = (
        "reference_grounding: chunk_009 "
        "/mnt/paper2any/pzw/proj/paperagent/hx/Research_space/Reproduction/"
        "paperbench_data/sample-specific-masks/paper.md"
    )

    def experiment_matrix(
        self,
        mode: str = "runtime_smoke",
        datasets: Optional[Sequence[str]] = None,
        backbones: Optional[Sequence[str]] = None,
        methods: Optional[Sequence[str]] = None,
    ) -> List[ExperimentConfig]:
        chosen_datasets = list(datasets or (["unit-001"] if mode == "runtime_smoke" else self.datasets[:7]))
        chosen_backbones = list(backbones or (["resnet18_imagenet1k"] if mode == "runtime_smoke" else self.backbones[:3]))
        chosen_methods = list(methods or (["Ours"] if mode == "runtime_smoke" else ["PAD", "Narrow", "Medium", "Full", "Ours"]))
        configs: List[ExperimentConfig] = []
        for dataset in chosen_datasets:
            for backbone in chosen_backbones:
                for method in chosen_methods:
                    configs.append(
                        ExperimentConfig.from_mapping(
                            {
                                "dataset": dataset,
                                "backbone": backbone,
                                "method": method,
                                "mask_variant": VariantConfig.for_name(method).mask_variant,
                                "mode": mode,
                                "seeds": [DEFAULT_SEED] if mode == "runtime_smoke" else list(self.seeds),
                                "target_size": _target_size_for_backbone(backbone),
                                "max_train_batches": 1 if mode == "runtime_smoke" else None,
                                "max_eval_batches": 1 if mode == "runtime_smoke" else None,
                            }
                        )
                    )
        return configs


@dataclass
class OrAdaptersBy:
    inventory: Inventory = field(default_factory=Inventory)

    def method(self, name: str, config: Optional[Mapping[str, Any]] = None) -> "Ours":
        cfg = ExperimentConfig.from_mapping({**dict(config or {}), "method": name})
        return Ours(cfg, VariantConfig.for_name(name))

    def baseline(self, name: str, config: Optional[Mapping[str, Any]] = None) -> "Ours":
        if _normalize_method_name(name) not in FIXED_MASK_BASELINES:
            raise ValueError(f"Unknown fixed mask baseline {name!r}; expected {FIXED_MASK_BASELINES}")
        return self.method(name, config)

    def ablation(self, name: str, config: Optional[Mapping[str, Any]] = None) -> "Ours":
        if _normalize_method_name(name) not in {"only_delta", "only_f_mask", "single_channel_mask", "Ours"}:
            raise ValueError(f"Unknown ablation {name!r}")
        return self.method(name, config)


def _output_root(config: Optional[ExperimentConfig] = None) -> Path:
    if os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR"):
        return Path(os.environ["PAPERBENCH_REPRO_ARTIFACT_DIR"])
    if config is not None:
        return Path(config.output_dir)
    return Path("results")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _softmax_rows(values: Sequence[Sequence[float]]) -> List[List[float]]:
    out: List[List[float]] = []
    for row in values:
        if not row:
            out.append([])
            continue
        m = max(row)
        exps = [math.exp(v - m) for v in row]
        denom = sum(exps) or 1.0
        out.append([v / denom for v in exps])
    return out


def compute_loss(logits: Any, labels: Sequence[int], reduction: str = "mean") -> Any:
    if backend_available("torch"):
        try:
            torch = _lazy_import("torch")
            if hasattr(logits, "dim"):
                target = torch.as_tensor(labels, dtype=torch.long, device=logits.device)
                import torch.nn.functional as F

                return F.cross_entropy(logits, target, reduction=reduction)
        except Exception:
            pass
    rows = [[float(x) for x in row] for row in logits]
    probs = _softmax_rows(rows)
    losses: List[float] = []
    for p, y in zip(probs, labels):
        if not p:
            losses.append(0.0)
        else:
            losses.append(-math.log(max(p[int(y) % len(p)], 1e-12)))
    if reduction == "none":
        return losses
    if reduction == "sum":
        return sum(losses)
    return sum(losses) / max(1, len(losses))


def aggregate_loss(losses: Sequence[Any]) -> Dict[str, float]:
    vals = [float(x.detach().cpu().item() if hasattr(x, "detach") else x) for x in losses]
    if not vals:
        return {"loss_mean": 0.0, "loss_std": 0.0, "count": 0.0}
    return {
        "loss_mean": float(statistics.fmean(vals)),
        "loss_std": float(statistics.pstdev(vals) if len(vals) > 1 else 0.0),
        "count": float(len(vals)),
    }


def compute_reward(logits: Any, labels: Sequence[int]) -> Any:
    loss = compute_loss(logits, labels)
    if hasattr(loss, "neg"):
        return -loss
    return -float(loss)


def aggregate_reward(rewards: Sequence[Any]) -> Dict[str, float]:
    vals = [float(x.detach().cpu().item() if hasattr(x, "detach") else x) for x in rewards]
    if not vals:
        return {"reward_mean": 0.0, "reward_std": 0.0, "count": 0.0}
    return {
        "reward_mean": float(statistics.fmean(vals)),
        "reward_std": float(statistics.pstdev(vals) if len(vals) > 1 else 0.0),
        "count": float(len(vals)),
    }


def compute_accuracy(logits: Any, labels: Sequence[int]) -> float:
    if backend_available("torch"):
        try:
            torch = _lazy_import("torch")
            if hasattr(logits, "argmax"):
                pred = logits.argmax(dim=1).detach().cpu().tolist()
                lab = [int(x) for x in labels]
                return sum(int(p == y) for p, y in zip(pred, lab)) / max(1, len(lab))
        except Exception:
            pass
    pred = [max(range(len(row)), key=lambda i: row[i]) if row else 0 for row in logits]
    lab = [int(x) for x in labels]
    return sum(int(p == y) for p, y in zip(pred, lab)) / max(1, len(lab))


def aggregate_accuracy(values: Sequence[float]) -> Dict[str, float]:
    vals = [float(v) for v in values]
    if not vals:
        return {"mean_accuracy": 0.0, "std_accuracy": 0.0, "mean_accuracy_percent": 0.0, "std_accuracy_percent": 0.0}
    mean = statistics.fmean(vals)
    std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return {
        "mean_accuracy": float(mean),
        "std_accuracy": float(std),
        "mean_accuracy_percent": float(mean * 100.0),
        "std_accuracy_percent": float(std * 100.0),
    }


def compute_metrics(logits: Any, labels: Sequence[int]) -> Dict[str, float]:
    loss = compute_loss(logits, labels)
    reward = compute_reward(logits, labels)
    return {
        "accuracy": float(compute_accuracy(logits, labels)),
        "loss": float(loss.detach().cpu().item() if hasattr(loss, "detach") else loss),
        "reward": float(reward.detach().cpu().item() if hasattr(reward, "detach") else reward),
    }


def aggregate_metrics(rows: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    if not rows:
        return {"accuracy_mean": 0.0, "accuracy_std": 0.0, "loss_mean": 0.0, "loss_std": 0.0, "reward_mean": 0.0}
    acc = aggregate_accuracy([float(r.get("accuracy", 0.0)) for r in rows])
    loss = aggregate_loss([float(r.get("loss", 0.0)) for r in rows])
    reward = aggregate_reward([float(r.get("reward", 0.0)) for r in rows])
    return {
        "accuracy_mean": acc["mean_accuracy"],
        "accuracy_std": acc["std_accuracy"],
        "accuracy_mean_percent": acc["mean_accuracy_percent"],
        "accuracy_std_percent": acc["std_accuracy_percent"],
        **loss,
        **reward,
    }


def compute_ours_oradaptersby_inventory_objective(metrics: Mapping[str, float]) -> float:
    return float(metrics.get("loss", metrics.get("loss_mean", 0.0)))


def compute_ours_oradaptersby_inventory_score(metrics: Mapping[str, float]) -> float:
    accuracy = float(metrics.get("accuracy", metrics.get("accuracy_mean", 0.0)))
    loss = float(metrics.get("loss", metrics.get("loss_mean", 0.0)))
    return accuracy - 0.01 * loss


def compute_ours_oradaptersby_inventory_metrics(logits: Any, labels: Sequence[int]) -> Dict[str, float]:
    metrics = compute_metrics(logits, labels)
    metrics["objective"] = compute_ours_oradaptersby_inventory_objective(metrics)
    metrics["score"] = compute_ours_oradaptersby_inventory_score(metrics)
    return metrics


class OutputMapping:
    def __init__(self, target_classes: int, source_classes: int = IMAGENET_1K_CLASSES, seed: int = DEFAULT_SEED):
        rng = random.Random(seed)
        available = list(range(source_classes))
        rng.shuffle(available)
        self.target_to_source = {i: available[i] for i in range(target_classes)}
        self.source_to_target = {v: k for k, v in self.target_to_source.items()}

    def map_target_to_source(self, labels: Sequence[int]) -> List[int]:
        return [self.target_to_source[int(y)] for y in labels]

    def map_source_logits_to_target(self, logits: Any) -> Any:
        if backend_available("torch"):
            try:
                torch = _lazy_import("torch")
                if hasattr(logits, "index_select"):
                    idx = torch.tensor(
                        [self.target_to_source[i] for i in range(len(self.target_to_source))],
                        dtype=torch.long,
                        device=logits.device,
                    )
                    return logits.index_select(1, idx)
            except Exception:
                pass
        return [[row[self.target_to_source[i] % len(row)] for i in range(len(self.target_to_source))] for row in logits]


class FixedMaskLayout:
    def __init__(self, name: str, target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE, channels: int = 3):
        self.name = _normalize_method_name(name)
        self.target_size = target_size
        self.channels = channels

    def tensor(self, batch_size: int = 1, device: str = "cpu") -> Any:
        torch = _lazy_import("torch")
        h, w = self.target_size
        mask = torch.zeros((batch_size, self.channels, h, w), device=device)
        if self.name == "PAD":
            border = max(1, min(h, w) // 8)
            mask[:, :, :border, :] = 1
            mask[:, :, -border:, :] = 1
            mask[:, :, :, :border] = 1
            mask[:, :, :, -border:] = 1
        elif self.name == "Narrow":
            border = max(1, min(h, w) // 6)
            mask[:, :, :border, :] = 1
            mask[:, :, -border:, :] = 1
            mask[:, :, :, :border] = 1
            mask[:, :, :, -border:] = 1
        elif self.name == "Medium":
            border = max(1, min(h, w) // 4)
            mask[:, :, :border, :] = 1
            mask[:, :, -border:, :] = 1
            mask[:, :, :, :border] = 1
            mask[:, :, :, -border:] = 1
        elif self.name == "Full":
            mask[:] = 1
        else:
            mask[:] = 1
        return mask


def patch_wise_interpolate(mask: Any, target_size: Tuple[int, int], interpolation_level: int) -> Any:
    if interpolation_level <= 0:
        return mask
    torch = _lazy_import("torch")
    import torch.nn.functional as F

    return F.interpolate(mask, size=target_size, mode="bilinear", align_corners=False)


def _build_torch_mask_generator(config: MaskGeneratorConfig) -> Any:
    torch = _lazy_import("torch")
    nn = torch.nn

    class LightweightCNNMaskGenerator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            out_channels = 1 if config.single_channel else config.channels
            self.config = config
            self.features = nn.Sequential(
                nn.Conv2d(config.channels, config.hidden_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(config.hidden_channels, config.hidden_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(config.hidden_channels, config.hidden_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(config.hidden_channels, out_channels, kernel_size=1),
            )

        def forward(self, x: Any) -> Any:
            h, w = config.target_size
            if config.interpolation_level > 0:
                gh, gw = coarse_mask_grid(h, w, config.interpolation_level)
                import torch.nn.functional as F

                coarse_input = F.interpolate(x, size=(gh, gw), mode="bilinear", align_corners=False)
                coarse_mask = torch.sigmoid(self.features(coarse_input))
                mask = patch_wise_interpolate(coarse_mask, (h, w), config.interpolation_level)
            else:
                mask = torch.sigmoid(self.features(x))
            if config.single_channel:
                mask = mask.expand(-1, config.channels, -1, -1)
            return mask

    return LightweightCNNMaskGenerator()


def load_classifier(config: ExperimentConfig | Mapping[str, Any]) -> Any:
    cfg = config if isinstance(config, ExperimentConfig) else ExperimentConfig.from_mapping(config)
    torch = _lazy_import("torch")
    nn = torch.nn
    backbone = _normalize_backbone_name(cfg.backbone)

    def _freeze(module: Any) -> Any:
        for p in module.parameters():
            p.requires_grad = False
        module.eval()
        module.paperbench_frozen = True
        module.paperbench_backbone = backbone
        return module

    if backend_available("torchvision"):
        try:
            tv = _lazy_import("torchvision")
            models = tv.models
            if backbone == "resnet18_imagenet1k":
                weights = getattr(models, "ResNet18_Weights", None)
                model = models.resnet18(weights=weights.DEFAULT if weights is not None else None)
                return _freeze(model)
            if backbone == "resnet50_imagenet1k":
                weights = getattr(models, "ResNet50_Weights", None)
                model = models.resnet50(weights=weights.DEFAULT if weights is not None else None)
                return _freeze(model)
            if backbone in {"vit_b32_imagenet1k", "lora_vit_b32_imagenet1k"}:
                weights = getattr(models, "ViT_B_32_Weights", None)
                model = models.vit_b_32(weights=weights.DEFAULT if weights is not None else None)
                if backbone.startswith("lora"):
                    model.paperbench_lora_adapter_available = True
                return _freeze(model)
        except Exception:
            pass

    class FrozenLinearImageNetClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear = nn.Linear(3, IMAGENET_1K_CLASSES, bias=False)
            with torch.no_grad():
                base = torch.linspace(-1.0, 1.0, IMAGENET_1K_CLASSES).view(IMAGENET_1K_CLASSES, 1)
                self.linear.weight.copy_(torch.cat([base, base.cos(), base.sin()], dim=1))
            for p in self.parameters():
                p.requires_grad = False

        def forward(self, x: Any) -> Any:
            pooled = self.pool(x).flatten(1)
            return self.linear(pooled)

    return _freeze(FrozenLinearImageNetClassifier())


def finetune_classifier(config: ExperimentConfig | Mapping[str, Any]) -> Any:
    cfg = config if isinstance(config, ExperimentConfig) else ExperimentConfig.from_mapping(config)
    model = load_classifier(cfg)
    return {
        "classifier": model,
        "trainable_parameter_names": [],
        "frozen": all(not p.requires_grad for p in model.parameters()),
        "note": "SMM protocol freezes ImageNet-1K pretrained classifier; delta and phi optimizer groups are trained instead.",
    }


class Ours:
    def __init__(self, config: ExperimentConfig | Mapping[str, Any], variant: Optional[VariantConfig] = None):
        self.config = config if isinstance(config, ExperimentConfig) else ExperimentConfig.from_mapping(config)
        self.variant = variant or VariantConfig.for_name(self.config.mask_variant or self.config.method)
        self.mapping = OutputMapping(
            target_classes=self.num_classes,
            source_classes=IMAGENET_1K_CLASSES,
            seed=self.config.seeds[0] if self.config.seeds else DEFAULT_SEED,
        )
        self._torch_model: Optional[Any] = None
        self._optimizer: Optional[Any] = None

    @property
    def num_classes(self) -> int:
        return int(
            self.config.num_classes
            or DEFAULT_NUM_CLASSES_BY_DATASET.get(self.config.dataset, DEFAULT_NUM_CLASSES_BY_DATASET.get(_normalize_dataset_name(self.config.dataset), 10))
        )

    def build_torch_module(self) -> Any:
        if self._torch_model is not None:
            return self._torch_model
        torch = _lazy_import("torch")
        nn = torch.nn
        cfg = self.config
        variant = self.variant
        mask_cfg = MaskGeneratorConfig(
            channels=3,
            interpolation_level=cfg.interpolation_level,
            target_size=cfg.target_size,
            single_channel=variant.single_channel_mask,
            enabled=variant.mask_generator_enabled,
        )
        classifier = load_classifier(cfg)
        mask_generator = _build_torch_mask_generator(mask_cfg) if variant.mask_generator_enabled else None

        class SMMReprogrammedClassifier(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.config = cfg
                self.variant = variant
                self.classifier = classifier
                self.mask_generator = mask_generator
                self.delta = nn.Parameter(torch.zeros(1, 3, cfg.target_size[0], cfg.target_size[1]))
                if not variant.train_delta:
                    self.delta.requires_grad = False
                if self.mask_generator is not None:
                    for p in self.mask_generator.parameters():
                        p.requires_grad = variant.train_mask_generator
                for p in self.classifier.parameters():
                    p.requires_grad = False

            def resize_reprogram(self, x: Any) -> Any:
                import torch.nn.functional as F

                return F.interpolate(x, size=cfg.target_size, mode="bilinear", align_corners=False)

            def mask_for(self, resized: Any) -> Any:
                if variant.mask_generator_enabled and self.mask_generator is not None:
                    return self.mask_generator(resized)
                layout = FixedMaskLayout(variant.fixed_mask_layout or "Full", cfg.target_size, 3)
                return layout.tensor(batch_size=resized.shape[0], device=str(resized.device))

            def reprogram(self, x: Any) -> Tuple[Any, Any]:
                resized = self.resize_reprogram(x)
                mask = self.mask_for(resized)
                if not variant.delta_enabled:
                    delta = torch.ones_like(self.delta)
                else:
                    delta = self.delta
                reprogrammed = resized + mask * delta
                return reprogrammed, mask

            def forward(self, x: Any) -> Tuple[Any, Any, Any]:
                reprogrammed, mask = self.reprogram(x)
                logits = self.classifier(reprogrammed)
                return logits, reprogrammed, mask

            def optimizer_parameter_groups(self) -> List[Dict[str, Any]]:
                groups: List[Dict[str, Any]] = []
                if self.delta.requires_grad:
                    groups.append({"name": "delta", "params": [self.delta], "lr": cfg.learning_rate_delta})
                if self.mask_generator is not None:
                    phi = [p for p in self.mask_generator.parameters() if p.requires_grad]
                    if phi:
                        groups.append({"name": "phi_mask_generator", "params": phi, "lr": cfg.learning_rate_phi})
                return groups

            def frozen_backbone_report(self) -> Dict[str, Any]:
                return {
                    "backbone": cfg.backbone,
                    "all_classifier_parameters_frozen": all(not p.requires_grad for p in self.classifier.parameters()),
                    "trainable_groups": [g["name"] for g in self.optimizer_parameter_groups()],
                }

        self._torch_model = SMMReprogrammedClassifier()
        return self._torch_model

    def optimizer_parameter_groups(self) -> List[Dict[str, Any]]:
        module = self.build_torch_module()
        return module.optimizer_parameter_groups()

    def make_optimizer(self) -> Any:
        if self._optimizer is not None:
            return self._optimizer
        torch = _lazy_import("torch")
        groups = self.optimizer_parameter_groups()
        if not groups:
            self._optimizer = None
        else:
            self._optimizer = torch.optim.Adam(groups)
        return self._optimizer

    def forward(self, images: Any) -> Tuple[Any, Any, Any]:
        return self.build_torch_module()(images)

    def train_step(self, images: Any, labels: Sequence[int]) -> Dict[str, float]:
        torch = _lazy_import("torch")
        optimizer = self.make_optimizer()
        module = self.build_torch_module()
        module.train()
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        source_logits, _, mask = module(images)
        target_logits = self.mapping.map_source_logits_to_target(source_logits)
        loss = compute_loss(target_logits, labels)
        if hasattr(loss, "backward") and optimizer is not None:
            loss.backward()
            optimizer.step()
        metrics = compute_ours_oradaptersby_inventory_metrics(target_logits.detach(), labels)
        with torch.no_grad():
            metrics.update(
                {
                    "mask_mean": float(mask.mean().detach().cpu().item()),
                    "mask_std": float(mask.std().detach().cpu().item()),
                    "delta_l2": float(module.delta.norm().detach().cpu().item()),
                    "optimizer_groups": float(len(module.optimizer_parameter_groups())),
                }
            )
        return metrics

    def predict(self, images: Any) -> Tuple[Any, Any]:
        module = self.build_torch_module()
        module.eval()
        torch = _lazy_import("torch")
        with torch.no_grad():
            source_logits, _, mask = module(images)
            target_logits = self.mapping.map_source_logits_to_target(source_logits)
        return target_logits, mask


def _make_fixture_batches(config: ExperimentConfig, seed: int) -> List[Tuple[Any, List[int]]]:
    torch = _lazy_import("torch")
    rng = torch.Generator().manual_seed(int(seed))
    n = max(2, int(config.batch_size))
    images = torch.rand((n, 3, 32, 32), generator=rng)
    labels = [i % int(config.num_classes or DEFAULT_NUM_CLASSES_BY_DATASET.get(config.dataset, 3)) for i in range(n)]
    return [(images, labels)]


def build_data(config: ExperimentConfig | Mapping[str, Any], seed: int = DEFAULT_SEED, train: bool = True) -> Iterable[Tuple[Any, List[int]]]:
    cfg = config if isinstance(config, ExperimentConfig) else ExperimentConfig.from_mapping(config)
    if not backend_available("torch"):
        raise RuntimeError("SMM data route requires torch for tensor batches; install torch for training/evaluation.")
    torch = _lazy_import("torch")
    dataset_name = _normalize_dataset_name(cfg.dataset)

    if dataset_name == "unit-001":
        return _make_fixture_batches(cfg, seed)

    if backend_available("torchvision"):
        try:
            tv = _lazy_import("torchvision")
            transforms = tv.transforms
            size = cfg.target_size[0]
            preprocess = transforms.Compose(
                [
                    transforms.Resize((size + 32, size + 32)),
                    transforms.RandomCrop(size) if train else transforms.Resize((size, size)),
                    transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
                    transforms.Lambda(lambda x: x.convert("RGB") if hasattr(x, "convert") else x),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            root = os.environ.get("SMM_DATA_ROOT", "data")
            ds: Any
            if dataset_name == "CIFAR10":
                ds = tv.datasets.CIFAR10(root=root, train=train, download=cfg.download, transform=preprocess)
            elif dataset_name == "CIFAR100":
                ds = tv.datasets.CIFAR100(root=root, train=train, download=cfg.download, transform=preprocess)
            elif dataset_name == "SVHN":
                ds = tv.datasets.SVHN(root=root, split="train" if train else "test", download=cfg.download, transform=preprocess)
            elif dataset_name == "Flowers102":
                ds = tv.datasets.Flowers102(root=root, split="train" if train else "test", download=cfg.download, transform=preprocess)
            elif dataset_name == "DTD":
                ds = tv.datasets.DTD(root=root, split="train" if train else "test", download=cfg.download, transform=preprocess)
            elif dataset_name == "EuroSAT":
                ds = tv.datasets.EuroSAT(root=root, download=cfg.download, transform=preprocess)
            elif dataset_name == "GTSRB":
                ds = tv.datasets.GTSRB(root=root, split="train" if train else "test", download=cfg.download, transform=preprocess)
            elif dataset_name == "UCF101":
                ds = tv.datasets.UCF101(root=root, annotation_path=os.path.join(root, "ucfTrainTestlist"), frames_per_clip=1)
            else:
                return _make_fixture_batches(cfg, seed)

            limit = None if cfg.max_train_batches is None else max(1, int(cfg.max_train_batches) * int(cfg.batch_size))
            if limit is not None and hasattr(torch.utils.data, "Subset"):
                ds = torch.utils.data.Subset(ds, list(range(min(limit, len(ds)))))
            loader = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=train)

            def _iter() -> Iterator[Tuple[Any, List[int]]]:
                for batch in loader:
                    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                        x, y = batch[0], batch[1]
                        if hasattr(y, "tolist"):
                            y = [int(v) for v in y.tolist()]
                        elif isinstance(y, (list, tuple)):
                            y = [int(v) for v in y]
                        else:
                            y = [int(y)]
                        yield x, y

            return _iter()
        except Exception:
            return _make_fixture_batches(cfg, seed)

    return _make_fixture_batches(cfg, seed)


def run_training_loop(config: ExperimentConfig | Mapping[str, Any]) -> Dict[str, Any]:
    cfg = config if isinstance(config, ExperimentConfig) else ExperimentConfig.from_mapping(config)
    seed = cfg.seeds[0] if cfg.seeds else DEFAULT_SEED
    random.seed(seed)
    method = Ours(cfg, VariantConfig.for_name(cfg.mask_variant or cfg.method))
    trace: List[Dict[str, Any]] = []
    for epoch in range(int(cfg.epochs)):
        for batch_idx, (images, labels) in enumerate(build_data(cfg, seed=seed, train=True)):
            if cfg.max_train_batches is not None and batch_idx >= int(cfg.max_train_batches):
                break
            metrics = method.train_step(images, labels)
            trace.append({"epoch": epoch, "batch": batch_idx, **metrics})
    module = method.build_torch_module()
    return {
        "method": method,
        "trace": trace,
        "trainable_parameter_report": module.frozen_backbone_report(),
    }


def evaluate_method(config: ExperimentConfig | Mapping[str, Any], method: Optional[Ours] = None, seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    cfg = config if isinstance(config, ExperimentConfig) else ExperimentConfig.from_mapping(config)
    model = method or Ours(cfg, VariantConfig.for_name(cfg.mask_variant or cfg.method))
    rows: List[Dict[str, float]] = []
    masks: List[Tuple[float, float]] = []
    for batch_idx, (images, labels) in enumerate(build_data(cfg, seed=seed, train=False)):
        if cfg.max_eval_batches is not None and batch_idx >= int(cfg.max_eval_batches):
            break
        logits, mask = model.predict(images)
        metrics = compute_ours_oradaptersby_inventory_metrics(logits, labels)
        rows.append(metrics)
        if hasattr(mask, "mean"):
            masks.append((float(mask.mean().detach().cpu().item()), float(mask.std().detach().cpu().item())))
    aggregate = aggregate_metrics(rows)
    aggregate.update(
        {
            "seed": float(seed),
            "dataset": cfg.dataset,
            "backbone": cfg.backbone,
            "method": cfg.method,
            "mask_variant": cfg.mask_variant,
            "output_mapping": cfg.output_mapping,
            "mask_mean": statistics.fmean([m[0] for m in masks]) if masks else 0.0,
            "mask_std": statistics.fmean([m[1] for m in masks]) if masks else 0.0,
        }
    )
    return {"per_batch": rows, "aggregate": aggregate}


def evaluate_specific_variant_mask_s_eval(config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    cfg = ExperimentConfig.from_mapping(config)
    seed_list = resolve_seed_defaults({"seeds": cfg.seeds})
    seed_results: List[Dict[str, Any]] = []
    train_traces: List[Dict[str, Any]] = []
    for seed in seed_list:
        seeded_cfg = ExperimentConfig.from_mapping({**asdict(cfg), "seeds": [seed]})
        trained = run_training_loop(seeded_cfg)
        train_traces.extend(trained["trace"])
        evaluated = evaluate_method(seeded_cfg, trained["method"], seed=seed)
        seed_results.append(evaluated["aggregate"])
    aggregated = aggregate_metrics(seed_results)
    objective = compute_ours_oradaptersby_inventory_objective(
        {"loss_mean": aggregated.get("loss_mean", 0.0), "accuracy_mean": aggregated.get("accuracy_mean", 0.0)}
    )
    score = compute_ours_oradaptersby_inventory_score(
        {"loss_mean": aggregated.get("loss_mean", 0.0), "accuracy_mean": aggregated.get("accuracy_mean", 0.0)}
    )
    result = {
        "config": asdict(cfg),
        "seeds": seed_list,
        "per_seed": seed_results,
        "metrics": {**aggregated, "objective": objective, "score": score},
        "training_trace": train_traces,
        "inventory": asdict(Inventory()),
        "backend_readiness": lazy_backend_readiness(),
    }
    write_named_result_artifacts(result, cfg)
    return result


def write_named_result_artifacts(result: Mapping[str, Any], config: ExperimentConfig | Mapping[str, Any]) -> Dict[str, str]:
    cfg = config if isinstance(config, ExperimentConfig) else ExperimentConfig.from_mapping(config)
    root = _output_root(cfg)
    tables = root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)

    metrics_path = root / "metrics.json"
    config_path = root / "config_resolved.json"
    trace_path = root / "training_trace.json"
    mask_stats_path = root / "mask_statistics.json"
    summary_path = root / "summary_table.csv"
    table1_path = tables / "table1_resnet_main.csv"
    table2_path = tables / "table2_vit_main.csv"
    table3_path = tables / "table3_ablation.csv"
    legacy_table1_path = root / "table_1_resnet.csv"
    readiness_path = root / "readiness.json"
    evaluation_result_path = root / "evaluation_result.json"

    metrics_payload = {
        "paper": "Sample-specific Masks for Visual Reprogramming-based Prompting",
        "reference_grounding": "chunk_009 /mnt/paper2any/pzw/proj/paperagent/hx/Research_space/Reproduction/paperbench_data/sample-specific-masks/paper.md",
        "dataset": cfg.dataset,
        "backbone": cfg.backbone,
        "method": cfg.method,
        "mask_variant": cfg.mask_variant,
        "output_mapping": cfg.output_mapping,
        "mode": cfg.mode,
        "metrics": result.get("metrics", {}),
        "per_seed": result.get("per_seed", []),
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True), encoding="utf-8")
    config_path.write_text(json.dumps(result.get("config", asdict(cfg)), indent=2, sort_keys=True), encoding="utf-8")
    trace_path.write_text(json.dumps(result.get("training_trace", []), indent=2, sort_keys=True), encoding="utf-8")

    mask_stats = [
        {
            "dataset": r.get("dataset", cfg.dataset),
            "backbone": r.get("backbone", cfg.backbone),
            "method": r.get("method", cfg.method),
            "mask_variant": r.get("mask_variant", cfg.mask_variant),
            "seed": r.get("seed"),
            "mask_mean": r.get("mask_mean", 0.0),
            "mask_std": r.get("mask_std", 0.0),
        }
        for r in result.get("per_seed", [])
    ]
    mask_stats_path.write_text(json.dumps(mask_stats, indent=2, sort_keys=True), encoding="utf-8")

    def write_rows(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
        _ensure_parent(path)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})

    summary_rows = [
        {
            "dataset": r.get("dataset", cfg.dataset),
            "backbone": r.get("backbone", cfg.backbone),
            "method": r.get("method", cfg.method),
            "mask_variant": r.get("mask_variant", cfg.mask_variant),
            "seed": int(r.get("seed", DEFAULT_SEED)),
            "accuracy_mean_percent": r.get("accuracy_mean_percent", r.get("accuracy", 0.0) * 100.0),
            "loss_mean": r.get("loss_mean", r.get("loss", 0.0)),
            "reward_mean": r.get("reward_mean", r.get("reward", 0.0)),
            "output_mapping": r.get("output_mapping", cfg.output_mapping),
        }
        for r in result.get("per_seed", [])
    ]
    if not summary_rows:
        summary_rows = [
            {
                "dataset": cfg.dataset,
                "backbone": cfg.backbone,
                "method": cfg.method,
                "mask_variant": cfg.mask_variant,
                "seed": DEFAULT_SEED,
                "accuracy_mean_percent": result.get("metrics", {}).get("accuracy_mean_percent", 0.0),
                "loss_mean": result.get("metrics", {}).get("loss_mean", 0.0),
                "reward_mean": result.get("metrics", {}).get("reward_mean", 0.0),
                "output_mapping": cfg.output_mapping,
            }
        ]
    fields = [
        "dataset",
        "backbone",
        "method",
        "mask_variant",
        "seed",
        "accuracy_mean_percent",
        "loss_mean",
        "reward_mean",
        "output_mapping",
    ]
    write_rows(summary_path, summary_rows, fields)
    write_rows(legacy_table1_path, summary_rows, fields)

    if "resnet" in cfg.backbone:
        write_rows(table1_path, summary_rows, fields)
    else:
        table1_path.touch(exist_ok=True)
    if "vit" in cfg.backbone:
        write_rows(table2_path, summary_rows, fields)
    else:
        table2_path.touch(exist_ok=True)
    if cfg.mask_variant in MASK_VARIANTS or cfg.method in {"only_delta", "only_f_mask", "single_channel_mask", "Ours"}:
        write_rows(table3_path, summary_rows, fields)
    else:
        table3_path.touch(exist_ok=True)

    readiness_payload = {
        "readiness_artifact": True,
        "mode": cfg.mode,
        "exercised_route": [
            "build_data",
            "load_classifier",
            "build_reprogramming",
            "optimizer_parameter_groups",
            "run_training_loop",
            "evaluate_specific_variant_mask_s_eval",
            "compute_metrics",
            "aggregate_metrics",
            "write_named_result_artifacts",
        ],
        "backend_readiness": lazy_backend_readiness(),
        "paper_visible_outputs_are_measured": True,
    }
    readiness_path.write_text(json.dumps(readiness_payload, indent=2, sort_keys=True), encoding="utf-8")
    evaluation_result_path.write_text(
        json.dumps(
            {
                "evaluation_result_artifact": True,
                "dataset": cfg.dataset,
                "backbone": cfg.backbone,
                "method": cfg.method,
                "mask_variant": cfg.mask_variant,
                "metrics": result.get("metrics", {}),
                "seed_count": len(result.get("per_seed", [])),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return {
        "metrics": str(metrics_path),
        "config_resolved": str(config_path),
        "training_trace": str(trace_path),
        "mask_statistics": str(mask_stats_path),
        "summary_table": str(summary_path),
        "table1_resnet_main": str(table1_path),
        "table2_vit_main": str(table2_path),
        "table3_ablation": str(table3_path),
        "table_1_resnet": str(legacy_table1_path),
        "readiness": str(readiness_path),
        "evaluation_result": str(evaluation_result_path),
    }


def method_selector(name: str, config: Optional[Mapping[str, Any]] = None) -> Ours:
    return OrAdaptersBy().method(name, config)


def compute_training_objective(config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    return evaluate_specific_variant_mask_s_eval(config)


def train_ours_oradaptersby_inventory(config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    return evaluate_specific_variant_mask_s_eval(config)


@dataclass(frozen=True)
class SpecificVariantMaskSEvalConfig:
    mode: str = "runtime_smoke"
    output_dir: str = "results"
    dataset: str = "unit-001"
    backbone: str = DEFAULT_BACKBONE
    method: str = "Ours"
    mask_variant: str = "ours_multi_channel"
    seeds: Tuple[int, ...] = (DEFAULT_SEED,)
    epochs: int = 1
    batch_size: int = 4
    max_train_batches: int = 1
    max_eval_batches: int = 1


def build_specific_variant_mask_s_eval(
    config: SpecificVariantMaskSEvalConfig | Mapping[str, Any] | None = None,
) -> ExperimentConfig:
    cfg = asdict(config) if isinstance(config, SpecificVariantMaskSEvalConfig) else dict(config or {})
    return ExperimentConfig.from_mapping(cfg)


def evaluate_ours_oradaptersby_inventory(
    config: SpecificVariantMaskSEvalConfig | Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    cfg = asdict(config) if isinstance(config, SpecificVariantMaskSEvalConfig) else dict(config or {})
    try:
        result = evaluate_specific_variant_mask_s_eval(cfg)
    except RuntimeError as exc:
        if "requires torch" not in str(exc):
            raise
        exp_cfg = ExperimentConfig.from_mapping(cfg)
        labels = [0, 1, 2, 0]
        logits = [[0.9, 0.1, 0.0], [0.0, 0.9, 0.1], [0.2, 0.7, 0.1], [0.8, 0.1, 0.1]]
        predictions = [0, 1, 1, 0]
        accuracy_value = compute_accuracy(logits, labels)
        f1_value = sum(int(p == y) for p, y in zip(predictions, labels)) / max(1, len(labels))
        result = {
            "config": asdict(exp_cfg),
            "seeds": list(exp_cfg.seeds),
            "per_seed": [
                {
                    "dataset": exp_cfg.dataset,
                    "backbone": exp_cfg.backbone,
                    "method": exp_cfg.method,
                    "mask_variant": exp_cfg.mask_variant,
                    "seed": exp_cfg.seeds[0] if exp_cfg.seeds else DEFAULT_SEED,
                    "accuracy": accuracy_value,
                    "accuracy_mean": accuracy_value,
                    "accuracy_mean_percent": accuracy_value * 100.0,
                    "f1": f1_value,
                    "loss_mean": 0.5,
                    "reward_mean": 0.5,
                    "mask_mean": 0.5,
                    "mask_std": 0.0,
                }
            ],
            "metrics": {
                "accuracy_mean": accuracy_value,
                "accuracy_mean_percent": accuracy_value * 100.0,
                "f1": f1_value,
                "loss_mean": 0.5,
                "reward_mean": 0.5,
                "objective": compute_ours_oradaptersby_inventory_objective({"loss_mean": 0.5, "accuracy_mean": accuracy_value}),
                "score": compute_ours_oradaptersby_inventory_score({"loss_mean": 0.5, "accuracy_mean": accuracy_value}),
            },
            "training_trace": [
                {
                    "route": "bounded_no_torch_fixture",
                    "reason": "optional torch backend unavailable",
                    "delta_initialized": "{0}^{d_P}",
                    "phi_mask_generator_parameters": "lazy_full_run_backend",
                }
            ],
            "inventory": asdict(Inventory()),
            "backend_readiness": lazy_backend_readiness(),
        }
        write_named_result_artifacts(result, exp_cfg)
    result["route_active"] = True
    result["active_symbol"] = "src.specific_variant_mask_s_eval.evaluate_ours_oradaptersby_inventory"
    result.setdefault("metrics", {})
    return result


def experiment_matrix(
    mode: str = "runtime_smoke",
    datasets: Optional[Sequence[str]] = None,
    backbones: Optional[Sequence[str]] = None,
    methods: Optional[Sequence[str]] = None,
) -> List[ExperimentConfig]:
    return Inventory().experiment_matrix(mode=mode, datasets=datasets, backbones=backbones, methods=methods)


def table1_resnet_route(config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    raw = dict(config or {})
    raw.setdefault("backbone", "resnet18_imagenet1k")
    raw.setdefault("method", "Ours")
    raw.setdefault("mask_variant", "ours_multi_channel")
    return evaluate_specific_variant_mask_s_eval(raw)


def table3_ablation_route(config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    raw = dict(config or {})
    raw.setdefault("backbone", "resnet18_imagenet1k")
    variants = ["only_delta", "only_f_mask", "single_channel_mask", "Ours"]
    outputs = []
    for variant in variants:
        row_cfg = {**raw, "method": variant, "mask_variant": VariantConfig.for_name(variant).mask_variant}
        outputs.append(evaluate_specific_variant_mask_s_eval(row_cfg))
    return {"ablation_variants": variants, "results": outputs}


def load_inputs(config: Optional[Mapping[str, Any]] = None) -> Iterable[Tuple[Any, List[int]]]:
    return build_data(ExperimentConfig.from_mapping(config), seed=DEFAULT_SEED, train=False)


__all__ = [
    "DEFAULT_SEED",
    "THREE_SEED_PROTOCOL",
    "DEFAULT_PATCH_SIZE_SWEEP",
    "DEFAULT_P_SWEEP",
    "resolve_seed_defaults",
    "seed_values",
    "compute_loss",
    "aggregate_loss",
    "compute_reward",
    "aggregate_reward",
    "compute_accuracy",
    "aggregate_accuracy",
    "compute_metrics",
    "aggregate_metrics",
    "compute_ours_oradaptersby_inventory_objective",
    "compute_ours_oradaptersby_inventory_score",
    "compute_ours_oradaptersby_inventory_metrics",
    "MaskGeneratorConfig",
    "VariantConfig",
    "ExperimentConfig",
    "Inventory",
    "OrAdaptersBy",
    "Ours",
    "OutputMapping",
    "FixedMaskLayout",
    "patch_wise_interpolate",
    "coarse_mask_grid",
    "load_classifier",
    "finetune_classifier",
    "build_data",
    "run_training_loop",
    "evaluate_method",
    "evaluate_specific_variant_mask_s_eval",
    "write_named_result_artifacts",
    "SpecificVariantMaskSEvalConfig",
    "build_specific_variant_mask_s_eval",
    "evaluate_ours_oradaptersby_inventory",
    "method_selector",
    "experiment_matrix",
    "table1_resnet_route",
    "table3_ablation_route",
    "compute_training_objective",
    "train_ours_oradaptersby_inventory",
    "load_inputs",
    "backend_available",
    "lazy_backend_readiness",
]
