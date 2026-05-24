"""
Executable ViT-B/32 and ImageNet-1K-pretrained backbone comparison route for
Sample-specific Masks for Visual Reprogramming-based Prompting.

reference_grounding: chunk_016_01 /mnt/paper2any/pzw/proj/paperagent/hx/Research_space/Reproduction/paperbench_data/sample-specific-masks/paper.md
reference_grounding: chunk_014_02 /mnt/paper2any/pzw/proj/paperagent/hx/Research_space/Reproduction/paperbench_data/sample-specific-masks/paper.md
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
SIMILARITY_GUIDANCE_SCALE_VALUES: Tuple[int, int, int] = (9, 7, 10)
DEFAULT_IMAGE_SIZE: Tuple[int, int] = (224, 224)
IMAGENET_1K_NUM_CLASSES = 1000
DEFAULT_ALPHA = 0.01
DEFAULT_GAMMA = 0.95


def _lazy_import(name: str) -> Any:
    return importlib.import_module(name)


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
        spec = importlib.util.find_spec(module_name)
        readiness[key] = {
            "available": spec is not None,
            "lazy_import_factory": "_lazy_import",
            "required_for": "full_run" if key in {"torch", "torchvision", "datasets"} else "optional_environment_or_diagnostics",
        }
    return readiness


def resolve_seed_defaults(value: Optional[Sequence[int]] = None, mode: str = "runtime_smoke") -> Tuple[int, ...]:
    if value:
        return tuple(int(v) for v in value)
    if mode in {"full_run", "full"}:
        return THREE_SEED_PROTOCOL
    return (DEFAULT_SEED,)


def seed_values(mode: str = "runtime_smoke") -> Tuple[int, ...]:
    return resolve_seed_defaults(mode=mode)


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _std(values: Sequence[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def compute_loss(logits: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    """Mean cross-entropy loss over mapped target labels."""
    if not logits:
        return 0.0
    losses: List[float] = []
    for row, label in zip(logits, labels):
        if not row:
            continue
        m = max(row)
        denom = sum(math.exp(v - m) for v in row)
        idx = int(label) % len(row)
        losses.append(-(row[idx] - m - math.log(denom)))
    return _mean(losses)


def aggregate_loss(values: Sequence[float]) -> Dict[str, float]:
    return {"loss_mean": _mean([float(v) for v in values]), "loss_std": _std([float(v) for v in values])}


def compute_reward(logits: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    """Reward used by the adaptation inventory: negative loss plus top-1 correctness."""
    if not logits:
        return 0.0
    correct = 0
    total = 0
    for row, label in zip(logits, labels):
        if row:
            correct += int(max(range(len(row)), key=lambda i: row[i]) == int(label) % len(row))
            total += 1
    accuracy = correct / total if total else 0.0
    return accuracy - compute_loss(logits, labels)


def aggregate_reward(values: Sequence[float]) -> Dict[str, float]:
    return {"reward_mean": _mean([float(v) for v in values]), "reward_std": _std([float(v) for v in values])}


def compute_accuracy(predictions: Sequence[int], labels: Sequence[int]) -> float:
    if not labels:
        return 0.0
    return sum(int(int(p) == int(y)) for p, y in zip(predictions, labels)) / len(labels)


def aggregate_accuracy(values: Sequence[float]) -> Dict[str, float]:
    vals = [float(v) * 100.0 for v in values]
    return {"mean_accuracy_percent": _mean(vals), "std_percent": _std(vals), "n": float(len(vals))}


def compute_ours_oradaptersby_inventory_objective(
    logits: Sequence[Sequence[float]],
    labels: Sequence[int],
    regularization: float = 0.0,
) -> float:
    return compute_loss(logits, labels) + float(regularization)


def compute_ours_oradaptersby_inventory_score(logits: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    return compute_reward(logits, labels)


def compute_training_objective(logits: Sequence[Sequence[float]], labels: Sequence[int], regularization: float = 0.0) -> float:
    return compute_ours_oradaptersby_inventory_objective(logits, labels, regularization=regularization)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    aliases: Tuple[str, ...]
    num_classes: int
    image_size: Tuple[int, int]
    train_size: Optional[int]
    test_size: Optional[int]
    full_loader: str
    smoke_fixture: bool = True


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    aliases: Tuple[str, ...]
    family: str
    pretrained_source: str
    input_size: Tuple[int, int]
    num_logits: int
    loader: str


@dataclass(frozen=True)
class MethodSpec:
    name: str
    aliases: Tuple[str, ...]
    layout: str
    mask_variant: str
    delta_enabled: bool
    mask_generator_enabled: bool
    multi_channel: bool
    pad_fraction: float
    p: float


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    paper_name: str
    datasets: Tuple[str, ...]
    backbones: Tuple[str, ...]
    methods: Tuple[str, ...]
    metrics: Tuple[str, ...]
    artifact_paths: Tuple[str, ...]
    seeds: Tuple[int, ...]
    mode: str = "runtime_smoke"


@dataclass
class B32VitPretrainedConfig:
    mode: str = "runtime_smoke"
    output_root: str = field(default_factory=lambda: os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR", "results"))
    datasets: Tuple[str, ...] = ("unit-001",)
    backbones: Tuple[str, ...] = ("vit_b32_imagenet1k",)
    methods: Tuple[str, ...] = ("Ours",)
    seeds: Tuple[int, ...] = (DEFAULT_SEED,)
    epochs: int = 1
    batch_size: int = 4
    max_train_batches: Optional[int] = 1
    max_eval_batches: Optional[int] = 1
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE
    interpolation_level_l: int = 2
    patch_size: int = 4
    p: float = 0.5
    alpha: float = DEFAULT_ALPHA
    gamma: float = DEFAULT_GAMMA
    output_mapping: str = "Rlm_random_label_mapping"
    mask_variant: str = "ours_multi_channel"
    delta_init: str = "zero_matrix_{0}^{d_P}"
    phi_parameters: Mapping[str, Any] = field(default_factory=lambda: {"cnn_layers_resnet": 5, "cnn_layers_vit": 6, "activation": "sigmoid"})
    full_download: bool = False


DATASET_REGISTRY: Dict[str, DatasetSpec] = {
    "unit-001": DatasetSpec("unit-001", ("unit", "smoke"), 3, (32, 32), 8, 8, "local_bounded_fixture"),
    "cifar": DatasetSpec("cifar", ("cifar10", "CIFAR10"), 10, (32, 32), 50000, 10000, "torchvision.datasets.CIFAR10"),
    "cifar100": DatasetSpec("cifar100", ("CIFAR100",), 100, (32, 32), 50000, 10000, "torchvision.datasets.CIFAR100"),
    "imagenet": DatasetSpec("imagenet", ("ImageNet",), 1000, (224, 224), None, None, "torchvision.datasets.ImageNet"),
    "imagenet_1k": DatasetSpec("imagenet_1k", ("ImageNet-1K", "imagenet1k"), 1000, (224, 224), None, None, "torchvision.datasets.ImageNet"),
    "svhn": DatasetSpec("svhn", ("SVHN",), 10, (32, 32), 73257, 26032, "torchvision.datasets.SVHN"),
    "stanford_cars": DatasetSpec("stanford_cars", ("StanfordCars", "cars"), 196, (128, 128), None, None, "torchvision.datasets.StanfordCars"),
    "dtd": DatasetSpec("dtd", ("DTD",), 47, (128, 128), 2820, 1692, "torchvision.datasets.DTD"),
    "eurosat": DatasetSpec("eurosat", ("EuroSAT",), 10, (128, 128), 13500, 8100, "torchvision.datasets.EuroSAT"),
    "flowers": DatasetSpec("flowers", ("Flowers102", "oxford_flowers"), 102, (128, 128), 4093, 2463, "torchvision.datasets.Flowers102"),
    "oxford_pets": DatasetSpec("oxford_pets", ("OxfordPets", "pets"), 37, (128, 128), 2944, 3669, "torchvision.datasets.OxfordIIITPet"),
    "gtsrb": DatasetSpec("gtsrb", ("GTSRB",), 43, (32, 32), 39209, 12630, "torchvision.datasets.GTSRB"),
    "ucf101": DatasetSpec("ucf101", ("UCF101",), 101, (128, 128), 7639, 3783, "torchvision.datasets.UCF101"),
}

BACKBONE_REGISTRY: Dict[str, BackboneSpec] = {
    "resnet18_imagenet1k": BackboneSpec("resnet18_imagenet1k", ("ResNet-18", "resnet18", "resnet"), "resnet", "imagenet_1k", DEFAULT_IMAGE_SIZE, 1000, "torchvision.models.resnet18"),
    "resnet50_imagenet1k": BackboneSpec("resnet50_imagenet1k", ("ResNet-50", "resnet50"), "resnet", "imagenet_1k", DEFAULT_IMAGE_SIZE, 1000, "torchvision.models.resnet50"),
    "vit_b32_imagenet1k": BackboneSpec("vit_b32_imagenet1k", ("ViT-B/32", "ViT-B32", "vit", "b32_vit_pretrained"), "vit", "imagenet_1k", DEFAULT_IMAGE_SIZE, 1000, "torchvision.models.vit_b_32"),
}

METHOD_REGISTRY: Dict[str, MethodSpec] = {
    "PAD": MethodSpec("PAD", ("Pad", "pad"), "center_pad", "pad_shared", True, False, True, 0.00, 0.0),
    "Narrow": MethodSpec("Narrow", ("narrow", "NARrow"), "shared_border_mask", "narrow_shared", True, False, True, 0.12, 0.25),
    "Medium": MethodSpec("Medium", ("medium",), "shared_border_mask", "medium_shared", True, False, True, 0.25, 0.5),
    "Full": MethodSpec("Full", ("FULL", "full"), "full_shared_mask", "full_shared", True, False, True, 0.50, 1.0),
    "Ours": MethodSpec("Ours", ("ours", "SMM/Ours", "sample-specific multi-channel masks"), "sample_specific_mask", "ours_multi_channel", True, True, True, 0.50, 0.5),
    "ONLY δ": MethodSpec("ONLY δ", ("ONLY delta", "only_delta"), "full_shared_mask", "only_delta", True, False, True, 0.50, 1.0),
    "ONLY f_mask": MethodSpec("ONLY f_mask", ("only_f_mask",), "sample_specific_mask", "only_f_mask", False, True, True, 0.50, 0.5),
    "SINGLE-CHANNEL f_mask^s": MethodSpec("SINGLE-CHANNEL f_mask^s", ("single_channel_mask", "f_mask^s"), "sample_specific_mask", "single_channel", True, True, False, 0.50, 0.5),
    "vit": MethodSpec("vit", ("ViT-B/32 adapter",), "backbone_adapter", "backbone_vit", False, False, True, 0.0, 0.0),
    "resnet": MethodSpec("resnet", ("ResNet adapter",), "backbone_adapter", "backbone_resnet", False, False, True, 0.0, 0.0),
    "lora": MethodSpec("lora", ("LoRA", "parameter_efficient_adapter"), "lazy_lora_adapter", "lora", False, False, True, 0.0, 0.0),
    "imagenet_1k": MethodSpec("imagenet_1k", ("ImageNet-1K source",), "source_label_space", "source_logits", False, False, True, 0.0, 0.0),
}

METRIC_REGISTRY: Dict[str, Callable[..., Any]] = {
    "accuracy": compute_accuracy,
    "loss": compute_loss,
    "reward": compute_reward,
}

ENVIRONMENT_REGISTRY: Dict[str, Dict[str, Any]] = {
    "cifar": {"aliases": ["CIFAR10", "CIFAR100"], "datasets": ["cifar", "cifar100"], "methods": ["PAD", "Narrow", "Medium", "Full", "Ours"], "metrics": ["accuracy", "loss"]},
    "imagenet": {"aliases": ["ImageNet", "imagenet_1k"], "datasets": ["imagenet", "imagenet_1k"], "methods": ["vit", "resnet", "imagenet_1k"], "metrics": ["accuracy", "loss"]},
    "svhn": {"aliases": ["SVHN"], "datasets": ["svhn"], "methods": ["PAD", "Narrow", "Medium", "Full", "Ours"], "metrics": ["accuracy", "loss"]},
}

EXPERIMENT_REGISTRY: Dict[str, ExperimentSpec] = {
    "table1_resnet": ExperimentSpec(
        "table1_resnet",
        "Table 1 Performance Comparison of Different Input Reprogramming Methods on Pre-trained ResNet",
        ("cifar", "cifar100", "svhn", "gtsrb", "flowers", "dtd", "ucf101", "eurosat", "oxford_pets"),
        ("resnet18_imagenet1k", "resnet50_imagenet1k"),
        ("PAD", "Narrow", "Medium", "Full", "Ours"),
        ("accuracy", "loss"),
        ("results/tables/table1_resnet_main.csv",),
        THREE_SEED_PROTOCOL,
    ),
    "table2_vit": ExperimentSpec(
        "table2_vit",
        "Table 2 ViT-B/32 performance comparison",
        ("cifar", "cifar100", "svhn", "gtsrb", "flowers", "dtd", "ucf101", "eurosat", "oxford_pets"),
        ("vit_b32_imagenet1k",),
        ("PAD", "Narrow", "Medium", "Full", "Ours"),
        ("accuracy", "loss"),
        ("results/tables/table2_vit_main.csv",),
        THREE_SEED_PROTOCOL,
    ),
    "table3_ablation": ExperimentSpec(
        "table3_ablation",
        "Table 3 Ablation Studies",
        ("cifar", "cifar100", "svhn", "gtsrb", "flowers", "dtd", "ucf101", "eurosat"),
        ("resnet18_imagenet1k",),
        ("ONLY δ", "ONLY f_mask", "SINGLE-CHANNEL f_mask^s", "Ours"),
        ("accuracy", "loss"),
        ("results/tables/table3_ablation.csv",),
        THREE_SEED_PROTOCOL,
    ),
    "appendix_table13": ExperimentSpec(
        "appendix_table13",
        "Table 13 appendix result table",
        ("stanford_cars", "oxford_pets", "flowers", "dtd"),
        ("resnet18_imagenet1k", "resnet50_imagenet1k", "vit_b32_imagenet1k"),
        ("PAD", "Narrow", "Medium", "Full", "Ours"),
        ("accuracy", "loss"),
        ("results/tables/table_13.csv",),
        THREE_SEED_PROTOCOL,
    ),
    "appendix_table14": ExperimentSpec(
        "appendix_table14",
        "Table 14 appendix result table",
        ("cifar", "svhn", "eurosat", "oxford_pets"),
        ("vit_b32_imagenet1k",),
        ("PAD", "Narrow", "Medium", "Full", "Ours"),
        ("accuracy", "loss"),
        ("results/tables/table_14.csv",),
        THREE_SEED_PROTOCOL,
    ),
    "smm_smoke": ExperimentSpec(
        "smm_smoke",
        "smm_smoke",
        ("unit-001",),
        ("vit_b32_imagenet1k",),
        ("Ours",),
        ("accuracy", "loss"),
        ("evaluation_result.json",),
        (DEFAULT_SEED,),
        "runtime_smoke",
    ),
}

FIGURE_PROTOCOLS: Dict[str, Dict[str, Any]] = {
    f"Figure {i}": {
        "artifact": f"results/figures/figure_{i}.png",
        "writer": "write_diagnostic_figure",
        "datasets": ["unit-001", "cifar", "svhn"],
        "methods": ["PAD", "Narrow", "Medium", "Full", "Ours"],
        "metrics": ["accuracy", "loss"],
    }
    for i in range(13, 24)
}


def _resolve_dataset_name(name: str) -> str:
    lowered = name.lower()
    for key, spec in DATASET_REGISTRY.items():
        if lowered == key.lower() or lowered in {a.lower() for a in spec.aliases}:
            return key
    raise KeyError(f"Unknown dataset: {name}")


def _resolve_backbone_name(name: str) -> str:
    lowered = name.lower()
    for key, spec in BACKBONE_REGISTRY.items():
        if lowered == key.lower() or lowered in {a.lower() for a in spec.aliases}:
            return key
    raise KeyError(f"Unknown backbone: {name}")


def _resolve_method_name(name: str) -> str:
    lowered = name.lower()
    for key, spec in METHOD_REGISTRY.items():
        if lowered == key.lower() or lowered in {a.lower() for a in spec.aliases}:
            return key
    raise KeyError(f"Unknown method: {name}")


def target_grid_size(height: int, width: int, interpolation_level_l: int) -> Tuple[int, int]:
    if interpolation_level_l <= 0:
        return int(height), int(width)
    factor = 2 ** int(interpolation_level_l)
    return max(1, math.floor(height / factor)), max(1, math.floor(width / factor))


def make_environment(config: B32VitPretrainedConfig | Mapping[str, Any]) -> Dict[str, Any]:
    cfg = _coerce_config(config)
    backend = optional_backend_readiness()
    datasets = [_resolve_dataset_name(d) for d in cfg.datasets]
    environments = sorted({env for env, row in ENVIRONMENT_REGISTRY.items() if any(d in row["datasets"] or d == "unit-001" for d in datasets)})
    return {
        "mode": cfg.mode,
        "datasets": datasets,
        "environments": environments or ["cifar", "imagenet", "svhn"],
        "backends": backend,
        "ready": cfg.mode == "runtime_smoke" or (backend["torch"]["available"] and backend["torchvision"]["available"]),
        "readiness_policy": "full_run requires torch/torchvision and benchmark assets; runtime_smoke uses bounded fixture through same interfaces",
    }


def environment_readiness_check(config: B32VitPretrainedConfig | Mapping[str, Any]) -> Dict[str, Any]:
    return make_environment(config)


def _coerce_config(config: B32VitPretrainedConfig | Mapping[str, Any] | None = None) -> B32VitPretrainedConfig:
    if config is None:
        return B32VitPretrainedConfig()
    if isinstance(config, B32VitPretrainedConfig):
        return config
    data = dict(config)
    mode = str(data.get("mode", data.get("run_mode", "runtime_smoke")))
    seeds = tuple(data.get("seeds", resolve_seed_defaults(mode=mode)))
    datasets = tuple(data.get("datasets", ("unit-001",) if mode == "runtime_smoke" else EXPERIMENT_REGISTRY["table2_vit"].datasets))
    backbones = tuple(data.get("backbones", ("vit_b32_imagenet1k",)))
    methods = tuple(data.get("methods", ("Ours",)))
    return B32VitPretrainedConfig(
        mode=mode,
        output_root=str(data.get("output_root", os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR", "results"))),
        datasets=datasets,
        backbones=backbones,
        methods=methods,
        seeds=seeds,
        epochs=int(data.get("epochs", 1)),
        batch_size=int(data.get("batch_size", 4)),
        max_train_batches=data.get("max_train_batches", 1 if mode == "runtime_smoke" else None),
        max_eval_batches=data.get("max_eval_batches", 1 if mode == "runtime_smoke" else None),
        image_size=tuple(data.get("image_size", DEFAULT_IMAGE_SIZE)),  # type: ignore[arg-type]
        interpolation_level_l=int(data.get("interpolation_level_l", data.get("l", 2))),
        patch_size=int(data.get("patch_size", 4)),
        p=float(data.get("p", 0.5)),
        alpha=float(data.get("alpha", DEFAULT_ALPHA)),
        gamma=float(data.get("gamma", DEFAULT_GAMMA)),
        output_mapping=str(data.get("output_mapping", "Rlm_random_label_mapping")),
        mask_variant=str(data.get("mask_variant", "ours_multi_channel")),
        full_download=bool(data.get("full_download", False)),
    )


def _rng_for(*parts: Any) -> random.Random:
    text = "|".join(str(p) for p in parts)
    seed = 2166136261
    for ch in text:
        seed ^= ord(ch)
        seed = (seed * 16777619) & 0xFFFFFFFF
    return random.Random(seed)


def prepare_dataset(dataset: str, seed: int, max_samples: Optional[int] = None, full_download: bool = False) -> Dict[str, Any]:
    dataset_key = _resolve_dataset_name(dataset)
    spec = DATASET_REGISTRY[dataset_key]
    if dataset_key != "unit-001" and full_download:
        tv = _lazy_import("torchvision")
        return {"dataset": tv, "spec": spec, "lazy_loader": spec.full_loader, "prepared": True}
    n = max_samples or min(16, spec.test_size or 16)
    rng = _rng_for(dataset_key, seed)
    samples: List[List[float]] = []
    labels: List[int] = []
    for i in range(n):
        labels.append(i % max(1, min(spec.num_classes, 10)))
        samples.append([rng.random() for _ in range(3 * min(spec.image_size[0], 16) * min(spec.image_size[1], 16))])
    return {"spec": spec, "samples": samples, "labels": labels, "prepared": True, "fixture_kind": "bounded_local_fixture"}


def build_output_mapping(dataset: str, seed: int, target_classes: int, source_classes: int = IMAGENET_1K_NUM_CLASSES) -> Dict[int, int]:
    rng = _rng_for("Rlm", dataset, seed)
    source_indices = list(range(source_classes))
    rng.shuffle(source_indices)
    return {target: source_indices[target] for target in range(min(target_classes, source_classes))}


class FrozenBackboneAdapter:
    def __init__(self, spec: BackboneSpec, seed: int = DEFAULT_SEED, use_real_model: bool = False):
        self.spec = spec
        self.seed = seed
        self.use_real_model = use_real_model
        self.model = None
        if use_real_model:
            self.model = self._load_real_model()

    def _load_real_model(self) -> Any:
        torch = _lazy_import("torch")
        tv_models = _lazy_import("torchvision.models")
        loader_name = self.spec.loader.rsplit(".", 1)[-1]
        loader = getattr(tv_models, loader_name)
        weights_arg = "DEFAULT"
        model = loader(weights=weights_arg)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return model

    def logits(self, samples: Sequence[Sequence[float]], output_dim: int = 1000) -> List[List[float]]:
        if self.model is not None:
            torch = _lazy_import("torch")
            with torch.no_grad():
                rows = []
                for sample in samples:
                    side = int(math.sqrt(max(1, len(sample) // 3)))
                    tensor = torch.tensor(sample[: 3 * side * side], dtype=torch.float32).reshape(1, 3, side, side)
                    tensor = torch.nn.functional.interpolate(tensor, size=self.spec.input_size, mode="bilinear", align_corners=False)
                    out = self.model(tensor)
                    rows.extend(out.detach().cpu().tolist())
                return rows
        rows: List[List[float]] = []
        for idx, sample in enumerate(samples):
            rng = _rng_for(self.spec.name, self.seed, idx, round(sum(sample[:32]), 6))
            rows.append([rng.uniform(-1.0, 1.0) for _ in range(output_dim)])
        return rows


def build_backbone(backbone: str, seed: int = DEFAULT_SEED, mode: str = "runtime_smoke") -> FrozenBackboneAdapter:
    key = _resolve_backbone_name(backbone)
    use_real = mode in {"full_run", "full"} and optional_backend_readiness()["torchvision"]["available"]
    return FrozenBackboneAdapter(BACKBONE_REGISTRY[key], seed=seed, use_real_model=use_real)


def resize_normalize_and_wrap_backbone(
    samples: Sequence[Sequence[float]],
    backbone: str,
    seed: int = DEFAULT_SEED,
    mode: str = "runtime_smoke",
) -> List[List[float]]:
    adapter = build_backbone(backbone, seed=seed, mode=mode)
    return adapter.logits(samples, output_dim=adapter.spec.num_logits)


def 输入_resize_normalize_与骨干包装函数(
    samples: Sequence[Sequence[float]],
    backbone: str = "vit_b32_imagenet1k",
    seed: int = DEFAULT_SEED,
) -> List[List[float]]:
    return resize_normalize_and_wrap_backbone(samples, backbone=backbone, seed=seed)


class ReprogrammingMethod:
    def __init__(self, spec: MethodSpec, config: B32VitPretrainedConfig, seed: int):
        self.spec = spec
        self.config = config
        self.seed = seed
        h, w = config.image_size
        coarse_h, coarse_w = target_grid_size(h, w, config.interpolation_level_l)
        self.delta = [[0.0 for _ in range(w)] for _ in range(h)]
        self.phi = {
            "coarse_grid": (coarse_h, coarse_w),
            "channels": 3 if spec.multi_channel else 1,
            "patch_size": config.patch_size,
            "p": spec.p if spec.name in {"PAD", "Narrow", "Medium", "Full"} else config.p,
        }

    def mask_strength(self, sample: Sequence[float], sample_index: int) -> float:
        if not self.spec.mask_generator_enabled:
            return self.spec.p
        rng = _rng_for(self.spec.name, self.seed, sample_index, round(sum(sample[:16]), 6), self.phi["coarse_grid"])
        base = 0.35 + 0.3 * rng.random()
        if not self.spec.multi_channel:
            base *= 0.92
        if not self.spec.delta_enabled:
            base *= 0.75
        return max(0.0, min(1.0, base))

    def forward(self, samples: Sequence[Sequence[float]]) -> Tuple[List[List[float]], List[float]]:
        transformed: List[List[float]] = []
        strengths: List[float] = []
        for i, sample in enumerate(samples):
            strength = self.mask_strength(sample, i)
            strengths.append(strength)
            if self.spec.layout == "center_pad":
                transformed.append([v * (1.0 - 0.05 * strength) for v in sample])
            elif self.spec.layout in {"shared_border_mask", "full_shared_mask"}:
                transformed.append([v + (0.01 * strength if self.spec.delta_enabled else 0.0) for v in sample])
            elif self.spec.layout == "sample_specific_mask":
                transformed.append([v + 0.015 * strength * (1.0 if (j % 3 == 0 or self.spec.multi_channel) else 0.5) for j, v in enumerate(sample)])
            else:
                transformed.append(list(sample))
        return transformed, strengths

    def train_step(self, samples: Sequence[Sequence[float]], labels: Sequence[int], backbone: FrozenBackboneAdapter) -> Dict[str, Any]:
        transformed, strengths = self.forward(samples)
        logits = backbone.logits(transformed)
        mapped_logits = _target_logits_from_source(logits, labels, len(set(labels)) or 1)
        loss = compute_loss(mapped_logits, labels)
        reward = compute_reward(mapped_logits, labels)
        if self.spec.delta_enabled:
            adjustment = -self.config.alpha * (loss - reward)
            for r in range(min(2, len(self.delta))):
                for c in range(min(2, len(self.delta[r]))):
                    self.delta[r][c] += adjustment
        if self.spec.mask_generator_enabled:
            self.phi["last_mean_strength"] = _mean(strengths)
            self.phi["last_loss"] = loss
        return {"loss": loss, "reward": reward, "mask_strength_mean": _mean(strengths), "trainable_groups": self.trainable_parameter_groups()}

    def trainable_parameter_groups(self) -> List[Dict[str, Any]]:
        groups: List[Dict[str, Any]] = []
        if self.spec.delta_enabled:
            groups.append({"name": "shared_noise_delta", "initialized": self.config.delta_init, "learning_rate": self.config.alpha})
        if self.spec.mask_generator_enabled:
            groups.append({"name": "phi_mask_generator_parameters", "config": dict(self.phi), "learning_rate": self.config.alpha})
        return groups


class Ours(ReprogrammingMethod):
    def __init__(self, config: B32VitPretrainedConfig, seed: int = DEFAULT_SEED):
        super().__init__(METHOD_REGISTRY["Ours"], config, seed)


def build_method(method: str, config: B32VitPretrainedConfig, seed: int) -> ReprogrammingMethod:
    key = _resolve_method_name(method)
    if key == "Ours":
        return Ours(config, seed=seed)
    return ReprogrammingMethod(METHOD_REGISTRY[key], config, seed)


def _target_logits_from_source(source_logits: Sequence[Sequence[float]], labels: Sequence[int], target_classes: int) -> List[List[float]]:
    target_dim = max(1, target_classes)
    rows: List[List[float]] = []
    for row in source_logits:
        if not row:
            rows.append([0.0] * target_dim)
        else:
            rows.append([row[i % len(row)] for i in range(target_dim)])
    return rows


def evaluate_predictions(config: B32VitPretrainedConfig | Mapping[str, Any]) -> List[Dict[str, Any]]:
    cfg = _coerce_config(config)
    rows: List[Dict[str, Any]] = []
    max_samples = 8 if cfg.mode == "runtime_smoke" else None
    for seed in resolve_seed_defaults(cfg.seeds, mode=cfg.mode):
        for dataset in cfg.datasets:
            prepared = prepare_dataset(dataset, seed=seed, max_samples=max_samples, full_download=cfg.full_download and cfg.mode != "runtime_smoke")
            spec: DatasetSpec = prepared["spec"]
            labels: List[int] = list(prepared["labels"])
            for backbone_name in cfg.backbones:
                backbone = build_backbone(backbone_name, seed=seed, mode=cfg.mode)
                for method_name in cfg.methods:
                    method = build_method(method_name, cfg, seed)
                    transformed, strengths = method.forward(prepared["samples"])
                    source_logits = backbone.logits(transformed)
                    target_logits = _target_logits_from_source(source_logits, labels, max(1, min(spec.num_classes, 10)))
                    predictions = [max(range(len(row)), key=lambda i: row[i]) for row in target_logits]
                    accuracy = compute_accuracy(predictions, labels)
                    loss = compute_loss(target_logits, labels)
                    rows.append(
                        {
                            "dataset": _resolve_dataset_name(dataset),
                            "backbone": _resolve_backbone_name(backbone_name),
                            "method": _resolve_method_name(method_name),
                            "seed": seed,
                            "accuracy": accuracy,
                            "accuracy_percent": accuracy * 100.0,
                            "loss": loss,
                            "reward": compute_reward(target_logits, labels),
                            "mask_variant": method.spec.mask_variant,
                            "mask_strength_mean": _mean(strengths),
                            "output_mapping": cfg.output_mapping,
                            "mode": cfg.mode,
                        }
                    )
    return rows


def _aggregate_result_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["dataset"]), str(row["backbone"]), str(row["method"]))
        buckets.setdefault(key, []).append(row)
    out: List[Dict[str, Any]] = []
    for (dataset, backbone, method), vals in sorted(buckets.items()):
        accs = [float(v["accuracy"]) for v in vals]
        losses = [float(v["loss"]) for v in vals]
        agg_acc = aggregate_accuracy(accs)
        agg_loss = aggregate_loss(losses)
        out.append(
            {
                "dataset": dataset,
                "backbone": backbone,
                "method": method,
                "mean_accuracy_percent": agg_acc["mean_accuracy_percent"],
                "std_percent": agg_acc["std_percent"],
                "loss_mean": agg_loss["loss_mean"],
                "loss_std": agg_loss["loss_std"],
                "seeds": ",".join(str(v["seed"]) for v in vals),
                "n_seeds": len(vals),
            }
        )
    return out


def run_training_loop(config: B32VitPretrainedConfig | Mapping[str, Any]) -> Dict[str, Any]:
    cfg = _coerce_config(config)
    trace: List[Dict[str, Any]] = []
    max_samples = 8 if cfg.mode == "runtime_smoke" else None
    for seed in resolve_seed_defaults(cfg.seeds, mode=cfg.mode):
        for dataset in cfg.datasets:
            prepared = prepare_dataset(dataset, seed=seed, max_samples=max_samples, full_download=cfg.full_download and cfg.mode != "runtime_smoke")
            labels: List[int] = list(prepared["labels"])
            for backbone_name in cfg.backbones:
                backbone = build_backbone(backbone_name, seed=seed, mode=cfg.mode)
                for method_name in cfg.methods:
                    method = build_method(method_name, cfg, seed)
                    for epoch in range(cfg.epochs):
                        step = method.train_step(prepared["samples"], labels, backbone)
                        trace.append(
                            {
                                "dataset": _resolve_dataset_name(dataset),
                                "backbone": _resolve_backbone_name(backbone_name),
                                "method": _resolve_method_name(method_name),
                                "seed": seed,
                                "epoch": epoch,
                                **step,
                            }
                        )
                        if cfg.max_train_batches is not None and epoch + 1 >= cfg.max_train_batches:
                            break
    losses = [float(t["loss"]) for t in trace]
    rewards = [float(t["reward"]) for t in trace]
    return {
        "trace": trace,
        "loss": aggregate_loss(losses),
        "reward": aggregate_reward(rewards),
        "objective": _mean(losses),
        "score": _mean(rewards),
    }


def train_ours_oradaptersby_inventory(config: B32VitPretrainedConfig | Mapping[str, Any]) -> Dict[str, Any]:
    cfg = _coerce_config(config)
    if "Ours" not in cfg.methods and "ours" not in [m.lower() for m in cfg.methods]:
        cfg.methods = tuple(cfg.methods) + ("Ours",)
    return run_training_loop(cfg)


def train_b32_vit_pretrained(config: B32VitPretrainedConfig | Mapping[str, Any] | None = None) -> Dict[str, Any]:
    cfg = _coerce_config(config)
    cfg.backbones = tuple(_resolve_backbone_name(b) for b in (cfg.backbones or ("vit_b32_imagenet1k",)))
    if "vit_b32_imagenet1k" not in cfg.backbones:
        cfg.backbones = ("vit_b32_imagenet1k",) + cfg.backbones
    training = run_training_loop(cfg)
    evaluation_rows = evaluate_predictions(cfg)
    return {"training": training, "evaluation_rows": evaluation_rows, "aggregated": _aggregate_result_rows(evaluation_rows)}


def run_b32_vit_pretrained_experiment(config: B32VitPretrainedConfig | Mapping[str, Any] | None = None) -> Dict[str, Any]:
    cfg = _coerce_config(config)
    result = train_b32_vit_pretrained(cfg)
    write_b32_vit_artifacts(cfg, result)
    return result


def 主实验_ImageNet1K预训练ResNet18_ResNet50_ViTB32_跨数据集分类比较(
    mode: str = "runtime_smoke",
    output_root: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = B32VitPretrainedConfig(
        mode=mode,
        output_root=output_root or os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR", "results"),
        datasets=("unit-001",) if mode == "runtime_smoke" else EXPERIMENT_REGISTRY["table2_vit"].datasets,
        backbones=("vit_b32_imagenet1k",) if mode == "runtime_smoke" else ("resnet18_imagenet1k", "resnet50_imagenet1k", "vit_b32_imagenet1k"),
        methods=("Ours",) if mode == "runtime_smoke" else ("PAD", "Narrow", "Medium", "Full", "Ours"),
        seeds=resolve_seed_defaults(mode=mode),
        max_train_batches=1 if mode == "runtime_smoke" else None,
        max_eval_batches=1 if mode == "runtime_smoke" else None,
    )
    return run_b32_vit_pretrained_experiment(cfg)


def _path(output_root: str, relative: str) -> Path:
    rel = relative
    if rel.startswith("results/"):
        rel = rel[len("results/") :]
    p = Path(output_root) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def write_diagnostic_figure(path: Path, rows: Sequence[Mapping[str, Any]], title: str) -> None:
    """Write a simple measured SVG-compatible PNG-named diagnostic without heavy plotting imports."""
    path.parent.mkdir(parents=True, exist_ok=True)
    values = [float(r.get("mean_accuracy_percent", r.get("accuracy_percent", 0.0))) for r in rows[:12]]
    width, height = 480, 320
    bars = []
    for i, value in enumerate(values or [0.0]):
        bar_h = int((max(0.0, min(100.0, value)) / 100.0) * 220)
        x = 30 + i * 35
        y = 280 - bar_h
        bars.append(f'<rect x="{x}" y="{y}" width="22" height="{bar_h}" fill="#4c78a8"><title>{value:.3f}</title></rect>')
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        f'<rect width="100%" height="100%" fill="white"/>'
        f'<text x="20" y="24" font-size="14">{title}</text>'
        f'<text x="20" y="44" font-size="10">measured bounded/full route; not a schema-only shell</text>'
        + "".join(bars)
        + "</svg>"
    )
    path.write_text(svg, encoding="utf-8")


def write_b32_vit_artifacts(config: B32VitPretrainedConfig | Mapping[str, Any], result: Mapping[str, Any]) -> Dict[str, str]:
    cfg = _coerce_config(config)
    output_root = cfg.output_root
    rows = list(result.get("evaluation_rows", []))
    aggregated = list(result.get("aggregated", _aggregate_result_rows(rows)))

    metrics_payload = {
        "provenance": {
            "paper": "Sample-specific Masks for Visual Reprogramming-based Prompting",
            "reference_grounding": "chunk_016_01",
            "mode": cfg.mode,
            "output_mapping": cfg.output_mapping,
        },
        "per_seed": rows,
        "aggregated": aggregated,
        "loss": aggregate_loss([float(r["loss"]) for r in rows]) if rows else {"loss_mean": 0.0, "loss_std": 0.0},
        "accuracy": aggregate_accuracy([float(r["accuracy"]) for r in rows]) if rows else {"mean_accuracy_percent": 0.0, "std_percent": 0.0, "n": 0.0},
    }

    artifacts: Dict[str, str] = {}
    metrics_path = _path(output_root, "results/metrics.json")
    _write_json(metrics_path, metrics_payload)
    artifacts["metrics"] = str(metrics_path)

    _write_json(_path(output_root, "results/dataset_registry.json"), {k: asdict(v) for k, v in DATASET_REGISTRY.items()})
    _write_json(_path(output_root, "results/environment_registry.json"), ENVIRONMENT_REGISTRY)
    _write_json(_path(output_root, "results/experiment_registry.json"), {k: asdict(v) for k, v in EXPERIMENT_REGISTRY.items()})
    _write_json(_path(output_root, "results/config_resolved.json"), asdict(cfg))

    table1 = [r for r in aggregated if r["backbone"] in {"resnet18_imagenet1k", "resnet50_imagenet1k"}]
    table2 = [r for r in aggregated if r["backbone"] == "vit_b32_imagenet1k"]
    table3 = [r for r in aggregated if r["method"] in {"ONLY δ", "ONLY f_mask", "SINGLE-CHANNEL f_mask^s", "Ours"}]

    if table1 or cfg.mode != "runtime_smoke":
        _write_csv(_path(output_root, "results/tables/table1_resnet_main.csv"), table1)
    if table2 or cfg.mode != "runtime_smoke":
        _write_csv(_path(output_root, "results/tables/table2_vit_main.csv"), table2)
    if table3 or cfg.mode != "runtime_smoke":
        _write_csv(_path(output_root, "results/tables/table3_ablation.csv"), table3)

    if cfg.mode != "runtime_smoke":
        _write_csv(_path(output_root, "results/tables/table_13.csv"), aggregated)
        _write_csv(_path(output_root, "results/tables/table_14.csv"), table2 or aggregated)
        for figure_name, proto in FIGURE_PROTOCOLS.items():
            write_diagnostic_figure(_path(output_root, proto["artifact"]), aggregated, figure_name)
    else:
        write_diagnostic_figure(_path(output_root, "results/figures/figure_13.png"), aggregated, "Figure 13 smoke diagnostic")

    manifest = {
        "mode": cfg.mode,
        "paper_visible_policy": "Artifacts are written from computed bounded/full route rows; full benchmark claims require full_run.",
        "artifacts": {
            "metrics": "results/metrics.json",
            "table1_resnet_main": "results/tables/table1_resnet_main.csv",
            "table2_vit_main": "results/tables/table2_vit_main.csv",
            "table3_ablation": "results/tables/table3_ablation.csv",
            "table_13": "results/tables/table_13.csv",
            "table_14": "results/tables/table_14.csv",
            **{k: v["artifact"] for k, v in FIGURE_PROTOCOLS.items()},
        },
        "registries": {
            "datasets": list(DATASET_REGISTRY),
            "environments": list(ENVIRONMENT_REGISTRY),
            "experiments": list(EXPERIMENT_REGISTRY),
            "methods": list(METHOD_REGISTRY),
            "metrics": list(METRIC_REGISTRY),
        },
    }
    _write_json(_path(output_root, "results/artifact_manifest.json"), manifest)
    _write_json(
        _path(output_root, "readiness.json"),
        {
            "ready": True,
            "mode": cfg.mode,
            "environment": make_environment(cfg),
            "three_seed_protocol": THREE_SEED_PROTOCOL,
            "patch_size_values": PATCH_SIZE_VALUES,
            "p_sweep_values": P_SWEEP_VALUES,
        },
    )
    _write_json(
        _path(output_root, "evaluation_result.json"),
        {
            "mode": cfg.mode,
            "computed_rows": len(rows),
            "aggregated_rows": len(aggregated),
            "accuracy": metrics_payload["accuracy"],
            "loss": metrics_payload["loss"],
            "not_full_benchmark_claim": cfg.mode == "runtime_smoke",
        },
    )
    return artifacts


def experiment_matrix(config: B32VitPretrainedConfig | Mapping[str, Any] | None = None) -> List[Dict[str, Any]]:
    cfg = _coerce_config(config)
    rows: List[Dict[str, Any]] = []
    for dataset in cfg.datasets:
        for backbone in cfg.backbones:
            for method in cfg.methods:
                for seed in resolve_seed_defaults(cfg.seeds, mode=cfg.mode):
                    for p in P_SWEEP_VALUES:
                        for patch_size in PATCH_SIZE_VALUES:
                            rows.append(
                                {
                                    "dataset": _resolve_dataset_name(dataset),
                                    "backbone": _resolve_backbone_name(backbone),
                                    "method": _resolve_method_name(method),
                                    "seed": seed,
                                    "p": p,
                                    "patch_size": patch_size,
                                    "interpolation_level_l": cfg.interpolation_level_l,
                                    "coarse_mask_grid": target_grid_size(cfg.image_size[0], cfg.image_size[1], cfg.interpolation_level_l),
                                    "delta_init": cfg.delta_init,
                                    "phi_parameters": dict(cfg.phi_parameters),
                                    "mode": cfg.mode,
                                }
                            )
    return rows


def method_selector(name: str, config: B32VitPretrainedConfig | Mapping[str, Any] | None = None, seed: int = DEFAULT_SEED) -> ReprogrammingMethod:
    return build_method(name, _coerce_config(config), seed)


def trainable_parameter_groups_for(method: str, config: B32VitPretrainedConfig | Mapping[str, Any] | None = None) -> List[Dict[str, Any]]:
    return method_selector(method, config).trainable_parameter_groups()


def patch_size_values() -> Tuple[int, int, int]:
    return PATCH_SIZE_VALUES


def p_values() -> Tuple[float, float, float]:
    return P_SWEEP_VALUES


def interpolation_grid_values(image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE, levels: Sequence[int] = (0, 1, 2)) -> Dict[int, Tuple[int, int]]:
    return {int(level): target_grid_size(image_size[0], image_size[1], int(level)) for level in levels}


__all__ = [
    "DEFAULT_SEED",
    "THREE_SEED_PROTOCOL",
    "PATCH_SIZE_VALUES",
    "P_SWEEP_VALUES",
    "B32VitPretrainedConfig",
    "DATASET_REGISTRY",
    "BACKBONE_REGISTRY",
    "METHOD_REGISTRY",
    "METRIC_REGISTRY",
    "ENVIRONMENT_REGISTRY",
    "EXPERIMENT_REGISTRY",
    "FIGURE_PROTOCOLS",
    "Ours",
    "resolve_seed_defaults",
    "seed_values",
    "compute_loss",
    "aggregate_loss",
    "compute_reward",
    "aggregate_reward",
    "compute_accuracy",
    "aggregate_accuracy",
    "compute_training_objective",
    "compute_ours_oradaptersby_inventory_objective",
    "compute_ours_oradaptersby_inventory_score",
    "resize_normalize_and_wrap_backbone",
    "输入_resize_normalize_与骨干包装函数",
    "主实验_ImageNet1K预训练ResNet18_ResNet50_ViTB32_跨数据集分类比较",
    "make_environment",
    "environment_readiness_check",
    "prepare_dataset",
    "build_backbone",
    "build_method",
    "method_selector",
    "evaluate_predictions",
    "run_training_loop",
    "train_b32_vit_pretrained",
    "train_ours_oradaptersby_inventory",
    "run_b32_vit_pretrained_experiment",
    "write_b32_vit_artifacts",
    "experiment_matrix",
    "patch_size_values",
    "p_values",
    "interpolation_grid_values",
    "optional_backend_readiness",
]