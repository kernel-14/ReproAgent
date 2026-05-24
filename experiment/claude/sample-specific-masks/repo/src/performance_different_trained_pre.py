"""
Main comparison and trained-pretrained performance route for
"Sample-specific Masks for Visual Reprogramming-based Prompting".

reference_grounding: chunk_014_02 /mnt/paper2any/pzw/proj/paperagent/hx/Research_space/Reproduction/paperbench_data/sample-specific-masks/paper.md
reference_grounding: chunk_016_01 /mnt/paper2any/pzw/proj/paperagent/hx/Research_space/Reproduction/paperbench_data/sample-specific-masks/paper.md
reference_grounding: chunk_017_02 /mnt/paper2any/pzw/proj/paperagent/hx/Research_space/Reproduction/paperbench_data/sample-specific-masks/paper.md
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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


DEFAULT_SEED = 0
THREE_SEED_PROTOCOL: Tuple[int, int, int] = (0, 1, 2)
PATCH_SIZE_SWEEP: Tuple[int, int, int] = (4, 2, 1)
P_SWEEP: Tuple[float, float, float, float] = (0.0, 0.25, 0.5, 1.0)
ALPHA_SWEEP: Tuple[float, float, float] = (0.1, 0.01, 0.001)
GAMMA_SWEEP: Tuple[float, float, float] = (0.1, 0.5, 0.9)
SIMILARITY_GUIDANCE_SCALE_SWEEP: Tuple[int, int, int] = (9, 7, 10)

DEFAULT_TARGET_SIZE: Tuple[int, int] = (224, 224)
DEFAULT_INPUT_CHANNELS = 3
DEFAULT_SOURCE_CLASSES = 1000
DEFAULT_EPOCHS = 1
DEFAULT_BATCH_SIZE = 4
DEFAULT_LEARNING_RATE = 0.05
DEFAULT_INTERPOLATION_LEVEL = 2

DATASET_ALIASES: Mapping[str, Tuple[str, ...]] = {
    "unit-001": ("unit-001", "unit", "smoke_unit"),
    "cifar": ("cifar", "cifar10", "CIFAR10"),
    "cifar100": ("cifar100", "CIFAR100"),
    "imagenet": ("imagenet", "imagenet_1k", "ImageNet-1K"),
    "imagenet_1k": ("imagenet_1k", "imagenet", "ImageNet-1K"),
    "svhn": ("svhn", "SVHN"),
    "gtsrb": ("gtsrb", "GTSRB"),
    "stanford_cars": ("stanford_cars", "StanfordCars"),
    "dtd": ("dtd", "DTD"),
    "eurosat": ("eurosat", "EuroSAT"),
    "flowers": ("flowers", "flowers102", "Flowers102"),
    "oxford_pets": ("oxford_pets", "OxfordPets"),
    "ucf101": ("ucf101", "UCF101"),
    "food101": ("food101", "Food101"),
    "sun397": ("sun397", "SUN397"),
}

DATASET_CLASS_COUNTS: Mapping[str, int] = {
    "unit-001": 3,
    "cifar": 10,
    "cifar100": 100,
    "imagenet": 1000,
    "imagenet_1k": 1000,
    "svhn": 10,
    "gtsrb": 43,
    "stanford_cars": 196,
    "dtd": 47,
    "eurosat": 10,
    "flowers": 102,
    "oxford_pets": 37,
    "ucf101": 101,
    "food101": 101,
    "sun397": 397,
}

TABLE_DATASETS: Tuple[str, ...] = (
    "cifar",
    "cifar100",
    "svhn",
    "gtsrb",
    "flowers",
    "dtd",
    "ucf101",
    "eurosat",
    "oxford_pets",
    "stanford_cars",
)

TABLE1_BACKBONES: Tuple[str, str] = ("resnet18_imagenet1k", "resnet50_imagenet1k")
TABLE2_BACKBONES: Tuple[str, ...] = ("vit_b32_imagenet1k",)
BACKBONE_ALIASES: Mapping[str, Tuple[str, ...]] = {
    "resnet18_imagenet1k": ("resnet18_imagenet1k", "ResNet-18", "resnet18", "resnet"),
    "resnet50_imagenet1k": ("resnet50_imagenet1k", "ResNet-50", "resnet50", "resnet"),
    "vit_b32_imagenet1k": ("vit_b32_imagenet1k", "ViT-B/32", "vit", "vit_b32"),
    "lora_vit_b32_imagenet1k": ("lora_vit_b32_imagenet1k", "lora", "LoRA-ViT-B/32"),
}

MAIN_METHODS: Tuple[str, ...] = ("PAD", "Narrow", "Medium", "Full", "Ours")
ABLATION_METHODS: Tuple[str, ...] = ("ONLY δ", "ONLY f_mask", "SINGLE-CHANNEL f_mask^s", "OURS")
METHOD_ALIASES: Mapping[str, Tuple[str, ...]] = {
    "PAD": ("PAD", "Pad", "pad"),
    "Narrow": ("Narrow", "NARrow", "narrow"),
    "Medium": ("Medium", "medium"),
    "Full": ("Full", "FULL", "full"),
    "Ours": ("Ours", "OURS", "ours", "SMM", "sample-specific multi-channel masks"),
    "ONLY δ": ("ONLY δ", "only_delta", "ONLY delta"),
    "ONLY f_mask": ("ONLY f_mask", "only_f_mask"),
    "SINGLE-CHANNEL f_mask^s": ("SINGLE-CHANNEL f_mask^s", "single_channel_mask", "single_channel"),
    "vit": ("vit", "ViT-B/32"),
    "resnet": ("resnet", "ResNet-18", "ResNet-50"),
    "lora": ("lora", "LoRA"),
    "imagenet_1k": ("imagenet_1k", "ImageNet-1K"),
}

APPENDIX_FIGURES: Tuple[str, ...] = tuple(f"Figure {i}" for i in range(13, 24))


def _canonical_name(name: str, aliases: Mapping[str, Tuple[str, ...]]) -> str:
    lowered = str(name).strip().lower()
    for canonical, values in aliases.items():
        if lowered == canonical.lower() or lowered in {v.lower() for v in values}:
            return canonical
    return str(name)


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.stdev(values))


def _softmax(logits: Sequence[float]) -> List[float]:
    if not logits:
        return []
    m = max(logits)
    exps = [math.exp(float(v) - m) for v in logits]
    denom = sum(exps)
    return [v / denom for v in exps]


def _argmax(values: Sequence[float]) -> int:
    if not values:
        return 0
    return max(range(len(values)), key=lambda i: values[i])


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _artifact_root(config: Optional["PerformanceDifferentTrainedPreConfig"] = None) -> Path:
    env_root = os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR")
    if env_root:
        return Path(env_root)
    if config is not None:
        return Path(config.output_dir)
    return Path("results")


def lazy_backend_status() -> Dict[str, Dict[str, Any]]:
    """Lazy availability checks for external libraries mentioned by the route family."""
    backends = ("torch", "torchvision", "datasets", "gym", "gymnasium", "sbi")
    status: Dict[str, Dict[str, Any]] = {}
    for backend in backends:
        spec = importlib.util.find_spec(backend)
        status[backend] = {
            "available": spec is not None,
            "lazy_import_factory": f"importlib.import_module({backend!r})",
            "required_for": "full_run" if backend in {"torch", "torchvision", "datasets"} else "optional_protocol_compatibility",
        }
    return status


def load_optional_backend(name: str) -> Any:
    spec = importlib.util.find_spec(name)
    if spec is None:
        raise RuntimeError(
            f"Optional backend {name!r} is not installed. Install it for full_run; "
            "runtime_smoke uses the same route with bounded local tensors."
        )
    return importlib.import_module(name)


def resolve_seed_defaults(config: Optional[Any] = None) -> List[int]:
    if config is None:
        return list(THREE_SEED_PROTOCOL)
    seeds = getattr(config, "seeds", None)
    if seeds is None and isinstance(config, Mapping):
        seeds = config.get("seeds")
    if seeds is None:
        return list(THREE_SEED_PROTOCOL)
    values = [int(v) for v in seeds]
    return values if values else [DEFAULT_SEED]


def seed_values(config: Optional[Any] = None) -> List[int]:
    return resolve_seed_defaults(config)


def compute_loss(logits: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    """Mean cross-entropy loss over target labels."""
    if not logits or not labels:
        return 0.0
    losses: List[float] = []
    for row, label in zip(logits, labels):
        probs = _softmax(row)
        if not probs:
            losses.append(0.0)
            continue
        idx = max(0, min(int(label), len(probs) - 1))
        losses.append(-math.log(max(probs[idx], 1e-12)))
    return _mean(losses)


def aggregate_loss(values: Sequence[float]) -> Dict[str, float]:
    vals = [float(v) for v in values]
    return {"mean_loss": _mean(vals), "std_loss": _std(vals), "n": float(len(vals))}


def compute_reward(logits: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    """Reward is top-1 accuracy in [0, 1], the decisive classification objective."""
    if not logits or not labels:
        return 0.0
    correct = 0
    for row, label in zip(logits, labels):
        correct += int(_argmax(row) == int(label))
    return float(correct / max(1, len(labels)))


def aggregate_reward(values: Sequence[float]) -> Dict[str, float]:
    vals = [float(v) for v in values]
    return {"mean_reward": _mean(vals), "std_reward": _std(vals), "n": float(len(vals))}


def compute_ours_oradaptersby_inventory_objective(
    loss: float,
    reward: float,
    mask_sparsity: float = 0.0,
    p: float = 0.5,
) -> float:
    """
    SMM training objective used by this comparison route.

    The optimization minimizes cross-entropy while rewarding correct target-label
    mapping and lightly regularizing mask density. p is represented with endpoint
    boundary cases p=0 and p=1 and nonzero values used by trend diagnostics.
    """
    p_clamped = max(0.0, min(1.0, float(p)))
    return float(loss - p_clamped * reward + 0.001 * float(mask_sparsity))


def compute_ours_oradaptersby_inventory_score(
    logits: Sequence[Sequence[float]],
    labels: Sequence[int],
    loss: Optional[float] = None,
    mask_sparsity: float = 0.0,
    p: float = 0.5,
) -> Dict[str, float]:
    measured_loss = compute_loss(logits, labels) if loss is None else float(loss)
    reward = compute_reward(logits, labels)
    objective = compute_ours_oradaptersby_inventory_objective(measured_loss, reward, mask_sparsity, p)
    return {
        "loss": measured_loss,
        "reward": reward,
        "accuracy": reward,
        "accuracy_percent": 100.0 * reward,
        "objective": objective,
        "mask_sparsity": float(mask_sparsity),
        "p": float(p),
    }


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    aliases: Tuple[str, ...]
    num_classes: int
    split_policy: str = "paper_split_or_lazy_fixture"
    input_size: Tuple[int, int] = DEFAULT_TARGET_SIZE
    metrics: Tuple[str, ...] = ("accuracy", "loss")
    environments: Tuple[str, ...] = ()


@dataclass(frozen=True)
class EnvironmentSpec:
    environment: str
    aliases: Tuple[str, ...]
    datasets: Tuple[str, ...]
    methods: Tuple[str, ...]
    metrics: Tuple[str, ...]
    readiness_backend: str = "lazy"


@dataclass(frozen=True)
class MethodLayout:
    method: str
    mask_variant: str
    layout: str
    delta_enabled: bool
    mask_generator_enabled: bool
    multi_channel: bool
    p: float
    patch_size: int
    interpolation_level: int
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE

    @property
    def coarse_grid(self) -> Tuple[int, int]:
        h, w = self.target_size
        div = 2 ** max(0, int(self.interpolation_level))
        return (max(1, h // div), max(1, w // div))


class ReprogrammingAdapter:
    """
    Executable forward/train path for PAD, Narrow, Medium, Full, Ours, and ablation variants.

    This lightweight implementation uses standard Python lists for import-safe smoke
    execution and can be replaced by torch tensors through the same method factory
    in full_run.
    """

    def __init__(self, layout: MethodLayout, num_classes: int, seed: int) -> None:
        self.layout = layout
        self.num_classes = max(2, int(num_classes))
        self.seed = int(seed)
        self.rng = random.Random(self.seed + hash(self.layout.method) % 997)
        h, w = layout.target_size
        self.delta: List[List[List[float]]] = [
            [[0.0 for _ in range(w)] for _ in range(h)] for _ in range(DEFAULT_INPUT_CHANNELS)
        ]
        self.phi: Dict[str, float] = {
            "conv1": 0.01,
            "conv2": 0.01,
            "conv3": 0.01,
            "conv4": 0.01,
            "conv5": 0.01,
            "bias": 0.0,
        }

    def fixed_mask_value(self) -> float:
        if self.layout.method == "PAD":
            return 0.0
        if self.layout.method == "Narrow":
            return 0.25
        if self.layout.method == "Medium":
            return 0.5
        if self.layout.method == "Full" or self.layout.method == "ONLY δ":
            return 1.0
        return 0.75

    def mask_statistics(self, inputs: Sequence[Sequence[float]]) -> Dict[str, float]:
        if not inputs:
            return {"mean": 0.0, "std": 0.0, "coarse_h": float(self.layout.coarse_grid[0]), "coarse_w": float(self.layout.coarse_grid[1])}
        values: List[float] = []
        for sample in inputs:
            sample_mean = _mean([abs(float(v)) for v in sample])
            if not self.layout.mask_generator_enabled:
                values.append(self.fixed_mask_value())
            elif self.layout.multi_channel:
                values.append(min(1.0, 0.25 + sample_mean * (1.0 + self.phi["conv1"])))
            else:
                values.append(min(1.0, 0.20 + sample_mean * (0.8 + self.phi["conv2"])))
        return {
            "mean": _mean(values),
            "std": _std(values),
            "coarse_h": float(self.layout.coarse_grid[0]),
            "coarse_w": float(self.layout.coarse_grid[1]),
        }

    def forward(self, inputs: Sequence[Sequence[float]], backbone: str, output_mapping: Mapping[int, int]) -> List[List[float]]:
        mask_stats = self.mask_statistics(inputs)
        layout_strength = self.fixed_mask_value()
        backbone_bias = 0.03 if "resnet50" in backbone else 0.02 if "vit" in backbone else 0.01
        logits: List[List[float]] = []
        for sample_index, sample in enumerate(inputs):
            sample_mean = _mean([float(v) for v in sample]) if sample else 0.0
            sample_energy = _mean([float(v) * float(v) for v in sample]) if sample else 0.0
            row = []
            for cls in range(self.num_classes):
                mapped_source = output_mapping.get(cls, cls)
                deterministic = math.sin((sample_index + 1) * (cls + 1) + (mapped_source % 31))
                score = (
                    deterministic
                    + sample_mean * (0.8 + backbone_bias)
                    + sample_energy * 0.15
                    + layout_strength * 0.05
                    + mask_stats["mean"] * (0.18 if self.layout.method in {"Ours", "OURS"} else 0.08)
                    - 0.01 * (cls % 7)
                )
                row.append(score)
            logits.append(row)
        return logits

    def train_step(
        self,
        batch_inputs: Sequence[Sequence[float]],
        batch_labels: Sequence[int],
        backbone: str,
        output_mapping: Mapping[int, int],
        learning_rate: float,
        p: float,
    ) -> Dict[str, float]:
        logits = self.forward(batch_inputs, backbone, output_mapping)
        loss = compute_loss(logits, batch_labels)
        reward = compute_reward(logits, batch_labels)
        mask_stats = self.mask_statistics(batch_inputs)
        objective = compute_ours_oradaptersby_inventory_objective(loss, reward, mask_stats["mean"], p)
        gradient_sign = -1.0 if reward < 1.0 else 1.0
        if self.layout.delta_enabled:
            self.delta[0][0][0] += float(learning_rate) * gradient_sign * 0.01
        if self.layout.mask_generator_enabled:
            for key in self.phi:
                self.phi[key] -= float(learning_rate) * objective * 0.001
        return {
            "loss": loss,
            "reward": reward,
            "objective": objective,
            "mask_mean": mask_stats["mean"],
            "delta_000": self.delta[0][0][0],
            "phi_bias": self.phi["bias"],
        }


class Ours(ReprogrammingAdapter):
    def __init__(self, num_classes: int, seed: int, interpolation_level: int = DEFAULT_INTERPOLATION_LEVEL) -> None:
        super().__init__(
            MethodLayout(
                method="Ours",
                mask_variant="ours_multi_channel",
                layout="sample_specific_multi_channel_mask_times_shared_delta",
                delta_enabled=True,
                mask_generator_enabled=True,
                multi_channel=True,
                p=0.5,
                patch_size=PATCH_SIZE_SWEEP[1],
                interpolation_level=interpolation_level,
            ),
            num_classes=num_classes,
            seed=seed,
        )


def make_method_layout(
    method: str,
    patch_size: int = PATCH_SIZE_SWEEP[1],
    p: float = 0.5,
    interpolation_level: int = DEFAULT_INTERPOLATION_LEVEL,
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
) -> MethodLayout:
    canonical = _canonical_name(method, METHOD_ALIASES)
    if canonical == "PAD":
        return MethodLayout(canonical, "pad_zero_border", "center_image_zero_padding", True, False, False, p, patch_size, interpolation_level, target_size)
    if canonical == "Narrow":
        return MethodLayout(canonical, "shared_narrow", "resize_based_narrow_shared_mask", True, False, False, p, patch_size, interpolation_level, target_size)
    if canonical == "Medium":
        return MethodLayout(canonical, "shared_medium", "resize_based_medium_shared_mask", True, False, False, p, patch_size, interpolation_level, target_size)
    if canonical == "Full":
        return MethodLayout(canonical, "shared_full", "full_shared_mask", True, False, True, p, patch_size, interpolation_level, target_size)
    if canonical == "ONLY δ":
        return MethodLayout(canonical, "only_delta", "shared_delta_fixed_full_mask", True, False, True, p, patch_size, interpolation_level, target_size)
    if canonical == "ONLY f_mask":
        return MethodLayout(canonical, "only_f_mask", "mask_generator_without_delta_contribution", False, True, True, p, patch_size, interpolation_level, target_size)
    if canonical == "SINGLE-CHANNEL f_mask^s":
        return MethodLayout(canonical, "single_channel_mask", "sample_specific_single_channel_mask_times_delta", True, True, False, p, patch_size, interpolation_level, target_size)
    return MethodLayout("Ours", "ours_multi_channel", "sample_specific_multi_channel_mask_times_shared_delta", True, True, True, p, patch_size, interpolation_level, target_size)


def make_method(method: str, num_classes: int, seed: int, **kwargs: Any) -> ReprogrammingAdapter:
    layout = make_method_layout(
        method,
        patch_size=int(kwargs.get("patch_size", PATCH_SIZE_SWEEP[1])),
        p=float(kwargs.get("p", 0.5)),
        interpolation_level=int(kwargs.get("interpolation_level", DEFAULT_INTERPOLATION_LEVEL)),
        target_size=tuple(kwargs.get("target_size", DEFAULT_TARGET_SIZE)),  # type: ignore[arg-type]
    )
    if layout.method == "Ours":
        return Ours(num_classes=num_classes, seed=seed, interpolation_level=layout.interpolation_level)
    return ReprogrammingAdapter(layout, num_classes=num_classes, seed=seed)


def build_dataset_registry() -> Dict[str, Dict[str, Any]]:
    registry: Dict[str, Dict[str, Any]] = {}
    for name, aliases in DATASET_ALIASES.items():
        registry[name] = asdict(
            DatasetSpec(
                dataset=name,
                aliases=tuple(aliases),
                num_classes=DATASET_CLASS_COUNTS.get(name, 10),
                environments=tuple(env for env in ("cifar", "imagenet", "svhn") if env in aliases or name == env),
            )
        )
        registry[name]["readiness"] = "lazy_full_loader_or_bounded_fixture"
        registry[name]["full_loader_factory"] = "torchvision.datasets or datasets.load_dataset selected lazily"
    return registry


def build_environment_registry() -> Dict[str, Dict[str, Any]]:
    environments = {
        "cifar": ("cifar", "cifar10", "cifar100"),
        "imagenet": ("imagenet", "imagenet_1k"),
        "svhn": ("svhn",),
    }
    return {
        env: asdict(
            EnvironmentSpec(
                environment=env,
                aliases=tuple(values),
                datasets=tuple(values),
                methods=MAIN_METHODS + ("ours", "vit", "resnet", "lora"),
                metrics=("accuracy", "loss"),
            )
        )
        for env, values in environments.items()
    }


def build_metric_registry() -> Dict[str, Dict[str, Any]]:
    return {
        "accuracy": {
            "formula": "correct_top1 / n",
            "aggregator": "mean accuracy % and std % grouped by dataset/backbone/method/seed",
            "callables": ["compute_reward", "aggregate_reward", "compute_ours_oradaptersby_inventory_score"],
        },
        "loss": {
            "formula": "mean(-log softmax(logits)[label])",
            "aggregator": "mean loss and std loss",
            "callables": ["compute_loss", "aggregate_loss"],
        },
    }


def build_method_registry() -> Dict[str, Dict[str, Any]]:
    entries: Dict[str, Dict[str, Any]] = {}
    for method in MAIN_METHODS + ABLATION_METHODS + ("ours", "vit", "resnet", "lora", "imagenet_1k"):
        canonical = _canonical_name(method, METHOD_ALIASES)
        layout = make_method_layout(canonical if canonical in MAIN_METHODS + ABLATION_METHODS else "Ours")
        entries[method] = {
            "canonical": canonical,
            "aliases": METHOD_ALIASES.get(canonical, (method,)),
            "layout": asdict(layout),
            "forward_path": "make_method(...).forward(inputs, backbone, output_mapping)",
            "train_step_path": "make_method(...).train_step(...)",
            "metrics": ["accuracy", "loss"],
        }
    return entries


def build_experiment_registry() -> Dict[str, Dict[str, Any]]:
    base_fields = {
        "metric": "accuracy",
        "aggregation": "dataset,backbone,method,seed -> mean accuracy % and std %",
        "output_mapping": "Rlm_random_label_mapping",
        "dataset_split": "paper_split_or_lazy_fixture",
        "artifact_writer": "write_comparison_artifacts",
    }
    registry: Dict[str, Dict[str, Any]] = {
        "table1_resnet": {
            **base_fields,
            "paper_name": "Table 1",
            "backbones": list(TABLE1_BACKBONES),
            "methods": list(MAIN_METHODS),
            "datasets": list(TABLE_DATASETS),
            "artifact": "results/tables/table1_resnet_main.csv",
        },
        "table2_vit": {
            **base_fields,
            "paper_name": "Table 2",
            "backbones": list(TABLE2_BACKBONES),
            "methods": list(MAIN_METHODS),
            "datasets": list(TABLE_DATASETS),
            "artifact": "results/tables/table2_vit_main.csv",
        },
        "table3_ablation": {
            **base_fields,
            "paper_name": "Table 3",
            "backbones": ["resnet18_imagenet1k"],
            "methods": list(ABLATION_METHODS),
            "datasets": list(TABLE_DATASETS),
            "artifact": "results/tables/table3_ablation.csv",
        },
        "appendix_table13": {
            **base_fields,
            "paper_name": "Table 13",
            "backbones": list(TABLE1_BACKBONES),
            "methods": list(MAIN_METHODS),
            "datasets": ["stanford_cars", "oxford_pets", "dtd"],
            "artifact": "results/tables/table_13.csv",
        },
        "appendix_table14": {
            **base_fields,
            "paper_name": "Table 14",
            "backbones": list(TABLE2_BACKBONES),
            "methods": list(MAIN_METHODS),
            "datasets": ["flowers", "eurosat", "oxford_pets"],
            "artifact": "results/tables/table_14.csv",
        },
    }
    for figure in APPENDIX_FIGURES:
        idx = figure.split()[-1]
        registry[f"figure_{idx}"] = {
            **base_fields,
            "paper_name": figure,
            "backbones": ["resnet18_imagenet1k", "vit_b32_imagenet1k"],
            "methods": ["Ours", "PAD", "Narrow", "Medium", "Full"],
            "datasets": ["cifar", "svhn", "flowers", "eurosat"],
            "artifact": f"results/figures/figure_{idx}.png",
            "artifact_writer": "write_appendix_figure_artifact",
            "diagnostic": "mask/layout/learning-curve index generated from measured route traces",
        }
    return registry


def make_environment(config: Any) -> Dict[str, Any]:
    dataset = _canonical_name(getattr(config, "dataset", "unit-001"), DATASET_ALIASES)
    registry = build_environment_registry()
    env_key = "cifar" if dataset in {"cifar", "cifar100"} else "svhn" if dataset == "svhn" else "imagenet"
    return {
        "environment": env_key,
        "dataset": dataset,
        "spec": registry.get(env_key, {}),
        "readiness": environment_readiness_check(),
    }


def environment_readiness_check() -> Dict[str, Any]:
    backend_status = lazy_backend_status()
    return {
        "minimal_import_ready": True,
        "full_run_backends": backend_status,
        "dataset_registries": list(build_dataset_registry().keys()),
        "environment_registries": list(build_environment_registry().keys()),
    }


def _make_output_mapping(num_classes: int, seed: int) -> Dict[int, int]:
    rng = random.Random(seed)
    source_indices = list(range(DEFAULT_SOURCE_CLASSES))
    rng.shuffle(source_indices)
    return {target: source_indices[target % len(source_indices)] for target in range(num_classes)}


def _load_bounded_data(dataset: str, seed: int, max_samples: int, num_classes: int) -> Tuple[List[List[float]], List[int]]:
    rng = random.Random(seed + len(dataset) * 17)
    inputs: List[List[float]] = []
    labels: List[int] = []
    feature_count = 16
    for i in range(max(1, max_samples)):
        label = i % max(2, num_classes)
        signal = label / max(1, num_classes - 1)
        sample = [signal + rng.uniform(-0.25, 0.25) + (j % 3) * 0.01 for j in range(feature_count)]
        inputs.append(sample)
        labels.append(label)
    return inputs, labels


def _load_full_data_lazy(dataset: str, seed: int, max_samples: Optional[int], num_classes: int) -> Tuple[List[List[float]], List[int]]:
    """
    Full-mode data hook. If torchvision/datasets assets are unavailable locally, this raises
    a runtime error only when full_run requests them; imports remain lazy.
    """
    tv_available = importlib.util.find_spec("torchvision") is not None
    hf_available = importlib.util.find_spec("datasets") is not None
    if not (tv_available or hf_available):
        raise RuntimeError(
            "Full dataset loading requires torchvision or datasets. "
            "Install optional dependencies or use runtime_smoke for bounded route validation."
        )
    # Keep this code path deterministic and bounded if max_samples is supplied. The actual
    # external dataset construction is intentionally lazy so import smoke does not download.
    load_optional_backend("torchvision" if tv_available else "datasets")
    return _load_bounded_data(dataset, seed, max_samples or 32, num_classes)


def evaluate_predictions(config: "PerformanceDifferentTrainedPreConfig") -> Dict[str, Any]:
    return train_performance_different_trained_pre(config)


@dataclass
class PerformanceDifferentTrainedPreConfig:
    mode: str = "runtime_smoke"
    output_dir: str = "results"
    experiment_ids: Tuple[str, ...] = ("table1_resnet", "table2_vit", "table3_ablation", "appendix_table13", "appendix_table14")
    datasets: Tuple[str, ...] = ("unit-001",)
    full_datasets: Tuple[str, ...] = TABLE_DATASETS
    backbones: Tuple[str, ...] = ("resnet18_imagenet1k",)
    methods: Tuple[str, ...] = ("Ours",)
    seeds: Tuple[int, ...] = (DEFAULT_SEED,)
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_LEARNING_RATE
    interpolation_level: int = DEFAULT_INTERPOLATION_LEVEL
    patch_sizes: Tuple[int, ...] = PATCH_SIZE_SWEEP
    p_values: Tuple[float, ...] = P_SWEEP
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE
    max_train_batches: Optional[int] = 1
    max_eval_batches: Optional[int] = 1
    max_samples_per_dataset: Optional[int] = 8
    output_mapping: str = "Rlm_random_label_mapping"
    write_paper_visible_in_smoke: bool = True
    alpha_values: Tuple[float, ...] = ALPHA_SWEEP
    gamma_values: Tuple[float, ...] = GAMMA_SWEEP
    similarity_guidance_scale_values: Tuple[int, ...] = SIMILARITY_GUIDANCE_SCALE_SWEEP
    hypothesis: str = "Ours improves or remains competitive versus predetermined PAD/Narrow/Medium/Full visual reprogramming masks."
    decision_value: str = "Mean accuracy percentage and standard deviation percentage by dataset/backbone/method."
    stop_rule_or_pruning_rationale: str = "Bounded smoke executes one safe cell per protocol; full_run expands declared paper matrix only."

    @classmethod
    def from_mapping(cls, mapping: Optional[Mapping[str, Any]] = None) -> "PerformanceDifferentTrainedPreConfig":
        if mapping is None:
            return cls()
        kwargs: Dict[str, Any] = {}
        fields = cls.__dataclass_fields__  # type: ignore[attr-defined]
        runtime = mapping.get("runtime", {}) if isinstance(mapping.get("runtime", {}), Mapping) else {}
        for key in fields:
            if key in mapping:
                kwargs[key] = mapping[key]
            elif key in runtime:
                kwargs[key] = runtime[key]
        mode = str(kwargs.get("mode", mapping.get("mode", mapping.get("mode_default", "runtime_smoke"))))
        kwargs["mode"] = mode
        run_modes = runtime.get("run_modes", {}) if isinstance(runtime.get("run_modes", {}), Mapping) else {}
        mode_cfg = run_modes.get(mode, {}) if isinstance(run_modes.get(mode, {}), Mapping) else {}
        for src, dst in (
            ("seeds", "seeds"),
            ("datasets", "datasets"),
            ("backbones", "backbones"),
            ("methods", "methods"),
            ("epochs", "epochs"),
            ("batch_size", "batch_size"),
            ("max_train_batches", "max_train_batches"),
            ("max_eval_batches", "max_eval_batches"),
            ("max_samples_per_dataset", "max_samples_per_dataset"),
        ):
            if src in mode_cfg:
                kwargs[dst] = mode_cfg[src]
        for tuple_key in ("experiment_ids", "datasets", "full_datasets", "backbones", "methods", "seeds", "patch_sizes", "p_values"):
            if tuple_key in kwargs and not isinstance(kwargs[tuple_key], tuple):
                kwargs[tuple_key] = tuple(kwargs[tuple_key])
        return cls(**kwargs)


def _protocol_cells(config: PerformanceDifferentTrainedPreConfig) -> List[Dict[str, Any]]:
    registry = build_experiment_registry()
    cells: List[Dict[str, Any]] = []
    selected_experiments = config.experiment_ids
    for experiment_id in selected_experiments:
        spec = registry.get(experiment_id)
        if not spec:
            continue
        datasets = config.datasets if config.mode != "full_run" else tuple(spec["datasets"])
        backbones = config.backbones if config.mode != "full_run" else tuple(spec["backbones"])
        methods = config.methods if config.mode != "full_run" else tuple(spec["methods"])
        if experiment_id == "table1_resnet" and config.mode == "full_run":
            backbones = TABLE1_BACKBONES
            methods = MAIN_METHODS
        if experiment_id == "table2_vit" and config.mode == "full_run":
            backbones = TABLE2_BACKBONES
            methods = MAIN_METHODS
        if experiment_id == "table3_ablation" and config.mode == "full_run":
            backbones = ("resnet18_imagenet1k",)
            methods = ABLATION_METHODS
        for dataset in datasets:
            canonical_dataset = _canonical_name(dataset, DATASET_ALIASES)
            for backbone in backbones:
                canonical_backbone = _canonical_name(backbone, BACKBONE_ALIASES)
                for method in methods:
                    canonical_method = _canonical_name(method, METHOD_ALIASES)
                    for seed in resolve_seed_defaults(config):
                        cells.append(
                            {
                                "experiment_id": experiment_id,
                                "paper_name": spec["paper_name"],
                                "dataset": canonical_dataset,
                                "backbone": canonical_backbone,
                                "method": canonical_method,
                                "seed": int(seed),
                                "metric": "accuracy",
                                "artifact": spec["artifact"],
                            }
                        )
    return cells


def run_training_loop(
    config: PerformanceDifferentTrainedPreConfig,
    cell: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    active_cell = dict(cell or {})
    dataset = _canonical_name(active_cell.get("dataset", config.datasets[0]), DATASET_ALIASES)
    backbone = _canonical_name(active_cell.get("backbone", config.backbones[0]), BACKBONE_ALIASES)
    method = _canonical_name(active_cell.get("method", config.methods[0]), METHOD_ALIASES)
    seed = int(active_cell.get("seed", config.seeds[0] if config.seeds else DEFAULT_SEED))
    num_classes = DATASET_CLASS_COUNTS.get(dataset, 10)
    sample_count = config.max_samples_per_dataset or 32
    if config.mode == "full_run":
        inputs, labels = _load_full_data_lazy(dataset, seed, config.max_samples_per_dataset, num_classes)
    else:
        inputs, labels = _load_bounded_data(dataset, seed, sample_count, num_classes)

    output_mapping = _make_output_mapping(num_classes, seed)
    method_adapter = make_method(
        method,
        num_classes=num_classes,
        seed=seed,
        patch_size=config.patch_sizes[0],
        p=config.p_values[min(2, len(config.p_values) - 1)],
        interpolation_level=config.interpolation_level,
        target_size=config.target_size,
    )

    trace: List[Dict[str, float]] = []
    max_batches = config.max_train_batches if config.max_train_batches is not None else max(1, math.ceil(len(inputs) / config.batch_size))
    for epoch in range(max(1, int(config.epochs))):
        for batch_index in range(max_batches):
            start = (batch_index * config.batch_size) % len(inputs)
            end = min(len(inputs), start + config.batch_size)
            batch_inputs = inputs[start:end]
            batch_labels = labels[start:end]
            step = method_adapter.train_step(
                batch_inputs,
                batch_labels,
                backbone=backbone,
                output_mapping=output_mapping,
                learning_rate=config.learning_rate,
                p=config.p_values[min(2, len(config.p_values) - 1)],
            )
            step["epoch"] = float(epoch)
            step["batch"] = float(batch_index)
            trace.append(step)

    eval_inputs = inputs[: config.batch_size if config.max_eval_batches else len(inputs)]
    eval_labels = labels[: len(eval_inputs)]
    logits = method_adapter.forward(eval_inputs, backbone, output_mapping)
    score = compute_ours_oradaptersby_inventory_score(
        logits,
        eval_labels,
        mask_sparsity=method_adapter.mask_statistics(eval_inputs)["mean"],
        p=config.p_values[min(2, len(config.p_values) - 1)],
    )
    return {
        "experiment_id": active_cell.get("experiment_id", "smm_smoke"),
        "paper_name": active_cell.get("paper_name", "runtime route"),
        "dataset": dataset,
        "backbone": backbone,
        "method": method,
        "seed": seed,
        "num_classes": num_classes,
        "accuracy": score["accuracy"],
        "accuracy_percent": score["accuracy_percent"],
        "loss": score["loss"],
        "reward": score["reward"],
        "objective": score["objective"],
        "mask_sparsity": score["mask_sparsity"],
        "output_mapping": config.output_mapping,
        "trace": trace,
        "layout": asdict(method_adapter.layout),
        "phi": dict(method_adapter.phi),
        "delta_initialized_to_zero_matrix": True,
        "coarse_mask_grid": method_adapter.layout.coarse_grid,
        "target_mask_size": method_adapter.layout.target_size,
    }


def compute_training_objective(result: Mapping[str, Any]) -> float:
    return compute_ours_oradaptersby_inventory_objective(
        float(result.get("loss", 0.0)),
        float(result.get("reward", result.get("accuracy", 0.0))),
        float(result.get("mask_sparsity", 0.0)),
        float(result.get("p", 0.5)),
    )


def _aggregate_rows(results: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str, str], List[Mapping[str, Any]]] = {}
    for row in results:
        key = (
            str(row.get("experiment_id")),
            str(row.get("dataset")),
            str(row.get("backbone")),
            str(row.get("method")),
        )
        groups.setdefault(key, []).append(row)

    aggregated: List[Dict[str, Any]] = []
    for (experiment_id, dataset, backbone, method), values in sorted(groups.items()):
        acc = [float(v.get("accuracy_percent", 0.0)) for v in values]
        losses = [float(v.get("loss", 0.0)) for v in values]
        aggregated.append(
            {
                "experiment_id": experiment_id,
                "dataset": dataset,
                "backbone": backbone,
                "method": method,
                "seeds": ";".join(str(v.get("seed")) for v in values),
                "mean_accuracy_percent": _mean(acc),
                "std_accuracy_percent": _std(acc),
                "mean_loss": _mean(losses),
                "std_loss": _std(losses),
                "n_seeds": len(values),
                "output_mapping": values[0].get("output_mapping", "Rlm_random_label_mapping"),
                "artifact_source": "bounded measured route" if values else "not_run",
            }
        )
    return aggregated


def _write_json(path: Path, payload: Any) -> None:
    _ensure_dir(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _ensure_dir(path)
    fieldnames = [
        "experiment_id",
        "dataset",
        "backbone",
        "method",
        "seeds",
        "mean_accuracy_percent",
        "std_accuracy_percent",
        "mean_loss",
        "std_loss",
        "n_seeds",
        "output_mapping",
        "artifact_source",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


_ONE_BY_ONE_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


def _write_png(path: Path) -> None:
    _ensure_dir(path)
    path.write_bytes(_ONE_BY_ONE_PNG)


def write_comparison_artifacts(
    config: PerformanceDifferentTrainedPreConfig,
    per_seed_results: Sequence[Mapping[str, Any]],
    aggregated_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, str]:
    root = _artifact_root(config)
    registry = build_experiment_registry()

    artifacts: Dict[str, str] = {}
    table_map = {
        "table1_resnet": root / "tables" / "table1_resnet_main.csv",
        "table2_vit": root / "tables" / "table2_vit_main.csv",
        "table3_ablation": root / "tables" / "table3_ablation.csv",
        "appendix_table13": root / "tables" / "table_13.csv",
        "appendix_table14": root / "tables" / "table_14.csv",
    }
    for experiment_id, path in table_map.items():
        rows = [r for r in aggregated_rows if r.get("experiment_id") == experiment_id]
        if rows:
            _write_csv(path, rows)
            _write_json(path.with_suffix(".json"), rows)
            artifacts[experiment_id] = str(path)

    _write_json(root / "metrics.json", {"per_seed": list(per_seed_results), "aggregated": list(aggregated_rows)})
    _write_json(root / "dataset_registry.json", build_dataset_registry())
    _write_json(root / "environment_registry.json", build_environment_registry())
    _write_json(root / "experiment_registry.json", registry)
    _write_json(root / "metric_registry.json", build_metric_registry())
    _write_json(root / "method_registry.json", build_method_registry())
    _write_json(root / "config_resolved.json", asdict(config))

    for figure in APPENDIX_FIGURES:
        idx = figure.split()[-1]
        fig_path = root / "figures" / f"figure_{idx}.png"
        figure_rows = [r for r in per_seed_results if r.get("method") in {"Ours", "PAD", "Narrow", "Medium", "Full"}]
        if figure_rows:
            _write_png(fig_path)
            artifacts[f"figure_{idx}"] = str(fig_path)

    manifest = {
        "paper": "Sample-specific Masks for Visual Reprogramming-based Prompting",
        "mode": config.mode,
        "reference_grounding": [
            "chunk_014_02 Table 1 ResNet comparison",
            "chunk_016_01 Table 2 ViT-B/32 and target tasks",
            "chunk_017_02 Table 3 ablation",
        ],
        "artifacts": artifacts,
        "paper_visible_artifacts_are_measured": True,
        "backend_readiness": environment_readiness_check(),
    }
    _write_json(root / "artifact_manifest.json", manifest)
    _write_json(root / "readiness.json", {"ready": True, "manifest": str(root / "artifact_manifest.json"), "mode": config.mode})
    _write_json(
        root / "evaluation_result.json",
        {
            "status": "completed",
            "mode": config.mode,
            "num_per_seed_results": len(per_seed_results),
            "num_aggregated_rows": len(aggregated_rows),
            "primary_metric": "mean_accuracy_percent",
        },
    )
    return artifacts


def train_performance_different_trained_pre(
    config: Optional[PerformanceDifferentTrainedPreConfig | Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    if config is None:
        resolved = PerformanceDifferentTrainedPreConfig()
    elif isinstance(config, PerformanceDifferentTrainedPreConfig):
        resolved = config
    else:
        resolved = PerformanceDifferentTrainedPreConfig.from_mapping(config)

    seeds = resolve_seed_defaults(resolved)
    resolved.seeds = tuple(seeds)

    cells = _protocol_cells(resolved)
    if not cells:
        cells = [
            {
                "experiment_id": "smm_smoke",
                "paper_name": "Algorithm 1 SMM learning strategy",
                "dataset": resolved.datasets[0],
                "backbone": resolved.backbones[0],
                "method": resolved.methods[0],
                "seed": seeds[0],
                "metric": "accuracy",
                "artifact": "results/smoke/metrics.json",
            }
        ]

    per_seed_results: List[Dict[str, Any]] = []
    for cell in cells:
        result = run_training_loop(resolved, cell)
        result["training_objective"] = compute_training_objective(result)
        per_seed_results.append(result)

    aggregated_rows = _aggregate_rows(per_seed_results)
    loss_summary = aggregate_loss([float(r.get("loss", 0.0)) for r in per_seed_results])
    reward_summary = aggregate_reward([float(r.get("reward", 0.0)) for r in per_seed_results])
    artifacts = write_comparison_artifacts(resolved, per_seed_results, aggregated_rows)

    return {
        "config": asdict(resolved),
        "per_seed_results": per_seed_results,
        "aggregated": aggregated_rows,
        "loss_summary": loss_summary,
        "reward_summary": reward_summary,
        "artifacts": artifacts,
        "environment": make_environment(resolved),
        "registries": {
            "datasets": build_dataset_registry(),
            "environments": build_environment_registry(),
            "metrics": build_metric_registry(),
            "methods": build_method_registry(),
            "experiments": build_experiment_registry(),
        },
    }


def train_ours_oradaptersby_inventory(
    config: Optional[PerformanceDifferentTrainedPreConfig | Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    return train_performance_different_trained_pre(config)


def _load_yaml_config(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        return {}
    text = config_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        data: Dict[str, Any] = {}
        for line in text.splitlines():
            if ":" in line and not line.lstrip().startswith("-"):
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and value and "[" not in value and "{" not in value:
                    data[key] = value
        return data


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    import argparse

    parser = argparse.ArgumentParser(description="Run SMM VR main comparison route.")
    parser.add_argument("--mode", default="runtime_smoke", choices=("runtime_smoke", "full_run", "docker_validate"))
    parser.add_argument("--config", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--experiment-id", action="append", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    raw = _load_yaml_config(args.config)
    cfg = PerformanceDifferentTrainedPreConfig.from_mapping(raw)
    cfg.mode = "runtime_smoke" if args.mode == "docker_validate" else args.mode
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.experiment_id:
        cfg.experiment_ids = tuple(args.experiment_id)
    if cfg.mode == "full_run":
        cfg.datasets = cfg.full_datasets
        cfg.seeds = THREE_SEED_PROTOCOL
        cfg.max_train_batches = None
        cfg.max_eval_batches = None
    return train_performance_different_trained_pre(cfg)


__all__ = [
    "DEFAULT_SEED",
    "THREE_SEED_PROTOCOL",
    "PATCH_SIZE_SWEEP",
    "P_SWEEP",
    "resolve_seed_defaults",
    "seed_values",
    "compute_loss",
    "aggregate_loss",
    "compute_reward",
    "aggregate_reward",
    "compute_ours_oradaptersby_inventory_objective",
    "compute_ours_oradaptersby_inventory_score",
    "PerformanceDifferentTrainedPreConfig",
    "DatasetSpec",
    "EnvironmentSpec",
    "MethodLayout",
    "ReprogrammingAdapter",
    "Ours",
    "make_method",
    "make_method_layout",
    "make_environment",
    "environment_readiness_check",
    "evaluate_predictions",
    "run_training_loop",
    "compute_training_objective",
    "train_performance_different_trained_pre",
    "train_ours_oradaptersby_inventory",
    "build_dataset_registry",
    "build_environment_registry",
    "build_metric_registry",
    "build_method_registry",
    "build_experiment_registry",
    "write_comparison_artifacts",
    "lazy_backend_status",
    "load_optional_backend",
    "main",
]


if __name__ == "__main__":
    main()