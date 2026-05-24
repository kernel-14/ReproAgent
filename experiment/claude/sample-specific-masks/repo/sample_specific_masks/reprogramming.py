"""
Core Sample-specific Multi-channel Mask (SMM) visual reprogramming components.

reference_grounding: chunk_009 /mnt/paper2any/pzw/proj/paperagent/hx/Research_space/Reproduction/paperbench_data/sample-specific-masks/paper.md
reference_grounding: chunk_016_01 /mnt/paper2any/pzw/proj/paperagent/hx/Research_space/Reproduction/paperbench_data/sample-specific-masks/paper.md
reference_grounding: chunk_014_02 /mnt/paper2any/pzw/proj/paperagent/hx/Research_space/Reproduction/paperbench_data/sample-specific-masks/paper.md

The implemented input reprogramming hypothesis is

    f_in(x_i | phi, delta) = r(x_i) + delta * f_mask(r(x_i) | phi)

where r is bilinear resizing to the frozen ImageNet-1K pretrained model input
space, delta is a shared trainable pattern initialized to zeros, and f_mask is a
lightweight CNN producing sample-specific masks.  The module also exposes the
paper's ablation variants (ONLY delta, ONLY f_mask, SINGLE-CHANNEL f_mask^s)
and fixed-mask VR baselines (PAD, Narrow, Medium, Full).
"""

from __future__ import annotations

import importlib
import math
import os
import random
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


DEFAULT_SEED = 0
THREE_SEED_PROTOCOL: Tuple[int, int, int] = (0, 1, 2)
PATCH_SIZE_SWEEP: Tuple[int, int, int] = (4, 2, 1)
P_SWEEP: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
ALPHA_SWEEP: Tuple[float, ...] = (0.0, 0.1, 0.5, 1.0)
GAMMA_SWEEP: Tuple[float, ...] = (0.0, 0.1, 1.0)
SIMILARITY_GUIDANCE_SCALE_SWEEP: Tuple[int, int, int] = (9, 7, 10)

IMAGENET_NORMALIZE: Mapping[str, Tuple[float, float, float]] = {
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
}

SUPPORTED_BACKBONES: Tuple[str, ...] = (
    "resnet18_imagenet1k",
    "resnet50_imagenet1k",
    "vit_b32_imagenet1k",
    "resnet18",
    "resnet50",
    "vit_b32",
    "vit",
    "resnet",
    "imagenet_1k",
)

SUPPORTED_TARGET_DATASETS: Tuple[str, ...] = (
    "CIFAR10",
    "CIFAR100",
    "SVHN",
    "GTSRB",
    "Flowers102",
    "DTD",
    "UCF101",
    "Food101",
    "EuroSAT",
    "OxfordPets",
    "SUN397",
    "StanfordCars",
    "unit-001",
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
    "ours",
    "Ours",
    "only_delta",
    "ONLY δ",
    "only_f_mask",
    "ONLY f_mask",
    "single_channel_mask",
    "SINGLE-CHANNEL f_mask^s",
    "PAD",
    "Pad",
    "pad",
    "Narrow",
    "narrow",
    "Medium",
    "medium",
    "Full",
    "full",
    "vit",
    "resnet",
    "lora",
    "imagenet_1k",
)


def _optional_import(module_name: str) -> Optional[Any]:
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def optional_backend_availability() -> Dict[str, bool]:
    """Lazy availability check for optional backends named by repo plans."""
    return {
        "torch": _optional_import("torch") is not None,
        "torchvision": _optional_import("torchvision") is not None,
        "datasets": _optional_import("datasets") is not None,
        "gym": (_optional_import("gymnasium") is not None) or (_optional_import("gym") is not None),
        "sbi": _optional_import("sbi") is not None,
    }


def _require_torch() -> Any:
    torch = _optional_import("torch")
    if torch is None:
        raise RuntimeError(
            "PyTorch is required to instantiate SMM reprogramming modules. "
            "Install torch for training/evaluation; static import remains lightweight."
        )
    return torch


def _torch_nn_functional() -> Tuple[Any, Any, Any]:
    torch = _require_torch()
    nn = importlib.import_module("torch.nn")
    functional = importlib.import_module("torch.nn.functional")
    return torch, nn, functional


def resolve_seed_defaults(
    seeds: Optional[Sequence[int]] = None,
    *,
    mode: str = "runtime_smoke",
    use_three_seed_protocol: Optional[bool] = None,
) -> List[int]:
    """Resolve executable seed protocol.

    Full mode defaults to the paper-visible three_seed_protocol, while smoke
    mode executes a bounded single-seed route unless explicitly overridden.
    """
    if seeds is not None:
        return [int(seed) for seed in seeds]
    if use_three_seed_protocol is True or mode in {"full", "full_run", "paper_full"}:
        return list(THREE_SEED_PROTOCOL)
    return [DEFAULT_SEED]


def seed_values(
    seeds: Optional[Sequence[int]] = None,
    *,
    mode: str = "runtime_smoke",
    use_three_seed_protocol: Optional[bool] = None,
) -> List[int]:
    return resolve_seed_defaults(
        seeds=seeds,
        mode=mode,
        use_three_seed_protocol=use_three_seed_protocol,
    )


def set_reprogramming_seed(seed: int = DEFAULT_SEED) -> None:
    random.seed(int(seed))
    os.environ.setdefault("PYTHONHASHSEED", str(int(seed)))
    torch = _optional_import("torch")
    if torch is not None:
        torch.manual_seed(int(seed))


@dataclass(frozen=True)
class ReprogrammingConfig:
    """Paper-visible SMM builder config with common benchmark aliases."""

    input_size: Tuple[int, int, int] = (3, 224, 224)
    pretrained_input_size: Tuple[int, int, int] = (3, 224, 224)
    mask_channels: int = 3
    interpolation_level: int = 1
    variant: str = "ours"
    seed: int = DEFAULT_SEED
    learning_rate_delta: float = 1.0e-2
    learning_rate_mask: float = 1.0e-3

    @property
    def image_size(self) -> Tuple[int, int]:
        return (int(self.pretrained_input_size[-2]), int(self.pretrained_input_size[-1]))

    @property
    def method(self) -> str:
        aliases = {
            "ours": "ours",
            "ours_multi_channel": "ours",
            "only_delta": "only_delta",
            "only_f_mask": "only_f_mask",
            "single_channel": "single_channel_mask",
            "single_channel_mask": "single_channel_mask",
        }
        return aliases.get(str(self.variant), str(self.variant))


def _to_package_config(config: ReprogrammingConfig | Mapping[str, Any] | None = None) -> Any:
    pkg = importlib.import_module("sample_specific_masks")
    SMMConfig = getattr(pkg, "SMMConfig")
    MaskGeneratorConfig = getattr(pkg, "MaskGeneratorConfig")
    if isinstance(config, ReprogrammingConfig):
        spec = config
    else:
        raw = dict(config or {})
        if "pretrained_input_size" not in raw and "image_size" in raw:
            image_size = raw["image_size"]
            if isinstance(image_size, Sequence) and not isinstance(image_size, str):
                raw["pretrained_input_size"] = tuple(image_size) if len(tuple(image_size)) == 3 else (int(raw.get("mask_channels", 3)), *tuple(image_size))
        if "input_size" not in raw:
            raw["input_size"] = raw.get("pretrained_input_size", (int(raw.get("mask_channels", 3)), 224, 224))
        accepted = {key: value for key, value in raw.items() if key in ReprogrammingConfig.__dataclass_fields__}
        spec = ReprogrammingConfig(**accepted)
    raw_backbone = ""
    if not isinstance(config, ReprogrammingConfig) and isinstance(config, Mapping):
        raw_backbone = str(config.get("backbone", config.get("model", ""))).lower()
    mask_depth = 6 if ("vit" in raw_backbone or "b32" in raw_backbone or "b_32" in raw_backbone) else 5
    mask = MaskGeneratorConfig(
        input_channels=int(spec.mask_channels),
        output_channels=1 if spec.method == "single_channel_mask" else int(spec.mask_channels),
        depth=mask_depth,
        interpolation_level=int(spec.interpolation_level),
        single_channel=spec.method == "single_channel_mask",
    )
    return SMMConfig(
        method=spec.method,
        backbone=raw_backbone or "resnet18",
        image_size=spec.image_size,
        channels=int(spec.mask_channels),
        seed=int(spec.seed),
        learning_rate_delta=float(spec.learning_rate_delta),
        learning_rate_mask=float(spec.learning_rate_mask),
        mask=mask,
    )


def build_reprogramming(config: ReprogrammingConfig | Mapping[str, Any] | None = None) -> Any:
    """Build the canonical SMM module f_in(x_i|phi, delta)."""

    pkg = importlib.import_module("sample_specific_masks")
    return pkg.build_method("ours" if config is None else _to_package_config(config).method, _to_package_config(config))


def build_method(name: str = "ours", config: ReprogrammingConfig | Mapping[str, Any] | None = None) -> Any:
    pkg = importlib.import_module("sample_specific_masks")
    smm_config = _to_package_config(config)
    return pkg.build_method(name or smm_config.method, smm_config)


def ours(config: ReprogrammingConfig | Mapping[str, Any] | None = None) -> Any:
    return build_method("ours", config)


def only_delta(config: ReprogrammingConfig | Mapping[str, Any] | None = None) -> Any:
    return build_method("only_delta", config)


def only_f_mask(config: ReprogrammingConfig | Mapping[str, Any] | None = None) -> Any:
    return build_method("only_f_mask", config)


def single_channel_mask(config: ReprogrammingConfig | Mapping[str, Any] | None = None) -> Any:
    return build_method("single_channel_mask", config)


Ours = importlib.import_module("sample_specific_masks").Ours
SMMReprogrammer = importlib.import_module("sample_specific_masks").SMMReprogrammer
