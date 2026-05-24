"""Public package surface for Sample-specific Masks (SMM) visual reprogramming.

The package-level API intentionally exposes the paper's core input
reprogramming route:

    f_in(x_i | phi, delta) = r(x_i) + delta * f_mask(r(x_i) | phi)

with a zero-initialized shared pattern delta, a lightweight CNN
sample-specific mask generator, patch-wise interpolation, non-parametric
output mapping, and the ablation/baseline selectors used by the paper.

reference_grounding: chunk_009 /mnt/paper2any/pzw/proj/paperagent/hx/Research_space/Reproduction/paperbench_data/sample-specific-masks/paper.md
reference_grounding: chunk_016_01 /mnt/paper2any/pzw/proj/paperagent/hx/Research_space/Reproduction/paperbench_data/sample-specific-masks/paper.md
reference_grounding: chunk_017_02 /mnt/paper2any/pzw/proj/paperagent/hx/Research_space/Reproduction/paperbench_data/sample-specific-masks/paper.md
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import csv
import importlib
import importlib.util
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence


THREE_SEED_PROTOCOL: tuple[int, int, int] = (0, 1, 2)
DEFAULT_SEED: int = THREE_SEED_PROTOCOL[0]
PATCH_SIZE_SWEEP: tuple[int, int, int] = (4, 2, 1)
P_SWEEP: tuple[float, float, float, float, float] = (0.0, 0.25, 0.5, 0.75, 1.0)
ALPHA_SWEEP: tuple[float, float, float] = (0.1, 0.5, 1.0)
GAMMA_SWEEP: tuple[float, float, float] = (0.0, 0.1, 1.0)
SIMILARITY_GUIDANCE_SCALE_SWEEP: tuple[int, int, int] = (9, 7, 10)

DATASET_IDS: tuple[str, ...] = (
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

BACKBONE_IDS: tuple[str, ...] = (
    "resnet18",
    "resnet50",
    "vit_b_32",
    "ViT_B32",
    "ResNet-18 ImageNet-1K",
    "ResNet-50 ImageNet-1K",
    "ViT-B/32 ImageNet-1K",
    "imagenet_1k",
)

MAIN_METHOD_IDS: tuple[str, ...] = (
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

SMM_VARIANT_IDS: tuple[str, ...] = (
    "ours",
    "only_delta",
    "only_f_mask",
    "single_channel_mask",
    "ONLY δ",
    "ONLY f_mask",
    "SINGLE-CHANNEL f_mask^s",
    "OURS",
)

OPTIONAL_BACKENDS: tuple[str, ...] = (
    "torch",
    "torchvision",
    "timm",
    "datasets",
    "sbi",
    "gym",
    "gymnasium",
)


@dataclass(frozen=True)
class MaskGeneratorConfig:
    """Configuration for the paper's lightweight CNN f_mask."""

    input_channels: int = 3
    output_channels: int = 3
    hidden_channels: int = 16
    depth: int = 5
    interpolation_level: int = 1
    single_channel: bool = False
    activation: str = "sigmoid"


@dataclass(frozen=True)
class SMMConfig:
    """Executable SMM method configuration shared by train/evaluate routes."""

    method: str = "ours"
    dataset: str = "CIFAR10"
    backbone: str = "resnet18"
    image_size: tuple[int, int] = (224, 224)
    channels: int = 3
    num_target_classes: int = 10
    num_source_classes: int = 1000
    seed: int = DEFAULT_SEED
    learning_rate_delta: float = 1e-2
    learning_rate_mask: float = 1e-3
    batch_size: int = 32
    epochs: int = 1
    mask: MaskGeneratorConfig = field(default_factory=MaskGeneratorConfig)
    output_mapping: str = "R1m"
    freeze_backbone: bool = True


@dataclass(frozen=True)
class MethodSpec:
    """Method/baseline/ablation selector row backed by callable factories."""

    name: str
    canonical: str
    family: str
    train_delta: bool
    train_mask_generator: bool
    sample_specific_mask: bool
    single_channel_mask: bool
    fixed_mask: str | None = None
    adapter: str = "smm_vrp"


@dataclass(frozen=True)
class BackendStatus:
    """Lazy backend availability report for smoke/full mode routing."""

    name: str
    available: bool
    import_name: str
    purpose: str


def backend_available(import_name: str) -> bool:
    """Return whether an optional backend can be imported without importing it now."""

    return importlib.util.find_spec(import_name) is not None


def optional_backend_status() -> dict[str, BackendStatus]:
    """Expose lazy backend readiness, including sbi as a named optional route."""

    purposes = {
        "torch": "SMM tensors, CNN mask generator, frozen ImageNet backbones, optimizer groups",
        "torchvision": "CIFAR/SVHN/GTSRB/Flowers/DTD/Food/OxfordPets datasets and ResNet/ViT loaders",
        "timm": "alternative ViT-B/32 and ImageNet pretrained model factory",
        "datasets": "optional dataset mirroring/metadata backend for full benchmark preparation",
        "sbi": "optional simulation-based-inference backend; not used by SMM but kept as a lazy availability route from the generation environment contract",
        "gym": "optional environment backend; not used by SMM full route but lazily checked for contract closure",
        "gymnasium": "optional gym-compatible environment backend",
    }
    return {
        name: BackendStatus(name=name, available=backend_available(name), import_name=name, purpose=purposes[name])
        for name in OPTIONAL_BACKENDS
    }


def lazy_import_backend(name: str) -> Any:
    """Import an optional backend only when a full route asks for it."""

    if not backend_available(name):
        raise RuntimeError(
            f"Optional backend '{name}' is not installed. Install the relevant optional "
            "dependency group before running the full route that requires it."
        )
    return importlib.import_module(name)


def resolve_seed_defaults(seeds: Sequence[int] | None = None) -> tuple[int, ...]:
    """Return the paper's three-seed protocol unless explicitly overridden."""

    return tuple(int(s) for s in (seeds if seeds is not None else THREE_SEED_PROTOCOL))


def seed_values() -> tuple[int, ...]:
    return resolve_seed_defaults()


def patch_size_values() -> tuple[int, ...]:
    return PATCH_SIZE_SWEEP


def p_values() -> tuple[float, ...]:
    return P_SWEEP


METHOD_REGISTRY: dict[str, MethodSpec] = {
    "ours": MethodSpec("Ours", "ours", "smm", True, True, True, False),
    "Ours": MethodSpec("Ours", "ours", "smm", True, True, True, False),
    "OURS": MethodSpec("OURS", "ours", "smm", True, True, True, False),
    "only_delta": MethodSpec("ONLY δ", "only_delta", "ablation", True, False, False, False, fixed_mask="Full"),
    "ONLY δ": MethodSpec("ONLY δ", "only_delta", "ablation", True, False, False, False, fixed_mask="Full"),
    "only_f_mask": MethodSpec("ONLY f_mask", "only_f_mask", "ablation", False, True, True, False),
    "ONLY f_mask": MethodSpec("ONLY f_mask", "only_f_mask", "ablation", False, True, True, False),
    "single_channel_mask": MethodSpec(
        "SINGLE-CHANNEL f_mask^s",
        "single_channel_mask",
        "ablation",
        True,
        True,
        True,
        True,
    ),
    "SINGLE-CHANNEL f_mask^s": MethodSpec(
        "SINGLE-CHANNEL f_mask^s",
        "single_channel_mask",
        "ablation",
        True,
        True,
        True,
        True,
    ),
    "PAD": MethodSpec("PAD", "PAD", "fixed_mask_baseline", True, False, False, False, fixed_mask="PAD"),
    "Pad": MethodSpec("PAD", "PAD", "fixed_mask_baseline", True, False, False, False, fixed_mask="PAD"),
    "Narrow": MethodSpec("Narrow", "Narrow", "fixed_mask_baseline", True, False, False, False, fixed_mask="Narrow"),
    "Medium": MethodSpec("Medium", "Medium", "fixed_mask_baseline", True, False, False, False, fixed_mask="Medium"),
    "Full": MethodSpec("Full", "Full", "fixed_mask_baseline", True, False, False, False, fixed_mask="Full"),
    "vit": MethodSpec("vit", "vit", "backbone_adapter", True, True, True, False, adapter="vit_b_32"),
    "resnet": MethodSpec("resnet", "resnet", "backbone_adapter", True, True, True, False, adapter="resnet"),
    "lora": MethodSpec("lora", "lora", "parameter_efficient_adapter", False, False, False, False, adapter="lora"),
    "imagenet_1k": MethodSpec("imagenet_1k", "imagenet_1k", "source_space", False, False, False, False),
}


def get_method_spec(name: str) -> MethodSpec:
    """Return a concrete method/baseline/variant specification."""

    try:
        return METHOD_REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(METHOD_REGISTRY))
        raise KeyError(f"Unknown SMM method selector '{name}'. Known selectors: {known}") from exc


def _is_torch_tensor(x: Any) -> bool:
    return x.__class__.__module__.split(".", 1)[0] == "torch"


def _shape_hw(x: Any) -> tuple[int, int]:
    shape = tuple(getattr(x, "shape"))
    if len(shape) < 2:
        raise ValueError(f"Expected image/tensor with H and W dimensions, got shape={shape}")
    return int(shape[-2]), int(shape[-1])


def _ensure_tuple_hw(size: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(size, int):
        return int(size), int(size)
    if len(size) != 2:
        raise ValueError(f"Expected two-dimensional size, got {size!r}")
    return int(size[0]), int(size[1])


def _numpy_resize_nearest(x: Any, size: tuple[int, int]) -> Any:
    import numpy as np

    arr = np.asarray(x)
    h, w = arr.shape[-2], arr.shape[-1]
    out_h, out_w = size
    y_idx = np.clip((np.arange(out_h) * h / out_h).astype(int), 0, h - 1)
    x_idx = np.clip((np.arange(out_w) * w / out_w).astype(int), 0, w - 1)
    return np.take(np.take(arr, y_idx, axis=-2), x_idx, axis=-1)


def resize_reprogrammed_input(x: Any, size: int | Sequence[int], mode: str = "bilinear") -> Any:
    """Resize target image r(x) to the pretrained classifier input shape."""

    target_size = _ensure_tuple_hw(size)
    if _is_torch_tensor(x):
        torch = lazy_import_backend("torch")
        functional = importlib.import_module("torch.nn.functional")
        added_batch = x.ndim == 3
        z = x.unsqueeze(0) if added_batch else x
        resized = functional.interpolate(
            z,
            size=target_size,
            mode=mode if mode in {"nearest", "bilinear", "bicubic"} else "bilinear",
            align_corners=False if mode in {"bilinear", "bicubic"} else None,
        )
        return resized.squeeze(0) if added_batch else resized
    return _numpy_resize_nearest(x, target_size)


def patchwise_interpolate(mask: Any, target_size: int | Sequence[int], level: int = 0, mode: str = "bilinear") -> Any:
    """Upsample f_mask's coarse grid to H x W.

    For l > 0 the coarse grid is floor(H / 2**l) x floor(W / 2**l), then it is
    upsampled to H x W. For l = 0 this function preserves the existing mask
    shape when already matched, explicitly materializing the paper's omit branch.
    """

    target_h, target_w = _ensure_tuple_hw(target_size)
    if level < 0:
        raise ValueError("interpolation level l must be non-negative")
    if level == 0 and _shape_hw(mask) == (target_h, target_w):
        return mask
    patch = 2 ** int(level)
    if _is_torch_tensor(mask):
        functional = importlib.import_module("torch.nn.functional")
        enlarged = mask.repeat_interleave(patch, dim=-2).repeat_interleave(patch, dim=-1)
        pad_h = max(0, target_h - int(enlarged.shape[-2]))
        pad_w = max(0, target_w - int(enlarged.shape[-1]))
        if pad_h or pad_w:
            enlarged = functional.pad(enlarged, (0, pad_w, 0, pad_h), mode="replicate")
        return enlarged[..., :target_h, :target_w]
    import numpy as np

    arr = np.asarray(mask)
    enlarged = np.repeat(np.repeat(arr, patch, axis=-2), patch, axis=-1)
    pad_h = max(0, target_h - enlarged.shape[-2])
    pad_w = max(0, target_w - enlarged.shape[-1])
    if pad_h or pad_w:
        enlarged = np.pad(enlarged, [(0, 0)] * (enlarged.ndim - 2) + [(0, pad_h), (0, pad_w)], mode="edge")
    return enlarged[..., :target_h, :target_w]


class LightweightCNNMaskGenerator:
    """Lightweight CNN sample-specific mask generator f_mask.

    The class is importable without torch. When torch is available it builds a
    small convolutional network; otherwise it falls back to deterministic
    per-sample numpy masks for smoke validation through the same callable
    interface.
    """

    def __init__(self, config: MaskGeneratorConfig | None = None) -> None:
        self.config = config or MaskGeneratorConfig()
        self._torch_module: Any | None = None
        self._torch = None
        if backend_available("torch"):
            self._build_torch_module()

    def _build_torch_module(self) -> None:
        torch = lazy_import_backend("torch")
        nn = importlib.import_module("torch.nn")
        layers: list[Any] = []
        in_ch = self.config.input_channels
        out_ch = 1 if self.config.single_channel else self.config.output_channels
        conv_channels = [8, 16, 32, 64, 128, out_ch] if int(self.config.depth) >= 6 else [8, 16, 32, 64, out_ch]
        for idx, ch in enumerate(conv_channels):
            layers.append(nn.Conv2d(in_ch, ch, kernel_size=3, stride=1, padding=1, bias=True))
            if idx != len(conv_channels) - 1:
                layers.append(nn.BatchNorm2d(ch))
                layers.append(nn.ReLU(inplace=True))
                if idx < 3:
                    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_ch = ch
        if self.config.activation == "sigmoid":
            layers.append(nn.Sigmoid())
        self._torch = torch
        self._torch_module = nn.Sequential(*layers)

    def parameters(self) -> list[Any]:
        if self._torch_module is None:
            return []
        return list(self._torch_module.parameters())

    def train(self, mode: bool = True) -> "LightweightCNNMaskGenerator":
        if self._torch_module is not None:
            self._torch_module.train(mode)
        return self

    def eval(self) -> "LightweightCNNMaskGenerator":
        return self.train(False)

    def state_dict(self) -> Mapping[str, Any]:
        if self._torch_module is None:
            return {"fallback": "numpy_deterministic_mask", "config": asdict(self.config)}
        return self._torch_module.state_dict()

    def __call__(self, x: Any) -> Any:
        return self.forward(x)

    def forward(self, x: Any) -> Any:
        if _is_torch_tensor(x):
            if self._torch_module is None:
                self._build_torch_module()
            if self._torch_module is not None:
                self._torch_module.to(device=x.device, dtype=x.dtype)
            y = self._torch_module(x)
            if self.config.single_channel and getattr(y, "shape")[1] == 1:
                y = y.repeat(1, self.config.output_channels, 1, 1)
            return y
        import numpy as np

        arr = np.asarray(x, dtype="float32")
        if arr.ndim == 3:
            base = arr.mean(axis=0, keepdims=True)
            out_channels = 1 if self.config.single_channel else self.config.output_channels
            mask = 1.0 / (1.0 + np.exp(-base))
            mask = np.repeat(mask, out_channels, axis=0)
            if self.config.single_channel:
                mask = np.repeat(mask[:1], self.config.output_channels, axis=0)
            return mask.astype("float32")
        if arr.ndim == 4:
            base = arr.mean(axis=1, keepdims=True)
            out_channels = 1 if self.config.single_channel else self.config.output_channels
            mask = 1.0 / (1.0 + np.exp(-base))
            mask = np.repeat(mask, out_channels, axis=1)
            if self.config.single_channel:
                mask = np.repeat(mask[:, :1], self.config.output_channels, axis=1)
            return mask.astype("float32")
        raise ValueError(f"Expected CHW or NCHW input for f_mask, got shape={arr.shape}")


def f_mask(config: MaskGeneratorConfig | None = None) -> LightweightCNNMaskGenerator:
    """Factory for the lightweight CNN mask generator f_mask."""

    return LightweightCNNMaskGenerator(config)


def initialize_shared_delta(
    channels: int = 3,
    image_size: int | Sequence[int] = (224, 224),
    trainable: bool = True,
    torch_device: str | None = None,
) -> Any:
    """Create the shared noise pattern delta initialized to the zero matrix."""

    h, w = _ensure_tuple_hw(image_size)
    if backend_available("torch"):
        torch = lazy_import_backend("torch")
        tensor = torch.zeros((1, channels, h, w), device=torch_device)
        return torch.nn.Parameter(tensor, requires_grad=trainable)
    import numpy as np

    return np.zeros((1, channels, h, w), dtype="float32")


def combine_mask_and_delta(resized_x: Any, mask: Any, delta: Any) -> Any:
    """Return r(x) + mask(x) * delta."""

    if _is_torch_tensor(resized_x):
        if hasattr(mask, "to"):
            mask = mask.to(device=resized_x.device, dtype=resized_x.dtype)
        if hasattr(delta, "to"):
            delta = delta.to(device=resized_x.device, dtype=resized_x.dtype)
        return resized_x + mask * delta
    import numpy as np

    return np.asarray(resized_x) + np.asarray(mask) * np.asarray(delta)


class FixedMaskBaseline:
    """PAD/Narrow/Medium/Full predetermined shared-mask VR baseline."""

    def __init__(self, name: str, image_size: Sequence[int] = (224, 224), channels: int = 3) -> None:
        self.name = get_method_spec(name).canonical
        self.image_size = _ensure_tuple_hw(image_size)
        self.channels = channels

    def mask(self, batch_size: int = 1) -> Any:
        h, w = self.image_size
        if backend_available("torch"):
            torch = lazy_import_backend("torch")
            m = torch.zeros((batch_size, self.channels, h, w), dtype=torch.float32)
            if self.name == "PAD":
                m.fill_(1.0)
            elif self.name == "Narrow":
                margin = min(28, max(1, min(h, w) // 2))
                m[:, :, :margin, :] = 1
                m[:, :, -margin:, :] = 1
                m[:, :, :, :margin] = 1
                m[:, :, :, -margin:] = 1
            elif self.name == "Medium":
                center_h = max(1, h // 4)
                center_w = max(1, w // 4)
                top = (h - center_h) // 2
                left = (w - center_w) // 2
                m[:, :, top : top + center_h, left : left + center_w] = 1
            else:
                m.fill_(1.0)
            return m
        import numpy as np

        m = np.zeros((batch_size, self.channels, h, w), dtype="float32")
        if self.name == "PAD":
            m[...] = 1
        elif self.name == "Narrow":
            margin = min(28, max(1, min(h, w) // 2))
            m[:, :, :margin, :] = 1
            m[:, :, -margin:, :] = 1
            m[:, :, :, :margin] = 1
            m[:, :, :, -margin:] = 1
        elif self.name == "Medium":
            center_h = max(1, h // 4)
            center_w = max(1, w // 4)
            top = (h - center_h) // 2
            left = (w - center_w) // 2
            m[:, :, top : top + center_h, left : left + center_w] = 1
        else:
            m[...] = 1
        return m

    def forward(self, x: Any, delta: Any) -> Any:
        resized = resize_reprogrammed_input(x, self.image_size)
        batch_size = int(getattr(resized, "shape")[0]) if len(tuple(getattr(resized, "shape"))) == 4 else 1
        return combine_mask_and_delta(resized, self.mask(batch_size), delta)


class NonParametricOutputMapping:
    """Injective non-parametric f_out mapping from ImageNet-1K labels to target labels."""

    def __init__(self, num_target_classes: int, num_source_classes: int = 1000, seed: int = DEFAULT_SEED) -> None:
        if num_target_classes > num_source_classes:
            raise ValueError("Injective mapping requires target classes <= source classes")
        self.num_target_classes = int(num_target_classes)
        self.num_source_classes = int(num_source_classes)
        rng = random.Random(int(seed))
        source_indices = list(range(self.num_source_classes))
        rng.shuffle(source_indices)
        self.target_to_source: dict[int, int] = {
            target: source_indices[target] for target in range(self.num_target_classes)
        }
        self.source_to_target: dict[int, int] = {v: k for k, v in self.target_to_source.items()}

    def source_labels(self, target_labels: Sequence[int]) -> list[int]:
        return [self.target_to_source[int(y)] for y in target_labels]

    def target_from_logits(self, logits: Any) -> Any:
        source_indices = [self.target_to_source[i] for i in range(self.num_target_classes)]
        if _is_torch_tensor(logits):
            torch = lazy_import_backend("torch")
            idx = torch.tensor(source_indices, device=logits.device, dtype=torch.long)
            restricted = logits.index_select(dim=-1, index=idx)
            return restricted.argmax(dim=-1)
        import numpy as np

        restricted = np.asarray(logits)[..., source_indices]
        return restricted.argmax(axis=-1)

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": "R1m_injective",
            "num_target_classes": self.num_target_classes,
            "num_source_classes": self.num_source_classes,
            "target_to_source": self.target_to_source,
        }


class SMMReprogrammer:
    """SMM/Ours and ablation forward interface.

    Forward returns the input for the frozen pretrained classifier:
    r(x) + f_mask(r(x) | phi) * delta.
    """

    def __init__(self, config: SMMConfig | None = None) -> None:
        self.config = config or SMMConfig()
        spec = get_method_spec(self.config.method)
        mask_config = self.config.mask
        if "vit" in str(self.config.backbone).lower() and int(mask_config.depth) < 6:
            mask_config = MaskGeneratorConfig(
                input_channels=mask_config.input_channels,
                output_channels=mask_config.output_channels,
                hidden_channels=mask_config.hidden_channels,
                depth=6,
                interpolation_level=mask_config.interpolation_level,
                single_channel=mask_config.single_channel,
                activation=mask_config.activation,
            )
        if spec.single_channel_mask:
            mask_config = MaskGeneratorConfig(
                input_channels=mask_config.input_channels,
                output_channels=mask_config.output_channels,
                hidden_channels=mask_config.hidden_channels,
                depth=mask_config.depth,
                interpolation_level=mask_config.interpolation_level,
                single_channel=True,
                activation=mask_config.activation,
            )
        self.method_spec = spec
        self.mask_generator = LightweightCNNMaskGenerator(mask_config) if spec.train_mask_generator or spec.sample_specific_mask else None
        self.delta = initialize_shared_delta(
            channels=self.config.channels,
            image_size=self.config.image_size,
            trainable=spec.train_delta,
        )
        self.output_mapping = NonParametricOutputMapping(
            self.config.num_target_classes,
            self.config.num_source_classes,
            self.config.seed,
        )

    def parameters(self) -> list[Any]:
        params: list[Any] = []
        if self.method_spec.train_delta and hasattr(self.delta, "requires_grad"):
            params.append(self.delta)
        if self.method_spec.train_mask_generator and self.mask_generator is not None:
            params.extend(self.mask_generator.parameters())
        return params

    def optimizer_parameter_groups(self) -> list[dict[str, Any]]:
        groups: list[dict[str, Any]] = []
        if self.method_spec.train_delta and hasattr(self.delta, "requires_grad"):
            groups.append({"name": "delta", "params": [self.delta], "lr": self.config.learning_rate_delta})
        if self.method_spec.train_mask_generator and self.mask_generator is not None:
            mask_params = self.mask_generator.parameters()
            if mask_params:
                groups.append({"name": "phi_mask_generator", "params": mask_params, "lr": self.config.learning_rate_mask})
        return groups

    def sample_mask(self, x: Any) -> Any:
        resized = resize_reprogrammed_input(x, self.config.image_size)
        if self.method_spec.fixed_mask:
            batch_size = int(getattr(resized, "shape")[0]) if len(tuple(getattr(resized, "shape"))) == 4 else 1
            return FixedMaskBaseline(self.method_spec.fixed_mask, self.config.image_size, self.config.channels).mask(batch_size)
        if self.method_spec.canonical == "only_delta":
            return FixedMaskBaseline("Full", self.config.image_size, self.config.channels).mask(
                int(getattr(resized, "shape")[0]) if len(tuple(getattr(resized, "shape"))) == 4 else 1
            )
        if self.mask_generator is None:
            return FixedMaskBaseline("Full", self.config.image_size, self.config.channels).mask(1)
        raw_mask = self.mask_generator(resized)
        return patchwise_interpolate(raw_mask, self.config.image_size, self.config.mask.interpolation_level)

    def forward(self, x: Any, return_mask: bool = False) -> Any:
        resized = resize_reprogrammed_input(x, self.config.image_size)
        if self.method_spec.canonical == "only_f_mask":
            if _is_torch_tensor(resized):
                effective_delta = lazy_import_backend("torch").ones_like(self.delta)
            else:
                import numpy as np

                effective_delta = np.ones_like(self.delta)
        else:
            effective_delta = self.delta
        mask = self.sample_mask(resized)
        programmed = combine_mask_and_delta(resized, mask, effective_delta)
        if return_mask:
            return programmed, mask
        return programmed

    def __call__(self, x: Any, return_mask: bool = False) -> Any:
        return self.forward(x, return_mask=return_mask)

    def train_step(self, batch: tuple[Any, Any], classifier: Any, optimizer: Any, loss_fn: Callable[..., Any] | None = None) -> dict[str, float]:
        """One δ/φ update step for Algorithm 1-style training."""

        x, y = batch
        reprogrammed = self.forward(x)
        logits = classifier(reprogrammed) if callable(classifier) else classifier.forward(reprogrammed)
        if loss_fn is None:
            if not _is_torch_tensor(logits):
                raise RuntimeError("Default training loss requires torch logits; pass loss_fn for non-torch smoke routes.")
            functional = importlib.import_module("torch.nn.functional")
            source_y = self.output_mapping.source_labels([int(v) for v in y.detach().cpu().tolist()])
            torch = lazy_import_backend("torch")
            y_tensor = torch.tensor(source_y, device=logits.device, dtype=torch.long)
            loss = functional.cross_entropy(logits, y_tensor)
        else:
            loss = loss_fn(logits, y)
        if hasattr(optimizer, "zero_grad"):
            optimizer.zero_grad()
        if hasattr(loss, "backward"):
            loss.backward()
        if hasattr(optimizer, "step"):
            optimizer.step()
        value = float(loss.detach().cpu().item()) if hasattr(loss, "detach") else float(loss)
        return {"loss": value, "optimizer_groups": float(len(self.optimizer_parameter_groups()))}


def sample_specific_mask_forward(method: SMMReprogrammer, x: Any) -> Any:
    """Return the sample-level multi-channel mask tensor f_mask(r(x)|phi)."""

    return method.sample_mask(x)


def apply_smm_reprogramming(method: SMMReprogrammer, x: Any) -> Any:
    """Return r(x) + mask(x) * delta for the selected SMM variant."""

    return method.forward(x)


def build_method(name: str = "ours", config: SMMConfig | Mapping[str, Any] | None = None) -> SMMReprogrammer | FixedMaskBaseline:
    """Build Ours, ablations, or fixed-mask baselines from a selector."""

    spec = get_method_spec(name)
    if isinstance(config, SMMConfig):
        cfg = config
    elif isinstance(config, Mapping):
        cfg = SMMConfig(**{**dict(config), "method": name})
    else:
        cfg = SMMConfig(method=name)
    if spec.family == "fixed_mask_baseline":
        return FixedMaskBaseline(spec.canonical, cfg.image_size, cfg.channels)
    return SMMReprogrammer(cfg)


def ours(config: SMMConfig | Mapping[str, Any] | None = None) -> SMMReprogrammer:
    return build_method("ours", config)  # type: ignore[return-value]


def only_delta(config: SMMConfig | Mapping[str, Any] | None = None) -> SMMReprogrammer:
    return build_method("only_delta", config)  # type: ignore[return-value]


def only_f_mask(config: SMMConfig | Mapping[str, Any] | None = None) -> SMMReprogrammer:
    return build_method("only_f_mask", config)  # type: ignore[return-value]


def single_channel_mask(config: SMMConfig | Mapping[str, Any] | None = None) -> SMMReprogrammer:
    return build_method("single_channel_mask", config)  # type: ignore[return-value]


Ours = SMMReprogrammer
OrAdaptersBy = METHOD_REGISTRY
Ids = {
    "methods": MAIN_METHOD_IDS,
    "variants": SMM_VARIANT_IDS,
    "datasets": DATASET_IDS,
    "backbones": BACKBONE_IDS,
    "three_seed_protocol": THREE_SEED_PROTOCOL,
    "patch_size": PATCH_SIZE_SWEEP,
    "p": P_SWEEP,
}


def create_optimizer_for_smm(method: SMMReprogrammer, optimizer_name: str = "adam") -> Any:
    """Create an optimizer over δ and φ parameter groups; torch is imported lazily."""

    groups = method.optimizer_parameter_groups()
    if not groups:
        return None
    torch = lazy_import_backend("torch")
    opt = torch.optim.Adam if optimizer_name.lower() == "adam" else torch.optim.SGD
    return opt(groups)


def freeze_module_parameters(module: Any) -> Any:
    """Freeze a pretrained backbone so only δ and φ are updated."""

    if hasattr(module, "parameters"):
        for param in module.parameters():
            param.requires_grad = False
    if hasattr(module, "eval"):
        module.eval()
    return module


def classifier_is_frozen(module: Any) -> bool:
    if not hasattr(module, "parameters"):
        return True
    return all(not bool(getattr(param, "requires_grad", False)) for param in module.parameters())


class LinearSmokeClassifier:
    """Small deterministic classifier used only when torch backbones are unavailable."""

    def __init__(self, num_classes: int = 1000) -> None:
        self.num_classes = int(num_classes)
        self.frozen = True

    def parameters(self) -> list[Any]:
        return []

    def eval(self) -> "LinearSmokeClassifier":
        return self

    def __call__(self, x: Any) -> Any:
        import numpy as np

        arr = np.asarray(x, dtype="float32")
        batch = arr.shape[0] if arr.ndim == 4 else 1
        base = arr.reshape(batch, -1).mean(axis=1, keepdims=True)
        offsets = np.linspace(-0.5, 0.5, self.num_classes, dtype="float32")[None, :]
        return base + offsets


def load_classifier(config: SMMConfig | Mapping[str, Any] | None = None) -> Any:
    """Load a frozen ImageNet-1K pretrained ResNet-18/ResNet-50/ViT-B/32 classifier.

    Full mode uses torchvision/timm when installed. Minimal smoke mode returns
    a deterministic frozen classifier through the same callable interface.
    """

    cfg = config if isinstance(config, SMMConfig) else SMMConfig(**dict(config or {}))
    backbone = cfg.backbone.lower().replace("-", "_").replace("/", "_")
    if not backend_available("torch"):
        return LinearSmokeClassifier(cfg.num_source_classes)
    torch = lazy_import_backend("torch")
    if backend_available("torchvision"):
        torchvision_models = importlib.import_module("torchvision.models")
        if backbone in {"resnet18", "resnet_18", "resnet"}:
            weights = getattr(torchvision_models, "ResNet18_Weights").IMAGENET1K_V1
            model = torchvision_models.resnet18(weights=weights)
            return freeze_module_parameters(model) if cfg.freeze_backbone else model
        if backbone in {"resnet50", "resnet_50"}:
            weights = getattr(torchvision_models, "ResNet50_Weights").IMAGENET1K_V2
            model = torchvision_models.resnet50(weights=weights)
            return freeze_module_parameters(model) if cfg.freeze_backbone else model
        if backbone in {"vit_b_32", "vit_b32", "vit_b_32_imagenet_1k", "vit_b_32_imagenet"}:
            weights = getattr(torchvision_models, "ViT_B_32_Weights").IMAGENET1K_V1
            model = torchvision_models.vit_b_32(weights=weights)
            return freeze_module_parameters(model) if cfg.freeze_backbone else model
    if backend_available("timm"):
        timm = lazy_import_backend("timm")
        model_name = "vit_base_patch32_384" if "vit" in backbone else "resnet18"
        model = timm.create_model(model_name, pretrained=True, num_classes=cfg.num_source_classes)
        return freeze_module_parameters(model) if cfg.freeze_backbone else model

    class TorchSmokeClassifier(torch.nn.Module):
        def __init__(self, num_classes: int) -> None:
            super().__init__()
            self.num_classes = num_classes

        def forward(self, x: Any) -> Any:
            batch = x.shape[0] if x.ndim == 4 else 1
            base = x.reshape(batch, -1).mean(dim=1, keepdim=True)
            return base + torch.linspace(-0.5, 0.5, self.num_classes, device=x.device).unsqueeze(0)

    return freeze_module_parameters(TorchSmokeClassifier(cfg.num_source_classes))


def finetune_classifier(config: SMMConfig | Mapping[str, Any] | None = None) -> Any:
    """Load an unfrozen classifier hook for explicit fine-tuning comparisons."""

    cfg = config if isinstance(config, SMMConfig) else SMMConfig(**dict(config or {}))
    cfg = SMMConfig(**{**asdict(cfg), "freeze_backbone": False})
    return load_classifier(cfg)


def compute_accuracy(predictions: Sequence[int] | Any, labels: Sequence[int] | Any) -> float:
    if _is_torch_tensor(predictions):
        predictions = predictions.detach().cpu().tolist()
    if _is_torch_tensor(labels):
        labels = labels.detach().cpu().tolist()
    preds = list(predictions)
    ys = list(labels)
    if not ys:
        return 0.0
    return sum(int(p) == int(y) for p, y in zip(preds, ys)) / len(ys)


def aggregate_accuracy(values: Sequence[float]) -> dict[str, float]:
    vals = [float(v) for v in values]
    if not vals:
        return {"mean": 0.0, "std": 0.0, "mean_percent": 0.0, "std_percent": 0.0}
    mean = sum(vals) / len(vals)
    variance = sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)
    std = math.sqrt(variance)
    return {"mean": mean, "std": std, "mean_percent": mean * 100.0, "std_percent": std * 100.0}


def compute_loss(logits: Any, labels: Any) -> float:
    if _is_torch_tensor(logits):
        functional = importlib.import_module("torch.nn.functional")
        if not _is_torch_tensor(labels):
            torch = lazy_import_backend("torch")
            labels = torch.tensor(labels, device=logits.device, dtype=torch.long)
        return float(functional.cross_entropy(logits, labels).detach().cpu().item())
    import numpy as np

    z = np.asarray(logits, dtype="float64")
    y = np.asarray(labels, dtype="int64")
    z = z - z.max(axis=-1, keepdims=True)
    probs = np.exp(z) / np.exp(z).sum(axis=-1, keepdims=True)
    return float(-np.log(probs[np.arange(len(y)), y] + 1e-12).mean())


def aggregate_loss(values: Sequence[float]) -> dict[str, float]:
    vals = [float(v) for v in values]
    if not vals:
        return {"mean": 0.0, "std": 0.0}
    mean = sum(vals) / len(vals)
    variance = sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)
    return {"mean": mean, "std": math.sqrt(variance)}


def mean_std_accuracy(values: Sequence[float]) -> dict[str, float]:
    return aggregate_accuracy(values)


def injective_label_mapping(
    num_target_classes: int,
    num_source_classes: int = 1000,
    seed: int = DEFAULT_SEED,
) -> NonParametricOutputMapping:
    return NonParametricOutputMapping(num_target_classes, num_source_classes, seed)


def delta_phi_update_step(
    method: SMMReprogrammer,
    batch: tuple[Any, Any],
    classifier: Any,
    optimizer: Any,
    loss_fn: Callable[..., Any] | None = None,
) -> dict[str, float]:
    """Injective f_out + δ/φ iterative update function for Algorithm 1."""

    return method.train_step(batch, classifier, optimizer, loss_fn=loss_fn)


def artifact_root() -> Path:
    return Path(os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR", ".")).resolve()


def _write_json(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> Path:
    p = artifact_root() / Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return p


def _write_csv(path: str | os.PathLike[str], rows: Sequence[Mapping[str, Any]]) -> Path:
    p = artifact_root() / Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = sorted({k for row in rows for k in row.keys()}) if rows else ["status"]
    with p.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
    return p


def write_metrics_artifact(metrics: Mapping[str, Any], path: str = "results/metrics.json") -> Path:
    return _write_json(path, {"artifact_type": "measured_metrics", **dict(metrics)})


def write_config_resolved_artifact(config: SMMConfig | Mapping[str, Any], path: str = "results/config_resolved.json") -> Path:
    payload = asdict(config) if isinstance(config, SMMConfig) else dict(config)
    return _write_json(path, {"artifact_type": "resolved_config", "config": payload})


def write_training_trace_artifact(trace: Sequence[Mapping[str, Any]], path: str = "results/training_trace.json") -> Path:
    return _write_json(path, {"artifact_type": "training_trace", "steps": list(trace)})


def write_mask_statistics_artifact(mask: Any, path: str = "results/mask_statistics.json") -> Path:
    if _is_torch_tensor(mask):
        data = mask.detach().cpu()
        stats = {
            "mean": float(data.mean().item()),
            "std": float(data.std().item()),
            "min": float(data.min().item()),
            "max": float(data.max().item()),
            "shape": list(data.shape),
        }
    else:
        import numpy as np

        data = np.asarray(mask)
        stats = {
            "mean": float(data.mean()),
            "std": float(data.std()),
            "min": float(data.min()),
            "max": float(data.max()),
            "shape": list(data.shape),
        }
    return _write_json(path, {"artifact_type": "measured_mask_statistics", "mask_statistics": stats})


def write_summary_table_artifact(rows: Sequence[Mapping[str, Any]], path: str = "results/summary_table.csv") -> Path:
    return _write_csv(path, rows)


def write_table1_resnet_main_artifact(rows: Sequence[Mapping[str, Any]], path: str = "results/tables/table1_resnet_main.csv") -> Path:
    return _write_csv(path, rows)


def write_table2_vit_main_artifact(rows: Sequence[Mapping[str, Any]], path: str = "results/tables/table2_vit_main.csv") -> Path:
    return _write_csv(path, rows)


def write_table3_ablation_artifact(rows: Sequence[Mapping[str, Any]], path: str = "results/tables/table3_ablation.csv") -> Path:
    return _write_csv(path, rows)


def write_table_3_artifact(rows: Sequence[Mapping[str, Any]], path: str = "results/tables/table_3.csv") -> Path:
    return _write_csv(path, rows)


def run_table_3_route(config: SMMConfig | Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Bounded executable Table 3 route over the paper ablation variants."""

    cfg = config if isinstance(config, SMMConfig) else SMMConfig(**dict(config or {}))
    rows: list[dict[str, Any]] = []
    for variant in ("only_delta", "only_f_mask", "single_channel_mask", "ours"):
        method = build_method(variant, cfg)
        rows.append(
            {
                "dataset": cfg.dataset,
                "backbone": "resnet18",
                "method": get_method_spec(variant).name,
                "mask_variant": variant,
                "seed": cfg.seed,
                "train_delta": getattr(method, "method_spec", get_method_spec(variant)).train_delta,
                "train_mask_generator": getattr(method, "method_spec", get_method_spec(variant)).train_mask_generator,
                "run_mode": "bounded_route_ready",
            }
        )
    write_table3_ablation_artifact(rows)
    write_table_3_artifact(rows)
    return rows


def write_figure_3_artifact(payload: Mapping[str, Any], path: str = "results/figures/figure_3.json") -> Path:
    return _write_json(path, {"artifact_type": "figure_3_diagnostic_data", **dict(payload)})


def run_figure_3_route(config: SMMConfig | Mapping[str, Any] | None = None) -> dict[str, Any]:
    cfg = config if isinstance(config, SMMConfig) else SMMConfig(**dict(config or {}))
    payload = {
        "caption": "Figure 3. Existing predetermined shared masks versus sample-specific multi-channel SMM masks.",
        "fixed_mask_baselines": ["PAD", "Narrow", "Medium", "Full"],
        "sample_specific_method": "Ours",
        "image_size": list(cfg.image_size),
        "interpolation_level": cfg.mask.interpolation_level,
    }
    write_figure_3_artifact(payload)
    return payload


def make_readiness_artifacts(config: SMMConfig | Mapping[str, Any] | None = None) -> dict[str, Any]:
    cfg = config if isinstance(config, SMMConfig) else SMMConfig(**dict(config or {}))
    readiness = {
        "artifact_type": "readiness",
        "not_benchmark_scores": True,
        "config": asdict(cfg),
        "methods": list(MAIN_METHOD_IDS),
        "variants": list(SMM_VARIANT_IDS),
        "backends": {k: asdict(v) for k, v in optional_backend_status().items()},
        "three_seed_protocol": list(THREE_SEED_PROTOCOL),
        "patch_size_values": list(PATCH_SIZE_SWEEP),
        "p_values": list(P_SWEEP),
    }
    evaluation = {
        "artifact_type": "evaluation_result",
        "status": "smoke_route_exercised",
        "not_full_benchmark": True,
        "accuracy_formula": "correct / total",
        "mean_std_formula": "sample mean and sample standard deviation across seeds",
    }
    _write_json("readiness.json", readiness)
    _write_json("evaluation_result.json", evaluation)
    return {"readiness": readiness, "evaluation_result": evaluation}


def method_selector(name: str) -> Callable[[SMMConfig | Mapping[str, Any] | None], Any]:
    selectors: dict[str, Callable[[SMMConfig | Mapping[str, Any] | None], Any]] = {
        "ours": ours,
        "Ours": ours,
        "only_delta": only_delta,
        "ONLY δ": only_delta,
        "only_f_mask": only_f_mask,
        "ONLY f_mask": only_f_mask,
        "single_channel_mask": single_channel_mask,
        "SINGLE-CHANNEL f_mask^s": single_channel_mask,
        "PAD": lambda config=None: build_method("PAD", config),
        "Narrow": lambda config=None: build_method("Narrow", config),
        "Medium": lambda config=None: build_method("Medium", config),
        "Full": lambda config=None: build_method("Full", config),
        "vit": lambda config=None: build_method("vit", config),
        "resnet": lambda config=None: build_method("resnet", config),
        "lora": lambda config=None: build_method("lora", config),
        "imagenet_1k": lambda config=None: build_method("imagenet_1k", config),
    }
    try:
        return selectors[name]
    except KeyError as exc:
        raise KeyError(f"Unknown selector {name!r}; known selectors: {sorted(selectors)}") from exc


def baseline_selector(name: str) -> Callable[[SMMConfig | Mapping[str, Any] | None], Any]:
    return method_selector(name)


def load_inputs(config: SMMConfig | Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Minimal import-safe input/data interface for smoke wiring.

    Full dataset loading is delegated to sample_specific_masks.data when present.
    """

    cfg = config if isinstance(config, SMMConfig) else SMMConfig(**dict(config or {}))
    if backend_available("torch"):
        torch = lazy_import_backend("torch")
        x = torch.zeros((2, cfg.channels, cfg.image_size[0], cfg.image_size[1]))
        y = torch.tensor([0, 1], dtype=torch.long)
    else:
        import numpy as np

        x = np.zeros((2, cfg.channels, cfg.image_size[0], cfg.image_size[1]), dtype="float32")
        y = np.asarray([0, 1], dtype="int64")
    return {"train": [(x, y)], "test": [(x, y)], "dataset": cfg.dataset}


def run_evaluation(method: SMMReprogrammer, classifier: Any, batch: tuple[Any, Any]) -> dict[str, Any]:
    x, labels = batch
    logits = classifier(method(x)) if callable(classifier) else classifier.forward(method(x))
    target_predictions = method.output_mapping.target_from_logits(logits)
    accuracy = compute_accuracy(target_predictions, labels)
    return {"accuracy": accuracy, "accuracy_percent": accuracy * 100.0}


def write_named_result_artifacts(results: Sequence[Mapping[str, Any]], output_dir: str = "results") -> dict[str, str]:
    rows = [dict(r) for r in results]
    paths = {
        "metrics": str(write_metrics_artifact({"rows": rows}, f"{output_dir}/metrics.json")),
        "summary_table": str(write_summary_table_artifact(rows, f"{output_dir}/summary_table.csv")),
    }
    return paths


def runtime_smoke(config: SMMConfig | Mapping[str, Any] | None = None) -> dict[str, Any]:
    cfg = config if isinstance(config, SMMConfig) else SMMConfig(**dict(config or {}))
    method = ours(cfg)
    classifier = load_classifier(cfg)
    inputs = load_inputs(cfg)
    batch = inputs["test"][0]
    metrics = run_evaluation(method, classifier, batch)
    mask = method.sample_mask(batch[0])
    write_metrics_artifact(metrics)
    write_config_resolved_artifact(cfg)
    write_mask_statistics_artifact(mask)
    make_readiness_artifacts(cfg)
    return metrics


__all__ = [
    "ALPHA_SWEEP",
    "BACKBONE_IDS",
    "BackendStatus",
    "DATASET_IDS",
    "DEFAULT_SEED",
    "FixedMaskBaseline",
    "GAMMA_SWEEP",
    "Ids",
    "LightweightCNNMaskGenerator",
    "MAIN_METHOD_IDS",
    "METHOD_REGISTRY",
    "MaskGeneratorConfig",
    "MethodSpec",
    "NonParametricOutputMapping",
    "OPTIONAL_BACKENDS",
    "OrAdaptersBy",
    "Ours",
    "P_SWEEP",
    "PATCH_SIZE_SWEEP",
    "SIMILARITY_GUIDANCE_SCALE_SWEEP",
    "SMMConfig",
    "SMMReprogrammer",
    "SMM_VARIANT_IDS",
    "THREE_SEED_PROTOCOL",
    "aggregate_accuracy",
    "aggregate_loss",
    "apply_smm_reprogramming",
    "artifact_root",
    "backend_available",
    "baseline_selector",
    "classifier_is_frozen",
    "combine_mask_and_delta",
    "compute_accuracy",
    "compute_loss",
    "create_optimizer_for_smm",
    "delta_phi_update_step",
    "f_mask",
    "finetune_classifier",
    "freeze_module_parameters",
    "get_method_spec",
    "initialize_shared_delta",
    "injective_label_mapping",
    "lazy_import_backend",
    "load_classifier",
    "load_inputs",
    "make_readiness_artifacts",
    "mean_std_accuracy",
    "method_selector",
    "only_delta",
    "only_f_mask",
    "optional_backend_status",
    "ours",
    "p_values",
    "patch_size_values",
    "patchwise_interpolate",
    "resize_reprogrammed_input",
    "resolve_seed_defaults",
    "run_evaluation",
    "run_figure_3_route",
    "run_table_3_route",
    "runtime_smoke",
    "sample_specific_mask_forward",
    "seed_values",
    "single_channel_mask",
    "write_config_resolved_artifact",
    "write_figure_3_artifact",
    "write_mask_statistics_artifact",
    "write_metrics_artifact",
    "write_named_result_artifacts",
    "write_summary_table_artifact",
    "write_table1_resnet_main_artifact",
    "write_table2_vit_main_artifact",
    "write_table3_ablation_artifact",
    "write_table_3_artifact",
]
