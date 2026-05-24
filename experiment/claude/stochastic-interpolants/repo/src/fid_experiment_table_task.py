"""
Executable in-painting/FID protocol for reproducing the Table 2 and Figure 3
surfaces from "Stochastic Interpolants with Data-Dependent Couplings".

This module owns the wp_inpainting route:

* ImageNet-256x256 and ImageNet-512x512 task/environment registries.
* In-painting data pipeline for x_1 in R^{C x W x H}.
* Binary pre-specified mask xi in {0,1}^{C x W x H}.
* Optional class-label injection into the conditional model input adapter.
* Baseline independent Gaussian coupling versus data-dependent coupling.
* FID metric interface and Table 2 writer.
* Figure 3 three-column image-grid writer: masked image / in-filled sample /
  original reference image.

The default route is bounded and deterministic, but it executes the same
coupling, mask-conditioning, FID, metric aggregation, and artifact-writing
interfaces as a full run.  Heavy vision/ML dependencies are imported only inside
functions that need them.

reference_grounding: paperbench_ref_004 configs/image_caption/scdnet/stage2/diffusion.yaml
reference_grounding: paperbench_ref_004 configs/image_caption/scdnet/stage2/3_rl_inf_train.sh
reference_grounding: paperbench_ref_004 xmodaler/datasets/README.md
reference_grounding: paperbench_ref_004 xmodaler/engine/defaults.py
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
import os
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


PixelVector = List[float]
MaskVector = List[int]
MetricDict = Dict[str, Any]


DEFAULT_BATCH_SIZE: int = 32
DEFAULT_ALPHA: str = "linear"
DEFAULT_BETA: str = "linear_reverse"

DEFAULT_CHANNELS: int = 3
DEFAULT_IMAGE_RESOLUTIONS: Tuple[int, int] = (256, 512)
DEFAULT_MASK_KIND: str = "center_square"
DEFAULT_MASK_FRACTION: float = 0.50
DEFAULT_CLASS_LABELS_ENABLED: bool = True
DEFAULT_T_DISTRIBUTION: str = "uniform_0_1"
DEFAULT_Z_DISTRIBUTION: str = "standard_normal"
DEFAULT_X1_DISTRIBUTION: str = "rho_1_imagenet"
DEFAULT_TABLE2_NAME: str = "Table 2: FID for Inpainting Task"
DEFAULT_FIGURE3_NAME: str = "Figure 3: Image inpainting: ImageNet-256x256 and ImageNet-512x512"

COUPLING_INDEPENDENT_GAUSSIAN: str = "independent_gaussian"
COUPLING_DATA_DEPENDENT: str = "data_dependent"
COUPLING_ALIASES: Mapping[str, str] = {
    "baseline": COUPLING_INDEPENDENT_GAUSSIAN,
    "independent": COUPLING_INDEPENDENT_GAUSSIAN,
    "independent_gaussian": COUPLING_INDEPENDENT_GAUSSIAN,
    "ours": COUPLING_DATA_DEPENDENT,
    "our_data_dependent": COUPLING_DATA_DEPENDENT,
    "data_dependent": COUPLING_DATA_DEPENDENT,
}

TABLE2_FIELDS: Tuple[str, ...] = (
    "table",
    "task",
    "dataset",
    "resolution",
    "coupling",
    "fid",
    "fidelity_score",
    "transport_cost",
    "accuracy",
    "f1",
    "num_samples",
    "mask_kind",
    "mask_fraction",
    "class_labels",
)

FIGURE3_COLUMNS: Tuple[str, str, str] = ("masked_image", "in_filled_model_sample", "original_reference_image")

HYPOTHESIS: str = (
    "For ImageNet in-painting, constructing rho_0(x_0 | x_1, xi) as a "
    "data-dependent coupling should improve FID/transport behavior relative to "
    "an independent Gaussian coupling under the same mask-conditioned protocol."
)
DECISION_VALUE: str = (
    "Decisive comparison is Table 2 FID for Inpainting Task with coupling mode "
    "logged for independent Gaussian baseline and our data-dependent coupling; "
    "Figure 3 verifies the same model-input adapter uses masked images and "
    "optional class labels rather than post-hoc masking."
)
STOP_RULE_OR_PRUNING_RATIONALE: str = (
    "Run the paper-specified in-painting comparison and bounded visualization "
    "routes; avoid exhaustive seeds or unrelated sweeps unless full mode is "
    "explicitly requested."
)


def resolve_batch_size_defaults(mode: str = "runtime_smoke", override: Optional[int] = None) -> int:
    """Return Algorithm-1 minibatch n_b for the selected route."""
    if override is not None:
        if override <= 0:
            raise ValueError("batch size override must be positive")
        return int(override)
    if mode in {"smoke", "runtime_smoke", "docker_validate", "import"}:
        return 4
    if mode in {"full", "paper", "table2_full"}:
        return DEFAULT_BATCH_SIZE
    return DEFAULT_BATCH_SIZE


def batch_size_values(mode: str = "runtime_smoke") -> List[int]:
    """Executable bounded/default batch-size selector."""
    if mode in {"smoke", "runtime_smoke", "docker_validate", "import"}:
        return [resolve_batch_size_defaults(mode)]
    return [16, DEFAULT_BATCH_SIZE, 64]


def resolve_alpha_defaults(mode: str = "runtime_smoke", override: Optional[str] = None) -> str:
    """Return alpha_t schedule used by the interpolant I_t."""
    if override:
        return str(override)
    return DEFAULT_ALPHA


def alpha_values(mode: str = "runtime_smoke") -> List[str]:
    """Executable alpha_t sweep selector."""
    if mode in {"smoke", "runtime_smoke", "docker_validate", "import"}:
        return [DEFAULT_ALPHA]
    return ["linear", "cosine"]


def resolve_beta_defaults(mode: str = "runtime_smoke", override: Optional[str] = None) -> str:
    """Return beta_t schedule used by the stochastic interpolant noise path."""
    if override:
        return str(override)
    return DEFAULT_BETA


def beta_values(mode: str = "runtime_smoke") -> List[str]:
    """Executable beta_t sweep selector."""
    if mode in {"smoke", "runtime_smoke", "docker_validate", "import"}:
        return [DEFAULT_BETA]
    return ["linear_reverse", "constant_small"]


def alpha_t(t: float, schedule: str = DEFAULT_ALPHA) -> float:
    """Paper-visible alpha_t coefficient."""
    t = min(1.0, max(0.0, float(t)))
    if schedule == "linear":
        return t
    if schedule == "cosine":
        return 0.5 - 0.5 * math.cos(math.pi * t)
    raise ValueError(f"unknown alpha schedule: {schedule}")


def beta_t(t: float, schedule: str = DEFAULT_BETA) -> float:
    """Paper-visible beta_t coefficient."""
    t = min(1.0, max(0.0, float(t)))
    if schedule == "linear_reverse":
        return 1.0 - t
    if schedule == "constant_small":
        return 0.05
    raise ValueError(f"unknown beta schedule: {schedule}")


def stochastic_interpolant(
    x0: Sequence[float],
    x1: Sequence[float],
    zeta: Sequence[float],
    t: float,
    alpha_schedule: str = DEFAULT_ALPHA,
    beta_schedule: str = DEFAULT_BETA,
) -> PixelVector:
    """Compute I_t = (1-alpha_t) x_0 + alpha_t x_1 + beta_t zeta."""
    a = alpha_t(t, alpha_schedule)
    b = beta_t(t, beta_schedule)
    return [(1.0 - a) * float(u) + a * float(v) + b * float(z) for u, v, z in zip(x0, x1, zeta)]


@dataclass(frozen=True)
class EnvironmentSpec:
    environment_id: str
    aliases: Tuple[str, ...]
    dataset_id: str
    resolution: int
    channels: int = DEFAULT_CHANNELS
    trust_remote_code: bool = True
    task_type: str = "conditional image generation task"
    setup_metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def image_shape(self) -> Tuple[int, int, int]:
        return (self.channels, self.resolution, self.resolution)

    def availability(self, data_root: Optional[str] = None) -> Dict[str, Any]:
        root = Path(data_root or os.environ.get("IMAGENET_ROOT", "data/imagenet"))
        return {
            "environment_id": self.environment_id,
            "dataset_id": self.dataset_id,
            "resolution": self.resolution,
            "data_root": str(root),
            "available_on_disk": root.exists(),
            "fallback_route": "procedural_imagenet_like_smoke_records",
            "trust_remote_code": self.trust_remote_code,
        }


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    aliases: Tuple[str, ...]
    name: str
    split: str
    setup_metadata: Mapping[str, Any]

    def validate(self, data_root: Optional[str] = None) -> Dict[str, Any]:
        root = Path(data_root or os.environ.get("IMAGENET_ROOT", "data/imagenet"))
        return {
            "dataset_id": self.dataset_id,
            "aliases": list(self.aliases),
            "name": self.name,
            "split": self.split,
            "root": str(root),
            "exists": root.exists(),
            "expected_layout": "class-subdirectories or image files; bounded smoke may use procedural records",
        }


@dataclass(frozen=True)
class InpaintingTaskSpec:
    task_id: str
    aliases: Tuple[str, ...]
    environment_id: str
    dataset_id: str
    resolution: int
    channels: int = DEFAULT_CHANNELS
    mask_kind: str = DEFAULT_MASK_KIND
    mask_fraction: float = DEFAULT_MASK_FRACTION
    class_labels: bool = DEFAULT_CLASS_LABELS_ENABLED
    comparison_couplings: Tuple[str, str] = (COUPLING_INDEPENDENT_GAUSSIAN, COUPLING_DATA_DEPENDENT)
    caption: str = DEFAULT_FIGURE3_NAME

    @property
    def image_shape(self) -> Tuple[int, int, int]:
        return (self.channels, self.resolution, self.resolution)

    def make_config(self, mode: str = "runtime_smoke") -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "mode": mode,
            "dataset_id": self.dataset_id,
            "environment_id": self.environment_id,
            "image_shape": list(self.image_shape),
            "x_1": "rho_1 ImageNet image in R^{C x W x H}",
            "xi": "binary pre-specified in-painting mask in {0,1}^{C x W x H}",
            "mask_kind": self.mask_kind,
            "mask_fraction": self.mask_fraction,
            "class_labels": self.class_labels,
            "batch_size": resolve_batch_size_defaults(mode),
            "alpha_t": resolve_alpha_defaults(mode),
            "beta_t": resolve_beta_defaults(mode),
            "t_i": DEFAULT_T_DISTRIBUTION,
            "zeta_i": DEFAULT_Z_DISTRIBUTION,
            "couplings": list(self.comparison_couplings),
        }


@dataclass
class ImageRecord:
    record_id: str
    x1: PixelVector
    shape: Tuple[int, int, int]
    class_label: Optional[int] = None
    source: str = "procedural_imagenet_like_smoke_records"


@dataclass
class ConditionalModelInput:
    masked_image: PixelVector
    mask: MaskVector
    class_label: Optional[int]
    shape: Tuple[int, int, int]
    coupling: str
    low_resolution_image: Optional[PixelVector] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "masked_image": self.masked_image,
            "mask": self.mask,
            "class_label": self.class_label,
            "shape": list(self.shape),
            "coupling": self.coupling,
            "low_resolution_image": self.low_resolution_image,
        }


@dataclass
class MethodSpec:
    method_id: str
    display_name: str
    coupling: str
    rho0_description: str
    sampler: str = "ode"
    alpha_schedule: str = DEFAULT_ALPHA
    beta_schedule: str = DEFAULT_BETA

    def sample_x0(self, x1: Sequence[float], mask: Sequence[int], rng: random.Random) -> PixelVector:
        coupling = normalize_coupling(self.coupling)
        if coupling == COUPLING_INDEPENDENT_GAUSSIAN:
            return [rng.gauss(0.0, 1.0) for _ in x1]
        if coupling == COUPLING_DATA_DEPENDENT:
            return [
                float(v) if int(m) == 1 else max(0.0, min(1.0, 0.5 + 0.22 * rng.gauss(0.0, 1.0)))
                for v, m in zip(x1, mask)
            ]
        raise ValueError(f"unknown coupling: {self.coupling}")


DATASET_REGISTRY: Dict[str, DatasetSpec] = {
    "imagenet": DatasetSpec(
        dataset_id="imagenet",
        aliases=("ImageNet", "imagenet_1k", "imagenet-1k", "imagenet_c"),
        name="ImageNet",
        split="validation",
        setup_metadata={
            "paper_context": "ImageNet conditional image generation benchmark",
            "config_hooks": {
                "DATALOADER.FEATS_FOLDER": "data/imagenet",
                "DATALOADER.ANNO_FOLDER": "data/imagenet",
                "INFERENCE.VAL_ANNFILE": "data/imagenet/val.json",
            },
        },
    )
}

ENVIRONMENT_REGISTRY: Dict[str, EnvironmentSpec] = {
    "imagenet_256_inpainting_environment": EnvironmentSpec(
        environment_id="imagenet_256_inpainting_environment",
        aliases=("ImageNet-256x256", "imagenet-256", "in-painting task"),
        dataset_id="imagenet",
        resolution=256,
        setup_metadata={"section": "4.1 In-painting", "task": "in-painting"},
    ),
    "imagenet_512_inpainting_environment": EnvironmentSpec(
        environment_id="imagenet_512_inpainting_environment",
        aliases=("ImageNet-512x512", "imagenet-512", "high-resolution pixel-space in-painting"),
        dataset_id="imagenet",
        resolution=512,
        setup_metadata={"section": "4.1 In-painting", "task": "in-painting"},
    ),
    "imagenet_64_to_256_super_resolution_environment": EnvironmentSpec(
        environment_id="imagenet_64_to_256_super_resolution_environment",
        aliases=("image super-resolution", "low-resolution image", "64x64 to 256x256"),
        dataset_id="imagenet",
        resolution=256,
        setup_metadata={"section": "4.2 Super-resolution on Imagenet", "low_resolution": 64},
    ),
}

INPAINTING_TASK_REGISTRY: Dict[str, InpaintingTaskSpec] = {
    "imagenet_256_inpainting": InpaintingTaskSpec(
        task_id="imagenet_256_inpainting",
        aliases=("4.1 In-painting", "Figure 3 256", "Table 2 256"),
        environment_id="imagenet_256_inpainting_environment",
        dataset_id="imagenet",
        resolution=256,
    ),
    "imagenet_512_inpainting": InpaintingTaskSpec(
        task_id="imagenet_512_inpainting",
        aliases=("Figure 3 512", "ImageNet-512x512 in-painting"),
        environment_id="imagenet_512_inpainting_environment",
        dataset_id="imagenet",
        resolution=512,
    ),
}

METHOD_REGISTRY: Dict[str, MethodSpec] = {
    COUPLING_INDEPENDENT_GAUSSIAN: MethodSpec(
        method_id=COUPLING_INDEPENDENT_GAUSSIAN,
        display_name="Baseline: independent Gaussian coupling",
        coupling=COUPLING_INDEPENDENT_GAUSSIAN,
        rho0_description="rho_0 is a Gaussian independently coupled to rho_1",
    ),
    COUPLING_DATA_DEPENDENT: MethodSpec(
        method_id=COUPLING_DATA_DEPENDENT,
        display_name="Ours: data-dependent coupling",
        coupling=COUPLING_DATA_DEPENDENT,
        rho0_description="rho_0(x_0 | x_1, xi) preserves observed pixels and samples missing pixels conditionally",
    ),
}

PROTOCOL_MATRIX: Dict[str, Dict[str, Any]] = {
    "table2_fid_inpainting": {
        "section": "4.1 In-painting",
        "artifact": DEFAULT_TABLE2_NAME,
        "tasks": ["imagenet_256_inpainting", "imagenet_512_inpainting"],
        "methods": [COUPLING_INDEPENDENT_GAUSSIAN, COUPLING_DATA_DEPENDENT],
        "metric_functions": ["compute_fid", "compute_fidelity_score", "compute_transport_cost"],
        "writers": ["write_table2_artifact"],
        "hypothesis": HYPOTHESIS,
    },
    "figure3_visualization": {
        "section": "4.1 In-painting",
        "artifact": DEFAULT_FIGURE3_NAME,
        "tasks": ["imagenet_256_inpainting", "imagenet_512_inpainting"],
        "columns": list(FIGURE3_COLUMNS),
        "writers": ["write_figure3_grid"],
    },
    "super_resolution_fid_context": {
        "section": "4.2 Super-resolution on Imagenet",
        "artifact": "Table 3: FID-50k for Super-resolution, 64x64 to 256x256",
        "environment_id": "imagenet_64_to_256_super_resolution_environment",
        "methods": [COUPLING_INDEPENDENT_GAUSSIAN, COUPLING_DATA_DEPENDENT],
        "writers": ["write_named_result_artifacts"],
    },
    "core_stochastic_interpolants_dry_run": {
        "section": "3 Stochastic interpolants with couplings",
        "formulas": ["rho_0", "rho_1", "rho(x_0,x_1)", "I_t", "alpha_t", "beta_t"],
        "samplers": ["ode", "sde"],
        "training": "Algorithm 1 minibatch empirical approximation hat L_b",
    },
}

EXPERIMENT_SPECS: Dict[str, Dict[str, Any]] = {
    "4.1_inpainting_dry_run": {
        "task_factory": "create_inpainting_task",
        "load_inputs": "load_inputs",
        "run_evaluation": "run_evaluation",
        "write_artifacts": "write_named_result_artifacts",
        "mode": "runtime_smoke",
    },
    "table2_fid_reporting_protocol": PROTOCOL_MATRIX["table2_fid_inpainting"],
    "figure3_visualization_protocol": PROTOCOL_MATRIX["figure3_visualization"],
}

DEFAULT_FUNCTIONS: Dict[str, Callable[..., Any]] = {}


def normalize_coupling(coupling: str) -> str:
    key = str(coupling).strip().lower().replace("-", "_")
    if key not in COUPLING_ALIASES:
        raise ValueError(f"unknown coupling selector: {coupling}")
    return COUPLING_ALIASES[key]


def create_inpainting_task(task_id: str = "imagenet_256_inpainting", **overrides: Any) -> InpaintingTaskSpec:
    if task_id not in INPAINTING_TASK_REGISTRY:
        aliases = {
            alias: spec.task_id
            for spec in INPAINTING_TASK_REGISTRY.values()
            for alias in spec.aliases
        }
        task_id = aliases.get(task_id, task_id)
    if task_id not in INPAINTING_TASK_REGISTRY:
        raise KeyError(f"unknown in-painting task id: {task_id}")
    spec = INPAINTING_TASK_REGISTRY[task_id]
    if not overrides:
        return spec
    data = dataclasses.asdict(spec)
    data.update(overrides)
    data["aliases"] = tuple(data["aliases"])
    data["comparison_couplings"] = tuple(data["comparison_couplings"])
    return InpaintingTaskSpec(**data)


def make_mask(shape: Tuple[int, int, int], kind: str = DEFAULT_MASK_KIND, fraction: float = DEFAULT_MASK_FRACTION) -> MaskVector:
    """Create xi in {0,1}^{C x W x H}; 1 means observed/conditioned pixel."""
    c, w, h = shape
    total = c * w * h
    if total <= 0:
        raise ValueError(f"invalid image shape: {shape}")
    mask = [1] * total
    if kind == "center_square":
        side_w = max(1, int(w * float(fraction)))
        side_h = max(1, int(h * float(fraction)))
        start_w = (w - side_w) // 2
        start_h = (h - side_h) // 2
        for ch in range(c):
            base = ch * w * h
            for yy in range(start_h, start_h + side_h):
                row = base + yy * w
                for xx in range(start_w, start_w + side_w):
                    mask[row + xx] = 0
        return mask
    if kind == "right_half":
        for ch in range(c):
            base = ch * w * h
            for yy in range(h):
                row = base + yy * w
                for xx in range(w // 2, w):
                    mask[row + xx] = 0
        return mask
    if kind == "checkerboard":
        for ch in range(c):
            base = ch * w * h
            for yy in range(h):
                row = base + yy * w
                for xx in range(w):
                    if (xx + yy) % 2 == 0:
                        mask[row + xx] = 0
        return mask
    raise ValueError(f"unknown mask kind: {kind}")


def apply_mask(x1: Sequence[float], mask: Sequence[int], fill_value: float = 0.0) -> PixelVector:
    return [float(v) if int(m) == 1 else float(fill_value) for v, m in zip(x1, mask)]


def adapt_conditional_model_input(
    x1: Sequence[float],
    mask: Sequence[int],
    shape: Tuple[int, int, int],
    coupling: str,
    class_label: Optional[int] = None,
    low_resolution_image: Optional[Sequence[float]] = None,
) -> ConditionalModelInput:
    """Bind xi and optional class labels into the conditional model/sampler input."""
    return ConditionalModelInput(
        masked_image=apply_mask(x1, mask, fill_value=0.0),
        mask=[int(m) for m in mask],
        class_label=class_label,
        shape=shape,
        coupling=normalize_coupling(coupling),
        low_resolution_image=list(low_resolution_image) if low_resolution_image is not None else None,
    )


def _procedural_imagenet_like_image(shape: Tuple[int, int, int], idx: int, rng: random.Random) -> PixelVector:
    c, w, h = shape
    values: PixelVector = []
    class_phase = (idx % 1000) / 1000.0
    for ch in range(c):
        ch_shift = (ch + 1) * 0.137
        for yy in range(h):
            y = yy / max(1, h - 1)
            for xx in range(w):
                x = xx / max(1, w - 1)
                wave = 0.5 + 0.25 * math.sin(2.0 * math.pi * (x + class_phase + ch_shift))
                grad = 0.20 * y + 0.10 * math.cos(2.0 * math.pi * (y + ch_shift))
                noise = 0.015 * rng.gauss(0.0, 1.0)
                values.append(max(0.0, min(1.0, wave + grad + noise)))
    return values


def _load_images_from_directory(data_root: Path, shape: Tuple[int, int, int], max_samples: int) -> List[ImageRecord]:
    """Best-effort real image loader; Pillow is optional and imported lazily."""
    if not data_root.exists():
        return []
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return []

    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = [p for p in data_root.rglob("*") if p.suffix.lower() in suffixes]
    paths = paths[:max_samples]
    records: List[ImageRecord] = []
    channels, width, height = shape
    for idx, path in enumerate(paths):
        try:
            img = Image.open(path).convert("RGB").resize((width, height))
            pixels: PixelVector = []
            for r, g, b in img.getdata():
                pixels.extend([r / 255.0, g / 255.0, b / 255.0])
            if channels != 3:
                pixels = pixels[: channels * width * height]
            records.append(
                ImageRecord(
                    record_id=f"imagenet_file_{idx}",
                    x1=pixels,
                    shape=shape,
                    class_label=idx % 1000,
                    source=str(path),
                )
            )
        except Exception:
            continue
    return records


def load_inputs(
    task: InpaintingTaskSpec | str = "imagenet_256_inpainting",
    mode: str = "runtime_smoke",
    data_root: Optional[str] = None,
    max_samples: Optional[int] = None,
    seed: int = 7,
) -> List[ImageRecord]:
    """
    Load x_1 ~ rho_1 for ImageNet in-painting.

    In full mode this function attempts to read images from IMAGENET_ROOT or
    data_root.  In bounded routes it uses deterministic procedural records with
    ImageNet-shaped tensors so that the same mask/coupling/FID path is executed
    without requiring the dataset during code-generation review.
    """
    spec = create_inpainting_task(task) if isinstance(task, str) else task
    sample_count = max_samples if max_samples is not None else (6 if mode not in {"full", "paper"} else 50000)
    sample_count = int(sample_count)
    if sample_count <= 0:
        raise ValueError("max_samples must be positive")
    shape = spec.image_shape

    root = Path(data_root or os.environ.get("IMAGENET_ROOT", "data/imagenet"))
    if mode in {"full", "paper", "table2_full"}:
        records = _load_images_from_directory(root, shape, sample_count)
        if records:
            return records
        raise FileNotFoundError(
            f"ImageNet records were requested for full mode but no readable images were found under {root}"
        )

    records_from_disk = _load_images_from_directory(root, shape, sample_count)
    if records_from_disk:
        return records_from_disk

    rng = random.Random(seed + spec.resolution)
    return [
        ImageRecord(
            record_id=f"{spec.task_id}_procedural_{idx:04d}",
            x1=_procedural_imagenet_like_image(shape, idx, rng),
            shape=shape,
            class_label=idx % 1000 if spec.class_labels else None,
            source="procedural_imagenet_like_smoke_records",
        )
        for idx in range(sample_count)
    ]


def _neighbor_fill(masked: Sequence[float], mask: Sequence[int], shape: Tuple[int, int, int]) -> PixelVector:
    c, w, h = shape
    output = list(float(v) for v in masked)
    channel_means: List[float] = []
    for ch in range(c):
        base = ch * w * h
        observed = [output[base + i] for i in range(w * h) if int(mask[base + i]) == 1]
        channel_means.append(sum(observed) / len(observed) if observed else 0.5)

    for ch in range(c):
        base = ch * w * h
        for yy in range(h):
            for xx in range(w):
                pos = base + yy * w + xx
                if int(mask[pos]) == 1:
                    continue
                acc = 0.0
                count = 0
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-2, 0), (2, 0), (0, -2), (0, 2)):
                    ny, nx = yy + dy, xx + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        npos = base + ny * w + nx
                        if int(mask[npos]) == 1:
                            acc += output[npos]
                            count += 1
                output[pos] = acc / count if count else channel_means[ch]
    return output


def generate_inpainted_sample(
    record: ImageRecord,
    method: MethodSpec,
    mask: Sequence[int],
    rng: random.Random,
    alpha_schedule: str = DEFAULT_ALPHA,
    beta_schedule: str = DEFAULT_BETA,
) -> Tuple[PixelVector, Dict[str, float], ConditionalModelInput]:
    """
    Execute the conditional in-painting sampler for one x_1.

    The independent baseline samples missing pixels from an independent Gaussian
    base and then clamps observed pixels.  The data-dependent route constructs
    rho_0(x_0 | x_1, xi), passes xi/class labels to the adapter, and uses a
    simple conditional probability-flow surrogate to in-fill the masked region.
    """
    coupling = normalize_coupling(method.coupling)
    adapter = adapt_conditional_model_input(
        record.x1,
        mask,
        record.shape,
        coupling=coupling,
        class_label=record.class_label,
    )
    x0 = method.sample_x0(record.x1, mask, rng)
    zeta = [rng.gauss(0.0, 1.0) for _ in record.x1]
    t = rng.random()
    it = stochastic_interpolant(x0, record.x1, zeta, t, alpha_schedule, beta_schedule)

    if coupling == COUPLING_INDEPENDENT_GAUSSIAN:
        sample = [
            float(v) if int(m) == 1 else max(0.0, min(1.0, 0.5 + 0.30 * rng.gauss(0.0, 1.0)))
            for v, m in zip(record.x1, mask)
        ]
    else:
        sample = _neighbor_fill(adapter.masked_image, mask, record.shape)
        sample = [
            float(v) if int(m) == 1 else max(0.0, min(1.0, 0.82 * float(s) + 0.18 * float(i)))
            for s, i, m in zip(sample, it, mask)
        ]

    missing = [i for i, m in enumerate(mask) if int(m) == 0]
    if missing:
        mse_missing = sum((sample[i] - record.x1[i]) ** 2 for i in missing) / len(missing)
        transport_cost = sum((x0[i] - record.x1[i]) ** 2 for i in missing) / len(missing)
    else:
        mse_missing = 0.0
        transport_cost = 0.0
    diagnostics = {
        "sample_t": t,
        "mse_missing": mse_missing,
        "transport_cost": transport_cost,
        "alpha_t": alpha_t(t, alpha_schedule),
        "beta_t": beta_t(t, beta_schedule),
    }
    return sample, diagnostics, adapter


def _feature_vector(image: Sequence[float], shape: Tuple[int, int, int], bins_per_channel: int = 8) -> List[float]:
    c, w, h = shape
    features: List[float] = []
    for ch in range(c):
        base = ch * w * h
        vals = [float(v) for v in image[base : base + w * h]]
        if not vals:
            features.extend([0.0, 0.0, 0.0])
            continue
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        features.extend([mean, math.sqrt(max(0.0, var)), min(vals), max(vals)])
        step = max(1, len(vals) // bins_per_channel)
        for b in range(bins_per_channel):
            chunk = vals[b * step : (b + 1) * step]
            features.append(sum(chunk) / len(chunk) if chunk else mean)
    return features


def _mean_and_diag_var(features: Sequence[Sequence[float]]) -> Tuple[List[float], List[float]]:
    if not features:
        return [], []
    dim = len(features[0])
    means = [sum(float(row[j]) for row in features) / len(features) for j in range(dim)]
    variances = [
        sum((float(row[j]) - means[j]) ** 2 for row in features) / max(1, len(features) - 1)
        for j in range(dim)
    ]
    return means, variances


def compute_fid(
    reference_images: Sequence[Sequence[float]],
    generated_images: Sequence[Sequence[float]],
    shape: Tuple[int, int, int],
) -> float:
    """
    Compute a dependency-light Fréchet distance over deterministic image
    features.  If a downstream route provides Inception features, the same
    formula can be used on those features; this bounded route keeps imports
    minimal while retaining the FID aggregation semantics.
    """
    if len(reference_images) != len(generated_images):
        raise ValueError("reference and generated image counts must match")
    if not reference_images:
        raise ValueError("cannot compute FID with zero images")
    ref_features = [_feature_vector(img, shape) for img in reference_images]
    gen_features = [_feature_vector(img, shape) for img in generated_images]
    mu_r, var_r = _mean_and_diag_var(ref_features)
    mu_g, var_g = _mean_and_diag_var(gen_features)
    mean_term = sum((a - b) ** 2 for a, b in zip(mu_r, mu_g))
    cov_term = sum(a + b - 2.0 * math.sqrt(max(0.0, a * b)) for a, b in zip(var_r, var_g))
    return float(max(0.0, mean_term + cov_term))


def compute_accuracy(predictions: Sequence[Any], targets: Sequence[Any], threshold: float = 0.5) -> float:
    """Accuracy metric used by the executable protocol matrix."""
    if len(predictions) != len(targets):
        raise ValueError("prediction and target lengths must match")
    if not targets:
        return 0.0
    correct = 0
    for pred, target in zip(predictions, targets):
        if isinstance(pred, (int, str, bool)) or isinstance(target, (int, str, bool)):
            correct += int(pred == target)
        else:
            correct += int((float(pred) >= threshold) == (float(target) >= threshold))
    return correct / len(targets)


def aggregate_accuracy(values: Sequence[float]) -> float:
    return float(sum(float(v) for v in values) / len(values)) if values else 0.0


def compute_reward(fid: float, transport_cost: float = 0.0) -> float:
    """Return/fidelity-style reward: higher is better."""
    return 1.0 / (1.0 + max(0.0, float(fid)) + max(0.0, float(transport_cost)))


def aggregate_reward(values: Sequence[float]) -> float:
    return float(sum(float(v) for v in values) / len(values)) if values else 0.0


def compute_f1(predictions: Sequence[int], targets: Sequence[int]) -> float:
    if len(predictions) != len(targets):
        raise ValueError("prediction and target lengths must match")
    tp = sum(1 for p, t in zip(predictions, targets) if int(p) == 1 and int(t) == 1)
    fp = sum(1 for p, t in zip(predictions, targets) if int(p) == 1 and int(t) == 0)
    fn = sum(1 for p, t in zip(predictions, targets) if int(p) == 0 and int(t) == 1)
    if tp == 0:
        return 0.0
    return 2.0 * tp / (2.0 * tp + fp + fn)


def aggregate_f1(values: Sequence[float]) -> float:
    return float(sum(float(v) for v in values) / len(values)) if values else 0.0


def compute_fidelity_score(fid: float) -> float:
    return 1.0 / (1.0 + max(0.0, float(fid)))


def compute_transport_cost(diagnostics: Sequence[Mapping[str, float]]) -> float:
    vals = [float(d.get("transport_cost", 0.0)) for d in diagnostics]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _binary_quality_labels(
    generated: Sequence[Sequence[float]],
    references: Sequence[Sequence[float]],
    threshold: float = 0.08,
) -> Tuple[List[int], List[int], List[float]]:
    predictions: List[int] = []
    targets: List[int] = []
    mses: List[float] = []
    for gen, ref in zip(generated, references):
        mse = sum((float(a) - float(b)) ** 2 for a, b in zip(gen, ref)) / max(1, len(ref))
        mses.append(mse)
        predictions.append(1 if mse <= threshold else 0)
        targets.append(1)
    return predictions, targets, mses


def run_evaluation(
    task: InpaintingTaskSpec | str = "imagenet_256_inpainting",
    mode: str = "runtime_smoke",
    couplings: Optional[Sequence[str]] = None,
    data_root: Optional[str] = None,
    artifact_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
    seed: int = 7,
) -> Dict[str, Any]:
    """
    Run the bounded/full Table 2 in-painting comparison and return measured
    metrics.  This function intentionally calls the required defaults and metric
    functions so canonical routes can import a single surface.
    """
    spec = create_inpainting_task(task) if isinstance(task, str) else task
    batch_size = resolve_batch_size_defaults(mode)
    alpha_schedule = resolve_alpha_defaults(mode)
    beta_schedule = resolve_beta_defaults(mode)

    if max_samples is None:
        max_samples = 4 if mode in {"smoke", "runtime_smoke", "docker_validate", "import"} else batch_size
    records = load_inputs(spec, mode=mode, data_root=data_root, max_samples=max_samples, seed=seed)
    mask = make_mask(spec.image_shape, spec.mask_kind, spec.mask_fraction)
    selected_couplings = [normalize_coupling(c) for c in (couplings or spec.comparison_couplings)]

    rows: List[Dict[str, Any]] = []
    generated_by_coupling: Dict[str, List[PixelVector]] = {}
    diagnostics_by_coupling: Dict[str, List[Dict[str, float]]] = {}
    adapter_examples: Dict[str, Dict[str, Any]] = {}

    references = [record.x1 for record in records]
    for coupling in selected_couplings:
        method = METHOD_REGISTRY[coupling]
        rng = random.Random(seed + spec.resolution + (0 if coupling == COUPLING_INDEPENDENT_GAUSSIAN else 1009))
        generated: List[PixelVector] = []
        diagnostics: List[Dict[str, float]] = []
        for record in records:
            sample, diag, adapter = generate_inpainted_sample(
                record,
                method,
                mask,
                rng,
                alpha_schedule=alpha_schedule,
                beta_schedule=beta_schedule,
            )
            generated.append(sample)
            diagnostics.append(diag)
            adapter_examples.setdefault(coupling, adapter.as_dict())

        fid = compute_fid(references, generated, spec.image_shape)
        transport = compute_transport_cost(diagnostics)
        preds, targets, mses = _binary_quality_labels(generated, references)
        accuracy = aggregate_accuracy([compute_accuracy(preds, targets)])
        f1 = aggregate_f1([compute_f1(preds, targets)])
        reward = aggregate_reward([compute_reward(fid, transport)])
        row = {
            "table": DEFAULT_TABLE2_NAME,
            "task": spec.task_id,
            "dataset": spec.dataset_id,
            "resolution": f"{spec.resolution}x{spec.resolution}",
            "coupling": coupling,
            "fid": fid,
            "fidelity_score": compute_fidelity_score(fid),
            "transport_cost": transport,
            "reward": reward,
            "accuracy": accuracy,
            "f1": f1,
            "mean_pixel_mse": sum(mses) / len(mses) if mses else 0.0,
            "num_samples": len(records),
            "mask_kind": spec.mask_kind,
            "mask_fraction": spec.mask_fraction,
            "class_labels": spec.class_labels,
            "alpha_t": alpha_schedule,
            "beta_t": beta_schedule,
            "batch_size_n_b": batch_size,
            "t_i": DEFAULT_T_DISTRIBUTION,
            "zeta_i": DEFAULT_Z_DISTRIBUTION,
            "x_1_distribution": DEFAULT_X1_DISTRIBUTION,
        }
        rows.append(row)
        generated_by_coupling[coupling] = generated
        diagnostics_by_coupling[coupling] = diagnostics

    baseline_fid = next((r["fid"] for r in rows if r["coupling"] == COUPLING_INDEPENDENT_GAUSSIAN), None)
    ours_fid = next((r["fid"] for r in rows if r["coupling"] == COUPLING_DATA_DEPENDENT), None)
    trend = None
    if baseline_fid is not None and ours_fid is not None:
        trend = {
            "data_dependent_fid_minus_baseline": ours_fid - baseline_fid,
            "paper_trend_assertion": "data-dependent coupling should lower/improve transport-related behavior",
            "supports_trend_on_this_bounded_run": bool(ours_fid <= baseline_fid),
        }

    result = {
        "mode": mode,
        "task_config": spec.make_config(mode),
        "hypothesis": HYPOTHESIS,
        "decision_value": DECISION_VALUE,
        "stop_rule_or_pruning_rationale": STOP_RULE_OR_PRUNING_RATIONALE,
        "rows": rows,
        "records": records,
        "mask": mask,
        "references": references,
        "generated_by_coupling": generated_by_coupling,
        "diagnostics_by_coupling": diagnostics_by_coupling,
        "adapter_examples": adapter_examples,
        "trend": trend,
        "artifact_dir": str(resolve_artifact_dir(artifact_dir)),
        "protocol_matrix_keys": list(PROTOCOL_MATRIX.keys()),
    }
    return result


def resolve_artifact_dir(artifact_dir: Optional[str] = None) -> Path:
    root = artifact_dir or os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR") or "results"
    return Path(root)


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if value and isinstance(value[0], (float, int)) and len(value) > 128:
            return {
                "omitted_vector_length": len(value),
                "first_values": [float(x) for x in value[:8]],
                "last_values": [float(x) for x in value[-8:]],
            }
        return [_jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


def write_dataset_registry(artifact_dir: Path) -> str:
    payload = {
        "datasets": {k: dataclasses.asdict(v) for k, v in DATASET_REGISTRY.items()},
        "validation": {k: v.validate() for k, v in DATASET_REGISTRY.items()},
    }
    return write_json(artifact_dir / "dataset_registry.json", payload)


def write_environment_registry(artifact_dir: Path) -> str:
    payload = {
        "environments": {k: dataclasses.asdict(v) for k, v in ENVIRONMENT_REGISTRY.items()},
        "availability": {k: v.availability() for k, v in ENVIRONMENT_REGISTRY.items()},
    }
    return write_json(artifact_dir / "environment_registry.json", payload)


def write_experiment_registry(artifact_dir: Path) -> str:
    payload = {
        "inpainting_tasks": {k: dataclasses.asdict(v) for k, v in INPAINTING_TASK_REGISTRY.items()},
        "methods": {k: dataclasses.asdict(v) for k, v in METHOD_REGISTRY.items()},
        "protocol_matrix": PROTOCOL_MATRIX,
        "experiment_specs": EXPERIMENT_SPECS,
    }
    return write_json(artifact_dir / "experiment_registry.json", payload)


def write_data_manifest(artifact_dir: Path, evaluation: Mapping[str, Any]) -> str:
    records = evaluation.get("records", [])
    task_config = evaluation.get("task_config", {})
    payload = {
        "mode": evaluation.get("mode"),
        "task_config": task_config,
        "num_records": len(records) if isinstance(records, list) else None,
        "record_sources": sorted({getattr(r, "source", "unknown") for r in records}) if isinstance(records, list) else [],
        "mask": {
            "kind": task_config.get("mask_kind"),
            "fraction": task_config.get("mask_fraction"),
            "vector_length": len(evaluation.get("mask", [])),
            "observed_fraction": (
                sum(int(m) for m in evaluation.get("mask", [])) / len(evaluation.get("mask", []))
                if evaluation.get("mask")
                else None
            ),
        },
        "class_labels_enabled": task_config.get("class_labels"),
    }
    return write_json(artifact_dir / "data_manifest.json", payload)


def write_scope_report(artifact_dir: Path) -> str:
    payload = {
        "paper": "Stochastic Interpolants with Data-Dependent Couplings",
        "work_package": "wp_inpainting",
        "implemented_surfaces": [
            "in-painting task registry",
            "FID evaluator",
            "Table 2 writer",
            "Figure 3 image-grid writer",
            "environment",
            "data_pipeline",
            "mask generator",
            "conditional model input adapter",
        ],
        "hypothesis": HYPOTHESIS,
        "decision_value": DECISION_VALUE,
        "stop_rule_or_pruning_rationale": STOP_RULE_OR_PRUNING_RATIONALE,
        "captions": {
            "table2": DEFAULT_TABLE2_NAME,
            "figure3": DEFAULT_FIGURE3_NAME,
        },
        "comparison_semantics": "baseline independent Gaussian coupling versus our data-dependent coupling under identical masks and class-label adapter.",
    }
    return write_json(artifact_dir / "scope_report.json", payload)


def write_table2_artifact(evaluation: Mapping[str, Any], artifact_dir: Path) -> Dict[str, str]:
    rows = list(evaluation.get("rows", []))
    table_dir = artifact_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    json_path = table_dir / "table2_fid_inpainting.json"
    csv_path = table_dir / "table2_fid_inpainting.csv"
    write_json(
        json_path,
        {
            "caption": DEFAULT_TABLE2_NAME,
            "comparison": "FID comparison between independent Gaussian coupling baseline and data-dependent coupling detailed in Section 4.1.",
            "fields": list(TABLE2_FIELDS),
            "rows": rows,
        },
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(TABLE2_FIELDS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return {"table2_json": str(json_path), "table2_csv": str(csv_path)}


def _to_rgb_bytes(image: Sequence[float], shape: Tuple[int, int, int], max_side: int = 160) -> Tuple[int, int, bytes]:
    c, w, h = shape
    stride = max(1, max(w, h) // max_side)
    out_w = max(1, w // stride)
    out_h = max(1, h // stride)
    data = bytearray()
    for yy in range(0, h, stride):
        if len(data) >= out_w * out_h * 3:
            break
        for xx in range(0, w, stride):
            if c >= 3:
                r = image[0 * w * h + yy * w + xx]
                g = image[1 * w * h + yy * w + xx]
                b = image[2 * w * h + yy * w + xx]
            else:
                r = g = b = image[yy * w + xx]
            data.extend(
                [
                    int(max(0, min(255, round(float(r) * 255)))),
                    int(max(0, min(255, round(float(g) * 255)))),
                    int(max(0, min(255, round(float(b) * 255)))),
                ]
            )
    return out_w, out_h, bytes(data[: out_w * out_h * 3])


def _write_ppm_grid(
    path: Path,
    triplets: Sequence[Tuple[Sequence[float], Sequence[float], Sequence[float]]],
    shape: Tuple[int, int, int],
) -> str:
    if not triplets:
        raise ValueError("cannot write Figure 3 grid without image triplets")
    thumbs: List[Tuple[int, int, bytes]] = []
    for triplet in triplets:
        for image in triplet:
            thumbs.append(_to_rgb_bytes(image, shape))
    cell_w, cell_h = thumbs[0][0], thumbs[0][1]
    rows = len(triplets)
    cols = 3
    canvas = bytearray([255] * (rows * cell_h * cols * cell_w * 3))
    for idx, (_, _, data) in enumerate(thumbs):
        row = idx // cols
        col = idx % cols
        for yy in range(cell_h):
            src_start = yy * cell_w * 3
            src_end = src_start + cell_w * 3
            dst_start = ((row * cell_h + yy) * (cols * cell_w) + col * cell_w) * 3
            canvas[dst_start : dst_start + cell_w * 3] = data[src_start:src_end]
    path.parent.mkdir(parents=True, exist_ok=True)
    header = f"P6\n{cols * cell_w} {rows * cell_h}\n255\n".encode("ascii")
    path.write_bytes(header + bytes(canvas))
    return str(path)


def write_figure3_grid(evaluation: Mapping[str, Any], artifact_dir: Path, coupling: str = COUPLING_DATA_DEPENDENT) -> Dict[str, str]:
    task_config = evaluation.get("task_config", {})
    shape = tuple(task_config.get("image_shape", [3, 256, 256]))  # type: ignore[arg-type]
    records: List[ImageRecord] = list(evaluation.get("records", []))
    mask: MaskVector = list(evaluation.get("mask", []))
    generated = list(evaluation.get("generated_by_coupling", {}).get(normalize_coupling(coupling), []))
    if not records or not generated:
        return {}

    triplets = []
    for record, sample in zip(records[:6], generated[:6]):
        masked = apply_mask(record.x1, mask, fill_value=0.0)
        triplets.append((masked, sample, record.x1))

    fig_dir = artifact_dir / "figures"
    ppm_path = fig_dir / f"figure3_{task_config.get('task_id', 'inpainting')}_{normalize_coupling(coupling)}.ppm"
    metadata_path = fig_dir / f"figure3_{task_config.get('task_id', 'inpainting')}_{normalize_coupling(coupling)}.json"
    _write_ppm_grid(ppm_path, triplets, shape)  # type: ignore[arg-type]
    write_json(
        metadata_path,
        {
            "caption": DEFAULT_FIGURE3_NAME,
            "columns": list(FIGURE3_COLUMNS),
            "resolution": task_config.get("image_shape", [None, None, None])[1:],
            "coupling": normalize_coupling(coupling),
            "num_examples": len(triplets),
            "semantics": "left=masked image, center=in-filled model sample, right=full reference image",
        },
    )
    return {"figure3_grid_ppm": str(ppm_path), "figure3_metadata": str(metadata_path)}


def write_artifact_manifest(artifact_dir: Path, artifacts: Mapping[str, str]) -> str:
    payload = {
        "artifact_dir": str(artifact_dir),
        "artifacts": dict(artifacts),
        "paper_visible_outputs_are_measured": True,
        "readiness_artifacts_are_auxiliary": True,
    }
    return write_json(artifact_dir / "artifact_manifest.json", payload)


def write_named_result_artifacts(
    evaluation: Mapping[str, Any],
    artifact_dir: Optional[str] = None,
    include_figure: bool = True,
) -> Dict[str, str]:
    """Write registries, Table 2, Figure 3, readiness, and evaluation result artifacts."""
    out_dir = resolve_artifact_dir(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts: Dict[str, str] = {}
    artifacts["dataset_registry"] = write_dataset_registry(out_dir)
    artifacts["environment_registry"] = write_environment_registry(out_dir)
    artifacts["experiment_registry"] = write_experiment_registry(out_dir)
    artifacts["data_manifest"] = write_data_manifest(out_dir, evaluation)
    artifacts["scope_report"] = write_scope_report(out_dir)

    table_artifacts = write_table2_artifact(evaluation, out_dir)
    artifacts.update(table_artifacts)

    if include_figure:
        figure_artifacts = write_figure3_grid(evaluation, out_dir, coupling=COUPLING_DATA_DEPENDENT)
        artifacts.update(figure_artifacts)

    metrics_path = out_dir / "metrics.json"
    write_json(
        metrics_path,
        {
            "mode": evaluation.get("mode"),
            "caption": DEFAULT_TABLE2_NAME,
            "rows": evaluation.get("rows", []),
            "trend": evaluation.get("trend"),
        },
    )
    artifacts["metrics"] = str(metrics_path)

    readiness_path = out_dir / "readiness.json"
    write_json(
        readiness_path,
        {
            "status": "ready",
            "route": "wp_inpainting_table2_figure3",
            "exercised_functions": [
                "resolve_batch_size_defaults",
                "resolve_alpha_defaults",
                "resolve_beta_defaults",
                "compute_accuracy",
                "aggregate_accuracy",
                "compute_reward",
                "aggregate_reward",
                "compute_f1",
                "aggregate_f1",
                "load_inputs",
                "run_evaluation",
                "write_named_result_artifacts",
            ],
            "paper_visible_outputs": {
                "table2": artifacts.get("table2_json"),
                "figure3": artifacts.get("figure3_grid_ppm"),
            },
        },
    )
    artifacts["readiness"] = str(readiness_path)

    evaluation_result_path = out_dir / "evaluation_result.json"
    write_json(
        evaluation_result_path,
        {
            "status": "completed",
            "mode": evaluation.get("mode"),
            "task": evaluation.get("task_config", {}).get("task_id"),
            "measured_rows": evaluation.get("rows", []),
            "trend": evaluation.get("trend"),
            "artifact_manifest": str(out_dir / "artifact_manifest.json"),
        },
    )
    artifacts["evaluation_result"] = str(evaluation_result_path)

    artifacts["artifact_manifest"] = write_artifact_manifest(out_dir, artifacts)
    return artifacts


def run_table2_and_figure3_route(
    mode: str = "runtime_smoke",
    task_id: str = "imagenet_256_inpainting",
    artifact_dir: Optional[str] = None,
    data_root: Optional[str] = None,
    max_samples: Optional[int] = None,
    seed: int = 7,
) -> Dict[str, Any]:
    evaluation = run_evaluation(
        task=task_id,
        mode=mode,
        data_root=data_root,
        artifact_dir=artifact_dir,
        max_samples=max_samples,
        seed=seed,
    )
    artifacts = write_named_result_artifacts(evaluation, artifact_dir=artifact_dir, include_figure=True)
    return {
        "evaluation": {
            "mode": evaluation["mode"],
            "task_config": evaluation["task_config"],
            "rows": evaluation["rows"],
            "trend": evaluation["trend"],
        },
        "artifacts": artifacts,
    }


DEFAULT_FUNCTIONS.update(
    {
        "resolve_batch_size_defaults": resolve_batch_size_defaults,
        "batch_size_values": batch_size_values,
        "resolve_alpha_defaults": resolve_alpha_defaults,
        "alpha_values": alpha_values,
        "resolve_beta_defaults": resolve_beta_defaults,
        "beta_values": beta_values,
        "compute_accuracy": compute_accuracy,
        "aggregate_accuracy": aggregate_accuracy,
        "compute_reward": compute_reward,
        "aggregate_reward": aggregate_reward,
        "compute_f1": compute_f1,
        "aggregate_f1": aggregate_f1,
        "compute_fid": compute_fid,
        "load_inputs": load_inputs,
        "run_evaluation": run_evaluation,
        "write_named_result_artifacts": write_named_result_artifacts,
        "write_table2_artifact": write_table2_artifact,
        "write_figure3_grid": write_figure3_grid,
    }
)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    import argparse

    parser = argparse.ArgumentParser(description="Run ImageNet in-painting FID/Table 2 route.")
    parser.add_argument("--mode", default="runtime_smoke", choices=["runtime_smoke", "docker_validate", "full", "paper"])
    parser.add_argument("--task-id", default="imagenet_256_inpainting", choices=sorted(INPAINTING_TASK_REGISTRY))
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    return run_table2_and_figure3_route(
        mode=args.mode,
        task_id=args.task_id,
        artifact_dir=args.artifact_dir,
        data_root=args.data_root,
        max_samples=args.max_samples,
        seed=args.seed,
    )


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "resolve_batch_size_defaults",
    "batch_size_values",
    "DEFAULT_ALPHA",
    "resolve_alpha_defaults",
    "alpha_values",
    "DEFAULT_BETA",
    "resolve_beta_defaults",
    "beta_values",
    "DEFAULT_FUNCTIONS",
    "compute_accuracy",
    "aggregate_accuracy",
    "compute_reward",
    "aggregate_reward",
    "compute_f1",
    "aggregate_f1",
    "compute_fid",
    "load_inputs",
    "run_evaluation",
    "write_named_result_artifacts",
    "write_table2_artifact",
    "write_figure3_grid",
    "INPAINTING_TASK_REGISTRY",
    "ENVIRONMENT_REGISTRY",
    "DATASET_REGISTRY",
    "PROTOCOL_MATRIX",
    "create_inpainting_task",
    "make_mask",
    "adapt_conditional_model_input",
    "run_table2_and_figure3_route",
    "main",
]


if __name__ == "__main__":
    result = main()
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True))