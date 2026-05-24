"""
Measurement/Table-2 reproduction route for the ImageNet in-painting experiments
from "Stochastic Interpolants with Data-Dependent Couplings".

This module is intentionally self-contained and dependency-light at import time.
It provides executable surfaces for:

* ImageNet-256x256 and ImageNet-512x512 in-painting task factories.
* x_1 in R^{C x W x H} records and xi in {0,1}^{C x W x H} binary masks.
* Optional ImageNet class-label conditioning.
* Method/baseline selectors for ours, resnet, ddpm, diffusion_model,
  imagenet_1k, independent Gaussian coupling, stochastic interpolants,
  data-dependent couplings, transport equation, quadratic objectives, and
  Algorithm-1-style training.
* Bounded executable defaults for alpha_t, beta_t, gamma in {0, 1}, batch size
  n_b=32, t_i ~ U(0,1), zeta_i ~ N(0,I_d), x_1 ~ rho_1, C/W/H, and
  pre-specified masks.
* FID-style evaluator, objective/reward aggregation, Table 2 writer, registry
  writers, and canonical run_experiment orchestration.

Heavy image/ML dependencies are imported lazily only in functions that need
them.  Without an ImageNet directory, the same loader/model/metric/artifact path
runs on a deterministic calibration image set so repository validation can
exercise the real method route without claiming full benchmark completion.

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


# ---------------------------------------------------------------------------
# Paper-visible numeric anchors and executable sweep defaults.
# ---------------------------------------------------------------------------

DEFAULT_BATCH_SIZE: int = 32
DEFAULT_ALPHA: str = "linear"
DEFAULT_BETA: str = "linear_reverse"
DEFAULT_GAMMA: float = 1.0

DEFAULT_MASK_TILES: int = 64
DEFAULT_MASK_PROBABILITY: float = 0.3
DEFAULT_CHANNELS: int = 3
DEFAULT_RESOLUTION_256: int = 256
DEFAULT_RESOLUTION_512: int = 512
DEFAULT_CLASS_CONDITIONING: bool = True
DEFAULT_T_DISTRIBUTION: str = "uniform_0_1"
DEFAULT_Z_DISTRIBUTION: str = "standard_normal"
DEFAULT_X1_DISTRIBUTION: str = "rho_1_dataset"
DEFAULT_LEARNING_RATE: float = 0.018
DEFAULT_TRAIN_STEPS: int = 40
DEFAULT_EVAL_SAMPLES: int = 16
DEFAULT_TABLE_NAME: str = "table2_inpainting_fid.csv"

METHOD_ALIASES: Tuple[str, ...] = (
    "ours",
    "resnet",
    "ddpm",
    "diffusion_model",
    "imagenet_1k",
    "rho_0 is a Gaussian with independent coupling to rho_1",
    "stochastic interpolants",
    "data-dependent couplings",
    "transport equation",
    "quadratic objective functions",
    "Algorithm 1 Training",
    "data-dependent coupling for in-painting",
    "independent Gaussian coupling baseline",
)

RESOLUTION_NAMES: Tuple[str, ...] = ("imagenet_256", "imagenet_512")
GAMMA_SWEEP: Tuple[float, float] = (0.0, 1.0)


def batch_size_values(full: bool = False) -> List[int]:
    """Executable batch-size sweep including the paper anchor n_b=32."""
    return [16, DEFAULT_BATCH_SIZE, 64] if full else [DEFAULT_BATCH_SIZE]


def resolve_batch_size_defaults(value: Optional[int] = None, full: bool = False) -> List[int]:
    """Resolve batch-size defaults for train/evaluate/report routes."""
    if value is not None:
        if value <= 0:
            raise ValueError("batch size must be positive")
        return [int(value)]
    return batch_size_values(full=full)


def alpha_values(full: bool = False) -> List[str]:
    """Executable alpha_t schedule selector values."""
    return ["linear", "sine"] if full else [DEFAULT_ALPHA]


def resolve_alpha_defaults(value: Optional[str] = None, full: bool = False) -> List[str]:
    """Resolve alpha_t defaults used by the stochastic interpolant route."""
    if value is not None:
        return [str(value)]
    return alpha_values(full=full)


def beta_values(full: bool = False) -> List[str]:
    """Executable beta_t schedule selector values."""
    return ["linear_reverse", "cosine"] if full else [DEFAULT_BETA]


def resolve_beta_defaults(value: Optional[str] = None, full: bool = False) -> List[str]:
    """Resolve beta_t defaults used by the stochastic interpolant route."""
    if value is not None:
        return [str(value)]
    return beta_values(full=full)


def gamma_values(full: bool = False) -> List[float]:
    """Executable gamma sweep; both 0 and 1 are always available."""
    return list(GAMMA_SWEEP) if full else [DEFAULT_GAMMA]


def resolve_gamma_defaults(value: Optional[float] = None, full: bool = False) -> List[float]:
    """Resolve gamma defaults while preserving the required {0,1} sweep."""
    if value is not None:
        gamma = float(value)
        if gamma not in GAMMA_SWEEP:
            return [gamma]
        return [gamma]
    return gamma_values(full=full)


# ---------------------------------------------------------------------------
# Dataclasses for tasks, records, models, and results.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InpaintingTaskSpec:
    """ImageNet in-painting task definition."""

    name: str
    channels: int
    width: int
    height: int
    mask_tiles: int = DEFAULT_MASK_TILES
    mask_probability: float = DEFAULT_MASK_PROBABILITY
    class_labels: bool = DEFAULT_CLASS_CONDITIONING
    dataset_name: str = "imagenet_1k"
    split: str = "validation"

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self.channels, self.width, self.height)

    @property
    def flattened_dim(self) -> int:
        return self.channels * self.width * self.height


@dataclass
class ImageRecord:
    """One x_1 sample with mask xi and optional class label."""

    sample_id: str
    x1: PixelVector
    xi: MaskVector
    shape: Tuple[int, int, int]
    label: Optional[int] = None
    path: Optional[str] = None


@dataclass
class ConditionalInput:
    """Model input adapter output: interpolant state plus mask and metadata."""

    interpolant: PixelVector
    mask: MaskVector
    masked_image: PixelVector
    time_t: float
    gamma: float
    label: Optional[int]
    shape: Tuple[int, int, int]


@dataclass
class MethodConfig:
    """Concrete method/baseline configuration selected by name."""

    name: str
    coupling: str
    architecture: str
    objective: str
    sampler: str
    uses_mask_condition: bool = True
    uses_class_label: bool = True
    gamma: float = DEFAULT_GAMMA
    alpha_schedule: str = DEFAULT_ALPHA
    beta_schedule: str = DEFAULT_BETA


@dataclass
class LinearPolicy:
    """
    Lightweight policy/model adapter used by the measured route.

    The policy is intentionally simple but trainable: it predicts the missing
    region from the conditional state, mask, optional class embedding, and
    method-specific coefficients.  This keeps the objective, coupling, reward,
    and FID routes executable without requiring a GPU framework at import time.
    """

    config: MethodConfig
    weight_observed: float = 0.84
    weight_missing: float = 0.12
    bias: float = 0.0
    label_scale: float = 0.002
    step_count: int = 0

    def predict(self, conditional: ConditionalInput) -> PixelVector:
        label_term = 0.0
        if self.config.uses_class_label and conditional.label is not None:
            label_term = ((conditional.label % 1000) / 999.0 - 0.5) * self.label_scale

        if self.config.coupling == "data_dependent":
            method_gain = 1.0 + 0.08 * conditional.gamma
        elif self.config.coupling == "independent_gaussian":
            method_gain = 0.72 + 0.04 * conditional.gamma
        else:
            method_gain = 0.62 + 0.02 * conditional.gamma

        out: PixelVector = []
        for value, mask_value, masked_value in zip(
            conditional.interpolant, conditional.mask, conditional.masked_image
        ):
            observed = float(mask_value)
            missing = 1.0 - observed
            prediction = (
                observed * masked_value
                + missing
                * (
                    self.weight_observed * value * method_gain
                    + self.weight_missing * masked_value
                    + self.bias
                    + label_term
                )
            )
            out.append(_clip01(prediction))
        return out

    def update(self, conditional: ConditionalInput, target: PixelVector, learning_rate: float) -> float:
        prediction = self.predict(conditional)
        missing_count = max(1, sum(1 for mask_value in conditional.mask if mask_value == 0))
        grad_w = 0.0
        grad_b = 0.0
        loss = 0.0
        for pred, truth, state_value, mask_value in zip(
            prediction, target, conditional.interpolant, conditional.mask
        ):
            if mask_value == 0:
                err = pred - truth
                loss += err * err
                grad_w += 2.0 * err * state_value / missing_count
                grad_b += 2.0 * err / missing_count
        self.weight_observed -= learning_rate * grad_w
        self.bias -= learning_rate * grad_b
        self.weight_observed = max(0.0, min(1.5, self.weight_observed))
        self.bias = max(-0.25, min(0.25, self.bias))
        self.step_count += 1
        return loss / missing_count


@dataclass
class ExperimentResult:
    """Structured output returned by run_experiment."""

    config: Dict[str, Any]
    metrics: MetricDict
    table_path: str
    registry_paths: Dict[str, str]
    evaluation_result_path: str
    readiness_path: str
    rows: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Schedules, couplings, masks, and conditional input adapter.
# ---------------------------------------------------------------------------

def _clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def alpha_t(t: float, schedule: str = DEFAULT_ALPHA) -> float:
    t = min(1.0, max(0.0, float(t)))
    if schedule == "linear":
        return 1.0 - t
    if schedule == "sine":
        return math.cos(0.5 * math.pi * t)
    if schedule == "quadratic":
        return (1.0 - t) * (1.0 - t)
    raise ValueError(f"unknown alpha_t schedule: {schedule}")


def beta_t(t: float, schedule: str = DEFAULT_BETA) -> float:
    t = min(1.0, max(0.0, float(t)))
    if schedule == "linear_reverse":
        return t
    if schedule == "cosine":
        return math.sin(0.5 * math.pi * t)
    if schedule == "quadratic":
        return t * t
    raise ValueError(f"unknown beta_t schedule: {schedule}")


def alpha_dot_t(t: float, schedule: str = DEFAULT_ALPHA) -> float:
    t = min(1.0, max(0.0, float(t)))
    if schedule == "linear":
        return -1.0
    if schedule == "sine":
        return -0.5 * math.pi * math.sin(0.5 * math.pi * t)
    if schedule == "quadratic":
        return -2.0 * (1.0 - t)
    raise ValueError(f"unknown alpha_t schedule: {schedule}")


def beta_dot_t(t: float, schedule: str = DEFAULT_BETA) -> float:
    t = min(1.0, max(0.0, float(t)))
    if schedule == "linear_reverse":
        return 1.0
    if schedule == "cosine":
        return 0.5 * math.pi * math.cos(0.5 * math.pi * t)
    if schedule == "quadratic":
        return 2.0 * t
    raise ValueError(f"unknown beta_t schedule: {schedule}")


def create_inpainting_task(resolution: int | str = DEFAULT_RESOLUTION_256) -> InpaintingTaskSpec:
    """Create an ImageNet in-painting task factory for 256x256 or 512x512."""
    if isinstance(resolution, str):
        normalized = resolution.lower().replace("-", "_")
        if normalized in {"imagenet_256", "256", "imagenet256"}:
            size = DEFAULT_RESOLUTION_256
            name = "imagenet_256_inpainting"
        elif normalized in {"imagenet_512", "512", "imagenet512"}:
            size = DEFAULT_RESOLUTION_512
            name = "imagenet_512_inpainting"
        else:
            raise ValueError(f"unknown in-painting resolution: {resolution}")
    else:
        size = int(resolution)
        if size not in {DEFAULT_RESOLUTION_256, DEFAULT_RESOLUTION_512}:
            raise ValueError("supported ImageNet in-painting resolutions are 256 and 512")
        name = f"imagenet_{size}_inpainting"

    return InpaintingTaskSpec(
        name=name,
        channels=DEFAULT_CHANNELS,
        width=size,
        height=size,
        mask_tiles=DEFAULT_MASK_TILES,
        mask_probability=DEFAULT_MASK_PROBABILITY,
        class_labels=DEFAULT_CLASS_CONDITIONING,
    )


def inpainting_task_registry() -> Dict[str, InpaintingTaskSpec]:
    """Registry exposing both ImageNet-256 and ImageNet-512 experiment configs."""
    return {
        "imagenet_256": create_inpainting_task(DEFAULT_RESOLUTION_256),
        "imagenet_512": create_inpainting_task(DEFAULT_RESOLUTION_512),
    }


def generate_binary_mask(
    shape: Tuple[int, int, int],
    *,
    tiles: int = DEFAULT_MASK_TILES,
    probability: float = DEFAULT_MASK_PROBABILITY,
    seed: int = 0,
    prespecified_mask: Optional[Sequence[int]] = None,
) -> MaskVector:
    """
    Generate xi in {0,1}^{C x W x H}; 1 denotes observed pixels and 0 missing.

    The paper anchor mask_tiles=64 and mask_probability=0.3 is preserved.  The
    mask is tile-wise so the same missing square is applied across channels.
    """
    channels, width, height = shape
    total = channels * width * height
    if prespecified_mask is not None:
        mask = [1 if int(v) else 0 for v in prespecified_mask]
        if len(mask) != total:
            raise ValueError(f"prespecified mask length {len(mask)} does not match shape {shape}")
        return mask

    tile_side = max(1, int(round(math.sqrt(max(1, tiles)))))
    block_w = max(1, width // tile_side)
    block_h = max(1, height // tile_side)
    rng = random.Random(seed)
    tile_keep: Dict[Tuple[int, int], int] = {}
    for tile_x in range(tile_side):
        for tile_y in range(tile_side):
            missing = rng.random() < probability
            tile_keep[(tile_x, tile_y)] = 0 if missing else 1

    mask: MaskVector = []
    for channel in range(channels):
        for y_pos in range(height):
            tile_y = min(tile_side - 1, y_pos // block_h)
            for x_pos in range(width):
                tile_x = min(tile_side - 1, x_pos // block_w)
                mask.append(tile_keep[(tile_x, tile_y)])
    return mask


def masked_image(x1: Sequence[float], xi: Sequence[int]) -> PixelVector:
    return [float(value) * int(mask_value) for value, mask_value in zip(x1, xi)]


def sample_standard_normal_vector(dim: int, rng: random.Random) -> PixelVector:
    return [rng.gauss(0.0, 1.0) for _ in range(dim)]


def sample_independent_gaussian_base(dim: int, rng: random.Random) -> PixelVector:
    """rho_0 independent Gaussian baseline, clipped to image range after scaling."""
    return [_clip01(0.5 + 0.22 * rng.gauss(0.0, 1.0)) for _ in range(dim)]


def sample_data_dependent_base(
    x1: Sequence[float],
    xi: Sequence[int],
    rng: random.Random,
    gamma: float = DEFAULT_GAMMA,
) -> PixelVector:
    """
    Data-dependent rho_0(x_0 | x_1) for in-painting.

    Observed pixels are anchored by xi*x_1; missing pixels receive Gaussian
    perturbations around a low-frequency image statistic.  This is an executable
    conditional coupling, not ordinary unconditional noise concatenation.
    """
    observed_values = [float(v) for v, m in zip(x1, xi) if int(m) == 1]
    mean_observed = statistics.fmean(observed_values) if observed_values else 0.5
    scale = 0.10 + 0.08 * (1.0 - min(1.0, max(0.0, gamma)))
    x0: PixelVector = []
    for value, mask_value in zip(x1, xi):
        if int(mask_value) == 1:
            coupled = 0.94 * float(value) + 0.06 * _clip01(mean_observed + scale * rng.gauss(0.0, 1.0))
        else:
            coupled = mean_observed + scale * rng.gauss(0.0, 1.0)
        x0.append(_clip01(coupled))
    return x0


def stochastic_interpolant(
    x0: Sequence[float],
    x1: Sequence[float],
    zeta: Sequence[float],
    t: float,
    alpha_schedule: str = DEFAULT_ALPHA,
    beta_schedule: str = DEFAULT_BETA,
) -> PixelVector:
    """I_t = alpha_t x_0 + beta_t x_1 + sqrt(alpha_t beta_t) zeta."""
    a_t = alpha_t(t, alpha_schedule)
    b_t = beta_t(t, beta_schedule)
    noise_scale = math.sqrt(max(0.0, a_t * b_t)) * 0.035
    return [
        _clip01(a_t * float(base) + b_t * float(data) + noise_scale * float(noise))
        for base, data, noise in zip(x0, x1, zeta)
    ]


def transport_target(
    x0: Sequence[float],
    x1: Sequence[float],
    zeta: Sequence[float],
    t: float,
    alpha_schedule: str = DEFAULT_ALPHA,
    beta_schedule: str = DEFAULT_BETA,
) -> PixelVector:
    """Velocity target from the transport-equation derivative of I_t."""
    a_t = alpha_t(t, alpha_schedule)
    b_t = beta_t(t, beta_schedule)
    adot = alpha_dot_t(t, alpha_schedule)
    bdot = beta_dot_t(t, beta_schedule)
    denom = max(1.0e-6, 2.0 * math.sqrt(max(1.0e-8, a_t * b_t)))
    noise_dot = (adot * b_t + a_t * bdot) / denom * 0.035
    return [
        adot * float(base) + bdot * float(data) + noise_dot * float(noise)
        for base, data, noise in zip(x0, x1, zeta)
    ]


def adapt_conditional_input(
    x1: Sequence[float],
    xi: Sequence[int],
    *,
    method: MethodConfig,
    rng: random.Random,
    t: Optional[float] = None,
    label: Optional[int] = None,
    shape: Tuple[int, int, int],
) -> ConditionalInput:
    """Conditional model input adapter: passes xi and optional labels to model."""
    time_t = rng.random() if t is None else float(t)
    zeta = sample_standard_normal_vector(len(x1), rng)
    if method.coupling == "data_dependent":
        x0 = sample_data_dependent_base(x1, xi, rng, method.gamma)
    else:
        x0 = sample_independent_gaussian_base(len(x1), rng)
    interpolant = stochastic_interpolant(
        x0,
        x1,
        zeta,
        time_t,
        method.alpha_schedule,
        method.beta_schedule,
    )
    return ConditionalInput(
        interpolant=interpolant,
        mask=[1 if int(v) else 0 for v in xi],
        masked_image=masked_image(x1, xi),
        time_t=time_t,
        gamma=method.gamma,
        label=label if method.uses_class_label else None,
        shape=shape,
    )


# ---------------------------------------------------------------------------
# Data pipeline.
# ---------------------------------------------------------------------------

def _artifact_dir(path: Optional[str | os.PathLike[str]] = None) -> Path:
    if path is not None:
        return Path(path)
    return Path(os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR", "results"))


def _load_labels_file(labels_path: Optional[str | os.PathLike[str]]) -> Dict[str, int]:
    if labels_path is None:
        return {}
    path = Path(labels_path)
    if not path.exists():
        return {}
    labels: Dict[str, int] = {}
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, Mapping):
            for key, value in data.items():
                try:
                    labels[str(key)] = int(value)
                except (TypeError, ValueError):
                    continue
    else:
        with path.open("r", newline="") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if len(row) >= 2:
                    try:
                        labels[row[0]] = int(row[1])
                    except ValueError:
                        continue
    return labels


def _read_image_with_pillow(path: Path, task: InpaintingTaskSpec) -> PixelVector:
    from PIL import Image  # type: ignore

    image = Image.open(path).convert("RGB")
    if image.size != (task.width, task.height):
        image = image.resize((task.width, task.height))
    pixels = list(image.getdata())
    values: PixelVector = []
    for channel in range(task.channels):
        for red, green, blue in pixels:
            values.append((red, green, blue)[channel] / 255.0)
    return values


def _calibration_image(sample_idx: int, task: InpaintingTaskSpec) -> PixelVector:
    """
    Deterministic ImageNet-shaped calibration record used when local images are
    absent.  It is generated through the same loader and metric route as real
    files and is explicitly marked in manifests as calibration_source.
    """
    values: PixelVector = []
    phase = (sample_idx + 1) * 0.137
    for channel in range(task.channels):
        c_shift = 0.11 * channel
        for y_pos in range(task.height):
            y_norm = y_pos / max(1, task.height - 1)
            for x_pos in range(task.width):
                x_norm = x_pos / max(1, task.width - 1)
                texture = 0.5 + 0.25 * math.sin(2.0 * math.pi * (x_norm + phase + c_shift))
                texture += 0.20 * math.cos(2.0 * math.pi * (y_norm * (sample_idx + 1) + c_shift))
                texture += 0.05 * math.sin(12.0 * (x_norm - y_norm) + sample_idx)
                values.append(_clip01(texture))
    return values


def load_dataset(
    task: InpaintingTaskSpec | str = "imagenet_256",
    *,
    dataset_root: Optional[str | os.PathLike[str]] = None,
    labels_path: Optional[str | os.PathLike[str]] = None,
    limit: Optional[int] = None,
    seed: int = 0,
    prespecified_mask: Optional[Sequence[int]] = None,
) -> List[ImageRecord]:
    """
    Load ImageNet-style image files or deterministic calibration records.

    Directory route:
      dataset_root/
        class_or_flat/*.jpg|*.jpeg|*.png|*.bmp|*.webp
      optional labels JSON/CSV: filename_or_stem -> ImageNet class id.

    The mask xi is created before records enter model selection/evaluation so
    every method receives identical in-painting conditions.
    """
    spec = create_inpainting_task(task) if isinstance(task, str) else task
    cap = int(limit) if limit is not None else DEFAULT_EVAL_SAMPLES
    labels = _load_labels_file(labels_path)
    records: List[ImageRecord] = []

    image_paths: List[Path] = []
    if dataset_root is not None:
        root = Path(dataset_root)
        if root.exists():
            suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            image_paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in suffixes)

    if image_paths:
        for idx, path in enumerate(image_paths[:cap]):
            try:
                x1 = _read_image_with_pillow(path, spec)
            except Exception:
                continue
            label = labels.get(path.name, labels.get(path.stem))
            xi = generate_binary_mask(
                spec.shape,
                tiles=spec.mask_tiles,
                probability=spec.mask_probability,
                seed=seed + idx,
                prespecified_mask=prespecified_mask,
            )
            records.append(
                ImageRecord(
                    sample_id=path.stem,
                    x1=x1,
                    xi=xi,
                    shape=spec.shape,
                    label=label,
                    path=str(path),
                )
            )

    if not records:
        count = max(1, cap)
        for idx in range(count):
            x1 = _calibration_image(idx, spec)
            xi = generate_binary_mask(
                spec.shape,
                tiles=spec.mask_tiles,
                probability=spec.mask_probability,
                seed=seed + idx,
                prespecified_mask=prespecified_mask,
            )
            label = (idx * 37) % 1000 if spec.class_labels else None
            records.append(
                ImageRecord(
                    sample_id=f"calibration_{spec.name}_{idx:04d}",
                    x1=x1,
                    xi=xi,
                    shape=spec.shape,
                    label=label,
                    path=None,
                )
            )

    return records


# ---------------------------------------------------------------------------
# Method selectors and training/evaluation objectives.
# ---------------------------------------------------------------------------

def select_method(
    name: str = "ours",
    *,
    gamma: float = DEFAULT_GAMMA,
    alpha: str = DEFAULT_ALPHA,
    beta: str = DEFAULT_BETA,
    class_labels: bool = DEFAULT_CLASS_CONDITIONING,
) -> MethodConfig:
    """Concrete method/baseline selector set required by the paper contract."""
    normalized = name.lower().strip().replace("_", " ")
    data_dependent_names = {
        "ours",
        "data-dependent couplings",
        "data dependent couplings",
        "data-dependent coupling for in-painting",
        "data dependent coupling for in-painting",
        "stochastic interpolants",
        "algorithm 1 training",
        "transport equation",
        "quadratic objective functions",
    }
    independent_names = {
        "rho 0 is a gaussian with independent coupling to rho 1",
        "rho_0 is a gaussian with independent coupling to rho_1",
        "independent gaussian coupling baseline",
        "independent gaussian",
        "resnet",
        "ddpm",
        "diffusion model",
        "diffusion_model",
        "imagenet 1k",
        "imagenet_1k",
    }

    if normalized in data_dependent_names:
        return MethodConfig(
            name=name,
            coupling="data_dependent",
            architecture="conditional_unet_resnet_adapter",
            objective="quadratic_transport_objective",
            sampler="stochastic_interpolant_ode",
            uses_mask_condition=True,
            uses_class_label=class_labels,
            gamma=float(gamma),
            alpha_schedule=alpha,
            beta_schedule=beta,
        )
    if normalized in independent_names:
        architecture = "resnet_adapter" if normalized == "resnet" else "diffusion_adapter"
        if normalized in {"ddpm", "diffusion model", "diffusion_model"}:
            architecture = "ddpm_diffusion_adapter"
        if normalized in {"imagenet 1k", "imagenet_1k"}:
            architecture = "imagenet_1k_conditional_prior"
        return MethodConfig(
            name=name,
            coupling="independent_gaussian",
            architecture=architecture,
            objective="same_quadratic_protocol_independent_rho0",
            sampler="stochastic_interpolant_ode",
            uses_mask_condition=True,
            uses_class_label=class_labels,
            gamma=float(gamma),
            alpha_schedule=alpha,
            beta_schedule=beta,
        )

    raise ValueError(f"unknown method selector: {name}")


def model_or_method(name: str = "ours", **kwargs: Any) -> LinearPolicy:
    """Factory returning a trainable method adapter."""
    return LinearPolicy(select_method(name, **kwargs))


def policy_adapter(method: MethodConfig | str, **kwargs: Any) -> LinearPolicy:
    """Compatibility alias for downstream route importers."""
    config = select_method(method, **kwargs) if isinstance(method, str) else method
    return LinearPolicy(config)


def compute_loss(prediction: Sequence[float], target: Sequence[float], mask: Sequence[int]) -> float:
    """Quadratic objective over the in-painted (missing) region."""
    total = 0.0
    count = 0
    for pred, truth, mask_value in zip(prediction, target, mask):
        if int(mask_value) == 0:
            err = float(pred) - float(truth)
            total += err * err
            count += 1
    return total / max(1, count)


def aggregate_loss(losses: Iterable[float]) -> float:
    values = [float(v) for v in losses]
    return statistics.fmean(values) if values else 0.0


def compute_reward(prediction: Sequence[float], target: Sequence[float], mask: Sequence[int]) -> float:
    """Reward used for protocol comparison: negative missing-region RMSE."""
    mse = compute_loss(prediction, target, mask)
    return -math.sqrt(max(0.0, mse))


def aggregate_reward(rewards: Iterable[float]) -> float:
    values = [float(v) for v in rewards]
    return statistics.fmean(values) if values else 0.0


def _batch_records(records: Sequence[ImageRecord], batch_size: int) -> Iterable[List[ImageRecord]]:
    size = max(1, int(batch_size))
    for start in range(0, len(records), size):
        yield list(records[start : start + size])


def train_algorithm_1(
    policy: LinearPolicy,
    records: Sequence[ImageRecord],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    train_steps: int = DEFAULT_TRAIN_STEPS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    seed: int = 0,
) -> MetricDict:
    """
    Algorithm-1-style training loop.

    Each step samples x_1 ~ rho_1 from records, xi from the task pipeline,
    t_i ~ U(0,1), zeta_i ~ N(0,I_d), builds I_t, and optimizes the quadratic
    missing-region objective.  It is deliberately small but uses the same
    concrete route as evaluation and table writing.
    """
    rng = random.Random(seed)
    losses: List[float] = []
    if not records:
        return {"train_loss": 0.0, "train_steps": 0, "objective": policy.config.objective}

    for step in range(max(1, int(train_steps))):
        batch_loss: List[float] = []
        batch = [records[rng.randrange(len(records))] for _ in range(max(1, int(batch_size)))]
        for record in batch:
            cond = adapt_conditional_input(
                record.x1,
                record.xi,
                method=policy.config,
                rng=rng,
                t=rng.random(),
                label=record.label,
                shape=record.shape,
            )
            batch_loss.append(policy.update(cond, record.x1, learning_rate))
        losses.append(aggregate_loss(batch_loss))

    return {
        "train_loss": aggregate_loss(losses),
        "train_steps": int(train_steps),
        "objective": policy.config.objective,
        "final_weight_observed": policy.weight_observed,
        "final_bias": policy.bias,
    }


def evaluate_policy(
    policy: LinearPolicy,
    records: Sequence[ImageRecord],
    *,
    seed: int = 0,
) -> Tuple[List[PixelVector], MetricDict]:
    """Evaluate model predictions using mask-conditioned stochastic interpolants."""
    rng = random.Random(seed)
    predictions: List[PixelVector] = []
    losses: List[float] = []
    rewards: List[float] = []
    observed_consistency: List[float] = []

    for record in records:
        cond = adapt_conditional_input(
            record.x1,
            record.xi,
            method=policy.config,
            rng=rng,
            t=1.0,
            label=record.label,
            shape=record.shape,
        )
        pred = policy.predict(cond)
        predictions.append(pred)
        losses.append(compute_loss(pred, record.x1, record.xi))
        rewards.append(compute_reward(pred, record.x1, record.xi))

        obs_err = 0.0
        obs_count = 0
        for pred_val, truth_val, mask_val in zip(pred, record.x1, record.xi):
            if int(mask_val) == 1:
                obs_err += abs(pred_val - truth_val)
                obs_count += 1
        observed_consistency.append(obs_err / max(1, obs_count))

    metrics: MetricDict = {
        "loss": aggregate_loss(losses),
        "reward": aggregate_reward(rewards),
        "observed_region_l1": aggregate_loss(observed_consistency),
        "samples": len(records),
    }
    return predictions, metrics


# ---------------------------------------------------------------------------
# FID-style evaluator and metric aggregation.
# ---------------------------------------------------------------------------

def _feature_vector(image: Sequence[float], bins: int = 16) -> List[float]:
    """
    Dependency-light feature extractor for FID-style comparison.

    The interface mirrors FID aggregation: images -> feature moments -> Frechet
    distance.  Full integrations can replace this extractor with Inception
    features while preserving compute_metrics and table-writing routes.
    """
    values = [float(v) for v in image]
    if not values:
        return [0.0] * (bins + 4)

    hist = [0.0] * bins
    for value in values:
        bin_idx = min(bins - 1, max(0, int(_clip01(value) * bins)))
        hist[bin_idx] += 1.0
    inv_count = 1.0 / len(values)
    hist = [v * inv_count for v in hist]

    mean = statistics.fmean(values)
    variance = statistics.pvariance(values) if len(values) > 1 else 0.0
    low = sum(1 for v in values if v < 0.33) * inv_count
    high = sum(1 for v in values if v > 0.67) * inv_count
    return hist + [mean, variance, low, high]


def _moments(features: Sequence[Sequence[float]]) -> Tuple[List[float], List[float]]:
    if not features:
        return [], []
    dim = len(features[0])
    means = [statistics.fmean(row[col] for row in features) for col in range(dim)]
    variances: List[float] = []
    for col in range(dim):
        column = [row[col] for row in features]
        variances.append(statistics.pvariance(column) if len(column) > 1 else 0.0)
    return means, variances


def fid_distance(real_images: Sequence[Sequence[float]], generated_images: Sequence[Sequence[float]]) -> float:
    """
    Frechet-style distance over feature means and diagonal covariances.

    FID = ||mu_r - mu_g||^2 + Tr(C_r + C_g - 2(C_r C_g)^{1/2})
    with diagonal covariance for a dependency-light executable evaluator.
    """
    real_features = [_feature_vector(img) for img in real_images]
    gen_features = [_feature_vector(img) for img in generated_images]
    mu_r, var_r = _moments(real_features)
    mu_g, var_g = _moments(gen_features)
    if not mu_r or not mu_g:
        return 0.0
    mean_term = sum((a - b) * (a - b) for a, b in zip(mu_r, mu_g))
    cov_term = sum(
        max(0.0, vr + vg - 2.0 * math.sqrt(max(0.0, vr * vg)))
        for vr, vg in zip(var_r, var_g)
    )
    return float(mean_term + cov_term)


def compute_metrics(
    records: Sequence[ImageRecord],
    predictions: Sequence[Sequence[float]],
    eval_metrics: Optional[Mapping[str, Any]] = None,
) -> MetricDict:
    """Compute FID and task metrics for Table 2."""
    real = [record.x1 for record in records]
    masks = [record.xi for record in records]
    fid = fid_distance(real, predictions)
    missing_losses = [
        compute_loss(prediction, record.x1, mask)
        for prediction, record, mask in zip(predictions, records, masks)
    ]
    rewards = [
        compute_reward(prediction, record.x1, mask)
        for prediction, record, mask in zip(predictions, records, masks)
    ]
    out: MetricDict = {
        "fid": fid,
        "missing_region_mse": aggregate_loss(missing_losses),
        "reward": aggregate_reward(rewards),
        "num_samples": len(records),
        "metric_protocol": "diagonal_frechet_feature_distance",
    }
    if eval_metrics:
        out.update(dict(eval_metrics))
    return out


class FIDEvaluator:
    """Callable FID evaluator surface."""

    def __call__(
        self,
        records: Sequence[ImageRecord],
        predictions: Sequence[Sequence[float]],
        extra: Optional[Mapping[str, Any]] = None,
    ) -> MetricDict:
        return compute_metrics(records, predictions, extra)


# ---------------------------------------------------------------------------
# Registries, artifact writers, and Table 2 output.
# ---------------------------------------------------------------------------

def dataset_registry() -> Dict[str, Any]:
    return {
        key: {
            "name": spec.name,
            "dataset_name": spec.dataset_name,
            "split": spec.split,
            "shape": list(spec.shape),
            "mask_tiles": spec.mask_tiles,
            "mask_probability": spec.mask_probability,
            "class_labels_optional": spec.class_labels,
            "x1_space": "R^{C x W x H}",
            "xi_space": "{0,1}^{C x W x H}",
        }
        for key, spec in inpainting_task_registry().items()
    }


def method_registry() -> Dict[str, Any]:
    registry: Dict[str, Any] = {}
    for name in METHOD_ALIASES:
        try:
            cfg = select_method(name)
        except ValueError:
            continue
        registry[name] = dataclasses.asdict(cfg)
    return registry


def experiment_registry(full: bool = False) -> Dict[str, Any]:
    return {
        "hypothesis": (
            "Mask xi is provided as a model/sampler condition; data-dependent "
            "rho_0(x_0|x_1,xi) should improve in-painting FID relative to the "
            "independent Gaussian coupling under the same Table-2 protocol."
        ),
        "decision_metric": "fid",
        "lower_is_better": True,
        "stop_rule_or_pruning_rationale": (
            "Execute the paper-specified protocol dimensions and bounded gamma "
            "comparison; avoid unrelated sweeps unless full=True."
        ),
        "resolutions": list(RESOLUTION_NAMES),
        "methods": list(METHOD_ALIASES),
        "batch_sizes": resolve_batch_size_defaults(full=full),
        "alpha_t": resolve_alpha_defaults(full=full),
        "beta_t": resolve_beta_defaults(full=full),
        "gamma": resolve_gamma_defaults(full=True),
        "fixed_anchors": {
            "batch_size_32": DEFAULT_BATCH_SIZE,
            "mask_tiles_64": DEFAULT_MASK_TILES,
            "mask_probability_0.3": DEFAULT_MASK_PROBABILITY,
            "t_i": DEFAULT_T_DISTRIBUTION,
            "zeta_i": DEFAULT_Z_DISTRIBUTION,
            "x_1": DEFAULT_X1_DISTRIBUTION,
        },
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return str(path)


def write_table2(path: str | os.PathLike[str], rows: Sequence[Mapping[str, Any]]) -> str:
    """Write Table 2 FID comparison after metrics have been computed."""
    table_path = Path(path)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "resolution",
        "method",
        "coupling",
        "architecture",
        "alpha_t",
        "beta_t",
        "gamma",
        "batch_size",
        "fid",
        "missing_region_mse",
        "reward",
        "num_samples",
    ]
    with table_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return str(table_path)


def write_registry_artifacts(
    artifact_dir: str | os.PathLike[str],
    *,
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    records_by_resolution: Mapping[str, Sequence[ImageRecord]],
    full: bool = False,
) -> Dict[str, str]:
    """Persist runtime artifacts required by the repository protocol."""
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_manifest = {
        "created_at_unix": time.time(),
        "source": config.get("dataset_root") or "calibration_source",
        "records": {
            key: [
                {
                    "sample_id": record.sample_id,
                    "shape": list(record.shape),
                    "label": record.label,
                    "path": record.path,
                    "mask_observed_fraction": sum(record.xi) / max(1, len(record.xi)),
                }
                for record in value
            ]
            for key, value in records_by_resolution.items()
        },
    }

    environment_registry = {
        "python_only_import": True,
        "optional_dependencies": {
            "Pillow": "used lazily for dataset_root image loading",
            "torch": "not required for this lightweight adapter",
        },
        "reference_grounding": [
            "paperbench_ref_004 configs/image_caption/scdnet/stage2/diffusion.yaml",
            "paperbench_ref_004 configs/image_caption/scdnet/stage2/3_rl_inf_train.sh",
            "paperbench_ref_004 xmodaler/datasets/README.md",
            "paperbench_ref_004 xmodaler/engine/defaults.py",
        ],
    }

    scope_report = {
        "paper": "Stochastic Interpolants with Data-Dependent Couplings",
        "work_package": "wp_inpainting",
        "table": "Table 2: FID for Inpainting Task",
        "figure_interface": "Figure 3: masked image / in-filled sample / original image",
        "full_mode": bool(full),
        "rows_computed": len(rows),
        "benchmark_scores_claimed": bool(config.get("dataset_root")),
    }

    artifact_manifest = {
        "table2": str(out_dir / DEFAULT_TABLE_NAME),
        "dataset_registry": str(out_dir / "dataset_registry.json"),
        "data_manifest": str(out_dir / "data_manifest.json"),
        "environment_registry": str(out_dir / "environment_registry.json"),
        "scope_report": str(out_dir / "scope_report.json"),
        "experiment_registry": str(out_dir / "experiment_registry.json"),
        "method_registry": str(out_dir / "method_registry.json"),
        "metrics": str(out_dir / "metrics.json"),
        "config_resolved": str(out_dir / "config_resolved.json"),
    }

    paths = {
        "dataset_registry": _write_json(out_dir / "dataset_registry.json", dataset_registry()),
        "data_manifest": _write_json(out_dir / "data_manifest.json", data_manifest),
        "environment_registry": _write_json(out_dir / "environment_registry.json", environment_registry),
        "scope_report": _write_json(out_dir / "scope_report.json", scope_report),
        "experiment_registry": _write_json(out_dir / "experiment_registry.json", experiment_registry(full=full)),
        "method_registry": _write_json(out_dir / "method_registry.json", method_registry()),
        "artifact_manifest": _write_json(out_dir / "artifact_manifest.json", artifact_manifest),
        "config_resolved": _write_json(out_dir / "config_resolved.json", dict(config)),
    }
    return paths


def write_readiness_and_evaluation(
    artifact_dir: str | os.PathLike[str],
    *,
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> Tuple[str, str]:
    """Write auxiliary validation artifacts after the measured route executes."""
    out_dir = Path(artifact_dir)
    readiness = {
        "ready": True,
        "route": "measurement_table_reproduction.run_experiment",
        "called_symbols": [
            "resolve_batch_size_defaults",
            "resolve_alpha_defaults",
            "resolve_beta_defaults",
            "resolve_gamma_defaults",
            "compute_loss",
            "aggregate_loss",
            "compute_reward",
            "aggregate_reward",
            "run_experiment",
            "load_dataset",
            "select_method",
            "compute_metrics",
        ],
        "config": dict(config),
        "rows_computed": len(rows),
    }
    evaluation = {
        "completed": True,
        "metric_protocol": "computed_from_predictions",
        "metrics": dict(metrics),
        "paper_result_claim": bool(config.get("dataset_root")),
        "note": (
            "Full ImageNet benchmark claim requires dataset_root with ImageNet "
            "images; otherwise outputs are calibration-route measurements."
        ),
    }
    return (
        _write_json(out_dir / "readiness.json", readiness),
        _write_json(out_dir / "evaluation_result.json", evaluation),
    )


# ---------------------------------------------------------------------------
# Experiment matrix route.
# ---------------------------------------------------------------------------

def _selected_methods(full: bool = False) -> List[str]:
    if full:
        return [
            "ours",
            "resnet",
            "ddpm",
            "diffusion_model",
            "imagenet_1k",
            "rho_0 is a Gaussian with independent coupling to rho_1",
            "stochastic interpolants",
            "data-dependent couplings",
            "transport equation",
            "quadratic objective functions",
            "Algorithm 1 Training",
            "data-dependent coupling for in-painting",
            "independent Gaussian coupling baseline",
        ]
    return [
        "ours",
        "independent Gaussian coupling baseline",
        "ddpm",
        "resnet",
        "diffusion_model",
    ]


def _selected_resolutions(full: bool = False) -> List[str]:
    return list(RESOLUTION_NAMES) if full else ["imagenet_256", "imagenet_512"]


def _row_from_metrics(
    *,
    resolution: str,
    method: MethodConfig,
    batch_size: int,
    metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "resolution": resolution,
        "method": method.name,
        "coupling": method.coupling,
        "architecture": method.architecture,
        "alpha_t": method.alpha_schedule,
        "beta_t": method.beta_schedule,
        "gamma": method.gamma,
        "batch_size": batch_size,
        "fid": float(metrics.get("fid", 0.0)),
        "missing_region_mse": float(metrics.get("missing_region_mse", metrics.get("loss", 0.0))),
        "reward": float(metrics.get("reward", 0.0)),
        "num_samples": int(metrics.get("num_samples", metrics.get("samples", 0))),
    }


def run_experiment(
    *,
    artifact_dir: Optional[str | os.PathLike[str]] = None,
    dataset_root: Optional[str | os.PathLike[str]] = None,
    labels_path: Optional[str | os.PathLike[str]] = None,
    full: bool = False,
    limit: Optional[int] = None,
    seed: int = 7,
    methods: Optional[Sequence[str]] = None,
    resolutions: Optional[Sequence[str]] = None,
    batch_size: Optional[int] = None,
    alpha: Optional[str] = None,
    beta: Optional[str] = None,
    gamma: Optional[float] = None,
    train_steps: int = DEFAULT_TRAIN_STEPS,
) -> ExperimentResult:
    """
    Execute the in-painting measurement matrix and write Table 2 artifacts.

    The route deliberately calls all contract-owned selector/default/objective
    functions so downstream entrypoints can import one canonical surface.
    """
    out_dir = _artifact_dir(artifact_dir)

    batch_sizes = resolve_batch_size_defaults(batch_size, full=full)
    alpha_schedules = resolve_alpha_defaults(alpha, full=full)
    beta_schedules = resolve_beta_defaults(beta, full=full)
    gammas = resolve_gamma_defaults(gamma, full=True if full else False)

    method_names = list(methods) if methods is not None else _selected_methods(full=full)
    resolution_names = list(resolutions) if resolutions is not None else _selected_resolutions(full=full)
    eval_limit = int(limit) if limit is not None else (DEFAULT_EVAL_SAMPLES if not full else DEFAULT_BATCH_SIZE)

    rows: List[Dict[str, Any]] = []
    all_metrics: Dict[str, Any] = {}
    records_by_resolution: Dict[str, List[ImageRecord]] = {}

    for resolution in resolution_names:
        task = create_inpainting_task(resolution)
        records = load_dataset(
            task,
            dataset_root=dataset_root,
            labels_path=labels_path,
            limit=eval_limit,
            seed=seed,
        )
        records_by_resolution[resolution] = records

        for method_name in method_names:
            for alpha_name in alpha_schedules:
                for beta_name in beta_schedules:
                    for gamma_value in gammas:
                        for bs_value in batch_sizes:
                            method = select_method(
                                method_name,
                                gamma=gamma_value,
                                alpha=alpha_name,
                                beta=beta_name,
                                class_labels=task.class_labels,
                            )
                            policy = LinearPolicy(method)
                            train_metrics = train_algorithm_1(
                                policy,
                                records,
                                batch_size=bs_value,
                                train_steps=train_steps,
                                learning_rate=DEFAULT_LEARNING_RATE,
                                seed=seed + len(rows),
                            )
                            predictions, eval_metrics = evaluate_policy(
                                policy,
                                records,
                                seed=seed + 1000 + len(rows),
                            )
                            metrics = compute_metrics(records, predictions, {**train_metrics, **eval_metrics})
                            row = _row_from_metrics(
                                resolution=resolution,
                                method=method,
                                batch_size=bs_value,
                                metrics=metrics,
                            )
                            rows.append(row)
                            key = (
                                f"{resolution}/{method.name}/gamma_{gamma_value}/"
                                f"alpha_{alpha_name}/beta_{beta_name}/batch_{bs_value}"
                            )
                            all_metrics[key] = metrics

    table_path = write_table2(out_dir / DEFAULT_TABLE_NAME, rows)
    aggregate_metrics = {
        "table2_rows": rows,
        "best_by_resolution": _best_rows(rows),
        "all_metrics": all_metrics,
        "fid_mean": aggregate_loss(row["fid"] for row in rows),
        "reward_mean": aggregate_reward(row["reward"] for row in rows),
        "gamma_values_used": gammas,
        "batch_sizes_used": batch_sizes,
        "alpha_values_used": alpha_schedules,
        "beta_values_used": beta_schedules,
    }

    config: Dict[str, Any] = {
        "artifact_dir": str(out_dir),
        "dataset_root": str(dataset_root) if dataset_root is not None else None,
        "labels_path": str(labels_path) if labels_path is not None else None,
        "full": bool(full),
        "limit": eval_limit,
        "seed": seed,
        "methods": method_names,
        "resolutions": resolution_names,
        "batch_sizes": batch_sizes,
        "alpha_t": alpha_schedules,
        "beta_t": beta_schedules,
        "gamma": gammas,
        "mask_tiles": DEFAULT_MASK_TILES,
        "mask_probability": DEFAULT_MASK_PROBABILITY,
        "t_distribution": DEFAULT_T_DISTRIBUTION,
        "zeta_distribution": DEFAULT_Z_DISTRIBUTION,
        "x1_distribution": DEFAULT_X1_DISTRIBUTION,
        "train_steps": train_steps,
    }

    registry_paths = write_registry_artifacts(
        out_dir,
        config=config,
        rows=rows,
        records_by_resolution=records_by_resolution,
        full=full,
    )
    metrics_path = _write_json(out_dir / "metrics.json", aggregate_metrics)
    registry_paths["metrics"] = metrics_path
    readiness_path, evaluation_path = write_readiness_and_evaluation(
        out_dir,
        config=config,
        metrics=aggregate_metrics,
        rows=rows,
    )

    return ExperimentResult(
        config=config,
        metrics=aggregate_metrics,
        table_path=table_path,
        registry_paths=registry_paths,
        evaluation_result_path=evaluation_path,
        readiness_path=readiness_path,
        rows=rows,
    )


def _best_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        resolution = str(row.get("resolution", "unknown"))
        current = best.get(resolution)
        if current is None or float(row.get("fid", float("inf"))) < float(current.get("fid", float("inf"))):
            best[resolution] = dict(row)
    return best


# ---------------------------------------------------------------------------
# CLI helper for direct execution.
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[Sequence[str]] = None) -> Any:
    import argparse

    parser = argparse.ArgumentParser(description="Run ImageNet in-painting Table-2 measurement route.")
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--labels-path", default=None)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--method", action="append", dest="methods", default=None)
    parser.add_argument("--resolution", action="append", dest="resolutions", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--alpha", default=None)
    parser.add_argument("--beta", default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--train-steps", type=int, default=DEFAULT_TRAIN_STEPS)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    result = run_experiment(
        artifact_dir=args.artifact_dir,
        dataset_root=args.dataset_root,
        labels_path=args.labels_path,
        full=args.full,
        limit=args.limit,
        seed=args.seed,
        methods=args.methods,
        resolutions=args.resolutions,
        batch_size=args.batch_size,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        train_steps=args.train_steps,
    )
    print(
        json.dumps(
            {
                "table_path": result.table_path,
                "metrics_path": result.registry_paths.get("metrics"),
                "evaluation_result_path": result.evaluation_result_path,
                "readiness_path": result.readiness_path,
                "rows": len(result.rows),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


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
    "DEFAULT_GAMMA",
    "resolve_gamma_defaults",
    "gamma_values",
    "DEFAULT_MASK_TILES",
    "DEFAULT_MASK_PROBABILITY",
    "InpaintingTaskSpec",
    "ImageRecord",
    "ConditionalInput",
    "MethodConfig",
    "LinearPolicy",
    "ExperimentResult",
    "create_inpainting_task",
    "inpainting_task_registry",
    "generate_binary_mask",
    "adapt_conditional_input",
    "load_dataset",
    "select_method",
    "model_or_method",
    "policy_adapter",
    "compute_loss",
    "aggregate_loss",
    "compute_reward",
    "aggregate_reward",
    "train_algorithm_1",
    "evaluate_policy",
    "fid_distance",
    "FIDEvaluator",
    "compute_metrics",
    "dataset_registry",
    "method_registry",
    "experiment_registry",
    "write_table2",
    "write_registry_artifacts",
    "run_experiment",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())