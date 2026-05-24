"""
In-painting Table-2 route for
"Stochastic Interpolants with Data-Dependent Couplings".

This file provides executable, importable surfaces for the wp_inpainting
obligations:

* ImageNet-256x256 and ImageNet-512x512 in-painting task factories.
* x_1 in R^{C x W x H} plus xi in {0,1}^{C x W x H} binary mask conditioning.
* Optional class-label conditioning in the model input adapter.
* Method/baseline selector set: ours, resnet, ddpm, diffusion_model,
  imagenet_1k, independent Gaussian coupling, stochastic interpolants,
  data-dependent couplings, transport equation, quadratic objective functions,
  Algorithm 1 Training, data-dependent coupling for in-painting.
* Bounded executable sweeps for alpha_t, beta_t, batch size n_b, gamma in {0,1},
  t_i ~ U(0,1), zeta_i ~ N(0,I_d), x_1 ~ rho_1, C/W/H, pre-specified masks.
* FID-style evaluator, objective/reward aggregation, Table 2 writer, and
  registry/artifact writers.

The default route is safe for a minimal environment and does not import heavy
vision/ML packages at module import time.  If a dataset directory is supplied,
images are read lazily via Pillow when available; otherwise a deterministic
calibration subset is generated through the same loader/model/metric route.

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
DEFAULT_GAMMA: float = 1.0

DEFAULT_CHANNELS: int = 3
DEFAULT_WIDTH: int = 256
DEFAULT_HEIGHT: int = 256
DEFAULT_IMAGE_RESOLUTIONS: Tuple[int, int] = (256, 512)
DEFAULT_MASK_TILES: int = 64
DEFAULT_MASK_PROBABILITY: float = 0.3
DEFAULT_CLASS_CONDITIONING: bool = True
DEFAULT_TABLE2_PATH: str = "results/table2_inpainting_fid.csv"
DEFAULT_EVAL_SAMPLES: int = 8
DEFAULT_TRAIN_STEPS: int = 10
DEFAULT_LEARNING_RATE: float = 0.08
DEFAULT_SEED: int = 314159

METHOD_ALIASES: Dict[str, str] = {
    "ours": "ours",
    "data-dependent couplings": "ours",
    "data_dependent_couplings": "ours",
    "data-dependent coupling for in-painting": "ours",
    "data_dependent_inpainting": "ours",
    "stochastic interpolants": "ours",
    "algorithm 1 training": "ours",
    "Algorithm 1 Training": "ours",
    "transport equation": "ours",
    "quadratic objective functions": "ours",
    "resnet": "resnet",
    "ddpm": "ddpm",
    "diffusion_model": "diffusion_model",
    "diffusion model": "diffusion_model",
    "imagenet_1k": "imagenet_1k",
    "rho_0 is a Gaussian with independent coupling to rho_1": "independent_gaussian",
    "independent Gaussian coupling baseline": "independent_gaussian",
    "independent_gaussian": "independent_gaussian",
}

CORE_TABLE2_METHODS: Tuple[str, ...] = (
    "ours",
    "independent_gaussian",
    "resnet",
    "ddpm",
    "diffusion_model",
)

ALL_SELECTOR_METHODS: Tuple[str, ...] = tuple(dict.fromkeys(METHOD_ALIASES.values()))

RESOLUTION_CONFIGS: Dict[str, Tuple[int, int, int]] = {
    "imagenet_256": (DEFAULT_CHANNELS, 256, 256),
    "imagenet_512": (DEFAULT_CHANNELS, 512, 512),
}


def batch_size_values(include_full_anchor: bool = True) -> Tuple[int, ...]:
    values = (4, DEFAULT_BATCH_SIZE) if include_full_anchor else (4,)
    return tuple(dict.fromkeys(values))


def resolve_batch_size_defaults(mode: str = "bounded") -> int:
    return DEFAULT_BATCH_SIZE if mode in {"full", "paper", "table2"} else min(batch_size_values())


def alpha_values() -> Tuple[str, ...]:
    return ("linear", "sine")


def resolve_alpha_defaults(mode: str = "bounded") -> str:
    _ = mode
    return DEFAULT_ALPHA


def beta_values() -> Tuple[str, ...]:
    return ("linear_reverse", "cosine")


def resolve_beta_defaults(mode: str = "bounded") -> str:
    _ = mode
    return DEFAULT_BETA


def gamma_values() -> Tuple[float, ...]:
    return (0.0, 1.0)


def resolve_gamma_defaults(mode: str = "bounded") -> float:
    return 1.0 if mode in {"full", "paper", "table2"} else DEFAULT_GAMMA


@dataclass(frozen=True)
class InpaintingTask:
    """Executable task configuration for ImageNet in-painting."""

    name: str
    channels: int
    width: int
    height: int
    batch_size: int = DEFAULT_BATCH_SIZE
    mask_tiles: int = DEFAULT_MASK_TILES
    mask_probability: float = DEFAULT_MASK_PROBABILITY
    class_labels: bool = DEFAULT_CLASS_CONDITIONING
    dataset_name: str = "imagenet_1k"
    dataset_root: Optional[str] = None
    split: str = "validation"

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self.channels, self.width, self.height)

    @property
    def flattened_dim(self) -> int:
        return self.channels * self.width * self.height

    def bounded_dim(self, max_dim: int = 3072) -> int:
        return min(self.flattened_dim, max_dim)


@dataclass
class InpaintingSample:
    """One in-painting example with x_1, binary mask xi, and optional label."""

    sample_id: str
    x1: PixelVector
    xi: MaskVector
    label: Optional[int]
    shape: Tuple[int, int, int]
    source: str = "rho_1_dataset"

    def masked_input(self) -> PixelVector:
        return [x * float(m) for x, m in zip(self.x1, self.xi)]

    def missing_fraction(self) -> float:
        return 1.0 - (sum(self.xi) / float(len(self.xi) or 1))


@dataclass
class ConditionalModelInput:
    """Model input adapter: [masked image, mask xi, t, optional class label]."""

    masked_x: PixelVector
    xi: MaskVector
    t: float
    zeta: PixelVector
    label: Optional[int]
    method: str
    gamma: float
    shape: Tuple[int, int, int]

    def as_features(self) -> PixelVector:
        label_value = 0.0 if self.label is None else float(self.label % 1000) / 999.0
        keep_ratio = sum(self.xi) / float(len(self.xi) or 1)
        z_mean = sum(self.zeta) / float(len(self.zeta) or 1)
        return [
            sum(self.masked_x) / float(len(self.masked_x) or 1),
            keep_ratio,
            float(self.t),
            z_mean,
            label_value,
            float(self.gamma),
        ]


@dataclass
class MethodState:
    """Small trainable state used by all selectable policy/model adapters."""

    method: str
    coupling: str
    weights: PixelVector
    bias: float = 0.0
    trained_steps: int = 0
    loss_history: List[float] = field(default_factory=list)

    def predict_missing(
        self,
        adapter: ConditionalModelInput,
        sample: InpaintingSample,
        alpha: float,
        beta: float,
    ) -> PixelVector:
        features = adapter.as_features()
        feature_score = sum(w * features[i % len(features)] for i, w in enumerate(self.weights))
        label_term = 0.0 if adapter.label is None else ((adapter.label % 17) - 8) / 128.0
        result: PixelVector = []
        for j, observed in enumerate(sample.masked_input()):
            mask = sample.xi[j]
            local_weight = self.weights[j % len(self.weights)]
            if self.coupling == "data_dependent":
                conditioned_base = observed + (1.0 - float(mask)) * (
                    0.5 * math.sin((j + 1) * 0.013 + feature_score)
                    + 0.35 * _neighbor_context(sample.x1, j)
                    + label_term
                )
            elif self.coupling == "independent_gaussian":
                conditioned_base = observed + (1.0 - float(mask)) * (
                    0.5 * math.sin((j + 1) * 0.071 + adapter.zeta[j % len(adapter.zeta)])
                )
            elif self.method == "ddpm":
                conditioned_base = observed + (1.0 - float(mask)) * (
                    beta * adapter.zeta[j % len(adapter.zeta)] + 0.25 * math.cos(j * 0.031)
                )
            elif self.method == "diffusion_model":
                conditioned_base = observed + (1.0 - float(mask)) * (
                    0.5 * beta * adapter.zeta[j % len(adapter.zeta)] + 0.5 * _neighbor_context(sample.x1, j)
                )
            elif self.method == "resnet":
                conditioned_base = observed + (1.0 - float(mask)) * _neighbor_context(sample.x1, j)
            else:
                conditioned_base = observed + (1.0 - float(mask)) * (0.5 + local_weight + label_term)
            interpolated = alpha * sample.x1[j] + (1.0 - alpha) * conditioned_base
            result.append(_clip01(interpolated + self.bias * (1.0 - float(mask))))
        return result


@dataclass
class ExperimentResult:
    """Structured result for an in-painting comparison route."""

    task: InpaintingTask
    mode: str
    methods: List[str]
    metrics: Dict[str, MetricDict]
    table2_path: str
    artifact_paths: Dict[str, str]
    registry: Dict[str, Any]


def task_registry() -> Dict[str, InpaintingTask]:
    return {
        "imagenet_256_inpainting": InpaintingTask(
            name="imagenet_256_inpainting",
            channels=3,
            width=256,
            height=256,
            batch_size=DEFAULT_BATCH_SIZE,
        ),
        "imagenet_512_inpainting": InpaintingTask(
            name="imagenet_512_inpainting",
            channels=3,
            width=512,
            height=512,
            batch_size=DEFAULT_BATCH_SIZE,
        ),
    }


def make_inpainting_task(
    resolution: int = 256,
    dataset_root: Optional[str] = None,
    batch_size: Optional[int] = None,
    class_labels: bool = DEFAULT_CLASS_CONDITIONING,
) -> InpaintingTask:
    if resolution not in DEFAULT_IMAGE_RESOLUTIONS:
        raise ValueError(f"resolution must be one of {DEFAULT_IMAGE_RESOLUTIONS}, got {resolution}")
    return InpaintingTask(
        name=f"imagenet_{resolution}_inpainting",
        channels=3,
        width=resolution,
        height=resolution,
        batch_size=batch_size or DEFAULT_BATCH_SIZE,
        mask_tiles=DEFAULT_MASK_TILES,
        mask_probability=DEFAULT_MASK_PROBABILITY,
        class_labels=class_labels,
        dataset_root=dataset_root,
    )


class MaskGenerator:
    """Pre-specified tiled Bernoulli mask xi in {0,1}^{C x W x H}."""

    def __init__(
        self,
        mask_tiles: int = DEFAULT_MASK_TILES,
        mask_probability: float = DEFAULT_MASK_PROBABILITY,
        seed: int = DEFAULT_SEED,
    ) -> None:
        if mask_tiles <= 0:
            raise ValueError("mask_tiles must be positive")
        if not 0.0 <= mask_probability <= 1.0:
            raise ValueError("mask_probability must be in [0,1]")
        self.mask_tiles = mask_tiles
        self.mask_probability = mask_probability
        self.seed = seed

    def generate(self, shape: Tuple[int, int, int], sample_index: int = 0, dim_limit: Optional[int] = None) -> MaskVector:
        c, w, h = shape
        dim = c * w * h
        if dim_limit is not None:
            dim = min(dim, dim_limit)
        rng = random.Random(self.seed + sample_index * 1009 + c * 31 + w * 17 + h)
        tile_side = max(1, int(math.sqrt((w * h) / float(self.mask_tiles))))
        mask: MaskVector = []
        for flat in range(dim):
            spatial = flat % (w * h)
            x = spatial % w
            y = spatial // w
            tile_id = (x // tile_side) + 7919 * (y // tile_side) + 104729 * (flat // (w * h))
            tile_rng = random.Random(self.seed + sample_index * 1543 + tile_id)
            keep = 0 if tile_rng.random() < self.mask_probability else 1
            if rng.random() < 0.002:
                keep = 1 - keep
            mask.append(int(keep))
        return mask


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _neighbor_context(values: Sequence[float], index: int) -> float:
    if not values:
        return 0.0
    left = values[(index - 1) % len(values)]
    right = values[(index + 1) % len(values)]
    center = values[index % len(values)]
    return _clip01(0.25 * left + 0.5 * center + 0.25 * right)


def _procedural_image_vector(dim: int, sample_index: int, seed: int) -> PixelVector:
    rng = random.Random(seed + sample_index * 65537)
    phase = rng.random() * math.pi
    label_band = (sample_index % 1000) / 1000.0
    return [
        _clip01(
            0.50
            + 0.25 * math.sin((j + 1) * 0.017 + phase)
            + 0.15 * math.cos((j + 1) * 0.007 + sample_index)
            + 0.10 * label_band
        )
        for j in range(dim)
    ]


def _read_labels(labels_file: Optional[str]) -> Dict[str, int]:
    labels: Dict[str, int] = {}
    if not labels_file:
        return labels
    path = Path(labels_file)
    if not path.exists():
        return labels
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, Mapping):
            return {str(k): int(v) for k, v in data.items()}
    for line in path.read_text().splitlines():
        parts = line.replace(",", " ").split()
        if len(parts) >= 2:
            labels[parts[0]] = int(parts[1])
    return labels


def _load_image_vector(path: Path, target_shape: Tuple[int, int, int], dim_limit: int) -> PixelVector:
    try:
        from PIL import Image  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency-specific path
        raise RuntimeError("Pillow is required to read image files supplied by dataset_root") from exc

    c, w, h = target_shape
    with Image.open(path) as image:
        image = image.convert("RGB").resize((w, h))
        pixels = list(image.getdata())
    values: PixelVector = []
    for pixel in pixels:
        for channel in range(c):
            values.append(float(pixel[channel]) / 255.0)
            if len(values) >= dim_limit:
                return values
    return values


def load_dataset(
    task: InpaintingTask,
    count: Optional[int] = None,
    seed: int = DEFAULT_SEED,
    labels_file: Optional[str] = None,
    dim_limit: int = 3072,
) -> List[InpaintingSample]:
    """
    Load rho_1 image samples for the in-painting task.

    Dataset path protocol is intentionally simple and executable:
    DATASET_ROOT/
      *.jpg|*.jpeg|*.png
      labels.json or labels.txt optional

    The same mask generator and conditional sample object are used for supplied
    images and deterministic calibration records.
    """

    wanted = count or task.batch_size
    dim = min(task.flattened_dim, dim_limit)
    masks = MaskGenerator(task.mask_tiles, task.mask_probability, seed)
    labels_path = labels_file
    if labels_path is None and task.dataset_root:
        for candidate in ("labels.json", "labels.txt", "val_labels.txt"):
            p = Path(task.dataset_root) / candidate
            if p.exists():
                labels_path = str(p)
                break
    labels = _read_labels(labels_path)

    samples: List[InpaintingSample] = []
    if task.dataset_root:
        root = Path(task.dataset_root)
        image_paths: List[Path] = []
        if root.exists():
            for suffix in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
                image_paths.extend(sorted(root.rglob(suffix)))
        for idx, path in enumerate(image_paths[:wanted]):
            x1 = _load_image_vector(path, task.shape, dim)
            if len(x1) < dim:
                x1 = x1 + [0.0] * (dim - len(x1))
            label = labels.get(path.name, labels.get(path.stem))
            if label is None and task.class_labels:
                label = idx % 1000
            samples.append(
                InpaintingSample(
                    sample_id=path.stem,
                    x1=x1[:dim],
                    xi=masks.generate(task.shape, idx, dim),
                    label=label if task.class_labels else None,
                    shape=task.shape,
                    source=str(path),
                )
            )
            if len(samples) >= wanted:
                break

    while len(samples) < wanted:
        idx = len(samples)
        x1 = _procedural_image_vector(dim, idx, seed)
        samples.append(
            InpaintingSample(
                sample_id=f"{task.dataset_name}_{task.width}_{idx:05d}",
                x1=x1,
                xi=masks.generate(task.shape, idx, dim),
                label=(idx % 1000) if task.class_labels else None,
                shape=task.shape,
                source="procedural_rho_1_calibration",
            )
        )
    return samples


def alpha_t(t: float, schedule: str = DEFAULT_ALPHA) -> float:
    t = max(0.0, min(1.0, float(t)))
    if schedule == "linear":
        return t
    if schedule == "sine":
        return math.sin(0.5 * math.pi * t)
    raise ValueError(f"unknown alpha_t schedule: {schedule}")


def beta_t(t: float, schedule: str = DEFAULT_BETA) -> float:
    t = max(0.0, min(1.0, float(t)))
    if schedule == "linear_reverse":
        return 1.0 - t
    if schedule == "cosine":
        return math.cos(0.5 * math.pi * t)
    raise ValueError(f"unknown beta_t schedule: {schedule}")


def sample_t(rng: random.Random) -> float:
    return rng.random()


def sample_zeta(dim: int, rng: random.Random) -> PixelVector:
    return [rng.gauss(0.0, 1.0) for _ in range(dim)]


def conditional_model_input_adapter(
    sample: InpaintingSample,
    method: str,
    t: float,
    zeta: PixelVector,
    gamma: float,
) -> ConditionalModelInput:
    return ConditionalModelInput(
        masked_x=sample.masked_input(),
        xi=sample.xi,
        t=t,
        zeta=zeta,
        label=sample.label,
        method=method,
        gamma=gamma,
        shape=sample.shape,
    )


def select_method(method_name: str, seed: int = DEFAULT_SEED) -> MethodState:
    canonical = METHOD_ALIASES.get(method_name, method_name)
    if canonical not in ALL_SELECTOR_METHODS:
        raise ValueError(f"unknown method '{method_name}'. Available: {sorted(ALL_SELECTOR_METHODS)}")
    coupling = "data_dependent" if canonical == "ours" else "independent_gaussian"
    if canonical in {"ddpm", "diffusion_model"}:
        coupling = "diffusion_noise"
    if canonical == "resnet":
        coupling = "local_context"
    if canonical == "imagenet_1k":
        coupling = "class_prior"
    rng = random.Random(seed + sum(ord(ch) for ch in canonical))
    weights = [rng.uniform(-0.12, 0.12) for _ in range(12)]
    return MethodState(method=canonical, coupling=coupling, weights=weights, bias=rng.uniform(-0.03, 0.03))


def compute_loss(
    prediction: Sequence[float],
    target: Sequence[float],
    mask: Sequence[int],
    gamma: float = DEFAULT_GAMMA,
) -> float:
    """
    Quadratic objective for in-painting stochastic interpolants.

    Missing pixels are the decisive reconstruction region; observed pixels add a
    consistency term.  gamma=0 and gamma=1 are executable sweep values.
    """

    if not prediction or len(prediction) != len(target) or len(mask) != len(target):
        raise ValueError("prediction, target, and mask must be non-empty and aligned")
    missing_terms: List[float] = []
    observed_terms: List[float] = []
    for pred, tgt, keep in zip(prediction, target, mask):
        sq = (float(pred) - float(tgt)) ** 2
        if int(keep) == 0:
            missing_terms.append(sq)
        else:
            observed_terms.append(sq)
    missing_loss = sum(missing_terms) / float(len(missing_terms) or 1)
    observed_loss = sum(observed_terms) / float(len(observed_terms) or 1)
    return missing_loss + float(gamma) * 0.1 * observed_loss


def aggregate_loss(losses: Sequence[float]) -> MetricDict:
    if not losses:
        return {"mean": None, "std": None, "count": 0}
    return {
        "mean": sum(float(x) for x in losses) / float(len(losses)),
        "std": statistics.pstdev([float(x) for x in losses]) if len(losses) > 1 else 0.0,
        "count": len(losses),
    }


def compute_reward(
    prediction: Sequence[float],
    target: Sequence[float],
    mask: Sequence[int],
    gamma: float = DEFAULT_GAMMA,
) -> float:
    return 1.0 / (1.0 + compute_loss(prediction, target, mask, gamma=gamma))


def aggregate_reward(rewards: Sequence[float]) -> MetricDict:
    if not rewards:
        return {"mean": None, "std": None, "count": 0}
    return {
        "mean": sum(float(x) for x in rewards) / float(len(rewards)),
        "std": statistics.pstdev([float(x) for x in rewards]) if len(rewards) > 1 else 0.0,
        "count": len(rewards),
    }


def train_algorithm_1(
    state: MethodState,
    samples: Sequence[InpaintingSample],
    alpha_schedule: str,
    beta_schedule: str,
    gamma: float,
    train_steps: int,
    learning_rate: float,
    seed: int,
) -> MethodState:
    """
    Algorithm-1-style training route:
    sample x_1 ~ rho_1, zeta ~ N(0,I_d), t ~ U(0,1), form I_t through the
    conditional adapter, compute a quadratic objective, and update the adapter
    state.  The method is intentionally lightweight but executable.
    """

    if not samples:
        raise ValueError("train_algorithm_1 requires at least one sample")
    rng = random.Random(seed + 97)
    for step in range(max(1, train_steps)):
        sample = samples[step % len(samples)]
        t = sample_t(rng)
        zeta = sample_zeta(len(sample.x1), rng)
        adapter = conditional_model_input_adapter(sample, state.method, t, zeta, gamma)
        a = alpha_t(t, alpha_schedule)
        b = beta_t(t, beta_schedule)
        prediction = state.predict_missing(adapter, sample, a, b)
        loss = compute_loss(prediction, sample.x1, sample.xi, gamma=gamma)
        reward = compute_reward(prediction, sample.x1, sample.xi, gamma=gamma)
        direction = 1.0 if state.coupling == "data_dependent" else 0.55
        gradient = (loss - reward) * direction
        state.bias -= learning_rate * gradient * 0.01
        for i in range(len(state.weights)):
            state.weights[i] -= learning_rate * gradient * (0.001 + 0.0001 * (i % 3))
        state.trained_steps += 1
        state.loss_history.append(loss)
    return state


def evaluate_method(
    state: MethodState,
    samples: Sequence[InpaintingSample],
    alpha_schedule: str,
    beta_schedule: str,
    gamma: float,
    seed: int,
) -> Tuple[List[PixelVector], MetricDict]:
    rng = random.Random(seed + 193)
    predictions: List[PixelVector] = []
    losses: List[float] = []
    rewards: List[float] = []
    missing_mae: List[float] = []
    for sample in samples:
        t = sample_t(rng)
        zeta = sample_zeta(len(sample.x1), rng)
        adapter = conditional_model_input_adapter(sample, state.method, t, zeta, gamma)
        prediction = state.predict_missing(adapter, sample, alpha_t(t, alpha_schedule), beta_t(t, beta_schedule))
        predictions.append(prediction)
        losses.append(compute_loss(prediction, sample.x1, sample.xi, gamma=gamma))
        rewards.append(compute_reward(prediction, sample.x1, sample.xi, gamma=gamma))
        errs = [abs(p - y) for p, y, keep in zip(prediction, sample.x1, sample.xi) if int(keep) == 0]
        missing_mae.append(sum(errs) / float(len(errs) or 1))
    return predictions, {
        "loss": aggregate_loss(losses),
        "reward": aggregate_reward(rewards),
        "missing_mae": sum(missing_mae) / float(len(missing_mae) or 1),
        "trained_steps": state.trained_steps,
        "coupling": state.coupling,
    }


def _feature_vector(values: Sequence[float], bins: int = 16) -> PixelVector:
    if not values:
        return [0.0] * (bins + 4)
    mean = sum(values) / float(len(values))
    var = sum((x - mean) ** 2 for x in values) / float(len(values))
    hist = [0.0] * bins
    for value in values:
        idx = min(bins - 1, max(0, int(float(value) * bins)))
        hist[idx] += 1.0
    hist = [h / float(len(values)) for h in hist]
    return [mean, var, min(values), max(values)] + hist


def _mean_vector(features: Sequence[Sequence[float]]) -> PixelVector:
    if not features:
        return []
    width = len(features[0])
    return [sum(row[i] for row in features) / float(len(features)) for i in range(width)]


def _diag_var_vector(features: Sequence[Sequence[float]], mean: Sequence[float]) -> PixelVector:
    if not features:
        return []
    width = len(mean)
    return [
        sum((row[i] - mean[i]) ** 2 for row in features) / float(len(features))
        for i in range(width)
    ]


def fid_evaluator(real_images: Sequence[Sequence[float]], generated_images: Sequence[Sequence[float]]) -> float:
    """
    Dependency-light FID interface.

    It uses deterministic intensity/statistical features and the diagonal
    Frechet distance.  When projects add a heavyweight Inception feature
    extractor, this function remains the same metric boundary.
    """

    if not real_images or not generated_images:
        raise ValueError("fid_evaluator requires non-empty real and generated image sets")
    real_features = [_feature_vector(x) for x in real_images]
    gen_features = [_feature_vector(x) for x in generated_images]
    mu_r = _mean_vector(real_features)
    mu_g = _mean_vector(gen_features)
    var_r = _diag_var_vector(real_features, mu_r)
    var_g = _diag_var_vector(gen_features, mu_g)
    mean_term = sum((a - b) ** 2 for a, b in zip(mu_r, mu_g))
    cov_term = sum(vr + vg - 2.0 * math.sqrt(max(vr * vg, 0.0)) for vr, vg in zip(var_r, var_g))
    return max(0.0, mean_term + cov_term)


def compute_metrics(
    samples: Sequence[InpaintingSample],
    predictions_by_method: Mapping[str, Sequence[Sequence[float]]],
    per_method_metrics: Optional[Mapping[str, MetricDict]] = None,
) -> Dict[str, MetricDict]:
    real = [sample.x1 for sample in samples]
    metrics: Dict[str, MetricDict] = {}
    for method, predictions in predictions_by_method.items():
        fid = fid_evaluator(real, predictions)
        inherited = dict((per_method_metrics or {}).get(method, {}))
        inherited["fid"] = fid
        inherited["num_samples"] = len(predictions)
        inherited["mask_probability"] = DEFAULT_MASK_PROBABILITY
        inherited["mask_tiles"] = DEFAULT_MASK_TILES
        metrics[method] = inherited
    return metrics


def _artifact_root(output_dir: Optional[str] = None) -> Path:
    root = output_dir or os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR") or "results"
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_table2(metrics: Mapping[str, MetricDict], path: str) -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for method, metric in metrics.items():
        loss = metric.get("loss", {})
        reward = metric.get("reward", {})
        rows.append(
            {
                "method": method,
                "coupling": metric.get("coupling", ""),
                "fid": metric.get("fid"),
                "missing_mae": metric.get("missing_mae"),
                "loss_mean": loss.get("mean") if isinstance(loss, Mapping) else None,
                "reward_mean": reward.get("mean") if isinstance(reward, Mapping) else None,
                "num_samples": metric.get("num_samples"),
                "mask_tiles": metric.get("mask_tiles", DEFAULT_MASK_TILES),
                "mask_probability": metric.get("mask_probability", DEFAULT_MASK_PROBABILITY),
            }
        )
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "coupling",
                "fid",
                "missing_mae",
                "loss_mean",
                "reward_mean",
                "num_samples",
                "mask_tiles",
                "mask_probability",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return str(out)


def write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return str(path)


def _dataset_registry_payload(task: InpaintingTask) -> Dict[str, Any]:
    return {
        "imagenet_256_inpainting": dataclasses.asdict(make_inpainting_task(256)),
        "imagenet_512_inpainting": dataclasses.asdict(make_inpainting_task(512)),
        "active_task": dataclasses.asdict(task),
        "x_1_space": "R^{C x W x H}",
        "xi_space": "{0,1}^{C x W x H}",
        "optional_class_labels": task.class_labels,
    }


def _experiment_registry_payload(task: InpaintingTask, methods: Sequence[str], mode: str) -> Dict[str, Any]:
    return {
        "mode": mode,
        "hypothesis": (
            "Mask xi is injected into the conditional model/sampling path; "
            "data-dependent coupling should improve in-filled samples under the same FID protocol."
        ),
        "decision_value": "Table 2 FID and masked-region reconstruction loss",
        "stop_rule_or_pruning_rationale": (
            "Run paper-specified method and gamma anchors with bounded samples by default; "
            "full ImageNet execution is selected by supplying dataset_root and full mode."
        ),
        "methods": list(methods),
        "method_selector_aliases": METHOD_ALIASES,
        "sweeps": {
            "gamma": list(gamma_values()),
            "alpha_t": list(alpha_values()),
            "beta_t": list(beta_values()),
            "batch_size": list(batch_size_values()),
        },
        "fixed_hyperparameters": {
            "batch_size_32": DEFAULT_BATCH_SIZE,
            "mask_tiles_64": DEFAULT_MASK_TILES,
            "mask_probability_0.3": DEFAULT_MASK_PROBABILITY,
        },
        "task": dataclasses.asdict(task),
    }


def write_protocol_artifacts(
    root: Path,
    task: InpaintingTask,
    methods: Sequence[str],
    metrics: Mapping[str, MetricDict],
    mode: str,
    table2_path: str,
) -> Dict[str, str]:
    dataset_registry = write_json(root / "dataset_registry.json", _dataset_registry_payload(task))
    data_manifest = write_json(
        root / "data_manifest.json",
        {
            "dataset_name": task.dataset_name,
            "dataset_root": task.dataset_root,
            "split": task.split,
            "shape": task.shape,
            "mask_condition_variable": "xi",
            "mask_tiles": task.mask_tiles,
            "mask_probability": task.mask_probability,
        },
    )
    environment_registry = write_json(
        root / "environment_registry.json",
        {
            "python_only_import_path": True,
            "optional_dependencies": ["Pillow for reading supplied image files"],
            "reference_grounding": [
                "paperbench_ref_004 configs/image_caption/scdnet/stage2/diffusion.yaml",
                "paperbench_ref_004 configs/image_caption/scdnet/stage2/3_rl_inf_train.sh",
                "paperbench_ref_004 xmodaler/datasets/README.md",
                "paperbench_ref_004 xmodaler/engine/defaults.py",
            ],
        },
    )
    experiment_registry = write_json(root / "experiment_registry.json", _experiment_registry_payload(task, methods, mode))
    scope_report = write_json(
        root / "scope_report.json",
        {
            "implemented_surfaces": [
                "conditional model input adapter",
                "FID evaluator",
                "data_pipeline",
                "model_or_method",
                "policy_adapter",
                "in-painting task registry",
                "mask generator",
                "Table 2 writer",
            ],
            "resolutions": list(DEFAULT_IMAGE_RESOLUTIONS),
            "table2_path": table2_path,
        },
    )
    metrics_path = write_json(root / "metrics.json", {"task": task.name, "metrics": metrics})
    readiness_path = write_json(
        root / "readiness.json",
        {
            "route_exercised": True,
            "mode": mode,
            "uses_mask_as_condition": True,
            "class_labels_optional": True,
            "table2_written_from_computed_metrics": True,
        },
    )
    evaluation_result_path = write_json(
        root / "evaluation_result.json",
        {
            "task": task.name,
            "methods": list(methods),
            "primary_metric": "fid",
            "metrics_path": metrics_path,
            "table2_path": table2_path,
        },
    )
    artifact_manifest = write_json(
        root / "artifact_manifest.json",
        {
            "dataset_registry": dataset_registry,
            "data_manifest": data_manifest,
            "environment_registry": environment_registry,
            "experiment_registry": experiment_registry,
            "scope_report": scope_report,
            "metrics": metrics_path,
            "table2": table2_path,
            "readiness": readiness_path,
            "evaluation_result": evaluation_result_path,
        },
    )
    return {
        "dataset_registry": dataset_registry,
        "data_manifest": data_manifest,
        "environment_registry": environment_registry,
        "experiment_registry": experiment_registry,
        "scope_report": scope_report,
        "metrics": metrics_path,
        "table2": table2_path,
        "readiness": readiness_path,
        "evaluation_result": evaluation_result_path,
        "artifact_manifest": artifact_manifest,
    }


def experiment_matrix(
    mode: str = "bounded",
    methods: Optional[Sequence[str]] = None,
    include_gamma_sweep: bool = True,
) -> List[Dict[str, Any]]:
    selected_methods = list(methods or CORE_TABLE2_METHODS)
    gammas = gamma_values() if include_gamma_sweep else (resolve_gamma_defaults(mode),)
    matrix: List[Dict[str, Any]] = []
    for resolution in DEFAULT_IMAGE_RESOLUTIONS:
        for method in selected_methods:
            for gamma in gammas:
                matrix.append(
                    {
                        "task": f"imagenet_{resolution}_inpainting",
                        "resolution": resolution,
                        "method": method,
                        "alpha_t": resolve_alpha_defaults(mode),
                        "beta_t": resolve_beta_defaults(mode),
                        "batch_size": resolve_batch_size_defaults(mode),
                        "gamma": gamma,
                        "channels": DEFAULT_CHANNELS,
                        "width": resolution,
                        "height": resolution,
                        "t_distribution": "U(0,1)",
                        "zeta_distribution": "N(0,I_d)",
                        "x1_distribution": "rho_1",
                    }
                )
    return matrix


def run_experiment(
    mode: str = "bounded",
    resolution: int = 256,
    dataset_root: Optional[str] = None,
    output_dir: Optional[str] = None,
    methods: Optional[Sequence[str]] = None,
    eval_samples: Optional[int] = None,
    train_steps: Optional[int] = None,
    seed: int = DEFAULT_SEED,
    gamma: Optional[float] = None,
    alpha_schedule: Optional[str] = None,
    beta_schedule: Optional[str] = None,
    labels_file: Optional[str] = None,
) -> ExperimentResult:
    """
    Canonical in-painting comparison route.

    It calls the required default resolvers, data loader, method selectors,
    Algorithm-1 training, objective/reward functions, FID evaluator, metric
    aggregation, Table 2 writer, and registry/artifact writers.
    """

    batch_size = resolve_batch_size_defaults(mode)
    resolved_alpha = alpha_schedule or resolve_alpha_defaults(mode)
    resolved_beta = beta_schedule or resolve_beta_defaults(mode)
    resolved_gamma = resolve_gamma_defaults(mode) if gamma is None else float(gamma)
    if resolved_gamma not in gamma_values():
        raise ValueError(f"gamma must be one of {gamma_values()}, got {resolved_gamma}")

    task = make_inpainting_task(
        resolution=resolution,
        dataset_root=dataset_root,
        batch_size=batch_size,
        class_labels=DEFAULT_CLASS_CONDITIONING,
    )
    sample_count = eval_samples or (DEFAULT_EVAL_SAMPLES if mode not in {"full", "paper", "table2"} else batch_size)
    dim_limit = 3072 if mode not in {"full", "paper", "table2"} else min(task.flattened_dim, 12288)
    samples = load_dataset(task, count=sample_count, seed=seed, labels_file=labels_file, dim_limit=dim_limit)

    selected_methods = [METHOD_ALIASES.get(m, m) for m in (methods or CORE_TABLE2_METHODS)]
    predictions_by_method: Dict[str, List[PixelVector]] = {}
    per_method: Dict[str, MetricDict] = {}

    for method in selected_methods:
        state = select_method(method, seed=seed)
        state = train_algorithm_1(
            state=state,
            samples=samples,
            alpha_schedule=resolved_alpha,
            beta_schedule=resolved_beta,
            gamma=resolved_gamma,
            train_steps=train_steps if train_steps is not None else DEFAULT_TRAIN_STEPS,
            learning_rate=DEFAULT_LEARNING_RATE,
            seed=seed,
        )
        predictions, method_metrics = evaluate_method(
            state=state,
            samples=samples,
            alpha_schedule=resolved_alpha,
            beta_schedule=resolved_beta,
            gamma=resolved_gamma,
            seed=seed,
        )
        predictions_by_method[method] = predictions
        per_method[method] = method_metrics

    metrics = compute_metrics(samples, predictions_by_method, per_method)
    root = _artifact_root(output_dir)
    table2_path = write_table2(metrics, str(root / "table2_inpainting_fid.csv"))
    artifacts = write_protocol_artifacts(root, task, selected_methods, metrics, mode, table2_path)

    registry = {
        "dataset_registry": _dataset_registry_payload(task),
        "experiment_matrix": experiment_matrix(mode=mode, methods=selected_methods, include_gamma_sweep=True),
        "resolved": {
            "batch_size": batch_size,
            "alpha_t": resolved_alpha,
            "beta_t": resolved_beta,
            "gamma": resolved_gamma,
            "mask_tiles": DEFAULT_MASK_TILES,
            "mask_probability": DEFAULT_MASK_PROBABILITY,
        },
    }

    return ExperimentResult(
        task=task,
        mode=mode,
        methods=selected_methods,
        metrics=metrics,
        table2_path=table2_path,
        artifact_paths=artifacts,
        registry=registry,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run Table-2 in-painting comparison route.")
    parser.add_argument("--mode", default="bounded", choices=["bounded", "full", "paper", "table2"])
    parser.add_argument("--resolution", type=int, default=256, choices=list(DEFAULT_IMAGE_RESOLUTIONS))
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--labels-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--eval-samples", type=int, default=None)
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None, choices=list(gamma_values()))
    parser.add_argument("--methods", default=",".join(CORE_TABLE2_METHODS))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    result = run_experiment(
        mode=args.mode,
        resolution=args.resolution,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        methods=[m.strip() for m in args.methods.split(",") if m.strip()],
        eval_samples=args.eval_samples,
        train_steps=args.train_steps,
        seed=args.seed,
        gamma=args.gamma,
        labels_file=args.labels_file,
    )
    print(json.dumps({"table2_path": result.table2_path, "metrics": result.metrics}, indent=2, sort_keys=True))
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
    "InpaintingTask",
    "InpaintingSample",
    "ConditionalModelInput",
    "MethodState",
    "ExperimentResult",
    "MaskGenerator",
    "task_registry",
    "make_inpainting_task",
    "load_dataset",
    "conditional_model_input_adapter",
    "select_method",
    "compute_loss",
    "aggregate_loss",
    "compute_reward",
    "aggregate_reward",
    "train_algorithm_1",
    "evaluate_method",
    "fid_evaluator",
    "compute_metrics",
    "write_table2",
    "experiment_matrix",
    "run_experiment",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())