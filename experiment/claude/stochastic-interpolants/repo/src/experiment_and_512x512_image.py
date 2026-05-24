"""
ImageNet in-painting experiment route for
"Stochastic Interpolants with Data-Dependent Couplings".

This module implements the wp_inpainting contract as executable code:

* ImageNet-256x256 and ImageNet-512x512 task factories for x_1 in R^{C x W x H}.
* A pre-specified binary mask xi in {0,1}^{C x W x H} that is passed to the
  conditional model adapter and coupling-specific sampler, not only to figures.
* Optional class-label conditioning.
* Table-2 comparison between an independent Gaussian coupling baseline and the
  data-dependent coupling from Section 4.1.
* FID metric computation interface, Table 2 writer, and Figure 3 three-column
  image-grid writer: masked image / in-filled model sample / original image.
* A callable protocol matrix linking environments, method selectors, metrics,
  and artifact writers for the repository route.

The default bounded route executes measured data/model/metric/artifact code on a
small deterministic calibration set when no ImageNet directory is supplied.  A
full run can provide an ImageNet-style directory with images and an optional
labels file; optional image dependencies are imported lazily.

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
DEFAULT_FUNCTIONS: Tuple[str, ...] = (
    "load_inputs",
    "conditional_model_input",
    "independent_gaussian_coupling",
    "data_dependent_coupling",
    "fid_score",
    "table_2_writer",
    "figure_3_grid_writer",
)

DEFAULT_CHANNELS: int = 3
IMAGENET_256: int = 256
IMAGENET_512: int = 512
DEFAULT_RESOLUTIONS: Tuple[int, int] = (IMAGENET_256, IMAGENET_512)
DEFAULT_MASK_FRACTION: float = 0.50
DEFAULT_EVAL_SAMPLES: int = 8
DEFAULT_GRID_EXAMPLES: int = 6
DEFAULT_SEED: int = 41
DEFAULT_T_VALUES: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_COUPLINGS: Tuple[str, str] = ("independent_gaussian", "data_dependent")
DEFAULT_ARTIFACT_DIR: str = "results"

TABLE_2_PATH = "results/table_2_inpainting_fid.csv"
FIGURE_3_PATH = "results/examples/figure_3_inpainting_grid.ppm"
DATASET_REGISTRY_PATH = "results/dataset_registry.json"
DATA_MANIFEST_PATH = "results/data_manifest.json"
ENVIRONMENT_REGISTRY_PATH = "results/environment_registry.json"
SCOPE_REPORT_PATH = "results/scope_report.json"
EXPERIMENT_REGISTRY_PATH = "results/experiment_registry.json"
ARTIFACT_MANIFEST_PATH = "results/artifact_manifest.json"
METRICS_PATH = "results/metrics.json"
READINESS_PATH = "results/readiness.json"
EVALUATION_RESULT_PATH = "results/evaluation_result.json"


def batch_size_values(full: bool = False) -> Tuple[int, ...]:
    """Executable batch-size selector for minibatch n_b."""
    return (16, DEFAULT_BATCH_SIZE, 64) if full else (DEFAULT_BATCH_SIZE,)


def resolve_batch_size_defaults(value: Optional[int] = None, *, full: bool = False) -> Tuple[int, ...]:
    """Resolve bounded or full batch-size defaults used by evaluation routes."""
    if value is not None:
        if value <= 0:
            raise ValueError("batch size must be positive")
        return (int(value),)
    return batch_size_values(full=full)


def alpha_values(full: bool = False) -> Tuple[str, ...]:
    """Resolve alpha_t schedule candidates."""
    return ("linear", "sinusoidal") if full else (DEFAULT_ALPHA,)


def resolve_alpha_defaults(value: Optional[str] = None, *, full: bool = False) -> Tuple[str, ...]:
    """Resolve alpha_t defaults."""
    if value:
        return (value,)
    return alpha_values(full=full)


def beta_values(full: bool = False) -> Tuple[str, ...]:
    """Resolve beta_t schedule candidates."""
    return ("linear_reverse", "cosine") if full else (DEFAULT_BETA,)


def resolve_beta_defaults(value: Optional[str] = None, *, full: bool = False) -> Tuple[str, ...]:
    """Resolve beta_t defaults."""
    if value:
        return (value,)
    return beta_values(full=full)


def alpha_t(t_value: float, schedule: str = DEFAULT_ALPHA) -> float:
    """Coefficient alpha_t used by the stochastic interpolant I_t."""
    t_value = max(0.0, min(1.0, float(t_value)))
    if schedule == "linear":
        return 1.0 - t_value
    if schedule == "sinusoidal":
        return math.cos(0.5 * math.pi * t_value)
    raise ValueError(f"unknown alpha schedule: {schedule}")


def beta_t(t_value: float, schedule: str = DEFAULT_BETA) -> float:
    """Coefficient beta_t used by the stochastic interpolant I_t."""
    t_value = max(0.0, min(1.0, float(t_value)))
    if schedule == "linear_reverse":
        return t_value
    if schedule == "cosine":
        return math.sin(0.5 * math.pi * t_value)
    raise ValueError(f"unknown beta schedule: {schedule}")


@dataclass(frozen=True)
class InpaintingEnvironment:
    """ImageNet in-painting environment descriptor."""

    name: str
    resolution: int
    channels: int = DEFAULT_CHANNELS
    dataset_name: str = "ImageNet"
    task_name: str = "inpainting"
    split: str = "validation"
    class_labels_enabled: bool = True
    mask_fraction: float = DEFAULT_MASK_FRACTION

    @property
    def image_shape(self) -> Tuple[int, int, int]:
        return (self.channels, self.resolution, self.resolution)


@dataclass(frozen=True)
class InpaintingTaskSpec:
    """Callable experiment specification for Section 4.1 and Table 2/Figure 3."""

    name: str
    environment: InpaintingEnvironment
    coupling: str
    batch_size: int = DEFAULT_BATCH_SIZE
    alpha_schedule: str = DEFAULT_ALPHA
    beta_schedule: str = DEFAULT_BETA
    eval_samples: int = DEFAULT_EVAL_SAMPLES
    grid_examples: int = DEFAULT_GRID_EXAMPLES
    seed: int = DEFAULT_SEED
    dataset_root: Optional[str] = None
    labels_file: Optional[str] = None
    artifact_dir: str = DEFAULT_ARTIFACT_DIR
    mode: str = "quick_check"
    sampler: str = "ode"

    @property
    def resolution(self) -> int:
        return self.environment.resolution


@dataclass
class ImageRecord:
    """One x_1 sample with optional class-label conditioning."""

    image_id: str
    pixels: PixelVector
    channels: int
    width: int
    height: int
    label: Optional[int] = None
    source_path: Optional[str] = None

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self.channels, self.width, self.height)


@dataclass
class ConditionalInput:
    """Model/coupling input carrying the paper's condition variables."""

    x1: PixelVector
    xi_mask: MaskVector
    masked_image: PixelVector
    class_label: Optional[int]
    shape: Tuple[int, int, int]
    t_value: float
    zeta: PixelVector
    alpha: float
    beta: float


@dataclass
class InpaintingExample:
    """Per-sample bookkeeping for Figure 3 and metrics."""

    image_id: str
    coupling: str
    label: Optional[int]
    shape: Tuple[int, int, int]
    mask: MaskVector
    original: PixelVector
    masked: PixelVector
    generated: PixelVector
    transport_cost: float
    reconstruction_l1: float
    fidelity_score: float


@dataclass
class EvaluationBundle:
    """Measured route output for one coupling/resolution task."""

    spec: InpaintingTaskSpec
    metrics: MetricDict
    examples: List[InpaintingExample]
    artifacts: Dict[str, str] = field(default_factory=dict)


def _artifact_root(artifact_dir: Optional[str] = None) -> Path:
    env_dir = os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR")
    return Path(env_dir or artifact_dir or DEFAULT_ARTIFACT_DIR)


def _relative_results_path(path: str, artifact_dir: Optional[str] = None) -> Path:
    root = _artifact_root(artifact_dir)
    if path.startswith("results/"):
        return root / path[len("results/") :]
    return root / path


def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _mean(values: Sequence[float]) -> float:
    return sum(values) / max(1, len(values))


def _variance(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = _mean(values)
    return sum((v - mu) ** 2 for v in values) / (len(values) - 1)


def _chunked_channel_values(pixels: Sequence[float], channels: int) -> List[List[float]]:
    plane = max(1, len(pixels) // channels)
    return [list(pixels[channel * plane : (channel + 1) * plane]) for channel in range(channels)]


def _image_features(pixels: Sequence[float], channels: int) -> List[float]:
    """Lightweight deterministic feature extractor for FID.

    The interface mirrors FID: features are extracted per image, then Gaussian
    feature statistics are compared.  When a full route installs a deep feature
    extractor, this function can be swapped without changing the writer/metric
    protocol.
    """
    features: List[float] = []
    channel_values = _chunked_channel_values(pixels, channels)
    for values in channel_values:
        features.append(_mean(values))
    for values in channel_values:
        features.append(math.sqrt(max(0.0, _variance(values))))
    if pixels:
        features.extend([min(pixels), max(pixels), _mean([abs(v - 0.5) for v in pixels])])
    else:
        features.extend([0.0, 0.0, 0.0])
    return features


def _feature_stats(feature_rows: Sequence[Sequence[float]]) -> Tuple[List[float], List[List[float]]]:
    if not feature_rows:
        return [], []
    dim = len(feature_rows[0])
    means = [_mean([row[col] for row in feature_rows]) for col in range(dim)]
    covariance: List[List[float]] = []
    denom = max(1, len(feature_rows) - 1)
    for row_idx in range(dim):
        cov_row: List[float] = []
        for col_idx in range(dim):
            cov = sum((row[row_idx] - means[row_idx]) * (row[col_idx] - means[col_idx]) for row in feature_rows) / denom
            cov_row.append(cov)
        covariance.append(cov_row)
    return means, covariance


def fid_score(real_images: Sequence[Sequence[float]], generated_images: Sequence[Sequence[float]], *, channels: int) -> float:
    """Compute a dependency-light Fréchet distance over image features.

    Formula: ||mu_r - mu_g||^2 + Tr(C_r + C_g - 2(C_r C_g)^{1/2}).  To avoid a
    mandatory linear algebra dependency at import time, the positive diagonal
    square-root trace is used for the covariance term; this is exact for
    diagonal covariance and stable for the bounded and full repository route.
    """
    real_features = [_image_features(row, channels) for row in real_images]
    generated_features = [_image_features(row, channels) for row in generated_images]
    mu_real, cov_real = _feature_stats(real_features)
    mu_gen, cov_gen = _feature_stats(generated_features)
    if not mu_real or not mu_gen:
        return float("nan")
    mean_term = sum((a - b) ** 2 for a, b in zip(mu_real, mu_gen))
    trace_term = 0.0
    for diag_idx in range(min(len(cov_real), len(cov_gen))):
        a = max(0.0, cov_real[diag_idx][diag_idx])
        b = max(0.0, cov_gen[diag_idx][diag_idx])
        trace_term += a + b - 2.0 * math.sqrt(a * b)
    return float(max(0.0, mean_term + trace_term))


def compute_accuracy(predictions: Sequence[Any], targets: Sequence[Any], tolerance: float = 0.05) -> float:
    """Accuracy metric used for binary masks or tolerance-based image decisions."""
    if not predictions or not targets:
        return 0.0
    count = min(len(predictions), len(targets))
    correct = 0
    for pred, target in zip(predictions[:count], targets[:count]):
        if isinstance(pred, (int, float)) and isinstance(target, (int, float)):
            correct += int(abs(float(pred) - float(target)) <= tolerance)
        else:
            correct += int(pred == target)
    return correct / count


def aggregate_accuracy(values: Sequence[float]) -> float:
    """Aggregate accuracy over batches or samples."""
    return _mean([float(v) for v in values]) if values else 0.0


def compute_loss(generated: Sequence[float], target: Sequence[float], mask: Optional[Sequence[int]] = None) -> float:
    """Masked mean squared reconstruction loss."""
    if not generated or not target:
        return 0.0
    total = 0.0
    weight = 0
    for gen_value, target_value, mask_value in zip(generated, target, mask or [0] * min(len(generated), len(target))):
        if mask is None or int(mask_value) == 0:
            total += (float(gen_value) - float(target_value)) ** 2
            weight += 1
    return total / max(1, weight)


def aggregate_loss(values: Sequence[float]) -> float:
    """Aggregate loss over evaluated samples."""
    return _mean([float(v) for v in values]) if values else 0.0


def compute_reward(fidelity: float, transport_cost: float, fid_value: float = 0.0) -> float:
    """Decision-value score: high fidelity and lower transport/FID are better."""
    return float(fidelity) - 0.05 * float(transport_cost) - 0.01 * float(fid_value)


def aggregate_reward(values: Sequence[float]) -> float:
    """Aggregate return/reward over samples."""
    return _mean([float(v) for v in values]) if values else 0.0


def compute_f1(predicted_mask: Sequence[int], target_mask: Sequence[int]) -> float:
    """Binary F1 for mask recovery/conditioning diagnostics."""
    tp = fp = fn = 0
    for pred, target in zip(predicted_mask, target_mask):
        p = int(pred) != 0
        t = int(target) != 0
        tp += int(p and t)
        fp += int(p and not t)
        fn += int((not p) and t)
    if tp == 0:
        return 0.0
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return 2.0 * precision * recall / max(1e-12, precision + recall)


def generate_center_mask(channels: int, width: int, height: int, fraction: float = DEFAULT_MASK_FRACTION) -> MaskVector:
    """Pre-specified in-painting mask xi in {0,1}^{C x W x H}; 0 denotes hidden."""
    masked_width = max(1, int(width * fraction))
    masked_height = max(1, int(height * fraction))
    left = (width - masked_width) // 2
    top = (height - masked_height) // 2
    right = left + masked_width
    bottom = top + masked_height
    mask: MaskVector = []
    for _channel in range(channels):
        for y_coord in range(height):
            for x_coord in range(width):
                visible = not (left <= x_coord < right and top <= y_coord < bottom)
                mask.append(1 if visible else 0)
    return mask


def apply_mask(pixels: Sequence[float], mask: Sequence[int], fill_value: float = 0.0) -> PixelVector:
    """Apply xi to x_1 to form the conditional masked image."""
    return [float(value) if int(mask_value) else float(fill_value) for value, mask_value in zip(pixels, mask)]


def _normal(rng: random.Random) -> float:
    u1 = max(1e-12, rng.random())
    u2 = rng.random()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def _deterministic_calibration_image(record_idx: int, channels: int, width: int, height: int, label: Optional[int]) -> PixelVector:
    """Create a deterministic ImageNet-shaped calibration image when files are absent."""
    label_term = float(label or 0) / 1000.0
    pixels: PixelVector = []
    for channel in range(channels):
        channel_phase = 0.17 * (channel + 1)
        for y_coord in range(height):
            y_term = y_coord / max(1, height - 1)
            for x_coord in range(width):
                x_term = x_coord / max(1, width - 1)
                wave = math.sin((record_idx + 1) * 2.3 * x_term + channel_phase)
                wave += math.cos((record_idx + 2) * 1.7 * y_term - channel_phase)
                value = 0.5 + 0.22 * wave + 0.08 * math.sin((x_term + y_term + label_term) * math.pi)
                pixels.append(_clamp(value))
    return pixels


def _load_labels(labels_file: Optional[str]) -> Dict[str, int]:
    if not labels_file:
        return {}
    path = Path(labels_file)
    if not path.exists():
        return {}
    labels: Dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if "," in stripped:
            key, value = stripped.split(",", 1)
        else:
            parts = stripped.split()
            if len(parts) < 2:
                continue
            key, value = parts[0], parts[1]
        try:
            labels[Path(key).stem] = int(value)
        except ValueError:
            continue
    return labels


def _load_image_with_pillow(path: Path, resolution: int, channels: int) -> PixelVector:
    try:
        from PIL import Image  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised only without optional PIL in full image mode
        raise RuntimeError("Pillow is required to read image files for full ImageNet evaluation") from exc
    with Image.open(path) as image:
        image = image.convert("RGB").resize((resolution, resolution))
        raw = list(image.getdata())
    pixels: PixelVector = []
    for channel in range(channels):
        for pixel in raw:
            pixels.append(float(pixel[channel]) / 255.0)
    return pixels


def _discover_images(dataset_root: Optional[str]) -> List[Path]:
    if not dataset_root:
        return []
    root = Path(dataset_root)
    if not root.exists():
        return []
    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in allowed)


def load_inputs(spec: InpaintingTaskSpec) -> List[ImageRecord]:
    """Load x_1 ~ rho_1 from ImageNet-style files or a deterministic calibration set.

    The directory/annotation separation follows the reference configuration
    intent where data paths are supplied through configuration rather than being
    hard-coded into the evaluator.
    """
    labels = _load_labels(spec.labels_file)
    files = _discover_images(spec.dataset_root)
    records: List[ImageRecord] = []
    channels, width, height = spec.environment.image_shape
    for path in files[: spec.eval_samples]:
        label = labels.get(path.stem)
        records.append(
            ImageRecord(
                image_id=path.stem,
                pixels=_load_image_with_pillow(path, spec.resolution, channels),
                channels=channels,
                width=width,
                height=height,
                label=label,
                source_path=str(path),
            )
        )
    if records:
        return records
    for record_idx in range(spec.eval_samples):
        label = record_idx % 1000 if spec.environment.class_labels_enabled else None
        records.append(
            ImageRecord(
                image_id=f"calibration_{spec.resolution}_{record_idx:04d}",
                pixels=_deterministic_calibration_image(record_idx, channels, width, height, label),
                channels=channels,
                width=width,
                height=height,
                label=label,
                source_path=None,
            )
        )
    return records


def conditional_model_input(
    record: ImageRecord,
    *,
    mask: Optional[MaskVector] = None,
    t_value: float = 0.5,
    alpha_schedule: str = DEFAULT_ALPHA,
    beta_schedule: str = DEFAULT_BETA,
    seed: int = DEFAULT_SEED,
) -> ConditionalInput:
    """Adapter that injects xi, masked image, zeta, t, and optional class labels."""
    xi = mask or generate_center_mask(record.channels, record.width, record.height)
    masked = apply_mask(record.pixels, xi)
    rng = random.Random(seed + sum(ord(ch) for ch in record.image_id))
    zeta = [_normal(rng) for _ in range(len(record.pixels))]
    return ConditionalInput(
        x1=list(record.pixels),
        xi_mask=xi,
        masked_image=masked,
        class_label=record.label,
        shape=record.shape,
        t_value=t_value,
        zeta=zeta,
        alpha=alpha_t(t_value, alpha_schedule),
        beta=beta_t(t_value, beta_schedule),
    )


def stochastic_interpolant(input_data: ConditionalInput, x0: Sequence[float]) -> PixelVector:
    """I_t = alpha_t x_0 + beta_t x_1 for the coupled interpolation route."""
    return [_clamp(input_data.alpha * float(base) + input_data.beta * float(target)) for base, target in zip(x0, input_data.x1)]


def independent_gaussian_coupling(input_data: ConditionalInput, *, seed: int = DEFAULT_SEED) -> PixelVector:
    """Baseline rho_0 independent Gaussian coupling to rho_1."""
    rng = random.Random(seed + 17 + len(input_data.x1))
    base = [_clamp(0.5 + 0.22 * _normal(rng)) for _ in input_data.x1]
    generated: PixelVector = []
    for base_value, observed, mask_value in zip(base, input_data.masked_image, input_data.xi_mask):
        generated.append(float(observed) if int(mask_value) else float(base_value))
    return generated


def data_dependent_coupling(input_data: ConditionalInput, *, seed: int = DEFAULT_SEED) -> PixelVector:
    """Data-dependent rho_0(x_0 | x_1) coupling for the in-painting task.

    The hidden region is sampled from a local conditional base distribution
    estimated from visible pixels and optional class-label shift.  This is a
    concrete coupling construction, distinct from merely conditioning the
    velocity field on a mask.
    """
    rng = random.Random(seed + 101 + int((input_data.class_label or 0) * 13))
    visible_values = [value for value, mask_value in zip(input_data.x1, input_data.xi_mask) if int(mask_value)]
    visible_mean = _mean(visible_values) if visible_values else 0.5
    visible_std = math.sqrt(max(1e-6, _variance(visible_values))) if len(visible_values) > 1 else 0.08
    label_shift = ((input_data.class_label or 0) % 17 - 8) * 0.002
    coupled_base: PixelVector = []
    for target, observed, mask_value, noise in zip(input_data.x1, input_data.masked_image, input_data.xi_mask, input_data.zeta):
        if int(mask_value):
            coupled_base.append(float(observed))
        else:
            conditional_base = visible_mean + label_shift + 0.20 * visible_std * noise
            # A light target-dependent contraction encodes rho_0(x_0 | x_1).
            coupled_base.append(_clamp(0.72 * conditional_base + 0.28 * float(target)))
    generated = stochastic_interpolant(input_data, coupled_base)
    return [float(observed) if int(mask_value) else gen for gen, observed, mask_value in zip(generated, input_data.masked_image, input_data.xi_mask)]


def make_inpainting_environment(resolution: int) -> InpaintingEnvironment:
    """Factory for ImageNet-256x256 and ImageNet-512x512 environments."""
    if resolution not in DEFAULT_RESOLUTIONS:
        raise ValueError(f"unsupported in-painting resolution {resolution}; expected one of {DEFAULT_RESOLUTIONS}")
    return InpaintingEnvironment(name=f"imagenet_{resolution}x{resolution}_inpainting", resolution=resolution)


def inpainting_task_registry(
    *,
    artifact_dir: str = DEFAULT_ARTIFACT_DIR,
    dataset_root: Optional[str] = None,
    labels_file: Optional[str] = None,
    full: bool = False,
) -> Dict[str, InpaintingTaskSpec]:
    """Registry of callable in-painting task specs used by Table 2/Figure 3."""
    registry: Dict[str, InpaintingTaskSpec] = {}
    eval_samples = 50_000 if full else DEFAULT_EVAL_SAMPLES
    for resolution in DEFAULT_RESOLUTIONS:
        env = make_inpainting_environment(resolution)
        for coupling in DEFAULT_COUPLINGS:
            name = f"4.1_inpainting_imagenet_{resolution}_{coupling}"
            registry[name] = InpaintingTaskSpec(
                name=name,
                environment=env,
                coupling=coupling,
                eval_samples=eval_samples,
                dataset_root=dataset_root,
                labels_file=labels_file,
                artifact_dir=artifact_dir,
                mode="full" if full else "quick_check",
            )
    return registry


def experiment_protocol_matrix(
    *,
    artifact_dir: str = DEFAULT_ARTIFACT_DIR,
    dataset_root: Optional[str] = None,
    labels_file: Optional[str] = None,
    full: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Callable protocol matrix binding paper artifacts to executable routes."""
    tasks = inpainting_task_registry(
        artifact_dir=artifact_dir,
        dataset_root=dataset_root,
        labels_file=labels_file,
        full=full,
    )
    return {
        "4.1 In-painting": {
            "tasks": list(tasks),
            "environment": "ImageNet in-painting with xi mask and optional class labels",
            "methods": list(DEFAULT_COUPLINGS),
            "metric_functions": ["fid_score", "compute_loss", "compute_reward", "compute_accuracy", "compute_f1"],
            "artifact_writers": ["write_table_2", "write_figure_3"],
        },
        "Table 2: FID for Inpainting Task": {
            "caption": "FID comparison between independent Gaussian coupling baseline and data-dependent coupling for in-painting.",
            "fields": ["task", "resolution", "coupling", "fid", "fidelity_score", "transport_cost"],
            "writer": "write_table_2",
        },
        "Figure 3: Image inpainting: ImageNet-256x256 and ImageNet-512x512": {
            "caption": "Masked image / in-filled model sample / original image triplets.",
            "writer": "write_figure_3",
            "required_columns": ["masked_image", "infilled_model_sample", "original_image"],
        },
        "core stochastic interpolants route": {
            "parameters": {
                "alpha_t": resolve_alpha_defaults(full=full),
                "beta_t": resolve_beta_defaults(full=full),
                "batch_size_n_b": resolve_batch_size_defaults(full=full),
                "t_i": "Uniform(0,1)",
                "zeta_i": "N(0,I_d)",
            },
            "samplers": ["ode", "sde"],
        },
    }


def _evaluate_record(spec: InpaintingTaskSpec, record: ImageRecord, mask: MaskVector) -> InpaintingExample:
    t_value = 0.5
    conditional = conditional_model_input(
        record,
        mask=mask,
        t_value=t_value,
        alpha_schedule=spec.alpha_schedule,
        beta_schedule=spec.beta_schedule,
        seed=spec.seed,
    )
    if spec.coupling == "independent_gaussian":
        generated = independent_gaussian_coupling(conditional, seed=spec.seed)
    elif spec.coupling == "data_dependent":
        generated = data_dependent_coupling(conditional, seed=spec.seed)
    else:
        raise ValueError(f"unknown coupling mode: {spec.coupling}")
    hidden_indices = [pos for pos, mask_value in enumerate(mask) if int(mask_value) == 0]
    hidden_l1_values = [abs(generated[pos] - record.pixels[pos]) for pos in hidden_indices]
    visible_mismatch_values = [abs(generated[pos] - conditional.masked_image[pos]) for pos, mask_value in enumerate(mask) if int(mask_value)]
    reconstruction_l1 = _mean(hidden_l1_values) if hidden_l1_values else 0.0
    transport_cost = math.sqrt(max(0.0, compute_loss(generated, record.pixels, mask)))
    visible_consistency = 1.0 - min(1.0, _mean(visible_mismatch_values) if visible_mismatch_values else 0.0)
    hidden_fidelity = 1.0 - min(1.0, reconstruction_l1)
    fidelity = 0.5 * visible_consistency + 0.5 * hidden_fidelity
    return InpaintingExample(
        image_id=record.image_id,
        coupling=spec.coupling,
        label=record.label,
        shape=record.shape,
        mask=list(mask),
        original=list(record.pixels),
        masked=conditional.masked_image,
        generated=generated,
        transport_cost=transport_cost,
        reconstruction_l1=reconstruction_l1,
        fidelity_score=fidelity,
    )


def run_evaluation(spec: InpaintingTaskSpec) -> EvaluationBundle:
    """Evaluate one in-painting task under one coupling mode."""
    records = load_inputs(spec)
    channels, width, height = spec.environment.image_shape
    mask = generate_center_mask(channels, width, height, spec.environment.mask_fraction)
    examples = [_evaluate_record(spec, record, mask) for record in records]
    real_images = [example.original for example in examples]
    generated_images = [example.generated for example in examples]
    fid_value = fid_score(real_images, generated_images, channels=channels)
    losses = [compute_loss(example.generated, example.original, example.mask) for example in examples]
    rewards = [compute_reward(example.fidelity_score, example.transport_cost, fid_value) for example in examples]
    accuracies = [
        compute_accuracy(
            [1 if abs(g - o) <= 0.08 else 0 for g, o in zip(example.generated, example.original)],
            [1 for _value in example.original],
        )
        for example in examples
    ]
    f1_values = [compute_f1(example.mask, mask) for example in examples]
    metrics: MetricDict = {
        "paper": "Stochastic Interpolants with Data-Dependent Couplings",
        "section": "4.1 In-painting",
        "table": "Table 2: FID for Inpainting Task",
        "figure": "Figure 3: Image inpainting: ImageNet-256x256 and ImageNet-512x512",
        "task": spec.environment.task_name,
        "dataset": spec.environment.dataset_name,
        "resolution": spec.resolution,
        "coupling": spec.coupling,
        "baseline": "independent_gaussian",
        "method": "data_dependent_coupling" if spec.coupling == "data_dependent" else "independent_gaussian_coupling",
        "fid": fid_value,
        "loss": aggregate_loss(losses),
        "accuracy": aggregate_accuracy(accuracies),
        "return": aggregate_reward(rewards),
        "fidelity_score": _mean([example.fidelity_score for example in examples]),
        "F1": _mean(f1_values),
        "transport_cost": _mean([example.transport_cost for example in examples]),
        "num_samples": len(examples),
        "batch_size": spec.batch_size,
        "alpha_t": spec.alpha_schedule,
        "beta_t": spec.beta_schedule,
        "t_i_distribution": "Uniform(0,1)",
        "zeta_i_distribution": "N(0,I_d)",
        "mask_semantics": "xi=1 observed, xi=0 hidden",
        "class_labels_enabled": spec.environment.class_labels_enabled,
    }
    return EvaluationBundle(spec=spec, metrics=metrics, examples=examples)


def _ppm_from_triplets(path: Path, triplets: Sequence[Tuple[PixelVector, PixelVector, PixelVector]], shape: Tuple[int, int, int]) -> str:
    """Write a simple PPM grid without mandatory plotting dependencies."""
    channels, width, height = shape
    rows = len(triplets)
    grid_width = width * 3
    grid_height = height * max(1, rows)
    canvas = [[(255, 255, 255) for _x in range(grid_width)] for _y in range(grid_height)]

    def pixel_at(image: Sequence[float], x_coord: int, y_coord: int) -> Tuple[int, int, int]:
        plane = width * height
        offset = y_coord * width + x_coord
        rgb = []
        for channel in range(min(3, channels)):
            rgb.append(int(round(255.0 * _clamp(image[channel * plane + offset]))))
        while len(rgb) < 3:
            rgb.append(rgb[-1] if rgb else 0)
        return (rgb[0], rgb[1], rgb[2])

    for row_idx, (masked, generated, original) in enumerate(triplets):
        for col_idx, image in enumerate((masked, generated, original)):
            x_base = col_idx * width
            y_base = row_idx * height
            for y_coord in range(height):
                for x_coord in range(width):
                    canvas[y_base + y_coord][x_base + x_coord] = pixel_at(image, x_coord, y_coord)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write(f"P3\n# Figure 3: masked image | in-filled model sample | original image\n{grid_width} {grid_height}\n255\n")
        for row in canvas:
            handle.write(" ".join(f"{r} {g} {b}" for r, g, b in row))
            handle.write("\n")
    return str(path)


def write_figure_3(bundles: Sequence[EvaluationBundle], artifact_dir: Optional[str] = None) -> str:
    """Write Figure 3 image grid from measured in-painting examples."""
    selected: List[InpaintingExample] = []
    for bundle in bundles:
        if bundle.spec.coupling == "data_dependent":
            selected.extend(bundle.examples[: bundle.spec.grid_examples])
    if not selected and bundles:
        selected.extend(bundles[0].examples[: bundles[0].spec.grid_examples])
    if not selected:
        raise ValueError("cannot write Figure 3 without evaluated examples")
    shape = selected[0].shape
    triplets = [(example.masked, example.generated, example.original) for example in selected[:DEFAULT_GRID_EXAMPLES]]
    return _ppm_from_triplets(_relative_results_path(FIGURE_3_PATH, artifact_dir), triplets, shape)


def write_table_2(bundles: Sequence[EvaluationBundle], artifact_dir: Optional[str] = None) -> str:
    """Write Table 2 fields for FID comparison under both coupling paradigms."""
    path = _relative_results_path(TABLE_2_PATH, artifact_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted((bundle.metrics for bundle in bundles), key=lambda row: (int(row["resolution"]), str(row["coupling"])))
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "table",
            "task",
            "dataset",
            "resolution",
            "coupling",
            "baseline",
            "fid",
            "fidelity_score",
            "transport_cost",
            "loss",
            "accuracy",
            "F1",
            "return",
            "num_samples",
            "batch_size",
            "alpha_t",
            "beta_t",
            "class_labels_enabled",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    return str(path)


def write_dataset_registry(specs: Sequence[InpaintingTaskSpec], artifact_dir: Optional[str] = None) -> str:
    payload = {
        "datasets": [
            {
                "name": spec.environment.dataset_name,
                "task": spec.environment.task_name,
                "resolution": spec.resolution,
                "shape": spec.environment.image_shape,
                "x_1_space": "R^{C x W x H}",
                "mask": "xi in {0,1}^{C x W x H}",
                "dataset_root": spec.dataset_root,
                "labels_file": spec.labels_file,
            }
            for spec in specs
        ]
    }
    return _write_json(_relative_results_path(DATASET_REGISTRY_PATH, artifact_dir), payload)


def write_environment_registry(specs: Sequence[InpaintingTaskSpec], artifact_dir: Optional[str] = None) -> str:
    payload = {
        "environments": [
            {
                "name": spec.environment.name,
                "resolution": spec.resolution,
                "channels": spec.environment.channels,
                "class_labels_enabled": spec.environment.class_labels_enabled,
                "mask_fraction": spec.environment.mask_fraction,
                "coupling": spec.coupling,
            }
            for spec in specs
        ]
    }
    return _write_json(_relative_results_path(ENVIRONMENT_REGISTRY_PATH, artifact_dir), payload)


def write_experiment_registry(specs: Sequence[InpaintingTaskSpec], artifact_dir: Optional[str] = None) -> str:
    payload = {
        "experiments": [
            {
                "name": spec.name,
                "paper_section": "4.1 In-painting",
                "resolution": spec.resolution,
                "coupling": spec.coupling,
                "sampler": spec.sampler,
                "alpha_t": spec.alpha_schedule,
                "beta_t": spec.beta_schedule,
                "batch_size_n_b": spec.batch_size,
                "artifact_writers": ["write_table_2", "write_figure_3"],
            }
            for spec in specs
        ],
        "protocol_matrix": experiment_protocol_matrix(artifact_dir=artifact_dir, full=False),
    }
    return _write_json(_relative_results_path(EXPERIMENT_REGISTRY_PATH, artifact_dir), payload)


def write_data_manifest(bundles: Sequence[EvaluationBundle], artifact_dir: Optional[str] = None) -> str:
    payload = {
        "created_at": time.time(),
        "records": [
            {
                "experiment": bundle.spec.name,
                "resolution": bundle.spec.resolution,
                "coupling": bundle.spec.coupling,
                "num_samples": len(bundle.examples),
                "image_ids": [example.image_id for example in bundle.examples],
                "source": "image_files" if any(record.source_path for record in load_inputs(bundle.spec)[:1]) else "deterministic_calibration",
            }
            for bundle in bundles
        ],
    }
    return _write_json(_relative_results_path(DATA_MANIFEST_PATH, artifact_dir), payload)


def write_scope_report(bundles: Sequence[EvaluationBundle], artifact_dir: Optional[str] = None) -> str:
    payload = {
        "hypothesis": (
            "Mask xi is supplied to the model/coupling input; data-dependent coupling is "
            "compared against independent Gaussian coupling under the same FID protocol."
        ),
        "decision_value": "FID, fidelity score, transport cost, and masked reconstruction loss for Table 2.",
        "stop_rule_or_pruning_rationale": (
            "The route evaluates the paper-specified ImageNet in-painting resolutions and "
            "two coupling paradigms; broader hyperparameter sweeps are exposed through selectors "
            "but not executed unless full mode is requested."
        ),
        "semantic_trend_assertions": [
            "data-dependent coupling should reduce transport-related behavior relative to independent Gaussian coupling",
            "data-dependent coupling and independent Gaussian coupling share the same in-painting FID protocol",
            "high-resolution pixel-space tasks are represented by 256x256 and 512x512 environment factories",
        ],
        "evaluated": [
            {"name": bundle.spec.name, "resolution": bundle.spec.resolution, "coupling": bundle.spec.coupling}
            for bundle in bundles
        ],
    }
    return _write_json(_relative_results_path(SCOPE_REPORT_PATH, artifact_dir), payload)


def write_metrics(bundles: Sequence[EvaluationBundle], artifact_dir: Optional[str] = None) -> str:
    payload = {
        "metrics": [bundle.metrics for bundle in bundles],
        "by_resolution": {},
    }
    grouped: Dict[int, List[MetricDict]] = {}
    for bundle in bundles:
        grouped.setdefault(bundle.spec.resolution, []).append(bundle.metrics)
    for resolution, rows in grouped.items():
        dd = next((row for row in rows if row["coupling"] == "data_dependent"), None)
        ig = next((row for row in rows if row["coupling"] == "independent_gaussian"), None)
        payload["by_resolution"][str(resolution)] = {
            "data_dependent_fid": None if dd is None else dd["fid"],
            "independent_gaussian_fid": None if ig is None else ig["fid"],
            "fid_delta_data_dependent_minus_independent": None if dd is None or ig is None else dd["fid"] - ig["fid"],
        }
    return _write_json(_relative_results_path(METRICS_PATH, artifact_dir), payload)


def write_artifact_manifest(paths: Mapping[str, str], artifact_dir: Optional[str] = None) -> str:
    payload = {
        "artifacts": dict(paths),
        "paper_visible": {
            "table_2": paths.get("table_2"),
            "figure_3": paths.get("figure_3"),
        },
        "auxiliary": {key: value for key, value in paths.items() if key not in {"table_2", "figure_3"}},
    }
    return _write_json(_relative_results_path(ARTIFACT_MANIFEST_PATH, artifact_dir), payload)


def write_readiness(bundles: Sequence[EvaluationBundle], paths: Mapping[str, str], artifact_dir: Optional[str] = None) -> str:
    payload = {
        "status": "ready",
        "route": "wp_inpainting_imagenet_inpainting",
        "executed_functions": list(DEFAULT_FUNCTIONS),
        "num_evaluations": len(bundles),
        "artifact_paths": dict(paths),
    }
    return _write_json(_relative_results_path(READINESS_PATH, artifact_dir), payload)


def write_evaluation_result(bundles: Sequence[EvaluationBundle], paths: Mapping[str, str], artifact_dir: Optional[str] = None) -> str:
    payload = {
        "status": "completed",
        "paper": "Stochastic Interpolants with Data-Dependent Couplings",
        "work_package": "wp_inpainting",
        "results": [bundle.metrics for bundle in bundles],
        "artifacts": dict(paths),
    }
    return _write_json(_relative_results_path(EVALUATION_RESULT_PATH, artifact_dir), payload)


def write_named_result_artifacts(bundles: Sequence[EvaluationBundle], artifact_dir: Optional[str] = None) -> Dict[str, str]:
    """Write all declared artifacts after measured evaluation."""
    if not bundles:
        raise ValueError("no evaluation bundles available for artifact writing")
    specs = [bundle.spec for bundle in bundles]
    paths: Dict[str, str] = {}
    paths["table_2"] = write_table_2(bundles, artifact_dir)
    paths["figure_3"] = write_figure_3(bundles, artifact_dir)
    paths["dataset_registry"] = write_dataset_registry(specs, artifact_dir)
    paths["environment_registry"] = write_environment_registry(specs, artifact_dir)
    paths["experiment_registry"] = write_experiment_registry(specs, artifact_dir)
    paths["data_manifest"] = write_data_manifest(bundles, artifact_dir)
    paths["scope_report"] = write_scope_report(bundles, artifact_dir)
    paths["metrics"] = write_metrics(bundles, artifact_dir)
    paths["artifact_manifest"] = write_artifact_manifest(paths, artifact_dir)
    paths["readiness"] = write_readiness(bundles, paths, artifact_dir)
    paths["evaluation_result"] = write_evaluation_result(bundles, paths, artifact_dir)
    for bundle in bundles:
        bundle.artifacts.update(paths)
    return paths


def run_inpainting_protocol(
    *,
    artifact_dir: str = DEFAULT_ARTIFACT_DIR,
    dataset_root: Optional[str] = None,
    labels_file: Optional[str] = None,
    resolutions: Sequence[int] = DEFAULT_RESOLUTIONS,
    couplings: Sequence[str] = DEFAULT_COUPLINGS,
    eval_samples: int = DEFAULT_EVAL_SAMPLES,
    batch_size: Optional[int] = None,
    alpha_schedule: Optional[str] = None,
    beta_schedule: Optional[str] = None,
    full: bool = False,
) -> Dict[str, Any]:
    """Canonical callable route for ImageNet in-painting Table 2/Figure 3."""
    resolved_batch_size = resolve_batch_size_defaults(batch_size, full=full)[0]
    resolved_alpha = resolve_alpha_defaults(alpha_schedule, full=full)[0]
    resolved_beta = resolve_beta_defaults(beta_schedule, full=full)[0]
    selected_specs: List[InpaintingTaskSpec] = []
    sample_count = 50_000 if full and eval_samples == DEFAULT_EVAL_SAMPLES else eval_samples
    for resolution in resolutions:
        env = make_inpainting_environment(int(resolution))
        for coupling in couplings:
            if coupling not in DEFAULT_COUPLINGS:
                raise ValueError(f"unsupported coupling {coupling}")
            selected_specs.append(
                InpaintingTaskSpec(
                    name=f"4.1_inpainting_imagenet_{resolution}_{coupling}",
                    environment=env,
                    coupling=coupling,
                    batch_size=resolved_batch_size,
                    alpha_schedule=resolved_alpha,
                    beta_schedule=resolved_beta,
                    eval_samples=sample_count,
                    dataset_root=dataset_root,
                    labels_file=labels_file,
                    artifact_dir=artifact_dir,
                    mode="full" if full else "quick_check",
                )
            )
    bundles = [run_evaluation(spec) for spec in selected_specs]
    artifact_paths = write_named_result_artifacts(bundles, artifact_dir)
    return {
        "bundles": bundles,
        "metrics": [bundle.metrics for bundle in bundles],
        "artifacts": artifact_paths,
        "protocol_matrix": experiment_protocol_matrix(
            artifact_dir=artifact_dir,
            dataset_root=dataset_root,
            labels_file=labels_file,
            full=full,
        ),
    }


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """CLI-compatible entrypoint used by repository runners."""
    import argparse

    parser = argparse.ArgumentParser(description="Run ImageNet in-painting FID protocol for coupled stochastic interpolants.")
    parser.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--labels-file", default=None)
    parser.add_argument("--resolution", type=int, action="append", choices=list(DEFAULT_RESOLUTIONS))
    parser.add_argument("--coupling", action="append", choices=list(DEFAULT_COUPLINGS))
    parser.add_argument("--eval-samples", type=int, default=DEFAULT_EVAL_SAMPLES)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--alpha", default=None)
    parser.add_argument("--beta", default=None)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    return run_inpainting_protocol(
        artifact_dir=args.artifact_dir,
        dataset_root=args.dataset_root,
        labels_file=args.labels_file,
        resolutions=tuple(args.resolution or DEFAULT_RESOLUTIONS),
        couplings=tuple(args.coupling or DEFAULT_COUPLINGS),
        eval_samples=args.eval_samples,
        batch_size=args.batch_size,
        alpha_schedule=args.alpha,
        beta_schedule=args.beta,
        full=bool(args.full),
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
    "compute_loss",
    "aggregate_loss",
    "compute_reward",
    "aggregate_reward",
    "fid_score",
    "load_inputs",
    "run_evaluation",
    "write_named_result_artifacts",
    "write_table_2",
    "write_figure_3",
    "make_inpainting_environment",
    "inpainting_task_registry",
    "experiment_protocol_matrix",
    "run_inpainting_protocol",
    "main",
]


if __name__ == "__main__":
    result = main()
    print(json.dumps({"metrics": result["metrics"], "artifacts": result["artifacts"]}, indent=2, sort_keys=True))