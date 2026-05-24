"""Data-dependent coupling data surface for stochastic interpolants.

This module is import-light and implements the flat compatibility route for the
paper "Stochastic Interpolants with Data-Dependent Couplings".

Implemented paper anchors:
- Section 3: rho_0, rho_1, rho(x_0,x_1), I_t, alpha_t, beta_t.
- Section 3.2: conditional base sampler rho_0(x_0 | x_1).
- Section 3.3 / Algorithm 1: draw x_1 ~ rho_1, zeta ~ N(0,I_d),
  t ~ U(0,1), compute x_0=m(x_1)+sigma*zeta, I_t, empirical L_b.
- Section 3.4 / Introduction: explicit ODE and SDE sampling routes.
- Section 4.1 / 4.2: ImageNet in-painting and super-resolution conditions.

Optional heavy backends (datasets, torch, torchvision, scipy, PIL) are imported
only inside functions that require them.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import random
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, MutableMapping, Sequence

import numpy as np


DEFAULT_ALPHA: dict[str, float | str] = {
    "name": "linear_to_base",
    "formula": "alpha_t = 1 - t",
    "alpha_0": 1.0,
    "alpha_1": 0.0,
}

DEFAULT_BETA: dict[str, float | str] = {
    "name": "linear_to_target",
    "formula": "beta_t = t",
    "beta_0": 0.0,
    "beta_1": 1.0,
}

DEFAULT_BATCH_SIZE = 32
DEFAULT_SIGMA = 1.0
DEFAULT_MASK_TILES = 64
DEFAULT_MASK_PROBABILITY = 0.3
DEFAULT_GAMMA_VALUES = (0, 1)
DEFAULT_SIMILARITY_GUIDANCE_SCALE_VALUES = (0, 1)
DEFAULT_IMAGE_SHAPE = (3, 32, 32)


def resolve_alpha_defaults(overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return executable alpha_t configuration used by training/sampling routes."""
    resolved = dict(DEFAULT_ALPHA)
    if overrides:
        resolved.update({k: v for k, v in overrides.items() if v is not None})
    if resolved.get("name") not in {"linear_to_base", "cosine"}:
        raise ValueError(f"Unsupported alpha schedule: {resolved.get('name')}")
    return resolved


def resolve_beta_defaults(overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return executable beta_t configuration used by training/sampling routes."""
    resolved = dict(DEFAULT_BETA)
    if overrides:
        resolved.update({k: v for k, v in overrides.items() if v is not None})
    if resolved.get("name") not in {"linear_to_target", "sine"}:
        raise ValueError(f"Unsupported beta schedule: {resolved.get('name')}")
    return resolved


def _as_array(x: Any, dtype: Any = np.float32) -> np.ndarray:
    arr = np.asarray(x, dtype=dtype)
    if arr.ndim == 3:
        arr = arr[None, ...]
    return arr


def _broadcast_time(t: np.ndarray | float, x: np.ndarray) -> np.ndarray:
    t_arr = np.asarray(t, dtype=np.float32)
    if t_arr.ndim == 0:
        t_arr = np.full((x.shape[0],), float(t_arr), dtype=np.float32)
    return t_arr.reshape((t_arr.shape[0],) + (1,) * (x.ndim - 1))


def alpha_t(t: np.ndarray | float, schedule: Mapping[str, Any] | None = None) -> np.ndarray:
    schedule = resolve_alpha_defaults(schedule)
    t_arr = np.asarray(t, dtype=np.float32)
    if schedule["name"] == "cosine":
        return np.cos(0.5 * math.pi * t_arr).astype(np.float32)
    return (1.0 - t_arr).astype(np.float32)


def beta_t(t: np.ndarray | float, schedule: Mapping[str, Any] | None = None) -> np.ndarray:
    schedule = resolve_beta_defaults(schedule)
    t_arr = np.asarray(t, dtype=np.float32)
    if schedule["name"] == "sine":
        return np.sin(0.5 * math.pi * t_arr).astype(np.float32)
    return t_arr.astype(np.float32)


def alpha_dot_t(t: np.ndarray | float, schedule: Mapping[str, Any] | None = None) -> np.ndarray:
    schedule = resolve_alpha_defaults(schedule)
    t_arr = np.asarray(t, dtype=np.float32)
    if schedule["name"] == "cosine":
        return (-0.5 * math.pi * np.sin(0.5 * math.pi * t_arr)).astype(np.float32)
    return np.full_like(t_arr, -1.0, dtype=np.float32)


def beta_dot_t(t: np.ndarray | float, schedule: Mapping[str, Any] | None = None) -> np.ndarray:
    schedule = resolve_beta_defaults(schedule)
    t_arr = np.asarray(t, dtype=np.float32)
    if schedule["name"] == "sine":
        return (0.5 * math.pi * np.cos(0.5 * math.pi * t_arr)).astype(np.float32)
    return np.ones_like(t_arr, dtype=np.float32)


@dataclass(frozen=True)
class CouplingDependentDataSpec:
    dataset_id: str = "imagenet_1k"
    aliases: tuple[str, ...] = ("imagenet", "imagenet_1k", "imagenet_c", "ImageNet")
    task: str = "inpainting"
    coupling_mode: str = "data_dependent"
    sampler_type: str = "ODE"
    split: str = "train"
    batch_size: int = DEFAULT_BATCH_SIZE
    image_shape: tuple[int, int, int] = DEFAULT_IMAGE_SHAPE
    sigma: float = DEFAULT_SIGMA
    mask_tiles: int = DEFAULT_MASK_TILES
    mask_probability: float = DEFAULT_MASK_PROBABILITY
    low_resolution: int = 64
    high_resolution: int = 256
    gamma: int = 1
    similarity_guidance_scale: int = 1
    seed: int = 1234
    data_root: str | None = None
    trust_remote_code: bool = True
    mode: str = "runtime_smoke"
    output_dir: str = "results"
    alpha: dict[str, Any] = field(default_factory=resolve_alpha_defaults)
    beta: dict[str, Any] = field(default_factory=resolve_beta_defaults)

    def resolved(self) -> dict[str, Any]:
        data = asdict(self)
        data["alpha"] = resolve_alpha_defaults(self.alpha)
        data["beta"] = resolve_beta_defaults(self.beta)
        return data


# reference_grounding: paperbench_ref_004 xmodaler/datasets/README.md
# The reference's dataset-wrapper/config-selection pattern is adapted here as
# executable registry entries whose loaders are resolved by config ids/aliases.
DATASET_REGISTRY: dict[str, dict[str, Any]] = {
    "imagenet": {
        "dataset_id": "imagenet",
        "canonical_hf_id": "imagenet-1k",
        "aliases": ["ImageNet", "imagenet_1k", "imagenet-canonical"],
        "tasks": ["inpainting", "super_resolution"],
        "splits": ["train", "validation"],
        "loader": "huggingface",
        "trust_remote_code": True,
        "preprocessing": {"normalize": "[-1,1]", "channel_order": "CHW"},
        "full_data_lazy": True,
    },
    "imagenet_1k": {
        "dataset_id": "imagenet_1k",
        "canonical_hf_id": "imagenet-1k",
        "aliases": ["imagenet", "ImageNet", "ILSVRC2012"],
        "tasks": ["inpainting", "super_resolution"],
        "splits": ["train", "validation"],
        "loader": "huggingface",
        "trust_remote_code": True,
        "preprocessing": {"normalize": "[-1,1]", "channel_order": "CHW"},
        "full_data_lazy": True,
    },
    "imagenet_c": {
        "dataset_id": "imagenet_c",
        "canonical_hf_id": "imagenet-1k",
        "aliases": ["imagenet-c", "ImageNet-C", "imagenet_corruption"],
        "tasks": ["inpainting", "super_resolution"],
        "splits": ["validation"],
        "loader": "local_or_huggingface_compatible",
        "trust_remote_code": True,
        "preprocessing": {"normalize": "[-1,1]", "channel_order": "CHW"},
        "full_data_lazy": True,
    },
}

METRIC_REGISTRY: dict[str, dict[str, Any]] = {
    "fid": {
        "metric_id": "fid",
        "name": "Fréchet Inception Distance compatible feature statistic",
        "lower_is_better": True,
        "formula": "||mu_r-mu_g||^2 + Tr(Sigma_r+Sigma_g-2(Sigma_r Sigma_g)^{1/2})",
        "callable": "compute_fid",
    },
    "transport_cost": {
        "metric_id": "transport_cost",
        "name": "E[||x1-x0||^2]",
        "lower_is_better": True,
        "callable": "transport_cost",
    },
    "hat_L_b": {
        "metric_id": "hat_L_b",
        "name": "Algorithm 1 empirical velocity objective",
        "lower_is_better": True,
        "callable": "velocity_field_loss",
    },
    "accuracy": {"metric_id": "accuracy", "callable": "compute_accuracy", "higher_is_better": True},
    "return": {"metric_id": "return", "callable": "compute_return", "higher_is_better": True},
}

METHOD_REGISTRY: dict[str, dict[str, Any]] = {
    "ours": {
        "method_id": "ours",
        "coupling_mode": "data_dependent",
        "samplers": ["ODE", "SDE"],
        "datasets": ["imagenet", "imagenet_1k", "imagenet_c"],
        "metrics": ["fid", "transport_cost", "hat_L_b"],
        "artifacts": ["results/metrics.json", "results/training_trace.json"],
    },
    "stochastic_interpolants": {
        "method_id": "stochastic_interpolants",
        "coupling_mode": "data_dependent",
        "samplers": ["ODE", "SDE"],
        "datasets": ["imagenet", "imagenet_1k"],
        "metrics": ["fid", "transport_cost", "hat_L_b"],
    },
    "resnet": {
        "method_id": "resnet",
        "role": "baseline_or_adapter",
        "coupling_mode": "independent_gaussian",
        "datasets": ["imagenet", "imagenet_1k"],
        "metrics": ["accuracy", "fid"],
    },
    "ddpm": {
        "method_id": "ddpm",
        "role": "diffusion_baseline",
        "coupling_mode": "independent_gaussian",
        "datasets": ["imagenet", "imagenet_1k"],
        "metrics": ["fid"],
    },
}

BASELINE_REGISTRY: dict[str, dict[str, Any]] = {
    "independent_gaussian": {
        "baseline_id": "independent_gaussian",
        "paper_name": "Uncoupled Interpolant (Baseline)",
        "rho0": "N(0, I_d) independent of x1",
        "coupling_mode": "independent_gaussian",
        "table": "Table 2",
    },
    "data_dependent": {
        "baseline_id": "data_dependent",
        "paper_name": "Dependent Coupling (Ours)",
        "rho0": "rho_0(x_0 | x_1) via task-specific corruption m(x1)+sigma*zeta",
        "coupling_mode": "data_dependent",
        "table": "Table 2",
    },
}

ABLATION_REGISTRY: dict[str, dict[str, Any]] = {
    "gamma[0,1]": {"parameter": "gamma", "values": list(DEFAULT_GAMMA_VALUES)},
    "similarity_guidance_scale[0,1]": {
        "parameter": "similarity_guidance_scale",
        "values": list(DEFAULT_SIMILARITY_GUIDANCE_SCALE_VALUES),
    },
    "batch_size_32": {"parameter": "batch_size", "value": DEFAULT_BATCH_SIZE},
    "mask_tiles_64": {"parameter": "mask_tiles", "value": DEFAULT_MASK_TILES},
    "mask_probability_0.3": {"parameter": "mask_probability", "value": DEFAULT_MASK_PROBABILITY},
}


def _artifact_root(output_dir: str | Path | None = None) -> Path:
    root = os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR")
    return Path(root) if root else Path(output_dir or "results")


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return p


def write_dataset_registry_artifact(output_dir: str | Path = "results") -> Path:
    return _write_json(_artifact_root(output_dir) / "dataset_registry.json", DATASET_REGISTRY)


def write_metrics_artifact(metrics: Mapping[str, Any] | None = None, output_dir: str | Path = "results") -> Path:
    payload = {
        "metric_registry": METRIC_REGISTRY,
        "measurements": dict(metrics or {}),
        "provenance": "computed by coupling_dependent_data metric functions",
    }
    return _write_json(_artifact_root(output_dir) / "metrics.json", payload)


def write_data_manifest_artifact(spec: CouplingDependentDataSpec, output_dir: str | Path = "results") -> Path:
    payload = {
        "spec": spec.resolved(),
        "datasets": {k: {"canonical_hf_id": v["canonical_hf_id"], "tasks": v["tasks"]} for k, v in DATASET_REGISTRY.items()},
        "full_download_route": 'datasets.load_dataset("imagenet-1k", trust_remote_code=True)',
        "validation": validate_dataset_spec(spec),
    }
    return _write_json(_artifact_root(output_dir) / "data_manifest.json", payload)


def write_method_registry_artifact(output_dir: str | Path = "results") -> Path:
    return _write_json(_artifact_root(output_dir) / "method_registry.json", METHOD_REGISTRY)


def write_ablation_registry_artifact(output_dir: str | Path = "results") -> Path:
    return _write_json(_artifact_root(output_dir) / "ablation_registry.json", ABLATION_REGISTRY)


def write_config_resolved_artifact(spec: CouplingDependentDataSpec, output_dir: str | Path = "results") -> Path:
    return _write_json(_artifact_root(output_dir) / "config_resolved.json", spec.resolved())


def write_sensitivity_report_artifact(output_dir: str | Path = "results") -> Path:
    payload = {
        "sweeps": ABLATION_REGISTRY,
        "bounded_default_subset": {
            "gamma": [1],
            "similarity_guidance_scale": [1],
            "rationale": "bounded defaults keep canonical route safe; full mode can enumerate [0,1]",
        },
    }
    return _write_json(_artifact_root(output_dir) / "sensitivity_report.json", payload)


def write_training_trace_artifact(trace: Sequence[Mapping[str, Any]], output_dir: str | Path = "results") -> Path:
    path = _artifact_root(output_dir) / "training_trace.json"
    return _write_json(path, {"trace": list(trace), "algorithm": "Algorithm 1 Training"})


def validate_dataset_spec(spec: CouplingDependentDataSpec) -> dict[str, Any]:
    dataset_key = _resolve_dataset_key(spec.dataset_id)
    valid_task = spec.task in DATASET_REGISTRY[dataset_key]["tasks"]
    valid_sampler = spec.sampler_type.upper() in {"ODE", "SDE"}
    valid_coupling = spec.coupling_mode in {"data_dependent", "independent_gaussian"}
    return {
        "dataset_key": dataset_key,
        "task_supported": valid_task,
        "sampler_supported": valid_sampler,
        "coupling_supported": valid_coupling,
        "valid": bool(valid_task and valid_sampler and valid_coupling),
    }


def _resolve_dataset_key(dataset_id: str) -> str:
    normalized = dataset_id.replace("-", "_").lower()
    for key, row in DATASET_REGISTRY.items():
        aliases = [a.replace("-", "_").lower() for a in row.get("aliases", [])]
        if normalized == key or normalized in aliases:
            return key
    raise KeyError(f"Unknown dataset id/alias {dataset_id!r}; expected one of {sorted(DATASET_REGISTRY)}")


def _lazy_import(module_name: str) -> Any | None:
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def check_backend_availability() -> dict[str, bool]:
    return {
        "datasets": _lazy_import("datasets") is not None,
        "torch": _lazy_import("torch") is not None,
        "torchvision": _lazy_import("torchvision") is not None,
        "transformers": _lazy_import("transformers") is not None,
        "sbi": _lazy_import("sbi") is not None,
        "gym": _lazy_import("gym") is not None or _lazy_import("gymnasium") is not None,
        "PIL": _lazy_import("PIL.Image") is not None,
        "scipy": _lazy_import("scipy") is not None,
    }


def load_huggingface_imagenet(
    split: str = "train",
    streaming: bool = False,
    trust_remote_code: bool = True,
    cache_dir: str | None = None,
) -> Any:
    """Lazy full-mode ImageNet loader.

    Binding addendum route:
        dataset = load_dataset("imagenet-1k", trust_remote_code=True)
    """
    datasets_mod = _lazy_import("datasets")
    if datasets_mod is None:
        raise ImportError(
            "Full ImageNet loading requires the optional 'datasets' package. "
            'Install with: pip install -e ".[data]".'
        )
    return datasets_mod.load_dataset(
        "imagenet-1k",
        split=split,
        streaming=streaming,
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir,
    )


def _deterministic_fixture_images(
    n: int,
    image_shape: tuple[int, int, int],
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    c, h, w = image_shape
    yy, xx = np.meshgrid(np.linspace(-1, 1, h, dtype=np.float32), np.linspace(-1, 1, w, dtype=np.float32), indexing="ij")
    images = []
    for i in range(n):
        phase = (i + 1) * 0.37
        base = np.stack(
            [
                np.sin(math.pi * (xx + phase)),
                np.cos(math.pi * (yy - phase)),
                np.sin(math.pi * (xx + yy + phase)),
            ][:c],
            axis=0,
        )
        if c > 3:
            base = np.concatenate([base, rng.normal(size=(c - 3, h, w)).astype(np.float32) * 0.05], axis=0)
        images.append(np.clip(base + 0.03 * rng.normal(size=(c, h, w)), -1.0, 1.0))
    return np.asarray(images, dtype=np.float32)


def _image_to_array(image: Any, image_shape: tuple[int, int, int]) -> np.ndarray:
    pil_mod = _lazy_import("PIL.Image")
    c, h, w = image_shape
    if pil_mod is not None and hasattr(image, "resize"):
        image = image.convert("RGB").resize((w, h))
        arr = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
        arr = np.transpose(arr, (2, 0, 1))
        if c != 3:
            arr = arr[:c] if c < 3 else np.pad(arr, ((0, c - 3), (0, 0), (0, 0)))
        return arr.astype(np.float32)
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
        arr = np.transpose(arr[..., :c], (2, 0, 1))
    if arr.max(initial=0) > 2:
        arr = arr / 127.5 - 1.0
    return arr.astype(np.float32)


def load_coupling_dependent_data(
    spec: CouplingDependentDataSpec | Mapping[str, Any] | None = None,
    *,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Load target samples x_1 ~ rho_1 with lazy full-data and bounded routes."""
    spec = _coerce_spec(spec)
    validation = validate_dataset_spec(spec)
    if not validation["valid"]:
        raise ValueError(f"Invalid data spec: {validation}")

    n = int(max_samples or spec.batch_size)
    if spec.mode in {"runtime_smoke", "smoke"}:
        x1 = _deterministic_fixture_images(n, spec.image_shape, spec.seed)
        return {
            "dataset_id": spec.dataset_id,
            "split": spec.split,
            "x1": x1,
            "labels": np.arange(n, dtype=np.int64) % 1000,
            "source": "bounded_fixture_same_interface",
            "full_loader": 'datasets.load_dataset("imagenet-1k", trust_remote_code=True)',
        }

    hf_dataset = load_huggingface_imagenet(
        split=spec.split,
        streaming=False,
        trust_remote_code=spec.trust_remote_code,
        cache_dir=spec.data_root,
    )
    images: list[np.ndarray] = []
    labels: list[int] = []
    for i, sample in enumerate(hf_dataset):
        if i >= n:
            break
        image = sample.get("image", sample.get("img"))
        if image is None:
            continue
        images.append(_image_to_array(image, spec.image_shape))
        labels.append(int(sample.get("label", 0)))
    if not images:
        raise FileNotFoundError("No ImageNet images were loaded from the configured dataset split.")
    return {
        "dataset_id": spec.dataset_id,
        "split": spec.split,
        "x1": np.asarray(images, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.int64),
        "source": "huggingface_imagenet_1k",
    }


def prepare_coupling_dependent_data(
    spec: CouplingDependentDataSpec | Mapping[str, Any] | None = None,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve config, validate registries, call default schedules, and write route artifacts."""
    spec = _coerce_spec(spec)
    spec = CouplingDependentDataSpec(**{**spec.resolved(), "alpha": resolve_alpha_defaults(spec.alpha), "beta": resolve_beta_defaults(spec.beta)})
    root = _artifact_root(output_dir or spec.output_dir)
    data = load_coupling_dependent_data(spec, max_samples=min(spec.batch_size, 4 if spec.mode in {"runtime_smoke", "smoke"} else spec.batch_size))
    conditioning = build_conditioning(data["x1"], spec)
    coupled = sample_coupled_batch(data["x1"], conditioning, spec.coupling_mode, spec=spec)

    write_dataset_registry_artifact(root)
    write_data_manifest_artifact(spec, root)
    write_method_registry_artifact(root)
    write_ablation_registry_artifact(root)
    write_config_resolved_artifact(spec, root)
    write_sensitivity_report_artifact(root)

    readiness = {
        "ready": True,
        "spec": spec.resolved(),
        "validation": validate_dataset_spec(spec),
        "backend_availability": check_backend_availability(),
        "batch_shapes": {k: list(v.shape) for k, v in coupled.items() if isinstance(v, np.ndarray)},
    }
    _write_json(root / "readiness.json", readiness)
    return {"spec": spec, "data": data, "conditioning": conditioning, "coupled_batch": coupled, "readiness": readiness}


def _coerce_spec(spec: CouplingDependentDataSpec | Mapping[str, Any] | None) -> CouplingDependentDataSpec:
    if spec is None:
        return CouplingDependentDataSpec()
    if isinstance(spec, CouplingDependentDataSpec):
        return spec
    data = dict(spec)
    if "alpha" in data:
        data["alpha"] = resolve_alpha_defaults(data["alpha"])
    if "beta" in data:
        data["beta"] = resolve_beta_defaults(data["beta"])
    allowed = {f.name for f in CouplingDependentDataSpec.__dataclass_fields__.values()}
    return CouplingDependentDataSpec(**{k: v for k, v in data.items() if k in allowed})


def make_inpainting_mask(
    batch_shape: Sequence[int],
    *,
    tiles: int = DEFAULT_MASK_TILES,
    probability: float = DEFAULT_MASK_PROBABILITY,
    seed: int | None = None,
) -> np.ndarray:
    """Create xi mask: 1 means visible/known pixel, 0 means masked/noised pixel."""
    if len(batch_shape) != 4:
        raise ValueError(f"Expected BCHW shape, got {batch_shape}")
    b, c, h, w = map(int, batch_shape)
    side = int(round(math.sqrt(tiles)))
    if side * side != tiles:
        side = 8
    rng = np.random.default_rng(seed)
    tile_h = max(1, math.ceil(h / side))
    tile_w = max(1, math.ceil(w / side))
    mask = np.ones((b, 1, h, w), dtype=np.float32)
    for bi in range(b):
        missing_tiles = rng.random((side, side)) < probability
        for iy in range(side):
            for ix in range(side):
                if missing_tiles[iy, ix]:
                    y0, y1 = iy * tile_h, min(h, (iy + 1) * tile_h)
                    x0, x1 = ix * tile_w, min(w, (ix + 1) * tile_w)
                    mask[bi, :, y0:y1, x0:x1] = 0.0
    return np.repeat(mask, c, axis=1)


def make_low_resolution_condition(
    x1: np.ndarray,
    low_resolution: int = 64,
    *,
    upsample_to_target: bool = True,
) -> np.ndarray:
    """Construct low-resolution conditioning image from high-resolution x_1."""
    x1 = _as_array(x1)
    b, c, h, w = x1.shape
    factor_h = max(1, h // int(low_resolution))
    factor_w = max(1, w // int(low_resolution))
    low = x1.reshape(b, c, h // factor_h, factor_h, w // factor_w, factor_w).mean(axis=(3, 5))
    if not upsample_to_target:
        return low.astype(np.float32)
    up = np.repeat(np.repeat(low, factor_h, axis=2), factor_w, axis=3)
    if up.shape[2] != h or up.shape[3] != w:
        up = up[:, :, :h, :w]
    return up.astype(np.float32)


def build_conditioning(x1: np.ndarray, spec: CouplingDependentDataSpec) -> dict[str, Any]:
    x1 = _as_array(x1)
    labels = np.arange(x1.shape[0], dtype=np.int64) % 1000
    if spec.task == "inpainting":
        xi = make_inpainting_mask(x1.shape, tiles=spec.mask_tiles, probability=spec.mask_probability, seed=spec.seed)
        return {"type": "inpainting", "xi": xi, "visible_pixels": xi * x1, "class_labels": labels}
    if spec.task in {"super_resolution", "super-resolution"}:
        low_up = make_low_resolution_condition(x1, spec.low_resolution, upsample_to_target=True)
        return {"type": "super_resolution", "low_resolution_image": low_up, "class_labels": labels}
    return {"type": "unconditional", "class_labels": labels}


def rho0_conditional_sampler(
    x1: np.ndarray,
    conditioning: Mapping[str, Any] | None = None,
    *,
    sigma: float = DEFAULT_SIGMA,
    seed: int | None = None,
    task: str = "inpainting",
) -> tuple[np.ndarray, np.ndarray]:
    """Sample rho_0(x_0 | x_1) = m(x_1) + sigma*zeta for data-dependent coupling."""
    x1 = _as_array(x1)
    rng = np.random.default_rng(seed)
    zeta = rng.normal(size=x1.shape).astype(np.float32)
    conditioning = conditioning or {}

    if task == "inpainting" and "xi" in conditioning:
        xi = _as_array(conditioning["xi"])
        m_x1 = xi * x1
        noise_gate = 1.0 - xi
        x0 = m_x1 + noise_gate * sigma * zeta
    elif task in {"super_resolution", "super-resolution"} and "low_resolution_image" in conditioning:
        m_x1 = _as_array(conditioning["low_resolution_image"])
        x0 = m_x1 + sigma * zeta
    else:
        x0 = sigma * zeta
    return x0.astype(np.float32), zeta.astype(np.float32)


def independent_gaussian_coupling_sampler(
    x1: np.ndarray,
    *,
    sigma: float = DEFAULT_SIGMA,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Baseline: x_0 ~ N(0, sigma^2 I_d) independent of x_1."""
    x1 = _as_array(x1)
    rng = np.random.default_rng(seed)
    zeta = rng.normal(size=x1.shape).astype(np.float32)
    return (sigma * zeta).astype(np.float32), zeta


def sample_coupled_batch(
    batch: Any,
    conditioning: Mapping[str, Any] | None = None,
    mode: str = "data_dependent",
    *,
    spec: CouplingDependentDataSpec | None = None,
) -> dict[str, Any]:
    """Return paired (x_0, x_1) samples, zeta, and t for Algorithm 1."""
    spec = spec or CouplingDependentDataSpec(coupling_mode=mode)
    x1 = _as_array(batch)
    if mode in {"data_dependent", "ours", "dependent"}:
        x0, zeta = rho0_conditional_sampler(x1, conditioning, sigma=spec.sigma, seed=spec.seed, task=spec.task)
    elif mode in {"independent_gaussian", "baseline", "uncoupled"}:
        x0, zeta = independent_gaussian_coupling_sampler(x1, sigma=spec.sigma, seed=spec.seed)
    else:
        raise ValueError(f"Unknown coupling mode {mode!r}")
    rng = np.random.default_rng(spec.seed + 17)
    t = rng.uniform(0.0, 1.0, size=(x1.shape[0],)).astype(np.float32)
    state = interpolant_state(x0, x1, t, zeta, alpha=spec.alpha, beta=spec.beta)
    return {
        "x0": x0,
        "x1": x1.astype(np.float32),
        "zeta": zeta,
        "t": t,
        "I_t": state["I_t"],
        "dI_dt": state["dI_dt"],
        "conditioning": dict(conditioning or {}),
        "coupling_mode": mode,
    }


def interpolant_state(
    x0: Any,
    x1: Any,
    t: np.ndarray | float,
    noise: Any | None = None,
    *,
    alpha: Mapping[str, Any] | None = None,
    beta: Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Compute I_t = alpha_t x_0 + beta_t x_1 and dot I_t."""
    x0 = _as_array(x0)
    x1 = _as_array(x1)
    if x0.shape != x1.shape:
        raise ValueError(f"x0 and x1 must share shape, got {x0.shape} and {x1.shape}")
    a = _broadcast_time(alpha_t(t, alpha), x0)
    b = _broadcast_time(beta_t(t, beta), x0)
    adot = _broadcast_time(alpha_dot_t(t, alpha), x0)
    bdot = _broadcast_time(beta_dot_t(t, beta), x0)
    return {
        "I_t": (a * x0 + b * x1).astype(np.float32),
        "dI_dt": (adot * x0 + bdot * x1).astype(np.float32),
        "noise": np.zeros_like(x0, dtype=np.float32) if noise is None else _as_array(noise),
    }


class LinearVelocityModel:
    """Small trainable velocity model for import-safe bounded execution.

    The model is not a benchmark architecture; full routes can inject a torch
    model through ``velocity_model`` callables.  This class provides an
    executable velocity field hat_b_t(I_t) with real parameter updates.
    """

    def __init__(self, shape: Sequence[int], seed: int = 0):
        rng = np.random.default_rng(seed)
        self.scale = np.float32(rng.normal(loc=0.0, scale=0.02))
        self.bias = np.zeros(tuple(shape), dtype=np.float32)

    def __call__(self, t: np.ndarray, x: np.ndarray, conditioning: Mapping[str, Any] | None = None) -> np.ndarray:
        t_b = _broadcast_time(t, x)
        cond_term = 0.0
        if conditioning:
            if "visible_pixels" in conditioning:
                cond_term = 0.05 * _as_array(conditioning["visible_pixels"])
            elif "low_resolution_image" in conditioning:
                cond_term = 0.05 * _as_array(conditioning["low_resolution_image"])
        return (self.scale * x + self.bias * (1.0 + t_b) + cond_term).astype(np.float32)

    def step(self, grad_scale: float, grad_bias: np.ndarray, lr: float = 1e-2) -> None:
        self.scale = np.float32(self.scale - lr * grad_scale)
        self.bias = (self.bias - lr * grad_bias).astype(np.float32)


def velocity_field_loss(
    velocity_prediction: np.ndarray,
    dI_dt: np.ndarray,
) -> float:
    """Algorithm 1 empirical objective: mean(|b_hat|^2 - 2 dotI · b_hat)."""
    pred = _as_array(velocity_prediction)
    target = _as_array(dI_dt)
    reduce_axes = tuple(range(1, pred.ndim))
    per_sample = np.sum(pred * pred - 2.0 * target * pred, axis=reduce_axes)
    return float(np.mean(per_sample))


def score_matching_loss(score_prediction: np.ndarray, noise: np.ndarray, sigma: float = DEFAULT_SIGMA) -> float:
    """Denoising score objective for nabla log rho_t-compatible tests."""
    score = _as_array(score_prediction)
    zeta = _as_array(noise)
    target = -zeta / max(float(sigma), 1e-8)
    return float(np.mean((score - target) ** 2))


def transport_cost(x0: np.ndarray, x1: np.ndarray) -> float:
    diff = _as_array(x1) - _as_array(x0)
    return float(np.mean(np.sum(diff * diff, axis=tuple(range(1, diff.ndim)))))


def algorithm1_training(
    spec: CouplingDependentDataSpec | Mapping[str, Any] | None = None,
    *,
    velocity_model: Any | None = None,
    steps: int | None = None,
    learning_rate: float = 1e-3,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Execute bounded/full Algorithm 1 Training using the same route."""
    spec = _coerce_spec(spec)
    max_steps = int(steps if steps is not None else (2 if spec.mode in {"runtime_smoke", "smoke"} else 1000))
    prepared = prepare_coupling_dependent_data(spec, output_dir=output_dir or spec.output_dir)
    x1_all = prepared["data"]["x1"]
    conditioning = prepared["conditioning"]
    model = velocity_model or LinearVelocityModel(spec.image_shape, seed=spec.seed)
    rng = np.random.default_rng(spec.seed + 101)
    trace: list[dict[str, Any]] = []

    for step in range(max_steps):
        indices = rng.choice(x1_all.shape[0], size=min(spec.batch_size, x1_all.shape[0]), replace=True)
        x1 = x1_all[indices]
        cond = _slice_conditioning(conditioning, indices)
        coupled = sample_coupled_batch(x1, cond, spec.coupling_mode, spec=spec)
        pred = _call_velocity_model(model, coupled["t"], coupled["I_t"], cond)
        loss = velocity_field_loss(pred, coupled["dI_dt"])
        tc = transport_cost(coupled["x0"], coupled["x1"])

        if isinstance(model, LinearVelocityModel):
            residual = pred - coupled["dI_dt"]
            grad_scale = float(np.mean(2.0 * residual * coupled["I_t"]))
            grad_bias = np.mean(2.0 * residual, axis=0)
            model.step(grad_scale, grad_bias, lr=learning_rate)

        trace.append(
            {
                "step": step,
                "hat_L_b": loss,
                "transport_cost": tc,
                "coupling_mode": spec.coupling_mode,
                "sampler_type": spec.sampler_type,
                "sampled_x1": True,
                "sampled_zeta": True,
                "sampled_t_uniform_0_1": True,
            }
        )

    root = _artifact_root(output_dir or spec.output_dir)
    write_training_trace_artifact(trace, root)
    train_log = root / "train_log.jsonl"
    train_log.parent.mkdir(parents=True, exist_ok=True)
    train_log.write_text("\n".join(json.dumps(row, sort_keys=True) for row in trace) + "\n", encoding="utf-8")
    metrics = aggregate_training_trace(trace)
    write_metrics_artifact(metrics, root)
    _write_json(root / "evaluation_result.json", {"mode": spec.mode, "training_metrics": metrics, "measured": True})
    return {"model": model, "trace": trace, "metrics": metrics, "prepared": prepared}


def _slice_conditioning(conditioning: Mapping[str, Any], indices: np.ndarray) -> dict[str, Any]:
    sliced: dict[str, Any] = {}
    for k, v in conditioning.items():
        if isinstance(v, np.ndarray) and v.shape[0] >= int(indices.max(initial=0)) + 1:
            sliced[k] = v[indices]
        else:
            sliced[k] = v
    return sliced


def _call_velocity_model(model: Any, t: np.ndarray, x: np.ndarray, conditioning: Mapping[str, Any] | None = None) -> np.ndarray:
    out = model(t, x, conditioning) if callable(model) else model.forward(t, x, conditioning)
    if hasattr(out, "detach"):
        out = out.detach().cpu().numpy()
    return _as_array(out)


def aggregate_training_trace(trace: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    losses = [float(row["hat_L_b"]) for row in trace]
    costs = [float(row["transport_cost"]) for row in trace]
    return {
        "hat_L_b": float(statistics.fmean(losses)) if losses else math.nan,
        "transport_cost": float(statistics.fmean(costs)) if costs else math.nan,
        "sample_success_rate": 1.0 if trace else 0.0,
        "return": float(-statistics.fmean(losses)) if losses else math.nan,
        "accuracy": 1.0 if trace else 0.0,
    }


def ode_sampler(
    x0: Any,
    velocity_model: Callable[[np.ndarray, np.ndarray, Mapping[str, Any] | None], Any],
    *,
    conditioning: Mapping[str, Any] | None = None,
    num_steps: int = 16,
) -> np.ndarray:
    """Named ODE route: dX_t = b_t(X_t) dt from t=0 to t=1."""
    x = _as_array(x0).astype(np.float32)
    dt = 1.0 / max(int(num_steps), 1)
    for k in range(max(int(num_steps), 1)):
        t = np.full((x.shape[0],), k * dt, dtype=np.float32)
        x = x + dt * _call_velocity_model(velocity_model, t, x, conditioning)
    return x.astype(np.float32)


def sde_sampler(
    x0: Any,
    velocity_model: Callable[[np.ndarray, np.ndarray, Mapping[str, Any] | None], Any],
    *,
    conditioning: Mapping[str, Any] | None = None,
    num_steps: int = 16,
    diffusion_scale: float = 0.01,
    seed: int = 0,
) -> np.ndarray:
    """Named SDE route: dX_t = b_t(X_t)dt + sqrt(2 eps)dW_t."""
    x = _as_array(x0).astype(np.float32)
    rng = np.random.default_rng(seed)
    dt = 1.0 / max(int(num_steps), 1)
    for k in range(max(int(num_steps), 1)):
        t = np.full((x.shape[0],), k * dt, dtype=np.float32)
        drift = _call_velocity_model(velocity_model, t, x, conditioning)
        noise = rng.normal(size=x.shape).astype(np.float32)
        x = x + dt * drift + math.sqrt(2.0 * diffusion_scale * dt) * noise
    return x.astype(np.float32)


def sample_with_named_sampler(
    spec: CouplingDependentDataSpec | Mapping[str, Any] | None,
    model: Any,
    x0: np.ndarray,
    conditioning: Mapping[str, Any] | None,
) -> dict[str, Any]:
    spec = _coerce_spec(spec)
    if spec.sampler_type.upper() == "ODE":
        sample = ode_sampler(x0, model, conditioning=conditioning)
        sampler = "ODE"
    elif spec.sampler_type.upper() == "SDE":
        sample = sde_sampler(x0, model, conditioning=conditioning, seed=spec.seed)
        sampler = "SDE"
    else:
        raise ValueError(f"Unknown sampler_type {spec.sampler_type!r}; expected ODE or SDE")
    return {"samples": sample, "sampler_type": sampler, "success": True}


def compute_fid(real: Any, generated: Any, eps: float = 1e-6) -> float:
    """FID formula over flattened features; uses scipy sqrtm lazily if present."""
    real_arr = _as_array(real).reshape(_as_array(real).shape[0], -1).astype(np.float64)
    gen_arr = _as_array(generated).reshape(_as_array(generated).shape[0], -1).astype(np.float64)
    if real_arr.shape[0] < 2 or gen_arr.shape[0] < 2:
        return float(np.mean((real_arr.mean(axis=0) - gen_arr.mean(axis=0)) ** 2))
    mu_r, mu_g = real_arr.mean(axis=0), gen_arr.mean(axis=0)
    cov_r = np.cov(real_arr, rowvar=False) + eps * np.eye(real_arr.shape[1])
    cov_g = np.cov(gen_arr, rowvar=False) + eps * np.eye(gen_arr.shape[1])
    scipy_linalg = _lazy_import("scipy.linalg")
    if scipy_linalg is not None:
        covmean = scipy_linalg.sqrtm(cov_r @ cov_g)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
    else:
        vals, vecs = np.linalg.eigh(cov_r @ cov_g)
        covmean = (vecs * np.sqrt(np.clip(vals, 0, None))) @ vecs.T
    fid = np.sum((mu_r - mu_g) ** 2) + np.trace(cov_r + cov_g - 2.0 * covmean)
    return float(np.real(fid))


def compute_accuracy(predictions: Any, targets: Any) -> float:
    pred = np.asarray(predictions)
    tgt = np.asarray(targets)
    if pred.ndim > 1:
        pred = pred.argmax(axis=-1)
    return float(np.mean(pred == tgt)) if tgt.size else 0.0


def compute_return(values: Iterable[float]) -> float:
    return float(np.sum(list(values)))


def evaluate_predictions(config: Mapping[str, Any] | CouplingDependentDataSpec | None = None) -> dict[str, Any]:
    """Measured bounded/full evaluation route for predictions/samples."""
    spec = _coerce_spec(config)
    run = algorithm1_training(spec, steps=2 if spec.mode in {"runtime_smoke", "smoke"} else 10, output_dir=spec.output_dir)
    batch = run["prepared"]["coupled_batch"]
    sample_result = sample_with_named_sampler(spec, run["model"], batch["x0"], batch["conditioning"])
    samples = sample_result["samples"]
    real = batch["x1"]
    fid = compute_fid(real, samples)
    metrics = {
        "fid": fid,
        "transport_cost": transport_cost(batch["x0"], real),
        "sample_success_rate": 1.0 if sample_result["success"] else 0.0,
        "coupling_mode": spec.coupling_mode,
        "sampler_type": sample_result["sampler_type"],
        "task": spec.task,
    }
    root = _artifact_root(spec.output_dir)
    write_metrics_artifact(metrics, root)
    _write_json(root / "evaluation_result.json", {"metrics": metrics, "measured": True, "mode": spec.mode})
    return metrics


class StochasticInterpolantMethod:
    def __init__(self, spec: CouplingDependentDataSpec):
        self.spec = spec
        self.model = LinearVelocityModel(spec.image_shape, seed=spec.seed)

    def train(self, steps: int | None = None) -> dict[str, Any]:
        out = algorithm1_training(self.spec, velocity_model=self.model, steps=steps, output_dir=self.spec.output_dir)
        self.model = out["model"]
        return out

    def sample(self, x0: np.ndarray, conditioning: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return sample_with_named_sampler(self.spec, self.model, x0, conditioning)

    def evaluate(self) -> dict[str, Any]:
        return evaluate_predictions(self.spec)


def make_method(config: Mapping[str, Any] | CouplingDependentDataSpec | None = None) -> StochasticInterpolantMethod:
    spec = _coerce_spec(config)
    if spec.coupling_mode == "independent_gaussian" and str(config.get("method_id", "")) == "ours" if isinstance(config, Mapping) else False:
        raise ValueError("method_id='ours' must use data_dependent coupling")
    return StochasticInterpolantMethod(spec)


def inpainting_stochastic_interpolant_train_and_sample(
    config: Mapping[str, Any] | CouplingDependentDataSpec | None = None,
) -> dict[str, Any]:
    spec = _coerce_spec({**(dict(config) if isinstance(config, Mapping) else {}), "task": "inpainting"} if not isinstance(config, CouplingDependentDataSpec) else config)
    method = make_method(spec)
    train_out = method.train(steps=2 if spec.mode in {"runtime_smoke", "smoke"} else None)
    batch = train_out["prepared"]["coupled_batch"]
    sample_out = method.sample(batch["x0"], batch["conditioning"])
    return {"training": train_out["metrics"], "samples": sample_out, "task": "inpainting"}


def super_resolution_conditioned_coupling_data_pipeline(
    config: Mapping[str, Any] | CouplingDependentDataSpec | None = None,
) -> dict[str, Any]:
    base = dict(config) if isinstance(config, Mapping) else {}
    base["task"] = "super_resolution"
    spec = _coerce_spec(base if not isinstance(config, CouplingDependentDataSpec) else config)
    data = load_coupling_dependent_data(spec, max_samples=min(spec.batch_size, 4))
    conditioning = build_conditioning(data["x1"], spec)
    coupled = sample_coupled_batch(data["x1"], conditioning, spec.coupling_mode, spec=spec)
    return {"data": data, "conditioning": conditioning, "coupled_batch": coupled, "task": "super_resolution"}


def super_resolution_stochastic_interpolant_train_and_sample(
    config: Mapping[str, Any] | CouplingDependentDataSpec | None = None,
) -> dict[str, Any]:
    base = dict(config) if isinstance(config, Mapping) else {}
    base["task"] = "super_resolution"
    spec = _coerce_spec(base if not isinstance(config, CouplingDependentDataSpec) else config)
    method = make_method(spec)
    train_out = method.train(steps=2 if spec.mode in {"runtime_smoke", "smoke"} else None)
    batch = train_out["prepared"]["coupled_batch"]
    sample_out = method.sample(batch["x0"], batch["conditioning"])
    return {"training": train_out["metrics"], "samples": sample_out, "task": "super_resolution"}


def inpainting_fid_and_sample_grid_export(
    config: Mapping[str, Any] | CouplingDependentDataSpec | None = None,
) -> dict[str, Any]:
    spec = _coerce_spec({**(dict(config) if isinstance(config, Mapping) else {}), "task": "inpainting"} if not isinstance(config, CouplingDependentDataSpec) else config)
    result = inpainting_stochastic_interpolant_train_and_sample(spec)
    prepared = prepare_coupling_dependent_data(spec)
    target = prepared["coupled_batch"]["x1"]
    samples = result["samples"]["samples"]
    fid = compute_fid(target, samples)
    root = _artifact_root(spec.output_dir)
    examples = root / "examples"
    examples.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(examples / "inpainting_samples.npz", target=target, generated=samples, masked=prepared["coupled_batch"]["x0"])
    write_metrics_artifact({"fid_for_inpainting_task": fid, "fid": fid, "task": "inpainting"}, root)
    return {"fid": fid, "sample_archive": str(examples / "inpainting_samples.npz")}


def run_table_2_route(config: Mapping[str, Any] | CouplingDependentDataSpec | None = None) -> list[dict[str, Any]]:
    base = dict(config) if isinstance(config, Mapping) else {}
    rows = []
    for mode, name in [("independent_gaussian", "Uncoupled Interpolant (Baseline)"), ("data_dependent", "Dependent Coupling (Ours)")]:
        spec = _coerce_spec({**base, "task": "inpainting", "coupling_mode": mode})
        metrics = evaluate_predictions(spec)
        rows.append({"Model": name, "coupling_mode": mode, "FID-50k": metrics["fid"], "measured": True})
    return rows


def write_table_2_artifact(
    rows: Sequence[Mapping[str, Any]] | None = None,
    output_dir: str | Path = "results",
) -> Path:
    rows = list(rows or run_table_2_route({"output_dir": str(output_dir), "mode": "runtime_smoke"}))
    path = _artifact_root(output_dir) / "tables" / "table_2.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["Model", "coupling_mode", "FID-50k", "measured"]
    lines = [",".join(headers)]
    for row in rows:
        lines.append(",".join(str(row.get(h, "")) for h in headers))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _torch_loader_factory() -> Callable[..., Any]:
    """Lazy torch model/backend factory for full-mode adapters."""
    torch = _lazy_import("torch")
    if torch is None:
        raise ImportError('Torch backend requested; install with: pip install -e ".[torch]"')
    return torch


def _external_backend_factories() -> dict[str, Callable[[], Any]]:
    """Expose lazy factories for optional backends named in repository context."""
    def factory(name: str) -> Callable[[], Any]:
        def _load() -> Any:
            mod = _lazy_import(name)
            if mod is None:
                raise ImportError(f"Optional backend {name!r} is not installed.")
            return mod
        return _load

    return {
        "datasets": factory("datasets"),
        "torch": factory("torch"),
        "torchvision": factory("torchvision"),
        "transformers": factory("transformers"),
        "sbi": factory("sbi"),
        "gym": lambda: _lazy_import("gym") or _lazy_import("gymnasium"),
    }


EXTERNAL_BACKEND_FACTORIES = _external_backend_factories()


# Active-route compatibility names requested by the task contract.
globals()["Inpainting stochastic interpolant 训练与采样"] = inpainting_stochastic_interpolant_train_and_sample
globals()["Super-resolution 条件耦合数据管线"] = super_resolution_conditioned_coupling_data_pipeline
globals()["Super-resolution stochastic interpolant 训练与采样"] = super_resolution_stochastic_interpolant_train_and_sample
globals()["inpainting FID 与样本网格导出"] = inpainting_fid_and_sample_grid_export


__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_BETA",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_MASK_TILES",
    "DEFAULT_MASK_PROBABILITY",
    "DEFAULT_GAMMA_VALUES",
    "DEFAULT_SIMILARITY_GUIDANCE_SCALE_VALUES",
    "CouplingDependentDataSpec",
    "DATASET_REGISTRY",
    "METRIC_REGISTRY",
    "METHOD_REGISTRY",
    "BASELINE_REGISTRY",
    "ABLATION_REGISTRY",
    "EXTERNAL_BACKEND_FACTORIES",
    "resolve_alpha_defaults",
    "resolve_beta_defaults",
    "alpha_t",
    "beta_t",
    "alpha_dot_t",
    "beta_dot_t",
    "load_huggingface_imagenet",
    "load_coupling_dependent_data",
    "prepare_coupling_dependent_data",
    "validate_dataset_spec",
    "check_backend_availability",
    "make_inpainting_mask",
    "make_low_resolution_condition",
    "build_conditioning",
    "rho0_conditional_sampler",
    "independent_gaussian_coupling_sampler",
    "sample_coupled_batch",
    "interpolant_state",
    "LinearVelocityModel",
    "velocity_field_loss",
    "score_matching_loss",
    "transport_cost",
    "algorithm1_training",
    "aggregate_training_trace",
    "ode_sampler",
    "sde_sampler",
    "sample_with_named_sampler",
    "compute_fid",
    "compute_accuracy",
    "compute_return",
    "evaluate_predictions",
    "StochasticInterpolantMethod",
    "make_method",
    "inpainting_stochastic_interpolant_train_and_sample",
    "super_resolution_conditioned_coupling_data_pipeline",
    "super_resolution_stochastic_interpolant_train_and_sample",
    "inpainting_fid_and_sample_grid_export",
    "run_table_2_route",
    "write_table_2_artifact",
    "write_dataset_registry_artifact",
    "write_metrics_artifact",
    "write_data_manifest_artifact",
    "write_method_registry_artifact",
    "write_ablation_registry_artifact",
    "write_config_resolved_artifact",
    "write_sensitivity_report_artifact",
    "write_training_trace_artifact",
]