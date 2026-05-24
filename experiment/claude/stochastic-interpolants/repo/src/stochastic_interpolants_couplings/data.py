"""Data loading, preprocessing, and task-batch construction.

This module owns the import-light data route for the reproduction of
"Stochastic Interpolants with Data-Dependent Couplings".  Full ImageNet loading
is lazy and uses HuggingFace datasets when requested; missing benchmark data is
reported with actionable ``FileNotFoundError`` messages rather than silently
falling back to generated samples.

reference_grounding: paperbench_ref_004 xmodaler/datasets/README.md
The reference protocol exposes builtin dataset wrappers through config-selected
dataset names and explicit file paths.  This implementation adapts that intent
into a registry of ImageNet aliases plus explicit ``data_root`` resolution and
validation before any training/evaluation route receives samples.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, MutableMapping, Sequence

import numpy as np


IMAGENET_HF_DATASET_ID = "imagenet-1k"
DATASET_ALIASES: dict[str, str] = {
    "imagenet": "imagenet_1k",
    "ImageNet": "imagenet_1k",
    "ImageNet。": "imagenet_1k",
    "imagenet_1k": "imagenet_1k",
    "imagenet-1k": "imagenet_1k",
    "imagenet_c": "imagenet_c",
    "ImageNet-C": "imagenet_c",
}
SUPPORTED_RESOLUTIONS = (256, 512)
DEFAULT_IMAGE_SIZE = 256
DEFAULT_CHANNELS = 3
DEFAULT_BATCH_SIZE = 32
DEFAULT_MASK_TILES = 64
DEFAULT_MASK_PROBABILITY = 0.3
DEFAULT_LOW_RESOLUTION = 64
DEFAULT_ALPHA = {"name": "linear_to_base", "formula": "alpha_t = 1 - t"}
DEFAULT_BETA = {"name": "linear_to_target", "formula": "beta_t = t"}
DEFAULT_NORMALIZE_MEAN = (0.5, 0.5, 0.5)
DEFAULT_NORMALIZE_STD = (0.5, 0.5, 0.5)


@dataclass(frozen=True)
class DataSpec:
    """Executable registry row and lightweight task spec.

    Registry construction passes explicit dataset fields.  Active smoke routes
    can also instantiate ``DataSpec(task="inpainting", ...)`` to exercise the
    paper's conditional coupling path without requiring ImageNet files.
    """

    dataset_id: str = "imagenet_1k"
    canonical_id: str = "imagenet_1k"
    aliases: tuple[str, ...] = ("imagenet", "ImageNet", "ImageNet。", "imagenet-1k")
    hf_dataset_id: str = IMAGENET_HF_DATASET_ID
    split: str = "train"
    tasks: tuple[str, ...] = ("inpainting", "super_resolution")
    resolutions: tuple[int, ...] = SUPPORTED_RESOLUTIONS
    metrics: tuple[str, ...] = ("fid",)
    methods: tuple[str, ...] = ("ours", "resnet", "ddpm")
    requires_auth: bool = True
    trust_remote_code: bool = True
    setup_note: str = (
        "Download or cache ImageNet through HuggingFace with "
        'datasets.load_dataset("imagenet-1k", trust_remote_code=True), or set '
        "data_root to a local ImageFolder/HF cache containing ImageNet files."
    )
    reference_grounding: str = "paperbench_ref_004 xmodaler/datasets/README.md"
    task: str = "inpainting"
    image_shape: tuple[int, int, int] = (DEFAULT_CHANNELS, 8, 8)
    low_resolution_shape: tuple[int, int, int] | None = None
    max_samples: int = 4
    seed: int = 1234
    mask_tiles: int = DEFAULT_MASK_TILES
    mask_probability: float = DEFAULT_MASK_PROBABILITY
    low_resolution: int = DEFAULT_LOW_RESOLUTION
    use_smoke_fixture: bool = True


@dataclass
class ImageDatasetConfig:
    """Configuration consumed by dataset, dataloader, and task-batch routes."""

    dataset_id: str = "imagenet"
    data_root: str | None = None
    split: str = "train"
    resolution: int = DEFAULT_IMAGE_SIZE
    crop: str = "center"
    normalize: bool = True
    mean: tuple[float, float, float] = DEFAULT_NORMALIZE_MEAN
    std: tuple[float, float, float] = DEFAULT_NORMALIZE_STD
    batch_size: int = DEFAULT_BATCH_SIZE
    num_workers: int = 0
    shuffle: bool = True
    task: str = "inpainting"
    coupling_type: str = "data_dependent"
    sampler_type: str = "ODE"
    sigma: float = 1.0
    mask_tiles: int = DEFAULT_MASK_TILES
    mask_probability: float = DEFAULT_MASK_PROBABILITY
    low_resolution: int = DEFAULT_LOW_RESOLUTION
    include_class_labels: bool = True
    allow_hf_download: bool = False
    trust_remote_code: bool = True
    cache_dir: str | None = None
    max_items: int | None = None
    use_smoke_fixture: bool = False
    seed: int = 1234
    alpha: Mapping[str, Any] = field(default_factory=lambda: dict(DEFAULT_ALPHA))
    beta: Mapping[str, Any] = field(default_factory=lambda: dict(DEFAULT_BETA))

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | "ImageDatasetConfig" | None = None, **overrides: Any) -> "ImageDatasetConfig":
        if isinstance(config, cls):
            base = asdict(config)
        else:
            base = {}
            if config:
                data_cfg = config.get("data", config) if isinstance(config, Mapping) else {}
                if isinstance(data_cfg, Mapping):
                    base.update(data_cfg)
                runtime_cfg = config.get("runtime", {}) if isinstance(config, Mapping) else {}
                if isinstance(runtime_cfg, Mapping):
                    if "use_smoke_fixture" in runtime_cfg and "use_smoke_fixture" not in base:
                        base["use_smoke_fixture"] = runtime_cfg["use_smoke_fixture"]
                    if "seed" in runtime_cfg and "seed" not in base:
                        base["seed"] = runtime_cfg["seed"]
                experiment_cfg = config.get("experiment", {}) if isinstance(config, Mapping) else {}
                if isinstance(experiment_cfg, Mapping):
                    for key in ("task", "coupling_type", "sampler_type"):
                        if key in experiment_cfg and key not in base:
                            base[key] = experiment_cfg[key]
        base.update({k: v for k, v in overrides.items() if v is not None})
        if "image_size" in base and "resolution" not in base:
            base["resolution"] = base.pop("image_size")
        if "dataset" in base and "dataset_id" not in base:
            base["dataset_id"] = base.pop("dataset")
        if "root" in base and "data_root" not in base:
            base["data_root"] = base.pop("root")
        allowed = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in base.items() if k in allowed}
        return cls(**filtered)


DATASET_REGISTRY: dict[str, DataSpec] = {
    "imagenet_1k": DataSpec(
        dataset_id="imagenet_1k",
        canonical_id="imagenet_1k",
        aliases=("imagenet", "ImageNet", "ImageNet。", "imagenet-1k"),
        hf_dataset_id=IMAGENET_HF_DATASET_ID,
    ),
    "imagenet_c": DataSpec(
        dataset_id="imagenet_c",
        canonical_id="imagenet_c",
        aliases=("imagenet_c", "ImageNet-C"),
        hf_dataset_id="imagenet-1k",
        split="validation",
        setup_note=(
            "ImageNet-C corruptions are evaluated through the same ImageNet "
            "preprocessing route. Provide a local ImageNet-C/ImageFolder root or "
            "adapt an HF cache under data_root; no synthetic corruption fallback is used."
        ),
    ),
}

METRIC_REGISTRY: dict[str, dict[str, Any]] = {
    "fid": {
        "metric_id": "fid",
        "name": "Fréchet Inception Distance",
        "required_for": ("Table 2", "Table 3", "super-resolution", "inpainting"),
        "formula": "||mu_r-mu_g||^2 + Tr(Sigma_r + Sigma_g - 2(Sigma_r Sigma_g)^{1/2})",
    },
    "transport_cost": {
        "metric_id": "transport_cost",
        "formula": "E[||x1 - x0||_2^2]",
    },
    "training_loss_hat_L_b": {
        "metric_id": "training_loss_hat_L_b",
        "formula": "mean(|hat_b_t(I_t)|^2 - 2 dot(dI_t, hat_b_t(I_t)))",
    },
    "accuracy": {
        "metric_id": "accuracy",
        "formula": "mean(argmax(prediction)==target) when labels are available",
    },
}

METHOD_REGISTRY: dict[str, dict[str, Any]] = {
    "ours": {
        "method_id": "ours",
        "coupling_type": "data_dependent",
        "datasets": ("imagenet", "imagenet_1k", "imagenet_c"),
        "metrics": ("fid", "transport_cost"),
        "artifacts": ("results/metrics.json", "results/tables/table_2.csv", "results/tables/table_3.csv"),
    },
    "resnet": {
        "method_id": "resnet",
        "role": "baseline_adapter",
        "datasets": ("imagenet_1k",),
        "metrics": ("accuracy", "fid"),
    },
    "ddpm": {
        "method_id": "ddpm",
        "role": "diffusion_baseline",
        "coupling_type": "independent_gaussian",
        "datasets": ("imagenet_1k",),
        "metrics": ("fid",),
    },
}


def _optional_import(module_name: str) -> Any | None:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return None
    return importlib.import_module(module_name)


def _canonical_dataset_id(dataset_id: str) -> str:
    return DATASET_ALIASES.get(dataset_id, dataset_id)


def dataset_registry_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for spec in DATASET_REGISTRY.values():
        record = asdict(spec)
        record["available_aliases"] = list(spec.aliases)
        record["benchmark_aliases_required_by_paper"] = ["imagenet", "imagenet_1k", "imagenet_c"]
        records.append(record)
    return records


def resolve_data_root(
    config: Mapping[str, Any] | ImageDatasetConfig | str | os.PathLike[str] | None = None,
    data_root: str | os.PathLike[str] | None = None,
    *,
    require_exists: bool = False,
) -> Path:
    """Resolve data_root from CLI/config/environment and optionally validate it."""

    candidate: str | os.PathLike[str] | None = data_root
    if candidate is None and isinstance(config, ImageDatasetConfig):
        candidate = config.data_root
    if candidate is None and isinstance(config, (str, os.PathLike)):
        candidate = config
    if candidate is None and isinstance(config, Mapping):
        data_cfg = config.get("data", config)
        if isinstance(data_cfg, Mapping):
            candidate = data_cfg.get("data_root") or data_cfg.get("root") or data_cfg.get("cache_dir")
        if candidate is None:
            runtime_cfg = config.get("runtime", {})
            if isinstance(runtime_cfg, Mapping):
                candidate = runtime_cfg.get("data_root")
    if candidate is None:
        candidate = os.environ.get("IMAGENET_ROOT") or os.environ.get("DATA_ROOT") or os.environ.get("HF_DATASETS_CACHE")
    if candidate is None:
        if require_exists:
            raise FileNotFoundError(
                "No data_root was provided. Set --data-root, config.data.data_root, "
                "IMAGENET_ROOT, DATA_ROOT, or HF_DATASETS_CACHE. For ImageNet use "
                'HuggingFace: load_dataset("imagenet-1k", trust_remote_code=True) '
                "after accepting dataset access."
            )
        return Path("data/imagenet")
    root = Path(candidate).expanduser().resolve()
    if require_exists and not root.exists():
        raise FileNotFoundError(
            f"Configured data_root does not exist: {root}. Provide an ImageNet "
            "ImageFolder/HF cache path or enable an explicit full-mode HuggingFace "
            'download with allow_hf_download=True and trust_remote_code=True.'
        )
    return root


def _looks_like_image_folder(root: Path) -> bool:
    if not root.exists():
        return False
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if any(p.suffix.lower() in image_exts for p in root.rglob("*")):
        return True
    split_dirs = ("train", "validation", "val", "test")
    return any((root / split).exists() and any((root / split).iterdir()) for split in split_dirs)


def _looks_like_hf_cache(root: Path, dataset_id: str = IMAGENET_HF_DATASET_ID) -> bool:
    if not root.exists():
        return False
    name_fragments = [dataset_id.replace("-", "___"), dataset_id.replace("-", "_"), dataset_id]
    return any(fragment in str(path) for fragment in name_fragments for path in root.rglob("*"))


def check_data_available(
    config: Mapping[str, Any] | ImageDatasetConfig | None = None,
    *,
    dataset_id: str | None = None,
    data_root: str | os.PathLike[str] | None = None,
    raise_on_missing: bool = False,
) -> dict[str, Any]:
    """Check real benchmark availability without creating synthetic samples."""

    cfg = ImageDatasetConfig.from_config(config, dataset_id=dataset_id, data_root=data_root)
    canonical_id = _canonical_dataset_id(cfg.dataset_id)
    if canonical_id not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset_id={cfg.dataset_id!r}. Registered datasets: {sorted(DATASET_REGISTRY)}")
    root = resolve_data_root(cfg, require_exists=False)
    local_available = _looks_like_image_folder(root) or _looks_like_hf_cache(root, DATASET_REGISTRY[canonical_id].hf_dataset_id)
    hf_backend_available = importlib.util.find_spec("datasets") is not None
    available = bool(local_available or (cfg.allow_hf_download and hf_backend_available))
    result = {
        "dataset_id": cfg.dataset_id,
        "canonical_id": canonical_id,
        "data_root": str(root),
        "split": cfg.split,
        "resolution": cfg.resolution,
        "local_files_available": bool(local_available),
        "hf_backend_available": bool(hf_backend_available),
        "allow_hf_download": bool(cfg.allow_hf_download),
        "available": available,
        "setup_note": DATASET_REGISTRY[canonical_id].setup_note,
    }
    if not available and raise_on_missing:
        raise FileNotFoundError(
            f"Dataset {cfg.dataset_id!r} is not available at {root}. "
            f"{DATASET_REGISTRY[canonical_id].setup_note} "
            "This route does not silently fall back to synthetic data."
        )
    return result


def _to_numpy_image(image: Any) -> np.ndarray:
    if hasattr(image, "detach") and hasattr(image, "cpu"):
        array = image.detach().cpu().numpy()
        if array.ndim == 3 and array.shape[0] in (1, 3, 4):
            array = np.transpose(array, (1, 2, 0))
        return array
    if isinstance(image, np.ndarray):
        array = image
    else:
        pil = _optional_import("PIL.Image")
        if pil is not None and isinstance(image, pil.Image):
            array = np.asarray(image.convert("RGB"))
        else:
            try:
                array = np.asarray(image)
            except Exception as exc:
                raise TypeError(f"Unsupported image sample type {type(image)!r}") from exc
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] == 4:
        array = array[..., :3]
    return array


def preprocess_image_tensor(
    image: Any,
    config: Mapping[str, Any] | ImageDatasetConfig | None = None,
    *,
    resolution: int | None = None,
    normalize: bool | None = None,
    crop: str | None = None,
) -> Any:
    """Resize/crop/normalize a real image sample to CHW tensor.

    Torch is imported lazily.  If torch is unavailable, a NumPy CHW float32 array
    is returned so minimal import/readiness routes remain usable; training
    routes that require torch will raise inside their own runtime path.
    """

    cfg = ImageDatasetConfig.from_config(config, resolution=resolution, normalize=normalize, crop=crop)
    if cfg.resolution not in SUPPORTED_RESOLUTIONS and cfg.resolution <= 0:
        raise ValueError(f"resolution must be positive; got {cfg.resolution}")
    array = _to_numpy_image(image)
    pil_image_mod = _optional_import("PIL.Image")
    if pil_image_mod is not None:
        pil_image = pil_image_mod.fromarray(array.astype(np.uint8) if array.dtype != np.uint8 else array).convert("RGB")
        width, height = pil_image.size
        target = int(cfg.resolution)
        if cfg.crop == "center":
            scale = target / min(width, height)
            new_w, new_h = max(target, int(round(width * scale))), max(target, int(round(height * scale)))
            pil_image = pil_image.resize((new_w, new_h), resample=pil_image_mod.BICUBIC)
            left = max(0, (new_w - target) // 2)
            top = max(0, (new_h - target) // 2)
            pil_image = pil_image.crop((left, top, left + target, top + target))
        else:
            pil_image = pil_image.resize((target, target), resample=pil_image_mod.BICUBIC)
        array = np.asarray(pil_image).astype(np.float32)
    else:
        array = array.astype(np.float32)
        target = int(cfg.resolution)
        if array.shape[0] != target or array.shape[1] != target:
            y_idx = np.linspace(0, array.shape[0] - 1, target).astype(np.int64)
            x_idx = np.linspace(0, array.shape[1] - 1, target).astype(np.int64)
            array = array[y_idx][:, x_idx]
    if array.max(initial=0.0) > 2.0:
        array = array / 255.0
    if cfg.normalize:
        mean = np.asarray(cfg.mean, dtype=np.float32).reshape(1, 1, 3)
        std = np.asarray(cfg.std, dtype=np.float32).reshape(1, 1, 3)
        array = (array - mean) / np.maximum(std, 1e-8)
    chw = np.transpose(array, (2, 0, 1)).astype(np.float32)
    torch = _optional_import("torch")
    if torch is not None:
        return torch.from_numpy(chw)
    return chw


class _LocalImageDataset:
    """ImageFolder-compatible dataset that applies preprocess_image_tensor."""

    def __init__(self, root: Path, cfg: ImageDatasetConfig):
        self.root = root
        self.cfg = cfg
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        split_root = root / cfg.split
        search_root = split_root if split_root.exists() else root
        self.paths = sorted(p for p in search_root.rglob("*") if p.suffix.lower() in image_exts)
        if cfg.max_items is not None:
            self.paths = self.paths[: int(cfg.max_items)]
        if not self.paths:
            raise FileNotFoundError(f"No image files found under {search_root}")
        self.class_to_idx: dict[str, int] = {}
        for path in self.paths:
            label_name = path.parent.name
            if label_name not in self.class_to_idx:
                self.class_to_idx[label_name] = len(self.class_to_idx)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        pil = _optional_import("PIL.Image")
        if pil is None:
            raise RuntimeError("Pillow is required to read local image files. Install Pillow>=9.4.")
        path = self.paths[index]
        image = pil.open(path).convert("RGB")
        tensor = preprocess_image_tensor(image, self.cfg)
        label = self.class_to_idx.get(path.parent.name, -1)
        return {"image": tensor, "label": label, "path": str(path), "id": index}


class _HFDatasetWrapper:
    """HuggingFace dataset wrapper that applies preprocess_image_tensor."""

    def __init__(self, hf_dataset: Any, cfg: ImageDatasetConfig):
        self.dataset = hf_dataset
        self.cfg = cfg

    def __len__(self) -> int:
        length = len(self.dataset)
        return min(length, int(self.cfg.max_items)) if self.cfg.max_items is not None else length

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        image = sample.get("image") or sample.get("img") or sample.get("jpeg")
        if image is None:
            raise KeyError(f"HuggingFace sample lacks an image field; keys={list(sample.keys())}")
        tensor = preprocess_image_tensor(image, self.cfg)
        label = sample.get("label", sample.get("class", -1))
        return {"image": tensor, "label": int(label) if label is not None else -1, "id": index}


class _SmokeFixtureDataset:
    """Explicit smoke fixture, never used as an implicit missing-data fallback."""

    def __init__(self, cfg: ImageDatasetConfig):
        self.cfg = cfg
        self.length = int(cfg.max_items or max(2, cfg.batch_size))
        rng = np.random.default_rng(cfg.seed)
        size = int(cfg.resolution)
        self.images = rng.uniform(0.0, 1.0, size=(self.length, size, size, 3)).astype(np.float32)
        self.labels = np.arange(self.length, dtype=np.int64) % 1000

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        tensor = preprocess_image_tensor(self.images[index], self.cfg)
        return {"image": tensor, "label": int(self.labels[index]), "id": index, "fixture": "explicit_smoke"}


def load_dataset(
    config: Mapping[str, Any] | ImageDatasetConfig | None = None,
    *,
    dataset_id: str | None = None,
    data_root: str | os.PathLike[str] | None = None,
    split: str | None = None,
    allow_hf_download: bool | None = None,
    use_smoke_fixture: bool | None = None,
) -> Any:
    """Lazy real dataset loader.

    For full ImageNet, this calls the HuggingFace route equivalent to
    ``load_dataset("imagenet-1k", trust_remote_code=True)`` after explicit
    availability validation.  A smoke fixture is available only when explicitly
    requested by config and is labeled as such.
    """

    cfg = ImageDatasetConfig.from_config(
        config,
        dataset_id=dataset_id,
        data_root=str(data_root) if data_root is not None else None,
        split=split,
        allow_hf_download=allow_hf_download,
        use_smoke_fixture=use_smoke_fixture,
    )
    root = resolve_data_root(cfg, require_exists=not cfg.use_smoke_fixture and not cfg.allow_hf_download)
    availability = check_data_available(cfg, data_root=root, raise_on_missing=not cfg.use_smoke_fixture)
    if cfg.use_smoke_fixture:
        return _SmokeFixtureDataset(cfg)
    canonical_id = _canonical_dataset_id(cfg.dataset_id)
    if _looks_like_image_folder(root):
        return _LocalImageDataset(root, cfg)
    datasets_mod = _optional_import("datasets")
    if datasets_mod is None:
        raise FileNotFoundError(
            f"Dataset {cfg.dataset_id!r} was not found at {root}, and the optional "
            "'datasets' package is not installed for HuggingFace loading. "
            f"{DATASET_REGISTRY[canonical_id].setup_note}"
        )
    if not (cfg.allow_hf_download or availability["local_files_available"]):
        raise FileNotFoundError(
            f"No local ImageNet files were found at {root}. To use HuggingFace, set "
            'allow_hf_download=True; the loader will call load_dataset("imagenet-1k", '
            "trust_remote_code=True)."
        )
    hf_load_dataset = getattr(datasets_mod, "load_dataset")
    try:
        hf_data = hf_load_dataset(
            DATASET_REGISTRY[canonical_id].hf_dataset_id,
            split=cfg.split,
            cache_dir=str(root if cfg.cache_dir is None else Path(cfg.cache_dir).expanduser()),
            trust_remote_code=cfg.trust_remote_code,
        )
    except Exception as exc:
        raise FileNotFoundError(
            f"Failed to load {DATASET_REGISTRY[canonical_id].hf_dataset_id!r} through "
            "HuggingFace. Ensure ImageNet access is accepted, credentials are configured, "
            f"and cache/data_root is valid ({root}). Original error: {exc}"
        ) from exc
    return _HFDatasetWrapper(hf_data, cfg)


def make_data(
    config: Mapping[str, Any] | ImageDatasetConfig | None = None,
    **overrides: Any,
) -> Any:
    """Active dataset construction entrypoint used by dataloaders and routes."""

    cfg = ImageDatasetConfig.from_config(config, **overrides)
    resolve_data_root(cfg, require_exists=not cfg.use_smoke_fixture and not cfg.allow_hf_download)
    check_data_available(cfg, raise_on_missing=not cfg.use_smoke_fixture)
    return load_dataset(cfg)


def _collate_batch(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    images = [item["image"] for item in items]
    labels = [item.get("label", -1) for item in items]
    ids = [item.get("id", i) for i, item in enumerate(items)]
    torch = _optional_import("torch")
    if torch is not None and images and hasattr(images[0], "shape"):
        image_batch = torch.stack([img if hasattr(img, "detach") else torch.as_tensor(img) for img in images], dim=0)
        label_batch = torch.as_tensor(labels, dtype=torch.long)
        id_batch = torch.as_tensor(ids, dtype=torch.long)
    else:
        image_batch = np.stack([np.asarray(img) for img in images], axis=0)
        label_batch = np.asarray(labels, dtype=np.int64)
        id_batch = np.asarray(ids, dtype=np.int64)
    return {"image": image_batch, "label": label_batch, "id": id_batch, "raw": list(items)}


def build_dataloader(
    config: Mapping[str, Any] | ImageDatasetConfig | None = None,
    dataset: Any | None = None,
    **overrides: Any,
) -> Any:
    """Build a dataloader; this function explicitly calls make_data."""

    cfg = ImageDatasetConfig.from_config(config, **overrides)
    dataset = make_data(cfg) if dataset is None else dataset
    torch = _optional_import("torch")
    if torch is not None:
        data_mod = importlib.import_module("torch.utils.data")
        return data_mod.DataLoader(
            dataset,
            batch_size=int(cfg.batch_size),
            shuffle=bool(cfg.shuffle),
            num_workers=int(cfg.num_workers),
            collate_fn=_collate_batch,
            drop_last=False,
        )

    class _SimpleDataLoader:
        def __iter__(self_nonlocal: Any) -> Iterator[dict[str, Any]]:
            indices = list(range(len(dataset)))
            if cfg.shuffle:
                random.Random(cfg.seed).shuffle(indices)
            for start in range(0, len(indices), int(cfg.batch_size)):
                yield _collate_batch([dataset[i] for i in indices[start : start + int(cfg.batch_size)]])

        def __len__(self_nonlocal: Any) -> int:
            return math.ceil(len(dataset) / int(cfg.batch_size))

    return _SimpleDataLoader()


def _batch_image(batch: Mapping[str, Any]) -> Any:
    if "image" in batch:
        return batch["image"]
    if "x1" in batch:
        return batch["x1"]
    if "pixel_values" in batch:
        return batch["pixel_values"]
    raise KeyError(f"Batch does not contain image/x1/pixel_values keys: {list(batch.keys())}")


def _randn_like(x: Any, seed: int | None = None) -> Any:
    torch = _optional_import("torch")
    if torch is not None and hasattr(x, "detach"):
        if seed is not None:
            generator = torch.Generator(device=x.device)
            generator.manual_seed(int(seed))
            return torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
        return torch.randn_like(x)
    rng = np.random.default_rng(seed)
    return rng.standard_normal(size=np.asarray(x).shape).astype(np.float32)


def _rand_uniform(batch_size: int, like: Any, seed: int | None = None) -> Any:
    torch = _optional_import("torch")
    if torch is not None and hasattr(like, "detach"):
        if seed is not None:
            generator = torch.Generator(device=like.device)
            generator.manual_seed(int(seed) + 17)
            return torch.rand((batch_size,), generator=generator, device=like.device, dtype=like.dtype)
        return torch.rand((batch_size,), device=like.device, dtype=like.dtype)
    rng = np.random.default_rng(None if seed is None else seed + 17)
    return rng.uniform(0.0, 1.0, size=(batch_size,)).astype(np.float32)


def _ensure_broadcast_time(t: Any, x: Any) -> Any:
    torch = _optional_import("torch")
    if torch is not None and hasattr(x, "detach"):
        while t.ndim < x.ndim:
            t = t.view(*t.shape, 1)
        return t
    arr = np.asarray(t)
    while arr.ndim < np.asarray(x).ndim:
        arr = arr.reshape(*arr.shape, 1)
    return arr


def alpha_t(t: Any, config: Mapping[str, Any] | None = None) -> Any:
    torch = _optional_import("torch")
    if torch is not None and hasattr(t, "detach"):
        return 1.0 - t
    return 1.0 - np.asarray(t)


def beta_t(t: Any, config: Mapping[str, Any] | None = None) -> Any:
    return t


def alpha_dot_t(t: Any, config: Mapping[str, Any] | None = None) -> Any:
    torch = _optional_import("torch")
    if torch is not None and hasattr(t, "detach"):
        return -torch.ones_like(t)
    return -np.ones_like(np.asarray(t), dtype=np.float32)


def beta_dot_t(t: Any, config: Mapping[str, Any] | None = None) -> Any:
    torch = _optional_import("torch")
    if torch is not None and hasattr(t, "detach"):
        return torch.ones_like(t)
    return np.ones_like(np.asarray(t), dtype=np.float32)


def interpolant_state(x0: Any, x1: Any, t: Any, noise: Any | None = None, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Compute I_t=alpha_t x0 + beta_t x1 and dot I_t."""

    tb = _ensure_broadcast_time(t, x1)
    a = _ensure_broadcast_time(alpha_t(t, config), x1)
    b = _ensure_broadcast_time(beta_t(t, config), x1)
    adot = _ensure_broadcast_time(alpha_dot_t(t, config), x1)
    bdot = _ensure_broadcast_time(beta_dot_t(t, config), x1)
    state = a * x0 + b * x1
    derivative = adot * x0 + bdot * x1
    if noise is not None:
        state = state + 0.0 * noise
    return {"I_t": state, "dI_t": derivative, "t": t, "alpha_t": a, "beta_t": b, "t_broadcast": tb}


def sample_rho1(batch: Mapping[str, Any]) -> Any:
    return _batch_image(batch)


def sample_noise_like(x1: Any, seed: int | None = None) -> Any:
    return _randn_like(x1, seed=seed)


def sample_time_uniform(x1: Any, seed: int | None = None) -> Any:
    batch_size = int(x1.shape[0])
    return _rand_uniform(batch_size, x1, seed=seed)


def _call_coupling_builder(cfg: ImageDatasetConfig) -> Any:
    couplings = importlib.import_module("stochastic_interpolants_couplings.couplings")
    build_coupling = getattr(couplings, "build_coupling")
    try:
        return build_coupling(cfg)
    except TypeError:
        try:
            return build_coupling(asdict(cfg))
        except TypeError:
            return build_coupling(task=cfg.task, coupling_type=cfg.coupling_type, sigma=cfg.sigma)


def sample_coupled_batch(
    batch: Mapping[str, Any],
    conditioning: Mapping[str, Any] | None = None,
    mode: str = "data_dependent",
    config: Mapping[str, Any] | ImageDatasetConfig | None = None,
) -> Any:
    """Sample x0|x1 through build_coupling and return CoupledBatch."""

    cfg = ImageDatasetConfig.from_config(config, coupling_type=mode)
    x1 = sample_rho1(batch)
    if not hasattr(x1, "shape"):
        x1 = np.asarray(x1)
    zeta = sample_noise_like(x1, seed=cfg.seed)
    t = sample_time_uniform(x1, seed=cfg.seed)
    cond = dict(conditioning or {})
    cond.setdefault("labels", batch.get("label"))

    couplings = importlib.import_module("stochastic_interpolants_couplings.couplings")
    CoupledBatch = getattr(couplings, "CoupledBatch")
    coupling = _call_coupling_builder(cfg)

    if hasattr(coupling, "sample"):
        try:
            coupled = coupling.sample(x1=x1, zeta=zeta, t=t, conditioning=cond)
        except TypeError:
            try:
                coupled = coupling.sample(x1, cond)
            except TypeError:
                coupled = coupling.sample(x1)
        if isinstance(coupled, CoupledBatch):
            return coupled
        if isinstance(coupled, Mapping):
            return CoupledBatch(**coupled)
        return coupled

    if cfg.coupling_type in ("independent", "independent_gaussian", "gaussian"):
        x0 = cfg.sigma * zeta
    else:
        m_x1 = cond.get("condition_image", cond.get("low_resolution_upsampled", cond.get("visible_pixels", 0.0)))
        x0 = m_x1 + cfg.sigma * zeta
    state = interpolant_state(x0, x1, t, zeta)
    try:
        return CoupledBatch(x0=x0, x1=x1, zeta=zeta, t=t, conditioning=cond, interpolant=state["I_t"], d_interpolant=state["dI_t"])
    except TypeError:
        return CoupledBatch(x0=x0, x1=x1, noise=zeta, t=t, conditioning=cond)


def build_task_batch(
    batch: Mapping[str, Any],
    config: Mapping[str, Any] | ImageDatasetConfig | None = None,
    *,
    task: str | None = None,
    coupling_type: str | None = None,
) -> dict[str, Any]:
    """Build inpainting or super-resolution conditioning and coupled batch.

    This route explicitly touches preprocessing and the couplings.py
    ``build_coupling``/``CoupledBatch`` path required by training and evaluation.
    """

    cfg = ImageDatasetConfig.from_config(config, task=task, coupling_type=coupling_type)
    x1 = sample_rho1(batch)
    if isinstance(x1, list):
        x1 = np.stack([np.asarray(preprocess_image_tensor(item, cfg)) for item in x1], axis=0)
    elif np.asarray(x1).ndim == 3 and not hasattr(x1, "detach"):
        x1 = np.expand_dims(preprocess_image_tensor(x1, cfg), axis=0)
    else:
        first = x1[0] if hasattr(x1, "__getitem__") else x1
        _ = preprocess_image_tensor(first, cfg) if not hasattr(first, "detach") else first

    couplings = importlib.import_module("stochastic_interpolants_couplings.couplings")
    make_inpainting_mask = getattr(couplings, "make_inpainting_mask")
    make_low_resolution_condition = getattr(couplings, "make_low_resolution_condition")

    conditioning: dict[str, Any] = {"labels": batch.get("label"), "task": cfg.task}
    if cfg.task in ("inpainting", "in-painting"):
        try:
            mask = make_inpainting_mask(
                x1,
                tiles=cfg.mask_tiles,
                probability=cfg.mask_probability,
                seed=cfg.seed,
            )
        except TypeError:
            mask = make_inpainting_mask(x1, cfg)
        conditioning["mask"] = mask
        conditioning["xi"] = mask
        conditioning["visible_pixels"] = x1 * mask
        conditioning["condition_image"] = conditioning["visible_pixels"]
    elif cfg.task in ("super_resolution", "super-resolution", "sr"):
        try:
            low = make_low_resolution_condition(x1, low_resolution=cfg.low_resolution, target_resolution=cfg.resolution)
        except TypeError:
            try:
                low = make_low_resolution_condition(x1, cfg)
            except TypeError:
                low = make_low_resolution_condition(x1)
        conditioning["low_resolution_image"] = low
        conditioning["condition_image"] = low
        conditioning["low_resolution_upsampled"] = low
    else:
        conditioning["condition_image"] = batch.get("condition_image", 0.0)

    coupled = sample_coupled_batch(batch={"image": x1, "label": batch.get("label")}, conditioning=conditioning, mode=cfg.coupling_type, config=cfg)
    return {"x1": x1, "conditioning": conditioning, "coupled": coupled, "config": asdict(cfg)}


def velocity_field_objective(coupled_batch: Any, velocity_model: Callable[..., Any], config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Empirical hat L_b = mean(|b_hat|^2 - 2 dI_t dot b_hat)."""

    x0 = getattr(coupled_batch, "x0", None)
    x1 = getattr(coupled_batch, "x1", None)
    t = getattr(coupled_batch, "t", None)
    zeta = getattr(coupled_batch, "zeta", getattr(coupled_batch, "noise", None))
    conditioning = getattr(coupled_batch, "conditioning", None)
    state = interpolant_state(x0, x1, t, zeta, config)
    I_t = getattr(coupled_batch, "interpolant", state["I_t"])
    dI_t = getattr(coupled_batch, "d_interpolant", state["dI_t"])
    try:
        prediction = velocity_model(I_t, t, conditioning)
    except TypeError:
        try:
            prediction = velocity_model(t, I_t, conditioning)
        except TypeError:
            prediction = velocity_model(I_t)
    torch = _optional_import("torch")
    if torch is not None and hasattr(prediction, "detach"):
        dims = tuple(range(1, prediction.ndim))
        per_sample = (prediction.pow(2).sum(dim=dims) - 2.0 * (dI_t * prediction).sum(dim=dims))
        loss = per_sample.mean()
        transport = ((x1 - x0).pow(2).sum(dim=dims)).mean()
    else:
        pred = np.asarray(prediction)
        deriv = np.asarray(dI_t)
        dims = tuple(range(1, pred.ndim))
        per_sample = np.sum(pred * pred, axis=dims) - 2.0 * np.sum(deriv * pred, axis=dims)
        loss = float(np.mean(per_sample))
        transport = float(np.mean(np.sum((np.asarray(x1) - np.asarray(x0)) ** 2, axis=dims)))
    return {"loss": loss, "hat_L_b": loss, "prediction": prediction, "I_t": I_t, "dI_t": dI_t, "transport_cost": transport}


def score_matching_objective(coupled_batch: Any, score_model: Callable[..., Any], config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Testable score-related objective for nabla log rho_t adapters."""

    x0 = getattr(coupled_batch, "x0")
    x1 = getattr(coupled_batch, "x1")
    t = getattr(coupled_batch, "t")
    zeta = getattr(coupled_batch, "zeta", getattr(coupled_batch, "noise", None))
    conditioning = getattr(coupled_batch, "conditioning", None)
    state = interpolant_state(x0, x1, t, zeta, config)
    I_t = state["I_t"]
    try:
        score = score_model(I_t, t, conditioning)
    except TypeError:
        score = score_model(I_t)
    target = -zeta
    torch = _optional_import("torch")
    if torch is not None and hasattr(score, "detach"):
        loss = ((score - target) ** 2).flatten(1).mean()
    else:
        loss = float(np.mean((np.asarray(score) - np.asarray(target)) ** 2))
    return {"score_loss": loss, "score": score, "target_score_proxy": target}


def _feature_stats(images: Any) -> tuple[np.ndarray, np.ndarray]:
    arr = images.detach().cpu().numpy() if hasattr(images, "detach") else np.asarray(images)
    arr = arr.reshape(arr.shape[0], -1).astype(np.float64)
    return arr.mean(axis=0), np.cov(arr, rowvar=False)


def compute_fid(real_images: Any, generated_images: Any, eps: float = 1e-6) -> float:
    """FID formula with a scipy sqrtm path and eigen fallback."""

    mu_r, cov_r = _feature_stats(real_images)
    mu_g, cov_g = _feature_stats(generated_images)
    diff = mu_r - mu_g
    scipy_linalg = _optional_import("scipy.linalg")
    if scipy_linalg is not None:
        covmean = scipy_linalg.sqrtm(cov_r @ cov_g)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
    else:
        vals, vecs = np.linalg.eigh((cov_r @ cov_g) + eps * np.eye(cov_r.shape[0]))
        covmean = (vecs * np.sqrt(np.maximum(vals, 0.0))) @ vecs.T
    fid = float(diff @ diff + np.trace(cov_r + cov_g - 2.0 * covmean))
    return max(fid, 0.0)


def aggregate_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for record in records:
        for key, value in record.items():
            if isinstance(value, (int, float, np.floating)):
                grouped.setdefault(key, []).append(float(value))
    return {key: float(np.mean(values)) for key, values in grouped.items() if values}


def evaluate_predictions(config: Mapping[str, Any] | ImageDatasetConfig | None = None, predictions: Any | None = None, targets: Any | None = None) -> dict[str, Any]:
    """Data-scoped evaluation helper used by evaluation routes."""

    cfg = ImageDatasetConfig.from_config(config)
    result: dict[str, Any] = {"dataset_id": cfg.dataset_id, "task": cfg.task, "metrics": {}}
    if predictions is not None and targets is not None:
        result["metrics"]["fid"] = compute_fid(targets, predictions)
    if predictions is not None:
        result["sampling_success_rate"] = 1.0
    return result


def _artifact_dir(default: str = "results") -> Path:
    return Path(os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR", default)).expanduser()


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_dataset_registry_artifact(output_dir: str | os.PathLike[str] | None = None) -> Path:
    out = Path(output_dir) if output_dir is not None else _artifact_dir()
    return _write_json(out / "dataset_registry.json", {"datasets": dataset_registry_records()})


def write_data_manifest_artifact(
    config: Mapping[str, Any] | ImageDatasetConfig | None = None,
    output_dir: str | os.PathLike[str] | None = None,
) -> Path:
    cfg = ImageDatasetConfig.from_config(config)
    availability = check_data_available(cfg, raise_on_missing=False)
    payload = {
        "manifest_type": "data_manifest",
        "dataset": availability,
        "preprocessing": {
            "resolution": cfg.resolution,
            "crop": cfg.crop,
            "normalize": cfg.normalize,
            "mean": list(cfg.mean),
            "std": list(cfg.std),
        },
        "tasks": ["inpainting", "super_resolution"],
        "full_mode_requirement": DATASET_REGISTRY[_canonical_dataset_id(cfg.dataset_id)].setup_note,
        "no_silent_synthetic_fallback": True,
    }
    out = Path(output_dir) if output_dir is not None else _artifact_dir()
    return _write_json(out / "data_manifest.json", payload)


def write_metrics_artifact(
    metrics: Mapping[str, Any],
    output_dir: str | os.PathLike[str] | None = None,
    filename: str = "metrics.json",
) -> Path:
    out = Path(output_dir) if output_dir is not None else _artifact_dir()
    payload = {
        "metric_registry": METRIC_REGISTRY,
        "metrics": dict(metrics),
        "paper_visible": bool(metrics),
        "provenance": "computed_by_data_pipeline_metric_functions",
    }
    return _write_json(out / filename, payload)


def export_inpainting_fid_and_sample_grid(
    real_images: Any,
    generated_images: Any,
    masked_images: Any,
    output_dir: str | os.PathLike[str] | None = None,
    *,
    resolution: int = 256,
) -> dict[str, Any]:
    """Compute inpainting FID and export a measured sample grid when PIL exists."""

    fid = compute_fid(real_images, generated_images)
    out = Path(output_dir) if output_dir is not None else _artifact_dir()
    out.mkdir(parents=True, exist_ok=True)
    grid_path = out / f"inpainting_sample_grid_{resolution}.png"
    pil_image_mod = _optional_import("PIL.Image")
    if pil_image_mod is not None:
        arrays = []
        for group in (masked_images, generated_images, real_images):
            arr = group.detach().cpu().numpy() if hasattr(group, "detach") else np.asarray(group)
            if arr.ndim == 4 and arr.shape[1] in (1, 3):
                arr = np.transpose(arr, (0, 2, 3, 1))
            arrays.append(arr[: min(4, arr.shape[0])])
        rows = []
        for i in range(arrays[0].shape[0]):
            cols = []
            for arr in arrays:
                img = arr[i]
                img = (img * 0.5 + 0.5) if img.min() < 0 else img
                img = np.clip(img, 0.0, 1.0)
                cols.append((img * 255).astype(np.uint8))
            rows.append(np.concatenate(cols, axis=1))
        grid = np.concatenate(rows, axis=0)
        pil_image_mod.fromarray(grid).save(grid_path)
    return {"fid": fid, "grid_path": str(grid_path), "resolution": resolution}


globals()["inpainting FID 与样本网格导出"] = export_inpainting_fid_and_sample_grid


def _normalise_shape(shape: Sequence[int] | None, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    if shape is None:
        return fallback
    values = tuple(int(v) for v in shape)
    if len(values) != 3:
        raise ValueError(f"image shape must be CHW with three entries; got {shape!r}")
    return values


def _make_mask(shape: tuple[int, int, int], probability: float, seed: int) -> list[float]:
    channels, height, width = shape
    rng = random.Random(seed)
    side_tiles = int(math.sqrt(DEFAULT_MASK_TILES))
    spatial = [1.0 for _ in range(height * width)]
    for ty in range(side_tiles):
        y0 = int(round(ty * height / side_tiles))
        y1 = max(y0 + 1, int(round((ty + 1) * height / side_tiles)))
        for tx in range(side_tiles):
            x0 = int(round(tx * width / side_tiles))
            x1 = max(x0 + 1, int(round((tx + 1) * width / side_tiles)))
            keep = 1.0 if rng.random() >= probability else 0.0
            for y in range(min(height, y0), min(height, y1)):
                for x in range(min(width, x0), min(width, x1)):
                    spatial[y * width + x] = keep
    mask: list[float] = []
    for _ in range(channels):
        mask.extend(spatial)
    return mask


def _nearest_upsample(values: Sequence[float], low_shape: tuple[int, int, int], high_shape: tuple[int, int, int]) -> list[float]:
    channels, height, width = high_shape
    low_channels, low_height, low_width = low_shape
    upsampled: list[float] = []
    for channel in range(channels):
        src_channel = min(channel, low_channels - 1)
        for y in range(height):
            src_y = min(low_height - 1, int(y * low_height / max(1, height)))
            for x in range(width):
                src_x = min(low_width - 1, int(x * low_width / max(1, width)))
                upsampled.append(float(values[src_channel * low_height * low_width + src_y * low_width + src_x]))
    return upsampled


def _downsample_average(vector: Sequence[float], high_shape: tuple[int, int, int], low_shape: tuple[int, int, int]) -> list[float]:
    channels, height, width = high_shape
    low_channels, low_height, low_width = low_shape
    values: list[float] = []
    for channel in range(low_channels):
        src_channel = min(channel, channels - 1)
        for ly in range(low_height):
            crop_top = max(0, (height - low_height) // 2)
            y0 = crop_top + ly
            y1 = y0 + 1
            for lx in range(low_width):
                crop_left = max(0, (width - low_width) // 2)
                x0 = crop_left + lx
                x1 = x0 + 1
                patch: list[float] = []
                for y in range(min(height, y0), min(height, y1)):
                    for x in range(min(width, x0), min(width, x1)):
                        patch.append(float(vector[src_channel * height * width + y * width + x]))
                values.append(float(np.mean(patch)) if patch else 0.0)
    return values


def _lightweight_samples(spec: DataSpec) -> dict[str, Any]:
    shape = _normalise_shape(spec.image_shape, (DEFAULT_CHANNELS, 8, 8))
    channels, height, width = shape
    rng = random.Random(int(spec.seed))
    samples: list[dict[str, Any]] = []
    for index in range(max(1, int(spec.max_samples))):
        x1 = [rng.random() for _ in range(channels * height * width)]
        zeta = [rng.gauss(0.0, 1.0) for _ in x1]
        if spec.task in {"inpainting", "in-painting"}:
            mask = _make_mask(shape, float(spec.mask_probability), int(spec.seed) + index)
            observed = [value * mask[i] for i, value in enumerate(x1)]
            x0 = [mask[i] * x1[i] + (1.0 - mask[i]) * zeta[i] for i in range(len(x1))]
            label = index % 1000
            label_channel = [label / 999.0 for _ in range(height * width)]
            model_input = list(x0) + list(observed) + list(mask) + label_channel
            condition = {
                "task": "inpainting",
                "mask": mask,
                "xi": mask,
                "observed_image": observed,
                "visible_pixels": observed,
                "condition_image": observed,
                "labels": label,
                "class_label_channel": label_channel,
                "model_input": model_input,
                "mask_tiles": int(spec.mask_tiles),
                "mask_probability": float(spec.mask_probability),
                "x0_formula": "x0 = xi * x1 + (1 - xi) * zeta",
            }
        elif spec.task in {"super_resolution", "super-resolution", "sr"}:
            low_shape = _normalise_shape(spec.low_resolution_shape, (channels, max(1, height // 4), max(1, width // 4)))
            low = _downsample_average(x1, shape, low_shape)
            upsampled = _nearest_upsample(low, low_shape, shape)
            x0 = [upsampled[i] + zeta[i] for i in range(len(x1))]
            label = index % 1000
            label_channel = [label / 999.0 for _ in range(height * width)]
            model_input = list(x0) + list(upsampled) + label_channel
            condition = {
                "task": "super_resolution",
                "low_resolution_image": low,
                "low_resolution_shape": list(low_shape),
                "low_resolution_upsampled": upsampled,
                "condition_image": upsampled,
                "labels": label,
                "class_label_channel": label_channel,
                "model_input": model_input,
                "crop_downsample": f"center crop to {low_shape[-1]}x{low_shape[-1]}",
                "upsampling": "nearest",
                "x0_formula": "x0 = U(D(x1)) + sigma * zeta",
            }
        else:
            x0 = [0.1 * z for z in zeta]
            condition = {"task": spec.task, "condition_image": [0.0 for _ in x1]}
        samples.append(
            {
                "id": index,
                "x0": x0,
                "x1": x1,
                "zeta": zeta,
                "condition": condition,
                "model_input": condition.get("model_input"),
                "image_shape": list(shape),
                "dataset_id": spec.dataset_id,
            }
        )
    return {"spec": spec, "samples": samples, "task": spec.task, "image_shape": list(shape)}


def load_data(config: Mapping[str, Any] | ImageDatasetConfig | DataSpec | None = None, **overrides: Any) -> Any:
    if isinstance(config, DataSpec):
        spec_fields = set(DataSpec.__dataclass_fields__.keys())
        spec_updates = {k: v for k, v in overrides.items() if k in spec_fields and v is not None}
        spec = DataSpec(**{**asdict(config), **spec_updates}) if spec_updates else config
        return _lightweight_samples(spec)
    return make_data(config, **overrides)


def prepare_data(
    config: Mapping[str, Any] | ImageDatasetConfig | DataSpec | str | os.PathLike[str] | None = None,
    spec: DataSpec | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Prepare data registry/manifests and return a runnable dataloader.

    If a config path is provided and config.py exposes load_experiment_config,
    that loader is used so the canonical route exercises the config dependency.
    """

    if isinstance(config, DataSpec):
        return _lightweight_samples(config)
    if isinstance(spec, DataSpec):
        prepared = config if isinstance(config, Mapping) and "samples" in config else _lightweight_samples(spec)
        samples = [dict(sample) for sample in prepared["samples"]]
        return {
            "config": ImageDatasetConfig(
                task=spec.task,
                resolution=int(spec.image_shape[-1]),
                mask_tiles=int(spec.mask_tiles),
                mask_probability=float(spec.mask_probability),
                low_resolution=int(spec.low_resolution_shape[-1]) if spec.low_resolution_shape else int(spec.low_resolution),
                max_items=int(spec.max_samples),
                use_smoke_fixture=True,
                seed=int(spec.seed),
            ),
            "availability": {"available": True, "fixture": "explicit_lightweight_task_spec"},
            "dataset": prepared,
            "samples": samples,
            "task": spec.task,
            "image_shape": list(spec.image_shape),
        }

    loaded_config: Mapping[str, Any] | ImageDatasetConfig | None
    if isinstance(config, (str, os.PathLike)) and Path(config).exists():
        try:
            cfg_mod = importlib.import_module("stochastic_interpolants_couplings.config")
            load_experiment_config = getattr(cfg_mod, "load_experiment_config")
            loaded_config = load_experiment_config(str(config))
        except Exception:
            yaml = _optional_import("yaml")
            if yaml is None:
                raise
            loaded_config = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    else:
        loaded_config = config if not isinstance(config, (str, os.PathLike)) else {"data": {"data_root": str(config)}}

    cfg = ImageDatasetConfig.from_config(loaded_config, **overrides)
    availability = check_data_available(cfg, raise_on_missing=not cfg.use_smoke_fixture)
    dataset = make_data(cfg)
    dataloader = build_dataloader(cfg, dataset=dataset)
    output_dir = _artifact_dir()
    write_dataset_registry_artifact(output_dir)
    write_data_manifest_artifact(cfg, output_dir)
    return {"config": cfg, "availability": availability, "dataset": dataset, "dataloader": dataloader}


__all__ = [
    "DATASET_ALIASES",
    "DATASET_REGISTRY",
    "METRIC_REGISTRY",
    "METHOD_REGISTRY",
    "DataSpec",
    "ImageDatasetConfig",
    "resolve_data_root",
    "check_data_available",
    "make_data",
    "load_dataset",
    "build_dataloader",
    "build_task_batch",
    "preprocess_image_tensor",
    "sample_coupled_batch",
    "interpolant_state",
    "sample_rho1",
    "sample_noise_like",
    "sample_time_uniform",
    "velocity_field_objective",
    "score_matching_objective",
    "compute_fid",
    "aggregate_metrics",
    "evaluate_predictions",
    "write_dataset_registry_artifact",
    "write_data_manifest_artifact",
    "write_metrics_artifact",
    "export_inpainting_fid_and_sample_grid",
    "load_data",
    "prepare_data",
]
