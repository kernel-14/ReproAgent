"""
src/data/data.py
----------------
DPMs-ANT data pipeline: paper-derived dataset/benchmark registry, FewShotDataset
(10-shot subset sampling + augmentation), dataset loaders, pretrained model loader
helpers, checkpoint management, and artifact writers.

reference_grounding: paper_semantic_chunk_012 10-shot image generation experiments
reference_grounding: paper_semantic_chunk_014_01 experimental_setup classifier_loader_finetuning
reference_grounding: paper_method_core dpms_ant data pipeline

Paper datasets registered (10 total):
  Source domains: imagenet | ffhq | lsun_church
  Target domains: babies | sunglasses | raphael_peale | sketches | modigliani
                  haunted_houses | landscape_drawings

Addendum clarifications implemented here:
  - Classifier fine-tuned from ImageNet-pretrained MobileNetV2; last layer
    replaced to output 2 classes (source vs target domain).
  - Adaptor: down-pool + GroupNorm + 3×3 conv → 4-head attention →
    MLP(→8 or 16) → upsample×4 + GroupNorm + 3×3 conv.
"""

from __future__ import annotations

import json
import os
import pathlib
import random
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# Dataset Registry
# Paper evidence contract: explicitly register all dataset/benchmark aliases.
# reference_grounding: paper_semantic_chunk_012 dataset list
# ---------------------------------------------------------------------------

DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    # ── Source domains ───────────────────────────────────────────────────────
    "imagenet": {
        "id": "imagenet",
        "role": "source",
        "description": "ImageNet ILSVRC-2012 large-scale image classification dataset",
        "image_size": 256,
        "default_split": "train",
        "num_classes": 1000,
        "aliases": ["imagenet", "ilsvrc", "imagenet1k"],
        "data_root_env": "IMAGENET_ROOT",
        "default_root": "data/imagenet",
        "url": "https://image-net.org/",
        "preprocessing": {"resize": 256, "center_crop": 256, "normalize": True},
    },
    "ffhq": {
        "id": "ffhq",
        "role": "source",
        "description": "Flickr-Faces-HQ: 70,000 high-quality face images at 256×256",
        "image_size": 256,
        "default_split": "train",
        "num_classes": None,
        "aliases": ["ffhq", "flickr_faces_hq", "faces_hq"],
        "data_root_env": "FFHQ_ROOT",
        "default_root": "data/ffhq",
        "url": "https://github.com/NVlabs/ffhq-dataset",
        "preprocessing": {"resize": 256, "center_crop": 256, "normalize": True},
    },
    "lsun_church": {
        "id": "lsun_church",
        "role": "source",
        "description": "LSUN Church outdoor category – architecture/landscape source domain",
        "image_size": 256,
        "default_split": "train",
        "num_classes": None,
        "aliases": ["lsun_church", "lsun-church", "church", "lsun_church_outdoor"],
        "data_root_env": "LSUN_ROOT",
        "default_root": "data/lsun",
        "url": "https://www.yf.io/p/lsun",
        "preprocessing": {"resize": 256, "center_crop": 256, "normalize": True},
    },
    # ── Target domains (7 few-shot target domains from paper Table 2) ─────────
    "babies": {
        "id": "babies",
        "role": "target",
        "description": "Baby face images – FFHQ→Babies 10-shot transfer target",
        "image_size": 256,
        "shot": 10,
        "source_domain": "ffhq",
        "framework": "ddpm",
        "aliases": ["babies", "baby_faces", "baby"],
        "data_root_env": "BABIES_ROOT",
        "default_root": "data/few_shot/babies",
        "augmentation": ["random_horizontal_flip"],
        "fid_reference": "data/fid_stats/babies.npz",
    },
    "sunglasses": {
        "id": "sunglasses",
        "role": "target",
        "description": "Sunglasses face images – FFHQ→Sunglasses 10-shot transfer target",
        "image_size": 256,
        "shot": 10,
        "source_domain": "ffhq",
        "framework": "ddpm",
        "aliases": ["sunglasses", "glasses"],
        "data_root_env": "SUNGLASSES_ROOT",
        "default_root": "data/few_shot/sunglasses",
        "augmentation": ["random_horizontal_flip"],
        "fid_reference": "data/fid_stats/sunglasses.npz",
    },
    "raphael_peale": {
        "id": "raphael_peale",
        "role": "target",
        "description": "Raphael Peale portrait paintings – FFHQ→Raphael Peale artistic transfer",
        "image_size": 256,
        "shot": 10,
        "source_domain": "ffhq",
        "framework": "ddpm",
        "aliases": ["raphael_peale", "raphael-peale", "peale_portraits", "raphael"],
        "data_root_env": "RAPHAEL_PEALE_ROOT",
        "default_root": "data/few_shot/raphael_peale",
        "augmentation": ["random_horizontal_flip"],
        "fid_reference": "data/fid_stats/raphael_peale.npz",
    },
    "sketches": {
        "id": "sketches",
        "role": "target",
        "description": "Sketch face drawings – FFHQ→Sketches style transfer",
        "image_size": 256,
        "shot": 10,
        "source_domain": "ffhq",
        "framework": "ddpm",
        "aliases": ["sketches", "face_sketches", "sketch"],
        "data_root_env": "SKETCHES_ROOT",
        "default_root": "data/few_shot/sketches",
        "augmentation": ["random_horizontal_flip"],
        "fid_reference": "data/fid_stats/sketches.npz",
    },
    "modigliani": {
        "id": "modigliani",
        "role": "target",
        "description": "Modigliani portrait paintings – FFHQ→Modigliani artistic style transfer",
        "image_size": 256,
        "shot": 10,
        "source_domain": "ffhq",
        "framework": "ddpm",
        "aliases": ["modigliani", "modigliani_paintings"],
        "data_root_env": "MODIGLIANI_ROOT",
        "default_root": "data/few_shot/modigliani",
        "augmentation": ["random_horizontal_flip"],
        "fid_reference": "data/fid_stats/modigliani.npz",
    },
    "haunted_houses": {
        "id": "haunted_houses",
        "role": "target",
        "description": "Haunted house images – LSUN-Church→Haunted Houses 10-shot transfer",
        "image_size": 256,
        "shot": 10,
        "source_domain": "lsun_church",
        "framework": "ddpm",
        "aliases": ["haunted_houses", "haunted-houses", "haunted"],
        "data_root_env": "HAUNTED_HOUSES_ROOT",
        "default_root": "data/few_shot/haunted_houses",
        "augmentation": ["random_horizontal_flip"],
        "fid_reference": "data/fid_stats/haunted_houses.npz",
    },
    "landscape_drawings": {
        "id": "landscape_drawings",
        "role": "target",
        "description": (
            "Landscape drawing artworks – LSUN-Church→Landscape Drawings 10-shot transfer"
        ),
        "image_size": 256,
        "shot": 10,
        "source_domain": "lsun_church",
        "framework": "ddpm",
        "aliases": [
            "landscape_drawings",
            "landscape-drawings",
            "landscapes",
            "landscape_drawing",
        ],
        "data_root_env": "LANDSCAPE_DRAWINGS_ROOT",
        "default_root": "data/few_shot/landscape_drawings",
        "augmentation": ["random_horizontal_flip"],
        "fid_reference": "data/fid_stats/landscape_drawings.npz",
    },
}

# Ordered target domain list (7 domains, paper Table 2)
TARGET_DOMAINS: List[str] = [
    "babies",
    "sunglasses",
    "raphael_peale",
    "sketches",
    "modigliani",
    "haunted_houses",
    "landscape_drawings",
]

# Source domain list
SOURCE_DOMAINS: List[str] = ["imagenet", "ffhq", "lsun_church"]

# Experiment pairs (source, target, framework) – Table 2 of the paper
DOMAIN_PAIRS: List[Dict[str, str]] = [
    {"source": "ffhq", "target": "babies", "framework": "ddpm"},
    {"source": "ffhq", "target": "sunglasses", "framework": "ddpm"},
    {"source": "ffhq", "target": "raphael_peale", "framework": "ddpm"},
    {"source": "ffhq", "target": "sketches", "framework": "ddpm"},
    {"source": "ffhq", "target": "modigliani", "framework": "ddpm"},
    {"source": "lsun_church", "target": "haunted_houses", "framework": "ddpm"},
    {"source": "lsun_church", "target": "landscape_drawings", "framework": "ddpm"},
]

# Paper training hyperparameters (addendum fixed values)
PAPER_HYPERPARAMETERS: Dict[str, Any] = {
    "total_iterations": 5000,
    "ablation_iterations": 300,
    "shot": 10,
    "batch_size": 64,
    "omega": 0.02,
    "adversarial_inner_steps": 10,
    "similarity_guidance_scale_gamma": 5,
    "adaptor_ddpm": {"c": 4, "d": 8},
    "adaptor_ldm": {"c": 2, "d": 8},
}


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


def get_dataset_info(name: str) -> Dict[str, Any]:
    """Return registry entry for a dataset by id or alias.

    Parameters
    ----------
    name : str
        Dataset id or alias string.

    Returns
    -------
    Dict[str, Any]
        Registry entry with id, role, description, etc.

    Raises
    ------
    KeyError
        If name is not found in registry or aliases.
    """
    if name in DATASET_REGISTRY:
        return DATASET_REGISTRY[name]
    for key, info in DATASET_REGISTRY.items():
        if name in info.get("aliases", []):
            return info
    raise KeyError(
        f"Dataset '{name}' not found in registry. "
        f"Available ids: {list(DATASET_REGISTRY.keys())}"
    )


def list_datasets(role: Optional[str] = None) -> List[str]:
    """List registered dataset ids, optionally filtered by role.

    Parameters
    ----------
    role : str, optional
        Filter by 'source' or 'target'. If None, returns all.
    """
    if role is None:
        return list(DATASET_REGISTRY.keys())
    return [k for k, v in DATASET_REGISTRY.items() if v.get("role") == role]


def get_domain_pairs(source: Optional[str] = None) -> List[Dict[str, str]]:
    """Return experiment domain pairs, optionally filtered by source domain."""
    if source is None:
        return DOMAIN_PAIRS
    return [p for p in DOMAIN_PAIRS if p["source"] == source]


# ---------------------------------------------------------------------------
# Identity transform fallback (no torchvision)
# ---------------------------------------------------------------------------


class _IdentityTransform:
    """Minimal fallback transform when torchvision is not installed."""

    def __call__(self, img: Any) -> Any:
        return img


# ---------------------------------------------------------------------------
# FewShotDataset
# Implements 10-shot subset sampling with data augmentation.
# reference_grounding: paper_semantic_chunk_012 few-shot setting
# ---------------------------------------------------------------------------


class FewShotDataset:
    """
    Generic 10-shot target-domain dataset.

    Loads up to ``shot`` images from a target-domain directory and applies
    standard augmentation (random horizontal flip + normalization to [-1, 1]).

    Parameters
    ----------
    domain : str
        Target domain id (must be in DATASET_REGISTRY with role='target').
    shot : int, optional
        Number of images to sample; paper default = 10.
    root : str, optional
        Root directory for image files. If None, uses registry default_root
        or the environment variable specified in data_root_env.
    image_size : int, optional
        Resize/crop target. If None, uses registry value.
    augment : bool, optional
        Apply random horizontal flip (paper default: True).
    seed : int, optional
        Random seed for reproducible subset selection.
    """

    def __init__(
        self,
        domain: str,
        shot: int = 10,
        root: Optional[str] = None,
        image_size: Optional[int] = None,
        augment: bool = True,
        seed: int = 42,
    ) -> None:
        self.domain = domain
        self.shot = shot
        self.augment = augment
        self.seed = seed

        info = get_dataset_info(domain)
        self.info = info

        # Resolve root directory
        if root is not None:
            self.root = pathlib.Path(root)
        else:
            env_key = info.get("data_root_env", "")
            env_val = os.environ.get(env_key, "") if env_key else ""
            self.root = (
                pathlib.Path(env_val) if env_val else pathlib.Path(info["default_root"])
            )

        # Resolve image size
        self.image_size = (
            image_size if image_size is not None else info.get("image_size", 256)
        )

        self._image_paths: Optional[List[pathlib.Path]] = None
        self._transform: Any = None

    # ── Transform (lazy import of torchvision) ────────────────────────────────

    def _get_transform(self) -> Any:
        """Build and cache torchvision transform (lazy import)."""
        if self._transform is not None:
            return self._transform
        try:
            import torchvision.transforms as T  # noqa: PLC0415

            ops: List[Any] = [
                T.Resize(self.image_size),
                T.CenterCrop(self.image_size),
            ]
            if self.augment:
                ops.append(T.RandomHorizontalFlip(p=0.5))
            ops += [
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
            self._transform = T.Compose(ops)
        except ImportError:
            self._transform = _IdentityTransform()
        return self._transform

    # ── File collection ───────────────────────────────────────────────────────

    def _collect_paths(self) -> List[pathlib.Path]:
        """Collect image file paths from root, then subsample to ``shot``."""
        _IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
        if not self.root.exists():
            return []
        paths = sorted(
            p for p in self.root.rglob("*") if p.suffix.lower() in _IMG_EXTS
        )
        if not paths:
            return []
        rng = random.Random(self.seed)
        if len(paths) > self.shot:
            paths = rng.sample(paths, self.shot)
        return paths

    @property
    def image_paths(self) -> List[pathlib.Path]:
        """Lazily populated list of image file paths."""
        if self._image_paths is None:
            self._image_paths = self._collect_paths()
        return self._image_paths

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Any:
        """Return a transformed image tensor, or a path dict on ImportError."""
        path = self.image_paths[idx]
        try:
            from PIL import Image as PILImage  # noqa: PLC0415

            img = PILImage.open(path).convert("RGB")
            return self._get_transform()(img)
        except ImportError:
            return {"path": str(path), "idx": idx, "domain": self.domain}

    # ── DataLoader factory ────────────────────────────────────────────────────

    def as_dataloader(
        self,
        batch_size: int = 10,
        shuffle: bool = True,
        num_workers: int = 0,
        drop_last: bool = False,
    ) -> Any:
        """Return a PyTorch DataLoader over this dataset.

        Parameters
        ----------
        batch_size : int
            Mini-batch size. Paper default for few-shot loader = ``shot`` (10).
        shuffle : bool
            Shuffle between epochs.
        num_workers : int
            DataLoader worker processes.
        drop_last : bool
            Drop incomplete final batch.

        Returns
        -------
        torch.utils.data.DataLoader

        Raises
        ------
        RuntimeError
            If PyTorch is not installed.
        """
        try:
            from torch.utils.data import DataLoader, Dataset as TorchDataset  # noqa: PLC0415

            outer = self

            class _TorchWrapper(TorchDataset):
                def __len__(self_) -> int:
                    return len(outer)

                def __getitem__(self_, idx: int) -> Any:
                    return outer[idx]

            return DataLoader(
                _TorchWrapper(),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                drop_last=drop_last,
            )
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is required to create a DataLoader. "
                "Install torch: pip install torch torchvision"
            ) from exc

    # ── Readiness ─────────────────────────────────────────────────────────────

    def readiness(self) -> Dict[str, Any]:
        """Return dataset readiness report dict."""
        paths = self.image_paths
        root_exists = self.root.exists()
        return {
            "domain": self.domain,
            "root": str(self.root),
            "root_exists": root_exists,
            "found_images": len(paths),
            "shot": self.shot,
            "image_size": self.image_size,
            "augment": self.augment,
            "ready": root_exists and len(paths) > 0,
            "status": "ready" if (root_exists and len(paths) > 0) else "data_missing",
            "source_domain": self.info.get("source_domain", "unknown"),
            "framework": self.info.get("framework", "ddpm"),
        }


# ---------------------------------------------------------------------------
# make_dataset / make_fewshot_dataset
# ---------------------------------------------------------------------------


def make_dataset(config: Dict[str, Any]) -> FewShotDataset:
    """
    Factory: create a FewShotDataset from a config dict.

    Expected config keys
    --------------------
    domain : str
        Target domain id (required).
    shot : int, optional
        Number of samples; default 10.
    root : str, optional
        Override data root directory.
    image_size : int, optional
        Override image size.
    augment : bool, optional
        Apply augmentation; default True.
    seed : int, optional
        Random seed; default 42.

    Returns
    -------
    FewShotDataset
    """
    domain = config["domain"]
    shot = int(config.get("shot", 10))
    root = config.get("root", None)
    image_size = config.get("image_size", None)
    augment = bool(config.get("augment", True))
    seed = int(config.get("seed", 42))
    return FewShotDataset(
        domain=domain,
        shot=shot,
        root=root,
        image_size=image_size,
        augment=augment,
        seed=seed,
    )


def make_fewshot_dataset(
    pair: Dict[str, str],
    config: Optional[Dict[str, Any]] = None,
) -> FewShotDataset:
    """
    Create a FewShotDataset from a domain-pair dict.

    Parameters
    ----------
    pair : dict
        Must contain ``'target'`` key; optionally ``'source'``, ``'framework'``.
    config : dict, optional
        Additional overrides: shot, root, image_size, augment, seed.

    Returns
    -------
    FewShotDataset
    """
    cfg: Dict[str, Any] = {} if config is None else dict(config)
    cfg["domain"] = pair["target"]
    if "shot" not in cfg:
        target_info = get_dataset_info(pair["target"])
        cfg["shot"] = target_info.get("shot", 10)
    return make_dataset(cfg)


# ---------------------------------------------------------------------------
# Dataset readiness check
# ---------------------------------------------------------------------------


def check_dataset_readiness(
    domains: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Check readiness for each registered dataset.

    Parameters
    ----------
    domains : list[str], optional
        Subset of domain ids to check. If None, checks all registered datasets.

    Returns
    -------
    dict
        Mapping domain_id → readiness report dict.
    """
    if domains is None:
        domains = list(DATASET_REGISTRY.keys())
    report: Dict[str, Dict[str, Any]] = {}
    for domain in domains:
        try:
            info = get_dataset_info(domain)
        except KeyError:
            report[domain] = {
                "domain": domain,
                "status": "unknown_domain",
                "ready": False,
                "root_exists": False,
            }
            continue

        env_key = info.get("data_root_env", "")
        env_val = os.environ.get(env_key, "") if env_key else ""
        root = (
            pathlib.Path(env_val) if env_val else pathlib.Path(info["default_root"])
        )
        root_exists = root.exists()

        if info.get("role") == "target":
            ds = FewShotDataset(
                domain=domain,
                shot=info.get("shot", 10),
                root=str(root),
            )
            report[domain] = ds.readiness()
        else:
            report[domain] = {
                "domain": domain,
                "root": str(root),
                "root_exists": root_exists,
                "status": "ready" if root_exists else "data_missing",
                "ready": root_exists,
                "role": info.get("role", "source"),
                "num_classes": info.get("num_classes"),
            }
    return report


# ---------------------------------------------------------------------------
# Source-domain dataset (for classifier training)
# reference_grounding: paper_semantic_chunk_014_01 classifier_loader_finetuning
# ---------------------------------------------------------------------------


class SourceDomainDataset:
    """
    Source-domain image dataset used during domain classifier training.

    The domain classifier (MobileNetV2 fine-tuned to 2 classes) requires
    balanced source/target batches.  Images from this dataset are labelled 0
    (source class).

    reference_grounding: paper_semantic_chunk_014_01 classifier_loader_finetuning
    """

    def __init__(
        self,
        domain: str,
        root: Optional[str] = None,
        image_size: int = 256,
        max_samples: Optional[int] = None,
        augment: bool = True,
        seed: int = 42,
    ) -> None:
        self.domain = domain
        self.image_size = image_size
        self.max_samples = max_samples
        self.augment = augment
        self.seed = seed
        self.label = 0  # source class label for binary classifier

        info = get_dataset_info(domain)
        self.info = info

        if root is not None:
            self.root = pathlib.Path(root)
        else:
            env_key = info.get("data_root_env", "")
            env_val = os.environ.get(env_key, "") if env_key else ""
            self.root = (
                pathlib.Path(env_val) if env_val else pathlib.Path(info["default_root"])
            )

        self._image_paths: Optional[List[pathlib.Path]] = None
        self._transform: Any = None

    def _get_transform(self) -> Any:
        if self._transform is not None:
            return self._transform
        try:
            import torchvision.transforms as T  # noqa: PLC0415

            ops: List[Any] = [T.Resize(self.image_size), T.CenterCrop(self.image_size)]
            if self.augment:
                ops.append(T.RandomHorizontalFlip(0.5))
            ops += [
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
            self._transform = T.Compose(ops)
        except ImportError:
            self._transform = _IdentityTransform()
        return self._transform

    @property
    def image_paths(self) -> List[pathlib.Path]:
        if self._image_paths is None:
            _IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
            if not self.root.exists():
                self._image_paths = []
            else:
                paths = sorted(
                    p for p in self.root.rglob("*") if p.suffix.lower() in _IMG_EXTS
                )
                if self.max_samples and len(paths) > self.max_samples:
                    rng = random.Random(self.seed)
                    paths = rng.sample(paths, self.max_samples)
                self._image_paths = paths
        return self._image_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        path = self.image_paths[idx]
        try:
            from PIL import Image as PILImage  # noqa: PLC0415

            img = PILImage.open(path).convert("RGB")
            return self._get_transform()(img), self.label
        except ImportError:
            return {"path": str(path), "label": self.label, "idx": idx}, self.label


# ---------------------------------------------------------------------------
# BinaryDomainDataset (paired source + target for classifier training)
# reference_grounding: paper_semantic_chunk_014_01 classifier_loader_finetuning
# ---------------------------------------------------------------------------


class BinaryDomainDataset:
    """
    Paired source + target dataset for binary domain classifier fine-tuning.

    Yields (image_tensor, label) where label=0 for source images and
    label=1 for target images.  Used to fine-tune MobileNetV2 so that the
    last FC layer produces 2-class logits (source vs target domain).

    reference_grounding: paper_semantic_chunk_014_01 classifier_loader_finetuning

    Parameters
    ----------
    source_domain : str
        Source domain id.
    target_domain : str
        Target domain id.
    source_root : str, optional
        Override source data root.
    target_root : str, optional
        Override target data root.
    image_size : int
        Resize/crop size; default 256.
    max_source_samples : int
        Cap on source images per epoch; default 500.
    target_shot : int
        Number of target images; paper default 10.
    augment : bool
        Apply horizontal flip; default True.
    seed : int
        RNG seed; default 42.
    """

    def __init__(
        self,
        source_domain: str,
        target_domain: str,
        source_root: Optional[str] = None,
        target_root: Optional[str] = None,
        image_size: int = 256,
        max_source_samples: int = 500,
        target_shot: int = 10,
        augment: bool = True,
        seed: int = 42,
    ) -> None:
        self.source_ds = SourceDomainDataset(
            domain=source_domain,
            root=source_root,
            image_size=image_size,
            max_samples=max_source_samples,
            augment=augment,
            seed=seed,
        )
        self.target_ds = FewShotDataset(
            domain=target_domain,
            shot=target_shot,
            root=target_root,
            image_size=image_size,
            augment=augment,
            seed=seed,
        )
        self._items: Optional[List[Tuple[Any, int]]] = None

    def _build(self) -> List[Tuple[Any, int]]:
        items: List[Tuple[Any, int]] = []
        for i in range(len(self.source_ds)):
            img, _ = self.source_ds[i]
            items.append((img, 0))
        for i in range(len(self.target_ds)):
            items.append((self.target_ds[i], 1))
        return items

    @property
    def items(self) -> List[Tuple[Any, int]]:
        if self._items is None:
            self._items = self._build()
        return self._items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        return self.items[idx]

    def as_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> Any:
        """Wrap in a DataLoader for classifier training."""
        try:
            from torch.utils.data import DataLoader, Dataset as TorchDataset  # noqa: PLC0415

            outer = self

            class _W(TorchDataset):
                def __len__(self_) -> int:
                    return len(outer)

                def __getitem__(self_, idx: int) -> Tuple[Any, int]:
                    return outer[idx]

            return DataLoader(_W(), batch_size=batch_size, shuffle=shuffle,
                              num_workers=num_workers)
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is required for DataLoader. pip install torch torchvision"
            ) from exc


# ---------------------------------------------------------------------------
# Pretrained model loader helpers
# reference_grounding: paper_semantic_chunk_014_01 pretrained model loading
# ---------------------------------------------------------------------------


def load_pretrained_ddpm(
    source_domain: str,
    checkpoint_path: Optional[str] = None,
    freeze_non_adaptor: bool = True,
) -> Any:
    """
    Load a pretrained DDPM model, optionally freezing non-adaptor parameters.

    All parameters except those in the Shift Adaptor (W_down / W_up bottleneck)
    are frozen, so only adaptor weights are updated during fine-tuning.

    reference_grounding: paper_method_core dpms_ant/adaptor/shift_adaptor.py

    Parameters
    ----------
    source_domain : str
        Source domain id: ``ffhq`` | ``lsun_church``.
    checkpoint_path : str, optional
        Explicit path to ``.pt`` checkpoint.  If None, resolved via env var.
    freeze_non_adaptor : bool
        Freeze all parameters except adaptor weights.

    Returns
    -------
    nn.Module
        DDPM model with adaptor.

    Raises
    ------
    RuntimeError
        If torch or model module is unavailable.
    FileNotFoundError
        If checkpoint file does not exist.
    """
    if checkpoint_path is None:
        _ckpt_env = {"ffhq": "DDPM_FFHQ_CKPT", "lsun_church": "DDPM_CHURCH_CKPT"}
        env_key = _ckpt_env.get(source_domain, "")
        checkpoint_path = os.environ.get(
            env_key, f"pretrained/ddpm_{source_domain}_256.pt"
        )

    try:
        import torch  # noqa: PLC0415
        from src.models.ddpm import DDPM  # type: ignore  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            f"torch or src.models.ddpm unavailable: {exc}. "
            "Install torch and ensure src/models/ddpm.py exists."
        ) from exc

    ckpt = pathlib.Path(checkpoint_path)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"DDPM checkpoint not found: {ckpt}. "
            "Download the pretrained checkpoint from the original source."
        )

    model = DDPM.load_from_checkpoint(str(ckpt))
    if freeze_non_adaptor:
        _freeze_non_adaptor_params(model)
    return model


def load_pretrained_ldm(
    source_domain: str,
    checkpoint_path: Optional[str] = None,
    freeze_non_adaptor: bool = True,
) -> Any:
    """
    Load a pretrained LDM model, optionally freezing non-adaptor parameters.

    reference_grounding: paper_method_core src/models/ldm.py

    Parameters
    ----------
    source_domain : str
        Source domain id: ``ffhq``.
    checkpoint_path : str, optional
        Explicit path to ``.pt`` checkpoint.
    freeze_non_adaptor : bool
        Freeze all parameters except adaptor weights.

    Returns
    -------
    nn.Module
        LDM model with adaptor.
    """
    if checkpoint_path is None:
        _ckpt_env = {"ffhq": "LDM_FFHQ_CKPT"}
        env_key = _ckpt_env.get(source_domain, "")
        checkpoint_path = os.environ.get(
            env_key, f"pretrained/ldm_{source_domain}_256.pt"
        )

    try:
        import torch  # noqa: PLC0415
        from src.models.ldm import LDM  # type: ignore  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            f"torch or src.models.ldm unavailable: {exc}. "
            "Install torch and ensure src/models/ldm.py exists."
        ) from exc

    ckpt = pathlib.Path(checkpoint_path)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"LDM checkpoint not found: {ckpt}. "
            "Download the pretrained checkpoint from the original source."
        )

    model = LDM.load_from_checkpoint(str(ckpt))
    if freeze_non_adaptor:
        _freeze_non_adaptor_params(model)
    return model


def _freeze_non_adaptor_params(model: Any) -> None:
    """
    Freeze all model parameters except those belonging to adaptor modules.

    Adaptor parameter names matching any of the following substrings remain
    trainable: ``adaptor``, ``shift_adaptor``, ``w_down``, ``w_up``,
    ``W_down``, ``W_up``, ``adaptor_down``, ``adaptor_up``.

    reference_grounding: paper_method_core dpms_ant/adaptor/shift_adaptor.py
    """
    _ADAPTOR_KEYS = (
        "adaptor",
        "shift_adaptor",
        "w_down",
        "w_up",
        "W_down",
        "W_up",
        "adaptor_down",
        "adaptor_up",
    )
    try:
        for name, param in model.named_parameters():
            is_adaptor = any(k in name for k in _ADAPTOR_KEYS)
            param.requires_grad = is_adaptor
    except AttributeError:
        pass  # model is not nn.Module – skip silently


def load_domain_classifier(
    checkpoint_path: Optional[str] = None,
    mobilenet_weights: Optional[str] = None,
    num_classes: int = 2,
) -> Any:
    """
    Load domain classifier (MobileNetV2 fine-tuned for 2-class source/target).

    Per addendum: pre-trained MobileNetV2 last layer is replaced with a linear
    layer outputting ``num_classes`` (default 2: source=0, target=1).

    reference_grounding: paper_semantic_chunk_014_01 classifier_loader_finetuning

    Parameters
    ----------
    checkpoint_path : str, optional
        Path to saved fine-tuned classifier ``.pt`` checkpoint.
        If None, checks env var ``DOMAIN_CLASSIFIER_CKPT``.
    mobilenet_weights : str, optional
        Path to ImageNet-pretrained MobileNetV2 weights.
        If None, checks env var ``MOBILENET_WEIGHTS``.
    num_classes : int
        Output classes for binary domain classification; paper default = 2.

    Returns
    -------
    nn.Module
        DomainClassifier with MobileNetV2 backbone.
    """
    if checkpoint_path is None:
        checkpoint_path = os.environ.get("DOMAIN_CLASSIFIER_CKPT", "")
    if mobilenet_weights is None:
        mobilenet_weights = os.environ.get(
            "MOBILENET_WEIGHTS", "pretrained/mobilenet_v2.pth"
        )

    try:
        import torch  # noqa: PLC0415
        from dpms_ant.classifier.domain_classifier import (  # type: ignore  # noqa: PLC0415
            DomainClassifier,
        )
    except ImportError as exc:
        raise RuntimeError(
            f"torch or DomainClassifier unavailable: {exc}. "
            "Install torch and ensure dpms_ant/classifier/domain_classifier.py exists."
        ) from exc

    classifier = DomainClassifier(
        backbone="mobilenet_v2",
        num_classes=num_classes,
        pretrained_weights=mobilenet_weights,
    )

    if checkpoint_path and pathlib.Path(checkpoint_path).exists():
        state = torch.load(checkpoint_path, map_location="cpu")
        classifier.load_state_dict(state.get("classifier", state))

    return classifier


# ---------------------------------------------------------------------------
# Checkpoint save / restore
# reference_grounding: paper_semantic_chunk_014_01 training checkpoint management
# ---------------------------------------------------------------------------


def save_checkpoint(
    step: int,
    adaptor_state: Dict[str, Any],
    classifier_state: Optional[Dict[str, Any]],
    optimizer_state: Optional[Dict[str, Any]],
    out_dir: Union[str, pathlib.Path],
    name: Optional[str] = None,
) -> pathlib.Path:
    """
    Save adaptor + classifier checkpoint to disk.

    Parameters
    ----------
    step : int
        Current training step.
    adaptor_state : dict
        ``state_dict`` of adaptor parameters.
    classifier_state : dict, optional
        ``state_dict`` of domain classifier.
    optimizer_state : dict, optional
        Optimizer ``state_dict`` for exact resumption.
    out_dir : str | Path
        Destination directory.
    name : str, optional
        Checkpoint file stem; default ``ckpt_{step:06d}.pt``.

    Returns
    -------
    pathlib.Path
        Written checkpoint path.
    """
    try:
        import torch  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "torch is required to save checkpoints. pip install torch"
        ) from exc

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = name if name else f"ckpt_{step:06d}.pt"
    ckpt_path = out_dir / fname
    payload: Dict[str, Any] = {
        "step": step,
        "adaptor": adaptor_state,
        "classifier": classifier_state if classifier_state is not None else {},
        "optimizer": optimizer_state if optimizer_state is not None else {},
    }
    torch.save(payload, ckpt_path)
    return ckpt_path


def load_checkpoint(
    checkpoint_path: Union[str, pathlib.Path],
    map_location: str = "cpu",
) -> Dict[str, Any]:
    """
    Load adaptor + classifier checkpoint.

    Parameters
    ----------
    checkpoint_path : str | Path
        Path to ``.pt`` checkpoint file.
    map_location : str
        PyTorch device map_location; default ``'cpu'``.

    Returns
    -------
    dict
        Keys: ``step`` (int), ``adaptor`` (dict), ``classifier`` (dict),
        ``optimizer`` (dict).
    """
    try:
        import torch  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "torch is required to load checkpoints. pip install torch"
        ) from exc

    path = pathlib.Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    raw = torch.load(path, map_location=map_location)
    return {
        "step": int(raw.get("step", 0)),
        "adaptor": raw.get("adaptor", {}),
        "classifier": raw.get("classifier", {}),
        "optimizer": raw.get("optimizer", {}),
    }


# ---------------------------------------------------------------------------
# Artifact writers
# reference_grounding: task_043 declared artifact paths
# ---------------------------------------------------------------------------

_DECLARED_ARTIFACTS = [
    "results/dataset_registry.json",
    "results/data_manifest.json",
    "results/domain_registry.json",
    "results/environment_registry.json",
    "results/scope_report.json",
    "results/config_resolved.json",
]


def _artifact_dir() -> pathlib.Path:
    """Resolve artifact output directory from env or default."""
    env = os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR", "")
    return pathlib.Path(env) if env else pathlib.Path("results")


def write_dataset_registry(out_dir: Optional[pathlib.Path] = None) -> pathlib.Path:
    """Write dataset registry JSON artifact.

    Returns
    -------
    pathlib.Path
        Written file path.
    """
    out = (out_dir or _artifact_dir()) / "dataset_registry.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "schema_version": "1.0",
        "artifact_type": "dataset_registry",
        "paper": (
            "Bridging Data Gaps in Diffusion Models with Adversarial "
            "Noise-Based Transfer Learning"
        ),
        "method": "DPMs-ANT",
        "description": (
            "All paper-registered datasets: imagenet, ffhq, lsun_church (source) and "
            "babies, sunglasses, raphael_peale, sketches, modigliani, "
            "haunted_houses, landscape_drawings (7 target domains, 10-shot)."
        ),
        "source_domains": SOURCE_DOMAINS,
        "target_domains": TARGET_DOMAINS,
        "domain_pairs": DOMAIN_PAIRS,
        "registry": DATASET_REGISTRY,
        "total_datasets": len(DATASET_REGISTRY),
        "total_source_domains": len(SOURCE_DOMAINS),
        "total_target_domains": len(TARGET_DOMAINS),
        "paper_shot_count": 10,
        "hyperparameters": PAPER_HYPERPARAMETERS,
    }
    out.write_text(json.dumps(payload, indent=2))
    return out


def write_data_manifest(out_dir: Optional[pathlib.Path] = None) -> pathlib.Path:
    """Write data manifest JSON artifact.

    Returns
    -------
    pathlib.Path
        Written file path.
    """
    out = (out_dir or _artifact_dir()) / "data_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    readiness_report = check_dataset_readiness()
    datasets_meta: Dict[str, Any] = {}
    for domain, info in DATASET_REGISTRY.items():
        datasets_meta[domain] = {
            "id": domain,
            "role": info.get("role", "unknown"),
            "default_root": info.get("default_root", ""),
            "image_size": info.get("image_size", 256),
            "shot": info.get("shot", None),
            "source_domain": info.get("source_domain", None),
            "framework": info.get("framework", None),
            "readiness": readiness_report.get(domain, {}),
        }
    payload: Dict[str, Any] = {
        "schema_version": "1.0",
        "artifact_type": "data_manifest",
        "description": (
            "Per-domain readiness status and dataset metadata for all 10 "
            "registered datasets in DPMs-ANT."
        ),
        "datasets": datasets_meta,
        "few_shot_domains": TARGET_DOMAINS,
        "domain_pairs": DOMAIN_PAIRS,
        "readiness_summary": {
            d: readiness_report.get(d, {}).get("status", "unknown")
            for d in DATASET_REGISTRY
        },
    }
    out.write_text(json.dumps(payload, indent=2))
    return out


def write_domain_registry(out_dir: Optional[pathlib.Path] = None) -> pathlib.Path:
    """Write domain registry JSON artifact.

    Returns
    -------
    pathlib.Path
        Written file path.
    """
    out = (out_dir or _artifact_dir()) / "domain_registry.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    experiment_pairs = []
    for p in DOMAIN_PAIRS:
        s_info = DATASET_REGISTRY[p["source"]]
        t_info = DATASET_REGISTRY[p["target"]]
        experiment_pairs.append(
            {
                "pair_id": f"{p['source']}_to_{p['target']}",
                "source": p["source"],
                "target": p["target"],
                "framework": p["framework"],
                "shot": t_info.get("shot", 10),
                "source_default_root": s_info.get("default_root", ""),
                "target_default_root": t_info.get("default_root", ""),
                "target_image_size": t_info.get("image_size", 256),
                "fid_reference": t_info.get("fid_reference", ""),
            }
        )
    payload: Dict[str, Any] = {
        "schema_version": "1.0",
        "artifact_type": "domain_registry",
        "description": (
            "Source-target domain pairs for Table 2 experiments "
            "(DDPM framework, 10-shot). 5 FFHQ→face/art targets + "
            "2 LSUN-Church→architecture/landscape targets."
        ),
        "source_domains": {d: DATASET_REGISTRY[d] for d in SOURCE_DOMAINS},
        "target_domains": {d: DATASET_REGISTRY[d] for d in TARGET_DOMAINS},
        "experiment_pairs": experiment_pairs,
        "total_pairs": len(experiment_pairs),
    }
    out.write_text(json.dumps(payload, indent=2))
    return out


def write_environment_registry(out_dir: Optional[pathlib.Path] = None) -> pathlib.Path:
    """Write environment registry JSON artifact.

    Returns
    -------
    pathlib.Path
        Written file path.
    """
    out = (out_dir or _artifact_dir()) / "environment_registry.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    data_env_vars: Dict[str, Any] = {}
    for domain, info in DATASET_REGISTRY.items():
        env_key = info.get("data_root_env", "")
        if env_key:
            data_env_vars[env_key] = {
                "domain": domain,
                "current_value": os.environ.get(env_key, ""),
                "default_root": info.get("default_root", ""),
                "description": f"Root directory for {domain} dataset images",
                "is_set": bool(os.environ.get(env_key, "")),
            }

    payload: Dict[str, Any] = {
        "schema_version": "1.0",
        "artifact_type": "environment_registry",
        "description": (
            "Environment variable names and current/default root paths for "
            "all registered datasets in DPMs-ANT."
        ),
        "data_environment_variables": data_env_vars,
        "classifier_environment": {
            "MOBILENET_WEIGHTS": {
                "description": (
                    "Path to ImageNet-pretrained MobileNetV2 weights for "
                    "domain classifier fine-tuning (last layer → 2 classes)."
                ),
                "current_value": os.environ.get("MOBILENET_WEIGHTS", ""),
                "default": "pretrained/mobilenet_v2.pth",
                "is_set": bool(os.environ.get("MOBILENET_WEIGHTS", "")),
            },
            "DOMAIN_CLASSIFIER_CKPT": {
                "description": "Fine-tuned domain classifier checkpoint path",
                "current_value": os.environ.get("DOMAIN_CLASSIFIER_CKPT", ""),
                "default": "checkpoints/domain_classifier.pt",
                "is_set": bool(os.environ.get("DOMAIN_CLASSIFIER_CKPT", "")),
            },
        },
        "pretrained_model_environment": {
            "DDPM_FFHQ_CKPT": {
                "description": "Pretrained DDPM FFHQ-256 checkpoint path",
                "current_value": os.environ.get("DDPM_FFHQ_CKPT", ""),
                "default": "pretrained/ddpm_ffhq_256.pt",
                "is_set": bool(os.environ.get("DDPM_FFHQ_CKPT", "")),
            },
            "DDPM_CHURCH_CKPT": {
                "description": "Pretrained DDPM LSUN-Church-256 checkpoint path",
                "current_value": os.environ.get("DDPM_CHURCH_CKPT", ""),
                "default": "pretrained/ddpm_lsun_church_256.pt",
                "is_set": bool(os.environ.get("DDPM_CHURCH_CKPT", "")),
            },
            "LDM_FFHQ_CKPT": {
                "description": "Pretrained LDM FFHQ-256 checkpoint path",
                "current_value": os.environ.get("LDM_FFHQ_CKPT", ""),
                "default": "pretrained/ldm_ffhq_256.pt",
                "is_set": bool(os.environ.get("LDM_FFHQ_CKPT", "")),
            },
        },
        "paperbench_artifact_dir": {
            "PAPERBENCH_REPRO_ARTIFACT_DIR": {
                "description": "Auxiliary artifact output directory override",
                "current_value": os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR", ""),
                "default": "results",
                "is_set": bool(os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR", "")),
            }
        },
    }
    out.write_text(json.dumps(payload, indent=2))
    return out


def write_scope_report(out_dir: Optional[pathlib.Path] = None) -> pathlib.Path:
    """Write scope report JSON artifact.

    Returns
    -------
    pathlib.Path
        Written file path.
    """
    out = (out_dir or _artifact_dir()) / "scope_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "schema_version": "1.0",
        "artifact_type": "scope_report",
        "paper": (
            "Bridging Data Gaps in Diffusion Models with Adversarial "
            "Noise-Based Transfer Learning"
        ),
        "method": "DPMs-ANT",
        "core_contributions": [
            "Similarity-Guided Training: domain classifier (MobileNetV2, 2-class) "
            "provides KL-divergence guidance over noisy images",
            "Adversarial Noise Selection (ANT): PGD inner loop selects worst-case "
            "noise perturbations maximising adaptation loss (omega=0.02, 10 steps)",
            "Shift Adaptor: bottleneck W_down/W_up parameter-efficient fine-tuning; "
            "down-pool + GroupNorm + 3x3 conv → 4-head attention → MLP(→8 or 16) "
            "→ upsample×4 + GroupNorm + 3x3 conv",
        ],
        "frameworks": ["ddpm", "ldm"],
        "source_domains": SOURCE_DOMAINS,
        "target_domains": TARGET_DOMAINS,
        "shot_count": 10,
        "experiment_pairs": DOMAIN_PAIRS,
        "evaluation_metrics": ["fid", "accuracy", "intra_lpips", "fidelity_score"],
        "baselines": [
            "fine_tune",
            "ewc",
            "ada",
            "cdc",
            "dcl",
            "ddpm_pa",
        ],
        "adaptor_config": {
            "ddpm": {"c": 4, "d": 8, "description": "DDPM: c=4 compression, d=8 layers"},
            "ldm": {"c": 2, "d": 8, "description": "LDM: c=2 compression, d=8 layers"},
            "architecture_detail": (
                "down-pool → GroupNorm → 3×3 conv → 4-head attention → "
                "MLP(feature→8 or 16) → upsample×4 → GroupNorm → 3×3 conv"
            ),
        },
        "classifier_config": {
            "backbone": "MobileNetV2",
            "pretrain": "ImageNet",
            "output_classes": 2,
            "class_0": "source domain",
            "class_1": "target domain",
            "description": (
                "ImageNet-pretrained MobileNetV2 fine-tuned with last FC layer "
                "replaced to output 2 classes (source vs target domain). "
                "reference_grounding: paper_semantic_chunk_014_01 "
                "classifier_loader_finetuning"
            ),
        },
        "training_hyperparameters": PAPER_HYPERPARAMETERS,
        "declared_artifacts": _DECLARED_ARTIFACTS,
    }
    out.write_text(json.dumps(payload, indent=2))
    return out


def write_config_resolved(
    config: Optional[Dict[str, Any]] = None,
    out_dir: Optional[pathlib.Path] = None,
) -> pathlib.Path:
    """Write resolved configuration JSON artifact.

    Parameters
    ----------
    config : dict, optional
        Runtime configuration overrides merged on top of defaults.
    out_dir : Path, optional
        Output directory override.

    Returns
    -------
    pathlib.Path
        Written file path.
    """
    out = (out_dir or _artifact_dir()) / "config_resolved.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    default_config: Dict[str, Any] = {
        "framework": "ddpm",
        "method": "dpms_ant",
        "source_domain": "ffhq",
        "target_domain": "babies",
        "shot": 10,
        "image_size": 256,
        "batch_size": 64,
        "total_iterations": 5000,
        "omega": 0.02,
        "adversarial_inner_steps": 10,
        "similarity_guidance_scale_gamma": 5,
        "adaptor_c": 4,
        "adaptor_d": 8,
        "augmentation": ["random_horizontal_flip"],
        "normalize_mean": [0.5, 0.5, 0.5],
        "normalize_std": [0.5, 0.5, 0.5],
        "classifier_backbone": "mobilenet_v2",
        "classifier_num_classes": 2,
        "mobilenet_weights": os.environ.get(
            "MOBILENET_WEIGHTS", "pretrained/mobilenet_v2.pth"
        ),
        "ddpm_ffhq_ckpt": os.environ.get("DDPM_FFHQ_CKPT", "pretrained/ddpm_ffhq_256.pt"),
        "ddpm_church_ckpt": os.environ.get(
            "DDPM_CHURCH_CKPT", "pretrained/ddpm_lsun_church_256.pt"
        ),
        "ldm_ffhq_ckpt": os.environ.get("LDM_FFHQ_CKPT", "pretrained/ldm_ffhq_256.pt"),
    }
    if config:
        default_config.update(config)

    payload: Dict[str, Any] = {
        "schema_version": "1.0",
        "artifact_type": "config_resolved",
        "description": "DPMs-ANT resolved runtime configuration with paper defaults.",
        "config": default_config,
        "paper_hyperparameters": PAPER_HYPERPARAMETERS,
        "domain_registry_size": len(DATASET_REGISTRY),
    }
    out.write_text(json.dumps(payload, indent=2))
    return out


def write_all_artifacts(
    config: Optional[Dict[str, Any]] = None,
    out_dir: Optional[pathlib.Path] = None,
) -> Dict[str, str]:
    """
    Write all declared artifacts for this module.

    Artifacts written
    -----------------
    - results/dataset_registry.json
    - results/data_manifest.json
    - results/domain_registry.json
    - results/environment_registry.json
    - results/scope_report.json
    - results/config_resolved.json

    Parameters
    ----------
    config : dict, optional
        Config overrides forwarded to ``write_config_resolved``.
    out_dir : Path, optional
        Output directory override.

    Returns
    -------
    dict
        Mapping artifact_key → absolute file path string.
    """
    d = out_dir or _artifact_dir()
    paths: Dict[str, str] = {}
    paths["dataset_registry"] = str(write_dataset_registry(d))
    paths["data_manifest"] = str(write_data_manifest(d))
    paths["domain_registry"] = str(write_domain_registry(d))
    paths["environment_registry"] = str(write_environment_registry(d))
    paths["scope_report"] = str(write_scope_report(d))
    paths["config_resolved"] = str(write_config_resolved(config, d))
    return paths


# ---------------------------------------------------------------------------
# Module-level smoke test / CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=== DPMs-ANT src/data/data.py smoke test ===")
    print(f"Registered datasets ({len(DATASET_REGISTRY)}): {list(DATASET_REGISTRY.keys())}")
    print(f"Source domains: {SOURCE_DOMAINS}")
    print(f"Target domains ({len(TARGET_DOMAINS)}): {TARGET_DOMAINS}")
    print(f"Domain pairs: {len(DOMAIN_PAIRS)}")

    # Write all artifacts
    print("\nWriting artifacts ...")
    artifacts = write_all_artifacts()
    for k, v in artifacts.items():
        size = pathlib.Path(v).stat().st_size if pathlib.Path(v).exists() else 0
        print(f"  [{k}] {v}  ({size} bytes)")

    # Dataset readiness check
    print("\nDataset readiness:")
    for domain in TARGET_DOMAINS:
        ds = FewShotDataset(domain=domain)
        r = ds.readiness()
        print(
            f"  {domain:22s}: status={r['status']!s:15s}  "
            f"root_exists={r['root_exists']}  found={r['found_images']}"
        )

    # FewShotDataset basic wiring
    ds = FewShotDataset(domain="babies", shot=10)
    print(f"\nFewShotDataset('babies', shot=10): len={len(ds)}")

    # make_dataset
    ds2 = make_dataset({"domain": "sunglasses", "shot": 10})
    print(f"make_dataset({{'domain': 'sunglasses'}}): len={len(ds2)}")

    # make_fewshot_dataset
    ds3 = make_fewshot_dataset({"source": "lsun_church", "target": "haunted_houses"})
    print(f"make_fewshot_dataset(lsun_church→haunted_houses): len={len(ds3)}")

    print("\n=== smoke test complete ===")
    sys.exit(0)