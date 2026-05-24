"""
src/dataset_registry.py
=======================
Paper: "Bridging Data Gaps in Diffusion Models with Adversarial Noise-Based Transfer Learning"

Dataset/benchmark registry for DPMs-ANT few-shot domain adaptation.

reference_grounding: paper_semantic_chunk_012 10-shot image generation experiments
reference_grounding: paper_semantic_chunk_014_01 DDPM + LDM framework evaluation
reference_grounding: paper_dataset_inventory dataset domains

Explicitly registered dataset/benchmark aliases (paper evidence contract):
  imagenet | ffhq | lsun_church | babies | sunglasses | raphael_peale |
  sketches | modigliani | haunted_houses | landscape_drawings

Implementation surfaces: data_pipeline, config, artifact_writer, evaluation, environment

Classifier note (addendum binding):
  MobileNet fine-tuned with last layer modified to 2 output classes
  (source vs target domain classification).
  reference_grounding: paper_semantic_chunk_014_01_classifier_loader_finetuning

Adaptor architecture note (addendum binding):
  Down-pooling + norm + 3x3 conv → 4-head attention → MLP (8 or 16) →
  up-sampling x4 + norm + 3x3 conv.
  reference_grounding: paper_semantic_chunk_014_01 adaptor architecture
"""

from __future__ import annotations

import copy
import json
import logging
import os
import pathlib
import random
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paper-derived source-domain registry entries
# reference_grounding: paper_dataset_inventory source domain definitions
# ---------------------------------------------------------------------------

SOURCE_DOMAINS: Dict[str, Dict[str, Any]] = {
    "imagenet": {
        "id": "imagenet",
        "alias": ["imagenet", "imagenet1k", "ilsvrc2012", "imagenet_1k"],
        "type": "source",
        "framework": ["ddpm", "ldm"],
        "image_size": 256,
        "num_classes": 1000,
        "description": (
            "ImageNet-1K classification dataset used for MobileNet pre-training "
            "and as classifier source-domain reference."
        ),
        "default_root": "data/imagenet",
        "split": "train",
        "classifier_pretrain": True,
    },
    "ffhq": {
        "id": "ffhq",
        "alias": ["ffhq", "ffhq256", "flickr_faces_hq", "ffhq_256"],
        "type": "source",
        "framework": ["ddpm", "ldm"],
        "image_size": 256,
        "num_classes": None,
        "description": (
            "Flickr-Faces-HQ (256×256) dataset; DDPM/LDM source domain "
            "for face-domain transfer experiments."
        ),
        "default_root": "data/ffhq",
        "split": "train",
        "target_domains": [
            "babies", "sunglasses", "raphael_peale", "sketches", "modigliani"
        ],
        "pretrained_checkpoint": "checkpoints/ddpm_ffhq_256.pt",
    },
    "lsun_church": {
        "id": "lsun_church",
        "alias": ["lsun_church", "lsun-church", "church", "lsun_churches"],
        "type": "source",
        "framework": ["ddpm"],
        "image_size": 256,
        "num_classes": None,
        "description": (
            "LSUN-Church (256×256) dataset; DDPM source domain for "
            "scene/landscape transfer experiments."
        ),
        "default_root": "data/lsun_church",
        "split": "train",
        "target_domains": ["haunted_houses", "landscape_drawings"],
        "pretrained_checkpoint": "checkpoints/ddpm_lsun_church_256.pt",
    },
}

# ---------------------------------------------------------------------------
# Paper-derived target-domain registry entries (7 few-shot domains)
# reference_grounding: paper_semantic_chunk_012 Table 2 – 7 target domains
# ---------------------------------------------------------------------------

TARGET_DOMAINS: Dict[str, Dict[str, Any]] = {
    "babies": {
        "id": "babies",
        "alias": ["babies", "baby_faces", "metfaces_babies", "baby"],
        "type": "target",
        "source_domain": "ffhq",
        "framework": ["ddpm", "ldm"],
        "image_size": 256,
        "shot": 10,
        "description": (
            "10-shot babies-face images; FFHQ→Babies adaptation target. "
            "reference_grounding: paper_semantic_chunk_012 Table 2"
        ),
        "default_root": "data/babies",
        "split": "train",
        "augment": True,
    },
    "sunglasses": {
        "id": "sunglasses",
        "alias": ["sunglasses", "glasses", "sunglasses_faces"],
        "type": "target",
        "source_domain": "ffhq",
        "framework": ["ddpm", "ldm"],
        "image_size": 256,
        "shot": 10,
        "description": (
            "10-shot sunglasses-face images; FFHQ→Sunglasses adaptation target. "
            "reference_grounding: paper_semantic_chunk_012 Table 2"
        ),
        "default_root": "data/sunglasses",
        "split": "train",
        "augment": True,
    },
    "raphael_peale": {
        "id": "raphael_peale",
        "alias": [
            "raphael_peale", "raphael", "peale", "portrait_paintings",
            "raphael-peale",
        ],
        "type": "target",
        "source_domain": "ffhq",
        "framework": ["ddpm", "ldm"],
        "image_size": 256,
        "shot": 10,
        "description": (
            "10-shot Raphael Peale portrait paintings; FFHQ→Raphael Peale adaptation. "
            "reference_grounding: paper_semantic_chunk_012 Table 2"
        ),
        "default_root": "data/raphael_peale",
        "split": "train",
        "augment": True,
    },
    "sketches": {
        "id": "sketches",
        "alias": ["sketches", "face_sketches", "pencil_sketches", "sketch"],
        "type": "target",
        "source_domain": "ffhq",
        "framework": ["ddpm", "ldm"],
        "image_size": 256,
        "shot": 10,
        "description": (
            "10-shot face sketch images; FFHQ→Sketches adaptation target. "
            "reference_grounding: paper_semantic_chunk_012 Table 2"
        ),
        "default_root": "data/sketches",
        "split": "train",
        "augment": True,
    },
    "modigliani": {
        "id": "modigliani",
        "alias": ["modigliani", "modigliani_paintings", "modigliani_faces"],
        "type": "target",
        "source_domain": "ffhq",
        "framework": ["ddpm", "ldm"],
        "image_size": 256,
        "shot": 10,
        "description": (
            "10-shot Modigliani painting images; FFHQ→Modigliani adaptation target. "
            "reference_grounding: paper_semantic_chunk_012 Table 2"
        ),
        "default_root": "data/modigliani",
        "split": "train",
        "augment": True,
    },
    "haunted_houses": {
        "id": "haunted_houses",
        "alias": [
            "haunted_houses", "haunted", "spooky_houses",
            "haunted-houses", "hauntedhouses",
        ],
        "type": "target",
        "source_domain": "lsun_church",
        "framework": ["ddpm"],
        "image_size": 256,
        "shot": 10,
        "description": (
            "10-shot haunted-house images; LSUN-Church→Haunted Houses adaptation. "
            "reference_grounding: paper_semantic_chunk_012 Table 2"
        ),
        "default_root": "data/haunted_houses",
        "split": "train",
        "augment": True,
    },
    "landscape_drawings": {
        "id": "landscape_drawings",
        "alias": [
            "landscape_drawings", "landscape", "drawings",
            "landscape-drawings", "landscapes",
        ],
        "type": "target",
        "source_domain": "lsun_church",
        "framework": ["ddpm"],
        "image_size": 256,
        "shot": 10,
        "description": (
            "10-shot landscape-drawing images; LSUN-Church→Landscape Drawings adaptation. "
            "reference_grounding: paper_semantic_chunk_012 Table 2"
        ),
        "default_root": "data/landscape_drawings",
        "split": "train",
        "augment": True,
    },
}

# ---------------------------------------------------------------------------
# Unified registry (all 10 domains: 3 source + 7 target)
# ---------------------------------------------------------------------------

DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {}
DATASET_REGISTRY.update(SOURCE_DOMAINS)
DATASET_REGISTRY.update(TARGET_DOMAINS)

# Alias lookup table (normalised to lowercase)
_ALIAS_MAP: Dict[str, str] = {}
for _did, _entry in DATASET_REGISTRY.items():
    _ALIAS_MAP[_did.lower()] = _did
    for _alias in _entry.get("alias", []):
        _ALIAS_MAP[_alias.lower()] = _did

# ---------------------------------------------------------------------------
# Domain-pair registry  (source → target)
# reference_grounding: paper_semantic_chunk_012 Table 2 experiment matrix
# ---------------------------------------------------------------------------

DOMAIN_PAIRS: List[Tuple[str, str]] = [
    ("ffhq", "babies"),
    ("ffhq", "sunglasses"),
    ("ffhq", "raphael_peale"),
    ("ffhq", "sketches"),
    ("ffhq", "modigliani"),
    ("lsun_church", "haunted_houses"),
    ("lsun_church", "landscape_drawings"),
]

# ---------------------------------------------------------------------------
# MobileNet domain-classifier configuration
# reference_grounding: paper_semantic_chunk_014_01_classifier_loader_finetuning
# Addendum: "fine-tuned by modifying the last layer to output two classes to
# classify whether images were coming from the source or the target dataset."
# ---------------------------------------------------------------------------

MOBILENET_CONFIG: Dict[str, Any] = {
    "architecture": "mobilenet_v2",
    "pretrained_source": "imagenet",
    "num_output_classes": 2,
    "fine_tune_strategy": "last_layer",
    "input_size": (3, 256, 256),
    "normalize_mean": [0.5, 0.5, 0.5],
    "normalize_std": [0.5, 0.5, 0.5],
    "description": (
        "MobileNet V2 pre-trained on ImageNet; last fully-connected layer "
        "replaced with Linear(in_features, 2) for binary source/target domain "
        "classification used in similarity-guided training. "
        "reference_grounding: paper_semantic_chunk_014_01_classifier_loader_finetuning"
    ),
}

# ---------------------------------------------------------------------------
# Adaptor architecture reference constants (addendum binding)
# reference_grounding: paper_semantic_chunk_014_01 adaptor architecture Section 4
# ---------------------------------------------------------------------------

ADAPTOR_ARCH_CONFIG: Dict[str, Any] = {
    "description": (
        "Shift Adaptor: down-pooling → GroupNorm → 3×3 Conv → "
        "4-head self-attention → MLP (bottleneck 8 or 16) → "
        "up-sample ×4 → GroupNorm → 3×3 Conv."
    ),
    "ddpm_c": 4,          # compression ratio c=4 (DDPM framework)
    "ddpm_d": 8,          # number of adaptor layers d=8 (DDPM framework)
    "ldm_c": 4,           # compression ratio c=4 (LDM framework)
    "ldm_d": 4,           # number of adaptor layers d=4 (LDM framework)
    "attention_heads": 4,
    "mlp_bottleneck_choices": [8, 16],
    "upsample_factor": 4,
    "reference": "paper_semantic_chunk_014_01 adaptor architecture Section 4",
}

# ---------------------------------------------------------------------------
# Registry lookup helpers
# ---------------------------------------------------------------------------


def resolve_dataset_id(name: str) -> str:
    """Resolve a dataset name or alias to its canonical registry ID."""
    key = name.lower().strip()
    if key in _ALIAS_MAP:
        return _ALIAS_MAP[key]
    raise KeyError(
        f"Unknown dataset '{name}'. "
        f"Available IDs/aliases: {sorted(_ALIAS_MAP.keys())}"
    )


def get_dataset_entry(name: str) -> Dict[str, Any]:
    """Return the full registry entry for a dataset by id or alias."""
    return DATASET_REGISTRY[resolve_dataset_id(name)]


def list_source_domains() -> List[str]:
    """Return all registered source-domain IDs."""
    return [k for k, v in DATASET_REGISTRY.items() if v["type"] == "source"]


def list_target_domains() -> List[str]:
    """Return all registered target-domain IDs (7 few-shot domains)."""
    return [k for k, v in DATASET_REGISTRY.items() if v["type"] == "target"]


def list_domain_pairs() -> List[Tuple[str, str]]:
    """Return all paper-registered (source, target) domain pairs."""
    return list(DOMAIN_PAIRS)


# ---------------------------------------------------------------------------
# FewShotDataset
# reference_grounding: paper_semantic_chunk_012 10-shot image generation experiments
# reference_grounding: paper_semantic_chunk_014_01 experimental setup – 10 target images
# ---------------------------------------------------------------------------

class FewShotDataset:
    """
    Generic few-shot dataset for DPMs-ANT target-domain adaptation.

    Supports:
    - Exactly ``shot`` images (default 10) sampled from the target-domain folder.
    - Random horizontal flip augmentation (paper default).
    - Reproducible seed-based subsetting.
    - Graceful synthetic-fallback for smoke/dry-run validation when real data
      is not present.

    Implements the map-style dataset protocol (``__len__`` / ``__getitem__``)
    so it can be wrapped by ``torch.utils.data.DataLoader`` via
    ``as_dataloader()``.

    Paper note (addendum):
      batch_size=64 training uses oversampled repetitions of the 10-shot set.

    reference_grounding: paper_semantic_chunk_012 10-shot image generation
    reference_grounding: paper_semantic_chunk_014_01 experimental setup
    """

    def __init__(
        self,
        domain: str,
        root: Optional[str] = None,
        shot: int = 10,
        image_size: int = 256,
        augment: bool = True,
        seed: int = 42,
        transform=None,
    ):
        self.domain_id = resolve_dataset_id(domain)
        self.entry = DATASET_REGISTRY[self.domain_id]
        self.shot = shot
        self.image_size = image_size
        self.augment = augment
        self.seed = seed
        self._transform = transform

        if root is not None:
            self.root = pathlib.Path(root)
        else:
            env_root = os.environ.get("DPMS_ANT_DATA_ROOT", "data")
            self.root = (
                pathlib.Path(env_root) /
                self.entry.get("default_root", self.domain_id)
            )

        self._image_paths: List[pathlib.Path] = []
        self._loaded: bool = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _discover_images(self) -> List[pathlib.Path]:
        """Scan root directory recursively for image files."""
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
        if not self.root.exists():
            logger.warning(
                "FewShotDataset[%s]: root '%s' does not exist – "
                "synthetic fallback will be used for missing samples.",
                self.domain_id, self.root,
            )
            return []
        return sorted(
            p for p in self.root.rglob("*") if p.suffix.lower() in exts
        )

    def _select_shots(self, paths: List[pathlib.Path]) -> List[pathlib.Path]:
        """Select exactly ``shot`` images with a reproducible RNG."""
        if not paths:
            return []
        rng = random.Random(self.seed)
        if len(paths) <= self.shot:
            return list(paths)
        return rng.sample(paths, self.shot)

    def _build_transform(self):
        """Build a torchvision transform pipeline (lazy import)."""
        try:
            import torchvision.transforms as T  # lazy
            ops: list = [T.Resize((self.image_size, self.image_size))]
            if self.augment:
                ops.append(T.RandomHorizontalFlip())
            ops += [
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
            return T.Compose(ops)
        except ImportError:
            return None

    def _synthetic_tensor(self, idx: int):
        """Return a deterministic synthetic tensor (smoke fallback)."""
        try:
            import torch  # lazy
            g = torch.Generator().manual_seed(self.seed + idx)
            return torch.randn(3, self.image_size, self.image_size, generator=g)
        except ImportError:
            return None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self) -> "FewShotDataset":
        """Discover and subset image files."""
        all_paths = self._discover_images()
        self._image_paths = self._select_shots(all_paths)
        self._loaded = True
        logger.info(
            "FewShotDataset[%s]: %d/%d images selected (shot=%d, root=%s)",
            self.domain_id, len(self._image_paths), len(all_paths),
            self.shot, self.root,
        )
        return self

    def __len__(self) -> int:
        if not self._loaded:
            self.load()
        # If no real images are found, report shot count so DataLoader can
        # iterate using synthetic fallback tensors.
        return max(len(self._image_paths), self.shot)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        """Return (image_tensor, label=1) – label 1 denotes target domain."""
        if not self._loaded:
            self.load()

        if idx >= len(self._image_paths):
            # Smoke / missing-data fallback
            tensor = self._synthetic_tensor(idx)
            return tensor, 1

        path = self._image_paths[idx]
        try:
            from PIL import Image as PILImage  # lazy
            img = PILImage.open(path).convert("RGB")
            if self._transform is not None:
                tensor = self._transform(img)
            else:
                tfm = self._build_transform()
                if tfm is not None:
                    tensor = tfm(img)
                else:
                    # Minimal numpy fallback
                    import numpy as np  # lazy
                    import torch          # lazy
                    arr = (
                        np.array(
                            img.resize((self.image_size, self.image_size)),
                            dtype="float32",
                        ) / 127.5 - 1.0
                    )
                    tensor = torch.from_numpy(arr.transpose(2, 0, 1))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "FewShotDataset[%s]: failed to load '%s': %s – using fallback.",
                self.domain_id, path, exc,
            )
            tensor = self._synthetic_tensor(idx)

        return tensor, 1  # label 1 = target domain

    def as_dataloader(
        self,
        batch_size: int = 10,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        """
        Wrap this dataset in a ``torch.utils.data.DataLoader``.

        Paper trains with batch_size=64 via oversampled repetitions of the
        10-shot set; the DataLoader batch size should be set accordingly by
        the trainer, not here.
        reference_grounding: paper_semantic_chunk_014_01 batch_size=64
        """
        try:
            import torch.utils.data as td  # lazy
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is required for DataLoader. "
                "Install via: pip install torch"
            ) from exc

        effective_batch = min(batch_size, len(self))
        return td.DataLoader(
            self,
            batch_size=effective_batch,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    def readiness_info(self) -> Dict[str, Any]:
        """Return a readiness/manifest dict for this dataset instance."""
        if not self._loaded:
            self.load()
        return {
            "domain_id": self.domain_id,
            "root": str(self.root),
            "root_exists": self.root.exists(),
            "num_discovered": len(self._image_paths),
            "shot": self.shot,
            "image_size": self.image_size,
            "augment": self.augment,
            "seed": self.seed,
            "registry_entry": self.entry,
        }


# ---------------------------------------------------------------------------
# SourceDomainDataset (binary classifier negative class)
# reference_grounding: paper_semantic_chunk_014_01_classifier_loader_finetuning
# ---------------------------------------------------------------------------

class SourceDomainDataset:
    """
    Source-domain images used as label-0 examples in the binary domain
    classifier (source=0, target=1).

    Addendum binding:
      "These pre-trained models were fine-tuned by modifying the last layer
      to output two classes to classify whether images were coming from the
      source or the target dataset."
    reference_grounding: paper_semantic_chunk_014_01_classifier_loader_finetuning
    """

    def __init__(
        self,
        domain: str,
        root: Optional[str] = None,
        image_size: int = 256,
        max_samples: int = 1000,
        augment: bool = True,
        seed: int = 0,
        transform=None,
    ):
        self.domain_id = resolve_dataset_id(domain)
        self.entry = DATASET_REGISTRY[self.domain_id]
        if self.entry["type"] != "source":
            raise ValueError(
                f"SourceDomainDataset expects a source domain; got '{self.domain_id}'"
            )
        self.image_size = image_size
        self.max_samples = max_samples
        self.augment = augment
        self.seed = seed
        self._transform = transform

        if root is not None:
            self.root = pathlib.Path(root)
        else:
            env_root = os.environ.get("DPMS_ANT_DATA_ROOT", "data")
            self.root = (
                pathlib.Path(env_root) /
                self.entry.get("default_root", self.domain_id)
            )

        self._image_paths: List[pathlib.Path] = []
        self._loaded: bool = False

    def load(self) -> "SourceDomainDataset":
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        if not self.root.exists():
            logger.warning(
                "SourceDomainDataset[%s]: root '%s' does not exist.",
                self.domain_id, self.root,
            )
            self._loaded = True
            return self
        paths = sorted(
            p for p in self.root.rglob("*") if p.suffix.lower() in exts
        )
        rng = random.Random(self.seed)
        if len(paths) > self.max_samples:
            paths = rng.sample(paths, self.max_samples)
        self._image_paths = paths
        self._loaded = True
        return self

    def __len__(self) -> int:
        if not self._loaded:
            self.load()
        return max(len(self._image_paths), 1)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        """Return (image_tensor, label=0) – label 0 denotes source domain."""
        if not self._loaded:
            self.load()

        if idx >= len(self._image_paths):
            try:
                import torch  # lazy
                g = torch.Generator().manual_seed(self.seed + idx + 99999)
                tensor = torch.randn(
                    3, self.image_size, self.image_size, generator=g
                )
            except ImportError:
                tensor = None
            return tensor, 0

        path = self._image_paths[idx]
        try:
            from PIL import Image as PILImage  # lazy
            import torchvision.transforms as T  # lazy
            img = PILImage.open(path).convert("RGB")
            ops = [T.Resize((self.image_size, self.image_size))]
            if self.augment:
                ops.append(T.RandomHorizontalFlip())
            ops += [
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
            tensor = T.Compose(ops)(img) if self._transform is None else self._transform(img)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "SourceDomainDataset[%s]: load error for '%s': %s",
                self.domain_id, path, exc,
            )
            try:
                import torch  # lazy
                tensor = torch.zeros(3, self.image_size, self.image_size)
            except ImportError:
                tensor = None

        return tensor, 0  # label 0 = source domain


# ---------------------------------------------------------------------------
# Factory functions
# reference_grounding: paper_dataset_inventory dataset factory interface
# ---------------------------------------------------------------------------


def make_dataset(config: Dict[str, Any]) -> Union[FewShotDataset, SourceDomainDataset]:
    """
    Factory: build a dataset from a configuration dictionary.

    Config keys
    -----------
    domain / target_domain : str   – registry id or alias (required)
    root : str                     – optional override for data root
    shot : int                     – few-shot count (default 10)
    image_size : int               – spatial resolution (default 256)
    augment : bool                 – horizontal flip augmentation
    seed : int                     – reproducibility seed
    type : str                     – 'fewshot' | 'source' (auto from registry)
    max_samples : int              – max source images (source datasets)

    Returns
    -------
    FewShotDataset or SourceDomainDataset
    """
    domain = config.get("domain") or config.get("target_domain")
    if domain is None:
        raise ValueError(
            "make_dataset: config must contain 'domain' or 'target_domain'."
        )

    canonical_id = resolve_dataset_id(domain)
    entry = DATASET_REGISTRY[canonical_id]
    dtype = config.get("type", entry["type"])

    common: Dict[str, Any] = dict(
        root=config.get("root"),
        image_size=config.get("image_size", entry.get("image_size", 256)),
        augment=config.get("augment", entry.get("augment", True)),
        seed=config.get("seed", 42),
    )

    if dtype == "source":
        return SourceDomainDataset(
            domain=canonical_id,
            max_samples=config.get("max_samples", 1000),
            **common,
        )
    return FewShotDataset(
        domain=canonical_id,
        shot=config.get("shot", entry.get("shot", 10)),
        **common,
    )


def make_fewshot_dataset(
    pair: Tuple[str, str],
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[FewShotDataset, SourceDomainDataset]:
    """
    Build a (target_fewshot_dataset, source_domain_dataset) pair for a
    given (source_domain, target_domain) experiment tuple.

    Parameters
    ----------
    pair : (source_domain_id, target_domain_id)
    config : optional override dict (image_size, shot, augment, seed,
             target_root, source_root, source_max_samples)

    Returns
    -------
    (FewShotDataset, SourceDomainDataset)

    reference_grounding: paper_semantic_chunk_012 domain pair training setup
    """
    cfg = config or {}
    source_id = resolve_dataset_id(pair[0])
    target_id = resolve_dataset_id(pair[1])

    target_ds = FewShotDataset(
        domain=target_id,
        root=cfg.get("target_root"),
        shot=cfg.get("shot", 10),
        image_size=cfg.get("image_size", 256),
        augment=cfg.get("augment", True),
        seed=cfg.get("seed", 42),
    )
    source_ds = SourceDomainDataset(
        domain=source_id,
        root=cfg.get("source_root"),
        image_size=cfg.get("image_size", 256),
        max_samples=cfg.get("source_max_samples", 1000),
        augment=cfg.get("augment", True),
        seed=cfg.get("seed", 0),
    )
    return target_ds, source_ds


# ---------------------------------------------------------------------------
# Readiness checks
# ---------------------------------------------------------------------------


def check_dataset_readiness(
    domain: str,
    root: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Check whether a dataset directory exists and contains enough images.

    Returns a readiness dict with keys:
      domain_id, ready, root, num_images, required, message.
    """
    canonical_id = resolve_dataset_id(domain)
    entry = DATASET_REGISTRY[canonical_id]

    if root is not None:
        data_root = pathlib.Path(root)
    else:
        env_root = os.environ.get("DPMS_ANT_DATA_ROOT", "data")
        data_root = (
            pathlib.Path(env_root) / entry.get("default_root", canonical_id)
        )

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if not data_root.exists():
        return {
            "domain_id": canonical_id,
            "ready": False,
            "root": str(data_root),
            "num_images": 0,
            "required": entry.get("shot", 1),
            "message": f"Root directory '{data_root}' does not exist.",
        }

    n_images = sum(
        1 for p in data_root.rglob("*") if p.suffix.lower() in exts
    )
    required = entry.get("shot", 1)
    ready = n_images >= required
    return {
        "domain_id": canonical_id,
        "ready": ready,
        "root": str(data_root),
        "num_images": n_images,
        "required": required,
        "message": (
            f"OK: {n_images} images found (need {required})."
            if ready else
            f"WARN: only {n_images} images found, need {required}."
        ),
    }


def check_all_readiness() -> Dict[str, Dict[str, Any]]:
    """Run readiness check for every registered dataset."""
    return {did: check_dataset_readiness(did) for did in DATASET_REGISTRY}


# ---------------------------------------------------------------------------
# MobileNet classifier helpers
# reference_grounding: paper_semantic_chunk_014_01_classifier_loader_finetuning
# ---------------------------------------------------------------------------


def get_mobilenet_config() -> Dict[str, Any]:
    """Return MobileNet domain-classifier configuration dict."""
    return copy.deepcopy(MOBILENET_CONFIG)


def build_mobilenet_classifier(
    pretrained: bool = True,
    num_classes: int = 2,
):
    """
    Build MobileNet V2 domain classifier with a 2-class output head.

    Addendum binding:
      "fine-tuned by modifying the last layer to output two classes to
      classify whether images were coming from the source or the target dataset."
    reference_grounding: paper_semantic_chunk_014_01_classifier_loader_finetuning

    Parameters
    ----------
    pretrained : bool  – Load ImageNet pre-trained weights from torchvision.
    num_classes : int  – Number of output logits (2 for source/target binary).

    Returns
    -------
    torch.nn.Module
    """
    try:
        import torch.nn as nn  # lazy
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required to build the domain classifier. "
            "Install via: pip install torch torchvision"
        ) from exc

    try:
        import torchvision.models as tvm  # lazy

        if pretrained:
            try:
                weights = tvm.MobileNet_V2_Weights.IMAGENET1K_V1
                model = tvm.mobilenet_v2(weights=weights)
            except AttributeError:
                # torchvision < 0.13
                model = tvm.mobilenet_v2(pretrained=True)  # type: ignore[call-arg]
        else:
            try:
                model = tvm.mobilenet_v2(weights=None)
            except TypeError:
                model = tvm.mobilenet_v2(pretrained=False)  # type: ignore[call-arg]

        # Replace last classifier layer with 2-class head (addendum requirement)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    except ImportError:
        # Minimal stub when torchvision is absent
        import torch  # lazy

        class _MobileNetStub(nn.Module):
            """Minimal stand-in when torchvision is unavailable."""
            def __init__(self) -> None:
                super().__init__()
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.head = nn.Linear(3, num_classes)

            def forward(self, x: "torch.Tensor") -> "torch.Tensor":
                return self.head(self.pool(x).flatten(1))

        logger.warning(
            "torchvision not available; returning minimal MobileNet stub. "
            "Install torchvision for full ImageNet pre-trained weights."
        )
        return _MobileNetStub()


# ---------------------------------------------------------------------------
# Artifact writers
# reference_grounding: task_051 writes_artifacts specification
# ---------------------------------------------------------------------------


def _artifact_dir() -> pathlib.Path:
    base = os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR", "results")
    path = pathlib.Path(base)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_dataset_registry_artifact() -> Dict[str, str]:
    """
    Write results/dataset_registry.json.
    reference_grounding: task_051 writes_artifacts
    """
    out = _artifact_dir() / "dataset_registry.json"
    payload = {
        "artifact_type": "dataset_registry",
        "paper": (
            "Bridging Data Gaps in Diffusion Models with "
            "Adversarial Noise-Based Transfer Learning"
        ),
        "reference_grounding": "paper_dataset_inventory src/dataset_registry.py",
        "source_domains": SOURCE_DOMAINS,
        "target_domains": TARGET_DOMAINS,
        "domain_pairs": [list(p) for p in DOMAIN_PAIRS],
        "all_domain_ids": list(DATASET_REGISTRY.keys()),
        "num_source_domains": len(list_source_domains()),
        "num_target_domains": len(list_target_domains()),
        "num_domain_pairs": len(DOMAIN_PAIRS),
        "mobilenet_config": MOBILENET_CONFIG,
        "adaptor_arch": ADAPTOR_ARCH_CONFIG,
    }
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    logger.info("Wrote %s", out)
    return {"dataset_registry": str(out)}


def write_data_manifest_artifact() -> Dict[str, str]:
    """
    Write results/data_manifest.json with per-domain readiness status.
    reference_grounding: task_051 writes_artifacts
    """
    out = _artifact_dir() / "data_manifest.json"
    readiness = check_all_readiness()
    n_ready = sum(1 for r in readiness.values() if r.get("ready"))
    payload = {
        "artifact_type": "data_manifest",
        "reference_grounding": "paper_dataset_inventory src/dataset_registry.py",
        "readiness": readiness,
        "summary": {
            "total_domains": len(readiness),
            "ready": n_ready,
            "not_ready": len(readiness) - n_ready,
        },
    }
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    logger.info("Wrote %s", out)
    return {"data_manifest": str(out)}


def write_domain_registry_artifact() -> Dict[str, str]:
    """
    Write results/domain_registry.json.
    reference_grounding: task_051 writes_artifacts
    """
    out = _artifact_dir() / "domain_registry.json"
    payload = {
        "artifact_type": "domain_registry",
        "reference_grounding": "paper_semantic_chunk_012 src/dataset_registry.py",
        "source_domains": list_source_domains(),
        "target_domains": list_target_domains(),
        "domain_pairs": [
            {
                "source": s,
                "target": t,
                "source_entry": DATASET_REGISTRY[s],
                "target_entry": DATASET_REGISTRY[t],
            }
            for s, t in DOMAIN_PAIRS
        ],
        "adaptor_architecture": ADAPTOR_ARCH_CONFIG,
        "classifier": MOBILENET_CONFIG,
    }
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    logger.info("Wrote %s", out)
    return {"domain_registry": str(out)}


def write_environment_registry_artifact() -> Dict[str, str]:
    """
    Write results/environment_registry.json.
    reference_grounding: task_051 writes_artifacts
    """
    out = _artifact_dir() / "environment_registry.json"
    payload = {
        "artifact_type": "environment_registry",
        "reference_grounding": "paper_dataset_inventory src/dataset_registry.py",
        "python_version": sys.version,
        "registered_datasets": list(DATASET_REGISTRY.keys()),
        "data_root_env": os.environ.get("DPMS_ANT_DATA_ROOT", "data (default)"),
        "artifact_dir_env": os.environ.get(
            "PAPERBENCH_REPRO_ARTIFACT_DIR", "results (default)"
        ),
        "domain_pair_count": len(DOMAIN_PAIRS),
        "dataset_registry_version": "1.0.0",
    }
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    logger.info("Wrote %s", out)
    return {"environment_registry": str(out)}


def write_scope_report_artifact(
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Write results/scope_report.json.
    reference_grounding: task_051 writes_artifacts
    """
    out = _artifact_dir() / "scope_report.json"
    payload = {
        "artifact_type": "scope_report",
        "reference_grounding": "paper_dataset_inventory src/dataset_registry.py",
        "registered_datasets": list(DATASET_REGISTRY.keys()),
        "domain_pairs_count": len(DOMAIN_PAIRS),
        "shot_protocol": 10,
        "image_size": 256,
        "frameworks": ["ddpm", "ldm"],
        "config_override": config or {},
    }
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    logger.info("Wrote %s", out)
    return {"scope_report": str(out)}


def write_config_resolved_artifact(
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Write results/config_resolved.json with merged default + override config.
    reference_grounding: task_051 writes_artifacts
    """
    out = _artifact_dir() / "config_resolved.json"

    default_cfg: Dict[str, Any] = {
        "shot": 10,
        "image_size": 256,
        "augment": True,
        "seed": 42,
        "source_max_samples": 1000,
        "mobilenet_pretrained": True,
        "mobilenet_weights": "imagenet",
        "classifier_output_classes": 2,
        "adaptor": ADAPTOR_ARCH_CONFIG,
        "training": {
            "batch_size": 64,
            "total_iterations": 5000,
            "omega": 0.02,
            "adversarial_inner_steps": 10,
            "similarity_guidance_scale": 5,
            "ablation_iterations": 300,
        },
    }

    merged = copy.deepcopy(default_cfg)
    if config:
        merged.update(config)

    payload = {
        "artifact_type": "config_resolved",
        "reference_grounding": "paper_semantic_chunk_014_01 src/dataset_registry.py",
        "resolved_config": merged,
        "dataset_registry_ids": list(DATASET_REGISTRY.keys()),
    }
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    logger.info("Wrote %s", out)
    return {"config_resolved": str(out)}


def write_all_artifacts(
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Write all declared artifact paths for this module."""
    results: Dict[str, str] = {}
    results.update(write_dataset_registry_artifact())
    results.update(write_data_manifest_artifact())
    results.update(write_domain_registry_artifact())
    results.update(write_environment_registry_artifact())
    results.update(write_scope_report_artifact(config))
    results.update(write_config_resolved_artifact(config))
    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Registry data
    "DATASET_REGISTRY",
    "SOURCE_DOMAINS",
    "TARGET_DOMAINS",
    "DOMAIN_PAIRS",
    "MOBILENET_CONFIG",
    "ADAPTOR_ARCH_CONFIG",
    # Lookups
    "resolve_dataset_id",
    "get_dataset_entry",
    "list_source_domains",
    "list_target_domains",
    "list_domain_pairs",
    # Dataset classes
    "FewShotDataset",
    "SourceDomainDataset",
    # Factories
    "make_dataset",
    "make_fewshot_dataset",
    # Readiness
    "check_dataset_readiness",
    "check_all_readiness",
    # Classifier helpers
    "get_mobilenet_config",
    "build_mobilenet_classifier",
    # Artifact writers
    "write_dataset_registry_artifact",
    "write_data_manifest_artifact",
    "write_domain_registry_artifact",
    "write_environment_registry_artifact",
    "write_scope_report_artifact",
    "write_config_resolved_artifact",
    "write_all_artifacts",
]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Writing dataset registry artifacts …")
    paths = write_all_artifacts()
    for name, path in paths.items():
        print(f"  {name}: {path}")
    print("Done.")