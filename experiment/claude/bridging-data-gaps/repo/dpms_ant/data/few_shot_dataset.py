# dpms_ant/data/few_shot_dataset.py
# =============================================================================
# DPMs-ANT – Few-Shot Dataset Loading and Domain Registry
# Paper: "Bridging Data Gaps in Diffusion Models with Adversarial Noise-Based
#         Transfer Learning"
#
# reference_grounding: paper_semantic_chunk_012 10-shot image generation experiments
# reference_grounding: paper_semantic_chunk_014_01 experimental_setup_subsection
# reference_grounding: paper_method_core Algorithm 1 data pipeline
#
# Implements:
#   - FewShotDataset: 10-shot subset sampling with augmentation
#   - DATASET_REGISTRY: all paper-derived domain aliases
#   - make_dataset(config): factory from config dict
#   - make_fewshot_dataset(pair, config): source→target pair loader
#   - dataset_readiness_check(): validates registry and writes artifacts
#   - write_dataset_artifacts(): writes results/dataset_registry.json etc.
#
# Paper evidence contract:
#   Source domains:  imagenet, ffhq, lsun_church
#   Target domains:  babies, sunglasses, raphael_peale, sketches, modigliani,
#                    haunted_houses, landscape_drawings
#   Shot count:      10 (fixed per paper Section 4 / addendum)
#   Classifier:      MobileNetV2 fine-tuned to 2-class (source vs target)
#                    per addendum: "modifying the last layer to output two classes"
# =============================================================================

from __future__ import annotations

import json
import os
import random
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Lazy import helpers – keep module importable without torch/torchvision
# ---------------------------------------------------------------------------

def _require_torch():
    try:
        import torch
        return torch
    except ImportError as e:
        raise ImportError(
            "torch is required for dataset loading. Install via: pip install torch torchvision"
        ) from e


def _require_torchvision():
    try:
        import torchvision
        return torchvision
    except ImportError as e:
        raise ImportError(
            "torchvision is required for dataset loading. Install via: pip install torchvision"
        ) from e


# ---------------------------------------------------------------------------
# Paper-derived Dataset Registry
# reference_grounding: paper_semantic_chunk_012 dataset_registry
# reference_grounding: paper_semantic_chunk_014_01 experimental_setup
#
# All 10 paper-registered domain IDs with metadata.
# Source domains: imagenet, ffhq, lsun_church
# Target domains: babies, sunglasses, raphael_peale, sketches, modigliani,
#                 haunted_houses, landscape_drawings
# ---------------------------------------------------------------------------

DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    # ── Source domains ──────────────────────────────────────────────────────
    "imagenet": {
        "id": "imagenet",
        "aliases": ["imagenet", "ImageNet", "ILSVRC"],
        "role": "source",
        "domain_type": "natural",
        "resolution": 256,
        "pretrained_model": "ddpm_imagenet256",
        "default_data_dir": "data/imagenet",
        "num_classes": 1000,
        "description": "ImageNet source domain for DDPM/LDM pretraining",
        "paper_table": "Table 2",
        # reference_grounding: paper_semantic_chunk_012 imagenet_source
    },
    "ffhq": {
        "id": "ffhq",
        "aliases": ["ffhq", "FFHQ", "ffhq256", "ffhq_256"],
        "role": "source",
        "domain_type": "face",
        "resolution": 256,
        "pretrained_model": "ddpm_ffhq256",
        "default_data_dir": "data/ffhq",
        "num_classes": None,
        "description": "FFHQ 256x256 source domain – 5 target domains in paper",
        "paper_table": "Table 2",
        # reference_grounding: paper_semantic_chunk_012 ffhq_source
    },
    "lsun_church": {
        "id": "lsun_church",
        "aliases": ["lsun_church", "lsun-church", "church", "LSUN_Church"],
        "role": "source",
        "domain_type": "architecture",
        "resolution": 256,
        "pretrained_model": "ddpm_lsun_church256",
        "default_data_dir": "data/lsun/church_outdoor_train_lmdb",
        "num_classes": None,
        "description": "LSUN Church 256x256 source domain – 2 target domains in paper",
        "paper_table": "Table 2",
        # reference_grounding: paper_semantic_chunk_012 lsun_church_source
    },
    # ── Target domains (FFHQ source) ────────────────────────────────────────
    "babies": {
        "id": "babies",
        "aliases": ["babies", "baby", "Babies"],
        "role": "target",
        "domain_type": "face",
        "resolution": 256,
        "source_domain": "ffhq",
        "default_data_dir": "data/babies",
        "shot_count": 10,
        "description": "10-shot babies target domain (FFHQ→Babies)",
        "paper_table": "Table 2",
        "paper_fid_ours": 56.4,
        # reference_grounding: paper_semantic_chunk_012 babies_target
    },
    "sunglasses": {
        "id": "sunglasses",
        "aliases": ["sunglasses", "Sunglasses", "glasses"],
        "role": "target",
        "domain_type": "face",
        "resolution": 256,
        "source_domain": "ffhq",
        "default_data_dir": "data/sunglasses",
        "shot_count": 10,
        "description": "10-shot sunglasses target domain (FFHQ→Sunglasses)",
        "paper_table": "Table 2",
        "paper_fid_ours": 63.2,
        # reference_grounding: paper_semantic_chunk_012 sunglasses_target
    },
    "raphael_peale": {
        "id": "raphael_peale",
        "aliases": ["raphael_peale", "raphael-peale", "RaphaelPeale", "peale"],
        "role": "target",
        "domain_type": "painting",
        "resolution": 256,
        "source_domain": "ffhq",
        "default_data_dir": "data/raphael_peale",
        "shot_count": 10,
        "description": "10-shot Raphael Peale portrait paintings (FFHQ→Raphael Peale)",
        "paper_table": "Table 2",
        # reference_grounding: paper_semantic_chunk_012 raphael_peale_target
    },
    "sketches": {
        "id": "sketches",
        "aliases": ["sketches", "sketch", "Sketches"],
        "role": "target",
        "domain_type": "sketch",
        "resolution": 256,
        "source_domain": "ffhq",
        "default_data_dir": "data/sketches",
        "shot_count": 10,
        "description": "10-shot face sketches target domain (FFHQ→Sketches)",
        "paper_table": "Table 2",
        # reference_grounding: paper_semantic_chunk_012 sketches_target
    },
    "modigliani": {
        "id": "modigliani",
        "aliases": ["modigliani", "Modigliani"],
        "role": "target",
        "domain_type": "painting",
        "resolution": 256,
        "source_domain": "ffhq",
        "default_data_dir": "data/modigliani",
        "shot_count": 10,
        "description": "10-shot Modigliani portrait paintings (FFHQ→Modigliani)",
        "paper_table": "Table 2",
        # reference_grounding: paper_semantic_chunk_012 modigliani_target
    },
    # ── Target domains (LSUN-Church source) ─────────────────────────────────
    "haunted_houses": {
        "id": "haunted_houses",
        "aliases": ["haunted_houses", "haunted-houses", "HauntedHouses", "haunted"],
        "role": "target",
        "domain_type": "architecture",
        "resolution": 256,
        "source_domain": "lsun_church",
        "default_data_dir": "data/haunted_houses",
        "shot_count": 10,
        "description": "10-shot haunted houses (LSUN-Church→Haunted Houses)",
        "paper_table": "Table 2",
        # reference_grounding: paper_semantic_chunk_012 haunted_houses_target
    },
    "landscape_drawings": {
        "id": "landscape_drawings",
        "aliases": ["landscape_drawings", "landscape-drawings", "LandscapeDrawings",
                    "landscape", "landscapes"],
        "role": "target",
        "domain_type": "drawing",
        "resolution": 256,
        "source_domain": "lsun_church",
        "default_data_dir": "data/landscape_drawings",
        "shot_count": 10,
        "description": "10-shot landscape drawings (LSUN-Church→Landscape Drawings)",
        "paper_table": "Table 2",
        # reference_grounding: paper_semantic_chunk_012 landscape_drawings_target
    },
}

# Alias lookup: maps any alias string → canonical registry id
_ALIAS_TO_ID: Dict[str, str] = {}
for _did, _dmeta in DATASET_REGISTRY.items():
    for _alias in _dmeta.get("aliases", []):
        _ALIAS_TO_ID[_alias.lower()] = _did

# Paper-defined source→target pairs (Table 2)
# reference_grounding: paper_semantic_chunk_012 experiment_pairs
DOMAIN_PAIRS: List[Dict[str, str]] = [
    {"source": "ffhq",         "target": "babies",            "framework": "ddpm"},
    {"source": "ffhq",         "target": "sunglasses",        "framework": "ddpm"},
    {"source": "ffhq",         "target": "raphael_peale",     "framework": "ddpm"},
    {"source": "ffhq",         "target": "sketches",          "framework": "ddpm"},
    {"source": "ffhq",         "target": "modigliani",        "framework": "ddpm"},
    {"source": "lsun_church",  "target": "haunted_houses",    "framework": "ddpm"},
    {"source": "lsun_church",  "target": "landscape_drawings","framework": "ddpm"},
]

# All 7 target domain IDs (paper Table 2)
TARGET_DOMAIN_IDS: List[str] = [p["target"] for p in DOMAIN_PAIRS]

# All 3 source domain IDs
SOURCE_DOMAIN_IDS: List[str] = ["imagenet", "ffhq", "lsun_church"]


# ---------------------------------------------------------------------------
# Utility: resolve domain id from alias
# ---------------------------------------------------------------------------

def resolve_domain_id(name: str) -> str:
    """Resolve a domain alias to its canonical registry ID."""
    key = name.lower().strip()
    if key in DATASET_REGISTRY:
        return key
    if key in _ALIAS_TO_ID:
        return _ALIAS_TO_ID[key]
    raise KeyError(
        f"Unknown domain '{name}'. Valid IDs/aliases: {sorted(_ALIAS_TO_ID.keys())}"
    )


def get_domain_meta(name: str) -> Dict[str, Any]:
    """Return registry metadata for a domain name or alias."""
    return DATASET_REGISTRY[resolve_domain_id(name)]


# ---------------------------------------------------------------------------
# FewShotDataset
# reference_grounding: paper_semantic_chunk_012 10-shot_setting
# reference_grounding: paper_semantic_chunk_014_01 experimental_setup
#
# Loads exactly `shot` images from a target domain directory.
# Supports standard augmentation (random horizontal flip + color jitter)
# consistent with few-shot fine-tuning practice.
# ---------------------------------------------------------------------------

class FewShotDataset:
    """
    10-shot target domain dataset for DPMs-ANT fine-tuning.

    Paper contract (Section 4 / addendum):
      - Exactly 10 images from the target domain are used for fine-tuning.
      - Images are loaded from a flat directory or ImageFolder structure.
      - Augmentation: random horizontal flip + mild color jitter.
      - Returns a DataLoader that cycles over the 10 images indefinitely
        (with replacement sampling) to support 5000-iteration training.

    reference_grounding: paper_semantic_chunk_012 10-shot_setting
    """

    def __init__(
        self,
        domain: str,
        shot: int = 10,
        data_root: Optional[str] = None,
        image_size: int = 256,
        augment: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            domain:     Domain name or alias (resolved via DATASET_REGISTRY).
            shot:       Number of images to use (paper default: 10).
            data_root:  Root directory for data. Falls back to registry default
                        or DATA_ROOT env var.
            image_size: Spatial resolution (paper: 256).
            augment:    Apply random flip + color jitter (True for training).
            seed:       Random seed for reproducible subset selection.
        """
        self.domain_id = resolve_domain_id(domain)
        self.meta = DATASET_REGISTRY[self.domain_id]
        self.shot = shot
        self.image_size = image_size
        self.augment = augment
        self.seed = seed

        # Resolve data directory
        if data_root is not None:
            self.data_dir = Path(data_root) / self.domain_id
        else:
            env_root = os.environ.get("DATA_ROOT", "data")
            self.data_dir = Path(env_root) / self.domain_id
            # Fall back to registry default
            if not self.data_dir.exists():
                self.data_dir = Path(self.meta["default_data_dir"])

        self._dataset = None  # lazy-loaded torchvision dataset
        self._indices: Optional[List[int]] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_transform(self):
        """Build torchvision transform pipeline."""
        tv = _require_torchvision()
        transforms = tv.transforms

        base = [
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
        ]
        if self.augment:
            base += [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.05, contrast=0.05, saturation=0.05, hue=0.02
                ),
            ]
        base += [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
        return transforms.Compose(base)

    def _load_underlying_dataset(self):
        """Load the full directory as an ImageFolder or flat image list."""
        tv = _require_torchvision()
        transform = self._build_transform()

        data_dir = self.data_dir
        if not data_dir.exists():
            warnings.warn(
                f"[FewShotDataset] Data directory not found: {data_dir}. "
                "Returning synthetic placeholder dataset for smoke/import validation.",
                RuntimeWarning,
                stacklevel=3,
            )
            return _SyntheticFewShotDataset(
                n=self.shot, image_size=self.image_size, domain_id=self.domain_id
            )

        # Try ImageFolder (subdirectory per class) first, then flat folder
        try:
            ds = tv.datasets.ImageFolder(root=str(data_dir), transform=transform)
        except FileNotFoundError:
            ds = tv.datasets.ImageFolder(root=str(data_dir.parent), transform=transform)

        return ds

    def _select_shot_indices(self, total: int) -> List[int]:
        """Deterministically select `shot` indices from [0, total)."""
        rng = random.Random(self.seed)
        if total <= self.shot:
            return list(range(total))
        return sorted(rng.sample(range(total), self.shot))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_dataset(self):
        """
        Return a torch Dataset containing exactly `shot` images.

        Returns a Subset wrapping the underlying ImageFolder, or a
        SyntheticFewShotDataset when the data directory is absent
        (smoke/import validation path).
        """
        torch = _require_torch()
        if self._dataset is None:
            underlying = self._load_underlying_dataset()
            if isinstance(underlying, _SyntheticFewShotDataset):
                self._dataset = underlying
            else:
                n = len(underlying)
                self._indices = self._select_shot_indices(n)
                self._dataset = torch.utils.data.Subset(underlying, self._indices)
        return self._dataset

    def get_dataloader(
        self,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        replacement: bool = True,
        num_samples: Optional[int] = None,
    ):
        """
        Return a DataLoader over the 10-shot subset.

        For training (replacement=True), uses a WeightedRandomSampler so the
        10 images are sampled with replacement across 5000 training iterations.

        Args:
            batch_size:   Batch size (paper training uses 64, but 10-shot
                          means effective batch is min(64, 10) per real step).
            num_workers:  DataLoader workers.
            pin_memory:   Pin memory for GPU transfer.
            replacement:  Sample with replacement (True for training loops).
            num_samples:  Total samples to draw when replacement=True.
                          Defaults to batch_size * 5000 (full training budget).
        """
        torch = _require_torch()
        dataset = self.get_dataset()
        n = len(dataset)

        if replacement and n > 0:
            if num_samples is None:
                num_samples = batch_size * 5000  # full training budget
            weights = [1.0 / n] * n
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=weights,
                num_samples=num_samples,
                replacement=True,
            )
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=True,
            )
        else:
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=not replacement,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=False,
            )

    def __len__(self) -> int:
        return self.shot

    def __repr__(self) -> str:
        return (
            f"FewShotDataset(domain={self.domain_id!r}, shot={self.shot}, "
            f"image_size={self.image_size}, data_dir={self.data_dir})"
        )


# ---------------------------------------------------------------------------
# Synthetic fallback dataset (smoke / import validation only)
# reference_grounding: paper_semantic_chunk_012 smoke_validation
# ---------------------------------------------------------------------------

class _SyntheticFewShotDataset:
    """
    Lightweight synthetic dataset used when real data is absent.
    Generates random tensors in [-1, 1] at the target resolution.
    Labeled as dry-run / smoke artifact – NOT real experiment data.
    """

    _DRY_RUN_LABEL = "synthetic_smoke_only"

    def __init__(self, n: int = 10, image_size: int = 256, domain_id: str = "unknown"):
        self.n = n
        self.image_size = image_size
        self.domain_id = domain_id

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        torch = _require_torch()
        img = torch.rand(3, self.image_size, self.image_size) * 2.0 - 1.0
        label = 0
        return img, label


# ---------------------------------------------------------------------------
# make_dataset – factory from config dict
# reference_grounding: paper_method_core make_dataset
# ---------------------------------------------------------------------------

def make_dataset(config: Dict[str, Any]) -> FewShotDataset:
    """
    Construct a FewShotDataset from a config dictionary.

    Expected config keys:
        target_domain (str):  Target domain name or alias.
        shot          (int):  Number of shots (default: 10).
        data_root     (str):  Optional root directory for data.
        image_size    (int):  Image resolution (default: 256).
        augment       (bool): Apply augmentation (default: True).
        seed          (int):  Random seed (default: 42).

    Returns:
        FewShotDataset instance.
    """
    domain = config.get("target_domain") or config.get("domain")
    if domain is None:
        raise ValueError("config must contain 'target_domain' or 'domain' key")

    return FewShotDataset(
        domain=domain,
        shot=int(config.get("shot", 10)),
        data_root=config.get("data_root", None),
        image_size=int(config.get("image_size", 256)),
        augment=bool(config.get("augment", True)),
        seed=int(config.get("seed", 42)),
    )


# ---------------------------------------------------------------------------
# make_fewshot_dataset – source→target pair loader
# reference_grounding: paper_semantic_chunk_012 domain_pair_loader
# ---------------------------------------------------------------------------

def make_fewshot_dataset(
    pair: Dict[str, str],
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[FewShotDataset, Dict[str, Any]]:
    """
    Build a FewShotDataset for a source→target domain pair.

    Args:
        pair:   Dict with keys 'source' and 'target' (domain names/aliases).
                Optionally 'framework' ('ddpm' | 'ldm').
        config: Optional config overrides (shot, image_size, data_root, etc.).

    Returns:
        (FewShotDataset, pair_meta) where pair_meta contains resolved IDs,
        registry metadata, and framework info.

    Example:
        ds, meta = make_fewshot_dataset(
            {"source": "ffhq", "target": "babies", "framework": "ddpm"},
            config={"shot": 10, "image_size": 256},
        )
    """
    cfg = config or {}
    source_id = resolve_domain_id(pair["source"])
    target_id = resolve_domain_id(pair["target"])
    framework = pair.get("framework", cfg.get("framework", "ddpm"))

    source_meta = DATASET_REGISTRY[source_id]
    target_meta = DATASET_REGISTRY[target_id]

    # Merge config: pair-level overrides > config > registry defaults
    merged_cfg = {
        "target_domain": target_id,
        "shot": cfg.get("shot", target_meta.get("shot_count", 10)),
        "image_size": cfg.get("image_size", target_meta.get("resolution", 256)),
        "data_root": cfg.get("data_root", None),
        "augment": cfg.get("augment", True),
        "seed": cfg.get("seed", 42),
    }

    dataset = make_dataset(merged_cfg)

    pair_meta = {
        "source_id": source_id,
        "target_id": target_id,
        "framework": framework,
        "source_meta": source_meta,
        "target_meta": target_meta,
        "shot": merged_cfg["shot"],
        "image_size": merged_cfg["image_size"],
        "pretrained_model": source_meta.get("pretrained_model"),
    }

    return dataset, pair_meta


# ---------------------------------------------------------------------------
# Dataset readiness check
# reference_grounding: paper_semantic_chunk_012 readiness_check
# ---------------------------------------------------------------------------

def dataset_readiness_check(
    data_root: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Check availability of all registered domains and return a readiness report.

    Does NOT require data to be present – missing directories are flagged as
    'unavailable' but do not raise errors. This allows smoke/import validation
    in environments without real datasets.

    Returns:
        Dict with keys:
            registry_size (int): Number of registered domains.
            available (list):    Domain IDs with data present.
            unavailable (list):  Domain IDs with data absent.
            pairs_valid (list):  Domain pairs where target data is present.
            status (str):        'ready' | 'partial' | 'unavailable'.
    """
    root = Path(data_root) if data_root else Path(os.environ.get("DATA_ROOT", "data"))

    available = []
    unavailable = []

    for did, meta in DATASET_REGISTRY.items():
        candidate = root / did
        fallback = Path(meta["default_data_dir"])
        if candidate.exists() or fallback.exists():
            available.append(did)
        else:
            unavailable.append(did)

    pairs_valid = []
    for pair in DOMAIN_PAIRS:
        if pair["target"] in available:
            pairs_valid.append(pair)

    if len(available) == len(DATASET_REGISTRY):
        status = "ready"
    elif len(available) > 0:
        status = "partial"
    else:
        status = "unavailable"

    report = {
        "registry_size": len(DATASET_REGISTRY),
        "registered_domains": list(DATASET_REGISTRY.keys()),
        "available": available,
        "unavailable": unavailable,
        "pairs_valid": pairs_valid,
        "total_pairs": len(DOMAIN_PAIRS),
        "status": status,
        "data_root_checked": str(root),
    }

    if verbose:
        print(f"[FewShotDataset] Readiness: {status} "
              f"({len(available)}/{len(DATASET_REGISTRY)} domains available)")

    return report


# ---------------------------------------------------------------------------
# Artifact writer
# reference_grounding: paper_semantic_chunk_012 artifact_writer
# Writes: results/dataset_registry.json, results/data_manifest.json,
#         results/domain_registry.json, results/environment_registry.json,
#         results/scope_report.json, results/config_resolved.json
# ---------------------------------------------------------------------------

def write_dataset_artifacts(
    output_dir: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    dry_run: bool = True,
) -> Dict[str, str]:
    """
    Write all declared dataset/domain registry artifacts.

    Args:
        output_dir: Output directory (default: results/ or PAPERBENCH_REPRO_ARTIFACT_DIR).
        config:     Optional resolved config to embed in config_resolved.json.
        dry_run:    If True, labels artifacts as dry-run/schema artifacts.

    Returns:
        Dict mapping artifact name → written file path.
    """
    artifact_dir = (
        output_dir
        or os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR")
        or "results"
    )
    out = Path(artifact_dir)
    out.mkdir(parents=True, exist_ok=True)

    dry_run_label = "dry_run_schema_artifact" if dry_run else "real_experiment_artifact"
    readiness = dataset_readiness_check(verbose=False)

    # ── dataset_registry.json ────────────────────────────────────────────────
    dataset_registry_payload = {
        "_artifact_type": dry_run_label,
        "_description": (
            "Paper-derived dataset/domain registry for DPMs-ANT. "
            "Contains all 10 registered domain IDs with metadata. "
            "reference_grounding: paper_semantic_chunk_012 dataset_registry"
        ),
        "_paper": "Bridging Data Gaps in Diffusion Models with Adversarial Noise-Based Transfer Learning",
        "registry": {
            did: {
                "id": did,
                "role": meta["role"],
                "domain_type": meta["domain_type"],
                "resolution": meta["resolution"],
                "aliases": meta["aliases"],
                "default_data_dir": meta["default_data_dir"],
                "shot_count": meta.get("shot_count", None),
                "source_domain": meta.get("source_domain", None),
                "pretrained_model": meta.get("pretrained_model", None),
                "description": meta["description"],
                "paper_table": meta.get("paper_table", "Table 2"),
            }
            for did, meta in DATASET_REGISTRY.items()
        },
        "source_domains": SOURCE_DOMAIN_IDS,
        "target_domains": TARGET_DOMAIN_IDS,
        "total_registered": len(DATASET_REGISTRY),
    }
    p_dataset_registry = out / "dataset_registry.json"
    p_dataset_registry.write_text(json.dumps(dataset_registry_payload, indent=2))

    # ── domain_registry.json ─────────────────────────────────────────────────
    domain_registry_payload = {
        "_artifact_type": dry_run_label,
        "_description": (
            "Domain pair registry for DPMs-ANT experiments. "
            "Lists all 7 source→target pairs from Table 2. "
            "reference_grounding: paper_semantic_chunk_012 domain_pairs"
        ),
        "domain_pairs": DOMAIN_PAIRS,
        "source_domains": SOURCE_DOMAIN_IDS,
        "target_domains": TARGET_DOMAIN_IDS,
        "total_pairs": len(DOMAIN_PAIRS),
        "shot_count": 10,
        "frameworks": ["ddpm", "ldm"],
    }
    p_domain_registry = out / "domain_registry.json"
    p_domain_registry.write_text(json.dumps(domain_registry_payload, indent=2))

    # ── data_manifest.json ───────────────────────────────────────────────────
    data_manifest_payload = {
        "_artifact_type": dry_run_label,
        "_description": (
            "Data availability manifest. "
            "Lists which domain directories are present on this machine. "
            "reference_grounding: paper_semantic_chunk_012 data_manifest"
        ),
        "readiness": readiness,
        "domain_data_paths": {
            did: {
                "default_path": meta["default_data_dir"],
                "env_path": str(Path(os.environ.get("DATA_ROOT", "data")) / did),
            }
            for did, meta in DATASET_REGISTRY.items()
        },
    }