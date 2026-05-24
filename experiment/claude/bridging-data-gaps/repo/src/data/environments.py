"""
src/data/environments.py
========================
DPMs-ANT – Environment, Dataset, and Task Registry
Paper: "Bridging Data Gaps in Diffusion Models with Adversarial Noise-Based
        Transfer Learning"

Exposes:
  * DATASET_REGISTRY  – paper-derived dataset entries with ids, aliases, metadata
  * ENVIRONMENT_REGISTRY – environment/task entries with ids, aliases, setup metadata
  * DOMAIN_SOURCE_MAP – source→target domain mappings
  * FewShotDataset    – 10-shot dataset class supporting all target domains
  * make_dataset(config) – factory for dataset from config dict
  * make_fewshot_dataset(pair, config) – factory for source+target pair
  * dataset_readiness_check(domain) – confirm dataset availability
  * write_registry_artifacts() – write all declared JSON artifacts

reference_grounding: paper_method_core src/data/environments.py
reference_grounding: paper_semantic_chunk_012 10-shot image generation experiments
reference_grounding: paper_semantic_chunk_014_01 classifier_loader_finetuning_experimental_setup_subsection_experimental_setup

Binding addendum clarifications:
  - Classifier: pre-trained MobileNetV2 fine-tuned, last layer modified to output 2 classes
    (source vs target domain binary classification).
  - Adaptor: down-pooling → GroupNorm → 3×3 conv → 4-head attention → MLP(→8 or 16)
    → 4× upsample → GroupNorm → 3×3 conv.
  - 10-shot protocol: exactly 10 target-domain images, random horizontal flip augmentation.
"""

from __future__ import annotations

import json
import logging
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Artifact output directory helpers
# ---------------------------------------------------------------------------

def _results_dir() -> Path:
    """Return the results directory, preferring PAPERBENCH_REPRO_ARTIFACT_DIR."""
    env_dir = os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR", "")
    if env_dir:
        return Path(env_dir)
    return Path("results")


def _ensure_results() -> Path:
    d = _results_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d

# ---------------------------------------------------------------------------
# Dataset Registry
# reference_grounding: paper_semantic_chunk_012 datasets used in experiments
# ---------------------------------------------------------------------------

# Source domains used in paper experiments
SOURCE_DOMAINS = ["ffhq", "lsun_church", "imagenet"]

# Target domains used in paper 10-shot experiments
TARGET_DOMAINS = [
    "babies",
    "sunglasses",
    "raphael_peale",
    "sketches",
    "modigliani",
    "haunted_houses",
    "landscape_drawings",
]

# All registered dataset ids
ALL_DATASET_IDS = SOURCE_DOMAINS + TARGET_DOMAINS

DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    # ------------------------------------------------------------------
    # Source domains (large-scale pre-training datasets)
    # ------------------------------------------------------------------
    "imagenet": {
        "id": "imagenet",
        "aliases": ["ImageNet", "ILSVRC", "imagenet1k", "imagenet_1k"],
        "role": "source",
        "description": "ImageNet-1K large-scale image classification dataset",
        "num_classes": 1000,
        "image_size": 256,
        "framework": ["ddpm", "ldm"],
        "data_root_env": "IMAGENET_ROOT",
        "default_root": "data/imagenet",
        "split": "train",
        "loader": "image_folder",
        "shot_count": None,
        "augmentation": ["random_horizontal_flip", "center_crop", "resize"],
        "paper_role": "source domain / classifier pre-training base",
        # reference_grounding: paper_semantic_chunk_014_01 imagenet source
    },
    "ffhq": {
        "id": "ffhq",
        "aliases": ["FFHQ", "Flickr-Faces-HQ", "flickr_faces_hq"],
        "role": "source",
        "description": "Flickr-Faces-HQ – 70,000 high-quality human face images at 1024×1024",
        "num_classes": None,
        "image_size": 256,
        "framework": ["ddpm", "ldm"],
        "data_root_env": "FFHQ_ROOT",
        "default_root": "data/ffhq",
        "split": "train",
        "loader": "image_folder",
        "shot_count": None,
        "augmentation": ["random_horizontal_flip", "resize", "center_crop"],
        "paper_role": "source domain for FFHQ→{Babies,Sunglasses,RaphaelPeale,Sketches,Modigliani}",
        # reference_grounding: paper_semantic_chunk_012 FFHQ source domain
    },
    "lsun_church": {
        "id": "lsun_church",
        "aliases": ["LSUN-Church", "lsun_church_outdoor", "LSUN_Church", "church_outdoor"],
        "role": "source",
        "description": "LSUN Church Outdoor category – large-scale scene images",
        "num_classes": None,
        "image_size": 256,
        "framework": ["ddpm"],
        "data_root_env": "LSUN_ROOT",
        "default_root": "data/lsun",
        "split": "train",
        "loader": "lsun",
        "shot_count": None,
        "augmentation": ["random_horizontal_flip", "resize", "center_crop"],
        "paper_role": "source domain for LSUN-Church→{HauntedHouses,LandscapeDrawings}",
        # reference_grounding: paper_semantic_chunk_012 LSUN-Church source domain
    },
    # ------------------------------------------------------------------
    # Target domains (10-shot fine-tuning sets, FFHQ-based)
    # ------------------------------------------------------------------
    "babies": {
        "id": "babies",
        "aliases": ["Babies", "baby_faces", "ffhq_babies"],
        "role": "target",
        "source_domain": "ffhq",
        "description": "Baby face images – FFHQ-based 10-shot target domain",
        "num_classes": None,
        "image_size": 256,
        "framework": ["ddpm", "ldm"],
        "data_root_env": "BABIES_ROOT",
        "default_root": "data/few_shot/babies",
        "split": "train",
        "loader": "image_folder",
        "shot_count": 10,
        "augmentation": ["random_horizontal_flip"],
        "paper_role": "10-shot FFHQ target domain (Table 2)",
        # reference_grounding: paper_semantic_chunk_012 Babies 10-shot target
    },
    "sunglasses": {
        "id": "sunglasses",
        "aliases": ["Sunglasses", "sunglasses_faces", "ffhq_sunglasses"],
        "role": "target",
        "source_domain": "ffhq",
        "description": "Faces with sunglasses – FFHQ-based 10-shot target domain",
        "num_classes": None,
        "image_size": 256,
        "framework": ["ddpm", "ldm"],
        "data_root_env": "SUNGLASSES_ROOT",
        "default_root": "data/few_shot/sunglasses",
        "split": "train",
        "loader": "image_folder",
        "shot_count": 10,
        "augmentation": ["random_horizontal_flip"],
        "paper_role": "10-shot FFHQ target domain (Table 2)",
        # reference_grounding: paper_semantic_chunk_012 Sunglasses 10-shot target
    },
    "raphael_peale": {
        "id": "raphael_peale",
        "aliases": ["RaphaelPeale", "Raphael_Peale", "raphael", "peale_portraits"],
        "role": "target",
        "source_domain": "ffhq",
        "description": "Portrait paintings by Raphael Peale – FFHQ-based 10-shot target domain",
        "num_classes": None,
        "image_size": 256,
        "framework": ["ddpm", "ldm"],
        "data_root_env": "RAPHAEL_PEALE_ROOT",
        "default_root": "data/few_shot/raphael_peale",
        "split": "train",
        "loader": "image_folder",
        "shot_count": 10,
        "augmentation": ["random_horizontal_flip"],
        "paper_role": "10-shot FFHQ target domain (Table 2)",
        # reference_grounding: paper_semantic_chunk_012 Raphael Peale 10-shot target
    },
    "sketches": {
        "id": "sketches",
        "aliases": ["Sketches", "face_sketches", "ffhq_sketches", "sketch"],
        "role": "target",
        "source_domain": "ffhq",
        "description": "Face sketch images – FFHQ-based 10-shot target domain",
        "num_classes": None,
        "image_size": 256,
        "framework": ["ddpm", "ldm"],
        "data_root_env": "SKETCHES_ROOT",
        "default_root": "data/few_shot/sketches",
        "split": "train",
        "loader": "image_folder",
        "shot_count": 10,
        "augmentation": ["random_horizontal_flip"],
        "paper_role": "10-shot FFHQ target domain (Table 2)",
        # reference_grounding: paper_semantic_chunk_012 Sketches 10-shot target
    },
    "modigliani": {
        "id": "modigliani",
        "aliases": ["Modigliani", "amedeo_modigliani", "AmedeoModigliani", "modigliani_portraits"],
        "role": "target",
        "source_domain": "ffhq",
        "description": "Portrait paintings by Amedeo Modigliani – FFHQ-based 10-shot target domain",
        "num_classes": None,
        "image_size": 256,
        "framework": ["ddpm", "ldm"],
        "data_root_env": "MODIGLIANI_ROOT",
        "default_root": "data/few_shot/modigliani",
        "split": "train",
        "loader": "image_folder",
        "shot_count": 10,
        "augmentation": ["random_horizontal_flip"],
        "paper_role": "10-shot FFHQ target domain (Table 2)",
        # reference_grounding: paper_semantic_chunk_012 Amedeo Modigliani 10-shot target
    },
    "haunted_houses": {
        "id": "haunted_houses",
        "aliases": ["HauntedHouses", "Haunted_Houses", "haunted_house", "haunted"],
        "role": "target",
        "source_domain": "lsun_church",
        "description": "Haunted house images – LSUN-Church-based 10-shot target domain",
        "num_classes": None,
        "image_size": 256,
        "framework": ["ddpm"],
        "data_root_env": "HAUNTED_HOUSES_ROOT",
        "default_root": "data/few_shot/haunted_houses",
        "split": "train",
        "loader": "image_folder",
        "shot_count": 10,
        "augmentation": ["random_horizontal_flip"],
        "paper_role": "10-shot LSUN-Church target domain (Table 2)",
        # reference_grounding: paper_semantic_chunk_012 Haunted Houses 10-shot target
    },
    "landscape_drawings": {
        "id": "landscape_drawings",
        "aliases": [
            "LandscapeDrawings", "Landscape_Drawings", "landscape", "landscape_drawing"
        ],
        "role": "target",
        "source_domain": "lsun_church",
        "description": "Landscape drawing images – LSUN-Church-based 10-shot target domain",
        "num_classes": None,
        "image_size": 256,
        "framework": ["ddpm"],
        "data_root_env": "LANDSCAPE_DRAWINGS_ROOT",
        "default_root": "data/few_shot/landscape_drawings",
        "split": "train",
        "loader": "image_folder",
        "shot_count": 10,
        "augmentation": ["random_horizontal_flip"],
        "paper_role": "10-shot LSUN-Church target domain (Table 2)",
        # reference_grounding: paper_semantic_chunk_012 Landscape Drawings 10-shot target
    },
}

# Alias lookup map: any alias → canonical id
_ALIAS_TO_ID: Dict[str, str] = {}
for _did, _dmeta in DATASET_REGISTRY.items():
    _ALIAS_TO_ID[_did] = _did
    for _alias in _dmeta.get("aliases", []):
        _ALIAS_TO_ID[_alias] = _did
        _ALIAS_TO_ID[_alias.lower()] = _did


def resolve_dataset_id(name: str) -> str:
    """Resolve a dataset name or alias to the canonical registry id."""
    canonical = _ALIAS_TO_ID.get(name) or _ALIAS_TO_ID.get(name.lower())
    if canonical is None:
        raise KeyError(
            f"Dataset '{name}' not found in DATASET_REGISTRY. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )
    return canonical


# Source→target domain mapping
DOMAIN_SOURCE_MAP: Dict[str, str] = {
    did: meta["source_domain"]
    for did, meta in DATASET_REGISTRY.items()
    if meta.get("role") == "target"
}

# Target domains per source domain
SOURCE_TO_TARGETS: Dict[str, List[str]] = {}
for _target_id, _source_id in DOMAIN_SOURCE_MAP.items():
    SOURCE_TO_TARGETS.setdefault(_source_id, []).append(_target_id)

# ---------------------------------------------------------------------------
# Environment / Task Registry
# reference_grounding: paper_semantic_chunk_014_01 experimental_setup
# ---------------------------------------------------------------------------

ENVIRONMENT_REGISTRY: Dict[str, Dict[str, Any]] = {
    # ------------------------------------------------------------------
    # Method environment
    # ------------------------------------------------------------------
    "ant": {
        "id": "ant",
        "aliases": ["ANT", "DPMs-ANT", "dpms_ant", "ours"],
        "name": "DPMs-ANT – Adversarial Noise-based Transfer learning",
        "description": (
            "Full DPMs-ANT method: Algorithm 1 with similarity-guided training "
            "and adversarial noise selection. Trains only ShiftAdaptor parameters."
        ),
        "method_id": "ours",
        "framework": ["ddpm", "ldm"],
        "training_iterations": 5000,
        "batch_size": 64,
        "shot_count": 10,
        "use_sim_guide": True,
        "use_adv_noise": True,
        "gamma": 5,
        "omega": 0.02,
        "adversarial_inner_steps": 10,
        "classifier_finetune_steps": 300,
        "shift_adaptor": {
            "ddpm": {"c": 4, "d": 8},
            "ldm": {"c": 2, "d": 8},
        },
        "factory": "make_ant_environment",
        "config_hook": "configs/experiments.yaml",
        # reference_grounding: paper_method_core Algorithm 1
    },
    "shot_image_generation": {
        "id": "shot_image_generation",
        "aliases": ["10-shot generation", "few_shot_generation", "10shot"],
        "name": "10-Shot Image Generation Task",
        "description": (
            "Evaluation protocol: generate images from diffusion model fine-tuned "
            "on exactly 10 target-domain images; measure FID, Intra-LPIPS, "
            "accuracy, and fidelity_score."
        ),
        "shot_count": 10,
        "evaluation_metrics": ["fid", "intra_lpips", "accuracy", "fidelity_score"],
        "num_generated_images": 5000,
        "factory": "make_few_shot_generation_env",
        "config_hook": "configs/experiments.yaml",
        # reference_grounding: paper_semantic_chunk_012 shot_image_generation
    },
    "intra_lpips": {
        "id": "intra_lpips",
        "aliases": ["Intra-LPIPS", "intra_lpips_metric", "diversity"],
        "name": "Intra-LPIPS Diversity Metric",
        "description": (
            "Measures pairwise LPIPS distance among generated images to quantify "
            "diversity/mode-coverage. Higher is more diverse."
        ),
        "metric_formula": "mean(LPIPS(x_i, x_j)) for i≠j in generated batch",
        "backbone": "alex",
        "factory": "make_intra_lpips_evaluator",
        "config_hook": "configs/experiments.yaml",
        # reference_grounding: paper_semantic_chunk_012 intra_lpips diversity metric
    },
    "experimental_setup": {
        "id": "experimental_setup",
        "aliases": ["ExperimentalSetup", "exp_setup", "Section 4"],
        "name": "Experimental Setup – Section 4",
        "description": (
            "Paper experimental setup: DDPM/LDM pre-trained on source domain, "
            "fine-tuned with DPMs-ANT on 10-shot target domain, evaluated against "
            "6 baselines with FID/Intra-LPIPS metrics."
        ),
        "source_domains": SOURCE_DOMAINS,
        "target_domains": TARGET_DOMAINS,
        "baselines": [
            "frozen", "naive_finetune", "diff_aug", "ada",
            "lecam", "few_shot_gan"
        ],
        "metrics": ["fid", "intra_lpips", "accuracy", "fidelity_score"],
        "classifier_setup": (
            "MobileNetV2 pre-trained on ImageNet, last layer modified to output "
            "2 classes (source vs target domain), fine-tuned 300 steps."
        ),
        "adaptor_architecture": (
            "down-pooling → GroupNorm → 3×3 conv → 4-head attention → MLP(→8|16) "
            "→ 4× upsample → GroupNorm → 3×3 conv"
        ),
        "factory": "make_experimental_setup",
        "config_hook": "configs/experiments.yaml",
        # reference_grounding: paper_semantic_chunk_014_01 experimental_setup
    },
    "lsun_church": {
        "id": "lsun_church",
        "aliases": ["LSUN-Church", "LSUN_Church", "church_outdoor", "lsun_church_env"],
        "name": "LSUN Church Outdoor Environment",
        "description": (
            "LSUN Church source-domain environment used in DDPM transfer to "
            "HauntedHouses and LandscapeDrawings."
        ),
        "source_domain": "lsun_church",
        "target_domains": ["haunted_houses", "landscape_drawings"],
        "framework": "ddpm",
        "factory": "make_lsun_church_env",
        "config_hook": "configs/ddpm_church.yaml",
        # reference_grounding: paper_semantic_chunk_012 LSUN-Church environment
    },
    "raphael_peale_env": {
        "id": "raphael_peale_env",
        "aliases": ["RaphaelPeale", "Raphael_Peale_env", "raphael"],
        "name": "Raphael Peale Portrait Target Environment",
        "description": (
            "10-shot Raphael Peale portrait target domain for FFHQ→RaphaelPeale transfer."
        ),
        "source_domain": "ffhq",
        "target_domain": "raphael_peale",
        "shot_count": 10,
        "framework": ["ddpm", "ldm"],
        "factory": "make_raphael_peale_env",
        "config_hook": "configs/ddpm_ffhq.yaml",
        # reference_grounding: paper_semantic_chunk_012 Raphael Peale target
    },
    "modigliani_env": {
        "id": "modigliani_env",
        "aliases": ["Modigliani", "AmedeoModigliani", "amedeo_modigliani_env"],
        "name": "Amedeo Modigliani Portrait Target Environment",
        "description": (
            "10-shot Amedeo Modigliani portrait target domain for FFHQ→Modigliani transfer."
        ),
        "source_domain": "ffhq",
        "target_domain": "modigliani",
        "shot_count": 10,
        "framework": ["ddpm", "ldm"],
        "factory": "make_modigliani_env",
        "config_hook": "configs/ddpm_ffhq.yaml",
        # reference_grounding: paper_semantic_chunk_012 Amedeo Modigliani target
    },
    "haunted_houses_env": {
        "id": "haunted_houses_env",
        "aliases": ["HauntedHouses", "Haunted_Houses_env", "haunted"],
        "name": "Haunted Houses Target Environment",
        "description": "10-shot Haunted Houses target domain for LSUN-Church→HauntedHouses transfer.",
        "source_domain": "lsun_church",
        "target_domain": "haunted_houses",
        "shot_count": 10,
        "framework": "ddpm",
        "factory": "make_haunted_houses_env",
        "config_hook": "configs/ddpm_church.yaml",
        # reference_grounding: paper_semantic_chunk_012 Haunted Houses target
    },
    "imagenet": {
        "id": "imagenet",
        "aliases": [
            "ImageNet", "ILSVRC", "imagenet1k", "imagenet_1k",
            "imagenet_classifier", "imagenet_pretrain",
        ],
        "name": "ImageNet Classification Environment",
        "description": (
            "ImageNet environment used for MobileNetV2 classifier pre-training. "
            "The classifier last layer is then replaced with a 2-class head "
            "(source vs target domain) and fine-tuned for 300 steps."
        ),
        "num_classes": 1000,
        "classifier_finetune": {
            "output_classes": 2,
            "finetune_steps": 300,
            "backbone": "mobilenet_v2",
            "last_layer_modification": (
                "Replace classifier[-1] with nn.Linear(in_features, 2) "
                "to distinguish source vs target domain."
            ),
        },
        "factory": "make_imagenet_env",
        "config_hook": "configs/experiments.yaml",
        # reference_grounding: paper_semantic_chunk_014_01 imagenet classifier env
    },
    "babies_env": {
        "id": "babies_env",
        "aliases": ["Babies", "babies_ffhq", "ffhq_babies_env"],
        "name": "Babies FFHQ 10-shot Target Environment",
        "description": "10-shot Babies target domain for FFHQ→Babies transfer (DDPM/LDM).",
        "source_domain": "ffhq",
        "target_domain": "babies",
        "shot_count": 10,
        "framework": ["ddpm", "ldm"],
        "factory": "make_babies_env",
        "config_hook": "configs/ddpm_ffhq.yaml",
        # reference_grounding: paper_semantic_chunk_012 Babies 10-shot FFHQ target
    },
    "sunglasses_env": {
        "id": "sunglasses_env",
        "aliases": ["Sunglasses", "sunglasses_ffhq", "ffhq_sunglasses_env"],
        "name": "Sunglasses FFHQ 10-shot Target Environment",
        "description": "10-shot Sunglasses target domain for FFHQ→Sunglasses transfer (DDPM/LDM).",
        "source_domain": "ffhq",
        "target_domain": "sunglasses",
        "shot_count": 10,
        "framework": ["ddpm", "ldm"],
        "factory": "make_sunglasses_env",
        "config_hook": "configs/ddpm_ffhq.yaml",
        # reference_grounding: paper_semantic_chunk_012 Sunglasses 10-shot FFHQ target
    },
}

# Alias lookup for environment registry
_ENV_ALIAS_TO_ID: Dict[str, str] = {}
for _eid, _emeta in ENVIRONMENT_REGISTRY.items():
    _ENV_ALIAS_TO_ID[_eid] = _eid
    for _alias in _emeta.get("aliases", []):
        _ENV_ALIAS_TO_ID[_alias] = _eid
        _ENV_ALIAS_TO_ID[_alias.lower()] = _eid


def resolve_env_id(name: str) -> str:
    """Resolve an environment name or alias to the canonical registry id."""
    canonical = _ENV_ALIAS_TO_ID.get(name) or _ENV_ALIAS_TO_ID.get(name.lower())
    if canonical is None:
        raise KeyError(
            f"Environment '{name}' not found in ENVIRONMENT_REGISTRY. "
            f"Available: {list(ENVIRONMENT_REGISTRY.keys())}"
        )
    return canonical


# ---------------------------------------------------------------------------
# Adaptor architecture metadata (addendum binding)
# reference_grounding: paper_semantic_chunk_014_01 adaptor_module_architecture
# ---------------------------------------------------------------------------

ADAPTOR_ARCHITECTURE_SPEC = {
    "description": (
        "Shift Adaptor inserted into DDPM/LDM UNet residual blocks. "
        "Architecture per insertion point:"
    ),
    "layers": [
        {"layer": 1, "op": "down_pooling", "details": "Average pooling to downsample spatial dims"},
        {"layer": 2, "op": "group_norm",   "details": "GroupNorm normalisation"},
        {"layer": 3, "op": "conv3x3",      "details": "3×3 convolution"},
        {"layer": 4, "op": "attention",    "details": "4-head multi-head self-attention"},
        {"layer": 5, "op": "mlp",          "details": "MLP reducing feature size to 8 or 16"},
        {"layer": 6, "op": "upsample",     "details": "4× bilinear/transposed-conv upsample"},
        {"layer": 7, "op": "group_norm",   "details": "GroupNorm normalisation"},
        {"layer": 8, "op": "conv3x3",      "details": "3×3 convolution"},
    ],
    "ddpm_params": {"c": 4, "d": 8},
    "ldm_params":  {"c": 2, "d": 8},
    "trainable_only": True,
    "frozen_backbone": True,
}

# ---------------------------------------------------------------------------
# Classifier architecture metadata (addendum binding)
# reference_grounding: paper_semantic_chunk_014_01 classifier_loader_finetuning
# ---------------------------------------------------------------------------

CLASSIFIER_SPEC = {
    "backbone": "mobilenet_v2",
    "pretrained": "imagenet",
    "output_classes": 2,
    "class_mapping": {0: "source_domain", 1: "target_domain"},
    "last_layer_modification": (
        "Replace MobileNetV2.classifier[-1] (nn.Linear(1280, 1000)) "
        "with nn.Linear(1280, 2) for binary source/target classification."
    ),
    "finetune_steps": 300,
    "optimizer": "Adam",
    "loss": "cross_entropy",
    "input": "noisy_image_x_t_at_timestep_t",
    "gradient_use": "∇log p_φ(y=S|x_t) and ∇log p_φ(y=T|x_t) for KL similarity loss",
}

# ---------------------------------------------------------------------------
# FewShotDataset – 10-shot dataset class
# reference_grounding: paper_semantic_chunk_012 10-shot dataset
# ---------------------------------------------------------------------------


class FewShotDataset:
    """
    Generic 10-shot (or N-shot) dataset for DPMs-ANT target domain fine-tuning.

    Supports all paper target domains:
      babies, sunglasses, raphael_peale, sketches, modigliani,
      haunted_houses, landscape_drawings

    Implements:
      - Loading exactly `shot` images from target domain directory
      - Random horizontal flip augmentation (paper default)
      - Resize + center crop to image_size
      - Returns a DataLoader when .get_dataloader() is called

    reference_grounding: paper_semantic_chunk_012 few_shot_target_dataset
    """

    def __init__(
        self,
        domain: str,
        shot: int = 10,
        image_size: int = 256,
        data_root: Optional[str] = None,
        seed: int = 42,
        augment: bool = True,
    ):
        """
        Args:
            domain: Target domain id or alias (resolved via DATASET_REGISTRY).
            shot: Number of images to use (default: 10, paper fixed value).
            image_size: Spatial resolution for resizing (default: 256).
            data_root: Override path to image directory; if None, uses registry default.
            seed: Random seed for reproducible subset sampling.
            augment: Enable random horizontal flip augmentation (paper default: True).
        """
        self.domain_id = resolve_dataset_id(domain)
        self.meta = DATASET_REGISTRY[self.domain_id]
        self.shot = shot
        self.image_size = image_size
        self.seed = seed
        self.augment = augment

        # Determine data root
        if data_root is not None:
            self.data_root = Path(data_root)
        else:
            env_var = self.meta.get("data_root_env", "")
            env_val = os.environ.get(env_var, "") if env_var else ""
            self.data_root = Path(env_val) if env_val else Path(self.meta["default_root"])

        self._image_paths: Optional[List[Path]] = None
        self._images = None  # lazy loaded tensor cache

    def _discover_image_paths(self) -> List[Path]:
        """Discover image files under data_root, sorted for reproducibility."""
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
        paths: List[Path] = []
        if self.data_root.exists():
            for p in sorted(self.data_root.rglob("*")):
                if p.suffix.lower() in exts:
                    paths.append(p)
        return paths

    def _sample_shot_paths(self, paths: List[Path]) -> List[Path]:
        """Reproducibly sample exactly `shot` paths from available images."""
        rng = random.Random(self.seed)
        if len(paths) >= self.shot:
            return rng.sample(paths, self.shot)
        elif len(paths) > 0:
            logger.warning(
                "Domain '%s': only %d images found, expected %d (shot=%d). "
                "Using all available images.",
                self.domain_id, len(paths), self.shot, self.shot
            )
            return paths
        else:
            logger.warning(
                "Domain '%s': no images found at '%s'. "
                "Dataset will be empty (smoke/dry-run mode).",
                self.domain_id, self.data_root
            )
            return []

    @property
    def image_paths(self) -> List[Path]:
        if self._image_paths is None:
            all_paths = self._discover_image_paths()
            self._image_paths = self._sample_shot_paths(all_paths)
        return self._image_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        """
        Returns a (image_tensor, label) tuple.
        label=1 for target domain (used in binary classifier training).
        Requires PIL and torchvision (lazy import).
        """
        try:
            from PIL import Image  # type: ignore
        except ImportError as e:
            raise ImportError("PIL/Pillow is required for FewShotDataset image loading.") from e

        try:
            import torch  # type: ignore
            import torchvision.transforms as T  # type: ignore
        except ImportError as e:
            raise ImportError(
                "torch and torchvision are required for FewShotDataset.__getitem__."
            ) from e

        path = self.image_paths[idx]

        # Build transform
        transforms = [
            T.Resize(self.image_size),
            T.CenterCrop(self.image_size),
        ]
        if self.augment:
            transforms.append(T.RandomHorizontalFlip())
        transforms += [
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
        transform = T.Compose(transforms)

        img = Image.open(path).convert("RGB")
        tensor = transform(img)
        label = 1  # target domain label for binary classifier
        return tensor, label

    def get_dataloader(
        self,
        batch_size: int = 10,
        num_workers: int = 0,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        """
        Return a DataLoader wrapping this dataset.
        Default batch_size=10 matches the 10-shot protocol (one batch = all images).

        reference_grounding: paper_semantic_chunk_012 10-shot DataLoader
        """
        try:
            from torch.utils.data import DataLoader  # type: ignore
        except ImportError as e:
            raise ImportError(
                "torch is required for FewShotDataset.get_dataloader()."
            ) from e

        return DataLoader(
            self,
            batch_size=min(batch_size, max(len(self), 1)),
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
        )

    def get_all_tensors(self):
        """
        Load all shot images and return as a single stacked tensor (N, C, H, W).
        Convenience method for adversarial noise selection inner loop.
        """
        try:
            import torch  # type: ignore
        except ImportError as e:
            raise ImportError("torch is required for FewShotDataset.get_all_tensors()") from e

        tensors = [self[i][0] for i in range(len(self))]
        if not tensors:
            import torch  # type: ignore
            return torch.zeros(0, 3, self.image_size, self.image_size)
        return torch.stack(tensors, dim=0)

    def __repr__(self) -> str:
        return (
            f"FewShotDataset(domain={self.domain_id!r}, shot={self.shot}, "
            f"image_size={self.image_size}, n_found={len(self)})"
        )


# ---------------------------------------------------------------------------
# Source domain dataset wrapper
# ---------------------------------------------------------------------------


class SourceDomainDataset:
    """
    Source domain dataset used alongside FewShotDataset for binary classifier
    training (source vs target domain, label=0 for source).

    reference_grounding: paper_semantic_chunk_014_01 classifier source domain data
    """

    def __init__(
        self,
        domain: str,
        image_size: int = 256,
        data_root: Optional[str] = None,
        augment: bool = True,
        max_samples: Optional[int] = None,
    ):
        self.domain_id = resolve_dataset_id(domain)
        self.meta = DATASET_REGISTRY[self.domain_id]
        self.image_size = image_size
        self.augment = augment
        self.max_samples = max_samples

        if data_root is not None:
            self.data_root = Path(data_root)
        else:
            env_var = self.meta.get("data_root_env", "")
            env_val = os.environ.get(env_var, "") if env_var else ""
            self.data_root = Path(env_val) if env_val else Path(self.meta["default_root"])

        self._image_paths: Optional[List[Path]] = None

    @property
    def image_paths(self) -> List[Path]:
        if self._image_paths is None:
            exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
            paths: List[Path] = []
            if self.data_root.exists():
                for p in sorted(self.data_root.rglob("*")):
                    if p.suffix.lower() in exts:
                        paths.append(p)
            if self.max_samples and len(paths) > self.max_samples:
                paths = paths[: self.max_samples]
            self._image_paths = paths
        return self._image_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        try:
            from PIL import Image  # type: ignore
            import torchvision.transforms as T  # type: ignore
        except ImportError as e:
            raise ImportError("PIL and torchvision required for SourceDomainDataset.") from e

        path = self.image_paths[idx]
        transforms = [T.Resize(self.image_size), T.CenterCrop(self.image_size)]
        if self.augment:
            transforms.append(T.RandomHorizontalFlip())
        transforms += [
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
        img = Image.open(path).convert("RGB")
        return T.Compose(transforms)(img), 0  # label=0 for source domain


# ---------------------------------------------------------------------------
# Dataset readiness check
# ---------------------------------------------------------------------------


def dataset_readiness_check(domain: str) -> Dict[str, Any]:
    """
    Check whether the dataset directory for a given domain exists and contains images.

    Returns a dict with keys: id, root, exists, image_count, ready.
    """
    domain_id = resolve_dataset_id(domain)
    meta = DATASET_REGISTRY[domain_id]
    env_var = meta.get("data_root_env", "")
    env_val = os.environ.get(env_var, "") if env_var else ""
    root = Path(env_val) if env_val else Path(meta["default_root"])

    exists = root.exists()
    image_count = 0
    if exists:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
        for p in root.rglob("*"):
            if p.suffix.lower() in exts:
                image_count += 1

    ready = exists and image_count >= (meta.get("shot_count") or 1)
    return {
        "id": domain_id,
        "root": str(root),
        "exists": exists,
        "image_count": image_count,
        "required": meta.get("shot_count"),
        "ready": ready,
    }


def check_all_datasets() -> Dict[str, Any]:
    """Run readiness checks for all registered datasets."""
    results = {}
    for did in ALL_DATASET_IDS:
        results[did] = dataset_readiness_check(did)
    return results


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def make_dataset(config: Dict[str, Any]) -> "FewShotDataset | SourceDomainDataset":
    """
    Factory: create a dataset from a config dict.

    Config keys:
      domain (str): dataset id or alias
      role (str): 'target' (FewShotDataset) or 'source' (SourceDomainDataset)
      shot (int): number of shots for target domain (default: 10)
      image_size (int): spatial resolution (default: 256)
      data_root (str|None): optional override path
      augment (bool): enable augmentation (default: True)
      seed (int): random seed (default: 42)
    """
    domain = config["domain"]
    role = config.get("role", "target")
    image_size = config.get("image_size", 256)
    data_root = config.get("data_root", None)
    augment = config.get("augment", True)

    if role == "target":
        shot = config.get("shot", 10)
        seed = config.get("seed", 42)
        return FewShotDataset(
            domain=domain,
            shot=shot,
            image_size=image_size,
            data_root=data_root,
            seed=seed,
            augment=augment,
        )
    else:
        max_samples = config.get("max_samples", None)
        return SourceDomainDataset(
            domain=domain,
            image_size=image_size,
            data_root=data_root,
            augment=augment,
            max_samples=max_samples,
        )


def make_fewshot_dataset(
    pair: Tuple[str, str],
    config: Optional[Dict[str, Any]] = None,
) -> Tuple["FewShotDataset", "SourceDomainDataset"]:
    """
    Factory: create a (target_dataset, source_dataset) pair for a source→target transfer.

    Args:
        pair: (source_domain_id, target_domain_id) tuple.
        config: optional config overrides (image_size, shot, augment, etc.)

    Returns:
        (target_ds, source_ds)

    reference_grounding: paper_semantic_chunk_012 source-target dataset pair
    """
    cfg = config or {}
    source_id, target_id = pair

    target_ds = FewShotDataset(
        domain=target_id,
        shot=cfg.get("shot", 10),
        image_size=cfg.get("image_size", 256),
        data_root=cfg.get("target_data_root", None),
        seed=cfg.get("seed", 42),
        augment=cfg.get("augment", True),
    )
    source_ds = SourceDomainDataset(
        domain=source_id,
        image_size=cfg.get("image_size", 256),
        data_root=cfg.get("source_data_root", None),
        augment=cfg.get("augment", True),
        max_samples=cfg.get("source_max_samples", None),
    )
    return target_ds, source_ds


def make_ant_environment(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Factory hook for the ANT training environment.
    Returns a metadata dict describing the environment setup.
    """
    cfg = config or {}
    env_meta = deepcopy(ENVIRONMENT_REGISTRY["ant"])
    env_meta["config_override"] = cfg
    return env_meta


def make_few_shot_generation_env(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Factory hook for the 10-shot image generation evaluation environment."""
    cfg = config or {}
    env_meta = deepcopy(ENVIRONMENT_REGISTRY["shot_image_generation"])
    env_meta["config_override"] = cfg
    return env_meta


def make_intra_lpips_evaluator(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Factory hook for Intra-LPIPS evaluator environment."""
    cfg = config or {}
    env_meta = deepcopy(ENVIRONMENT_REGISTRY["intra_lpips"])
    env_meta["config_override"] = cfg
    return env_meta


def make_experimental_setup(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Factory hook for the full experimental setup (Section 4)."""
    cfg = config or {}
    env_meta = deepcopy(ENVIRONMENT_REGISTRY["experimental_setup"])
    env_meta["config_override"] = cfg
    return env_meta


def make_lsun_church_env(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Factory hook for LSUN-Church source environment."""
    cfg = config or {}
    env_meta = deepcopy(ENVIRONMENT_REGISTRY["lsun_church"])
    env_meta["config_override"] = cfg
    return env_meta


def make_raphael_peale_env(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Factory hook for Raphael Peale target environment."""
    cfg = config or {}
    env_meta = deepcopy(ENVIRONMENT_REGISTRY["raphael_peale_env"])
    env_meta["config_override"] = cfg
    return env_meta


def make_modigliani_env(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Factory hook for Amedeo Modigliani target environment."""
    cfg = config or {}
    env_meta = deepcopy(ENVIRONMENT_REGISTRY["modigliani_env"])
    env_meta["config_override"] = cfg
    return env_meta


def make_haunted_houses_env(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Factory hook for Haunted Houses target environment."""
    cfg = config or {}
    env_meta = deepcopy(ENVIRONMENT_REGISTRY["haunted_houses_env"])
    env_meta["config_override"] = cfg
    return env_meta


def make_imagenet_env(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Factory hook for ImageNet classifier pre-training environment."""
    cfg = config or {}
    env_meta = deepcopy(ENVIRONMENT_REGISTRY["imagenet"])
    env_meta["config_override"] = cfg
    return env_meta


def make_babies_env(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Factory hook for Babies FFHQ 10-shot target environment."""
    cfg = config or {}
    env_meta = deepcopy(ENVIRONMENT_REGISTRY["babies_env"])
    env_meta["config_override"] = cfg
    return env_meta


def make_sunglasses_env(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Factory hook for Sunglasses FFHQ 10-shot target environment."""
    cfg = config or {}
    env_meta = deepcopy(ENVIRONMENT_REGISTRY["sunglasses_env"])
    env_meta["config_override"] = cfg
    return env_meta


# Registry of factory functions
FACTORY_REGISTRY: Dict[str, Callable] = {
    "make_ant_environment": make_ant_environment,
    "make_few_shot_generation_env": make_few_shot_generation_env,
    "make_intra_lpips_evaluator": make_intra_lpips_evaluator,
    "make_experimental_setup": make_experimental_setup,
    "make_lsun_church_env": make_lsun_church_env,
    "make_raphael_peale_env": make_raphael_peale_env,
    "make_modigliani_env": make_modigliani_env,
    "make_haunted_houses_env": make_haunted_houses_env,
    "make_imagenet_env": make_imagenet_env,
    "make_babies_env": make_babies_env,
    "make_sunglasses_env": make_sunglasses_env,
}


# ---------------------------------------------------------------------------
# Artifact writers
# reference_grounding: paper_method_core artifact_writers
# ---------------------------------------------------------------------------


def write_registry_artifacts(output_dir: Optional[Union[str, Path]] = None) -> Dict[str, Path]:
    """
    Write all declared JSON registry artifacts to the results directory.

    Writes:
      - dataset_registry.json
      - data_manifest.json
      - domain_registry.json
      - environment_registry.json
      - scope_report.json
      - config_resolved.json

    Returns dict of artifact_name → written_path.
    """
    if output_dir is None:
        out = _ensure_results()
    else:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

    written: Dict[str, Path] = {}

    # ── dataset_registry.json ──────────────────────────────────────────────
    dataset_reg_path = out / "dataset_registry.json"
    dataset_registry_payload = {
        "_schema": "dpms_ant_dataset_registry_v1",
        "_contract_label": "dataset_registry",
        "_source": "src/data/environments.py",
        "datasets": DATASET_REGISTRY,
        "source_domains": SOURCE_DOMAINS,
        "target_domains": TARGET_DOMAINS,
        "domain_source_map": DOMAIN_SOURCE_MAP,
        "source_to_targets": SOURCE_TO_TARGETS,
        "alias_map": _ALIAS_TO_ID,
    }
    dataset_reg_path.write_text(json.dumps(dataset_registry_payload, indent=2), encoding="utf-8")
    written["dataset_registry"] = dataset_reg_path

    # ── data_manifest.json ─────────────────────────────────────────────────
    data_manifest_path = out / "data_manifest.json"
    readiness = check_all_datasets()
    data_manifest_payload = {
        "_schema": "dpms_ant_data_manifest_v1",
        "_contract_label": "data_manifest",
        "_source": "src/data/environments.py",
        "datasets": readiness,
        "total": len(readiness),
        "ready_count": sum(1 for v in readiness.values() if v["ready"]),
        "missing_count": sum(1 for v in readiness.values() if not v["ready"]),
    }
    data_manifest_path.write_text(
        json.dumps(data_manifest_payload, indent=2), encoding="utf-8"
    )
    written["data_manifest"] = data_manifest_path

    # ── domain_registry.json ───────────────────────────────────────────────
    domain_reg_path = out / "domain_registry.json"
    domain_registry_payload = {
        "_schema": "dpms_ant_domain_registry_v1",
        "_contract_label": "domain_registry",
        "_source": "src/data/environments.py",
        "source_domains": {
            d: {"id": d, **DATASET_REGISTRY[d]}
            for d in SOURCE_DOMAINS
        },
        "target_domains": {
            d: {"id": d, **DATASET_REGISTRY[d]}
            for d in TARGET_DOMAINS
        },
        "adaptor_architecture": ADAPTOR_ARCHITECTURE_SPEC,
        "classifier_spec": CLASSIFIER_SPEC,
    }
    domain_reg_path.write_text(
        json.dumps(domain_registry_payload, indent=2), encoding="utf-8"
    )
    written["domain_registry"] = domain_reg_path

    # ── environment_registry.json ──────────────────────────────────────────
    env_reg_path = out / "environment_registry.json"
    # Make env registry JSON-serializable (strip callable references)
    serializable_env = {}
    for eid, emeta in ENVIRONMENT_REGISTRY.items():
        serializable_env[eid] = {k: v for k, v in emeta.items() if not callable(v)}
    env_registry_payload = {
        "_schema": "dpms_ant_environment_registry_v1",
        "_contract_label": "environment_registry",
        "_source": "src/data/environments.py",
        "environments": serializable_env,
        "factory_registry": list(FACTORY_REGISTRY.keys()),
        "env_alias_map": _ENV_ALIAS_TO_ID,
    }
    env_reg_path.write_text(
        json.dumps(env_registry_payload, indent=2), encoding="utf-8"
    )
    written["environment_registry"] = env_reg_path

    # ── scope_report.json ──────────────────────────────────────────────────
    scope_report_path = out / "scope_report.json"
    scope_report_payload = {
        "_schema": "dpms_ant_scope_report_v1",
        "_contract_label": "scope_report",
        "_source": "src/data/environments.py",
        "paper": "Bridging Data Gaps in Diffusion Models with Adversarial Noise-Based Transfer Learning",
        "method": "DPMs-ANT",
        "source_domains": SOURCE_DOMAINS,
        "target_domains": TARGET_DOMAINS,
        "num_target_domains": len(TARGET_DOMAINS),
        "shot_count": 10,
        "frameworks": ["ddpm", "ldm"],
        "baselines": [
            "frozen", "naive_finetune", "diff_aug", "ada", "lecam", "few_shot_gan"
        ],
        "metrics": ["fid", "intra_lpips", "accuracy", "fidelity_score"],
        "addendum_fixed_hparams": {
            "total_iterations": 5000,
            "ablation_iterations": 300,
            "batch_size": 64,
            "omega": 0.02,
            "adversarial_inner_steps": 10,
            "gamma": 5,
            "classifier_finetune_steps": 300,
        },
        "adaptor_architecture": ADAPTOR_ARCHITECTURE_SPEC,
        "classifier_spec": CLASSIFIER_SPEC,
        "registered_environments": list(ENVIRONMENT_REGISTRY.keys()),
        "registered_datasets": list(DATASET_REGISTRY.keys()),
    }
    scope_report_path.write_text(
        json.dumps(scope_report_payload, indent=2), encoding="utf-8"
    )
    written["scope_report"] = scope_report_path

    # ── config_resolved.json ───────────────────────────────────────────────
    config_resolved_path = out / "config_resolved.json"
    config_resolved_payload = {
        "_schema": "dpms_ant_config_resolved_v1",
        "_contract_label": "config_resolved",
        "_source": "src/data/environments.py",
        "resolved_dataset_defaults": {
            did: {
                "root": str(
                    Path(os.environ.get(meta.get("data_root_env", ""), ""))
                    if os.environ.get(meta.get("data_root_env", ""), "")
                    else Path(meta["default_root"])
                ),
                "image_size": meta["image_size"],
                "shot_count": meta.get("shot_count"),
                "role": meta["role"],
                "loader": meta["loader"],
            }
            for did, meta in DATASET_REGISTRY.items()
        },
        "source_to_targets": SOURCE_TO_TARGETS,
        "experiment_pairs": [
            {"source": src, "target": tgt}
            for src, targets in SOURCE_TO_TARGETS.items()
            for tgt in targets
        ],
    }
    config_resolved_path.write_text(
        json.dumps(config_resolved_payload, indent=2), encoding="utf-8"
    )
    written["config_resolved"] = config_resolved_path

    logger.info(
        "Registry artifacts written to %s: %s",
        out,
        [str(p) for p in written.values()],
    )
    return written


# ---------------------------------------------------------------------------
# Module-level self-registration (runs on import)
# ---------------------------------------------------------------------------


def _auto_write_if_env() -> None:
    """
    If PAPERBENCH_REPRO_ARTIFACT_DIR is set, write registry artifacts immediately
    on import. This ensures downstream validation can find the artifacts.
    """
    artifact_dir = os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR", "")
    if artifact_dir:
        try:
            write_registry_artifacts(artifact_dir)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Auto artifact write failed: %s", exc)


_auto_write_if_env()


# ---------------------------------------------------------------------------
# Convenience: list all experiment source→target pairs
# ---------------------------------------------------------------------------


def list_experiment_pairs() -> List[Tuple[str, str]]:
    """Return all (source, target) domain pairs from the paper experiments."""
    pairs = []
    for src, targets in SOURCE_TO_TARGETS.items():
        for tgt in targets:
            pairs.append((src, tgt))
    return pairs


def get_dataset_meta(domain: str) -> Dict[str, Any]:
    """Get full metadata for a dataset by id or alias."""
    return deepcopy(DATASET_REGISTRY[resolve_dataset_id(domain)])


def get_env_meta(env_name: str) -> Dict[str, Any]:
    """Get full metadata for an environment by id or alias."""
    return deepcopy(ENVIRONMENT_REGISTRY[resolve_env_id(env_name)])


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    # Registries
    "DATASET_REGISTRY",
    "ENVIRONMENT_REGISTRY",
    "DOMAIN_SOURCE_MAP",
    "SOURCE_TO_TARGETS",
    "SOURCE_DOMAINS",
    "TARGET_DOMAINS",
    "ALL_DATASET_IDS",
    "ADAPTOR_ARCHITECTURE_SPEC",
    "CLASSIFIER_SPEC",
    "FACTORY_REGISTRY",
    # Dataset classes
    "FewShotDataset",
    "SourceDomainDataset",
    # Resolve helpers
    "resolve_dataset_id",
    "resolve_env_id",
    # Factory functions
    "make_dataset",
    "make_fewshot_dataset",
    "make_ant_environment",
    "make_few_shot_generation_env",
    "make_intra_lpips_evaluator",
    "make_experimental_setup",
    "make_lsun_church_env",
    "make_raphael_peale_env",
    "make_modigliani_env",
    "make_haunted_houses_env",
    "make_imagenet_env",
    "make_babies_env",
    "make_sunglasses_env",
    # Readiness
    "dataset_readiness_check",
    "check_all_datasets",
    # Artifact writer
    "write_registry_artifacts",
    # Utilities
    "list_experiment_pairs",
    "get_dataset_meta",
    "get_env_meta",
]