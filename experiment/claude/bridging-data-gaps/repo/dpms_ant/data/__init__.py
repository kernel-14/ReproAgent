"""
dpms_ant/data/__init__.py
=========================
DPMs-ANT Data Package – Dataset/Benchmark Registry and Few-Shot Loader.

Paper: "Bridging Data Gaps in Diffusion Models with Adversarial Noise-Based
Transfer Learning"

Registers all paper-derived datasets with ids, setup metadata, and
loader/config hooks for:

  SOURCE DOMAINS:
    imagenet      – ImageNet pretraining for MobileNetV2 classifier
    ffhq          – Flickr-Faces-HQ (DDPM/LDM source, 256×256 faces)
    lsun_church   – LSUN Church Outdoor (DDPM source, 256×256)

  TARGET DOMAINS (10-shot few-shot settings, Table 2):
    babies            – FFHQ → Babies
    sunglasses        – FFHQ → Sunglasses
    raphael_peale     – FFHQ → Raphael Peale portraits
    sketches          – FFHQ → Face Sketches
    modigliani        – FFHQ → Modigliani paintings
    haunted_houses    – LSUN-Church → Haunted Houses
    landscape_drawings – LSUN-Church → Landscape Drawings

Classifier context (addendum):
  MobileNetV2 pre-trained on ImageNet, last layer modified to output 2
  classes (source vs target domain), fine-tuned for 300 steps.
  Accepts noisy image inputs (x_t, t) for domain discrimination.
  Provides gradients ∇log p_φ(y=S|x_t) and ∇log p_φ(y=T|x_t).

Adaptor context (addendum):
  ShiftAdaptor: down-pool → GroupNorm → 3×3 Conv → 4-head Attention
  → MLP (→8 or 16) → 4× upsample → GroupNorm → 3×3 Conv.
  DDPM: c=4, LDM: c=2, d=8 insertion layers (Section 4, Algorithm 1).

reference_grounding: paper_method_core dpms_ant/data/__init__.py
reference_grounding: paper_semantic_chunk_012 10-shot_image_generation_experiments
reference_grounding: paper_semantic_chunk_003_02 classifier_loader_finetuning_introduction_figure_two_sets
reference_grounding: paper_semantic_chunk_010_classifier_loader_finetuning_adversarial_noise_selection_subsection_adversarial_noise
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# =============================================================================
# Dataset / Benchmark Registry
# All paper-derived dataset entries with ids, metadata, and loader hooks.
# reference_grounding: paper_semantic_chunk_012 10-shot image generation
# =============================================================================

DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    # ── Source / pretraining domains ──────────────────────────────────────────
    "imagenet": {
        "id": "imagenet",
        "aliases": ["imagenet", "ImageNet", "ILSVRC", "ilsvrc"],
        "role": "classifier_pretraining",
        "description": (
            "ImageNet ILSVRC-2012 used for pretraining the MobileNetV2 domain "
            "classifier. The last linear layer is then replaced with a 2-class "
            "head and fine-tuned 300 steps on source+target domain images."
        ),
        "image_size": 224,
        "num_classes": 1000,
        "channels": 3,
        "split_default": "train",
        "loader_class": "ImageFolderDataset",
        "few_shot": False,
        "shot_count": None,
        "framework": ["ddpm", "ldm"],
        "source": True,
        "target": False,
        "classifier_pretraining": True,
        "env_var": "IMAGENET_DATA_PATH",
        "paper_reference": (
            "MobileNetV2 classifier initialized from ImageNet pretrained weights, "
            "last layer modified to 2-class output, fine-tuned 300 steps."
        ),
    },
    "ffhq": {
        "id": "ffhq",
        "aliases": ["ffhq", "FFHQ", "FlickrFacesHQ", "flickr_faces_hq"],
        "role": "source_domain",
        "description": (
            "Flickr-Faces-HQ: 70,000 high-quality face images at 256×256. "
            "Source domain for DDPM and LDM frameworks in DPMs-ANT experiments."
        ),
        "image_size": 256,
        "num_classes": None,
        "channels": 3,
        "split_default": "train",
        "loader_class": "ImageFolderDataset",
        "few_shot": False,
        "shot_count": None,
        "framework": ["ddpm", "ldm"],
        "source": True,
        "target": False,
        "pretrained_model_id": "ddpm_ffhq_256",
        "target_domains": ["babies", "sunglasses", "raphael_peale", "sketches", "modigliani"],
        "env_var": "FFHQ_DATA_PATH",
        "paper_reference": (
            "Source domain for DDPM/LDM transfer: "
            "FFHQ → {Babies, Sunglasses, Raphael Peale, Sketches, Modigliani}"
        ),
    },
    "lsun_church": {
        "id": "lsun_church",
        "aliases": [
            "lsun_church", "LSUN-Church", "lsun-church", "church",
            "lsun_church_outdoor",
        ],
        "role": "source_domain",
        "description": (
            "LSUN Church Outdoor category at 256×256. "
            "Source domain for DDPM framework in DPMs-ANT experiments."
        ),
        "image_size": 256,
        "num_classes": None,
        "channels": 3,
        "split_default": "train",
        "loader_class": "ImageFolderDataset",
        "few_shot": False,
        "shot_count": None,
        "framework": ["ddpm"],
        "source": True,
        "target": False,
        "pretrained_model_id": "ddpm_lsun_church_256",
        "target_domains": ["haunted_houses", "landscape_drawings"],
        "env_var": "LSUN_DATA_PATH",
        "paper_reference": (
            "Source domain for DDPM transfer: "
            "LSUN-Church → {Haunted Houses, Landscape Drawings}"
        ),
    },
    # ── Target / few-shot domains (10-shot, Table 2) ─────────────────────────
    "babies": {
        "id": "babies",
        "aliases": ["babies", "Babies", "baby_faces", "baby"],
        "role": "target_domain",
        "description": (
            "Baby face images. 10-shot target domain for FFHQ→Babies "
            "few-shot transfer learning experiments."
        ),
        "image_size": 256,
        "num_classes": None,
        "channels": 3,
        "split_default": "train",
        "loader_class": "FewShotDataset",
        "few_shot": True,
        "shot_count": 10,
        "source_domain": "ffhq",
        "framework": ["ddpm", "ldm"],
        "source": False,
        "target": True,
        "env_var": "BABIES_DATA_PATH",
        "paper_reference": "Table 2: FFHQ→Babies (10-shot), DPMs-ANT vs baselines",
    },
    "sunglasses": {
        "id": "sunglasses",
        "aliases": ["sunglasses", "Sunglasses", "faces_with_sunglasses"],
        "role": "target_domain",
        "description": (
            "Faces wearing sunglasses. 10-shot target domain for "
            "FFHQ→Sunglasses few-shot transfer learning experiments."
        ),
        "image_size": 256,
        "num_classes": None,
        "channels": 3,
        "split_default": "train",
        "loader_class": "FewShotDataset",
        "few_shot": True,
        "shot_count": 10,
        "source_domain": "ffhq",
        "framework": ["ddpm", "ldm"],
        "source": False,
        "target": True,
        "env_var": "SUNGLASSES_DATA_PATH",
        "paper_reference": "Table 2: FFHQ→Sunglasses (10-shot), DPMs-ANT vs baselines",
    },
    "raphael_peale": {
        "id": "raphael_peale",
        "aliases": [
            "raphael_peale", "Raphael Peale", "raphael-peale",
            "raphael_peale_portraits", "peale",
        ],
        "role": "target_domain",
        "description": (
            "Raphael Peale portrait paintings. 10-shot target domain for "
            "FFHQ→Raphael Peale few-shot transfer learning experiments."
        ),
        "image_size": 256,
        "num_classes": None,
        "channels": 3,
        "split_default": "train",
        "loader_class": "FewShotDataset",
        "few_shot": True,
        "shot_count": 10,
        "source_domain": "ffhq",
        "framework": ["ddpm", "ldm"],
        "source": False,
        "target": True,
        "env_var": "RAPHAEL_PEALE_DATA_PATH",
        "paper_reference": "Table 2: FFHQ→Raphael Peale (10-shot), DPMs-ANT vs baselines",
    },
    "sketches": {
        "id": "sketches",
        "aliases": ["sketches", "Sketches", "face_sketches", "sketch"],
        "role": "target_domain",
        "description": (
            "Face sketch drawings. 10-shot target domain for "
            "FFHQ→Sketches few-shot transfer learning experiments."
        ),
        "image_size": 256,
        "num_classes": None,
        "channels": 3,
        "split_default": "train",
        "loader_class": "FewShotDataset",
        "few_shot": True,
        "shot_count": 10,
        "source_domain": "ffhq",
        "framework": ["ddpm", "ldm"],
        "source": False,
        "target": True,
        "env_var": "SKETCHES_DATA_PATH",
        "paper_reference": "Table 2: FFHQ→Sketches (10-shot), DPMs-ANT vs baselines",
    },
    "modigliani": {
        "id": "modigliani",
        "aliases": [
            "modigliani", "Modigliani", "modigliani_paintings",
            "amedeo_modigliani",
        ],
        "role": "target_domain",
        "description": (
            "Modigliani portrait paintings. 10-shot target domain for "
            "FFHQ→Modigliani few-shot transfer learning experiments."
        ),
        "image_size": 256,
        "num_classes": None,
        "channels": 3,
        "split_default": "train",
        "loader_class": "FewShotDataset",
        "few_shot": True,
        "shot_count": 10,
        "source_domain": "ffhq",
        "framework": ["ddpm", "ldm"],
        "source": False,
        "target": True,
        "env_var": "MODIGLIANI_DATA_PATH",
        "paper_reference": "Table 2: FFHQ→Modigliani (10-shot), DPMs-ANT vs baselines",
    },
    "haunted_houses": {
        "id": "haunted_houses",
        "aliases": [
            "haunted_houses", "Haunted Houses", "haunted-houses",
            "haunted_house",
        ],
        "role": "target_domain",
        "description": (
            "Haunted house exterior images. 10-shot target domain for "
            "LSUN-Church→Haunted Houses few-shot transfer learning experiments."
        ),
        "image_size": 256,
        "num_classes": None,
        "channels": 3,
        "split_default": "train",
        "loader_class": "FewShotDataset",
        "few_shot": True,
        "shot_count": 10,
        "source_domain": "lsun_church",
        "framework": ["ddpm"],
        "source": False,
        "target": True,
        "env_var": "HAUNTED_HOUSES_DATA_PATH",
        "paper_reference": "Table 2: LSUN-Church→Haunted Houses (10-shot), DPMs-ANT vs baselines",
    },
    "landscape_drawings": {
        "id": "landscape_drawings",
        "aliases": [
            "landscape_drawings", "Landscape Drawings", "landscape-drawings",
            "landscapes", "landscape_drawing",
        ],
        "role": "target_domain",
        "description": (
            "Landscape drawings and paintings. 10-shot target domain for "
            "LSUN-Church→Landscape Drawings few-shot transfer learning experiments."
        ),
        "image_size": 256,
        "num_classes": None,
        "channels": 3,
        "split_default": "train",
        "loader_class": "FewShotDataset",
        "few_shot": True,
        "shot_count": 10,
        "source_domain": "lsun_church",
        "framework": ["ddpm"],
        "source": False,
        "target": True,
        "env_var": "LANDSCAPE_DRAWINGS_DATA_PATH",
        "paper_reference": "Table 2: LSUN-Church→Landscape Drawings (10-shot), DPMs-ANT vs baselines",
    },
}

# Source-to-target domain mapping (paper Table 2, all 7 experiment pairs)
# reference_grounding: paper_semantic_chunk_012 source-target experiment pairs
SOURCE_TARGET_PAIRS: List[Tuple[str, str]] = [
    ("ffhq", "babies"),
    ("ffhq", "sunglasses"),
    ("ffhq", "raphael_peale"),
    ("ffhq", "sketches"),
    ("ffhq", "modigliani"),
    ("lsun_church", "haunted_houses"),
    ("lsun_church", "landscape_drawings"),
]

# Alias index for O(1) lookup
_ALIAS_TO_ID: Dict[str, str] = {}
for _ds_id, _ds_info in DATASET_REGISTRY.items():
    _ALIAS_TO_ID[_ds_id.lower()] = _ds_id
    for _alias in _ds_info.get("aliases", []):
        _ALIAS_TO_ID[_alias.lower()] = _ds_id


# =============================================================================
# Registry Access Helpers
# =============================================================================

def get_dataset_info(dataset_id: str) -> Dict[str, Any]:
    """
    Resolve a dataset_id or alias to its full registry entry.

    Parameters
    ----------
    dataset_id : str
        Registered dataset id or any alias (case-insensitive).

    Returns
    -------
    Dict[str, Any]
        Full registry entry for the dataset.

    Raises
    ------
    KeyError
        If the dataset_id is not registered.
    """
    resolved = _ALIAS_TO_ID.get(dataset_id.lower())
    if resolved is None:
        raise KeyError(
            f"Unknown dataset '{dataset_id}'. "
            f"Valid ids: {sorted(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[resolved]


def get_dataset_ids() -> List[str]:
    """Return all registered dataset IDs (canonical form)."""
    return list(DATASET_REGISTRY.keys())


def get_target_domain_ids() -> List[str]:
    """Return all registered target-domain dataset IDs (10-shot few-shot)."""
    return [k for k, v in DATASET_REGISTRY.items() if v.get("target")]


def get_source_domain_ids() -> List[str]:
    """Return all registered source-domain dataset IDs."""
    return [k for k, v in DATASET_REGISTRY.items() if v.get("source")]


def get_target_domains(source_domain: str) -> List[str]:
    """Return all target domain IDs for a given source domain."""
    return [t for s, t in SOURCE_TARGET_PAIRS if s == source_domain]


def get_source_domain(target_domain: str) -> Optional[str]:
    """Return the source domain ID for a given target domain ID."""
    for s, t in SOURCE_TARGET_PAIRS:
        if t == target_domain:
            return s
    return None


def resolve_alias(alias: str) -> str:
    """Resolve any dataset alias to its canonical id."""
    resolved = _ALIAS_TO_ID.get(alias.lower())
    if resolved is None:
        raise KeyError(f"Unresolvable dataset alias: '{alias}'")
    return resolved


# =============================================================================
# Classifier Context
# Addendum: "These pre-trained models were fine-tuned by modifying the last
# layer to output two classes to classify whether images where coming from
# the source or the target dataset."
# reference_grounding: paper_semantic_chunk_003_02 classifier_loader_finetuning
# =============================================================================

CLASSIFIER_CONTEXT: Dict[str, Any] = {
    "backbone": "MobileNetV2",
    "pretrained_on": "imagenet",
    "num_output_classes": 2,
    "class_to_label": {0: "source", 1: "target"},
    "label_to_class": {"source": 0, "target": 1},
    "fine_tune_steps": 300,
    "addendum_description": (
        "Pre-trained models were fine-tuned by modifying the last layer to "
        "output two classes to classify whether images were coming from the "
        "source or the target dataset."
    ),
    "input_supports_noisy_images": True,
    "noisy_input_note": (
        "Classifier φ accepts (x_t, t) noisy image inputs where t is the "
        "diffusion timestep. Used to compute domain logits for noisy images."
    ),
    "gradient_computation": {
        "grad_log_p_source": "∇log p_φ(y=S|x_t) – gradient of log source-class probability",
        "grad_log_p_target": "∇log p_φ(y=T|x_t) – gradient of log target-class probability",
        "similarity_loss": "L_sim = γ · KL(∇log p_φ(y=S|x_t) ‖ ∇log p_φ(y=T|x_t)), γ=5",
    },
    "training_data": "Balanced batch of source + target domain images per fine-tuning step",
    "optimizer": "Adam",
}

# =============================================================================
# Adaptor Architecture Context
# Addendum: "The adaptor module is composed of a down-pooling layer followed
# by a normalization layer with 3x3 convolution. Then there is a 4 head
# attention layer followed by an MLP layer reducing feature size to 8 or 16.
# Then there is an up-sampling layer with a factor of 4, a normalization
# layer, and 3x3 convolutions. This is in relation to Section 4, algorithm 1."
# reference_grounding: paper_method_core dpms_ant/adaptor/shift_adaptor.py
# =============================================================================

ADAPTOR_CONTEXT: Dict[str, Any] = {
    "name": "ShiftAdaptor",
    "paper_section": "Section 4, Algorithm 1",
    "architecture_description": (
        "The adaptor module is composed of a down-pooling layer followed by a "
        "normalization layer with 3×3 convolution. Then there is a 4 head "
        "attention layer followed by an MLP layer reducing feature size to 8 "
        "or 16. Then there is an up-sampling layer with a factor of 4, a "
        "normalization layer, and 3×3 convolutions."
    ),
    "layer_sequence": [
        {"name": "down_pool",    "type": "AvgPool2d", "params": {"stride": 2}},
        {"name": "norm1",        "type": "GroupNorm"},
        {"name": "conv1",        "type": "Conv2d",    "params": {"kernel_size": 3, "padding": 1}},
        {"name": "attention",    "type": "MultiheadAttention", "params": {"num_heads": 4}},
        {"name": "mlp_compress", "type": "Linear",   "params": {"out_features": "8 or 16"}},
        {"name": "upsample",     "type": "Upsample",  "params": {"scale_factor": 4}},
        {"name": "norm2",        "type": "GroupNorm"},
        {"name": "conv2",        "type": "Conv2d",    "params": {"kernel_size": 3, "padding": 1}},
    ],
    "compression_ratio_c": {"ddpm": 4, "ldm": 2},
    "insertion_layers_d": 8,
    "trainable": "adaptor_parameters_only",
    "frozen": "pretrained_diffusion_backbone",
}

# =============================================================================
# Algorithm 1 Hyperparameter Anchors (from addendum)
# reference_grounding: paper_semantic_chunk_012 fixed hyperparameters
# =============================================================================

ALGORITHM1_HYPERPARAMS: Dict[str, Any] = {
    "total_iterations": 5000,
    "classifier_finetune_steps": 300,
    "shot_count": 10,
    "batch_size": 64,
    "similarity_guidance_gamma": 5,
    "adversarial_inner_steps": 10,
    "adversarial_step_size_omega": 0.02,
    "use_sim_guide_default": True,
    "use_adv_noise_default": True,
    "ablation_switches": {
        "use_sim_guide": [True, False],
        "use_adv_noise": [True, False],
    },
}


# =============================================================================
# FewShotDatasetConfig – loader configuration object
# =============================================================================

class FewShotDatasetConfig:
    """
    Configuration and loader hook for a 10-shot few-shot target domain dataset.

    Encapsulates all metadata needed to instantiate a FewShotDataset for
    Algorithm 1 training. Resolves data path from environment variable when
    explicit path is not provided.

    Parameters
    ----------
    dataset_id : str
        Registered dataset id or alias.
    data_path : str, optional
        Explicit filesystem path to images. Falls back to env variable.
    shot_count : int
        Number of target domain images. Default: 10 (paper anchor).
    image_size : int
        Spatial resolution for resizing. Default: 256.
    augment : bool
        Whether to apply random horizontal flip augmentation.
    transform_config : dict, optional
        Additional transform parameters.

    reference_grounding: paper_semantic_chunk_012 10-shot_image_generation_experiments
    """

    def __init__(
        self,
        dataset_id: str,
        data_path: Optional[str] = None,
        shot_count: int = 10,
        image_size: int = 256,
        augment: bool = True,
        transform_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        info = get_dataset_info(dataset_id)
        self.dataset_id: str = info["id"]
        self.role: str = info.get("role", "unknown")
        self.source_domain: Optional[str] = info.get("source_domain")
        self.shot_count: int = shot_count
        self.image_size: int = image_size
        self.augment: bool = augment
        self.transform_config: Dict[str, Any] = transform_config or {}
        env_key = info.get("env_var", "")
        self.data_path: str = data_path or os.environ.get(env_key, "")
        self._info: Dict[str, Any] = info

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "role": self.role,
            "source_domain": self.source_domain,
            "data_path": self.data_path,
            "shot_count": self.shot_count,
            "image_size": self.image_size,
            "augment": self.augment,
            "transform_config": self.transform_config,
        }

    def __repr__(self) -> str:
        return (
            f"FewShotDatasetConfig(id={self.dataset_id!r}, "
            f"shot_count={self.shot_count}, "
            f"image_size={self.image_size}, "
            f"source={self.source_domain!r})"
        )


def build_dataset_config(
    dataset_id: str,
    data_path: Optional[str] = None,
    shot_count: int = 10,
    image_size: int = 256,
    augment: bool = True,
    **kwargs: Any,
) -> FewShotDatasetConfig:
    """
    Build a FewShotDatasetConfig for a named paper dataset.

    Convenience factory for config-driven dataset instantiation in Algorithm 1.

    Parameters
    ----------
    dataset_id : str
        Registered dataset id or alias (e.g., 'babies', 'sunglasses').
    data_path : str, optional
        Explicit path to dataset images directory.
    shot_count : int
        Number of target-domain training images (paper anchor: 10).
    image_size : int
        Image spatial resolution (paper anchor: 256).

    Returns
    -------
    FewShotDatasetConfig
    """
    return FewShotDatasetConfig(
        dataset_id=dataset_id,
        data_path=data_path,
        shot_count=shot_count,
        image_size=image_size,
        augment=augment,
        transform_config=kwargs.get("transform_config"),
    )


def build_transform(image_size: int = 256, augment: bool = True):
    """
    Build image preprocessing transform pipeline for few-shot dataset loading.

    Constructs the standard DPMs-ANT preprocessing pipeline:
      Resize → [RandomHorizontalFlip] → ToTensor → Normalize(0.5, 0.5)

    Lazy import of torchvision to support minimal static import environments.

    Parameters
    ----------
    image_size : int
        Target spatial resolution. Paper anchor: 256.
    augment : bool
        Whether to include random horizontal flip.

    Returns
    -------
    torchvision.transforms.Compose or None if torchvision unavailable.
    """
    try:
        import torchvision.transforms as T  # lazy import
    except ImportError:
        return None

    ops: List[Any] = [T.Resize((image_size, image_size))]
    if augment:
        ops.append(T.RandomHorizontalFlip(p=0.5))
    ops.append(T.ToTensor())
    ops.append(T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    return T.Compose(ops)


def get_dataloader(
    dataset_id: str,
    data_path: str,
    batch_size: int = 64,
    shot_count: int = 10,
    image_size: int = 256,
    num_workers: int = 4,
    shuffle: bool = True,
    augment: bool = True,
    repeat_to_batch: bool = True,
):
    """
    Build a DataLoader for a registered few-shot target-domain dataset.

    Implements the data loading path required by Algorithm 1 of DPMs-ANT.
    When the dataset has fewer images than batch_size (as in the 10-shot
    setting), samples are repeated cyclically to fill each batch.

    Parameters
    ----------
    dataset_id : str
        Registered dataset id (e.g., 'babies', 'sunglasses').
    data_path : str
        Path to the image folder for this dataset.
    batch_size : int
        Training batch size. Paper anchor: 64.
    shot_count : int
        Number of target-domain images. Paper anchor: 10.
    image_size : int
        Spatial resolution. Paper anchor: 256.
    num_workers : int
        DataLoader worker processes.
    shuffle : bool
        Whether to shuffle the dataset.
    augment : bool
        Whether to apply random horizontal flip.
    repeat_to_batch : bool
        If True, use a RepeatSampler so the 10 images fill each batch of 64.

    Returns
    -------
    torch.utils.data.DataLoader

    reference_grounding: paper_semantic_chunk_012 10-shot_image_generation_experiments
    """
    try:
        import torch  # noqa: F401  lazy
        from torch.utils.data import DataLoader, RandomSampler
    except ImportError as exc:
        raise RuntimeError(
            f"torch is required for get_dataloader: {exc}"
        ) from exc

    try:
        from dpms_ant.data.few_shot_dataset import FewShotDataset
    except ImportError as exc:
        raise RuntimeError(
            f"dpms_ant.data.few_shot_dataset is required: {exc}"
        ) from exc

    transform = build_transform(image_size=image_size, augment=augment)
    dataset = FewShotDataset(
        root=data_path,
        shot_count=shot_count,
        transform=transform,
        dataset_id=dataset_id,
        repeat_to_batch=repeat_to_batch,
        batch_size=batch_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, max(len(dataset), 1)),
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
    )
    return loader


# =============================================================================
# Artifact Writer
# Writes results/dataset_registry.json with full registry content.
# reference_grounding: paper_method_core results/dataset_registry.json
# =============================================================================

def write_dataset_registry_artifact(
    output_dir: Optional[str] = None,
) -> str:
    """
    Write the dataset registry to results/dataset_registry.json.

    Produces a machine-readable artifact containing all paper-derived dataset
    ids, metadata, source-target pairs, classifier context, and adaptor context.

    Parameters
    ----------
    output_dir : str, optional
        Directory for output. Defaults to PAPERBENCH_REPRO_ARTIFACT_DIR env
        variable, then falls back to 'results/'.

    Returns
    -------
    str
        Absolute path to the written JSON file.
    """
    if output_dir is None:
        output_dir = os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR", "results")
    os.makedirs(output_dir, exist_ok=True)
    artifact_path = os.path.join(output_dir, "dataset_registry.json")

    payload: Dict[str, Any] = {
        "artifact_type": "dataset_registry",
        "artifact_version": "1.0",
        "paper": (
            "Bridging Data Gaps in Diffusion Models with "
            "Adversarial Noise-Based Transfer Learning"
        ),
        "dataset_registry": DATASET_REGISTRY,
        "source_target_pairs": [list(p) for p in SOURCE_TARGET_PAIRS],
        "classifier_context": CLASSIFIER_CONTEXT,
        "adaptor_context": ADAPTOR_CONTEXT,
        "algorithm1_hyperparams": ALGORITHM1_HYPERPARAMS,
        "summary": {
            "total_datasets": len(DATASET_REGISTRY),
            "source_domains": sorted(get_source_domain_ids()),
            "target_domains": sorted(get_target_domain_ids()),
            "few_shot_datasets": sorted(
                k for k, v in DATASET_REGISTRY.items() if v.get("few_shot")
            ),
            "experiment_pairs_count": len(SOURCE_TARGET_PAIRS),
        },
    }

    with open(artifact_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    return artifact_path


# =============================================================================
# Public API
# =============================================================================

__all__: List[str] = [
    # Registry data
    "DATASET_REGISTRY",
    "SOURCE_TARGET_PAIRS",
    "CLASSIFIER_CONTEXT",
    "ADAPTOR_CONTEXT",
    "ALGORITHM1_HYPERPARAMS",
    # Registry helpers
    "get_dataset_info",
    "get_dataset_ids",
    "get_target_domain_ids",
    "get_source_domain_ids",
    "get_target_domains",
    "get_source_domain",
    "resolve_alias",
    # Dataset config / loader builders
    "FewShotDatasetConfig",
    "build_dataset_config",
    "build_transform",
    "get_dataloader",
    # Artifact writer
    "write_dataset_registry_artifact",
]