"""
src/experiment_registry.py
==========================
Experiment registry, dataset registry, domain registry, and environment
registry for DPMs-ANT: Bridging Data Gaps in Diffusion Models with
Adversarial Noise-Based Transfer Learning.

reference_grounding: paper_semantic_chunk_012 experiment_registry
reference_grounding: paper_semantic_chunk_014_01 DDPM+LDM framework evaluation
reference_grounding: paper_method_core src/experiment_registry.py

Evidence obligation matrix (paper-derived rows preserved here):
  项目骨架    -> 统一入口 train/generate/evaluate -> DDPM+LDM两框架 -> 全部7个目标域
  配置系统    -> framework选择+domain映射+超参数 -> results/metrics.json
  addendum约束 -> batch_size=64, omega=0.02, adversarial_inner_steps=10,
                  5000_iterations, 300_training_iterations -> 配置文件
  DDPM框架+ShiftAdaptor(c=4,d=8) -> 10-shot FFHQ目标域(5个) -> FID评估
  DDPM框架+ShiftAdaptor(c=4,d=8) -> 10-shot LSUN Church目标域(2个) -> FID评估
  LDM框架+ShiftAdaptor(c=2,d=8) -> 10-shot FFHQ目标域 -> FID评估
  相似性引导训练(γ=5) -> MobileNet分类器(ImageNet预训练,300步微调)
  对抗噪声选择(PGD, inner_steps=10, omega=0.02, alpha扰动预算)
  experiment_did: DPMs-ANT(ours) -> Algorithm 1 -> DDPM+LDM两框架 -> 7目标域
  消融Ablation-SimGuide -> use_sim_guide=False -> FID上升
  消融Ablation-AdvNoise -> use_adv_noise=False -> FID上升
  FFHQ源域预训练DDPM -> 10-shot迁移: Babies/Sunglasses/Raphael Peale/Sketches/Modigliani

Protocol matrix (all named experiments materialised below):
  Experiment-TableMain  | FFHQ→Babies/Sunglasses全方法对比(Table 2) | FID↓
  Experiment-FullDomain | 全7目标域DDPM框架 | FID↓/Intra-LPIPS↑
  Experiment-LDM        | LDM框架FID对比 | FID↓/Intra-LPIPS↑
  Ablation-SimGuide     | use_sim_guide=False | FID上升
  Ablation-AdvNoise     | use_adv_noise=False | FID上升
  Ablation-AdaptorHyper | 不同c/d配置对比 | FID/Intra-LPIPS
  SensitivityAnalysis-Gamma      | γ参数扫描 | FID/Intra-LPIPS
  SensitivityAnalysis-Omega      | ω参数扫描 | FID/Intra-LPIPS
  SensitivityAnalysis-Alpha      | α参数扫描 | FID/Intra-LPIPS
  SensitivityAnalysis-Iterations | iteration_count扫描 | FID
"""

from __future__ import annotations

import copy
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Addendum-constrained hyperparameters (must NOT be overridden in sweeps)
# reference_grounding: paper_semantic_chunk_014_01 addendum constraints
# ---------------------------------------------------------------------------
ADDENDUM_FIXED_HYPERPARAMS: Dict[str, Any] = {
    "batch_size": 64,
    "omega": 0.02,                      # adversarial noise perturbation budget ω
    "adversarial_inner_steps": 10,       # PGD inner loop steps
    "total_iterations": 5000,            # main fine-tuning budget
    "ablation_iterations": 300,          # ablation / sensitivity iteration cap
    "default_shot_count": 10,            # few-shot setting
    "similarity_guidance_scale": 5,      # γ = 5
    "classifier_lr": 1e-4,
    "classifier_batch_size": 64,
    "classifier_training_iterations": 300,
    "classifier_optimizer": "adam",
    "classifier_num_output_classes": 2,  # binary: source vs target domain
}

# ---------------------------------------------------------------------------
# Classifier pretrained model URLs (addendum clarification, Section 5.2)
# reference_grounding: paper_semantic_chunk_014_01 classifier_loader_finetuning
# These models are fine-tuned with last layer replaced to output 2 classes.
# ---------------------------------------------------------------------------
CLASSIFIER_PRETRAINED_URLS: Dict[str, str] = {
    "ddpm": (
        "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/"
        "256x256_classifier.pt"
    ),
    "ldm": (
        "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/"
        "64x64_classifier.pt"
    ),
}

# ---------------------------------------------------------------------------
# Dataset Registry
# paper evidence contract: explicitly register dataset/benchmark aliases for
# imagenet, ffhq, lsun_church, babies, sunglasses, raphael_peale, sketches,
# modigliani, haunted_houses, landscape_drawings
# reference_grounding: paper_semantic_chunk_012 dataset_registry
# ---------------------------------------------------------------------------
DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "imagenet": {
        "aliases": [
            "imagenet", "imagenet1k", "ilsvrc2012",
            "imagenet_1k", "imagenet_classification",
        ],
        "role": "pretraining_classification",
        "image_size": 224,
        "num_classes": 1000,
        "split_keys": ["train", "val"],
        "data_dir_env": "IMAGENET_DIR",
        "default_data_dir": "data/imagenet",
        "description": (
            "ImageNet ILSVRC-2012 used for MobileNetV2 domain classifier "
            "pretraining in similarity-guided training (Section 5.2). "
            "Classifier is initialised from ImageNet-pretrained weights then "
            "fine-tuned on source+target 10-shot pairs."
        ),
        "paper_context": "MobileNet分类器(ImageNet预训练,300步微调)",
    },
    "ffhq": {
        "aliases": ["ffhq", "ffhq_256", "ffhq256", "flickr_faces_hq"],
        "role": "source_domain",
        "image_size": 256,
        "num_classes": None,
        "split_keys": ["train"],
        "data_dir_env": "FFHQ_DIR",
        "default_data_dir": "data/ffhq",
        "description": (
            "Flickr-Faces-HQ (FFHQ) 256×256 resolution. Source domain for "
            "DDPM-based transfer experiments (Tables 1, 2, 3, 4, 5, 6, 7)."
        ),
        "paper_context": "FFHQ源域预训练DDPM",
        "pretrained_ddpm_url": (
            "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/"
            "ffhq_10m.pt"
        ),
    },
    "lsun_church": {
        "aliases": [
            "lsun_church", "lsun-church",
            "church_outdoor", "lsun_church_outdoor",
        ],
        "role": "source_domain",
        "image_size": 256,
        "num_classes": None,
        "split_keys": ["train"],
        "data_dir_env": "LSUN_CHURCH_DIR",
        "default_data_dir": "data/lsun_church",
        "description": (
            "LSUN Church-Outdoor 256×256. Source domain for "
            "Church→Landscape and Church→Haunted Houses (Table 1, 4)."
        ),
        "paper_context": "LSUN Church源域预训练DDPM",
        "pretrained_ddpm_url": (
            "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/"
            "lsun_uncond_100M_1200K_bs128.pt"
        ),
    },
    "babies": {
        "aliases": ["babies", "baby_faces", "ffhq_babies"],
        "role": "target_domain",
        "image_size": 256,
        "num_classes": None,
        "shot_count": 10,
        "split_keys": ["train"],
        "data_dir_env": "BABIES_DIR",
        "default_data_dir": "data/babies",
        "source_domain": "ffhq",
        "description": (
            "10-shot baby face images. Target domain for "
            "FFHQ→Babies adaptation (Table 1, 2 – Figure 5)."
        ),
        "paper_context": "FFHQ→Babies 10-shot target domain",
    },
    "sunglasses": {
        "aliases": ["sunglasses", "ffhq_sunglasses", "sunglasses_faces"],
        "role": "target_domain",
        "image_size": 256,
        "num_classes": None,
        "shot_count": 10,
        "split_keys": ["train"],
        "data_dir_env": "SUNGLASSES_DIR",
        "default_data_dir": "data/sunglasses",
        "source_domain": "ffhq",
        "description": (
            "10-shot sunglasses-face images. Primary ablation target domain "
            "(Tables 1–7, Figures 1, 4, 5, 6)."
        ),
        "paper_context": "FFHQ→Sunglasses 10-shot target domain (primary ablation)",
    },
    "raphael_peale": {
        "aliases": [
            "raphael_peale", "raphael", "raphael_paintings",
            "raphael_peale_paintings", "portraits_raphael",
        ],
        "role": "target_domain",
        "image_size": 256,
        "num_classes": None,
        "shot_count": 10,
        "split_keys": ["train"],
        "data_dir_env": "RAPHAEL_DIR",
        "default_data_dir": "data/raphael_peale",
        "source_domain": "ffhq",
        "description": (
            "10-shot Raphael Peale portrait paintings. Target domain for "
            "FFHQ→Raphael's paintings (Table 1, 4 – Figure 3 bottom)."
        ),
        "paper_context": "FFHQ→Raphael Peale 10-shot target domain",
    },
    "sketches": {
        "aliases": ["sketches", "face_sketches", "sketch_faces"],
        "role": "target_domain",
        "image_size": 256,
        "num_classes": None,
        "shot_count": 10,
        "split_keys": ["train"],
        "data_dir_env": "SKETCHES_DIR",
        "default_data_dir": "data/sketches",
        "source_domain": "ffhq",
        "description": (
            "10-shot face sketches. Target domain for "
            "FFHQ→Sketches (Table 1, 4)."
        ),
        "paper_context": "FFHQ→Sketches 10-shot target domain",
    },
    "modigliani": {
        "aliases": ["modigliani", "modigliani_portraits", "modigliani_paintings"],
        "role": "target_domain",
        "image_size": 256,
        "num_classes": None,
        "shot_count": 10,
        "split_keys": ["train"],
        "data_dir_env": "MODIGLIANI_DIR",
        "default_data_dir": "data/modigliani",
        "source_domain": "ffhq",
        "description": (
            "10-shot Modigliani portrait paintings. Target domain for "
            "FFHQ→Modigliani (Table 1, 4)."
        ),
        "paper_context": "FFHQ→Modigliani 10-shot target domain",
    },
    "haunted_houses": {
        "aliases": ["haunted_houses", "haunted", "spooky_houses"],
        "role": "target_domain",
        "image_size": 256,
        "num_classes": None,
        "shot_count": 10,
        "split_keys": ["train"],
        "data_dir_env": "HAUNTED_HOUSES_DIR",
        "default_data_dir": "data/haunted_houses",
        "source_domain": "lsun_church",
        "description": (
            "10-shot haunted house images. Target domain for "
            "LSUN Church→Haunted Houses (Table 1, 4)."
        ),
        "paper_context": "LSUN Church→Haunted Houses 10-shot target domain",
    },
    "landscape_drawings": {
        "aliases": [
            "landscape_drawings", "landscape", "landscapes",
            "landscape_painting", "landscape_paintings",
        ],
        "role": "target_domain",
        "image_size": 256,
        "num_classes": None,
        "shot_count": 10,
        "split_keys": ["train"],
        "data_dir_env": "LANDSCAPE_DIR",
        "default_data_dir": "data/landscape_drawings",
        "source_domain": "lsun_church",
        "description": (
            "10-shot landscape drawing images. Target domain for "
            "LSUN Church→Landscape drawings (Table 1, 4 – Figure 3 top)."
        ),
        "paper_context": "LSUN Church→Landscape drawings 10-shot target domain",
    },
}

# Build alias → canonical key lookup table
_DATASET_ALIAS_MAP: Dict[str, str] = {}
for _key, _meta in DATASET_REGISTRY.items():
    _DATASET_ALIAS_MAP[_key] = _key
    for _alias in _meta.get("aliases", []):
        _DATASET_ALIAS_MAP[_alias] = _key
        _DATASET_ALIAS_MAP[_alias.lower().replace("-", "_")] = _key


def resolve_dataset_name(name: str) -> str:
    """Resolve a dataset alias to its canonical registry key."""
    normalised = name.lower().replace("-", "_")
    key = _DATASET_ALIAS_MAP.get(normalised) or _DATASET_ALIAS_MAP.get(name)
    if key is None:
        raise KeyError(
            f"Unknown dataset '{name}'. "
            f"Registered keys: {sorted(DATASET_REGISTRY.keys())}"
        )
    return key


# ---------------------------------------------------------------------------
# Domain Registry – source/target transfer pairs
# reference_grounding: paper_semantic_chunk_012 domain_registry
# ---------------------------------------------------------------------------
DOMAIN_REGISTRY: Dict[str, Dict[str, Any]] = {
    "ffhq_to_babies": {
        "source": "ffhq",
        "target": "babies",
        "framework": "ddpm",
        "table_refs": ["Table1", "Table2", "Table4", "Figure5"],
        "adaptor": {"c": 4, "d": 8},
        "metrics": ["fid", "intra_lpips"],
        "checkpoint_dir": "checkpoints/ddpm/ffhq_to_babies",
        "result_dir": "results/ffhq_to_babies",
    },
    "ffhq_to_sunglasses": {
        "source": "ffhq",
        "target": "sunglasses",
        "framework": "ddpm",
        "table_refs": [
            "Table1", "Table2", "Table3", "Table4", "Table5", "Table6",
            "Table7", "Figure1", "Figure4", "Figure5", "Figure6",
        ],
        "adaptor": {"c": 4, "d": 8},
        "metrics": ["fid", "intra_lpips"],
        "checkpoint_dir": "checkpoints/ddpm/ffhq_to_sunglasses",
        "result_dir": "results/ffhq_to_sunglasses",
        "is_primary_ablation": True,
    },
    "ffhq_to_raphael_peale": {
        "source": "ffhq",
        "target": "raphael_peale",
        "framework": "ddpm",
        "table_refs": ["Table1", "Table4", "Figure3"],
        "adaptor": {"c": 4, "d": 8},
        "metrics": ["fid", "intra_lpips"],
        "checkpoint_dir": "checkpoints/ddpm/ffhq_to_raphael_peale",
        "result_dir": "results/ffhq_to_raphael_peale",
    },
    "ffhq_to_sketches": {
        "source": "ffhq",
        "target": "sketches",
        "framework": "ddpm",
        "table_refs": ["Table1", "Table4"],
        "adaptor": {"c": 4, "d": 8},
        "metrics": ["fid", "intra_lpips"],
        "checkpoint_dir": "checkpoints/ddpm/ffhq_to_sketches",
        "result_dir": "results/ffhq_to_sketches",
    },
    "ffhq_to_modigliani": {
        "source": "ffhq",
        "target": "modigliani",
        "framework": "ddpm",
        "table_refs": ["Table1", "Table4"],
        "adaptor": {"c": 4, "d": 8},
        "metrics": ["fid", "intra_lpips"],
        "checkpoint_dir": "checkpoints/ddpm/ffhq_to_modigliani",
        "result_dir": "results/ffhq_to_modigliani",
    },
    "lsun_church_to_haunted_houses": {
        "source": "lsun_church",
        "target": "haunted_houses",
        "framework": "ddpm",
        "table_refs": ["Table1", "Table4"],
        "adaptor": {"c": 4, "d": 8},
        "metrics": ["fid", "intra_lpips"],
        "checkpoint_dir": "checkpoints/ddpm/lsun_church_to_haunted_houses",
        "result_dir": "results/lsun_church_to_haunted_houses",
    },
    "lsun_church_to_landscape_drawings": {
        "source": "lsun_church",
        "target": "landscape_drawings",
        "framework": "ddpm",
        "table_refs": ["Table1", "Table4", "Figure3"],
        "adaptor": {"c": 4, "d": 8},
        "metrics": ["fid", "intra_lpips"],
        "checkpoint_dir": "checkpoints/ddpm/lsun_church_to_landscape_drawings",
        "result_dir": "results/lsun_church_to_landscape_drawings",
    },
    # LDM framework variants
    "ffhq_to_sunglasses_ldm": {
        "source": "ffhq",
        "target": "sunglasses",
        "framework": "ldm",
        "table_refs": ["Table2"],
        "adaptor": {"c": 2, "d": 8},
        "metrics": ["fid", "intra_lpips"],
        "checkpoint_dir": "checkpoints/ldm/ffhq_to_sunglasses",
        "result_dir": "results/ldm/ffhq_to_sunglasses",
    },
    "ffhq_to_babies_ldm": {
        "source": "ffhq",
        "target": "babies",
        "framework": "ldm",
        "table_refs": ["Table2"],
        "adaptor": {"c": 2, "d": 8},
        "metrics": ["fid", "intra_lpips"],
        "checkpoint_dir": "checkpoints/ldm/ffhq_to_babies",
        "result_dir": "results/ldm/ffhq_to_babies",
    },
}


# ---------------------------------------------------------------------------
# Environment Registry
# paper evidence contract: explicitly register environment/task aliases
# for imagenet
# reference_grounding: paper_semantic_chunk_012 environment_registry
# ---------------------------------------------------------------------------
ENVIRONMENT_REGISTRY: Dict[str, Dict[str, Any]] = {
    "imagenet": {
        "aliases": [
            "imagenet", "imagenet1k", "ilsvrc2012",
            "imagenet_classification", "imagenet_1k",
        ],
        "task_type": "image_classification",
        "framework": "pytorch",
        "image_size": 224,
        "num_classes": 1000,
        "role_in_paper": (
            "Pretraining environment for MobileNetV2 domain classifier. "
            "The classifier is then fine-tuned on source+target 10-shot pairs "
            "to provide similarity-guided training signal. Section 5.2."
        ),
        "classifier_init": "mobilenet_v2_imagenet_pretrained",
        "fine_tune_last_layer_only": True,
        "fine_tune_output_classes": 2,
        "fine_tune_optimizer": "adam",
        "fine_tune_lr": ADDENDUM_FIXED_HYPERPARAMS["classifier_lr"],
        "fine_tune_batch_size": ADDENDUM_FIXED_HYPERPARAMS["classifier_batch_size"],
        "fine_tune_iterations": ADDENDUM_FIXED_HYPERPARAMS["classifier_training_iterations"],
    },
    "diffusion_256_ddpm": {
        "aliases": ["diffusion_256", "ddpm_256", "improved_diffusion_256"],
        "task_type": "unconditional_image_generation",
        "framework": "ddpm",
        "image_size": 256,
        "num_classes": None,
        "role_in_paper": (
            "256×256 DDPM environment; source for FFHQ and LSUN-Church "
            "pre-trained model weights. Hosts Shift Adaptor (c=4, d=8)."
        ),
        "pretrained_models": {
            "ffhq": (
                "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/"
                "ffhq_10m.pt"
            ),
            "lsun_church": (
                "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/"
                "lsun_uncond_100M_1200K_bs128.pt"
            ),
        },
        "classifier_pretrained_url": CLASSIFIER_PRETRAINED_URLS["ddpm"],
    },
    "diffusion_64_ldm": {
        "aliases": ["ldm_64", "diffusion_64", "latent_diffusion_64"],
        "task_type": "latent_image_generation",
        "framework": "ldm",
        "image_size": 64,
        "num_classes": None,
        "role_in_paper": (
            "64×64 LDM classifier environment used for LDM-based experiments. "
            "Hosts Shift Adaptor (c=2, d=8). See Table 2."
        ),
        "classifier_pretrained_url": CLASSIFIER_PRETRAINED_URLS["ldm"],
    },
}

# Build environment alias map
_ENV_ALIAS_MAP: Dict[str, str] = {}
for _ekey, _emeta in ENVIRONMENT_REGISTRY.items():
    _ENV_ALIAS_MAP[_ekey] = _ekey
    for _ealias in _emeta.get("aliases", []):
        _ENV_ALIAS_MAP[_ealias] = _ekey


def resolve_environment_name(name: str) -> str:
    """Resolve an environment alias to its canonical registry key."""
    key = _ENV_ALIAS_MAP.get(name) or _ENV_ALIAS_MAP.get(
        name.lower().replace("-", "_")
    )
    if key is None:
        raise KeyError(
            f"Unknown environment '{name}'. "
            f"Registered: {sorted(ENVIRONMENT_REGISTRY.keys())}"
        )
    return key


# ---------------------------------------------------------------------------
# Baselines Registry
# reference_grounding: paper_semantic_chunk_012 baselines Table1 Table2
# ---------------------------------------------------------------------------
BASELINES_REGISTRY: Dict[str, Dict[str, Any]] = {
    "baseline_finetune": {
        "name": "Baseline (full fine-tune)",
        "method_id": "baseline_finetune",
        "framework": "ddpm",
        "use_adaptor": False,
        "use_sim_guide": False,
        "use_adv_noise": False,
        "description": (
            "Direct fine-tuning of the entire DDPM model. "
            "First row in Figures 4 and 6."
        ),
        "table_refs": ["Table1", "Table2", "Figure4", "Figure6"],
        "param_rate": 1.0,
        "is_baseline": True,
    },
    "adaptor_only": {
        "name": "Adaptor (adaptor fine-tune only)",
        "method_id": "adaptor_only",
        "framework": "ddpm",
        "use_adaptor": True,
        "use_sim_guide": False,
        "use_adv_noise": False,
        "description": (
            "Fine-tuning only the Shift Adaptor parameters (c=4, d=8). "
            "Second row in Figure 4. FID ≈ 41.88 vs baseline 38.65."
        ),
        "table_refs": ["Figure4"],
        "is_baseline": True,
    },
    "dpms_ant_no_an": {
        "name": "DPMs-ANT w/o AN",
        "method_id": "dpms_ant_no_an",
        "framework": "ddpm",
        "use_adaptor": True,
        "use_sim_guide": True,
        "use_adv_noise": False,
        "description": (
            "DPMs-ANT with similarity-guided training only (no adversarial noise). "
            "Third row in Figures 4 and 6."
        ),
        "table_refs": ["Figure4", "Figure6"],
        "is_ablation": True,
    },
    "dpms_ant": {
        "name": "DPMs-ANT (ours)",
        "method_id": "dpms_ant",
        "framework": "ddpm",
        "use_adaptor": True,
        "use_sim_guide": True,
        "use_adv_noise": True,
        "description": (
            "Full DPMs-ANT: Shift Adaptor + Similarity-Guided Training (γ=5) "
            "+ Adversarial Noise Selection (PGD, ω=0.02, inner_steps=10). "
            "Algorithm 1 in the paper."
        ),
        "table_refs": [
            "Table1", "Table2", "Table3", "Table4",
            "Figure3", "Figure4", "Figure5", "Figure6",
        ],
        "is_proposed_method": True,
    },
    "ldm_ant": {
        "name": "LDM-ANT (ours)",
        "method_id": "ldm_ant",
        "framework": "ldm",
        "use_adaptor": True,
        "use_sim_guide": True,
        "use_adv_noise": True,
        "description": "DPMs-ANT applied to the LDM framework (Table 2).",
        "table_refs": ["Table2"],
        "is_proposed_method": True,
    },
    "ddpm_pa": {
        "name": "DDPM-PA",
        "method_id": "ddpm_pa",
        "framework": "ddpm",
        "use_adaptor": False,
        "use_sim_guide": False,
        "use_adv_noise": False,
        "description": (
            "DDPM pairwise alignment (existing DDPM few-shot method). "
            "Tables 1–4, 9; user study comparison."
        ),
        "table_refs": ["Table1", "Table2", "Table3", "Table4", "Table9"],
        "is_baseline": True,
    },
    "tgan": {
        "name": "TransferGAN (TGAN)",
        "method_id": "tgan",
        "framework": "gan",
        "description": "Transfer GAN baseline (Tables 1, 4).",
        "table_refs": ["Table1", "Table4"],
        "is_baseline": True,
    },
    "ada": {
        "name": "ADA",
        "method_id": "ada",
        "framework": "gan",
        "description": "Adaptive Discriminator Augmentation GAN baseline (Tables 1, 4).",
        "table_refs": ["Table1", "Table4"],
        "is_baseline": True,
    },
    "ewc": {
        "name": "EWC",
        "method_id": "ewc",
        "framework": "gan",
        "description": "Elastic Weight Consolidation GAN baseline (Tables 1, 4).",
        "table_refs": ["Table1", "Table4"],
        "is_baseline": True,
    },
    "cdc": {
        "name": "CDC",
        "method_id": "cdc",
        "framework": "gan",
        "description": "Cross-Domain Correspondence GAN baseline (Tables 1, 4).",
        "table_refs": ["Table1", "Table4"],
        "is_baseline": True,
    },
    "dcl": {
        "name": "DCL",
        "method_id": "dcl",
        "framework": "gan",
        "description": "Dual Contrastive Loss GAN baseline (Tables 1, 4).",
        "table_refs": ["Table1", "Table4"],
        "is_baseline": True,
    },
}


# ---------------------------------------------------------------------------
# Experiment Protocol Matrix
# Materialises the paper evidence contract for all named experiments/ablations.
# reference_grounding: paper_semantic_chunk_012 protocol_matrix
# ---------------------------------------------------------------------------
EXPERIMENT_PROTOCOL_MATRIX: List[Dict[str, Any]] = [
    # ── Core contribution experiments ─────────────────────────────────────
    {
        "experiment_id": "Experiment-TableMain",
        "description": (
            "FFHQ→Babies and FFHQ→Sunglasses full method comparison "
            "(Table 2). FID(↓) for DDPM and LDM frameworks."
        ),
        "paper_ref": "Table 2",
        "paper_context": (
            "Table 2: FID(↓) results of each method on 10-shot "
            "FFHQ→Babies and Sunglasses. Best results in bold."
        ),
        "framework": ["ddpm", "ldm"],
        "source_domain": "ffhq",
        "target_domains": ["babies", "sunglasses"],
        "methods": ["dpms_ant", "ldm_ant", "ddpm_pa", "tgan", "ada"],
        "metrics": ["fid"],
        "adaptor": {"ddpm": {"c": 4, "d": 8}, "ldm": {"c": 2, "d": 8}},
        "hyperparams": {
            "total_iterations": 5000,
            "shot_count": 10,
            "batch_size": 64,
            "omega": 0.02,
            "adversarial_inner_steps": 10,
            "gamma": 5,
        },
        "artifact_paths": [
            "results/metrics.json",
            "results/table2_fid.json",
        ],
        "checkpoint_dir": "checkpoints/ddpm",
        "is_core_contribution": True,
        "hypothesis": (
            "DPMs-ANT achieves lower FID than all baselines on "
            "FFHQ→Babies and FFHQ→Sunglasses under the 10-shot setting."
        ),
    },
    {
        "experiment_id": "Experiment-FullDomain",
        "description": (
            "All 7 target domains, DDPM framework, FID and Intra-LPIPS "
            "(Table 1 and Table 4). DDPM-ANT vs GAN-based and DDPM baselines."
        ),
        "paper_ref": ["Table 1", "Table 4"],
        "paper_context": (
            "Table 1: Intra-LPIPS(↑) for both DDPM and GAN-based baselines "
            "for 10-shot image generation tasks from FFHQ and LSUN Church. "
            "Table 4: Intra-LPIPS(↑) results for DDPM-based and GAN-based baselines."
        ),
        "framework": "ddpm",
        "source_domain": ["ffhq", "lsun_church"],
        "target_domains": [
            "babies", "sunglasses", "raphael_peale",
            "sketches", "modigliani",
            "haunted_houses", "landscape_drawings",
        ],
        "methods": [
            "dpms_ant", "ddpm_pa",
            "tgan", "ada", "ewc", "cdc", "dcl",
        ],
        "metrics": ["fid", "intra_lpips"],
        "adaptor": {"c": 4, "d": 8},
        "hyperparams": {
            "total_iterations": 5000,
            "shot_count": 10,
            "batch_size": 64,
            "omega": 0.02,
            "adversarial_inner_steps": 10,
            "gamma": 5,
        },
        "artifact_paths": [
            "results/metrics.json",
            "results/table1_intra_lpips.json",
            "results/table4_intra_lpips.json",
        ],
        "is_core_contribution": True,
        "hypothesis": (
            "DDPM-ANT yields considerable improvement in Intra-LPIPS across "
            "most tasks vs GAN-based and DDPM-based methods. "
            "LDM-ANT exceeds state-of-the-art GAN-based approaches."
        ),
    },
    {
        "experiment_id": "Experiment-LDM",
        "description": (
            "LDM framework FID comparison on FFHQ→Babies and FFHQ→Sunglasses "
            "(Table 2). LDM-ANT vs DDPM-PA baseline."
        ),
        "paper_ref": "Table 2",
        "paper_context": (
            "Table 2 (LDM rows): LDM-ANT excels beyond state-of-the-art "
            "GAN-based approaches demonstrating potent diversity preservation."
        ),
        "framework": "ldm",
        "source_domain": "ffhq",
        "target_domains": ["babies", "sunglasses"],
        "methods": ["ldm_ant", "ddpm_pa"],
        "metrics": ["fid", "intra_lpips"],
        "adaptor": {"c": 2, "d": 8},
        "hyperparams": {
            "total_iterations": 5000,
            "shot_count": 10,
            "batch_size": 64,
            "omega": 0.02,
            "adversarial_inner_steps": 10,
            "gamma": 5,
        },
        "artifact_paths": [
            "results/metrics.json",
            "results/table2_fid.json",
        ],
        "is_core_contribution": True,
        "hypothesis": (
            "LDM-ANT achieves competitive or superior FID vs DDPM-PA while "
            "better preserving diversity (Intra-LPIPS)."
        ),
    },
    # ── Ablation experiments ───────────────────────────────────────────────
    {
        "experiment_id": "Ablation-SimGuide",
        "description": (
            "Ablation: remove similarity-guided training (use_sim_guide=False). "
            "Verifies that similarity guidance is necessary for quality improvement."
        ),
        "paper_ref": ["Figure 4", "Figure 6"],
        "paper_context": (
            "Figure 4: ablation on 10-shot sunglasses, 300 iterations – "
            "third row DPMs-ANT w/o AN uses only similarity-guided training. "
            "Figure 6: different iteration counts showing convergence curves."
        ),
        "framework": "ddpm",
        "source_domain": "ffhq",
        "target_domains": ["sunglasses"],
        "methods": ["adaptor_only", "dpms_ant_no_an", "dpms_ant"],
        "metrics": ["fid"],
        "adaptor": {"c": 4, "d": 8},
        "hyperparams": {
            "total_iterations": 300,
            "shot_count": 10,
            "batch_size": 64,
            "omega": 0.02,
            "adversarial_inner_steps": 10,
            "gamma": 5,
            "use_sim_guide": False,
        },
        "artifact_paths": ["results/ablation_sim_guide.json"],
        "hypothesis": (
            "Removing similarity-guided training raises FID vs full DPMs-ANT, "
            "validating the strategy effectiveness."
        ),
        "expected_trend": "fid_without_sim_guide > fid_with_sim_guide",
    },
    {
        "experiment_id": "Ablation-AdvNoise",
        "description": (
            "Ablation: remove adversarial noise selection (use_adv_noise=False). "
            "Verifies that the ANT strategy is necessary."
        ),
        "paper_ref": ["Figure 4", "Figure 6"],
        "paper_context": (
            "Figure 4/6: DPMs-ANT w/o AN (only similarity-guided) vs full "
            "DPMs-ANT. Adversarial noise accelerates transfer and balances "
            "training pace."
        ),
        "framework": "ddpm",
        "source_domain": "ffhq",
        "target_domains": ["sunglasses"],
        "methods": ["dpms_ant_no_an", "dpms_ant"],
        "metrics": ["fid"],
        "adaptor": {"c": 4, "d": 8},
        "hyperparams": {
            "total_iterations": 300,
            "shot_count": 10,
            "batch_size": 64,
            "omega": 0.02,
            "adversarial_inner_steps": 10,
            "gamma": 5,
            "use_adv_noise": False,
        },
        "artifact_paths": ["results/ablation_adv_noise.json"],
        "hypothesis": (
            "Removing adversarial noise selection raises FID vs full DPMs-ANT, "
            "validating the ANT strategy."
        ),
        "expected_trend": "fid_without_adv_noise > fid_with_adv_noise",
    },
    {
        "experiment_id": "Ablation-AdaptorHyper",
        "description": (
            "Ablation: Shift Adaptor hyperparameter sweep – different (c, d) "
            "combinations. Paper anchors: DDPM c=4/d=8, LDM c=2/d=8."
        ),
        "paper_ref": "supplementary / implicit in Tables",
        "framework": "ddpm",
        "source_domain": "ffhq",
        "target_domains": ["sunglasses"],
        "methods": ["dpms_ant"],
        "metrics": ["fid", "intra_lpips"],
        "adaptor_variants": [
            {"c": 2, "d": 4},
            {"c": 4, "d": 8},   # paper anchor for DDPM
            {"c": 8, "d": 8},
        ],
        "hyperparams": {
            "total_iterations": 300,
            "shot_count": 10,
        },
        "artifact_paths": ["results/ablation_adaptor_hyper.json"],
        "hypothesis": "c=4, d=8 provides the best FID/Intra-LPIPS trade-off for DDPM.",
    },
    {
        "experiment_id": "Ablation-ClassifierDataSize",
        "description": (
            "Classifier trained on 10 vs 100 images (Table 3). "
            "Evaluates sensitivity of similarity-guidance quality."
        ),
        "paper_ref": "Table 3",
        "paper_context": (
            "Table 3: FID and Intra-LPIPS for DPM-ANT from "
            "FFHQ→Sunglasses with classifiers trained on 10 and 100 images."
        ),
        "framework": "ddpm",
        "source_domain": "ffhq",
        "target_domains": ["sunglasses"],
        "methods": ["dpms_ant"],
        "metrics": ["fid", "intra_lpips"],
        "classifier_training_sizes": [10, 100],
        "hyperparams": {
            "total_iterations": 5000,
            "shot_count": 10,
            "classifier_iterations": 300,
        },
        "artifact_paths": ["results/table3_classifier_size.json"],
        "hypothesis": (
            "Classifier trained on 100 images provides competitive or better "
            "guidance signal vs 10-image classifier."
        ),
    },
    # ── Sensitivity analysis experiments ──────────────────────────────────
    {
        "experiment_id": "SensitivityAnalysis-Gamma",
        "description": "Effects of similarity guidance scale γ on FID and Intra-LPIPS (Table 5).",
        "paper_ref": "Table 5",
        "paper_context": (
            "Table 5: Effects of γ in FFHQ→Sunglasses case "
            "in terms of FID and Intra-LPIPS."
        ),
        "framework": "ddpm",
        "source_domain": "ffhq",
        "target_domains": ["sunglasses"],
        "methods": ["dpms_ant"],
        "metrics": ["fid", "intra_lpips"],
        "sweep_param": "gamma",
        "sweep_values": [1, 2, 5, 10, 20],
        "default_value": 5,
        "hyperparams": {
            "total_iterations": 5000,
            "shot_count": 10,
        },
        "artifact_paths": ["results/sensitivity_gamma.json"],
        "hypothesis": "γ=5 provides the optimal FID and Intra-LPIPS balance.",
    },
    {
        "experiment_id": "SensitivityAnalysis-Omega",
        "description": (
            "Effects of adversarial noise perturbation budget ω on "
            "FID and Intra-LPIPS (Table 6)."
        ),
        "paper_ref": "Table 6",
        "paper_context": (
            "Table 6: Effects of ω in FFHQ→Sunglasses case "
            "in terms of FID and Intra-LPIPS."
        ),
        "framework": "ddpm",
        "source_domain": "ffhq",
        "target_domains": ["sunglasses"],
        "methods": ["dpms_ant"],
        "metrics": ["fid", "intra_lpips"],
        "sweep_param": "omega",
        "sweep_values": [0.005, 0.01, 0.02, 0.05, 0.1],
        "default_value": 0.02,
        "hyperparams": {
            "total_iterations": 5000,
            "shot_count": 10,
        },
        "artifact_paths": ["results/sensitivity_omega.json"],
        "hypothesis": "ω=0.02 yields the best FID and Intra-LPIPS.",
    },
    {
        "experiment_id": "SensitivityAnalysis-Alpha",
        "description": (
            "Sensitivity analysis on PGD adversarial perturbation step size α. "
            "Controls per-step noise displacement in the inner loop."
        ),
        "paper_ref": "supplementary",
        "framework": "ddpm",
        "source_domain": "ffhq",
        "target_domains": ["sunglasses"],
        "methods": ["dpms_ant"],
        "metrics": ["fid", "intra_lpips"],
        "sweep_param": "alpha",
        "sweep_values": [0.001, 0.005, 0.01, 0.02],
        "default_value": 0.01,
        "hyperparams": {
            "total_iterations": 5000,
            "shot_count": 10,
            "omega": 0.02,
        },
        "artifact_paths": ["results/sensitivity_alpha.json"],
        "hypothesis": (
            "Moderate α provides the best balance between "
            "exploration and training stability."
        ),
    },
    {
        "experiment_id": "SensitivityAnalysis-Iterations",
        "description": (
            "Effects of total training iterations on FID convergence "
            "(Table 7, Figure 6). Compared across baseline, w/o AN, and full ANT."
        ),
        "paper_ref": ["Table 7", "Figure 6"],
        "paper_context": (
            "Table 7: Effects of training iteration in FFHQ→Sunglasses. "
            "Figure 6: all models trained for different iterations on 10-shot "
            "sunglasses – baseline vs DPMs-ANT w/o AN vs DPMs-ANT."
        ),
        "framework": "ddpm",
        "source_domain": "ffhq",
        "target_domains": ["sunglasses"],
        "methods": ["baseline_finetune", "dpms_ant_no_an", "dpms_ant"],
        "metrics": ["fid"],
        "sweep_param": "total_iterations",
        "sweep_values": [100, 300, 500, 1000, 2000, 5000],
        "default_value": 5000,
        "hyperparams": {"shot_count": 10},
        "artifact_paths": ["results/sensitivity_iterations.json"],
        "hypothesis": (
            "DPMs-ANT converges faster (lower FID at fewer iterations) "
            "than baseline and DPMs-ANT w/o AN."
        ),
    },
]


# ---------------------------------------------------------------------------
# FewShotDataset
# reference_grounding: paper_semantic_chunk_014_01 few_shot_dataset
# ---------------------------------------------------------------------------
class FewShotDataset:
    """
    Few-shot target domain dataset for DPMs-ANT transfer learning.

    Loads exactly ``shot`` images from the registered target domain directory
    and returns a DataLoader compatible with the DPMs-ANT training loop.

    Parameters
    ----------
    domain : str
        Target domain name (canonical name or alias).
    shot : int
        Number of images to load (default 10, paper-specified few-shot setting).
    data_root : str or Path, optional
        Root directory; subdirectory ``<data_root>/<canonical_domain>`` is used.
        Falls back to the registry ``default_data_dir`` or env-var override.
    image_size : int, optional
        Spatial resolution for resizing. Falls back to registry value (256).
    augment : bool
        Apply random horizontal flip augmentation during training.
    seed : int
        Random seed for reproducible shot subset selection.
    """

    def __init__(
        self,
        domain: str,
        shot: int = 10,
        data_root: Optional[Union[str, Path]] = None,
        image_size: Optional[int] = None,
        augment: bool = True,
        seed: int = 42,
    ) -> None:
        canonical = resolve_dataset_name(domain)
        meta = DATASET_REGISTRY[canonical]

        self.domain = canonical
        self.shot = shot
        self.seed = seed
        self.augment = augment
        self.image_size = image_size or meta.get("image_size", 256)
        self.meta = meta

        # Resolve data directory with priority: explicit > env-var > registry default
        if data_root is not None:
            self.data_dir = Path(data_root) / canonical
        else:
            env_var = meta.get("data_dir_env", "")
            if env_var and os.environ.get(env_var):
                self.data_dir = Path(os.environ[env_var])
            else:
                self.data_dir = Path(
                    meta.get("default_data_dir", f"data/{canonical}")
                )

        self._images: Optional[List[Any]] = None  # lazy cache

    # ------------------------------------------------------------------
    # Core loading logic
    # ------------------------------------------------------------------

    def _load_images(self) -> List[Any]:
        """
        Load up to ``self.shot`` images from ``self.data_dir``.

        Applies resize + normalize transforms (lazy torchvision import).
        Returns a list of (C, H, W) tensors in [-1, 1].
        """
        try:
            import torch
            from torchvision import transforms as T
            from PIL import Image as PILImage
        except ImportError as exc:
            raise RuntimeError(
                "torch, torchvision, and Pillow are required to load images. "
                f"Missing package: {exc}"
            ) from exc

        transform_ops: List[Any] = []
        if self.augment:
            transform_ops.append(T.RandomHorizontalFlip())
        transform_ops += [
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
        transform = T.Compose(transform_ops)

        img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
        if not self.data_dir.exists():
            logger.warning(
                "Data directory '%s' not found for domain '%s'.",
                self.data_dir, self.domain,
            )
            return []

        all_paths = sorted(
            p for p in self.data_dir.rglob("*")
            if p.suffix.lower() in img_exts
        )
        if not all_paths:
            logger.warning(
                "No images found in '%s' for domain '%s'.",
                self.data_dir, self.domain,
            )
            return []

        import random as _random
        rng = _random.Random(self.seed)
        selected = rng.sample(all_paths, min(self.shot, len(all_paths)))

        images: List[Any] = []
        for p in selected:
            try:
                img = PILImage.open(p).convert("RGB")
                images.append(transform(img))
            except Exception as exc:
                logger.warning("Could not load image '%s': %s", p, exc)
        return images

    def get_images(self) -> List[Any]:
        """Return the loaded image tensors (cached after first call)."""
        if self._images is None:
            self._images = self._load_images()
        return self._images

    def get_dataloader(
        self,
        batch_size: int = 64,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
    ) -> Any:
        """
        Wrap the few-shot images in a ``torch.utils.data.DataLoader``.

        Parameters
        ----------
        batch_size : int
            Batch size (addendum-fixed default 64 for classifier training).
        shuffle : bool
            Shuffle each epoch.
        num_workers : int
            DataLoader worker count.
        pin_memory : bool
            Pin tensors for GPU transfer.
        drop_last : bool
            Drop incomplete final batch.
        """
        try:
            import torch
            from torch.utils.data import TensorDataset, DataLoader
        except ImportError as exc:
            raise RuntimeError(
                f"torch is required for DataLoader: {exc}"
            ) from exc

        images = self.get_images()
        if images:
            tensor = torch.stack(images)  # (N, C, H, W)
        else:
            tensor = torch.zeros(0, 3, self.image_size, self.image_size)
        dataset = TensorDataset(tensor)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )

    def __len__(self) -> int:
        return len(self.get_images())

    def __repr__(self) -> str:
        return (
            f"FewShotDataset(domain='{self.domain}', shot={self.shot}, "
            f"image_size={self.image_size}, data_dir='{self.data_dir}')"
        )


# ---------------------------------------------------------------------------
# make_dataset factory
# reference_grounding: paper_semantic_chunk_012 dataset_factory
# ---------------------------------------------------------------------------
def make_dataset(
    config: Dict[str, Any],
    split: str = "train",
) -> FewShotDataset:
    """
    Instantiate a dataset from a configuration dict.

    Parameters
    ----------
    config : dict
        Required key: ``target_domain`` or ``domain``.
        Optional keys: ``shot_count``, ``data_root``, ``image_size``,
        ``augment``, ``seed``.
    split : str
        Dataset split; only ``'train'`` is supported for few-shot loading.

    Returns
    -------
    FewShotDataset
    """
    domain = config.get("target_domain") or config.get("domain")
    if not domain:
        raise ValueError(
            "config must contain a 'target_domain' or 'domain' key. "
            f"Received keys: {sorted(config.keys())}"
        )
    return FewShotDataset(
        domain=domain,
        shot=int(
            config.get(
                "shot_count",
                ADDENDUM_FIXED_HYPERPARAMS["default_shot_count"],
            )
        ),
        data_root=config.get("data_root"),
        image_size=config.get("image_size"),
        augment=bool(config.get("augment", True)),
        seed=int(config.get("seed", 42)),
    )


# ---------------------------------------------------------------------------
# Dataset readiness checks
# ---------------------------------------------------------------------------
def check_dataset_readiness(
    domain: str,
    data_root: Optional[Union[str, Path]] = None,
    min_images: int = 1,
) -> Dict[str, Any]:
    """
    Check whether the data directory for ``domain`` exists and contains images.

    Returns
    -------
    dict with keys: ready, domain, data_dir, num_images_found,
                    min_images_required, message
    """
    canonical = resolve_dataset_name(domain)
    meta = DATASET_REGISTRY[canonical]

    if data_root is not None:
        data_dir = Path(data_root) / canonical
    else:
        env_var = meta.get("data_dir_env", "")
        if env_var and os.environ.get(env_var):
            data_dir = Path(os.environ[env_var])
        else:
            data_dir = Path(
                meta.get("default_data_dir", f"data/{canonical}")
            )

    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    num_found = 0
    ready = False
    message = ""

    if data_dir.exists():
        num_found = sum(
            1 for p in data_dir.rglob("*")
            if p.suffix.lower() in img_exts
        )
        ready = num_found >= min_images
        if ready:
            message = f"Found {num_found} images (required ≥ {min_images})."
        else:
            message = f"Only {num_found} images found (required ≥ {min_images})."
    else:
        message = f"Data directory '{data_dir}' does not exist."

    return {
        "ready": ready,
        "domain": canonical,
        "data_dir": str(data_dir),
        "num_images_found": num_found,
        "min_images_required": min_images,
        "message": message,
    }


def check_all_target_domains(
    data_root: Optional[Union[str, Path]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Check readiness for all seven paper target domains."""
    target_domains = [
        name
        for name, meta in DATASET_REGISTRY.items()
        if meta.get("role") == "target_domain"
    ]
    return {
        domain: check_dataset_readiness(domain, data_root=data_root)
        for domain in target_domains
    }


# ---------------------------------------------------------------------------
# ExperimentRegistry class
# ---------------------------------------------------------------------------
class ExperimentRegistry:
    """
    Registry binding named experiments to environments, methods, measurements,
    and artifact paths.

    Implements the full protocol matrix from the DPMs-ANT paper evidence contract.
    """

    def __init__(self) -> None:
        self._experiments: Dict[str, Dict[str, Any]] = {
            exp["experiment_id"]: exp
            for exp in EXPERIMENT_PROTOCOL_MATRIX
        }

    def list_experiments(self) -> List[str]:
        """Return all registered experiment IDs."""
        return list(self._experiments.keys())

    def get_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Retrieve experiment configuration by ID."""
        if experiment_id not in self._experiments:
            raise KeyError(
                f"Unknown experiment '{experiment_id}'. "
                f"Available: {self.list_experiments()}"
            )
        return copy.deepcopy(self._experiments[experiment_id])

    def get_core_experiments(self) -> List[Dict[str, Any]]:
        """Return experiments marked as core paper contributions."""
        return [
            copy.deepcopy(exp)
            for exp in self._experiments.values()
            if exp.get("is_core_contribution", False)
        ]

    def get_ablations(self) -> List[Dict[str, Any]]:
        """Return ablation study experiments."""
        return [
            copy.deepcopy(exp)
            for exp in self._experiments.values()
            if exp["experiment_id"].startswith("Ablation-")
        ]

    def get_sensitivity_analyses(self) -> List[Dict[str, Any]]:
        """Return sensitivity analysis experiments."""
        return [
            copy.deepcopy(exp)
            for exp in self._experiments.values()
            if exp["experiment_id"].startswith("SensitivityAnalysis-")
        ]

    def resolve_config(
        self,
        experiment_id: str,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Produce a fully resolved configuration for an experiment by merging
        the addendum-fixed hyperparameters with experiment-specific params.

        Parameters
        ----------
        experiment_id : str
        overrides : dict, optional
            Key-value pairs that take precedence over resolved defaults.

        Returns
        -------
        dict
        """
        exp = self.get_experiment(experiment_id)
        resolved: Dict[str, Any] = copy.deepcopy(ADDENDUM_FIXED_HYPERPARAMS)
        resolved.update(exp.get("hyperparams", {}))
        resolved["experiment_id"] = experiment_id
        resolved["framework"] = exp.get("framework", "ddpm")
        resolved["adaptor"] = exp.get("adaptor", {"c": 4, "d": 8})
        resolved["metrics"] = exp.get("metrics", ["fid"])
        resolved["artifact_paths"] = exp.get("artifact_paths", [])
        if overrides:
            resolved.update(overrides)
        return resolved

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the full registry to a plain dict."""
        return {
            "experiments": list(self._experiments.values()),
            "total_experiments": len(self._experiments),
            "core_experiments": [
                e["experiment_id"] for e in self.get_core_experiments()
            ],
            "ablations": [e["experiment_id"] for e in self.get_ablations()],
            "sensitivity_analyses": [
                e["experiment_id"] for e in self.get_sensitivity_analyses()
            ],
        }


# Module-level singleton
REGISTRY = ExperimentRegistry()


# ---------------------------------------------------------------------------
# Artifact writers
# ---------------------------------------------------------------------------

def _ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_registry_artifacts(
    output_dir: Union[str, Path] = "results",
) -> Dict[str, str]:
    """
    Write all declared registry artifacts to ``output_dir``.

    Produces:
      - dataset_registry.json
      - data_manifest.json
      - domain_registry.json
      - environment_registry.json
      - scope_report.json
      - config_resolved.json
      - experiment_registry.json

    Returns
    -------
    dict mapping artifact label → written file path
    """
    # Check for environment-variable override of output directory
    artifact_env = os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR")
    if artifact_env:
        out = _ensure_dir(artifact_env)
    else:
        out = _ensure_dir(output_dir)

    written: Dict[str, str] = {}

    # ── 1. dataset_registry.json ─────────────────────────────────────────
    dataset_reg_path = out / "dataset_registry.json"
    dataset_reg_payload: Dict[str, Any] = {
        "schema": "dpms_ant_dataset_registry_v1",
        "description": (
            "Dataset registry for DPMs-ANT paper reproduction. "
            "Registers all source and target domains used in the paper."
        ),
        "datasets": DATASET_REGISTRY,
        "alias_map": _DATASET_ALIAS_MAP,
        "classifier_pretrained_urls": CLASSIFIER_PRETRAINED_URLS,
        "classifier_finetuning": {
            "optimizer": ADDENDUM_FIXED_HYPERPARAMS["classifier_optimizer"],
            "lr": ADDENDUM_FIXED_HYPERPARAMS["classifier_lr"],
            "batch_size": ADDENDUM_FIXED_HYPERPARAMS["classifier_batch_size"],
            "iterations": ADDENDUM_FIXED_HYPERPARAMS["classifier_training_iterations"],
            "output_classes": ADDENDUM_FIXED_HYPERPARAMS["classifier_num_output_classes"],
            "description": (
                "Pre-trained classifier fine-tuned to classify source vs target "
                "images. Last layer replaced to output 2 classes. Section 5.2."
            ),
        },
    }
    with open(dataset_reg_path, "w", encoding="utf-8") as fh:
        json.dump(dataset_reg_payload, fh, indent=2)
    written["dataset_registry"] = str(dataset_reg_path)

    # ── 2. data_manifest.json ────────────────────────────────────────────
    manifest_path = out / "data_manifest.json"
    manifest_payload: Dict[str, Any] = {
        "schema": "dpms_ant_data_manifest_v1",
        "description": (
            "Data manifest listing all 10-shot target domains and their "
            "expected data directories for DPMs-ANT experiments."
        ),
        "shot_count": ADDENDUM_FIXED_HYPERPARAMS["default_shot_count"],
        "source_domains": [
            {
                "name": k,
                "default_data_dir": v.get("default_data_dir"),
                "data_dir_env": v.get("data_dir_env"),
                "paper_context": v.get("paper_context"),
            }
            for k, v in DATASET_REGISTRY.items()
            if v.get("role") == "source_domain"
        ],
        "target_domains": [
            {
                "name": k,
                "source_domain": v.get("source_domain"),
                "default_data_dir": v.get("default_data_dir"),
                "data_dir_env": v.get("data_dir_env"),
                "paper_context": v.get("paper_context"),
            }
            for k, v in DATASET_REGISTRY.items()
            if v.get("role") == "target_domain"
        ],
    }
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest_payload, fh, indent=2)
    written["data_manifest"] = str(manifest_path)

    # ── 3. domain_registry.json ──────────────────────────────────────────
    domain_reg_path = out / "domain_registry.json"
    domain_reg_payload: Dict[str, Any] = {
        "schema": "dpms_ant_domain_registry_v1",
        "description": (
            "Transfer domain pairs used in DPMs-ANT experiments. "
            "7 target domains across 2 source domains."
        ),
        "total_domain_pairs": len(DOMAIN_REGISTRY),
        "source_domains": ["ffhq", "lsun_church"],
        "target_domains": [
            "babies", "sunglasses", "raphael_peale",
            "sketches", "modigliani",
            "haunted_houses", "landscape_drawings",
        ],
        "domain_pairs": DOMAIN_REGISTRY,
        "adaptor_config_per_framework": {
            "ddpm": {"c": 4, "d": 8},
            "ldm": {"c": 2, "d": 8},
        },
    }
    with open(domain_reg_path, "w", encoding="utf-8") as fh:
        json.dump(domain_reg_payload, fh, indent=2)
    written["domain_registry"] = str(domain_reg_path)

    # ── 4. environment_registry.json ─────────────────────────────────────
    env_reg_path = out / "environment_registry.json"
    env_reg_payload: Dict[str, Any] = {
        "schema": "dpms_ant_environment_registry_v1",
        "description": (
            "Environment and task registry for DPMs-ANT. "
            "Includes imagenet environment alias for classifier pretraining."
        ),
        "environments": ENVIRONMENT_REGISTRY,
        "imagenet_aliases": ENVIRONMENT_REGISTRY["imagenet"]["aliases"],
        "environment_alias_map": _ENV_ALIAS_MAP,
    }
    with open(env_reg_path, "w", encoding="utf-8") as fh:
        json.dump(env_reg_payload, fh, indent=2)
    written["environment_registry"] = str(env_reg_path)

    # ── 5. scope_report.json ─────────────────────────────────────────────
    scope_path = out / "scope_report.json"
    scope_payload: Dict[str, Any] = {
        "schema": "dpms_ant_scope_report_v1",
        "description": (
            "Scope report for DPMs-ANT paper reproduction. "
            "Lists all datasets, domains, experiments, paper figures/tables."
        ),
        "paper": (
            "Bridging Data Gaps in Diffusion Models with "
            "Adversarial Noise-Based Transfer Learning"
        ),
        "frameworks": ["ddpm", "ldm"],
        "total_target_domains": 7,
        "target_domain_list": [
            "babies", "sunglasses", "raphael_peale",
            "sketches", "modigliani",
            "haunted_houses", "landscape_drawings",
        ],
        "source_domain_list": ["ffhq", "lsun_church"],
        "total_experiments": len(EXPERIMENT_PROTOCOL_MATRIX),
        "core_experiments": [
            e["experiment_id"]
            for e in EXPERIMENT_PROTOCOL_MATRIX
            if e.get("is_core_contribution")
        ],
        "ablation_experiments": [
            e["experiment_id"]
            for e in EXPERIMENT_PROTOCOL_MATRIX
            if e["experiment_id"].startswith("Ablation-")
        ],
        "sensitivity_experiments": [
            e["experiment_id"]
            for e in EXPERIMENT_PROTOCOL_MATRIX
            if e["experiment_id"].startswith("SensitivityAnalysis-")
        ],
        "paper_figures": [
            "Figure1", "Figure2", "Figure3",
            "Figure4", "Figure5", "Figure6",
        ],
        "paper_tables": [
            "Table1", "Table2", "Table3", "Table4",
            "Table5", "Table6", "Table7", "Table8", "Table9",
        ],
        "baselines": list(BASELINES_REGISTRY.keys()),
        "addendum_fixed_hyperparams": ADDENDUM_FIXED_HYPERPARAMS,
        "classifier_pretrained_urls": CLASSIFIER_PRETRAINED_URLS,
        "metrics_to_compute": [
            "fid", "intra_lpips", "accuracy", "fidelity_score",
        ],
        "artifact_output_root": str(out),
    }
    with open(scope_path, "w", encoding="utf-8") as fh:
        json.dump(scope_payload, fh, indent=2)
    written["scope_report"] = str(scope_path)

    # ── 6. config_resolved.json ──────────────────────────────────────────
    config_resolved_path = out / "config_resolved.json"
    config_resolved_payload: Dict[str, Any] = {
        "schema": "dpms_ant_config_resolved_v1",
        "description": (
            "Resolved configurations for all registered experiments, "
            "merging addendum-fixed hyperparameters with per-experiment params."
        ),
        "addendum_fixed_hyperparams": ADDENDUM_FIXED_HYPERPARAMS,
        "resolved_experiments": {
            exp["experiment_id"]: REGISTRY.resolve_config(exp["experiment_id"])
            for exp in EXPERIMENT_PROTOCOL_MATRIX
        },
    }
    with open(config_resolved_path, "w", encoding="utf-8") as fh:
        json.dump(config_resolved_payload, fh, indent=2)
    written["config_resolved"] = str(config_resolved_path)

    # ── 7. experiment_registry.json ──────────────────────────────────────
    exp_reg_path = out / "experiment_registry.json"
    exp_reg_payload: Dict[str, Any] = {
        "schema": "dpms_ant_experiment_registry_v1",
        "description": (
            "Full experiment protocol matrix for DPMs-ANT paper reproduction. "
            "Links named experiments to environments, methods, metrics, and "
            "artifact output paths."
        ),
        "registry": REGISTRY.to_dict(),
        "baselines": BASELINES_REGISTRY,
        "protocol_matrix": EXPERIMENT_PROTOCOL_MATRIX,
    }
    with open(exp_reg_path, "w", encoding="utf-8") as fh:
        json.dump(exp_reg_payload, fh, indent=2)
    written["experiment_registry"] = str(exp_reg_path)

    logger.info(
        "Registry artifacts written to '%s': %s",
        out, list(written.keys()),
    )
    return written


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
__all__ = [
    # Registries
    "DATASET_REGISTRY",
    "DOMAIN_REGISTRY",
    "ENVIRONMENT_REGISTRY",
    "BASELINES_REGISTRY",
    "EXPERIMENT_PROTOCOL_MATRIX",
    "ADDENDUM_FIXED_HYPERPARAMS",
    "CLASSIFIER_PRETRAINED_URLS",
    # Classes
    "FewShotDataset",
    "ExperimentRegistry",
    "REGISTRY",
    # Functions
    "make_dataset",
    "resolve_dataset_name",
    "resolve_environment_name",
    "check_dataset_readiness",
    "check_all_target_domains",
    "write_registry_artifacts",
]


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    paths = write_registry_artifacts(output_dir=out_dir)
    print(json.dumps(paths, indent=2))