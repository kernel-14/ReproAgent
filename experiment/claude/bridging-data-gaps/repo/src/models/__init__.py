# =============================================================================
# src/models/__init__.py
# DPMs-ANT – Models Package Public API
# Paper: "Bridging Data Gaps in Diffusion Models with Adversarial Noise-Based
#         Transfer Learning"
#
# reference_grounding: paper_semantic_chunk_005 adversarial_noise_selection
# reference_grounding: paper_semantic_chunk_008 adapter_shift_module_transfer_learning
#
# Exposes:
#   - Model factory: build_model(config) -> DDPMWithAdaptor | LDMWithAdaptor
#   - Method/baseline registry: METHOD_REGISTRY
#   - Sweep/config registry: SWEEP_REGISTRY, FIXED_HYPERPARAMETERS
#   - Classifier URL registry: CLASSIFIER_URLS (addendum Section 5.2)
#   - freeze_pretrained() utility
#   - write_model_registry_artifact()
# =============================================================================

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Method / Baseline / Attack Selector Registry
# Paper evidence contract: expose selectors for all named methods and baselines.
# reference_grounding: paper_semantic_chunk_008 method_registry
# ---------------------------------------------------------------------------

METHOD_REGISTRY: Dict[str, Dict[str, Any]] = {
    # ── Ours ──────────────────────────────────────────────────────────────
    "ours": {
        "display_name": "DPMs-ANT (Ours)",
        "description": (
            "Full DPMs-ANT: Shift Adaptor + Similarity-Guided Training "
            "+ Adversarial Noise Selection (Algorithm 1)"
        ),
        "uses_adaptor": True,
        "uses_similarity_guidance": True,
        "uses_adversarial_noise": True,
        "framework": ["ddpm", "ldm"],
    },
    "dpms_ant": {
        "display_name": "DPMs-ANT",
        "description": "Alias for 'ours'.",
        "uses_adaptor": True,
        "uses_similarity_guidance": True,
        "uses_adversarial_noise": True,
        "framework": ["ddpm", "ldm"],
    },
    "ldm_ant": {
        "display_name": "LDM-ANT",
        "description": "DPMs-ANT configured for the LDM backbone.",
        "uses_adaptor": True,
        "uses_similarity_guidance": True,
        "uses_adversarial_noise": True,
        "framework": ["ldm"],
    },
    # ── Ablation variants ─────────────────────────────────────────────────
    "similarity_guided_training": {
        "display_name": "DPMs-ANT w/ Similarity Guidance only",
        "description": "Shift Adaptor + Similarity-Guided Training; no ANT.",
        "uses_adaptor": True,
        "uses_similarity_guidance": True,
        "uses_adversarial_noise": False,
        "framework": ["ddpm", "ldm"],
    },
    "dpms_ant_wo_an": {
        "display_name": "DPMs-ANT w/o AN",
        "description": "DPMs-ANT ablation with similarity guidance only.",
        "uses_adaptor": True,
        "uses_similarity_guidance": True,
        "uses_adversarial_noise": False,
        "framework": ["ddpm", "ldm"],
    },
    "ddpm_ant_wo_an": {
        "display_name": "DDPM-ANT w/o AN",
        "description": "DDPM-branded alias for the similarity-guidance-only ablation.",
        "uses_adaptor": True,
        "uses_similarity_guidance": True,
        "uses_adversarial_noise": False,
        "framework": ["ddpm"],
    },
    "adversarial_noise_selection": {
        "display_name": "DPMs-ANT w/ ANT only",
        "description": "Shift Adaptor + Adversarial Noise Selection; no similarity guidance.",
        "uses_adaptor": True,
        "uses_similarity_guidance": False,
        "uses_adversarial_noise": True,
        "framework": ["ddpm", "ldm"],
    },
    # ── Diffusion baselines ───────────────────────────────────────────────
    "diffusion_model": {
        "display_name": "Diffusion Model (fine-tune all)",
        "description": "Full fine-tuning of the diffusion model without adaptors.",
        "uses_adaptor": False,
        "uses_similarity_guidance": False,
        "uses_adversarial_noise": False,
        "framework": ["ddpm", "ldm"],
    },
    "ddpm": {
        "display_name": "DDPM",
        "description": "DDPM backbone (no transfer adaptation).",
        "uses_adaptor": False,
        "uses_similarity_guidance": False,
        "uses_adversarial_noise": False,
        "framework": ["ddpm"],
    },
    "ldm": {
        "display_name": "LDM",
        "description": "Latent Diffusion Model backbone (no transfer adaptation).",
        "uses_adaptor": False,
        "uses_similarity_guidance": False,
        "uses_adversarial_noise": False,
        "framework": ["ldm"],
    },
    "ddpm_pa": {
        "display_name": "DDPM-PA",
        "description": "DDPM with Patch-level Attention adaptation baseline.",
        "uses_adaptor": False,
        "uses_similarity_guidance": False,
        "uses_adversarial_noise": False,
        "framework": ["ddpm"],
    },
    # ── GAN-based baselines ───────────────────────────────────────────────
    "tgan": {
        "display_name": "TGAN",
        "description": "Transfer GAN baseline for few-shot image generation.",
        "uses_adaptor": False,
        "uses_similarity_guidance": False,
        "uses_adversarial_noise": False,
        "framework": ["gan"],
    },
    "ada": {
        "display_name": "ADA",
        "description": "Adaptive Discriminator Augmentation (StyleGAN2-ADA) baseline.",
        "uses_adaptor": False,
        "uses_similarity_guidance": False,
        "uses_adversarial_noise": False,
        "framework": ["gan"],
    },
    "ewc": {
        "display_name": "EWC",
        "description": "Elastic Weight Consolidation continual-learning baseline.",
        "uses_adaptor": False,
        "uses_similarity_guidance": False,
        "uses_adversarial_noise": False,
        "framework": ["gan"],
    },
    "cdc": {
        "display_name": "CDC",
        "description": "Cross-Domain Correspondence GAN baseline.",
        "uses_adaptor": False,
        "uses_similarity_guidance": False,
        "uses_adversarial_noise": False,
        "framework": ["gan"],
    },
    "dcl": {
        "display_name": "DCL",
        "description": "Dual Contrastive Learning GAN baseline.",
        "uses_adaptor": False,
        "uses_similarity_guidance": False,
        "uses_adversarial_noise": False,
        "framework": ["gan"],
    },
    # ── Attack / sampler selectors ────────────────────────────────────────
    "pgd": {
        "display_name": "PGD",
        "description": (
            "Projected Gradient Descent adversarial attack used in "
            "Adversarial Noise Selection inner loop."
        ),
        "uses_adaptor": False,
        "uses_similarity_guidance": False,
        "uses_adversarial_noise": True,
        "framework": ["ddpm", "ldm"],
    },
    "ddim": {
        "display_name": "DDIM",
        "description": "Deterministic DDIM sampler for fast inference.",
        "uses_adaptor": False,
        "uses_similarity_guidance": False,
        "uses_adversarial_noise": False,
        "framework": ["ddpm", "ldm"],
    },
}

# ---------------------------------------------------------------------------
# Fixed Hyperparameter Anchors
# Paper evidence contract: preserve exact anchors from addendum / Section 5.2.
# reference_grounding: paper_semantic_chunk_012 fixed_hyperparameters
# ---------------------------------------------------------------------------

FIXED_HYPERPARAMETERS: Dict[str, Any] = {
    # anchor: 5000_iterations
    "total_iterations": 5000,
    # anchor: 300_training_iterations
    "ablation_iterations": 300,
    # anchor: 10_shot_setting
    "default_shot_count": 10,
    # anchor: gamma_5  (similarity guidance scale)
    "gamma": 5,
    # anchor: omega_0.02  (adversarial noise perturbation budget)
    "omega": 0.02,
    # anchor: adversarial_inner_steps_10
    "adversarial_inner_steps": 10,
    # anchor: batch_size_64
    "batch_size": 64,
}

# ---------------------------------------------------------------------------
# Bounded Parameter Sweep Registry
# Paper evidence contract: expose all named sweep axes with their value sets.
# reference_grounding: paper_semantic_chunk_012 sweep_registry
# ---------------------------------------------------------------------------

SWEEP_REGISTRY: Dict[str, Any] = {
    # Similarity guidance scale γ (ablation / sensitivity)
    "similarity_guidance_scale": {
        "values": [1, 2, 3, 5, 7, 9, 10],
        "default": 5,
        "fixed_anchor": "gamma_5",
    },
    # Adversarial noise scale ω (ablation / sensitivity)
    "adversarial_noise_scale": {
        "values": [0.01, 0.02, 0.03, 0.04, 0.05],
        "default": 0.02,
        "fixed_anchor": "omega_0.02",
    },
    # Training iteration count (ablation)
    "training_iteration_count": {
        "values": [0, 50, 100, 150, 200, 250, 300, 350],
        "default": 300,
        "fixed_anchor": "300_training_iterations",
    },
    # Shot count (data regime)
    "shot_count": {
        "values": [10, 100],
        "default": 10,
        "fixed_anchor": "10_shot_setting",
    },
    # Adversarial inner steps (PGD iterations)
    "iteration_count": {
        "values": [1, 5, 10, 20],
        "default": 10,
        "fixed_anchor": "adversarial_inner_steps_10",
    },
    # Alpha – PGD step size
    "alpha": {
        "values": [0.001, 0.005, 0.01, 0.02],
        "default": 0.01,
    },
    # Gamma – similarity guidance scale (alias)
    "gamma": {
        "values": [1, 3, 5, 7, 9],
        "default": 5,
        "fixed_anchor": "gamma_5",
    },
    # Epsilon – PGD perturbation budget (alias for omega)
    "epsilon": {
        "values": [0.01, 0.02, 0.03, 0.04, 0.05],
        "default": 0.02,
        "fixed_anchor": "omega_0.02",
    },
    # Batch size
    "batch_size": {
        "values": [8, 16, 32, 64],
        "default": 64,
        "fixed_anchor": "batch_size_64",
    },
}

# ---------------------------------------------------------------------------
# Classifier URL Registry
# Addendum Section 5.2: pre-trained classifiers used for similarity guidance.
# reference_grounding: paper_addendum_section_5_2 classifier_urls
# ---------------------------------------------------------------------------

CLASSIFIER_URLS: Dict[str, str] = {
    # DDPM framework: 256×256 classifier
    "ddpm": (
        "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/"
        "256x256_classifier.pt"
    ),
    # LDM framework: 64×64 classifier
    "ldm": (
        "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/"
        "64x64_classifier.pt"
    ),
}

# ---------------------------------------------------------------------------
# Adaptor hyperparameters per framework
# Interface contract: DDPM c=4,d=8; LDM c=2,d=8; all weights init to zero.
# reference_grounding: paper_semantic_chunk_008 shift_adaptor_params
# ---------------------------------------------------------------------------

ADAPTOR_CONFIG: Dict[str, Dict[str, int]] = {
    "ddpm": {"c": 4, "d": 8},
    "ldm": {"c": 2, "d": 8},
}


# ---------------------------------------------------------------------------
# Lazy model imports – heavy torch/model deps loaded only when needed
# ---------------------------------------------------------------------------

def _import_unet():
    """Lazy import of UNet to avoid top-level torch dependency."""
    from src.models.unet import UNetModel  # noqa: PLC0415
    return UNetModel


def _import_ddpm():
    """Lazy import of DDPM wrapper."""
    from src.models.ddpm import DDPMWithAdaptor  # noqa: PLC0415
    return DDPMWithAdaptor


def _import_ldm():
    """Lazy import of LDM wrapper."""
    from src.models.ldm import LDMWithAdaptor  # noqa: PLC0415
    return LDMWithAdaptor


def _import_ddim():
    """Lazy import of DDIM sampler."""
    from src.models.ddim import DDIMSampler  # noqa: PLC0415
    return DDIMSampler


def _import_shift_adaptor():
    """Lazy import of ShiftAdaptor."""
    from dpms_ant.adaptor.shift_adaptor import ShiftAdaptor  # noqa: PLC0415
    return ShiftAdaptor


# ---------------------------------------------------------------------------
# freeze_pretrained utility
# Interface contract: freeze all non-adaptor parameters.
# reference_grounding: paper_semantic_chunk_008 freeze_pretrained
# ---------------------------------------------------------------------------

def freeze_pretrained(model: Any) -> None:
    """Freeze all parameters except those belonging to ShiftAdaptor layers.

    After calling this function only adaptor (W_down / W_up) parameters
    have requires_grad=True, satisfying the paper's parameter-efficient
    fine-tuning contract.

    Args:
        model: A DDPMWithAdaptor or LDMWithAdaptor instance (or any nn.Module
               that contains ShiftAdaptor sub-modules).
    """
    try:
        import torch.nn as nn  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "freeze_pretrained requires PyTorch. Install torch to use this function."
        ) from exc

    ShiftAdaptor = _import_shift_adaptor()

    # First freeze everything
    for param in model.parameters():
        param.requires_grad_(False)

    # Then unfreeze adaptor parameters
    adaptor_param_count = 0
    for module in model.modules():
        if isinstance(module, ShiftAdaptor):
            for param in module.parameters():
                param.requires_grad_(True)
                adaptor_param_count += param.numel()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "adaptor_params": adaptor_param_count,
        "frozen_params": total_params - trainable_params,
    }


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(config: Dict[str, Any]) -> Any:
    """Build a DDPMWithAdaptor or LDMWithAdaptor from a config dict.

    Args:
        config: Dict with at minimum:
            - framework: "ddpm" | "ldm"
            - method: key from METHOD_REGISTRY
            - model: sub-dict with UNet architecture params

    Returns:
        Instantiated model with ShiftAdaptors inserted (if method uses adaptor).
    """
    method_key = config.get("method", "dpms_ant").lower()

    if method_key not in METHOD_REGISTRY:
        raise ValueError(
            f"Unknown method '{method_key}'. "
            f"Valid keys: {sorted(METHOD_REGISTRY.keys())}"
        )

    method_cfg = METHOD_REGISTRY[method_key]
    framework = config.get("framework")
    if framework is None:
        declared_frameworks = method_cfg.get("framework", ["ddpm"])
        framework = declared_frameworks[0] if len(declared_frameworks) == 1 else "ddpm"
    framework = str(framework).lower()
    adaptor_cfg = ADAPTOR_CONFIG.get(framework, ADAPTOR_CONFIG["ddpm"])

    if framework == "ddpm":
        DDPMWithAdaptor = _import_ddpm()
        model = DDPMWithAdaptor(
            unet_config=config.get("model", {}),
            adaptor_config=adaptor_cfg if method_cfg["uses_adaptor"] else None,
            use_adaptor=method_cfg["uses_adaptor"],
        )
    elif framework == "ldm":
        LDMWithAdaptor = _import_ldm()
        model = LDMWithAdaptor(
            unet_config=config.get("model", {}),
            adaptor_config=adaptor_cfg if method_cfg["uses_adaptor"] else None,
            use_adaptor=method_cfg["uses_adaptor"],
        )
    else:
        raise ValueError(f"Unknown framework '{framework}'. Use 'ddpm' or 'ldm'.")

    return model


def get_ddim_sampler(model: Any, **kwargs) -> Any:
    """Construct a DDIMSampler wrapping the given diffusion model."""
    DDIMSampler = _import_ddim()
    return DDIMSampler(model, **kwargs)


# ---------------------------------------------------------------------------
# Artifact writer
# Writes results/model_registry.json (declared artifact for this file).
# reference_grounding: paper_method_core artifact_contract
# ---------------------------------------------------------------------------

def write_model_registry_artifact(
    output_dir: Optional[str] = None,
    dry_run: bool = True,
) -> str:
    """Write results/model_registry.json.

    Args:
        output_dir: Directory to write into. Falls back to
                    PAPERBENCH_REPRO_ARTIFACT_DIR env var, then 'results/'.
        dry_run: If True, labels the artifact as a readiness/schema artifact.

    Returns:
        Path to the written file.
    """
    if output_dir is None:
        output_dir = os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR", "results")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "model_registry.json")

    payload: Dict[str, Any] = {
        "_artifact_type": "model_registry",
        "_dry_run": dry_run,
        "_label": (
            "DRY-RUN READINESS ARTIFACT – not a trained model or benchmark result"
            if dry_run
            else "model_registry"
        ),
        "method_registry": METHOD_REGISTRY,
        "sweep_registry": SWEEP_REGISTRY,
        "fixed_hyperparameters": FIXED_HYPERPARAMETERS,
        "adaptor_config": ADAPTOR_CONFIG,
        "classifier_urls": CLASSIFIER_URLS,
        "framework_options": ["ddpm", "ldm"],
        "method_keys": sorted(METHOD_REGISTRY.keys()),
    }

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    return out_path


# ---------------------------------------------------------------------------
# Public re-exports (lazy – only resolved when actually accessed)
# ---------------------------------------------------------------------------

__all__: List[str] = [
    # Registries
    "METHOD_REGISTRY",
    "SWEEP_REGISTRY",
    "FIXED_HYPERPARAMETERS",
    "ADAPTOR_CONFIG",
    "CLASSIFIER_URLS",
    # Factory / utilities
    "build_model",
    "get_ddim_sampler",
    "freeze_pretrained",
    # Artifact writer
    "write_model_registry_artifact",
    # Lazy accessors
    "_import_unet",
    "_import_ddpm",
    "_import_ldm",
    "_import_ddim",
    "_import_shift_adaptor",
]
