"""
src/methods/baselines.py
========================
Method / Baseline Registry and Selectable Adapters for DPMs-ANT.

Paper: "Bridging Data Gaps in Diffusion Models with Adversarial Noise-Based
Transfer Learning"

Complete method/baseline selector set (paper evidence contract):
  ours | diffusion_model | ddpm | ldm | dpms_ant |
  similarity_guided_training | adversarial_noise_selection |
  ddpm_pa | tgan | ada | ewc | cdc | dcl | pgd | ddim |
  gan | ffhq | lpips

reference_grounding: paper_semantic_chunk_012 10-shot image generation experiments
reference_grounding: paper_semantic_chunk_014_01 DDPM + LDM framework evaluation
reference_grounding: paper_method_core Algorithm 1 DPMs-ANT full training procedure
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ============================================================================
# SECTION 1 – FIXED HYPERPARAMETERS
# Addendum-mandated anchor values; must not be overridden in sweeps.
# reference_grounding: paper_semantic_chunk_012 fixed hyperparameters
# ============================================================================

FIXED_HYPERPARAMETERS: Dict[str, Any] = {
    # anchor: 5000_iterations  – total fine-tuning budget
    "total_iterations": 5000,
    # anchor: 300_training_iterations  – classifier/ablation training steps
    "classifier_training_iterations": 300,
    # anchor: 10_shot_setting
    "shot_count": 10,
    # anchor: gamma_5  – similarity guidance weight
    "similarity_guidance_scale": 5,
    # anchor: omega_0.02  – PGD adversarial step-size
    "adversarial_step_size": 0.02,
    # anchor: adversarial_inner_steps_10
    "adversarial_inner_steps": 10,
    # anchor: batch_size_64
    "batch_size": 64,
    # DDPM Shift Adaptor bottleneck: c=4, d=8
    "ddpm_adaptor_c": 4,
    "ddpm_adaptor_d": 8,
    # LDM Shift Adaptor bottleneck: c=2, d=8
    "ldm_adaptor_c": 2,
    "ldm_adaptor_d": 8,
    # All adaptor weights initialised to zero
    "adaptor_init_value": 0.0,
    # Non-adaptor parameters fully frozen
    "freeze_non_adaptor": True,
}

# ============================================================================
# SECTION 2 – PARAMETER SWEEP REGISTRY
# Bounded configuration values for sensitivity / ablation analyses.
# reference_grounding: paper_semantic_chunk_012 parameter sweeps
# ============================================================================

SWEEP_REGISTRY: Dict[str, List[Any]] = {
    # Shot count: main (10) and sensitivity (100)
    "shot_count": [10, 100],
    # Classifier training iteration count for ablation study
    "training_iteration_count": [0, 50, 100, 150, 200, 250, 300, 350],
    # Similarity guidance scale γ sensitivity
    "similarity_guidance_scale": [1, 2, 3, 5, 7, 9, 10],
    # Adversarial noise scale (omega/epsilon) sensitivity
    "adversarial_noise_scale": [0.01, 0.02, 0.03, 0.04, 0.05],
    # Batch size (fixed at 64 in main experiments)
    "batch_size": [64],
    # General loss-weighting alpha sweep
    "alpha": [0.1, 0.5, 1.0, 2.0, 5.0],
    # Gamma sweep (similarity guidance weight)
    "gamma": [1, 3, 5, 7, 9],
    # Epsilon (perturbation budget) sweep
    "epsilon": [0.01, 0.02, 0.03, 0.04, 0.05],
    # PGD inner iteration count sweep
    "iteration_count": [5, 10, 15, 20],
}

# ============================================================================
# SECTION 3 – METHOD CONFIGURATION REGISTRY
# Complete method / baseline selector set per paper evidence contract.
# reference_grounding: paper_semantic_chunk_014_01 DDPM + LDM framework evaluation
# ============================================================================


@dataclass
class MethodConfig:
    """
    Configuration descriptor for a paper method or comparison baseline.

    Fields
    ------
    method_id    : canonical identifier (lower-case, underscores)
    display_name : human-readable label
    description  : concise description with loss formula where applicable
    framework    : ddpm | ldm | gan | general | metric | dataset
    category     : ours | ablation | baseline | baseline_family | attack | sampler | metric | dataset
    hyperparams  : method-specific overrides (merged with FIXED_HYPERPARAMETERS)
    adaptor_only : True → only Shift Adaptor parameters are trained
    use_sim_guide: True → Similarity-Guided Training loss included
    use_adv_noise: True → Adversarial Noise Selection (PGD) included
    is_pgd_based : True → PGD inner optimisation is used
    paper_ref    : paper section / table reference
    """

    method_id: str
    display_name: str
    description: str
    framework: str
    category: str
    hyperparams: Dict[str, Any] = field(default_factory=dict)
    adaptor_only: bool = False
    use_sim_guide: bool = False
    use_adv_noise: bool = False
    is_pgd_based: bool = False
    paper_ref: str = ""

    def merged_hyperparams(self) -> Dict[str, Any]:
        """Return FIXED_HYPERPARAMETERS merged with method-specific overrides."""
        merged = dict(FIXED_HYPERPARAMETERS)
        merged.update(self.hyperparams)
        return merged

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method_id": self.method_id,
            "display_name": self.display_name,
            "description": self.description,
            "framework": self.framework,
            "category": self.category,
            "hyperparams": self.merged_hyperparams(),
            "adaptor_only": self.adaptor_only,
            "use_sim_guide": self.use_sim_guide,
            "use_adv_noise": self.use_adv_noise,
            "is_pgd_based": self.is_pgd_based,
            "paper_ref": self.paper_ref,
        }


METHOD_REGISTRY: Dict[str, MethodConfig] = {
    # ------------------------------------------------------------------
    # OUR METHOD – DPMs-ANT full (Algorithm 1)
    # ------------------------------------------------------------------
    "ours": MethodConfig(
        method_id="ours",
        display_name="DPMs-ANT (Ours)",
        description=(
            "Full DPMs-ANT method: Shift Adaptor (adaptor-only fine-tuning, "
            "all adaptor params init=0, non-adaptor frozen) + "
            "Similarity-Guided Training (MobileNetV2 classifier φ fine-tuned "
            "300 steps, L_sim = γ·KL(p_φ(S|x_t), p_φ(T|x_t)), γ=5) + "
            "Adversarial Noise Selection (PGD K=10, omega=0.02, "
            "ε* = argmax L_simple(x_0+ε)). Implements Algorithm 1."
        ),
        framework="ddpm",
        category="ours",
        hyperparams={
            "total_iterations": 5000,
            "classifier_training_iterations": 300,
            "similarity_guidance_scale": 5,
            "adversarial_step_size": 0.02,
            "adversarial_inner_steps": 10,
            "batch_size": 64,
        },
        adaptor_only=True,
        use_sim_guide=True,
        use_adv_noise=True,
        is_pgd_based=True,
        paper_ref="Algorithm 1 / Table 2 (Ours)",
    ),
    "dpms_ant": MethodConfig(
        method_id="dpms_ant",
        display_name="DPMs-ANT",
        description="Alias for 'ours'. Full DPMs-ANT method with Algorithm 1.",
        framework="ddpm",
        category="ours",
        hyperparams={
            "total_iterations": 5000,
            "classifier_training_iterations": 300,
            "similarity_guidance_scale": 5,
            "adversarial_step_size": 0.02,
            "adversarial_inner_steps": 10,
            "batch_size": 64,
        },
        adaptor_only=True,
        use_sim_guide=True,
        use_adv_noise=True,
        is_pgd_based=True,
        paper_ref="Algorithm 1 / Table 2 (Ours)",
    ),
    "ddpm_ant": MethodConfig(
        method_id="ddpm_ant",
        display_name="DDPM-ANT",
        description=(
            "DPMs-ANT applied to DDPM backbone. "
            "Shift Adaptor: c=4, d=8. Full Algorithm 1."
        ),
        framework="ddpm",
        category="ours",
        hyperparams={
            "total_iterations": 5000,
            "classifier_training_iterations": 300,
            "similarity_guidance_scale": 5,
            "adversarial_step_size": 0.02,
            "adversarial_inner_steps": 10,
            "batch_size": 64,
            "ddpm_adaptor_c": 4,
            "ddpm_adaptor_d": 8,
        },
        adaptor_only=True,
        use_sim_guide=True,
        use_adv_noise=True,
        is_pgd_based=True,
        paper_ref="Table 2 (Ours, DDPM framework)",
    ),
    # ------------------------------------------------------------------
    # ABLATION VARIANTS
    # ------------------------------------------------------------------
    "similarity_guided_training": MethodConfig(
        method_id="similarity_guided_training",
        display_name="Similarity-Guided Training (ablation)",
        description=(
            "Ablation – Shift Adaptor + Similarity-Guided Training only. "
            "use_adv_noise=False, use_sim_guide=True (γ=5, 300 steps). "
            "L_total = L_simple + γ·L_sim."
        ),
        framework="ddpm",
        category="ablation",
        hyperparams={
            "total_iterations": 5000,
            "classifier_training_iterations": 300,
            "similarity_guidance_scale": 5,
            "batch_size": 64,
        },
        adaptor_only=True,
        use_sim_guide=True,
        use_adv_noise=False,
        paper_ref="Table 4 (Ablation: w/o ANS)",
    ),
    "adversarial_noise_selection": MethodConfig(
        method_id="adversarial_noise_selection",
        display_name="Adversarial Noise Selection (ablation)",
        description=(
            "Ablation – Shift Adaptor + Adversarial Noise Selection only. "
            "use_adv_noise=True (K=10, omega=0.02), use_sim_guide=False. "
            "L_total = L_simple(x_0 + ε*)."
        ),
        framework="ddpm",
        category="ablation",
        hyperparams={
            "total_iterations": 5000,
            "adversarial_step_size": 0.02,
            "adversarial_inner_steps": 10,
            "batch_size": 64,
        },
        adaptor_only=True,
        use_sim_guide=False,
        use_adv_noise=True,
        is_pgd_based=True,
        paper_ref="Table 4 (Ablation: w/o SGT)",
    ),
    # ------------------------------------------------------------------
    # DIFFUSION MODEL BASELINES
    # ------------------------------------------------------------------
    "diffusion_model": MethodConfig(
        method_id="diffusion_model",
        display_name="Vanilla Diffusion Fine-tune",
        description=(
            "Vanilla diffusion model full-parameter fine-tuning. "
            "All DDPM parameters updated, no adaptor/regularisation/adversarial training. "
            "L = E[||ε - ε_θ(x_t, t)||²]."
        ),
        framework="ddpm",
        category="baseline",
        hyperparams={"total_iterations": 5000, "batch_size": 64},
        adaptor_only=False,
        use_sim_guide=False,
        use_adv_noise=False,
        paper_ref="Table 2 (Vanilla DM fine-tune)",
    ),
    "ddpm": MethodConfig(
        method_id="ddpm",
        display_name="DDPM",
        description=(
            "Pre-trained DDPM (Ho et al. 2020) without target-domain fine-tuning. "
            "Source domain generation reference."
        ),
        framework="ddpm",
        category="baseline",
        hyperparams={"batch_size": 64},
        adaptor_only=False,
        use_sim_guide=False,
        use_adv_noise=False,
        paper_ref="Table 2 (DDPM baseline)",
    ),
    "ldm": MethodConfig(
        method_id="ldm",
        display_name="LDM",
        description=(
            "Pre-trained Latent Diffusion Model (Rombach et al. 2022) "
            "without target-domain fine-tuning. Source domain reference."
        ),
        framework="ldm",
        category="baseline",
        hyperparams={"batch_size": 64},
        adaptor_only=False,
        use_sim_guide=False,
        use_adv_noise=False,
        paper_ref="Table 2 (LDM baseline)",
    ),
    "ddpm_pa": MethodConfig(
        method_id="ddpm_pa",
        display_name="DDPM-PA",
        description=(
            "DDPM with full Parameter Adaptation: all DDPM parameters fine-tuned "
            "on target domain (no adaptor, no adversarial training, no similarity guidance). "
            "Direct full-parameter fine-tuning baseline."
        ),
        framework="ddpm",
        category="baseline",
        hyperparams={"total_iterations": 5000, "batch_size": 64},
        adaptor_only=False,
        use_sim_guide=False,
        use_adv_noise=False,
        paper_ref="Table 2 (DDPM-PA)",
    ),
    # ------------------------------------------------------------------
    # GAN-BASED BASELINES
    # ------------------------------------------------------------------
    "gan": MethodConfig(
        method_id="gan",
        display_name="GAN (family)",
        description="GAN family baselines: TGAN, ADA, CDC, DCL.",
        framework="gan",
        category="baseline_family",
        paper_ref="Table 2 (GAN family baselines)",
    ),
    "tgan": MethodConfig(
        method_id="tgan",
        display_name="TransferGAN (TGAN)",
        description=(
            "TransferGAN (Wang et al. 2018): pre-trained StyleGAN2 fine-tuned on "
            "10-shot target images using standard adversarial training."
        ),
        framework="gan",
        category="baseline",
        hyperparams={"total_iterations": 5000, "batch_size": 64, "shot_count": 10},
        adaptor_only=False,
        use_sim_guide=False,
        use_adv_noise=False,
        paper_ref="Table 2 (TGAN)",
    ),
    "ada": MethodConfig(
        method_id="ada",
        display_name="ADA",
        description=(
            "Adaptive Discriminator Augmentation (Karras et al. 2020): "
            "StyleGAN2 training with adaptive augmentation for limited data. "
            "p ← p + sign(r_t − r_target)·C_update."
        ),
        framework="gan",
        category="baseline",
        hyperparams={
            "total_iterations": 5000,
            "batch_size": 64,
            "shot_count": 10,
            "ada_target": 0.6,
            "ada_interval": 4,
        },
        adaptor_only=False,
        use_sim_guide=False,
        use_adv_noise=False,
        paper_ref="Table 2 (ADA)",
    ),
    # ------------------------------------------------------------------
    # CONTINUAL LEARNING / REGULARISATION BASELINES
    # ------------------------------------------------------------------
    "ewc": MethodConfig(
        method_id="ewc",
        display_name="EWC",
        description=(
            "Elastic Weight Consolidation (Kirkpatrick et al. 2017): "
            "L_total = L_simple + λ/2 · Σ_i F_i·(θ_i − θ*_i)². "
            "λ=1000, Fisher F estimated on source domain data."
        ),
        framework="ddpm",
        category="baseline",
        hyperparams={
            "total_iterations": 5000,
            "batch_size": 64,
            "ewc_lambda": 1000.0,
        },
        adaptor_only=False,
        use_sim_guide=False,
        use_adv_noise=False,
        paper_ref="Table 2 (EWC)",
    ),
    # ------------------------------------------------------------------
    # DOMAIN ADAPTATION / CORRESPONDENCE BASELINES
    # ------------------------------------------------------------------
    "cdc": MethodConfig(
        method_id="cdc",
        display_name="CDC",
        description=(
            "Cross-Domain Correspondence (Ojha et al. 2021): "
            "L_CDC = ||D_dist(G_T(z_i), G_T(z_j)) − D_dist(G_S(z_i), G_S(z_j))||². "
            "Preserves relative distances across domains."
        ),
        framework="gan",
        category="baseline",
        hyperparams={
            "total_iterations": 5000,
            "batch_size": 64,
            "shot_count": 10,
            "cdc_weight": 1.0,
        },
        adaptor_only=False,
        use_sim_guide=False,
        use_adv_noise=False,
        paper_ref="Table 2 (CDC)",
    ),
    "dcl": MethodConfig(
        method_id="dcl",
        display_name="DCL",
        description=(
            "Domain Consistent Loss: few-shot generative transfer with domain "
            "consistency constraints. L_DCL = ||φ(G_T(z)) − φ(G_S(z))||², "
            "φ = perceptual feature extractor."
        ),
        framework="gan",
        category="baseline",
        hyperparams={
            "total_iterations": 5000,
            "batch_size": 64,
            "shot_count": 10,
            "dcl_weight": 1.0,
        },
        adaptor_only=False,
        use_sim_guide=False,
        use_adv_noise=False,
        paper_ref="Table 2 (DCL)",
    ),
    # ------------------------------------------------------------------
    # ATTACK / OPTIMISATION PRIMITIVES
    # ------------------------------------------------------------------
    "pgd": MethodConfig(
        method_id="pgd",
        display_name="PGD",
        description=(
            "Projected Gradient Descent (Madry et al. 2018) inner-loop optimisation "
            "used in DPMs-ANT Adversarial Noise Selection. "
            "ε* = argmax_{||ε||_∞≤δ} L_simple(x_0+ε). "
            "K=10 steps, step-size omega=0.02."
        ),
        framework="general",
        category="attack",
        hyperparams={
            "adversarial_step_size": 0.02,
            "adversarial_inner_steps": 10,
        },
        is_pgd_based=True,
        paper_ref="Algorithm 1 Step 2 (PGD inner loop)",
    ),
    # ------------------------------------------------------------------
    # SAMPLERS
    # ------------------------------------------------------------------
    "ddim": MethodConfig(
        method_id="ddim",
        display_name="DDIM",
        description=(
            "Denoising Diffusion Implicit Models (Song et al. 2020): "
            "deterministic accelerated sampling, ~50 steps instead of 1000. "
            "Used for fast evaluation-time generation."
        ),
        framework="ddpm",
        category="sampler",
        hyperparams={"ddim_steps": 50, "ddim_eta": 0.0},
        paper_ref="Sampling (DDIM fast inference)",
    ),
    # ------------------------------------------------------------------
    # DATASET / METRIC SELECTORS (used in evaluation pipeline routing)
    # ------------------------------------------------------------------
    "ffhq": MethodConfig(
        method_id="ffhq",
        display_name="FFHQ",
        description=(
            "Flickr Faces High Quality (FFHQ) source domain: "
            "70K high-quality 256×256 face images. Pre-trained DDPM/LDM source."
        ),
        framework="ddpm",
        category="dataset",
        paper_ref="Source domain FFHQ (Table 2)",
    ),
    "lpips": MethodConfig(
        method_id="lpips",
        display_name="LPIPS",
        description=(
            "Learned Perceptual Image Patch Similarity (Zhang et al. 2018). "
            "Used as Intra-LPIPS diversity: mean pairwise LPIPS across generated images."
        ),
        framework="general",
        category="metric",
        paper_ref="Evaluation metric: Intra-LPIPS diversity",
    ),
}

# ============================================================================
# SECTION 4 – ABSTRACT BASE ADAPTER
# ============================================================================


class BaselineAdapter(ABC):
    """
    Abstract base class for all method / baseline training adapters.

    Subclasses implement:
      - build_model(pretrained_path, model_config) → model
      - compute_loss(batch, model, step) → (loss_tensor, metric_dict)

    The base class provides:
      - train(dataloader, n_steps, …) → history dict
      - generate_samples(n_samples, output_dir, …) → list of file paths
      - get_config_dict() → serialisable dict
    """

    def __init__(self, config: MethodConfig, device: str = "cpu") -> None:
        self.config = config
        self.device = device
        self.hyperparams = config.merged_hyperparams()
        self._model: Any = None
        self._optimizer: Any = None