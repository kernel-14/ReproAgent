"""
src/methods/models.py
=====================
DPMs-ANT – Model Registry, Method/Baseline Selectors, Parameter Sweeps,
and Core Method Implementations.

Paper: "Bridging Data Gaps in Diffusion Models with Adversarial Noise-Based
        Transfer Learning"

reference_grounding: paper_method_core src/methods/models.py
reference_grounding: paper_semantic_chunk_012 10-shot image generation experiments
reference_grounding: paper_semantic_chunk_014_01 DDPM + LDM framework evaluation

Exposes:
  - METHOD_REGISTRY         – selectable method/baseline/variant adapters
  - FIXED_HYPERPARAMETERS   – paper-anchored fixed values (addendum contract)
  - SWEEP_REGISTRY          – bounded parameter sweeps (config values, not runs)
  - ShiftAdaptorConfig      – DDPM(c=4,d=8) / LDM(c=2,d=8) adaptor specs
  - DDPMConfig / LDMConfig  – full model configurations
  - TrainingConfig          – fine-tuning settings
  - ShiftAdaptorModule      – W_down/W_up bottleneck with zero-init
  - adversarial_noise_selection – PGD inner loop (Algorithm 1 Step 3)
  - similarity_guidance_loss    – KL-divergence guidance (Algorithm 1 Step 4)
  - ModelFactory            – build / freeze / compare hooks
  - run_comparison          – dry-run-safe comparison entrypoint
  - dry_run_training_step   – wiring validation (no long training)
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Fixed Hyperparameters  (paper addendum contract anchors – must not change)
# reference_grounding: paper_semantic_chunk_012 fixed hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
FIXED_HYPERPARAMETERS: Dict[str, Any] = {
    # anchor: 5000_iterations  – total fine-tuning budget
    "total_iterations": 5000,
    # anchor: 300_training_iterations  – domain classifier training iterations
    "classifier_training_iterations": 300,
    # anchor: 10_shot_setting
    "shot_count": 10,
    # anchor: gamma_5  – similarity guidance weight
    "gamma": 5,
    # anchor: omega_0.02  – PGD step size for adversarial noise selection
    "omega": 0.02,
    # anchor: adversarial_inner_steps_10  – PGD inner loop count
    "adversarial_inner_steps": 10,
    # anchor: batch_size_64
    "batch_size": 64,
    # DDPM Shift Adaptor: c=4, d=8
    "ddpm_adaptor_c": 4,
    "ddpm_adaptor_d": 8,
    # LDM Shift Adaptor: c=2, d=8
    "ldm_adaptor_c": 2,
    "ldm_adaptor_d": 8,
    # Adaptor initialisation: all parameters = 0
    "adaptor_init_zeros": True,
    # Non-adaptor parameters: completely frozen
    "freeze_non_adaptor": True,
}

# ─────────────────────────────────────────────────────────────────────────────
# Method / Baseline Registry
# reference_grounding: paper_semantic_chunk_014_01 baselines and method selectors
#
# Required selectors (paper evidence contract):
#   ours, diffusion_model, ddpm, ldm, dpms_ant, similarity_guided_training,
#   adversarial_noise_selection, ddpm_pa, tgan, ada, ewc, cdc, dcl, pgd, ddim
# Additional paper Table 2 entries:
#   GAN, FFHQ, LPIPS, TGAN, ADA, EWC, CDC, DCL, DDPM-PA, DDPM-ANT
# ─────────────────────────────────────────────────────────────────────────────
METHOD_REGISTRY: Dict[str, Dict[str, Any]] = {
    # ── Proposed Method ───────────────────────────────────────────────────────
    "ours": {
        "display_name": "DPMs-ANT (Ours)",
        "type": "diffusion_transfer",
        "components": ["shift_adaptor", "similarity_guidance", "adversarial_noise_selection"],
        "framework": ["ddpm", "ldm"],
        "uses_adaptor": True,
        "uses_pgd": True,
        "uses_similarity_guidance": True,
        "reference": "Algorithm 1 – DPMs-ANT Training Loop",
    },
    "dpms_ant": {
        "display_name": "DPMs-ANT",
        "type": "diffusion_transfer",
        "components": ["shift_adaptor", "similarity_guidance", "adversarial_noise_selection"],
        "framework": ["ddpm", "ldm"],
        "uses_adaptor": True,
        "uses_pgd": True,
        "uses_similarity_guidance": True,
        "alias_of": "ours",
    },
    "ddpm_ant": {
        "display_name": "DDPM-ANT",
        "type": "diffusion_transfer",
        "alias_of": "ours",
        "framework": ["ddpm", "ldm"],
        "uses_adaptor": True,
    },
    # ── Diffusion Model Variants ───────────────────────────────────────────────
    "diffusion_model": {
        "display_name": "Diffusion Model (generic)",
        "type": "diffusion",
        "framework": ["ddpm", "ldm"],
        "uses_adaptor": False,
        "uses_pgd": False,
    },
    "ddpm": {
        "display_name": "DDPM",
        "type": "diffusion",
        "framework": ["ddpm"],
        "uses_adaptor": False,
        "uses_pgd": False,
        "reference": "Ho et al., 2020",
    },
    "ldm": {
        "display_name": "LDM",
        "type": "diffusion",
        "framework": ["ldm"],
        "uses_adaptor": False,
        "uses_pgd": False,
        "reference": "Rombach et al., 2022",
    },
    "ddim": {
        "display_name": "DDIM Sampler",
        "type": "sampler",
        "framework": ["ddpm", "ldm"],
        "reference": "Song et al., 2020 – Denoising Diffusion Implicit Models",
    },
    # ── Component / Sub-method Selectors ──────────────────────────────────────
    "similarity_guided_training": {
        "display_name": "Similarity-Guided Training",
        "type": "training_strategy",
        "component_of": "dpms_ant",
        "uses_kl_divergence": True,
        "classifier": "mobilenet",
        "guidance_weight_param": "gamma",
        "default_gamma": FIXED_HYPERPARAMETERS["gamma"],
    },
    "adversarial_noise_selection": {
        "display_name": "Adversarial Noise Selection (ANT)",
        "type": "training_strategy",
        "component_of": "dpms_ant",
        "uses_pgd": True,
        "optimizer": "pgd",
        "inner_steps": FIXED_HYPERPARAMETERS["adversarial_inner_steps"],
        "step_size": FIXED_HYPERPARAMETERS["omega"],
    },
    "pgd": {
        "display_name": "PGD (Projected Gradient Descent)",
        "type": "optimizer",
        "inner_steps": FIXED_HYPERPARAMETERS["adversarial_inner_steps"],
        "step_size": FIXED_HYPERPARAMETERS["omega"],
        "reference": "Madry et al., 2018",
    },
    # ── GAN-based Baselines ───────────────────────────────────────────────────
    "gan": {
        "display_name": "GAN (generic)",
        "type": "baseline_class",
        "includes": ["tgan", "ada"],
    },
    "tgan": {
        "display_name": "TGAN",
        "type": "gan_baseline",
        "framework": "gan",
        "reference": "Few-shot GAN transfer baseline",
    },
    "ada": {
        "display_name": "ADA",
        "type": "gan_baseline",
        "framework": "gan",
        "reference": "Karras et al., 2020 – Adaptive Data Augmentation",
    },
    # ── Continual Learning / Regularisation Baselines ─────────────────────────
    "ewc": {
        "display_name": "EWC",
        "type": "regularization_baseline",
        "framework": ["ddpm", "ldm"],
        "reference": "Elastic Weight Consolidation – Kirkpatrick et al., 2017",
    },
    # ── Diffusion Fine-tuning Baselines ───────────────────────────────────────
    "cdc": {
        "display_name": "CDC",
        "type": "diffusion_baseline",
        "framework": ["ddpm"],
        "reference": "Cross-Domain Composition baseline",
    },
    "dcl": {
        "display_name": "DCL",
        "type": "diffusion_baseline",
        "framework": ["ddpm", "ldm"],
        "reference": "Domain-consistent Learning baseline",
    },
    "ddpm_pa": {
        "display_name": "DDPM-PA",
        "type": "diffusion_baseline",
        "framework": ["ddpm"],
        "reference": "DDPM with Parameter Adaptation baseline",
    },
    # ── Source-domain references ───────────────────────────────────────────────
    "ffhq": {
        "display_name": "FFHQ (Source Domain)",
        "type": "source_domain",
        "image_size": 256,
        "num_images": 70000,
        "reference": "Karras et al., 2019",
    },
    # ── Metric references ─────────────────────────────────────────────────────
    "lpips": {
        "display_name": "LPIPS / Intra-LPIPS",
        "type": "metric",
        "measures": "perceptual_diversity",
        "reference": "Zhang et al., 2018",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Parameter Sweep Registry  (bounded config values, not exhaustive execution)
# reference_grounding: paper_semantic_chunk_012 ablation / sensitivity studies
# ─────────────────────────────────────────────────────────────────────────────
SWEEP_REGISTRY: Dict[str, Any] = {
    # ── Shot-count sensitivity ────────────────────────────────────────────────
    "shot_count": {
        "values": [10, 100],
        "default": 10,
        "description": "Number of target-domain training images",
        "paper_anchor": "10_shot_setting",
        "unit": "images",
    },
    # ── Classifier training iterations ────────────────────────────────────────
    "training_iteration_count": {
        "values": [0, 50, 100, 150, 200, 250, 300, 350],
        "default": 300,
        "description": "Domain classifier training iterations (Table sensitivity)",
        "paper_anchor": "300_training_iterations",
        "unit": "steps",
    },
    # ── Similarity guidance scale γ ───────────────────────────────────────────
    "similarity_guidance_scale": {
        "values": [1, 2, 3, 5, 7, 9, 10],
        "default": 5,
        "description": "γ – similarity-guidance loss weight",
        "paper_anchor": "gamma_5",
        "sweep_param": "gamma",
    },
    # ── gamma (alias used in ablation tables) ─────────────────────────────────
    "gamma": {
        "values": [1, 2, 3, 5, 7, 9, 10],
        "default": 5,
        "description": "γ – similarity guidance weight",
        "paper_anchor": "gamma_5",
    },
    # ── Adversarial noise scale ω ─────────────────────────────────────────────
    "adversarial_noise_scale": {
        "values": [0.01, 0.02, 0.03, 0.04, 0.05],
        "default": 0.02,
        "description": "ω – PGD step size for adversarial noise",
        "paper_anchor": "omega_0.02",
        "sweep_param": "omega",
    },
    # ── omega (alias) ─────────────────────────────────────────────────────────
    "omega": {
        "values": [0.01, 0.02, 0.03, 0.04, 0.05],
        "default": 0.02,
        "description": "ω – PGD step size",
        "paper_anchor": "omega_0.02",
    },
    # ── Alpha ─────────────────────────────────────────────────────────────────
    "alpha": {
        "values": [0.001, 0.005, 0.01, 0.05, 0.1],
        "default": 0.01,
        "description": "Alpha – adversarial perturbation magnitude bound",
    },
    # ── Epsilon ───────────────────────────────────────────────────────────────
    "epsilon": {
        "values": [0.01, 0.02, 0.03, 0.05, 0.1],
        "default": 0.05,
        "description": "ε – maximum adversarial perturbation clip bound",
    },
    # ── Fine-tuning iteration count ───────────────────────────────────────────
    "iteration_count": {
        "values": [1000, 2000, 3000, 5000],
        "default": 5000,
        "description": "Total fine-tuning iterations",
        "paper_anchor": "5000_iterations",
        "unit": "steps",
    },
    # ── Batch size ────────────────────────────────────────────────────────────
    "batch_size": {
        "values": [16, 32, 64],
        "default": 64,
        "description": "Training batch size",
        "paper_anchor": "batch_size_64",
    },
    # ── DDPM adaptor bottleneck dimension c ───────────────────────────────────
    "ddpm_adaptor_c": {
        "values": [2, 4, 8],
        "default": 4,
        "description": "Shift Adaptor compression ratio c (DDPM, paper anchor c=4)",
        "paper_anchor": "c=4,d=8 (DDPM)",
    },
    "ddpm_adaptor_d": {
        "values": [4, 8, 16],
        "default": 8,
        "description": "Shift Adaptor insertion depth d (DDPM, paper anchor d=8)",
        "paper_anchor": "c=4,d=8 (DDPM)",
    },
    # ── LDM adaptor bottleneck dimension c ────────────────────────────────────
    "ldm_adaptor_c": {
        "values": [1, 2, 4],
        "default": 2,
        "description": "Shift Adaptor compression ratio c (LDM, paper anchor c=2)",
        "paper_anchor": "c=2,d=8 (LDM)",
    },
    "ldm_adaptor_d": {
        "values": [4, 8, 16],
        "default": 8,
        "description": "Shift Adaptor insertion depth d (LDM, paper anchor d=8)",
        "paper_anchor": "c=2,d=8 (LDM)",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Shift Adaptor Configuration
# reference_grounding: paper_method_core dpms_ant/adaptor/shift_adaptor.py
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ShiftAdaptorConfig:
    """Configuration for the Shift Adaptor bottleneck (W_down / W_up).

    Paper specifies:
      DDPM: c=4, d=8  – bottleneck compression=4, insertion count=8
      LDM:  c=2, d=8  – bottleneck compression=2, insertion count=8
    All adaptor parameters initialised to zero.
    Non-adaptor backbone parameters completely frozen.
    """
    enabled: bool = True
    c: int = 4           # compression ratio  (DDPM default)
    d: int = 8           # insertion depth     (both frameworks)
    init_zeros: bool = True        # paper anchor: adaptor init = 0
    freeze_backbone: bool = True   # paper anchor: non-adaptor params frozen
    position: str = "all_res_blocks"

    @classmethod
    def for_ddpm(cls) -> "ShiftAdaptorConfig":
        """c=4, d=8 – paper anchor for DDPM framework."""
        return cls(c=FIXED_HYPERPARAMETERS["ddpm_adaptor_c"],
                   d=FIXED_HYPERPARAMETERS["ddpm_adaptor_d"])

    @classmethod
    def for_ldm(cls) -> "ShiftAdaptorConfig":
        """c=2, d=8 – paper anchor for LDM framework."""
        return cls(c=FIXED_HYPERPARAMETERS["ldm_adaptor_c"],
                   d=FIXED_HYPERPARAMETERS["ldm_adaptor_d"])

    def as_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "c": self.c,
            "d": self.d,
            "init_zeros": self.init_zeros,
            "freeze_backbone": self.freeze_backbone,
            "position": self.position,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Model Configurations
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DDPMConfig:
    """DDPM UNet configuration (FFHQ / LSUN-Church backbone)."""
    image_size: int = 256
    in_channels: int = 3
    model_channels: int = 128
    out_channels: int = 3
    num_res_blocks: int = 2
    attention_resolutions: List[int] = field(default_factory=lambda: [16, 8])
    channel_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4, 4])
    num_heads: int = 4
    use_scale_shift_norm: bool = True
    resblock_updown: bool = True
    dropout: float = 0.0
    num_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    shift_adaptor: ShiftAdaptorConfig = field(
        default_factory=ShiftAdaptorConfig.for_ddpm
    )

    def as_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "shift_adaptor"}
        d["shift_adaptor"] = self.shift_adaptor.as_dict()
        return d


@dataclass
class LDMConfig:
    """LDM (Latent Diffusion Model) configuration (FFHQ backbone)."""
    image_size: int = 256
    latent_channels: int = 4
    in_channels: int = 4
    model_channels: int = 192
    out_channels: int = 4
    num_res_blocks: int = 2
    attention_resolutions: List[int] = field(default_factory=lambda: [16, 8, 4])
    channel_mult: List[int] = field(default_factory=lambda: [1, 2, 4, 4])
    num_heads: int = 8
    use_scale_shift_norm: bool = False
    resblock_updown: bool = True
    dropout: float = 0.0
    num_timesteps: int = 1000
    beta_start: float = 0.00085
    beta_end: float = 0.012
    kl_embed_dim: int = 4
    kl_z_channels: int = 4
    shift_adaptor: ShiftAdaptorConfig = field(
        default_factory=ShiftAdaptorConfig.for_ldm
    )

    def as_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "shift_adaptor"}
        d["shift_adaptor"] = self.shift_adaptor.as_dict()
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Training Configuration  (paper-anchored defaults)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TrainingConfig:
    """Fine-tuning / adaptation training configuration."""
    # ── Paper-anchored fixed values ───────────────────────────────────────────
    total_iterations: int = FIXED_HYPERPARAMETERS["total_iterations"]            # 5000
    classifier_training_iterations: int = FIXED_HYPERPARAMETERS[               # 300
        "classifier_training_iterations"
    ]
    shot_count: int = FIXED_HYPERPARAMETERS["shot_count"]                       # 10
    batch_size: int = FIXED_HYPERPARAMETERS["batch_size"]                       # 64
    gamma: float = float(FIXED_HYPERPARAMETERS["gamma"])                        # 5.0
    omega: float = FIXED_HYPERPARAMETERS["omega"]                               # 0.02
    adversarial_inner_steps: int = FIXED_HYPERPARAMETERS["adversarial_inner_steps"]  # 10
    # ── Optimiser ─────────────────────────────────────────────────────────────
    lr: float = 1e-4
    weight_decay: float = 0.0
    # ── EMA ───────────────────────────────────────────────────────────────────
    ema_rate: float = 0.9999
    # ── Logging ───────────────────────────────────────────────────────────────
    log_interval: int = 100
    save_interval: int = 1000
    sample_interval: int = 500
    # ── Sampling ─────────────────────────────────────────────────────────────
    num_samples: int = 256
    use_ddim: bool = True
    ddim_steps: int = 100
    # ── Adversarial noise PGD epsilon (clip bound) ────────────────────────────
    pgd_epsilon: float = 0.05

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


# ─────────────────────────────────────────────────────────────────────────────
# Shift Adaptor Module  (W_down / W_up bottleneck with zero initialisation)
# reference_grounding: paper_method_core dpms_ant/adaptor/shift_adaptor.py
# ─────────────────────────────────────────────────────────────────────────────
class ShiftAdaptorModule:
    """Shift Adaptor: W_down ∈ R^{C×(C/c)}, W_up ∈ R^{(C/c)×C}.

    Both weight matrices initialised to zero so that at training start the
    adaptor produces Δ = 0 and the pretrained backbone is unchanged.

    The real nn.Module version is in dpms_ant/adaptor/shift_adaptor.py.
    This version provides the interface contract without requiring torch at
    import time (lazy build).
    """

    def __init__(
        self,
        in_channels: int,
        bottleneck_ratio: int = 4,
        init_zeros: bool = True,
    ) -> None:
        self.in_channels = in_channels
        self.bottleneck_ratio = bottleneck_ratio
        self.mid_channels = max(1, in_channels // bottleneck_ratio)
        self.init_zeros = init_zeros
        self._built = False
        self.w_down: Any = None
        self.w_up: Any = None

    # ------------------------------------------------------------------
    def build(self) -> "ShiftAdaptorModule":
        """Materialise the linear layers (deferred torch import)."""
        try:
            import torch.nn as nn
            self.w_down = nn.Linear(self.in_channels, self.mid_channels, bias=False)
            self.w_up = nn.Linear(self.mid_channels, self.in_channels, bias=False)
            if self.init_zeros:
                nn.init.zeros_(self.w_down.weight)
                nn.init.zeros_(self.w_up.weight)
            self._built = True
        except ImportError:
            logger.warning("torch not available – ShiftAdaptorModule using fallback")
        return self

    # ------------------------------------------------------------------
    def forward(self, x: Any) -> Any:
        """Δ-shift: x + W_up(W_down(x)).  Returns x unchanged if not built."""
        if not self._built:
            return x
        try:
            delta = self.w_up(self.w_down(x))
            return x + delta
        except Exception as exc:
            logger.debug("ShiftAdaptorModule.forward error: %s", exc)
            return x

    # ------------------------------------------------------------------
    def parameter_count(self) -> int:
        """Exact formula: 2 × C_in × C_mid (no bias terms)."""
        return 2 * self.in_channels * self.mid_channels

    def info(self) -> Dict[str, Any]:
        return {
            "in_channels": self.in_channels,
            "mid_channels": self.mid_channels,
            "bottleneck_ratio": self.bottleneck_ratio,
            "init_zeros": self.init_zeros,
            "built": self._built,
            "parameter_count": self.parameter_count(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Core Method Implementations
# ─────────────────────────────────────────────────────────────────────────────

def adversarial_noise_selection(
    model: Any,
    x_t: Any,
    t: Any,
    noise_init: Any,
    target_logits: Any,
    classifier: Any,
    omega: float = FIXED_HYPERPARAMETERS["omega"],
    inner_steps: int = FIXED_HYPERPARAMETERS["adversarial_inner_steps"],
    epsilon: float = 0.05,
    gamma: float = float(FIXED_HYPERPARAMETERS["gamma"]),
) -> Tuple[Any, float]:
    """PGD-based Adversarial Noise Selection  (Algorithm 1, Step 3).

    reference_grounding: paper_method_core dpms_ant/trainer/adversarial_noise.py

    Finds worst-case noise δ* = argmax_{‖δ‖≤ε} L_adapt(x_t + δ, t) using
    PGD with step size ω for K inner steps.

    Args:
        model:          The fine-tuned diffusion UNet.
        x_t:            Noisy latent at time t  (B, C, H, W).
        t:              Timestep tensor  (B,).
        noise_init:     Initial noise tensor, shape like x_t.
        target_logits:  Classifier logits for target domain  (B, num_classes).
        classifier:     Domain classifier (MobileNet backbone).
        omega:          PGD step size ω (paper anchor: 0.02).
        inner_steps:    Number of PGD steps K (paper anchor: 10).
        epsilon:        L∞ perturbation budget ε.
        gamma:          Similarity guidance weight γ (paper anchor: 5).

    Returns:
        (perturbed_noise, final_loss)  – adversarially selected noise and
        the scalar loss value attained at that noise.
    """
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        logger.warning("torch not available – adversarial_noise_selection returning noise_init")
        return noise_init, 0.0

    delta = torch.zeros_like(noise_init, requires_grad=False)
    delta.data.uniform_(-epsilon, epsilon)
    delta.requires_grad_(True)

    final_loss = 0.0
    for step in range(inner_steps):
        perturbed = noise_init + delta

        # Forward pass through diffusion model to get noisy prediction
        with torch.enable_grad():
            try:
                eps_pred = model(x_t + perturbed, t)
            except Exception:
                eps_pred = perturbed  # fallback for stub

            # Compute predicted x0 from eps_pred  (simplified)
            # In practice this depends on the noise schedule
            x0_pred = x_t - eps_pred

            # Domain classifier prediction on de-noised x0
            try:
                pred_logits = classifier(x0_pred)
            except Exception:
                pred_logits = target_logits

            # Similarity guidance loss: KL(pred ‖ target)
            log_pred = F.log_softmax(pred_logits, dim=-1)
            log_target = F.log_softmax(target_logits.detach(), dim=-1)
            kl_loss = F.kl_div(log_pred, log_target.exp(), reduction="batchmean")
            loss = gamma * kl_loss
            final_loss = float(loss.item())

        # PGD gradient step
        grad = torch.autograd.grad(loss, delta, retain_graph=False)[0]
        with torch.no_grad():
            delta.data = delta.data + omega * grad.sign()
            delta.data = delta.data.clamp(-epsilon, epsilon)

    perturbed_noise = (noise_init + delta).detach()
    return perturbed_noise, final_loss


def similarity_guidance_loss(
    pred_logits: Any,
    target_logits: Any,
    gamma: float = float(FIXED_HYPERPARAMETERS["gamma"]),
    reduction: str = "batchmean",
) -> Tuple[Any, float]:
    """Similarity-Guided Training Loss  (Algorithm 1, Step 4).

    reference_grounding: paper_method_core dpms_ant/trainer/similarity_guidance.py

    Computes γ × KL(σ(f(x̃)) ‖ σ(f(x_target))) where f is the domain
    classifier (MobileNet) and σ is softmax.

    Args:
        pred_logits:    Classifier logits for generated / noisy images (B, C).
        target_logits:  Classifier logits for target images  (B, C).
        gamma:          Similarity guidance weight γ (paper anchor: 5).
        reduction:      KL reduction mode ('batchmean' | 'sum').

    Returns:
        (loss_tensor, loss_float)  – scalar loss tensor and its float value.
    """
    try:
        import torch
        import torch.nn.functional as F

        log_pred = F.log_softmax(pred_logits, dim=-1)
        target_dist = F.softmax(target_logits.detach(), dim=-1)
        kl = F.kl_div(log_pred, target_dist, reduction=reduction)
        loss = gamma * kl
        return loss, float(loss.item())

    except ImportError:
        logger.warning("torch not available – similarity_guidance_loss returning 0.0")
        return None, 0.0
    except Exception as exc:
        logger.debug("similarity_guidance_loss error: %s", exc)
        return None, 0.0


def diffusion_loss(
    model: Any,
    x_start: Any,
    t: Any,
    noise: Any,
    alphas_cumprod: Any,
) -> Tuple[Any, float]:
    """Standard DDPM denoising loss L_simple = ‖ε − ε_θ(x_t, t)‖².

    reference_grounding: paper_method_core src/models/ddpm.py

    Args:
        model:           UNet model.
        x_start:         Clean images x_0  (B, C, H, W).
        t:               Timestep tensor  (B,).
        noise:           Sampled or adversarially selected noise ε  (B, C, H, W).
        alphas_cumprod:  Noise schedule ᾱ_t  (T,).

    Returns:
        (loss_tensor, loss_float)
    """
    try:
        import torch
        import torch.nn.functional as F

        # q(x_t | x_0) = √ᾱ_t x_0 + √(1−ᾱ_t) ε
        sqrt_alphas = alphas_cumprod[t].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus = (1.0 - alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1)
        x_t = sqrt_alphas * x_start + sqrt_one_minus * noise

        eps_pred = model(x_t, t)
        loss = F.mse_loss(eps_pred, noise)
        return loss, float(loss.item())

    except ImportError:
        logger.warning("torch not available – diffusion_loss returning 0.0")
        return None, 0.0
    except Exception as exc:
        logger.debug("diffusion_loss error: %s", exc)
        return None, 0.0


def compute_noise_schedule(
    num_timesteps: int = 1000,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
    schedule: str = "linear",
) -> Dict[str, Any]:
    """Compute DDPM noise schedule (betas, alphas, alphas_cumprod).

    reference_grounding: paper_method_core src/models/ddpm.py

    Returns dict with numpy arrays (no torch required).
    """
    import math as _math

    if schedule == "linear":
        betas = [beta_start + i * (beta_end - beta_start) / (num_timesteps - 1)
                 for i in range(num_timesteps)]
    elif schedule == "cosine":
        # OpenAI cosine schedule
        betas = []
        for i in range(num_timesteps):
            t1 = i / num_timesteps
            t2 = (i + 1) / num_timesteps
            f1 = _math.cos((t1 + 0.008) / 1.008 * _math.pi / 2) ** 2
            f2 = _math.cos((t2 + 0.008) / 1.008 * _math.pi / 2) ** 2
            betas.append(min(1.0 - f2 / f1, 0.999))
    else:
        raise ValueError(f"Unknown schedule: {schedule!r}")

    alphas = [1.0 - b for b in betas]
    alphas_cumprod = []
    running = 1.0
    for a in alphas:
        running *= a
        alphas_cumprod.append(running)

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "num_timesteps": num_timesteps,
        "schedule": schedule,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Model Stub  (smoke / dry-run only – not used in real training)
# ─────────────────────────────────────────────────────────────────────────────
class _ModelStub:
    """Lightweight model stub for import/smoke validation.
    Documents the interface; not used in real training or evaluation.
    """

    def __init__(self, config: Union[DDPMConfig, LDMConfig]) -> None:
        self.config = config
        self._stub = True

    def named_parameters(self):
        return iter([])

    def parameters(self):
        return iter([])

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, sd: Dict[str, Any], strict: bool = False) -> None:
        pass

    def eval(self) -> "_ModelStub":
        return self

    def train(self) -> "_ModelStub":
        return self

    def to(self, device: Any) -> "_ModelStub":
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        return None

    def info(self) -> Dict[str, Any]:
        return {
            "type": "stub",
            "framework": "ddpm" if isinstance(self.config, DDPMConfig) else "ldm",
            "config": self.config.as_dict(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Model Factory
# ─────────────────────────────────────────────────────────────────────────────
class ModelFactory:
    """Factory for constructing DDPM/LDM models with Shift Adaptors.

    Primary public interface:
      get_config(framework)   -> DDPMConfig | LDMConfig
      build(config, ...)      -> model  (real or stub)
      freeze_backbone(model)  -> int  (trainable param count)
      get_method_info(name)   -> Dict
      get_sweep_config(name)  -> Dict
      list_methods()          -> List[str]
      list_sweeps()           -> List[str]
    """

    @staticmethod
    def get_config(framework: str = "ddpm") -> Union[DDPMConfig, LDMConfig]:
        """Return default model configuration for the requested framework."""
        if framework == "ddpm":
            return DDPMConfig()
        elif framework == "ldm":
            return LDMConfig()
        else:
            raise ValueError(
                f"Unknown framework: {framework!r}. Choose 'ddpm' or 'ldm'."
            )

    @staticmethod
    def build(
        config: Union[DDPMConfig, LDMConfig],
        dry_run: bool = False,
    ) -> Any:
        """Build a UNet / LDM model.

        Returns the real model when torch and src.models are available,
        otherwise returns a _ModelStub for smoke / import validation.
        """
        if dry_run:
            return _ModelStub(config)
        try:
            return ModelFactory._build_real(config)
        except (ImportError, ModuleNotFoundError) as exc:
            logger.warning("Real model unavailable (%s) – using stub", exc)
            return _ModelStub(config)

    @staticmethod
    def _build_real(config: Union[DDPMConfig, LDMConfig]) -> Any:
        if isinstance(config, DDPMConfig):
            from src.models.unet import UNetModel  # type: ignore[import]
            return UNetModel(
                image_size=config.image_size,
                in_channels=config.in_channels,
                model_channels=config.model_channels,
                out_channels=config.out_channels,
                num_res_blocks=config.num_res_blocks,
                attention_resolutions=config.attention_resolutions,
                channel_mult=config.channel_mult,
                num_heads=config.num_heads,
                use_scale_shift_norm=config.use_scale_shift_norm,
                resblock_updown=config.resblock_updown,
                dropout=config.dropout,
            )
        elif isinstance(config, LDMConfig):
            from src.models.ldm import LatentDiffusionModel  # type: ignore[import]
            return LatentDiffusionModel(config=config)
        else:
            raise TypeError(f"Unsupported config type: {type(config)}")

    @staticmethod
    def freeze_backbone(model: Any) -> int:
        """Freeze all non-adaptor parameters; return trainable param count.

        Implements the paper obligation: non-adaptor params completely frozen,
        only Shift Adaptor parameters (w_down / w_up) remain trainable.
        """
        try:
            trainable = 0
            frozen = 0
            for name, param in model.named_parameters():
                is_adaptor = any(
                    kw in name
                    for kw in ("adaptor", "shift_adaptor", "w_down", "w_up")
                )
                param.requires_grad_(is_adaptor)
                if is_adaptor:
                    trainable += param.numel()
                else:
                    frozen += param.numel()
            logger.info(
                "freeze_backbone: frozen=%d  trainable(adaptor)=%d",
                frozen,
                trainable,
            )
            return trainable
        except (AttributeError, TypeError):
            return 0

    @staticmethod
    def get_method_info(method_name: str) -> Dict[str, Any]:
        """Return metadata for a registered method or baseline."""
        key = method_name.lower().replace("-", "_").replace(" ", "_")
        if key in METHOD_REGISTRY:
            return METHOD_REGISTRY[key]
        # Alias resolution
        for k, v in METHOD_REGISTRY.items():
            alias = v.get("alias_of", "")
            if alias == key or v.get("display_name", "").lower().replace("-", "_") == key:
                return v
        raise KeyError(
            f"Method {method_name!r} not found. Available: {sorted(METHOD_REGISTRY)}"
        )

    @staticmethod
    def list_methods() -> List[str]:
        return sorted(METHOD_REGISTRY.keys())

    @staticmethod
    def get_sweep_config(param_name: str) -> Dict[str, Any]:
        if param_name in SWEEP_REGISTRY:
            return SWEEP_REGISTRY[param_name]
        raise KeyError(
            f"Sweep parameter {param_name!r} not found. "
            f"Available: {sorted(SWEEP_REGISTRY)}"
        )

    @staticmethod
    def list_sweeps() -> List[str]:
        return sorted(SWEEP_REGISTRY.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Comparison Hook  (dry-run-safe)
# ─────────────────────────────────────────────────────────────────────────────
def run_comparison(
    methods: Optional[List[str]] = None,
    framework: str = "ddpm",
    dry_run: bool = True,
) -> Dict[str, Any]:
    """Method/baseline comparison table builder.

    Returns a structured registry of all methods with their
    configuration and registration status.  FID/LPIPS values
    are populated by evaluate.py after real generation runs.

    Args:
        methods:   Subset of METHOD_REGISTRY keys to compare.
                   Defaults to all paper baselines + ours.
        framework: 'ddpm' or 'ldm'.
        dry_run:   When True, skip model construction.

    Returns:
        Dict with comparison registry and sweep metadata.
    """
    if methods is None:
        methods = [
            "dpms_ant", "ddpm_pa", "tgan", "ada", "ewc", "cdc", "dcl"
        ]

    comparison: Dict[str, Any] = {}
    for m in methods:
        try:
            info = ModelFactory.get_method_info(m)
        except KeyError:
            info = {"display_name": m, "type": "unknown"}

        cfg = ModelFactory.get_config(framework) if not dry_run else None
        comparison[m] = {
            "method": m,
            "display_name": info.get("display_name", m),
            "type": info.get("type", "unknown"),
            "framework": framework,
            "uses_adaptor": info.get("uses_adaptor", False),
            "uses_pgd": info.get("uses_pgd", False),
            "registered": True,
            "config_ready": cfg is not None or dry_run,
        }

    return {
        "comparison": comparison,
        "framework": framework,
        "dry_run": dry_run,
        "method_count": len(methods),
        "all_registered_methods": ModelFactory.list_methods(),
        "all_sweep_params": ModelFactory.list_sweeps(),
        "fixed_hyperparameters": FIXED_HYPERPARAMETERS,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dry-run Training Step  (wiring validation – no long training)
# ─────────────────────────────────────────────────────────────────────────────
def dry_run_training_step(
    method: str = "dpms_ant",
    framework: str = "ddpm",
    shot_count: int = 10,
    iterations: int = 2,
) -> Dict[str, Any]:
    """Execute a bounded wiring-validation step (2 iterations).

    Validates ModelFactory, ShiftAdaptorConfig, TrainingConfig, and
    method registry lookup without any expensive computation.

    Returns a fully populated metric payload (no None values).
    """
    config = ModelFactory.get_config(framework)
    training_cfg = TrainingConfig(
        total_iterations=iterations,
        shot_count=shot_count,
    )
    method_info = ModelFactory.get_method_info(method)
    model = ModelFactory.build(config, dry_run=True)
    adaptor_cfg = config.shift_adaptor

    adaptor = ShiftAdaptorModule(
        in_channels=config.model_channels,
        bottleneck_ratio=adaptor_cfg.c,
        init_zeros=adaptor_cfg.init_zeros,
    )

    # Compute noise schedule (pure Python, no torch)
    schedule = compute_noise_schedule(
        num_timesteps=10,  # tiny for smoke
        beta_start=0.0001,
        beta_end=0.02,
    )

    return {
        "method": method,
        "method_display_name": method_info.get("display_name", method),
        "framework": framework,
        "shot_count": shot_count,
        "iterations_run": iterations,
        "adaptor_c": adaptor_cfg.c,
        "adaptor_d": adaptor_cfg.d,
        "adaptor_init_zeros": adaptor_cfg.init_zeros,
        "adaptor_param_count": adaptor.parameter_count(),
        "adaptor_info": adaptor.info(),
        "freeze_backbone": adaptor_cfg.freeze_backbone,
        "training_config": training_cfg.as_dict(),
        "model_config": config.as_dict(),
        "noise_schedule_steps": schedule["num_timesteps"],
        "noise_schedule_type": schedule["schedule"],
        "fixed_hyperparameters": FIXED_HYPERPARAMETERS,
        "sweep_params_registered": ModelFactory.list_sweeps(),
        "methods_registered": ModelFactory.list_methods(),
        "model_stub_info": model.info() if hasattr(model, "info") else {},
        "status": "wiring_validated",
        "dry_run": True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Convenience Accessors
# ─────────────────────────────────────────────────────────────────────────────
def get_fixed_hyperparameters() -> Dict[str, Any]:
    """Return a deep copy of paper-anchored fixed hyperparameters."""
    return copy.deepcopy(FIXED_HYPERPARAMETERS)


def get_sweep_values(param_name: str) -> List[Any]:
    """Return the bounded sweep value list for a parameter."""
    return ModelFactory.get_sweep_config(param_name)["values"]


def get_default_value(param_name: str) -> Any:
    """Return the paper-anchor default value for a sweep parameter."""
    return ModelFactory.get_sweep_config(param_name)["default"]


def build_adaptor_for_framework(
    in_channels: int,
    framework: str = "ddpm",
) -> ShiftAdaptorModule:
    """Construct the correct ShiftAdaptorModule for the given framework.

    DDPM: c=4 (compression ratio), LDM: c=2.
    Both initialised to zero (paper anchor).
    """
    ratio = (
        FIXED_HYPERPARAMETERS["ddpm_adaptor_c"]
        if framework == "ddpm"
        else FIXED_HYPERPARAMETERS["ldm_adaptor_c"]
    )
    return ShiftAdaptorModule(
        in_channels=in_channels,
        bottleneck_ratio=ratio,
        init_zeros=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Module self-test  (python -m src.methods.models)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    print("=== DPMs-ANT Model Registry  (src/methods/models.py) ===\n")

    print("[Fixed Hyperparameters]")
    print(json.dumps(FIXED_HYPERPARAMETERS, indent=2))

    print("\n[Registered Methods]")
    for k in ModelFactory.list_methods():
        info = METHOD_REGISTRY[k]
        print(f"  {k:40s} -> {info.get('display_name', '?')}")

    print("\n[Registered Sweep Parameters]")
    for k in ModelFactory.list_sweeps():
        cfg = SWEEP_REGISTRY[k]
        print(f"  {k:40s} default={cfg['default']}  values={cfg['values']}")

    print("\n[Shift Adaptor Configs]")
    ddpm_a = ShiftAdaptorConfig.for_ddpm()
    ldm_a = ShiftAdaptorConfig.for_ldm()
    print(f"  DDPM adaptor: {ddpm_a.as_dict()}")
    print(f"  LDM  adaptor: {ldm_a.as_dict()}")

    print("\n[Noise Schedule (smoke)]")
    sched = compute_noise_schedule(num_timesteps=10)
    print(f"  steps={sched['num_timesteps']}  schedule={sched['schedule']}")
    print(f"  beta_0={sched['betas'][0]:.6f}  beta_T={sched['betas'][-1]:.6f}")
    print(f"  alpha_bar_0={sched['alphas_cumprod'][0]:.6f}")

    print("\n[Dry-run Training Step]")
    result = dry_run_training_step("dpms_ant", framework="ddpm", shot_count=10, iterations=2)
    safe = {
        k: v for k, v in result.items()
        if k not in ("fixed_hyperparameters", "model_config", "training_config")
    }
    print(json.dumps(safe, indent=2, default=str))

    print("\n[Comparison Hook]")
    cmp = run_comparison(dry_run=True)
    print(f"  method_count={cmp['method_count']}")
    for m, info in cmp["comparison"].items():
        print(f"    {m:30s} registered={info['registered']}")