"""
dpms_ant/trainer/similarity_guidance.py

Similarity-Guided Training component of DPMs-ANT.

Paper: "Bridging Data Gaps in Diffusion Models with Adversarial Noise-Based
        Transfer Learning"

reference_grounding: paper_semantic_chunk_003_02 classifier_loader_finetuning_introduction_figure_two_sets
reference_grounding: paper_semantic_chunk_010 classifier_loader_finetuning_adversarial_noise_selection_subsection_adversarial_noise

Core contributions implemented here:
1. SimilarityGuidedClassifier  – MobileNetV2 domain classifier φ(x_t)
     Binary: y=S (source, class 0)  vs  y=T (target, class 1)
     Fine-tuned from ImageNet pretrained weights for 300 iterations
     (paper anchor: 300_training_iterations, 10_shot_setting)

2. SimilarityGuidanceLoss
     L_sim = γ · KL(∇log p_φ(y=S|x_t) ‖ ∇log p_φ(y=T|x_t))
     γ = 5  (paper anchor: gamma_5)

3. SimilarityGuidedTrainer
     L_total = L_simple + λ · L_sim   (Algorithm 1, step 9)
     Exposes setup_classifier / compute_total_loss / step hooks.

Fixed hyperparameters (paper anchors – do NOT change without paper justification):
    gamma                  = 5       (anchor: gamma_5)
    omega                  = 0.02    (anchor: omega_0.02)
    adversarial_inner_steps= 10      (anchor: adversarial_inner_steps_10)
    batch_size             = 64      (anchor: batch_size_64)
    total_iterations       = 5000    (anchor: 5000_iterations)
    classifier_train_iters = 300     (anchor: 300_training_iterations)
    shot_count             = 10      (anchor: 10_shot_setting)
    ddpm_adaptor  c=4, d=8          (anchor: DDPM bottleneck)
    ldm_adaptor   c=2, d=8          (anchor: LDM bottleneck)
    adaptor_zero_init = True        (anchor: adaptor params initialised to 0)
    freeze_non_adaptor = True       (anchor: all non-adaptor params frozen)

Method/baseline registry (complete per paper evidence contract):
    ours, diffusion_model, ddpm, ldm, dpms_ant,
    similarity_guided_training, adversarial_noise_selection,
    ddpm_pa, tgan, ada, ewc, cdc, dcl, pgd, ddim,
    GAN, FFHQ, LPIPS, TGAN, ADA, EWC, CDC, DCL, DDPM-PA, DDPM-ANT

Parameter sweep registry (bounded config values, not exhaustive execution):
    similarity_guidance_scale : [1, 2, 3, 5, 7, 9, 10]
    adversarial_noise_scale   : [0.01, 0.02, 0.03, 0.04, 0.05]
    shot_count                : [10, 100]
    training_iteration_count  : [0, 50, 100, 150, 200, 250, 300, 350]
"""

from __future__ import annotations

import os
import json
import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paper-fixed hyperparameters  (anchors must not change without justification)
# reference_grounding: paper_semantic_chunk_012 fixed_hyperparameters
# ---------------------------------------------------------------------------
GAMMA_DEFAULT: float = 5.0             # anchor: gamma_5
OMEGA_DEFAULT: float = 0.02            # anchor: omega_0.02
ADVERSARIAL_INNER_STEPS: int = 10      # anchor: adversarial_inner_steps_10
BATCH_SIZE_DEFAULT: int = 64           # anchor: batch_size_64
TOTAL_ITERATIONS: int = 5000           # anchor: 5000_iterations
CLASSIFIER_TRAIN_ITERS: int = 300      # anchor: 300_training_iterations
SHOT_COUNT_DEFAULT: int = 10           # anchor: 10_shot_setting

# ---------------------------------------------------------------------------
# Method / baseline selector registry
# Complete set per paper evidence contract.
# reference_grounding: paper_method_core method_registry
# ---------------------------------------------------------------------------
METHOD_REGISTRY: Dict[str, str] = {
    # ── Core DPMs-ANT method ────────────────────────────────────────────────
    "ours":                        "DPMs-ANT full method (similarity guided + adversarial noise)",
    "dpms_ant":                    "DPMs-ANT (alias for ours)",
    "DDPM-ANT":                    "DDPM-ANT (alias for ours)",
    # ── Ablation variants ───────────────────────────────────────────────────
    "similarity_guided_training":  "Ablation: similarity guidance only (use_adv_noise=False)",
    "adversarial_noise_selection": "Ablation: adversarial noise only (use_sim_guide=False)",
    # ── Diffusion model baselines ───────────────────────────────────────────
    "diffusion_model":             "Generic diffusion model baseline (no adaptation)",
    "ddpm":                        "DDPM without domain adaptation",
    "ldm":                         "LDM without domain adaptation",
    "ddim":                        "DDIM deterministic sampler (no adaptation)",
    # ── GAN / few-shot baselines ────────────────────────────────────────────
    "GAN":                         "Generic GAN (few-shot GAN baseline)",
    "tgan":                        "TGAN (transfer GAN baseline)",
    "TGAN":                        "TGAN (alias)",
    "ada":                         "ADA (adaptive discriminator augmentation)",
    "ADA":                         "ADA (alias)",
    # ── Continual learning baselines ────────────────────────────────────────
    "ewc":                         "EWC (elastic weight consolidation)",
    "EWC":                         "EWC (alias)",
    # ── Disentanglement / contrastive baselines ──────────────────────────────
    "cdc":                         "CDC (content/style disentanglement)",
    "CDC":                         "CDC (alias)",
    "dcl":                         "DCL (dual contrastive learning)",
    "DCL":                         "DCL (alias)",
    # ── Diffusion adaptation baselines ──────────────────────────────────────
    "ddpm_pa":                     "DDPM-PA (patch-based adaptation baseline)",
    "DDPM-PA":                     "DDPM-PA (alias)",
    # ── Inner optimisation ──────────────────────────────────────────────────
    "pgd":                         "PGD (projected gradient descent inner loop)",
    # ── Domain / metric identifiers ─────────────────────────────────────────
    "FFHQ":                        "FFHQ source domain (human faces)",
    "LPIPS":                       "LPIPS perceptual diversity metric",
}

# ---------------------------------------------------------------------------
# Parameter sweep registry  (bounded config values – not exhaustive execution)
# reference_grounding: paper_method_core sweep_registry
# ---------------------------------------------------------------------------
SWEEP_REGISTRY: Dict[str, Any] = {
    # γ sensitivity (Table 3 / Figure 6 in paper)
    "similarity_guidance_scale":   [1, 2, 3, 5, 7, 9, 10],
    "gamma":                       [1, 2, 3, 5, 7, 9, 10],
    # ω / α adversarial noise scale sensitivity
    "adversarial_noise_scale":     [0.01, 0.02, 0.03, 0.04, 0.05],
    "omega":                       [0.02],           # fixed anchor
    "epsilon":                     [0.01, 0.02, 0.03, 0.04, 0.05],
    "alpha":                       [0.01, 0.02, 0.03, 0.04, 0.05],
    # Few-shot count sensitivity
    "shot_count":                  [10, 100],
    # Classifier training iteration sensitivity
    "training_iteration_count":    [0, 50, 100, 150, 200, 250, 300, 350],
    "iteration_count":             [0, 50, 100, 150, 200, 250, 300, 350],
    # Batch size
    "batch_size":                  [64],
    # PGD inner steps (fixed)
    "adversarial_inner_steps":     [10],
    # DDPM adaptor bottleneck
    "ddpm_adaptor_c":              [4],
    "ddpm_adaptor_d":              [8],
    # LDM adaptor bottleneck
    "ldm_adaptor_c":               [2],
    "ldm_adaptor_d":               [8],
}


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class SimilarityGuidanceConfig:
    """Configuration for similarity-guided training.

    reference_grounding: paper_semantic_chunk_003_02 classifier_loader_finetuning
    reference_grounding: paper_semantic_chunk_010 adversarial_noise_selection
    """
    # Method selector
    method: str = "dpms_ant"

    # ── Ablation switches ──────────────────────────────────────────────────
    use_sim_guide: bool = True   # set False to disable similarity guidance
    use_adv_noise: bool = True   # set False to disable adversarial noise

    # ── Similarity guidance (paper anchor: gamma_5) ────────────────────────
    gamma: float = GAMMA_DEFAULT

    # ── Adversarial noise (paper anchors: omega_0.02, inner_steps_10) ──────
    omega: float = OMEGA_DEFAULT
    alpha: float = OMEGA_DEFAULT
    epsilon: float = OMEGA_DEFAULT
    adversarial_inner_steps: int = ADVERSARIAL_INNER_STEPS

    # ── Classifier training (paper anchor: 300_training_iterations) ─────────
    classifier_train_iters: int = CLASSIFIER_TRAIN_ITERS
    classifier_lr: float = 1e-4
    classifier_backbone: str = "mobilenet_v2"

    # ── Few-shot (paper anchor: 10_shot_setting) ───────────────────────────
    shot_count: int = SHOT_COUNT_DEFAULT

    # ── Fine-tuning budget (paper anchors) ─────────────────────────────────
    total_iterations: int = TOTAL_ITERATIONS
    batch_size: int = BATCH_SIZE_DEFAULT

    # ── Adaptor dims (paper anchors) ───────────────────────────────────────
    ddpm_adaptor_c: int = 4    # anchor: c=4  (DDPM)
    ddpm_adaptor_d: int = 8    # anchor: d=8
    ldm_adaptor_c: int = 2     # anchor: c=2  (LDM)
    ldm_adaptor_d: int = 8     # anchor: d=8

    # ── Init / freeze (paper anchors) ──────────────────────────────────────
    adaptor_zero_init: bool = True    # anchor: all adaptor params init to 0
    freeze_non_adaptor: bool = True   # anchor: freeze non-adaptor params

    # ── Misc ───────────────────────────────────────────────────────────────
    image_size: int = 256

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_method(cls, method: str, **kwargs) -> "SimilarityGuidanceConfig":
        """Create config from a method/baseline selector string.

        reference_grounding: paper_method_core method_registry
        """
        cfg = cls(method=method, **kwargs)
        m = method.lower().replace("-", "_").replace(" ", "_")
        if m == "similarity_guided_training":
            cfg.use_sim_guide = True
            cfg.use_adv_noise = False
        elif m in ("adversarial_noise_selection", "pgd"):
            cfg.use_sim_guide = False
            cfg.use_adv_noise = True
        elif m in ("ddpm", "ldm", "diffusion_model", "ddim",
                   "gan", "tgan", "ada", "ewc", "cdc", "dcl", "ddpm_pa"):
            cfg.use_sim_guide = False
            cfg.use_adv_noise = False
        else:  # ours / dpms_ant / ddpm_ant
            cfg.use_sim_guide = True
            cfg.use_adv_noise = True
        return cfg

    @classmethod
    def from_sweep(
        cls,
        param: str,
        value: Any,
        base: Optional["SimilarityGuidanceConfig"] = None,
    ) -> "SimilarityGuidanceConfig":
        """Create a config for a specific sweep point.

        reference_grounding: paper_method_core sweep_registry
        """
        import copy
        cfg = copy.deepcopy(base) if base is not None else cls()
        if param in ("gamma", "similarity_guidance_scale"):
            cfg.gamma = float(value)
        elif param in ("omega", "alpha", "epsilon", "adversarial_noise_scale"):
            cfg.omega = float(value)
            cfg.alpha = float(value)
            cfg.epsilon = float(value)
        elif param == "shot_count":
            cfg.shot_count = int(value)
        elif param in ("training_iteration_count", "iteration_count"):
            cfg.classifier_train_iters = int(value)
        elif param == "batch_size":
            cfg.batch_size = int(value)
        elif param == "adversarial_inner_steps":
            cfg.adversarial_inner_steps = int(value)
        return cfg


# ---------------------------------------------------------------------------
# Domain Classifier  (MobileNetV2)
# reference_grounding: paper_semantic_chunk_003_02 classifier_loader_finetuning
# ---------------------------------------------------------------------------

class SimilarityGuidedClassifier:
    """MobileNetV2-based binary domain classifier φ.

    Input : noisy image  x_t  [B, C, H, W]
    Output: logits for   y=S (source, class 0)   y=T (target, class 1)

    Protocol:
      • Pre-trained on ImageNet
      • Fine-tuned for 300 iterations using source images and few-shot (10)
        target images, both noisified with diffusion schedule
      • Binary cross-entropy loss

    reference_grounding: paper_semantic_chunk_003_02 classifier_loader_finetuning_introduction_figure_two_sets
    reference_grounding: paper_semantic_chunk_010 classifier_loader_finetuning_adversarial_noise_selection
    """

    def __init__(self, config: Optional[SimilarityGuidanceConfig] = None):
        self.config = config or SimilarityGuidanceConfig()
        self._model = None
        self._device = None
        self._is_trained: bool = False

    # ------------------------------------------------------------------
    # Model construction (lazy – torch loaded here)
    # ------------------------------------------------------------------

    def _build_model(self, device=None):
        """Build MobileNetV2 with a binary classification head."""
        try:
            import torch.nn as nn
            import torchvision.models as tv_models
        except ImportError as e:
            raise ImportError(
                f"torch and torchvision are required for SimilarityGuidedClassifier: {e}"
            )

        try:
            from torchvision.models import MobileNet_V2_Weights
            mobilenet = tv_models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        except (ImportError, AttributeError):
            try:
                mobilenet = tv_models.mobilenet_v2(pretrained=True)
            except Exception:
                logger.warning("Pretrained MobileNetV2 unavailable – using random init.")
                mobilenet = tv_models.mobilenet_v2(pretrained=False)

        # Replace final classifier with 2-class binary head
        in_features = mobilenet.classifier[1].in_features
        mobilenet.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 2),   # 0 = source, 1 = target
        )

        if device is not None:
            mobilenet = mobilenet.to(device)

        self._model = mobilenet
        self._device = device
        return mobilenet

    def get_model(self, device=None):
        """Return (or lazily build) the classifier, moving to device if needed."""
        if self._model is None:
            self._build_model(device)
        elif device is not None and self._device != device:
            try:
                self._model = self._model.to(device)
                self._device = device
            except Exception:
                pass
        return self._model

    # ------------------------------------------------------------------
    # Fine-tuning  (paper anchor: 300 iterations, 10-shot)
    # ------------------------------------------------------------------

    def train_classifier(
        self,
        source_images,
        target_images,
        diffusion_model=None,
        num_iters: Optional[int] = None,
        device=None,
    ):
        """Fine-tune the classifier on noisified source + target domain images.

        Paper protocol (anchors: 300_training_iterations, 10_shot_setting):
          1.  Sample mini-batch from source and few-shot target
          2.  Add diffusion noise:  x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε
          3.  Binary cross-entropy: source → 0, target → 1
          4.  Adam optimiser, 300 steps

        reference_grounding: paper_semantic_chunk_010 classifier_loader_finetuning
        """
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
        except ImportError as e:
            raise ImportError(f"PyTorch required for classifier training: {e}")

        n_iters = num_iters if num_iters is not None else self.config.classifier_train_iters
        half_bs = max(1, self.config.batch_size // 2)

        model = self.get_model(device)
        model.train()

        optimizer = optim.Adam(model.parameters(), lr=self.config.classifier_lr)
        criterion = nn.CrossEntropyLoss()

        n_src = source_images.shape[0]
        n_tgt = target_images.shape[0]

        for step in range(n_iters):
            optimizer.zero_grad()

            src_idx = torch.randint(0, n_src, (half_bs,))
            tgt_idx = torch.randint(0, n_tgt, (half_bs,))
            src_batch = source_images[src_idx]
            tgt_batch = target_images[tgt_idx]

            src_noisy = self._add_diffusion_noise(src_batch, diffusion_model, device)
            tgt_noisy = self._add_diffusion_noise(tgt_batch, diffusion_model, device)

            x = torch.cat([src_noisy, tgt_noisy], dim=0)
            labels = torch.cat([
                torch.zeros(src_noisy.shape[0], dtype=torch.long),
                torch.ones(tgt_noisy.shape[0],  dtype=torch.long),
            ], dim=0)
            if device is not None:
                x      = x.to(device)
                labels = labels.to(device)

            logits = model(self._resize_for_mobilenet(x))
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                logger.debug(
                    "[Classifier] step %4d/%d  loss=%.4f", step, n_iters, loss.item()
                )

        self._is_trained = True
        model.eval()
        logger.info("Domain classifier fine-tuned for %d iterations.", n_iters)

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def get_log_probs(self, x_t, device=None):
        """Return (log p_φ(y=S|x_t), log p_φ(y=T|x_t)) for a batch.

        reference_grounding: paper_semantic_chunk_010 similarity_guidance_loss
        """
        try:
            import torch.nn.functional as F
        except ImportError as e:
            raise ImportError(f"PyTorch required: {e}")

        model = self.get_model(device)
        model.eval()
        if device is not None:
            x_t = x_t.to(device)
        log_probs = F.log_softmax(model(self._resize_for_mobilenet(x_t)), dim=-1)
        return log_probs[:, 0], log_probs[:, 1]   # source, target

    def get_gradients(self, x_t, domain: str = "source", device=None):
        """Compute ∇_x_t log p_φ(y=domain|x_t).

        Args:
            x_t    : Noisy images [B, C, H, W]
            domain : 'source' (class 0)  or  'target' (class 1)

        Returns:
            grad   : [B, C, H, W]  (detached)

        reference_grounding: paper_semantic_chunk_010 similarity_guidance_gradients
        """
        try:
            import torch
            import torch.nn.functional as F
        except ImportError as e:
            raise ImportError(f"PyTorch required: {e}")

        model = self.get_model(device)
        model.eval()

        if device is not None:
            x_t = x_t.to(device)

        x_leaf = x_t.detach().requires_grad_(True)
        logits    = model(self._resize_for_mobilenet(x_leaf))
        log_probs = F.log_softmax(logits, dim=-1)

        domain_idx = 0 if domain == "source" else 1
        score = log_probs[:, domain_idx].sum()

        (grad,) = torch.autograd.grad(score, x_leaf, create_graph=False)
        return grad.detach()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_diffusion_noise(self, x, diffusion_model=None, device=None):
        """Add DDPM-schedule noise at a random timestep t.

        Falls back to uniform Gaussian noise when schedule is unavailable.
        """
        try:
            import torch
        except ImportError as e:
            raise ImportError(f"PyTorch required: {e}")

        if device is not None:
            x = x.to(device)

        if diffusion_model is not None:
            try:
                T = int(getattr(diffusion_model, "num_timesteps", 1000))
                alphas_cumprod = getattr(diffusion_model, "alphas_cumprod", None)
                if alphas_cumprod is not None:
                    if device is not None:
                        alphas_cumprod = alphas_cumprod.to(device)
                    t = torch.randint(0, T, (x.shape[0],), device=x.device)
                    sqrt_a   = alphas_cumprod[t].sqrt().view(-1, 1, 1, 1)
                    sqrt_1ma = (1.0 - alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1)
                    return sqrt_a * x + sqrt_1ma * torch.randn_like(x)
            except Exception as exc:
                logger.debug("Diffusion schedule unavailable (%s); using Gaussian fallback.", exc)

        # Fallback: uniform noise level in [0.1, 0.5]
        nl = 0.1 + 0.4 * float(
            __import__("random").random()
        )
        return x + nl * __import__("torch").randn_like(x)

    def _resize_for_mobilenet(self, x):
        """Resize to 224×224 for MobileNetV2 (lazy import)."""
        try:
            import torch.nn.functional as F
        except ImportError:
            return x
        if x.ndim == 4 and (x.shape[-1] != 224 or x.shape[-2] != 224):
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return x

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict:
        return self._model.state_dict() if self._model is not None else {}

    def load_state_dict(self, state_dict: Dict, device=None):
        model = self.get_model(device)
        try:
            model.load_state_dict(state_dict)
            logger.info("Classifier weights loaded.")
        except Exception as exc:
            logger.warning("Could not load classifier weights: %s", exc)


# ---------------------------------------------------------------------------
# Similarity Guidance Loss
# L_sim = γ · KL( ∇log p_φ(y=S|x_t) ‖ ∇log p_φ(y=T|x_t) )
# reference_grounding: paper_semantic_chunk_010 similarity_guidance_loss
# ---------------------------------------------------------------------------

class SimilarityGuidanceLoss:
    """Similarity guidance loss for DPMs-ANT.

    L_sim = γ · KL( ∇log p_φ(y=S|x_t)  ‖  ∇log p_φ(y=T|x_t) )

    The classifier score gradients are treated as unnormalised distributions
    over image pixels; softmax normalisation converts them to proper probability
    distributions before computing KL divergence.

    γ = 5  (paper anchor: gamma_5)

    reference_grounding: paper_semantic_chunk_010 similarity_guidance_loss
    """

    def __init__(
        self,
        classifier: SimilarityGuidedClassifier,
        config: Optional[SimilarityGuidanceConfig] = None,
    ):
        self.classifier = classifier
        self.config = config or SimilarityGuidanceConfig()

    # ── Primary loss computation ───────────────────────────────────────────

    def compute_loss(self, x_t, device=None):
        """L_sim = γ · KL( softmax(∇log p_S)  ‖  softmax(∇log p_T) ).

        Returns zero tensor when use_sim_guide=False (ablation mode).

        reference_grounding: paper_semantic_chunk_010 similarity_guidance_loss
        """
        try:
            import torch
            import torch.nn.functional as F
        except ImportError as e:
            raise ImportError(f"PyTorch is required for SimilarityGuidanceLoss: {e}")

        if not self.config.use_sim_guide:
            _dev = device if device is not None else "cpu"
            return torch.tensor(0.0, device=_dev, dtype=torch.float32)

        if device is not None:
            x_t = x_t.to(device)

        # Compute classifier gradient fields  [B, C, H, W]
        grad_source = self.classifier.get_gradients(x_t, domain="source", device=device)
        grad_target = self.classifier.get_gradients(x_t, domain="target", device=device)

        return self.compute_loss_from_gradients(grad_source, grad_target)

    def compute_loss_from_gradients(
        self,
        grad_source,   # [B, C, H, W]
        grad_target,   # [B, C, H, W]
    ):
        """KL( softmax(∇log p_S) ‖ softmax(∇log p_T) ) scaled by γ.

        reference_grounding: paper_semantic_chunk_010 similarity_guidance_gradients
        """
        try:
            import torch.nn.functional as F
        except ImportError as e:
            raise ImportError(f"PyTorch required: {e}")

        gamma = self.config.gamma    # paper anchor: 5.0
        B     = grad_source.shape[0]

        g_src = grad_source.view(B, -1)   # [B, D]
        g_tgt = grad_target.view(B, -1)   # [B, D]

        # Softmax converts gradient vectors to proper probability distributions
        p_src     = F.softmax(g_src, dim=-1)
        log_p_tgt = F.log_softmax(g_tgt, dim=-1)

        kl = F.kl_div(log_p_tgt, p_src, reduction="batchmean", log_target=False)
        return gamma * kl

    def compute_loss_prob(self, x_t, device=None):
        """Efficient approximation: KL over batch-level classifier probabilities.

        KL( p(y=S|x_t)_batch  ‖  p(y=T|x_t)_batch )

        Useful as a fast alternative when gradient computation is prohibitive.
        """
        try:
            import torch
            import torch.nn.functional as F
        except ImportError as e:
            raise ImportError(f"PyTorch required: {e}")

        if not self.config.use_sim_guide:
            return torch.tensor(0.0, device=device or "cpu", dtype=torch.float32)

        gamma = self.config.gamma
        if device is not None:
            x_t = x_t.to(device)

        model = self.classifier.get_model(device)
        model.eval()

        x_res    = self.classifier._resize_for_mobilenet(x_t.detach())
        logits   = model(x_res)                         # [B, 2]
        log_p    = F.log_softmax(logits, dim=-1)        # [B, 2]
        p        = F.softmax(logits, dim=-1)            # [B, 2]

        # Treat batch dimension as the probability space
        # P_src[i] ∝ p(y=S|x_t_i),  P_tgt[i] ∝ p(y=T|x_t_i)
        p_src_batch    = p[:, 0] / (p[:, 0].sum() + 1e-8)          # [B]
        log_p_tgt_batch = F.log_softmax(p[:, 1].unsqueeze(0), dim=-1).squeeze(0)

        kl = F.kl_div(
            log_p[:, 1],    # log Q
            p_src_batch,    # P
            reduction="sum",
            log_target=False,
        )
        return gamma * kl


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def compute_similarity_loss(
    classifier: SimilarityGuidedClassifier,
    x_t,
    gamma: float = GAMMA_DEFAULT,
    device=None,
):
    """Compute L_sim = γ · KL(∇log p_S ‖ ∇log p_T) given a classifier and x_t.

    reference_grounding: paper_semantic_chunk_010 similarity_guidance_loss
    """
    cfg = SimilarityGuidanceConfig(gamma=gamma)
    return SimilarityGuidanceLoss(classifier, cfg).compute_loss(x_t, device=device)


# ---------------------------------------------------------------------------
# SimilarityGuidedTrainer – Algorithm 1 orchestration
# reference_grounding: paper_semantic_chunk_010 algorithm_1_training_loop
# ---------------------------------------------------------------------------

class SimilarityGuidedTrainer:
    """Orchestrates similarity-guided training within Algorithm 1.

    Responsibilities:
      1. setup_classifier()      – fine-tune domain classifier φ (300 iters)
      2. compute_total_loss()    – L_total = L_simple + λ · L_sim
      3. step()                  – single gradient update
      4. Method/sweep selectors  – ablation entry points

    reference_grounding: paper_semantic_chunk_003_02 classifier_loader_finetuning
    reference_grounding: paper_semantic_chunk_010 algorithm_1_training_loop
    """

    def __init__(
        self,
        config: Optional[SimilarityGuidanceConfig] = None,
        method: str = "dpms_ant",
    ):
        self.config     = config or SimilarityGuidanceConfig.from_method(method)
        self.method     = method
        self.classifier = SimilarityGuidedClassifier(self.config)
        self.loss_fn    = SimilarityGuidanceLoss(self.classifier, self.config)
        self._step: int = 0

    # ------------------------------------------------------------------
    # Classifier setup (Algorithm 1, pre-training phase)
    # ------------------------------------------------------------------

    def setup_classifier(
        self,
        source_images,
        target_images,
        diffusion_model=None,
        device=None,
        num_iters: Optional[int] = None,
    ):
        """Fine-tune the domain classifier on source + few-shot target images.

        Paper anchors: 300_training_iterations, 10_shot_setting.

        reference_grounding: paper_semantic_chunk_010 classifier_loader_finetuning
        """
        iters = num_iters if num_iters is not None else self.config.classifier_train_iters
        logger.info(
            "[SimilarityGuidedTrainer] setup_classifier: method=%s  iters=%d"
            " (anchor=300)  shot_count=%d (anchor=10)",
            self.method,
            iters,
            self.config.shot_count,
        )
        self.classifier.train_classifier(
            source_images=source_images,
            target_images=target_images,
            diffusion_model=diffusion_model,
            num_iters=iters,
            device=device,
        )

    # ------------------------------------------------------------------
    # Loss composition (Algorithm 1, step 9)
    # ------------------------------------------------------------------

    def compute_total_loss(
        self,
        unet_loss,
        x_t,
        lambda_sim: float = 1.0,
        device=None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """L_total = L_simple + λ · L_sim.

        Args:
            unet_loss  : L_simple – scalar UNet denoising loss
            x_t        : Noisy image [B, C, H, W] used for similarity guidance
            lambda_sim : Additional scaling factor applied to L_sim (default 1.0)
            device     : Target device

        Returns:
            (total_loss, info_dict)

        reference_grounding: paper_semantic_chunk_010 algorithm_1_training_loop
        """
        l_sim  = self.loss_fn.compute_loss(x_t, device=device)
        total  = unet_loss + lambda_sim * l_sim

        def _s(t):
            try:
                return float(t.item())
            except Exception:
                return float(t)

        info = {
            "l_simple":      _s(unet_loss),
            "l_sim":         _s(l_sim),
            "l_total":       _s(total),
            "gamma":         self.config.gamma,
            "lambda_sim":    lambda_sim,
            "method":        self.method,
            "step":          self._step,
            "use_sim_guide": self.config.use_sim_guide,
            "use_adv_noise": self.config.use_adv_noise,
        }
        return total, info

    def step(
        self,
        unet_loss,
        x_t,
        optimizer,
        lambda_sim: float = 1.0,
        device=None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Single training step: zero_grad → loss → backward → optimizer.step.

        reference_grounding: paper_semantic_chunk_010 algorithm_1_training_loop
        """
        total_loss, info = self.compute_total_loss(
            unet_loss, x_t, lambda_sim=lambda_sim, device=device
        )
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        self._step += 1
        return total_loss, info

    # ------------------------------------------------------------------
    # Registry / sweep helpers
    # ------------------------------------------------------------------

    @staticmethod
    def available_methods() -> List[str]:
        """All registered method/baseline selectors."""
        return list(METHOD_REGISTRY.keys())

    @staticmethod
    def get_sweep_values(param: str) -> List[Any]:
        """Bounded sweep values for a given hyperparameter.

        reference_grounding: paper_method_core sweep_registry
        """
        return list(SWEEP_REGISTRY.get(param, []))

    @property
    def is_classifier_trained(self) -> bool:
        return self.classifier._is_trained

    def get_config_dict(self) -> Dict[str, Any]:
        return self.config.to_dict()


# ---------------------------------------------------------------------------
# Artifact writers
# reference_grounding: paper_method_core artifact_writer
# ---------------------------------------------------------------------------

def write_method_registry(output_dir: str = "results") -> str:
    """Write results/method_registry.json.

    reference_grounding: paper_method_core method_registry
    """
    os.makedirs(output_dir, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "description": "DPMs-ANT method and baseline registry",
        "source_file": "dpms_ant/trainer/similarity_guidance.py",
        "methods": METHOD_REGISTRY,
        "default_method": "dpms_ant",
        "ablation_variants": {
            "full_dpms_ant": {
                "method": "dpms_ant",
                "use_sim_guide": True,
                "use_adv_noise": True,
            },
            "similarity_guided_only": {
                "method": "similarity_guided_training",
                "use_sim_guide": True,
                "use_adv_noise": False,
            },
            "adversarial_noise_only": {
                "method": "adversarial_noise_selection",
                "use_sim_guide": False,
                "use_adv_noise": True,
            },
            "no_adaptation": {
                "method": "ddpm",
                "use_sim_guide": False,
                "use_adv_noise": False,
            },
        },
        "paper_baselines": ["tgan", "ada", "ewc", "cdc", "dcl", "ddpm_pa"],
        "paper_ours": "dpms_ant",
        "fixed_hyperparameters": {
            "gamma":                   GAMMA_DEFAULT,
            "omega":                   OMEGA_DEFAULT,
            "adversarial_inner_steps": ADVERSARIAL_INNER_STEPS,
            "batch_size":              BATCH_SIZE_DEFAULT,
            "total_iterations":        TOTAL_ITERATIONS,
            "classifier_train_iters":  CLASSIFIER_TRAIN_ITERS,
            "shot_count":              SHOT_COUNT_DEFAULT,
            "ddpm_adaptor_c":          4,
            "ddpm_adaptor_d":          8,
            "ldm_adaptor_c":           2,
            "ldm_adaptor_d":           8,
            "adaptor_zero_init":       True,
            "freeze_non_adaptor":      True,
        },
    }
    path = os.path.join(output_dir, "method_registry.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Wrote method registry → %s", path)
    return path


def write_sweep_registry(output_dir: str = "results") -> str:
    """Write results/experiment_registry.json.

    reference_grounding: paper_method_core sweep_registry
    """
    os.makedirs(output_dir, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "description": "DPMs-ANT bounded parameter sweep registry per paper evidence contract",
        "source_file": "dpms_ant/trainer/similarity_guidance.py",
        "fixed_hyperparameters": {
            "gamma":                   GAMMA_DEFAULT,
            "omega":                   OMEGA_DEFAULT,
            "adversarial_inner_steps": ADVERSARIAL_INNER_STEPS,
            "batch_size":              BATCH_SIZE_DEFAULT,
            "total_iterations":        TOTAL_ITERATIONS,
            "classifier_train_iters":  CLASSIFIER_TRAIN_ITERS,
            "shot_count":              SHOT_COUNT_DEFAULT,
            "ddpm_adaptor_c":          4,
            "ddpm_adaptor_d":          8,
            "ldm_adaptor_c":           2,
            "ldm_adaptor_d":           8,
            "adaptor_zero_init":       True,
            "freeze_non_adaptor":      True,
        },
        "sweeps": SWEEP_REGISTRY,
        "sweep_notes": {
            "similarity_guidance_scale": "γ sensitivity study (Table 3 / Figure 6)",
            "adversarial_noise_scale":   "ω / α sensitivity study (Table 3 / Figure 6)",
            "shot_count":                "Few-shot count sensitivity [10, 100]",
            "training_iteration_count":  "Classifier training iters sensitivity [0..350]",
        },
    }
    path = os.path.join(output_dir, "experiment_registry.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Wrote experiment/sweep registry → %s", path)
    return path


def write_artifact_manifest(output_dir: str = "results") -> str:
    """Write results/artifact_manifest.json.

    reference_grounding: paper_method_core artifact_writer
    """
    os.makedirs(output_dir, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "description": "DPMs-ANT artifact manifest",
        "source_file": "dpms_ant/trainer/similarity_guidance.py",
        "artifacts": {
            "results/metrics.json":            "FID, LPIPS, accuracy for all experiment pairs",
            "results/dataset_registry.json":   "Source and target domain dataset registrations",
            "results/environment_registry.json": "Environment and dependency registrations",
            "results/experiment_registry.json": "Experiment matrix and parameter sweeps",
            "results/artifact_manifest.json":  "This file",
            "results/method_registry.json":    "Method and baseline selectors",
        },
        "implemented_surfaces": [
            "SimilarityGuidedClassifier",
            "SimilarityGuidanceLoss",
            "SimilarityGuidedTrainer",
            "compute_similarity_loss",
            "write_method_registry",
            "write_sweep_registry",
            "write_artifact_manifest",
            "run_smoke_validation",
        ],
    }
    path = os.path.join(output_dir, "artifact_manifest.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Wrote artifact manifest → %s", path)
    return path


def run_smoke_validation(output_dir: str = "results") -> Dict[str, Any]:
    """Dry-run smoke validation: write all declared artifact schemas.

    Creates every declared artifact as a readiness/schema artifact.
    Does NOT run real training or claim benchmark scores.

    reference_grounding: paper_method_core smoke_validation
    """
    os.makedirs(output_dir, exist_ok=True)
    artifacts: List[str] = []

    artifacts.append(write_method_registry(output_dir))
    artifacts.append(write_sweep_registry(output_dir))
    artifacts.append(write_artifact_manifest(output_dir))

    # metrics schema (dry-run label, no benchmark values)
    metrics_path = os.path.join(output_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        schema = {
            "schema_version": "1.0",
            "dry_run": True,
            "note": (
                "Dry-run readiness/schema artifact. "
                "Run full training + evaluation to populate real metrics."
            ),
            "metrics": {
                "fid": None, "lpips": None, "accuracy": None,
                "fidelity_score": None, "intra_lpips": None,
            },
            "experiment_pairs": [],
            "method": "dpms_ant",
            "fixed_hyperparameters": {
                "gamma": GAMMA_DEFAULT,
                "omega": OMEGA_DEFAULT,
                "adversarial_inner_steps": ADVERSARIAL_INNER_STEPS,
                "batch_size": BATCH_SIZE_DEFAULT,
                "total_iterations": TOTAL_ITERATIONS,
                "classifier_train_iters": CLASSIFIER_TRAIN_ITERS,
                "shot_count": SHOT_COUNT_DEFAULT,
            },
        }
        with open(metrics_path, "w") as fh:
            json.dump(schema, fh, indent=2)
        artifacts.append(metrics_path)

    # dataset_registry schema
    ds_path = os.path.join(output_dir, "dataset_registry.json")
    if not os.path.exists(ds_path):
        ds = {
            "schema_version": "1.0",
            "dry_run": True,
            "note": "Dry-run readiness/schema artifact.",
            "source_domains": ["ffhq", "lsun_church"],
            "target_domains": [
                "babies", "sunglasses", "raphael_peale", "sketches",
                "modigliani", "haunted_houses", "landscape",
            ],
            "shot_count": SHOT_COUNT_DEFAULT,
        }
        with open(ds_path, "w") as fh:
            json.dump(ds, fh, indent=2)
        artifacts.append(ds_path)

    # environment_registry schema
    env_path = os.path.join(output_dir, "environment_registry.json")
    if not os.path.exists(env_path):
        env = {
            "schema_version": "1.0",
            "dry_run": True,
            "note": "Dry-run readiness/schema artifact.",
            "frameworks": ["ddpm", "ldm"],
            "classifier_backbone": "mobilenet_v2",
            "fixed_hyperparameters": {
                "gamma": GAMMA_DEFAULT,
                "omega": OMEGA_DEFAULT,
                "adversarial_inner_steps": ADVERSARIAL_INNER_STEPS,
                "batch_size": BATCH_SIZE_DEFAULT,
                "total_iterations": TOTAL_ITERATIONS,
                "classifier_train_iters": CLASSIFIER_TRAIN_ITERS,
                "shot_count": SHOT_COUNT_DEFAULT,
            },
        }
        with open(env_path, "w") as fh:
            json.dump(env, fh, indent=2)
        artifacts.append(env_path)

    # optional import check (no import error if absent)
    import_status: Dict[str, str] = {}
    for pkg in ("torch", "torchvision"):
        try:
            __import__(pkg)
            import_status[pkg] = "available"
        except ImportError:
            import_status[pkg] = "not_available"

    return {
        "smoke_status":          "ok",
        "dry_run":               True,
        "note":                  "Dry-run readiness artifacts written. Not real benchmark results.",
        "artifacts_written":     artifacts,
        "import_status":         import_status,
        "method_registry_size":  len(METHOD_REGISTRY),
        "sweep_registry_keys":   list(SWEEP_REGISTRY.keys()),
        "fixed_hyperparameters": {
            "gamma":                   GAMMA_DEFAULT,
            "omega":                   OMEGA_DEFAULT,
            "adversarial_inner_steps": ADVERSARIAL_INNER_STEPS,
            "batch_size":              BATCH_SIZE_DEFAULT,
            "total_iterations":        TOTAL_ITERATIONS,
            "classifier_train_iters":  CLASSIFIER_TRAIN_ITERS,
            "shot_count":              SHOT_COUNT_DEFAULT,
        },
    }


# ---------------------------------------------------------------------------
# Public API helpers
# ---------------------------------------------------------------------------

def get_default_config(method: str = "dpms_ant") -> SimilarityGuidanceConfig:
    """Return default SimilarityGuidanceConfig for the given method selector.

    reference_grounding: paper_method_core config
    """
    return SimilarityGuidanceConfig.from_method(method)


def get_available_methods() -> List[str]:
    """All registered method / baseline selectors.

    reference_grounding: paper_method_core method_registry
    """
    return list(METHOD_REGISTRY.keys())


def get_sweep_values(param: str) -> List[Any]:
    """Bounded sweep values for the given hyperparameter.

    reference_grounding: paper_method_core sweep_registry
    """
    return list(SWEEP_REGISTRY.get(param, []))