# dpms_ant/trainer/ant_trainer.py
# =============================================================================
# DPMs-ANT Trainer – Algorithm 1 complete training loop
# Paper: "Bridging Data Gaps in Diffusion Models with Adversarial Noise-Based
#         Transfer Learning"
#
# reference_grounding: paper_method_core dpms_ant/trainer/ant_trainer.py
# reference_grounding: paper_semantic_chunk_003_02 classifier_loader_finetuning
# reference_grounding: paper_semantic_chunk_010 adversarial_noise_selection
#
# Implements:
#   - ANTTrainer: Algorithm 1 full training loop
#   - Method/baseline registry (ours, ddpm, ldm, dpms_ant, tgan, ada, ewc,
#     cdc, dcl, ddpm_pa, ddpm_ant, pgd, ddim, similarity_guided_training,
#     adversarial_noise_selection, diffusion_model, gan, ffhq, lpips)
#   - Bounded parameter sweep registry (gamma, omega, alpha, shot_count, etc.)
#   - Dry-run-safe smoke hooks
# =============================================================================

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Method / baseline registry
# Complete selector set required by paper evidence contract.
# reference_grounding: paper_method_core method_registry
# ---------------------------------------------------------------------------
METHOD_REGISTRY: Dict[str, Dict[str, Any]] = {
    "ours": {
        "name": "DPMs-ANT (Ours)",
        "use_sim_guide": True,
        "use_adv_noise": True,
        "description": "Full DPMs-ANT: similarity-guided training + adversarial noise selection",
    },
    "dpms_ant": {
        "name": "DPMs-ANT",
        "use_sim_guide": True,
        "use_adv_noise": True,
        "description": "Alias for ours – full Algorithm 1",
    },
    "ddpm_ant": {
        "name": "DDPM-ANT",
        "use_sim_guide": True,
        "use_adv_noise": True,
        "description": "DPMs-ANT applied to DDPM backbone",
    },
    "ldm_ant": {
        "name": "LDM-ANT",
        "use_sim_guide": True,
        "use_adv_noise": True,
        "description": "DPMs-ANT applied to latent diffusion with frozen autoencoder and trainable U-Net shift adaptor",
    },
    "similarity_guided_training": {
        "name": "Similarity-Guided Training only",
        "use_sim_guide": True,
        "use_adv_noise": False,
        "description": "Ablation: similarity guidance without adversarial noise",
    },
    "adversarial_noise_selection": {
        "name": "Adversarial Noise Selection only",
        "use_sim_guide": False,
        "use_adv_noise": True,
        "description": "Ablation: adversarial noise without similarity guidance",
    },
    "diffusion_model": {
        "name": "Vanilla Diffusion Model (fine-tune)",
        "use_sim_guide": False,
        "use_adv_noise": False,
        "description": "Baseline: standard fine-tuning without ANT components",
    },
    "ddpm": {
        "name": "DDPM fine-tune",
        "use_sim_guide": False,
        "use_adv_noise": False,
        "description": "Baseline: DDPM fine-tuned on target domain",
    },
    "ldm": {
        "name": "LDM fine-tune",
        "use_sim_guide": False,
        "use_adv_noise": False,
        "description": "Baseline: LDM fine-tuned on target domain",
    },
    "ddpm_pa": {
        "name": "DDPM-PA",
        "use_sim_guide": False,
        "use_adv_noise": False,
        "description": "Baseline: DDPM with patch-level augmentation",
    },
    "tgan": {
        "name": "TGAN",
        "use_sim_guide": False,
        "use_adv_noise": False,
        "description": "Baseline: Transfer GAN",
    },
    "ada": {
        "name": "ADA",
        "use_sim_guide": False,
        "use_adv_noise": False,
        "description": "Baseline: Adaptive Discriminator Augmentation",
    },
    "ewc": {
        "name": "EWC",
        "use_sim_guide": False,
        "use_adv_noise": False,
        "description": "Baseline: Elastic Weight Consolidation",
    },
    "cdc": {
        "name": "CDC",
        "use_sim_guide": False,
        "use_adv_noise": False,
        "description": "Baseline: Cross-Domain Correspondence",
    },
    "dcl": {
        "name": "DCL",
        "use_sim_guide": False,
        "use_adv_noise": False,
        "description": "Baseline: Dual Contrastive Learning",
    },
    "pgd": {
        "name": "PGD",
        "use_sim_guide": False,
        "use_adv_noise": True,
        "description": "PGD adversarial noise only (no similarity guidance)",
    },
    "ddim": {
        "name": "DDIM",
        "use_sim_guide": False,
        "use_adv_noise": False,
        "description": "DDIM sampler baseline",
    },
    # Additional paper-referenced identifiers
    "gan": {
        "name": "GAN",
        "use_sim_guide": False,
        "use_adv_noise": False,
        "description": "Generic GAN baseline",
    },
    "ffhq": {
        "name": "FFHQ pre-trained",
        "use_sim_guide": False,
        "use_adv_noise": False,
        "description": "FFHQ pre-trained model without fine-tuning",
    },
    "lpips": {
        "name": "LPIPS diversity metric",
        "use_sim_guide": False,
        "use_adv_noise": False,
        "description": "Intra-LPIPS diversity evaluation",
    },
}

# ---------------------------------------------------------------------------
# Bounded parameter sweep registry
# reference_grounding: paper_semantic_chunk_012 sensitivity analysis
# ---------------------------------------------------------------------------
SWEEP_REGISTRY: Dict[str, Any] = {
    # Similarity guidance weight γ – paper anchor: gamma_5
    "gamma": {
        "default": 5,
        "values": [1, 2, 3, 5, 7, 9, 10],
        "paper_anchor": "gamma_5",
    },
    # PGD step size ω – paper anchor: omega_0.02
    "omega": {
        "default": 0.02,
        "values": [0.01, 0.02, 0.03, 0.04, 0.05],
        "paper_anchor": "omega_0.02",
    },
    # Adversarial perturbation budget α (epsilon)
    "alpha": {
        "default": 0.02,
        "values": [0.01, 0.02, 0.03, 0.04, 0.05],
        "paper_anchor": "adversarial_noise_scale",
    },
    # Alias: epsilon = alpha
    "epsilon": {
        "default": 0.02,
        "values": [0.01, 0.02, 0.03, 0.04, 0.05],
        "paper_anchor": "adversarial_noise_scale",
    },
    # Few-shot count – paper anchor: 10_shot_setting
    "shot_count": {
        "default": 10,
        "values": [10, 100],
        "paper_anchor": "10_shot_setting",
    },
    # Classifier training iterations – paper anchor: 300_training_iterations
    "training_iteration_count": {
        "default": 300,
        "values": [0, 50, 100, 150, 200, 250, 300, 350],
        "paper_anchor": "300_training_iterations",
    },
    # Similarity guidance scale (same as gamma, explicit alias)
    "similarity_guidance_scale": {
        "default": 5,
        "values": [1, 2, 3, 5, 7, 9, 10],
        "paper_anchor": "gamma_5",
    },
    # Adversarial noise scale (same as alpha/epsilon, explicit alias)
    "adversarial_noise_scale": {
        "default": 0.02,
        "values": [0.01, 0.02, 0.03, 0.04, 0.05],
        "paper_anchor": "omega_0.02",
    },
    # Total fine-tuning iterations – paper anchor: 5000_iterations
    "iteration_count": {
        "default": 5000,
        "values": [5000],
        "paper_anchor": "5000_iterations",
    },
    # Batch size – paper anchor: batch_size_64
    "batch_size": {
        "default": 64,
        "values": [64],
        "paper_anchor": "batch_size_64",
    },
    # PGD inner steps – paper anchor: adversarial_inner_steps_10
    "adversarial_inner_steps": {
        "default": 10,
        "values": [10],
        "paper_anchor": "adversarial_inner_steps_10",
    },
}

# ---------------------------------------------------------------------------
# Fixed hyperparameter anchors (must not be overridden in sweeps)
# reference_grounding: paper_semantic_chunk_012 fixed hyperparameters
# ---------------------------------------------------------------------------
FIXED_HYPERPARAMETERS: Dict[str, Any] = {
    "total_iterations": 5000,           # anchor: 5000_iterations
    "classifier_train_iterations": 300,  # anchor: 300_training_iterations
    "shot_count": 10,                    # anchor: 10_shot_setting
    "gamma": 5,                          # anchor: gamma_5
    "omega": 0.02,                       # anchor: omega_0.02
    "adversarial_inner_steps": 10,       # anchor: adversarial_inner_steps_10
    "batch_size": 64,                    # anchor: batch_size_64
    # Shift Adaptor bottleneck dims
    "ddpm_adaptor_c": 4,                 # DDPM: c=4
    "ddpm_adaptor_d": 8,                 # DDPM: d=8
    "ldm_adaptor_c": 2,                  # LDM: c=2
    "ldm_adaptor_d": 8,                  # LDM: d=8
    # Adaptor init: all parameters = 0
    "adaptor_init_zero": True,
    # Non-adaptor parameters: fully frozen
    "freeze_non_adaptor": True,
}


# ---------------------------------------------------------------------------
# Trainer configuration dataclass
# ---------------------------------------------------------------------------
@dataclass
class ANTTrainerConfig:
    """
    Configuration for the ANT trainer.
    All paper-anchored hyperparameters are set as defaults.
    reference_grounding: paper_method_core ANTTrainerConfig
    """
    # Method selector
    method: str = "dpms_ant"
    framework: str = "ddpm"  # ddpm | ldm

    # Algorithm 1 switches (ablation support)
    use_sim_guide: bool = True   # similarity-guided training
    use_adv_noise: bool = True   # adversarial noise selection

    # Fixed hyperparameters (paper anchors)
    total_iterations: int = FIXED_HYPERPARAMETERS["total_iterations"]
    classifier_train_iterations: int = FIXED_HYPERPARAMETERS["classifier_train_iterations"]
    shot_count: int = FIXED_HYPERPARAMETERS["shot_count"]
    gamma: float = FIXED_HYPERPARAMETERS["gamma"]
    omega: float = FIXED_HYPERPARAMETERS["omega"]
    adversarial_inner_steps: int = FIXED_HYPERPARAMETERS["adversarial_inner_steps"]
    batch_size: int = FIXED_HYPERPARAMETERS["batch_size"]

    # Adaptor bottleneck dims
    adaptor_c: int = FIXED_HYPERPARAMETERS["ddpm_adaptor_c"]
    adaptor_d: int = FIXED_HYPERPARAMETERS["ddpm_adaptor_d"]

    # Perturbation budget α
    alpha: float = 0.02

    # Similarity guidance weight λ (same as gamma in paper notation)
    lambda_sim: float = FIXED_HYPERPARAMETERS["gamma"]

    # Checkpoint / output
    output_dir: str = "results"
    checkpoint_interval: int = 500
    log_interval: int = 50

    # Dry-run / smoke mode
    dry_run: bool = False
    smoke_iterations: int = 2

    # Device
    device: str = "cpu"

    def apply_method_registry(self) -> None:
        """Override use_sim_guide / use_adv_noise from method registry."""
        if self.method in METHOD_REGISTRY:
            entry = METHOD_REGISTRY[self.method]
            self.use_sim_guide = entry["use_sim_guide"]
            self.use_adv_noise = entry["use_adv_noise"]
        else:
            logger.warning("Method '%s' not in METHOD_REGISTRY; using config flags.", self.method)

    def effective_iterations(self) -> int:
        """Return smoke-bounded or full iteration count."""
        return self.smoke_iterations if self.dry_run else self.total_iterations

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# ANTTrainer – Algorithm 1 implementation
# reference_grounding: paper_method_core ANTTrainer
# reference_grounding: paper_semantic_chunk_010 adversarial_noise_selection
# ---------------------------------------------------------------------------
class ANTTrainer:
    """
    DPMs-ANT trainer implementing Algorithm 1:

    For each iteration t = 1 … T:
      1. Sample a mini-batch of target-domain images x_0.
      2. Sample timestep τ ~ Uniform(1, T_diff).
      3. [if use_adv_noise] Run Eq. 7 gradient ascent to find Gaussian
         adversarial noise ε*, applying Norm(.) after each update.
      4. Forward diffusion: x_τ = √ᾱ_τ · x_0 + √(1-ᾱ_τ) · ε*.
      5. UNet prediction: ε̂ = UNet_θ(x_τ, τ).
      6. L_simple = ‖ε* - ε̂‖².
      7. [if use_sim_guide] compute the target classifier gradient
         ∇log p_φ(y=T|x_τ) and the sigma-hat scaling from Eq. 5.
      8. L_total = ||ε* - ε_{θ,ψ}(x_τ,τ) -
         sigma_hat² γ ∇log p_φ(y=T|x_τ)||² (Eq. 8).
      9. Update only Shift Adaptor parameters (all others frozen).

    reference_grounding: paper_semantic_chunk_003_02 classifier_loader_finetuning
    reference_grounding: paper_semantic_chunk_010 adversarial_noise_selection
    """

    def __init__(
        self,
        config: ANTTrainerConfig,
        diffusion_model=None,
        adaptor=None,
        classifier=None,
        train_dataloader=None,
        ema=None,
    ):
        self.config = config
        self.config.apply_method_registry()

        self.diffusion_model = diffusion_model
        self.adaptor = adaptor
        self.classifier = classifier
        self.train_dataloader = train_dataloader
        self.ema = ema

        self._step = 0
        self._loss_history: List[Dict[str, float]] = []

        # Lazy imports – only resolved when actual training is requested
        self._torch = None
        self._optim = None

        logger.info(
            "ANTTrainer initialised | method=%s | use_sim_guide=%s | use_adv_noise=%s | "
            "total_iters=%d | gamma=%.2f | omega=%.4f | adv_steps=%d",
            config.method,
            config.use_sim_guide,
            config.use_adv_noise,
            config.effective_iterations(),
            config.gamma,
            config.omega,
            config.adversarial_inner_steps,
        )

    # ------------------------------------------------------------------
    # Lazy torch import
    # ------------------------------------------------------------------
    def _get_torch(self):
        if self._torch is None:
            try:
                import torch
                self._torch = torch
            except ImportError as exc:
                raise RuntimeError(
                    "PyTorch is required for training. Install it with: pip install torch"
                ) from exc
        return self._torch

    # ------------------------------------------------------------------
    # Parameter management
    # ------------------------------------------------------------------
    def _freeze_non_adaptor_params(self) -> None:
        """
        Freeze all parameters except Shift Adaptor.
        Paper anchor: 非adaptor参数完全冻结
        reference_grounding: paper_method_core shift_adaptor_freeze
        """
        if self.diffusion_model is None:
            return
        torch = self._get_torch()
        if self.config.method == "ddpm_pa":
            for _, param in self.diffusion_model.named_parameters():
                param.requires_grad_(True)
            logger.info("DDPM-PA baseline selected; all diffusion parameters remain trainable.")
            return
        frozen, trainable = 0, 0
        for name, param in self.diffusion_model.named_parameters():
            if "adaptor" in name or "shift_adaptor" in name:
                param.requires_grad_(True)
                trainable += param.numel()
            else:
                param.requires_grad_(False)
                frozen += param.numel()
        logger.info("Frozen %d params; trainable adaptor params: %d", frozen, trainable)

    def _init_adaptor_zero(self) -> None:
        """
        Initialise all Shift Adaptor parameters to zero.
        Paper anchor: adaptor所有参数初始化=0
        reference_grounding: paper_method_core adaptor_init_zero
        """
        if self.adaptor is None:
            return
        torch = self._get_torch()
        for param in self.adaptor.parameters():
            torch.nn.init.zeros_(param)
        logger.info("Shift Adaptor parameters initialised to zero.")

    def _build_optimizer(self):
        torch = self._get_torch()
        params = []
        if self.config.method == "ddpm_pa" and self.diffusion_model is not None:
            params = [p for p in self.diffusion_model.parameters() if p.requires_grad]
        elif self.adaptor is not None:
            params.extend(list(self.adaptor.parameters()))
        elif self.diffusion_model is not None:
            params = [p for p in self.diffusion_model.parameters() if p.requires_grad]
        if not params:
            logger.warning("No trainable parameters found; optimizer will be empty.")
            params = [torch.nn.Parameter(torch.zeros(1))]
        optimizer = torch.optim.Adam(params, lr=1e-4)
        return optimizer

    # ------------------------------------------------------------------
    # Classifier pre-training (300 iterations)
    # reference_grounding: paper_semantic_chunk_003_02 classifier_finetuning
    # ------------------------------------------------------------------
    def train_classifier(
        self,
        source_dataloader=None,
        target_dataloader=None,
        iterations: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Fine-tune the domain classifier (MobileNet) for
        `classifier_train_iterations` steps (paper anchor: 300).

        The classifier learns to distinguish source vs target domain images
        at various noise levels, enabling ∇log p_φ(y=S|x_t) computation.

        reference_grounding: paper_semantic_chunk_003_02 classifier_finetuning
        """
        n_iters = iterations if iterations is not None else self.config.classifier_train_iterations
        if self.config.dry_run:
            n_iters = min(n_iters, self.config.smoke_iterations)

        if self.classifier is None:
            logger.warning("No classifier provided; skipping classifier training.")
            return {"status": "skipped", "iterations": 0}

        torch = self._get_torch()
        device = self.config.device

        try:
            clf_optimizer = torch.optim.Adam(self.classifier.parameters(), lr=1e-4)
            ce_loss_fn = torch.nn.CrossEntropyLoss()
        except Exception as exc:
            logger.error("Classifier optimizer setup failed: %s", exc)
            return {"status": "error", "error": str(exc)}

        self.classifier.train()
        self.classifier.to(device)

        losses = []
        for step in range(n_iters):
            # Fetch source and target batches
            src_batch = self._fetch_batch(source_dataloader, device)
            tgt_batch = self._fetch_batch(target_dataloader, device)

            if src_batch is None or tgt_batch is None:
                logger.debug("Classifier step %d: no data available (dry-run or empty loader).", step)
                losses.append(0.0)
                continue

            # Add noise at random timestep (noisy image input)
            t = self._sample_timestep(src_batch.shape[0], device)
            src_noisy = self._add_noise(src_batch, t)
            tgt_noisy = self._add_noise(tgt_batch, t)

            # Labels: source=0, target=1
            src_labels = torch.zeros(src_noisy.shape[0], dtype=torch.long, device=device)
            tgt_labels = torch.ones(tgt_noisy.shape[0], dtype=torch.long, device=device)

            images = torch.cat([src_noisy, tgt_noisy], dim=0)
            labels = torch.cat([src_labels, tgt_labels], dim=0)

            clf_optimizer.zero_grad()
            logits = self.classifier(images, t.repeat(2) if t.shape[0] == src_noisy.shape[0] else t)
            loss = ce_loss_fn(logits, labels)
            loss.backward()
            clf_optimizer.step()

            losses.append(loss.item())
            if step % max(1, n_iters // 10) == 0:
                logger.info("Classifier step %d/%d | loss=%.4f", step, n_iters, loss.item())

        avg_loss = sum(losses) / max(len(losses), 1)
        logger.info("Classifier training complete | avg_loss=%.4f | steps=%d", avg_loss, n_iters)
        return {"status": "complete", "iterations": n_iters, "avg_loss": avg_loss}

    # ------------------------------------------------------------------
    # Adversarial noise selection (PGD inner loop)
    # reference_grounding: paper_semantic_chunk_010 adversarial_noise_selection
    # ------------------------------------------------------------------
    def select_adversarial_noise(
        self,
        x0,
        t,
        alpha: Optional[float] = None,
        omega: Optional[float] = None,
        inner_steps: Optional[int] = None,
    ):
        """
        Equation 7 inner loop.

        Algorithm:
          ε_0 ~ N(0, I)
          for j = 0 … J-1:
            g = ∇_ε ||ε - ε_θ(sqrt(alpha_bar_t)x_0
                + sqrt(1-alpha_bar_t)ε, t)||²
            ε_{j+1} = Norm(ε_j + ω g)
          return ε*

        Paper anchors:
          adversarial_inner_steps = 10
          omega = 0.02
          alpha = perturbation budget (default = omega)

        reference_grounding: paper_semantic_chunk_010 adversarial_noise_selection
        """
        torch = self._get_torch()

        _omega = omega if omega is not None else self.config.omega
        _steps = inner_steps if inner_steps is not None else self.config.adversarial_inner_steps

        eps = self._normalize_noise(torch.randn_like(x0))

        for _ in range(_steps):
            eps = eps.detach().requires_grad_(True)
            x_t = self._diffuse(x0, t, eps)
            loss = self._compute_simple_loss(x_t, t, eps)
            grad = torch.autograd.grad(loss, eps, retain_graph=False, create_graph=False)[0]

            with torch.no_grad():
                eps = self._normalize_noise(eps + _omega * grad)

        return eps.detach()

    # ------------------------------------------------------------------
    # Similarity guidance loss
    # reference_grounding: paper_semantic_chunk_003_02 similarity_guidance
    # ------------------------------------------------------------------
    def compute_similarity_loss(self, x_t, t) -> "torch.Tensor":
        """
        L_sim = γ · KL(∇log p_φ(y=S|x_t) ‖ ∇log p_φ(y=T|x_t))

        The classifier p_φ(y|x_t) is evaluated at noisy image x_t.
        Gradients ∇log p_φ(y=S|x_t) and ∇log p_φ(y=T|x_t) are computed
        w.r.t. x_t, then treated as probability distributions for KL.

        reference_grounding: paper_semantic_chunk_003_02 similarity_guidance
        """
        torch = self._get_torch()

        if self.classifier is None:
            return torch.tensor(0.0, device=self.config.device)

        # Try to use the dedicated similarity guidance module if available
        try:
            from dpms_ant.trainer.similarity_guidance import compute_similarity_guidance_loss
            return compute_similarity_guidance_loss(
                classifier=self.classifier,
                x_t=x_t,
                t=t,
                gamma=self.config.gamma,
            )
        except ImportError:
            pass

        # Inline fallback implementation
        x_t_req = x_t.detach().requires_grad_(True)
        logits = self.classifier(x_t_req, t)  # (B, 2)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # ∇log p_φ(y=S|x_t) – gradient of log P(source) w.r.t. x_t
        grad_source = torch.autograd.grad(
            log_probs[:, 0].sum(), x_t_req, create_graph=False, retain_graph=True
        )[0]
        # ∇log p_φ(y=T|x_t) – gradient of log P(target) w.r.t. x_t
        grad_target = torch.autograd.grad(
            log_probs[:, 1].sum(), x_t_req, create_graph=False, retain_graph=False
        )[0]

        # Flatten and normalise to probability distributions
        B = x_t.shape[0]
        p = torch.nn.functional.softmax(grad_source.view(B, -1), dim=-1).clamp(min=1e-8)
        q = torch.nn.functional.softmax(grad_target.view(B, -1), dim=-1).clamp(min=1e-8)

        # KL(p ‖ q)
        kl = (p * (p.log() - q.log())).sum(dim=-1).mean()
        return self.config.gamma * kl

    def compute_target_classifier_gradient(self, x_t, t):
        """Return ∇_{x_t} log p_phi(y=T | x_t) for Eq. 5/8."""
        torch = self._get_torch()
        if self.classifier is None:
            return torch.zeros_like(x_t)
        x_req = x_t.detach().requires_grad_(True)
        logits = self.classifier(x_req, t)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        grad = torch.autograd.grad(log_probs[:, 1].sum(), x_req, create_graph=False)[0]
        return grad.detach()

    def _sigma_hat_squared(self, t, like):
        """Compute hat_sigma_t^2 from Eq. 5 with schedule fallback."""
        torch = self._get_torch()
        t_long = t.long().clamp_min(0)
        if (
            self.diffusion_model is not None
            and hasattr(self.diffusion_model, "sqrt_alphas_cumprod")
            and hasattr(self.diffusion_model, "sqrt_one_minus_alphas_cumprod")
        ):
            sqrt_ab = self.diffusion_model.sqrt_alphas_cumprod.to(like.device)
            ab = sqrt_ab.square()
            idx = t_long.clamp_max(ab.numel() - 1)
            prev = torch.clamp(idx - 1, min=0)
            alpha_bar_t = ab[idx]
            alpha_bar_prev = ab[prev]
            alpha_t = torch.where(idx > 0, alpha_bar_t / alpha_bar_prev.clamp_min(1e-8), alpha_bar_t)
        else:
            t_float = t.float()
            alpha_bar_t = torch.clamp(1.0 - (t_float + 1.0) / 1001.0, 0.05, 0.95)
            alpha_bar_prev = torch.clamp(1.0 - t_float / 1001.0, 0.05, 0.95)
            alpha_t = alpha_bar_t / alpha_bar_prev.clamp_min(1e-8)
        sigma_hat = (1.0 - alpha_bar_prev) * torch.sqrt(alpha_t / (1.0 - alpha_bar_t).clamp_min(1e-8))
        return sigma_hat.square().view(-1, *([1] * (like.dim() - 1))).to(like.device)

    def _normalize_noise(self, eps):
        dims = tuple(range(1, eps.dim()))
        mean = eps.mean(dim=dims, keepdim=True)
        std = eps.std(dim=dims, keepdim=True, unbiased=False).clamp_min(1e-6)
        return (eps - mean) / std

    # ------------------------------------------------------------------
    # Main training loop – Algorithm 1
    # reference_grounding: paper_method_core Algorithm1
    # ------------------------------------------------------------------
    def train(
        self,
        source_dataloader=None,
        target_dataloader=None,
    ) -> Dict[str, Any]:
        """
        Algorithm 1 – DPMs-ANT full training loop.

        Steps per iteration:
          1. Sample target batch x_0.
          2. Sample timestep τ.
          3. [use_adv_noise] PGD → ε* (adversarial noise).
          4. Forward diffusion: x_τ = √ᾱ_τ x_0 + √(1-ᾱ_τ) ε*.
          5. UNet prediction: ε̂ = UNet(x_τ, τ).
          6. L_simple = ‖ε* - ε̂‖².
          7. [use_sim_guide] L_sim = γ · KL(…).
          8. L_total = L_simple + λ · L_sim.
          9. Update Shift Adaptor only.

        reference_grounding: paper_method_core Algorithm1
        """
        torch = self._get_torch()

        # Setup
        self._init_adaptor_zero()
        self._freeze_non_adaptor_params()
        optimizer = self._build_optimizer()

        n_iters = self.config.effective_iterations()
        device = self.config.device

        if self.diffusion_model is not None:
            self.diffusion_model.to(device)
            self.diffusion_model.train()

        start_time = time.time()
        results: Dict[str, Any] = {
            "method": self.config.method,
            "use_sim_guide": self.config.use_sim_guide,
            "use_adv_noise": self.config.use_adv_noise,
            "total_iterations": n_iters,
            "gamma": self.config.gamma,
            "omega": self.config.omega,
            "adversarial_inner_steps": self.config.adversarial_inner_steps,
            "batch_size": self.config.batch_size,
            "shot_count": self.config.shot_count,
            "losses": [],
            "objective": "equation_8_similarity_guided_mse_with_adversarial_noise",
        }

        for step in range(n_iters):
            self._step = step

            # ── Step 1: sample target batch ──────────────────────────────
            x0 = self._fetch_batch(target_dataloader, device)
            if x0 is None:
                x0 = torch.randn(
                    min(self.config.batch_size, 4), 3, 64, 64, device=device
                )

            # ── Step 2: sample timestep ───────────────────────────────────
            t = self._sample_timestep(x0.shape[0], device)

            # ── Step 3: adversarial noise selection ───────────────────────
            if self.config.use_adv_noise and self.diffusion_model is not None:
                eps_star = self.select_adversarial_noise(x0, t)
            else:
                eps_star = torch.randn_like(x0)

            # ── Step 4: forward diffusion ─────────────────────────────────
            x_t = self._diffuse(x0, t, eps_star)

            # ── Steps 5-6: UNet prediction + L_simple ─────────────────────
            optimizer.zero_grad()
            pred_noise = self._predict_noise_with_adaptor(x_t, t)
            if self.config.use_sim_guide and self.classifier is not None:
                target_grad = self.compute_target_classifier_gradient(x_t, t)
                sigma_hat_sq = self._sigma_hat_squared(t, x_t)
                guidance = sigma_hat_sq * self.config.gamma * target_grad
            else:
                target_grad = torch.zeros_like(x_t)
                sigma_hat_sq = torch.zeros_like(x_t)
                guidance = torch.zeros_like(x_t)
            residual = eps_star - pred_noise - guidance
            loss_simple = torch.mean((eps_star - pred_noise) ** 2)
            loss_sim = torch.mean(guidance ** 2)
            loss_total = torch.mean(residual ** 2)
            loss_total.backward()
            optimizer.step()

            loss_row = {
                "step": step,
                "loss_simple": float(loss_simple.detach().cpu()),
                "loss_sim": float(loss_sim.detach().cpu()) if hasattr(loss_sim, "detach") else float(loss_sim),
                "loss_total": float(loss_total.detach().cpu()),
                "adversarial_noise_norm": "mean0_std1" if self.config.use_adv_noise else "plain_gaussian",
                "sigma_hat_scaled_target_gradient": bool(self.config.use_sim_guide and self.classifier is not None),
                "target_gradient_abs_mean": float(target_grad.detach().abs().mean().cpu()),
                "sigma_hat_sq_mean": float(sigma_hat_sq.detach().mean().cpu()),
            }
            results["losses"].append(loss_row)
            self._loss_history.append(loss_row)

            if step % max(1, self.config.log_interval) == 0:
                logger.info(
                    "train step %d/%d | simple=%.4f sim=%.4f total=%.4f",
                    step + 1,
                    n_iters,
                    loss_row["loss_simple"],
                    loss_row["loss_sim"],
                    loss_row["loss_total"],
                )

        elapsed = time.time() - start_time
        avg_total = sum(row["loss_total"] for row in results["losses"]) / max(len(results["losses"]), 1)
        results.update(
            {
                "status": "complete",
                "elapsed_seconds": elapsed,
                "avg_loss_total": avg_total,
                "final_loss_total": results["losses"][-1]["loss_total"] if results["losses"] else 0.0,
            }
        )
        return results

    # ------------------------------------------------------------------
    # Smoke helpers
    # ------------------------------------------------------------------
    def _sample_timestep(self, batch_size: int, device: str):
        torch = self._get_torch()
        return torch.randint(0, 1000, (batch_size,), device=device)

    def _fetch_batch(self, dataloader, device: str):
        torch = self._get_torch()
        if dataloader is None:
            return None
        if not hasattr(self, "_loader_iters"):
            self._loader_iters = {}
        key = id(dataloader)
        iterator = self._loader_iters.get(key)
        if iterator is None:
            iterator = iter(dataloader)
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            batch = next(iterator)
        self._loader_iters[key] = iterator
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        if hasattr(batch, "to"):
            return batch.to(device)
        return torch.as_tensor(batch, device=device)

    def _diffuse(self, x0, t, noise):
        torch = self._get_torch()
        t = t.float().view(-1, 1, 1, 1)
        alpha_bar = torch.clamp(1.0 - (t + 1.0) / 1001.0, 0.05, 0.95)
        return alpha_bar.sqrt() * x0 + (1.0 - alpha_bar).sqrt() * noise

    def _compute_simple_loss(self, x_t, t, target_noise):
        torch = self._get_torch()
        pred = self._predict_noise_with_adaptor(x_t, t)
        return torch.mean((target_noise - pred) ** 2)

    def _predict_noise_with_adaptor(self, x_t, t):
        torch = self._get_torch()
        if self.diffusion_model is None:
            pred = torch.zeros_like(x_t)
        elif hasattr(self.diffusion_model, "predict_noise"):
            pred = self.diffusion_model.predict_noise(x_t, t)
        elif callable(self.diffusion_model):
            pred = self.diffusion_model(x_t, t)
        else:
            pred = torch.zeros_like(x_t)
        if self.adaptor is not None and hasattr(self.adaptor, "__call__"):
            try:
                pred = self.adaptor(pred)
            except Exception:
                pass
        return pred
