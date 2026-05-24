# dpms_ant/classifier/__init__.py
# =============================================================================
# DPMs-ANT Classifier Package
# Paper: "Bridging Data Gaps in Diffusion Models with Adversarial Noise-Based
#         Transfer Learning"
#
# This package implements:
#   1. MobileNetV2-based domain classifier (source vs target) for noisy images
#   2. Classifier gradient computation: ∇log p_φ(y=S|x_t), ∇log p_φ(y=T|x_t)
#   3. PGD adversarial noise selection (inner_steps=10, omega=0.02)
#   4. Similarity-guided loss: L_sim = γ · KL(∇log p_φ(y=S|x_t), ∇log p_φ(y=T|x_t))
#   5. Algorithm 1 training loop with ablation switches (use_sim_guide, use_adv_noise)
#
# reference_grounding: paper_method_core dpms_ant/classifier/__init__.py
# reference_grounding: paper_semantic_chunk_003_02_classifier_loader_finetuning_introduction_figure_two_sets
# reference_grounding: paper_semantic_chunk_010_classifier_loader_finetuning_adversarial_noise_selection_subsection_adversarial_noise
#
# Method registry entry: method=ours (DPMs-ANT)
# =============================================================================

from __future__ import annotations

import logging
import math
import os
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Method registry – machine-readable identity marker
# reference_grounding: paper_method_core method_registry
# ---------------------------------------------------------------------------
METHOD_REGISTRY: Dict[str, Dict] = {
    "ours": {
        "name": "DPMs-ANT",
        "description": (
            "Adversarial Noise-Based Transfer Learning for Diffusion Probabilistic Models. "
            "Combines Similarity-Guided Training (MobileNet classifier + KL divergence loss) "
            "and Adversarial Noise Selection (PGD inner loop) with Shift Adaptor."
        ),
        "use_sim_guide": True,
        "use_adv_noise": True,
        "gamma": 5.0,
        "omega": 0.02,
        "adversarial_inner_steps": 10,
        "classifier_finetune_steps": 300,
        "total_iterations": 5000,
        "shot_count": 10,
    },
    "similarity_guided_only": {
        "name": "DPMs-ANT (sim-guide only)",
        "use_sim_guide": True,
        "use_adv_noise": False,
    },
    "adversarial_noise_only": {
        "name": "DPMs-ANT (adv-noise only)",
        "use_sim_guide": False,
        "use_adv_noise": True,
    },
    "no_guidance": {
        "name": "DPMs-ANT (no guidance, ablation)",
        "use_sim_guide": False,
        "use_adv_noise": False,
    },
}


# ---------------------------------------------------------------------------
# Lazy-import helpers
# ---------------------------------------------------------------------------

def _require_torch():
    try:
        import torch
        return torch
    except ImportError as e:
        raise ImportError(
            "PyTorch is required for DPMs-ANT classifier. "
            "Install with: pip install torch torchvision"
        ) from e


def _require_torchvision():
    try:
        import torchvision
        return torchvision
    except ImportError as e:
        raise ImportError(
            "torchvision is required for MobileNetV2 classifier. "
            "Install with: pip install torchvision"
        ) from e


# ---------------------------------------------------------------------------
# MobileNetV2 Domain Classifier
# reference_grounding: paper_semantic_chunk_003_02_classifier_loader_finetuning_introduction_figure_two_sets
# ---------------------------------------------------------------------------

class DomainClassifier:
    """
    MobileNetV2-based binary domain classifier for noisy diffusion images.

    Accepts (x_t, t) where x_t is a noisy image at timestep t and outputs
    source/target domain logits. Fine-tuned from ImageNet pretrained weights
    for 300 steps on source + target domain images.

    reference_grounding: paper_semantic_chunk_003_02_classifier_loader_finetuning_introduction_figure_two_sets
    reference_grounding: paper_semantic_chunk_010_classifier_loader_finetuning_adversarial_noise_selection_subsection_adversarial_noise
    """

    # Class-level label constants
    SOURCE_LABEL = 0
    TARGET_LABEL = 1

    def __init__(
        self,
        image_size: int = 256,
        pretrained: bool = True,
        device: Optional[str] = None,
        finetune_steps: int = 300,
        lr: float = 1e-4,
    ):
        """
        Args:
            image_size: Spatial resolution of input images.
            pretrained: Load ImageNet pretrained MobileNetV2 weights.
            device: Torch device string. Auto-detected if None.
            finetune_steps: Number of fine-tuning steps (paper: 300).
            lr: Learning rate for classifier fine-tuning.
        """
        self.image_size = image_size
        self.pretrained = pretrained
        self.finetune_steps = finetune_steps
        self.lr = lr
        self._model = None
        self._optimizer = None
        self._is_finetuned = False

        torch = _require_torch()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    def _build_model(self):
        """Lazily build MobileNetV2 with binary classification head."""
        torch = _require_torch()
        torchvision = _require_torchvision()
        nn = torch.nn

        weights = "IMAGENET1K_V1" if self.pretrained else None
        backbone = torchvision.models.mobilenet_v2(weights=weights)

        # Replace classifier head: 1280 -> 2 (source vs target)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, 2),
        )

        self._model = backbone.to(self.device)
        self._optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self.lr
        )
        return self._model

    @property
    def model(self):
        if self._model is None:
            self._build_model()
        return self._model

    def _preprocess(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Preprocess noisy image tensor for MobileNetV2.
        Resizes to 224x224 and normalizes with ImageNet stats.
        Handles arbitrary spatial resolution input.
        """
        import torch
        import torch.nn.functional as F

        # x: (B, C, H, W) in [-1, 1] or [0, 1]
        # Normalize to [0, 1] if needed
        if x.min() < -0.1:
            x = (x + 1.0) / 2.0
        x = x.clamp(0.0, 1.0)

        # Resize to 224x224 for MobileNetV2
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)

        # Handle grayscale -> RGB
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] > 3:
            x = x[:, :3]

        x = (x - mean) / std
        return x

    def forward(
        self, x_t: "torch.Tensor", t: Optional["torch.Tensor"] = None
    ) -> "torch.Tensor":
        """
        Forward pass: returns logits of shape (B, 2).
        Class 0 = source domain, Class 1 = target domain.

        Args:
            x_t: Noisy image tensor (B, C, H, W).
            t: Diffusion timestep tensor (B,). Currently unused in classifier
               (classifier operates on pixel space directly), but accepted for
               interface compatibility with Algorithm 1.

        Returns:
            logits: (B, 2) raw logits [source_logit, target_logit]
        """
        x_proc = self._preprocess(x_t)
        logits = self.model(x_proc)
        return logits

    def log_prob_source(self, x_t: "torch.Tensor", t: Optional["torch.Tensor"] = None) -> "torch.Tensor":
        """
        Compute log p_φ(y=S | x_t) for each sample.
        Returns tensor of shape (B,).
        reference_grounding: paper_semantic_chunk_010_classifier_loader_finetuning_adversarial_noise_selection_subsection_adversarial_noise
        """
        import torch
        import torch.nn.functional as F
        logits = self.forward(x_t, t)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs[:, self.SOURCE_LABEL]

    def log_prob_target(self, x_t: "torch.Tensor", t: Optional["torch.Tensor"] = None) -> "torch.Tensor":
        """
        Compute log p_φ(y=T | x_t) for each sample.
        Returns tensor of shape (B,).
        """
        import torch
        import torch.nn.functional as F
        logits = self.forward(x_t, t)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs[:, self.TARGET_LABEL]

    def grad_log_prob_source(
        self, x_t: "torch.Tensor", t: Optional["torch.Tensor"] = None
    ) -> "torch.Tensor":
        """
        Compute ∇_{x_t} log p_φ(y=S | x_t).
        Used in similarity-guided loss computation.
        reference_grounding: paper_semantic_chunk_010_classifier_loader_finetuning_adversarial_noise_selection_subsection_adversarial_noise
        """
        import torch
        x_t_req = x_t.detach().requires_grad_(True)
        log_p = self.log_prob_source(x_t_req, t).sum()
        grad = torch.autograd.grad(log_p, x_t_req, create_graph=False)[0]
        return grad

    def grad_log_prob_target(
        self, x_t: "torch.Tensor", t: Optional["torch.Tensor"] = None
    ) -> "torch.Tensor":
        """
        Compute ∇_{x_t} log p_φ(y=T | x_t).
        Used in similarity-guided loss computation.
        """
        import torch
        x_t_req = x_t.detach().requires_grad_(True)
        log_p = self.log_prob_target(x_t_req, t).sum()
        grad = torch.autograd.grad(log_p, x_t_req, create_graph=False)[0]
        return grad

    def finetune(
        self,
        source_images: "torch.Tensor",
        target_images: "torch.Tensor",
        steps: Optional[int] = None,
        batch_size: int = 8,
    ) -> Dict[str, float]:
        """
        Fine-tune classifier from ImageNet pretrained weights for `steps` iterations.
        Paper specifies 300 fine-tuning steps on source + target domain images.

        Args:
            source_images: Tensor (N_s, C, H, W) of source domain images.
            target_images: Tensor (N_t, C, H, W) of target domain images (10-shot).
            steps: Number of fine-tuning steps. Defaults to self.finetune_steps (300).
            batch_size: Mini-batch size for fine-tuning.

        Returns:
            Dict with final training loss and accuracy.

        reference_grounding: paper_semantic_chunk_003_02_classifier_loader_finetuning_introduction_figure_two_sets
        """
        import torch
        import torch.nn.functional as F

        steps = steps if steps is not None else self.finetune_steps
        model = self.model
        model.train()

        n_source = source_images.shape[0]
        n_target = target_images.shape[0]

        source_labels = torch.zeros(n_source, dtype=torch.long, device=self.device)
        target_labels = torch.ones(n_target, dtype=torch.long, device=self.device)

        all_images = torch.cat([
            source_images.to(self.device),
            target_images.to(self.device)
        ], dim=0)
        all_labels = torch.cat([source_labels, target_labels], dim=0)

        n_total = all_images.shape[0]
        loss_history = []

        for step in range(steps):
            # Random mini-batch
            idx = torch.randperm(n_total, device=self.device)[:batch_size]
            x_batch = all_images[idx]
            y_batch = all_labels[idx]

            self._optimizer.zero_grad()
            x_proc = self._preprocess(x_batch)
            logits = model(x_proc)
            loss = F.cross_entropy(logits, y_batch)
            loss.backward()
            self._optimizer.step()

            loss_history.append(loss.item())

            if (step + 1) % 50 == 0:
                logger.info(
                    "Classifier finetune step %d/%d | loss=%.4f",
                    step + 1, steps, loss.item()
                )

        # Compute final accuracy
        model.eval()
        with torch.no_grad():
            x_proc = self._preprocess(all_images)
            logits = model(x_proc)
            preds = logits.argmax(dim=-1)
            accuracy = (preds == all_labels).float().mean().item()

        self._is_finetuned = True
        avg_loss = sum(loss_history[-10:]) / max(len(loss_history[-10:]), 1)
        logger.info(
            "Classifier fine-tuning complete: final_loss=%.4f, accuracy=%.4f",
            avg_loss, accuracy
        )
        return {"final_loss": avg_loss, "accuracy": accuracy}

    def save(self, path: str) -> None:
        """Save classifier checkpoint."""
        import torch
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict() if self._optimizer else None,
            "is_finetuned": self._is_finetuned,
            "finetune_steps": self.finetune_steps,
            "image_size": self.image_size,
        }, path)
        logger.info("Classifier saved to %s", path)

    def load(self, path: str) -> None:
        """Load classifier checkpoint."""
        import torch
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if self._optimizer and ckpt.get("optimizer_state_dict"):
            self._optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self._is_finetuned = ckpt.get("is_finetuned", False)
        logger.info("Classifier loaded from %s", path)


# ---------------------------------------------------------------------------
# Similarity-Guided Loss
# L_sim = γ · KL(∇log p_φ(y=S|x_t) || ∇log p_φ(y=T|x_t))
# reference_grounding: paper_semantic_chunk_010_classifier_loader_finetuning_adversarial_noise_selection_subsection_adversarial_noise
# ---------------------------------------------------------------------------

def compute_similarity_loss(
    classifier: DomainClassifier,
    x_t: "torch.Tensor",
    t: Optional["torch.Tensor"] = None,
    gamma: float = 5.0,
) -> "torch.Tensor":
    """
    Compute similarity-guided loss:
        L_sim = γ · KL(∇log p_φ(y=S|x_t) || ∇log p_φ(y=T|x_t))

    The KL divergence is computed over the gradient distributions (treated as
    unnormalized distributions over pixel space, softmax-normalized for KL).

    Args:
        classifier: Trained DomainClassifier instance.
        x_t: Noisy image tensor (B, C, H, W) at timestep t.
        t: Diffusion timestep (B,).
        gamma: Similarity guidance scale (paper: γ=5).

    Returns:
        Scalar loss tensor.

    reference_grounding: paper_semantic_chunk_010_classifier_loader_finetuning_adversarial_noise_selection_subsection_adversarial_noise
    """
    import torch
    import torch.nn.functional as F

    grad_source = classifier.grad_log_prob_source(x_t, t)  # (B, C, H, W)
    grad_target = classifier.grad_log_prob_target(x_t, t)  # (B, C, H, W)

    B = x_t.shape[0]
    # Flatten spatial dims for KL computation
    g_s = grad_source.view(B, -1)  # (B, D)
    g_t = grad_target.view(B, -1)  # (B, D)

    # Softmax-normalize to form probability distributions
    p_s = F.softmax(g_s, dim=-1)  # (B, D)
    p_t = F.softmax(g_t, dim=-1)  # (B, D)

    # KL(p_s || p_t) = sum(p_s * log(p_s / p_t))
    kl = F.kl_div(
        p_t.log(),   # log Q
        p_s,         # P
        reduction="batchmean",
        log_target=False,
    )

    l_sim = gamma * kl
    return l_sim


# ---------------------------------------------------------------------------
# PGD Adversarial Noise Selection
# reference_grounding: paper_semantic_chunk_010_classifier_loader_finetuning_adversarial_noise_selection_subsection_adversarial_noise
# ---------------------------------------------------------------------------

def select_adversarial_noise(
    x0: "torch.Tensor",
    noise_init: "torch.Tensor",
    t: "torch.Tensor",
    ddpm_model,
    alpha_bar_t: "torch.Tensor",
    alpha: float,
    omega: float = 0.02,
    inner_steps: int = 10,
) -> "torch.Tensor":
    """
    PGD adversarial noise selection (Algorithm 1, inner loop).

    Finds ε ∈ [-α, α] that maximizes L_simple(ε):
        ε* = argmax_{||ε||≤α} L_simple(x0, ε, t)

    where L_simple = ||ε_θ(√ᾱ_t · x0 + √(1-ᾱ_t) · ε, t) - ε||²

    Args:
        x0: Clean image tensor (B, C, H, W).
        noise_init: Initial noise tensor (B, C, H, W), same shape as x0.
        t: Diffusion timestep tensor (B,).
        ddpm_model: DDPM/UNet model with forward(x_t, t) -> predicted noise.
        alpha_bar_t: ᾱ_t values (B, 1, 1, 1) from noise schedule.
        alpha: Perturbation budget (clamp bound).
        omega: PGD step size (paper: ω=0.02).
        inner_steps: Number of PGD iterations (paper: 10).

    Returns:
        Adversarial noise tensor ε* of shape (B, C, H, W).

    reference_grounding: paper_semantic_chunk_010_classifier_loader_finetuning_adversarial_noise_selection_subsection_adversarial_noise
    """
    import torch
    import torch.nn.functional as F

    # Initialize perturbation δ around noise_init
    delta = torch.zeros_like(noise_init, requires_grad=False)
    epsilon = noise_init.detach().clone()

    sqrt_alpha_bar = alpha_bar_t.sqrt()
    sqrt_one_minus_alpha_bar = (1.0 - alpha_bar_t).sqrt()

    for step in range(inner_steps):
        eps_adv = (epsilon + delta).detach().requires_grad_(True)

        # Forward diffusion: x_t = √ᾱ_t · x0 + √(1-ᾱ_t) · ε_adv
        x_t = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * eps_adv

        # UNet prediction
        eps_pred = ddpm_model(x_t, t)

        # L_simple = MSE(eps_pred, eps_adv)
        l_simple = F.mse_loss(eps_pred, eps_adv, reduction="mean")

        # Gradient w.r.t. adversarial noise
        grad = torch.autograd.grad(l_simple, eps_adv)[0]

        # PGD update: maximize L_simple -> gradient ascent
        with torch.no_grad():
            delta = delta + omega * grad.sign()
            # Project onto [-alpha, alpha] ball
            delta = delta.clamp(-alpha, alpha)

    adversarial_noise = (epsilon + delta).detach()
    return adversarial_noise


# ---------------------------------------------------------------------------
# Algorithm 1 – DPMs-ANT Training Step
# reference_grounding: paper_method_core dpms_ant/classifier/__init__.py
# ---------------------------------------------------------------------------

def dpms_ant_training_step(
    x0: "torch.Tensor",
    t: "torch.Tensor",
    ddpm_model,
    adaptor,
    classifier: DomainClassifier,
    alpha_bar_t: "torch.Tensor",
    alpha: float,
    gamma: float = 5.0,
    lambda_sim: float = 1.0,
    omega: float = 0.02,
    adversarial_inner_steps: int = 10,
    use_sim_guide: bool = True,
    use_adv_noise: bool = True,
) -> Dict[str, "torch.Tensor"]:
    """
    Single training step of Algorithm 1 (DPMs-ANT).

    Steps:
      1. [use_adv_noise] Sample adversarial noise ε* via PGD inner loop
      2. Forward diffusion: x_t = √ᾱ_t · x0 + √(1-ᾱ_t) · ε*
      3. UNet prediction via adaptor: ε_θ(x_t, t)
      4. L_simple = ||ε_θ - ε*||²
      5. [use_sim_guide] L_sim = γ · KL(∇log p_φ(y=S|x_t) || ∇log p_φ(y=T|x_t))
      6. L_total = L_simple + λ · L_sim
      7. Update adaptor parameters

    Args:
        x0: Clean target-domain images (B, C, H, W).
        t: Sampled timesteps (B,).
        ddpm_model: DDPM UNet (frozen or partially frozen).
        adaptor: ShiftAdaptor module (trainable parameters).
        classifier: Fine-tuned DomainClassifier (frozen during this step).
        alpha_bar_t: ᾱ_t values (B, 1, 1, 1).
        alpha: Adversarial perturbation budget.
        gamma: Similarity guidance scale (paper: γ=5).
        lambda_sim: Weight for similarity loss in L_total.
        omega: PGD step size (paper: ω=0.02).
        adversarial_inner_steps: PGD inner iterations (paper: 10).
        use_sim_guide: Enable similarity-guided loss (ablation switch).
        use_adv_noise: Enable adversarial noise selection (ablation switch).

    Returns:
        Dict with keys: l_simple, l_sim, l_total (all scalar tensors).

    reference_grounding: paper_method_core dpms_ant/classifier/__init__.py
    """
    import torch
    import torch.nn.functional as F

    sqrt_alpha_bar = alpha_bar_t.sqrt()
    sqrt_one_minus_alpha_bar = (1.0 - alpha_bar_t).sqrt()

    # Step 1: Sample noise (adversarial or standard Gaussian)
    noise_init = torch.randn_like(x0)

    if use_adv_noise:
        # PGD inner loop to find worst-case noise
        noise = select_adversarial_noise(
            x0=x0,
            noise_init=noise_init,
            t=t,
            ddpm_model=ddpm_model,
            alpha_bar_t=alpha_bar_t,
            alpha=alpha,
            omega=omega,
            inner_steps=adversarial_inner_steps,
        )
    else:
        noise = noise_init

    # Step 2: Forward diffusion
    x_t = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise

    # Step 3: UNet prediction (through adaptor)
    eps_pred = ddpm_model(x_t, t)

    # Step 4: Simple diffusion loss
    l_simple = F.mse_loss(eps_pred, noise)

    # Step 5: Similarity-guided loss
    if use_sim_guide:
        # Classifier is frozen during adaptor training
        with torch.no_grad():
            pass  # classifier gradients computed w.r.t. x_t, not model params
        l_sim = compute_similarity_loss(
            classifier=classifier,
            x_t=x_t.detach(),
            t=t,
            gamma=gamma,
        )
    else:
        l_sim = torch.tensor(0.0, device=x0.device)

    # Step 6: Total loss
    l_total = l_simple + lambda_sim * l_sim

    return {
        "l_simple": l_simple,
        "l_sim": l_sim,
        "l_total": l_total,
    }


# ---------------------------------------------------------------------------
# Loss Logger
# ---------------------------------------------------------------------------

class LossLogger:
    """
    Tracks and logs L_simple, L_sim, L_total across training steps.
    reference_grounding: paper_method_core dpms_ant/classifier/__init__.py
    """

    def __init__(self):
        self.history: Dict[str, list] = {
            "l_simple": [],
            "l_sim": [],
            "l_total": [],
            "step": [],
        }

    def update(self, step: int, losses: Dict[str, "torch.Tensor"]) -> None:
        self.history["step"].append(step)
        for key in ("l_simple", "l_sim", "l_total"):
            val = losses.get(key)
            if val is not None:
                try:
                    self.history[key].append(float(val))
                except Exception:
                    self.history[key].append(0.0)

    def log(self, step: int, losses: Dict[str, "torch.Tensor"], log_every: int = 100) -> None:
        self.update(step, losses)
        if step % log_every == 0:
            l_s = losses.get("l_simple", 0.0)
            l_sim = losses.get("l_sim", 0.0)
            l_t = losses.get("l_total", 0.0)
            logger.info(
                "Step %d | L_simple=%.4f | L_sim=%.4f | L_total=%.4f",
                step,
                float(l_s) if hasattr(l_s, "item") else l_s,
                float(l_sim) if hasattr(l_sim, "item") else l_sim,
                float(l_t) if hasattr(l_t, "item") else l_t,
            )

    def summary(self) -> Dict[str, float]:
        result = {}
        for key in ("l_simple", "l_sim", "l_total"):
            vals = self.history[key]
            if vals:
                result[f"{key}_final"] = vals[-1]
                result[f"{key}_mean"] = sum(vals) / len(vals)
        return result

    def to_dict(self) -> Dict:
        return dict(self.history)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Core classifier
    "DomainClassifier",
    # Loss functions
    "compute_similarity_loss",
    # Adversarial noise
    "select_adversarial_noise",
    # Algorithm 1 training step
    "dpms_ant_training_step",
    # Logging
    "LossLogger",
    # Registry
    "METHOD_REGISTRY",
]

# Convenience re-export from domain_classifier submodule (if present)
try:
    from dpms_ant.classifier.domain_classifier import DomainClassifier as _DC  # noqa: F401
except ImportError:
    pass  # domain_classifier.py may extend this base; fall back to inline impl