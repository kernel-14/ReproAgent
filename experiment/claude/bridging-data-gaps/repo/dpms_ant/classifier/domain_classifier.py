"""
dpms_ant/classifier/domain_classifier.py

Domain classifier for DPMs-ANT: MobileNet-based source/target binary classifier
that operates on noisy images x_t to provide similarity guidance and adversarial
noise selection signals.

reference_grounding: paper_semantic_chunk_003_02_classifier_loader_finetuning_introduction_figure_two_sets
reference_grounding: paper_semantic_chunk_010_classifier_loader_finetuning_adversarial_noise_selection_subsection_adversarial_noise
reference_grounding: paper_method_core dpms_ant/classifier/domain_classifier.py

Method: DPMs-ANT (ours)
  - Similarity-guided training: L_sim = γ · KL(∇log p_φ(y=S|x_t) ‖ ∇log p_φ(y=T|x_t))
  - Adversarial noise selection: PGD inner loop, inner_steps=10, omega=0.02
  - Classifier fine-tuned from ImageNet MobileNet weights for 300 steps
  - Supports noisy image input (x_t at arbitrary diffusion timestep t)
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry marker – method=ours (DPMs-ANT)
# reference_grounding: paper_method_core method_registry
# ---------------------------------------------------------------------------
METHOD_ID = "dpms_ant"
METHOD_DISPLAY_NAME = "DPMs-ANT (ours)"
CLASSIFIER_FINETUNE_STEPS = 300   # paper addendum: 300 training iterations for classifier
SIMILARITY_GUIDANCE_SCALE = 5.0   # γ = 5  (paper eq. for L_sim)
ADV_INNER_STEPS = 10              # PGD inner steps (paper addendum)
ADV_OMEGA = 0.02                  # PGD step size ω = 0.02 (paper addendum)

# Source class index = 0, Target class index = 1
CLASS_SOURCE = 0
CLASS_TARGET = 1


# ---------------------------------------------------------------------------
# Lazy import helpers
# ---------------------------------------------------------------------------

def _require_torch():
    try:
        import torch
        return torch
    except ImportError as e:
        raise ImportError(
            "PyTorch is required for DomainClassifier. "
            "Install it with: pip install torch torchvision"
        ) from e


def _require_torchvision():
    try:
        import torchvision
        return torchvision
    except ImportError as e:
        raise ImportError(
            "torchvision is required for DomainClassifier. "
            "Install it with: pip install torchvision"
        ) from e


# ---------------------------------------------------------------------------
# DomainClassifier
# ---------------------------------------------------------------------------

class DomainClassifier:
    """
    MobileNetV2-based binary domain classifier φ.

    Accepts noisy images x_t (arbitrary diffusion timestep) and outputs
    log-probabilities for source (y=S) and target (y=T) classes.

    Fine-tuned from ImageNet pretrained weights for CLASSIFIER_FINETUNE_STEPS
    steps on a mix of source-domain samples and target few-shot samples.

    reference_grounding: paper_semantic_chunk_003_02_classifier_loader_finetuning_introduction_figure_two_sets
    """

    def __init__(
        self,
        device: Optional[str] = None,
        pretrained: bool = True,
        finetune_steps: int = CLASSIFIER_FINETUNE_STEPS,
        lr: float = 1e-4,
        image_size: int = 256,
    ):
        torch = _require_torch()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.finetune_steps = finetune_steps
        self.lr = lr
        self.image_size = image_size
        self._model = None
        self._pretrained = pretrained
        self._is_finetuned = False

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _build_model(self):
        """Build MobileNetV2 with a 2-class head for source/target classification."""
        torch = _require_torch()
        tv = _require_torchvision()
        import torch.nn as nn

        weights = tv.models.MobileNet_V2_Weights.IMAGENET1K_V1 if self._pretrained else None
        backbone = tv.models.mobilenet_v2(weights=weights)

        # Replace classifier head: 1280 → 2 (source vs target)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, 2),
        )
        backbone = backbone.to(self.device)
        return backbone

    def get_model(self):
        """Lazy-initialize and return the underlying nn.Module."""
        if self._model is None:
            self._model = self._build_model()
        return self._model

    # ------------------------------------------------------------------
    # Fine-tuning
    # ------------------------------------------------------------------

    def finetune(
        self,
        source_loader,
        target_loader,
        steps: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Fine-tune the classifier on source (label=0) and target (label=1) images
        for `steps` gradient steps (default: self.finetune_steps = 300).

        reference_grounding: paper_semantic_chunk_003_02_classifier_loader_finetuning_introduction_figure_two_sets

        Args:
            source_loader: DataLoader yielding source-domain images (tensors in [-1,1]).
            target_loader: DataLoader yielding target few-shot images (tensors in [-1,1]).
            steps: Number of fine-tuning gradient steps (paper: 300).

        Returns:
            dict with final training loss.
        """
        torch = _require_torch()
        import torch.nn as nn

        steps = steps if steps is not None else self.finetune_steps
        model = self.get_model()
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        source_iter = iter(source_loader)
        target_iter = iter(target_loader)

        total_loss = 0.0
        for step in range(steps):
            # Fetch source batch
            try:
                src_imgs = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                src_imgs = next(source_iter)

            # Fetch target batch
            try:
                tgt_imgs = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                tgt_imgs = next(target_iter)

            if isinstance(src_imgs, (list, tuple)):
                src_imgs = src_imgs[0]
            if isinstance(tgt_imgs, (list, tuple)):
                tgt_imgs = tgt_imgs[0]

            src_imgs = src_imgs.to(self.device)
            tgt_imgs = tgt_imgs.to(self.device)

            # Resize to MobileNet input if needed
            src_imgs = self._resize_if_needed(src_imgs)
            tgt_imgs = self._resize_if_needed(tgt_imgs)

            # Labels: source=0, target=1
            src_labels = torch.zeros(src_imgs.size(0), dtype=torch.long, device=self.device)
            tgt_labels = torch.ones(tgt_imgs.size(0), dtype=torch.long, device=self.device)

            imgs = torch.cat([src_imgs, tgt_imgs], dim=0)
            labels = torch.cat([src_labels, tgt_labels], dim=0)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (step + 1) % 50 == 0:
                logger.info(
                    "[DomainClassifier] finetune step %d/%d  loss=%.4f",
                    step + 1, steps, loss.item()
                )

        self._is_finetuned = True
        avg_loss = total_loss / max(steps, 1)
        logger.info("[DomainClassifier] fine-tuning complete. avg_loss=%.4f", avg_loss)
        return {"finetune_avg_loss": avg_loss, "finetune_steps": steps}

    # ------------------------------------------------------------------
    # Forward / log-probability
    # ------------------------------------------------------------------

    def log_prob(self, x_t) -> Tuple:
        """
        Compute log p_φ(y=S|x_t) and log p_φ(y=T|x_t) for noisy images x_t.

        reference_grounding: paper_method_core similarity_guidance

        Args:
            x_t: Tensor [B, C, H, W] of noisy images (arbitrary timestep t).

        Returns:
            (log_p_source, log_p_target): each Tensor [B]
        """
        torch = _require_torch()
        import torch.nn.functional as F

        model = self.get_model()
        model.eval()

        x_t_resized = self._resize_if_needed(x_t)
        logits = model(x_t_resized)                    # [B, 2]
        log_probs = F.log_softmax(logits, dim=-1)      # [B, 2]
        return log_probs[:, CLASS_SOURCE], log_probs[:, CLASS_TARGET]

    def log_prob_with_grad(self, x_t):
        """
        Compute log p_φ(y=S|x_t) and log p_φ(y=T|x_t) while retaining the
        computation graph so that gradients w.r.t. x_t can be computed.

        Used by similarity_guidance.py to compute ∇log p_φ(y=S|x_t) and
        ∇log p_φ(y=T|x_t) for the KL-divergence similarity loss.

        reference_grounding: paper_semantic_chunk_010_classifier_loader_finetuning_adversarial_noise_selection_subsection_adversarial_noise

        Args:
            x_t: Tensor [B, C, H, W] with requires_grad=True.

        Returns:
            (log_p_source, log_p_target): each Tensor [B], graph retained.
        """
        import torch.nn.functional as F

        model = self.get_model()
        # Keep model in eval mode but allow gradient flow through x_t
        x_t_resized = self._resize_if_needed(x_t)
        logits = model(x_t_resized)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs[:, CLASS_SOURCE], log_probs[:, CLASS_TARGET]

    def classifier_gradients(self, x_t):
        """
        Compute ∇_x log p_φ(y=S|x_t) and ∇_x log p_φ(y=T|x_t).

        These gradients are used in the similarity-guided loss:
            L_sim = γ · KL(∇log p_φ(y=S|x_t) ‖ ∇log p_φ(y=T|x_t))

        reference_grounding: paper_method_core similarity_guidance
        reference_grounding: paper_semantic_chunk_010_classifier_loader_finetuning_adversarial_noise_selection_subsection_adversarial_noise

        Args:
            x_t: Tensor [B, C, H, W] (will be detached and re-attached with grad).

        Returns:
            (grad_source, grad_target): each Tensor [B, C, H, W]
        """
        torch = _require_torch()

        x_t_s = x_t.detach().requires_grad_(True)
        log_p_s, log_p_t = self.log_prob_with_grad(x_t_s)

        # ∇_x log p_φ(y=S|x_t)
        grad_source = torch.autograd.grad(
            log_p_s.sum(), x_t_s,
            create_graph=False, retain_graph=True
        )[0]

        # ∇_x log p_φ(y=T|x_t)
        grad_target = torch.autograd.grad(
            log_p_t.sum(), x_t_s,
            create_graph=False, retain_graph=False
        )[0]

        return grad_source, grad_target

    # ------------------------------------------------------------------
    # Similarity-guided loss
    # ------------------------------------------------------------------

    def similarity_loss(self, x_t, gamma: float = SIMILARITY_GUIDANCE_SCALE):
        """
        Compute L_sim = γ · KL(∇log p_φ(y=S|x_t) ‖ ∇log p_φ(y=T|x_t))

        The KL divergence is computed over the spatial gradient distributions
        (flattened per sample), following the paper formulation.

        reference_grounding: paper_method_core similarity_guidance
        reference_grounding: paper_semantic_chunk_003_02_classifier_loader_finetuning_introduction_figure_two_sets

        Args:
            x_t: Tensor [B, C, H, W]
            gamma: similarity guidance scale (default γ=5)

        Returns:
            Scalar loss tensor.
        """
        torch = _require_torch()
        import torch.nn.functional as F

        grad_s, grad_t = self.classifier_gradients(x_t)  # [B, C, H, W]

        B = grad_s.size(0)
        # Flatten spatial dims → [B, D]
        gs_flat = grad_s.view(B, -1)
        gt_flat = grad_t.view(B, -1)

        # Convert to probability distributions via softmax over spatial dims
        p_s = F.softmax(gs_flat, dim=-1).clamp(min=1e-8)
        p_t = F.softmax(gt_flat, dim=-1).clamp(min=1e-8)

        # KL(p_s ‖ p_t) per sample, then mean over batch
        kl = F.kl_div(p_t.log(), p_s, reduction="batchmean")
        return gamma * kl

    # ------------------------------------------------------------------
    # Adversarial noise selection (PGD inner loop)
    # ------------------------------------------------------------------

    def select_adversarial_noise(
        self,
        noise_init,
        x0,
        t,
        diffusion_model,
        alpha_bar_t,
        delta: float = 0.1,
        inner_steps: int = ADV_INNER_STEPS,
        omega: float = ADV_OMEGA,
    ):
        """
        PGD inner loop to select adversarial noise ε* that maximises L_simple.

        Algorithm 1 (adversarial noise selection step):
            ε ← ε_init
            for k = 1..inner_steps:
                g ← ∇_ε L_simple(x0, ε, t)
                ε ← ε + ω · sign(g)
                ε ← clip(ε, -δ, δ)
            return ε*

        reference_grounding: paper_semantic_chunk_010_classifier_loader_finetuning_adversarial_noise_selection_subsection_adversarial_noise
        reference_grounding: paper_method_core adversarial_noise

        Args:
            noise_init: Initial noise tensor [B, C, H, W] ~ N(0,I).
            x0: Clean source images [B, C, H, W].
            t: Diffusion timestep tensor [B].
            diffusion_model: Object with .q_sample(x0, t, noise) and
                             .p_losses(x0, t, noise) methods.
            alpha_bar_t: ᾱ_t values [B] for the DDPM noise schedule.
            delta: Perturbation budget (clamp bound).
            inner_steps: Number of PGD steps (paper: 10).
            omega: PGD step size (paper: 0.02).

        Returns:
            Adversarial noise tensor ε* [B, C, H, W] (detached).
        """
        torch = _require_torch()

        eps = noise_init.detach().clone().requires_grad_(True)

        for k in range(inner_steps):
            if eps.grad is not None:
                eps.grad.zero_()

            # Compute noisy image x_t = sqrt(ᾱ_t)·x0 + sqrt(1-ᾱ_t)·ε
            # Use diffusion_model's q_sample if available, else manual
            if hasattr(diffusion_model, "q_sample"):
                x_t = diffusion_model.q_sample(x0, t, noise=eps)
            else:
                # Manual DDPM forward process
                alpha_bar = alpha_bar_t.view(-1, 1, 1, 1)
                x_t = (alpha_bar ** 0.5) * x0 + ((1 - alpha_bar) ** 0.5) * eps

            # Compute L_simple = E[‖ε - ε_θ(x_t, t)‖²]
            if hasattr(diffusion_model, "p_losses"):
                loss = diffusion_model.p_losses(x0, t, noise=eps)
            else:
                # Fallback: MSE between eps and model prediction
                eps_pred = diffusion_model(x_t, t)
                loss = ((eps - eps_pred) ** 2).mean()

            loss.backward()

            with torch.no_grad():
                # PGD ascent step (maximise loss)
                eps = eps + omega * eps.grad.sign()
                # Project back to ε ∈ [-δ, δ]
                eps = eps.clamp(-delta, delta)
                eps = eps.detach().requires_grad_(True)

        return eps.detach()

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Save classifier weights to disk."""
        torch = _require_torch()
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(
            {
                "state_dict": self.get_model().state_dict(),
                "finetune_steps": self.finetune_steps,
                "is_finetuned": self._is_finetuned,
                "method_id": METHOD_ID,
            },
            path,
        )
        logger.info("[DomainClassifier] saved to %s", path)

    def load(self, path: str):
        """Load classifier weights from disk."""
        torch = _require_torch()
        ckpt = torch.load(path, map_location=self.device)
        self.get_model().load_state_dict(ckpt["state_dict"])
        self._is_finetuned = ckpt.get("is_finetuned", True)
        logger.info("[DomainClassifier] loaded from %s", path)

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def evaluate_accuracy(self, source_loader, target_loader) -> float:
        """
        Compute binary classification accuracy on source + target images.

        reference_grounding: paper_method_core evaluation
        """
        torch = _require_torch()

        model = self.get_model()
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs in source_loader:
                if isinstance(imgs, (list, tuple)):
                    imgs = imgs[0]
                imgs = imgs.to(self.device)
                imgs = self._resize_if_needed(imgs)
                logits = model(imgs)
                preds = logits.argmax(dim=-1)
                labels = torch.zeros(imgs.size(0), dtype=torch.long, device=self.device)
                correct += (preds == labels).sum().item()
                total += imgs.size(0)

            for imgs in target_loader:
                if isinstance(imgs, (list, tuple)):
                    imgs = imgs[0]
                imgs = imgs.to(self.device)
                imgs = self._resize_if_needed(imgs)
                logits = model(imgs)
                preds = logits.argmax(dim=-1)
                labels = torch.ones(imgs.size(0), dtype=torch.long, device=self.device)
                correct += (preds == labels).sum().item()
                total += imgs.size(0)

        accuracy = correct / max(total, 1)
        logger.info("[DomainClassifier] accuracy=%.4f (%d/%d)", accuracy, correct, total)
        return accuracy

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resize_if_needed(self, x):
        """Resize images to 224×224 for MobileNet if they differ in size."""
        torch = _require_torch()
        import torch.nn.functional as F

        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        # Normalise from [-1,1] to ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        # Map [-1,1] → [0,1] first
        x = (x + 1.0) / 2.0
        # Handle grayscale → RGB
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] > 3:
            x = x[:, :3]
        x = (x - mean) / std
        return x

    def set_eval(self):
        """Set model to eval mode (freeze BN etc.)."""
        if self._model is not None:
            self._model.eval()

    def set_train(self):
        """Set model to train mode."""
        if self._model is not None:
            self._model.train()

    def parameters(self):
        """Expose model parameters for external optimisers."""
        return self.get_model().parameters()

    def to(self, device: str):
        """Move model to device."""
        self.device = device
        if self._model is not None:
            self._model = self._model.to(device)
        return self


# ---------------------------------------------------------------------------
# Factory / registry
# ---------------------------------------------------------------------------

def build_domain_classifier(
    cfg: Optional[Dict] = None,
    device: Optional[str] = None,
    pretrained: bool = True,
) -> DomainClassifier:
    """
    Factory function for DomainClassifier.

    method=ours (DPMs-ANT) registry entry.
    reference_grounding: paper_method_core method_registry

    Args:
        cfg: Optional config dict with keys: device, finetune_steps, lr, image_size.
        device: Override device string.
        pretrained: Whether to load ImageNet pretrained MobileNet weights.

    Returns:
        DomainClassifier instance.
    """
    cfg = cfg or {}
    dev = device or cfg.get("device", None)
    finetune_steps = cfg.get("classifier_finetune_steps", CLASSIFIER_FINETUNE_STEPS)
    lr = cfg.get("classifier_lr", 1e-4)
    image_size = cfg.get("image_size", 256)

    classifier = DomainClassifier(
        device=dev,
        pretrained=pretrained,
        finetune_steps=finetune_steps,
        lr=lr,
        image_size=image_size,
    )
    logger.info(
        "[build_domain_classifier] method=%s  finetune_steps=%d  device=%s",
        METHOD_ID, finetune_steps, classifier.device,
    )
    return classifier


# ---------------------------------------------------------------------------
# Method registry entry (machine-readable)
# reference_grounding: paper_method_core method_registry
# ---------------------------------------------------------------------------
DOMAIN_CLASSIFIER_REGISTRY = {
    "method_id": METHOD_ID,
    "display_name": METHOD_DISPLAY_NAME,
    "classifier_class": "DomainClassifier",
    "backbone": "MobileNetV2",
    "pretrained_source": "ImageNet",
    "finetune_steps": CLASSIFIER_FINETUNE_STEPS,
    "similarity_guidance_scale_gamma": SIMILARITY_GUIDANCE_SCALE,
    "adv_inner_steps": ADV_INNER_STEPS,
    "adv_omega": ADV_OMEGA,
    "class_source": CLASS_SOURCE,
    "class_target": CLASS_TARGET,
    "loss_sim_formula": "gamma * KL(grad_log_p_source || grad_log_p_target)",
    "adv_noise_formula": "eps = eps + omega * sign(grad_eps L_simple); clip(eps, -delta, delta)",
    "paper_ref": "Bridging Data Gaps in Diffusion Models with Adversarial Noise-Based Transfer Learning",
}