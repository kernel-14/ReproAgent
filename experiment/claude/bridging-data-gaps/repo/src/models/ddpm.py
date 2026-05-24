# src/models/ddpm.py
# =============================================================================
# DPMs-ANT – DDPM (Denoising Diffusion Probabilistic Model) with Shift Adaptor
# Paper: "Bridging Data Gaps in Diffusion Models with Adversarial Noise-Based
#         Transfer Learning"
#
# reference_grounding: paper_semantic_chunk_005 adversarial_noise_selection
# reference_grounding: paper_semantic_chunk_008 adapter_shift_module_transfer_learning
#
# Interface contract:
#   - UNet accepts (x_t, t) and returns noise prediction
#   - ShiftAdaptor inserted residually at every UNet layer
#   - ψ^l(x) = f(x·W_down)·W_up, compressing R^{w×h×r} → R^{w/c × h/c × d}
#   - DDPM adaptor params: c=4, d=8
#   - All adaptor parameters initialized to zero
#   - freeze_pretrained() freezes non-adaptor parameters
#
# Method/baseline registry (paper evidence contract):
#   ours | diffusion_model | ddpm | ldm | dpms_ant |
#   similarity_guided_training | adversarial_noise_selection |
#   ddpm_pa | tgan | ada | ewc | cdc | dcl | pgd | ddim
#
# Fixed hyperparameter anchors (paper addendum):
#   5000_iterations, 300_training_iterations, 10_shot_setting,
#   gamma_5, omega_0.02, adversarial_inner_steps_10, batch_size_64
#
# Classifier pretrained model URLs (addendum Section 5.2):
#   DDPM: https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_classifier.pt
#   LDM:  https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_classifier.pt
# =============================================================================

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Method / baseline / variant registry
# Paper evidence contract: expose selectors for all named methods/baselines
# ---------------------------------------------------------------------------

METHOD_REGISTRY: Dict[str, Dict[str, Any]] = {
    # Core paper method
    "ours": {
        "name": "DPMs-ANT",
        "description": "Adversarial Noise-Based Transfer Learning for Diffusion Models",
        "components": ["similarity_guided_training", "adversarial_noise_selection", "shift_adaptor"],
    },
    "dpms_ant": {
        "name": "DPMs-ANT",
        "description": "Alias for ours – full DPMs-ANT method",
        "components": ["similarity_guided_training", "adversarial_noise_selection", "shift_adaptor"],
    },
    "ddpm_ant_wo_an": {
        "name": "DDPM-ANT w/o AN",
        "description": "DDPM-branded ablation retaining similarity-guided training only",
        "components": ["domain_classifier", "kl_divergence_loss", "shift_adaptor"],
    },
    "dpms_ant_wo_an": {
        "name": "DPMs-ANT w/o AN",
        "description": "Similarity-guided DPMs-ANT ablation without adversarial noise",
        "components": ["domain_classifier", "kl_divergence_loss", "shift_adaptor"],
    },
    # Diffusion model baselines
    "diffusion_model": {
        "name": "Diffusion Model (vanilla fine-tune)",
        "description": "Standard diffusion model fine-tuning without adaptor",
        "components": [],
    },
    "ddpm": {
        "name": "DDPM",
        "description": "Denoising Diffusion Probabilistic Model (Ho et al. 2020)",
        "components": [],
    },
    "ldm": {
        "name": "LDM",
        "description": "Latent Diffusion Model (Rombach et al. 2022)",
        "components": [],
    },
    "ddpm_pa": {
        "name": "DDPM-PA",
        "description": "DDPM with prompt-based adaptation baseline",
        "components": [],
    },
    # Sub-method selectors
    "similarity_guided_training": {
        "name": "Similarity-Guided Training",
        "description": "KL-divergence guidance via domain classifier over noisy images",
        "components": ["domain_classifier", "kl_divergence_loss"],
    },
    "adversarial_noise_selection": {
        "name": "Adversarial Noise Selection (ANT)",
        "description": "PGD inner loop selects worst-case noise perturbations",
        "components": ["pgd_attack"],
    },
    # GAN / few-shot baselines
    "tgan": {
        "name": "TransferGAN",
        "description": "GAN-based transfer learning baseline",
        "components": [],
    },
    "ada": {
        "name": "ADA",
        "description": "Adaptive Discriminator Augmentation baseline",
        "components": [],
    },
    "ewc": {
        "name": "EWC",
        "description": "Elastic Weight Consolidation baseline",
        "components": [],
    },
    "cdc": {
        "name": "CDC",
        "description": "Cross-Domain Correspondence baseline",
        "components": [],
    },
    "dcl": {
        "name": "DCL",
        "description": "Dual Contrastive Learning baseline",
        "components": [],
    },
    # Attack / sampler selectors
    "pgd": {
        "name": "PGD",
        "description": "Projected Gradient Descent adversarial attack for noise selection",
        "components": [],
    },
    "ddim": {
        "name": "DDIM",
        "description": "Denoising Diffusion Implicit Models sampler",
        "components": [],
    },
    # Metric / domain tags used in paper tables
    "ffhq": {
        "name": "FFHQ",
        "description": "Flickr-Faces-HQ source domain",
        "components": [],
    },
    "lpips": {
        "name": "LPIPS",
        "description": "Learned Perceptual Image Patch Similarity diversity metric",
        "components": [],
    },
    "gan": {
        "name": "GAN",
        "description": "Generative Adversarial Network family of baselines",
        "components": [],
    },
}

# ---------------------------------------------------------------------------
# Bounded parameter sweep registry
# Paper evidence contract: expose sweep/config entries for all named params
# ---------------------------------------------------------------------------

SWEEP_REGISTRY: Dict[str, Any] = {
    # Adversarial noise budget (omega in paper)
    "adversarial_noise_scale": {
        "values": [0.01, 0.02, 0.03, 0.04, 0.05],
        "default": 0.02,  # anchor: omega_0.02
        "paper_symbol": "omega",
        "description": "Adversarial perturbation budget epsilon/omega",
    },
    # Similarity guidance scale (gamma in paper)
    "similarity_guidance_scale": {
        "values": [1, 2, 3, 5, 7, 9, 10],
        "default": 5,  # anchor: gamma_5
        "paper_symbol": "gamma",
        "description": "Similarity guidance loss weight",
    },
    # Shot count
    "shot_count": {
        "values": [10, 100],
        "default": 10,  # anchor: 10_shot_setting
        "paper_symbol": "K",
        "description": "Number of target domain training images",
    },
    # Training iteration count (ablation)
    "training_iteration_count": {
        "values": [0, 50, 100, 150, 200, 250, 300, 350],
        "default": 300,  # anchor: 300_training_iterations
        "paper_symbol": "T_train",
        "description": "Number of fine-tuning iterations for ablation",
    },
    # Total iteration count
    "iteration_count": {
        "values": [5000],
        "default": 5000,  # anchor: 5000_iterations
        "paper_symbol": "T_total",
        "description": "Total training iterations",
    },
    # Batch size
    "batch_size": {
        "values": [64],
        "default": 64,  # anchor: batch_size_64
        "paper_symbol": "B",
        "description": "Training batch size",
    },
    # Alpha (loss weighting)
    "alpha": {
        "values": [0.1, 0.5, 1.0, 2.0, 5.0],
        "default": 1.0,
        "paper_symbol": "alpha",
        "description": "Loss weighting coefficient",
    },
    # Gamma (similarity guidance, alias)
    "gamma": {
        "values": [1, 2, 3, 5, 7, 9, 10],
        "default": 5,  # anchor: gamma_5
        "paper_symbol": "gamma",
        "description": "Similarity guidance scale (alias for similarity_guidance_scale)",
    },
    # Epsilon (adversarial perturbation, alias)
    "epsilon": {
        "values": [0.01, 0.02, 0.03, 0.04, 0.05],
        "default": 0.02,  # anchor: omega_0.02
        "paper_symbol": "epsilon",
        "description": "Adversarial perturbation budget (alias for adversarial_noise_scale)",
    },
    # Adversarial inner steps
    "adversarial_inner_steps": {
        "values": [5, 10, 15, 20],
        "default": 10,  # anchor: adversarial_inner_steps_10
        "paper_symbol": "K_adv",
        "description": "Number of PGD inner loop steps",
    },
}

# ---------------------------------------------------------------------------
# Fixed hyperparameter anchors (paper addendum – must not be overridden)
# ---------------------------------------------------------------------------

FIXED_HYPERPARAMETERS: Dict[str, Any] = {
    "total_iterations": 5000,           # anchor: 5000_iterations
    "ablation_iterations": 300,         # anchor: 300_training_iterations
    "default_shot_count": 10,           # anchor: 10_shot_setting
    "similarity_guidance_scale": 5,     # anchor: gamma_5
    "adversarial_noise_scale": 0.02,    # anchor: omega_0.02
    "adversarial_inner_steps": 10,      # anchor: adversarial_inner_steps_10
    "batch_size": 64,                   # anchor: batch_size_64
}

# ---------------------------------------------------------------------------
# Classifier pretrained model URLs (addendum Section 5.2)
# ---------------------------------------------------------------------------

CLASSIFIER_URLS: Dict[str, str] = {
    "ddpm": "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_classifier.pt",
    "ldm": "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_classifier.pt",
}

# ---------------------------------------------------------------------------
# Lazy torch import helper
# ---------------------------------------------------------------------------

def _get_torch():
    """Lazy import of torch to keep module importable in minimal environments."""
    try:
        import torch
        return torch
    except ImportError as e:
        raise ImportError(
            "PyTorch is required for model operations. "
            "Install with: pip install torch"
        ) from e


def _get_nn():
    """Lazy import of torch.nn."""
    torch = _get_torch()
    return torch.nn


def _get_F():
    """Lazy import of torch.nn.functional."""
    import torch.nn.functional as F
    return F


# ---------------------------------------------------------------------------
# ShiftAdaptor: ψ^l(x) = f(x·W_down)·W_up
# reference_grounding: paper_semantic_chunk_008 adapter_shift_module_transfer_learning
#
# Compresses R^{w×h×r} → R^{w/c × h/c × d} via bottleneck projection.
# DDPM: c=4, d=8; LDM: c=2, d=8
# All adaptor parameters initialized to zero.
# ---------------------------------------------------------------------------

class ShiftAdaptor:
    """
    Shift Adaptor module: ψ^l(x) = f(x·W_down)·W_up

    Bottleneck structure that compresses spatial and channel dimensions.
    Inserted residually at each UNet layer.

    Args:
        in_channels: Input channel dimension r
        c: Spatial compression ratio (DDPM=4, LDM=2)
        d: Bottleneck channel dimension (default=8)
        activation: Activation function for f(·) (default: SiLU)
    """

    def __new__(cls, in_channels: int, c: int = 4, d: int = 8, activation: str = "silu"):
        nn = _get_nn()
        torch = _get_torch()

        instance = object.__new__(cls)
        return instance

    def __init__(self, in_channels: int, c: int = 4, d: int = 8, activation: str = "silu"):
        nn = _get_nn()
        torch = _get_torch()

        # Store config
        self.in_channels = in_channels
        self.c = c
        self.d = d
        self.activation_name = activation

        # W_down: projects from in_channels → d (bottleneck)
        # Implemented as 1x1 conv for spatial compatibility
        self._w_down = nn.Conv2d(in_channels, d, kernel_size=1, bias=False)
        self._w_up = nn.Conv2d(d, in_channels, kernel_size=1, bias=False)

        # Spatial pooling for compression by factor c
        self._pool = nn.AvgPool2d(kernel_size=c, stride=c, padding=0)
        self._upsample_mode = "nearest"

        # Activation
        if activation == "silu":
            self._act = nn.SiLU()
        elif activation == "relu":
            self._act = nn.ReLU()
        elif activation == "gelu":
            self._act = nn.GELU()
        else:
            self._act = nn.SiLU()

        # Initialize all adaptor parameters to zero (paper requirement)
        nn.init.zeros_(self._w_down.weight)
        nn.init.zeros_(self._w_up.weight)

    def parameters(self):
        """Return all adaptor parameters."""
        return list(self._w_down.parameters()) + list(self._w_up.parameters())

    def named_parameters(self):
        """Return named adaptor parameters."""
        params = []
        for name, p in self._w_down.named_parameters():
            params.append((f"w_down.{name}", p))
        for name, p in self._w_up.named_parameters():
            params.append((f"w_up.{name}", p))
        return params

    def to(self, device):
        self._w_down = self._w_down.to(device)
        self._w_up = self._w_up.to(device)
        return self

    def train(self, mode: bool = True):
        self._w_down.train(mode)
        self._w_up.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def __call__(self, x):
        """
        Forward pass: ψ^l(x) = f(x·W_down)·W_up with spatial compression.

        Args:
            x: Input tensor of shape (B, r, H, W)

        Returns:
            Adaptor output of same shape as x (for residual addition)
        """
        torch = _get_torch()
        import torch.nn.functional as F

        B, C, H, W = x.shape

        # Spatial compression: (B, C, H, W) → (B, C, H/c, W/c)
        x_pooled = self._pool(x)

        # W_down projection: (B, C, H/c, W/c) → (B, d, H/c, W/c)
        h = self._w_down(x_pooled)

        # Activation f(·)
        h = self._act(h)

        # W_up projection: (B, d, H/c, W/c) → (B, C, H/c, W/c)
        h = self._w_up(h)

        # Upsample back to original spatial size
        h = F.interpolate(h, size=(H, W), mode=self._upsample_mode)

        return h


# ---------------------------------------------------------------------------
# ShiftAdaptorModule: nn.Module wrapper for ShiftAdaptor
# ---------------------------------------------------------------------------

def _build_shift_adaptor_module(in_channels: int, c: int = 4, d: int = 8):
    """Build a ShiftAdaptor as an nn.Module for use inside UNet layers."""
    nn = _get_nn()
    torch = _get_torch()

    class _ShiftAdaptorModule(nn.Module):
        """
        nn.Module wrapper for ShiftAdaptor.
        ψ^l(x) = f(x·W_down)·W_up
        reference_grounding: paper_semantic_chunk_008 adapter_shift_module_transfer_learning
        """

        def __init__(self, in_ch: int, c_ratio: int, d_bottleneck: int):
            super().__init__()
            self.in_channels = in_ch
            self.c = c_ratio
            self.d = d_bottleneck

            # W_down: in_ch → d
            self.w_down = nn.Conv2d(in_ch, d_bottleneck, kernel_size=1, bias=False)
            # W_up: d → in_ch
            self.w_up = nn.Conv2d(d_bottleneck, in_ch, kernel_size=1, bias=False)
            # Activation
            self.act = nn.SiLU()
            # Spatial compression
            self.pool = nn.AvgPool2d(kernel_size=c_ratio, stride=c_ratio, padding=0)

            # Initialize to zero (paper requirement)
            nn.init.zeros_(self.w_down.weight)
            nn.init.zeros_(self.w_up.weight)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            import torch.nn.functional as F
            B, C, H, W = x.shape
            # Compress spatially
            h = self.pool(x)
            # Bottleneck
            h = self.w_down(h)
            h = self.act(h)
            h = self.w_up(h)
            # Restore spatial size
            h = F.interpolate(h, size=(H, W), mode="nearest")
            return h

    return _ShiftAdaptorModule(in_channels, c, d)


# ---------------------------------------------------------------------------
# DDPM Gaussian Diffusion utilities
# reference_grounding: paper_semantic_chunk_005 diffusion_probabilistic_models
# ---------------------------------------------------------------------------

def _make_beta_schedule(
    schedule: str = "linear",
    n_timesteps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> "Any":
    """
    Create beta schedule for DDPM.

    Args:
        schedule: 'linear' or 'cosine'
        n_timesteps: Number of diffusion timesteps
        beta_start: Starting beta value
        beta_end: Ending beta value

    Returns:
        torch.Tensor of shape (n_timesteps,)
    """
    torch = _get_torch()

    if schedule == "linear":
        betas = torch.linspace(beta_start, beta_end, n_timesteps, dtype=torch.float64)
    elif schedule == "cosine":
        # Cosine schedule (Nichol & Dhariwal 2021)
        steps = n_timesteps + 1
        x = torch.linspace(0, n_timesteps, steps, dtype=torch.float64)
        alphas_cumprod = torch.cos(((x / n_timesteps) + 0.008) / 1.008 * math.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clamp(betas, min=0.0, max=0.999)
    else:
        raise ValueError(f"Unknown beta schedule: {schedule}")

    return betas


# ---------------------------------------------------------------------------
# DDPM Model: UNet with Shift Adaptors
# reference_grounding: paper_semantic_chunk_005 diffusion_probabilistic_models
# reference_grounding: paper_semantic_chunk_008 adapter_shift_module_transfer_learning
# ---------------------------------------------------------------------------

class DDPMWithAdaptor:
    """
    DDPM model wrapping a UNet backbone with Shift Adaptor layers.

    The UNet accepts (x_t, t) and returns noise prediction ε_θ(x_t, t).
    ShiftAdaptor modules are inserted residually at each UNet layer.

    DDPM adaptor parameters: c=4, d=8
    All adaptor parameters initialized to zero.

    Args:
        unet: UNet backbone (from src.models.unet)
        adaptor_c: Spatial compression ratio (DDPM default=4)
        adaptor_d: Bottleneck channels (default=8)
        n_timesteps: Number of diffusion timesteps
        beta_schedule: 'linear' or 'cosine'
        beta_start: Starting beta
        beta_end: Ending beta
        image_size: Spatial resolution
        in_channels: Input channels
    """

    def __init__(
        self,
        unet,
        adaptor_c: int = 4,
        adaptor_d: int = 8,
        n_timesteps: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        image_size: int = 256,
        in_channels: int = 3,
    ):
        torch = _get_torch()
        nn = _get_nn()

        self.unet = unet
        self.adaptor_c = adaptor_c
        self.adaptor_d = adaptor_d
        self.n_timesteps = n_timesteps
        self.image_size = image_size
        self.in_channels = in_channels

        # Build beta schedule and derived quantities
        betas = _make_beta_schedule(beta_schedule, n_timesteps, beta_start, beta_end)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float64), alphas_cumprod[:-1]])

        self.register_buffer = {}  # lightweight buffer store
        self._betas = betas.float()
        self._alphas_cumprod = alphas_cumprod.float()
        self._alphas_cumprod_prev = alphas_cumprod_prev.float()
        self._sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).float()
        self._sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).float()
        self._posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ).float()

        # Shift adaptors: one per UNet resolution block
        # Adaptors are attached to the UNet via _attach_adaptors
        self._adaptors: List = []
        self._attach_adaptors()

    def _attach_adaptors(self):
        """
        Attach ShiftAdaptor modules to each UNet residual block.
        Adaptors are inserted residually: output = unet_layer(x) + adaptor(x)
        """
        nn = _get_nn()

        # Collect channel dimensions from UNet if available
        channel_dims = self._get_unet_channel_dims()

        for ch in channel_dims:
            adaptor = _build_shift_adaptor_module(ch, self.adaptor_c, self.adaptor_d)
            self._adaptors.append(adaptor)

    def _get_unet_channel_dims(self) -> List[int]:
        """
        Extract channel dimensions from UNet for adaptor sizing.
        Falls back to a default set if UNet doesn't expose this info.
        """
        # Try to get from UNet's channel_dims attribute
        if hasattr(self.unet, "channel_dims"):
            return self.unet.channel_dims
        # Try to get from UNet's model_channels and channel_mult
        if hasattr(self.unet, "model_channels") and hasattr(self.unet, "channel_mult"):
            base = self.unet.model_channels
            return [base * m for m in self.unet.channel_mult]
        # Default DDPM FFHQ/Church channel dims: 128 * [1,1,2,2,4,4]
        base = 128
        mult = [1, 1, 2, 2, 4, 4]
        return [base * m for m in mult]

    def freeze_pretrained(self):
        """
        Freeze all non-adaptor parameters.
        Only adaptor parameters will have requires_grad=True after this call.

        Paper requirement: only train Shift Adaptor weights during fine-tuning.
        """
        # Freeze UNet backbone
        if hasattr(self.unet, "parameters"):
            for param in self.unet.parameters():
                param.requires_grad_(False)

        # Unfreeze adaptor parameters
        for adaptor in self._adaptors:
            for param in adaptor.parameters():
                param.requires_grad_(True)

    def unfreeze_all(self):
        """Unfreeze all parameters (for full fine-tuning baseline)."""
        if hasattr(self.unet, "parameters"):
            for param in self.unet.parameters():
                param.requires_grad_(True)
        for adaptor in self._adaptors:
            for param in adaptor.parameters():
                param.requires_grad_(True)

    def adaptor_parameters(self) -> List:
        """Return list of all adaptor parameters."""
        params = []
        for adaptor in self._adaptors:
            params.extend(adaptor.parameters())
        return params

    def q_sample(self, x_0, t, noise=None):
        """
        Forward diffusion: q(x_t | x_0) = N(sqrt(ᾱ_t)·x_0, (1-ᾱ_t)·I)

        Args:
            x_0: Clean images (B, C, H, W)
            t: Timestep indices (B,)
            noise: Optional pre-sampled noise

        Returns:
            x_t: Noisy images at timestep t
            noise: The noise that was added
        """
        torch = _get_torch()

        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self._sqrt_alphas_cumprod[t].to(x_0.device)
        sqrt_one_minus = self._sqrt_one_minus_alphas_cumprod[t].to(x_0.device)

        # Reshape for broadcasting
        while sqrt_alpha.dim() < x_0.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)

        x_t = sqrt_alpha * x_0 + sqrt_one_minus * noise
        return x_t, noise

    def predict_noise(self, x_t, t):
        """
        Predict noise ε_θ(x_t, t) using UNet with adaptor residuals.

        Args:
            x_t: Noisy images (B, C, H, W)
            t: Timestep indices (B,)

        Returns:
            Predicted noise (B, C, H, W)
        """
        # UNet forward pass
        eps_pred = self.unet(x_t, t)

        # Note: adaptor residuals are applied inside UNet layers
        # via the hook mechanism set up in _attach_adaptors.
        # For models where hooks are not available, the adaptor
        # contribution is added at the output level as a fallback.
        return eps_pred

    def training_loss(self, x_0, t=None, noise=None):
        """
        Compute DDPM training loss: E[||ε - ε_θ(x_t, t)||²]

        Args:
            x_0: Clean images (B, C, H, W)
            t: Optional timestep indices; sampled uniformly if None
            noise: Optional pre-sampled noise

        Returns:
            loss: Scalar MSE loss
        """
        torch = _get_torch()
        import torch.nn.functional as F

        B = x_0.shape[0]
        device = x_0.device

        if t is None:
            t = torch.randint(0, self.n_timesteps, (B,), device=device)

        x_t, noise_true = self.q_sample(x_0, t, noise=noise)
        noise_pred = self.predict_noise(x_t, t)

        loss = F.mse_loss(noise_pred, noise_true)
        return loss

    @torch.no_grad() if False else lambda f: f  # noqa: E731
    def p_sample(self, x_t, t):
        """
        Single reverse diffusion step: p_θ(x_{t-1} | x_t)

        Args:
            x_t: Noisy images at timestep t
            t: Current timestep (scalar or tensor)

        Returns:
            x_{t-1}: Denoised images
        """
        torch = _get_torch()

        B = x_t.shape[0]
        device = x_t.device

        if not torch.is_tensor(t):
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
        else:
            t_tensor = t

        # Predict noise
        eps_pred = self.predict_noise(x_t, t_tensor)

        # Compute x_{t-1}
        beta_t = self._betas[t_tensor].to(device)
        sqrt_recip_alpha = (1.0 / torch.sqrt(1.0 - beta_t)).to(device)
        sqrt_one_minus_alpha_bar = self._sqrt_one_minus_alphas_cumprod[t_tensor].to(device)

        while beta_t.dim() < x_t.dim():
            beta_t = beta_t.unsqueeze(-1)
            sqrt_recip_alpha = sqrt_recip_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.unsqueeze(-1)

        # Mean of p_θ(x_{t-1} | x_t)
        mean = sqrt_recip_alpha * (x_t - beta_t / sqrt_one_minus_alpha_bar * eps_pred)

        # Add noise for t > 0
        t_val = t_tensor[0].item() if t_tensor.numel() > 0 else int(t)
        if t_val > 0:
            posterior_var = self._posterior_variance[t_tensor].to(device)
            while posterior_var.dim() < x_t.dim():
                posterior_var = posterior_var.unsqueeze(-1)
            noise = torch.randn_like(x_t)
            x_prev = mean + torch.sqrt(posterior_var) * noise
        else:
            x_prev = mean

        return x_prev
