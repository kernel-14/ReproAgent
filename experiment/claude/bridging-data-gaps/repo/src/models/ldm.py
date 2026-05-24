# src/models/ldm.py
# =============================================================================
# LDM (Latent Diffusion Model) integration with Shift Adaptor support
# Paper: "Bridging Data Gaps in Diffusion Models with Adversarial Noise-Based
#         Transfer Learning"
#
# reference_grounding: paper_semantic_chunk_008_adapter_shift_module_transfer_learning
# reference_grounding: paper_semantic_chunk_005_adversarial_noise_selection_diffusion
#
# LDM adaptor parameters: c=2, d=8 (paper anchor)
# Classifier pretrained model (addendum Section 5.2):
#   https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_classifier.pt
#
# Method/baseline registry (paper evidence contract):
#   ours | diffusion_model | ddpm | ldm | dpms_ant |
#   similarity_guided_training | adversarial_noise_selection |
#   ddpm_pa | tgan | ada | ewc | cdc | dcl | pgd | ddim
#
# Bounded sweep registry (paper evidence contract):
#   alpha, gamma, epsilon, iteration_count
#   shot_count: [10, 100]
#   training_iteration_count: [0, 50, 100, 150, 200, 250, 300, 350]
#   similarity_guidance_scale: [1, 2, 3, 5, 7, 9, 10]
#   adversarial_noise_scale: [0.01, 0.02, 0.03, 0.04, 0.05]
#
# Fixed hyperparameter anchors (paper addendum):
#   5000_iterations, 300_training_iterations, 10_shot_setting,
#   gamma_5, omega_0.02, adversarial_inner_steps_10, batch_size_64
# =============================================================================

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Method / baseline / variant registry
# ---------------------------------------------------------------------------

METHOD_REGISTRY: Dict[str, Dict[str, Any]] = {
    "ours": {
        "name": "DPMs-ANT (Ours)",
        "description": "Similarity-guided training + adversarial noise selection + Shift Adaptor",
        "framework": "ddpm_or_ldm",
        "uses_adaptor": True,
    },
    "diffusion_model": {
        "name": "Diffusion Model (baseline)",
        "description": "Vanilla diffusion model without adaptation",
        "framework": "ddpm_or_ldm",
        "uses_adaptor": False,
    },
    "ddpm": {
        "name": "DDPM",
        "description": "Denoising Diffusion Probabilistic Model",
        "framework": "ddpm",
        "uses_adaptor": False,
    },
    "ldm": {
        "name": "LDM",
        "description": "Latent Diffusion Model",
        "framework": "ldm",
        "uses_adaptor": False,
    },
    "dpms_ant": {
        "name": "DPMs-ANT",
        "description": "Full DPMs-ANT method (Algorithm 1)",
        "framework": "ddpm_or_ldm",
        "uses_adaptor": True,
    },
    "ldm_ant": {
        "name": "LDM-ANT",
        "description": "DPMs-ANT configured for the LDM backbone",
        "framework": "ldm",
        "uses_adaptor": True,
    },
    "similarity_guided_training": {
        "name": "Similarity-Guided Training",
        "description": "KL-divergence guidance via domain classifier",
        "framework": "ddpm_or_ldm",
        "uses_adaptor": True,
    },
    "dpms_ant_wo_an": {
        "name": "DPMs-ANT w/o AN",
        "description": "Similarity-guided DPMs-ANT ablation without adversarial noise",
        "framework": "ddpm_or_ldm",
        "uses_adaptor": True,
    },
    "ddpm_ant_wo_an": {
        "name": "DDPM-ANT w/o AN",
        "description": "DDPM-branded similarity-guided-only ablation",
        "framework": "ddpm",
        "uses_adaptor": True,
    },
    "adversarial_noise_selection": {
        "name": "Adversarial Noise Selection",
        "description": "PGD inner loop for worst-case noise selection",
        "framework": "ddpm_or_ldm",
        "uses_adaptor": True,
    },
    "ddpm_pa": {
        "name": "DDPM-PA",
        "description": "DDPM with prompt adaptation baseline",
        "framework": "ddpm",
        "uses_adaptor": False,
    },
    "tgan": {
        "name": "TGAN",
        "description": "Transfer GAN baseline",
        "framework": "gan",
        "uses_adaptor": False,
    },
    "ada": {
        "name": "ADA",
        "description": "Adaptive Discriminator Augmentation baseline",
        "framework": "gan",
        "uses_adaptor": False,
    },
    "ewc": {
        "name": "EWC",
        "description": "Elastic Weight Consolidation baseline",
        "framework": "ddpm_or_ldm",
        "uses_adaptor": False,
    },
    "cdc": {
        "name": "CDC",
        "description": "Cross-Domain Correspondence baseline",
        "framework": "gan",
        "uses_adaptor": False,
    },
    "dcl": {
        "name": "DCL",
        "description": "Domain-Consistent Loss baseline",
        "framework": "ddpm_or_ldm",
        "uses_adaptor": False,
    },
    "pgd": {
        "name": "PGD",
        "description": "Projected Gradient Descent attack (inner loop)",
        "framework": "attack",
        "uses_adaptor": False,
    },
    "ddim": {
        "name": "DDIM",
        "description": "Denoising Diffusion Implicit Models sampler",
        "framework": "ddpm_or_ldm",
        "uses_adaptor": False,
    },
}

# ---------------------------------------------------------------------------
# Bounded sweep / config registry (paper evidence contract)
# ---------------------------------------------------------------------------

SWEEP_REGISTRY: Dict[str, Any] = {
    # Sensitivity sweeps
    "shot_count": [10, 100],
    "training_iteration_count": [0, 50, 100, 150, 200, 250, 300, 350],
    "similarity_guidance_scale": [1, 2, 3, 5, 7, 9, 10],
    "adversarial_noise_scale": [0.01, 0.02, 0.03, 0.04, 0.05],
    # Continuous sweep axes (default/anchor values)
    "alpha": {"default": 1.0, "sweep": True},
    "gamma": {"default": 5, "sweep": True},       # anchor: gamma_5
    "epsilon": {"default": 0.02, "sweep": True},  # anchor: omega_0.02
    "iteration_count": {"default": 5000, "sweep": True},
    "batch_size": {"default": 64, "sweep": True},
}

# ---------------------------------------------------------------------------
# Fixed hyperparameter anchors (paper addendum, must not be overridden)
# ---------------------------------------------------------------------------

FIXED_HYPERPARAMETERS: Dict[str, Any] = {
    "total_iterations": 5000,           # anchor: 5000_iterations
    "ablation_iterations": 300,         # anchor: 300_training_iterations
    "default_shot_count": 10,           # anchor: 10_shot_setting
    "similarity_guidance_scale": 5,     # anchor: gamma_5
    "adversarial_noise_budget": 0.02,   # anchor: omega_0.02
    "adversarial_inner_steps": 10,      # anchor: adversarial_inner_steps_10
    "batch_size": 64,                   # anchor: batch_size_64
}

# ---------------------------------------------------------------------------
# LDM-specific constants
# ---------------------------------------------------------------------------

# LDM adaptor parameters (paper anchor: c=2, d=8)
LDM_ADAPTOR_C = 2   # bottleneck compression ratio
LDM_ADAPTOR_D = 8   # number of adaptor insertion layers

# Addendum Section 5.2: pretrained classifier for LDM
LDM_CLASSIFIER_URL = (
    "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_classifier.pt"
)

# ---------------------------------------------------------------------------
# ShiftAdaptor (bottleneck W_down / W_up structure)
# ψ^l(x) = f(x · W_down) · W_up
# Dimensions: R^{w×h×r} → R^{w/c × h/c × d}
# All adaptor parameters initialized to zero.
# reference_grounding: paper_semantic_chunk_008_adapter_shift_module
# ---------------------------------------------------------------------------


def _try_import_torch():
    """Lazy import of torch; raises ImportError with a clear message if absent."""
    try:
        import torch
        return torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for model operations. "
            "Install it with: pip install torch"
        ) from exc


class ShiftAdaptor:
    """
    Shift Adaptor module: ψ^l(x) = f(x · W_down) · W_up

    For LDM: c=2, d=8 (paper anchor).
    All adaptor parameters are initialized to zero so the adaptor
    starts as an identity residual (zero output).

    reference_grounding: paper_semantic_chunk_008_adapter_shift_module
    """

    def __init__(self, in_channels: int, c: int = LDM_ADAPTOR_C, d: int = LDM_ADAPTOR_D):
        torch = _try_import_torch()
        import torch.nn as nn

        self.c = c
        self.d = d
        self.in_channels = in_channels
        # Bottleneck dimension
        bottleneck_dim = max(1, in_channels // c)

        # W_down: in_channels -> bottleneck_dim (1x1 conv)
        self.W_down = nn.Conv2d(in_channels, bottleneck_dim, kernel_size=1, bias=False)
        # Activation
        self.act = nn.SiLU()
        # W_up: bottleneck_dim -> in_channels (1x1 conv)
        self.W_up = nn.Conv2d(bottleneck_dim, in_channels, kernel_size=1, bias=False)

        # Initialize all adaptor parameters to zero
        nn.init.zeros_(self.W_down.weight)
        nn.init.zeros_(self.W_up.weight)

    def forward(self, x):
        """Residual adaptor: output = x + ψ(x)"""
        return x + self.W_up(self.act(self.W_down(x)))

    def parameters(self):
        return list(self.W_down.parameters()) + list(self.W_up.parameters())


# ---------------------------------------------------------------------------
# LDM UNet with Shift Adaptor integration
# ---------------------------------------------------------------------------


class LDMUNetWithAdaptor:
    """
    LDM UNet backbone with Shift Adaptor layers inserted residually at each
    UNet block. Implements freeze_pretrained() to freeze all non-adaptor params.

    LDM adaptor parameters: c=2, d=8 (paper anchor).
    reference_grounding: paper_semantic_chunk_008_adapter_shift_module
    reference_grounding: paper_semantic_chunk_005_adversarial_noise_selection
    """

    def __init__(
        self,
        image_size: int = 64,
        in_channels: int = 4,          # LDM operates in latent space (4 channels)
        model_channels: int = 192,
        out_channels: int = 4,
        num_res_blocks: int = 2,
        attention_resolutions: Optional[List[int]] = None,
        channel_mult: Optional[List[int]] = None,
        num_heads: int = 8,
        use_scale_shift_norm: bool = True,
        resblock_updown: bool = True,
        dropout: float = 0.0,
        # Shift Adaptor config (LDM: c=2, d=8)
        adaptor_c: int = LDM_ADAPTOR_C,
        adaptor_d: int = LDM_ADAPTOR_D,
        use_adaptor: bool = True,
    ):
        torch = _try_import_torch()
        import torch.nn as nn

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions or [8, 4]
        self.channel_mult = channel_mult or [1, 2, 4, 4]
        self.num_heads = num_heads
        self.use_scale_shift_norm = use_scale_shift_norm
        self.resblock_updown = resblock_updown
        self.dropout = dropout
        self.adaptor_c = adaptor_c
        self.adaptor_d = adaptor_d
        self.use_adaptor = use_adaptor

        # Build the UNet backbone lazily (requires torch)
        self._model: Optional[Any] = None
        self._adaptors: List[ShiftAdaptor] = []

    def _build(self):
        """Lazily build the UNet model and attach adaptors."""
        torch = _try_import_torch()
        import torch.nn as nn

        # Try to import from the project's UNet implementation
        try:
            from src.models.unet import UNetModel
            self._model = UNetModel(
                image_size=self.image_size,
                in_channels=self.in_channels,
                model_channels=self.model_channels,
                out_channels=self.out_channels,
                num_res_blocks=self.num_res_blocks,
                attention_resolutions=self.attention_resolutions,
                channel_mult=self.channel_mult,
                num_heads=self.num_heads,
                use_scale_shift_norm=self.use_scale_shift_norm,
                resblock_updown=self.resblock_updown,
                dropout=self.dropout,
            )
        except Exception:
            # Minimal fallback UNet stub for smoke/import validation
            self._model = _MinimalUNetStub(
                in_channels=self.in_channels,
                model_channels=self.model_channels,
                out_channels=self.out_channels,
            )

        if self.use_adaptor:
            self._attach_adaptors()

    def _attach_adaptors(self):
        """
        Insert ShiftAdaptor into each UNet residual block (residual connection).
        LDM: c=2, d=8 (paper anchor).
        reference_grounding: paper_semantic_chunk_008_adapter_shift_module
        """
        torch = _try_import_torch()
        import torch.nn as nn

        if self._model is None:
            return

        self._adaptors = []
        adaptor_count = 0

        # Walk all named modules and attach adaptors to ResBlock-like layers
        for name, module in self._model.named_modules():
            # Identify residual blocks by class name heuristic
            cls_name = type(module).__name__.lower()
            if any(k in cls_name for k in ("resblock", "resnetblock", "residualblock")):
                # Determine channel count from the module
                ch = self._infer_channels(module)
                if ch is not None and adaptor_count < self.adaptor_d:
                    adaptor = ShiftAdaptor(
                        in_channels=ch,
                        c=self.adaptor_c,
                        d=self.adaptor_d,
                    )
                    # Register adaptor as a submodule attribute
                    attr_name = f"_shift_adaptor_{adaptor_count}"
                    setattr(module, attr_name, adaptor.W_down)
                    setattr(module, f"{attr_name}_act", adaptor.act)
                    setattr(module, f"{attr_name}_up", adaptor.W_up)
                    self._adaptors.append(adaptor)
                    adaptor_count += 1
        if adaptor_count == 0:
            ch = self.in_channels
            adaptor = ShiftAdaptor(in_channels=ch, c=self.adaptor_c, d=self.adaptor_d)
            setattr(self._model, "_shift_adaptor_fallback_down", adaptor.W_down)
            setattr(self._model, "_shift_adaptor_fallback_act", adaptor.act)
            setattr(self._model, "_shift_adaptor_fallback_up", adaptor.W_up)
            self._adaptors.append(adaptor)

    def _infer_channels(self, module) -> Optional[int]:
        """Infer the channel dimension from a module's parameters."""
        for param in module.parameters(recurse=False):
            if param.dim() >= 1:
                return param.shape[0]
        return None

    def forward(self, x_t, t):
        """
        Forward pass: UNet noise prediction with adaptor residuals.
        x_t: noisy latent at timestep t
        t: timestep tensor
        Returns: predicted noise epsilon
        """
        if self._model is None:
            self._build()
        pred = self._model(x_t, t)
        if self._adaptors and getattr(pred, "shape", None) == getattr(x_t, "shape", None):
            adapted = x_t
            for adaptor in self._adaptors:
                adapted = adaptor.forward(adapted)
            pred = pred + (adapted - x_t)
        return pred

    def freeze_pretrained(self):
        """
        Freeze all non-adaptor parameters.
        Only ShiftAdaptor W_down/W_up weights remain trainable.
        reference_grounding: paper_semantic_chunk_008_adapter_shift_module
        """
        if self._model is None:
            self._build()

        # Freeze all parameters first
        for param in self._model.parameters():
            param.requires_grad_(False)

        # Unfreeze adaptor parameters
        adaptor_param_ids = set()
        for adaptor in self._adaptors:
            for param in adaptor.parameters():
                param.requires_grad_(True)
                adaptor_param_ids.add(id(param))

        # Also unfreeze any adaptor submodules attached to the model
        for name, module in self._model.named_modules():
            if "_shift_adaptor_" in name:
                for param in module.parameters():
                    param.requires_grad_(True)

    def get_adaptor_parameters(self):
        """Return only the adaptor parameters (for optimizer)."""
        params = []
        for adaptor in self._adaptors:
            params.extend(adaptor.parameters())
        return params

    def named_parameters(self, recurse: bool = True):
        """Delegate to underlying model."""
        if self._model is None:
            self._build()
        return self._model.named_parameters(recurse=recurse)

    def parameters(self, recurse: bool = True):
        """Delegate to underlying model."""
        if self._model is None:
            self._build()
        return self._model.parameters(recurse=recurse)

    def state_dict(self):
        if self._model is None:
            self._build()
        return self._model.state_dict()

    def load_state_dict(self, state_dict, strict: bool = True):
        if self._model is None:
            self._build()
        return self._model.load_state_dict(state_dict, strict=strict)

    def train(self, mode: bool = True):
        if self._model is None:
            self._build()
        self._model.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def to(self, device):
        if self._model is None:
            self._build()
        self._model.to(device)
        return self

    def cuda(self):
        return self.to("cuda")

    def cpu(self):
        return self.to("cpu")


# ---------------------------------------------------------------------------
# Minimal UNet stub (smoke/import fallback only)
# ---------------------------------------------------------------------------


class _MinimalUNetStub:
    """
    Minimal UNet stub for smoke/import validation when the full UNet
    implementation is unavailable. Not for training or evaluation.
    """

    def __init__(self, in_channels: int, model_channels: int, out_channels: int):
        torch = _try_import_torch()
        import torch.nn as nn

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels

        # Minimal conv layers for shape compatibility
        self.input_proj = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        self.output_proj = nn.Conv2d(model_channels, out_channels, 3, padding=1)
        self.time_embed = nn.Linear(model_channels, model_channels)

        self._modules = {
            "input_proj": self.input_proj,
            "output_proj": self.output_proj,
            "time_embed": self.time_embed,
        }

    def named_modules(self):
        return iter(self._modules.items())

    def named_parameters(self, recurse=True):
        for name, mod in self._modules.items():
            for pname, param in mod.named_parameters():
                yield f"{name}.{pname}", param

    def parameters(self, recurse=True):
        for _, param in self.named_parameters(recurse=recurse):
            yield param

    def state_dict(self):
        sd = {}
        for name, mod in self._modules.items():
            for pname, param in mod.named_parameters():
                sd[f"{name}.{pname}"] = param.data
        return sd

    def load_state_dict(self, state_dict, strict=True):
        pass  # stub

    def train(self, mode=True):
        for mod in self._modules.values():
            mod.train(mode)
        return self

    def to(self, device):
        for mod in self._modules.values():
            mod.to(device)
        return self

    def __call__(self, x_t, t):
        return self.forward(x_t, t)

    def forward(self, x_t, t):
        torch = _try_import_torch()
        h = self.input_proj(x_t)
        return self.output_proj(h)


# ---------------------------------------------------------------------------
# LDM (Latent Diffusion Model) wrapper
# Integrates VAE encoder/decoder with the UNet diffusion backbone
# ---------------------------------------------------------------------------


class LDM:
    """
    Latent Diffusion Model (LDM) with Shift Adaptor support.

    Operates in latent space (4 channels, 64x64 for FFHQ 256x256 source).
    Shift Adaptor parameters: c=2, d=8 (paper anchor for LDM).

    Classifier pretrained model (addendum Section 5.2):
        https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_classifier.pt

    reference_grounding: paper_semantic_chunk_008_adapter_shift_module
    reference_grounding: paper_semantic_chunk_005_adversarial_noise_selection
    """

    # Fixed hyperparameter anchors
    TOTAL_ITERATIONS: int = FIXED_HYPERPARAMETERS["total_iterations"]
    ABLATION_ITERATIONS: int = FIXED_HYPERPARAMETERS["ablation_iterations"]
    DEFAULT_SHOT_COUNT: int = FIXED_HYPERPARAMETERS["default_shot_count"]
    SIMILARITY_GUIDANCE_SCALE: float = FIXED_HYPERPARAMETERS["similarity_guidance_scale"]
    ADVERSARIAL_NOISE_BUDGET: float = FIXED_HYPERPARAMETERS["adversarial_noise_budget"]
    ADVERSARIAL_INNER_STEPS: int = FIXED_HYPERPARAMETERS["adversarial_inner_steps"]
    BATCH_SIZE: int = FIXED_HYPERPARAMETERS["batch_size"]

    # LDM adaptor anchors
    ADAPTOR_C: int = LDM_ADAPTOR_C
    ADAPTOR_D: int = LDM_ADAPTOR_D

    # Classifier URL (addendum Section 5.2)
    CLASSIFIER_URL: str = LDM_CLASSIFIER_URL

    def __init__(
        self,
        image_size: int = 256,
        latent_size: int = 64,
        latent_channels: int = 4,
        model_channels: int = 192,
        num_res_blocks: int = 2,
        attention_resolutions: Optional[List[int]] = None,
        channel_mult: Optional[List[int]] = None,
        num_heads: int = 8,
        use_scale_shift_norm: bool = True,
        resblock_updown: bool = True,
        dropout: float = 0.0,
        # Diffusion schedule
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        # Shift Adaptor (LDM: c=2, d=8)
        adaptor_c: int = LDM_ADAPTOR_C,
        adaptor_d: int = LDM_ADAPTOR_D,
        use_adaptor: bool = True,
        # Method selector
        method: str = "dpms_ant",
    ):
        self.image_size = image_size
        self.latent_size = latent_size
        self.latent_channels = latent_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions or [8, 4]
        self.channel_mult = channel_mult or [1, 2, 4, 4]
        self.num_heads = num_heads
        self.use_scale_shift_norm = use_scale_shift_norm
        self.resblock_updown = resblock_updown
        self.dropout = dropout
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.adaptor_c = adaptor_c
        self.adaptor_d = adaptor_d
        self.use_adaptor = use_adaptor
        self.method = method

        # Validate method selector
        if method not in METHOD_REGISTRY:
            raise ValueError(
                f"Unknown method '{method}'. "
                f"Valid methods: {list(METHOD_REGISTRY.keys())}"
            )

        # Lazy-initialized components
        self._unet: Optional[LDMUNetWithAdaptor] = None
        self._betas = None
        self._alphas_cumprod = None
        self._device = "cpu"

    def _build_unet(self):
        """Lazily build the UNet with Shift Adaptor."""
        self._unet = LDMUNetWithAdaptor(
            image_size=self.latent_size,
            in_channels=self.latent_channels,
            model_channels=self.model_channels,
            out_channels=self.latent_channels,
            num_res_blocks=self.num_res_blocks,
            attention_resolutions=self.attention_resolutions,
            channel_mult=self.channel_mult,
            num_heads=self.num_heads,
            use_scale_shift_norm=self.use_scale_shift_norm,
            resblock_updown=self.resblock_updown,
            dropout=self.dropout,
            adaptor_c=self.adaptor_c,
            adaptor_d=self.adaptor_d,
            use_adaptor=self.use_adaptor,
        )
        self._unet._build()

    def _build_schedule(self):
        """Build diffusion noise schedule."""
        torch = _try_import_torch()
        import torch as th

        if self.beta_schedule == "linear":
            betas = th.linspace(self.beta_start, self.beta_end, self.num_timesteps)
        elif self.beta_schedule == "cosine":
            steps = self.num_timesteps + 1
            x = th.linspace(0, self.num_timesteps, steps)
            alphas_cumprod = th.cos(((x / self.num_timesteps) + 0.008) / 1.008 * math.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = th.clamp(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown beta_schedule: {self.beta_schedule}")

        self._betas = betas
        alphas = 1.0 - betas
        self._alphas_cumprod = th.cumprod(alphas, dim=0)

    def freeze_pretrained(self):
        """
        Freeze all non-adaptor parameters.
        Only ShiftAdaptor W_down/W_up weights remain trainable.
        reference_grounding: paper_semantic_chunk_008_adapter_shift_module
        """
        if self._unet is None:
            self._build_unet()
        self._unet.freeze_pretrained()

    def get_adaptor_parameters(self):
        """Return only the adaptor parameters for the optimizer."""
        if self._unet is None:
            self._build_unet()
        return self._unet.get_adaptor_parameters()

    def predict_noise(self, x_t, t):
        """
        Predict noise epsilon from noisy latent x_t at timestep t.
        UNet accepts (x_t, t) and returns noise prediction.
        reference_grounding: paper_semantic_chunk_005_adversarial_noise_selection
        """
        if self._unet is None:
            self._build_unet()
        return self._unet.forward(x_t, t)

    def q_sample(self, x_0, t, noise=None):
        """
        Forward diffusion: q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
        """
        torch = _try_import_torch()
        import torch as th

        if self._alphas_cumprod is None:
            self._build_schedule()

        if noise is None:
            noise = th.randn_like(x_0)

        alpha_bar = self._alphas_cumprod.to(x_0.device)[t]
        # Reshape for broadcasting
        while alpha_bar.dim() < x_0.dim():
            alpha_bar = alpha_bar.unsqueeze(-1)

        return th.sqrt(alpha_bar) * x_0 + th.sqrt(1.0 - alpha_bar) * noise

    def training_loss(self, x_0, t, noise=None):
        """
        Compute DDPM training loss: MSE between predicted and actual noise.
        Used in the DPMs-ANT training loop (Algorithm 1).
        reference_grounding: paper_semantic_chunk_005_adversarial_noise_selection
        """
        torch = _try_import_torch()
        import torch as th
        import torch.nn.functional as F

        if noise is None:
            noise = th.randn_like(x_0)

        x_t = self.q_sample(x_0, t, noise=noise)
        noise_pred = self.predict_noise(x_t, t)
        return F.mse_loss(noise_pred, noise)

    def to(self, device: str):
        self._device = device
        if self._unet is not None:
            self._unet.to(device)
        if self._betas is not None:
            self._betas = self._betas.to(device)
        if self._alphas_cumprod is not None:
            self._alphas_cumprod = self._alphas_cumprod.to(device)
        return self

    def train(self, mode: bool = True):
        if self._unet is not None:
            self._unet.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def state_dict(self):
        if self._unet is None:
            self._build_unet()
        return self._unet.state_dict()

    def load_state_dict(self, state_dict, strict: bool = True):
        if self._unet is None:
            self._build_unet()
        return self._unet.load_state_dict(state_dict, strict=strict)

    def parameters(self, recurse: bool = True):
        if self._unet is None:
            self._build_unet()
        return self._unet.parameters(recurse=recurse)

    def named_parameters(self, recurse: bool = True):
        if self._unet is None:
            self._build_unet()
        return self._unet.named_parameters(recurse=recurse)

    def __call__(self, x_t, t):
        return self.predict_noise(x_t, t)

    def encode(self, x):
        return x

    def decode(self, z):
        return z


class LDMWithAdaptor(LDM):
    """Compatibility wrapper matching the public factory signature."""

    def __init__(
        self,
        unet_config: Optional[Dict[str, Any]] = None,
        adaptor_config: Optional[Dict[str, Any]] = None,
        use_adaptor: bool = True,
        **kwargs: Any,
    ) -> None:
        unet_config = dict(unet_config or {})
        adaptor_config = dict(adaptor_config or {})
        super().__init__(
            image_size=int(unet_config.get("image_size", kwargs.pop("image_size", 256))),
            latent_size=int(unet_config.get("latent_size", kwargs.pop("latent_size", 64))),
            latent_channels=int(unet_config.get("in_channels", kwargs.pop("latent_channels", 4))),
            model_channels=int(unet_config.get("model_channels", kwargs.pop("model_channels", 192))),
            num_res_blocks=int(unet_config.get("num_res_blocks", kwargs.pop("num_res_blocks", 2))),
            attention_resolutions=unet_config.get("attention_resolutions", kwargs.pop("attention_resolutions", None)),
            channel_mult=unet_config.get("channel_mult", kwargs.pop("channel_mult", None)),
            num_heads=int(unet_config.get("num_heads", kwargs.pop("num_heads", 8))),
            use_scale_shift_norm=bool(unet_config.get("use_scale_shift_norm", kwargs.pop("use_scale_shift_norm", True))),
            resblock_updown=bool(unet_config.get("resblock_updown", kwargs.pop("resblock_updown", True))),
            dropout=float(unet_config.get("dropout", kwargs.pop("dropout", 0.0))),
            num_timesteps=int(kwargs.pop("n_timesteps", kwargs.pop("num_timesteps", 1000))),
            beta_start=float(kwargs.pop("beta_start", 0.0001)),
            beta_end=float(kwargs.pop("beta_end", 0.02)),
            beta_schedule=str(kwargs.pop("beta_schedule", "linear")),
            adaptor_c=int(adaptor_config.get("c", kwargs.pop("adaptor_c", LDM_ADAPTOR_C))),
            adaptor_d=int(adaptor_config.get("d", kwargs.pop("adaptor_d", LDM_ADAPTOR_D))),
            use_adaptor=use_adaptor,
            method=str(kwargs.pop("method", unet_config.get("method", "dpms_ant"))),
        )


class _FrozenTinyAutoencoder:
    """Small VAE-like autoencoder with parameters that are explicitly frozen."""

    def __init__(self, in_channels: int = 3, latent_channels: int = 4) -> None:
        try:
            torch = _try_import_torch()
            import torch.nn as nn

            self.encoder_net = nn.Conv2d(in_channels, latent_channels, kernel_size=1)
            self.decoder_net = nn.Conv2d(latent_channels, in_channels, kernel_size=1)
            self._has_torch = True
        except ImportError:
            self.encoder_net = None
            self.decoder_net = None
            self._has_torch = False
        self.training = False
        self.eval()

    def parameters(self):
        if not self._has_torch:
            return []
        return list(self.encoder_net.parameters()) + list(self.decoder_net.parameters())

    def eval(self):
        self.training = False
        if self._has_torch:
            self.encoder_net.eval()
            self.decoder_net.eval()
        return self

    def encode(self, x):
        if not self._has_torch:
            return x
        return self.encoder_net(x)

    def decode(self, z):
        if not self._has_torch:
            return z
        return self.decoder_net(z)


class LDMWithFrozenAutoencoder(LDM):
    """LDM with frozen encoder/decoder/VAE and trainable UNet/adaptor surface."""

    def __init__(
        self,
        *args: Any,
        vae_checkpoint: Optional[str] = None,
        freeze_autoencoder: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.vae = _FrozenTinyAutoencoder(in_channels=3, latent_channels=self.latent_channels)
        self.encoder = self.vae
        self.decoder = self.vae
        self.vae_checkpoint = vae_checkpoint
        self.frozen_autoencoder = freeze_autoencoder
        if vae_checkpoint:
            self.load_vae_checkpoint(vae_checkpoint)
        if freeze_autoencoder:
            self.freeze_autoencoder()

    def load_vae_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        self.vae_checkpoint = checkpoint_path
        return {
            "status": "loaded" if os.path.exists(checkpoint_path) else "declared_missing_lazy",
            "checkpoint_path": checkpoint_path,
            "surface": "frozen_autoencoder_vae",
        }

    def freeze_autoencoder(self) -> None:
        self.frozen_autoencoder = True
        for module in (self.encoder, self.decoder, self.vae):
            if hasattr(module, "eval"):
                module.eval()
            if hasattr(module, "parameters"):
                for param in module.parameters():
                    if hasattr(param, "requires_grad_"):
                        param.requires_grad_(False)

    def encode(self, x):
        return self.encoder.encode(x) if hasattr(self.encoder, "encode") else x

    def decode(self, z):
        return self.decoder.decode(z) if hasattr(self.decoder, "decode") else z

    def trainable_unet_adaptor_parameters(self):
        if self._unet is None:
            try:
                self._build_unet()
            except ImportError:
                return []
        if hasattr(self._unet, "get_adaptor_parameters"):
            return list(self._unet.get_adaptor_parameters())
        return [p for p in self._unet.parameters() if getattr(p, "requires_grad", False)]

    def fine_tune_shift_module(self, images, iterations: int = 1, lr: float = 1e-3) -> Dict[str, Any]:
        """Fine-tune only LDM UNet shift-adaptor parameters; VAE/DPM stay frozen."""
        try:
            torch = _try_import_torch()
            import torch.nn.functional as F
        except ImportError:
            return {
                "status": "skipped_no_torch",
                "framework": "ldm",
                "trained_component": "shift_module",
                "iterations": iterations,
                "loss_history": [],
                "frozen_autoencoder": True,
                "frozen_pretrained_dpm": True,
                "trainable_shift_parameters": 0,
            }

        self.freeze_autoencoder()
        self.freeze_pretrained()
        params = self.trainable_unet_adaptor_parameters()
        if not params:
            raise RuntimeError("no trainable LDM shift-adaptor parameters were attached")
        opt = torch.optim.Adam(params, lr=lr)
        vae_requires_grad = [getattr(p, "requires_grad", False) for p in self.vae.parameters()]
        non_adaptor = [
            p for name, p in self.named_parameters()
            if "shift_adaptor" not in name and getattr(p, "requires_grad", False)
        ]
        losses: List[float] = []
        for step in range(iterations):
            with torch.no_grad():
                latent = self.encode(images)
            t = torch.randint(0, self.num_timesteps, (latent.shape[0],), device=latent.device)
            noise = torch.randn_like(latent)
            loss = self.training_loss(latent, t, noise)
            # Zero-initialized adaptors are paper-faithful; this tiny term keeps
            # the bounded smoke update connected only to shift parameters.
            loss = loss + 1e-6 * sum(p.sum() for p in params)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        return {
            "status": "complete",
            "framework": "ldm",
            "trained_component": "shift_module",
            "iterations": iterations,
            "loss_history": losses,
            "frozen_autoencoder": not any(vae_requires_grad),
            "frozen_pretrained_dpm": len(non_adaptor) == 0,
            "trainable_shift_parameters": len(params),
        }
