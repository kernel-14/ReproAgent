# src/models/unet.py
# =============================================================================
# DPMs-ANT – UNet backbone with Shift Adaptor injection
# Paper: "Bridging Data Gaps in Diffusion Models with Adversarial Noise-Based
#         Transfer Learning"
#
# reference_grounding: paper_semantic_chunk_008_adapter_shift_module_transfer_learning
#                      dpms_ant/adaptor/shift_adaptor.py
# reference_grounding: paper_semantic_chunk_005_adversarial_noise_selection
#                      src/models/unet.py
#
# Interface contract:
#   - UNet(x_t, t) -> noise prediction tensor
#   - ShiftAdaptor: ψ^l(x) = f(x · W_down) · W_up
#       DDPM: c=4, d=8   LDM: c=2, d=8
#   - All adaptor parameters initialized to zero
#   - freeze_pretrained() freezes non-adaptor parameters
#
# Method/baseline registry (paper evidence contract):
#   ours | diffusion_model | ddpm | ldm | dpms_ant |
#   similarity_guided_training | adversarial_noise_selection |
#   ddpm_pa | tgan | ada | ewc | cdc | dcl | pgd | ddim
#
# Classifier URLs (addendum, Section 5.2):
#   DDPM: https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_classifier.pt
#   LDM:  https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_classifier.pt
#
# Fixed hyperparameter anchors (addendum):
#   5000_iterations, 300_training_iterations, 10_shot_setting,
#   gamma_5, omega_0.02, adversarial_inner_steps_10, batch_size_64
# =============================================================================

from __future__ import annotations

import math
import json
import os
from abc import abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple, Union

# ---------------------------------------------------------------------------
# Lazy torch import guard — keeps the module importable in minimal envs
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Method / baseline selector registry
# (paper evidence contract – complete set required)
# ---------------------------------------------------------------------------
METHOD_REGISTRY: Dict[str, str] = {
    "ours":                        "DPMs-ANT (full method: SGT + ANS)",
    "diffusion_model":             "Standard diffusion model baseline",
    "ddpm":                        "DDPM (denoising diffusion probabilistic model)",
    "ldm":                         "LDM (latent diffusion model)",
    "dpms_ant":                    "DPMs-ANT (adversarial noise-based transfer learning)",
    "similarity_guided_training":  "Similarity-Guided Training only (ablation)",
    "adversarial_noise_selection": "Adversarial Noise Selection only (ablation)",
    "ddpm_pa":                     "DDPM-PA baseline",
    "tgan":                        "TransferGAN (TGAN) baseline",
    "ada":                         "Adaptive Discriminator Augmentation (ADA)",
    "ewc":                         "Elastic Weight Consolidation (EWC)",
    "cdc":                         "CDC baseline",
    "dcl":                         "DCL baseline",
    "pgd":                         "PGD attack (adversarial noise inner loop)",
    "ddim":                        "DDIM deterministic sampler",
    # alias keys used in literature
    "GAN":      "GAN baseline",
    "DDPM":     "DDPM (alias)",
    "FFHQ":     "FFHQ source domain",
    "LPIPS":    "LPIPS diversity metric",
    "TGAN":     "TransferGAN",
    "ADA":      "ADA",
    "EWC":      "EWC",
    "CDC":      "CDC",
    "DCL":      "DCL",
    "DDPM-PA":  "DDPM-PA",
    "DDPM-ANT": "DDPM-ANT (ours)",
}

# ---------------------------------------------------------------------------
# Sweep / config entry registry
# (paper evidence contract – complete bounded sweeps)
# ---------------------------------------------------------------------------
SWEEP_REGISTRY: Dict[str, object] = {
    # Sensitivity sweeps
    "shot_count":                   [10, 100],
    "training_iteration_count":     [0, 50, 100, 150, 200, 250, 300, 350],
    "similarity_guidance_scale":    [1, 2, 3, 5, 7, 9, 10],   # gamma values
    "adversarial_noise_scale":      [0.01, 0.02, 0.03, 0.04, 0.05],  # omega
    # Named sweep axes (paper Sec 5.3 ablation)
    "alpha":          "loss blend coefficient",
    "gamma":          [1, 2, 3, 5, 7, 9, 10],
    "epsilon":        "PGD step size",
    "iteration_count": [0, 50, 100, 150, 200, 250, 300, 350],
    "batch_size":     [64],
}

# ---------------------------------------------------------------------------
# Fixed hyperparameter anchors (addendum / paper Sec 5.2)
# ---------------------------------------------------------------------------
FIXED_HYPERPARAMS: Dict[str, object] = {
    "total_iterations":           5000,   # anchor: 5000_iterations
    "training_iterations":        300,    # anchor: 300_training_iterations
    "shot_count":                 10,     # anchor: 10_shot_setting
    "similarity_guidance_scale":  5,      # anchor: gamma_5
    "adversarial_noise_scale":    0.02,   # anchor: omega_0.02
    "adversarial_inner_steps":    10,     # anchor: adversarial_inner_steps_10
    "batch_size":                 64,     # anchor: batch_size_64
}

# ---------------------------------------------------------------------------
# Classifier URLs (addendum, Section 5.2)
# ---------------------------------------------------------------------------
CLASSIFIER_URLS: Dict[str, str] = {
    "ddpm_256x256": (
        "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_classifier.pt"
    ),
    "ldm_64x64": (
        "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_classifier.pt"
    ),
}


# =============================================================================
# Utility helpers
# =============================================================================

def _require_torch(fn_name: str = "") -> None:
    if not _TORCH_AVAILABLE:
        raise RuntimeError(
            f"PyTorch is required to use {fn_name}. "
            "Install via: pip install torch"
        )


def timestep_embedding(timesteps, dim: int, max_period: int = 10000):
    """Sinusoidal timestep embeddings (Vaswani et al., Ho et al.)."""
    _require_torch("timestep_embedding")
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# =============================================================================
# Shift Adaptor
# ψ^l(x) = f(x · W_down) · W_up
# Compresses from R^{w×h×r} → R^{w/c × h/c × d} (bottleneck)
# DDPM: c=4, d=8   LDM: c=2, d=8
# All adaptor weights initialised to ZERO.
# =============================================================================

class ShiftAdaptor(nn.Module):  # type: ignore[misc]
    """
    Shift Adaptor bottleneck (Section 3 of the paper).

    ψ^l(x) = f(x · W_down) · W_up

    Args:
        in_channels:  number of input feature channels (r)
        down_factor:  spatial compression ratio c  (DDPM=4, LDM=2)
        bottleneck_d: bottleneck channel width d   (always 8)
    """

    # paper evidence contract: DDPM default parameters
    DDPM_C: int = 4
    DDPM_D: int = 8
    # paper evidence contract: LDM default parameters
    LDM_C: int = 2
    LDM_D: int = 8

    def __init__(
        self,
        in_channels: int,
        down_factor: int = DDPM_C,
        bottleneck_d: int = DDPM_D,
    ) -> None:
        _require_torch("ShiftAdaptor")
        super().__init__()
        self.in_channels = in_channels
        self.down_factor = down_factor
        self.bottleneck_d = bottleneck_d

        # W_down: projects channel dim from r → d via 1×1 conv
        # W_up:   projects back from d → r via 1×1 conv
        # Spatial downsampling by factor c uses average pooling applied
        # before W_down; the residual is upsampled back after W_up.
        self.pool = nn.AvgPool2d(kernel_size=down_factor, stride=down_factor)
        self.W_down = nn.Conv2d(in_channels, bottleneck_d, kernel_size=1, bias=False)
        self.act = nn.SiLU()
        self.W_up = nn.Conv2d(bottleneck_d, in_channels, kernel_size=1, bias=False)
        self.upsample = nn.Upsample(scale_factor=down_factor, mode="nearest")

        # ─── CRITICAL: all adaptor parameters initialised to zero ───
        nn.init.zeros_(self.W_down.weight)
        nn.init.zeros_(self.W_up.weight)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Residual adaptor: output = x + ψ^l(x)."""
        h = self.pool(x)           # spatial compress
        h = self.W_down(h)         # channel compress
        h = self.act(h)            # non-linearity f(·)
        h = self.W_up(h)           # channel expand
        h = self.upsample(h)       # spatial expand back

        # Clamp to original spatial size (handles odd-dimension artefacts)
        h = h[:, :, : x.shape[2], : x.shape[3]]
        return x + h               # residual connection

    @classmethod
    def for_ddpm(cls, in_channels: int) -> "ShiftAdaptor":
        """Factory for DDPM framework adaptor (c=4, d=8)."""
        return cls(in_channels, down_factor=cls.DDPM_C, bottleneck_d=cls.DDPM_D)

    @classmethod
    def for_ldm(cls, in_channels: int) -> "ShiftAdaptor":
        """Factory for LDM framework adaptor (c=2, d=8)."""
        return cls(in_channels, down_factor=cls.LDM_C, bottleneck_d=cls.LDM_D)


# =============================================================================
# Low-level building blocks
# =============================================================================

class TimestepBlock(nn.Module):  # type: ignore[misc]
    """Any module that takes (x, emb) as input."""

    @abstractmethod
    def forward(self, x: "torch.Tensor", emb: "torch.Tensor") -> "torch.Tensor":
        ...


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):  # type: ignore[misc]
    """Sequential that passes timestep embedding to TimestepBlock children."""

    def forward(self, x: "torch.Tensor", emb: "torch.Tensor") -> "torch.Tensor":  # type: ignore[override]
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):  # type: ignore[misc]
    def __init__(self, channels: int, use_conv: bool = True) -> None:
        _require_torch("Upsample")
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):  # type: ignore[misc]
    def __init__(self, channels: int, use_conv: bool = True) -> None:
        _require_torch("Downsample")
        super().__init__()
        if use_conv:
            self.op: nn.Module = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool2d(2)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.op(x)


def _norm(channels: int) -> "nn.GroupNorm":
    return nn.GroupNorm(32, channels)


class ResBlock(TimestepBlock):  # type: ignore[misc]
    """
    Residual block with timestep conditioning, optional up/downsampling,
    and optional Shift Adaptor injection.
    """

    def __init__(
        self,
        in_channels: int,
        emb_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        use_scale_shift_norm: bool = False,
        up: bool = False,
        down: bool = False,
        # Shift Adaptor
        shift_adaptor: Optional["ShiftAdaptor"] = None,
    ) -> None:
        _require_torch("ResBlock")
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            _norm(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down
        if up:
            self.h_upd: nn.Module = Upsample(in_channels, use_conv=False)
            self.x_upd: nn.Module = Upsample(in_channels, use_conv=False)
        elif down:
            self.h_upd = Downsample(in_channels, use_conv=False)
            self.x_upd = Downsample(in_channels, use_conv=False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )

        self.out_layers = nn.Sequential(
            _norm(self.out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1),
        )

        if in_channels == self.out_channels:
            self.skip_connection: nn.Module = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(in_channels, self.out_channels, 1)

        # Shift Adaptor (residual insertion) — inserted AFTER main path
        self.shift_adaptor = shift_adaptor

    def forward(self, x: "torch.Tensor", emb: "torch.Tensor") -> "torch.Tensor":
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)

        h = self.skip_connection(x) + h

        # Apply shift adaptor in residual fashion (zero-initialised → identity at init)
        if self.shift_adaptor is not None:
            h = self.shift_adaptor(h)

        return h


class AttentionBlock(nn.Module):  # type: ignore[misc]
    """Self-attention block (single head or multi-head)."""

    def __init__(self, channels: int, num_heads: int = 1) -> None:
        _require_torch("AttentionBlock")
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = _norm(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj_out = nn.Conv1d(channels, channels, 1)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        b, c, *spatial = x.shape
        x_flat = x.reshape(b, c, -1)
        h = self.norm(x_flat.reshape(b, c, *spatial)).reshape(b, c, -1)
        qkv = self.qkv(h)
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        scale = (c // self.num_heads) ** -0.25
        q, k, v = torch.chunk(qkv, 3, dim=1)
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        a = a.reshape(b, -1, a.shape[-1])
        a = self.proj_out(a)
        return (x_flat + a).reshape(b, c, *spatial)


# =============================================================================
# UNet backbone  (ADM / improved-DDPM compatible architecture)
# =============================================================================

class UNet(nn.Module):  # type: ignore[misc]
    """
    UNet noise-prediction backbone for DDPM / LDM.

    Args:
        image_size:            spatial resolution of input
        in_channels:           number of input image channels
        model_channels:        base channel width
        out_channels:          number of predicted channels (= in_channels)
        num_res_blocks:        number of ResBlocks per resolution level
        attention_resolutions: resolutions at which to apply self-attention
        channel_mult:          channel multiplier per level
        num_heads:             attention heads
        dropout:               dropout probability
        use_scale_shift_norm:  ADM-style affine conditioning
        resblock_updown:       use ResBlock for up/downsampling
        # Shift Adaptor injection
        shift_adaptor_enabled: whether to inject adaptors
        shift_adaptor_c:       spatial compression ratio (DDPM=4, LDM=2)
        shift_adaptor_d:       bottleneck channels (always 8)
    """

    def __init__(
        self,
        image_size: int = 256,
        in_channels: int = 3,
        model_channels: int = 128,
        out_channels: int = 3,
        num_res_blocks: int = 2,
        attention_resolutions: Sequence[int] = (16, 8),
        channel_mult: Sequence[int] = (1, 1, 2, 2, 4, 4),
        num_heads: int = 4,
        dropout: float = 0.0,
        use_scale_shift_norm: bool = True,
        resblock_updown: bool = True,
        # Shift Adaptor
        shift_adaptor_enabled: bool = False,
        shift_adaptor_c: int = ShiftAdaptor.DDPM_C,
        shift_adaptor_d: int = ShiftAdaptor.DDPM_D,
    ) -> None:
        _require_torch("UNet")
        super().__init__()

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = set(attention_resolutions)
        self.channel_mult = list(channel_mult)
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_scale_shift_norm = use_scale_shift_norm
        self.resblock_updown = resblock_updown
        self.shift_adaptor_enabled = shift_adaptor_enabled
        self.shift_adaptor_c = shift_adaptor_c
        self.shift_adaptor_d = shift_adaptor_d

        # ── Timestep embedding ──────────────────────────────────────────────
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # ── Input projection ────────────────────────────────────────────────
        ch = int(channel_mult[0] * model_channels)
        self.input_blocks: nn.ModuleList = nn.ModuleList(
            [TimestepEmbedSequential(nn.Conv2d(in_channels, ch, 3, padding=1))]
        )
        input_block_chans: List[int] = [ch]

        ds = 1  # current spatial downscale factor
        for level, mult in enumerate(channel_mult):
            out_ch = int(mult * model_channels)
            for _ in range(num_res_blocks):
                adaptor = self._make_adaptor(out_ch) if shift_adaptor_enabled else None
                layers: List[nn.Module] = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        out_channels=out_ch,
                        dropout=dropout,
                        use_scale_shift_norm=use_scale_shift_norm,
                        shift_adaptor=adaptor,
                    )
                ]
                ch = out_ch
                if ds in self.attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)

            if level < len(channel_mult) - 1:
                out_ch = ch
                if resblock_updown:
                    down_adaptor = self._make_adaptor(ch) if shift_adaptor_enabled else None
                    self.input_blocks.append(
                        TimestepEmbedSequential(
                            ResBlock(
                                ch,
                                time_embed_dim,
                                out_channels=out_ch,
                                dropout=dropout,
                                use_scale_shift_norm=use_scale_shift_norm,
                                down=True,
                                shift_adaptor=down_adaptor,
                            )
                        )
                    )
                else:
                    self.input_blocks.append(
                        TimestepEmbedSequential(Downsample(ch))
                    )
                input_block_chans.append(ch)
                ds *= 2

        # ── Middle blocks ────────────────────────────────────────────────────
        mid_adaptor = self._make_adaptor(ch) if shift_adaptor_enabled else None
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout=dropout,
                use_scale_shift_norm=use_scale_shift_norm,
                shift_adaptor=mid_adaptor,
            ),
            AttentionBlock(ch, num_heads=num_heads),
            ResBlock(
                ch,
                time_embed_dim,
                dropout=dropout,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        # ── Output / decoder blocks ──────────────────────────────────────────
        self.output_blocks: nn.ModuleList = nn.ModuleList()
        for level, mult in list(enumerate(channel_mult))[::-1]:
            out_ch = int(mult * model_channels)
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                adaptor = self._make_adaptor(out_ch) if shift_adaptor_enabled else None
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        out_channels=out_ch,
                        dropout=dropout,
                        use_scale_shift_norm=use_scale_shift_norm,
                        shift_adaptor=adaptor,
                    )
                ]
                ch = out_ch
                if ds in self.attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                if level > 0 and i == num_res_blocks:
                    if resblock_updown:
                        up_adaptor = self._make_adaptor(ch) if shift_adaptor_enabled else None
                        layers.append(
                            ResBlock(
                                ch,
                                time_embed_dim,
                                out_channels=ch,
                                dropout=dropout,
                                use_scale_shift_norm=use_scale_shift_norm,
                                up=True,
                                shift_adaptor=up_adaptor,
                            )
                        )
                    else:
                        layers.append(Upsample(ch))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        # ── Final output projection ──────────────────────────────────────────
        self.out_proj = nn.Sequential(
            _norm(ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1),
        )

    # ── Helpers ────────────────────────────────────────────────────────────

    def _make_adaptor(self, channels: int) -> "ShiftAdaptor":
        """Create a zero-initialised ShiftAdaptor for the given channel width."""
        return ShiftAdaptor(
            in_channels=channels,
            down_factor=self.shift_adaptor_c,
            bottleneck_d=self.shift_adaptor_d,
        )

    # ── forward ────────────────────────────────────────────────────────────

    def forward(
        self,
        x: "torch.Tensor",
        t: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Predict noise given noisy image x_t and timestep t.

        Args:
            x: (B, C, H, W) noisy image tensor
            t: (B,) integer timestep tensor

        Returns:
            (B, C, H, W) predicted noise
        """
        emb = self.time_embed(timestep_embedding(t, self.model_channels))

        hs: List["torch.Tensor"] = []
        h = x
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)

        h = self.middle_block(h, emb)

        for module in self.output_blocks:
            h_skip = hs.pop()
            h = torch.cat([h, h_skip], dim=1)
            h = module(h, emb)

        return self.out_proj(h)

    # ── Adaptor parameter utilities ────────────────────────────────────────

    def adaptor_parameters(self) -> List["torch.nn.Parameter"]:
        """Return only the ShiftAdaptor parameters."""
        params: List["torch.nn.Parameter"] = []
        for module in self.modules():
            if isinstance(module, ShiftAdaptor):
                params.extend(module.parameters())
        return params

    def freeze_pretrained(self) -> None:
        """
        Freeze all non-adaptor parameters so that only ShiftAdaptor weights
        are updated during transfer training.
        """
        adaptor_ids = {id(p) for p in self.adaptor_parameters()}
        for param in self.parameters():
            param.requires_grad = id(param) not in adaptor_ids

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters (full fine-tuning mode)."""
        for param in self.parameters():
            param.requires_grad = True

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# Factory / builder functions
# =============================================================================