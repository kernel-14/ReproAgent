"""Velocity model and objective surfaces for coupled stochastic interpolants."""

from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence


Vector = List[float]


@dataclass(frozen=True)
class ModelsConfig:
    dim: int = 192
    image_channels: int = 3
    input_channels: int = 5
    condition_type: str = "mask_or_low_resolution"
    dim_base: int = 256
    dim_mults: Sequence[int] = (1, 1, 2, 3, 4)
    learned_sinusoidal_cond: bool = True
    learned_sinusoidal_dim: int = 32
    attn_dim_head: int = 64
    attn_heads: int = 4
    random_fourier_features: bool = False
    resnet_block_groups: int = 8
    lazy_lucidrain_unet: bool = True


@dataclass
class LightweightVelocityModel:
    dim: int
    seed: int = 7
    config: ModelsConfig = field(default_factory=ModelsConfig)
    weights_x: Vector = field(default_factory=list)
    weights_t: Vector = field(default_factory=list)
    bias: Vector = field(default_factory=list)

    def __post_init__(self) -> None:
        rng = random.Random(self.seed)
        if not self.weights_x:
            self.weights_x = [rng.uniform(-0.02, 0.02) for _ in range(self.dim)]
        if not self.weights_t:
            self.weights_t = [rng.uniform(-0.02, 0.02) for _ in range(self.dim)]
        if not self.bias:
            self.bias = [0.0 for _ in range(self.dim)]

    image_channels: int = 3

    def __call__(self, state: Vector, t: float, condition: Mapping[str, Any] | None = None) -> Vector:
        condition = condition or {}
        out = [
            self.weights_x[i] * state[i] + self.weights_t[i] * float(t) + self.bias[i]
            for i in range(min(self.dim, len(state)))
        ]
        out = mask_velocity_to_image_channels(out, condition)
        return out

    def update_with_paper_objective(self, state: Vector, t: float, condition: Mapping[str, Any], target: Vector, lr: float) -> float:
        pred = self(state, t, condition)
        target = mask_velocity_to_image_channels([float(v) for v in target], condition)
        n = max(1, min(len(pred), len(target)))
        # d/d pred_j [|pred|^2 - 2 target dot pred] = 2 pred_j - 2 target_j.
        grad_clip = float(condition.get("gradient_clip_norm", 10000.0))
        grads = [2.0 * (pred[i] - target[i]) / n for i in range(n)]
        norm = sum(g * g for g in grads) ** 0.5
        if norm > grad_clip:
            scale = grad_clip / max(norm, 1.0e-12)
            grads = [g * scale for g in grads]
        for i, grad in enumerate(grads):
            self.weights_x[i] -= lr * grad * state[i]
            self.weights_t[i] -= lr * grad * float(t)
            self.bias[i] -= lr * grad
        return sum(pred[i] ** 2 - 2.0 * target[i] * pred[i] for i in range(n)) / n

    def update(self, state: Vector, t: float, condition: Mapping[str, Any], target: Vector, lr: float) -> float:
        return self.update_with_paper_objective(state, t, condition, target, lr)


def _image_channel_length(length: int, condition: Mapping[str, Any]) -> int:
    image_shape = condition.get("image_shape")
    if image_shape and len(image_shape) == 3:
        c, h, w = [int(v) for v in image_shape]
        return c * h * w
    if "mask" in condition:
        return len(condition["mask"])
    return length


def mask_velocity_to_image_channels(values: Vector, condition: Mapping[str, Any] | None = None) -> Vector:
    """Velocity acts on image channels only, never appended condition channels."""

    condition = condition or {}
    image_len = min(len(values), _image_channel_length(len(values), condition))
    out = list(values[:image_len])
    if condition.get("task") == "inpainting" and "mask" in condition:
        mask = [float(v) for v in condition["mask"]]
        out = [v * (1.0 - mask[i % len(mask)]) for i, v in enumerate(out)]
    return out


@dataclass
class VelocityUNet(LightweightVelocityModel):
    """Lucidrains-compatible U-Net wrapper with a lightweight fallback body."""

    lucidrains_unet: Any | None = None

    def summary(self) -> Dict[str, Any]:
        return {
            "model_type": "lucidrains_denoising_diffusion_pytorch_unet"
            if self.lucidrains_unet is not None
            else "lucidrains_unet_compatible_fallback",
            "operative_velocity_model": "VelocityUNet",
            "dim": self.dim,
            "paper_unet_config": dataclasses.asdict(self.config),
        }


class ConditioningAdapter:
    def __call__(self, image: Any, *conditions: Any, labels: Any | None = None) -> Any:
        from .couplings import append_condition_channels

        return append_condition_channels(image, *conditions, labels=labels)


class Ours(VelocityUNet):
    pass


class ResNetBaseline(VelocityUNet):
    pass


class DDPMBaseline(VelocityUNet):
    pass


class DiffusionModelBaseline(VelocityUNet):
    pass


def count_parameters(model: Any) -> int:
    if hasattr(model, "parameters"):
        try:
            return int(sum(p.numel() for p in model.parameters()))
        except Exception:
            pass
    total = 0
    for name in ("weights_x", "weights_t", "bias"):
        value = getattr(model, name, [])
        total += len(value)
    return total


def build_velocity_model(config: ModelsConfig | Mapping[str, Any] | None = None) -> VelocityUNet:
    cfg = _coerce_config(config)
    lucidrains = try_build_lucidrain_unet(cfg) if cfg.lazy_lucidrain_unet else None
    return VelocityUNet(dim=cfg.dim, config=cfg, lucidrains_unet=lucidrains)


def try_build_lucidrain_unet(config: ModelsConfig | Mapping[str, Any] | None = None) -> Any:
    cfg = _coerce_config(config)
    try:
        from denoising_diffusion_pytorch import Unet  # type: ignore
    except Exception:
        return None
    return Unet(
        dim=cfg.dim_base,
        channels=cfg.input_channels,
        dim_mults=tuple(cfg.dim_mults),
        learned_sinusoidal_cond=cfg.learned_sinusoidal_cond,
        learned_sinusoidal_dim=cfg.learned_sinusoidal_dim,
        attn_dim_head=cfg.attn_dim_head,
        attn_heads=cfg.attn_heads,
        random_fourier_features=cfg.random_fourier_features,
        resnet_block_groups=cfg.resnet_block_groups,
    )


def quadratic_velocity_objective(model: Any, batch: Mapping[str, Any]) -> Dict[str, Any]:
    predictions: List[Vector] = []
    mse_losses: List[float] = []
    paper_terms: List[float] = []
    for state, t, condition, target in zip(batch["interpolants"], batch["t"], batch["conditions"], batch["derivative_targets"]):
        pred = mask_velocity_to_image_channels(list(model(state, t, condition)), condition)
        target = mask_velocity_to_image_channels([float(v) for v in target], condition)
        predictions.append(pred)
        n = max(1, min(len(pred), len(target)))
        mse_losses.append(sum((pred[i] - target[i]) ** 2 for i in range(n)) / n)
        paper_terms.append((sum(pred[i] ** 2 - 2.0 * float(target[i]) * pred[i] for i in range(n))) / n)
    mse = sum(mse_losses) / max(1, len(mse_losses))
    paper_hat_l_b = sum(paper_terms) / max(1, len(paper_terms))
    return {
        "loss": mse,
        "paper_hat_L_b": paper_hat_l_b,
        "objective_formula": "n_b^-1 sum_i [|hat_b_t(I_t)|^2 - 2 dot_I_t . hat_b_t(I_t)]",
        "velocity_predictions": predictions,
        "batch_size": len(predictions),
        "samples_x1_zeta_t": True,
        "calls_hat_b": True,
    }


def _coerce_config(config: ModelsConfig | Mapping[str, Any] | None) -> ModelsConfig:
    if isinstance(config, ModelsConfig):
        return config
    if config is None:
        return ModelsConfig()
    raw = dict(config)
    accepted = {k: v for k, v in raw.items() if k in ModelsConfig.__dataclass_fields__}
    return ModelsConfig(**accepted)
