"""Data-dependent coupling implementations for ImageNet inpainting and SR."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


def _optional_torch() -> Any | None:
    try:
        import torch  # type: ignore
    except Exception:
        return None
    return torch


def _is_torch(x: Any) -> bool:
    return hasattr(x, "detach") and hasattr(x, "shape")


def _randn_like(x: Any, seed: int | None = None) -> Any:
    torch = _optional_torch()
    if torch is not None and _is_torch(x):
        if seed is None:
            return torch.randn_like(x)
        generator = torch.Generator(device=x.device)
        generator.manual_seed(int(seed))
        return torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
    rng = np.random.default_rng(seed)
    return rng.standard_normal(np.asarray(x).shape).astype(np.float32)


def _as_shape(x_or_shape: Any) -> tuple[int, int, int, int]:
    if isinstance(x_or_shape, tuple):
        if len(x_or_shape) == 4:
            return tuple(int(v) for v in x_or_shape)  # type: ignore[return-value]
        if len(x_or_shape) == 3:
            c, h, w = (int(v) for v in x_or_shape)
            return (1, c, h, w)
    shape = tuple(int(v) for v in getattr(x_or_shape, "shape", np.asarray(x_or_shape).shape))
    if len(shape) == 4:
        return shape  # type: ignore[return-value]
    if len(shape) == 3:
        c, h, w = shape
        return (1, c, h, w)
    if len(shape) == 1:
        return (1, 1, 1, shape[0])
    raise ValueError(f"expected BCHW/CHW/flat image shape, got {shape!r}")


def _reshape_like(mask: Any, x: Any) -> Any:
    shape = tuple(getattr(x, "shape", np.asarray(x).shape))
    if len(shape) == 3 and getattr(mask, "ndim", np.asarray(mask).ndim) == 4:
        return mask[0]
    return mask


def _tile_bounds(index: int, side: int, length: int) -> tuple[int, int]:
    start = int(round(index * length / side))
    end = int(round((index + 1) * length / side))
    return start, max(start + 1, end)


def make_inpainting_mask(
    x_or_shape: Any,
    tiles: int = 64,
    probability: float = 0.3,
    seed: int | None = None,
    *,
    observed_value: float = 1.0,
    missing_value: float = 0.0,
) -> Any:
    """Return an 8x8 tile Bernoulli mask shared across channels.

    The paper uses 64 equal spatial tiles and samples each tile into the mask
    with probability p=0.3.  The same binary xi value is then repeated for every
    channel at that spatial location.
    """

    batch, channels, height, width = _as_shape(x_or_shape)
    side = int(math.sqrt(int(tiles)))
    if side * side != int(tiles):
        raise ValueError("inpainting mask requires a square tile count; paper anchor is 64")
    rng = np.random.default_rng(seed)
    spatial = np.empty((batch, height, width), dtype=np.float32)
    for b in range(batch):
        for ty in range(side):
            y0, y1 = _tile_bounds(ty, side, height)
            for tx in range(side):
                x0, x1 = _tile_bounds(tx, side, width)
                keep = observed_value if rng.random() >= probability else missing_value
                spatial[b, y0:y1, x0:x1] = keep
    mask_np = np.repeat(spatial[:, None, :, :], channels, axis=1)
    torch = _optional_torch()
    if torch is not None and _is_torch(x_or_shape):
        mask = torch.as_tensor(mask_np, device=x_or_shape.device, dtype=x_or_shape.dtype)
        return _reshape_like(mask, x_or_shape)
    mask_np = _reshape_like(mask_np, x_or_shape)
    return mask_np


def _center_crop_numpy(array: np.ndarray, crop_h: int, crop_w: int) -> np.ndarray:
    h, w = array.shape[-2], array.shape[-1]
    top = max(0, (h - crop_h) // 2)
    left = max(0, (w - crop_w) // 2)
    return array[..., top : top + crop_h, left : left + crop_w]


def _nearest_upsample_numpy(array: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    src_h, src_w = array.shape[-2], array.shape[-1]
    y_idx = np.minimum(src_h - 1, (np.arange(target_h) * src_h / max(1, target_h)).astype(np.int64))
    x_idx = np.minimum(src_w - 1, (np.arange(target_w) * src_w / max(1, target_w)).astype(np.int64))
    return array[..., y_idx, :][..., x_idx]


def make_low_resolution_condition(
    x1: Any,
    low_resolution: int | None = None,
    target_resolution: int | None = None,
) -> Any:
    """Crop to 64/256 and nearest-neighbour upsample back to original size."""

    shape = tuple(getattr(x1, "shape", np.asarray(x1).shape))
    added_batch = False
    if len(shape) == 3:
        added_batch = True
        input_shape = (1, *shape)
    elif len(shape) == 4:
        input_shape = shape
    else:
        raise ValueError(f"super-resolution condition expects CHW or BCHW, got {shape!r}")
    _, _, height, width = input_shape
    target = int(target_resolution or height)
    low = int(low_resolution or (64 if target <= 256 else 256))

    torch = _optional_torch()
    if torch is not None and _is_torch(x1):
        x = x1.unsqueeze(0) if added_batch else x1
        crop_h, crop_w = min(low, height), min(low, width)
        top = max(0, (height - crop_h) // 2)
        left = max(0, (width - crop_w) // 2)
        cropped = x[..., top : top + crop_h, left : left + crop_w]
        upsampled = torch.nn.functional.interpolate(cropped, size=(height, width), mode="nearest")
        return upsampled[0] if added_batch else upsampled

    arr = np.asarray(x1, dtype=np.float32).reshape(input_shape)
    cropped_np = _center_crop_numpy(arr, min(low, height), min(low, width))
    upsampled_np = _nearest_upsample_numpy(cropped_np, height, width)
    return upsampled_np[0] if added_batch else upsampled_np


def append_class_label_channel(x: Any, labels: Any | None, num_classes: int = 1000) -> Any:
    """Append a constant per-sample class-value channel to BCHW/CHW input."""

    if labels is None:
        return x
    shape = tuple(getattr(x, "shape", np.asarray(x).shape))
    added_batch = len(shape) == 3
    if added_batch:
        shape = (1, *shape)
    batch, _, height, width = shape
    torch = _optional_torch()
    if torch is not None and _is_torch(x):
        tensor = x.unsqueeze(0) if added_batch else x
        label_tensor = labels
        if not hasattr(label_tensor, "detach"):
            label_tensor = torch.as_tensor(labels, device=tensor.device)
        label_tensor = label_tensor.to(device=tensor.device, dtype=tensor.dtype).view(batch, 1, 1, 1)
        label_tensor = label_tensor / float(max(1, num_classes - 1))
        channel = label_tensor.expand(batch, 1, height, width)
        out = torch.cat([tensor, channel], dim=1)
        return out[0] if added_batch else out
    arr = np.asarray(x, dtype=np.float32).reshape(shape)
    label_arr = np.asarray(labels, dtype=np.float32).reshape(batch, 1, 1, 1) / float(max(1, num_classes - 1))
    channel_np = np.broadcast_to(label_arr, (batch, 1, height, width)).astype(np.float32)
    out_np = np.concatenate([arr, channel_np], axis=1)
    return out_np[0] if added_batch else out_np


def append_condition_channels(x: Any, *conditions: Any, labels: Any | None = None) -> Any:
    """Concatenate image, condition images/masks, and optional class channel."""

    shape = tuple(getattr(x, "shape", np.asarray(x).shape))
    added_batch = len(shape) == 3
    torch = _optional_torch()
    if torch is not None and _is_torch(x):
        base = x.unsqueeze(0) if added_batch else x
        tensors = [base]
        for cond in conditions:
            if cond is None:
                continue
            c = cond
            if not hasattr(c, "detach"):
                c = torch.as_tensor(cond, device=base.device, dtype=base.dtype)
            if c.ndim == 3:
                c = c.unsqueeze(0)
            tensors.append(c.to(device=base.device, dtype=base.dtype))
        combined = torch.cat(tensors, dim=1)
        combined = append_class_label_channel(combined, labels)
        return combined[0] if added_batch else combined
    base_np = np.asarray(x, dtype=np.float32)
    if added_batch:
        base_np = base_np[None, ...]
    arrays = [base_np]
    for cond in conditions:
        if cond is None:
            continue
        arr = np.asarray(cond, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[None, ...]
        arrays.append(arr)
    combined_np = np.concatenate(arrays, axis=1)
    combined_np = append_class_label_channel(combined_np, labels)
    return combined_np[0] if added_batch else combined_np


@dataclass
class CoupledBatch:
    x0: Any
    x1: Any
    zeta: Any | None = None
    t: Any | None = None
    conditioning: Mapping[str, Any] = field(default_factory=dict)
    interpolant: Any | None = None
    d_interpolant: Any | None = None
    model_input: Any | None = None


@dataclass
class IndependentGaussianCoupling:
    sigma: float = 1.0

    def sample(self, x1: Any, zeta: Any | None = None, t: Any | None = None, conditioning: Mapping[str, Any] | None = None) -> CoupledBatch:
        zeta = _randn_like(x1) if zeta is None else zeta
        x0 = zeta
        return CoupledBatch(x0=x0, x1=x1, zeta=zeta, t=t, conditioning=dict(conditioning or {}))


@dataclass
class InpaintingCoupling:
    mask_tiles: int = 64
    mask_probability: float = 0.3
    sigma: float = 1.0
    seed: int = 1234

    def sample(self, x1: Any, zeta: Any | None = None, t: Any | None = None, conditioning: Mapping[str, Any] | None = None) -> CoupledBatch:
        cond = dict(conditioning or {})
        mask = cond.get("mask")
        if mask is None:
            mask = make_inpainting_mask(x1, self.mask_tiles, self.mask_probability, self.seed)
        zeta = _randn_like(x1, self.seed) if zeta is None else zeta
        visible = x1 * mask
        x0 = visible + (1.0 - mask) * zeta
        cond.update(
            {
                "task": "inpainting",
                "mask": mask,
                "xi": mask,
                "observed_image": visible,
                "visible_pixels": visible,
                "condition_image": visible,
                "mask_tiles": self.mask_tiles,
                "mask_probability": self.mask_probability,
                "x0_formula": "x0 = xi * x1 + (1 - xi) * zeta, zeta ~ N(0,I_d)",
            }
        )
        model_input = append_condition_channels(x0, visible, mask, labels=cond.get("labels"))
        return CoupledBatch(x0=x0, x1=x1, zeta=zeta, t=t, conditioning=cond, model_input=model_input)


@dataclass
class SuperResolutionCoupling:
    low_resolution: int = 64
    target_resolution: int = 256
    sigma: float = 1.0
    seed: int = 1234

    def sample(self, x1: Any, zeta: Any | None = None, t: Any | None = None, conditioning: Mapping[str, Any] | None = None) -> CoupledBatch:
        cond = dict(conditioning or {})
        low_up = cond.get("low_resolution_upsampled")
        if low_up is None:
            low_up = make_low_resolution_condition(x1, self.low_resolution, self.target_resolution)
        zeta = _randn_like(x1, self.seed) if zeta is None else zeta
        x0 = low_up + self.sigma * zeta
        cond.update(
            {
                "task": "super_resolution",
                "low_resolution_upsampled": low_up,
                "condition_image": low_up,
                "crop_downsample": f"center crop to {self.low_resolution}x{self.low_resolution}",
                "upsampling": "nearest",
                "x0_formula": "x0 = U(D(x1)) + sigma * zeta, zeta ~ N(0,I_d)",
            }
        )
        model_input = append_condition_channels(x0, low_up, labels=cond.get("labels"))
        return CoupledBatch(x0=x0, x1=x1, zeta=zeta, t=t, conditioning=cond, model_input=model_input)


DataDependentCoupling = InpaintingCoupling


def build_coupling(config: Mapping[str, Any] | Any | None = None, **overrides: Any) -> Any:
    raw = {}
    if config is not None:
        if hasattr(config, "__dict__"):
            raw.update(vars(config))
        elif isinstance(config, Mapping):
            raw.update(config)
    raw.update({k: v for k, v in overrides.items() if v is not None})
    task = str(raw.get("task", "inpainting")).replace("-", "_")
    coupling_type = str(raw.get("coupling_type", raw.get("coupling", "data_dependent")))
    if coupling_type in {"independent", "independent_gaussian", "gaussian"}:
        return IndependentGaussianCoupling(sigma=float(raw.get("sigma", 1.0)))
    if task in {"super_resolution", "sr"}:
        target = int(raw.get("resolution", raw.get("target_resolution", 256)))
        return SuperResolutionCoupling(
            low_resolution=int(raw.get("low_resolution", 64 if target <= 256 else 256)),
            target_resolution=target,
            sigma=float(raw.get("sigma", 1.0)),
            seed=int(raw.get("seed", 1234)),
        )
    return InpaintingCoupling(
        mask_tiles=int(raw.get("mask_tiles", 64)),
        mask_probability=float(raw.get("mask_probability", 0.3)),
        sigma=float(raw.get("sigma", 1.0)),
        seed=int(raw.get("seed", 1234)),
    )


def sample_coupled_batch(x1: Any, config: Mapping[str, Any] | Any | None = None, conditioning: Mapping[str, Any] | None = None) -> CoupledBatch:
    coupling = build_coupling(config)
    return coupling.sample(x1=x1, conditioning=conditioning)
