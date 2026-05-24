"""Core stochastic-interpolant schedules and objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


def _optional_torch() -> Any | None:
    try:
        import torch  # type: ignore
    except Exception:
        return None
    return torch


def _is_torch(x: Any) -> bool:
    return hasattr(x, "detach") and hasattr(x, "shape")


def sample_standard_normal(x_like: Any, seed: int | None = None) -> Any:
    torch = _optional_torch()
    if torch is not None and _is_torch(x_like):
        if seed is None:
            return torch.randn_like(x_like)
        generator = torch.Generator(device=x_like.device)
        generator.manual_seed(int(seed))
        return torch.randn(x_like.shape, generator=generator, device=x_like.device, dtype=x_like.dtype)
    rng = np.random.default_rng(seed)
    return rng.standard_normal(np.asarray(x_like).shape).astype(np.float32)


def sample_time_uniform(batch_size: int, x_like: Any | None = None, seed: int | None = None) -> Any:
    torch = _optional_torch()
    if torch is not None and x_like is not None and _is_torch(x_like):
        if seed is None:
            return torch.rand((batch_size,), device=x_like.device, dtype=x_like.dtype)
        generator = torch.Generator(device=x_like.device)
        generator.manual_seed(int(seed))
        return torch.rand((batch_size,), generator=generator, device=x_like.device, dtype=x_like.dtype)
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=(batch_size,)).astype(np.float32)


def alpha_t(t: Any) -> Any:
    return t


def beta_t(t: Any) -> Any:
    return 1.0 - t


def dalpha_t(t: Any) -> Any:
    torch = _optional_torch()
    if torch is not None and _is_torch(t):
        return torch.ones_like(t)
    return np.ones_like(np.asarray(t), dtype=np.float32)


def dbeta_t(t: Any) -> Any:
    torch = _optional_torch()
    if torch is not None and _is_torch(t):
        return -torch.ones_like(t)
    return -np.ones_like(np.asarray(t), dtype=np.float32)


def _broadcast_time(t: Any, x: Any) -> Any:
    if _is_torch(x):
        while t.ndim < x.ndim:
            t = t.view(*t.shape, 1)
        return t
    arr = np.asarray(t)
    while arr.ndim < np.asarray(x).ndim:
        arr = arr.reshape(*arr.shape, 1)
    return arr


@dataclass
class InterpolantState:
    I_t: Any
    dI_t: Any
    t: Any
    formula: str = "I_t = t * x0 + (1 - t) * x1"
    derivative_formula: str = "dot_I_t = x0 - x1"


class StochasticInterpolant:
    """Paper interpolant I_t = t x0 + (1-t) x1."""

    def __call__(self, x0: Any, x1: Any, t: Any) -> InterpolantState:
        return interpolant_state(x0, x1, t)


def interpolant_state(x0: Any, x1: Any, t: Any) -> InterpolantState:
    tb = _broadcast_time(t, x1)
    I_t = tb * x0 + (1.0 - tb) * x1
    dI_t = x0 - x1
    return InterpolantState(I_t=I_t, dI_t=dI_t, t=t)


def interpolant_derivative(x0: Any, x1: Any, t: Any | None = None) -> Any:
    del t
    return x0 - x1


def _sum_nonbatch(x: Any) -> Any:
    if _is_torch(x):
        if x.ndim <= 1:
            return x.sum()
        return x.flatten(1).sum(dim=1)
    arr = np.asarray(x)
    if arr.ndim <= 1:
        return np.sum(arr)
    return np.sum(arr.reshape(arr.shape[0], -1), axis=1)


def velocity_field_objective(coupled_batch: Any, velocity_model: Any, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    del config
    I_t = getattr(coupled_batch, "interpolant", None)
    dI_t = getattr(coupled_batch, "d_interpolant", None)
    if I_t is None or dI_t is None:
        state = interpolant_state(getattr(coupled_batch, "x0"), getattr(coupled_batch, "x1"), getattr(coupled_batch, "t"))
        I_t, dI_t = state.I_t, state.dI_t
    condition = getattr(coupled_batch, "conditioning", {})
    try:
        pred = velocity_model(I_t, getattr(coupled_batch, "t"), condition)
    except TypeError:
        pred = velocity_model(I_t)
    term = _sum_nonbatch(pred * pred) - 2.0 * _sum_nonbatch(dI_t * pred)
    loss = term.mean() if hasattr(term, "mean") else float(np.mean(term))
    return {"loss": loss, "hat_L_b": loss, "prediction": pred, "I_t": I_t, "dI_t": dI_t}


score_field_objective = velocity_field_objective


def transport_cost(x0: Any, x1: Any) -> float:
    value = (x1 - x0) ** 2
    if _is_torch(value):
        return float(value.mean().detach().cpu())
    return float(np.mean(value))


def compute_loss(prediction: Any, target: Any) -> float:
    value = (prediction - target) ** 2
    if _is_torch(value):
        return float(value.mean().detach().cpu())
    return float(np.mean(value))


def aggregate_loss(values: Any) -> dict[str, float]:
    vals = [float(v) for v in values]
    return {"mean": float(np.mean(vals)) if vals else 0.0, "count": float(len(vals))}
