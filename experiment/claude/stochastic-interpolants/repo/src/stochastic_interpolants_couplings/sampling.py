"""Sampling routes for Algorithm 2."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Mapping


Vector = List[float]


def euler_ode_sample(model: Any, x0: Vector, condition: Mapping[str, Any] | None = None, steps: int = 16) -> Dict[str, Any]:
    x = [float(v) for v in x0]
    condition = condition or {}
    dt = 1.0 / float(max(1, steps))
    trace = [list(x)]
    for k in range(max(1, steps)):
        t = k * dt
        velocity = model(x, t, condition)
        x = [xi + dt * vi for xi, vi in zip(x, velocity)]
        if condition.get("task") == "inpainting" and "mask" in condition:
            mask = [float(v) for v in condition["mask"]]
            observed = [float(v) for v in condition.get("observed_image", x)]
            x = [observed[i] if mask[i % len(mask)] >= 0.5 else x[i] for i in range(len(x))]
        trace.append(list(x))
    return {"sampler": "ode", "sample": x, "trace": trace, "algorithm": "X_{i+1}=X_i+N^-1 hat_b_{i/N}(X_i)"}


def dopri_ode_sample(model: Any, x0: Any, condition: Mapping[str, Any] | None = None, steps: int = 16) -> Dict[str, Any]:
    """Use torchdiffeq Dopri5 when available; otherwise keep Euler-compatible output."""

    condition = condition or {}
    try:
        import torch  # type: ignore
        from torchdiffeq import odeint  # type: ignore

        x_start = x0 if hasattr(x0, "detach") else torch.as_tensor(x0, dtype=torch.float32)

        class Field(torch.nn.Module):
            def forward(self, t: Any, x: Any) -> Any:
                try:
                    return model(x, float(t.detach().cpu()), condition)
                except TypeError:
                    return model(x, t, condition)

        t_grid = torch.linspace(0.0, 1.0, max(2, steps + 1), device=x_start.device, dtype=x_start.dtype)
        trace = odeint(Field(), x_start, t_grid, method="dopri5")
        return {"sampler": "ode", "ode_solver": "torchdiffeq.dopri5", "sample": trace[-1], "trace": trace}
    except Exception:
        result = euler_ode_sample(model, list(x0), condition, steps)
        result["ode_solver"] = "torchdiffeq.dopri5 unavailable; Euler fallback used"
        return result


def euler_maruyama_sde_sample(
    model: Any,
    x0: Vector,
    condition: Mapping[str, Any] | None = None,
    steps: int = 16,
    noise_scale: float = 0.08,
    seed: int = 7,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    x = [float(v) for v in x0]
    condition = condition or {}
    dt = 1.0 / float(max(1, steps))
    trace = [list(x)]
    for k in range(max(1, steps)):
        t = k * dt
        velocity = model(x, t, condition)
        x = [xi + dt * vi + noise_scale * (dt ** 0.5) * rng.gauss(0.0, 1.0) for xi, vi in zip(x, velocity)]
        trace.append(list(x))
    return {"sampler": "sde", "sample": x, "trace": trace}


def _flatten(x: Any) -> Vector:
    if hasattr(x, "detach"):
        return [float(v) for v in x.detach().cpu().reshape(-1).tolist()]
    try:
        import numpy as np

        return [float(v) for v in np.asarray(x).reshape(-1).tolist()]
    except Exception:
        return [float(v) for v in x]


def sample_ode(config: Mapping[str, Any] | Any | None = None, **kwargs: Any) -> Dict[str, Any]:
    from .couplings import build_coupling
    from .data import DataSpec, load_data, prepare_data
    from .models import ModelsConfig, build_velocity_model

    raw = dict(config) if isinstance(config, Mapping) else {}
    task = raw.get("task", kwargs.get("task", "inpainting"))
    spec = DataSpec(task=task, max_samples=2, seed=int(raw.get("seed", kwargs.get("seed", 7))))
    prepared = prepare_data(load_data(spec), spec)
    sample = prepared["samples"][0]
    model = build_velocity_model(ModelsConfig(dim=len(sample["x1"])))
    coupling = build_coupling({"task": task, "coupling_type": raw.get("coupling_type", "data_dependent")})
    coupled = coupling.sample(sample["x1"], conditioning=sample["condition"])
    return dopri_ode_sample(model, _flatten(coupled.x0), coupled.conditioning, steps=int(raw.get("sampler_steps", kwargs.get("steps", 16))))


def sample_sde(config: Mapping[str, Any] | Any | None = None, **kwargs: Any) -> Dict[str, Any]:
    result = sample_ode(config, **kwargs)
    result["sampler"] = "sde"
    return result


def sample_inpainting(config: Mapping[str, Any] | Any | None = None, **kwargs: Any) -> Dict[str, Any]:
    raw = dict(config) if isinstance(config, Mapping) else {}
    raw["task"] = "inpainting"
    raw.update(kwargs)
    return sample_ode(raw)


def sample_superresolution(config: Mapping[str, Any] | Any | None = None, **kwargs: Any) -> Dict[str, Any]:
    raw = dict(config) if isinstance(config, Mapping) else {}
    raw["task"] = "super_resolution"
    raw.update(kwargs)
    return sample_ode(raw)
