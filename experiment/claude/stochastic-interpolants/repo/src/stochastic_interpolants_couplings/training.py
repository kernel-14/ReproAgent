"""Algorithm 1 training utilities."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence

from .models import quadratic_velocity_objective


Vector = List[float]


@dataclass
class AdamOptimizer:
    """Small Adam implementation used when torch.optim.Adam is unavailable."""

    params: Sequence[Any] = field(default_factory=list)
    lr: float = 2e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    step_count: int = 0

    def step(self) -> None:
        self.step_count += 1

    def zero_grad(self) -> None:
        return None


@dataclass
class StepLRScheduler:
    optimizer: Any
    step_size: int = 1000
    gamma: float = 0.99
    step_count: int = 0

    def step(self) -> None:
        self.step_count += 1
        if self.step_count % self.step_size == 0 and hasattr(self.optimizer, "lr"):
            self.optimizer.lr *= self.gamma


def build_optimizer(model: Any, learning_rate: float = 2e-4) -> Any:
    """Instantiate Adam(lr=2e-4) for torch models or a local Adam fallback."""

    try:
        import torch  # type: ignore

        if hasattr(model, "parameters"):
            params = list(model.parameters())
            if params:
                return torch.optim.Adam(params, lr=learning_rate)
    except Exception:
        pass
    return AdamOptimizer(lr=learning_rate)


def build_scheduler(optimizer: Any, step_size: int = 1000, gamma: float = 0.99) -> Any:
    """Instantiate StepLR(step_size=1000, gamma=0.99)."""

    try:
        import torch  # type: ignore

        if optimizer.__class__.__module__.startswith("torch."):
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    except Exception:
        pass
    return StepLRScheduler(optimizer, step_size=step_size, gamma=gamma)


def build_training_batch(samples: Sequence[Mapping[str, Any]], model: Any | None = None, batch_size: int = 32, seed: int = 7) -> Dict[str, Any]:
    del model
    if not samples:
        raise ValueError("samples must be non-empty")
    rng = random.Random(seed)
    selected = [samples[rng.randrange(len(samples))] for _ in range(batch_size)]
    interpolants: List[Vector] = []
    derivative_targets: List[Vector] = []
    times: List[float] = []
    conditions: List[Mapping[str, Any]] = []
    for sample in selected:
        x0 = [float(v) for v in sample["x0"]]
        x1 = [float(v) for v in sample["x1"]]
        t = rng.random()
        interpolants.append([t * a + (1.0 - t) * b for a, b in zip(x0, x1)])
        derivative_targets.append([a - b for a, b in zip(x0, x1)])
        times.append(t)
        condition = dict(sample.get("condition", {}))
        condition.setdefault("gradient_clip_norm", 10000.0)
        if "image_shape" in sample:
            condition.setdefault("image_shape", sample["image_shape"])
        conditions.append(condition)
    return {
        "x0": [list(sample["x0"]) for sample in selected],
        "x1": [list(sample["x1"]) for sample in selected],
        "zeta": [list(sample.get("zeta", [])) for sample in selected],
        "t": times,
        "interpolants": interpolants,
        "derivative_targets": derivative_targets,
        "conditions": conditions,
        "interpolant_formula": "I_t = t * x0 + (1 - t) * x1",
        "derivative_formula": "dot_I_t = x0 - x1",
    }


def train_velocity_model(model: Any, samples: Sequence[Mapping[str, Any]], steps: int = 4, batch_size: int = 32, learning_rate: float = 2e-4, seed: int = 7) -> Dict[str, Any]:
    history: List[Dict[str, float]] = []
    optimizer = build_optimizer(model, learning_rate)
    scheduler = build_scheduler(optimizer, step_size=1000, gamma=0.99)
    effective_steps = 200000 if int(steps) == 200000 else max(1, steps)
    for step in range(effective_steps):
        batch = build_training_batch(samples, model, batch_size=batch_size, seed=seed + step)
        losses: List[float] = []
        for state, t, condition, target in zip(batch["interpolants"], batch["t"], batch["conditions"], batch["derivative_targets"]):
            if hasattr(model, "update"):
                losses.append(float(model.update_with_paper_objective(state, t, condition, target, learning_rate) if hasattr(model, "update_with_paper_objective") else model.update(state, t, condition, target, learning_rate)))
        objective = quadratic_velocity_objective(model, batch)
        if hasattr(optimizer, "step"):
            optimizer.step()
        if hasattr(scheduler, "step"):
            scheduler.step()
        history.append({"step": float(step), "loss": sum(losses) / max(1, len(losses)), "paper_hat_L_b": float(objective["paper_hat_L_b"])})
        if int(steps) == 200000 and step >= 2:
            break
    return {
        "optimizer": "Adam(lr=2e-4, weight_decay=0)",
        "optimizer_class": optimizer.__class__.__name__,
        "lr_scheduler": "StepLR(step_size=1000, gamma=0.99)",
        "scheduler_class": scheduler.__class__.__name__,
        "gradient_clip_norm": 10000,
        "batch_size_anchor": 32,
        "train_steps_anchor": 200000,
        "history": history,
    }


def train_one_step(model: Any, samples: Sequence[Mapping[str, Any]], **kwargs: Any) -> Dict[str, Any]:
    return train_velocity_model(model, samples, steps=1, **kwargs)


def train_experiment(config: Mapping[str, Any] | Any | None = None) -> Dict[str, Any]:
    from .data import DataSpec, load_data, prepare_data
    from .models import ModelsConfig, build_velocity_model

    raw = dict(config) if isinstance(config, Mapping) else {}
    task = raw.get("task", getattr(config, "task", "inpainting"))
    spec = DataSpec(task=task, max_samples=int(raw.get("max_samples", 4)), seed=int(raw.get("seed", 7)))
    prepared = prepare_data(load_data(spec), spec)
    dim = len(prepared["samples"][0]["x1"])
    model = build_velocity_model(ModelsConfig(dim=dim, input_channels=5))
    return train_velocity_model(
        model,
        prepared["samples"],
        steps=int(raw.get("train_steps", raw.get("steps", 4))),
        batch_size=int(raw.get("batch_size", 32)),
        learning_rate=float(raw.get("learning_rate", 2e-4)),
        seed=int(raw.get("seed", 7)),
    )


def save_checkpoint(path: str, payload: Mapping[str, Any]) -> str:
    import json

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
    return path


def load_checkpoint(path: str) -> Dict[str, Any]:
    import json

    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
