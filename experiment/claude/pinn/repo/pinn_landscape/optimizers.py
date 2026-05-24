"""Optimizer registry wrappers for the PINN reproduction package."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping

from pinn_landscape.models import normalize_optimizer_name
from src.method_registry import available_optimizers, optimizer as registry_optimizer


@dataclass(frozen=True)
class OptimizerConfig:
    name: str
    learning_rate: float = 1e-3
    history_size: int = 100
    line_search: str = "strong_wolfe"
    switch_iteration: int = 11_000
    max_steps: int = 41_000
    mode: str = "runtime_smoke"
    second_order_rank: int = 16


ADAM_LR_GRID = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
ADAM_LBFGS_SWITCH_ITERATIONS = (1_000, 11_000, 31_000)
FULL_TRAINING_ITERATIONS = 41_000


def optimizer_registry() -> Dict[str, Dict[str, Any]]:
    return available_optimizers()


def build_optimizer_config(name: str, mode: str = "runtime_smoke", **overrides: Any) -> OptimizerConfig:
    canonical = normalize_optimizer_name(name)
    base = OptimizerConfig(name=canonical, mode=mode)
    payload = asdict(base)
    payload.update(overrides)
    return OptimizerConfig(**payload)


def build_optimizer(name: str) -> Any:
    return registry_optimizer(name)


def adam_config(mode: str = "runtime_smoke", learning_rate: float = 1e-3) -> OptimizerConfig:
    return OptimizerConfig(name="Adam", learning_rate=learning_rate, mode=mode)


def lbfgs_config(mode: str = "runtime_smoke", learning_rate: float = 1.0) -> OptimizerConfig:
    return OptimizerConfig(name="L-BFGS", learning_rate=learning_rate, history_size=100, line_search="strong_wolfe", mode=mode)


def adam_lbfgs_config(mode: str = "runtime_smoke", learning_rate: float = 1e-3, switch_iteration: int = 11_000) -> OptimizerConfig:
    return OptimizerConfig(name="Adam+L-BFGS", learning_rate=learning_rate, history_size=100, line_search="strong_wolfe", switch_iteration=switch_iteration, mode=mode)


def nncg_config(mode: str = "runtime_smoke", rank: int = 16) -> OptimizerConfig:
    return OptimizerConfig(name="NysNewton-CG", mode=mode, second_order_rank=rank)


def build_adam_lbfgs_schedule(
    switch_iteration: int,
    *,
    adam_lrs: Any = ADAM_LR_GRID,
    adam_lr: float | None = None,
    lbfgs_lr: float = 1.0,
    history_size: int = 100,
    line_search: str = "strong_wolfe",
    total_iterations: int = FULL_TRAINING_ITERATIONS,
    smoke_steps: int | None = None,
) -> Dict[str, Any]:
    """Return an executable Adam-to-L-BFGS protocol description."""

    lr = float(adam_lr if adam_lr is not None else list(adam_lrs)[2])
    active_total = int(smoke_steps) if smoke_steps is not None else int(total_iterations)
    active_switch = min(int(switch_iteration), max(1, active_total - 1)) if active_total > 1 else 1
    return {
        "name": "Adam+L-BFGS",
        "total_iterations": int(total_iterations),
        "active_iterations": active_total,
        "switch_iteration": int(switch_iteration),
        "active_switch_iteration": active_switch,
        "adam_learning_rate": lr,
        "adam_lr_grid": [float(v) for v in adam_lrs],
        "lbfgs_lr": float(lbfgs_lr),
        "history_size": int(history_size),
        "line_search_fn": line_search,
        "line_search": line_search,
        "phases": [
            {"optimizer": "Adam", "start_iteration": 0, "end_iteration": int(switch_iteration), "learning_rate": lr},
            {
                "optimizer": "L-BFGS",
                "start_iteration": int(switch_iteration),
                "end_iteration": int(total_iterations),
                "learning_rate": float(lbfgs_lr),
                "history_size": int(history_size),
                "line_search_fn": line_search,
            },
        ],
    }


def adam_lbfgs_1k_config(**kwargs: Any) -> Dict[str, Any]:
    return build_adam_lbfgs_schedule(1_000, **kwargs)


def adam_lbfgs_11k_config(**kwargs: Any) -> Dict[str, Any]:
    return build_adam_lbfgs_schedule(11_000, **kwargs)


def adam_lbfgs_31k_config(**kwargs: Any) -> Dict[str, Any]:
    return build_adam_lbfgs_schedule(31_000, **kwargs)


__all__ = [
    "ADAM_LBFGS_SWITCH_ITERATIONS",
    "ADAM_LR_GRID",
    "FULL_TRAINING_ITERATIONS",
    "OptimizerConfig",
    "adam_lbfgs_1k_config",
    "adam_lbfgs_11k_config",
    "adam_lbfgs_31k_config",
    "optimizer_registry",
    "build_optimizer_config",
    "build_optimizer",
    "build_adam_lbfgs_schedule",
    "adam_config",
    "lbfgs_config",
    "adam_lbfgs_config",
    "nncg_config",
]
