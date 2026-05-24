"""Configuration helpers for the PINN reproduction package."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from pinn_landscape.models import ModelConfig as PinnModelConfig
from pinn_landscape.models import get_problem_registry, get_sweep_registry
from src.method_registry import expand_experiment_registry


DEFAULT_PROBLEMS: Tuple[str, ...] = ("convection", "reaction", "wave")
DEFAULT_WIDTHS: Tuple[int, ...] = (50, 100, 200, 400)
DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)
DEFAULT_FULL_ITERATIONS = 41_000
DEFAULT_SMOKE_ITERATIONS = 3
DEFAULT_RESIDUAL_POINTS = 10_000
DEFAULT_INTERIOR_GRID = (255, 100)
DEFAULT_ADAM_LR_GRID: Tuple[float, ...] = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
DEFAULT_LBFGS_LR = 1.0
DEFAULT_LBFGS_MEMORY_SIZE = 100
DEFAULT_LBFGS_LINE_SEARCH = "strong_wolfe"
DEFAULT_SWITCH_ITERATIONS: Tuple[int, ...] = (1_000, 11_000, 31_000)


@dataclass(frozen=True)
class OptimizerConfig:
    name: str
    learning_rate: float = 1e-3
    history_size: int = DEFAULT_LBFGS_MEMORY_SIZE
    line_search: str = DEFAULT_LBFGS_LINE_SEARCH
    switch_iteration: int = 11_000
    second_order_rank: int = 16
    max_steps: int = DEFAULT_FULL_ITERATIONS
    mode: str = "runtime_smoke"


@dataclass(frozen=True)
class ExperimentConfig:
    problem: str = "convection"
    optimizer: str = "Adam"
    width: int = 200
    seed: int = 0
    max_steps: int = DEFAULT_SMOKE_ITERATIONS
    dry_run: bool = True
    output_dir: str = "results"
    mode: str = "runtime_smoke"
    n_residual_points: int = 32
    interior_grid: Tuple[int, int] = DEFAULT_INTERIOR_GRID
    model: PinnModelConfig = field(default_factory=PinnModelConfig)
    optimizer_config: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(name="Adam"))


def default_config(mode: str = "runtime_smoke") -> Dict[str, Any]:
    smoke = mode in {"runtime_smoke", "docker_validate", "dry_run", "smoke"}
    return {
        "problem": "convection",
        "optimizer": "Adam",
        "width": 200,
        "seed": 0,
        "max_steps": DEFAULT_SMOKE_ITERATIONS if smoke else DEFAULT_FULL_ITERATIONS,
        "dry_run": smoke,
        "output_dir": "results",
        "mode": mode,
        "n_residual_points": 32 if smoke else DEFAULT_RESIDUAL_POINTS,
        "interior_grid": (8, 4) if smoke else DEFAULT_INTERIOR_GRID,
        "model": asdict(PinnModelConfig(width=200, seed=0)),
        "optimizer_config": asdict(OptimizerConfig(name="Adam", mode=mode)),
    }


def _load_raw_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            return dict(data or {})
        except Exception:
            return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_config(config: Mapping[str, Any] | None = None, mode: str = "runtime_smoke") -> Dict[str, Any]:
    payload = default_config(mode)
    if config:
        payload.update(dict(config))
    payload["mode"] = mode
    payload["dry_run"] = bool(payload.get("dry_run", True) or mode in {"runtime_smoke", "docker_validate", "dry_run", "smoke"})
    return payload


def load_config(path: str | Path | None = None, mode: str = "runtime_smoke") -> Dict[str, Any]:
    if path is None:
        return default_config(mode)
    raw = _load_raw_config(Path(path))
    return resolve_config(raw, mode=raw.get("mode", mode) if isinstance(raw, Mapping) else mode)


def expand_experiment_matrix(
    problems: Sequence[str] = DEFAULT_PROBLEMS,
    optimizers: Sequence[str] = ("Adam", "L-BFGS", "Adam+L-BFGS", "NysNewton-CG"),
    widths: Sequence[int] = DEFAULT_WIDTHS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    mode: str = "runtime_smoke",
    max_experiments: Optional[int] = None,
) -> List[ExperimentConfig]:
    rows = expand_experiment_registry(
        problems=problems,
        optimizers=optimizers,
        widths=widths,
        seeds=seeds,
        mode=mode,
        max_experiments=max_experiments,
    )
    configs: List[ExperimentConfig] = []
    for row in rows:
        configs.append(
            ExperimentConfig(
                problem=row.problem,
                optimizer=row.optimizer,
                width=row.width,
                seed=row.seed,
                max_steps=row.iteration_count,
                dry_run=mode in {"runtime_smoke", "docker_validate", "dry_run", "smoke"},
                output_dir="results",
                mode=mode,
                n_residual_points=32 if mode in {"runtime_smoke", "docker_validate", "dry_run", "smoke"} else DEFAULT_RESIDUAL_POINTS,
                interior_grid=(8, 4) if mode in {"runtime_smoke", "docker_validate", "dry_run", "smoke"} else DEFAULT_INTERIOR_GRID,
            )
        )
    return configs


def validate_config(config: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    if config.get("problem") not in get_problem_registry():
        errors.append(f"unknown problem: {config.get('problem')!r}")
    if int(config.get("width", 0)) <= 0:
        errors.append("width must be positive")
    if int(config.get("seed", 0)) < 0:
        errors.append("seed must be non-negative")
    return errors


def problem_defaults() -> Dict[str, Dict[str, Any]]:
    return get_problem_registry()


def sweep_defaults() -> Dict[str, Dict[str, Any]]:
    return get_sweep_registry()


__all__ = [
    "OptimizerConfig",
    "ExperimentConfig",
    "default_config",
    "load_config",
    "resolve_config",
    "expand_experiment_matrix",
    "validate_config",
    "problem_defaults",
    "sweep_defaults",
    "DEFAULT_PROBLEMS",
    "DEFAULT_WIDTHS",
    "DEFAULT_SEEDS",
    "DEFAULT_FULL_ITERATIONS",
    "DEFAULT_SMOKE_ITERATIONS",
]
