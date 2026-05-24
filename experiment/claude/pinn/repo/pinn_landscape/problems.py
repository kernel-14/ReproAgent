"""Problem registry and sampling wrappers for the PINN reproduction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

from pinn_landscape import sampling


@dataclass
class Problem:
    name: str
    mode: str = "runtime_smoke"
    seed: int = 0

    @property
    def spec(self) -> sampling.ProblemSpec:
        return sampling.get_problem(self.name)

    def sample_train_batch(self, **kwargs: Any) -> sampling.SampleBatch:
        return sampling.sample_problem(self.name, mode=self.mode, seed=self.seed, **kwargs)

    def sample_eval_grid(self, **kwargs: Any) -> sampling.SampleBatch:
        kwargs.setdefault("n_reference_points", 64)
        return sampling.sample_problem(self.name, mode=self.mode, seed=self.seed, **kwargs)

    def reference_solution(self, points: Any) -> Any:
        return sampling.reference_solution(self.name, points)

    def loss_components(self, model: Any, batch: Optional[sampling.SampleBatch] = None) -> Dict[str, Any]:
        return sampling.compute_loss_components(model, batch or self.sample_train_batch())


def problem_registry() -> Dict[str, sampling.ProblemSpec]:
    return sampling.get_problem_registry()


def make_problem(name: str, config: Mapping[str, Any] | None = None) -> Problem:
    cfg = dict(config or {})
    return Problem(
        name=name,
        mode=str(cfg.get("mode", "runtime_smoke")),
        seed=int(cfg.get("seed", 0)),
    )


def sample_train_batch(name: str, config: Mapping[str, Any] | None = None, **kwargs: Any) -> sampling.SampleBatch:
    cfg = dict(config or {})
    cfg.update(kwargs)
    return sampling.sample_problem(
        name,
        mode=str(cfg.get("mode", "runtime_smoke")),
        seed=int(cfg.get("seed", 0)),
        n_residual_points=cfg.get("n_residual_points"),
        n_initial_points=cfg.get("n_initial_points"),
        n_boundary_points=cfg.get("n_boundary_points"),
        n_reference_points=cfg.get("n_reference_points"),
    )


def sample_eval_grid(name: str, config: Mapping[str, Any] | None = None, **kwargs: Any) -> sampling.SampleBatch:
    cfg = dict(config or {})
    cfg.update(kwargs)
    cfg.setdefault("n_reference_points", 64)
    return sampling.sample_problem(
        name,
        mode=str(cfg.get("mode", "runtime_smoke")),
        seed=int(cfg.get("seed", 0)),
        n_residual_points=cfg.get("n_residual_points"),
        n_initial_points=cfg.get("n_initial_points"),
        n_boundary_points=cfg.get("n_boundary_points"),
        n_reference_points=cfg.get("n_reference_points"),
    )


def reference_solution(name: str, points: Any) -> Any:
    return sampling.reference_solution(name, points)


def build_problem_manifest(mode: str = "runtime_smoke", seed: int = 0) -> Dict[str, Any]:
    batches = {name: sampling.sample_problem(name, mode=mode, seed=seed) for name in problem_registry()}
    return {
        "mode": mode,
        "seed": seed,
        "problems": {name: batch.manifest() for name, batch in batches.items()},
    }


__all__ = [
    "Problem",
    "problem_registry",
    "make_problem",
    "sample_train_batch",
    "sample_eval_grid",
    "reference_solution",
    "build_problem_manifest",
]
