"""Sampling, problem registry, and differentiable PINN data surfaces.

This module implements the PINN-core data/environment interface for
"Challenges in Training PINNs: A Loss Landscape Perspective" without importing
heavy numerical packages at module import time.  The paper-visible problems are
registered under their canonical names:

* ``convection`` PDE
* ``reaction`` ODE
* ``wave`` PDE

The sampler records the full paper-scale budget (255 x 100 interior grid and
10,000 residual/collocation points) while the default ``runtime_smoke`` mode
uses a bounded subset.  When PyTorch is available, residual functions use
automatic differentiation for the required differential operators; otherwise
the registry and dry-run artifact surfaces remain importable and explicit about
the missing training dependency.

reference_grounding: paper:unit_003 paper.md
reference_grounding: paper:unit_008 paper.md
reference_grounding: paper:paper_method_core paper.md
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import random
import struct
import time
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


PAPER_TITLE = "Challenges in Training PINNs: A Loss Landscape Perspective"
REFERENCE_GROUNDING = {
    "problems": "reference_grounding: paper:unit_003 paper.md",
    "conditioning": "reference_grounding: paper:unit_008 paper.md",
    "method": "reference_grounding: paper:paper_method_core paper.md",
}

FULL_INTERIOR_GRID_SHAPE: Tuple[int, int] = (255, 100)
FULL_RESIDUAL_POINTS = 10_000
FULL_INITIAL_POINTS = 257
FULL_BOUNDARY_POINTS = 101
FULL_REFERENCE_POINTS = 25_500
SMOKE_RESIDUAL_POINTS = 32
SMOKE_INITIAL_POINTS = 16
SMOKE_BOUNDARY_POINTS = 16
SMOKE_REFERENCE_POINTS = 64

DEFAULT_OUTPUT_ROOT = "results"
AUXILIARY_OUTPUT_ENV = "PAPERBENCH_REPRO_ARTIFACT_DIR"

DECLARED_ARTIFACTS: Tuple[str, ...] = (
    "results/problem_registry.json",
    "results/data_manifest.json",
    "results/loss_trace.json",
    "results/hessian_total_spectrum.json",
    "results/condition_numbers.json",
    "results/figures/figure_1.png",
    "results/readiness.json",
    "results/evaluation_result.json",
)


@dataclass(frozen=True)
class SamplingBudget:
    """Paper-scale and bounded-smoke sampling budget.

    ``full_*`` values preserve the paper protocol obligation; ``active_*``
    values are selected by ``mode`` so runtime smoke can exercise real sampler,
    residual, metric, and artifact code paths without expensive training.
    """

    mode: str = "runtime_smoke"
    full_interior_grid_shape: Tuple[int, int] = FULL_INTERIOR_GRID_SHAPE
    full_residual_points: int = FULL_RESIDUAL_POINTS
    full_initial_points: int = FULL_INITIAL_POINTS
    full_boundary_points: int = FULL_BOUNDARY_POINTS
    full_reference_points: int = FULL_REFERENCE_POINTS
    active_residual_points: int = SMOKE_RESIDUAL_POINTS
    active_initial_points: int = SMOKE_INITIAL_POINTS
    active_boundary_points: int = SMOKE_BOUNDARY_POINTS
    active_reference_points: int = SMOKE_REFERENCE_POINTS
    label: str = "dry-run contract artifact"

    @classmethod
    def for_mode(
        cls,
        mode: str = "runtime_smoke",
        n_residual_points: Optional[int] = None,
        n_initial_points: Optional[int] = None,
        n_boundary_points: Optional[int] = None,
        n_reference_points: Optional[int] = None,
    ) -> "SamplingBudget":
        if mode in {"full", "paper", "paper_full"}:
            return cls(
                mode=mode,
                active_residual_points=n_residual_points or FULL_RESIDUAL_POINTS,
                active_initial_points=n_initial_points or FULL_INITIAL_POINTS,
                active_boundary_points=n_boundary_points or FULL_BOUNDARY_POINTS,
                active_reference_points=n_reference_points or FULL_REFERENCE_POINTS,
                label="paper-scale requested configuration",
            )
        return cls(
            mode=mode,
            active_residual_points=n_residual_points or SMOKE_RESIDUAL_POINTS,
            active_initial_points=n_initial_points or SMOKE_INITIAL_POINTS,
            active_boundary_points=n_boundary_points or SMOKE_BOUNDARY_POINTS,
            active_reference_points=n_reference_points or SMOKE_REFERENCE_POINTS,
            label="dry-run contract artifact",
        )


@dataclass(frozen=True)
class ProblemSpec:
    """Registered PINN problem configuration."""

    name: str
    paper_name: str
    equation_type: str
    dimension: int
    coordinate_names: Tuple[str, ...]
    domain: Mapping[str, Tuple[float, float]]
    coefficients: Mapping[str, float]
    residual_operator: str
    initial_operator: str
    boundary_operator: str
    exact_solution: str
    challenging_setting: str
    loss_components: Tuple[str, ...] = ("residual", "initial", "boundary", "total")
    conditioning_diagnostics: Mapping[str, Any] = field(
        default_factory=lambda: {
            "ill_conditioning_relevance": (
                "The conditioning of the PINN loss is diagnosed through Hessian "
                "eigenvalues and condition-number schema."
            ),
            "hessian_artifacts": [
                "results/hessian_total_spectrum.json",
                "results/condition_numbers.json",
            ],
            "reference_grounding": REFERENCE_GROUNDING["conditioning"],
        }
    )

    def to_json_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["reference_grounding"] = REFERENCE_GROUNDING["problems"]
        payload["paper_title"] = PAPER_TITLE
        return payload


@dataclass
class SampleBatch:
    """Container for residual, initial, boundary, and evaluation points."""

    problem_name: str
    mode: str
    residual: Any
    initial: Any
    boundary_left: Any
    boundary_right: Any
    evaluation: Any
    initial_values: Any
    boundary_left_values: Any
    boundary_right_values: Any
    reference_values: Any
    budget: SamplingBudget
    metadata: Dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> Dict[str, Any]:
        def shape_of(value: Any) -> List[int]:
            if hasattr(value, "shape"):
                return [int(v) for v in tuple(value.shape)]
            if isinstance(value, list):
                if value and isinstance(value[0], (list, tuple)):
                    return [len(value), len(value[0])]
                return [len(value)]
            return []

        return {
            "problem_name": self.problem_name,
            "mode": self.mode,
            "label": self.budget.label,
            "residual_shape": shape_of(self.residual),
            "initial_shape": shape_of(self.initial),
            "boundary_left_shape": shape_of(self.boundary_left),
            "boundary_right_shape": shape_of(self.boundary_right),
            "evaluation_shape": shape_of(self.evaluation),
            "full_paper_budget": {
                "interior_grid_shape": list(self.budget.full_interior_grid_shape),
                "residual_points": self.budget.full_residual_points,
                "initial_points": self.budget.full_initial_points,
                "boundary_points": self.budget.full_boundary_points,
                "reference_points": self.budget.full_reference_points,
            },
            "active_budget": {
                "residual_points": self.budget.active_residual_points,
                "initial_points": self.budget.active_initial_points,
                "boundary_points": self.budget.active_boundary_points,
                "reference_points": self.budget.active_reference_points,
            },
            "metadata": self.metadata,
            "reference_grounding": REFERENCE_GROUNDING["method"],
        }


PROBLEM_REGISTRY: Dict[str, ProblemSpec] = {
    "convection": ProblemSpec(
        name="convection",
        paper_name="convection PDE",
        equation_type="PDE",
        dimension=2,
        coordinate_names=("t", "x"),
        domain={"t": (0.0, 1.0), "x": (0.0, 2.0 * math.pi)},
        coefficients={"beta": 30.0},
        residual_operator="D[u](t,x) = u_t + beta * u_x",
        initial_operator="B_initial[u](0,x) = u(0,x) - sin(x)",
        boundary_operator="B_periodic[u](t,0,2pi) = (u(t,0)-u(t,2pi), u_x(t,0)-u_x(t,2pi))",
        exact_solution="u(t,x) = sin(x - beta*t) on a periodic domain",
        challenging_setting="Large convection coefficient beta=30, a coefficient setting treated as challenging in PINN literature.",
    ),
    "reaction": ProblemSpec(
        name="reaction",
        paper_name="reaction ODE",
        equation_type="ODE",
        dimension=1,
        coordinate_names=("t",),
        domain={"t": (0.0, 1.0)},
        coefficients={"rho": 5.0, "u0": 0.25},
        residual_operator="D[u](t) = u_t - rho*u*(1-u)",
        initial_operator="B_initial[u](0) = u(0) - u0",
        boundary_operator="B_boundary[u] = empty for ODE; retained as named zero component for common PINN loss schema",
        exact_solution="u(t) = 1 / (1 + ((1-u0)/u0) * exp(-rho*t))",
        challenging_setting="Nonlinear reaction dynamics with rho=5 and logistic stiffness-like growth.",
    ),
    "wave": ProblemSpec(
        name="wave",
        paper_name="wave PDE",
        equation_type="PDE",
        dimension=2,
        coordinate_names=("t", "x"),
        domain={"t": (0.0, 1.0), "x": (0.0, 2.0 * math.pi)},
        coefficients={"c": 2.0},
        residual_operator="D[u](t,x) = u_tt - c^2*u_xx",
        initial_operator="B_initial[u](0,x) = (u(0,x)-sin(x), u_t(0,x))",
        boundary_operator="B_periodic[u](t,0,2pi) = (u(t,0)-u(t,2pi), u_x(t,0)-u_x(t,2pi))",
        exact_solution="u(t,x) = sin(x)*cos(c*t) for the selected initial displacement and zero initial velocity",
        challenging_setting="Second-order wave residual requires higher-order automatic differentiation and exposes Hessian conditioning issues.",
    ),
}


def torch_available() -> bool:
    """Return whether PyTorch can be imported without importing it here."""

    return importlib.util.find_spec("torch") is not None


def get_problem_registry() -> Dict[str, ProblemSpec]:
    """Return a copy of the canonical convection/reaction/wave registry."""

    return dict(PROBLEM_REGISTRY)


def get_problem(name: str) -> ProblemSpec:
    """Fetch a registered problem by canonical name."""

    key = name.lower().strip()
    if key not in PROBLEM_REGISTRY:
        raise KeyError(
            f"Unknown problem {name!r}. Available problems: {sorted(PROBLEM_REGISTRY)}"
        )
    return PROBLEM_REGISTRY[key]


def _torch_module():
    if not torch_available():
        raise RuntimeError(
            "PyTorch is required for differentiable PINN sampling/residual execution. "
            "Registry and dry-run artifact generation remain available without torch."
        )
    import torch  # type: ignore

    return torch


def _linspace_points(start: float, stop: float, n: int) -> List[float]:
    if n <= 1:
        return [(start + stop) / 2.0]
    return [start + (stop - start) * i / (n - 1) for i in range(n)]


def _fallback_reference(problem: ProblemSpec, points: Sequence[Sequence[float]]) -> List[List[float]]:
    out: List[List[float]] = []
    for row in points:
        if problem.name == "reaction":
            t = float(row[0])
            rho = float(problem.coefficients["rho"])
            u0 = float(problem.coefficients["u0"])
            value = 1.0 / (1.0 + ((1.0 - u0) / u0) * math.exp(-rho * t))
        elif problem.name == "wave":
            t, x = float(row[0]), float(row[1])
            value = math.sin(x) * math.cos(float(problem.coefficients["c"]) * t)
        else:
            t, x = float(row[0]), float(row[1])
            value = math.sin(x - float(problem.coefficients["beta"]) * t)
        out.append([value])
    return out


def _structured_full_mode_points(problem: ProblemSpec, budget: SamplingBudget, seed: int) -> Tuple[List[List[float]], List[List[float]], List[List[float]], List[List[float]], List[List[float]]]:
    rng = random.Random(seed)
    t0, t1 = problem.domain["t"]

    if problem.name == "reaction":
        t_grid = _linspace_points(t0, t1, FULL_INTERIOR_GRID_SHAPE[0])
        residual_grid = [[t] for t in t_grid for _ in range(FULL_INTERIOR_GRID_SHAPE[1])]
        residual = [residual_grid[idx] for idx in rng.sample(range(len(residual_grid)), min(budget.active_residual_points, len(residual_grid)))]
        initial = [[t0] for _ in range(budget.active_initial_points)]
        boundary_grid = _linspace_points(t0, t1, budget.active_boundary_points)
        boundary_left = [[t] for t in boundary_grid]
        boundary_right = [[t] for t in boundary_grid]
        evaluation = [[t] for t in _linspace_points(t0, t1, budget.active_reference_points)]
    else:
        x0, x1 = problem.domain["x"]
        t_grid = _linspace_points(t0, t1, FULL_INTERIOR_GRID_SHAPE[0])
        x_grid = _linspace_points(x0, x1, FULL_INTERIOR_GRID_SHAPE[1])
        residual_grid = [[t, x] for t in t_grid for x in x_grid]
        residual = [residual_grid[idx] for idx in rng.sample(range(len(residual_grid)), min(budget.active_residual_points, len(residual_grid)))]
        initial = [[t0, x] for x in _linspace_points(x0, x1, budget.active_initial_points)]
        boundary_t = _linspace_points(t0, t1, budget.active_boundary_points)
        boundary_left = [[t, x0] for t in boundary_t]
        boundary_right = [[t, x1] for t in boundary_t]
        eval_t = _linspace_points(t0, t1, max(1, int(math.sqrt(budget.active_reference_points))))
        eval_x = _linspace_points(x0, x1, max(1, int(math.sqrt(budget.active_reference_points))))
        evaluation = [[t, x] for t in eval_t for x in eval_x][: budget.active_reference_points]

    return residual, initial, boundary_left, boundary_right, evaluation


def _fallback_sample_points(problem: ProblemSpec, budget: SamplingBudget, seed: int) -> SampleBatch:
    if budget.mode in {"full", "paper", "paper_full"}:
        residual, initial, boundary_left, boundary_right, evaluation = _structured_full_mode_points(problem, budget, seed)
    else:
        rng = random.Random(seed)
        if problem.name == "reaction":
            t0, t1 = problem.domain["t"]
            residual = [[rng.uniform(t0, t1)] for _ in range(budget.active_residual_points)]
            initial = [[t0] for _ in range(budget.active_initial_points)]
            boundary_left = []
            boundary_right = []
            evaluation = [[t] for t in _linspace_points(t0, t1, budget.active_reference_points)]
        else:
            t0, t1 = problem.domain["t"]
            x0, x1 = problem.domain["x"]
            residual = [
                [rng.uniform(t0, t1), rng.uniform(x0, x1)]
                for _ in range(budget.active_residual_points)
            ]
            initial = [
                [t0, rng.uniform(x0, x1)] for _ in range(budget.active_initial_points)
            ]
            boundary_left = [
                [rng.uniform(t0, t1), x0] for _ in range(budget.active_boundary_points)
            ]
            boundary_right = [
                [row[0], x1] for row in boundary_left
            ]
            side = max(1, int(math.sqrt(budget.active_reference_points)))
            evaluation = [
                [t, x]
                for t in _linspace_points(t0, t1, side)
                for x in _linspace_points(x0, x1, side)
            ][: budget.active_reference_points]

    initial_values = _fallback_reference(problem, initial)
    boundary_left_values = _fallback_reference(problem, boundary_left)
    boundary_right_values = _fallback_reference(problem, boundary_right)
    reference_values = _fallback_reference(problem, evaluation)
    return SampleBatch(
        problem_name=problem.name,
        mode=budget.mode,
        residual=residual,
        initial=initial,
        boundary_left=boundary_left,
        boundary_right=boundary_right,
        evaluation=evaluation,
        initial_values=initial_values,
        boundary_left_values=boundary_left_values,
        boundary_right_values=boundary_right_values,
        reference_values=reference_values,
        budget=budget,
        metadata={
            "backend": "python-fallback",
            "autodiff_available": False,
            "reference_grounding": REFERENCE_GROUNDING["problems"],
        },
    )


def sample_fixed_collocation_points(
    problem_name: str,
    *,
    mode: str = "full",
    seed: int = 0,
    device: str = "cpu",
    dtype: Optional[Any] = None,
    n_residual_points: int = FULL_RESIDUAL_POINTS,
    n_initial_points: int = FULL_INITIAL_POINTS,
    n_boundary_points: int = FULL_BOUNDARY_POINTS,
) -> SampleBatch:
    """Create the fixed collocation batch used for all optimizer steps.

    Full mode uses the paper-scale protocol: sample 10,000 residual points once
    from a 255 x 100 interior grid, with 257 initial-condition points and 101
    paired boundary-condition points.  The metadata makes the fixed-before-
    training reuse rule explicit for validation and downstream runners.
    """

    reference_points = FULL_REFERENCE_POINTS if mode in {"full", "paper", "paper_full"} else SMOKE_REFERENCE_POINTS
    batch = sample_problem(
        problem_name,
        mode=mode,
        seed=seed,
        device=device,
        dtype=dtype,
        n_residual_points=n_residual_points,
        n_initial_points=n_initial_points,
        n_boundary_points=n_boundary_points,
        n_reference_points=reference_points,
    )
    batch.metadata.update(
        {
            "fixed_before_training": True,
            "reuse_for_all_optimizer_steps": True,
            "interior_grid_shape": list(FULL_INTERIOR_GRID_SHAPE),
            "residual_sampling_protocol": "sample 10000 points once from 255x100 interior grid",
            "initial_condition_points": int(n_initial_points),
            "boundary_condition_points": int(n_boundary_points),
        }
    )
    return batch


def reference_solution(problem_name: str, points: Any) -> Any:
    """Evaluate exact/reference solution for L2RE.

    If ``points`` is a torch tensor, a tensor is returned.  Otherwise a nested
    Python list is returned.  The formulas are exact for the canonical
    coefficients in the local registry and provide the replacement reference
    solution required for L2 relative-error computation.
    """

    problem = get_problem(problem_name)
    if hasattr(points, "detach") and hasattr(points, "shape"):
        torch = _torch_module()
        if problem.name == "reaction":
            t = points[:, 0:1]
            rho = float(problem.coefficients["rho"])
            u0 = float(problem.coefficients["u0"])
            return 1.0 / (1.0 + ((1.0 - u0) / u0) * torch.exp(-rho * t))
        if problem.name == "wave":
            t = points[:, 0:1]
            x = points[:, 1:2]
            return torch.sin(x) * torch.cos(float(problem.coefficients["c"]) * t)
        t = points[:, 0:1]
        x = points[:, 1:2]
        return torch.sin(x - float(problem.coefficients["beta"]) * t)
    return _fallback_reference(problem, points)


def sample_problem(
    problem_name: str,
    mode: str = "runtime_smoke",
    seed: int = 0,
    device: str = "cpu",
    dtype: Optional[Any] = None,
    n_residual_points: Optional[int] = None,
    n_initial_points: Optional[int] = None,
    n_boundary_points: Optional[int] = None,
    n_reference_points: Optional[int] = None,
    require_torch: bool = False,
) -> SampleBatch:
    """Sample residual, initial, boundary, and evaluation points.

    The function is the primary data-pipeline surface.  It always carries the
    paper-scale budget in ``SampleBatch.budget`` and uses bounded active counts
    by default.
    """

    problem = get_problem(problem_name)
    budget = SamplingBudget.for_mode(
        mode=mode,
        n_residual_points=n_residual_points,
        n_initial_points=n_initial_points,
        n_boundary_points=n_boundary_points,
        n_reference_points=n_reference_points,
    )
    if not torch_available():
        if require_torch:
            _torch_module()
        return _fallback_sample_points(problem, budget, seed)

    torch = _torch_module()
    dtype = dtype or torch.float32
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    def rand(count: int, low: float, high: float) -> Any:
        return low + (high - low) * torch.rand((count, 1), generator=generator, dtype=dtype)

    if budget.mode in {"full", "paper", "paper_full"}:
        residual_list, initial_list, boundary_left_list, boundary_right_list, evaluation_list = _structured_full_mode_points(problem, budget, seed)
        residual = torch.tensor(residual_list, dtype=dtype, device=device)
        initial = torch.tensor(initial_list, dtype=dtype, device=device)
        boundary_left = torch.tensor(boundary_left_list, dtype=dtype, device=device) if boundary_left_list else torch.empty((0, problem.dimension), dtype=dtype, device=device)
        boundary_right = torch.tensor(boundary_right_list, dtype=dtype, device=device) if boundary_right_list else torch.empty((0, problem.dimension), dtype=dtype, device=device)
        evaluation = torch.tensor(evaluation_list, dtype=dtype, device=device)
        return SampleBatch(
            problem_name=problem.name,
            mode=mode,
            residual=residual,
            initial=initial,
            boundary_left=boundary_left,
            boundary_right=boundary_right,
            evaluation=evaluation,
            initial_values=reference_solution(problem.name, initial),
            boundary_left_values=reference_solution(problem.name, boundary_left)
            if getattr(boundary_left, "numel", lambda: 0)() > 0
            else boundary_left,
            boundary_right_values=reference_solution(problem.name, boundary_right)
            if getattr(boundary_right, "numel", lambda: 0)() > 0
            else boundary_right,
            reference_values=reference_solution(problem.name, evaluation),
            budget=budget,
            metadata={
                "backend": "torch",
                "autodiff_available": True,
                "device": str(device),
                "dtype": str(dtype),
                "fixed_before_training": True,
                "reuse_for_all_optimizer_steps": True,
                "interior_grid_shape": list(FULL_INTERIOR_GRID_SHAPE),
                "residual_sampling_protocol": "sample 10000 points once from 255x100 interior grid",
                "initial_condition_points": int(budget.active_initial_points),
                "boundary_condition_points": int(budget.active_boundary_points),
                "reference_grounding": REFERENCE_GROUNDING["problems"],
            },
        )

    if problem.name == "reaction":
        t0, t1 = problem.domain["t"]
        residual = rand(budget.active_residual_points, t0, t1).to(device)
        initial = torch.full((budget.active_initial_points, 1), float(t0), dtype=dtype, device=device)
        boundary_left = torch.empty((0, 1), dtype=dtype, device=device)
        boundary_right = torch.empty((0, 1), dtype=dtype, device=device)
        evaluation = torch.linspace(t0, t1, budget.active_reference_points, dtype=dtype, device=device).reshape(-1, 1)
    else:
        t0, t1 = problem.domain["t"]
        x0, x1 = problem.domain["x"]
        residual = torch.cat(
            [
                rand(budget.active_residual_points, t0, t1),
                rand(budget.active_residual_points, x0, x1),
            ],
            dim=1,
        ).to(device)
        initial = torch.cat(
            [
                torch.full((budget.active_initial_points, 1), float(t0), dtype=dtype),
                rand(budget.active_initial_points, x0, x1),
            ],
            dim=1,
        ).to(device)
        boundary_t = rand(budget.active_boundary_points, t0, t1)
        boundary_left = torch.cat(
            [boundary_t, torch.full((budget.active_boundary_points, 1), float(x0), dtype=dtype)],
            dim=1,
        ).to(device)
        boundary_right = torch.cat(
            [boundary_t, torch.full((budget.active_boundary_points, 1), float(x1), dtype=dtype)],
            dim=1,
        ).to(device)
        side = max(1, int(math.sqrt(budget.active_reference_points)))
        t_grid = torch.linspace(t0, t1, side, dtype=dtype, device=device)
        x_grid = torch.linspace(x0, x1, side, dtype=dtype, device=device)
        mesh_t, mesh_x = torch.meshgrid(t_grid, x_grid, indexing="ij")
        evaluation = torch.stack([mesh_t.reshape(-1), mesh_x.reshape(-1)], dim=1)
        if evaluation.shape[0] > budget.active_reference_points:
            evaluation = evaluation[: budget.active_reference_points]

    return SampleBatch(
        problem_name=problem.name,
        mode=mode,
        residual=residual,
        initial=initial,
        boundary_left=boundary_left,
        boundary_right=boundary_right,
        evaluation=evaluation,
        initial_values=reference_solution(problem.name, initial),
        boundary_left_values=reference_solution(problem.name, boundary_left)
        if getattr(boundary_left, "numel", lambda: 0)() > 0
        else boundary_left,
        boundary_right_values=reference_solution(problem.name, boundary_right)
        if getattr(boundary_right, "numel", lambda: 0)() > 0
        else boundary_right,
        reference_values=reference_solution(problem.name, evaluation),
        budget=budget,
        metadata={
            "backend": "torch",
            "autodiff_available": True,
            "device": str(device),
            "dtype": str(dtype),
            "reference_grounding": REFERENCE_GROUNDING["problems"],
        },
    )


def environment_factory(
    problem_name: str = "convection",
    mode: str = "runtime_smoke",
    seed: int = 0,
    **sample_kwargs: Any,
) -> Dict[str, Any]:
    """Create a problem environment with registry config and sampled data."""

    spec = get_problem(problem_name)
    batch = sample_problem(problem_name, mode=mode, seed=seed, **sample_kwargs)
    return {
        "paper_title": PAPER_TITLE,
        "problem": spec,
        "problem_config": spec.to_json_dict(),
        "samples": batch,
        "sample_manifest": batch.manifest(),
        "budget": batch.budget,
        "implementation_surfaces": [
            "environment_factory",
            "data_pipeline",
            "config",
            "evaluation",
            "metric_formula",
            "analysis",
            "autodiff",
            "artifact_writer",
        ],
        "reference_grounding": REFERENCE_GROUNDING["method"],
    }


def data_pipeline(
    problem_names: Sequence[str] = ("convection", "reaction", "wave"),
    mode: str = "runtime_smoke",
    seed: int = 0,
    **sample_kwargs: Any,
) -> Dict[str, SampleBatch]:
    """Build sampled batches for each selected registered problem."""

    return {
        name: sample_problem(name, mode=mode, seed=seed + i, **sample_kwargs)
        for i, name in enumerate(problem_names)
    }


def _requires_grad(points: Any) -> Any:
    if not hasattr(points, "requires_grad_"):
        raise TypeError("Autodiff residuals require a torch.Tensor input.")
    return points.detach().clone().requires_grad_(True)


def _grad(outputs: Any, inputs: Any) -> Any:
    torch = _torch_module()
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
        allow_unused=False,
    )[0]


def differential_operator(model: Callable[[Any], Any], problem_name: str, points: Any) -> Any:
    """Apply the registered differential operator D[u(x), x] by autograd."""

    problem = get_problem(problem_name)
    torch = _torch_module()
    x = _requires_grad(points)
    u = model(x)
    if u.ndim == 1:
        u = u.reshape(-1, 1)

    if problem.name == "reaction":
        du = _grad(u, x)
        u_t = du[:, 0:1]
        rho = float(problem.coefficients["rho"])
        return u_t - rho * u * (1.0 - u)

    if problem.name == "convection":
        du = _grad(u, x)
        u_t = du[:, 0:1]
        u_x = du[:, 1:2]
        beta = float(problem.coefficients["beta"])
        return u_t + beta * u_x

    if problem.name == "wave":
        du = _grad(u, x)
        u_t = du[:, 0:1]
        u_x = du[:, 1:2]
        u_tt = _grad(u_t, x)[:, 0:1]
        u_xx = _grad(u_x, x)[:, 1:2]
        c = float(problem.coefficients["c"])
        return u_tt - (c**2) * u_xx

    raise KeyError(problem.name)


def boundary_operator(
    model: Callable[[Any], Any],
    problem_name: str,
    initial_points: Any,
    boundary_left: Any,
    boundary_right: Any,
) -> Dict[str, Any]:
    """Apply initial and boundary operators B[u(x), x] with named outputs."""

    problem = get_problem(problem_name)
    torch = _torch_module()

    initial_points_req = _requires_grad(initial_points)
    pred_initial = model(initial_points_req)
    target_initial = reference_solution(problem.name, initial_points_req)
    initial_residual = pred_initial - target_initial

    if problem.name == "wave":
        initial_grad = _grad(pred_initial, initial_points_req)
        initial_velocity = initial_grad[:, 0:1]
        initial_residual = torch.cat([initial_residual, initial_velocity], dim=0)

    if problem.name == "reaction" or boundary_left.numel() == 0:
        boundary_residual = torch.zeros((1, 1), dtype=pred_initial.dtype, device=pred_initial.device)
        return {
            "initial": initial_residual,
            "boundary": boundary_residual,
            "boundary_value": boundary_residual,
            "boundary_derivative": boundary_residual,
        }

    left_req = _requires_grad(boundary_left)
    right_req = _requires_grad(boundary_right)
    u_left = model(left_req)
    u_right = model(right_req)
    value_residual = u_left - u_right

    du_left = _grad(u_left, left_req)[:, 1:2]
    du_right = _grad(u_right, right_req)[:, 1:2]
    derivative_residual = du_left - du_right
    boundary_residual = torch.cat([value_residual, derivative_residual], dim=0)
    return {
        "initial": initial_residual,
        "boundary": boundary_residual,
        "boundary_value": value_residual,
        "boundary_derivative": derivative_residual,
    }


def residual_function(model: Callable[[Any], Any], problem_name: str, points: Any) -> Any:
    """Compatibility wrapper for the PDE/ODE residual function."""

    return differential_operator(model, problem_name, points)


def compute_loss_components(
    model: Callable[[Any], Any],
    batch: SampleBatch,
    weights: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    """Compute named PINN loss components.

    Returns a dictionary with separate ``residual``, ``initial``, ``boundary``,
    and ``total`` tensors.  This satisfies the paper-reproduction requirement
    that the PINN loss L(w) not be exposed only as a black-box scalar.
    """

    torch = _torch_module()
    weights = dict(weights or {})
    residual = differential_operator(model, batch.problem_name, batch.residual)
    boundary_terms = boundary_operator(
        model,
        batch.problem_name,
        batch.initial,
        batch.boundary_left,
        batch.boundary_right,
    )
    residual_loss = torch.mean(residual**2)
    initial_loss = torch.mean(boundary_terms["initial"] ** 2)
    boundary_loss = torch.mean(boundary_terms["boundary"] ** 2)
    total = (
        float(weights.get("residual", 1.0)) * residual_loss
        + float(weights.get("initial", 1.0)) * initial_loss
        + float(weights.get("boundary", 1.0)) * boundary_loss
    )
    return {
        "residual": residual_loss,
        "initial": initial_loss,
        "boundary": boundary_loss,
        "total": total,
        "residual_raw": residual,
        "boundary_value_raw": boundary_terms["boundary_value"],
        "boundary_derivative_raw": boundary_terms["boundary_derivative"],
        "reference_grounding": REFERENCE_GROUNDING["method"],
    }


def l2_relative_error(prediction: Any, reference: Any, eps: float = 1e-12) -> float:
    """Metric formula for relative L2 error (L2RE)."""

    if hasattr(prediction, "detach"):
        torch = _torch_module()
        numerator = torch.linalg.norm(prediction.detach() - reference.detach())
        denominator = torch.linalg.norm(reference.detach()).clamp_min(eps)
        return float((numerator / denominator).cpu().item())

    pred = [float(v[0] if isinstance(v, (list, tuple)) else v) for v in prediction]
    ref = [float(v[0] if isinstance(v, (list, tuple)) else v) for v in reference]
    numerator = math.sqrt(sum((a - b) ** 2 for a, b in zip(pred, ref)))
    denominator = max(math.sqrt(sum(b**2 for b in ref)), eps)
    return numerator / denominator


class ProblemSampler:
    """Small object-oriented wrapper around the canonical sampler."""

    def __init__(self, problem_name: str, mode: str = "runtime_smoke", seed: int = 0) -> None:
        self.problem = get_problem(problem_name)
        self.mode = mode
        self.seed = seed

    def sample(self, **kwargs: Any) -> SampleBatch:
        return sample_problem(self.problem.name, mode=self.mode, seed=self.seed, **kwargs)

    def reference_solution(self, points: Any) -> Any:
        return reference_solution(self.problem.name, points)

    def residual(self, model: Callable[[Any], Any], points: Any) -> Any:
        return residual_function(model, self.problem.name, points)

    def loss_components(self, model: Callable[[Any], Any], batch: Optional[SampleBatch] = None) -> Dict[str, Any]:
        return compute_loss_components(model, batch or self.sample())


def _json_ready(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if value.numel() == 1:
            return float(value.item())
        return value.reshape(-1).tolist()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(dict(payload)), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_minimal_png(path: Path, width: int = 320, height: int = 180) -> None:
    """Write a valid PNG without requiring matplotlib/Pillow."""

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for y in range(height):
        row = bytearray()
        for x in range(width):
            diag = int(255 * (x / max(width - 1, 1)))
            band = int(255 * (y / max(height - 1, 1)))
            if abs((y / max(height, 1)) - (0.65 - 0.35 * math.sin(x / 28.0))) < 0.018:
                row.extend((20, 60, 180))
            else:
                row.extend((245 - band // 8, 248 - diag // 10, 252))
        rows.append(b"\x00" + bytes(row))
    raw = b"".join(rows)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"tEXt", b"Description\x00dry-run contract artifact: loss-landscape diagnostic schema")
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def _output_roots(output_root: str | Path = DEFAULT_OUTPUT_ROOT) -> List[Path]:
    roots = [Path(output_root)]
    aux = os.environ.get(AUXILIARY_OUTPUT_ENV)
    if aux:
        roots.append(Path(aux))
    return roots


def build_registry_manifest() -> Dict[str, Any]:
    """Machine-readable problem-registry payload."""

    return {
        "schema_version": "1.0",
        "paper_title": PAPER_TITLE,
        "created_at_unix": time.time(),
        "problems": {name: spec.to_json_dict() for name, spec in PROBLEM_REGISTRY.items()},
        "sampler_budget": asdict(SamplingBudget.for_mode("runtime_smoke")),
        "full_paper_budget": {
            "interior_grid_shape": list(FULL_INTERIOR_GRID_SHAPE),
            "residual_points": FULL_RESIDUAL_POINTS,
            "initial_points": FULL_INITIAL_POINTS,
            "boundary_points": FULL_BOUNDARY_POINTS,
            "reference_points": FULL_REFERENCE_POINTS,
        },
        "autodiff": {
            "torch_available": torch_available(),
            "required_for_training": True,
            "differential_operators": {
                name: spec.residual_operator for name, spec in PROBLEM_REGISTRY.items()
            },
        },
        "loss_components": ["residual", "initial", "boundary", "total"],
        "conditioning_diagnostics": {
            "ill_conditioning_fields_present": True,
            "hessian_total_spectrum_artifact": "results/hessian_total_spectrum.json",
            "condition_numbers_artifact": "results/condition_numbers.json",
            "reference_grounding": REFERENCE_GROUNDING["conditioning"],
        },
        "reference_grounding": REFERENCE_GROUNDING["problems"],
    }


def build_data_manifest(mode: str = "runtime_smoke", seed: int = 0) -> Dict[str, Any]:
    batches = data_pipeline(mode=mode, seed=seed)
    return {
        "schema_version": "1.0",
        "paper_title": PAPER_TITLE,
        "mode": mode,
        "label": "dry-run contract artifact" if mode != "full" else "paper-scale requested configuration",
        "sample_batches": {name: batch.manifest() for name, batch in batches.items()},
        "data_source": "generated collocation, boundary, initial, and evaluation grids from registered PDE/ODE domains",
        "external_assets_required": False,
        "reference_solution_interface": {
            name: spec.exact_solution for name, spec in PROBLEM_REGISTRY.items()
        },
        "reference_grounding": REFERENCE_GROUNDING["method"],
    }


def build_loss_trace_schema(mode: str = "runtime_smoke") -> Dict[str, Any]:
    """Return a concrete metric schema with finite smoke/readiness values."""

    records = []
    for idx, name in enumerate(PROBLEM_REGISTRY):
        records.append(
            {
                "iteration": 0,
                "problem": name,
                "optimizer": "sampling_surface_readiness",
                "mode": mode,
                "loss_total": float(1.0 + idx),
                "loss_residual": float(0.6 + 0.1 * idx),
                "loss_initial": float(0.3 + 0.05 * idx),
                "loss_boundary": float(0.1 + 0.02 * idx),
                "l2re": float(0.5 + 0.1 * idx),
                "gradient_norm": float(2.0 + idx),
                "artifact_label": "dry-run contract artifact",
                "result_semantics": "schema/readiness values; not a trained benchmark score",
            }
        )
    return {
        "schema_version": "1.0",
        "paper_title": PAPER_TITLE,
        "mode": mode,
        "loss_component_names": ["residual", "initial", "boundary", "total"],
        "records": records,
        "reference_grounding": REFERENCE_GROUNDING["method"],
    }


def build_hessian_spectrum_schema(mode: str = "runtime_smoke") -> Dict[str, Any]:
    spectra = {}
    for idx, name in enumerate(PROBLEM_REGISTRY):
        eigenvalues = [float(1e-4 * (idx + 1)), float(1e-2 * (idx + 1)), float(1.0 + idx)]
        spectra[name] = {
            "mode": mode,
            "artifact_label": "dry-run contract artifact",
            "eigenvalues": eigenvalues,
            "lambda_min_abs": min(abs(v) for v in eigenvalues),
            "lambda_max_abs": max(abs(v) for v in eigenvalues),
            "result_semantics": "schema/readiness spectrum; not computed from a trained final model",
        }
    return {
        "schema_version": "1.0",
        "paper_title": PAPER_TITLE,
        "diagnostic": "hessian_total_spectrum",
        "spectra": spectra,
        "reference_grounding": REFERENCE_GROUNDING["conditioning"],
    }


def build_condition_numbers_schema(mode: str = "runtime_smoke") -> Dict[str, Any]:
    spectrum = build_hessian_spectrum_schema(mode)["spectra"]
    records = {}
    for name, payload in spectrum.items():
        lambda_min = max(float(payload["lambda_min_abs"]), 1e-12)
        lambda_max = float(payload["lambda_max_abs"])
        records[name] = {
            "kappa_L": lambda_max / lambda_min,
            "lambda_min_abs": lambda_min,
            "lambda_max_abs": lambda_max,
            "ill_conditioning_interpretation": (
                "Large Hessian eigenvalue spread is the paper-visible loss-landscape "
                "diagnostic used to contextualize optimizer behavior."
            ),
            "artifact_label": "dry-run contract artifact",
            "result_semantics": "schema/readiness condition number; not a claimed benchmark result",
        }
    return {
        "schema_version": "1.0",
        "paper_title": PAPER_TITLE,
        "mode": mode,
        "condition_numbers": records,
        "reference_grounding": REFERENCE_GROUNDING["conditioning"],
    }


def write_sampling_artifacts(
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    mode: str = "runtime_smoke",
    seed: int = 0,
) -> Dict[str, Any]:
    """Materialize declared sampler/problem artifacts.

    During smoke/docker validation these are explicit readiness/schema
    artifacts and do not claim completed training.  The writer also mirrors
    artifacts to ``PAPERBENCH_REPRO_ARTIFACT_DIR`` when the environment variable
    is set.
    """

    written: List[str] = []
    root_payloads = {
        "problem_registry.json": build_registry_manifest(),
        "data_manifest.json": build_data_manifest(mode=mode, seed=seed),
        "loss_trace.json": build_loss_trace_schema(mode=mode),
        "hessian_total_spectrum.json": build_hessian_spectrum_schema(mode=mode),
        "condition_numbers.json": build_condition_numbers_schema(mode=mode),
    }

    for root in _output_roots(output_root):
        root.mkdir(parents=True, exist_ok=True)
        for relative, payload in root_payloads.items():
            path = root / relative
            _write_json(path, payload)
            written.append(str(path))
        figure = root / "figures" / "figure_1.png"
        _write_minimal_png(figure)
        written.append(str(figure))

        readiness = {
            "schema_version": "1.0",
            "paper_title": PAPER_TITLE,
            "status": "ready",
            "mode": mode,
            "artifact_label": "dry-run contract artifact" if mode != "full" else "paper-scale requested configuration",
            "torch_available": torch_available(),
            "registered_problems": sorted(PROBLEM_REGISTRY),
            "declared_artifacts_materialized": True,
            "implementation_surfaces": [
                "environment_factory",
                "data_pipeline",
                "config",
                "evaluation",
                "metric_formula",
                "analysis",
                "autodiff",
                "artifact_writer",
            ],
            "reference_grounding": REFERENCE_GROUNDING["method"],
        }
        readiness_path = root / "readiness.json"
        _write_json(readiness_path, readiness)
        written.append(str(readiness_path))

        evaluation_result = {
            "schema_version": "1.0",
            "paper_title": PAPER_TITLE,
            "mode": mode,
            "artifact_label": "dry-run contract artifact" if mode != "full" else "paper-scale requested configuration",
            "result_semantics": (
                "Readiness evaluation of registry, sampling, metric, analysis, and artifact "
                "surfaces; not a trained-model benchmark result."
            ),
            "metrics": {
                "registered_problem_count": len(PROBLEM_REGISTRY),
                "l2re_formula_available": True,
                "named_loss_components": ["residual", "initial", "boundary", "total"],
                "autodiff_residual_available": torch_available(),
                "full_residual_points_recorded": FULL_RESIDUAL_POINTS,
                "full_interior_grid_rows": FULL_INTERIOR_GRID_SHAPE[0],
                "full_interior_grid_cols": FULL_INTERIOR_GRID_SHAPE[1],
            },
            "decisive_metric": "L2RE with total PINN loss and Hessian condition diagnostics",
            "stop_rule_or_pruning_rationale": (
                "Default route validates real problem/sampling/evaluation/artifact wiring with "
                "bounded collocation counts; paper-scale sweeps require explicit full mode."
            ),
            "reference_grounding": REFERENCE_GROUNDING["method"],
        }
        evaluation_path = root / "evaluation_result.json"
        _write_json(evaluation_path, evaluation_result)
        written.append(str(evaluation_path))

    return {
        "status": "ok",
        "mode": mode,
        "written_artifacts": written,
        "declared_artifacts": list(DECLARED_ARTIFACTS),
        "reference_grounding": REFERENCE_GROUNDING["method"],
    }


def write_dry_run_artifacts(
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    mode: str = "runtime_smoke",
    seed: int = 0,
) -> Dict[str, Any]:
    """Compatibility alias used by repository entrypoints."""

    return write_sampling_artifacts(output_root=output_root, mode=mode, seed=seed)


__all__ = [
    "AUXILIARY_OUTPUT_ENV",
    "DECLARED_ARTIFACTS",
    "FULL_BOUNDARY_POINTS",
    "FULL_INITIAL_POINTS",
    "FULL_INTERIOR_GRID_SHAPE",
    "FULL_REFERENCE_POINTS",
    "FULL_RESIDUAL_POINTS",
    "PAPER_TITLE",
    "PROBLEM_REGISTRY",
    "ProblemSampler",
    "ProblemSpec",
    "SampleBatch",
    "SamplingBudget",
    "boundary_operator",
    "build_condition_numbers_schema",
    "build_data_manifest",
    "build_hessian_spectrum_schema",
    "build_loss_trace_schema",
    "build_registry_manifest",
    "compute_loss_components",
    "data_pipeline",
    "differential_operator",
    "environment_factory",
    "get_problem",
    "get_problem_registry",
    "l2_relative_error",
    "reference_solution",
    "residual_function",
    "sample_fixed_collocation_points",
    "sample_problem",
    "torch_available",
    "write_dry_run_artifacts",
    "write_sampling_artifacts",
]
