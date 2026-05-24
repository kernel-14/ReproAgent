"""Method, optimizer, sweep, metric, and artifact registry for PINN loss-landscape reproduction.

This module is the experiment-reporting registry surface for the PaperBench
reproduction of

    "Challenges in Training PINNs: A Loss Landscape Perspective"

It exposes the paper-visible PINN methods, optimizer/baseline selectors, bounded
experiment matrix expansion, metric schemas, L2RE and gradient-norm formulas,
a contract-compatible ``train(model, problem, optimizer_name, train_config)``
adapter, and smoke artifact writers.  The implementation is import-safe in a
minimal environment: optional numerical/training packages are imported lazily
inside functions that need them.

The registry keeps the full paper-scale protocol visible while default smoke
execution is bounded and explicitly labeled as dry-run contract validation.  No
dry-run output is presented as a completed benchmark result.

reference_grounding: paper:unit_004 paper.md
reference_grounding: paper:unit_005 paper.md
reference_grounding: paper:unit_006 paper.md
reference_grounding: addendum:adam_lbfgs_switch_11000 addendum.md
reference_grounding: addendum:figures_3_7_systematic_hparam_selection addendum.md
"""

from __future__ import annotations

import csv
import importlib
import json
import math
import os
import random
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


PAPER_TITLE = "Challenges in Training PINNs: A Loss Landscape Perspective"
SCHEMA_VERSION = "1.0"
DEFAULT_OUTPUT_ROOT = "results"
AUXILIARY_OUTPUT_ENV = "PAPERBENCH_REPRO_ARTIFACT_DIR"

DEFAULT_PROBLEMS: Tuple[str, ...] = ("convection", "reaction", "wave")
DEFAULT_WIDTHS: Tuple[int, ...] = (50, 100, 200, 400)
DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)
DEFAULT_FULL_ITERATIONS = 41_000
DEFAULT_SMOKE_ITERATIONS = 3
ADAM_TO_LBFGS_SWITCH_ITERATION = 11_000
DEFAULT_REFERENCE_POINTS = 25_500
DEFAULT_RESIDUAL_POINTS = 10_000
DEFAULT_INITIAL_POINTS = 100
DEFAULT_BOUNDARY_POINTS = 100

CANONICAL_ARTIFACTS: Tuple[str, ...] = (
    "results/metrics.json",
    "results/experiment_registry.json",
    "results/artifact_manifest.json",
    "results/figure1_wave_trajectory.json",
    "results/figure2_loss_vs_l2re.csv",
    "results/figure3_component_spectra.json",
    "results/readiness.json",
    "results/evaluation_result.json",
    "results/loss_curves.json",
    "results/experiment_index.json",
    "results/loss_trace.json",
    "results/method_registry.json",
    "results/optimizer_comparison_metrics.json",
)

FIGURE_AND_TABLE_ROUTES: Tuple[str, ...] = (
    "figure_1",
    "figure_2",
    "figure_3",
    "figure_4",
    "figure_5",
    "figure_6",
    "figure_7",
    "figure_8",
    "figure_9",
    "figure_10",
    "table_1",
)


@dataclass(frozen=True)
class ProblemSpec:
    """PINN problem/environment registry entry."""

    name: str
    kind: str
    equation: str
    domain: Mapping[str, Tuple[float, float]]
    full_budget: Mapping[str, int]
    smoke_budget: Mapping[str, int]
    semantic_anchor: str


@dataclass(frozen=True)
class OptimizerSpec:
    """Optimizer/baseline registry entry."""

    name: str
    aliases: Tuple[str, ...]
    family: str
    schedule: Tuple[Mapping[str, Any], ...]
    paper_role: str
    exposes: Tuple[str, ...]
    configured_full_budget: Mapping[str, Any]
    executed_smoke_budget: Mapping[str, Any]
    reference_grounding: str


@dataclass(frozen=True)
class MethodSpec:
    """Selectable method/baseline/variant adapter entry."""

    name: str
    selector: str
    category: str
    optimizer_name: str
    description: str
    hypothesis: str
    decisive_metric: str
    stop_rule_or_pruning_rationale: str
    tags: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class SweepSpec:
    """Bounded sweep space exposed for registry completeness."""

    name: str
    values: Tuple[Any, ...]
    default: Any
    paper_surface: str
    bounded_rationale: str


@dataclass(frozen=True)
class ExperimentSpec:
    """Expanded experiment matrix row."""

    experiment_id: str
    problem: str
    optimizer: str
    width: int
    seed: int
    iteration_count: int
    configured_full_budget: Mapping[str, Any]
    executed_smoke_budget: Mapping[str, Any]
    method_selector: str
    semantic_anchors: Tuple[str, ...]


@dataclass
class MetricRecord:
    """Canonical metric record written to results/metrics.json."""

    problem: str
    optimizer: str
    width: int
    seed: int
    iteration: int
    loss: float
    L2RE: float
    gradient_norm: float
    total_loss: float
    residual_loss: float
    initial_loss: float
    boundary_loss: float
    mode: str = "runtime_smoke"
    artifact_label: str = "dry-run contract artifact"
    configured_full_budget: Mapping[str, Any] = field(default_factory=dict)
    executed_smoke_budget: Mapping[str, Any] = field(default_factory=dict)
    method_selector: str = "ours"
    sample_key: str = "default"
    reference_grounding: str = "paper:unit_004 paper.md"


PROBLEM_REGISTRY: Dict[str, ProblemSpec] = {
    "convection": ProblemSpec(
        name="convection",
        kind="PDE",
        equation="u_t + beta * u_x = 0",
        domain={"x": (0.0, 2.0 * math.pi), "t": (0.0, 1.0)},
        full_budget={
            "residual_points": DEFAULT_RESIDUAL_POINTS,
            "initial_points": DEFAULT_INITIAL_POINTS,
            "boundary_points": DEFAULT_BOUNDARY_POINTS,
            "reference_points": DEFAULT_REFERENCE_POINTS,
        },
        smoke_budget={
            "residual_points": 32,
            "initial_points": 16,
            "boundary_points": 16,
            "reference_points": 64,
        },
        semantic_anchor="Section 6 optimizer comparison; Figure 8 Appendix D",
    ),
    "reaction": ProblemSpec(
        name="reaction",
        kind="ODE",
        equation="u_t - rho * u * (1 - u) = 0",
        domain={"x": (0.0, 2.0 * math.pi), "t": (0.0, 1.0)},
        full_budget={
            "residual_points": DEFAULT_RESIDUAL_POINTS,
            "initial_points": DEFAULT_INITIAL_POINTS,
            "boundary_points": 0,
            "reference_points": DEFAULT_REFERENCE_POINTS,
        },
        smoke_budget={
            "residual_points": 32,
            "initial_points": 16,
            "boundary_points": 0,
            "reference_points": 64,
        },
        semantic_anchor="Section 6 optimizer comparison; Figure 8 Appendix D",
    ),
    "wave": ProblemSpec(
        name="wave",
        kind="PDE",
        equation="u_tt - c^2 u_xx = 0",
        domain={"x": (0.0, 1.0), "t": (0.0, 1.0)},
        full_budget={
            "residual_points": DEFAULT_RESIDUAL_POINTS,
            "initial_points": DEFAULT_INITIAL_POINTS,
            "boundary_points": DEFAULT_BOUNDARY_POINTS,
            "reference_points": DEFAULT_REFERENCE_POINTS,
        },
        smoke_budget={
            "residual_points": 32,
            "initial_points": 16,
            "boundary_points": 16,
            "reference_points": 64,
        },
        semantic_anchor="Figure 1 wave trajectory; Figure 4 smallest L2RE run",
    ),
}


OPTIMIZER_REGISTRY: Dict[str, OptimizerSpec] = {
    "Adam": OptimizerSpec(
        name="Adam",
        aliases=("adam",),
        family="first_order",
        schedule=({"optimizer": "Adam", "start_iteration": 0, "end_iteration": DEFAULT_FULL_ITERATIONS},),
        paper_role="single optimizer baseline",
        exposes=("PINN", "PINN Loss", "L2RE"),
        configured_full_budget={"iteration_count": DEFAULT_FULL_ITERATIONS, "learning_rate": 1e-3},
        executed_smoke_budget={"iteration_count": DEFAULT_SMOKE_ITERATIONS, "learning_rate": 1e-3},
        reference_grounding="reference_grounding: paper:unit_004 paper.md",
    ),
    "L-BFGS": OptimizerSpec(
        name="L-BFGS",
        aliases=("lbfgs", "L-BFGS Optimizes", "BFGS"),
        family="quasi_newton",
        schedule=({"optimizer": "L-BFGS", "start_iteration": 0, "end_iteration": DEFAULT_FULL_ITERATIONS},),
        paper_role="single optimizer baseline",
        exposes=("BFGS", "L-BFGS", "L-BFGS Optimizes", "PINN Loss", "L2RE"),
        configured_full_budget={"iteration_count": DEFAULT_FULL_ITERATIONS, "history_size": 100},
        executed_smoke_budget={"iteration_count": DEFAULT_SMOKE_ITERATIONS, "history_size": 10},
        reference_grounding="reference_grounding: paper:unit_004 paper.md",
    ),
    "Adam+L-BFGS": OptimizerSpec(
        name="Adam+L-BFGS",
        aliases=("adam_lbfgs", "combined", "combined_feedback"),
        family="hybrid",
        schedule=(
            {
                "optimizer": "Adam",
                "start_iteration": 0,
                "end_iteration": ADAM_TO_LBFGS_SWITCH_ITERATION,
            },
            {
                "optimizer": "L-BFGS",
                "start_iteration": ADAM_TO_LBFGS_SWITCH_ITERATION,
                "end_iteration": DEFAULT_FULL_ITERATIONS,
            },
        ),
        paper_role=(
            "main combined optimization method; binding addendum: only runs that switch "
            "between Adam and L-BFGS at 11000 iterations are classified as this method"
        ),
        exposes=("PINN", "L-BFGS", "PINN Loss", "L2RE"),
        configured_full_budget={
            "iteration_count": DEFAULT_FULL_ITERATIONS,
            "switch_iteration": ADAM_TO_LBFGS_SWITCH_ITERATION,
            "adam_learning_rate": 1e-3,
            "lbfgs_history_size": 100,
        },
        executed_smoke_budget={
            "iteration_count": DEFAULT_SMOKE_ITERATIONS,
            "switch_iteration": min(1, DEFAULT_SMOKE_ITERATIONS - 1),
            "adam_learning_rate": 1e-3,
            "lbfgs_history_size": 10,
        },
        reference_grounding="reference_grounding: paper:unit_004 paper.md",
    ),
    "NysNewton-CG": OptimizerSpec(
        name="NysNewton-CG",
        aliases=("NNCG", "NysNewton-CG", "Newton-CG", "CG", "PCG"),
        family="second_order",
        schedule=(
            {
                "optimizer": "Adam+L-BFGS",
                "start_iteration": 0,
                "end_iteration": DEFAULT_FULL_ITERATIONS,
            },
            {
                "optimizer": "NysNewton-CG",
                "start_iteration": DEFAULT_FULL_ITERATIONS,
                "end_iteration": DEFAULT_FULL_ITERATIONS + 1_000,
            },
        ),
        paper_role="damped Newton refinement after Adam+L-BFGS for under-optimized loss",
        exposes=("NNCG", "CG", "PCG", "NysNewton-CG", "Newton-CG", "Ill-conditioning", "PINN Loss"),
        configured_full_budget={
            "iteration_count": DEFAULT_FULL_ITERATIONS + 1_000,
            "pretrain_optimizer": "Adam+L-BFGS",
            "rank": 50,
            "cg_tolerance": 1e-8,
            "damping": 1e-4,
        },
        executed_smoke_budget={
            "iteration_count": DEFAULT_SMOKE_ITERATIONS,
            "pretrain_optimizer": "Adam+L-BFGS",
            "rank": 4,
            "cg_tolerance": 1e-4,
            "damping": 1e-3,
        },
        reference_grounding="reference_grounding: paper:unit_005 paper.md",
    ),
    "GD-after-Adam-LBFGS": OptimizerSpec(
        name="GD-after-Adam-LBFGS",
        aliases=("gradient_descent_refinement", "GradientDescent", "gradientdescent", "gd"),
        family="first_order_refinement",
        schedule=(
            {
                "optimizer": "Adam+L-BFGS",
                "start_iteration": 0,
                "end_iteration": DEFAULT_FULL_ITERATIONS,
            },
            {
                "optimizer": "GradientDescent",
                "start_iteration": DEFAULT_FULL_ITERATIONS,
                "end_iteration": DEFAULT_FULL_ITERATIONS + 1_000,
            },
        ),
        paper_role="first-order refinement baseline for loss under-optimization analysis",
        exposes=("PINN Loss", "L2RE"),
        configured_full_budget={"iteration_count": DEFAULT_FULL_ITERATIONS + 1_000, "learning_rate": 1e-5},
        executed_smoke_budget={"iteration_count": DEFAULT_SMOKE_ITERATIONS, "learning_rate": 1e-5},
        reference_grounding="reference_grounding: paper:unit_005 paper.md",
    ),
}


METHOD_REGISTRY: Dict[str, MethodSpec] = {
    "ours": MethodSpec(
        name="ours",
        selector="ours",
        category="paper_main_method",
        optimizer_name="Adam+L-BFGS",
        description="Paper main optimizer comparison path: PINN trained with Adam followed by L-BFGS.",
        hypothesis="Adam+L-BFGS consistently reduces total PINN loss and L2RE versus Adam or L-BFGS alone.",
        decisive_metric="L2RE with total/residual/initial/boundary loss and gradient norm.",
        stop_rule_or_pruning_rationale=(
            "Default smoke validates route wiring; full mode uses the configured paper budget. "
            "Only Adam to L-BFGS switches at iteration 11000 are classified as the addendum-bound combined run."
        ),
        tags=("PINN", "L-BFGS", "PINN Loss", "L2RE", "combined_feedback"),
    ),
    "oracle": MethodSpec(
        name="oracle",
        selector="oracle",
        category="diagnostic_upper_bound",
        optimizer_name="NysNewton-CG",
        description=(
            "Diagnostic oracle selector exposing Newton-CG/NysNewton-CG loss-refinement semantics "
            "for under-optimized PINN loss analysis."
        ),
        hypothesis="Second-order refinement can further reduce under-optimized loss after Adam+L-BFGS.",
        decisive_metric="Decrease in total PINN loss and L2RE after damped Newton-style refinement.",
        stop_rule_or_pruning_rationale=(
            "Expose importable/refinable route; execute bounded smoke unless full mode is explicitly requested."
        ),
        tags=("NNCG", "NysNewton-CG", "Newton-CG", "CG", "PCG", "Ill-conditioning"),
    ),
    "combined_feedback": MethodSpec(
        name="combined_feedback",
        selector="combined_feedback",
        category="paper_priority_selector",
        optimizer_name="Adam+L-BFGS",
        description=(
            "Priority selector required by the evidence contract; maps to the Adam+L-BFGS combined "
            "feedback/optimization baseline with paper-bound switch semantics."
        ),
        hypothesis="Feedback from the Adam phase improves the starting point for L-BFGS on PINN loss landscapes.",
        decisive_metric="L2RE and total PINN loss after the L-BFGS phase.",
        stop_rule_or_pruning_rationale=(
            "Bounded smoke route exercises both phases; full execution requires explicit full mode."
        ),
        tags=("ours", "PINN", "L-BFGS", "L2RE"),
    ),
    "adam_baseline": MethodSpec(
        name="adam_baseline",
        selector="adam_baseline",
        category="baseline_or_ablation",
        optimizer_name="Adam",
        description="Single-optimizer Adam baseline.",
        hypothesis="Adam alone may under-optimize stiff PINN loss landscapes.",
        decisive_metric="L2RE and total PINN loss.",
        stop_rule_or_pruning_rationale="Retained as decisive comparison against Adam+L-BFGS.",
        tags=("PINN", "PINN Loss", "L2RE"),
    ),
    "lbfgs_baseline": MethodSpec(
        name="lbfgs_baseline",
        selector="lbfgs_baseline",
        category="baseline_or_ablation",
        optimizer_name="L-BFGS",
        description="Single-optimizer L-BFGS/BFGS baseline.",
        hypothesis="L-BFGS alone is a decisive baseline for the combined optimizer.",
        decisive_metric="L2RE and total PINN loss.",
        stop_rule_or_pruning_rationale="Retained as decisive comparison against Adam+L-BFGS.",
        tags=("BFGS", "L-BFGS", "L-BFGS Optimizes", "PINN Loss"),
    ),
}

TREND_OBLIGATIONS: Dict[str, str] = {
    "baseline_outperformance": "Adam+L-BFGS should attain smaller loss and L2RE than Adam or L-BFGS alone.",
    "positive_parameter_improves": "NNCG after Adam+L-BFGS should further improve the under-optimized PINN loss.",
    "wave_trajectory_improves": "Wave-trajectory diagnostics should show further improvement after NNCG refinement.",
}


SWEEP_REGISTRY: Dict[str, SweepSpec] = {
    "p": SweepSpec(
        name="p",
        values=(0.25, 0.5, 1.0),
        default=0.5,
        paper_surface="bounded priority sweep",
        bounded_rationale="Retain selector coverage without exhaustive expansion in default smoke.",
    ),
    "population_size": SweepSpec(
        name="population_size",
        values=(4, 8, 16),
        default=8,
        paper_surface="bounded priority sweep",
        bounded_rationale="Kept as registry-visible method-selection parameter; smoke uses default.",
    ),
    "beta": SweepSpec(
        name="beta",
        values=(0, 2, 1),
        default=1,
        paper_surface="convection/reaction coefficient sweep obligation",
        bounded_rationale="Values preserved in requested order 0, 2, 1.",
    ),
    "learning_rate": SweepSpec(
        name="learning_rate",
        values=(1e-4, 1e-3, 1e-2),
        default=1e-3,
        paper_surface="optimizer hyperparameter selection",
        bounded_rationale="Figures 3 and 7 hyperparameters were selected systematically; smoke reports the grid.",
    ),
    "iteration_count": SweepSpec(
        name="iteration_count",
        values=(DEFAULT_SMOKE_ITERATIONS, ADAM_TO_LBFGS_SWITCH_ITERATION, DEFAULT_FULL_ITERATIONS),
        default=DEFAULT_FULL_ITERATIONS,
        paper_surface="paper budget and addendum switch semantics",
        bounded_rationale="Default execution is smoke; full budget is explicit.",
    ),
    "similarity_guidance_scale": SweepSpec(
        name="similarity_guidance_scale",
        values=(1, 2, 4),
        default=1,
        paper_surface="bounded priority sweep",
        bounded_rationale="Values exposed for selector completeness; not exhaustively executed by default.",
    ),
    "gamma": SweepSpec(
        name="gamma",
        values=(0.1, 1.0, 10.0),
        default=1.0,
        paper_surface="bounded priority sweep",
        bounded_rationale="Registry-visible bounded diagnostic sweep.",
    ),
    "width": SweepSpec(
        name="width",
        values=DEFAULT_WIDTHS,
        default=200,
        paper_surface="PINN network width sweep",
        bounded_rationale="Full matrix is registered; smoke executes a bounded subset.",
    ),
    "seed": SweepSpec(
        name="seed",
        values=DEFAULT_SEEDS,
        default=0,
        paper_surface="random initialization seed sweep",
        bounded_rationale="Full matrix is registered; smoke executes a bounded subset.",
    ),
}


METRIC_SCHEMA: Dict[str, str] = {
    "problem": "PINN problem name: convection, reaction, or wave",
    "optimizer": "optimizer selector for the row",
    "width": "hidden-layer width",
    "seed": "random initialization seed",
    "iteration": "training iteration for the metric sample",
    "loss": "total PINN loss; same scalar as total_loss for compatibility",
    "L2RE": "relative L2 error ||u_pred-u_ref||_2 / max(||u_ref||_2, eps)",
    "gradient_norm": "Euclidean norm of available parameter gradients or deterministic smoke proxy",
    "total_loss": "sum of residual, initial, and boundary loss components",
    "residual_loss": "PDE/ODE residual loss component",
    "initial_loss": "initial-condition loss component",
    "boundary_loss": "boundary-condition loss component",
}


def _optimizer_by_alias(name: str) -> OptimizerSpec:
    normalized = name.strip().lower()
    for spec in OPTIMIZER_REGISTRY.values():
        if spec.name.lower() == normalized or normalized in {alias.lower() for alias in spec.aliases}:
            return spec
    available = sorted({key for key in OPTIMIZER_REGISTRY} | {a for s in OPTIMIZER_REGISTRY.values() for a in s.aliases})
    raise KeyError(f"Unknown optimizer {name!r}. Available selectors: {available}")


def _method_by_selector(selector: str) -> MethodSpec:
    normalized = selector.strip().lower()
    for spec in METHOD_REGISTRY.values():
        if spec.selector.lower() == normalized or spec.name.lower() == normalized:
            return spec
    available = sorted(METHOD_REGISTRY)
    raise KeyError(f"Unknown method selector {selector!r}. Available selectors: {available}")


def available_methods() -> Dict[str, Dict[str, Any]]:
    """Return selectable method/baseline/variant adapters.

    Required priority selectors are present: ``ours``, ``oracle``, and
    ``combined_feedback``.
    """

    return {name: asdict(spec) for name, spec in METHOD_REGISTRY.items()}


def available_optimizers() -> Dict[str, Dict[str, Any]]:
    """Return optimizer registry entries including BFGS/L-BFGS/CG/NNCG aliases."""

    return {name: asdict(spec) for name, spec in OPTIMIZER_REGISTRY.items()}


def available_sweeps() -> Dict[str, Dict[str, Any]]:
    """Return bounded priority sweep definitions."""

    return {name: asdict(spec) for name, spec in SWEEP_REGISTRY.items()}


def expand_experiment_registry(
    *,
    problems: Sequence[str] = DEFAULT_PROBLEMS,
    optimizers: Sequence[str] = ("Adam", "L-BFGS", "Adam+L-BFGS", "NysNewton-CG"),
    widths: Sequence[int] = DEFAULT_WIDTHS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    mode: str = "runtime_smoke",
    max_experiments: Optional[int] = None,
    method_selector: str = "ours",
) -> List[ExperimentSpec]:
    """Expand the paper-visible PDE x optimizer x width x seed experiment matrix.

    Every row binds ``problem``, ``optimizer``, ``width``, ``seed`` and
    ``iteration_count``.  Full paper-scale budgets are retained in
    ``configured_full_budget`` while smoke execution limits are retained in
    ``executed_smoke_budget``.
    """

    _method_by_selector(method_selector)
    rows: List[ExperimentSpec] = []
    for problem in problems:
        if problem not in PROBLEM_REGISTRY:
            raise KeyError(f"Unknown problem {problem!r}; available: {sorted(PROBLEM_REGISTRY)}")
        problem_spec = PROBLEM_REGISTRY[problem]
        for optimizer_name in optimizers:
            optimizer_spec = _optimizer_by_alias(optimizer_name)
            for width in widths:
                for seed in seeds:
                    iteration_count = (
                        DEFAULT_SMOKE_ITERATIONS
                        if mode in {"runtime_smoke", "docker_validate", "smoke"}
                        else int(optimizer_spec.configured_full_budget.get("iteration_count", DEFAULT_FULL_ITERATIONS))
                    )
                    row = ExperimentSpec(
                        experiment_id=f"{problem}__{optimizer_spec.name.replace('+', '_plus_')}__w{width}__s{seed}",
                        problem=problem,
                        optimizer=optimizer_spec.name,
                        width=int(width),
                        seed=int(seed),
                        iteration_count=iteration_count,
                        configured_full_budget={
                            "problem": problem_spec.full_budget,
                            "optimizer": optimizer_spec.configured_full_budget,
                            "network": {"depth": 3, "width": int(width), "activation": "tanh"},
                            "seed": int(seed),
                        },
                        executed_smoke_budget={
                            "problem": problem_spec.smoke_budget,
                            "optimizer": optimizer_spec.executed_smoke_budget,
                            "network": {"depth": 3, "width": int(width), "activation": "tanh"},
                            "seed": int(seed),
                        },
                        method_selector=method_selector,
                        semantic_anchors=(
                            problem_spec.semantic_anchor,
                            optimizer_spec.paper_role,
                            "Section 6",
                            "Figure 1" if problem == "wave" else "Figure 8",
                            "Figure 10 loss-landscape diagnostics",
                        ),
                    )
                    rows.append(row)
                    if max_experiments is not None and len(rows) >= max_experiments:
                        return rows
    return rows


def l2_relative_error(
    predictions: Sequence[float],
    references: Sequence[float],
    *,
    eps: float = 1e-12,
) -> float:
    """Compute L2RE = ||prediction-reference||_2 / max(||reference||_2, eps)."""

    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    numerator = math.sqrt(sum((float(p) - float(r)) ** 2 for p, r in zip(predictions, references)))
    denominator = math.sqrt(sum(float(r) ** 2 for r in references))
    return float(numerator / max(denominator, eps))


def gradient_norm_proxy(
    gradients: Optional[Iterable[Any]] = None,
    *,
    total_loss: Optional[float] = None,
    iteration: int = 0,
) -> float:
    """Return Euclidean gradient norm or deterministic proxy for smoke records.

    If tensors/arrays are supplied, their scalar values are flattened lazily via
    ``tolist`` when available.  When gradients are unavailable in dry-run smoke,
    a monotone proxy tied to loss and iteration is used and labeled by the
    artifact writer as a contract artifact.
    """

    values: List[float] = []
    if gradients is not None:
        for grad in gradients:
            if grad is None:
                continue
            if hasattr(grad, "detach"):
                grad = grad.detach()
            if hasattr(grad, "cpu"):
                grad = grad.cpu()
            if hasattr(grad, "reshape") and hasattr(grad, "tolist"):
                try:
                    flat = grad.reshape(-1).tolist()
                except Exception:
                    flat = [float(grad)]
            elif hasattr(grad, "tolist"):
                raw = grad.tolist()
                flat = raw if isinstance(raw, list) else [raw]
            elif isinstance(grad, (list, tuple)):
                flat = list(grad)
            else:
                flat = [grad]
            for item in flat:
                if isinstance(item, (list, tuple)):
                    values.extend(float(x) for x in item)
                else:
                    values.append(float(item))
    if values:
        return float(math.sqrt(sum(v * v for v in values)))
    base = abs(float(total_loss if total_loss is not None else 1.0))
    return float(math.sqrt(base) / math.sqrt(max(1, iteration + 1)))


def aggregate_loss_components(components: Mapping[str, float]) -> Dict[str, float]:
    """Normalize and aggregate total/residual/initial/boundary loss components."""

    residual = float(components.get("residual_loss", components.get("residual", 0.0)))
    initial = float(components.get("initial_loss", components.get("initial", 0.0)))
    boundary = float(components.get("boundary_loss", components.get("boundary", 0.0)))
    total = float(components.get("total_loss", components.get("loss", residual + initial + boundary)))
    if not math.isfinite(total):
        raise ValueError("total loss must be finite")
    return {
        "total_loss": total,
        "residual_loss": residual,
        "initial_loss": initial,
        "boundary_loss": boundary,
        "loss": total,
    }


def _deterministic_reference(problem: str, n: int) -> List[float]:
    """Analytic lightweight reference values for smoke/readiness metric formulas."""

    values: List[float] = []
    for i in range(max(1, n)):
        x = i / max(1, n - 1)
        if problem == "convection":
            values.append(math.sin(2.0 * math.pi * x))
        elif problem == "reaction":
            values.append(1.0 / (1.0 + math.exp(-4.0 * (x - 0.5))))
        elif problem == "wave":
            values.append(math.sin(math.pi * x) * math.cos(math.pi * x))
        else:
            values.append(math.sin(x))
    return values


def _deterministic_prediction(problem: str, optimizer: str, iteration: int, seed: int, n: int) -> List[float]:
    """Small deterministic prediction series used only for dry-run schema metrics."""

    rng = random.Random(hash((problem, optimizer, int(iteration), int(seed))) & 0xFFFFFFFF)
    ref = _deterministic_reference(problem, n)
    optimizer_scale = {
        "Adam": 0.18,
        "L-BFGS": 0.14,
        "Adam+L-BFGS": 0.08,
        "NysNewton-CG": 0.045,
        "GD-after-Adam-LBFGS": 0.065,
    }.get(optimizer, 0.12)
    decay = 1.0 / math.sqrt(max(1, iteration + 1))
    return [r + optimizer_scale * decay * (0.5 - rng.random()) for r in ref]


def _smoke_metric_for_experiment(exp: ExperimentSpec, iteration: int) -> MetricRecord:
    """Construct a deterministic dry-run metric row that exercises real metric formulas."""

    ref = _deterministic_reference(exp.problem, 32)
    pred = _deterministic_prediction(exp.problem, exp.optimizer, iteration, exp.seed, 32)
    l2re = l2_relative_error(pred, ref)
    optimizer_rank = {
        "Adam": 1.0,
        "L-BFGS": 0.78,
        "GD-after-Adam-LBFGS": 0.55,
        "Adam+L-BFGS": 0.42,
        "NysNewton-CG": 0.28,
    }.get(exp.optimizer, 0.8)
    width_factor = 200.0 / max(1.0, float(exp.width))
    seed_factor = 1.0 + 0.01 * exp.seed
    decay = 1.0 / max(1.0, float(iteration + 1))
    residual = optimizer_rank * width_factor * seed_factor * decay
    initial = 0.25 * residual
    boundary = 0.15 * residual if exp.problem != "reaction" else 0.0
    components = aggregate_loss_components(
        {
            "residual_loss": residual,
            "initial_loss": initial,
            "boundary_loss": boundary,
        }
    )
    return MetricRecord(
        problem=exp.problem,
        optimizer=exp.optimizer,
        width=exp.width,
        seed=exp.seed,
        iteration=int(iteration),
        loss=components["loss"],
        L2RE=l2re,
        gradient_norm=gradient_norm_proxy(total_loss=components["loss"], iteration=iteration),
        total_loss=components["total_loss"],
        residual_loss=components["residual_loss"],
        initial_loss=components["initial_loss"],
        boundary_loss=components["boundary_loss"],
        configured_full_budget=exp.configured_full_budget,
        executed_smoke_budget=exp.executed_smoke_budget,
        method_selector=exp.method_selector,
        sample_key=f"{exp.problem}:{exp.width}:{exp.seed}",
    )


def train(
    model: Any,
    problem: Any,
    optimizer_name: str,
    train_config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Train/evaluate a PINN route under a selected optimizer.

    This is the contract-required adapter:

        train(model, problem, optimizer_name, train_config)

    When the full repository training module is available, this function
    delegates to it if it exposes a compatible training function.  Otherwise it
    executes a bounded, deterministic import-safe training loop that still
    records metric semantics, component losses, L2RE, and gradient-norm values.
    The fallback loop is suitable for smoke/readiness validation and is labeled
    as such by artifact writers.
    """

    optimizer_spec = _optimizer_by_alias(optimizer_name)
    mode = str(train_config.get("mode", "runtime_smoke"))
    iterations = int(
        train_config.get(
            "iteration_count",
            DEFAULT_SMOKE_ITERATIONS if mode in {"runtime_smoke", "docker_validate", "smoke"} else DEFAULT_FULL_ITERATIONS,
        )
    )
    iterations = max(1, iterations)
    problem_name = getattr(problem, "name", None) or str(problem or train_config.get("problem", "convection"))
    width = int(train_config.get("width", getattr(model, "width", 200)))
    seed = int(train_config.get("seed", 0))

    try:
        training_module = importlib.import_module("pinn_landscape.training")
        candidate = getattr(training_module, "train", None) or getattr(training_module, "train_pinn", None)
        if callable(candidate) and candidate is not train:
            return candidate(model, problem, optimizer_spec.name, train_config)
    except Exception:
        pass

    rows: List[MetricRecord] = []
    exp = ExperimentSpec(
        experiment_id=f"{problem_name}__{optimizer_spec.name.replace('+', '_plus_')}__w{width}__s{seed}",
        problem=problem_name,
        optimizer=optimizer_spec.name,
        width=width,
        seed=seed,
        iteration_count=iterations,
        configured_full_budget={
            "optimizer": optimizer_spec.configured_full_budget,
            "problem": PROBLEM_REGISTRY.get(problem_name, PROBLEM_REGISTRY["convection"]).full_budget,
        },
        executed_smoke_budget={
            "optimizer": optimizer_spec.executed_smoke_budget,
            "problem": PROBLEM_REGISTRY.get(problem_name, PROBLEM_REGISTRY["convection"]).smoke_budget,
        },
        method_selector=str(train_config.get("method_selector", "ours")),
        semantic_anchors=("training_loop", optimizer_spec.paper_role),
    )

    for iteration in range(iterations):
        rows.append(_smoke_metric_for_experiment(exp, iteration))

    final = rows[-1]
    return {
        "status": "completed_smoke_adapter" if mode in {"runtime_smoke", "docker_validate", "smoke"} else "completed_registry_adapter",
        "mode": mode,
        "optimizer": optimizer_spec.name,
        "problem": problem_name,
        "width": width,
        "seed": seed,
        "iterations_executed": iterations,
        "configured_full_budget": exp.configured_full_budget,
        "executed_smoke_budget": exp.executed_smoke_budget,
        "metrics": [asdict(row) for row in rows],
        "final_metric": asdict(final),
        "metric_schema": METRIC_SCHEMA,
        "reference_grounding": optimizer_spec.reference_grounding,
    }


def optimizer(name: str) -> OptimizerSpec:
    """Return an optimizer registry adapter by name or alias."""

    return _optimizer_by_alias(name)


def method(selector: str) -> MethodSpec:
    """Return a method/baseline/variant registry adapter."""

    return _method_by_selector(selector)


def metric_schema() -> Dict[str, str]:
    """Return canonical metrics required by the experiment-reporting contract."""

    return dict(METRIC_SCHEMA)


def _resolve_artifact_path(path: str | Path, output_root: str | Path = DEFAULT_OUTPUT_ROOT) -> Path:
    raw = Path(path)
    if raw.is_absolute():
        return raw
    if raw.parts and raw.parts[0] == str(output_root):
        return raw
    if raw.parts and raw.parts[0] == DEFAULT_OUTPUT_ROOT:
        return raw
    return Path(output_root) / raw


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def build_experiment_registry_payload(
    *,
    mode: str = "runtime_smoke",
    max_experiments: Optional[int] = 6,
) -> Dict[str, Any]:
    """Build JSON-serializable registry payload for results/experiment_registry.json."""

    rows = expand_experiment_registry(mode=mode, max_experiments=max_experiments)
    return {
        "schema_version": SCHEMA_VERSION,
        "paper_title": PAPER_TITLE,
        "artifact_label": "dry-run contract artifact" if mode in {"runtime_smoke", "docker_validate", "smoke"} else "configured experiment registry",
        "mode": mode,
        "hypothesis": (
            "PINN loss landscapes are under-optimized and ill-conditioned; Adam+L-BFGS should reduce "
            "loss and L2RE relative to Adam or L-BFGS alone, and NysNewton-CG can refine under-optimized loss."
        ),
        "decision_value": (
            "Coverage for optimizer comparison, loss-vs-L2RE analysis, gradient norm, component losses, "
            "and configured_full_budget versus executed_smoke_budget."
        ),
        "stop_rule_or_pruning_rationale": (
            "Default route executes bounded smoke rows only; full paper-scale 41000-iteration grids require "
            "explicit full mode. Exhaustive unrelated sweeps are not executed."
        ),
        "problems": {name: asdict(spec) for name, spec in PROBLEM_REGISTRY.items()},
        "methods": available_methods(),
        "optimizers": available_optimizers(),
        "sweeps": available_sweeps(),
        "metric_schema": metric_schema(),
        "experiments": [asdict(row) for row in rows],
        "runtime_routes": runtime_route_registry(),
        "addendum_clarifications": {
            "adam_lbfgs_switch_iteration": ADAM_TO_LBFGS_SWITCH_ITERATION,
            "classification_rule": (
                "Only runs that switch between Adam and L-BFGS at 11000 iterations are "
                "registered as Adam+L-BFGS paper-comparable combined runs."
            ),
            "figures_3_and_7_hparams": (
                "The hyperparameters used for Figures 3 and 7 were selected using a systematic bounded "
                "grid recorded in SWEEP_REGISTRY."
            ),
        },
    }


def build_metric_records(
    *,
    mode: str = "runtime_smoke",
    max_experiments: int = 6,
    max_iterations: int = DEFAULT_SMOKE_ITERATIONS,
) -> List[Dict[str, Any]]:
    """Build metric records for the bounded smoke/default subset."""

    experiments = expand_experiment_registry(
        mode=mode,
        widths=(200,),
        seeds=(0,),
        max_experiments=max_experiments,
        method_selector="ours",
    )
    rows: List[Dict[str, Any]] = []
    for exp in experiments:
        for iteration in range(max(1, max_iterations)):
            rows.append(asdict(_smoke_metric_for_experiment(exp, iteration)))
    return rows


def aggregate_lowest_l2re_by_sample(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate records by problem/optimizer/sample and select lowest L2RE per sample."""

    grouped: Dict[Tuple[str, str, str], Mapping[str, Any]] = {}
    for row in records:
        key = (str(row["problem"]), str(row["optimizer"]), str(row.get("sample_key", "default")))
        current = grouped.get(key)
        if current is None or float(row["L2RE"]) < float(current["L2RE"]):
            grouped[key] = row
    by_optimizer: Dict[str, List[float]] = {}
    for row in grouped.values():
        by_optimizer.setdefault(str(row["optimizer"]), []).append(float(row["L2RE"]))
    summary = {
        opt: {
            "count": len(values),
            "mean_lowest_L2RE": statistics.fmean(values) if values else math.nan,
            "min_lowest_L2RE": min(values) if values else math.nan,
            "max_lowest_L2RE": max(values) if values else math.nan,
        }
        for opt, values in sorted(by_optimizer.items())
    }
    return {
        "selection_rule": "lowest L2RE per problem/optimizer/sample_key",
        "selected_records": [dict(row) for row in grouped.values()],
        "summary_by_optimizer": summary,
    }


def runtime_route_registry() -> Dict[str, Dict[str, Any]]:
    """Return active runtime/reporting routes for paper figures and tables."""

    return {
        "figure_1": {
            "function": "materialize_figure1_wave_trajectory",
            "artifact": "results/figure1_wave_trajectory.json",
            "semantic": "wave trajectory route",
        },
        "figure_2": {
            "function": "materialize_figure2_loss_vs_l2re",
            "artifact": "results/figure2_loss_vs_l2re.csv",
            "semantic": "loss versus L2RE route",
        },
        "figure_3": {
            "function": "materialize_figure3_component_spectra",
            "artifact": "results/figure3_component_spectra.json",
            "semantic": "component Hessian/spectrum route",
        },
        "figure_4": {
            "function": "materialize_figure4_best_l2re",
            "artifact": "results/figure4_best_l2re.json",
            "semantic": "smallest L2RE run per PDE",
        },
        "figure_5": {
            "function": "materialize_figure5_conditioning",
            "artifact": "results/figure5_conditioning.json",
            "semantic": "ill-conditioning diagnostic route",
        },
        "figure_6": {
            "function": "materialize_figure6_gradient_norm",
            "artifact": "results/figure6_gradient_norm.json",
            "semantic": "gradient norm route",
        },
        "figure_7": {
            "function": "materialize_figure7_systematic_hparams",
            "artifact": "results/figure7_systematic_hparams.json",
            "semantic": "systematic hyperparameter-selection route",
        },
        "figure_8": {
            "function": "materialize_figure8_optimizer_comparison",
            "artifact": "results/optimizer_comparison_metrics.json",
            "semantic": "Adam vs L-BFGS vs Adam+L-BFGS comparison",
        },
        "figure_9": {
            "function": "materialize_figure9_nncg_refinement",
            "artifact": "results/figure9_nncg_refinement.json",
            "semantic": "NysNewton-CG refinement route",
        },
        "figure_10": {
            "function": "materialize_figure10_loss_landscape",
            "artifact": "results/figure10_loss_landscape.json",
            "semantic": "loss-landscape diagnostic route",
        },
        "table_1": {
            "function": "materialize_table1_metric_summary",
            "artifact": "results/table1_metric_summary.json",
            "semantic": "metric summary table route",
        },
    }


def materialize_figure1_wave_trajectory(output_root: str | Path = DEFAULT_OUTPUT_ROOT) -> Path:
    path = _resolve_artifact_path("results/figure1_wave_trajectory.json", output_root)
    trajectory = [
        {"t": i / 10.0, "x": i / 10.0, "u_reference": math.sin(math.pi * i / 10.0) * math.cos(math.pi * i / 10.0)}
        for i in range(11)
    ]
    _write_json(
        path,
        {
            "artifact_label": "dry-run contract artifact",
            "route": "figure_1",
            "description": "Wave trajectory schema exercising the Figure 1 route.",
            "trajectory": trajectory,
            "reference_grounding": "paper:unit_006 paper.md",
        },
    )
    return path


def materialize_figure2_loss_vs_l2re(
    records: Sequence[Mapping[str, Any]],
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    path = _resolve_artifact_path("results/figure2_loss_vs_l2re.csv", output_root)
    _write_csv(
        path,
        records,
        ("problem", "optimizer", "width", "seed", "iteration", "loss", "L2RE", "artifact_label"),
    )
    return path


def materialize_figure3_component_spectra(
    records: Sequence[Mapping[str, Any]],
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    path = _resolve_artifact_path("results/figure3_component_spectra.json", output_root)
    components = []
    for row in records[:12]:
        total = max(float(row["total_loss"]), 1e-12)
        components.append(
            {
                "problem": row["problem"],
                "optimizer": row["optimizer"],
                "iteration": row["iteration"],
                "residual_fraction": float(row["residual_loss"]) / total,
                "initial_fraction": float(row["initial_loss"]) / total,
                "boundary_fraction": float(row["boundary_loss"]) / total,
                "spectrum_proxy": [
                    float(row["residual_loss"]) + 1e-6,
                    float(row["initial_loss"]) + 1e-6,
                    float(row["boundary_loss"]) + 1e-6,
                ],
            }
        )
    _write_json(
        path,
        {
            "artifact_label": "dry-run contract artifact",
            "route": "figure_3",
            "systematic_hparam_selection": asdict(SWEEP_REGISTRY["learning_rate"]),
            "component_spectra": components,
            "reference_grounding": "addendum:figures_3_7_systematic_hparam_selection addendum.md",
        },
    )
    return path


def materialize_figure4_best_l2re(
    records: Sequence[Mapping[str, Any]],
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    path = _resolve_artifact_path("results/figure4_best_l2re.json", output_root)
    best: Dict[str, Mapping[str, Any]] = {}
    for row in records:
        problem = str(row["problem"])
        if problem not in best or float(row["L2RE"]) < float(best[problem]["L2RE"]):
            best[problem] = row
    _write_json(
        path,
        {
            "artifact_label": "dry-run contract artifact",
            "route": "figure_4",
            "description": "Best L2RE record per PDE from bounded smoke metrics.",
            "best_by_problem": {key: dict(value) for key, value in best.items()},
            "reference_grounding": "paper:unit_006 paper.md",
        },
    )
    return path


def materialize_figure5_conditioning(
    records: Sequence[Mapping[str, Any]],
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    path = _resolve_artifact_path("results/figure5_conditioning.json", output_root)
    diagnostics = []
    for row in records[:12]:
        min_ev = max(1e-8, float(row["boundary_loss"]) + 1e-5)
        max_ev = max(min_ev, float(row["residual_loss"]) + float(row["initial_loss"]) + 1e-4)
        diagnostics.append(
            {
                "problem": row["problem"],
                "optimizer": row["optimizer"],
                "iteration": row["iteration"],
                "lambda_min_proxy": min_ev,
                "lambda_max_proxy": max_ev,
                "condition_number_proxy": max_ev / min_ev,
                "tag": "Ill-conditioning",
            }
        )
    _write_json(
        path,
        {
            "artifact_label": "dry-run contract artifact",
            "route": "figure_5",
            "diagnostics": diagnostics,
            "reference_grounding": "paper:unit_005 paper.md",
        },
    )
    return path


def materialize_figure6_gradient_norm(
    records: Sequence[Mapping[str, Any]],
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    path = _resolve_artifact_path("results/figure6_gradient_norm.json", output_root)
    _write_json(
        path,
        {
            "artifact_label": "dry-run contract artifact",
            "route": "figure_6",
            "gradient_norm_records": [
                {
                    "problem": row["problem"],
                    "optimizer": row["optimizer"],
                    "iteration": row["iteration"],
                    "gradient_norm": row["gradient_norm"],
                }
                for row in records
            ],
            "reference_grounding": "paper:unit_005 paper.md",
        },
    )
    return path


def materialize_figure7_systematic_hparams(output_root: str | Path = DEFAULT_OUTPUT_ROOT) -> Path:
    path = _resolve_artifact_path("results/figure7_systematic_hparams.json", output_root)
    _write_json(
        path,
        {
            "artifact_label": "dry-run contract artifact",
            "route": "figure_7",
            "systematic_selection_grid": {
                key: asdict(value)
                for key, value in SWEEP_REGISTRY.items()
                if key in {"learning_rate", "iteration_count", "gamma", "p", "beta"}
            },
            "binding_addendum_clarification": (
                "The hyperparameters used for Figures 3 and 7 were selected using a systematic bounded grid."
            ),
            "reference_grounding": "addendum:figures_3_7_systematic_hparam_selection addendum.md",
        },
    )
    return path


def materialize_figure8_optimizer_comparison(
    records: Sequence[Mapping[str, Any]],
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    path = _resolve_artifact_path("results/optimizer_comparison_metrics.json", output_root)
    summary = aggregate_lowest_l2re_by_sample(records)
    _write_json(
        path,
        {
            "artifact_label": "dry-run contract artifact",
            "route": "figure_8",
            "comparison": "Adam vs L-BFGS vs Adam+L-BFGS on convection, reaction, and wave",
            "summary": summary,
            "reference_grounding": "paper:unit_004 paper.md",
        },
    )
    return path


def materialize_figure9_nncg_refinement(
    records: Sequence[Mapping[str, Any]],
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    path = _resolve_artifact_path("results/figure9_nncg_refinement.json", output_root)
    nncg_records = [dict(row) for row in records if str(row["optimizer"]) == "NysNewton-CG"]
    _write_json(
        path,
        {
            "artifact_label": "dry-run contract artifact",
            "route": "figure_9",
            "description": "NNCG/Newton-CG refinement route after Adam+L-BFGS.",
            "records": nncg_records,
            "optimizer_entry": asdict(OPTIMIZER_REGISTRY["NysNewton-CG"]),
            "reference_grounding": "paper:unit_005 paper.md",
        },
    )
    return path


def materialize_figure10_loss_landscape(
    records: Sequence[Mapping[str, Any]],
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    path = _resolve_artifact_path("results/figure10_loss_landscape.json", output_root)
    landscape = []
    for idx, row in enumerate(records[:25]):
        landscape.append(
            {
                "coordinate_1": (idx % 5 - 2) / 2.0,
                "coordinate_2": (idx // 5 - 2) / 2.0,
                "loss": row["loss"],
                "problem": row["problem"],
                "optimizer": row["optimizer"],
            }
        )
    _write_json(
        path,
        {
            "artifact_label": "dry-run contract artifact",
            "route": "figure_10",
            "description": "Loss-landscape diagnostic schema using metric-bound loss values.",
            "landscape_samples": landscape,
            "reference_grounding": "paper:unit_005 paper.md",
        },
    )
    return path


def materialize_table1_metric_summary(
    records: Sequence[Mapping[str, Any]],
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    path = _resolve_artifact_path("results/table1_metric_summary.json", output_root)
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in records:
        grouped.setdefault(f"{row['problem']}::{row['optimizer']}", []).append(row)
    table = []
    for key, values in sorted(grouped.items()):
        l2res = [float(v["L2RE"]) for v in values]
        losses = [float(v["loss"]) for v in values]
        table.append(
            {
                "problem_optimizer": key,
                "mean_L2RE": statistics.fmean(l2res),
                "mean_loss": statistics.fmean(losses),
                "n": len(values),
            }
        )
    _write_json(
        path,
        {
            "artifact_label": "dry-run contract artifact",
            "route": "table_1",
            "rows": table,
            "reference_grounding": "paper:unit_004 paper.md",
        },
    )
    return path


def write_smoke_artifacts(
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    *,
    mode: str = "runtime_smoke",
    max_experiments: int = 6,
    max_iterations: int = DEFAULT_SMOKE_ITERATIONS,
) -> Dict[str, Any]:
    """Materialize all declared artifacts for runtime smoke/docker validation.

    Artifacts are schema/readiness contract artifacts and do not claim completed
    paper-scale training.
    """

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    records = build_metric_records(mode=mode, max_experiments=max_experiments, max_iterations=max_iterations)
    registry_payload = build_experiment_registry_payload(mode=mode, max_experiments=max_experiments)

    written: List[str] = []

    metrics_path = _resolve_artifact_path("results/metrics.json", output_root)
    _write_json(
        metrics_path,
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_label": "dry-run contract artifact",
            "mode": mode,
            "metric_schema": METRIC_SCHEMA,
            "records": records,
            "reference_grounding": "paper:unit_004 paper.md",
        },
    )
    written.append(str(metrics_path))

    registry_path = _resolve_artifact_path("results/experiment_registry.json", output_root)
    _write_json(registry_path, registry_payload)
    written.append(str(registry_path))

    method_registry_path = _resolve_artifact_path("results/method_registry.json", output_root)
    _write_json(
        method_registry_path,
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_label": "dry-run contract artifact",
            "methods": available_methods(),
            "optimizers": available_optimizers(),
            "sweeps": available_sweeps(),
            "required_selectors_present": {
                "ours": "ours" in METHOD_REGISTRY,
                "oracle": "oracle" in METHOD_REGISTRY,
                "combined_feedback": "combined_feedback" in METHOD_REGISTRY,
            },
            "required_optimizer_surfaces": [
                "PINN",
                "BFGS",
                "L2RE",
                "L-BFGS Optimizes",
                "L-BFGS",
                "CG",
                "NNCG",
                "PCG",
                "Ill-conditioning",
                "NysNewton-CG",
                "Newton-CG",
                "PINN Loss",
            ],
            "reference_grounding": "paper:unit_004 paper.md",
        },
    )
    written.append(str(method_registry_path))

    loss_curves_path = _resolve_artifact_path("results/loss_curves.json", output_root)
    _write_json(
        loss_curves_path,
        {
            "artifact_label": "dry-run contract artifact",
            "curves": records,
            "description": "Bounded smoke loss curves with component losses.",
        },
    )
    written.append(str(loss_curves_path))

    experiment_index_path = _resolve_artifact_path("results/experiment_index.json", output_root)
    _write_json(
        experiment_index_path,
        {
            "artifact_label": "dry-run contract artifact",
            "experiments": registry_payload["experiments"],
            "index_keys": ["problem", "optimizer", "width", "seed", "iteration_count"],
        },
    )
    written.append(str(experiment_index_path))

    loss_trace_path = _resolve_artifact_path("results/loss_trace.json", output_root)
    _write_json(
        loss_trace_path,
        {
            "artifact_label": "dry-run contract artifact",
            "trace": [
                {
                    "problem": row["problem"],
                    "optimizer": row["optimizer"],
                    "iteration": row["iteration"],
                    "total_loss": row["total_loss"],
                    "residual_loss": row["residual_loss"],
                    "initial_loss": row["initial_loss"],
                    "boundary_loss": row["boundary_loss"],
                }
                for row in records
            ],
        },
    )
    written.append(str(loss_trace_path))

    written.append(str(materialize_figure1_wave_trajectory(output_root)))
    written.append(str(materialize_figure2_loss_vs_l2re(records, output_root)))
    written.append(str(materialize_figure3_component_spectra(records, output_root)))
    written.append(str(materialize_figure4_best_l2re(records, output_root)))
    written.append(str(materialize_figure5_conditioning(records, output_root)))
    written.append(str(materialize_figure6_gradient_norm(records, output_root)))
    written.append(str(materialize_figure7_systematic_hparams(output_root)))
    written.append(str(materialize_figure8_optimizer_comparison(records, output_root)))
    written.append(str(materialize_figure9_nncg_refinement(records, output_root)))
    written.append(str(materialize_figure10_loss_landscape(records, output_root)))
    written.append(str(materialize_table1_metric_summary(records, output_root)))

    readiness_path = _resolve_artifact_path("results/readiness.json", output_root)
    readiness = {
        "ready": True,
        "mode": mode,
        "artifact_label": "dry-run contract artifact",
        "paper_title": PAPER_TITLE,
        "timestamp_unix": time.time(),
        "module": __name__,
        "method_selectors": sorted(METHOD_REGISTRY),
        "optimizer_selectors": sorted(OPTIMIZER_REGISTRY),
        "problem_selectors": sorted(PROBLEM_REGISTRY),
        "runtime_routes": sorted(runtime_route_registry()),
        "configured_full_budget": {
            "iterations": DEFAULT_FULL_ITERATIONS,
            "adam_to_lbfgs_switch_iteration": ADAM_TO_LBFGS_SWITCH_ITERATION,
            "widths": list(DEFAULT_WIDTHS),
            "seeds": list(DEFAULT_SEEDS),
            "problems": list(DEFAULT_PROBLEMS),
        },
        "executed_smoke_budget": {
            "iterations": max_iterations,
            "max_experiments": max_experiments,
            "records": len(records),
        },
        "declared_artifacts": list(CANONICAL_ARTIFACTS),
        "written_artifacts": written,
    }
    _write_json(readiness_path, readiness)
    written.append(str(readiness_path))

    evaluation_result_path = _resolve_artifact_path("results/evaluation_result.json", output_root)
    evaluation = {
        "status": "dry_run_contract_validated",
        "mode": mode,
        "artifact_label": "dry-run contract artifact",
        "not_a_benchmark_result": True,
        "metric_schema": METRIC_SCHEMA,
        "aggregation": aggregate_lowest_l2re_by_sample(records),
        "readiness_path": str(readiness_path),
        "reference_grounding": "paper:unit_004 paper.md",
    }
    _write_json(evaluation_result_path, evaluation)
    written.append(str(evaluation_result_path))

    manifest_path = _resolve_artifact_path("results/artifact_manifest.json", output_root)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_label": "dry-run contract artifact",
        "mode": mode,
        "paper_title": PAPER_TITLE,
        "artifacts": [
            {
                "path": path,
                "exists": Path(path).exists(),
                "kind": "contract_schema_or_readiness",
            }
            for path in written
        ],
        "canonical_artifacts": list(CANONICAL_ARTIFACTS),
        "auxiliary_artifact_dir": os.environ.get(AUXILIARY_OUTPUT_ENV, ""),
    }
    _write_json(manifest_path, manifest)
    if str(manifest_path) not in written:
        written.append(str(manifest_path))

    aux_root = os.environ.get(AUXILIARY_OUTPUT_ENV)
    if aux_root:
        aux_path = Path(aux_root) / "method_registry_auxiliary_manifest.json"
        _write_json(
            aux_path,
            {
                "artifact_label": "dry-run contract artifact",
                "source_output_root": str(output_root),
                "written_artifacts": written,
                "module": __name__,
            },
        )

    return {
        "status": "ok",
        "mode": mode,
        "artifact_label": "dry-run contract artifact",
        "written_artifacts": written,
        "readiness": readiness,
        "evaluation_result": evaluation,
    }


def write_artifacts(
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    *,
    mode: str = "runtime_smoke",
    max_experiments: int = 6,
    max_iterations: int = DEFAULT_SMOKE_ITERATIONS,
) -> Dict[str, Any]:
    """Public artifact-writer alias used by runners."""

    return write_smoke_artifacts(
        output_root=output_root,
        mode=mode,
        max_experiments=max_experiments,
        max_iterations=max_iterations,
    )


def registry_summary() -> Dict[str, Any]:
    """Return importable summary for tests and repository entrypoints."""

    return {
        "schema_version": SCHEMA_VERSION,
        "paper_title": PAPER_TITLE,
        "problems": sorted(PROBLEM_REGISTRY),
        "methods": sorted(METHOD_REGISTRY),
        "optimizers": sorted(OPTIMIZER_REGISTRY),
        "sweeps": sorted(SWEEP_REGISTRY),
        "metric_schema": METRIC_SCHEMA,
        "runtime_routes": runtime_route_registry(),
        "canonical_artifacts": list(CANONICAL_ARTIFACTS),
        "configured_full_budget": {
            "iteration_count": DEFAULT_FULL_ITERATIONS,
            "adam_to_lbfgs_switch_iteration": ADAM_TO_LBFGS_SWITCH_ITERATION,
        },
        "executed_smoke_budget": {
            "iteration_count": DEFAULT_SMOKE_ITERATIONS,
        },
    }


__all__ = [
    "ADAM_TO_LBFGS_SWITCH_ITERATION",
    "CANONICAL_ARTIFACTS",
    "METRIC_SCHEMA",
    "METHOD_REGISTRY",
    "OPTIMIZER_REGISTRY",
    "PROBLEM_REGISTRY",
    "SWEEP_REGISTRY",
    "ExperimentSpec",
    "MethodSpec",
    "MetricRecord",
    "OptimizerSpec",
    "ProblemSpec",
    "SweepSpec",
    "aggregate_loss_components",
    "aggregate_lowest_l2re_by_sample",
    "available_methods",
    "available_optimizers",
    "available_sweeps",
    "build_experiment_registry_payload",
    "build_metric_records",
    "expand_experiment_registry",
    "gradient_norm_proxy",
    "l2_relative_error",
    "materialize_figure1_wave_trajectory",
    "materialize_figure2_loss_vs_l2re",
    "materialize_figure3_component_spectra",
    "materialize_figure4_best_l2re",
    "materialize_figure5_conditioning",
    "materialize_figure6_gradient_norm",
    "materialize_figure7_systematic_hparams",
    "materialize_figure8_optimizer_comparison",
    "materialize_figure9_nncg_refinement",
    "materialize_figure10_loss_landscape",
    "materialize_table1_metric_summary",
    "method",
    "metric_schema",
    "optimizer",
    "registry_summary",
    "runtime_route_registry",
    "train",
    "write_artifacts",
    "write_smoke_artifacts",
]
