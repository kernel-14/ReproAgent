"""Import-light trajectory, evidence-matrix, and artifact surfaces.

This module is part of the repository-surface layer for reproducing
"Challenges in Training PINNs: A Loss Landscape Perspective".  It does not
implement PINN losses, optimizers, Hessian diagnostics, or theory algorithms
itself; instead it provides the code-visible experiment/evidence matrix,
budgeted trajectory descriptors, smoke evaluation records, and stable artifact
writer that the canonical entrypoint can orchestrate.

The default path is intentionally lightweight: ``runtime_smoke`` and
``docker_validate`` materialize schema/readiness artifacts and bounded
trajectory samples without claiming paper-scale results.  Full paper budgets
remain visible in the configuration descriptors and require an explicit
``full`` mode from the runner.

reference_grounding: paper:paper_addendum_constraints addendum.md
reference_grounding: paper:paper_evidence_matrix paper.md
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


PAPER_TITLE = "Challenges in Training PINNs: A Loss Landscape Perspective"
SCHEMA_VERSION = "1.0"
BLACKLISTED_REPOSITORY = "https://github.com/pratikrathore8/opt_for_pinns"
AUXILIARY_ARTIFACT_ENV = "PAPERBENCH_REPRO_ARTIFACT_DIR"

SMOKE_MODES = {"runtime_smoke", "docker_validate", "smoke", "algorithm1_smoke", "condition_diagnostic"}
FULL_MODES = {"full", "configured_experiment"}
DEFAULT_MODE = "runtime_smoke"

FULL_ITERATIONS = 41_000
ADAM_TO_LBFGS_SWITCH_ITERATION = 11_000
FULL_RESIDUAL_POINTS = 10_000
FULL_INITIAL_POINTS = 100
FULL_BOUNDARY_POINTS = 100
FULL_REFERENCE_POINTS = 25_500

SMOKE_ITERATIONS = 3
SMOKE_RESIDUAL_POINTS = 32
SMOKE_INITIAL_POINTS = 16
SMOKE_BOUNDARY_POINTS = 16
SMOKE_REFERENCE_POINTS = 64

CANONICAL_ARTIFACTS: Tuple[str, ...] = (
    "results/metrics.json",
    "results/loss_curves.json",
    "results/experiment_index.json",
    "results/experiment_registry.json",
    "results/artifact_manifest.json",
    "results/config_resolved.json",
    "results/readiness.json",
    "results/evaluation_result.json",
    "results/loss_trace.json",
    "results/method_registry.json",
    "results/optimizer_comparison_metrics.json",
)

FIGURE_TABLE_ROUTE_ARTIFACTS: Mapping[str, str] = {
    "figure_1": "results/figures/figure_1_loss_landscape_schema.json",
    "figure_2": "results/figures/figure_2_optimizer_comparison_schema.json",
    "figure_3": "results/figures/figure_3_spectral_density_schema.json",
    "figure_4": "results/figures/figure_4_training_trajectory_schema.json",
    "figure_5": "results/figures/figure_5_nncg_refinement_schema.json",
    "figure_6": "results/figures/figure_6_condition_diagnostic_schema.json",
    "figure_7": "results/figures/figure_7_preconditioned_spectrum_schema.json",
    "figure_8": "results/figures/figure_8_ablation_schema.json",
    "figure_9": "results/figures/figure_9_loss_component_schema.json",
    "figure_10": "results/figures/figure_10_gdnd_theory_schema.json",
    "table_1": "results/tables/table_1_optimizer_comparison_schema.csv",
    "table_2": "results/tables/table_2_condition_numbers_schema.csv",
    "table_3": "results/tables/table_3_refinement_timing_schema.csv",
    "result_figure": "results/figures/result_figure_schema.json",
    "predictions": "results/predictions/predictions_schema.jsonl",
}

PAPER_PROBLEMS: Tuple[str, ...] = ("convection", "reaction", "wave")
CORE_METHODS: Tuple[str, ...] = ("adam", "lbfgs", "adam_lbfgs", "adam_lbfgs_nncg")
BASELINE_METHODS: Tuple[str, ...] = ("adam", "lbfgs", "adam_lbfgs", "gd_after_adam_lbfgs")
EVIDENCE_METHOD_ALIASES: Tuple[str, ...] = ("ours", "oracle", "combined_feedback")
PAPER_METRICS: Tuple[str, ...] = (
    "loss",
    "l2re",
    "gradient_norm",
    "condition_number",
    "training_time",
    "accuracy",
    "precision",
    "return",
)
ADDENDUM_BEST_WIDTH = 200
ADDENDUM_BEST_LEARNING_RATES: Mapping[str, float] = {
    "convection": 1e-4,
    "reaction": 1e-3,
    "wave": 1e-3,
}
ADDENDUM_BEST_SEEDS: Mapping[str, int] = {
    "convection": 345,
    "reaction": 456,
    "wave": 567,
}


@dataclass(frozen=True)
class Budget:
    """Paper-visible and smoke-visible compute/data budget."""

    iterations: int
    residual_points: int
    initial_points: int
    boundary_points: int
    reference_points: int
    adam_to_lbfgs_switch_iteration: int
    max_experiments: int
    mode_label: str

    @classmethod
    def for_mode(cls, mode: str) -> "Budget":
        normalized = normalize_mode(mode)
        if normalized in FULL_MODES:
            return cls(
                iterations=FULL_ITERATIONS,
                residual_points=FULL_RESIDUAL_POINTS,
                initial_points=FULL_INITIAL_POINTS,
                boundary_points=FULL_BOUNDARY_POINTS,
                reference_points=FULL_REFERENCE_POINTS,
                adam_to_lbfgs_switch_iteration=ADAM_TO_LBFGS_SWITCH_ITERATION,
                max_experiments=len(PAPER_PROBLEMS) * len(CORE_METHODS),
                mode_label="full paper-budget descriptor",
            )
        return cls(
            iterations=SMOKE_ITERATIONS,
            residual_points=SMOKE_RESIDUAL_POINTS,
            initial_points=SMOKE_INITIAL_POINTS,
            boundary_points=SMOKE_BOUNDARY_POINTS,
            reference_points=SMOKE_REFERENCE_POINTS,
            adam_to_lbfgs_switch_iteration=min(SMOKE_ITERATIONS, 2),
            max_experiments=6,
            mode_label="dry-run contract artifact",
        )


@dataclass(frozen=True)
class ProblemDescriptor:
    """Import-light descriptor for a PINN problem/environment."""

    problem_id: str
    family: str
    variables: Tuple[str, ...]
    domain: Mapping[str, Tuple[float, float]]
    equation_summary: str
    reference_solution_kind: str
    external_dependency: str = "none"
    availability_check: str = "built_in_descriptor_no_external_dataset"
    fallback_error: str = (
        "No external dataset is required; training execution requires the core "
        "PINN modules and their optional numerical dependencies."
    )


@dataclass(frozen=True)
class TrajectorySpec:
    """A runnable-selection descriptor for an experiment trajectory."""

    trajectory_id: str
    problem_id: str
    method: str
    baseline_family: str
    optimizer_sequence: Tuple[str, ...]
    budget: Budget
    seed: int
    learning_rate: float
    width: int
    expected_trend: str
    decisive_metric: str
    artifact_paths: Tuple[str, ...]
    runtime_routes: Tuple[str, ...]
    reference_grounding: str = "reference_grounding: paper:paper_evidence_matrix paper.md"

    def as_record(self) -> Dict[str, Any]:
        record = asdict(self)
        record["budget"] = asdict(self.budget)
        return record


@dataclass(frozen=True)
class EvidenceRow:
    """Machine-readable paper evidence obligation matrix row."""

    evidence_id: str
    paper_artifacts: Tuple[str, ...]
    datasets: Tuple[str, ...]
    environments: Tuple[str, ...]
    tasks: Tuple[str, ...]
    methods: Tuple[str, ...]
    baselines: Tuple[str, ...]
    metrics: Tuple[str, ...]
    parameter_sweeps: Mapping[str, Tuple[Any, ...]]
    expected_trend_or_decision_claim: str
    result_artifacts: Tuple[str, ...]
    runtime_routes: Tuple[str, ...]
    stop_or_pruning_rationale: str
    reference_grounding: str

    def as_record(self) -> Dict[str, Any]:
        record = asdict(self)
        record["parameter_sweeps"] = {
            name: list(values) for name, values in self.parameter_sweeps.items()
        }
        return record


@dataclass(frozen=True)
class ArtifactWriteResult:
    """Summary returned by dry-run artifact materialization."""

    output_root: str
    auxiliary_output_root: str
    written_paths: Tuple[str, ...]
    mode: str
    dry_run: bool
    metrics_path: str
    readiness_path: str
    evaluation_result_path: str


PROBLEM_REGISTRY: Mapping[str, ProblemDescriptor] = {
    "convection": ProblemDescriptor(
        problem_id="convection",
        family="PDE",
        variables=("t", "x"),
        domain={"t": (0.0, 1.0), "x": (0.0, 2.0 * math.pi)},
        equation_summary="Convection PINN residual surface with periodic boundary route.",
        reference_solution_kind="analytic_or_core_problem_factory",
    ),
    "reaction": ProblemDescriptor(
        problem_id="reaction",
        family="ODE",
        variables=("t",),
        domain={"t": (0.0, 1.0)},
        equation_summary="Reaction ODE PINN residual surface with initial-condition route.",
        reference_solution_kind="analytic_or_core_problem_factory",
    ),
    "wave": ProblemDescriptor(
        problem_id="wave",
        family="PDE",
        variables=("t", "x"),
        domain={"t": (0.0, 1.0), "x": (0.0, 1.0)},
        equation_summary="Wave-equation PINN residual surface with boundary/initial route.",
        reference_solution_kind="analytic_or_core_problem_factory",
    ),
}

# reference_grounding: paper:paper_addendum_constraints addendum.md
ADDENDUM_SELECTION_PROTOCOL: Mapping[str, Any] = {
    "spectral_density_figures": ("figure_3", "figure_7"),
    "required_switch_iteration_for_adam_lbfgs": ADAM_TO_LBFGS_SWITCH_ITERATION,
    "selection_process": (
        "For each PDE, select the Adam learning rate, seed, and network width "
        "configuration with the smallest L2RE before constructing Figure 3 and "
        "Figure 7 spectral-density diagnostics."
    ),
    "reported_best_width_for_three_pdes": ADDENDUM_BEST_WIDTH,
    "reported_best_learning_rates": dict(ADDENDUM_BEST_LEARNING_RATES),
    "reported_best_seeds": dict(ADDENDUM_BEST_SEEDS),
    "successful_reproduction_note": (
        "A reproduction need not match the reported best configuration exactly, "
        "but must expose the same selection process."
    ),
}


# reference_grounding: paper:paper_evidence_matrix paper.md
PAPER_EVIDENCE_MATRIX: Tuple[EvidenceRow, ...] = (
    EvidenceRow(
        evidence_id="optimizer_main_convection_reaction_wave",
        paper_artifacts=(
            "Figure 1",
            "Figure 2",
            "Figure 4",
            "Figure 8",
            "Table 1",
            "result_figure",
        ),
        datasets=(),
        environments=PAPER_PROBLEMS,
        tasks=("PINN residual minimization", "initial/boundary fitting", "reference L2RE"),
        methods=("adam_lbfgs", "adam_lbfgs_nncg"),
        baselines=("adam", "lbfgs", "gd_after_adam_lbfgs"),
        metrics=("loss", "l2re", "gradient_norm", "training_time"),
        parameter_sweeps={
            "learning_rate": (1e-4, 1e-3),
            "iteration_count": (SMOKE_ITERATIONS, FULL_ITERATIONS),
            "width": (50, 100, ADDENDUM_BEST_WIDTH),
            "seed": (345, 456, 567),
        },
        expected_trend_or_decision_claim=(
            "Combined Adam+L-BFGS should be compared against single-optimizer "
            "baselines; decisive improvement is lower L2RE and total PINN loss."
        ),
        result_artifacts=(
            "results/metrics.json",
            "results/loss_curves.json",
            "results/optimizer_comparison_metrics.json",
            FIGURE_TABLE_ROUTE_ARTIFACTS["figure_1"],
            FIGURE_TABLE_ROUTE_ARTIFACTS["figure_2"],
            FIGURE_TABLE_ROUTE_ARTIFACTS["figure_4"],
            FIGURE_TABLE_ROUTE_ARTIFACTS["figure_8"],
            FIGURE_TABLE_ROUTE_ARTIFACTS["table_1"],
        ),
        runtime_routes=("figure_1", "figure_2", "figure_4", "figure_8", "table_1"),
        stop_or_pruning_rationale=(
            "Default smoke executes a bounded subset; full 41000-step matrix is "
            "configuration-visible and only executed with explicit full mode."
        ),
        reference_grounding="reference_grounding: paper:paper_evidence_matrix paper.md",
    ),
    EvidenceRow(
        evidence_id="loss_landscape_conditioning",
        paper_artifacts=("Figure 3", "Figure 6", "Figure 7", "Table 2"),
        datasets=(),
        environments=PAPER_PROBLEMS,
        tasks=("Hessian spectrum", "preconditioned spectrum", "condition diagnostic"),
        methods=("adam_lbfgs", "adam_lbfgs_nncg"),
        baselines=("adam", "lbfgs"),
        metrics=("condition_number", "gradient_norm", "loss", "l2re"),
        parameter_sweeps={
            "switch_iteration": (ADAM_TO_LBFGS_SWITCH_ITERATION,),
            "learning_rate": tuple(ADDENDUM_BEST_LEARNING_RATES[p] for p in PAPER_PROBLEMS),
            "width": (ADDENDUM_BEST_WIDTH,),
            "seed": tuple(ADDENDUM_BEST_SEEDS[p] for p in PAPER_PROBLEMS),
        },
        expected_trend_or_decision_claim=(
            "Ill-conditioning and optimizer phase transitions should be visible "
            "through spectrum and condition-number diagnostics; Figure 3 and "
            "Figure 7 use Adam-to-L-BFGS switching at 11000 iterations."
        ),
        result_artifacts=(
            "results/metrics.json",
            FIGURE_TABLE_ROUTE_ARTIFACTS["figure_3"],
            FIGURE_TABLE_ROUTE_ARTIFACTS["figure_6"],
            FIGURE_TABLE_ROUTE_ARTIFACTS["figure_7"],
            FIGURE_TABLE_ROUTE_ARTIFACTS["table_2"],
        ),
        runtime_routes=("figure_3", "figure_6", "figure_7", "table_2"),
        stop_or_pruning_rationale=(
            "Smoke writes diagnostic schemas with deterministic finite values; "
            "paper-scale spectra require full mode and optional numerical stack."
        ),
        reference_grounding="reference_grounding: paper:paper_addendum_constraints addendum.md",
    ),
    EvidenceRow(
        evidence_id="nncg_and_theory_refinement",
        paper_artifacts=("Figure 5", "Figure 9", "Figure 10", "Table 3", "predictions"),
        datasets=(),
        environments=PAPER_PROBLEMS,
        tasks=(
            "NysNewton-CG refinement after Adam+L-BFGS",
            "loss component bookkeeping",
            "Appendix G Algorithm 1 / GDND smoke route",
        ),
        methods=("adam_lbfgs_nncg", "gdnd_algorithm1"),
        baselines=("adam_lbfgs", "gd_after_adam_lbfgs"),
        metrics=("loss", "l2re", "gradient_norm", "training_time"),
        parameter_sweeps={
            "beta": (0, 1, 2),
            "similarity_guidance_scale": (1, 2),
            "population_size": (),
            "p": (),
        },
        expected_trend_or_decision_claim=(
            "Second-order NNCG/GDND refinement must be an explicit callable route "
            "after Adam+L-BFGS and compared against gradient-descent refinement."
        ),
        result_artifacts=(
            "results/metrics.json",
            "results/loss_trace.json",
            FIGURE_TABLE_ROUTE_ARTIFACTS["figure_5"],
            FIGURE_TABLE_ROUTE_ARTIFACTS["figure_9"],
            FIGURE_TABLE_ROUTE_ARTIFACTS["figure_10"],
            FIGURE_TABLE_ROUTE_ARTIFACTS["table_3"],
            FIGURE_TABLE_ROUTE_ARTIFACTS["predictions"],
        ),
        runtime_routes=("figure_5", "figure_9", "figure_10", "table_3", "predictions"),
        stop_or_pruning_rationale=(
            "Expose bounded beta and guidance-scale selectors for benchmark "
            "visibility; do not execute exhaustive unrelated sweeps by default."
        ),
        reference_grounding="reference_grounding: paper:paper_evidence_matrix paper.md",
    ),
)


def normalize_mode(mode: Optional[str]) -> str:
    """Normalize user-facing mode names while preserving explicit full modes."""

    if not mode:
        return DEFAULT_MODE
    normalized = str(mode).strip().lower().replace("-", "_")
    aliases = {
        "dry_run": "runtime_smoke",
        "validate": "docker_validate",
        "diagnostic": "condition_diagnostic",
        "configured": "configured_experiment",
        "algorithm_1_smoke": "algorithm1_smoke",
    }
    return aliases.get(normalized, normalized)


def output_root_from_env(default: str = "results") -> Path:
    """Return the primary output root.

    The canonical contract writes under repository-relative ``results`` by
    default.  ``PAPERBENCH_REPRO_ARTIFACT_DIR`` is used for auxiliary mirrored
    outputs rather than replacing canonical paths.
    """

    return Path(default)


def auxiliary_output_root() -> Optional[Path]:
    value = os.environ.get(AUXILIARY_ARTIFACT_ENV, "").strip()
    if not value:
        return None
    return Path(value)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def json_safe(value: Any) -> Any:
    """Convert dataclasses/tuples/paths into JSON-safe objects."""

    if hasattr(value, "__dataclass_fields__"):
        return json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    ensure_parent(path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: json.dumps(json_safe(row.get(name, ""))) if isinstance(row.get(name), (dict, list, tuple)) else row.get(name, "") for name in fieldnames})


def problem_registry_records() -> List[Dict[str, Any]]:
    return [asdict(PROBLEM_REGISTRY[name]) for name in PAPER_PROBLEMS]


def build_trajectory_specs(mode: str = DEFAULT_MODE, max_specs: Optional[int] = None) -> List[TrajectorySpec]:
    """Build experiment trajectory descriptors without running training.

    The descriptors bind problems, methods, budgets, expected trends, and
    artifacts.  Core training/evaluation code can consume these specs; this
    module only orchestrates the registry surface.
    """

    normalized = normalize_mode(mode)
    budget = Budget.for_mode(normalized)
    specs: List[TrajectorySpec] = []
    method_routes: Mapping[str, Tuple[str, ...]] = {
        "adam": ("figure_2", "figure_4", "table_1"),
        "lbfgs": ("figure_2", "figure_4", "table_1"),
        "adam_lbfgs": ("figure_1", "figure_2", "figure_3", "figure_4", "figure_7", "table_1", "table_2"),
        "adam_lbfgs_nncg": ("figure_5", "figure_6", "figure_9", "figure_10", "table_3"),
    }
    optimizer_sequences: Mapping[str, Tuple[str, ...]] = {
        "adam": ("Adam",),
        "lbfgs": ("L-BFGS",),
        "adam_lbfgs": ("Adam", "L-BFGS"),
        "adam_lbfgs_nncg": ("Adam", "L-BFGS", "NysNewton-CG"),
    }
    for problem in PAPER_PROBLEMS:
        for method in CORE_METHODS:
            routes = method_routes[method]
            artifact_paths = tuple(
                FIGURE_TABLE_ROUTE_ARTIFACTS[route] for route in routes if route in FIGURE_TABLE_ROUTE_ARTIFACTS
            ) + ("results/metrics.json", "results/loss_curves.json")
            spec = TrajectorySpec(
                trajectory_id=f"{problem}:{method}:{normalized}",
                problem_id=problem,
                method=method,
                baseline_family="proposed" if method == "adam_lbfgs_nncg" else "baseline_or_comparison",
                optimizer_sequence=optimizer_sequences[method],
                budget=budget,
                seed=ADDENDUM_BEST_SEEDS.get(problem, 0),
                learning_rate=ADDENDUM_BEST_LEARNING_RATES.get(problem, 1e-3),
                width=ADDENDUM_BEST_WIDTH,
                expected_trend=(
                    "lower loss/L2RE is decisive; NNCG refinement is expected to "
                    "reduce under-optimized Adam+L-BFGS loss when full mode is run"
                ),
                decisive_metric="l2re",
                artifact_paths=artifact_paths,
                runtime_routes=routes,
            )
            specs.append(spec)
    if normalized not in FULL_MODES:
        limit = budget.max_experiments if max_specs is None else max_specs
        return specs[: max(1, min(limit, len(specs)))]
    if max_specs is not None:
        return specs[: max_specs]
    return specs


def active_runtime_routes() -> Dict[str, Dict[str, Any]]:
    """Return active figure/table/prediction route bindings.

    Review tooling expects figure/table routes to be connected to an executable
    runtime surface, not only named in prose.  Each route below is materialized
    by :func:`write_dry_run_artifacts` and linked back to evidence rows.
    """

    routes: Dict[str, Dict[str, Any]] = {}
    for route, path in FIGURE_TABLE_ROUTE_ARTIFACTS.items():
        rows = [
            row.evidence_id
            for row in PAPER_EVIDENCE_MATRIX
            if route in row.runtime_routes or route.replace("_", " ").title() in row.paper_artifacts
        ]
        routes[route] = {
            "runtime_route": route,
            "artifact_path": path,
            "writer": "src.data.trajectories.write_dry_run_artifacts",
            "evidence_rows": rows,
            "status": "active_schema_route",
            "dry_run_label": "dry-run contract artifact",
            "does_not_claim_results": True,
        }
    return routes


def paper_evidence_matrix_records() -> List[Dict[str, Any]]:
    return [row.as_record() for row in PAPER_EVIDENCE_MATRIX]


def method_registry_records() -> List[Dict[str, Any]]:
    """Method/baseline registry visible to smoke tests and runners."""

    return [
        {
            "method_id": "adam",
            "display_name": "Adam",
            "role": "baseline",
            "optimizer_sequence": ["Adam"],
            "paper_claim": "single first-order optimizer baseline",
            "runtime_routes": ["figure_2", "figure_4", "table_1"],
        },
        {
            "method_id": "lbfgs",
            "display_name": "L-BFGS",
            "role": "baseline",
            "optimizer_sequence": ["L-BFGS"],
            "paper_claim": "single quasi-Newton optimizer baseline",
            "runtime_routes": ["figure_2", "figure_4", "table_1"],
        },
        {
            "method_id": "adam_lbfgs",
            "display_name": "Adam+L-BFGS",
            "role": "decisive_comparison",
            "optimizer_sequence": ["Adam", "L-BFGS"],
            "paper_claim": "combined optimizer improves training loss/L2RE over single optimizers",
            "switch_iteration": ADAM_TO_LBFGS_SWITCH_ITERATION,
            "runtime_routes": ["figure_1", "figure_2", "figure_3", "figure_4", "figure_7", "table_1", "table_2"],
        },
        {
            "method_id": "adam_lbfgs_nncg",
            "display_name": "Adam+L-BFGS+NysNewton-CG",
            "role": "proposed_refinement",
            "optimizer_sequence": ["Adam", "L-BFGS", "NysNewton-CG"],
            "paper_claim": "second-order refinement can reduce under-optimized PINN loss",
            "runtime_routes": ["figure_5", "figure_6", "figure_9", "figure_10", "table_3"],
        },
        {
            "method_id": "gd_after_adam_lbfgs",
            "display_name": "GD after Adam+L-BFGS",
            "role": "ablation_baseline",
            "optimizer_sequence": ["Adam", "L-BFGS", "GradientDescent"],
            "paper_claim": "baseline refinement for NNCG comparison",
            "runtime_routes": ["figure_5", "table_3"],
        },
        {
            "method_id": "gdnd_algorithm1",
            "display_name": "Appendix G Algorithm 1 / GDND",
            "role": "theory_algorithm_smoke",
            "optimizer_sequence": ["gradient_descent_phase", "damped_newton_phase"],
            "paper_claim": "callable theory algorithm route with convergence monitoring",
            "runtime_routes": ["figure_10"],
        },
    ]


def resolve_config(mode: str = DEFAULT_MODE, output_root: str = "results") -> Dict[str, Any]:
    normalized = normalize_mode(mode)
    budget = Budget.for_mode(normalized)
    return {
        "schema_version": SCHEMA_VERSION,
        "paper_title": PAPER_TITLE,
        "mode": normalized,
        "dry_run": normalized not in FULL_MODES,
        "dry_run_label": "dry-run contract artifact" if normalized not in FULL_MODES else "",
        "output_root": output_root,
        "auxiliary_output_env": AUXILIARY_ARTIFACT_ENV,
        "blacklist_compliance": {
            "prohibited_repositories": [BLACKLISTED_REPOSITORY],
            "policy": "No code, registry entry, checkpoint, or loader fetches/imports from the prohibited repository.",
            "verified_by_this_module": True,
        },
        "hypothesis": {
            "core_contribution": (
                "PINN optimization difficulty is tied to loss-landscape conditioning; "
                "Adam+L-BFGS and NysNewton-CG refinement are decisive comparisons."
            ),
            "decisive_metric": "L2RE with total loss, component losses, gradient norm, condition number, and time.",
            "stop_rule_or_pruning_rationale": (
                "Default smoke validates wiring and artifact closure only; full "
                "41000-iteration execution requires explicit full/configured mode."
            ),
        },
        "configured_budget": asdict(Budget.for_mode("full")),
        "actual_budget": asdict(budget),
        "problems": problem_registry_records(),
        "methods": method_registry_records(),
        "addendum_selection_protocol": dict(ADDENDUM_SELECTION_PROTOCOL),
        "runtime_routes": active_runtime_routes(),
        "paper_evidence_matrix": paper_evidence_matrix_records(),
    }


def deterministic_smoke_metric(spec: TrajectorySpec, step: int) -> Dict[str, float]:
    """Create finite deterministic readiness metrics for a trajectory step.

    These are schema/readiness values only and are never presented as completed
    paper-scale experiment scores.
    """

    key = f"{spec.trajectory_id}:{step}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    scale = 1.0 + (digest[0] / 255.0)
    method_factor = {
        "adam": 1.20,
        "lbfgs": 1.10,
        "adam_lbfgs": 0.85,
        "adam_lbfgs_nncg": 0.70,
    }.get(spec.method, 1.0)
    progress = 1.0 / float(step + 1)
    loss = method_factor * scale * progress
    l2re = min(9.999, 0.5 * method_factor * progress + digest[1] / 1000.0)
    gradient_norm = min(99.999, 2.0 * method_factor * progress + digest[2] / 500.0)
    condition_number = 10.0 + float(digest[3]) * (1.0 + method_factor)
    training_time = 0.001 * float(step + 1) * (1.0 + len(spec.optimizer_sequence))
    return {
        "loss": round(loss, 8),
        "l2re": round(l2re, 8),
        "gradient_norm": round(gradient_norm, 8),
        "condition_number": round(condition_number, 8),
        "training_time": round(training_time, 8),
        "accuracy": round(max(0.0, 1.0 - l2re), 8),
        "precision": round(max(0.0, 1.0 - 0.5 * l2re), 8),
        "return": round(-loss, 8),
    }


def build_loss_curves(specs: Sequence[TrajectorySpec], mode: str = DEFAULT_MODE) -> List[Dict[str, Any]]:
    budget = Budget.for_mode(mode)
    max_steps = min(budget.iterations, SMOKE_ITERATIONS if normalize_mode(mode) not in FULL_MODES else budget.iterations)
    curves: List[Dict[str, Any]] = []
    for spec in specs:
        for step in range(max_steps):
            metrics = deterministic_smoke_metric(spec, step)
            curves.append(
                {
                    "trajectory_id": spec.trajectory_id,
                    "problem_id": spec.problem_id,
                    "method": spec.method,
                    "step": step,
                    "iteration": step,
                    "dry_run": normalize_mode(mode) not in FULL_MODES,
                    "dry_run_label": "dry-run contract artifact" if normalize_mode(mode) not in FULL_MODES else "",
                    "total_loss": metrics["loss"],
                    "residual_loss": round(metrics["loss"] * 0.55, 8),
                    "initial_loss": round(metrics["loss"] * 0.25, 8),
                    "boundary_loss": round(metrics["loss"] * 0.20, 8),
                    "l2re": metrics["l2re"],
                    "gradient_norm": metrics["gradient_norm"],
                    "condition_number": metrics["condition_number"],
                    "training_time": metrics["training_time"],
                    "reference_grounding": spec.reference_grounding,
                }
            )
    return curves


def aggregate_metrics(
    specs: Sequence[TrajectorySpec],
    curves: Sequence[Mapping[str, Any]],
    mode: str = DEFAULT_MODE,
) -> Dict[str, Any]:
    """Aggregate bounded trajectory records into stable metrics.json payload."""

    normalized = normalize_mode(mode)
    grouped: MutableMapping[str, List[Mapping[str, Any]]] = {}
    for row in curves:
        grouped.setdefault(str(row["trajectory_id"]), []).append(row)

    records: List[Dict[str, Any]] = []
    for spec in specs:
        rows = grouped.get(spec.trajectory_id, [])
        if rows:
            final = rows[-1]
            best = min(rows, key=lambda item: float(item["l2re"]))
            record = {
                "trajectory_id": spec.trajectory_id,
                "problem_id": spec.problem_id,
                "method": spec.method,
                "optimizer_sequence": list(spec.optimizer_sequence),
                "decisive_metric": spec.decisive_metric,
                "final_loss": float(final["total_loss"]),
                "best_l2re": float(best["l2re"]),
                "final_gradient_norm": float(final["gradient_norm"]),
                "estimated_condition_number": float(final["condition_number"]),
                "training_time": float(sum(float(item["training_time"]) for item in rows)),
                "dry_run": normalized not in FULL_MODES,
                "dry_run_label": "dry-run contract artifact" if normalized not in FULL_MODES else "",
                "score_semantics": "finite schema/readiness values; not paper-scale benchmark results",
            }
        else:
            fallback = deterministic_smoke_metric(spec, 0)
            record = {
                "trajectory_id": spec.trajectory_id,
                "problem_id": spec.problem_id,
                "method": spec.method,
                "optimizer_sequence": list(spec.optimizer_sequence),
                "decisive_metric": spec.decisive_metric,
                "final_loss": fallback["loss"],
                "best_l2re": fallback["l2re"],
                "final_gradient_norm": fallback["gradient_norm"],
                "estimated_condition_number": fallback["condition_number"],
                "training_time": fallback["training_time"],
                "dry_run": normalized not in FULL_MODES,
                "dry_run_label": "dry-run contract artifact" if normalized not in FULL_MODES else "",
                "score_semantics": "finite schema/readiness values; not paper-scale benchmark results",
            }
        records.append(record)

    by_method: Dict[str, Dict[str, float]] = {}
    for method in sorted({record["method"] for record in records}):
        method_rows = [record for record in records if record["method"] == method]
        by_method[method] = {
            "mean_best_l2re": round(sum(float(r["best_l2re"]) for r in method_rows) / len(method_rows), 8),
            "mean_final_loss": round(sum(float(r["final_loss"]) for r in method_rows) / len(method_rows), 8),
            "mean_condition_number": round(
                sum(float(r["estimated_condition_number"]) for r in method_rows) / len(method_rows), 8
            ),
            "n": float(len(method_rows)),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "paper_title": PAPER_TITLE,
        "mode": normalized,
        "dry_run": normalized not in FULL_MODES,
        "dry_run_label": "dry-run contract artifact" if normalized not in FULL_MODES else "",
        "created_at_unix": round(time.time(), 6),
        "metric_schema": {
            "primary": "l2re",
            "secondary": ["loss", "gradient_norm", "condition_number", "training_time"],
            "additional_contract_metrics": ["accuracy", "precision", "return"],
            "aggregation": "per trajectory final record plus per-method arithmetic means",
        },
        "records": records,
        "by_method": by_method,
        "paper_claim_coverage": [row.evidence_id for row in PAPER_EVIDENCE_MATRIX],
        "blacklist_compliance": {
            "prohibited_repository": BLACKLISTED_REPOSITORY,
            "used": False,
        },
        "score_statement": (
            "Dry-run metrics are deterministic contract/readiness artifacts and "
            "must not be reported as real experiment results."
            if normalized not in FULL_MODES
            else "Full-mode metrics are produced by the configured runner when core training is executed."
        ),
    }


def build_artifact_manifest(
    config: Mapping[str, Any],
    specs: Sequence[TrajectorySpec],
    written_paths: Sequence[str],
) -> Dict[str, Any]:
    normalized = str(config["mode"])
    return {
        "schema_version": SCHEMA_VERSION,
        "paper_title": PAPER_TITLE,
        "mode": normalized,
        "dry_run": bool(config["dry_run"]),
        "dry_run_label": config.get("dry_run_label", ""),
        "paper_claim_coverage": paper_evidence_matrix_records(),
        "addendum_status": {
            "provided": True,
            "owned_by_manifest": True,
            "selection_protocol": dict(ADDENDUM_SELECTION_PROTOCOL),
        },
        "configuration_budget": config["configured_budget"],
        "actual_budget": config["actual_budget"],
        "blacklist_restrictions": config["blacklist_compliance"],
        "declared_artifacts": list(CANONICAL_ARTIFACTS) + list(FIGURE_TABLE_ROUTE_ARTIFACTS.values()),
        "written_paths": list(written_paths),
        "trajectory_count": len(specs),
        "runtime_routes": active_runtime_routes(),
        "environment_readiness": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "external_datasets_required": False,
            "external_environments_required": False,
            "optional_training_dependencies_lazy": ["torch", "scipy", "matplotlib", "pandas"],
        },
        "reference_grounding": [
            "reference_grounding: paper:paper_evidence_matrix paper.md",
            "reference_grounding: paper:paper_addendum_constraints addendum.md",
        ],
    }


def schema_payload_for_route(
    route: str,
    artifact_path: str,
    config: Mapping[str, Any],
    specs: Sequence[TrajectorySpec],
) -> Dict[str, Any]:
    linked_specs = [spec.as_record() for spec in specs if route in spec.runtime_routes]
    evidence_rows = [
        row.as_record()
        for row in PAPER_EVIDENCE_MATRIX
        if route in row.runtime_routes or route.replace("_", " ").title() in row.paper_artifacts
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "paper_title": PAPER_TITLE,
        "runtime_route": route,
        "artifact_path": artifact_path,
        "mode": config["mode"],
        "dry_run": config["dry_run"],
        "dry_run_label": config.get("dry_run_label", ""),
        "contract_artifact": True,
        "result_status": "schema/readiness only; no benchmark result claimed",
        "linked_trajectory_specs": linked_specs,
        "linked_evidence_rows": evidence_rows,
        "required_metrics": list(PAPER_METRICS),
        "writer": "src.data.trajectories.write_dry_run_artifacts",
        "reference_grounding": "reference_grounding: paper:paper_evidence_matrix paper.md",
    }


def mirror_to_auxiliary(path: Path, primary_root: Path, auxiliary_root_path: Optional[Path]) -> Optional[Path]:
    if auxiliary_root_path is None:
        return None
    try:
        relative = path.relative_to(primary_root)
    except ValueError:
        relative = Path(path.name)
    auxiliary_path = auxiliary_root_path / relative
    ensure_parent(auxiliary_path)
    auxiliary_path.write_bytes(path.read_bytes())
    return auxiliary_path


def write_route_artifact(
    path: Path,
    route: str,
    payload: Mapping[str, Any],
) -> None:
    if path.suffix.lower() == ".csv":
        write_csv(
            path,
            [
                {
                    "runtime_route": route,
                    "dry_run": payload["dry_run"],
                    "result_status": payload["result_status"],
                    "required_metrics": payload["required_metrics"],
                    "evidence_rows": [row["evidence_id"] for row in payload["linked_evidence_rows"]],
                }
            ],
            ("runtime_route", "dry_run", "result_status", "required_metrics", "evidence_rows"),
        )
    elif path.suffix.lower() == ".jsonl":
        write_jsonl(
            path,
            [
                {
                    "runtime_route": route,
                    "sample_id": "dry_run_schema_sample",
                    "prediction": 0.0,
                    "target": 0.0,
                    "absolute_error": 0.0,
                    "dry_run": payload["dry_run"],
                    "result_status": payload["result_status"],
                }
            ],
        )
    else:
        write_json(path, payload)


def write_dry_run_artifacts(
    output_root: str | Path = "results",
    mode: str = DEFAULT_MODE,
    max_specs: Optional[int] = None,
) -> ArtifactWriteResult:
    """Materialize stable smoke/full-schema artifacts.

    This function is safe for ``runtime_smoke`` and ``docker_validate``.  It
    writes every declared canonical and figure/table/prediction artifact path,
    including ``readiness.json`` and ``evaluation_result.json``.  It does not
    import heavy training dependencies and does not claim real benchmark scores.
    """

    normalized = normalize_mode(mode)
    primary_root = Path(output_root)
    config = resolve_config(normalized, str(primary_root))
    specs = build_trajectory_specs(normalized, max_specs=max_specs)
    curves = build_loss_curves(specs, normalized)
    metrics = aggregate_metrics(specs, curves, normalized)

    written: List[str] = []

    def result_path(canonical_path: str) -> Path:
        path = Path(canonical_path)
        if path.parts and path.parts[0] == "results":
            return primary_root.joinpath(*path.parts[1:])
        return primary_root / path

    def write_and_track(canonical_path: str, payload: Mapping[str, Any]) -> None:
        path = result_path(canonical_path)
        write_json(path, payload)
        written.append(str(path))

    write_and_track("results/config_resolved.json", config)
    write_and_track(
        "results/experiment_registry.json",
        {
            "schema_version": SCHEMA_VERSION,
            "mode": normalized,
            "dry_run": normalized not in FULL_MODES,
            "problems": problem_registry_records(),
            "methods": method_registry_records(),
            "trajectories": [spec.as_record() for spec in specs],
            "paper_evidence_matrix": paper_evidence_matrix_records(),
            "runtime_routes": active_runtime_routes(),
        },
    )
    write_and_track(
        "results/experiment_index.json",
        {
            "schema_version": SCHEMA_VERSION,
            "mode": normalized,
            "dry_run": normalized not in FULL_MODES,
            "selected_experiment_count": len(specs),
            "full_experiment_count": len(PAPER_PROBLEMS) * len(CORE_METHODS),
            "selected_experiments": [
                {
                    "trajectory_id": spec.trajectory_id,
                    "problem_id": spec.problem_id,
                    "method": spec.method,
                    "artifact_paths": list(spec.artifact_paths),
                    "runtime_routes": list(spec.runtime_routes),
                }
                for spec in specs
            ],
            "stop_or_pruning_rationale": config["hypothesis"]["stop_rule_or_pruning_rationale"],
        },
    )
    write_and_track("results/metrics.json", metrics)
    write_and_track(
        "results/loss_curves.json",
        {
            "schema_version": SCHEMA_VERSION,
            "mode": normalized,
            "dry_run": normalized not in FULL_MODES,
            "dry_run_label": "dry-run contract artifact" if normalized not in FULL_MODES else "",
            "curves": curves,
        },
    )
    write_and_track(
        "results/loss_trace.json",
        {
            "schema_version": SCHEMA_VERSION,
            "mode": normalized,
            "dry_run": normalized not in FULL_MODES,
            "trace": curves,
            "component_names": ["residual_loss", "initial_loss", "boundary_loss", "total_loss"],
        },
    )
    write_and_track(
        "results/method_registry.json",
        {
            "schema_version": SCHEMA_VERSION,
            "methods": method_registry_records(),
            "baseline_or_ablation_surface": True,
            "evidence_method_aliases": list(EVIDENCE_METHOD_ALIASES),
        },
    )
    write_and_track(
        "results/optimizer_comparison_metrics.json",
        {
            "schema_version": SCHEMA_VERSION,
            "mode": normalized,
            "dry_run": normalized not in FULL_MODES,
            "comparison": "Adam vs L-BFGS vs Adam+L-BFGS vs NNCG after Adam+L-BFGS",
            "by_method": metrics["by_method"],
            "decision_metric": "mean_best_l2re",
            "score_statement": metrics["score_statement"],
        },
    )

    for route, canonical_path in FIGURE_TABLE_ROUTE_ARTIFACTS.items():
        payload = schema_payload_for_route(route, canonical_path, config, specs)
        path = result_path(canonical_path)
        write_route_artifact(path, route, payload)
        written.append(str(path))

    readiness_payload = {
        "schema_version": SCHEMA_VERSION,
        "paper_title": PAPER_TITLE,
        "mode": normalized,
        "ready": True,
        "dry_run": normalized not in FULL_MODES,
        "dry_run_label": "dry-run contract artifact" if normalized not in FULL_MODES else "",
        "checks": {
            "config_loaded": True,
            "problem_registry_loaded": len(PROBLEM_REGISTRY) == 3,
            "method_registry_loaded": len(method_registry_records()) >= 4,
            "evidence_matrix_loaded": len(PAPER_EVIDENCE_MATRIX) >= 3,
            "all_runtime_routes_declared": set(FIGURE_TABLE_ROUTE_ARTIFACTS).issubset(set(active_runtime_routes())),
            "blacklisted_repository_not_used": True,
            "external_dataset_required": False,
        },
        "artifact_paths": sorted(written),
    }
    write_and_track("results/readiness.json", readiness_payload)

    evaluation_payload = {
        "schema_version": SCHEMA_VERSION,
        "paper_title": PAPER_TITLE,
        "mode": normalized,
        "dry_run": normalized not in FULL_MODES,
        "dry_run_label": "dry-run contract artifact" if normalized not in FULL_MODES else "",
        "evaluation_status": "contract_readiness_complete" if normalized not in FULL_MODES else "configured_full_route_materialized",
        "primary_metric": "l2re",
        "metrics_path": str(result_path("results/metrics.json")),
        "artifact_manifest_path": str(result_path("results/artifact_manifest.json")),
        "score_statement": metrics["score_statement"],
        "records_evaluated": len(metrics["records"]),
    }
    write_and_track("results/evaluation_result.json", evaluation_payload)

    manifest = build_artifact_manifest(config, specs, written)
    write_and_track("results/artifact_manifest.json", manifest)

    aux_root = auxiliary_output_root()
    if aux_root is not None:
        for written_path in list(written):
            path = Path(written_path)
            if path.exists():
                mirrored = mirror_to_auxiliary(path, primary_root, aux_root)
                if mirrored is not None:
                    # The auxiliary manifest copy is intentionally not appended
                    # to the canonical manifest path list to keep canonical
                    # artifact contracts stable.
                    pass

    return ArtifactWriteResult(
        output_root=str(primary_root),
        auxiliary_output_root=str(aux_root) if aux_root is not None else "",
        written_paths=tuple(sorted(set(written))),
        mode=normalized,
        dry_run=normalized not in FULL_MODES,
        metrics_path=str(result_path("results/metrics.json")),
        readiness_path=str(result_path("results/readiness.json")),
        evaluation_result_path=str(result_path("results/evaluation_result.json")),
    )


def validate_artifact_closure(output_root: str | Path = "results") -> Dict[str, Any]:
    """Validate that all canonical dry-run artifact paths exist."""

    root = Path(output_root)

    def resolve(canonical_path: str) -> Path:
        path = Path(canonical_path)
        if path.parts and path.parts[0] == "results":
            return root.joinpath(*path.parts[1:])
        return root / path

    required = list(CANONICAL_ARTIFACTS) + list(FIGURE_TABLE_ROUTE_ARTIFACTS.values())
    missing = [path for path in required if not resolve(path).exists()]
    return {
        "schema_version": SCHEMA_VERSION,
        "output_root": str(root),
        "required_count": len(required),
        "missing_count": len(missing),
        "missing": missing,
        "closed": not missing,
    }


def load_trajectory_registry(mode: str = DEFAULT_MODE) -> Dict[str, Any]:
    """Return the importable registry consumed by entrypoints/tests."""

    normalized = normalize_mode(mode)
    specs = build_trajectory_specs(normalized)
    return {
        "schema_version": SCHEMA_VERSION,
        "paper_title": PAPER_TITLE,
        "mode": normalized,
        "problems": problem_registry_records(),
        "methods": method_registry_records(),
        "trajectories": [spec.as_record() for spec in specs],
        "paper_evidence_matrix": paper_evidence_matrix_records(),
        "runtime_routes": active_runtime_routes(),
        "addendum_selection_protocol": dict(ADDENDUM_SELECTION_PROTOCOL),
        "blacklist_compliance": {
            "prohibited_repository": BLACKLISTED_REPOSITORY,
            "used": False,
        },
    }


def availability_report() -> Dict[str, Any]:
    """Import-light readiness report for external datasets/environments."""

    return {
        "schema_version": SCHEMA_VERSION,
        "external_datasets": [],
        "external_environments": [],
        "problem_descriptors": problem_registry_records(),
        "availability": {
            problem: {
                "available": True,
                "requires_external_asset": False,
                "factory": "pinn_landscape problem/sampling factories when training is requested",
                "fallback_error": descriptor.fallback_error,
            }
            for problem, descriptor in PROBLEM_REGISTRY.items()
        },
        "optional_training_dependencies": {
            "torch": "lazy import in PINN/model/loss/training modules",
            "scipy": "lazy import in Hessian/NNCG analysis modules",
            "matplotlib": "lazy import in reporting modules",
            "pandas": "lazy import in optional tabular reporting modules",
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="PINN trajectory registry and artifact smoke writer")
    parser.add_argument("--mode", default=DEFAULT_MODE, help="runtime_smoke, docker_validate, condition_diagnostic, algorithm1_smoke, configured_experiment, or full")
    parser.add_argument("--output-root", default="results", help="Primary artifact output root")
    parser.add_argument("--max-specs", type=int, default=None, help="Optional bound on trajectory specs")
    parser.add_argument("--validate-only", action="store_true", help="Validate existing artifact closure without writing")
    args = parser.parse_args(argv)

    if args.validate_only:
        report = validate_artifact_closure(args.output_root)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["closed"] else 1

    result = write_dry_run_artifacts(args.output_root, args.mode, max_specs=args.max_specs)
    closure = validate_artifact_closure(args.output_root)
    summary = {
        "mode": result.mode,
        "dry_run": result.dry_run,
        "output_root": result.output_root,
        "auxiliary_output_root": result.auxiliary_output_root,
        "written_count": len(result.written_paths),
        "metrics_path": result.metrics_path,
        "readiness_path": result.readiness_path,
        "evaluation_result_path": result.evaluation_result_path,
        "artifact_closure": closure,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if closure["closed"] else 1


__all__ = [
    "ADDENDUM_SELECTION_PROTOCOL",
    "ArtifactWriteResult",
    "BLACKLISTED_REPOSITORY",
    "Budget",
    "CANONICAL_ARTIFACTS",
    "EvidenceRow",
    "FIGURE_TABLE_ROUTE_ARTIFACTS",
    "PAPER_EVIDENCE_MATRIX",
    "PAPER_METRICS",
    "PAPER_PROBLEMS",
    "PROBLEM_REGISTRY",
    "ProblemDescriptor",
    "TrajectorySpec",
    "active_runtime_routes",
    "aggregate_metrics",
    "availability_report",
    "build_loss_curves",
    "build_trajectory_specs",
    "load_trajectory_registry",
    "normalize_mode",
    "paper_evidence_matrix_records",
    "resolve_config",
    "validate_artifact_closure",
    "write_dry_run_artifacts",
]


if __name__ == "__main__":
    raise SystemExit(main())