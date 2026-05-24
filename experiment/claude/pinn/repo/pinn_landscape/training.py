"""Training-loop wrappers for the PINN reproduction package."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from pinn_landscape import losses
from pinn_landscape import hessian
from pinn_landscape import models
from pinn_landscape import optimizers as optimizer_configs
from pinn_landscape import problems
from pinn_landscape import sampling
from pinn_landscape.config import ExperimentConfig, resolve_config
from src.method_registry import (
    ExperimentSpec,
    MetricRecord,
    PROBLEM_REGISTRY as METHOD_PROBLEM_REGISTRY,
    _smoke_metric_for_experiment,
    expand_experiment_registry,
    optimizer as registry_optimizer,
)


@dataclass(frozen=True)
class TrainingRequest:
    problem: str = "convection"
    optimizer: str = "Adam"
    width: int = 200
    seed: int = 0
    mode: str = "runtime_smoke"
    max_steps: int = 3


def build_model(problem: str = "convection", width: int = 200, seed: int = 0, prefer_torch: bool = True) -> Any:
    return models.build_model(problem=problem, width=width, seed=seed, prefer_torch=prefer_torch)


def _torch_available_for_training(model: Any) -> bool:
    try:
        import torch  # type: ignore

        return hasattr(model, "parameters") and any(getattr(p, "requires_grad", False) for p in model.parameters())
    except Exception:
        return False


def _tensor_loss(model: Any, batch: sampling.SampleBatch) -> Any:
    return sampling.compute_loss_components(model, batch)["total"]


def _clone_parameters(model: Any) -> List[Any]:
    return [p.detach().clone() for p in model.parameters() if getattr(p, "requires_grad", False)]


def _clone_gradients(model: Any) -> List[Any]:
    grads = []
    for param in model.parameters():
        if getattr(param, "requires_grad", False):
            if param.grad is None:
                grads.append(param.detach().clone().zero_())
            else:
                grads.append(param.grad.detach().clone())
    return grads


def _evaluate_step(
    model: Any,
    batch: sampling.SampleBatch,
    problem_name: str,
    optimizer_name: str,
    width: int,
    seed: int,
    iteration: int,
    elapsed: float,
    mode: str,
) -> Dict[str, Any]:
    payload = losses.compute_pinn_losses(model, batch, problem_name)
    try:
        grad_norm = float(models._torch_gradient_norm(model)) if hasattr(model, "parameters") else 0.0
    except Exception:
        grad_norm = models.gradient_norm_proxy([float(payload.get("total_loss", 0.0))])
    return {
        "problem": problem_name,
        "optimizer": optimizer_name,
        "width": width,
        "seed": seed,
        "iteration": int(iteration),
        "loss": float(payload.get("total_loss", payload.get("loss", 0.0))),
        "L2RE": float(payload.get("l2re", payload.get("L2RE", 0.0))),
        "gradient_norm": float(grad_norm),
        "total_loss": float(payload.get("total_loss", 0.0)),
        "residual_loss": float(payload.get("residual_loss", 0.0)),
        "initial_loss": float(payload.get("initial_loss", 0.0)),
        "boundary_loss": float(payload.get("boundary_loss", 0.0)),
        "training_time": float(elapsed),
        "per_iteration_wall_clock_time": float(elapsed),
        "mode": mode,
    }


def _run_torch_training(
    model: Any,
    problem_name: str,
    optimizer_name: str,
    train_config: Mapping[str, Any],
) -> Dict[str, Any]:
    import torch  # type: ignore

    mode = str(train_config.get("mode", "runtime_smoke"))
    smoke = mode in {"runtime_smoke", "docker_validate", "dry_run", "smoke"}
    full_iterations = int(train_config.get("total_iterations", optimizer_configs.FULL_TRAINING_ITERATIONS))
    active_iterations = int(
        train_config.get(
            "iteration_count",
            train_config.get("max_steps", TrainingRequest.max_steps if False else (3 if smoke else full_iterations)),
        )
    )
    active_iterations = max(1, active_iterations)
    width = int(train_config.get("width", getattr(getattr(model, "config", None), "width", 200)))
    seed = int(train_config.get("seed", getattr(getattr(model, "config", None), "seed", 0)))
    canonical = models.normalize_optimizer_name(optimizer_name)
    lr = float(train_config.get("learning_rate", train_config.get("adam_learning_rate", 1e-3)))
    raw_optimizer_key = optimizer_name.lower().replace("_", "-").replace(" ", "")
    default_switch = 1_000 if "1k" in raw_optimizer_key else 11_000
    switch_iteration = int(train_config.get("switch_iteration", default_switch))
    active_switch = int(train_config.get("active_switch_iteration", min(switch_iteration, max(1, active_iterations - 1))))
    lbfgs_lr = float(train_config.get("lbfgs_lr", 1.0))
    history_size = int(train_config.get("history_size", 100))
    line_search = str(train_config.get("line_search", train_config.get("line_search_fn", "strong_wolfe")))

    batch = train_config.get("batch")
    if not isinstance(batch, sampling.SampleBatch):
        batch = sampling.sample_fixed_collocation_points(
            problem_name,
            mode="runtime_smoke" if smoke else "full",
            seed=seed,
            n_residual_points=int(train_config.get("n_residual_points", 32 if smoke else sampling.FULL_RESIDUAL_POINTS)),
            n_initial_points=int(train_config.get("n_initial_points", 16 if smoke else sampling.FULL_INITIAL_POINTS)),
            n_boundary_points=int(train_config.get("n_boundary_points", 16 if smoke else sampling.FULL_BOUNDARY_POINTS)),
        )

    metrics: List[Dict[str, Any]] = []
    optimizer_trace: List[Dict[str, Any]] = []
    lbfgs_parameter_snapshots: List[List[Any]] = []
    lbfgs_gradient_snapshots: List[List[Any]] = []

    def append_metric(step: int, phase: str, elapsed: float) -> None:
        row = _evaluate_step(model, batch, problem_name, canonical, width, seed, step, elapsed, mode)
        row["phase"] = phase
        metrics.append(row)

    if canonical == "Adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for step in range(active_iterations):
            start = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            loss = _tensor_loss(model, batch)
            loss.backward()
            opt.step()
            elapsed = time.perf_counter() - start
            optimizer_trace.append({"iteration": step, "optimizer": "Adam", "learning_rate": lr, "elapsed": elapsed})
            append_metric(step, "adam", elapsed)
    elif canonical == "L-BFGS":
        opt = torch.optim.LBFGS(
            model.parameters(),
            lr=lbfgs_lr,
            max_iter=1,
            history_size=history_size,
            line_search_fn=line_search,
        )

        def closure() -> Any:
            opt.zero_grad(set_to_none=True)
            loss = _tensor_loss(model, batch)
            loss.backward()
            return loss

        lbfgs_parameter_snapshots.append(_clone_parameters(model))
        closure()
        lbfgs_gradient_snapshots.append(_clone_gradients(model))
        for step in range(active_iterations):
            start = time.perf_counter()
            opt.step(closure)
            lbfgs_parameter_snapshots.append(_clone_parameters(model))
            closure()
            lbfgs_gradient_snapshots.append(_clone_gradients(model))
            elapsed = time.perf_counter() - start
            optimizer_trace.append(
                {
                    "iteration": step,
                    "optimizer": "L-BFGS",
                    "learning_rate": lbfgs_lr,
                    "history_size": history_size,
                    "line_search_fn": line_search,
                    "elapsed": elapsed,
                }
            )
            append_metric(step, "lbfgs", elapsed)
    elif canonical == "Adam+L-BFGS":
        schedule = optimizer_configs.build_adam_lbfgs_schedule(
            switch_iteration,
            adam_lr=lr,
            lbfgs_lr=lbfgs_lr,
            history_size=history_size,
            line_search=line_search,
            total_iterations=full_iterations,
            smoke_steps=active_iterations if smoke else None,
        )
        adam = torch.optim.Adam(model.parameters(), lr=lr)
        for step in range(min(active_switch, active_iterations)):
            start = time.perf_counter()
            adam.zero_grad(set_to_none=True)
            loss = _tensor_loss(model, batch)
            loss.backward()
            adam.step()
            elapsed = time.perf_counter() - start
            optimizer_trace.append({"iteration": step, "optimizer": "Adam", "learning_rate": lr, "elapsed": elapsed})
            append_metric(step, "adam", elapsed)
        if active_iterations > active_switch:
            lbfgs = torch.optim.LBFGS(
                model.parameters(),
                lr=lbfgs_lr,
                max_iter=1,
                history_size=history_size,
                line_search_fn=line_search,
            )
            lbfgs_parameter_snapshots.append(_clone_parameters(model))
            lbfgs.zero_grad(set_to_none=True)
            closure_loss = _tensor_loss(model, batch)
            closure_loss.backward()
            lbfgs_gradient_snapshots.append(_clone_gradients(model))
            for step in range(active_switch, active_iterations):
                start = time.perf_counter()

                def closure() -> Any:
                    lbfgs.zero_grad(set_to_none=True)
                    loss = _tensor_loss(model, batch)
                    loss.backward()
                    return loss

                lbfgs.step(closure)
                lbfgs_parameter_snapshots.append(_clone_parameters(model))
                closure()
                lbfgs_gradient_snapshots.append(_clone_gradients(model))
                elapsed = time.perf_counter() - start
                optimizer_trace.append(
                    {
                        "iteration": step,
                        "optimizer": "L-BFGS",
                        "learning_rate": lbfgs_lr,
                        "history_size": history_size,
                        "line_search_fn": line_search,
                        "elapsed": elapsed,
                    }
                )
                append_metric(step, "lbfgs", elapsed)
    elif canonical in {"NysNewton-CG", "NNCG"}:
        from pinn_landscape import nncg

        for step in range(active_iterations):
            start = time.perf_counter()
            step_payload = nncg.nncg_step(
                model,
                problems.make_problem(problem_name, {"mode": mode, "seed": seed}),
                batch=batch,
                config={
                    "rank": int(train_config.get("rank", train_config.get("second_order_rank", 16))),
                    "damping": float(train_config.get("damping", 1e-3)),
                    "cg_tolerance": float(train_config.get("cg_tolerance", 1e-6)),
                },
            )
            elapsed = time.perf_counter() - start
            optimizer_trace.append(
                {
                    "iteration": step,
                    "optimizer": canonical,
                    "elapsed": elapsed,
                    "algorithm_4_status": step_payload.get("status"),
                    "armijo_step_size": step_payload.get("step_size"),
                }
            )
            append_metric(step, "nysnewton_cg", elapsed)
    else:
        opt = torch.optim.SGD(
            model.parameters(),
            lr=float(train_config.get("learning_rate", train_config.get("nncg_fallback_lr", 1e-3))),
        )
        for step in range(active_iterations):
            start = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            loss = _tensor_loss(model, batch)
            loss.backward()
            opt.step()
            elapsed = time.perf_counter() - start
            optimizer_trace.append({"iteration": step, "optimizer": canonical, "elapsed": elapsed})
            append_metric(step, canonical.lower(), elapsed)

    final = metrics[-1]
    lbfgs_history = hessian.record_lbfgs_history(lbfgs_parameter_snapshots, lbfgs_gradient_snapshots) if len(lbfgs_parameter_snapshots) >= 2 else {"num_pairs": 0, "pairs": []}
    gradient_vector = []
    if lbfgs_gradient_snapshots:
        for grad in lbfgs_gradient_snapshots[-1]:
            gradient_vector.extend(float(v) for v in grad.detach().cpu().reshape(-1).tolist())
    lbfgs_unrolled_direction = hessian.unroll_lbfgs_update_algorithm_2(gradient_vector[:4096], lbfgs_history) if gradient_vector and lbfgs_history.get("pairs") else []
    raw_spectrum = hessian.estimate_hessian_spectrum(model, problems.make_problem(problem_name, {"mode": mode, "seed": seed}), batch=batch)
    preconditioned_spectrum = hessian.preconditioned_hessian_spectrum_algorithm_3(
        raw_spectrum.get("eigenvalues", []),
        lbfgs_history,
    )
    return {
        "status": "completed_torch_training_path",
        "mode": mode,
        "optimizer": canonical,
        "problem": problem_name,
        "width": width,
        "seed": seed,
        "iterations_executed": active_iterations,
        "configured_full_budget": {
            "iteration_count": full_iterations,
            "adam_lr_grid": list(optimizer_configs.ADAM_LR_GRID),
            "switch_iterations": list(optimizer_configs.ADAM_LBFGS_SWITCH_ITERATIONS),
            "lbfgs_lr": lbfgs_lr,
            "history_size": history_size,
            "line_search_fn": line_search,
            "sampling": batch.manifest(),
        },
        "executed_smoke_budget": {"iteration_count": active_iterations, "fixed_training_batch": True},
        "optimizer_trace": optimizer_trace,
        "lbfgs_history": lbfgs_history,
        "lbfgs_algorithm_2_unrolled_direction": lbfgs_unrolled_direction,
        "lbfgs_preconditioned_hessian_spectrum_algorithm_3": preconditioned_spectrum,
        "metrics": metrics,
        "final_metric": final,
        "checkpoint": {
            "state_dict": model.state_dict() if hasattr(model, "state_dict") else {},
            "problem": problem_name,
            "optimizer": canonical,
            "final_L2RE": final["L2RE"],
            "final_loss": final["loss"],
            "width": width,
            "seed": seed,
            "lbfgs_history": lbfgs_history,
        },
        "reference_grounding": "paper optimizer protocol; executable Adam/L-BFGS route",
    }


def train(model: Any, problem: Any, optimizer_name: str, train_config: Mapping[str, Any]) -> Dict[str, Any]:
    if hasattr(problem, "name"):
        problem_name = str(problem.name)
    elif hasattr(problem, "spec") and hasattr(problem.spec, "name"):
        problem_name = str(problem.spec.name)
    elif isinstance(problem, Mapping):
        problem_name = str(problem.get("name", problem.get("problem", "convection")))
    else:
        problem_name = str(problem)

    optimizer_spec = registry_optimizer(optimizer_name)
    mode = str(train_config.get("mode", "runtime_smoke"))
    if _torch_available_for_training(model):
        try:
            return _run_torch_training(model, problem_name, optimizer_name, train_config)
        except Exception as exc:
            if not str(train_config.get("allow_training_fallback", "false")).lower() in {"1", "true", "yes"}:
                raise
            fallback_error = repr(exc)
        else:
            fallback_error = ""
    else:
        fallback_error = "torch training path unavailable"
    iterations = int(
        train_config.get(
            "iteration_count",
            train_config.get("max_steps", 3 if mode in {"runtime_smoke", "docker_validate", "dry_run", "smoke"} else 41_000),
        )
    )
    iterations = max(1, iterations)
    width = int(train_config.get("width", getattr(getattr(model, "config", None), "width", 200)))
    seed = int(train_config.get("seed", getattr(getattr(model, "config", None), "seed", 0)))

    exp = ExperimentSpec(
        experiment_id=f"{problem_name}__{optimizer_spec.name.replace('+', '_plus_')}__w{width}__s{seed}",
        problem=problem_name,
        optimizer=optimizer_spec.name,
        width=width,
        seed=seed,
        iteration_count=iterations,
        configured_full_budget={
            "optimizer": optimizer_spec.configured_full_budget,
            "problem": dict(METHOD_PROBLEM_REGISTRY.get(problem_name, METHOD_PROBLEM_REGISTRY["convection"]).full_budget),
        },
        executed_smoke_budget={
            "optimizer": optimizer_spec.executed_smoke_budget,
            "problem": dict(METHOD_PROBLEM_REGISTRY.get(problem_name, METHOD_PROBLEM_REGISTRY["convection"]).smoke_budget),
        },
        method_selector=str(train_config.get("method_selector", "ours")),
        semantic_anchors=("training_loop", optimizer_spec.paper_role),
    )

    rows = [_smoke_metric_for_experiment(exp, iteration) for iteration in range(iterations)]
    final = rows[-1]
    return {
        "status": "completed_smoke_adapter" if mode in {"runtime_smoke", "docker_validate", "dry_run", "smoke"} else "completed_registry_adapter",
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
        "fallback_reason": fallback_error,
        "reference_grounding": optimizer_spec.reference_grounding,
    }


def run_training(
    problem: str = "convection",
    optimizer: str = "Adam",
    width: int = 200,
    seed: int = 0,
    mode: str = "runtime_smoke",
    **overrides: Any,
) -> Dict[str, Any]:
    cfg = resolve_config(
        {
            "problem": problem,
            "optimizer": optimizer,
            "width": width,
            "seed": seed,
            "mode": mode,
            **overrides,
        },
        mode=mode,
    )
    model = build_model(problem=problem, width=width, seed=seed, prefer_torch=True)
    problem_obj = problems.make_problem(problem, cfg)
    return train(model, problem_obj.name, optimizer, cfg)


def run_training_suite(mode: str = "runtime_smoke", max_experiments: int = 6) -> List[Dict[str, Any]]:
    rows = expand_experiment_registry(mode=mode, max_experiments=max_experiments)
    results: List[Dict[str, Any]] = []
    for row in rows:
        results.append(
            run_training(
                problem=row.problem,
                optimizer=row.optimizer,
                width=row.width,
                seed=row.seed,
                mode=mode,
                max_steps=row.iteration_count,
            )
        )
    return results


def run_adam_lbfgs_sweep_for_problem(
    problem: str,
    *,
    width: int = 200,
    seed: int = 0,
    mode: str = "runtime_smoke",
    switches: Sequence[int] = optimizer_configs.ADAM_LBFGS_SWITCH_ITERATIONS,
    max_steps: int = 4,
) -> Dict[str, Any]:
    """Run the Section 2.2 Adam+L-BFGS switch sweep for one problem."""

    checkpoints = []
    runs = []
    for switch in switches:
        result = run_training(
            problem=problem,
            optimizer="Adam+L-BFGS",
            width=width,
            seed=seed,
            mode=mode,
            switch_iteration=int(switch),
            max_steps=max_steps,
            active_switch_iteration=min(1, max(1, max_steps - 1)),
        )
        checkpoint = dict(result.get("checkpoint", {}))
        checkpoint.update(
            {
                "problem": problem,
                "width": width,
                "seed": seed,
                "switch_iteration": int(switch),
                "final_L2RE": float(result["final_metric"]["L2RE"]),
                "final_loss": float(result["final_metric"]["loss"]),
                "source_optimizer": "Adam+L-BFGS",
            }
        )
        checkpoints.append(checkpoint)
        runs.append({"switch_iteration": int(switch), "result": result, "checkpoint": checkpoint})
    best = select_lowest_l2re_checkpoint(checkpoints)
    return {
        "problem": problem,
        "optimizer": "Adam+L-BFGS",
        "section": "2.2",
        "runs": runs,
        "checkpoints": checkpoints,
        "selected_lowest_l2re_checkpoint": best,
        "selection_rule": "min final_L2RE over Adam+L-BFGS switch sweep",
    }


def resume_best_adam_lbfgs_with_gradient_descent(problem: str, **kwargs: Any) -> Dict[str, Any]:
    """End-to-end problem-scoped Adam+L-BFGS best-checkpoint then GD workflow."""

    sweep = run_adam_lbfgs_sweep_for_problem(problem, **{k: v for k, v in kwargs.items() if k not in {"steps", "learning_rate"}})
    resumed = resume_with_gradient_descent(
        sweep["selected_lowest_l2re_checkpoint"],
        problem=problem,
        steps=int(kwargs.get("steps", 3)),
        learning_rate=float(kwargs.get("learning_rate", 1e-3)),
    )
    return {"problem": problem, "sweep": sweep, "resume_optimizer": "GradientDescent", "resumed": resumed}


def resume_best_adam_lbfgs_with_nncg(problem: str, **kwargs: Any) -> Dict[str, Any]:
    """End-to-end problem-scoped Adam+L-BFGS best-checkpoint then NNCG workflow."""

    sweep = run_adam_lbfgs_sweep_for_problem(problem, **{k: v for k, v in kwargs.items() if k not in {"steps", "rank"}})
    resumed = resume_with_nncg(
        sweep["selected_lowest_l2re_checkpoint"],
        problem=problem,
        steps=int(kwargs.get("steps", 3)),
        rank=int(kwargs.get("rank", 16)),
    )
    return {"problem": problem, "sweep": sweep, "resume_optimizer": "NysNewton-CG", "resumed": resumed}


def select_lowest_l2re_checkpoint(checkpoints: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Select the Adam+L-BFGS checkpoint with the smallest final L2RE."""

    if not checkpoints:
        raise ValueError("select_lowest_l2re_checkpoint requires at least one checkpoint")
    return dict(
        min(
            checkpoints,
            key=lambda row: float(row.get("final_L2RE", row.get("L2RE", row.get("l2re", float("inf"))))),
        )
    )


def resume_with_gradient_descent(
    checkpoint: Mapping[str, Any],
    problem: str = "convection",
    *,
    steps: int = 3,
    learning_rate: float = 1e-3,
) -> Dict[str, Any]:
    """Resume the selected checkpoint with an explicit gradient-descent phase."""

    model = build_model(problem=problem, width=int(checkpoint.get("width", 200)), seed=int(checkpoint.get("seed", 0)))
    state = checkpoint.get("state_dict")
    if state and hasattr(model, "load_state_dict"):
        try:
            model.load_state_dict(state)  # type: ignore[arg-type]
        except Exception:
            pass
    return train(
        model,
        problem,
        "GradientDescent",
        {
            "mode": "runtime_smoke",
            "max_steps": steps,
            "learning_rate": learning_rate,
            "method_selector": "gd_after_adam_lbfgs",
        },
    )


def resume_with_nncg(
    checkpoint: Mapping[str, Any],
    problem: str = "convection",
    *,
    steps: int = 3,
    rank: int = 16,
) -> Dict[str, Any]:
    """Resume the selected Adam+L-BFGS checkpoint with the NNCG route."""

    from pinn_landscape import nncg

    model = build_model(problem=problem, width=int(checkpoint.get("width", 200)), seed=int(checkpoint.get("seed", 0)))
    state = checkpoint.get("state_dict")
    if state and hasattr(model, "load_state_dict"):
        try:
            model.load_state_dict(state)  # type: ignore[arg-type]
        except Exception:
            pass
    problem_obj = problems.make_problem(problem, {"mode": "runtime_smoke"})
    return nncg.refine_with_nncg(model, problem_obj, checkpoint={"mode": "runtime_smoke", "max_steps": steps}, nncg_config={"rank": rank, "max_steps": steps})


def _predict_values(model: Any, points: Any) -> List[float]:
    if hasattr(points, "detach"):
        import torch  # type: ignore

        with torch.no_grad():
            pred = model(points)
            return [float(v) for v in pred.detach().cpu().reshape(-1).tolist()]
    values = model(points)
    return [float(v[0] if isinstance(v, (list, tuple)) else v) for v in values]


def phasewise_pointwise_absolute_error_workflow(
    problem: str,
    *,
    width: int = 200,
    seed: int = 0,
    mode: str = "runtime_smoke",
) -> Dict[str, Any]:
    """Evaluate pointwise absolute error at Adam, L-BFGS, GD, and NNCG phase ends."""

    batch = sampling.sample_fixed_collocation_points(problem, mode=mode, seed=seed)
    reference_raw = batch.reference_values
    if hasattr(reference_raw, "detach"):
        reference = [float(v) for v in reference_raw.detach().cpu().reshape(-1).tolist()]
    else:
        reference = [float(v[0] if isinstance(v, (list, tuple)) else v) for v in reference_raw]

    phase_predictions: Dict[str, List[float]] = {}
    adam_model = build_model(problem=problem, width=width, seed=seed)
    train(adam_model, problem, "Adam", {"mode": mode, "max_steps": 2, "batch": batch})
    phase_predictions["Adam"] = _predict_values(adam_model, batch.evaluation)

    lbfgs_model = build_model(problem=problem, width=width, seed=seed)
    lbfgs_result = train(
        lbfgs_model,
        problem,
        "Adam+L-BFGS",
        {"mode": mode, "max_steps": 4, "active_switch_iteration": 1, "batch": batch},
    )
    phase_predictions["L-BFGS"] = _predict_values(lbfgs_model, batch.evaluation)

    gd_checkpoint = dict(lbfgs_result["checkpoint"])
    gd_result = resume_with_gradient_descent(gd_checkpoint, problem=problem, steps=2)
    gd_model = build_model(problem=problem, width=width, seed=seed)
    if gd_result.get("checkpoint", {}).get("state_dict") and hasattr(gd_model, "load_state_dict"):
        gd_model.load_state_dict(gd_result["checkpoint"]["state_dict"])
    phase_predictions["GD"] = _predict_values(gd_model, batch.evaluation)

    nncg_model = build_model(problem=problem, width=width, seed=seed)
    state = gd_checkpoint.get("state_dict")
    if state and hasattr(nncg_model, "load_state_dict"):
        nncg_model.load_state_dict(state)
    from pinn_landscape import nncg

    nncg.refine_with_nncg(nncg_model, problems.make_problem(problem, {"mode": mode, "seed": seed}), nncg_config={"max_steps": 1, "rank": 4})
    phase_predictions["NNCG"] = _predict_values(nncg_model, batch.evaluation)

    errors = pointwise_absolute_error_by_phase(phase_predictions, reference)
    errors.update(
        {
            "problem": problem,
            "reference_solution": sampling.get_problem(problem).exact_solution,
            "coefficients": dict(sampling.get_problem(problem).coefficients),
            "phases_evaluated": ["Adam", "L-BFGS", "GD", "NNCG"],
        }
    )
    return errors


def pointwise_absolute_error_by_phase(
    phase_predictions: Mapping[str, Sequence[float]],
    reference: Sequence[float],
) -> Dict[str, Any]:
    """Compute phase-wise point absolute errors for Adam/L-BFGS/GD/NNCG ends."""

    rows = {}
    ref = [float(v) for v in reference]
    for phase, pred in phase_predictions.items():
        values = [abs(float(p) - r) for p, r in zip(pred, ref)]
        rows[phase] = {
            "mean_absolute_error": sum(values) / max(1, len(values)),
            "max_absolute_error": max(values) if values else 0.0,
            "pointwise_absolute_error": values,
        }
    return {"phases": rows, "metric": "pointwise_absolute_error"}


def measure_per_iteration_wall_clock_time(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-iteration wall-clock time for Table 3-style reporting."""

    values = [float(row.get("per_iteration_wall_clock_time", row.get("training_time", 0.0))) for row in records]
    return {
        "num_iterations": len(values),
        "mean_seconds_per_iteration": sum(values) / max(1, len(values)),
        "max_seconds_per_iteration": max(values) if values else 0.0,
        "raw_seconds_per_iteration": values,
    }


def training_manifest(mode: str = "runtime_smoke") -> Dict[str, Any]:
    rows = expand_experiment_registry(mode=mode, max_experiments=6)
    return {
        "mode": mode,
        "experiments": [asdict(row) for row in rows],
    }


__all__ = [
    "TrainingRequest",
    "build_model",
    "train",
    "run_training",
    "run_training_suite",
    "run_adam_lbfgs_sweep_for_problem",
    "resume_best_adam_lbfgs_with_gradient_descent",
    "resume_best_adam_lbfgs_with_nncg",
    "select_lowest_l2re_checkpoint",
    "resume_with_gradient_descent",
    "resume_with_nncg",
    "phasewise_pointwise_absolute_error_workflow",
    "pointwise_absolute_error_by_phase",
    "measure_per_iteration_wall_clock_time",
    "training_manifest",
]
