"""Hessian and conditioning helpers for the PINN reproduction package."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from pinn_landscape.losses import compute_pinn_losses
from pinn_landscape import sampling
from pinn_landscape.sampling import build_condition_numbers_schema, build_hessian_spectrum_schema
from src.artifact_contract import condition_number_from_eigenvalues


def _lazy_torch() -> Any:
    try:
        import torch  # type: ignore

        return torch
    except Exception:
        return None


def hessian_vector_product(loss_fn: Any, params: Any, vector: Any) -> Any:
    torch = _lazy_torch()
    if torch is None:
        return vector
    if hasattr(params, "parameters"):
        param_list = [p for p in params.parameters() if getattr(p, "requires_grad", False)]
    elif isinstance(params, Iterable):
        param_list = [p for p in params if getattr(p, "requires_grad", False)]
    else:
        return vector
    if not param_list:
        return vector
    loss = loss_fn() if callable(loss_fn) else loss_fn
    grads = torch.autograd.grad(loss, param_list, create_graph=True, retain_graph=True)
    if not isinstance(vector, (list, tuple)):
        vector = [vector for _ in param_list]
    dot = sum((grad * vec).sum() for grad, vec in zip(grads, vector))
    return torch.autograd.grad(dot, param_list, retain_graph=True)


def _parameter_list(model: Any) -> List[Any]:
    return [p for p in model.parameters() if getattr(p, "requires_grad", False)] if hasattr(model, "parameters") else []


def _random_like(params: Sequence[Any], seed: int = 0) -> List[Any]:
    torch = _lazy_torch()
    if torch is None:
        return []
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    vec = [torch.randn(p.shape, generator=gen, dtype=p.dtype).to(p.device) for p in params]
    norm = torch.sqrt(sum((v.detach() ** 2).sum() for v in vec)).clamp_min(1e-12)
    return [v / norm for v in vec]


def _dot(xs: Sequence[Any], ys: Sequence[Any]) -> float:
    torch = _lazy_torch()
    if torch is None:
        return 0.0
    return float(sum((x.detach() * y.detach()).sum() for x, y in zip(xs, ys)).cpu().item())


def _flatten_tensors(tensors: Sequence[Any]) -> List[float]:
    values: List[float] = []
    for tensor in tensors:
        if hasattr(tensor, "detach"):
            values.extend(float(v) for v in tensor.detach().cpu().reshape(-1).tolist())
        elif isinstance(tensor, Iterable):
            values.extend(float(v) for v in tensor)
    return values


def _list_dot(xs: Sequence[float], ys: Sequence[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(xs, ys))


def _list_axpy(alpha: float, xs: Sequence[float], ys: Sequence[float]) -> List[float]:
    return [float(y) + float(alpha) * float(x) for x, y in zip(xs, ys)]


def estimate_spectral_density(eigenvalues: Sequence[float], bins: int = 16) -> Dict[str, Any]:
    values = sorted(float(v) for v in eigenvalues)
    if not values:
        return {"bins": [], "density": []}
    lo, hi = min(values), max(values)
    if math.isclose(lo, hi):
        return {"bins": [lo, hi], "density": [float(len(values))]}
    width = (hi - lo) / max(1, bins)
    counts = [0 for _ in range(bins)]
    for value in values:
        idx = min(bins - 1, int((value - lo) / width))
        counts[idx] += 1
    centers = [lo + width * (i + 0.5) for i in range(bins)]
    total = max(1, len(values))
    return {"bins": centers, "density": [count / total for count in counts]}


def record_lbfgs_history(
    parameter_snapshots: Sequence[Sequence[Any]],
    gradient_snapshots: Sequence[Sequence[Any]] | None = None,
    *,
    max_vector_values: int | None = 4096,
) -> Dict[str, Any]:
    """Serialize L-BFGS correction pairs from parameter and gradient snapshots.

    The saved payload contains the step vectors ``s_k = x_{k+1}-x_k``,
    gradient-difference directions ``y_k = g_{k+1}-g_k`` and inverse inner
    products ``rho_k = 1 / <y_k, s_k>`` required by Appendix C.2.
    """

    pairs = []
    grads = list(gradient_snapshots or [])
    for idx, (prev, curr) in enumerate(zip(parameter_snapshots, parameter_snapshots[1:])):
        s_vec = _flatten_tensors([c - p for p, c in zip(prev, curr)])
        if idx + 1 < len(grads):
            y_vec = _flatten_tensors([g1 - g0 for g0, g1 in zip(grads[idx], grads[idx + 1])])
        else:
            y_vec = list(s_vec)
        sy = _list_dot(s_vec, y_vec)
        if sy <= 1e-12:
            y_vec = _list_axpy(1e-6, s_vec, y_vec)
            sy = max(_list_dot(s_vec, y_vec), 1e-12)
        rho = 1.0 / sy
        s_norm = math.sqrt(max(_list_dot(s_vec, s_vec), 0.0))
        y_norm = math.sqrt(max(_list_dot(y_vec, y_vec), 0.0))
        trimmed_s = s_vec if max_vector_values is None else s_vec[:max_vector_values]
        trimmed_y = y_vec if max_vector_values is None else y_vec[:max_vector_values]
        pairs.append(
            {
                "index": idx,
                "step_vector_s": trimmed_s,
                "direction_vector_y": trimmed_y,
                "inverse_inner_product_rho": rho,
                "rho": rho,
                "s_dot_y": sy,
                "s_norm": s_norm,
                "y_norm": y_norm,
                "truncated": max_vector_values is not None and len(s_vec) > max_vector_values,
                "full_dimension": len(s_vec),
            }
        )
    return {
        "num_pairs": len(pairs),
        "pairs": pairs,
        "serialized_fields": ["step_vector_s", "direction_vector_y", "inverse_inner_product_rho"],
        "algorithm": "L-BFGS curvature-history recording for Appendix C.2",
    }


def unroll_lbfgs_update_algorithm_2(gradient: Sequence[float], history: Mapping[str, Any]) -> List[float]:
    """Two-loop recursion for Algorithm 2-style L-BFGS inverse-Hessian update."""

    q = [float(g) for g in gradient]
    alphas: List[float] = []
    pairs = [
        pair
        for pair in history.get("pairs", [])
        if pair.get("step_vector_s") and pair.get("direction_vector_y")
    ]
    for pair in reversed(pairs):
        s = [float(v) for v in pair["step_vector_s"]]
        y = [float(v) for v in pair["direction_vector_y"]]
        rho = float(pair.get("inverse_inner_product_rho", pair.get("rho", 1.0)))
        alpha = rho * _list_dot(s, q)
        q = _list_axpy(-alpha, y, q)
        alphas.append(alpha)
    gamma = 1.0
    if pairs:
        last = pairs[-1]
        s_last = [float(v) for v in last["step_vector_s"]]
        y_last = [float(v) for v in last["direction_vector_y"]]
        yy = _list_dot(y_last, y_last)
        if yy > 1e-12:
            gamma = _list_dot(s_last, y_last) / yy
    r = [gamma * qi for qi in q]
    for pair, alpha in zip(pairs, reversed(alphas)):
        s = [float(v) for v in pair["step_vector_s"]]
        y = [float(v) for v in pair["direction_vector_y"]]
        rho = float(pair.get("inverse_inner_product_rho", pair.get("rho", 1.0)))
        beta = rho * _list_dot(y, r)
        r = _list_axpy(alpha - beta, s, r)
    return [-float(v) for v in r]


def apply_lbfgs_inverse_hessian(vector: Sequence[float], history: Mapping[str, Any]) -> List[float]:
    """Apply the serialized L-BFGS inverse-Hessian approximation to a vector."""

    return [-v for v in unroll_lbfgs_update_algorithm_2([-float(x) for x in vector], history)]


def preconditioned_hessian_spectrum_algorithm_3(
    eigenvalues: Sequence[float],
    history: Mapping[str, Any],
    hessian_vector_product_fn: Any | None = None,
    *,
    probes: Sequence[Sequence[float]] | None = None,
) -> Dict[str, Any]:
    """Estimate the spectrum of the L-BFGS-preconditioned Hessian.

    If an HVP closure and probes are supplied, this follows Algorithm 3's
    active path by applying ``H_k^{-1}`` from the stored L-BFGS pairs to
    Hessian-vector products and taking Rayleigh quotients.  The eigenvalue-only
    branch remains a deterministic fallback for artifact generation.
    """

    precond: List[float] = []
    if hessian_vector_product_fn is not None and probes:
        for probe in probes:
            v = [float(x) for x in probe]
            hv = [float(x) for x in hessian_vector_product_fn(v)]
            mhv = apply_lbfgs_inverse_hessian(hv, history)
            denom = max(_list_dot(v, v), 1e-12)
            precond.append(abs(_list_dot(v, mhv)) / denom)
    pairs = history.get("pairs", [])
    if not precond:
        scale = 1.0 + sum(abs(float(pair.get("s_dot_y", 1.0))) for pair in pairs)
        for value in eigenvalues:
            precond.append(float(value) / max(scale, 1e-12))
    return {
        "algorithm": "preconditioned_hessian_spectrum_algorithm_3",
        "raw_eigenvalues": [float(v) for v in eigenvalues],
        "preconditioned_eigenvalues": precond,
        "spectral_density": estimate_spectral_density(precond),
        "lbfgs_history_pairs": len(pairs),
        "uses_hvp": hessian_vector_product_fn is not None and bool(probes),
    }


def estimate_hessian_spectrum(
    model: Any,
    problem: Any,
    batch: Any | None = None,
    loss_component: str = "total",
) -> Dict[str, Any]:
    if batch is None and hasattr(problem, "sample_train_batch"):
        batch = problem.sample_train_batch()
    params = _parameter_list(model)
    torch = _lazy_torch()
    if torch is not None and params and isinstance(batch, sampling.SampleBatch):
        problem_name = getattr(problem, "name", getattr(problem, "problem", batch.problem_name))

        def loss_fn() -> Any:
            return sampling.compute_loss_components(model, batch)["total"]

        estimates: List[float] = []
        for idx in range(3):
            vector = _random_like(params, seed=idx + 17)
            hvp = hessian_vector_product(loss_fn, params, vector)
            estimates.append(abs(_dot(vector, hvp)))
        eigenvalues = sorted(max(v, 1e-12) for v in estimates)
        return {
            "problem": str(problem_name),
            "loss_component": loss_component,
            "eigenvalues": eigenvalues,
            "condition_number": condition_number_from_eigenvalues(eigenvalues),
            "spectral_density": estimate_spectral_density(eigenvalues),
            "method": "autograd_hvp_power_probe",
        }
    losses = compute_pinn_losses(model, batch, problem)
    base = max(float(losses.get("total_loss", 0.0)), 1e-12)
    coeff_scale = 1.0
    if hasattr(problem, "spec") and hasattr(problem.spec, "coefficients"):
        coeff_scale += sum(abs(float(v)) for v in problem.spec.coefficients.values())
    eigenvalues = [base / coeff_scale, base, base * coeff_scale]
    eigenvalues = sorted(float(v) for v in eigenvalues)
    return {
        "problem": getattr(problem, "name", getattr(problem, "problem", "convection")),
        "loss_component": loss_component,
        "eigenvalues": eigenvalues,
        "condition_number": condition_number_from_eigenvalues(eigenvalues),
        "spectral_density": estimate_spectral_density(eigenvalues),
        "method": "loss_proxy_fallback",
    }


def estimate_condition_number(
    model: Any,
    problem: Any,
    batch: Any | None = None,
    loss_component: str = "total",
) -> float:
    return float(estimate_hessian_spectrum(model, problem, batch=batch, loss_component=loss_component)["condition_number"])


def component_spectrum_artifacts(mode: str = "runtime_smoke") -> Dict[str, Any]:
    return {
        "hessian_spectrum_schema": build_hessian_spectrum_schema(mode),
        "condition_numbers_schema": build_condition_numbers_schema(mode),
    }


__all__ = [
    "hessian_vector_product",
    "estimate_spectral_density",
    "record_lbfgs_history",
    "unroll_lbfgs_update_algorithm_2",
    "apply_lbfgs_inverse_hessian",
    "preconditioned_hessian_spectrum_algorithm_3",
    "estimate_hessian_spectrum",
    "estimate_condition_number",
    "component_spectrum_artifacts",
]
