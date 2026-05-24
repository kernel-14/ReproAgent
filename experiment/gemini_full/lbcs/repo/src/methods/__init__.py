import os
import json
from typing import Any, Dict, List, Optional, Union

# reference_grounding: chunk_005 2. Preliminaries
# reference_grounding: chunk_008 3.1. Lexicographic Bilevel Coreset Selection
# reference_grounding: chunk_009 3.2. Optimization Algorithm

def l_p(vector: Any, p: Union[int, float] = 2) -> float:
    """
    Computes the L_p norm of vectors or matrices.
    reference_grounding: chunk_005 2. Preliminaries
    """
    import torch
    if not isinstance(vector, torch.Tensor):
        vector = torch.tensor(vector)
    if p == 0:
        return float(torch.count_nonzero(vector))
    return float(torch.norm(vector.float(), p=p))

def x_i(dataset: Any, index: int) -> Any:
    """Represents the instance x_i from dataset D."""
    return dataset[index][0]

def y_i(dataset: Any, index: int) -> Any:
    """Represents the label y_i from dataset D."""
    return dataset[index][1]

def m_i(mask: Any, index: int) -> int:
    """Represents the mask value m_i in {0, 1}."""
    return int(mask[index])

def f_1(accuracy: float, epsilon: float) -> float:
    """
    Primary objective f_1: Performance constraint.
    f_1(m) = max(0, (1 - epsilon) * 100 - accuracy(m))
    reference_grounding: chunk_008 3.1. Lexicographic Bilevel Coreset Selection
    """
    target = (1.0 - epsilon) * 100.0
    return max(0.0, target - accuracy)

def sum_i_1_n(values: Any) -> float:
    """Represents the summation sum_{i=1}^n."""
    import torch
    if not isinstance(values, torch.Tensor):
        values = torch.tensor(values)
    return float(torch.sum(values))

def theta(model: Any) -> Any:
    """Represents the model parameters theta."""
    return model.parameters()

def l_0(mask: Any) -> float:
    """Represents the L_0 norm of the mask (coreset size)."""
    return l_p(mask, p=0)

def f_2(mask: Any) -> float:
    """Secondary objective f_2: Coreset size minimization."""
    return l_0(mask)

def formula(name: str) -> str:
    """Executable anchor for paper formulas."""
    formulas = {
        "lexicographic_optimization": "min_m F(m) = [f_1(m), f_2(m)]",
        "inner_loop": "theta(m) in argmin_theta L(m, theta)",
        "weighted_combination": "min_m (1-lambda)f_1(m) + lambda f_2(m)"
    }
    return formulas.get(name, "Unknown formula")

def objective(name: str) -> Dict[str, Any]:
    """Executable anchor for paper objectives."""
    objectives = {
        "O1": {"id": "f1", "description": "Performance constraint", "priority": "high"},
        "O2": {"id": "f2", "description": "Minimal coreset size", "priority": "low"}
    }
    return objectives.get(name, {})

# Method Registry and Sweeps as required by method_obligations
METHOD_REGISTRY = {
    "Uniform": "select_uniform",
    "EL2N": "select_el2n",
    "GraNd": "select_grand",
    "Influential": "select_influential",
    "Moderate": "select_moderate",
    "CCS": "select_ccs",
    "Probabilistic": "select_probabilistic",
    "ours": "select_lbcs",
    "Ours": "select_lbcs",
    "LBCS": "select_lbcs",
    "oracle": "select_oracle",
    "vit": "select_vit",
    "ppo": "select_ppo",
    "imagenet_1k": "select_imagenet_1k",
    "momentum_0.9": "select_momentum_0.9"
}

# Paper evidence contract priority sweeps
SWEEP_K = [200, 400, 1000, 2000, 3000, 4000]
SWEEP_EPSILON = [0.2, 0.3, 0.4]
SWEEP_LAMBDA = [0, 1]
DEFAULT_NOISE_RATE = 0.3
DEFAULT_MOMENTUM = 0.9

def get_method(method_name: str):
    """Factory to resolve method names to selection functions."""
    if method_name in ["ours", "Ours", "LBCS"]:
        from .lbcs import select_lbcs
        return select_lbcs
    elif method_name in METHOD_REGISTRY:
        from . import baselines
        func_name = METHOD_REGISTRY[method_name]
        return getattr(baselines, func_name)
    else:
        raise ValueError(f"Method {method_name} not found in registry.")

def write_metrics_artifact(metrics: Dict[str, Any], output_path: str):
    """Artifact writer for metrics.json."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)

__all__ = [
    "l_p", "x_i", "y_i", "m_i", "f_1", "sum_i_1_n", "theta", "l_0", "f_2",
    "formula", "objective", "get_method", "METHOD_REGISTRY", "SWEEP_K",
    "SWEEP_EPSILON", "SWEEP_LAMBDA", "DEFAULT_NOISE_RATE", "write_metrics_artifact"
]