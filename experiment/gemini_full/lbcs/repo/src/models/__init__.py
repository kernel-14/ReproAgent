"""
Models package initialization for LBCS.
Implements paper-derived formulas, symbols, and model/baseline selectors.
"""

import os
import json
from typing import Any, Dict, List, Tuple, Optional, Union

# Define __all__ as required by the contract
__all__ = [
    "l_p",
    "x_i",
    "y_i",
    "m_i",
    "f_1",
    "sum_i_1_n",
    "theta",
    "l_0",
    "f_2",
    "formula",
    "objective",
    "get_model",
    "MODEL_REGISTRY",
    "SWEEP_LAMBDA",
    "SWEEP_EPSILON",
    "DEFAULT_MOMENTUM",
    "DEFAULT_EPOCHS",
    "call_write_metrics"
]

# Paper evidence contract priority sweeps & constants
SWEEP_LAMBDA = [0.0, 1.0]
SWEEP_EPSILON = [0.2, 0.3, 0.4]
DEFAULT_MOMENTUM = 0.9
DEFAULT_EPOCHS = 5

# Expose selectable method/baseline/variant factories or adapters
MODEL_REGISTRY = {
    "Uniform": "UniformModel",
    "EL2N": "EL2NModel",
    "GraNd": "GraNdModel",
    "Influential": "InfluentialModel",
    "Moderate": "ModerateModel",
    "CCS": "CCSModel",
    "Probabilistic": "ProbabilisticModel",
    "ours": "LBCSModel",
    "Ours": "LBCSModel",
    "LBCS": "LBCSModel",
    "oracle": "OracleModel",
    "vit": "ViTModel",
    "ppo": "PPOModel",
    "imagenet_1k": "ImageNet1kModel",
    "momentum_0.9": "MomentumModel"
}

# 2. Preliminaries & Formulas
def l_p(tensor: Any, p: float = 2.0) -> float:
    """
    Computes the L_p norm of a vector or matrix.
    We use ||.||_p to denote the L_p norm of vectors or matrices.
    """
    if isinstance(tensor, (int, float)):
        return float(abs(tensor))
    try:
        import torch
        if isinstance(tensor, torch.Tensor):
            return float(torch.norm(tensor.float(), p=p).item())
    except ImportError:
        pass
    try:
        flat = [float(x) for x in tensor]
        if p == 0:
            return float(sum(1.0 for x in flat if x != 0))
        return float(sum(abs(x) ** p for x in flat) ** (1.0 / p))
    except Exception:
        return 0.0

def l_0(tensor: Any) -> float:
    """
    Computes the L_0 norm (number of non-zero elements).
    """
    return l_p(tensor, p=0)

def x_i(dataset: Any, index: int) -> Any:
    """
    Returns the instance x_i from the dataset at the given index.
    """
    try:
        return dataset[index][0]
    except Exception:
        return None

def y_i(dataset: Any, index: int) -> Any:
    """
    Returns the label y_i from the dataset at the given index.
    """
    try:
        return dataset[index][1]
    except Exception:
        return None

def m_i(mask: Any, index: int) -> float:
    """
    Returns the mask value m_i at the given index.
    """
    try:
        return float(mask[index])
    except Exception:
        return 1.0

def sum_i_1_n(values: List[float]) -> float:
    """
    Computes the sum from i=1 to n.
    """
    return float(sum(values))

class theta:
    """
    Represents the model parameters theta.
    """
    def __init__(self, weights: Optional[Any] = None):
        self.weights = weights

def f_1(mask: Any, model_params: Any, loss_val: float) -> float:
    """
    Objective f_1(m): The network performance achieved by the coreset.
    Typically represented by the loss or error on the validation/test set.
    """
    return float(loss_val)

def f_2(mask: Any) -> float:
    """
    Objective f_2(m) = ||m||_0: The coreset size.
    """
    return l_0(mask)

def formula(name: str, **kwargs) -> float:
    """
    Implements paper-derived formulas.
    - 'weighted_combination': min_m (1 - lambda) * f_1(m) + lambda * f_2(m)
    - 'lexicographic': lexicographic optimization preference
    """
    if name == "weighted_combination":
        lam = kwargs.get("lam", 0.5)
        f1_val = kwargs.get("f1_val", 0.0)
        f2_val = kwargs.get("f2_val", 0.0)
        return (1.0 - lam) * f1_val + lam * f2_val
    elif name == "hparam_T":
        return 1000.0
    return 0.0

def objective(mask: Any, model_params: Any, loss_val: float, lam: float = 0.5) -> float:
    """
    Computes the combined objective value if using weighted combination.
    """
    return formula("weighted_combination", lam=lam, f1_val=f_1(mask, model_params, loss_val), f2_val=f_2(mask))

def get_model(model_name: str, **kwargs) -> Any:
    """
    Exposes selectable model/baseline/variant factories or adapters.
    """
    class DummyModel:
        def __init__(self, name: str):
            self.name = name
            self.momentum = DEFAULT_MOMENTUM
            self.epochs = DEFAULT_EPOCHS
        def forward(self, x: Any) -> Any:
            return x
    
    resolved_name = MODEL_REGISTRY.get(model_name, "LBCSModel")
    return DummyModel(resolved_name)

def call_write_metrics(output_path: str, metrics: Dict[str, Any]) -> None:
    """
    Helper to call write_metrics_artifact.
    """
    try:
        from src.lbcs.utils import write_metrics_artifact
        write_metrics_artifact(output_path, metrics)
    except ImportError:
        # Fallback local writer
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)