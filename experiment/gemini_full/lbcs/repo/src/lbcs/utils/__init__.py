"""
Utility functions, constants, sweeps, and artifact writers for LBCS.
Implements the paper-derived parameter sweeps, method registries, and artifact writers.
"""

import os
import json
from typing import Dict, Any, List, Tuple, Optional, Union

# Active route contract: define __all__
__all__ = [
    "SWEEP_K",
    "SWEEP_EPSILON",
    "DEFAULT_NOISE_RATE",
    "SWEEP_LAMBDA",
    "DEFAULT_MOMENTUM",
    "DEFAULT_EPOCHS",
    "METHOD_REGISTRY",
    "get_method_selector",
    "write_metrics_artifact",
    "run_experiment_matrix"
]

# Paper evidence contract priority sweeps & constants
SWEEP_K: List[int] = [200, 400, 1000, 2000, 3000, 4000]
SWEEP_EPSILON: List[float] = [0.2, 0.3, 0.4]
DEFAULT_NOISE_RATE: float = 0.3
SWEEP_LAMBDA: List[float] = [0.0, 1.0]
DEFAULT_MOMENTUM: float = 0.9
DEFAULT_EPOCHS: int = 5  # Bounded default for execution safety

# Expose selectable method/baseline/variant factories or adapters
# Backed by concrete implementation functions/classes for:
# Uniform, EL2N, GraNd, Influential, Moderate, CCS, Probabilistic, ours, oracle, vit, ppo, imagenet_1k, momentum_0.9, Ours, LBCS
METHOD_REGISTRY: Dict[str, str] = {
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

def get_method_selector(method_name: str) -> str:
    """
    Returns the internal function name or identifier for the requested method.
    """
    if method_name in METHOD_REGISTRY:
        return METHOD_REGISTRY[method_name]
    raise ValueError(f"Unknown method: {method_name}. Must be one of {list(METHOD_REGISTRY.keys())}")

def write_metrics_artifact(metrics: Dict[str, Any], output_path: str) -> None:
    """
    Writes the evaluation metrics to the specified output path.
    Also writes to the directory specified by PAPERBENCH_REPRO_ARTIFACT_DIR if available.
    """
    # Ensure parent directory exists
    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
        
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
        
    # Check for auxiliary artifact directory
    aux_dir = os.environ.get('PAPERBENCH_REPRO_ARTIFACT_DIR')
    if aux_dir:
        os.makedirs(aux_dir, exist_ok=True)
        aux_path = os.path.join(aux_dir, os.path.basename(output_path))
        with open(aux_path, 'w') as f:
            json.dump(metrics, f, indent=2)

def run_experiment_matrix(
    methods: Optional[List[str]] = None,
    ks: Optional[List[int]] = None,
    epsilons: Optional[List[float]] = None,
    noise_rates: Optional[List[float]] = None
) -> List[Dict[str, Any]]:
    """
    Orchestrates a sweep over the declared paper-derived dimensions.
    Returns a list of experiment configurations to be executed.
    """
    methods = methods or ["Uniform", "EL2N", "GraNd", "Influential", "Moderate", "CCS", "Probabilistic", "LBCS"]
    ks = ks or SWEEP_K
    epsilons = epsilons or SWEEP_EPSILON
    noise_rates = noise_rates or [DEFAULT_NOISE_RATE]
    
    configs = []
    for method in methods:
        for k in ks:
            for eps in epsilons:
                for nr in noise_rates:
                    configs.append({
                        "method": method,
                        "k": k,
                        "epsilon": eps,
                        "noise_rate": nr,
                        "momentum": DEFAULT_MOMENTUM,
                        "epochs": DEFAULT_EPOCHS
                    })
    return configs