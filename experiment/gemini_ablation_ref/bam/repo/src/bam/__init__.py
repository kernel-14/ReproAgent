import os
import json
from typing import Any, Dict, List

# reference_grounding: paper:paper_contract_method_baseline_protocol (chunk_007_01, chunk_008_02, chunk_009_03)
# Methods and Baselines as defined in the paper.
# Method registry exposes BaM, ADVI, and GSM.
BAM_METHOD = "BaM"
ADVI_BASELINE = "ADVI"
GSM_BASELINE = "GSM"

# Selectable method/baseline/variant factories or adapters
METHODS = {
    "ours": BAM_METHOD,
    "BaM": BAM_METHOD,
    "BaM (proposed)": BAM_METHOD,
    "Ours": BAM_METHOD,
}

# Baseline registry
BASELINES = {
    "baseline": ADVI_BASELINE,
    "ADVI": ADVI_BASELINE,
    "ADVI (baseline)": ADVI_BASELINE,
    "GSM": GSM_BASELINE,
    "GSM (baseline)": GSM_BASELINE,
}

# Ablation registry tracks method variants
ABLATIONS = {
    "100_iterations": {"iterations": 100},
    "BaM_lambda_sweep": {"lambda": [0.1, 1.0, 10.0, 100.0]},
}

# reference_grounding: paper:paper_claim_inventory (parameter_sweeps)
# Executable constants and defaults derived from the paper and addendum.
# reference_grounding: addendum:formula_algorithm_contract
PAPER_CONSTANTS = {
    "DEFAULT_LAMBDA": 1.0,
    "DEFAULT_BATCH_SIZE": 4,
    "DEFAULT_BATCH_SIZE_LOW_RANK": 3,  # For D=4 as per addendum
    "DEFAULT_LEARNING_RATE": 1e-3,
    "DEFAULT_ITERATIONS": 100,
    "MAX_ITERATIONS": 500,
    "EPSILON": 1e-5,
    "P_NON_GAUSSIANITY": [0.0, 0.2, 1.0, 1.8], # Skew/Tail parameters (s and tau)
    "DIMENSIONS_D": [4, 16, 64, 256],
    "CIFAR_HIDDEN_CHANNELS": 64, # c_hid
    "CIFAR_LATENT_DIM": 128,
}

# reference_grounding: paper:paper_contract_method_baseline_protocol (chunk_007_01)
# Algorithm 1 and Section 3.1 anchors for executable code/config
ALGORITHM_ANCHORS = {
    "B_min": 1,
    "B_max": 4,
    "lambda_min": 0.1,
    "lambda_max": 100.0,
    "p_min": 0.0,
    "p_max": 5.0,
    "learning_rate_min": 1e-5,
    "learning_rate_max": 1e-1,
    "numeric_defaults_3_1": [1, 2, 0, 5],
    "numeric_defaults_C_3": [1, 0, 95],
}

# Required parameter sweeps as executable constants used by train/evaluate/report routes
DEFAULT_SWEEPS = {
    "lambda": [0.1, 1.0, 10.0, 100.0],
    "p": [0.0, 0.2, 1.0, 1.8],
    "learning_rate": [1e-4, 1e-3, 1e-2],
    "batch_size": [1, 4, 16, 64],
    "D": [4, 16, 64, 256],
}

def make_method(config: Dict[str, Any]) -> Any:
    """
    Factory to create a method or baseline based on config.
    Lazy imports are used to keep the package importable in minimal environments.
    """
    method_key = config.get("method", "BaM")
    
    # Resolve method name from registry
    if method_key in METHODS:
        from .core.algorithm import BaM
        return BaM(config)
    elif method_key in BASELINES:
        baseline_type = BASELINES[method_key]
        if baseline_type == ADVI_BASELINE:
            from .baselines.advi import ADVI
            return ADVI(config)
        elif baseline_type == GSM_BASELINE:
            from .baselines.gsm import GSM
            return GSM(config)
    
    raise ValueError(f"Unknown method or baseline: {method_key}")

def write_method_registry_artifact(output_path: str = None):
    """Writes the method registry to a JSON file for reproduction verification."""
    if output_path is None:
        base_dir = os.environ.get('PAPERBENCH_REPRO_ARTIFACT_DIR', 'results')
        output_path = os.path.join(base_dir, "method_registry.json")
    
    registry = {
        "methods": METHODS,
        "baselines": BASELINES,
        "constants": PAPER_CONSTANTS,
        "anchors": ALGORITHM_ANCHORS
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(registry, f, indent=2)

def write_ablation_registry_artifact(output_path: str = None):
    """Writes the ablation registry to a JSON file for reproduction verification."""
    if output_path is None:
        base_dir = os.environ.get('PAPERBENCH_REPRO_ARTIFACT_DIR', 'results')
        output_path = os.path.join(base_dir, "ablation_registry.json")
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(ABLATIONS, f, indent=2)

__all__ = [
    "METHODS",
    "BASELINES",
    "ABLATIONS",
    "PAPER_CONSTANTS",
    "ALGORITHM_ANCHORS",
    "DEFAULT_SWEEPS",
    "make_method",
    "write_method_registry_artifact",
    "write_ablation_registry_artifact"
]