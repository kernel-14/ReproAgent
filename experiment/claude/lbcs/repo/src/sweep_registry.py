"""
Sweep registry for Refined Coreset Selection experiments.

Exposes bounded parameter sweeps as config/registry values for all experiments
corresponding to paper tables and figures. Does not implement exhaustive
execution loops; provides configuration surfaces for downstream orchestration.

reference_grounding: paperbench_ref_003 train.py
reference_grounding: paperbench_ref_003 selection.py
reference_grounding: paperbench_ref_004 cnn_mnist_probability_1step_pixel_shared_rein.py
reference_grounding: paperbench_ref_004 noisy_label.py
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

# ============================================================================
# Paper Artifact Mapping
# Paper evidence contract: preserve table/figure captions, named baselines,
# comparison semantics, and output mapping
# ============================================================================

PAPER_ARTIFACTS = {
    "figure_1": {
        "caption": "Illustrations of phenomena of several trivial solutions discussed in §2.1",
        "experiment": "outer_iteration_convergence",
        "output_path": "results/figures/figure_1.png",
        "subfigures": ["f1_vs_iterations", "f2_vs_iterations"],
        "setup": "Appendix C.3",
        "dataset": "cifar10",
        "lambda_value": 0.5,
    },
    "table_1": {
        "caption": "Results (mean ± std.) to illustrate the utility of our method in optimizing the objectives f1(m) and f2(m)",
        "experiment": "preliminary_lbcs_superiority",
        "output_path": "results/tables/table_1.csv",
        "metrics": ["f1_initial", "f1_achieved", "f2_initial", "f2_achieved"],
    },
    "table_2": {
        "caption": "Mean and standard deviation of test accuracy (%) on different benchmarks with various predefined coreset sizes",
        "experiment": "baseline_comparison",
        "output_path": "results/tables/table_2.csv",
        "datasets": ["cifar10", "cifar100", "fmnist"],
        "baselines": ["Uniform", "EL2N", "GraNd", "Influential", "Moderate", "CCS", "Probabilistic", "LBCS"],
    },
    "table_3": {
        "caption": "Mean and standard deviation of test accuracy (%) on different benchmarks with coreset sizes achieved by the proposed LBCS",
        "experiment": "lbcs_optimized_coreset",
        "output_path": "results/tables/table_3.csv",
        "datasets": ["cifar10", "cifar100", "fmnist"],
    },
    "figure_2": {
        "caption": "Illustrations of coreset selection under imperfect supervision",
        "experiment": "robustness_noisy_labels",
        "output_path": "results/figures/figure_2.png",
        "subfigures": ["noisy_30_percent", "class_imbalanced"],
        "dataset": "fmnist",
        "noise_rate": 0.3,
        "noise_type": "symmetric",
    },
    "table_4": {
        "caption": "Top-5 test accuracy (%) on ImageNet-1k",
        "experiment": "imagenet_evaluation",
        "output_path": "results/tables/table_4.csv",
        "note": "ImageNet-1k experiments excluded per addendum clarifications",
        "excluded": True,
    },
}

# ============================================================================
# Baseline Method Registry
# reference_grounding: paperbench_ref_003 selection.py
# ============================================================================

BASELINE_METHODS = {
    "Uniform": {
        "id": "uniform",
        "name": "Uniform",
        "requires_training": False,
        "description": "Uniform random sampling",
    },
    "EL2N": {
        "id": "el2n",
        "name": "EL2N",
        "requires_training": True,
        "description": "Error L2-Norm based selection",
    },
    "GraNd": {
        "id": "grand",
        "name": "GraNd",
        "requires_training": True,
        "description": "Gradient Norm Distance based selection",
    },
    "Influential": {
        "id": "influential",
        "name": "Influential",
        "requires_training": True,
        "description": "Influence function based selection",
    },
    "Moderate": {
        "id": "moderate",
        "name": "Moderate",
        "requires_training": True,
        "description": "Moderate difficulty score based selection",
    },
    "CCS": {
        "id": "ccs",
        "name": "CCS",
        "requires_training": True,
        "description": "Coverage-based Coreset Selection",
    },
    "Probabilistic": {
        "id": "probabilistic",
        "name": "Probabilistic",
        "requires_training": False,
        "description": "Probabilistic bilevel coreset selection",
    },
    "LBCS": {
        "id": "lbcs",
        "name": "LBCS",
        "requires_training": True,
        "description": "Lexicographic Bilevel Coreset Selection (ours)",
        "is_proposed_method": True,
    },
}

# ============================================================================
# Experiment Sweep Registry
# Paper evidence contract: expose bounded sweep/config entries for lambda,
# epsilon, batch_size, initial_k, coreset_sizes per dataset
# reference_grounding: paperbench_ref_003 train.py
# ============================================================================

@dataclass
class ExperimentSweepConfig:
    """Configuration for a single experiment sweep."""
    experiment_id: str
    experiment_name: str
    datasets: List[str]
    methods: List[str]
    coreset_sizes: Dict[str, List[int]]
    epsilon_values: List[float]
    initial_k_values: List[int]
    lambda_values: List[float]
    batch_sizes: List[int]
    search_times: List[int]
    noise_rates: List[float]
    noise_types: List[str]
    seeds: List[int]
    output_artifact: str

# ============================================================================
# Table 1: Preliminary LBCS superiority
# Epsilon sweep {0.2, 0.3, 0.4}, initial_k {200, 400, 600, 800, 1000}
# ============================================================================

TABLE_1_SWEEP = ExperimentSweepConfig(
    experiment_id="table_1",
    experiment_name="Preliminary LBCS Algorithm Superiority",
    datasets=["cifar10"],
    methods=["LBCS"],
    coreset_sizes={
        "cifar10": [200, 400, 600, 800, 1000],
    },
    epsilon_values=[0.2, 0.3, 0.4],
    initial_k_values=[200, 400, 600, 800, 1000],
    lambda_values=[0.5],
    batch_sizes=[128],
    search_times=[10, 20, 30],
    noise_rates=[0.0],
    noise_types=["none"],
    seeds=[42, 43, 44],
    output_artifact="results/tables/table_1.csv",
)

# ============================================================================
# Table 2: Baseline comparison across datasets
# CIFAR-10 k ∈ {956, 1912, 2868, 3824}
# CIFAR-100 k ∈ {2500, 5000, 7500, 10000}
# F-MNIST k ∈ {1000, 2000, 3000, 4000}
# ============================================================================

TABLE_2_SWEEP = ExperimentSweepConfig(
    experiment_id="table_2",
    experiment_name="Baseline Comparison on Different Benchmarks",
    datasets=["cifar10", "cifar100", "fmnist"],
    methods=["Uniform", "EL2N", "GraNd", "Influential", "Moderate", "CCS", "Probabilistic", "LBCS"],
    coreset_sizes={
        "cifar10": [956, 1912, 2868, 3824],
        "cifar100": [2500, 5000, 7500, 10000],
        "fmnist": [1000, 2000, 3000, 4000],
    },
    epsilon_values=[0.3],
    initial_k_values=[],  # Use coreset_sizes as initial k
    lambda_values=[0.5],
    batch_sizes=[128],
    search_times=[20],
    noise_rates=[0.0],
    noise_types=["none"],
    seeds=[42, 43, 44],
    output_artifact="results/tables/table_2.csv",
)

# ============================================================================
# Table 3: LBCS optimized coreset sizes
# ============================================================================

TABLE_3_SWEEP = ExperimentSweepConfig(
    experiment_id="table_3",
    experiment_name="LBCS with Optimized Coreset Sizes",
    datasets=["cifar10", "cifar100", "fmnist"],
    methods=["LBCS"],
    coreset_sizes={
        "cifar10": [956, 1912, 2868, 3824],
        "cifar100": [2500, 5000, 7500, 10000],
        "fmnist": [1000, 2000, 3000, 4000],
    },
    epsilon_values=[0.3],
    initial_k_values=[],
    lambda_values=[0.5],
    batch_sizes=[128],
    search_times=[20],
    noise_rates=[0.0],
    noise_types=["none"],
    seeds=[42, 43, 44],
    output_artifact="results/tables/table_3.csv",
)

# ============================================================================
# Figure 1: Outer iteration convergence
# Lambda = 0.5 per addendum clarifications
# ============================================================================

FIGURE_1_SWEEP = ExperimentSweepConfig(
    experiment_id="figure_1",
    experiment_name="Outer Iteration Convergence Analysis",
    datasets=["cifar10"],
    methods=["LBCS"],
    coreset_sizes={
        "cifar10": [600],
    },
    epsilon_values=[0.3],
    initial_k_values=[600],
    lambda_values=[0.5],
    batch_sizes=[128],
    search_times=[50],
    noise_rates=[0.0],
    noise_types=["none"],
    seeds=[42],
    output_artifact="results/figures/figure_1.png",
)

# ============================================================================
# Figure 2: Robustness against noisy labels
# 30% symmetric label noise on F-MNIST
# reference_grounding: paperbench_ref_004 noisy_label.py
# ============================================================================

FIGURE_2_SWEEP = ExperimentSweepConfig(
    experiment_id="figure_2",
    experiment_name="Robustness Against Noisy Labels",
    datasets=["fmnist"],
    methods=["Uniform", "Moderate", "LBCS"],
    coreset_sizes={
        "fmnist": [1000, 2000, 3000, 4000],
    },
    epsilon_values=[0.3],
    initial_k_values=[],
    lambda_values=[0.5],
    batch_sizes=[128],
    search_times=[20],
    noise_rates=[0.3],
    noise_types=["symmetric"],
    seeds=[42, 43, 44],
    output_artifact="results/figures/figure_2.png",
)

# ============================================================================
# Master Sweep Registry
# ============================================================================

EXPERIMENT_SWEEP_REGISTRY = {
    "table_1": TABLE_1_SWEEP,
    "table_2": TABLE_2_SWEEP,
    "table_3": TABLE_3_SWEEP,
    "figure_1": FIGURE_1_SWEEP,
    "figure_2": FIGURE_2_SWEEP,
}

# ============================================================================
# Training Configuration Registry
# reference_grounding: paperbench_ref_003 train.py
# ============================================================================

TRAINING_CONFIG = {
    "batch_size_options": [128, 256],
    "learning_rate_options": [0.1, 0.01],
    "epochs_options": [200, 300],
    "optimizer_options": ["sgd", "adam"],
    "scheduler_options": ["cosine", "step"],
    "eval_freq_options": [2000, 5000],
}

# ============================================================================
# LBCS Algorithm Configuration
# ============================================================================

LBCS_CONFIG = {
    "epsilon_range": [0.2, 0.3, 0.4],
    "lambda_range": [0.0, 0.5, 1.0],
    "search_times_range": [5, 10, 20, 30, 50],
    "mask_initialization_methods": ["random", "uniform", "moderate"],
    "inner_loop_epochs": 10,
    "outer_loop_max_iterations": 50,
    "convergence_tolerance": 1e-4,
}

SWEEP_REGISTRY = {
    "epsilon": {"values": [0.2, 0.3, 0.4], "source": "Table 1"},
    "initial_k": {"values": [200, 400, 600, 800, 1000], "source": "Table 1"},
    "lambda_values": {"values": [0.0, 0.5, 1.0], "source": "LBCS objective sweep"},
    "coreset_sizes": {
        "cifar10": [956, 1912, 2868, 3824],
        "cifar100": [2500, 5000, 7500, 10000],
        "fmnist": [1000, 2000, 3000, 4000],
        "svhn": [1000, 2000, 3000, 4000],
    },
    "experiments": EXPERIMENT_SWEEP_REGISTRY,
}

# ============================================================================
# Main entrypoint
# Interface contract: main(config) callable
# ============================================================================

def main(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Main entrypoint for sweep registry.
    
    Returns sweep configurations for requested experiments or all experiments
    if no specific experiment is requested.
    
    Args:
        config: Optional configuration dictionary with keys:
            - experiment_id: str, specific experiment to retrieve
            - mode: str, execution mode (full, runtime_smoke, docker_validate)
    
    Returns:
        Dictionary containing sweep configurations and metadata
    """
    if config is None:
        config = {}
    
    experiment_id = config.get("experiment_id", None)
    mode = config.get("mode", "full")
    
    result = {
        "sweep_registry_version": "1.0.0",
        "mode": mode,
        "paper_artifacts": PAPER_ARTIFACTS,
        "baseline_methods": BASELINE_METHODS,
        "training_config": TRAINING_CONFIG,
        "lbcs_config": LBCS_CONFIG,
    }
    
    if experiment_id:
        if experiment_id in EXPERIMENT_SWEEP_REGISTRY:
            result["experiment_sweep"] = asdict(EXPERIMENT_SWEEP_REGISTRY[experiment_id])
        else:
            result["error"] = f"Unknown experiment_id: {experiment_id}"
            result["available_experiments"] = list(EXPERIMENT_SWEEP_REGISTRY.keys())
    else:
        # Return all experiment sweeps
        result["experiment_sweeps"] = {
            exp_id: asdict(sweep_config)
            for exp_id, sweep_config in EXPERIMENT_SWEEP_REGISTRY.items()
        }
    
    # Add bounded sweep summary
    result["bounded_sweep_summary"] = {
        "epsilon_values": LBCS_CONFIG["epsilon_range"],
        "lambda_values": LBCS_CONFIG["lambda_range"],
        "initial_k_values": TABLE_1_SWEEP.initial_k_values,
        "coreset_sizes_per_dataset": {
            "cifar10": TABLE_2_SWEEP.coreset_sizes["cifar10"],
            "cifar100": TABLE_2_SWEEP.coreset_sizes["cifar100"],
            "fmnist": TABLE_2_SWEEP.coreset_sizes["fmnist"],
        },
        "batch_size_options": TRAINING_CONFIG["batch_size_options"],
        "search_times_range": LBCS_CONFIG["search_times_range"],
    }
    
    return result


def get_experiment_sweep(experiment_id: str) -> Optional[ExperimentSweepConfig]:
    """
    Retrieve experiment sweep configuration by ID.
    
    Args:
        experiment_id: Experiment identifier (e.g., "table_1", "figure_2")
    
    Returns:
        ExperimentSweepConfig or None if not found
    """
    return EXPERIMENT_SWEEP_REGISTRY.get(experiment_id)


def get_baseline_method_config(method_name: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve baseline method configuration by name.
    
    Args:
        method_name: Method name (e.g., "LBCS", "Uniform", "EL2N")
    
    Returns:
        Method configuration dictionary or None if not found
    """
    return BASELINE_METHODS.get(method_name)


def get_coreset_sizes_for_dataset(dataset: str) -> List[int]:
    """
    Get default coreset sizes for a dataset from Table 2 sweep.
    
    Args:
        dataset: Dataset identifier (cifar10, cifar100, fmnist)
    
    Returns:
        List of coreset sizes
    """
    return TABLE_2_SWEEP.coreset_sizes.get(dataset, [])


def get_paper_artifact_info(artifact_id: str) -> Optional[Dict[str, Any]]:
    """
    Get paper artifact metadata (table/figure caption, output path, etc.).
    
    Args:
        artifact_id: Artifact identifier (e.g., "table_1", "figure_2")
    
    Returns:
        Artifact metadata dictionary or None if not found
    """
    return PAPER_ARTIFACTS.get(artifact_id)


# ============================================================================
# Registry validation and utility functions
# ============================================================================

def validate_sweep_registry() -> bool:
    """
    Validate that all sweep configurations are complete and consistent.
    
    Returns:
        True if valid, raises ValueError otherwise
    """
    # Check that all experiments have valid output artifacts
    for exp_id, sweep in EXPERIMENT_SWEEP_REGISTRY.items():
        if not sweep.output_artifact:
            raise ValueError(f"Experiment {exp_id} missing output_artifact")
        
        # Check datasets are valid
        valid_datasets = ["cifar10", "cifar100", "fmnist", "svhn", "imagenet1k"]
        for dataset in sweep.datasets:
            if dataset not in valid_datasets:
                raise ValueError(f"Invalid dataset {dataset} in experiment {exp_id}")
        
        # Check methods are valid
        for method in sweep.methods:
            if method not in BASELINE_METHODS:
                raise ValueError(f"Unknown method {method} in experiment {exp_id}")
    
    return True


def get_all_experiment_ids() -> List[str]:
    """Get list of all registered experiment IDs."""
    return list(EXPERIMENT_SWEEP_REGISTRY.keys())


def get_all_baseline_method_names() -> List[str]:
    """Get list of all registered baseline method names."""
    return list(BASELINE_METHODS.keys())


if __name__ == "__main__":
    # Validate registry on import
    validate_sweep_registry()
    
    # Example usage
    result = main({"mode": "runtime_smoke"})
    print(f"Sweep registry loaded: {len(result['experiment_sweeps'])} experiments")
    print(f"Available experiments: {get_all_experiment_ids()}")
    print(f"Available methods: {get_all_baseline_method_names()}")
