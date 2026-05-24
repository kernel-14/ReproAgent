"""
Sequential Neural Posterior Score Estimation - Parameter Sweep Registry

This module implements the paper evidence obligation matrix and parameter sweep registry
for SNPSE experiments. It binds paper-stated experiments to datasets, methods, parameters,
expected trends, and result artifacts.

Reference grounding:
- paperbench_ref_001 l5pc/docs/config.md: Configuration structure for multi-round inference
- paperbench_ref_001 sbi/sbi/inference/snpe/snpe_a.py: SNPE-A training parameters (learning_rate=5e-4)
- paperbench_ref_001 sbi/sbi/inference/snle/mnle.py: MNLE method interface
- paperbench_ref_001 sbi/sbi/inference/snle/snle_base.py: SNLE base class structure

Paper: Sequential Neural Score Estimation: Likelihood-Free Inference with
       Conditional Score Based Diffusion Models

Implementation surfaces: config, evaluation, baseline_or_ablation, artifact_writer

Method obligations:
- Paper evidence contract: expose bounded sweep/config entries for learning_rate=1e-4, optimizer=Adam
- Implement a code/config-visible paper evidence obligation matrix
- Each matrix row must bind paper/addendum-stated experiment to datasets, methods, parameters, trends, and artifacts
- Binding addendum clarification: The `sbibm` library should be used for C2ST with default hyperparameters
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple
import warnings


# ============================================================================
# Parameter Sweep Registry
# reference_grounding: paperbench_ref_001 sbi/sbi/inference/snpe/snpe_a.py
# Paper evidence contract: expose bounded sweep/config entries for learning_rate
# ============================================================================

PARAMETER_SWEEP_REGISTRY: Dict[str, Dict[str, Any]] = {
    "learning_rate": {
        "param_id": "learning_rate",
        "paper_value": 1e-4,
        "reference_value": 5e-4,  # From paperbench_ref_001:sbi/sbi/inference/snpe/snpe_a.py
        "sweep_values": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
        "bounded_sweep": [1e-4],  # Paper-stated value, not exhaustive
        "default": 1e-4,
        "description": "Learning rate for score network training (Adam optimizer)",
        "paper_section": "Section 5, Appendix",
        "sweep_type": "log_scale",
        "decision_value": "Paper uses 1e-4 as standard across experiments"
    },
    "optimizer": {
        "param_id": "optimizer",
        "paper_value": "Adam",
        "sweep_values": ["Adam", "AdamW", "SGD"],
        "bounded_sweep": ["Adam"],  # Paper-stated optimizer
        "default": "Adam",
        "description": "Optimizer for score network training",
        "paper_section": "Section 5",
        "sweep_type": "categorical",
        "decision_value": "Adam optimizer used throughout paper experiments"
    },
    "batch_size": {
        "param_id": "batch_size",
        "paper_value": 100,
        "sweep_values": [50, 100, 200],
        "bounded_sweep": [100],
        "default": 100,
        "description": "Training batch size",
        "paper_section": "Appendix",
        "sweep_type": "discrete",
        "decision_value": "Standard batch size for NPSE training"
    },
    "num_rounds": {
        "param_id": "num_rounds",
        "paper_value": 10,
        "sweep_values": [1, 5, 10],
        "bounded_sweep": [1, 10],  # Non-sequential vs sequential
        "default": 10,
        "description": "Number of sequential rounds for TSNPSE",
        "paper_section": "Algorithm 1, Section 5",
        "sweep_type": "discrete",
        "decision_value": "10 rounds for sequential methods, 1 for non-sequential"
    },
    "simulation_budget": {
        "param_id": "simulation_budget",
        "paper_value": [1000, 10000, 100000],
        "sweep_values": [1000, 10000, 100000],
        "bounded_sweep": [1000, 10000, 100000],  # All three budgets tested
        "default": 10000,
        "description": "Total simulation budget for SBI experiments",
        "paper_section": "Figure 2, Figure 3, Figure 9",
        "sweep_type": "discrete",
        "decision_value": "Three budgets test sample efficiency across methods"
    },
    "sde_type": {
        "param_id": "sde_type",
        "paper_value": "VE",
        "sweep_values": ["VE", "VP"],
        "bounded_sweep": ["VE"],  # Variance Exploding SDE used in paper
        "default": "VE",
        "description": "SDE type for diffusion process (VE: Variance Exploding, VP: Variance Preserving)",
        "paper_section": "Section 3, Section 4",
        "sweep_type": "categorical",
        "decision_value": "VE SDE provides stable score estimation"
    },
    "num_diffusion_steps": {
        "param_id": "num_diffusion_steps",
        "paper_value": 1000,
        "sweep_values": [100, 500, 1000],
        "bounded_sweep": [1000],
        "default": 1000,
        "description": "Number of discretization steps for reverse diffusion sampling",
        "paper_section": "Section 4",
        "sweep_type": "discrete",
        "decision_value": "1000 steps for accurate posterior sampling"
    },
}


# ============================================================================
# Paper Evidence Obligation Matrix
# reference_grounding: paperbench_ref_001 l5pc/docs/config.md
# Method obligation: Each matrix row binds experiment to datasets, methods, parameters, trends, artifacts
# ============================================================================

EVIDENCE_OBLIGATION_MATRIX: List[Dict[str, Any]] = [
    {
        "experiment_id": "figure_1_two_moons",
        "experiment_name": "Two Moons Posterior Visualization",
        "paper_reference": "Figure 1",
        "description": "Visualisation of posterior inference using Neural Posterior Score Estimation (NPSE) in the 'Two Moons' experiment",
        "datasets": ["two_moons"],
        "methods": ["NPSE"],
        "baselines": [],
        "parameters": {
            "learning_rate": 1e-4,
            "optimizer": "Adam",
            "simulation_budget": 10000,
            "sde_type": "VE"
        },
        "expected_trend": "Forward process transforms samples from target posterior to reference distribution; backward process transports samples from reference to posterior",
        "decision_claim": "NPSE can accurately approximate bimodal posterior distribution",
        "result_artifacts": [
            "results/figures/figure_1_two_moons_posterior.pdf",
            "results/figures/figure_1_forward_backward_process.pdf"
        ],
        "metrics": ["visual_quality", "posterior_coverage"],
        "paper_section": "Section 5.1, Figure 1"
    },
    {
        "experiment_id": "figure_2_non_sequential_benchmark",
        "experiment_name": "Non-Sequential Methods on 8 Benchmark Tasks",
        "paper_reference": "Figure 2",
        "description": "Results on eight benchmark tasks (non-sequential methods)",
        "datasets": ["slcp", "gaussian_linear", "gaussian_mixture", "two_moons", "bernoulli_glm", "lotka_volterra", "sir", "gaussian_linear_uniform"],
        "methods": ["NPSE", "NPE", "NLE", "NRE"],
        "baselines": ["NPE", "NLE", "NRE"],
        "parameters": {
            "learning_rate": 1e-4,
            "optimizer": "Adam",
            "simulation_budget": [1000, 10000, 100000],
            "num_rounds": 1
        },
        "expected_trend": "NPSE performance comparable or superior to NPE/NLE/NRE across simulation budgets",
        "decision_claim": "NPSE is competitive with established non-sequential methods",
        "result_artifacts": [
            "results/figures/figure_2_non_sequential_benchmark.pdf",
            "results/benchmark_metrics_non_sequential.csv"
        ],
        "metrics": ["c2st", "mmd", "posterior_log_prob"],
        "paper_section": "Section 5.2, Figure 2"
    },
    {
        "experiment_id": "figure_3_sequential_benchmark",
        "experiment_name": "Sequential Methods on 8 Benchmark Tasks",
        "paper_reference": "Figure 3",
        "description": "Results on eight benchmark tasks (sequential methods)",
        "datasets": ["slcp", "gaussian_linear", "gaussian_mixture", "two_moons", "bernoulli_glm", "lotka_volterra", "sir", "gaussian_linear_uniform"],
        "methods": ["TSNPSE", "SNPE-A", "SNPE-B", "SNPE-C"],
        "baselines": ["SNPE-A", "SNPE-B", "SNPE-C"],
        "parameters": {
            "learning_rate": 1e-4,
            "optimizer": "Adam",
            "simulation_budget": [1000, 10000, 100000],
            "num_rounds": 10
        },
        "expected_trend": "TSNPSE improves sample efficiency over non-sequential NPSE, competitive with SNPE variants",
        "decision_claim": "Sequential approach (TSNPSE) provides significant gains over non-sequential NPSE",
        "result_artifacts": [
            "results/figures/figure_3_sequential_benchmark.pdf",
            "results/benchmark_metrics_sequential.csv"
        ],
        "metrics": ["c2st", "mmd", "posterior_log_prob"],
        "paper_section": "Section 5.2, Figure 3"
    },
    {
        "experiment_id": "figure_4_pyloric",
        "experiment_name": "Pyloric Neuron Experiment",
        "paper_reference": "Figure 4",
        "description": "Results for the Pyloric experiment",
        "datasets": ["pyloric"],
        "methods": ["TSNPSE"],
        "baselines": ["SNPE-C", "SNVI"],
        "parameters": {
            "learning_rate": 1e-4,
            "optimizer": "Adam",
            "simulation_budget": 100000,
            "num_rounds": 10
        },
        "expected_trend": "TSNPSE achieves comparable or better posterior approximation quality on high-dimensional realistic task",
        "decision_claim": "TSNPSE scales to realistic neuroscience simulation-based inference problems",
        "result_artifacts": [
            "results/figures/figure_4_pyloric_metrics.pdf",
            "results/figures/figure_7_pyloric_marginals.pdf",
            "results/figures/figure_8_pyloric_coverage.pdf"
        ],
        "metrics": ["c2st", "coverage", "posterior_mean_error"],
        "paper_section": "Section 5.3, Figure 4, Figure 7, Figure 8"
    },
    {
        "experiment_id": "figure_5_npse_vs_nlse",
        "experiment_name": "NPSE vs NLSE Comparison",
        "paper_reference": "Figure 5",
        "description": "Comparison between NPSE and NLSE on four benchmark tasks",
        "datasets": ["slcp", "gaussian_linear", "two_moons", "bernoulli_glm"],
        "methods": ["NPSE", "NLSE"],
        "baselines": ["NLSE"],
        "parameters": {
            "learning_rate": 1e-4,
            "optimizer": "Adam",
            "simulation_budget": [1000, 10000, 100000],
            "num_rounds": 1
        },
        "expected_trend": "NPSE (score-based) outperforms NLSE (likelihood-based) on most tasks",
        "decision_claim": "Score estimation provides advantages over likelihood estimation for posterior inference",
        "result_artifacts": [
            "results/figures/figure_5_npse_vs_nlse.pdf",
            "results/comparison_npse_nlse_metrics.csv"
        ],
        "metrics": ["c2st", "posterior_log_prob"],
        "paper_section": "Appendix E.2, Figure 5"
    },
    {
        "experiment_id": "figure_6_sequential_variants",
        "experiment_name": "Sequential Variant Comparison (SNPSE-A/B vs TSNPSE)",
        "paper_reference": "Figure 6",
        "description": "Comparison between SNPSE-A, SNPSE-B, and TSNPSE on two benchmark tasks",
        "datasets": ["slcp", "gaussian_linear_uniform"],
        "methods": ["TSNPSE", "SNPSE-A", "SNPSE-B"],
        "baselines": ["SNPSE-A", "SNPSE-B"],
        "parameters": {
            "learning_rate": 1e-4,
            "optimizer": "Adam",
            "simulation_budget": [1000, 10000, 100000],
            "num_rounds": 10
        },
        "expected_trend": "TSNPSE performs better than SNPSE-A and SNPSE-B; SNPSE-C fails (C2ST ≈ 1)",
        "decision_claim": "Truncated proposal (TSNPSE) is superior to alternative sequential approaches",
        "result_artifacts": [
            "results/figures/figure_6_sequential_variants.pdf",
            "results/ablation_sequential_variants_metrics.csv"
        ],
        "metrics": ["c2st"],
        "paper_section": "Appendix E.3, Figure 6",
        "notes": "SNPSE-C omitted from results due to failure (C2ST ≈ 1)"
    },
    {
        "experiment_id": "figure_9_npse_vs_fmpe",
        "experiment_name": "NPSE vs FMPE Comparison",
        "paper_reference": "Figure 9",
        "description": "Comparison between NPSE and FMPE on eight benchmark tasks",
        "datasets": ["slcp", "gaussian_linear", "gaussian_mixture", "two_moons", "bernoulli_glm", "lotka_volterra", "sir", "gaussian_linear_uniform"],
        "methods": ["NPSE", "FMPE"],
        "baselines": ["FMPE"],
        "parameters": {
            "learning_rate": 1e-4,
            "optimizer": "Adam",
            "simulation_budget": [1000, 10000, 100000],
            "num_rounds": 1
        },
        "expected_trend": "NPSE and FMPE show comparable performance across tasks",
        "decision_claim": "Score-based diffusion (NPSE) is competitive with flow matching approaches (FMPE)",
        "result_artifacts": [
            "results/figures/figure_9_npse_vs_fmpe.pdf",
            "results/comparison_npse_fmpe_metrics.csv"
        ],
        "metrics": ["c2st", "mmd"],
        "paper_section": "Appendix E.4, Figure 9"
    },
    {
        "experiment_id": "c2st_metric_validation",
        "experiment_name": "C2ST Metric Implementation Validation",
        "paper_reference": "Section 5.3, Addendum",
        "description": "C2ST metric uses sbibm library with default hyperparameters (per addendum)",
        "datasets": ["all_benchmark_tasks"],
        "methods": ["all_methods"],
        "baselines": [],
        "parameters": {
            "c2st_classifier": "random_forest",
            "c2st_n_folds": 5,
            "c2st_use_default_sbibm_hyperparameters": True
        },
        "expected_trend": "C2ST values close to 0.5 indicate good posterior approximation",
        "decision_claim": "C2ST is primary metric for posterior quality assessment",
        "result_artifacts": [
            "results/c2st_validation_report.json"
        ],
        "metrics": ["c2st"],
        "paper_section": "Section 5.3, Addendum",
        "binding_clarification": "The `sbibm` library should be used to implement the C2ST method, using default hyperparameters"
    }
]


# ============================================================================
# Experiment Registry
# ============================================================================

EXPERIMENT_REGISTRY: Dict[str, Dict[str, Any]] = {
    experiment["experiment_id"]: experiment
    for experiment in EVIDENCE_OBLIGATION_MATRIX
}


# ============================================================================
# Dataset-Method-Metric Coverage Matrix
# ============================================================================

def get_coverage_matrix() -> Dict[str, Any]:
    """
    Generate a coverage matrix showing which datasets, methods, and metrics
    are tested across all paper experiments.
    
    Returns:
        Dict containing coverage statistics and matrix
    """
    datasets_covered: Set[str] = set()
    methods_covered: Set[str] = set()
    baselines_covered: Set[str] = set()
    metrics_covered: Set[str] = set()
    
    for experiment in EVIDENCE_OBLIGATION_MATRIX:
        datasets_covered.update(experiment["datasets"])
        methods_covered.update(experiment["methods"])
        baselines_covered.update(experiment.get("baselines", []))
        metrics_covered.update(experiment.get("metrics", []))
    
    return {
        "datasets": sorted(list(datasets_covered)),
        "methods": sorted(list(methods_covered)),
        "baselines": sorted(list(baselines_covered)),
        "metrics": sorted(list(metrics_covered)),
        "total_experiments": len(EVIDENCE_OBLIGATION_MATRIX),
        "total_datasets": len(datasets_covered),
        "total_methods": len(methods_covered),
        "total_baselines": len(baselines_covered),
        "total_metrics": len(metrics_covered)
    }


# ============================================================================
# Query Functions
# ============================================================================

def get_experiments_by_dataset(dataset_id: str) -> List[Dict[str, Any]]:
    """Get all experiments that use a specific dataset."""
    return [
        exp for exp in EVIDENCE_OBLIGATION_MATRIX
        if dataset_id in exp["datasets"]
    ]


def get_experiments_by_method(method_id: str) -> List[Dict[str, Any]]:
    """Get all experiments that test a specific method."""
    return [
        exp for exp in EVIDENCE_OBLIGATION_MATRIX
        if method_id in exp["methods"]
    ]


def get_experiments_by_metric(metric_id: str) -> List[Dict[str, Any]]:
    """Get all experiments that use a specific metric."""
    return [
        exp for exp in EVIDENCE_OBLIGATION_MATRIX
        if metric_id in exp.get("metrics", [])
    ]


def get_parameter_sweep_values(param_id: str, bounded_only: bool = True) -> List[Any]:
    """
    Get sweep values for a parameter.
    
    Args:
        param_id: Parameter identifier
        bounded_only: If True, return only bounded sweep values (paper-tested)
                     If False, return all possible sweep values
    
    Returns:
        List of parameter values for sweep
    """
    if param_id not in PARAMETER_SWEEP_REGISTRY:
        raise ValueError(f"Unknown parameter: {param_id}")
    
    param = PARAMETER_SWEEP_REGISTRY[param_id]
    
    if bounded_only:
        return param["bounded_sweep"]
    else:
        return param["sweep_values"]


def get_experiments_by_paper_figure(figure_ref: str) -> List[Dict[str, Any]]:
    """Get all experiments associated with a paper figure."""
    return [
        exp for exp in EVIDENCE_OBLIGATION_MATRIX
        if exp["paper_reference"] == figure_ref
    ]


# ============================================================================
# Artifact Writing Functions
# ============================================================================

def write_evidence_contract_matrix(output_path: str = "results/evidence_contract_matrix.json") -> None:
    """
    Write the evidence obligation matrix to JSON artifact.
    
    Args:
        output_path: Path for output JSON file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    output = {
        "matrix": EVIDENCE_OBLIGATION_MATRIX,
        "coverage": get_coverage_matrix(),
        "metadata": {
            "total_experiments": len(EVIDENCE_OBLIGATION_MATRIX),
            "paper_title": "Sequential Neural Score Estimation: Likelihood-Free Inference with Conditional Score Based Diffusion Models",
            "implementation_note": "Evidence obligation matrix binds paper experiments to datasets, methods, parameters, trends, and artifacts"
        }
    }
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)


def write_experiment_registry(output_path: str = "results/experiment_registry.json") -> None:
    """
    Write the experiment registry to JSON artifact.
    
    Args:
        output_path: Path for output JSON file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(EXPERIMENT_REGISTRY, f, indent=2)


def write_parameter_sweep_registry(output_path: str = "results/parameter_sweep_registry.json") -> None:
    """
    Write the parameter sweep registry to JSON artifact.
    
    Args:
        output_path: Path for output JSON file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(PARAMETER_SWEEP_REGISTRY, f, indent=2)


def write_sensitivity_report(output_path: str = "results/sensitivity_report.json") -> None:
    """
    Write a sensitivity analysis report showing parameter sweep coverage.
    
    Args:
        output_path: Path for output JSON file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    sensitivity_data = {
        "parameter_sweeps": {},
        "experiment_parameter_matrix": {},
        "metadata": {
            "total_parameters": len(PARAMETER_SWEEP_REGISTRY),
            "bounded_sweep_rationale": "Bounded sweeps contain only paper-tested values, not exhaustive combinations"
        }
    }
    
    # Add parameter sweep information
    for param_id, param_info in PARAMETER_SWEEP_REGISTRY.items():
        sensitivity_data["parameter_sweeps"][param_id] = {
            "paper_value": param_info["paper_value"],
            "bounded_sweep": param_info["bounded_sweep"],
            "full_sweep": param_info["sweep_values"],
            "decision_value": param_info.get("decision_value", ""),
            "sweep_type": param_info["sweep_type"]
        }
    
    # Add experiment-parameter mappings
    for experiment in EVIDENCE_OBLIGATION_MATRIX:
        exp_id = experiment["experiment_id"]
        sensitivity_data["experiment_parameter_matrix"][exp_id] = {
            "parameters": experiment.get("parameters", {}),
            "expected_trend": experiment.get("expected_trend", ""),
            "decision_claim": experiment.get("decision_claim", "")
        }
    
    with open(output_path, "w") as f:
        json.dump(sensitivity_data, f, indent=2)


def write_artifact_manifest(output_path: str = "results/artifact_manifest.json") -> None:
    """
    Write manifest of all declared result artifacts from experiments.
    
    Args:
        output_path: Path for output JSON file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    artifact_manifest = {
        "experiments": {},
        "all_artifacts": [],
        "artifact_types": {
            "figures": [],
            "metrics_csv": [],
            "metrics_json": [],
            "other": []
        }
    }
    
    all_artifacts_set = set()
    
    for experiment in EVIDENCE_OBLIGATION_MATRIX:
        exp_id = experiment["experiment_id"]
        artifacts = experiment.get("result_artifacts", [])
        
        artifact_manifest["experiments"][exp_id] = {
            "name": experiment["experiment_name"],
            "paper_reference": experiment["paper_reference"],
            "artifacts": artifacts
        }
        
        all_artifacts_set.update(artifacts)
        
        # Categorize artifacts
        for artifact in artifacts:
            if artifact.endswith(".pdf") or artifact.endswith(".png"):
                artifact_manifest["artifact_types"]["figures"].append(artifact)
            elif artifact.endswith(".csv"):
                artifact_manifest["artifact_types"]["metrics_csv"].append(artifact)
            elif artifact.endswith(".json"):
                artifact_manifest["artifact_types"]["metrics_json"].append(artifact)
            else:
                artifact_manifest["artifact_types"]["other"].append(artifact)
    
    artifact_manifest["all_artifacts"] = sorted(list(all_artifacts_set))
    
    with open(output_path, "w") as f:
        json.dump(artifact_manifest, f, indent=2)


def write_all_sweep_registry_artifacts(dry_run: bool = True) -> None:
    """
    Write all sweep registry artifacts for smoke/validation modes.
    
    Args:
        dry_run: If True, label artifacts as dry-run/schema artifacts
    """
    write_evidence_contract_matrix()
    write_experiment_registry()
    write_parameter_sweep_registry()
    write_sensitivity_report()
    write_artifact_manifest()
    
    # Write a summary metadata file
    summary_path = "results/sweep_registry_summary.json"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    
    summary = {
        "artifacts_written": [
            "results/evidence_contract_matrix.json",
            "results/experiment_registry.json",
            "results/parameter_sweep_registry.json",
            "results/sensitivity_report.json",
            "results/artifact_manifest.json"
        ],
        "dry_run": dry_run,
        "coverage": get_coverage_matrix(),
        "implementation_note": "Sweep registry implements paper evidence obligation matrix with bounded parameter sweeps"
    }
    
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


# ============================================================================
# Main Entry Point (for testing/validation)
# ============================================================================

def main():
    """Generate all sweep registry artifacts."""
    print("Writing sweep registry artifacts...")
    write_all_sweep_registry_artifacts(dry_run=True)
    print("Sweep registry artifacts written successfully.")
    print(f"\nCoverage Summary:")
    coverage = get_coverage_matrix()
    print(f"  Total Experiments: {coverage['total_experiments']}")
    print(f"  Datasets Covered: {coverage['total_datasets']}")
    print(f"  Methods Covered: {coverage['total_methods']}")
    print(f"  Baselines Covered: {coverage['total_baselines']}")
    print(f"  Metrics Covered: {coverage['total_metrics']}")


if __name__ == "__main__":
    main()