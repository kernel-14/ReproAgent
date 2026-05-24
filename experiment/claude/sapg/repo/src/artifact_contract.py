"""
src/artifact_contract.py
Artifact Contract Registry and Evidence Obligation Matrix for SAPG Reproduction
reference_grounding: wp_001 src/artifact_contract.py

This module implements the paper evidence obligation matrix, experiment registry,
metric schemas, and artifact writer surfaces for reproducing all figures and tables
from the SAPG paper.

Paper artifacts to reproduce:
- Figure 1: SAPG algorithm illustration (conceptual)
- Figure 2: Performance vs batch size for PPO
- Figure 3: SAPG architecture diagram (conceptual)
- Figure 4: Data aggregation schemes (conceptual)
- Figure 5: Performance curves SAPG vs baselines (PPO, PBT, PQL)
- Table 1: Performance after 2e10 samples across tasks
- Figure 6: Ablation study curves
- Figure 7: PCA reconstruction error curves
- Figure 8: MLP reconstruction error curves

Binding addendum clarifications:
- Figure 6: Blue plot is SAPG, others are ablations (symmetric aggregation, no off-policy, etc.)
- Figure 8: Two-layer MLP with same hidden size, ReLU activation, Adam optimizer
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum


class MetricType(Enum):
    """Metric types from paper evidence contract"""
    REWARD = "reward"
    SUCCESS_RATE = "success_rate"
    RETURN = "return"
    LOSS = "loss"
    ACCURACY = "accuracy"
    FIDELITY_SCORE = "fidelity_score"
    RECONSTRUCTION_ERROR = "reconstruction_error"


class ArtifactType(Enum):
    """Artifact types from paper"""
    FIGURE = "figure"
    TABLE = "table"
    METRICS_JSON = "metrics_json"
    CONFIG = "config"
    PREDICTIONS = "predictions"


@dataclass
class MetricSchema:
    """Schema for a metric measurement"""
    name: str
    metric_type: MetricType
    aggregation: str  # mean, std, min, max, final, etc.
    unit: Optional[str] = None
    higher_is_better: bool = True
    
    def to_dict(self):
        return {
            "name": self.name,
            "metric_type": self.metric_type.value,
            "aggregation": self.aggregation,
            "unit": self.unit,
            "higher_is_better": self.higher_is_better
        }


@dataclass
class ArtifactSpec:
    """Specification for a result artifact"""
    artifact_id: str
    artifact_type: ArtifactType
    output_path: str
    description: str
    paper_reference: str
    metrics: List[str]
    methods: List[str]
    
    def to_dict(self):
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type.value,
            "output_path": self.output_path,
            "description": self.description,
            "paper_reference": self.paper_reference,
            "metrics": self.metrics,
            "methods": self.methods
        }


@dataclass
class ExperimentSpec:
    """Specification for an experiment from the paper"""
    experiment_id: str
    name: str
    method: str
    tasks: List[str]
    metrics: List[str]
    artifacts: List[str]
    hyperparameters: Dict[str, Any]
    description: str
    
    def to_dict(self):
        return asdict(self)


# ============================================================================
# Evidence Contract Matrix: Paper-derived obligations
# ============================================================================

METRIC_SCHEMAS = {
    "reward": MetricSchema(
        name="reward",
        metric_type=MetricType.REWARD,
        aggregation="mean_over_episodes",
        unit="reward_units",
        higher_is_better=True
    ),
    "success_rate": MetricSchema(
        name="success_rate",
        metric_type=MetricType.SUCCESS_RATE,
        aggregation="mean_over_episodes",
        unit="proportion",
        higher_is_better=True
    ),
    "return": MetricSchema(
        name="return",
        metric_type=MetricType.RETURN,
        aggregation="mean_over_episodes",
        unit="cumulative_reward",
        higher_is_better=True
    ),
    "loss": MetricSchema(
        name="loss",
        metric_type=MetricType.LOSS,
        aggregation="mean_over_batches",
        unit="loss_units",
        higher_is_better=False
    ),
    "accuracy": MetricSchema(
        name="accuracy",
        metric_type=MetricType.ACCURACY,
        aggregation="mean",
        unit="proportion",
        higher_is_better=True
    ),
    "fidelity_score": MetricSchema(
        name="fidelity_score",
        metric_type=MetricType.FIDELITY_SCORE,
        aggregation="mean",
        unit="score",
        higher_is_better=True
    ),
    "reconstruction_error": MetricSchema(
        name="reconstruction_error",
        metric_type=MetricType.RECONSTRUCTION_ERROR,
        aggregation="mean_over_states",
        unit="mse",
        higher_is_better=False
    ),
}


ARTIFACT_SPECS = {
    "figure_1": ArtifactSpec(
        artifact_id="figure_1",
        artifact_type=ArtifactType.FIGURE,
        output_path="results/figures/figure_1.png",
        description="SAPG algorithm illustration showing split and aggregate mechanism",
        paper_reference="Figure 1",
        metrics=[],
        methods=["sapg"]
    ),
    "figure_2": ArtifactSpec(
        artifact_id="figure_2",
        artifact_type=ArtifactType.FIGURE,
        output_path="results/figures/figure_2.png",
        description="Performance vs batch size plot for PPO showing saturation",
        paper_reference="Figure 2",
        metrics=["reward"],
        methods=["ppo"]
    ),
    "figure_3": ArtifactSpec(
        artifact_id="figure_3",
        artifact_type=ArtifactType.FIGURE,
        output_path="results/figures/figure_3.png",
        description="SAPG architecture with leader and M-1 followers, shared backbone B_theta",
        paper_reference="Figure 3",
        metrics=[],
        methods=["sapg"]
    ),
    "figure_4": ArtifactSpec(
        artifact_id="figure_4",
        artifact_type=ArtifactType.FIGURE,
        output_path="results/figures/figure_4.png",
        description="Data aggregation schemes: leader-based vs symmetric",
        paper_reference="Figure 4",
        metrics=[],
        methods=["sapg"]
    ),
    "figure_5": ArtifactSpec(
        artifact_id="figure_5",
        artifact_type=ArtifactType.FIGURE,
        output_path="results/figures/figure_5.png",
        description="Performance curves SAPG vs PPO, PBT, PQL baselines across tasks",
        paper_reference="Figure 5",
        metrics=["reward", "success_rate"],
        methods=["sapg", "ppo", "pbt", "pql"]
    ),
    "table_1": ArtifactSpec(
        artifact_id="table_1",
        artifact_type=ArtifactType.TABLE,
        output_path="results/tables/table_1.csv",
        description="Performance after 2e10 samples with standard error across methods and tasks",
        paper_reference="Table 1",
        metrics=["success_rate", "reward"],
        methods=["sapg", "ppo", "pbt", "pql"]
    ),
    "figure_6": ArtifactSpec(
        artifact_id="figure_6",
        artifact_type=ArtifactType.FIGURE,
        output_path="results/figures/figure_6.png",
        description="Ablation study: symmetric aggregation, no off-policy, entropy variations, off-policy ratio",
        paper_reference="Figure 6",
        metrics=["reward", "success_rate"],
        methods=["sapg", "sapg_symmetric", "sapg_no_offpolicy", "sapg_entropy_0", "sapg_entropy_0.005", "sapg_entropy_0.01"]
    ),
    "figure_7": ArtifactSpec(
        artifact_id="figure_7",
        artifact_type=ArtifactType.FIGURE,
        output_path="results/figures/figure_7.png",
        description="PCA reconstruction error for states visited during training",
        paper_reference="Figure 7",
        metrics=["reconstruction_error"],
        methods=["sapg", "ppo", "random"]
    ),
    "figure_8": ArtifactSpec(
        artifact_id="figure_8",
        artifact_type=ArtifactType.FIGURE,
        output_path="results/figures/figure_8.png",
        description="MLP reconstruction error with varying hidden layer dimensions (two-layer, ReLU, Adam)",
        paper_reference="Figure 8",
        metrics=["reconstruction_error"],
        methods=["sapg", "ppo", "random"]
    ),
    "experiment_results": ArtifactSpec(
        artifact_id="experiment_results",
        artifact_type=ArtifactType.FIGURE,
        output_path="results/figures/experiment_results.png",
        description="General experiment results figure",
        paper_reference="result_figure",
        metrics=["reward", "success_rate"],
        methods=["sapg", "ppo", "pbt", "pql"]
    ),
    "metrics_json": ArtifactSpec(
        artifact_id="metrics_json",
        artifact_type=ArtifactType.METRICS_JSON,
        output_path="results/metrics.json",
        description="Aggregated metrics in JSON format",
        paper_reference="metrics_json",
        metrics=["reward", "success_rate", "loss", "return"],
        methods=["sapg", "ppo", "pbt", "pql"]
    ),
    "result_table": ArtifactSpec(
        artifact_id="result_table",
        artifact_type=ArtifactType.TABLE,
        output_path="results/tables/experiment_results.csv",
        description="Experiment results in tabular format",
        paper_reference="result_table",
        metrics=["reward", "success_rate", "return"],
        methods=["sapg", "ppo", "pbt", "pql"]
    ),
    "config": ArtifactSpec(
        artifact_id="config",
        artifact_type=ArtifactType.CONFIG,
        output_path="results/config_resolved.json",
        description="Resolved configuration for experiment",
        paper_reference="config",
        metrics=[],
        methods=[]
    ),
    "predictions": ArtifactSpec(
        artifact_id="predictions",
        artifact_type=ArtifactType.PREDICTIONS,
        output_path="results/predictions.jsonl",
        description="Per-sample predictions and outcomes",
        paper_reference="predictions",
        metrics=["reward", "success_rate"],
        methods=["sapg", "ppo"]
    ),
}


EXPERIMENT_REGISTRY = {
    "sapg_main": ExperimentSpec(
        experiment_id="sapg_main",
        name="SAPG Main Experiments",
        method="sapg",
        tasks=["ShadowHandOver", "ShadowHandCatchUnderarm", "ShadowHandCatchAbreast",
               "ShadowHandReOrientation", "AllegroHandReOrientation", "AllegroKuka"],
        metrics=["reward", "success_rate"],
        artifacts=["figure_5", "table_1"],
        hyperparameters={
            "num_policies": 6,
            "num_envs": 24576,
            "aggregation_coefficient": 1.0,
            "entropy_coefficient": 0.0,
            "total_samples": 2e10
        },
        description="Main SAPG experiments across all manipulation tasks"
    ),
    "ppo_baseline": ExperimentSpec(
        experiment_id="ppo_baseline",
        name="PPO Baseline",
        method="ppo",
        tasks=["ShadowHandOver", "ShadowHandCatchUnderarm", "ShadowHandCatchAbreast",
               "ShadowHandReOrientation", "AllegroHandReOrientation", "AllegroKuka"],
        metrics=["reward", "success_rate"],
        artifacts=["figure_2", "figure_5", "table_1"],
        hyperparameters={
            "num_policies": 1,
            "num_envs": 24576,
            "entropy_coefficient": 0.01,
            "total_samples": 2e10
        },
        description="PPO baseline for comparison with SAPG"
    ),
    "pbt_baseline": ExperimentSpec(
        experiment_id="pbt_baseline",
        name="PBT Baseline",
        method="pbt",
        tasks=["ShadowHandOver", "AllegroKuka"],
        metrics=["reward", "success_rate"],
        artifacts=["figure_5", "table_1"],
        hyperparameters={
            "population_size": 6,
            "num_envs": 24576,
            "total_samples": 2e10
        },
        description="Population-Based Training baseline"
    ),
    "pql_baseline": ExperimentSpec(
        experiment_id="pql_baseline",
        name="PQL Baseline",
        method="pql",
        tasks=["ShadowHandOver", "AllegroKuka"],
        metrics=["reward", "success_rate"],
        artifacts=["figure_5", "table_1"],
        hyperparameters={
            "num_policies": 6,
            "num_envs": 24576,
            "total_samples": 2e10
        },
        description="Parallel Q-Learning baseline"
    ),
    "ablation_symmetric": ExperimentSpec(
        experiment_id="ablation_symmetric",
        name="SAPG Ablation: Symmetric Aggregation",
        method="sapg_symmetric",
        tasks=["ShadowHandOver", "AllegroKuka"],
        metrics=["reward", "success_rate"],
        artifacts=["figure_6"],
        hyperparameters={
            "num_policies": 6,
            "num_envs": 24576,
            "aggregation_scheme": "symmetric",
            "total_samples": 2e10
        },
        description="SAPG without designated leader, symmetric data aggregation"
    ),
    "ablation_no_offpolicy": ExperimentSpec(
        experiment_id="ablation_no_offpolicy",
        name="SAPG Ablation: No Off-Policy",
        method="sapg_no_offpolicy",
        tasks=["ShadowHandOver", "AllegroKuka"],
        metrics=["reward", "success_rate"],
        artifacts=["figure_6"],
        hyperparameters={
            "num_policies": 6,
            "num_envs": 24576,
            "aggregation_coefficient": 0.0,
            "total_samples": 2e10
        },
        description="SAPG without off-policy data aggregation"
    ),
    "ablation_entropy": ExperimentSpec(
        experiment_id="ablation_entropy",
        name="SAPG Ablation: Entropy Variations",
        method="sapg",
        tasks=["ShadowHandReOrientation", "AllegroHandReOrientation"],
        metrics=["reward", "success_rate"],
        artifacts=["figure_6"],
        hyperparameters={
            "num_policies": 6,
            "num_envs": 24576,
            "entropy_coefficient_sweep": [0.0, 0.003, 0.005],
            "total_samples": 2e10
        },
        description="SAPG with different entropy coefficients"
    ),
    "state_coverage_pca": ExperimentSpec(
        experiment_id="state_coverage_pca",
        name="State Coverage Analysis: PCA",
        method="sapg",
        tasks=["ShadowHandOver"],
        metrics=["reconstruction_error"],
        artifacts=["figure_7"],
        hyperparameters={
            "num_policies": 6,
            "num_envs": 24576,
            "pca_components_sweep": [1, 2, 5, 10, 20, 50, 100],
            "total_samples": 2e10
        },
        description="PCA reconstruction error analysis for state space coverage"
    ),
    "state_coverage_mlp": ExperimentSpec(
        experiment_id="state_coverage_mlp",
        name="State Coverage Analysis: MLP",
        method="sapg",
        tasks=["ShadowHandOver"],
        metrics=["reconstruction_error"],
        artifacts=["figure_8"],
        hyperparameters={
            "num_policies": 6,
            "num_envs": 24576,
            "mlp_hidden_size_sweep": [16, 32, 64, 128, 256, 512],
            "mlp_layers": 2,
            "mlp_activation": "relu",
            "total_samples": 2e10
        },
        description="MLP reconstruction error analysis (two-layer, ReLU, Adam)"
    ),
}


# ============================================================================
# Artifact Writer Functions
# ============================================================================

def ensure_output_dir(output_path: str) -> Path:
    """Ensure output directory exists"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_evidence_contract_matrix(output_path: str = "results/evidence_contract_matrix.json", mode: str = "smoke"):
    """
    Write the complete evidence contract matrix to JSON.
    
    This matrix captures all paper-derived obligations: experiments, baselines,
    metrics, artifacts, and their relationships.
    """
    path = ensure_output_dir(output_path)
    
    contract_matrix = {
        "paper_title": "SAPG: Split and Aggregate Policy Gradients",
        "reproduction_mode": mode,
        "metrics": {k: v.to_dict() for k, v in METRIC_SCHEMAS.items()},
        "artifacts": {k: v.to_dict() for k, v in ARTIFACT_SPECS.items()},
        "experiments": {k: v.to_dict() for k, v in EXPERIMENT_REGISTRY.items()},
        "methods": ["sapg", "ppo", "pbt", "pql", "ddpg"],
        "tasks": [
            "ShadowHandOver", "ShadowHandCatchUnderarm", "ShadowHandCatchAbreast",
            "ShadowHandReOrientation", "AllegroHandReOrientation", "AllegroKuka",
            "harder_AllegroKuka", "Throw", "Regrasping", "Reorientation"
        ],
        "paper_figures": ["Figure 1", "Figure 2", "Figure 3", "Figure 4", "Figure 5", 
                         "Figure 6", "Figure 7", "Figure 8"],
        "paper_tables": ["Table 1", "Table 2", "Table 3", "Table 4"],
        "addendum_clarifications": {
            "figure_6": "Blue plot is SAPG, others are ablations (symmetric aggregation, no off-policy, etc.)",
            "figure_8": "Two-layer MLP with same hidden size, ReLU activation, Adam optimizer with default PyTorch hyperparameters"
        }
    }
    
    with open(path, 'w') as f:
        json.dump(contract_matrix, f, indent=2)
    
    return str(path)


def write_experiment_registry(output_path: str = "results/experiment_registry.json", mode: str = "smoke"):
    """Write experiment registry with all paper experiments"""
    path = ensure_output_dir(output_path)
    
    registry = {
        "experiments": {k: v.to_dict() for k, v in EXPERIMENT_REGISTRY.items()},
        "mode": mode,
        "total_experiments": len(EXPERIMENT_REGISTRY)
    }
    
    with open(path, 'w') as f:
        json.dump(registry, f, indent=2)
    
    return str(path)


def write_metrics_json(metrics_data: Dict[str, Any], output_path: str = "results/metrics.json", mode: str = "smoke"):
    """
    Write aggregated metrics to JSON.
    
    Args:
        metrics_data: Dictionary containing metric measurements
        output_path: Output file path
        mode: Execution mode (smoke, default, full)
    """
    path = ensure_output_dir(output_path)
    
    if mode == "smoke":
        # Dry-run schema artifact
        metrics_output = {
            "_artifact_type": "dry_run_schema",
            "_description": "Metrics schema for smoke validation - not real experiment results",
            "schemas": {k: v.to_dict() for k, v in METRIC_SCHEMAS.items()},
            "sample_data": {
                "reward": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
                "success_rate": {"mean": 0.0, "std": 0.0},
                "return": {"mean": 0.0, "std": 0.0},
                "loss": {"mean": 0.0, "std": 0.0}
            }
        }
    else:
        # Real metrics data
        metrics_output = {
            "_artifact_type": "experiment_metrics",
            "schemas": {k: v.to_dict() for k, v in METRIC_SCHEMAS.items()},
            "data": metrics_data
        }
    
    with open(path, 'w') as f:
        json.dump(metrics_output, f, indent=2)
    
    return str(path)


def write_artifact_manifest(output_path: str = "results/artifact_manifest.json", mode: str = "smoke"):
    """Write manifest of all declared artifacts with their paths and status"""
    path = ensure_output_dir(output_path)
    
    manifest = {
        "mode": mode,
        "artifacts": {k: v.to_dict() for k, v in ARTIFACT_SPECS.items()},
        "total_artifacts": len(ARTIFACT_SPECS),
        "artifact_paths": [spec.output_path for spec in ARTIFACT_SPECS.values()]
    }
    
    with open(path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    return str(path)


def write_sensitivity_report(sensitivity_data: Dict[str, Any], output_path: str = "results/sensitivity_report.json", mode: str = "smoke"):
    """
    Write sensitivity analysis report.
    
    Covers parameter sweeps from the paper:
    - Entropy coefficient variations (0, 0.005, 0.01)
    - Off-policy ratio variations
    - Number of policies (M)
    - Batch size effects
    """
    path = ensure_output_dir(output_path)
    
    if mode == "smoke":
        # Dry-run schema
        report = {
            "_artifact_type": "dry_run_schema",
            "_description": "Sensitivity analysis schema - not real experiment results",
            "parameter_sweeps": {
                "entropy_coefficient": [0.0, 0.003, 0.005],
                "aggregation_coefficient": [1.0],
                "num_policies": [6],
                "batch_size": [512, 1024, 2048, 4096, 8192, 16384, 24576]
            },
            "sample_results": {}
        }
    else:
        # Real sensitivity data
        report = {
            "_artifact_type": "sensitivity_analysis",
            "parameter_sweeps": sensitivity_data.get("sweeps", {}),
            "results": sensitivity_data.get("results", {})
        }
    
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)
    
    return str(path)


def write_figure_artifact(figure_data: Any, output_path: str, mode: str = "smoke"):
    """
    Write a figure artifact (PNG).
    
    Args:
        figure_data: Matplotlib figure or plot data
        output_path: Output file path
        mode: Execution mode
    """
    path = ensure_output_dir(output_path)
    
    if mode == "smoke":
        # Create minimal diagnostic image for smoke validation
        try:
            import importlib.util
            if importlib.util.find_spec("matplotlib") is not None:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.text(0.5, 0.5, f"DRY-RUN ARTIFACT\n{Path(output_path).name}\nSchema validation only",
                       ha='center', va='center', fontsize=14, color='red')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                plt.savefig(path, dpi=100, bbox_inches='tight')
                plt.close(fig)
            else:
                # Fallback: write a marker file
                with open(path.with_suffix('.txt'), 'w') as f:
                    f.write(f"DRY-RUN ARTIFACT MARKER: {output_path}\n")
                    f.write("Matplotlib not available for smoke validation\n")
        except Exception as e:
            # Fallback: write a marker file
            with open(path.with_suffix('.txt'), 'w') as f:
                f.write(f"DRY-RUN ARTIFACT MARKER: {output_path}\n")
                f.write(f"Error creating diagnostic image: {e}\n")
    else:
        # Real figure data - save using matplotlib
        if hasattr(figure_data, 'savefig'):
            figure_data.savefig(path, dpi=300, bbox_inches='tight')
        else:
            raise ValueError(f"Invalid figure data for {output_path}")
    
    return str(path)


def write_table_artifact(table_data: Any, output_path: str, mode: str = "smoke"):
    """
    Write a table artifact (CSV).
    
    Args:
        table_data: DataFrame or dict with table data
        output_path: Output file path
        mode: Execution mode
    """
    path = ensure_output_dir(output_path)
    
    if mode == "smoke":
        # Dry-run schema
        import csv
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["_artifact_type", "dry_run_schema"])
            writer.writerow(["_description", "Table schema for smoke validation - not real results"])
            writer.writerow([])
            writer.writerow(["method", "task", "metric", "value", "std_error"])
            writer.writerow(["sapg", "ShadowHandOver", "success_rate", "0.0", "0.0"])
            writer.writerow(["ppo", "ShadowHandOver", "success_rate", "0.0", "0.0"])
    else:
        # Real table data
        if hasattr(table_data, 'to_csv'):
            table_data.to_csv(path, index=False)
        else:
            import csv
            with open(path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=table_data[0].keys())
                writer.writeheader()
                writer.writerows(table_data)
    
    return str(path)


def write_all_smoke_artifacts():
    """
    Write all declared artifacts in smoke/dry-run mode.
    
    This creates schema/contract artifacts for every declared output path
    to validate artifact closure without running expensive experiments.
    """
    mode = "smoke"
    written_paths = []
    
    # Core contract artifacts
    written_paths.append(write_evidence_contract_matrix(mode=mode))
    written_paths.append(write_experiment_registry(mode=mode))
    written_paths.append(write_metrics_json({}, mode=mode))
    written_paths.append(write_artifact_manifest(mode=mode))
    written_paths.append(write_sensitivity_report({}, mode=mode))
    
    # Figure artifacts
