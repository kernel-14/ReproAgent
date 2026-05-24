"""
src/reporting/artifacts.py
Artifact writers and evidence contract registry for SAPG reproduction.
reference_grounding: wp_001 src/reporting/artifacts.py

Paper evidence contract: This module implements measurement schemas, aggregation
outputs, and artifact writers for all paper figures and tables:
- Figure 1: SAPG concept introduction
- Figure 2: Performance vs batch size for PPO
- Figure 3: SAPG architecture (leader/followers)
- Figure 4: Data aggregation schemes
- Figure 5: Performance curves (SAPG vs PPO, PBT, PQL)
- Table 1: Performance after 2e10 samples
- Figure 6: Ablation study
- Figure 7: PCA reconstruction error
- Figure 8: MLP reconstruction error

Binding addendum clarification: Figure 6 blue plot is SAPG, other curves are
ablations (symmetric aggregation, no off-policy, entropy variations).

Result-trend assertions preserved:
- baseline_outperformance: SAPG should outperform PPO, PBT, PQL baselines
- positive_parameter_improves: nonzero entropy/aggregation coefficients improve performance
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime


# ============================================================================
# Metric Schemas
# ============================================================================

@dataclass
class MetricSchema:
    """Schema for a single metric measurement."""
    name: str
    aggregation: str  # mean, sum, max, min, std, median
    unit: Optional[str] = None
    higher_is_better: bool = True
    description: str = ""


METRIC_SCHEMAS = {
    "reward": MetricSchema(
        name="reward",
        aggregation="mean",
        unit="scalar",
        higher_is_better=True,
        description="Episode reward (mean across environments)"
    ),
    "return": MetricSchema(
        name="return",
        aggregation="mean",
        unit="scalar",
        higher_is_better=True,
        description="Cumulative discounted return"
    ),
    "success_rate": MetricSchema(
        name="success_rate",
        aggregation="mean",
        unit="percentage",
        higher_is_better=True,
        description="Task success rate (0-1)"
    ),
    "accuracy": MetricSchema(
        name="accuracy",
        aggregation="mean",
        unit="percentage",
        higher_is_better=True,
        description="Prediction accuracy"
    ),
    "loss": MetricSchema(
        name="loss",
        aggregation="mean",
        unit="scalar",
        higher_is_better=False,
        description="Training loss (policy + value)"
    ),
    "fidelity_score": MetricSchema(
        name="fidelity_score",
        aggregation="mean",
        unit="scalar",
        higher_is_better=True,
        description="State space coverage fidelity (PCA/MLP reconstruction)"
    ),
    "policy_loss": MetricSchema(
        name="policy_loss",
        aggregation="mean",
        unit="scalar",
        higher_is_better=False,
        description="Policy gradient loss"
    ),
    "value_loss": MetricSchema(
        name="value_loss",
        aggregation="mean",
        unit="scalar",
        higher_is_better=False,
        description="Value function loss"
    ),
    "entropy": MetricSchema(
        name="entropy",
        aggregation="mean",
        unit="scalar",
        higher_is_better=True,
        description="Policy entropy (exploration measure)"
    ),
    "reconstruction_error": MetricSchema(
        name="reconstruction_error",
        aggregation="mean",
        unit="scalar",
        higher_is_better=False,
        description="State reconstruction error (PCA/MLP)"
    ),
}


# ============================================================================
# Artifact Path Registry
# ============================================================================

ARTIFACT_PATHS = {
    # Figures
    "figure_1": "results/figures/figure_1.png",
    "figure_2": "results/figures/figure_2.png",
    "figure_3": "results/figures/figure_3.png",
    "figure_4": "results/figures/figure_4.png",
    "figure_5": "results/figures/figure_5.png",
    "figure_6": "results/figures/figure_6.png",
    "figure_7": "results/figures/figure_7.png",
    "figure_8": "results/figures/figure_8.png",
    "result_figure": "results/figures/experiment_results.png",
    
    # Tables
    "table_1": "results/tables/table_1.csv",
    "result_table": "results/tables/experiment_results.csv",
    
    # JSON outputs
    "metrics_json": "results/metrics.json",
    "config": "results/config_resolved.json",
    "predictions": "results/predictions.jsonl",
    "evidence_contract_matrix": "results/evidence_contract_matrix.json",
    "experiment_registry": "results/experiment_registry.json",
    "artifact_manifest": "results/artifact_manifest.json",
    "sensitivity_report": "results/sensitivity_report.json",
    "readiness": "results/readiness.json",
    "evaluation_result": "results/evaluation_result.json",
}


# ============================================================================
# Paper Artifact Metadata
# ============================================================================

PAPER_ARTIFACTS = {
    "figure_1": {
        "caption": "We introduce a new class of on-policy RL algorithms that can scale to tens of thousands of parallel environments. In contrast to regular on-policy RL, such as PPO, which learns a single policy across environments leading to wasted environment capacity, our method learns diverse followers and combines data.",
        "type": "conceptual_diagram",
        "baselines": [],
        "metrics": [],
    },
    "figure_2": {
        "caption": "Performance vs batch size plot for PPO runs (blue curve) across two environments. The curve shows how PPO training runs can not take benefit of large batch size resulting from massively parallelized environments and their asymptotic performance saturates after a certain point. The dashed red line is the performance of our method.",
        "type": "performance_curve",
        "baselines": ["PPO"],
        "metrics": ["reward"],
        "x_axis": "batch_size",
        "y_axis": "performance",
    },
    "figure_3": {
        "caption": "We illustrate one particular variant of SAPG which performs well. There is one leader and M-1 followers (M=3 in figure). Each policy has the same backbone with shared parameters B_θ but is conditioned on local learned parameters φ_i. Each policy gets a block of N/M environments.",
        "type": "architecture_diagram",
        "baselines": [],
        "metrics": [],
    },
    "figure_4": {
        "caption": "Two data aggregation schemes we consider in this paper. (Left) one policy is a leader and uses data from each of the followers (Right) a symmetric scheme where each policy uses data from all others. In each case, the policy also uses its own on-policy data.",
        "type": "architecture_diagram",
        "baselines": [],
        "metrics": [],
    },
    "figure_5": {
        "caption": "Performance curves of SAPG with respect to PPO, PBT and PQL baselines. On AllegroKuka tasks, PPO and PQL barely make progress and SAPG beats PBT. On Shadow Hand and Allegro Kuka Reorientation, SAPG performs best with an entropy coefficient of 0.005 while the coefficient is 0 for other environments.",
        "type": "performance_curve",
        "baselines": ["PPO", "PBT", "PQL"],
        "metrics": ["reward", "success_rate"],
        "x_axis": "samples",
        "y_axis": "performance",
        "trend_assertion": "baseline_outperformance",
    },
    "table_1": {
        "caption": "Performance after 2e10 samples for different methods with standard error. This is measured by successes for the AllegroKuka tasks and by episode rewards for in-hand reorientation tasks. Across environments, we find that our method performs better than baselines.",
        "type": "performance_table",
        "baselines": ["PPO", "PBT", "PQL"],
        "metrics": ["success_rate", "reward"],
        "trend_assertion": "baseline_outperformance",
    },
    "figure_6": {
        "caption": "Performance curves for ablations of our method. The variants of our method with a symmetric aggregation scheme or without an off-policy combination perform significantly worse. Entropy regularization affects performance across environments, giving a benefit in reorientation. Using a high off-policy ratio works well. The blue plot is SAPG (ours), other curves are ablations: symmetric aggregation (no designated leader), no off-policy, entropy coefficient variations.",
        "type": "ablation_study",
        "baselines": ["SAPG", "symmetric_aggregation", "no_offpolicy", "entropy_0", "entropy_0.005", "entropy_0.01"],
        "metrics": ["reward", "success_rate"],
        "x_axis": "samples",
        "y_axis": "performance",
        "trend_assertion": "positive_parameter_improves",
    },
    "figure_7": {
        "caption": "Curves comparing reconstruction error for states visited during training using top-k PCA components for SAPG (Ours), PPO and a randomly initialized policy",
        "type": "analysis_curve",
        "baselines": ["SAPG", "PPO", "random"],
        "metrics": ["reconstruction_error", "fidelity_score"],
        "x_axis": "pca_components",
        "y_axis": "reconstruction_error",
    },
    "figure_8": {
        "caption": "Curves comparing reconstruction error for states visited during training using MLPs with varying hidden layer dimensions for SAPG (Ours), PPO and a randomly initialized policy. The neural network was a two layer MLP of the same size (size shown on x-axis). Activation function used was ReLU, trained with Adam optimizer using default hyperparameters from PyTorch.",
        "type": "analysis_curve",
        "baselines": ["SAPG", "PPO", "random"],
        "metrics": ["reconstruction_error", "fidelity_score"],
        "x_axis": "mlp_hidden_dim",
        "y_axis": "reconstruction_error",
    },
}


# ============================================================================
# Trend Assertions
# ============================================================================

TREND_ASSERTIONS = {
    "baseline_outperformance": {
        "description": "Proposed method (SAPG) should outperform explicit baselines (PPO, PBT, PQL)",
        "comparison": "SAPG > baseline",
        "baselines": ["PPO", "PBT", "PQL"],
        "metrics": ["reward", "success_rate"],
        "artifacts": ["figure_5", "table_1"],
    },
    "positive_parameter_improves": {
        "description": "Nonzero/positive parameter values (entropy coefficient, aggregation coefficient) should preserve reported improvement trend",
        "comparison": "parameter > 0 improves performance",
        "parameters": ["entropy_coefficient", "aggregation_coefficient"],
        "metrics": ["reward", "success_rate"],
        "artifacts": ["figure_6"],
    },
}


# ============================================================================
# Experiment Registry
# ============================================================================

EXPERIMENT_REGISTRY = {
    "sapg_main": {
        "method": "sapg",
        "description": "Main SAPG method with leader-follower aggregation",
        "config": {
            "num_policies": 6,
            "aggregation_coefficient": 1.0,
            "entropy_coefficient": 0.0,
            "shared_backbone": True,
        },
        "artifacts": ["figure_5", "table_1", "figure_7", "figure_8"],
    },
    "ppo_baseline": {
        "method": "ppo",
        "description": "Standard PPO baseline (single policy)",
        "config": {
            "num_policies": 1,
            "entropy_coefficient": 0.01,
        },
        "artifacts": ["figure_2", "figure_5", "table_1", "figure_7", "figure_8"],
    },
    "pbt_baseline": {
        "method": "pbt",
        "description": "Population-Based Training baseline",
        "config": {
            "population_size": 6,
        },
        "artifacts": ["figure_5", "table_1"],
    },
    "pql_baseline": {
        "method": "pql",
        "description": "Parallel Q-Learning baseline",
        "config": {},
        "artifacts": ["figure_5", "table_1"],
    },
    "ablation_symmetric": {
        "method": "sapg",
        "description": "SAPG ablation: symmetric aggregation (no designated leader)",
        "config": {
            "num_policies": 6,
            "aggregation_scheme": "symmetric",
            "aggregation_coefficient": 1.0,
        },
        "artifacts": ["figure_6"],
    },
    "ablation_no_offpolicy": {
        "method": "sapg",
        "description": "SAPG ablation: no off-policy data aggregation",
        "config": {
            "num_policies": 6,
            "aggregation_coefficient": 0.0,
        },
        "artifacts": ["figure_6"],
    },
    "ablation_entropy_0": {
        "method": "sapg",
        "description": "SAPG ablation: zero entropy coefficient",
        "config": {
            "num_policies": 6,
            "entropy_coefficient": 0.0,
        },
        "artifacts": ["figure_6"],
    },
    "ablation_entropy_0005": {
        "method": "sapg",
        "description": "SAPG ablation: entropy coefficient 0.005",
        "config": {
            "num_policies": 6,
            "entropy_coefficient": 0.005,
        },
        "artifacts": ["figure_6"],
    },
}


# ============================================================================
# Artifact Writers
# ============================================================================

def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists for given file path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_evidence_contract_matrix(output_path: Optional[str] = None, mode: str = "smoke") -> Dict[str, Any]:
    """
    Write evidence contract matrix mapping paper artifacts to implementation.
    
    Args:
        output_path: Override default output path
        mode: Execution mode (smoke, default, full)
    
    Returns:
        Evidence contract matrix dictionary
    """
    if output_path is None:
        output_path = ARTIFACT_PATHS["evidence_contract_matrix"]
    
    matrix = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "mode": mode,
            "paper_title": "SAPG: Split and Aggregate Policy Gradients",
            "reproduction_status": "schema_artifact" if mode == "smoke" else "experimental_results",
        },
        "metric_schemas": {k: asdict(v) for k, v in METRIC_SCHEMAS.items()},
        "artifact_paths": ARTIFACT_PATHS,
        "paper_artifacts": PAPER_ARTIFACTS,
        "trend_assertions": TREND_ASSERTIONS,
        "experiment_registry": EXPERIMENT_REGISTRY,
    }
    
    path = ensure_dir(output_path)
    with open(path, 'w') as f:
        json.dump(matrix, f, indent=2)
    
    return matrix


def write_experiment_registry(output_path: Optional[str] = None, mode: str = "smoke") -> Dict[str, Any]:
    """
    Write experiment registry with all configured experiments.
    
    Args:
        output_path: Override default output path
        mode: Execution mode (smoke, default, full)
    
    Returns:
        Experiment registry dictionary
    """
    if output_path is None:
        output_path = ARTIFACT_PATHS["experiment_registry"]
    
    registry = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "mode": mode,
            "total_experiments": len(EXPERIMENT_REGISTRY),
        },
        "experiments": EXPERIMENT_REGISTRY,
        "baselines": ["ppo", "pbt", "pql"],
        "ablations": [
            "ablation_symmetric",
            "ablation_no_offpolicy",
            "ablation_entropy_0",
            "ablation_entropy_0005",
        ],
    }
    
    path = ensure_dir(output_path)
    with open(path, 'w') as f:
        json.dump(registry, f, indent=2)
    
    return registry


def write_metrics_json(metrics: Dict[str, Any], output_path: Optional[str] = None, mode: str = "smoke") -> None:
    """
    Write metrics JSON with aggregated results.
    
    Args:
        metrics: Dictionary of metric values
        output_path: Override default output path
        mode: Execution mode (smoke, default, full)
    """
    if output_path is None:
        output_path = ARTIFACT_PATHS["metrics_json"]
    
    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "mode": mode,
            "artifact_type": "schema_artifact" if mode == "smoke" else "experimental_metrics",
        },
        "metrics": metrics,
        "schemas": {k: asdict(v) for k, v in METRIC_SCHEMAS.items()},
    }
    
    path = ensure_dir(output_path)
    with open(path, 'w') as f:
        json.dump(output, f, indent=2)


def write_artifact_manifest(mode: str = "smoke", output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Write artifact manifest listing all expected outputs.
    
    Args:
        mode: Execution mode (smoke, default, full)
        output_path: Override default output path
    
    Returns:
        Artifact manifest dictionary
    """
    if output_path is None:
        output_path = ARTIFACT_PATHS["artifact_manifest"]
    
    manifest = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "mode": mode,
            "artifact_type": "manifest",
        },
        "artifacts": {
            name: {
                "path": path,
                "exists": os.path.exists(path),
                "type": "figure" if "figure" in name else "table" if "table" in name else "json",
            }
            for name, path in ARTIFACT_PATHS.items()
        },
        "paper_artifacts": list(PAPER_ARTIFACTS.keys()),
        "experiments": list(EXPERIMENT_REGISTRY.keys()),
    }
    
    path = ensure_dir(output_path)
    with open(path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    return manifest


def write_sensitivity_report(
    sensitivity_results: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None,
    mode: str = "smoke"
) -> Dict[str, Any]:
    """
    Write sensitivity analysis report.
    
    Args:
        sensitivity_results: Sensitivity analysis results (None for smoke mode)
        output_path: Override default output path
        mode: Execution mode (smoke, default, full)
    
    Returns:
        Sensitivity report dictionary
    """
    if output_path is None:
        output_path = ARTIFACT_PATHS["sensitivity_report"]
    
    if sensitivity_results is None:
        # Smoke mode: schema artifact
        sensitivity_results = {
            "entropy_coefficient": {
                "values": [0.0, 0.005, 0.01],
                "metric": "reward",
                "trend": "positive_parameter_improves",
                "results": "pending_execution",
            },
            "aggregation_coefficient": {
                "values": [0.0, 0.25, 0.5, 0.75],
                "metric": "reward",
                "trend": "positive_parameter_improves",
                "results": "pending_execution",
            },
            "num_policies": {
                "values": [1, 2, 4, 8],
                "metric": "reward",
                "trend": "baseline_outperformance",
                "results": "pending_execution",
            },
        }
    
    report = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "mode": mode,
            "artifact_type": "schema_artifact" if mode == "smoke" else "sensitivity_analysis",
        },
        "parameters": sensitivity_results,
        "trend_assertions": TREND_ASSERTIONS,
    }
    
    path = ensure_dir(output_path)
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)
    
    return report


def write_figure_artifact(
    figure_id: str,
    data: Optional[Any] = None,
    output_path: Optional[str] = None,
    mode: str = "smoke"
) -> str:
    """
    Write figure artifact (PNG file).
    
    Args:
        figure_id: Figure identifier (e.g., "figure_1")
        data: Figure data (None for smoke mode creates schema artifact)
        output_path: Override default output path
        mode: Execution mode (smoke, default, full)
    
    Returns:
        Output file path
    """
    if output_path is None:
        output_path = ARTIFACT_PATHS.get(figure_id, f"results/figures/{figure_id}.png")
    
    path = ensure_dir(output_path)
    
    if mode == "smoke" or data is None:
        # Smoke mode: create minimal schema artifact
        try:
            import numpy as np
            # Lazy import to avoid requiring matplotlib in minimal environments
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.text(0.5, 0.5, f"Schema Artifact: {figure_id}\n(Dry-run contract validation)",
                       ha='center', va='center', fontsize=14, transform=ax.transAxes)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                
                metadata = PAPER_ARTIFACTS.get(figure_id, {})
                caption = metadata.get("caption", "")
                if caption:
                    fig.text(0.5, 0.02, f"Caption: {caption[:100]}...", 
                            ha='center', fontsize=8, wrap=True)
                
                plt.savefig(path, dpi=100, bbox_inches='tight')
                plt.close(fig)
            except ImportError:
                # Fallback: write a minimal PNG header if matplotlib unavailable
                # This creates a valid but minimal PNG file
                png_header = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
                with open(path, 'wb') as f:
                    f.write(png_header)
        except Exception:
            # Ultimate fallback: write marker file
            with open(path, 'w') as f:
                f.write(f"# Schema artifact for {figure_id} (dry-run validation)\n")
    else:
        # Real mode: write actual figure
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            if hasattr(data, 'savefig'):
                # data is a matplotlib figure
                data.savefig(path, dpi=300, bbox_inches='tight')
                plt.close(data)
            else:
                # data is raw plot data, create figure
                fig, ax = plt.subplots(figsize=(10, 6))
                # Plotting logic would go here based on data structure
                plt.savefig(path, dpi=300, bbox_inches='tight')
                plt.close(fig)
        except ImportError:
            # If matplotlib not available in real mode, write marker
            with open(path, 'w') as f:
                f.write(f"# Figure {figure_id} requires matplotlib for rendering\n")
    
    return str(path)


def write_table_artifact(
    table_id: str,
    data: Optional[Any] = None,
    output_path: Optional[str] = None,
    mode: str = "smoke"
) -> str:
    """
    Write table artifact (CSV file).
    
    Args:
        table_id: Table identifier (e.g., "table_1")
        data: Table data (None for smoke mode creates schema artifact)
        output_path: Override default output path
        mode: Execution mode (smoke, default, full)
    
    Returns:
        Output file path
    """
    if output_path is None:
        output_path = ARTIFACT_PATHS.get(table_id, f"results/tables/{table_id}.csv")
    
    path = ensure_dir(output_path)
    
    if mode == "smoke" or data is None:
        # Smoke mode: create schema artifact with expected structure
        metadata = PAPER_ARTIFACTS.get(table_id, {})
        baselines = metadata.get("baselines", ["SAPG", "PPO", "PBT", "PQL"])
        metrics = metadata.get("metrics", ["reward", "success_rate"])
        
        with open(path, 'w') as f:
            f.write("# Schema artifact for dry-run validation\n")
            f.write(f"# {metadata.get('caption', '')}\n")
            f.write("Method," + ",".join(metrics) + "\n")
            for baseline in baselines:
                f.write(f"{baseline}," + ",".join(["pending"] * len(metrics)) + "\n")
    else:
        # Real mode: write actual table data
        try:
            import pandas as pd
            if isinstance(data, pd.DataFrame):
                data.to_csv(path, index=False)
            else:
                # data is dict or list of dicts
                df = pd.DataFrame(data)
                df.to_csv(path, index=False)
        except ImportError:
            # Fallback without pandas
            with open(path, 'w') as f:
                if isinstance(data, dict):
                    headers = list(data.keys())
                    f.write(",".join(headers) + "\n")
                    # Assume all values are lists of same length
                    for i in range(len(data[headers[0]])):
                        row = [str(data[h][i]) for h in headers]
                        f.write(",".join(row) + "\n")
                else:
                    f.write(str(data))
    
    return str(path)


def write_readiness_json(mode: str = "smoke", output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Write readiness.json for smoke validation.
    
    Args:
        mode: Execution mode
        output_path: Override default output path
    
    Returns:
        Readiness status dictionary
    """
    if output_path is None:
        output_path = ARTIFACT_PATHS["readiness"]
    
    readiness = {
        "status": "ready",
        "mode": mode,
        "timestamp": datetime.now().isoformat(),
        "artifacts_declared": list(ARTIFACT_PATHS.keys()),
        "experiments_declared": list(EXPERIMENT_REGISTRY.keys()),
        "metrics_declared": list(METRIC_SCHEMAS.keys()),
        "paper_artifacts_declared": list(PAPER_ARTIFACTS.keys()),
        "trend_assertions_declared": list(TREND_ASSERTIONS.keys()),
    }
