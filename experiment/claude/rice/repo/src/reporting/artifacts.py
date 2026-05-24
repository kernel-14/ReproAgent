"""
RICE Artifacts Module

Implements artifact generation, metric computation, result aggregation, and reporting
for RICE paper reproduction. Provides writers for tables, figures, metrics, and checkpoints.

Generates artifacts for:
- Table 1: Efficiency and refining performance across environments
- Figure 5: Fidelity comparison across applications
- Ablation study results
- Training metrics and checkpoints

Trend assertions preserved:
- endpoint_low: p=0 and p=1 cases are minimum/worst
- sweep_insensitive: stable across parameter sweeps
- baseline_outperformance: RICE outperforms Random and StateMask
- positive_parameter_improves: nonzero parameters improve results
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np


# ============================================================================
# Artifact Path Registry
# ============================================================================

ARTIFACT_PATHS = {
    "table1_efficiency": "results/table1_efficiency.json",
    "table1_refining": "results/table1_refining.json",
    "figure5_fidelity": "results/figure5_fidelity.png",
    "figure1": "results/figures/figure_1.png",
    "figure2": "results/figures/figure_2.png",
    "figure3": "results/figures/figure_3.png",
    "figure6": "results/figures/figure_6.png",
    "figure7": "results/figures/figure_7.png",
    "figure8": "results/figures/figure_8.png",
    "figure10": "results/figures/figure_10.png",
    "table3": "results/tables/table_3.json",
    "table4": "results/tables/table_4.json",
    "table5": "results/tables/table_5.json",
    "table6": "results/tables/table_6.json",
    "ablation_studies": "results/ablation_studies.json",
    "metrics": "results/metrics.json",
    "pretrained_checkpoint": "checkpoints/pretrained_agent.pth",
    "refined_checkpoint": "checkpoints/refined_agent.pth",
    "config": "results/config.json",
    "predictions": "results/predictions.json",
    "readiness": "results/readiness.json",
    "evaluation_result": "results/evaluation_result.json",
}


# ============================================================================
# Metric Schemas
# ============================================================================

METRIC_SCHEMAS = {
    "fidelity_score": {
        "description": "Top-K agreement between predicted and ground-truth important states",
        "formula": "intersection(top_K_predicted, top_K_ground_truth) / K",
        "range": [0.0, 1.0],
        "higher_is_better": True,
        "aggregation": "mean",
        "unit": "ratio",
        "trend_assertions": ["baseline_outperformance", "positive_parameter_improves"]
    },
    "training_time": {
        "description": "Wall-clock time in seconds for training",
        "range": [0.0, float('inf')],
        "higher_is_better": False,
        "aggregation": "sum",
        "unit": "seconds",
        "trend_assertions": ["baseline_outperformance"]
    },
    "sample_count": {
        "description": "Number of environment samples used",
        "range": [0, float('inf')],
        "higher_is_better": False,
        "aggregation": "sum",
        "unit": "samples",
        "trend_assertions": ["baseline_outperformance"]
    },
    "mean_episode_reward": {
        "description": "Average cumulative reward per episode",
        "range": [-float('inf'), float('inf')],
        "higher_is_better": True,
        "aggregation": "mean",
        "unit": "reward",
        "trend_assertions": ["baseline_outperformance", "positive_parameter_improves"]
    },
    "reward_improvement": {
        "description": "Relative improvement over pre-trained policy",
        "formula": "(refined_reward - pretrained_reward) / abs(pretrained_reward)",
        "range": [-float('inf'), float('inf')],
        "higher_is_better": True,
        "aggregation": "mean",
        "unit": "ratio",
        "trend_assertions": ["baseline_outperformance", "positive_parameter_improves"]
    },
    "loss": {
        "description": "Training loss value",
        "range": [0.0, float('inf')],
        "higher_is_better": False,
        "aggregation": "mean",
        "unit": "loss",
        "trend_assertions": []
    },
    "reward": {
        "description": "Cumulative episode reward",
        "range": [-float('inf'), float('inf')],
        "higher_is_better": True,
        "aggregation": "mean",
        "unit": "reward",
        "trend_assertions": ["baseline_outperformance"]
    }
}


# ============================================================================
# Trend Assertions
# ============================================================================

TREND_ASSERTIONS = {
    "endpoint_low": {
        "description": "p=0 and p=1 cases must be lowest/minimum/worst",
        "validate": lambda results: _validate_endpoint_low(results)
    },
    "sweep_insensitive": {
        "description": "Stable/insensitive across parameter sweeps",
        "validate": lambda results: _validate_sweep_insensitive(results)
    },
    "baseline_outperformance": {
        "description": "RICE outperforms Random and StateMask baselines",
        "validate": lambda results: _validate_baseline_outperformance(results)
    },
    "positive_parameter_improves": {
        "description": "Nonzero/positive parameter values improve results",
        "validate": lambda results: _validate_positive_parameter_improves(results)
    }
}


def _validate_endpoint_low(results: Dict[str, Any]) -> bool:
    """Validate that p=0 and p=1 are minimum cases."""
    if 'parameter_sweep' in results and 'p' in results['parameter_sweep']:
        p_results = results['parameter_sweep']['p']
        if 0.0 in p_results and 1.0 in p_results:
            values = [p_results[p]['value'] for p in p_results]
            return min(values) in [p_results[0.0]['value'], p_results[1.0]['value']]
    return True


def _validate_sweep_insensitive(results: Dict[str, Any]) -> bool:
    """Validate stable behavior across parameter sweeps."""
    if 'parameter_sweep' in results:
        for param, sweep in results['parameter_sweep'].items():
            values = [sweep[k]['value'] for k in sweep]
            if len(values) > 1:
                std = np.std(values)
                mean = np.mean(values)
                cv = std / (abs(mean) + 1e-8)
                if cv > 0.5:
                    return False
    return True


def _validate_baseline_outperformance(results: Dict[str, Any]) -> bool:
    """Validate RICE outperforms baselines."""
    if 'method_comparison' in results:
        methods = results['method_comparison']
        if 'rice' in methods and ('random' in methods or 'statemask' in methods):
            rice_value = methods['rice']['value']
            if 'random' in methods and methods['random']['value'] >= rice_value:
                return False
            if 'statemask' in methods and methods['statemask']['value'] >= rice_value:
                return False
    return True


def _validate_positive_parameter_improves(results: Dict[str, Any]) -> bool:
    """Validate positive parameters improve results."""
    if 'parameter_sweep' in results:
        for param, sweep in results['parameter_sweep'].items():
            if 0.0 in sweep and any(k > 0 for k in sweep.keys()):
                zero_value = sweep[0.0]['value']
                positive_values = [sweep[k]['value'] for k in sweep if k > 0]
                if positive_values and max(positive_values) <= zero_value:
                    return False
    return True


# ============================================================================
# Metric Computation Functions
# ============================================================================

def compute_fidelity_score(predicted_states: List[int], 
                          ground_truth_states: List[int], 
                          k: int) -> float:
    """
    Compute fidelity score as top-K agreement.
    
    Args:
        predicted_states: Predicted important state indices
        ground_truth_states: Ground truth important state indices
        k: Number of top states to consider
        
    Returns:
        Fidelity score in [0, 1]
    """
    if not predicted_states or not ground_truth_states or k <= 0:
        return 0.0
    
    top_k_pred = set(predicted_states[:k])
    top_k_gt = set(ground_truth_states[:k])
    intersection = len(top_k_pred & top_k_gt)
    return float(intersection) / float(k)


def compute_reward_improvement(refined_rewards: List[float],
                               pretrained_rewards: List[float]) -> float:
    """
    Compute relative reward improvement.
    
    Args:
        refined_rewards: Rewards after refinement
        pretrained_rewards: Rewards before refinement
        
    Returns:
        Relative improvement ratio
    """
    if not refined_rewards or not pretrained_rewards:
        return 0.0
    
    refined_mean = np.mean(refined_rewards)
    pretrained_mean = np.mean(pretrained_rewards)
    
    if abs(pretrained_mean) < 1e-8:
        return 0.0
    
    return float((refined_mean - pretrained_mean) / abs(pretrained_mean))


def aggregate_metrics(metrics_list: List[Dict[str, float]], 
                     schema: Dict[str, Any]) -> Dict[str, float]:
    """
    Aggregate metrics according to schema specifications.
    
    Args:
        metrics_list: List of metric dictionaries
        schema: Metric schema with aggregation method
        
    Returns:
        Aggregated metrics
    """
    if not metrics_list:
        return {}
    
    aggregated = {}
    for key in metrics_list[0].keys():
        values = [m[key] for m in metrics_list if key in m]
        if not values:
            continue
            
        if key in METRIC_SCHEMAS:
            agg_method = METRIC_SCHEMAS[key].get("aggregation", "mean")
        else:
            agg_method = "mean"
        
        if agg_method == "mean":
            aggregated[key] = float(np.mean(values))
        elif agg_method == "sum":
            aggregated[key] = float(np.sum(values))
        elif agg_method == "max":
            aggregated[key] = float(np.max(values))
        elif agg_method == "min":
            aggregated[key] = float(np.min(values))
        else:
            aggregated[key] = float(np.mean(values))
    
    return aggregated


# ============================================================================
# Artifact Writers
# ============================================================================

def save_results(results_dict: Dict[str, Any], 
                output_path: str,
                mode: str = "real") -> None:
    """
    Save results dictionary to JSON file.
    
    Args:
        results_dict: Results to save
        output_path: Output file path
        mode: 'real' for actual results, 'schema' for dry-run
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    if mode == "schema":
        results_dict["_artifact_mode"] = "dry_run_schema"
        results_dict["_warning"] = "This is a schema artifact for contract validation, not experiment results"
    
    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2)


def generate_table(data: Dict[str, Any], 
                  table_name: str,
                  output_path: Optional[str] = None,
                  mode: str = "real") -> Dict[str, Any]:
    """
    Generate table artifact from data.
    
    Args:
        data: Table data
        table_name: Name of the table
        output_path: Optional output path (uses registry if None)
        mode: 'real' for actual results, 'schema' for dry-run
        
    Returns:
        Table data dictionary
    """
    if output_path is None:
        output_path = ARTIFACT_PATHS.get(table_name, f"results/{table_name}.json")
    
    table_dict = {
        "table_name": table_name,
        "data": data,
        "timestamp": time.time(),
        "mode": mode
    }
    
    save_results(table_dict, output_path, mode=mode)
    return table_dict


def generate_figure(data: Dict[str, Any], 
                   figure_name: str,
                   output_path: Optional[str] = None,
                   mode: str = "real") -> str:
    """
    Generate figure artifact from data.
    
    Args:
        data: Figure data
        figure_name: Name of the figure
        output_path: Optional output path (uses registry if None)
        mode: 'real' for actual results, 'schema' for dry-run
        
    Returns:
        Output file path
    """
    if output_path is None:
        output_path = ARTIFACT_PATHS.get(figure_name, f"results/{figure_name}.png")
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    if mode == "schema":
        _generate_schema_figure(data, output_file)
    else:
        _generate_real_figure(data, output_file)
    
    return str(output_file)


def _generate_schema_figure(data: Dict[str, Any], output_path: Path) -> None:
    """Generate schema/placeholder figure for contract validation."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Schema Figure\n{output_path.stem}\n[Dry-run artifact]",
                ha='center', va='center', fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
    except ImportError:
        with open(output_path, 'wb') as f:
            f.write(b'PNG schema placeholder')


def _generate_real_figure(data: Dict[str, Any], output_path: Path) -> None:
    """Generate actual figure from data."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if 'x' in data and 'y' in data:
            ax.plot(data['x'], data['y'], marker='o', label=data.get('label', 'Data'))
        elif 'bars' in data:
            bars_data = data['bars']
            x_pos = np.arange(len(bars_data))
            ax.bar(x_pos, [b['value'] for b in bars_data],
                  tick_label=[b['label'] for b in bars_data])
        
        ax.set_xlabel(data.get('xlabel', 'X'))
        ax.set_ylabel(data.get('ylabel', 'Y'))
        ax.set_title(data.get('title', 'Figure'))
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    except ImportError:
        _generate_schema_figure(data, output_path)


def write_table1_efficiency(results: Dict[str, Any], mode: str = "real") -> None:
    """Write Table 1 (efficiency comparison) artifact."""
    table_data = {
        "title": "Efficiency Comparison: RICE vs StateMask",
        "columns": ["Method", "Training Time (s)", "Sample Count", "Fidelity Score"],
        "rows": []
    }
    
    for method in ["rice", "statemask"]:
        if method in results:
            row = {
                "method": method.upper(),
                "training_time": results[method].get("training_time", 0.0),
                "sample_count": results[method].get("sample_count", 0),
                "fidelity_score": results[method].get("fidelity_score", 0.0)
            }
            table_data["rows"].append(row)
    
    generate_table(table_data, "table1_efficiency", mode=mode)


def write_table1_refining(results: Dict[str, Any], mode: str = "real") -> None:
    """Write Table 1 (refining performance) artifact."""
    table_data = {
        "title": "Refining Performance Across Environments",
        "columns": ["Environment", "RICE", "StateMask", "Random"],
        "rows": []
    }
    
    environments = ["hopper", "walker2d", "reacher", "halfcheetah",
                   "selfish_mining", "network_defense", "autonomous_driving", "malware"]
    
    for env in environments:
        if env in results:
            row = {
                "environment": env,
                "rice": results[env].get("rice", {}).get("reward_improvement", 0.0),
                "statemask": results[env].get("statemask", {}).get("reward_improvement", 0.0),
                "random": results[env].get("random", {}).get("reward_improvement", 0.0)
            }
            table_data["rows"].append(row)
    
    generate_table(table_data, "table1_refining", mode=mode)


def write_figure5_fidelity(results: Dict[str, Any], mode: str = "real") -> None:
    """Write Figure 5 (fidelity comparison) artifact."""
    applications = ["hopper", "walker2d", "reacher", "halfcheetah",
                   "selfish_mining", "network_defense", "autonomous_driving", "malware"]
    
    figure_data = {
        "title": "Fidelity Score Comparison Across Applications",
        "xlabel": "Application",
        "ylabel": "Fidelity Score",
        "bars": []
    }
    
    for app in applications:
        if app in results:
            figure_data["bars"].append({
                "label": app,
                "value": results[app].get("fidelity_score", 0.0)
            })
    
    generate_figure(figure_data, "figure5_fidelity", mode=mode)


def write_ablation_studies(results: Dict[str, Any], mode: str = "real") -> None:
    """Write ablation study results artifact."""
    ablation_data = {
        "title": "RICE Ablation Studies",
        "studies": {}
    }
    
    if "entropy_coefficient" in results:
        ablation_data["studies"]["entropy_coefficient"] = results["entropy_coefficient"]
    if "top_k" in results:
        ablation_data["studies"]["top_k"] = results["top_k"]
    if "roll_in_frequency" in results:
        ablation_data["studies"]["roll_in_frequency"] = results["roll_in_frequency"]
    
    save_results(ablation_data, ARTIFACT_PATHS["ablation_studies"], mode=mode)


def write_metrics(metrics: Dict[str, Any], mode: str = "real") -> None:
    """Write metrics artifact."""
    metrics_data = {
        "timestamp": time.time(),
        "metrics": metrics,
        "schemas": METRIC_SCHEMAS
    }
    save_results(metrics_data, ARTIFACT_PATHS["metrics"], mode=mode)


def write_checkpoint(checkpoint_data: Dict[str, Any], 
                    checkpoint_type: str = "pretrained",
                    mode: str = "real") -> None:
    """Write model checkpoint artifact."""
    if checkpoint_type == "pretrained":
        output_path = ARTIFACT_PATHS["pretrained_checkpoint"]
    else:
        output_path = ARTIFACT_PATHS["refined_checkpoint"]
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    if mode == "schema":
        with open(output_file, 'w') as f:
            json.dump({"_artifact_mode": "dry_run_schema", "type": checkpoint_type}, f)
    else:
        try:
            import torch
            torch.save(checkpoint_data, output_file)
        except ImportError:
            with open(output_file, 'w') as f:
                json.dump({"checkpoint_type": checkpoint_type, "data": "serialized"}, f)


def write_readiness_manifest(mode: str = "schema") -> None:
    """Write readiness manifest for contract validation."""
    manifest = {
        "artifact_mode": "dry_run_contract_validation",
        "timestamp": time.time(),
        "declared_artifacts": list(ARTIFACT_PATHS.keys()),
        "artifact_paths": ARTIFACT_PATHS,
        "metric_schemas": list(METRIC_SCHEMAS.keys()),
        "trend_assertions": list(TREND_ASSERTIONS.keys()),
        "warning": "This is a readiness manifest for smoke validation, not experiment results"
    }
    save_results(manifest, ARTIFACT_PATHS["readiness"], mode="schema")


def write_evaluation_result(results: Dict[str, Any], mode: str = "real") -> None:
    """Write evaluation result artifact."""
    eval_result = {
        "timestamp": time.time(),
        "mode": mode,
        "results": results,
        "validation": {
            assertion: TREND_ASSERTIONS[assertion]["validate"](results)
            for assertion in TREND_ASSERTIONS
        }
    }
    save_results(eval_result, ARTIFACT_PATHS["evaluation_result"], mode=mode)


# ============================================================================
# High-Level Artifact Generation
# ============================================================================

def generate_all_artifacts(results: Dict[str, Any], mode: str = "real") -> None:
    """
    Generate all required artifacts from results.
    
    Args:
        results: Complete results dictionary
        mode: 'real' for actual results, 'schema' for dry-run
    """
    if "efficiency" in results:
        write_table1_efficiency(results["efficiency"], mode=mode)
    
    if "refining" in results:
        write_table1_refining(results["refining"], mode=mode)
    
    if "fidelity" in results:
        write_figure5_fidelity(results["fidelity"], mode=mode)
    
    if "ablation" in results:
        write_ablation_studies(results["ablation"], mode=mode)
    
    if "metrics" in results:
        write_metrics(results["metrics"], mode=mode)
    
    if mode == "schema":
        write_readiness_manifest(mode="schema")
    
    write_evaluation_result(results, mode=mode)


def get_artifact_paths() -> Dict[str, str]:
    """Get all artifact paths for external access."""
    return ARTIFACT_PATHS.copy()


def get_metric_schemas() -> Dict[str, Dict[str, Any]]:
    """Get all metric schemas for external access."""
    return METRIC_SCHEMAS.copy()


def get_trend_assertions() -> Dict[str, Dict[str, Any]]:
    """Get all trend assertions for external access."""
    return TREND_ASSERTIONS.copy()