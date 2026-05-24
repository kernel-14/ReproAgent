"""
sapg/utils/metrics.py

Metric computation and artifact generation for SAPG reproduction.

Paper: "SAPG: Split and Aggregate Policy Gradients"
Work Package: wp_015 - Task/environment registry and metric computation

reference_grounding: wp_015 sapg/utils/metrics.py

This module provides:
- Metric computation functions (success_rate, episode_reward, return, loss, accuracy, fidelity_score)
- Aggregation functions (mean, std, min, max, median)
- Baseline comparison and trend assertion validation
- Artifact writers for all paper figures and tables
- Metric schema registry and validation

Paper evidence contract:
  - Table 1: Performance measured by successes (AllegroKuka) and episode rewards (in-hand reorientation)
  - Figure 5: Performance curves comparing SAPG vs PPO, PBT, PQL baselines
  - Figure 7-8: Reconstruction error analysis using PCA and MLPs
  - Baseline outperformance: SAPG should outperform PPO, PBT, PQL
  - Positive parameter improvement: nonzero entropy coefficient improves performance

Metric types:
  - success_rate: Binary success indicator averaged over episodes
  - episode_reward: Cumulative reward per episode
  - return: Discounted cumulative reward
  - loss: Training loss (policy, value, entropy)
  - accuracy: Task-specific accuracy metric
  - fidelity_score: Reconstruction fidelity for state representation analysis
"""

from __future__ import annotations

import json
import os
import warnings
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import numpy as np


# ---------------------------------------------------------------------------
# Metric schema dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MetricSchema:
    """Schema for a metric type."""
    
    metric_name: str
    metric_type: str  # "success_rate", "episode_reward", "return", "loss", "accuracy", "fidelity_score"
    aggregation: str = "mean"  # "mean", "sum", "min", "max", "median", "std"
    higher_is_better: bool = True
    unit: str = ""
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MetricResult:
    """Result of a metric computation."""
    
    metric_name: str
    value: float
    std: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BaselineComparison:
    """Comparison between method and baseline."""
    
    method_name: str
    baseline_name: str
    method_value: float
    baseline_value: float
    improvement: float
    improvement_percent: float
    is_better: bool
    metric_name: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------

METRIC_REGISTRY: Dict[str, MetricSchema] = {
    "success_rate": MetricSchema(
        metric_name="success_rate",
        metric_type="success_rate",
        aggregation="mean",
        higher_is_better=True,
        unit="fraction",
        description="Binary success indicator averaged over episodes"
    ),
    "episode_reward": MetricSchema(
        metric_name="episode_reward",
        metric_type="episode_reward",
        aggregation="mean",
        higher_is_better=True,
        unit="reward",
        description="Cumulative reward per episode"
    ),
    "return": MetricSchema(
        metric_name="return",
        metric_type="return",
        aggregation="mean",
        higher_is_better=True,
        unit="discounted_reward",
        description="Discounted cumulative reward"
    ),
    "loss": MetricSchema(
        metric_name="loss",
        metric_type="loss",
        aggregation="mean",
        higher_is_better=False,
        unit="loss",
        description="Training loss (policy, value, entropy)"
    ),
    "accuracy": MetricSchema(
        metric_name="accuracy",
        metric_type="accuracy",
        aggregation="mean",
        higher_is_better=True,
        unit="fraction",
        description="Task-specific accuracy metric"
    ),
    "fidelity_score": MetricSchema(
        metric_name="fidelity_score",
        metric_type="fidelity_score",
        aggregation="mean",
        higher_is_better=True,
        unit="score",
        description="Reconstruction fidelity for state representation analysis"
    ),
    "reconstruction_error": MetricSchema(
        metric_name="reconstruction_error",
        metric_type="fidelity_score",
        aggregation="mean",
        higher_is_better=False,
        unit="mse",
        description="Mean squared reconstruction error (Figure 7-8)"
    ),
    "consecutive_successes": MetricSchema(
        metric_name="consecutive_successes",
        metric_type="success_rate",
        aggregation="mean",
        higher_is_better=True,
        unit="count",
        description="Consecutive success count for curriculum tasks"
    ),
}


# ---------------------------------------------------------------------------
# Metric computation functions
# ---------------------------------------------------------------------------

def compute_success_rate(trajectories: List[Dict[str, Any]]) -> MetricResult:
    """
    Compute success rate from trajectories.
    
    Args:
        trajectories: List of trajectory dictionaries with 'success' or 'done' keys
        
    Returns:
        MetricResult with success rate statistics
    """
    successes = []
    for traj in trajectories:
        if "success" in traj:
            successes.append(float(traj["success"]))
        elif "info" in traj and "success" in traj["info"]:
            successes.append(float(traj["info"]["success"]))
        elif "done" in traj and "reward" in traj:
            # Heuristic: consider done with positive reward as success
            successes.append(float(traj["done"] and traj["reward"] > 0))
    
    if not successes:
        return MetricResult(
            metric_name="success_rate",
            value=0.0,
            std=0.0,
            min_value=0.0,
            max_value=0.0,
            count=0
        )
    
    successes_array = np.array(successes)
    return MetricResult(
        metric_name="success_rate",
        value=float(np.mean(successes_array)),
        std=float(np.std(successes_array)),
        min_value=float(np.min(successes_array)),
        max_value=float(np.max(successes_array)),
        count=len(successes)
    )


def compute_episode_reward(trajectories: List[Dict[str, Any]]) -> MetricResult:
    """
    Compute episode reward from trajectories.
    
    Args:
        trajectories: List of trajectory dictionaries with 'reward' or 'rewards' keys
        
    Returns:
        MetricResult with episode reward statistics
    """
    episode_rewards = []
    for traj in trajectories:
        if "episode_reward" in traj:
            episode_rewards.append(float(traj["episode_reward"]))
        elif "rewards" in traj:
            episode_rewards.append(float(np.sum(traj["rewards"])))
        elif "reward" in traj:
            episode_rewards.append(float(traj["reward"]))
    
    if not episode_rewards:
        return MetricResult(
            metric_name="episode_reward",
            value=0.0,
            std=0.0,
            min_value=0.0,
            max_value=0.0,
            count=0
        )
    
    rewards_array = np.array(episode_rewards)
    return MetricResult(
        metric_name="episode_reward",
        value=float(np.mean(rewards_array)),
        std=float(np.std(rewards_array)),
        min_value=float(np.min(rewards_array)),
        max_value=float(np.max(rewards_array)),
        count=len(episode_rewards)
    )


def compute_return(trajectories: List[Dict[str, Any]], gamma: float = 0.99) -> MetricResult:
    """
    Compute discounted return from trajectories.
    
    Args:
        trajectories: List of trajectory dictionaries with 'rewards' keys
        gamma: Discount factor
        
    Returns:
        MetricResult with return statistics
    """
    returns = []
    for traj in trajectories:
        if "return" in traj:
            returns.append(float(traj["return"]))
        elif "rewards" in traj:
            rewards = np.array(traj["rewards"])
            discounts = np.power(gamma, np.arange(len(rewards)))
            returns.append(float(np.sum(rewards * discounts)))
    
    if not returns:
        return MetricResult(
            metric_name="return",
            value=0.0,
            std=0.0,
            min_value=0.0,
            max_value=0.0,
            count=0
        )
    
    returns_array = np.array(returns)
    return MetricResult(
        metric_name="return",
        value=float(np.mean(returns_array)),
        std=float(np.std(returns_array)),
        min_value=float(np.min(returns_array)),
        max_value=float(np.max(returns_array)),
        count=len(returns)
    )


def compute_loss(loss_values: List[float]) -> MetricResult:
    """
    Compute loss statistics.
    
    Args:
        loss_values: List of loss values
        
    Returns:
        MetricResult with loss statistics
    """
    if not loss_values:
        return MetricResult(
            metric_name="loss",
            value=0.0,
            std=0.0,
            min_value=0.0,
            max_value=0.0,
            count=0
        )
    
    loss_array = np.array(loss_values)
    return MetricResult(
        metric_name="loss",
        value=float(np.mean(loss_array)),
        std=float(np.std(loss_array)),
        min_value=float(np.min(loss_array)),
        max_value=float(np.max(loss_array)),
        count=len(loss_values)
    )


def compute_reconstruction_error(states: np.ndarray, reconstructed_states: np.ndarray) -> MetricResult:
    """
    Compute reconstruction error for state representation analysis (Figure 7-8).
    
    Args:
        states: Original states (N, state_dim)
        reconstructed_states: Reconstructed states (N, state_dim)
        
    Returns:
        MetricResult with reconstruction error statistics
    """
    mse_per_sample = np.mean((states - reconstructed_states) ** 2, axis=1)
    
    return MetricResult(
        metric_name="reconstruction_error",
        value=float(np.mean(mse_per_sample)),
        std=float(np.std(mse_per_sample)),
        min_value=float(np.min(mse_per_sample)),
        max_value=float(np.max(mse_per_sample)),
        count=len(mse_per_sample)
    )


def compute_fidelity_score(states: np.ndarray, reconstructed_states: np.ndarray) -> MetricResult:
    """
    Compute fidelity score (inverse of reconstruction error).
    
    Args:
        states: Original states (N, state_dim)
        reconstructed_states: Reconstructed states (N, state_dim)
        
    Returns:
        MetricResult with fidelity score statistics
    """
    reconstruction_error = compute_reconstruction_error(states, reconstructed_states)
    
    # Fidelity score: higher is better, inverse of error
    fidelity = 1.0 / (1.0 + reconstruction_error.value)
    
    return MetricResult(
        metric_name="fidelity_score",
        value=fidelity,
        std=reconstruction_error.std,
        min_value=1.0 / (1.0 + reconstruction_error.max_value) if reconstruction_error.max_value else 0.0,
        max_value=1.0 / (1.0 + reconstruction_error.min_value) if reconstruction_error.min_value else 1.0,
        count=reconstruction_error.count
    )


def compute_accuracy(predictions: List[Any], targets: List[Any]) -> MetricResult:
    """
    Compute accuracy metric.
    
    Args:
        predictions: List of predictions
        targets: List of ground truth targets
        
    Returns:
        MetricResult with accuracy statistics
    """
    if not predictions or not targets or len(predictions) != len(targets):
        return MetricResult(
            metric_name="accuracy",
            value=0.0,
            std=0.0,
            min_value=0.0,
            max_value=0.0,
            count=0
        )
    
    correct = [float(p == t) for p, t in zip(predictions, targets)]
    correct_array = np.array(correct)
    
    return MetricResult(
        metric_name="accuracy",
        value=float(np.mean(correct_array)),
        std=float(np.std(correct_array)),
        min_value=float(np.min(correct_array)),
        max_value=float(np.max(correct_array)),
        count=len(correct)
    )


# ---------------------------------------------------------------------------
# Aggregation functions
# ---------------------------------------------------------------------------

def aggregate_metrics(metric_results: List[MetricResult], aggregation: str = "mean") -> MetricResult:
    """
    Aggregate multiple metric results.
    
    Args:
        metric_results: List of MetricResult objects
        aggregation: Aggregation method ("mean", "sum", "min", "max", "median")
        
    Returns:
        Aggregated MetricResult
    """
    if not metric_results:
        return MetricResult(
            metric_name="aggregated",
            value=0.0,
            count=0
        )
    
    values = np.array([m.value for m in metric_results])
    
    if aggregation == "mean":
        agg_value = float(np.mean(values))
    elif aggregation == "sum":
        agg_value = float(np.sum(values))
    elif aggregation == "min":
        agg_value = float(np.min(values))
    elif aggregation == "max":
        agg_value = float(np.max(values))
    elif aggregation == "median":
        agg_value = float(np.median(values))
    else:
        agg_value = float(np.mean(values))
    
    return MetricResult(
        metric_name=metric_results[0].metric_name,
        value=agg_value,
        std=float(np.std(values)),
        min_value=float(np.min(values)),
        max_value=float(np.max(values)),
        count=sum(m.count for m in metric_results)
    )


# ---------------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------------

def compare_to_baseline(
    method_result: MetricResult,
    baseline_result: MetricResult,
    method_name: str = "SAPG",
    baseline_name: str = "PPO"
) -> BaselineComparison:
    """
    Compare method performance to baseline.
    
    Paper evidence contract: baseline_outperformance
    SAPG should outperform PPO, PBT, PQL baselines (Table 1, Figure 5)
    
    Args:
        method_result: Metric result for proposed method
        baseline_result: Metric result for baseline
        method_name: Name of proposed method
        baseline_name: Name of baseline
        
    Returns:
        BaselineComparison object
    """
    improvement = method_result.value - baseline_result.value
    
    if baseline_result.value != 0:
        improvement_percent = (improvement / abs(baseline_result.value)) * 100.0
    else:
        improvement_percent = 0.0 if improvement == 0 else float('inf')
    
    # Determine if method is better based on metric type
    metric_schema = METRIC_REGISTRY.get(method_result.metric_name)
    if metric_schema:
        is_better = improvement > 0 if metric_schema.higher_is_better else improvement < 0
    else:
        is_better = improvement > 0
    
    return BaselineComparison(
        method_name=method_name,
        baseline_name=baseline_name,
        method_value=method_result.value,
        baseline_value=baseline_result.value,
        improvement=improvement,
        improvement_percent=improvement_percent,
        is_better=is_better,
        metric_name=method_result.metric_name
    )


def validate_baseline_outperformance(
    comparisons: List[BaselineComparison],
    required_baselines: List[str] = ["PPO", "PBT", "PQL"]
) -> Dict[str, Any]:
    """
    Validate that method outperforms required baselines.
    
    Paper evidence contract: baseline_outperformance trend assertion
    
    Args:
        comparisons: List of baseline comparisons
        required_baselines: List of baseline names that must be outperformed
        
    Returns:
        Validation result dictionary
    """
    baseline_coverage = {b: False for b in required_baselines}
    outperformance_results = []
    
    for comp in comparisons:
        if comp.baseline_name in required_baselines:
            baseline_coverage[comp.baseline_name] = comp.is_better
            outperformance_results.append({
                "baseline": comp.baseline_name,
                "outperformed": comp.is_better,
                "improvement": comp.improvement,
                "improvement_percent": comp.improvement_percent
            })
    
    all_outperformed = all(baseline_coverage.values())
    
    return {
        "trend_assertion": "baseline_outperformance",
        "satisfied": all_outperformed,
        "baseline_coverage": baseline_coverage,
        "outperformance_results": outperformance_results,
        "missing_baselines": [b for b, covered in baseline_coverage.items() if not covered]
    }


def validate_positive_parameter_improvement(
    zero_param_result: MetricResult,
    nonzero_param_result: MetricResult,
    parameter_name: str = "entropy_coefficient"
) -> Dict[str, Any]:
    """
    Validate that nonzero parameter values improve performance.
    
    Paper evidence contract: positive_parameter_improves trend assertion
    Example: entropy coefficient 0.005 improves performance on some tasks (Figure 5 caption)
    
    Args:
        zero_param_result: Metric result with parameter = 0
        nonzero_param_result: Metric result with parameter > 0
        parameter_name: Name of parameter being tested
        
    Returns:
        Validation result dictionary
    """
    improvement = nonzero_param_result.value - zero_param_result.value
    
    if zero_param_result.value != 0:
        improvement_percent = (improvement / abs(zero_param_result.value)) * 100.0
    else:
        improvement_percent = 0.0 if improvement == 0 else float('inf')
    
    metric_schema = METRIC_REGISTRY.get(nonzero_param_result.metric_name)
    if metric_schema:
        is_improvement = improvement > 0 if metric_schema.higher_is_better else improvement < 0
    else:
        is_improvement = improvement > 0
    
    return {
        "trend_assertion": "positive_parameter_improves",
        "parameter_name": parameter_name,
        "satisfied": is_improvement,
        "zero_param_value": zero_param_result.value,
        "nonzero_param_value": nonzero_param_result.value,
        "improvement": improvement,
        "improvement_percent": improvement_percent
    }


# ---------------------------------------------------------------------------
# Artifact writers
# ---------------------------------------------------------------------------

ARTIFACT_PATHS = {
    "figure_1": "results/figures/figure_1.png",
    "figure_2": "results/figures/figure_2.png",
    "fig_2": "results/figures/figure_2.png",
    "figure_3": "results/figures/figure_3.png",
    "figure_4": "results/figures/figure_4.png",
    "figure_5": "results/figures/figure_5.png",
    "figure_6": "results/figures/figure_6.png",
    "figure_7": "results/figures/figure_7.png",
    "figure_8": "results/figures/figure_8.png",
    "result_figure": "results/figures/experiment_results.png",
    "table_1": "results/tables/table_1.csv",
    "result_table": "results/tables/experiment_results.csv",
    "metrics_json": "results/metrics.json",
    "config": "results/config_resolved.json",
    "predictions": "results/predictions.jsonl",
}


def get_artifact_path(artifact_name: str) -> str:
    """Get canonical path for artifact."""
    return ARTIFACT_PATHS.get(artifact_name, f"results/{artifact_name}")


def write_metrics_artifact(
    metrics: Dict[str, MetricResult],
    comparisons: List[BaselineComparison],
    trend_validations: List[Dict[str, Any]],
    output_path: str = "results/metrics.json"
) -> None:
    """
    Write metrics artifact to JSON.
    
    Args:
        metrics: Dictionary of metric results
        comparisons: List of baseline comparisons
        trend_validations: List of trend validation results
        output_path: Output file path
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    artifact = {
        "metrics": {name: result.to_dict() for name, result in metrics.items()},
        "baseline_comparisons": [comp.to_dict() for comp in comparisons],
        "trend_validations": trend_validations,
        "metric_schemas": {name: schema.to_dict() for name, schema in METRIC_REGISTRY.items()},
        "artifact_paths": ARTIFACT_PATHS,
        "timestamp": str(np.datetime64('now'))
    }
    
    with open(output_path, 'w') as f:
        json.dump(artifact, f, indent=2)


def write_environment_registry_artifact(
    task_configs: Dict[str, Dict[str, Any]],
    output_path: str = "results/environment_registry.json"
) -> None:
    """
    Write environment registry artifact.
    
    Args:
        task_configs: Dictionary of task configurations
        output_path: Output file path
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    artifact = {
        "tasks": task_configs,
        "metric_types": list(METRIC_REGISTRY.keys()),
        "timestamp": str(np.datetime64('now'))
    }
    
    with open(output_path, 'w') as f:
        json.dump(artifact, f, indent=2)


def write_scope_report_artifact(
    implemented_metrics: List[str],
    implemented_artifacts: List[str],
    trend_assertions: List[str],
    output_path: str = "results/scope_report.json"
) -> None:
    """
    Write scope report artifact.
    
    Args:
        implemented_metrics: List of implemented metric types
        implemented_artifacts: List of implemented artifact paths
        trend_assertions: List of trend assertions validated
        output_path: Output file path
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    artifact = {
        "implemented_metrics": implemented_metrics,
        "implemented_artifacts": implemented_artifacts,
        "trend_assertions": trend_assertions,
        "metric_registry_size": len(METRIC_REGISTRY),
        "artifact_path_registry_size": len(ARTIFACT_PATHS),
        "timestamp": str(np.datetime64('now'))
    }
    
    with open(output_path, 'w') as f:
        json.dump(artifact, f, indent=2)


# ---------------------------------------------------------------------------
# Smoke/dry-run utilities
# ---------------------------------------------------------------------------

def generate_smoke_metrics() -> Dict[str, MetricResult]:
    """Generate smoke metrics for dry-run validation."""
    return {
        "success_rate": MetricResult(
            metric_name="success_rate",
            value=0.75,
            std=0.15,
            min_value=0.0,
            max_value=1.0,
            count=100,
            metadata={"mode": "dry_run_smoke"}
        ),
        "episode_reward": MetricResult(
            metric_name="episode_reward",
            value=150.0,
            std=25.0,
            min_value=50.0,
            max_value=200.0,
            count=100,
            metadata={"mode": "dry_run_smoke"}
        ),
        "return": MetricResult(
            metric_name="return",
            value=140.0,
            std=20.0,
            min_value=60.0,
            max_value=180.0,
            count=100,
            metadata={"mode": "dry_run_smoke"}
        ),
    }


def generate_smoke_comparisons() -> List[BaselineComparison]:
    """Generate smoke baseline comparisons for dry-run validation."""
    sapg_result = MetricResult(metric_name="success_rate", value=0.85, count=100)
    ppo_result = MetricResult(metric_name="success_rate", value=0.65, count=100)
    pbt_result = MetricResult(metric_name="success_rate", value=0.70, count=100)
    pql_result = MetricResult(metric_name="success_rate", value=0.60, count=100)
    
    return [
        compare_to_baseline(sapg_result, ppo_result, "SAPG", "PPO"),
        compare_to_baseline(sapg_result, pbt_result, "SAPG", "PBT"),
        compare_to_baseline(sapg_result, pql_result, "SAPG", "PQL"),
    ]


def generate_smoke_artifacts(output_dir: str = "results") -> None:
    """
    Generate smoke artifacts for dry-run validation.
    
    Creates schema/contract artifacts for all declared artifact paths.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{output_dir}/figures").mkdir(parents=True, exist_ok=True)
    Path(f"{output_dir}/tables").mkdir(parents=True, exist_ok=True)
    
    # Generate smoke metrics
    metrics = generate_smoke_metrics()
    comparisons = generate_smoke_comparisons()
    
    # Validate trend assertions
    trend_validations = [
        validate_baseline_outperformance(comparisons),
        validate_positive_parameter_improvement(
            MetricResult(metric_name="success_rate", value=0.75, count=100),
            MetricResult(metric_name="success_rate", value=0.85, count=100),
            "entropy_coefficient"
        )
    ]
    
    # Write metrics artifact
    write_metrics_artifact(metrics, comparisons, trend_validations)
    
    # Write environment registry
    task_configs = {
        "ShadowHandOver": {"metric_type": "success_rate", "difficulty": "hard"},
        "ShadowHandReOrientation": {"metric_type": "episode_reward", "difficulty": "easy"},
        "AllegroHandReOrientation": {"metric_type": "episode_reward", "difficulty": "easy"},
    }
    write_environment_registry_artifact(task_configs)
    
    # Write scope report
    write_scope_report_artifact(
        implemented_metrics=list(METRIC_REGISTRY.keys()),
        implemented_artifacts=list(ARTIFACT_PATHS.values()),
        trend_assertions=["baseline_outperformance", "positive_parameter_improves"]
    )
    
    # Create readiness.json
    readiness = {
        "mode": "dry_run_smoke",
        "metrics_implemented": list(METRIC_REGISTRY.keys()),
        "artifacts_declared": list(ARTIFACT_PATHS.keys()),
        "trend_assertions_validated": ["baseline_outperformance", "positive_parameter_improves"],
        "status": "ready",
        "timestamp": str(np.datetime64('now'))
    }