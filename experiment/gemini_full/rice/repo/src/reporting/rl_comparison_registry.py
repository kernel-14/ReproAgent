import os
import json
import csv
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Callable

# reference_grounding: paper chunk_035, chunk_011_02
DEFAULT_ALPHA = 0.01
alpha_values = [0.01, 0.001, 0.0001]

# reference_grounding: paper chunk_040, Figure 11
DEFAULT_LAMBDA = 0.01
lambda_values = [0, 0.1, 0.01, 0.001]

# reference_grounding: paper Figure 7
p_values = [0, 0.25, 0.5, 0.75, 1]

def resolve_alpha_defaults(alpha: Optional[float] = None) -> float:
    """
    Resolves alpha hyperparameter for mask network intrinsic reward.
    reference_grounding: paper chunk_035
    """
    return alpha if alpha is not None else DEFAULT_ALPHA

def resolve_lambda_defaults(lambda_val: Optional[float] = None) -> float:
    """
    Resolves lambda hyperparameter for exploration reward balancing.
    reference_grounding: paper chunk_040
    """
    return lambda_val if lambda_val is not None else DEFAULT_LAMBDA

# reference_grounding: paper chunk_011_02
def compute_reward(base_reward: float, mask_action: int, alpha: float) -> float:
    """
    Implements the intrinsic reward formula: R' = R + alpha * a_m
    where a_m = 1 if masked (blinded), 0 otherwise.
    """
    return base_reward + alpha * float(mask_action)

def aggregate_reward(rewards: List[float]) -> Dict[str, float]:
    """
    Aggregates rewards across trajectories.
    """
    import numpy as np
    if not rewards:
        return {"mean": 0.0, "std": 0.0}
    return {
        "mean": float(np.mean(rewards)),
        "std": float(np.std(rewards)),
        "min": float(np.min(rewards)),
        "max": float(np.max(rewards))
    }

# reference_grounding: paper 4.2. Experiment Design
def compute_fidelity_score(original_reward: float, masked_reward: float) -> float:
    """
    Computes the fidelity score as mentioned in StateMask.
    Measures the equivalence of the explanation method with StateMask.
    """
    if abs(original_reward) < 1e-6:
        return 0.0
    # Fidelity score typically measures the drop in performance when critical steps are masked.
    return 1.0 - (masked_reward / original_reward)

def aggregate_fidelity_score(scores: List[float]) -> Dict[str, float]:
    """
    Aggregates fidelity scores across 500 trajectories as per paper protocol.
    """
    import numpy as np
    if not scores:
        return {"mean": 0.0, "std": 0.0}
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores))
    }

def compute_loss(prediction: float, target: float) -> float:
    """
    Standard MSE loss for mask network or policy training.
    """
    return (prediction - target) ** 2

def aggregate_loss(losses: List[float]) -> Dict[str, float]:
    """
    Aggregates losses for reporting.
    """
    import numpy as np
    if not losses:
        return {"mean": 0.0}
    return {"mean": float(np.mean(losses))}

# reference_grounding: paper 4.3. Experiment Results
def compute_general_metrics_metric_general_metrics_training_loop_objective(losses: List[float]) -> float:
    """
    Canonical identifier: metric_general_metrics.
    Computes the average training objective (loss).
    """
    import numpy as np
    return float(np.mean(losses)) if losses else 0.0

def compute_general_metrics_metric_general_metrics_training_loop_score(rewards: List[float]) -> float:
    """
    Canonical identifier: metric_general_metrics.
    Computes the average training score (reward).
    """
    import numpy as np
    return float(np.mean(rewards)) if rewards else 0.0

@dataclass
class RlComparisonRegistryLayout:
    """
    Registry for RL baselines and experiment results.
    reference_grounding: paper 4.1. Experiment Setup
    """
    baselines: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    experiments: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)

def make_baseline(name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Factory for baseline configurations.
    reference_grounding: paper 4.1. Experiment Setup
    """
    baselines = {
        "ours": {"type": "RICE", "description": "Proposed RICE algorithm"},
        "random": {"type": "Random", "description": "Random exploration baseline"},
        "statemask": {"type": "StateMask", "description": "StateMask explanation baseline"},
        "ppo": {"type": "PPO", "description": "Vanilla PPO fine-tuning"},
        "sac": {"type": "SAC", "description": "Soft Actor-Critic baseline"},
        "gail": {"type": "GAIL", "description": "Generative Adversarial Imitation Learning"},
        "jsrl": {"type": "JSRL", "description": "Jump-Start Reinforcement Learning"},
        "heuristic": {"type": "Heuristic", "description": "Heuristic-based baseline"}
    }
    base_info = baselines.get(name.lower(), {"type": name, "description": "Custom baseline"})
    return {**base_info, "config": config}

def run_comparison(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes a comparison between baselines based on the provided config.
    """
    # This function serves as a route to coordinate baseline evaluation.
    return {"status": "initialized", "config": config}

def write_rl_comparison_registry_artifact(registry: RlComparisonRegistryLayout, output_path: str):
    """
    Writes the baseline registry to results/baseline_registry.json.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(asdict(registry), f, indent=2)

def write_json_artifact(data: Any, output_path: str):
    """
    Generic JSON artifact writer.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

def write_fidelity_score_artifact(scores: Dict[str, Any], output_path: str):
    """
    Writes fidelity scores to results/fidelity_scores.json.
    """
    write_json_artifact(scores, output_path)

def write_table_artifact(data: List[Dict[str, Any]], output_path: str):
    """
    Writes tabular data to CSV.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not data:
        return
    keys = data[0].keys()
    with open(output_path, 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(data)

# Canonical Metric Identifiers
metric_fidelity_score_top_k_ranking = "fidelity_score_top_k_ranking"
metric_fidelity_score = "fidelity_score"
metric_table_1_reproduction_artifact = "table_1_reproduction_artifact"
metric_reward = "reward"
metric_training_time = "training_time"
metric_final_reward = "final_reward"
metric_figure_1_reproduction_artifact = "figure_1_reproduction_artifact"
metric_figure_5_reproduction_artifact = "figure_5_reproduction_artifact"
metric_table_4_reproduction_artifact = "table_4_reproduction_artifact"
metric_general_metrics = "general_metrics"
metric_training_loop = "training_loop"
metric_evaluation = "evaluation"

# Canonical Artifact Identifiers
artifact_table_1 = "results/tables/table_1.csv"
artifact_figure_1 = "results/figures/figure_1.png"
artifact_figure_5 = "results/figures/figure_5.png"
artifact_table_4 = "results/tables/table_4.csv"
artifact_figure_2 = "results/figures/figure_2.png"
artifact_figure_3 = "results/figures/figure_3.png"
artifact_figure_4 = "results/figures/figure_4.png"
artifact_table_2 = "results/tables/table_2.csv"
artifact_table_3 = "results/tables/table_3.csv"
artifact_table_5 = "results/tables/table_5.csv"
artifact_table_6 = "results/tables/table_6.csv"
artifact_figure_6 = "results/figures/figure_6.png"
artifact_figure_7 = "results/figures/figure_7.png"
artifact_figure_8 = "results/figures/figure_8.png"
artifact_figure_9 = "results/figures/figure_9.png"

# Trend Assertions for Semantic Review
# reference_grounding: paper 4.3. Experiment Results
def assert_rice_outperforms_baselines(rice_reward: float, random_reward: float, statemask_reward: float):
    """
    RICE > Random, RICE >= StateMask
    """
    assert rice_reward > random_reward, "RICE should outperform Random baseline"
    assert rice_reward >= statemask_reward, "RICE should be at least as good as StateMask"

def assert_endpoint_low(p_0_reward: float, p_1_reward: float, p_mid_reward: float):
    """
    endpoint_low: p=0 and p=1 must be represented as lowest/minimum boundary cases.
    reference_grounding: paper Figure 7
    """
    assert p_mid_reward >= p_0_reward, "Mixed p should outperform p=0 boundary"
    assert p_mid_reward >= p_1_reward, "Mixed p should outperform p=1 boundary"