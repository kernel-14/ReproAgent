# src/data/data.py
# Data pipeline, metric registry, and evaluation contracts for SAPG reproduction
# reference_grounding: wp_004 src/data/data.py
#
# Paper evidence contract: Implements dataset registry, metric formulas, and
# evaluation surfaces for SAPG experiments on manipulation tasks.
#
# Method obligations:
# - Write verifiable artifacts: results/dataset_registry.json, results/metrics.json, results/data_manifest.json
# - Represent environments through import-light descriptors with availability checks
# - Create explicit benchmark registry entries from paper
# - Implement metric formula/aggregation for: reward, success_rate, episode_length
# - Keep full data downloads lazy, expose smoke fixtures for every benchmark
# - Preserve explicit environment/task coverage and initialization surfaces

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple
import numpy as np


# ============================================================================
# Dataset/Benchmark Registry
# ============================================================================

DATASET_REGISTRY = {
    "ShadowHandOver": {
        "task_id": "ShadowHandOver",
        "aliases": ["Shadow Hand", "ShadowHand", "HandOver"],
        "difficulty": "hard",
        "environment_type": "isaacgym",
        "max_episode_length": 1000,
        "num_envs": 24576,
        "observation_dim": 211,
        "action_dim": 20,
        "success_threshold": 0.8,
        "metric_type": "success_rate",
        "description": "Shadow Hand object manipulation - pass object between hands",
        "paper_reference": "Table 1, Figure 5",
        "availability": "isaacgym_required"
    },
    "ShadowHandCatchUnderarm": {
        "task_id": "ShadowHandCatchUnderarm",
        "aliases": ["CatchUnderarm", "Underarm"],
        "difficulty": "hard",
        "environment_type": "isaacgym",
        "max_episode_length": 1000,
        "num_envs": 24576,
        "observation_dim": 211,
        "action_dim": 20,
        "success_threshold": 0.8,
        "metric_type": "success_rate",
        "description": "Shadow Hand catch object thrown underarm",
        "paper_reference": "Table 1",
        "availability": "isaacgym_required"
    },
    "ShadowHandCatchAbreast": {
        "task_id": "ShadowHandCatchAbreast",
        "aliases": ["CatchAbreast", "Abreast"],
        "difficulty": "hard",
        "environment_type": "isaacgym",
        "max_episode_length": 1000,
        "num_envs": 24576,
        "observation_dim": 211,
        "action_dim": 20,
        "success_threshold": 0.8,
        "metric_type": "success_rate",
        "description": "Shadow Hand catch object thrown abreast",
        "paper_reference": "Table 1",
        "availability": "isaacgym_required"
    },
    "ShadowHandReOrientation": {
        "task_id": "ShadowHandReOrientation",
        "aliases": ["Shadow Hand Reorientation", "ShadowReorient", "Reorientation"],
        "difficulty": "hard",
        "environment_type": "isaacgym",
        "max_episode_length": 1000,
        "num_envs": 24576,
        "observation_dim": 211,
        "action_dim": 20,
        "success_threshold": 0.8,
        "metric_type": "success_rate",
        "description": "Shadow Hand object reorientation",
        "paper_reference": "Table 1",
        "availability": "isaacgym_required"
    },
    "AllegroHandReOrientation": {
        "task_id": "AllegroHandReOrientation",
        "aliases": ["Allegro Reorientation", "AllegroReorient"],
        "difficulty": "hard",
        "environment_type": "isaacgym",
        "max_episode_length": 1000,
        "num_envs": 24576,
        "observation_dim": 42,
        "action_dim": 16,
        "success_threshold": 0.8,
        "metric_type": "success_rate",
        "description": "Allegro Hand object reorientation",
        "paper_reference": "Table 1",
        "availability": "isaacgym_required"
    },
    "AllegroKuka": {
        "task_id": "AllegroKuka",
        "aliases": ["Allegro Kuka", "AllegroKukaReorientation"],
        "difficulty": "hard",
        "environment_type": "isaacgym",
        "max_episode_length": 1000,
        "num_envs": 24576,
        "observation_dim": 42,
        "action_dim": 16,
        "success_threshold": 0.8,
        "metric_type": "success_rate",
        "description": "Allegro Hand with Kuka arm manipulation",
        "paper_reference": "Table 1, Figure 5",
        "availability": "isaacgym_required"
    },
    "harder_AllegroKuka": {
        "task_id": "harder_AllegroKuka",
        "aliases": ["Harder Allegro Kuka", "HarderAllegroKuka"],
        "difficulty": "very_hard",
        "environment_type": "isaacgym",
        "max_episode_length": 1000,
        "num_envs": 24576,
        "observation_dim": 42,
        "action_dim": 16,
        "success_threshold": 0.9,
        "metric_type": "success_rate",
        "description": "Harder variant of Allegro Kuka manipulation",
        "paper_reference": "Table 1",
        "availability": "isaacgym_required"
    },
    "Throw": {
        "task_id": "Throw",
        "aliases": ["ThrowTask"],
        "difficulty": "hard",
        "environment_type": "isaacgym",
        "max_episode_length": 1000,
        "num_envs": 24576,
        "observation_dim": 211,
        "action_dim": 20,
        "success_threshold": 0.8,
        "metric_type": "success_rate",
        "description": "Object throwing task",
        "paper_reference": "Table 1",
        "availability": "isaacgym_required"
    },
    "Regrasping": {
        "task_id": "Regrasping",
        "aliases": ["RegraspTask"],
        "difficulty": "hard",
        "environment_type": "isaacgym",
        "max_episode_length": 1000,
        "num_envs": 24576,
        "observation_dim": 211,
        "action_dim": 20,
        "success_threshold": 0.8,
        "metric_type": "success_rate",
        "description": "Object regrasping task",
        "paper_reference": "Table 1",
        "availability": "isaacgym_required"
    }
}


# ============================================================================
# Metric Registry and Formulas
# ============================================================================

METRIC_REGISTRY = {
    "reward": {
        "metric_id": "reward",
        "display_name": "Episode Reward",
        "aggregation": "mean",
        "higher_is_better": True,
        "formula": "mean(episode_rewards)",
        "description": "Mean episode reward across all environments",
        "paper_reference": "Primary metric in all experiments"
    },
    "success_rate": {
        "metric_id": "success_rate",
        "display_name": "Success Rate",
        "aggregation": "mean",
        "higher_is_better": True,
        "formula": "mean(episode_successes)",
        "description": "Fraction of episodes that achieve success threshold",
        "paper_reference": "Table 1 - performance after 2e10 samples"
    },
    "episode_length": {
        "metric_id": "episode_length",
        "display_name": "Episode Length",
        "aggregation": "mean",
        "higher_is_better": False,
        "formula": "mean(episode_lengths)",
        "description": "Mean episode length (steps to completion or timeout)",
        "paper_reference": "Auxiliary metric for training efficiency"
    },
    "value_loss": {
        "metric_id": "value_loss",
        "display_name": "Value Loss",
        "aggregation": "mean",
        "higher_is_better": False,
        "formula": "mean(value_losses)",
        "description": "Value function loss during training",
        "paper_reference": "Training diagnostic"
    },
    "policy_loss": {
        "metric_id": "policy_loss",
        "display_name": "Policy Loss",
        "aggregation": "mean",
        "higher_is_better": False,
        "formula": "mean(policy_losses)",
        "description": "Policy gradient loss during training",
        "paper_reference": "Training diagnostic"
    },
    "entropy": {
        "metric_id": "entropy",
        "display_name": "Policy Entropy",
        "aggregation": "mean",
        "higher_is_better": True,
        "formula": "mean(policy_entropies)",
        "description": "Policy entropy for exploration",
        "paper_reference": "Figure 6 - entropy coefficient ablation"
    },
    "importance_weight": {
        "metric_id": "importance_weight",
        "display_name": "Importance Weight",
        "aggregation": "mean",
        "higher_is_better": False,
        "formula": "mean(importance_weights)",
        "description": "Mean importance sampling weight for off-policy aggregation",
        "paper_reference": "SAPG aggregation mechanism"
    },
    "state_coverage": {
        "metric_id": "state_coverage",
        "display_name": "State Space Coverage",
        "aggregation": "reconstruction_error",
        "higher_is_better": False,
        "formula": "reconstruction_error(states, reconstructed_states)",
        "description": "State space coverage via PCA/MLP reconstruction error",
        "paper_reference": "Figure 7, Figure 8 - state space coverage analysis"
    }
}


# ============================================================================
# Metric Computation Functions
# ============================================================================

def compute_reward_metric(episode_data: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute reward metrics from episode data.
    
    Args:
        episode_data: List of episode dictionaries with 'rewards' key
        
    Returns:
        Dictionary with reward statistics
    """
    if not episode_data:
        return {
            "mean_reward": 0.0,
            "std_reward": 0.0,
            "min_reward": 0.0,
            "max_reward": 0.0,
            "median_reward": 0.0
        }
    
    rewards = [ep.get("reward", 0.0) for ep in episode_data]
    rewards_array = np.array(rewards)
    
    return {
        "mean_reward": float(np.mean(rewards_array)),
        "std_reward": float(np.std(rewards_array)),
        "min_reward": float(np.min(rewards_array)),
        "max_reward": float(np.max(rewards_array)),
        "median_reward": float(np.median(rewards_array))
    }


def compute_success_rate_metric(episode_data: List[Dict[str, Any]], 
                                success_threshold: float = 0.8) -> Dict[str, float]:
    """
    Compute success rate metrics from episode data.
    
    Args:
        episode_data: List of episode dictionaries with 'success' or 'reward' key
        success_threshold: Threshold for considering episode successful
        
    Returns:
        Dictionary with success rate statistics
    """
    if not episode_data:
        return {
            "success_rate": 0.0,
            "num_successes": 0,
            "num_episodes": 0
        }
    
    successes = []
    for ep in episode_data:
        if "success" in ep:
            successes.append(float(ep["success"]))
        elif "reward" in ep:
            # Infer success from reward threshold
            successes.append(float(ep["reward"] >= success_threshold))
        else:
            successes.append(0.0)
    
    successes_array = np.array(successes)
    
    return {
        "success_rate": float(np.mean(successes_array)),
        "num_successes": int(np.sum(successes_array)),
        "num_episodes": len(episode_data)
    }


def compute_episode_length_metric(episode_data: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute episode length metrics from episode data.
    
    Args:
        episode_data: List of episode dictionaries with 'length' key
        
    Returns:
        Dictionary with episode length statistics
    """
    if not episode_data:
        return {
            "mean_length": 0.0,
            "std_length": 0.0,
            "min_length": 0.0,
            "max_length": 0.0
        }
    
    lengths = [ep.get("length", 0) for ep in episode_data]
    lengths_array = np.array(lengths)
    
    return {
        "mean_length": float(np.mean(lengths_array)),
        "std_length": float(np.std(lengths_array)),
        "min_length": float(np.min(lengths_array)),
        "max_length": float(np.max(lengths_array))
    }


def compute_training_metrics(training_data: Dict[str, List[float]]) -> Dict[str, float]:
    """
    Compute training diagnostic metrics.
    
    Args:
        training_data: Dictionary with lists of training metrics
        
    Returns:
        Dictionary with aggregated training metrics
    """
    metrics = {}
    
    for key in ["value_loss", "policy_loss", "entropy", "importance_weight"]:
        if key in training_data and training_data[key]:
            values = np.array(training_data[key])
            metrics[f"mean_{key}"] = float(np.mean(values))
            metrics[f"std_{key}"] = float(np.std(values))
        else:
            metrics[f"mean_{key}"] = 0.0
            metrics[f"std_{key}"] = 0.0
    
    return metrics


def compute_state_coverage_metric(states: np.ndarray, 
                                  reconstructed_states: np.ndarray) -> Dict[str, float]:
    """
    Compute state space coverage via reconstruction error.
    
    Paper reference: Figure 7 (PCA), Figure 8 (MLP) - state space coverage analysis
    
    Args:
        states: Original state observations (N, state_dim)
        reconstructed_states: Reconstructed states from dimensionality reduction (N, state_dim)
        
    Returns:
        Dictionary with reconstruction error metrics
    """
    if states.shape != reconstructed_states.shape:
        return {
            "reconstruction_error": float('inf'),
            "normalized_error": float('inf')
        }
    
    # Mean squared error
    mse = np.mean((states - reconstructed_states) ** 2)
    
    # Normalized by state variance
    state_var = np.var(states)
    normalized_error = mse / (state_var + 1e-8)
    
    return {
        "reconstruction_error": float(mse),
        "normalized_error": float(normalized_error),
        "state_variance": float(state_var)
    }


# ============================================================================
# Evaluation Interface
# ============================================================================

def evaluate_predictions(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate predictions/trajectories according to paper metrics.
    
    This function aggregates episode data and computes all registered metrics.
    
    Args:
        config: Configuration dictionary with:
            - episode_data: List of episode dictionaries
            - training_data: Dictionary of training metrics (optional)
            - task_id: Task identifier for benchmark-specific thresholds
            - mode: Execution mode (smoke, default, full)
            
    Returns:
        Dictionary with computed metrics and evaluation results
    """
    mode = config.get("mode", "smoke")
    task_id = config.get("task_id", "ShadowHandOver")
    episode_data = config.get("episode_data", [])
    training_data = config.get("training_data", {})
    
    # Get task-specific configuration
    task_config = DATASET_REGISTRY.get(task_id, {})
    success_threshold = task_config.get("success_threshold", 0.8)
    
    # Compute all metrics
    results = {
        "task_id": task_id,
        "mode": mode,
        "num_episodes": len(episode_data),
        "metrics": {}
    }
    
    # Reward metrics
    reward_metrics = compute_reward_metric(episode_data)
    results["metrics"].update(reward_metrics)
    
    # Success rate metrics
    success_metrics = compute_success_rate_metric(episode_data, success_threshold)
    results["metrics"].update(success_metrics)
    
    # Episode length metrics
    length_metrics = compute_episode_length_metric(episode_data)
    results["metrics"].update(length_metrics)
    
    # Training metrics (if available)
    if training_data:
        training_metrics = compute_training_metrics(training_data)
        results["metrics"].update(training_metrics)
    
    # Add metadata
    results["metadata"] = {
        "task_config": task_config,
        "success_threshold": success_threshold,
        "metric_registry": list(METRIC_REGISTRY.keys())
    }
    
    return results


# ============================================================================
# Data Manifest and Registry Writers
# ============================================================================

def write_dataset_registry(output_dir: str = "results") -> str:
    """
    Write dataset registry to JSON artifact.
    
    Returns:
        Path to written artifact
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "dataset_registry.json")
    
    registry_data = {
        "registry_version": "1.0",
        "paper_reference": "SAPG: Split and Aggregate Policy Gradients",
        "num_tasks": len(DATASET_REGISTRY),
        "tasks": DATASET_REGISTRY,
        "task_ids": list(DATASET_REGISTRY.keys())
    }
    
    with open(output_path, "w") as f:
        json.dump(registry_data, f, indent=2)
    
    return output_path


def write_metrics_registry(output_dir: str = "results") -> str:
    """
    Write metrics registry to JSON artifact.
    
    Returns:
        Path to written artifact
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "metrics.json")
    
    metrics_data = {
        "registry_version": "1.0",
        "paper_reference": "SAPG: Split and Aggregate Policy Gradients",
        "num_metrics": len(METRIC_REGISTRY),
        "metrics": METRIC_REGISTRY,
        "metric_ids": list(METRIC_REGISTRY.keys()),
        "primary_metrics": ["reward", "success_rate"],
        "training_metrics": ["value_loss", "policy_loss", "entropy", "importance_weight"],
        "analysis_metrics": ["state_coverage"]
    }
    
    with open(output_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
    
    return output_path


def write_data_manifest(config: Dict[str, Any], output_dir: str = "results") -> str:
    """
    Write data manifest describing available datasets and their status.
    
    Args:
        config: Configuration with mode and task selection
        output_dir: Output directory for manifest
        
    Returns:
        Path to written artifact
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "data_manifest.json")
    
    mode = config.get("mode", "smoke")
    
    # Check environment availability (lazy check, no actual import)
    try:
        import importlib.util
        isaacgym_available = importlib.util.find_spec("isaacgym") is not None
    except (ImportError, AttributeError):
        isaacgym_available = False
    
    manifest_data = {
        "manifest_version": "1.0",
        "mode": mode,
        "paper_reference": "SAPG: Split and Aggregate Policy Gradients",
        "environment_availability": {
            "isaacgym": isaacgym_available,
            "mujoco": False,  # Not used in paper
            "pybullet": False  # Not used in paper
        },
        "datasets": {},
        "readiness": {
            "smoke_mode": True,
            "default_mode": isaacgym_available,
            "full_mode": isaacgym_available
        }
    }
    
    # Add dataset availability status
    for task_id, task_config in DATASET_REGISTRY.items():
        env_type = task_config.get("environment_type", "isaacgym")
        available = manifest_data["environment_availability"].get(env_type, False)
        
        manifest_data["datasets"][task_id] = {
            "task_id": task_id,
            "environment_type": env_type,
            "available": available,
            "smoke_fixture": True,  # All tasks have smoke fixtures
            "difficulty": task_config.get("difficulty", "unknown"),
            "paper_reference": task_config.get("paper_reference", "")
        }
    
    with open(output_path, "w") as f:
        json.dump(manifest_data, f, indent=2)
    
    return output_path


# ============================================================================
# Smoke/Dry-Run Artifact Generation
# ============================================================================

def generate_smoke_artifacts(config: Dict[str, Any]) -> Dict[str, str]:
    """
    Generate smoke/dry-run artifacts for contract validation.
    
    This function creates all declared artifacts with schema/readiness content
    during smoke mode, without requiring actual training or environment execution.
    
    Args:
        config: Configuration dictionary with mode and output settings
        
    Returns:
        Dictionary mapping artifact names to output paths
    """
    output_dir = config.get("output_dir", "results")
    os.makedirs(output_dir, exist_ok=True)
    
    artifacts = {}
    
    # Write dataset registry
    artifacts["dataset_registry"] = write_dataset_registry(output_dir)
    
    # Write metrics registry
    artifacts["metrics_registry"] = write_metrics_registry(output_dir)
    
    # Write data manifest
    artifacts["data_manifest"] = write_data_manifest(config, output_dir)
    
    # Write readiness artifact
    readiness_path = os.path.join(output_dir, "readiness.json")
    readiness_data = {
        "mode": "smoke",
        "status": "ready",
        "artifacts_generated": list(artifacts.keys()),
        "dataset_registry_tasks": len(DATASET_REGISTRY),
        "metric_registry_metrics": len(METRIC_REGISTRY),
        "note": "Dry-run contract artifact - not real experiment results"
    }
    with open(readiness_path, "w") as f:
        json.dump(readiness_data, f, indent=2)
    artifacts["readiness"] = readiness_path
    
    # Write evaluation result schema
    eval_result_path = os.path.join(output_dir, "evaluation_result.json")
    eval_result_data = {
        "mode": "smoke",
        "task_id": "ShadowHandOver",
        "num_episodes": 0,
        "metrics": {
            "mean_reward": 0.0,
            "success_rate": 0.0,
            "mean_length": 0.0
        },
        "metadata": {
            "note": "Dry-run schema artifact - not real experiment results"
        }
    }
    with open(eval_result_path, "w") as f:
        json.dump(eval_result_data, f, indent=2)
    artifacts["evaluation_result"] = eval_result_path
    
    return artifacts


# ============================================================================
# Public API
# ============================================================================

def get_dataset_info(task_id: str) -> Optional[Dict[str, Any]]:
    """Get dataset/task information from registry."""
    return DATASET_REGISTRY.get(task_id)


def get_metric_info(metric_id: str) -> Optional[Dict[str, Any]]:
    """Get metric information from registry."""
    return METRIC_REGISTRY.get(metric_id)


def list_available_tasks() -> List[str]:
    """List all available task IDs."""
    return list(DATASET_REGISTRY.keys())


def list_available_metrics() -> List[str]:
    """List all available metric IDs."""
    return list(METRIC_REGISTRY.keys())


# ============================================================================
# Module Initialization
# ============================================================================

if __name__ == "__main__":
    # Smoke test: generate artifacts
    config = {"mode": "smoke", "output_dir": "results"}
    artifacts = generate_smoke_artifacts(config)
    print(f"Generated {len(artifacts)} smoke artifacts:")
    for name, path in artifacts.items():
        print(f"  {name}: {path}")