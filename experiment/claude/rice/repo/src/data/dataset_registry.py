"""
RICE Dataset and Benchmark Registry

Registers dataset/benchmark aliases for MuJoCo, selfish_mining, cage (network defense),
autonomous driving, and malware mutation environments.

Provides import-light descriptors/factories with clear availability checks and faithful
fallback errors for external environments and datasets.

Registry surfaces:
- get_dataset(name): Retrieve dataset descriptor
- get_environment(name): Create environment instance
- list_datasets(): List available datasets
- list_environments(): List available environments
- check_availability(name): Check if environment/dataset is available
- generate_random_explanations(trajectory_length, num_explanations): Random baseline

Evaluation surfaces:
- evaluate_refining(refined_agent, env, num_episodes): Evaluate refined agent performance
- evaluate_pretrained(agent, env, num_episodes): Evaluate pretrained agent performance
- compute_improvement(refined_reward, pretrained_reward): Compute improvement metrics
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import numpy as np
import json
import time


# ============================================================================
# Environment/Dataset Availability Checks
# ============================================================================

def check_mujoco_available() -> bool:
    """Check if MuJoCo environments are available."""
    try:
        import gymnasium as gym
        import mujoco
        return True
    except ImportError:
        try:
            import gym
            import mujoco
            return True
        except ImportError:
            return False


def check_cage_available() -> bool:
    """Check if CAGE (CybORG) network defense environment is available."""
    try:
        from CybORG import CybORG
        return True
    except ImportError:
        return False


def check_metadrive_available() -> bool:
    """Check if MetaDrive autonomous driving environment is available."""
    try:
        import metadrive
        return True
    except ImportError:
        return False


def check_pytorch_available() -> bool:
    """Check if PyTorch is available."""
    try:
        import torch
        return True
    except ImportError:
        return False


# ============================================================================
# Dataset/Benchmark Registry
# ============================================================================

DATASET_REGISTRY = {
    # MuJoCo robotics environments
    "mujoco": {
        "aliases": ["mujoco", "robotics", "continuous_control"],
        "environments": ["hopper", "walker2d", "reacher", "halfcheetah"],
        "category": "robotics",
        "availability_check": check_mujoco_available,
        "description": "MuJoCo continuous control robotics environments",
    },
    "hopper": {
        "aliases": ["hopper", "hopper-v3", "Hopper", "Hopper-v3"],
        "env_id": "Hopper-v3",
        "category": "mujoco",
        "observation_dim": 11,
        "action_dim": 3,
        "action_space": "continuous",
        "availability_check": check_mujoco_available,
        "description": "MuJoCo Hopper-v3 environment",
    },
    "walker2d": {
        "aliases": ["walker2d", "walker2d-v3", "Walker2d", "Walker2d-v3"],
        "env_id": "Walker2d-v3",
        "category": "mujoco",
        "observation_dim": 17,
        "action_dim": 6,
        "action_space": "continuous",
        "availability_check": check_mujoco_available,
        "description": "MuJoCo Walker2d-v3 environment",
    },
    "reacher": {
        "aliases": ["reacher", "reacher-v2", "Reacher", "Reacher-v2"],
        "env_id": "Reacher-v2",
        "category": "mujoco",
        "observation_dim": 11,
        "action_dim": 2,
        "action_space": "continuous",
        "availability_check": check_mujoco_available,
        "description": "MuJoCo Reacher-v2 environment",
    },
    "halfcheetah": {
        "aliases": ["halfcheetah", "halfcheetah-v3", "HalfCheetah", "HalfCheetah-v3"],
        "env_id": "HalfCheetah-v3",
        "category": "mujoco",
        "observation_dim": 17,
        "action_dim": 6,
        "action_space": "continuous",
        "availability_check": check_mujoco_available,
        "description": "MuJoCo HalfCheetah-v3 environment",
    },
    
    # Real-world application: Selfish Mining
    "selfish_mining": {
        "aliases": ["selfish_mining", "blockchain", "mining"],
        "category": "blockchain",
        "observation_dim": 4,
        "action_dim": 2,
        "action_space": "discrete",
        "availability_check": lambda: True,  # Custom environment
        "description": "Blockchain selfish mining attack simulation",
    },
    
    # Real-world application: CAGE (Network Defense)
    "cage": {
        "aliases": ["cage", "cyborg", "network_defense", "cybersecurity"],
        "category": "cybersecurity",
        "observation_dim": 52,
        "action_dim": 41,
        "action_space": "discrete",
        "availability_check": check_cage_available,
        "description": "CAGE Challenge 2 network defense environment (CybORG)",
    },
    
    # Real-world application: Autonomous Driving
    "autonomous_driving": {
        "aliases": ["autonomous_driving", "metadrive", "driving", "carla"],
        "category": "autonomous_driving",
        "observation_dim": "variable",
        "action_dim": 2,
        "action_space": "continuous",
        "availability_check": check_metadrive_available,
        "description": "MetaDrive autonomous driving simulation",
    },
    
    # Real-world application: Malware Mutation
    "malware_mutation": {
        "aliases": ["malware_mutation", "malware", "adversarial"],
        "category": "adversarial_ml",
        "observation_dim": "variable",
        "action_dim": "variable",
        "action_space": "discrete",
        "availability_check": lambda: True,  # Custom environment
        "description": "Malware mutation adversarial environment",
    },
}


def resolve_alias(name: str) -> Optional[str]:
    """Resolve environment alias to canonical name."""
    name_lower = name.lower()
    for key, info in DATASET_REGISTRY.items():
        if "aliases" in info and name_lower in [a.lower() for a in info["aliases"]]:
            return key
    return name if name in DATASET_REGISTRY else None


def get_dataset(name: str) -> Dict[str, Any]:
    """Get dataset descriptor by name or alias."""
    canonical_name = resolve_alias(name)
    if canonical_name is None:
        raise ValueError(f"Unknown dataset/benchmark: {name}")
    return DATASET_REGISTRY[canonical_name]


def list_datasets() -> List[str]:
    """List all registered datasets."""
    return list(DATASET_REGISTRY.keys())


def list_environments() -> List[str]:
    """List all registered environments."""
    return [k for k, v in DATASET_REGISTRY.items() if "env_id" in v or "category" in v]


def check_availability(name: str) -> bool:
    """Check if environment/dataset is available."""
    try:
        dataset = get_dataset(name)
        if "availability_check" in dataset:
            return dataset["availability_check"]()
        return True
    except (ValueError, KeyError):
        return False


# ============================================================================
# Environment Factory
# ============================================================================

def create_environment(name: str, **kwargs) -> Any:
    """
    Create environment instance with lazy loading and availability checks.
    
    Returns environment or raises clear error if unavailable.
    """
    dataset = get_dataset(name)
    
    if not check_availability(name):
        raise RuntimeError(
            f"Environment {name} is not available. "
            f"Required dependencies may not be installed. "
            f"Description: {dataset.get('description', 'N/A')}"
        )
    
    category = dataset.get("category", "")
    
    # MuJoCo environments
    if category == "mujoco" and "env_id" in dataset:
        try:
            import gymnasium as gym
        except ImportError:
            import gym
        return gym.make(dataset["env_id"], **kwargs)
    
    # Selfish Mining
    elif name == "selfish_mining":
        from src.environments import SelfishMiningEnv
        return SelfishMiningEnv(**kwargs)
    
    # CAGE Network Defense
    elif name == "cage":
        from src.environments import CageEnv
        return CageEnv(**kwargs)
    
    # Autonomous Driving
    elif name == "autonomous_driving":
        from src.environments import AutonomousDrivingEnv
        return AutonomousDrivingEnv(**kwargs)
    
    # Malware Mutation
    elif name == "malware_mutation":
        from src.environments import MalwareMutationEnv
        return MalwareMutationEnv(**kwargs)
    
    else:
        raise ValueError(f"Unknown environment type for {name}")


# ============================================================================
# Random Explanation Baseline
# ============================================================================

def generate_random_explanations(
    trajectory_length: int,
    num_explanations: int,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate random explanations as baseline.
    
    Uniformly samples timesteps from trajectory as critical states.
    
    Args:
        trajectory_length: Length of trajectory
        num_explanations: Number of critical states to select
        seed: Random seed for reproducibility
        
    Returns:
        Array of indices representing randomly selected critical states
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Uniform random sampling of timesteps
    num_explanations = min(num_explanations, trajectory_length)
    critical_indices = np.random.choice(
        trajectory_length,
        size=num_explanations,
        replace=False
    )
    return np.sort(critical_indices)


def generate_random_importance_scores(
    trajectory_length: int,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate random importance scores for all states.
    
    Args:
        trajectory_length: Length of trajectory
        seed: Random seed for reproducibility
        
    Returns:
        Array of random importance scores (uniform [0, 1])
    """
    if seed is not None:
        np.random.seed(seed)
    
    return np.random.uniform(0, 1, size=trajectory_length)


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_agent(
    agent: Any,
    env: Any,
    num_episodes: int = 10,
    max_steps: int = 1000,
    render: bool = False,
    seed: Optional[int] = None
) -> Dict[str, float]:
    """
    Evaluate agent performance in environment.
    
    Args:
        agent: Agent to evaluate
        env: Environment instance
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        render: Whether to render episodes
        seed: Random seed
        
    Returns:
        Dictionary with evaluation metrics
    """
    if seed is not None:
        np.random.seed(seed)
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset(seed=seed + episode if seed is not None else None)
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        while not done and episode_length < max_steps:
            # Get action from agent
            if hasattr(agent, 'predict'):
                action, _ = agent.predict(obs, deterministic=True)
            elif hasattr(agent, 'act'):
                action = agent.act(obs, deterministic=True)
            else:
                # Fallback: assume agent is callable
                action = agent(obs)
            
            # Step environment
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
            if render:
                env.render()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    
    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "std_length": float(np.std(episode_lengths)),
    }


def evaluate_refining(
    refined_agent: Any,
    env: Any,
    num_episodes: int = 10,
    max_steps: int = 1000,
    seed: Optional[int] = None
) -> float:
    """
    Evaluate refined agent performance.
    
    Args:
        refined_agent: Refined agent to evaluate
        env: Environment instance
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        seed: Random seed
        
    Returns:
        Mean reward over evaluation episodes
    """
    results = evaluate_agent(refined_agent, env, num_episodes, max_steps, seed=seed)
    return results["mean_reward"]


def evaluate_pretrained(
    agent: Any,
    env: Any,
    num_episodes: int = 10,
    max_steps: int = 1000,
    seed: Optional[int] = None
) -> float:
    """
    Evaluate pretrained agent performance.
    
    Args:
        agent: Pretrained agent to evaluate
        env: Environment instance
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        seed: Random seed
        
    Returns:
        Mean reward over evaluation episodes
    """
    results = evaluate_agent(agent, env, num_episodes, max_steps, seed=seed)
    return results["mean_reward"]


def compute_improvement(
    refined_reward: float,
    pretrained_reward: float
) -> Dict[str, float]:
    """
    Compute improvement metrics between refined and pretrained agents.
    
    Args:
        refined_reward: Reward of refined agent
        pretrained_reward: Reward of pretrained agent
        
    Returns:
        Dictionary with improvement metrics
    """
    absolute_improvement = refined_reward - pretrained_reward
    
    # Avoid division by zero
    if abs(pretrained_reward) < 1e-8:
        relative_improvement = 0.0 if abs(absolute_improvement) < 1e-8 else float('inf')
    else:
        relative_improvement = (absolute_improvement / abs(pretrained_reward)) * 100.0
    
    return {
        "absolute_improvement": float(absolute_improvement),
        "relative_improvement": float(relative_improvement),
        "refined_reward": float(refined_reward),
        "pretrained_reward": float(pretrained_reward),
    }


# ============================================================================
# Artifact Writing
# ============================================================================

def write_refining_results(
    results: Dict[str, Any],
    output_path: str = "results/table1_refining.json"
) -> None:
    """Write refining experiment results to artifact file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)


def write_metrics(
    metrics: Dict[str, Any],
    output_path: str = "results/metrics.json"
) -> None:
    """Write evaluation metrics to artifact file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)


def write_efficiency_results(
    results: Dict[str, Any],
    output_path: str = "results/table1_efficiency.json"
) -> None:
    """Write efficiency comparison results to artifact file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)


# ============================================================================
# Dry-Run Artifact Generation
# ============================================================================

def generate_dry_run_artifacts(artifact_dir: Optional[str] = None) -> None:
    """
    Generate dry-run artifacts for smoke testing.
    
    Creates schema/contract artifacts without running expensive experiments.
    """
    if artifact_dir is None:
        artifact_dir = os.environ.get('PAPERBENCH_REPRO_ARTIFACT_DIR', 'results')
    
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    
    # Dry-run refining results
    dry_run_refining = {
        "_dry_run": True,
        "_schema": "table1_refining",
        "_description": "Refining performance comparison (dry-run schema)",
        "environments": {
            "hopper": {
                "rice": {"mean": 0.0, "std": 0.0},
                "statemask": {"mean": 0.0, "std": 0.0},
                "random": {"mean": 0.0, "std": 0.0},
            }
        }
    }
    write_refining_results(dry_run_refining, artifact_dir / "table1_refining.json")
    
    # Dry-run metrics
    dry_run_metrics = {
        "_dry_run": True,
        "_schema": "evaluation_metrics",
        "_description": "Evaluation metrics (dry-run schema)",
        "mean_reward": 0.0,
        "std_reward": 0.0,
        "improvement": 0.0,
    }
    write_metrics(dry_run_metrics, artifact_dir / "metrics.json")
    
    # Dry-run efficiency results
    dry_run_efficiency = {
        "_dry_run": True,
        "_schema": "table1_efficiency",
        "_description": "Efficiency comparison (dry-run schema)",
        "rice_time": 0.0,
        "statemask_time": 0.0,
    }
    write_efficiency_results(dry_run_efficiency, artifact_dir / "table1_efficiency.json")
    
    print(f"Dry-run artifacts written to {artifact_dir}")


# ============================================================================
# Main (for testing)
# ============================================================================

if __name__ == "__main__":
    print("RICE Dataset Registry")
    print("=" * 80)
    
    print("\nRegistered datasets:")
    for name in list_datasets():
        dataset = get_dataset(name)
        available = check_availability(name)
        print(f"  - {name}: {dataset.get('description', 'N/A')} [{'✓' if available else '✗'}]")
    
    print("\nTesting random explanation generation:")
    random_explanations = generate_random_explanations(100, 10, seed=42)
    print(f"  Generated {len(random_explanations)} critical states: {random_explanations[:5]}...")
    
    print("\nGenerating dry-run artifacts...")
    generate_dry_run_artifacts()
    
    print("\nDataset registry ready.")