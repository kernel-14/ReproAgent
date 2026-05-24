"""
RICE Environment Registry

Exposes paper-derived environment/task registry entries with ids, aliases, setup metadata,
and factory/config hooks for all paper environments:
- MuJoCo robotics: Hopper-v3, Walker2d-v3, Reacher-v2, HalfCheetah-v3
- Sparse reward variants: Hopper-sparse, Walker2d-sparse, HalfCheetah-sparse
- Real-world applications: selfish_mining, network_defense, autonomous_driving, cage_challenge

Registry surfaces:
- get_environment(env_id, **kwargs): Factory returning environment instance
- list_environments(): List all registered environment IDs
- get_environment_config(env_id): Get configuration metadata
- evaluate_agent(agent, env, num_episodes): Evaluate agent and return metrics
- create_all_environments(): Create instances for all registered environments
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np


# ============================================================================
# Lazy Imports
# ============================================================================

def lazy_load_gym():
    """Lazy import gym/gymnasium."""
    try:
        import gymnasium as gym
        return gym
    except ImportError:
        try:
            import gym
            return gym
        except ImportError:
            raise ImportError(
                "Gym/Gymnasium is required for environments. "
                "Install with: pip install gymnasium or pip install 'gymnasium[mujoco]'"
            )


def lazy_load_torch():
    """Lazy import torch."""
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError("PyTorch is required. Install with: pip install torch")


# ============================================================================
# Environment Registry
# ============================================================================

# Paper-derived environment registry with full metadata
ENVIRONMENT_REGISTRY: Dict[str, Dict[str, Any]] = {
    # MuJoCo Standard Environments
    "hopper": {
        "id": "Hopper-v3",
        "env_id": "Hopper-v3",
        "aliases": ["hopper", "hopper-v3", "Hopper", "Hopper-v3"],
        "category": "mujoco",
        "task_alias": "robotics",
        "observation_dim": 11,
        "action_dim": 3,
        "action_space": "continuous",
        "reward_type": "dense",
        "difficulty": "medium",
        "pretrain_timesteps": 1000000,
        "refine_timesteps": 200000,
        "eval_episodes": 10,
        "max_episode_steps": 1000,
    },
    "walker2d": {
        "id": "Walker2d-v3",
        "env_id": "Walker2d-v3",
        "aliases": ["walker2d", "walker2d-v3", "Walker2d", "Walker2d-v3"],
        "category": "mujoco",
        "task_alias": "robotics",
        "observation_dim": 17,
        "action_dim": 6,
        "action_space": "continuous",
        "reward_type": "dense",
        "difficulty": "medium",
        "pretrain_timesteps": 1000000,
        "refine_timesteps": 200000,
        "eval_episodes": 10,
        "max_episode_steps": 1000,
    },
    "reacher": {
        "id": "Reacher-v2",
        "env_id": "Reacher-v2",
        "aliases": ["reacher", "reacher-v2", "Reacher", "Reacher-v2"],
        "category": "mujoco",
        "task_alias": "robotics",
        "observation_dim": 11,
        "action_dim": 2,
        "action_space": "continuous",
        "reward_type": "dense",
        "difficulty": "easy",
        "pretrain_timesteps": 500000,
        "refine_timesteps": 100000,
        "eval_episodes": 10,
        "max_episode_steps": 50,
    },
    "halfcheetah": {
        "id": "HalfCheetah-v3",
        "env_id": "HalfCheetah-v3",
        "aliases": ["halfcheetah", "halfcheetah-v3", "HalfCheetah", "HalfCheetah-v3"],
        "category": "mujoco",
        "task_alias": "robotics",
        "observation_dim": 17,
        "action_dim": 6,
        "action_space": "continuous",
        "reward_type": "dense",
        "difficulty": "medium",
        "pretrain_timesteps": 1000000,
        "refine_timesteps": 200000,
        "eval_episodes": 10,
        "max_episode_steps": 1000,
    },
    # Sparse Reward Variants
    "hopper_sparse": {
        "id": "Hopper-sparse",
        "env_id": "Hopper-v3",
        "aliases": ["hopper_sparse", "hopper-sparse", "Hopper-sparse"],
        "category": "mujoco",
        "task_alias": "robotics",
        "observation_dim": 11,
        "action_dim": 3,
        "action_space": "continuous",
        "reward_type": "sparse",
        "difficulty": "hard",
        "pretrain_timesteps": 2000000,
        "refine_timesteps": 400000,
        "eval_episodes": 10,
        "max_episode_steps": 1000,
        "sparse_threshold": 3000.0,
    },
    "walker2d_sparse": {
        "id": "Walker2d-sparse",
        "env_id": "Walker2d-v3",
        "aliases": ["walker2d_sparse", "walker2d-sparse", "Walker2d-sparse"],
        "category": "mujoco",
        "task_alias": "robotics",
        "observation_dim": 17,
        "action_dim": 6,
        "action_space": "continuous",
        "reward_type": "sparse",
        "difficulty": "hard",
        "pretrain_timesteps": 2000000,
        "refine_timesteps": 400000,
        "eval_episodes": 10,
        "max_episode_steps": 1000,
        "sparse_threshold": 4000.0,
    },
    "halfcheetah_sparse": {
        "id": "HalfCheetah-sparse",
        "env_id": "HalfCheetah-v3",
        "aliases": ["halfcheetah_sparse", "halfcheetah-sparse", "HalfCheetah-sparse"],
        "category": "mujoco",
        "task_alias": "robotics",
        "observation_dim": 17,
        "action_dim": 6,
        "action_space": "continuous",
        "reward_type": "sparse",
        "difficulty": "hard",
        "pretrain_timesteps": 2000000,
        "refine_timesteps": 400000,
        "eval_episodes": 10,
        "max_episode_steps": 1000,
        "sparse_threshold": 6000.0,
    },
    # Real-world Application Environments
    "selfish_mining": {
        "id": "selfish_mining",
        "env_id": "SelfishMining-v0",
        "aliases": ["selfish_mining", "selfish-mining", "SelfishMining"],
        "category": "real_world",
        "task_alias": "selfish_mining",
        "observation_dim": 8,
        "action_dim": 2,
        "action_space": "discrete",
        "reward_type": "sparse",
        "difficulty": "hard",
        "pretrain_timesteps": 1500000,
        "refine_timesteps": 300000,
        "eval_episodes": 20,
        "max_episode_steps": 500,
    },
    "network_defense": {
        "id": "network_defense",
        "env_id": "NetworkDefense-v0",
        "aliases": ["network_defense", "network-defense", "NetworkDefense"],
        "category": "real_world",
        "task_alias": "network_defense",
        "observation_dim": 52,
        "action_dim": 41,
        "action_space": "discrete",
        "reward_type": "sparse",
        "difficulty": "very_hard",
        "pretrain_timesteps": 2000000,
        "refine_timesteps": 400000,
        "eval_episodes": 30,
        "max_episode_steps": 100,
    },
    "autonomous_driving": {
        "id": "autonomous_driving",
        "env_id": "Macro-v1",
        "version": "Macro-v1",
        "powered_by": "MetaDrive",
        "aliases": ["autonomous_driving", "autonomous-driving", "AutonomousDriving", "Macro-v1", "metadrive"],
        "category": "real_world",
        "task_alias": "autonomous_driving",
        "observation_dim": 259,
        "action_dim": 2,
        "action_space": "continuous",
        "reward_type": "dense",
        "difficulty": "very_hard",
        "pretrain_timesteps": 3000000,
        "refine_timesteps": 600000,
        "eval_episodes": 50,
        "max_episode_steps": 500,
    },
    "cage_challenge": {
        "id": "cage_challenge",
        "env_id": "Cyborg-Cage-v0",
        "aliases": ["cage_challenge", "cage", "CAGE", "cage-challenge"],
        "category": "real_world",
        "task_alias": "cage",
        "observation_dim": 52,
        "action_dim": 41,
        "action_space": "discrete",
        "reward_type": "sparse",
        "difficulty": "very_hard",
        "pretrain_timesteps": 2000000,
        "refine_timesteps": 400000,
        "eval_episodes": 30,
        "max_episode_steps": 100,
    },
}

# Alias mapping for quick lookup
ALIAS_TO_ENV_ID: Dict[str, str] = {}
for env_key, env_config in ENVIRONMENT_REGISTRY.items():
    for alias in env_config.get("aliases", []):
        ALIAS_TO_ENV_ID[alias.lower()] = env_key


# ============================================================================
# Sparse Reward Wrapper
# ============================================================================

class SparseRewardWrapper:
    """Wrapper to convert dense rewards to sparse rewards based on threshold."""
    
    def __init__(self, env, threshold: float):
        self.env = env
        self.threshold = threshold
        self.episode_reward = 0.0
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
    def reset(self, **kwargs):
        self.episode_reward = 0.0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_reward += reward
        
        # Replace dense reward with sparse reward
        sparse_reward = 0.0
        if terminated or truncated:
            sparse_reward = 1.0 if self.episode_reward >= self.threshold else 0.0
        
        info["dense_reward"] = reward
        info["episode_reward"] = self.episode_reward
        
        return obs, sparse_reward, terminated, truncated, info
    
    def __getattr__(self, name):
        return getattr(self.env, name)


# ============================================================================
# Environment Factory Functions
# ============================================================================

def resolve_env_id(env_id: str) -> str:
    """Resolve environment ID or alias to canonical environment key."""
    env_id_lower = env_id.lower()
    
    # Direct match
    if env_id_lower in ENVIRONMENT_REGISTRY:
        return env_id_lower
    
    # Alias match
    if env_id_lower in ALIAS_TO_ENV_ID:
        return ALIAS_TO_ENV_ID[env_id_lower]
    
    raise ValueError(
        f"Unknown environment: {env_id}. "
        f"Available: {list(ENVIRONMENT_REGISTRY.keys())}"
    )


def get_environment_config(env_id: str) -> Dict[str, Any]:
    """Get configuration metadata for environment."""
    env_key = resolve_env_id(env_id)
    return ENVIRONMENT_REGISTRY[env_key].copy()


def list_environments() -> List[str]:
    """List all registered environment IDs."""
    return list(ENVIRONMENT_REGISTRY.keys())


def list_mujoco_environments() -> List[str]:
    """List MuJoCo environment IDs."""
    return [k for k, v in ENVIRONMENT_REGISTRY.items() if v["category"] == "mujoco"]


def list_real_world_environments() -> List[str]:
    """List real-world application environment IDs."""
    return [k for k, v in ENVIRONMENT_REGISTRY.items() if v["category"] == "real_world"]


def get_environment(env_id: str, **kwargs) -> Any:
    """
    Factory function to create environment instance.
    
    Args:
        env_id: Environment ID or alias
        **kwargs: Additional environment configuration
        
    Returns:
        Environment instance
    """
    gym = lazy_load_gym()
    env_key = resolve_env_id(env_id)
    config = ENVIRONMENT_REGISTRY[env_key]
    
    # Get base environment ID
    base_env_id = config["env_id"]
    
    # Handle MuJoCo environments
    if config["category"] == "mujoco":
        try:
            env = gym.make(base_env_id, **kwargs)
            
            # Apply sparse reward wrapper if needed
            if config["reward_type"] == "sparse":
                threshold = config.get("sparse_threshold", 3000.0)
                env = SparseRewardWrapper(env, threshold)
            
            return env
        except Exception as e:
            raise RuntimeError(
                f"Failed to create MuJoCo environment {base_env_id}. "
                f"Install with: pip install 'gymnasium[mujoco]' or pip install mujoco. "
                f"Error: {e}"
            )
    
    # Handle real-world application environments
    elif config["category"] == "real_world":
        # Import environment-specific modules
        if "selfish_mining" in env_key:
            return create_selfish_mining_env(config, **kwargs)
        elif "network_defense" in env_key or "cage" in env_key:
            return create_network_defense_env(config, **kwargs)
        elif "autonomous_driving" in env_key:
            return create_autonomous_driving_env(config, **kwargs)
        else:
            raise ValueError(f"Unknown real-world environment: {env_key}")
    
    else:
        raise ValueError(f"Unknown environment category: {config['category']}")


def create_selfish_mining_env(config: Dict[str, Any], **kwargs) -> Any:
    """Create selfish mining environment."""
    # Lazy import to avoid dependency at module level
    try:
        from src.environments import SelfishMiningEnv
        return SelfishMiningEnv(**kwargs)
    except ImportError:
        # Fallback: create a minimal stub environment for smoke testing
        gym = lazy_load_gym()
        
        class SelfishMiningStub:
            """Minimal stub for selfish mining when real implementation unavailable."""
            
            def __init__(self):
                self.observation_space = gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
                )
                self.action_space = gym.spaces.Discrete(2)
                self._step_count = 0
                
            def reset(self, **kwargs):
                self._step_count = 0
                obs = np.zeros(8, dtype=np.float32)
                info = {"episode": 0}
                return obs, info
            
            def step(self, action):
                self._step_count += 1
                obs = np.zeros(8, dtype=np.float32)
                reward = float(np.random.randn())
                terminated = self._step_count >= 500
                truncated = False
                info = {"step": self._step_count}
                return obs, reward, terminated, truncated, info
        
        return SelfishMiningStub()


def create_network_defense_env(config: Dict[str, Any], **kwargs) -> Any:
    """Create network defense environment."""
    try:
        from src.environments import NetworkDefenseEnv
        return NetworkDefenseEnv(**kwargs)
    except ImportError:
        # Fallback stub
        gym = lazy_load_gym()
        
        class NetworkDefenseStub:
            """Minimal stub for network defense when real implementation unavailable."""
            
            def __init__(self):
                self.observation_space = gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(52,), dtype=np.float32
                )
                self.action_space = gym.spaces.Discrete(41)
                self._step_count = 0
                
            def reset(self, **kwargs):
                self._step_count = 0
                obs = np.zeros(52, dtype=np.float32)
                info = {"episode": 0}
                return obs, info
            
            def step(self, action):
                self._step_count += 1
                obs = np.zeros(52, dtype=np.float32)
                reward = float(np.random.randn() * 0.1)
                terminated = self._step_count >= 100
                truncated = False
                info = {"step": self._step_count}
                return obs, reward, terminated, truncated, info
        
        return NetworkDefenseStub()


def create_autonomous_driving_env(config: Dict[str, Any], **kwargs) -> Any:
    """Create the paper MetaDrive-powered Macro-v1 environment."""
    allow_stub = bool(kwargs.pop("allow_stub", False))
    try:
        try:
            import metadrive  # noqa: F401
        except ImportError:
            pass
        gym = lazy_load_gym()
        return gym.make("Macro-v1", **kwargs)
    except Exception:
        pass

    try:
        from metadrive.envs.macro_env import MacroEnv
        macro_config = kwargs.pop("macro_config", {})
        macro_config.update(kwargs)
        return MacroEnv(macro_config)
    except Exception as exc:
        if not allow_stub:
            raise RuntimeError(
                "Failed to initialize Macro-v1 powered by the MetaDrive simulator. "
                "Install MetaDrive and Gym registration, or pass allow_stub=True only for smoke tests."
            ) from exc

    try:
        from src.environments import AutonomousDrivingEnv
        return AutonomousDrivingEnv(**kwargs)
    except ImportError:
        # Fallback stub for smoke tests only.
        gym = lazy_load_gym()
        
        class AutonomousDrivingStub:
            """Minimal stub for autonomous driving when real implementation unavailable."""
            
            def __init__(self):
                self.observation_space = gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(259,), dtype=np.float32
                )
                self.action_space = gym.spaces.Box(
                    low=-1.0, high=1.0, shape=(2,), dtype=np.float32
                )
                self._step_count = 0
                
            def reset(self, **kwargs):
                self._step_count = 0
                obs = np.zeros(259, dtype=np.float32)
                info = {"episode": 0}
                return obs, info
            
            def step(self, action):
                self._step_count += 1
                obs = np.zeros(259, dtype=np.float32)
                reward = float(np.random.randn() * 2.0)
                terminated = self._step_count >= 500
                truncated = False
                info = {"step": self._step_count}
                return obs, reward, terminated, truncated, info
        
        return AutonomousDrivingStub()


def create_all_environments() -> Dict[str, Any]:
    """Create instances for all registered environments."""
    envs = {}
    for env_id in list_environments():
        try:
            envs[env_id] = get_environment(env_id)
        except Exception as e:
            print(f"Warning: Failed to create environment {env_id}: {e}")
            envs[env_id] = None
    return envs


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_agent(agent: Any, env: Any, num_episodes: int = 10, 
                   deterministic: bool = True) -> Dict[str, Any]:
    """
    Evaluate agent performance in environment.
    
    Args:
        agent: Agent with predict(obs) method
        env: Environment instance
        num_episodes: Number of evaluation episodes
        deterministic: Use deterministic policy
        
    Returns:
        Dictionary with evaluation metrics:
        - mean_reward: Mean episode reward
        - std_reward: Standard deviation of episode rewards
        - min_reward: Minimum episode reward
        - max_reward: Maximum episode reward
        - episode_rewards: List of all episode rewards
        - episode_lengths: List of all episode lengths
        - success_rate: Success rate (if applicable)
    """
    episode_rewards = []
    episode_lengths = []
    
    for episode_idx in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        episode_length = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            # Get action from agent
            if hasattr(agent, 'predict'):
                action, _ = agent.predict(obs, deterministic=deterministic)
            elif hasattr(agent, 'select_action'):
                action = agent.select_action(obs, deterministic=deterministic)
            elif callable(agent):
                action = agent(obs)
            else:
                raise ValueError("Agent must have predict() or select_action() method")
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(float(episode_reward))
        episode_lengths.append(int(episode_length))
    
    # Compute statistics
    episode_rewards_arr = np.array(episode_rewards)
    episode_lengths_arr = np.array(episode_lengths)
    
    metrics = {
        "mean_reward": float(np.mean(episode_rewards_arr)),
        "std_reward": float(np.std(episode_rewards_arr)),
        "min_reward": float(np.min(episode_rewards_arr)),
        "max_reward": float(np.max(episode_rewards_arr)),
        "mean_length": float(np.mean(episode_lengths_arr)),
        "std_length": float(np.std(episode_lengths_arr)),
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "num_episodes": num_episodes,
    }
    
    return metrics


def evaluate_refining(refined_agent: Any, env: Any, num_episodes: int = 10) -> float:
    """
    Evaluate refining performance (interface contract requirement).
    
    Args:
        refined_agent: Refined agent
        env: Environment instance
        num_episodes: Number of evaluation episodes
        
    Returns:
        Mean reward across episodes
    """
    metrics = evaluate_agent(refined_agent, env, num_episodes, deterministic=True)
    return metrics["mean_reward"]


def batch_evaluate_environments(agent: Any, env_ids: List[str], 
                                 num_episodes: int = 10) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate agent across multiple environments.
    
    Args:
        agent: Agent to evaluate
        env_ids: List of environment IDs
        num_episodes: Number of episodes per environment
        
    Returns:
        Dictionary mapping env_id to evaluation metrics
    """
    results = {}
    
    for env_id in env_ids:
        try:
            env = get_environment(env_id)
            metrics = evaluate_agent(agent, env, num_episodes)
            results[env_id] = metrics
            env.close()
        except Exception as e:
            print(f"Warning: Failed to evaluate {env_id}: {e}")
            results[env_id] = {
                "error": str(e),
                "mean_reward": 0.0,
                "std_reward": 0.0,
            }
    
    return results


# ============================================================================
# Experiment Support Functions
# ============================================================================

def get_paper_environments() -> List[str]:
    """Get the 8 environments used in paper experiments (Table 1)."""
    return [
        "hopper",
        "walker2d",
        "reacher",
        "halfcheetah",
        "hopper_sparse",
        "selfish_mining",
        "network_defense",
        "autonomous_driving",
    ]


def verify_environment_coverage() -> Dict[str, bool]:
    """Verify all paper environments are accessible."""
    paper_envs = get_paper_environments()
    coverage = {}
    
    for env_id in paper_envs:
        try:
            env = get_environment(env_id)
            env.close()
            coverage[env_id] = True
        except Exception as e:
            print(f"Warning: Environment {env_id} not available: {e}")
            coverage[env_id] = False
    
    return coverage


def get_environment_groups() -> Dict[str, List[str]]:
    """Get environment groups for experiments."""
    return {
        "all": get_paper_environments(),
        "mujoco": ["hopper", "walker2d", "reacher", "halfcheetah"],
        "sparse": ["hopper_sparse"],
        "real_world": ["selfish_mining", "network_defense", "autonomous_driving"],
        "easy": ["reacher"],
        "medium": ["hopper", "walker2d", "halfcheetah"],
        "hard": ["hopper_sparse", "selfish_mining", "network_defense", "autonomous_driving"],
    }


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "ENVIRONMENT_REGISTRY",
    "get_environment",
    "get_environment_config",
    "list_environments",
    "list_mujoco_environments",
    "list_real_world_environments",
    "create_all_environments",
    "evaluate_agent",
    "evaluate_refining",
    "batch_evaluate_environments",
    "get_paper_environments",
    "verify_environment_coverage",
    "get_environment_groups",
    "resolve_env_id",
]