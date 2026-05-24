"""
RICE Environment Registry and Factory Module

Implements environment/task registry with factory functions for:
- MuJoCo continuous control tasks (Hopper, Walker2d, Reacher, HalfCheetah)
- Sparse reward variants
- Real-world applications (selfish mining, network defense, autonomous driving, malware)
- Environment wrappers and configuration support
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
import numpy as np


# ============================================================================
# Environment Registry
# ============================================================================

ENVIRONMENT_REGISTRY = {
    # MuJoCo Standard Environments
    "hopper": {
        "id": "Hopper-v3",
        "aliases": ["hopper", "hopper-v3", "Hopper", "Hopper-v3"],
        "category": "mujoco",
        "observation_dim": 11,
        "action_dim": 3,
        "action_space": "continuous",
        "reward_type": "dense",
        "difficulty": "medium",
        "factory": "make_mujoco_env"
    },
    "walker2d": {
        "id": "Walker2d-v3",
        "aliases": ["walker2d", "walker2d-v3", "Walker2d", "Walker2d-v3"],
        "category": "mujoco",
        "observation_dim": 17,
        "action_dim": 6,
        "action_space": "continuous",
        "reward_type": "dense",
        "difficulty": "medium",
        "factory": "make_mujoco_env",
        "normalize_observations": True,
        "normalization_scope": "training_drl_agents"
    },
    "reacher": {
        "id": "Reacher-v2",
        "aliases": ["reacher", "reacher-v2", "Reacher", "Reacher-v2"],
        "category": "mujoco",
        "observation_dim": 11,
        "action_dim": 2,
        "action_space": "continuous",
        "reward_type": "dense",
        "difficulty": "easy",
        "factory": "make_mujoco_env"
    },
    "halfcheetah": {
        "id": "HalfCheetah-v3",
        "aliases": ["halfcheetah", "halfcheetah-v3", "HalfCheetah", "HalfCheetah-v3", "cheetah"],
        "category": "mujoco",
        "observation_dim": 17,
        "action_dim": 6,
        "action_space": "continuous",
        "reward_type": "dense",
        "difficulty": "medium",
        "factory": "make_mujoco_env",
        "normalize_observations": True,
        "normalization_scope": "training_drl_agents"
    },
    
    # Sparse Reward Variants
    "hopper-sparse": {
        "id": "Hopper-v3",
        "aliases": ["hopper-sparse", "Hopper-sparse"],
        "category": "mujoco",
        "observation_dim": 11,
        "action_dim": 3,
        "action_space": "continuous",
        "reward_type": "sparse",
        "difficulty": "hard",
        "factory": "make_sparse_mujoco_env",
        "base_env": "Hopper-v3",
        "sparse_threshold": 3000.0
    },
    "walker2d-sparse": {
        "id": "Walker2d-v3",
        "aliases": ["walker2d-sparse", "Walker2d-sparse"],
        "category": "mujoco",
        "observation_dim": 17,
        "action_dim": 6,
        "action_space": "continuous",
        "reward_type": "sparse",
        "difficulty": "hard",
        "factory": "make_sparse_mujoco_env",
        "base_env": "Walker2d-v3",
        "sparse_threshold": 4000.0
    },
    "halfcheetah-sparse": {
        "id": "HalfCheetah-v3",
        "aliases": ["halfcheetah-sparse", "HalfCheetah-sparse"],
        "category": "mujoco",
        "observation_dim": 17,
        "action_dim": 6,
        "action_space": "continuous",
        "reward_type": "sparse",
        "difficulty": "hard",
        "factory": "make_sparse_mujoco_env",
        "base_env": "HalfCheetah-v3",
        "sparse_threshold": 10000.0,
        "normalize_observations": True,
        "normalization_scope": "training_drl_agents"
    },
    
    # Real-world Application Environments
    "selfish_mining": {
        "id": "SelfishMining-v0",
        "aliases": ["selfish_mining", "selfish-mining", "bitcoin", "mining"],
        "category": "blockchain",
        "observation_dim": 6,
        "action_dim": 2,
        "action_space": "discrete",
        "reward_type": "dense",
        "difficulty": "hard",
        "factory": "make_selfish_mining_env",
        "source": "pto-selfish-mining",
        "description": "Bitcoin selfish mining attack simulation compatible with pto-selfish-mining"
    },
    "network_defense": {
        "id": "CyberDefense-v0",
        "aliases": ["network_defense", "network-defense", "cage", "cyber", "cyborg"],
        "category": "cybersecurity",
        "observation_dim": "variable",
        "action_dim": "variable",
        "action_space": "discrete",
        "reward_type": "sparse",
        "difficulty": "hard",
        "factory": "make_network_defense_env",
        "description": "CAGE Challenge 2 network defense environment"
    },
    "autonomous_driving": {
        "id": "Macro-v1",
        "aliases": ["autonomous_driving", "autonomous-driving", "metadrive", "Macro-v1", "macro-v1", "driving"],
        "category": "robotics",
        "observation_dim": "variable",
        "action_dim": 2,
        "action_space": "continuous",
        "reward_type": "dense",
        "difficulty": "hard",
        "factory": "make_autonomous_driving_env",
        "powered_by": "MetaDrive",
        "description": "Macro-v1 autonomous driving environment powered by the MetaDrive simulator"
    },
    "malware_mutation": {
        "id": "MalwareMutation-v0",
        "aliases": ["malware_mutation", "malware-mutation", "malware", "evasion"],
        "category": "cybersecurity",
        "observation_dim": 2381,
        "action_dim": 50,
        "action_space": "discrete",
        "reward_type": "sparse",
        "difficulty": "hard",
        "factory": "make_malware_env",
        "description": "Malware mutation for antivirus evasion"
    }
}


# Dataset/Benchmark Registry
DATASET_REGISTRY = {
    "mujoco": {
        "environments": ["hopper", "walker2d", "reacher", "halfcheetah", 
                        "hopper-sparse", "walker2d-sparse", "halfcheetah-sparse"],
        "type": "continuous_control",
        "source": "gym/mujoco",
        "requires_license": False
    },
    "selfish_mining": {
        "environments": ["selfish_mining"],
        "type": "blockchain_simulation",
        "source": "custom",
        "requires_license": False
    },
    "cage": {
        "environments": ["network_defense"],
        "type": "cybersecurity",
        "source": "CAGE Challenge 2",
        "requires_license": False
    },
    "robotics": {
        "environments": ["autonomous_driving"],
        "type": "robotics_simulation",
        "source": "CARLA/custom",
        "requires_license": False
    }
}


# ============================================================================
# Environment Factory Functions
# ============================================================================

def make_mujoco_env(env_id: str, config: Optional[Dict[str, Any]] = None) -> Any:
    """Factory for standard MuJoCo environments."""
    try:
        import gymnasium as gym
    except ImportError:
        import gym
    
    if config is None:
        config = {}
    
    seed = config.get("seed", None)
    render_mode = config.get("render_mode", None)
    
    try:
        env = gym.make(env_id, render_mode=render_mode)
    except TypeError:
        # Fallback for older gym versions
        env = gym.make(env_id)
    
    if seed is not None:
        env.reset(seed=seed) if hasattr(env, 'reset') else env.seed(seed)
    
    # Apply normalization wrapper if requested
    if config.get("normalize_observations", False):
        env = NormalizationWrapper(env, obs_norm=True)
    
    if config.get("normalize_rewards", False):
        env = NormalizationWrapper(env, reward_norm=True)
    
    return env


def make_sparse_mujoco_env(env_id: str, config: Optional[Dict[str, Any]] = None) -> Any:
    """Factory for sparse reward MuJoCo environments."""
    if config is None:
        config = {}
    
    # Create base environment
    base_env = make_mujoco_env(env_id, config)
    
    # Get threshold from config or registry
    env_key = config.get("env_key", "hopper-sparse")
    threshold = config.get("sparse_threshold", 
                          ENVIRONMENT_REGISTRY.get(env_key, {}).get("sparse_threshold", 3000.0))
    
    # Wrap with sparse reward wrapper
    sparse_env = SparseRewardWrapper(base_env, threshold=threshold)
    
    return sparse_env


def make_selfish_mining_env(env_id: str, config: Optional[Dict[str, Any]] = None) -> Any:
    """Factory for selfish mining blockchain environment."""
    if config is None:
        config = {}
    
    # Lazy import to avoid dependency at module load
    try:
        from .environments.selfish_mining import SelfishMiningEnv
    except ImportError:
        # Fallback implementation
        return SelfishMiningFallbackEnv(config)
    
    alpha = config.get("alpha", 0.3)  # Attacker mining power
    gamma = config.get("gamma", 0.5)  # Network propagation advantage
    max_fork_length = config.get("max_fork_length", 50)
    
    env = SelfishMiningEnv(alpha=alpha, gamma=gamma, max_fork_length=max_fork_length)
    return env


def make_network_defense_env(env_id: str, config: Optional[Dict[str, Any]] = None) -> Any:
    """Factory for CAGE Challenge network defense environment."""
    if config is None:
        config = {}
    
    try:
        from CybORG import CybORG
        from CybORG.Agents import B_lineAgent, SleepAgent
        scenario = config.get("scenario", "Scenario2")
        cyborg = CybORG(scenario_file=scenario, environment='sim')
        env = CybORG.get_wrapper(cyborg)
        return env
    except ImportError:
        # Fallback implementation
        return NetworkDefenseFallbackEnv(config)


def make_autonomous_driving_env(env_id: str, config: Optional[Dict[str, Any]] = None) -> Any:
    """Factory for autonomous driving environment."""
    if config is None:
        config = {}
    
    try:
        import metadrive  # noqa: F401
        import gymnasium as gym
        return gym.make("Macro-v1")
    except Exception:
        try:
            from .environments.autonomous_driving import AutonomousDrivingEnv
        except ImportError:
            # Fallback implementation preserves the Macro-v1/MetaDrive contract.
            return AutonomousDrivingFallbackEnv(config)
    
    town = config.get("town", "Town01")
    weather = config.get("weather", "ClearNoon")
    env = AutonomousDrivingEnv(town=town, weather=weather)
    return env


def make_malware_env(env_id: str, config: Optional[Dict[str, Any]] = None) -> Any:
    """Factory for malware mutation environment."""
    if config is None:
        config = {}
    
    try:
        import gym
        env = gym.make("malware-v0")
        return env
    except:
        # Fallback implementation
        return MalwareMutationFallbackEnv(config)


# ============================================================================
# Environment Wrappers
# ============================================================================

class NormalizationWrapper:
    """Wrapper for observation and reward normalization."""
    
    def __init__(self, env, obs_norm: bool = False, reward_norm: bool = False):
        self.env = env
        self.obs_norm = obs_norm
        self.reward_norm = reward_norm
        
        if obs_norm:
            self.obs_mean = np.zeros(env.observation_space.shape)
            self.obs_std = np.ones(env.observation_space.shape)
            self.obs_count = 0
        
        if reward_norm:
            self.reward_mean = 0.0
            self.reward_std = 1.0
            self.reward_count = 0
    
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if self.obs_norm:
            obs = self._normalize_obs(obs)
        return obs
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        
        if self.obs_norm:
            obs = self._normalize_obs(obs)
        
        if self.reward_norm:
            reward = self._normalize_reward(reward)
        
        return obs, reward, done, truncated, info
    
    def _normalize_obs(self, obs):
        if isinstance(obs, tuple):
            obs = obs[0]
        if self.obs_count > 0:
            obs = (obs - self.obs_mean) / (self.obs_std + 1e-8)
        self.obs_count += 1
        return obs
    
    def _normalize_reward(self, reward):
        if self.reward_count > 0:
            reward = (reward - self.reward_mean) / (self.reward_std + 1e-8)
        self.reward_count += 1
        return reward
    
    def __getattr__(self, name):
        return getattr(self.env, name)


class SparseRewardWrapper:
    """Wrapper that converts dense rewards to sparse rewards."""
    
    def __init__(self, env, threshold: float):
        self.env = env
        self.threshold = threshold
        self.episode_reward = 0.0
    
    def reset(self, **kwargs):
        self.episode_reward = 0.0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Accumulate episode reward
        self.episode_reward += reward
        
        # Only give reward at episode end if threshold exceeded
        if done or truncated:
            sparse_reward = 1.0 if self.episode_reward >= self.threshold else 0.0
            return obs, sparse_reward, done, truncated, info
        else:
            return obs, 0.0, done, truncated, info
    
    def __getattr__(self, name):
        return getattr(self.env, name)


# ============================================================================
# Fallback Environment Implementations
# ============================================================================

class SelfishMiningFallbackEnv:
    """Lightweight fallback for selfish mining when full environment unavailable."""
    
    def __init__(self, config: Dict[str, Any]):
        self.alpha = config.get("alpha", 0.3)
        self.gamma = config.get("gamma", 0.5)
        self.max_fork_length = config.get("max_fork_length", 50)
        
        self.observation_space = self._make_observation_space()
        self.action_space = self._make_action_space()
        
        self.state = None
        self.steps = 0
    
    def _make_observation_space(self):
        try:
            import gymnasium as gym
            return gym.spaces.Box(low=0, high=self.max_fork_length, shape=(6,), dtype=np.float32)
        except ImportError:
            import gym
            return gym.spaces.Box(low=0, high=self.max_fork_length, shape=(6,), dtype=np.float32)
    
    def _make_action_space(self):
        try:
            import gymnasium as gym
            return gym.spaces.Discrete(2)
        except ImportError:
            import gym
            return gym.spaces.Discrete(2)
    
    def reset(self, **kwargs):
        self.state = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
        self.steps = 0
        return self.state, {}
    
    def step(self, action):
        self.steps += 1
        
        # Simple dynamics simulation
        reward = np.random.randn() * 0.1
        done = self.steps >= 1000
        truncated = False
        
        self.state = self.state + np.random.randn(6) * 0.1
        self.state = np.clip(self.state, 0, self.max_fork_length)
        
        return self.state, reward, done, truncated, {}


class NetworkDefenseFallbackEnv:
    """Lightweight fallback for network defense when CAGE unavailable."""
    
    def __init__(self, config: Dict[str, Any]):
        self.num_hosts = config.get("num_hosts", 10)
        self.observation_space = self._make_observation_space()
        self.action_space = self._make_action_space()
        self.state = None
        self.steps = 0
    
    def _make_observation_space(self):
        try:
            import gymnasium as gym
            return gym.spaces.Box(low=0, high=1, shape=(self.num_hosts * 5,), dtype=np.float32)
        except ImportError:
            import gym
            return gym.spaces.Box(low=0, high=1, shape=(self.num_hosts * 5,), dtype=np.float32)
    
    def _make_action_space(self):
        try:
            import gymnasium as gym
            return gym.spaces.Discrete(self.num_hosts * 3)
        except ImportError:
            import gym
            return gym.spaces.Discrete(self.num_hosts * 3)
    
    def reset(self, **kwargs):
        self.state = np.random.rand(self.num_hosts * 5).astype(np.float32)
        self.steps = 0
        return self.state, {}
    
    def step(self, action):
        self.steps += 1
        reward = np.random.randn() * 0.1
        done = self.steps >= 500
        truncated = False
        self.state = np.random.rand(self.num_hosts * 5).astype(np.float32)
        return self.state, reward, done, truncated, {}


class AutonomousDrivingFallbackEnv:
    """Lightweight fallback for autonomous driving when CARLA unavailable."""
    
    def __init__(self, config: Dict[str, Any]):
        self.observation_space = self._make_observation_space()
        self.action_space = self._make_action_space()
        self.state = None
        self.steps = 0
    
    def _make_observation_space(self):
        try:
            import gymnasium as gym
            return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        except ImportError:
            import gym
            return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
    
    def _make_action_space(self):
        try:
            import gymnasium as gym
            return gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        except ImportError:
            import gym
            return gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
    
    def reset(self, **kwargs):
        self.state = np.random.randn(10).astype(np.float32)
        self.steps = 0
        return self.state, {}
    
    def step(self, action):
        self.steps += 1
        reward = np.random.randn() * 0.1
        done = self.steps >= 1000
        truncated = False
        self.state = self.state + np.random.randn(10) * 0.1
        return self.state, reward, done, truncated, {}


class MalwareMutationFallbackEnv:
    """Lightweight fallback for malware mutation when full environment unavailable."""
    
    def __init__(self, config: Dict[str, Any]):
        self.feature_dim = config.get("feature_dim", 2381)
        self.action_dim = config.get("action_dim", 50)
        
        self.observation_space = self._make_observation_space()
        self.action_space = self._make_action_space()
        self.state = None
        self.steps = 0
    
    def _make_observation_space(self):
        try:
            import gymnasium as gym
            return gym.spaces.Box(low=0, high=1, shape=(self.feature_dim,), dtype=np.float32)
        except ImportError:
            import gym
            return gym.spaces.Box(low=0, high=1, shape=(self.feature_dim,), dtype=np.float32)
    
    def _make_action_space(self):
        try:
            import gymnasium as gym
            return gym.spaces.Discrete(self.action_dim)
        except ImportError:
            import gym
            return gym.spaces.Discrete(self.action_dim)
    
    def reset(self, **kwargs):
        self.state = np.random.rand(self.feature_dim).astype(np.float32)
        self.steps = 0
        return self.state, {}
    
    def step(self, action):
        self.steps += 1
        reward = np.random.randn() * 0.1
        done = self.steps >= 100
        truncated = False
        self.state = np.random.rand(self.feature_dim).astype(np.float32)
        return self.state, reward, done, truncated, {}


# ============================================================================
# Environment Resolution and Lookup
# ============================================================================

def resolve_environment(env_name: str) -> Dict[str, Any]:
    """Resolve environment name to registry entry."""
    env_name_lower = env_name.lower().replace("_", "-")
    
    # Direct lookup
    if env_name_lower in ENVIRONMENT_REGISTRY:
        return ENVIRONMENT_REGISTRY[env_name_lower]
    
    # Search by alias
    for key, meta in ENVIRONMENT_REGISTRY.items():
        aliases = [a.lower().replace("_", "-") for a in meta.get("aliases", [])]
        if env_name_lower in aliases:
            return meta
    
    raise ValueError(f"Environment '{env_name}' not found in registry")


def make_environment(env_name: str, config: Optional[Dict[str, Any]] = None) -> Any:
    """Main factory function to create any registered environment."""
    meta = resolve_environment(env_name)
    factory_name = meta.get("factory", "make_mujoco_env")
    
    factory_fn = globals().get(factory_name)
    if factory_fn is None:
        raise ValueError(f"Factory function '{factory_name}' not found")
    
    env_id = meta.get("id", env_name)
    
    if config is None:
        config = {}
    
    config["env_key"] = env_name.lower()
    
    env = factory_fn(env_id, config)
    return env


def list_environments(category: Optional[str] = None) -> List[str]:
    """List all registered environments, optionally filtered by category."""
    if category is None:
        return list(ENVIRONMENT_REGISTRY.keys())
    else:
        return [k for k, v in ENVIRONMENT_REGISTRY.items() 
                if v.get("category") == category]


def get_environment_info(env_name: str) -> Dict[str, Any]:
    """Get metadata for a registered environment."""
    return resolve_environment(env_name)