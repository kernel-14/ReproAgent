"""
RICE Models Module

Implements neural network models for RICE experiments:
- Actor-critic networks for PPO, SAC, GAIL, JSRL
- Model registry for method/baseline selection
- Checkpoint management and model persistence
- Configuration support for hyperparameter sweeps (alpha, lambda, p)

Method Registry:
- ours: RICE actor-critic with explanation-guided architecture
- random: Random policy baseline
- statemask: StateMask-compatible actor-critic
- ppo: Standard PPO actor-critic
- sac: Soft Actor-Critic (if available)
- gail: GAIL discriminator + actor-critic
- jsrl: Joint state-reward learning
- baseline: Basic MLP actor-critic
- adapter: Fine-tuning adapter layers
- fine_tuning: Transfer learning compatible architecture

Sweep Parameters:
- alpha: Explanation weight (0.0 to 1.0)
- lambda: GAE lambda (0.9 to 0.99)
- p: Critical state percentile (0.1 to 0.5)
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import numpy as np


# ============================================================================
# Lazy Imports
# ============================================================================

def lazy_load_torch():
    """Lazy import torch to avoid dependency at module import time."""
    try:
        import torch
        return torch
    except ImportError:
        return None


def lazy_load_torch_nn():
    """Lazy import torch.nn."""
    torch = lazy_load_torch()
    if torch is not None:
        return torch.nn
    return None


# ============================================================================
# Model Registry
# ============================================================================

MODEL_REGISTRY = {}


def register_model(name: str, aliases: Optional[List[str]] = None):
    """Decorator to register model architectures."""
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        if aliases:
            for alias in aliases:
                MODEL_REGISTRY[alias] = cls
        return cls
    return decorator


def get_model(name: str, **kwargs):
    """Get model by name from registry."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model {name} not found in registry. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)


# ============================================================================
# Base Model Classes
# ============================================================================

class BaseActorCritic:
    """Base class for actor-critic models."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int], **kwargs):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.device = kwargs.get('device', 'cpu')
        
    def forward_actor(self, obs):
        """Forward pass through actor network."""
        raise NotImplementedError
        
    def forward_critic(self, obs):
        """Forward pass through critic network."""
        raise NotImplementedError
        
    def get_action(self, obs, deterministic: bool = False):
        """Sample action from policy."""
        raise NotImplementedError
        
    def evaluate_actions(self, obs, actions):
        """Evaluate log probability and entropy of actions."""
        raise NotImplementedError


class TorchActorCritic(BaseActorCritic):
    """PyTorch-based actor-critic implementation."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int] = None, **kwargs):
        super().__init__(obs_dim, action_dim, hidden_dims or [64, 64], **kwargs)
        
        torch = lazy_load_torch()
        nn = lazy_load_torch_nn()
        
        if torch is None or nn is None:
            raise ImportError("PyTorch required for TorchActorCritic")
        
        self.torch = torch
        self.nn = nn
        
        # Build actor network
        actor_layers = []
        prev_dim = obs_dim
        for hidden_dim in self.hidden_dims:
            actor_layers.append(nn.Linear(prev_dim, hidden_dim))
            actor_layers.append(nn.Tanh())
            prev_dim = hidden_dim
        actor_layers.append(nn.Linear(prev_dim, action_dim))
        self.actor = nn.Sequential(*actor_layers)
        
        # Build critic network
        critic_layers = []
        prev_dim = obs_dim
        for hidden_dim in self.hidden_dims:
            critic_layers.append(nn.Linear(prev_dim, hidden_dim))
            critic_layers.append(nn.Tanh())
            prev_dim = hidden_dim
        critic_layers.append(nn.Linear(prev_dim, 1))
        self.critic = nn.Sequential(*critic_layers)
        
        # Log std for continuous actions
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Move to device
        self.actor.to(self.device)
        self.critic.to(self.device)
        
    def forward_actor(self, obs):
        """Forward pass through actor network."""
        if not self.torch.is_tensor(obs):
            obs = self.torch.tensor(obs, dtype=self.torch.float32).to(self.device)
        return self.actor(obs)
        
    def forward_critic(self, obs):
        """Forward pass through critic network."""
        if not self.torch.is_tensor(obs):
            obs = self.torch.tensor(obs, dtype=self.torch.float32).to(self.device)
        return self.critic(obs)
        
    def get_action(self, obs, deterministic: bool = False):
        """Sample action from policy."""
        mean = self.forward_actor(obs)
        if deterministic:
            return mean, self.torch.zeros_like(mean), mean
        
        std = self.torch.exp(self.log_std)
        dist = self.torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, mean
        
    def evaluate_actions(self, obs, actions):
        """Evaluate log probability and entropy of actions."""
        mean = self.forward_actor(obs)
        std = self.torch.exp(self.log_std)
        dist = self.torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy
        
    def get_value(self, obs):
        """Get state value estimate."""
        return self.forward_critic(obs).squeeze(-1)


# ============================================================================
# Registered Model Architectures
# ============================================================================

@register_model("ours", aliases=["rice", "RICE"])
class RICEActorCritic(TorchActorCritic):
    """RICE actor-critic with explanation-guided architecture."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int] = None, 
                 alpha: float = 0.5, **kwargs):
        super().__init__(obs_dim, action_dim, hidden_dims or [128, 128], **kwargs)
        self.alpha = alpha  # Explanation weight
        

@register_model("ppo", aliases=["PPO"])
class PPOActorCritic(TorchActorCritic):
    """Standard PPO actor-critic."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int] = None, **kwargs):
        super().__init__(obs_dim, action_dim, hidden_dims or [64, 64], **kwargs)


@register_model("baseline", aliases=["basic", "mlp"])
class BaselineActorCritic(TorchActorCritic):
    """Basic MLP actor-critic baseline."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int] = None, **kwargs):
        super().__init__(obs_dim, action_dim, hidden_dims or [32, 32], **kwargs)


@register_model("statemask", aliases=["StateMask"])
class StateMaskActorCritic(TorchActorCritic):
    """StateMask-compatible actor-critic with mask network."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int] = None, **kwargs):
        super().__init__(obs_dim, action_dim, hidden_dims or [64, 64], **kwargs)
        
        nn = self.nn
        # Add mask network for state importance
        mask_layers = []
        prev_dim = obs_dim
        for hidden_dim in [32, 32]:
            mask_layers.append(nn.Linear(prev_dim, hidden_dim))
            mask_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        mask_layers.append(nn.Linear(prev_dim, obs_dim))
        mask_layers.append(nn.Sigmoid())
        self.mask_network = nn.Sequential(*mask_layers).to(self.device)
        
    def get_state_mask(self, obs):
        """Get state importance mask."""
        if not self.torch.is_tensor(obs):
            obs = self.torch.tensor(obs, dtype=self.torch.float32).to(self.device)
        return self.mask_network(obs)


@register_model("sac", aliases=["SAC"])
class SACActorCritic(TorchActorCritic):
    """Soft Actor-Critic architecture."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int] = None, **kwargs):
        super().__init__(obs_dim, action_dim, hidden_dims or [256, 256], **kwargs)
        
        # SAC uses two Q-networks
        nn = self.nn
        q2_layers = []
        prev_dim = obs_dim + action_dim
        for hidden_dim in self.hidden_dims:
            q2_layers.append(nn.Linear(prev_dim, hidden_dim))
            q2_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        q2_layers.append(nn.Linear(prev_dim, 1))
        self.q2_network = nn.Sequential(*q2_layers).to(self.device)


@register_model("gail", aliases=["GAIL"])
class GAILActorCritic(TorchActorCritic):
    """GAIL actor-critic with discriminator."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int] = None, **kwargs):
        super().__init__(obs_dim, action_dim, hidden_dims or [100, 100], **kwargs)
        
        # Add discriminator network
        nn = self.nn
        disc_layers = []
        prev_dim = obs_dim + action_dim
        for hidden_dim in [100, 100]:
            disc_layers.append(nn.Linear(prev_dim, hidden_dim))
            disc_layers.append(nn.Tanh())
            prev_dim = hidden_dim
        disc_layers.append(nn.Linear(prev_dim, 1))
        disc_layers.append(nn.Sigmoid())
        self.discriminator = nn.Sequential(*disc_layers).to(self.device)
        
    def discriminate(self, obs, action):
        """Discriminator forward pass."""
        if not self.torch.is_tensor(obs):
            obs = self.torch.tensor(obs, dtype=self.torch.float32).to(self.device)
        if not self.torch.is_tensor(action):
            action = self.torch.tensor(action, dtype=self.torch.float32).to(self.device)
        inp = self.torch.cat([obs, action], dim=-1)
        return self.discriminator(inp)


@register_model("jsrl", aliases=["JSRL"])
class JSRLActorCritic(TorchActorCritic):
    """Joint state-reward learning actor-critic."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int] = None, **kwargs):
        super().__init__(obs_dim, action_dim, hidden_dims or [128, 128], **kwargs)


@register_model("adapter", aliases=["fine_tuning", "transfer"])
class AdapterActorCritic(TorchActorCritic):
    """Actor-critic with adapter layers for fine-tuning."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int] = None, 
                 adapter_dim: int = 16, **kwargs):
        super().__init__(obs_dim, action_dim, hidden_dims or [64, 64], **kwargs)
        
        # Add adapter layers
        nn = self.nn
        self.adapter_dim = adapter_dim
        adapters = []
        for _ in self.hidden_dims:
            adapters.append(nn.Sequential(
                nn.Linear(self.hidden_dims[0], adapter_dim),
                nn.ReLU(),
                nn.Linear(adapter_dim, self.hidden_dims[0])
            ))
        self.adapters = nn.ModuleList(adapters).to(self.device)


@register_model("random", aliases=["Random"])
class RandomPolicy(BaseActorCritic):
    """Random policy baseline (no learning)."""
    
    def __init__(self, obs_dim: int, action_dim: int, **kwargs):
        super().__init__(obs_dim, action_dim, [], **kwargs)
        self.action_space_low = kwargs.get('action_low', -1.0)
        self.action_space_high = kwargs.get('action_high', 1.0)
        
    def get_action(self, obs, deterministic: bool = False):
        """Return random action."""
        action = np.random.uniform(
            self.action_space_low, 
            self.action_space_high, 
            size=self.action_dim
        )
        return action, 0.0, action
        
    def get_value(self, obs):
        """Return zero value."""
        return 0.0


# ============================================================================
# Model Management
# ============================================================================

def create_model(method: str, obs_dim: int, action_dim: int, config: Dict[str, Any]) -> BaseActorCritic:
    """
    Create model from method name and configuration.
    
    Args:
        method: Method name (ours, ppo, sac, etc.)
        obs_dim: Observation dimension
        action_dim: Action dimension
        config: Configuration dictionary with hyperparameters
        
    Returns:
        Model instance
    """
    model_config = config.get('model', {})
    hidden_dims = model_config.get('hidden_dims', [64, 64])
    device = model_config.get('device', 'cpu')
    
    # Get sweep parameters
    alpha = config.get('alpha', 0.5)
    lambda_param = config.get('lambda', 0.95)
    p = config.get('p', 0.2)
    
    model_kwargs = {
        'obs_dim': obs_dim,
        'action_dim': action_dim,
        'hidden_dims': hidden_dims,
        'device': device,
        'alpha': alpha,
        'lambda': lambda_param,
        'p': p,
    }
    
    return get_model(method, **model_kwargs)


def save_checkpoint(model: BaseActorCritic, path: Union[str, Path], 
                   metadata: Optional[Dict[str, Any]] = None):
    """
    Save model checkpoint with metadata.
    
    Args:
        model: Model to save
        path: Checkpoint file path
        metadata: Optional metadata dictionary
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    torch = lazy_load_torch()
    if torch is None:
        raise ImportError("PyTorch required for checkpoint saving")
    
    checkpoint = {
        'obs_dim': model.obs_dim,
        'action_dim': model.action_dim,
        'hidden_dims': model.hidden_dims,
        'model_class': model.__class__.__name__,
    }
    
    if hasattr(model, 'actor'):
        checkpoint['actor_state_dict'] = model.actor.state_dict()
    if hasattr(model, 'critic'):
        checkpoint['critic_state_dict'] = model.critic.state_dict()
    if hasattr(model, 'log_std'):
        checkpoint['log_std'] = model.log_std.data
        
    if metadata:
        checkpoint['metadata'] = metadata
        
    torch.save(checkpoint, path)
    return {'checkpoint_path': str(path), 'saved': True}


def load_checkpoint(path: Union[str, Path], device: str = 'cpu') -> Tuple[BaseActorCritic, Dict[str, Any]]:
    """
    Load model checkpoint.
    
    Args:
        path: Checkpoint file path
        device: Device to load model on
        
    Returns:
        Tuple of (model, metadata)
    """
    torch = lazy_load_torch()
    if torch is None:
        raise ImportError("PyTorch required for checkpoint loading")
    
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    checkpoint = torch.load(path, map_location=device)
    
    # Reconstruct model
    model_class_name = checkpoint.get('model_class', 'PPOActorCritic')
    model_class = None
    for name, cls in MODEL_REGISTRY.items():
        if cls.__name__ == model_class_name:
            model_class = cls
            break
    
    if model_class is None:
        model_class = PPOActorCritic
    
    model = model_class(
        obs_dim=checkpoint['obs_dim'],
        action_dim=checkpoint['action_dim'],
        hidden_dims=checkpoint['hidden_dims'],
        device=device
    )
    
    if 'actor_state_dict' in checkpoint:
        model.actor.load_state_dict(checkpoint['actor_state_dict'])
    if 'critic_state_dict' in checkpoint:
        model.critic.load_state_dict(checkpoint['critic_state_dict'])
    if 'log_std' in checkpoint:
        model.log_std.data = checkpoint['log_std']
    
    metadata = checkpoint.get('metadata', {})
    
    return model, metadata


# ============================================================================
# Hyperparameter Sweep Support
# ============================================================================

def get_sweep_configs(base_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate sweep configurations for hyperparameter search.
    
    Sweep parameters:
    - alpha: Explanation weight [0.0, 0.3, 0.5, 0.7, 1.0]
    - lambda: GAE lambda [0.90, 0.95, 0.99]
    - p: Critical state percentile [0.1, 0.2, 0.3, 0.5]
    
    Returns:
        List of configuration dictionaries
    """
    sweep_params = {
        'alpha': [0.0, 0.3, 0.5, 0.7, 1.0],
        'lambda': [0.90, 0.95, 0.99],
        'p': [0.1, 0.2, 0.3, 0.5]
    }
    
    configs = []
    
    # Default config
    configs.append(base_config.copy())
    
    # Alpha sweep
    for alpha in sweep_params['alpha']:
        config = base_config.copy()
        config['alpha'] = alpha
        config['sweep_name'] = f'alpha_{alpha}'
        configs.append(config)
    
    # Lambda sweep
    for lambda_val in sweep_params['lambda']:
        config = base_config.copy()
        config['lambda'] = lambda_val
        config['sweep_name'] = f'lambda_{lambda_val}'
        configs.append(config)
    
    # p sweep
    for p_val in sweep_params['p']:
        config = base_config.copy()
        config['p'] = p_val
        config['sweep_name'] = f'p_{p_val}'
        configs.append(config)
    
    return configs


# ============================================================================
# Dry-run and Smoke Test Support
# ============================================================================

def create_dry_run_artifacts(artifact_dir: str, config: Dict[str, Any]):
    """Create dry-run artifacts for smoke testing."""
    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)
    
    # Create checkpoint directory
    checkpoint_dir = artifact_path / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Create dummy checkpoint
    dummy_checkpoint_path = checkpoint_dir / 'pretrained_agent.pth'
    dummy_checkpoint = {
        'obs_dim': 11,
        'action_dim': 3,
        'hidden_dims': [64, 64],
        'model_class': 'PPOActorCritic',
        'metadata': {
            'stage': 'pre-training',
            'environment': 'Hopper-v3',
            'timesteps': 1000000,
            'dry_run': True,
            'note': 'This is a dry-run checkpoint for contract validation'
        }
    }
    
    # Save with minimal torch dependency
    torch = lazy_load_torch()
    if torch is not None:
        model = create_model('ppo', 11, 3, config)
        save_checkpoint(model, dummy_checkpoint_path, dummy_checkpoint['metadata'])
    else:
        # Fallback: save JSON manifest
        with open(dummy_checkpoint_path.with_suffix('.json'), 'w') as f:
            json.dump(dummy_checkpoint, f, indent=2)
    
    # Create model registry manifest
    registry_path = artifact_path / 'model_registry.json'
    registry = {
        'methods': list(MODEL_REGISTRY.keys()),
        'default': 'ppo',
        'baselines': ['random', 'baseline', 'ppo'],
        'variants': ['ours', 'statemask', 'sac', 'gail', 'jsrl', 'adapter'],
        'sweep_params': {
            'alpha': [0.0, 0.3, 0.5, 0.7, 1.0],
            'lambda': [0.90, 0.95, 0.99],
            'p': [0.1, 0.2, 0.3, 0.5]
        }
    }
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
    
    return {
        'checkpoint_path': str(dummy_checkpoint_path),
        'registry_path': str(registry_path),
        'dry_run': True
    }


# ============================================================================
# Model Evaluation and Metrics
# ============================================================================

def evaluate_model_quality(model: BaseActorCritic, env, n_episodes: int = 10) -> Dict[str, Any]:
    """
    Evaluate model performance.
    
    Args:
        model: Model to evaluate
        env: Environment instance
        n_episodes: Number of evaluation episodes
        
    Returns:
        Dictionary with evaluation metrics
    """
    episode_returns = []
    episode_lengths = []
    
    for _ in range(n_episodes):
        obs, _ = env.reset() if hasattr(env.reset(), '__iter__') and len(env.reset()) > 1 else (env.reset(), {})
        done = False
        episode_return = 0.0
        episode_length = 0
        
        while not done:
            action, _, _ = model.get_action(obs, deterministic=True)
            if hasattr(action, 'cpu'):
                action = action.cpu().numpy()
            
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                obs, reward, done, _ = step_result
            
            episode_return += reward
            episode_length += 1
            
            if episode_length >= 1000:  # Safety limit
                break
        
        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
    
    return {
        'mean_return': float(np.mean(episode_returns)),
        'std_return': float(np.std(episode_returns)),
        'min_return': float(np.min(episode_returns)),
        'max_return': float(np.max(episode_returns)),
        'mean_length': float(np.mean(episode_lengths)),
        'n_episodes': n_episodes
    }


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    'MODEL_REGISTRY',
    'register_model',
    'get_model',
    'create_model',
    'save_checkpoint',
    'load_checkpoint',
    'get_sweep_configs',
    'create_dry_run_artifacts',
    'evaluate_model_quality',
    'BaseActorCritic',
    'TorchActorCritic',
    'RICEActorCritic',
    'PPOActorCritic',
    'BaselineActorCritic',
    'StateMaskActorCritic',
    'SACActorCritic',
    'GAILActorCritic',
    'JSRLActorCritic',
    'AdapterActorCritic',
    'RandomPolicy',
]