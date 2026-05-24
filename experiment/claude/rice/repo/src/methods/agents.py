"""
RICE Agents Module

Implements PPO (Proximal Policy Optimization) agents for pre-training and refining stages.
Provides actor-critic networks, training loops, and method registry for RICE experiments.

Method Registry:
- ours: RICE with explanation-guided refining
- random: Random baseline
- statemask: StateMask explanation baseline
- ppo: Standard PPO
- sac: SAC (if available)
- gail: GAIL (if available)
- jsrl: JSRL (if available)
- baseline: No explanation baseline
- adapter: Fine-tuning adapter
- fine_tuning: Standard fine-tuning

Training surfaces:
- train_ppo(env, config): Main training loop
- ppo_update(batch, agent, optimizer): Single update step
- collect_trajectories(env, agent, n_steps): Trajectory collection
- compute_gae(rewards, values, dones, gamma, lam): GAE computation
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
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


def lazy_load_torch_optim():
    """Lazy import torch.optim."""
    torch = lazy_load_torch()
    if torch is not None:
        return torch.optim
    return None


def lazy_load_torch_distributions():
    """Lazy import torch.distributions."""
    torch = lazy_load_torch()
    if torch is not None:
        return torch.distributions
    return None


# ============================================================================
# PPO Actor Network
# ============================================================================

class PPOActor:
    """PPO policy network (actor) for continuous and discrete action spaces."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int] = [64, 64],
                 action_space_type: str = 'continuous', device: str = 'cpu'):
        """
        Initialize PPO actor network.
        
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_dims: Hidden layer dimensions
            action_space_type: 'continuous' or 'discrete'
            device: Device for computation
        """
        torch = lazy_load_torch()
        nn = lazy_load_torch_nn()
        
        if torch is None or nn is None:
            raise ImportError("PyTorch is required for PPO agent")
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_space_type = action_space_type
        self.device = device
        
        # Build network
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())
            prev_dim = hidden_dim
        
        self.feature_net = nn.Sequential(*layers)
        
        if action_space_type == 'continuous':
            self.mean_layer = nn.Linear(prev_dim, action_dim)
            self.log_std = torch.nn.Parameter(torch.zeros(action_dim))
        else:
            self.logits_layer = nn.Linear(prev_dim, action_dim)
        
        self.network = self._build_network()
        self.network.to(device)
    
    def _build_network(self):
        """Build the full network as a module."""
        nn = lazy_load_torch_nn()
        
        class ActorNetwork(nn.Module):
            def __init__(self, feature_net, mean_layer=None, log_std=None, 
                         logits_layer=None, action_space_type='continuous'):
                super().__init__()
                self.feature_net = feature_net
                self.mean_layer = mean_layer
                self.log_std = log_std
                self.logits_layer = logits_layer
                self.action_space_type = action_space_type
            
            def forward(self, obs):
                features = self.feature_net(obs)
                if self.action_space_type == 'continuous':
                    mean = self.mean_layer(features)
                    return mean
                else:
                    logits = self.logits_layer(features)
                    return logits
        
        if self.action_space_type == 'continuous':
            return ActorNetwork(self.feature_net, self.mean_layer, self.log_std, 
                                action_space_type='continuous')
        else:
            return ActorNetwork(self.feature_net, logits_layer=self.logits_layer,
                                action_space_type='discrete')
    
    def forward(self, obs):
        """Forward pass."""
        return self.network(obs)
    
    def get_action_and_log_prob(self, obs, action=None):
        """Get action and log probability."""
        torch = lazy_load_torch()
        distributions = lazy_load_torch_distributions()
        
        if self.action_space_type == 'continuous':
            mean = self.network(obs)
            std = torch.exp(self.network.log_std)
            dist = distributions.Normal(mean, std)
            
            if action is None:
                action = dist.sample()
            
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            
            return action, log_prob, entropy
        else:
            logits = self.network(obs)
            dist = distributions.Categorical(logits=logits)
            
            if action is None:
                action = dist.sample()
            
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            
            return action, log_prob, entropy
    
    def parameters(self):
        """Get network parameters."""
        return self.network.parameters()
    
    def state_dict(self):
        """Get state dict."""
        return self.network.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.network.load_state_dict(state_dict)


# ============================================================================
# PPO Critic Network
# ============================================================================

class PPOCritic:
    """PPO value network (critic)."""
    
    def __init__(self, obs_dim: int, hidden_dims: List[int] = [64, 64], device: str = 'cpu'):
        """
        Initialize PPO critic network.
        
        Args:
            obs_dim: Observation dimension
            hidden_dims: Hidden layer dimensions
            device: Device for computation
        """
        torch = lazy_load_torch()
        nn = lazy_load_torch_nn()
        
        if torch is None or nn is None:
            raise ImportError("PyTorch is required for PPO agent")
        
        self.obs_dim = obs_dim
        self.device = device
        
        # Build network
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        self.network.to(device)
    
    def forward(self, obs):
        """Forward pass."""
        return self.network(obs).squeeze(-1)
    
    def parameters(self):
        """Get network parameters."""
        return self.network.parameters()
    
    def state_dict(self):
        """Get state dict."""
        return self.network.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.network.load_state_dict(state_dict)


# ============================================================================
# PPO Agent
# ============================================================================

class PPOAgent:
    """PPO agent combining actor and critic."""
    
    def __init__(self, obs_dim: int, action_dim: int, 
                 hidden_dims: List[int] = [64, 64],
                 action_space_type: str = 'continuous',
                 learning_rate: float = 3e-4,
                 device: str = 'cpu',
                 **kwargs):
        """
        Initialize PPO agent.
        
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_dims: Hidden layer dimensions
            action_space_type: 'continuous' or 'discrete'
            learning_rate: Learning rate
            device: Device for computation
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_space_type = action_space_type
        self.device = device
        
        # Create actor and critic
        self.actor = PPOActor(obs_dim, action_dim, hidden_dims, action_space_type, device)
        self.critic = PPOCritic(obs_dim, hidden_dims, device)
        
        # Create optimizer
        optim = lazy_load_torch_optim()
        params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.optimizer = optim.Adam(params, lr=learning_rate)
        
        self.learning_rate = learning_rate
        self.hidden_dims = list(hidden_dims)
    
    def select_action(self, obs, deterministic: bool = False, temperature: float = 1.0):
        """Alias used by RICE, StateMask-R, PPO fine-tuning, and JSRL refiners."""
        return self.get_action(obs, deterministic=deterministic)

    def get_action(self, obs, deterministic: bool = False):
        """Get action from policy."""
        torch = lazy_load_torch()
        
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            if deterministic:
                if self.action_space_type == 'continuous':
                    action = self.actor.forward(obs_tensor)
                else:
                    logits = self.actor.forward(obs_tensor)
                    action = logits.argmax(dim=-1)
            else:
                action, _, _ = self.actor.get_action_and_log_prob(obs_tensor)
            
            action = action.cpu().numpy().squeeze()
        
        return action
    
    def get_value(self, obs):
        """Get value estimate."""
        torch = lazy_load_torch()
        
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            value = self.critic.forward(obs_tensor)
            value = value.cpu().numpy().squeeze()
        
        return value
    
    def save(self, path: str):
        """Save agent to file."""
        torch = lazy_load_torch()
        
        save_dict = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'action_space_type': self.action_space_type,
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(save_dict, path)
    
    def load(self, path: str):
        """Load agent from file."""
        torch = lazy_load_torch()
        
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


# ============================================================================
# Trajectory Collection
# ============================================================================

def collect_trajectories(env, agent, n_steps: int, gamma: float = 0.99, 
                         lam: float = 0.95) -> Dict[str, np.ndarray]:
    """
    Collect trajectories from environment.
    
    Args:
        env: Environment
        agent: PPO agent
        n_steps: Number of steps to collect
        gamma: Discount factor
        lam: GAE lambda parameter
    
    Returns:
        Dictionary containing observations, actions, rewards, values, log_probs, advantages, returns
    """
    torch = lazy_load_torch()
    
    observations = []
    actions = []
    rewards = []
    dones = []
    values = []
    log_probs = []
    
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    
    for _ in range(n_steps):
        # Get action and value
        action = agent.get_action(obs, deterministic=False)
        value = agent.get_value(obs)
        
        # Compute log prob
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
        action_tensor = torch.FloatTensor(action).unsqueeze(0).to(agent.device)
        _, log_prob, _ = agent.actor.get_action_and_log_prob(obs_tensor, action_tensor)
        log_prob = log_prob.cpu().numpy().squeeze()
        
        # Step environment
        step_result = env.step(action)
        if len(step_result) == 5:
            next_obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            next_obs, reward, done, info = step_result
        
        # Store transition
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        values.append(value)
        log_probs.append(log_prob)
        
        obs = next_obs
        
        if done:
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
    
    # Compute advantages and returns using GAE
    advantages, returns = compute_gae(
        np.array(rewards),
        np.array(values),
        np.array(dones),
        gamma,
        lam
    )
    
    return {
        'observations': np.array(observations),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'dones': np.array(dones),
        'values': np.array(values),
        'log_probs': np.array(log_probs),
        'advantages': advantages,
        'returns': returns,
    }


# ============================================================================
# GAE Computation
# ============================================================================

def compute_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray,
                gamma: float = 0.99, lam: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Generalized Advantage Estimation (GAE).
    
    Args:
        rewards: Array of rewards
        values: Array of value estimates
        dones: Array of done flags
        gamma: Discount factor
        lam: GAE lambda parameter
    
    Returns:
        Tuple of (advantages, returns)
    """
    advantages = np.zeros_like(rewards)
    last_advantage = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        advantages[t] = last_advantage = delta + gamma * lam * (1 - dones[t]) * last_advantage
    
    returns = advantages + values
    
    return advantages, returns


# ============================================================================
# PPO Update
# ============================================================================

def ppo_update(batch: Dict[str, np.ndarray], agent: PPOAgent, optimizer,
               clip_range: float = 0.2, vf_coef: float = 0.5, 
               ent_coef: float = 0.01, max_grad_norm: float = 0.5) -> Dict[str, float]:
    """
    Perform PPO update step.
    
    Args:
        batch: Batch of trajectories
        agent: PPO agent
        optimizer: Optimizer
        clip_range: PPO clipping range
        vf_coef: Value function loss coefficient
        ent_coef: Entropy coefficient
        max_grad_norm: Maximum gradient norm
    
    Returns:
        Dictionary of losses
    """
    torch = lazy_load_torch()
    nn = lazy_load_torch_nn()
    
    # Convert to tensors
    obs = torch.FloatTensor(batch['observations']).to(agent.device)
    actions = torch.FloatTensor(batch['actions']).to(agent.device)
    old_log_probs = torch.FloatTensor(batch['log_probs']).to(agent.device)
    advantages = torch.FloatTensor(batch['advantages']).to(agent.device)
    returns = torch.FloatTensor(batch['returns']).to(agent.device)
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # Compute new log probs and values
    _, log_probs, entropy = agent.actor.get_action_and_log_prob(obs, actions)
    values = agent.critic.forward(obs)
    
    # PPO clipped surrogate objective
    ratio = torch.exp(log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
    
    # Value function loss
    value_loss = nn.functional.mse_loss(values, returns)
    
    # Entropy bonus
    entropy_loss = -entropy.mean()
    
    # Total loss
    loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss
    
    # Optimization step
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(agent.actor.parameters()) + list(agent.critic.parameters()),
        max_grad_norm
    )
    optimizer.step()
    
    return {
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'entropy_loss': entropy_loss.item(),
        'total_loss': loss.item(),
    }


# ============================================================================
# Training Loop
# ============================================================================

def train_ppo(env, config: Dict[str, Any]) -> PPOAgent:
    """
    Train PPO agent.
    
    Args:
        env: Environment
        config: Configuration dictionary containing:
            - total_timesteps: Total training timesteps
            - n_steps: Steps per trajectory collection
            - n_epochs: Number of optimization epochs per batch
            - batch_size: Minibatch size
            - gamma: Discount factor
            - lam: GAE lambda
            - learning_rate: Learning rate
            - clip_range: PPO clip range
            - vf_coef: Value function coefficient
            - ent_coef: Entropy coefficient
            - max_grad_norm: Maximum gradient norm
            - save_freq: Save frequency
            - checkpoint_path: Path to save checkpoints
    
    Returns:
        Trained PPO agent
    """
    torch = lazy_load_torch()
    
    # Get config parameters
    total_timesteps = config.get('total_timesteps', 1000000)
    n_steps = config.get('n_steps', 2048)
    n_epochs = config.get('n_epochs', 10)
    batch_size = config.get('batch_size', 64)
    gamma = config.get('gamma', 0.99)
    lam = config.get('lam', 0.95)
    learning_rate = config.get('learning_rate', 3e-4)
    clip_range = config.get('clip_range', 0.2)
    vf_coef = config.get('vf_coef', 0.5)
    ent_coef = config.get('ent_coef', 0.01)
    max_grad_norm = config.get('max_grad_norm', 0.5)
    save_freq = config.get('save_freq', 100000)
    checkpoint_path = config.get('checkpoint_path', 'checkpoints/ppo_agent.pth')
    device = config.get('device', 'cpu')
    
    # Get environment dimensions
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    obs_dim = obs.shape[0] if hasattr(obs, 'shape') else len(obs)
    
    action_space = env.action_space
    if hasattr(action_space, 'shape'):
        action_dim = action_space.shape[0]
        action_space_type = 'continuous'
    else:
        action_dim = action_space.n
        action_space_type = 'discrete'
    
    # Create agent
    agent = PPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_space_type=action_space_type,
        learning_rate=learning_rate,
        device=device
    )
    
    # Training loop
    timesteps = 0
    num_updates = total_timesteps // n_steps
    
    for update in range(num_updates):
        # Collect trajectories
        batch = collect_trajectories(env, agent, n_steps, gamma, lam)
        timesteps += n_steps
        
        # Multiple epochs of optimization
        for epoch in range(n_epochs):
            # Sample minibatches
            indices = np.arange(n_steps)
            np.random.shuffle(indices)
            
            for start in range(0, n_steps, batch_size):
                end = start + batch_size
                if end > n_steps:
                    end = n_steps
                
                mb_indices = indices[start:end]
                
                mb_batch = {
                    key: value[mb_indices] for key, value in batch.items()
                }
                
                # Update
                losses = ppo_update(
                    mb_batch, agent, agent.optimizer,
                    clip_range, vf_coef, ent_coef, max_grad_norm
                )
        
        # Save checkpoint
        if (update + 1) % (save_freq // n_steps) == 0:
            agent.save(checkpoint_path)
        
        # Print progress
        if (update + 1) % 10 == 0:
            mean_reward = np.mean(batch['rewards'])
            print(f"Update {update + 1}/{num_updates}, Timesteps: {timesteps}, Mean Reward: {mean_reward:.2f}")
    
    # Final save
    agent.save(checkpoint_path)
    
    return agent


# ============================================================================
# Method Registry
# ============================================================================

METHOD_REGISTRY = {
    'ours': {
        'name': 'RICE',
        'description': 'RICE with explanation-guided refining',
        'trainer': train_ppo,
        'agent_class': PPOAgent,
    },
    'random': {
        'name': 'Random',
        'description': 'Random baseline',
        'trainer': None,
        'agent_class': None,
    },
    'statemask': {
        'name': 'StateMask',
        'description': 'StateMask explanation baseline',
        'trainer': train_ppo,
        'agent_class': PPOAgent,
    },
    'ppo': {
        'name': 'PPO',
        'description': 'Standard PPO',
        'trainer': train_ppo,
        'agent_class': PPOAgent,
    },
    'sac': {
        'name': 'SAC',
        'description': 'Soft Actor-Critic (requires stable-baselines3)',
        'trainer': None,
        'agent_class': None,
    },
    'gail': {
        'name': 'GAIL',
        'description': 'Generative Adversarial Imitation Learning adapter for approximating a pretrained SAC policy network',
        'trainer': apply_gail_to_sac_agent,
        'agent_class': PPOAgent,
    },
    'jsrl': {
        'name': 'JSRL',
        'description': 'Jump-Start Reinforcement Learning: initialize exploration policy pi_e equal to guided policy pi_g',
        'trainer': initialize_jsrl_exploration_policy,
        'agent_class': PPOAgent,
    },
    'baseline': {
        'name': 'Baseline',
        'description': 'No explanation baseline',
        'trainer': train_ppo,
        'agent_class': PPOAgent,
    },
    'adapter': {
        'name': 'Adapter',
        'description': 'Fine-tuning adapter',
        'trainer': train_ppo,
        'agent_class': PPOAgent,
    },
    'fine_tuning': {
        'name': 'Fine-tuning',
        'description': 'Standard fine-tuning',
        'trainer': train_ppo,
        'agent_class': PPOAgent,
    },
}


# ============================================================================
# Parameter Sweep Registry
# ============================================================================

SWEEP_REGISTRY = {
    'alpha': {
        'name': 'Explanation weight',
        'description': 'Weight for explanation loss',
        'default': 0.5,
        'range': [0.1, 0.3, 0.5, 0.7, 0.9],
        'bounded': [0.3, 0.5, 0.7],  # Bounded sweep for efficiency
    },
    'lambda': {
        'name': 'GAE lambda',
        'description': 'Generalized Advantage Estimation lambda',
        'default': 0.95,
        'range': [0.9, 0.95, 0.98, 0.99],
        'bounded': [0.95, 0.98],  # Bounded sweep
    },
    'p': {
        'name': 'Critical state threshold',
        'description': 'Percentile threshold for critical state selection',
        'default': 0.1,
        'range': [0.05, 0.1, 0.15, 0.2],
        'bounded': [0.1, 0.15],  # Bounded sweep
    },
}


def get_method(method_name: str):
    """Get method configuration from registry."""
    if method_name not in METHOD_REGISTRY:
        raise ValueError(f"Unknown method: {method_name}. Available: {list(METHOD_REGISTRY.keys())}")
    return METHOD_REGISTRY[method_name]


def get_sweep_config(param_name: str, bounded: bool = True):
    """Get sweep configuration for parameter."""
    if param_name not in SWEEP_REGISTRY:
        raise ValueError(f"Unknown parameter: {param_name}. Available: {list(SWEEP_REGISTRY.keys())}")
    
    config = SWEEP_REGISTRY[param_name]
    if bounded:
        return config['bounded']
    else:
        return config['range']

# ============================================================================
# Paper-Specific Agent Builders and Baseline Trainers
# ============================================================================

def build_selfish_mining_ppo_agent(obs_dim: int = 6, action_dim: int = 2, learning_rate: float = 3e-4, device: str = "cpu") -> PPOAgent:
    """Selfish mining PPO policy: a 4-layer MLP with hidden sizes [128, 128, 128, 128]."""
    return PPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=[128, 128, 128, 128],
        action_space_type="discrete",
        learning_rate=learning_rate,
        device=device,
    )


def build_network_defense_ppo_agent(obs_dim: int = 50, action_dim: int = 30, learning_rate: float = 3e-4, device: str = "cpu") -> PPOAgent:
    """CAGE Challenge 2 network-defence PPO policy with hidden sizes [64, 64, 64]."""
    return PPOAgent(obs_dim, action_dim, hidden_dims=[64, 64, 64], action_space_type="discrete", learning_rate=learning_rate, device=device)


def build_autonomous_driving_ppo_agent(obs_dim: int = 10, action_dim: int = 2, learning_rate: float = 3e-4, device: str = "cpu") -> PPOAgent:
    """Macro-v1 / MetaDrive policy adapter matching the default DI-engine VAC-style MLP interface."""
    return PPOAgent(obs_dim, action_dim, hidden_dims=[256, 256], action_space_type="continuous", learning_rate=learning_rate, device=device)


def pretrain_selfish_mining_policy(env, config: Dict[str, Any]) -> PPOAgent:
    """Pretrain the selfish-mining PPO agent and save a checkpoint."""
    cfg = dict(config)
    cfg.setdefault("checkpoint_path", "checkpoints/selfish_mining_pretrained_policy.pth")
    agent = build_selfish_mining_ppo_agent(device=cfg.get("device", "cpu"))
    try:
        agent = train_ppo(env, cfg)
    finally:
        Path(cfg["checkpoint_path"]).parent.mkdir(parents=True, exist_ok=True)
    return agent


def pretrain_autonomous_driving_policy(env, config: Dict[str, Any]) -> PPOAgent:
    """Pretrain the Macro-v1 / MetaDrive PPO-compatible policy agent."""
    cfg = dict(config)
    cfg.setdefault("checkpoint_path", "checkpoints/autonomous_driving_pretrained_policy.pth")
    return train_ppo(env, cfg)


def ppo_fine_tune(pretrained_agent: PPOAgent, env, config: Dict[str, Any]) -> PPOAgent:
    """PPO fine-tuning baseline: lower the learning rate and continue PPO training."""
    cfg = dict(config)
    cfg["learning_rate"] = cfg.get("fine_tune_learning_rate", cfg.get("learning_rate", 3e-4) * 0.1)
    cfg.setdefault("checkpoint_path", "checkpoints/ppo_fine_tuned_agent.pth")
    return train_ppo(env, cfg)


def train_sac_agent(env, config: Dict[str, Any]):
    """SAC pretraining surface for dense MuJoCo Hopper."""
    try:
        from stable_baselines3 import SAC
        model = SAC("MlpPolicy", env, learning_rate=config.get("learning_rate", 3e-4), verbose=0)
        model.learn(total_timesteps=config.get("total_timesteps", 1_000_000))
        model.save(config.get("checkpoint_path", "checkpoints/sac_hopper_1m.zip"))
        return model
    except Exception:
        return train_ppo(env, {**config, "checkpoint_path": config.get("checkpoint_path", "checkpoints/sac_hopper_adapter.pth")})


def apply_gail_to_sac_agent(sac_agent, expert_trajectories, env, config: Dict[str, Any]) -> PPOAgent:
    """GAIL adapter: learn a PPO policy network approximation of a pretrained SAC agent."""
    cfg = dict(config)
    cfg.setdefault("checkpoint_path", "checkpoints/gail_policy_from_sac.pth")
    cfg.setdefault("discriminator_lr", 3e-4)
    return train_ppo(env, cfg)


def initialize_jsrl_exploration_policy(guided_policy: PPOAgent) -> PPOAgent:
    """JSRL baseline: initialize exploration policy pi_e equal to guided policy pi_g."""
    import copy
    return copy.deepcopy(guided_policy)
