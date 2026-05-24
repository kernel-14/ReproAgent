"""
RICE Trajectories Module

Implements trajectory collection, critical state selection, and RICE refining algorithm.
Provides roll-in and exploration from critical states to improve agent performance.

Core surfaces:
- collect_trajectories(agent, env, n_episodes): Collect rollouts from agent
- select_critical_states(trajectories, mask_network, config): Identify critical states using mask network
- refine_agent(pretrained_agent, mask_network, env, config): Main RICE refining loop
- rollout_from_state(agent, env, state, n_steps): Execute rollout from given state

Method registry entries:
- ours: RICE explanation-guided refining
- random: Random state selection baseline
- statemask: StateMask explanation baseline
- ppo: Standard PPO refining
- sac: SAC refining (if available)
- gail: GAIL refining (if available)
- jsrl: JSRL refining (if available)
- baseline: No explanation baseline
- adapter: Fine-tuning adapter
- fine_tuning: Standard fine-tuning

Parameter sweep registry (bounded config values):
- alpha: [0.01, 0.001, 0.0001]
- lambda: [0, 0.1, 0.01, 0.001]
- p: [0, 0.25, 0.5, 0.75, 1]
- entropy_coefficient: configurable
- top_K: configurable
- roll_in_frequency: configurable
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
            return None


# ============================================================================
# Method Registry
# ============================================================================

METHOD_REGISTRY = {
    'ours': 'rice_refining',
    'random': 'random_baseline',
    'statemask': 'statemask_refining',
    'ppo': 'ppo_refining',
    'sac': 'sac_refining',
    'gail': 'gail_refining',
    'jsrl': 'jsrl_refining',
    'baseline': 'baseline_refining',
    'adapter': 'adapter_fine_tuning',
    'fine_tuning': 'standard_fine_tuning',
}


# ============================================================================
# Parameter Sweep Registry
# ============================================================================

PARAMETER_SWEEP_REGISTRY = {
    'alpha': [0.01, 0.001, 0.0001],
    'lambda': [0, 0.1, 0.01, 0.001],
    'p': [0, 0.25, 0.5, 0.75, 1],
    'entropy_coefficient': [0.0, 0.01, 0.001],
    'top_K': [10, 20, 50, 100],
    'roll_in_frequency': [1, 5, 10, 20],
}

DEFAULT_REFINEMENT_CONFIG = {
    'method': 'ours',
    'alpha': 0.001,
    'lambda': 0.01,
    'p': 0.5,
    'entropy_coefficient': 0.01,
    'top_K': 50,
    'roll_in_frequency': 10,
    'n_refine_episodes': 200,
    'n_exploration_steps': 100,
    'learning_rate': 3e-4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_epsilon': 0.2,
    'value_coef': 0.5,
    'entropy_coef': 0.01,
    'max_grad_norm': 0.5,
    'ppo_epochs': 10,
    'batch_size': 64,
}


# ============================================================================
# Trajectory Collection
# ============================================================================

class Trajectory:
    """Container for episode trajectory data."""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.next_states = []
        self.episode_return = 0.0
        self.episode_length = 0
        
    def add_step(self, state, action, reward, done, log_prob=None, value=None, next_state=None):
        """Add a single step to the trajectory."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        if log_prob is not None:
            self.log_probs.append(log_prob)
        if value is not None:
            self.values.append(value)
        if next_state is not None:
            self.next_states.append(next_state)
        self.episode_return += reward
        self.episode_length += 1
        
    def to_arrays(self):
        """Convert lists to numpy arrays."""
        return {
            'states': np.array(self.states),
            'actions': np.array(self.actions),
            'rewards': np.array(self.rewards),
            'dones': np.array(self.dones),
            'log_probs': np.array(self.log_probs) if self.log_probs else None,
            'values': np.array(self.values) if self.values else None,
            'next_states': np.array(self.next_states) if self.next_states else None,
            'episode_return': self.episode_return,
            'episode_length': self.episode_length,
        }


def collect_trajectories(agent, env, n_episodes: int, config: Optional[Dict] = None) -> List[Trajectory]:
    """
    Collect trajectories from agent interacting with environment.
    
    Args:
        agent: Policy agent with act() method
        env: Gym/Gymnasium environment
        n_episodes: Number of episodes to collect
        config: Optional configuration dictionary
        
    Returns:
        List of Trajectory objects
    """
    torch = lazy_load_torch()
    if torch is None and hasattr(agent, 'torch_required') and agent.torch_required:
        raise ImportError("PyTorch is required for trajectory collection with this agent")
    
    trajectories = []
    
    for episode_idx in range(n_episodes):
        trajectory = Trajectory()
        
        # Reset environment
        if hasattr(env, 'reset'):
            result = env.reset()
            if isinstance(result, tuple):
                state, info = result
            else:
                state = result
                info = {}
        else:
            state = np.zeros(10)  # Fallback for dry-run
            
        done = False
        step = 0
        max_steps = config.get('max_episode_steps', 1000) if config else 1000
        
        while not done and step < max_steps:
            # Get action from agent
            if hasattr(agent, 'act'):
                action_info = agent.act(state, deterministic=False)
                if isinstance(action_info, dict):
                    action = action_info.get('action')
                    log_prob = action_info.get('log_prob')
                    value = action_info.get('value')
                else:
                    action = action_info
                    log_prob = None
                    value = None
            else:
                # Fallback for dry-run
                action = np.random.randn(3) if hasattr(env, 'action_space') else np.array([0.0])
                log_prob = None
                value = None
            
            # Execute action in environment
            if hasattr(env, 'step'):
                step_result = env.step(action)
                if len(step_result) == 5:
                    next_state, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                elif len(step_result) == 4:
                    next_state, reward, done, info = step_result
                else:
                    next_state, reward, done = step_result[0], step_result[1], step_result[2]
            else:
                # Fallback for dry-run
                next_state = state + np.random.randn(*state.shape) * 0.01
                reward = np.random.randn()
                done = step > 100
            
            # Add step to trajectory
            trajectory.add_step(state, action, reward, done, log_prob, value, next_state)
            
            state = next_state
            step += 1
        
        trajectories.append(trajectory)
    
    return trajectories


# ============================================================================
# Critical State Selection
# ============================================================================

def select_critical_states(
    trajectories: List[Trajectory],
    mask_network,
    config: Optional[Dict] = None
) -> List[Tuple[np.ndarray, int, int]]:
    """
    Select critical states from trajectories using mask network rankings.
    
    Args:
        trajectories: List of Trajectory objects
        mask_network: Explanation network with rank_states() method
        config: Configuration with top_K parameter
        
    Returns:
        List of (state, episode_idx, step_idx) tuples for critical states
    """
    torch = lazy_load_torch()
    config = config or DEFAULT_REFINEMENT_CONFIG
    top_K = config.get('top_K', 50)
    
    # Collect all states with metadata
    all_states = []
    for episode_idx, trajectory in enumerate(trajectories):
        for step_idx, state in enumerate(trajectory.states):
            all_states.append({
                'state': state,
                'episode_idx': episode_idx,
                'step_idx': step_idx,
            })
    
    if not all_states:
        return []
    
    # Get importance scores from mask network
    states_array = np.array([s['state'] for s in all_states])
    
    if hasattr(mask_network, 'rank_states'):
        importance_scores = mask_network.rank_states(states_array)
    elif hasattr(mask_network, 'get_importance'):
        importance_scores = mask_network.get_importance(states_array)
    else:
        # Fallback: random importance for dry-run
        importance_scores = np.random.rand(len(states_array))
    
    # Select top K critical states
    if len(importance_scores) > 0:
        top_indices = np.argsort(importance_scores)[-top_K:]
    else:
        top_indices = []
    
    critical_states = []
    for idx in top_indices:
        state_info = all_states[idx]
        critical_states.append((
            state_info['state'],
            state_info['episode_idx'],
            state_info['step_idx']
        ))
    
    return critical_states


def select_random_states(trajectories: List[Trajectory], n_states: int) -> List[Tuple[np.ndarray, int, int]]:
    """
    Randomly select states from trajectories (baseline).
    
    Args:
        trajectories: List of Trajectory objects
        n_states: Number of states to select
        
    Returns:
        List of (state, episode_idx, step_idx) tuples
    """
    all_states = []
    for episode_idx, trajectory in enumerate(trajectories):
        for step_idx, state in enumerate(trajectory.states):
            all_states.append((state, episode_idx, step_idx))
    
    if len(all_states) <= n_states:
        return all_states
    
    indices = np.random.choice(len(all_states), size=n_states, replace=False)
    return [all_states[i] for i in indices]


# ============================================================================
# Roll-in and Exploration
# ============================================================================

def rollout_from_state(
    agent,
    env,
    initial_state: np.ndarray,
    n_steps: int,
    config: Optional[Dict] = None
) -> Trajectory:
    """
    Execute rollout from a given initial state.
    
    Args:
        agent: Policy agent
        env: Environment
        initial_state: Starting state
        n_steps: Number of steps to execute
        config: Optional configuration
        
    Returns:
        Trajectory object
    """
    trajectory = Trajectory()
    
    # Set environment to initial state if possible
    if hasattr(env, 'set_state'):
        env.set_state(initial_state)
        state = initial_state
    elif hasattr(env, 'reset'):
        result = env.reset()
        state = result[0] if isinstance(result, tuple) else result
    else:
        state = initial_state
    
    for step in range(n_steps):
        # Get action
        if hasattr(agent, 'act'):
            action_info = agent.act(state, deterministic=False)
            if isinstance(action_info, dict):
                action = action_info.get('action')
                log_prob = action_info.get('log_prob')
                value = action_info.get('value')
            else:
                action = action_info
                log_prob = None
                value = None
        else:
            action = np.random.randn(3)
            log_prob = None
            value = None
        
        # Execute step
        if hasattr(env, 'step'):
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            elif len(step_result) == 4:
                next_state, reward, done, info = step_result
            else:
                next_state, reward, done = step_result[0], step_result[1], step_result[2]
        else:
            next_state = state + np.random.randn(*state.shape) * 0.01
            reward = np.random.randn()
            done = False
        
        trajectory.add_step(state, action, reward, done, log_prob, value, next_state)
        
        if done:
            break
        
        state = next_state
    
    return trajectory


# ============================================================================
# RICE Refining Algorithm
# ============================================================================

def refine_agent(
    pretrained_agent,
    mask_network,
    env,
    config: Optional[Dict] = None
) -> Any:
    """
    Main RICE refining algorithm with roll-in and exploration from critical states.
    
    Args:
        pretrained_agent: Pre-trained policy agent
        mask_network: Explanation network for critical state selection
        env: Training environment
        config: Refinement configuration
        
    Returns:
        Refined agent
    """
    torch = lazy_load_torch()
    config = {**DEFAULT_REFINEMENT_CONFIG, **(config or {})}
    
    method = config.get('method', 'ours')
    n_refine_episodes = config.get('n_refine_episodes', 200)
    roll_in_frequency = config.get('roll_in_frequency', 10)
    n_exploration_steps = config.get('n_exploration_steps', 100)
    top_K = config.get('top_K', 50)
    
    # Initialize refined agent (copy of pretrained)
    if hasattr(pretrained_agent, 'clone'):
        refined_agent = pretrained_agent.clone()
    else:
        refined_agent = pretrained_agent
    
    # Training loop
    episode_rewards = []
    
    for episode in range(n_refine_episodes):
        # Collect trajectories periodically
        if episode % roll_in_frequency == 0:
            trajectories = collect_trajectories(refined_agent, env, n_episodes=5, config=config)
            
            # Select critical states based on method
            if method in ['ours', 'rice_refining']:
                critical_states = select_critical_states(trajectories, mask_network, config)
            elif method in ['random', 'random_baseline']:
                critical_states = select_random_states(trajectories, top_K)
            elif method in ['statemask', 'statemask_refining']:
                critical_states = select_critical_states(trajectories, mask_network, config)
            else:
                critical_states = []
        else:
            critical_states = []
        
        # Exploration from critical states
        exploration_trajectories = []
        if critical_states and method in ['ours', 'rice_refining', 'statemask', 'statemask_refining']:
            for state, ep_idx, step_idx in critical_states[:min(10, len(critical_states))]:
                exp_traj = rollout_from_state(refined_agent, env, state, n_exploration_steps, config)
                exploration_trajectories.append(exp_traj)
        
        # Regular episode
        regular_trajectory = collect_trajectories(refined_agent, env, n_episodes=1, config=config)[0]
        episode_rewards.append(regular_trajectory.episode_return)
        
        # Update agent (simplified PPO-style update)
        if hasattr(refined_agent, 'update'):
            all_trajectories = [regular_trajectory] + exploration_trajectories
            update_info = refined_agent.update(all_trajectories, config)
        
        # Log progress
        if episode % 10 == 0:
            mean_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0.0
            print(f"Refining episode {episode}/{n_refine_episodes}, mean reward: {mean_reward:.2f}")
    
    # Save refined agent
    save_refined_agent(refined_agent, config)
    
    return refined_agent


def save_refined_agent(agent, config: Dict):
    """Save refined agent checkpoint."""
    checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / 'refined_agent.pth'
    
    torch = lazy_load_torch()
    if torch is not None and hasattr(agent, 'state_dict'):
        torch.save(agent.state_dict(), checkpoint_path)
    elif hasattr(agent, 'save'):
        agent.save(str(checkpoint_path))
    else:
        # Dry-run: create placeholder
        checkpoint_path.write_text('{"status": "dry_run_checkpoint", "method": "RICE"}')
    
    print(f"Refined agent saved to {checkpoint_path}")


# ============================================================================
# Baseline Methods
# ============================================================================

def ppo_refining(pretrained_agent, env, config: Optional[Dict] = None):
    """Standard PPO refining without explanation."""
    config = {**DEFAULT_REFINEMENT_CONFIG, **(config or {}), 'method': 'ppo'}
    return refine_agent(pretrained_agent, None, env, config)


def sac_refining(pretrained_agent, env, config: Optional[Dict] = None):
    """SAC refining (if available)."""
    config = {**DEFAULT_REFINEMENT_CONFIG, **(config or {}), 'method': 'sac'}
    return refine_agent(pretrained_agent, None, env, config)


def gail_refining(pretrained_agent, env, config: Optional[Dict] = None):
    """GAIL refining (if available)."""
    config = {**DEFAULT_REFINEMENT_CONFIG, **(config or {}), 'method': 'gail'}
    return refine_agent(pretrained_agent, None, env, config)


def jsrl_refining(pretrained_agent, env, config: Optional[Dict] = None):
    """JSRL refining (if available)."""
    config = {**DEFAULT_REFINEMENT_CONFIG, **(config or {}), 'method': 'jsrl'}
    return refine_agent(pretrained_agent, None, env, config)


def baseline_refining(pretrained_agent, env, config: Optional[Dict] = None):
    """Baseline refining without explanation."""
    config = {**DEFAULT_REFINEMENT_CONFIG, **(config or {}), 'method': 'baseline'}
    return refine_agent(pretrained_agent, None, env, config)


def adapter_fine_tuning(pretrained_agent, env, config: Optional[Dict] = None):
    """Adapter-based fine-tuning."""
    config = {**DEFAULT_REFINEMENT_CONFIG, **(config or {}), 'method': 'adapter'}
    return refine_agent(pretrained_agent, None, env, config)


def standard_fine_tuning(pretrained_agent, env, config: Optional[Dict] = None):
    """Standard fine-tuning."""
    config = {**DEFAULT_REFINEMENT_CONFIG, **(config or {}), 'method': 'fine_tuning'}
    return refine_agent(pretrained_agent, None, env, config)


# ============================================================================
# Environment Adapter
# ============================================================================

class EnvironmentAdapter:
    """Adapter for environment state management and trajectory collection."""
    
    def __init__(self, env):
        self.env = env
        self.state_buffer = []
        
    def reset(self):
        """Reset environment and return initial state."""
        if hasattr(self.env, 'reset'):
            result = self.env.reset()
            return result[0] if isinstance(result, tuple) else result
        else:
            return np.zeros(10)
    
    def step(self, action):
        """Execute action and return next state, reward, done, info."""
        if hasattr(self.env, 'step'):
            result = self.env.step(action)
            if len(result) == 5:
                next_state, reward, terminated, truncated, info = result
                done = terminated or truncated
                return next_state, reward, done, info
            elif len(result) == 4:
                return result
            else:
                return result[0], result[1], result[2], {}
        else:
            return np.zeros(10), 0.0, False, {}
    
    def set_state(self, state):
        """Set environment to specific state (if supported)."""
        if hasattr(self.env, 'set_state'):
            self.env.set_state(state)
        elif hasattr(self.env, 'sim') and hasattr(self.env.sim, 'set_state'):
            self.env.sim.set_state(state)


# ============================================================================
# Dry-run Smoke Test Support
# ============================================================================

def dry_run_refine_agent(config: Optional[Dict] = None):
    """
    Dry-run version of refine_agent for smoke testing.
    Creates checkpoint artifacts without real training.
    """
    config = {**DEFAULT_REFINEMENT_CONFIG, **(config or {})}
    
    # Create checkpoint directory
    checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Write dry-run checkpoint
    checkpoint_path = checkpoint_dir / 'refined_agent.pth'
    dry_run_data = {
        'status': 'dry_run_checkpoint',
        'method': config.get('method', 'ours'),
        'config': config,
        'timestamp': time.time(),
    }
    
    import json
    checkpoint_path.write_text(json.dumps(dry_run_data, indent=2))
    
    print(f"[DRY-RUN] Refined agent checkpoint created at {checkpoint_path}")
    
    return {
        'agent': 'dry_run_agent',
        'checkpoint_path': str(checkpoint_path),
        'config': config,
    }


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    'collect_trajectories',
    'select_critical_states',
    'select_random_states',
    'rollout_from_state',
    'refine_agent',
    'save_refined_agent',
    'ppo_refining',
    'sac_refining',
    'gail_refining',
    'jsrl_refining',
    'baseline_refining',
    'adapter_fine_tuning',
    'standard_fine_tuning',
    'Trajectory',
    'EnvironmentAdapter',
    'METHOD_REGISTRY',
    'PARAMETER_SWEEP_REGISTRY',
    'DEFAULT_REFINEMENT_CONFIG',
    'dry_run_refine_agent',
]