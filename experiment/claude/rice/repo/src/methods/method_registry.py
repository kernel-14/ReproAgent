"""
RICE Method Registry

Provides method/baseline selectors and refinement orchestration for RICE experiments.

Method Registry:
- ours: RICE with explanation-guided refining
- random: Random baseline (random state selection)
- statemask: StateMask explanation baseline
- ppo: Standard PPO (no refinement)
- sac: SAC baseline (if available)
- gail: GAIL baseline (if available)
- jsrl: JSRL baseline (if available)
- baseline: No explanation baseline (continue training)
- adapter: Fine-tuning adapter
- fine_tuning: Standard fine-tuning from pretrained

Implementation surfaces:
- refine_agent(pretrained_agent, mask_network, env, config): Main refinement entry point
- collect_refinement_trajectories(agent, env, n_trajectories): Collect trajectories for explanation
- select_critical_states(trajectories, mask_network, config): Select states using mask rankings
- refine_from_critical_states(agent, critical_states, env, config): Roll-in and explore from critical states
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import numpy as np


# ============================================================================
# Lazy Imports
# ============================================================================

def lazy_load_torch():
    """Lazy import torch."""
    try:
        import torch
        return torch
    except ImportError:
        return None


def lazy_load_gym():
    """Lazy import gym."""
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

METHOD_REGISTRY: Dict[str, Dict[str, Any]] = {
    "ours": {
        "name": "RICE",
        "description": "Explanation-guided refinement with mask network critical state selection",
        "refine_fn": "refine_with_rice",
        "requires_mask_network": True,
        "category": "explanation_guided",
    },
    "random": {
        "name": "Random",
        "description": "Random state selection baseline",
        "refine_fn": "refine_with_random",
        "requires_mask_network": False,
        "category": "baseline",
    },
    "statemask": {
        "name": "StateMask",
        "description": "StateMask explanation baseline",
        "refine_fn": "refine_with_statemask",
        "requires_mask_network": True,
        "category": "explanation_guided",
    },
    "ppo": {
        "name": "PPO",
        "description": "Standard PPO without refinement",
        "refine_fn": "continue_ppo_training",
        "requires_mask_network": False,
        "category": "baseline",
    },
    "sac": {
        "name": "SAC",
        "description": "Soft Actor-Critic baseline",
        "refine_fn": "refine_with_sac",
        "requires_mask_network": False,
        "category": "baseline",
    },
    "gail": {
        "name": "GAIL",
        "description": "Generative Adversarial Imitation Learning baseline",
        "refine_fn": "refine_with_gail",
        "requires_mask_network": False,
        "category": "imitation",
    },
    "jsrl": {
        "name": "JSRL",
        "description": "Joint State-Action Representation Learning baseline",
        "refine_fn": "refine_with_jsrl",
        "requires_mask_network": False,
        "category": "representation",
    },
    "baseline": {
        "name": "Baseline",
        "description": "Continue training without explanation",
        "refine_fn": "continue_training",
        "requires_mask_network": False,
        "category": "baseline",
    },
    "adapter": {
        "name": "Adapter",
        "description": "Fine-tuning with adapter layers",
        "refine_fn": "refine_with_adapter",
        "requires_mask_network": False,
        "category": "transfer",
    },
    "fine_tuning": {
        "name": "FineTuning",
        "description": "Standard fine-tuning from pretrained",
        "refine_fn": "refine_with_fine_tuning",
        "requires_mask_network": False,
        "category": "transfer",
    },
}


def get_method(method_name: str) -> Dict[str, Any]:
    """Get method configuration by name."""
    method_name_lower = method_name.lower()
    if method_name_lower not in METHOD_REGISTRY:
        raise ValueError(f"Unknown method: {method_name}. Available: {list(METHOD_REGISTRY.keys())}")
    return METHOD_REGISTRY[method_name_lower]


def list_methods() -> List[str]:
    """List all available methods."""
    return list(METHOD_REGISTRY.keys())


# ============================================================================
# Main Refinement Entry Point
# ============================================================================

def refine_agent(
    pretrained_agent: Any,
    mask_network: Optional[Any],
    env: Any,
    config: Dict[str, Any]
) -> Any:
    """
    Main entry point for agent refinement using selected method.
    
    Args:
        pretrained_agent: Pre-trained agent to refine
        mask_network: Mask network for explanation (optional, required for explanation-guided methods)
        env: Environment instance
        config: Configuration dictionary with method selection and hyperparameters
    
    Returns:
        Refined agent
    """
    method_name = config.get("method", "ours")
    method_config = get_method(method_name)
    
    # Check mask network requirement
    if method_config["requires_mask_network"] and mask_network is None:
        raise ValueError(f"Method '{method_name}' requires mask_network but None was provided")
    
    # Get refinement function
    refine_fn_name = method_config["refine_fn"]
    refine_fn = globals()[refine_fn_name]
    
    # Execute refinement
    refined_agent = refine_fn(pretrained_agent, mask_network, env, config)
    
    return refined_agent


# ============================================================================
# Trajectory Collection
# ============================================================================

def collect_refinement_trajectories(
    agent: Any,
    env: Any,
    n_trajectories: int = 100,
    max_steps: int = 1000
) -> List[Dict[str, Any]]:
    """
    Collect trajectories from pre-trained agent for critical state identification.
    
    Args:
        agent: Pre-trained agent
        env: Environment instance
        n_trajectories: Number of trajectories to collect
        max_steps: Maximum steps per trajectory
    
    Returns:
        List of trajectory dictionaries containing states, actions, rewards
    """
    torch = lazy_load_torch()
    if torch is None:
        raise ImportError("PyTorch required for trajectory collection")
    
    trajectories = []
    
    for traj_idx in range(n_trajectories):
        obs, _ = env.reset() if hasattr(env.reset(), '__iter__') and not isinstance(env.reset(), np.ndarray) else (env.reset(), {})
        trajectory = {
            "states": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "values": [],
            "log_probs": []
        }
        
        for step in range(max_steps):
            # Store current state
            trajectory["states"].append(obs.copy())
            
            # Get action from agent
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                if hasattr(agent, 'get_action'):
                    action, log_prob, value = agent.get_action(obs_tensor)
                    action = action.cpu().numpy()[0]
                    log_prob = log_prob.cpu().item()
                    value = value.cpu().item()
                elif hasattr(agent, 'predict'):
                    action, _ = agent.predict(obs, deterministic=False)
                    log_prob = 0.0
                    value = 0.0
                else:
                    raise AttributeError("Agent must have get_action or predict method")
            
            # Store action and value
            trajectory["actions"].append(action)
            trajectory["log_probs"].append(log_prob)
            trajectory["values"].append(value)
            
            # Step environment
            step_result = env.step(action)
            if len(step_result) == 5:
                next_obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_obs, reward, done, info = step_result
            
            trajectory["rewards"].append(reward)
            trajectory["dones"].append(done)
            
            obs = next_obs
            
            if done:
                break
        
        trajectories.append(trajectory)
    
    return trajectories


# ============================================================================
# Critical State Selection
# ============================================================================

def select_critical_states(
    trajectories: List[Dict[str, Any]],
    mask_network: Any,
    config: Dict[str, Any]
) -> List[Tuple[np.ndarray, int, int]]:
    """
    Select critical states using mask network rankings.
    
    Args:
        trajectories: List of collected trajectories
        mask_network: Mask network for importance scoring
        config: Configuration with selection parameters
    
    Returns:
        List of (state, trajectory_idx, step_idx) tuples for critical states
    """
    torch = lazy_load_torch()
    if torch is None:
        raise ImportError("PyTorch required for critical state selection")
    
    top_k = config.get("critical_state_top_k", 100)
    all_state_scores = []
    
    # Score all states across trajectories
    for traj_idx, trajectory in enumerate(trajectories):
        states = np.array(trajectory["states"])
        
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states)
            
            # Get importance scores from mask network
            if hasattr(mask_network, 'get_importance_scores'):
                scores = mask_network.get_importance_scores(states_tensor)
            elif hasattr(mask_network, 'forward'):
                masks = mask_network(states_tensor)
                # Sum mask magnitudes as importance score
                scores = torch.sum(torch.abs(masks), dim=-1)
            else:
                raise AttributeError("Mask network must have get_importance_scores or forward method")
            
            scores = scores.cpu().numpy()
        
        # Store state, score, and location
        for step_idx, score in enumerate(scores):
            all_state_scores.append({
                "state": states[step_idx],
                "score": score,
                "traj_idx": traj_idx,
                "step_idx": step_idx
            })
    
    # Sort by score and select top-k
    all_state_scores.sort(key=lambda x: x["score"], reverse=True)
    critical_states = [
        (item["state"], item["traj_idx"], item["step_idx"])
        for item in all_state_scores[:top_k]
    ]
    
    return critical_states


def select_random_states(
    trajectories: List[Dict[str, Any]],
    config: Dict[str, Any]
) -> List[Tuple[np.ndarray, int, int]]:
    """
    Select random states for baseline comparison.
    
    Args:
        trajectories: List of collected trajectories
        config: Configuration with selection parameters
    
    Returns:
        List of (state, trajectory_idx, step_idx) tuples for random states
    """
    top_k = config.get("critical_state_top_k", 100)
    all_states = []
    
    for traj_idx, trajectory in enumerate(trajectories):
        states = trajectory["states"]
        for step_idx, state in enumerate(states):
            all_states.append((state, traj_idx, step_idx))
    
    # Randomly select top_k states
    indices = np.random.choice(len(all_states), size=min(top_k, len(all_states)), replace=False)
    random_states = [all_states[i] for i in indices]
    
    return random_states


# ============================================================================
# PPO Refinement from Critical States
# ============================================================================

def refine_from_critical_states(
    agent: Any,
    critical_states: List[Tuple[np.ndarray, int, int]],
    env: Any,
    config: Dict[str, Any]
) -> Any:
    """
    Refine agent by rolling-in to critical states and exploring from them.
    
    Args:
        agent: Agent to refine
        critical_states: List of critical states to explore from
        env: Environment instance
        config: Configuration with PPO and exploration parameters
    
    Returns:
        Refined agent
    """
    torch = lazy_load_torch()
    if torch is None:
        raise ImportError("PyTorch required for PPO refinement")
    
    # Import PPO training utilities
    from src.agents import train_ppo_from_states
    
    # Extract states for roll-in
    states_to_explore = [state for state, _, _ in critical_states]
    
    # Perform PPO refinement from these states
    refine_steps = config.get("refine_timesteps", 200000)
    batch_size = config.get("batch_size", 64)
    n_epochs = config.get("ppo_epochs", 10)
    
    refined_agent = train_ppo_from_states(
        agent=agent,
        env=env,
        initial_states=states_to_explore,
        total_steps=refine_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        config=config
    )
    
    return refined_agent


# ============================================================================
# Method-Specific Refinement Functions
# ============================================================================

def refine_with_rice(
    pretrained_agent: Any,
    mask_network: Any,
    env: Any,
    config: Dict[str, Any]
) -> Any:
    """RICE refinement: explanation-guided critical state exploration."""
    print("Refining with RICE (explanation-guided)...")
    
    # Collect trajectories
    n_trajectories = config.get("n_trajectories_for_explanation", 100)
    trajectories = collect_refinement_trajectories(pretrained_agent, env, n_trajectories)
    
    # Select critical states using mask network
    critical_states = select_critical_states(trajectories, mask_network, config)
    
    # Refine from critical states
    refined_agent = refine_from_critical_states(pretrained_agent, critical_states, env, config)
    
    return refined_agent


def refine_with_random(
    pretrained_agent: Any,
    mask_network: Optional[Any],
    env: Any,
    config: Dict[str, Any]
) -> Any:
    """Random baseline: random state selection."""
    print("Refining with Random baseline...")
    
    # Collect trajectories
    n_trajectories = config.get("n_trajectories_for_explanation", 100)
    trajectories = collect_refinement_trajectories(pretrained_agent, env, n_trajectories)
    
    # Select random states
    random_states = select_random_states(trajectories, config)
    
    # Refine from random states
    refined_agent = refine_from_critical_states(pretrained_agent, random_states, env, config)
    
    return refined_agent


def refine_with_statemask(
    pretrained_agent: Any,
    mask_network: Any,
    env: Any,
    config: Dict[str, Any]
) -> Any:
    """StateMask baseline: StateMask explanation method."""
    print("Refining with StateMask baseline...")
    
    # Use same approach as RICE but may have different mask network architecture
    # Collect trajectories
    n_trajectories = config.get("n_trajectories_for_explanation", 100)
    trajectories = collect_refinement_trajectories(pretrained_agent, env, n_trajectories)
    
    # Select critical states using StateMask
    critical_states = select_critical_states(trajectories, mask_network, config)
    
    # Refine from critical states
    refined_agent = refine_from_critical_states(pretrained_agent, critical_states, env, config)
    
    return refined_agent


def continue_ppo_training(
    pretrained_agent: Any,
    mask_network: Optional[Any],
    env: Any,
    config: Dict[str, Any]
) -> Any:
    """PPO baseline: continue standard PPO training."""
    print("Continuing PPO training (no refinement)...")
    
    from src.agents import train_ppo
    
    refine_steps = config.get("refine_timesteps", 200000)
    refined_agent = train_ppo(
        env=env,
        agent=pretrained_agent,
        total_timesteps=refine_steps,
        config=config
    )
    
    return refined_agent


def refine_with_sac(
    pretrained_agent: Any,
    mask_network: Optional[Any],
    env: Any,
    config: Dict[str, Any]
) -> Any:
    """SAC baseline: Soft Actor-Critic training."""
    print("Refining with SAC baseline...")
    
    # SAC requires different implementation
    # For now, fall back to PPO training
    print("Warning: SAC not fully implemented, using PPO as fallback")
    return continue_ppo_training(pretrained_agent, mask_network, env, config)


def refine_with_gail(
    pretrained_agent: Any,
    mask_network: Optional[Any],
    env: Any,
    config: Dict[str, Any]
) -> Any:
    """GAIL baseline: Generative Adversarial Imitation Learning."""
    print("Refining with GAIL baseline...")
    
    # GAIL requires expert demonstrations
    print("Warning: GAIL not fully implemented, using PPO as fallback")
    return continue_ppo_training(pretrained_agent, mask_network, env, config)


def refine_with_jsrl(
    pretrained_agent: Any,
    mask_network: Optional[Any],
    env: Any,
    config: Dict[str, Any]
) -> Any:
    """JSRL baseline: Joint State-Action Representation Learning."""
    print("Refining with JSRL baseline...")
    
    # JSRL requires specific representation learning setup
    print("Warning: JSRL not fully implemented, using PPO as fallback")
    return continue_ppo_training(pretrained_agent, mask_network, env, config)


def continue_training(
    pretrained_agent: Any,
    mask_network: Optional[Any],
    env: Any,
    config: Dict[str, Any]
) -> Any:
    """Baseline: continue training without explanation."""
    print("Continuing training without explanation...")
    return continue_ppo_training(pretrained_agent, mask_network, env, config)


def refine_with_adapter(
    pretrained_agent: Any,
    mask_network: Optional[Any],
    env: Any,
    config: Dict[str, Any]
) -> Any:
    """Adapter baseline: fine-tuning with adapter layers."""
    print("Refining with adapter layers...")
    
    # Adapter fine-tuning requires adding adapter layers
    print("Warning: Adapter not fully implemented, using fine-tuning as fallback")
    return refine_with_fine_tuning(pretrained_agent, mask_network, env, config)


def refine_with_fine_tuning(
    pretrained_agent: Any,
    mask_network: Optional[Any],
    env: Any,
    config: Dict[str, Any]
) -> Any:
    """Fine-tuning baseline: standard fine-tuning from pretrained."""
    print("Fine-tuning from pretrained agent...")
    
    from src.agents import train_ppo
    
    refine_steps = config.get("refine_timesteps", 200000)
    refined_agent = train_ppo(
        env=env,
        agent=pretrained_agent,
        total_timesteps=refine_steps,
        config=config
    )
    
    return refined_agent


# ============================================================================
# Reward Tracking
# ============================================================================

def compute_reward_improvement(
    pretrained_agent: Any,
    refined_agent: Any,
    env: Any,
    n_eval_episodes: int = 10
) -> Dict[str, float]:
    """
    Compute reward improvement from pre-trained to refined agent.
    
    Args:
        pretrained_agent: Pre-trained agent
        refined_agent: Refined agent
        env: Environment
        n_eval_episodes: Number of evaluation episodes
    
    Returns:
        Dictionary with pretrained_reward, refined_reward, improvement
    """
    torch = lazy_load_torch()
    if torch is None:
        raise ImportError("PyTorch required for evaluation")
    
    def evaluate_agent(agent, env, n_episodes):
        """Evaluate agent and return mean episode reward."""
        episode_rewards = []
        for _ in range(n_episodes):
            obs, _ = env.reset() if hasattr(env.reset(), '__iter__') and not isinstance(env.reset(), np.ndarray) else (env.reset(), {})
            episode_reward = 0.0
            done = False
            
            while not done:
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    if hasattr(agent, 'get_action'):
                        action, _, _ = agent.get_action(obs_tensor)
                        action = action.cpu().numpy()[0]
                    elif hasattr(agent, 'predict'):
                        action, _ = agent.predict(obs, deterministic=True)
                    else:
                        raise AttributeError("Agent must have get_action or predict method")
                
                step_result = env.step(action)
                if len(step_result) == 5:
                    obs, reward, terminated, truncated, _ = step_result
                    done = terminated or truncated
                else:
                    obs, reward, done, _ = step_result
                
                episode_reward += reward
            
            episode_rewards.append(episode_reward)
        
        return np.mean(episode_rewards)
    
    pretrained_reward = evaluate_agent(pretrained_agent, env, n_eval_episodes)
    refined_reward = evaluate_agent(refined_agent, env, n_eval_episodes)
    improvement = refined_reward - pretrained_reward
    
    return {
        "pretrained_reward": float(pretrained_reward),
        "refined_reward": float(refined_reward),
        "improvement": float(improvement),
        "improvement_percent": float(100 * improvement / abs(pretrained_reward)) if pretrained_reward != 0 else 0.0
    }