"""
sapg/utils/importance_sampling.py

Importance sampling utilities for SAPG multi-policy training.

Paper: "SAPG: Split and Aggregate Policy Gradients"
Work Package: wp_012 - The paper's multi-policy on-policy RL update

reference_grounding: wp_012 sapg/utils/importance_sampling.py

This module provides:
- Importance sampling ratio computation for off-policy aggregation
- PPO-style clipped surrogate loss for on-policy updates
- Off-policy importance-weighted aggregation term from paper
- Leader/follower policy update configuration
- Critic target handling for n-step (on-policy) and 1-step (off-policy) returns
- Update diagnostics and artifact generation

Method registry (paper evidence contract):
  ours, sapg, ppo, pbt, pql, ddpg, baseline, Ours, OURS, COEF=0, PPO, PBT, PQL

Architecture:
  - On-policy loss: PPO clipped surrogate objective for policy i on its own data D_i
  - Off-policy loss: Importance-weighted aggregation from other policies' data
  - Aggregation coefficient λ controls off-policy contribution
  - Supports both SAPG (multi-policy with aggregation) and PPO (single-policy) modes

Binding addendum clarification:
  The paper uses importance sampling to reuse data across M concurrent policies while
  maintaining an on-policy update structure. Each policy i updates using:
    L_i = L_on_policy(D_i) + λ * L_off_policy(D_{-i})
  where D_{-i} represents data from other policies with importance weighting.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import numpy as np


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ImportanceSamplingConfig:
    """Configuration for importance sampling and policy updates."""
    
    # PPO clipping parameters
    clip_epsilon: float = 0.2  # PPO clipping range
    clip_value_loss: bool = True  # Whether to clip value loss
    value_clip_epsilon: float = 0.2  # Value function clipping range
    
    # Off-policy aggregation parameters
    lambda_off_policy: float = 1.0  # Paper leader coefficient λ=1 for off-policy aggregation
    use_off_policy: bool = True  # Whether to use off-policy aggregation
    off_policy_subsample_ratio: float = 1.0  # Fraction of off-policy data to use
    
    # Importance sampling parameters
    importance_ratio_clip: float = 10.0  # Clip importance ratios to prevent instability
    normalize_advantages: bool = True  # Normalize advantages before computing loss
    
    # Policy update configuration
    num_policies: int = 6  # M policies in main SAPG/PBT/DexPBT comparisons
    leader_policy_indices: List[int] = field(default_factory=lambda: [0])  # Leader policies
    follower_policy_indices: List[int] = field(default_factory=lambda: [1, 2, 3])  # Follower policies
    
    # Critic target configuration
    use_gae: bool = True  # Use Generalized Advantage Estimation
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda parameter
    n_step_returns: int = 128  # Horizon for n-step returns (on-policy)
    use_one_step_off_policy: bool = True  # Use 1-step returns for off-policy
    
    # Update diagnostics
    track_policy_divergence: bool = True  # Track KL divergence between policies
    track_importance_weights: bool = True  # Track importance weight statistics
    
    # Method selection
    method: str = "sapg"  # "sapg", "ppo", "pbt", "pql", "ddpg"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> ImportanceSamplingConfig:
        """Create config from dictionary."""
        return cls(**config_dict)


@dataclass
class UpdateDiagnostics:
    """Diagnostics for policy update step."""
    
    # Loss components
    on_policy_loss: float = 0.0
    off_policy_loss: float = 0.0
    critic_loss: float = 0.0
    total_loss: float = 0.0
    
    # Policy-specific metrics
    policy_index: int = 0
    is_leader: bool = False
    
    # Importance sampling statistics
    mean_importance_ratio: float = 1.0
    max_importance_ratio: float = 1.0
    min_importance_ratio: float = 1.0
    clipped_ratio_fraction: float = 0.0
    
    # Advantage statistics
    mean_advantage: float = 0.0
    std_advantage: float = 1.0
    
    # Off-policy aggregation metadata
    num_off_policy_sources: int = 0
    off_policy_source_indices: List[int] = field(default_factory=list)
    off_policy_sample_count: int = 0
    
    # Policy divergence
    kl_divergence: Optional[float] = None
    entropy: Optional[float] = None
    
    # Timestamp
    timestamp: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert diagnostics to dictionary."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Importance sampling ratio computation
# ---------------------------------------------------------------------------

def compute_importance_ratio(
    log_probs_new: np.ndarray,
    log_probs_old: np.ndarray,
    clip_ratio: Optional[float] = None
) -> np.ndarray:
    """
    Compute importance sampling ratio π_new(a|s) / π_old(a|s).
    
    Args:
        log_probs_new: Log probabilities under new policy, shape (batch_size,)
        log_probs_old: Log probabilities under old policy, shape (batch_size,)
        clip_ratio: Optional clipping threshold for stability
    
    Returns:
        Importance ratios, shape (batch_size,)
    """
    # Compute ratio in log space for numerical stability
    log_ratio = log_probs_new - log_probs_old
    ratio = np.exp(log_ratio)
    
    # Clip ratios if specified
    if clip_ratio is not None:
        ratio = np.clip(ratio, 1.0 / clip_ratio, clip_ratio)
    
    return ratio


def compute_clipped_surrogate_loss(
    log_probs_new: np.ndarray,
    log_probs_old: np.ndarray,
    advantages: np.ndarray,
    clip_epsilon: float = 0.2,
    normalize_advantages: bool = True
) -> Tuple[float, Dict[str, float]]:
    """
    Compute PPO clipped surrogate loss.
    
    L^CLIP(θ) = E[min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)]
    where r(θ) = π_θ(a|s) / π_θ_old(a|s)
    
    Args:
        log_probs_new: Log probabilities under new policy, shape (batch_size,)
        log_probs_old: Log probabilities under old policy, shape (batch_size,)
        advantages: Advantage estimates, shape (batch_size,)
        clip_epsilon: PPO clipping range
        normalize_advantages: Whether to normalize advantages
    
    Returns:
        loss: Scalar loss value (negative for gradient ascent)
        diagnostics: Dictionary with loss components and statistics
    """
    # Normalize advantages
    if normalize_advantages and len(advantages) > 1:
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
    
    # Compute importance ratio
    ratio = compute_importance_ratio(log_probs_new, log_probs_old)
    
    # Compute surrogate losses
    surr1 = ratio * advantages
    surr2 = np.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    
    # Take minimum (pessimistic bound)
    clipped_loss = np.minimum(surr1, surr2)
    
    # Compute mean loss (negative for gradient ascent)
    loss = -np.mean(clipped_loss)
    
    # Compute diagnostics
    clipped_fraction = np.mean(np.abs(ratio - 1.0) > clip_epsilon)
    
    diagnostics = {
        "loss": loss,
        "mean_ratio": float(np.mean(ratio)),
        "max_ratio": float(np.max(ratio)),
        "min_ratio": float(np.min(ratio)),
        "clipped_fraction": float(clipped_fraction),
        "mean_advantage": float(np.mean(advantages)),
        "std_advantage": float(np.std(advantages)),
    }
    
    return loss, diagnostics


def compute_off_policy_aggregation_loss(
    target_log_probs: np.ndarray,
    source_log_probs_list: List[np.ndarray],
    source_old_log_probs_list: List[np.ndarray],
    advantages_list: List[np.ndarray],
    clip_epsilon: float = 0.2,
    importance_ratio_clip: float = 10.0,
    normalize_advantages: bool = True,
    subsample_ratio: float = 1.0
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute off-policy aggregation loss with importance weighting.
    
    L_off_policy = E_{D_{-i}}[w(s,a) * min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)]
    where w(s,a) = π_target(a|s) / π_source(a|s) is the importance weight
    
    Args:
        target_log_probs: Log probs under target policy, shape (total_batch_size,)
        source_log_probs_list: List of log probs under source policies (old)
        source_old_log_probs_list: List of log probs under source policies (behavior)
        advantages_list: List of advantage estimates from source policies
        clip_epsilon: PPO clipping range
        importance_ratio_clip: Clip importance weights for stability
        normalize_advantages: Whether to normalize advantages
        subsample_ratio: Fraction of off-policy data to use
    
    Returns:
        loss: Scalar loss value (negative for gradient ascent)
        diagnostics: Dictionary with loss components and statistics
    """
    if not source_log_probs_list:
        return 0.0, {"num_sources": 0, "total_samples": 0}
    
    # Concatenate all source data
    source_log_probs = np.concatenate(source_log_probs_list, axis=0)
    source_old_log_probs = np.concatenate(source_old_log_probs_list, axis=0)
    advantages = np.concatenate(advantages_list, axis=0)
    
    # Subsample if requested
    if subsample_ratio < 1.0:
        n_samples = int(len(advantages) * subsample_ratio)
        indices = np.random.choice(len(advantages), size=n_samples, replace=False)
        target_log_probs = target_log_probs[indices]
        source_log_probs = source_log_probs[indices]
        source_old_log_probs = source_old_log_probs[indices]
        advantages = advantages[indices]
    
    # Normalize advantages
    if normalize_advantages and len(advantages) > 1:
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
    
    # Compute importance weight: π_target / π_source
    importance_weight = compute_importance_ratio(
        target_log_probs, source_log_probs, clip_ratio=importance_ratio_clip
    )
    
    # Compute policy ratio for PPO clipping: π_target / π_source_old
    policy_ratio = compute_importance_ratio(target_log_probs, source_old_log_probs)
    
    # Compute clipped surrogate with importance weighting
    surr1 = importance_weight * policy_ratio * advantages
    surr2 = importance_weight * np.clip(
        policy_ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon
    ) * advantages
    
    clipped_loss = np.minimum(surr1, surr2)
    
    # Compute mean loss (negative for gradient ascent)
    loss = -np.mean(clipped_loss)
    
    # Compute diagnostics
    diagnostics = {
        "loss": loss,
        "num_sources": len(source_log_probs_list),
        "total_samples": len(advantages),
        "mean_importance_weight": float(np.mean(importance_weight)),
        "max_importance_weight": float(np.max(importance_weight)),
        "min_importance_weight": float(np.min(importance_weight)),
        "mean_policy_ratio": float(np.mean(policy_ratio)),
        "clipped_fraction": float(np.mean(np.abs(policy_ratio - 1.0) > clip_epsilon)),
    }
    
    return loss, diagnostics


def compute_value_loss(
    values_new: np.ndarray,
    values_old: np.ndarray,
    returns: np.ndarray,
    clip_value_loss: bool = True,
    value_clip_epsilon: float = 0.2
) -> Tuple[float, Dict[str, float]]:
    """
    Compute value function loss with optional clipping.
    
    Args:
        values_new: Value predictions under new parameters, shape (batch_size,)
        values_old: Value predictions under old parameters, shape (batch_size,)
        returns: Target returns, shape (batch_size,)
        clip_value_loss: Whether to clip value loss
        value_clip_epsilon: Clipping range for value loss
    
    Returns:
        loss: Scalar loss value
        diagnostics: Dictionary with loss components
    """
    if clip_value_loss:
        # Clipped value loss (PPO-style)
        value_pred_clipped = values_old + np.clip(
            values_new - values_old, -value_clip_epsilon, value_clip_epsilon
        )
        value_loss_unclipped = (values_new - returns) ** 2
        value_loss_clipped = (value_pred_clipped - returns) ** 2
        value_loss = 0.5 * np.mean(np.maximum(value_loss_unclipped, value_loss_clipped))
    else:
        # Standard MSE loss
        value_loss = 0.5 * np.mean((values_new - returns) ** 2)
    
    diagnostics = {
        "loss": float(value_loss),
        "mean_value": float(np.mean(values_new)),
        "mean_return": float(np.mean(returns)),
        "explained_variance": float(1.0 - np.var(returns - values_new) / (np.var(returns) + 1e-8)),
    }
    
    return value_loss, diagnostics


def compute_off_policy_critic_loss_eq8(
    critic_values_new: np.ndarray,
    rewards: np.ndarray,
    next_critic_values: np.ndarray,
    dones: np.ndarray,
    *,
    gamma: float = 0.99,
    importance_weights: Optional[np.ndarray] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Equation 8 off-policy critic loss for SAPG.

    The off-policy critic target is the one-step bootstrapped return
    ``r_t + gamma * (1 - done_t) * V(s_{t+1})`` from follower data, reweighted
    by the leader/follower importance weight when it is available. The optimized
    critic objective is mean squared error against the target.
    """
    values = np.asarray(critic_values_new, dtype=float)
    rewards = np.asarray(rewards, dtype=float)
    next_values = np.asarray(next_critic_values, dtype=float)
    dones = np.asarray(dones, dtype=float)
    targets = rewards + gamma * (1.0 - dones) * next_values
    squared_error = (values - targets) ** 2
    if importance_weights is not None:
        weights = np.asarray(importance_weights, dtype=float)
        squared_error = squared_error * weights
    else:
        weights = np.ones_like(squared_error)
    loss = 0.5 * float(np.mean(squared_error))
    return loss, {
        "equation": "Eq.8 off-policy critic loss",
        "loss": loss,
        "target_mean": float(np.mean(targets)),
        "critic_value_mean": float(np.mean(values)),
        "importance_weight_mean": float(np.mean(weights)),
        "mse": float(np.mean(squared_error)),
    }


def combine_actor_critic_loss_eq9(
    on_policy_actor_loss: float,
    off_policy_actor_loss: float,
    on_policy_critic_loss: float,
    off_policy_critic_loss: float,
    *,
    lambda_off_policy: float = 1.0,
    value_loss_coefficient: float = 0.5,
) -> Dict[str, float]:
    """Equation 9: combine on/off-policy actor and critic losses with λ=1 by default."""
    actor_total = float(on_policy_actor_loss) + lambda_off_policy * float(off_policy_actor_loss)
    critic_total = float(on_policy_critic_loss) + lambda_off_policy * float(off_policy_critic_loss)
    total = actor_total + value_loss_coefficient * critic_total
    return {
        "actor_total": actor_total,
        "critic_total": critic_total,
        "total_loss": total,
        "lambda_off_policy": float(lambda_off_policy),
        "value_loss_coefficient": float(value_loss_coefficient),
    }


def compute_gae_advantages(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Generalized Advantage Estimation (GAE).
    
    Args:
        rewards: Rewards, shape (batch_size,)
        values: Value predictions, shape (batch_size + 1,) (includes bootstrap)
        dones: Done flags, shape (batch_size,)
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
    
    Returns:
        advantages: GAE advantages, shape (batch_size,)
        returns: GAE returns, shape (batch_size,)
    """
    advantages = np.zeros_like(rewards)
    last_gae = 0.0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = values[t + 1]
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1.0 - dones[t]) - values[t]
        last_gae = delta + gamma * gae_lambda * (1.0 - dones[t]) * last_gae
        advantages[t] = last_gae
    
    returns = advantages + values[:-1]
    
    return advantages, returns


def compute_n_step_returns(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    n_steps: int = 128
) -> np.ndarray:
    """
    Compute n-step returns for on-policy updates.
    
    Args:
        rewards: Rewards, shape (batch_size,)
        values: Value predictions, shape (batch_size + 1,)
        dones: Done flags, shape (batch_size,)
        gamma: Discount factor
        n_steps: Number of steps for n-step returns
    
    Returns:
        returns: N-step returns, shape (batch_size,)
    """
    returns = np.zeros_like(rewards)
    
    for t in range(len(rewards)):
        ret = 0.0
        discount = 1.0
        
        for k in range(n_steps):
            if t + k >= len(rewards):
                break
            
            ret += discount * rewards[t + k]
            discount *= gamma * (1.0 - dones[t + k])
            
            if dones[t + k]:
                break
        
        # Add bootstrap value
        if t + n_steps < len(values):
            ret += discount * values[t + n_steps]
        
        returns[t] = ret
    
    return returns


def compute_one_step_returns(
    rewards: np.ndarray,
    next_values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99
) -> np.ndarray:
    """
    Compute 1-step returns for off-policy updates.
    
    Args:
        rewards: Rewards, shape (batch_size,)
        next_values: Next state values, shape (batch_size,)
        dones: Done flags, shape (batch_size,)
        gamma: Discount factor
    
    Returns:
        returns: 1-step returns, shape (batch_size,)
    """
    returns = rewards + gamma * next_values * (1.0 - dones)
    return returns


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

METHOD_REGISTRY = {
    "sapg": {
        "name": "SAPG",
        "description": "Split and Aggregate Policy Gradients (ours)",
        "use_off_policy": True,
        "num_policies": 6,
        "lambda_off_policy": 1.0,
    },
    "ours": {
        "name": "SAPG",
        "description": "Split and Aggregate Policy Gradients (ours)",
        "use_off_policy": True,
        "num_policies": 6,
        "lambda_off_policy": 1.0,
    },
    "ppo": {
        "name": "PPO",
        "description": "Proximal Policy Optimization baseline",
        "use_off_policy": False,
        "num_policies": 1,
        "lambda_off_policy": 0.0,
    },
    "pbt": {
        "name": "PBT",
        "description": "Population Based Training baseline",
        "use_off_policy": False,
        "num_policies": 6,
        "lambda_off_policy": 0.0,
    },
    "pql": {
        "name": "PQL",
        "description": "Policy Quality Learning baseline",
        "use_off_policy": True,
        "num_policies": 6,
        "lambda_off_policy": 0.3,
    },
    "ddpg": {
        "name": "DDPG",
        "description": "Deep Deterministic Policy Gradient baseline",
        "use_off_policy": True,
        "num_policies": 1,
        "lambda_off_policy": 1.0,
    },
    "baseline": {
        "name": "PPO",
        "description": "PPO baseline",
        "use_off_policy": False,
        "num_policies": 1,
        "lambda_off_policy": 0.0,
    },
}

# Aliases
METHOD_REGISTRY["Ours"] = METHOD_REGISTRY["ours"]
METHOD_REGISTRY["OURS"] = METHOD_REGISTRY["ours"]
METHOD_REGISTRY["COEF=0"] = METHOD_REGISTRY["ppo"]
METHOD_REGISTRY["PPO"] = METHOD_REGISTRY["ppo"]
METHOD_REGISTRY["PBT"] = METHOD_REGISTRY["pbt"]
METHOD_REGISTRY["PQL"] = METHOD_REGISTRY["pql"]


def get_method_config(method_name: str) -> Dict[str, Any]:
    """Get configuration for a registered method."""
    if method_name not in METHOD_REGISTRY:
        warnings.warn(f"Unknown method '{method_name}', defaulting to 'sapg'")
        method_name = "sapg"
    
    return METHOD_REGISTRY[method_name].copy()


# ---------------------------------------------------------------------------
# Artifact generation
# ---------------------------------------------------------------------------

def write_method_registry(output_path: str = "results/method_registry.json"):
    """Write method registry to JSON artifact."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    registry_data = {
        "methods": METHOD_REGISTRY,
        "timestamp": __import__("time").time(),
        "module": "sapg.utils.importance_sampling",
    }
    
    with open(output_path, "w") as f:
        json.dump(registry_data, f, indent=2)


def write_config_resolved(
    config: ImportanceSamplingConfig,
    output_path: str = "results/config_resolved.json"
):
    """Write resolved configuration to JSON artifact."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    config_data = {
        "importance_sampling_config": config.to_dict(),
        "method_config": get_method_config(config.method),
        "timestamp": __import__("time").time(),
        "module": "sapg.utils.importance_sampling",
    }
    
    with open(output_path, "w") as f:
        json.dump(config_data, f, indent=2)


def write_update_traces(
    diagnostics_list: List[UpdateDiagnostics],
    output_path: str = "results/update_traces.json"
):
    """Write update diagnostics traces to JSON artifact."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    traces_data = {
        "traces": [diag.to_dict() for diag in diagnostics_list],
        "num_updates": len(diagnostics_list),
        "timestamp": __import__("time").time(),
        "module": "sapg.utils.importance_sampling",
    }
    
    with open(output_path, "w") as f:
        json.dump(traces_data, f, indent=2)


# ---------------------------------------------------------------------------
# Smoke test / dry-run validation
# ---------------------------------------------------------------------------

def run_smoke_test():
    """Run smoke test to validate importance sampling utilities."""
    print("Running importance sampling smoke test...")
    
    # Create synthetic batch data
    batch_size = 128
    log_probs_new = np.random.randn(batch_size) * 0.1
    log_probs_old = np.random.randn(batch_size) * 0.1
    advantages = np.random.randn(batch_size)
    
    # Test on-policy loss
    on_policy_loss, on_policy_diag = compute_clipped_surrogate_loss(
        log_probs_new, log_probs_old, advantages
    )
    print(f"On-policy loss: {on_policy_loss:.4f}")
    print(f"  Mean ratio: {on_policy_diag['mean_ratio']:.4f}")
    print(f"  Clipped fraction: {on_policy_diag['clipped_fraction']:.4f}")
    
    # Test off-policy loss
    source_log_probs_list = [np.random.randn(batch_size) * 0.1 for _ in range(3)]
    source_old_log_probs_list = [np.random.randn(batch_size) * 0.1 for _ in range(3)]
    advantages_list = [np.random.randn(batch_size) for _ in range(3)]
    
    off_policy_loss, off_policy_diag = compute_off_policy_aggregation_loss(
        log_probs_new, source_log_probs_list, source_old_log_probs_list, advantages_list
    )
    print(f"Off-policy loss: {off_policy_loss:.4f}")
    print(f"  Num sources: {off_policy_diag['num_sources']}")
    print(f"  Mean importance weight: {off_policy_diag['mean_importance_weight']:.4f}")
    
    # Test value loss
    values_new = np.random.randn(batch_size)
    values_old = np.random.randn(batch_size)
    returns = np.random.randn(batch_size)
    
    value_loss, value_diag = compute_value_loss(values_new, values_old, returns)
    print(f"Value loss: {value_loss:.4f}")
    print(f"  Explained variance: {value_diag['explained_variance']:.4f}")
    
    # Test GAE computation
    rewards = np.random.randn(batch_size)
    values = np.random.randn(batch_size + 1)
    dones = np.random.rand(batch_size) < 0.1
    
    advantages_gae, returns_gae = compute_gae_advantages(rewards, values, dones)
    print(f"GAE advantages: mean={np.mean(advantages_gae):.4f}, std={np.std(advantages_gae):.4f}")
    
    # Test method registry
    print("\nMethod registry:")
    for method_name in ["sapg", "ppo", "pbt", "pql"]:
        config = get_method_config(method_name)
        print(f"  {method_name}: {config['name']} (off_policy={config['use_off_policy']})")
    
    # Write artifacts
    print("\nWriting artifacts...")
    write_method_registry()
    
    config = ImportanceSamplingConfig(method="sapg", num_policies=4)
    write_config_resolved(config)
    
    diagnostics = UpdateDiagnostics(
        on_policy_loss=on_policy_loss,
        off_policy_loss=off_policy_loss,
        critic_loss=value_loss,
        total_loss=on_policy_loss + 0.5 * off_policy_loss + 0.5 * value_loss,
        policy_index=0,
        is_leader=True,
        mean_importance_ratio=off_policy_diag["mean_importance_weight"],
        timestamp=__import__("time").time(),
    )
    write_update_traces([diagnostics])
    
    print("Smoke test completed successfully!")
    print("Artifacts written:")
    print("  - results/method_registry.json")
    print("  - results/config_resolved.json")
    print("  - results/update_traces.json")


if __name__ == "__main__":
    run_smoke_test()
