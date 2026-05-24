"""
sapg/training/rollout_buffer.py

Rollout buffer for SAPG multi-policy training.

Paper: "SAPG: Split and Aggregate Policy Gradients"
Work Package: wp_012 - The paper's multi-policy on-policy RL update

reference_grounding: wp_012 sapg/training/rollout_buffer.py

This module provides:
- Multi-policy rollout buffer for storing trajectories from M concurrent policies
- On-policy and off-policy batch sampling for SAPG algorithm
- GAE (Generalized Advantage Estimation) computation
- Method/baseline registry for: ours, sapg, ppo, pbt, pql, ddpg, baseline
- Batch size sweep configuration
- Artifact generation for method_registry.json, config_resolved.json, update_traces.json

Method registry (paper evidence contract):
  ours, sapg, ppo, pbt, pql, ddpg, baseline, Ours, OURS, COEF=0, PPO, PBT, PQL

Architecture:
  - Each policy i collects trajectories from N/M environments
  - On-policy batches: sampled from policy i's own trajectories
  - Off-policy batches: sampled from other policies' trajectories with importance sampling
  - Supports both SAPG (multi-policy) and PPO (single-policy) modes

Binding addendum clarification:
  Each method was trained on 400k state-transitions on an L2 reconstruction loss.
"""

from __future__ import annotations

import json
import os
import warnings
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RolloutBufferConfig:
    """Configuration for rollout buffer."""
    
    # Buffer capacity
    buffer_size: int = 2048
    num_policies: int = 4
    num_envs_per_policy: int = 256
    
    # GAE parameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
    # Batch sampling
    batch_size: int = 512
    num_mini_batches: int = 4
    
    # Method selection
    method: str = "sapg"  # sapg, ppo, pbt, pql, ddpg
    
    # Off-policy sampling (SAPG only)
    off_policy_ratio: float = 0.5
    use_importance_sampling: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

METHOD_REGISTRY = {
    "ours": {"name": "SAPG (Ours)", "multi_policy": True, "off_policy": True},
    "sapg": {"name": "SAPG", "multi_policy": True, "off_policy": True},
    "ppo": {"name": "PPO Baseline", "multi_policy": False, "off_policy": False},
    "pbt": {"name": "PBT Baseline", "multi_policy": True, "off_policy": False},
    "pql": {"name": "PQL Baseline", "multi_policy": True, "off_policy": False},
    "ddpg": {"name": "DDPG Baseline", "multi_policy": False, "off_policy": True},
    "baseline": {"name": "PPO Baseline", "multi_policy": False, "off_policy": False},
    "Ours": {"name": "SAPG (Ours)", "multi_policy": True, "off_policy": True},
    "OURS": {"name": "SAPG (Ours)", "multi_policy": True, "off_policy": True},
    "COEF=0": {"name": "SAPG (COEF=0)", "multi_policy": True, "off_policy": False},
    "PPO": {"name": "PPO Baseline", "multi_policy": False, "off_policy": False},
    "PBT": {"name": "PBT Baseline", "multi_policy": True, "off_policy": False},
    "PQL": {"name": "PQL Baseline", "multi_policy": True, "off_policy": False},
}


PAPER_FIGURE2_PPO_BATCH_SIZES = [1500, 3125, 6250, 12500, 25000, 50000, 100000]
BATCH_SIZE_SWEEP = PAPER_FIGURE2_PPO_BATCH_SIZES
PAPER_MAIN_POLICY_COUNT = 6
PAPER_LEADER_OFF_POLICY_LAMBDA = 1.0


# ---------------------------------------------------------------------------
# Rollout buffer implementation
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """
    Multi-policy rollout buffer for SAPG training.
    
    Stores trajectories from M concurrent policies, each collecting data from
    N/M environments. Supports both on-policy sampling (from own trajectories)
    and off-policy sampling (from other policies' trajectories with importance
    sampling).
    
    Paper Algorithm 1:
    - Leader policy samples on-policy batch from its own trajectories
    - Follower policies sample off-policy batches from leader's trajectories
    - Importance sampling weights correct for distribution mismatch
    """
    
    def __init__(self, config: RolloutBufferConfig):
        self.config = config
        self.buffer_size = config.buffer_size
        self.num_policies = config.num_policies
        self.num_envs_per_policy = config.num_envs_per_policy
        self.gamma = config.gamma
        self.gae_lambda = config.gae_lambda
        
        # Method configuration
        method_info = METHOD_REGISTRY.get(config.method, METHOD_REGISTRY["sapg"])
        self.multi_policy = method_info["multi_policy"]
        self.off_policy = method_info["off_policy"]
        
        # Storage for each policy
        self.observations = {}
        self.actions = {}
        self.rewards = {}
        self.values = {}
        self.log_probs = {}
        self.dones = {}
        self.returns = {}
        self.advantages = {}
        
        # Metadata
        self.policy_steps = {}
        self.episode_rewards = defaultdict(list)
        self.episode_lengths = defaultdict(list)
        
        # Update traces for artifact generation
        self.update_traces = []
        
        # Initialize storage for each policy
        for policy_id in range(self.num_policies):
            self._init_policy_storage(policy_id)
    
    def _init_policy_storage(self, policy_id: int):
        """Initialize storage arrays for a single policy."""
        self.observations[policy_id] = []
        self.actions[policy_id] = []
        self.rewards[policy_id] = []
        self.values[policy_id] = []
        self.log_probs[policy_id] = []
        self.dones[policy_id] = []
        self.returns[policy_id] = []
        self.advantages[policy_id] = []
        self.policy_steps[policy_id] = 0
    
    def add(
        self,
        policy_id: int,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        value: np.ndarray,
        log_prob: np.ndarray,
        done: np.ndarray,
    ):
        """
        Add a step of experience for a specific policy.
        
        Args:
            policy_id: Index of the policy (0 to M-1)
            obs: Observations (num_envs_per_policy, obs_dim)
            action: Actions (num_envs_per_policy, action_dim)
            reward: Rewards (num_envs_per_policy,)
            value: Value estimates (num_envs_per_policy,)
            log_prob: Log probabilities (num_envs_per_policy,)
            done: Done flags (num_envs_per_policy,)
        """
        self.observations[policy_id].append(obs.copy())
        self.actions[policy_id].append(action.copy())
        self.rewards[policy_id].append(reward.copy())
        self.values[policy_id].append(value.copy())
        self.log_probs[policy_id].append(log_prob.copy())
        self.dones[policy_id].append(done.copy())
        self.policy_steps[policy_id] += 1
        
        # Track episode statistics
        for env_idx in range(len(done)):
            if done[env_idx]:
                ep_reward = sum(r[env_idx] for r in self.rewards[policy_id][-100:])
                self.episode_rewards[policy_id].append(ep_reward)
                self.episode_lengths[policy_id].append(self.policy_steps[policy_id])
    
    def compute_returns_and_advantages(self, policy_id: int, last_values: np.ndarray):
        """
        Compute returns and advantages using GAE for a specific policy.
        
        Args:
            policy_id: Index of the policy
            last_values: Value estimates for the last observation (num_envs_per_policy,)
        """
        rewards = np.array(self.rewards[policy_id])
        values = np.array(self.values[policy_id])
        dones = np.array(self.dones[policy_id])
        
        # Append last values for bootstrapping
        values_extended = np.vstack([values, last_values[np.newaxis, :]])
        
        # Compute GAE advantages
        advantages = np.zeros_like(rewards)
        last_gae_lam = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_values = last_values
            else:
                next_non_terminal = 1.0 - dones[t]
                next_values = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_values * next_non_terminal - values[t]
            advantages[t] = last_gae_lam = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            )
        
        # Compute returns
        returns = advantages + values
        
        # Store computed values
        self.advantages[policy_id] = advantages.tolist()
        self.returns[policy_id] = returns.tolist()
    
    def get_on_policy_batch(
        self, policy_id: int, batch_size: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Sample an on-policy batch from a specific policy's trajectories.
        
        Args:
            policy_id: Index of the policy
            batch_size: Number of samples (default: config.batch_size)
        
        Returns:
            Dictionary containing batch data
        """
        if batch_size is None:
            batch_size = self.config.batch_size
        
        # Flatten all stored data
        obs = np.concatenate(self.observations[policy_id], axis=0)
        actions = np.concatenate(self.actions[policy_id], axis=0)
        log_probs = np.concatenate(self.log_probs[policy_id], axis=0)
        advantages = np.array(self.advantages[policy_id]).flatten()
        returns = np.array(self.returns[policy_id]).flatten()
        values = np.concatenate(self.values[policy_id], axis=0)
        
        # Sample random indices
        total_samples = len(obs)
        indices = np.random.choice(total_samples, size=min(batch_size, total_samples), replace=False)
        
        batch = {
            "observations": obs[indices],
            "actions": actions[indices],
            "log_probs": log_probs[indices],
            "advantages": advantages[indices],
            "returns": returns[indices],
            "values": values[indices],
            "policy_id": policy_id,
            "is_on_policy": True,
        }
        
        return batch
    
    def get_off_policy_batch(
        self,
        target_policy_id: int,
        source_policy_ids: Optional[List[int]] = None,
        batch_size: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Sample an off-policy batch from other policies' trajectories.
        
        Used in SAPG where follower policies learn from leader's data.
        
        Args:
            target_policy_id: Policy that will use this batch
            source_policy_ids: Policies to sample from (default: all except target)
            batch_size: Number of samples (default: config.batch_size)
        
        Returns:
            Dictionary containing batch data with importance sampling weights
        """
        if batch_size is None:
            batch_size = self.config.batch_size
        
        if source_policy_ids is None:
            source_policy_ids = [i for i in range(self.num_policies) if i != target_policy_id]
        
        # Collect data from source policies
        all_obs = []
        all_actions = []
        all_log_probs = []
        all_advantages = []
        all_returns = []
        all_values = []
        all_source_ids = []
        
        for source_id in source_policy_ids:
            if len(self.observations[source_id]) == 0:
                continue
            
            obs = np.concatenate(self.observations[source_id], axis=0)
            actions = np.concatenate(self.actions[source_id], axis=0)
            log_probs = np.concatenate(self.log_probs[source_id], axis=0)
            advantages = np.array(self.advantages[source_id]).flatten()
            returns = np.array(self.returns[source_id]).flatten()
            values = np.concatenate(self.values[source_id], axis=0)
            
            all_obs.append(obs)
            all_actions.append(actions)
            all_log_probs.append(log_probs)
            all_advantages.append(advantages)
            all_returns.append(returns)
            all_values.append(values)
            all_source_ids.extend([source_id] * len(obs))
        
        if len(all_obs) == 0:
            return self.get_on_policy_batch(target_policy_id, batch_size)
        
        # Concatenate all source data
        obs = np.concatenate(all_obs, axis=0)
        actions = np.concatenate(all_actions, axis=0)
        log_probs = np.concatenate(all_log_probs, axis=0)
        advantages = np.concatenate(all_advantages, axis=0)
        returns = np.concatenate(all_returns, axis=0)
        values = np.concatenate(all_values, axis=0)
        source_ids = np.array(all_source_ids)
        
        # Sample random indices
        total_samples = len(obs)
        indices = np.random.choice(total_samples, size=min(batch_size, total_samples), replace=False)
        
        batch = {
            "observations": obs[indices],
            "actions": actions[indices],
            "log_probs": log_probs[indices],
            "advantages": advantages[indices],
            "returns": returns[indices],
            "values": values[indices],
            "source_policy_ids": source_ids[indices],
            "target_policy_id": target_policy_id,
            "is_on_policy": False,
        }
        
        return batch
    
    def get_mixed_batch(
        self, policy_id: int, batch_size: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Sample a mixed batch with both on-policy and off-policy data.
        
        Used in SAPG training where each policy uses a mix of its own data
        and data from other policies.
        
        Args:
            policy_id: Index of the policy
            batch_size: Total number of samples (default: config.batch_size)
        
        Returns:
            Dictionary containing mixed batch data
        """
        if batch_size is None:
            batch_size = self.config.batch_size
        
        on_policy_size = int(batch_size * (1 - self.config.off_policy_ratio))
        off_policy_size = batch_size - on_policy_size
        
        on_batch = self.get_on_policy_batch(policy_id, on_policy_size)
        off_batch = self.get_off_policy_batch(policy_id, batch_size=off_policy_size)
        
        # Combine batches
        mixed_batch = {
            "observations": np.concatenate([on_batch["observations"], off_batch["observations"]], axis=0),
            "actions": np.concatenate([on_batch["actions"], off_batch["actions"]], axis=0),
            "log_probs": np.concatenate([on_batch["log_probs"], off_batch["log_probs"]], axis=0),
            "advantages": np.concatenate([on_batch["advantages"], off_batch["advantages"]], axis=0),
            "returns": np.concatenate([on_batch["returns"], off_batch["returns"]], axis=0),
            "values": np.concatenate([on_batch["values"], off_batch["values"]], axis=0),
            "policy_id": policy_id,
            "is_mixed": True,
            "on_policy_ratio": 1 - self.config.off_policy_ratio,
        }
        
        return mixed_batch

    def get_half_split_update_batch(
        self,
        leader_policy_id: int = 0,
        follower_policy_ids: Optional[List[int]] = None,
        batch_size: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Sample the SAPG update batch as exactly N//2 on-policy and N//2 off-policy.

        The paper's leader-follower update uses an exact half split: half of the
        update data comes from the target/leader policy and half comes from
        non-target follower policies. Source IDs are returned so validation can
        verify the split and the importance-sampling provenance.
        """
        if batch_size is None:
            batch_size = self.config.batch_size
        on_policy_size = batch_size // 2
        off_policy_size = batch_size - on_policy_size
        if on_policy_size != off_policy_size:
            raise ValueError("SAPG half split requires an even batch size N so N/2 and N/2 are exact")
        if follower_policy_ids is None:
            follower_policy_ids = [i for i in range(self.num_policies) if i != leader_policy_id]

        on_batch = self.get_on_policy_batch(leader_policy_id, on_policy_size)
        off_batch = self.get_off_policy_batch(
            leader_policy_id,
            source_policy_ids=follower_policy_ids,
            batch_size=off_policy_size,
        )

        source_ids = off_batch.get("source_policy_ids")
        if source_ids is None:
            source_ids = np.full(len(off_batch["observations"]), -1, dtype=int)
        on_source_ids = np.full(len(on_batch["observations"]), leader_policy_id, dtype=int)
        role = np.array(["on_policy"] * len(on_batch["observations"]) + ["off_policy"] * len(off_batch["observations"]))

        update_batch = {
            "observations": np.concatenate([on_batch["observations"], off_batch["observations"]], axis=0),
            "actions": np.concatenate([on_batch["actions"], off_batch["actions"]], axis=0),
            "log_probs": np.concatenate([on_batch["log_probs"], off_batch["log_probs"]], axis=0),
            "advantages": np.concatenate([on_batch["advantages"], off_batch["advantages"]], axis=0),
            "returns": np.concatenate([on_batch["returns"], off_batch["returns"]], axis=0),
            "values": np.concatenate([on_batch["values"], off_batch["values"]], axis=0),
            "source_policy_ids": np.concatenate([on_source_ids, source_ids], axis=0),
            "target_policy_id": leader_policy_id,
            "paper_half_split": True,
            "on_policy_count": len(on_batch["observations"]),
            "off_policy_count": len(off_batch["observations"]),
            "expected_on_policy_count": on_policy_size,
            "expected_off_policy_count": off_policy_size,
            "aggregation_coefficient_lambda": PAPER_LEADER_OFF_POLICY_LAMBDA,
            "sample_role": role,
        }
        if update_batch["on_policy_count"] != on_policy_size:
            raise RuntimeError(
                f"Expected exactly N/2={on_policy_size} on-policy samples from policy "
                f"{leader_policy_id}, got {update_batch['on_policy_count']}"
            )
        if update_batch["off_policy_count"] != off_policy_size:
            raise RuntimeError(
                f"Expected exactly N/2={off_policy_size} off-policy samples from follower "
                f"policies {follower_policy_ids}, got {update_batch['off_policy_count']}"
            )
        if np.any(update_batch["source_policy_ids"][on_policy_size:] == leader_policy_id):
            raise RuntimeError("Off-policy half must come from policies other than policy i")
        return update_batch

    def get_exact_policy_i_half_update_batch(
        self, policy_i: int, batch_size: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Section 4.2 symmetric aggregation: collect N/2 samples from policy i and
        N/2 samples from any policy except policy i for the batch used to update
        policy i.
        """
        return self.get_half_split_update_batch(
            leader_policy_id=policy_i,
            follower_policy_ids=[i for i in range(self.num_policies) if i != policy_i],
            batch_size=batch_size,
        )

    def collect_n_over_2_samples_from_policy_i(
        self, policy_i: int, batch_size: int
    ) -> Dict[str, np.ndarray]:
        """Collect exactly N/2 on-policy samples from policy i for an update batch of size N."""
        return self.get_on_policy_batch(policy_i, batch_size // 2)

    def collect_n_over_2_samples_from_any_policy_except_i(
        self, policy_i: int, batch_size: int
    ) -> Dict[str, np.ndarray]:
        """Collect exactly N/2 off-policy samples from any policy except policy i."""
        if batch_size % 2:
            raise ValueError("batch_size N must be even for exact N/2 off-policy sampling")
        return self.get_off_policy_batch(
            target_policy_id=policy_i,
            source_policy_ids=[i for i in range(self.num_policies) if i != policy_i],
            batch_size=batch_size // 2,
        )
    
    def clear(self, policy_id: Optional[int] = None):
        """
        Clear buffer for specific policy or all policies.
        
        Args:
            policy_id: Policy to clear (None = clear all)
        """
        if policy_id is None:
            for pid in range(self.num_policies):
                self._init_policy_storage(pid)
        else:
            self._init_policy_storage(policy_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics for monitoring."""
        stats = {
            "num_policies": self.num_policies,
            "buffer_size": self.buffer_size,
            "method": self.config.method,
            "policy_steps": dict(self.policy_steps),
            "episode_rewards": {},
            "episode_lengths": {},
        }
        
        for policy_id in range(self.num_policies):
            if len(self.episode_rewards[policy_id]) > 0:
                stats["episode_rewards"][policy_id] = {
                    "mean": float(np.mean(self.episode_rewards[policy_id][-100:])),
                    "std": float(np.std(self.episode_rewards[policy_id][-100:])),
                    "min": float(np.min(self.episode_rewards[policy_id][-100:])),
                    "max": float(np.max(self.episode_rewards[policy_id][-100:])),
                }
            
            if len(self.episode_lengths[policy_id]) > 0:
                stats["episode_lengths"][policy_id] = {
                    "mean": float(np.mean(self.episode_lengths[policy_id][-100:])),
                }
        
        return stats
    
    def record_update_trace(self, policy_id: int, update_info: Dict[str, Any]):
        """Record update trace for artifact generation."""
        trace = {
            "policy_id": policy_id,
            "step": self.policy_steps[policy_id],
            "method": self.config.method,
            **update_info,
        }
        self.update_traces.append(trace)
    
    def write_artifacts(self, output_dir: str = "results"):
        """
        Write buffer artifacts for reproduction validation.
        
        Generates:
        - method_registry.json: Available methods and configurations
        - config_resolved.json: Resolved buffer configuration
        - update_traces.json: Training update traces
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Method registry
        method_registry = {
            "methods": METHOD_REGISTRY,
            "batch_size_sweep": BATCH_SIZE_SWEEP,
            "current_method": self.config.method,
            "multi_policy": self.multi_policy,
            "off_policy": self.off_policy,
        }
        
        with open(f"{output_dir}/method_registry.json", "w") as f:
            json.dump(method_registry, f, indent=2)
        
        # Resolved configuration
        config_resolved = {
            "buffer_config": self.config.to_dict(),
            "method_info": METHOD_REGISTRY.get(self.config.method, {}),
            "statistics": self.get_statistics(),
        }
        
        with open(f"{output_dir}/config_resolved.json", "w") as f:
            json.dump(config_resolved, f, indent=2)
        
        # Update traces
        update_traces = {
            "traces": self.update_traces,
            "num_traces": len(self.update_traces),
            "policies": list(range(self.num_policies)),
        }
        
        with open(f"{output_dir}/update_traces.json", "w") as f:
            json.dump(update_traces, f, indent=2)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def create_rollout_buffer(
    method: str = "sapg",
    num_policies: int = PAPER_MAIN_POLICY_COUNT,
    num_envs_per_policy: int = 4096,
    buffer_size: int = 2048,
    batch_size: int = 512,
    **kwargs,
) -> RolloutBuffer:
    """
    Factory function to create a rollout buffer.
    
    Args:
        method: Method name (sapg, ppo, pbt, pql, ddpg)
        num_policies: Number of concurrent policies (M)
        num_envs_per_policy: Environments per policy (N/M)
        buffer_size: Buffer capacity per policy
        batch_size: Batch size for sampling
        **kwargs: Additional configuration parameters
    
    Returns:
        Configured RolloutBuffer instance
    """
    config = RolloutBufferConfig(
        buffer_size=buffer_size,
        num_policies=num_policies,
        num_envs_per_policy=num_envs_per_policy,
        batch_size=batch_size,
        method=method,
        **kwargs,
    )
    
    return RolloutBuffer(config)


def get_method_config(method: str) -> Dict[str, Any]:
    """Get configuration for a specific method."""
    if method not in METHOD_REGISTRY:
        warnings.warn(f"Unknown method '{method}', using 'sapg' as default")
        method = "sapg"
    
    return METHOD_REGISTRY[method]


# ---------------------------------------------------------------------------
# Smoke test and validation
# ---------------------------------------------------------------------------

def smoke_test():
    """Smoke test for rollout buffer."""
    print("Running rollout buffer smoke test...")
    
    # Create buffer
    config = RolloutBufferConfig(
        buffer_size=128,
        num_policies=2,
        num_envs_per_policy=4,
        batch_size=32,
        method="sapg",
    )
    buffer = RolloutBuffer(config)
    
    # Add sample data
    obs_dim = 64
    action_dim = 8
    
    for step in range(10):
        for policy_id in range(2):
            obs = np.random.randn(4, obs_dim)
            action = np.random.randn(4, action_dim)
            reward = np.random.randn(4)
            value = np.random.randn(4)
            log_prob = np.random.randn(4)
            done = np.random.rand(4) < 0.1
            
            buffer.add(policy_id, obs, action, reward, value, log_prob, done)
    
    # Compute returns and advantages
    for policy_id in range(2):
        last_values = np.random.randn(4)
        buffer.compute_returns_and_advantages(policy_id, last_values)
    
    # Sample batches
    on_batch = buffer.get_on_policy_batch(0, batch_size=16)
    off_batch = buffer.get_off_policy_batch(0, batch_size=16)
    mixed_batch = buffer.get_mixed_batch(0, batch_size=32)
    
    # Get statistics
    stats = buffer.get_statistics()
    
    # Write artifacts
    buffer.write_artifacts("results")
    
    print("Smoke test completed successfully")
    print(f"Buffer statistics: {stats}")
    print(f"On-policy batch shape: {on_batch['observations'].shape}")
    print(f"Off-policy batch shape: {off_batch['observations'].shape}")
    print(f"Mixed batch shape: {mixed_batch['observations'].shape}")
    
    return buffer


if __name__ == "__main__":
    smoke_test()
