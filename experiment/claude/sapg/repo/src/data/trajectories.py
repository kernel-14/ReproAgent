"""
src/data/trajectories.py
SAPG: Split and Aggregate Policy Gradients - Trajectory Data Pipeline

Implements trajectory collection, storage, metric aggregation, and artifact
writing for the SAPG multi-policy on-policy RL setup.

Paper context:
  - N total parallel environments split across M policies (N/M each)
  - Each policy collects a rollout buffer of T steps
  - Leader aggregates off-policy data from followers via importance sampling
  - Tasks: ShadowHandOver, AllegroKuka, ShadowHandCatchUnderarm, etc.

reference_grounding: wp_004 src/data/trajectories.py

Artifact outputs (dry-run and real):
  results/dataset_registry.json
  results/metrics.json
  results/data_manifest.json
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Paper-derived benchmark / task registry
# ---------------------------------------------------------------------------

# All tasks named in the paper (Table 1, Figure 5, Figure 7, Figure 8)
BENCHMARK_REGISTRY: Dict[str, Dict[str, Any]] = {
    "ShadowHandOver": {
        "task_id": "ShadowHandOver",
        "difficulty": "hard",
        "obs_dim": 211,
        "act_dim": 20,
        "num_envs_paper": 24576,
        "max_episode_length": 1000,
        "success_threshold": 0.9,
        "paper_table": "Table 1",
        "environment_type": "isaacgym",
        "description": "Shadow Hand object passing between two hands",
    },
    "ShadowHandCatchUnderarm": {
        "task_id": "ShadowHandCatchUnderarm",
        "difficulty": "hard",
        "obs_dim": 211,
        "act_dim": 20,
        "num_envs_paper": 24576,
        "max_episode_length": 1000,
        "success_threshold": 0.9,
        "paper_table": "Table 1",
        "environment_type": "isaacgym",
        "description": "Shadow Hand underarm catch",
    },
    "ShadowHandCatchAbreast": {
        "task_id": "ShadowHandCatchAbreast",
        "difficulty": "hard",
        "obs_dim": 211,
        "act_dim": 20,
        "num_envs_paper": 24576,
        "max_episode_length": 1000,
        "success_threshold": 0.9,
        "paper_table": "Table 1",
        "environment_type": "isaacgym",
        "description": "Shadow Hand abreast catch",
    },
    "ShadowHandReOrientation": {
        "task_id": "ShadowHandReOrientation",
        "difficulty": "hard",
        "obs_dim": 211,
        "act_dim": 20,
        "num_envs_paper": 24576,
        "max_episode_length": 1000,
        "success_threshold": 0.9,
        "paper_table": "Table 1",
        "environment_type": "isaacgym",
        "description": "Shadow Hand object reorientation",
    },
    "AllegroHandReOrientation": {
        "task_id": "AllegroHandReOrientation",
        "difficulty": "hard",
        "obs_dim": 160,
        "act_dim": 16,
        "num_envs_paper": 24576,
        "max_episode_length": 1000,
        "success_threshold": 0.9,
        "paper_table": "Table 1",
        "environment_type": "isaacgym",
        "description": "Allegro Hand object reorientation",
    },
    "AllegroKuka": {
        "task_id": "AllegroKuka",
        "difficulty": "hard",
        "obs_dim": 160,
        "act_dim": 16,
        "num_envs_paper": 24576,
        "max_episode_length": 1000,
        "success_threshold": 0.9,
        "paper_table": "Table 1",
        "paper_figure": "Figure 5",
        "environment_type": "isaacgym",
        "description": "Allegro Hand + Kuka arm manipulation",
    },
    "harder_AllegroKuka": {
        "task_id": "harder_AllegroKuka",
        "difficulty": "hard",
        "obs_dim": 160,
        "act_dim": 16,
        "num_envs_paper": 24576,
        "max_episode_length": 1000,
        "success_threshold": 0.9,
        "paper_table": "Table 1",
        "paper_figure": "Figure 5",
        "environment_type": "isaacgym",
        "description": "Harder variant of AllegroKuka",
    },
    "Throw": {
        "task_id": "Throw",
        "difficulty": "hard",
        "obs_dim": 160,
        "act_dim": 16,
        "num_envs_paper": 24576,
        "max_episode_length": 1000,
        "success_threshold": 0.9,
        "paper_table": "Table 1",
        "environment_type": "isaacgym",
        "description": "Object throwing task",
    },
    "Regrasping": {
        "task_id": "Regrasping",
        "difficulty": "hard",
        "obs_dim": 160,
        "act_dim": 16,
        "num_envs_paper": 24576,
        "max_episode_length": 1000,
        "success_threshold": 0.9,
        "paper_table": "Table 1",
        "environment_type": "isaacgym",
        "description": "Object regrasping task",
    },
    "Reorientation": {
        "task_id": "Reorientation",
        "difficulty": "hard",
        "obs_dim": 160,
        "act_dim": 16,
        "num_envs_paper": 24576,
        "max_episode_length": 1000,
        "success_threshold": 0.9,
        "paper_table": "Table 1",
        "environment_type": "isaacgym",
        "description": "Object reorientation task",
    },
}

# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------

# Paper-derived metrics (reward is the primary metric; success rate secondary)
METRIC_REGISTRY: Dict[str, Dict[str, Any]] = {
    "episode_reward_mean": {
        "metric_id": "episode_reward_mean",
        "formula": "mean(sum(r_t) for each episode)",
        "aggregation": "mean over environments",
        "paper_reference": "Table 1, Figure 5",
        "primary": True,
        "unit": "reward",
        "higher_is_better": True,
    },
    "episode_reward_std": {
        "metric_id": "episode_reward_std",
        "formula": "std(sum(r_t) for each episode)",
        "aggregation": "std over environments",
        "paper_reference": "Table 1",
        "primary": False,
        "unit": "reward",
        "higher_is_better": None,
    },
    "episode_reward_max": {
        "metric_id": "episode_reward_max",
        "formula": "max(sum(r_t) for each episode)",
        "aggregation": "max over environments",
        "paper_reference": "Table 1",
        "primary": False,
        "unit": "reward",
        "higher_is_better": True,
    },
    "success_rate": {
        "metric_id": "success_rate",
        "formula": "mean(episode_success) over environments",
        "aggregation": "mean binary success flag",
        "paper_reference": "Table 1, Figure 5",
        "primary": True,
        "unit": "fraction [0, 1]",
        "higher_is_better": True,
    },
    "policy_entropy": {
        "metric_id": "policy_entropy",
        "formula": "-sum(pi(a|s) * log(pi(a|s)))",
        "aggregation": "mean over batch",
        "paper_reference": "Figure 6 ablation (entropy coefficient)",
        "primary": False,
        "unit": "nats",
        "higher_is_better": None,
    },
    "importance_weight_mean": {
        "metric_id": "importance_weight_mean",
        "formula": "mean(pi_leader(a|s) / pi_follower_i(a|s))",
        "aggregation": "mean over off-policy batch",
        "paper_reference": "Algorithm 1 importance sampling",
        "primary": False,
        "unit": "ratio",
        "higher_is_better": None,
    },
    "kl_divergence": {
        "metric_id": "kl_divergence",
        "formula": "KL(pi_old || pi_new)",
        "aggregation": "mean over batch",
        "paper_reference": "PPO clipping surrogate",
        "primary": False,
        "unit": "nats",
        "higher_is_better": None,
    },
    "value_loss": {
        "metric_id": "value_loss",
        "formula": "mean((V(s) - V_target)^2)",
        "aggregation": "mean over batch",
        "paper_reference": "PPO value loss",
        "primary": False,
        "unit": "loss",
        "higher_is_better": False,
    },
    "policy_loss": {
        "metric_id": "policy_loss",
        "formula": "mean(-min(r*A, clip(r,1-eps,1+eps)*A))",
        "aggregation": "mean over batch",
        "paper_reference": "PPO clipped surrogate objective",
        "primary": False,
        "unit": "loss",
        "higher_is_better": False,
    },
}

# ---------------------------------------------------------------------------
# Trajectory data structures
# ---------------------------------------------------------------------------


@dataclass
class Trajectory:
    """
    Single-policy trajectory segment collected over T steps from N/M environments.

    Paper context: Each of M follower policies collects a trajectory from N/M
    environments. The leader then aggregates these off-policy trajectories using
    importance sampling weights.
    """

    policy_id: int  # Index i in [0, M-1]
    task_id: str
    num_envs: int  # N/M environments for this policy
    num_steps: int  # T rollout steps

    # Core rollout tensors stored as nested Python lists or numpy arrays.
    # Torch tensors are only created inside methods that explicitly import torch.
    observations: Any = None       # shape: (T, N/M, obs_dim)
    actions: Any = None            # shape: (T, N/M, act_dim)
    rewards: Any = None            # shape: (T, N/M)
    log_probs: Any = None          # shape: (T, N/M) - log pi_i(a|s)
    values: Any = None             # shape: (T, N/M)
    dones: Any = None              # shape: (T, N/M)
    infos: List[Dict[str, Any]] = field(default_factory=list)

    # Episode-level aggregates (populated by compute_episode_stats)
    episode_rewards: Any = None    # shape: (num_completed_episodes,)
    episode_lengths: Any = None    # shape: (num_completed_episodes,)
    success_flags: Any = None      # shape: (num_completed_episodes,)

    # Metadata
    collection_time_s: float = 0.0
    total_steps_collected: int = 0
    timestamp: str = ""

    def compute_episode_stats(self) -> Dict[str, float]:
        """
        Compute reward metric aggregations from raw per-step rewards.

        Formula (paper Table 1 / Figure 5):
          episode_reward = sum(r_t) over episode steps
          episode_reward_mean = mean over completed episodes
          success_rate = mean(success_flag) over completed episodes
        """
        if self.rewards is None:
            return {}

        # Lazy import - only needed when actual tensor data is present
        try:
            import numpy as np
            rewards_arr = np.asarray(self.rewards)  # (T, N/M)
        except ImportError:
            # Fallback: pure-Python aggregation over nested lists
            rewards_arr = None

        stats: Dict[str, float] = {}

        if rewards_arr is not None:
            # Sum rewards over time axis to get per-env episode reward estimate
            # (approximate: treats full rollout as one episode segment)
            per_env_returns = rewards_arr.sum(axis=0)  # (N/M,)
            stats["episode_reward_mean"] = float(per_env_returns.mean())
            stats["episode_reward_std"] = float(per_env_returns.std())
            stats["episode_reward_max"] = float(per_env_returns.max())
            stats["episode_reward_min"] = float(per_env_returns.min())
        elif isinstance(self.rewards, list):
            # Pure-Python fallback
            flat: List[float] = []
            for step_rewards in self.rewards:
                if isinstance(step_rewards, (list, tuple)):
                    flat.extend(float(r) for r in step_rewards)
                else:
                    flat.append(float(step_rewards))
            if flat:
                n = len(flat)
                mean_r = sum(flat) / n
                std_r = (sum((r - mean_r) ** 2 for r in flat) / n) ** 0.5
                stats["episode_reward_mean"] = mean_r
                stats["episode_reward_std"] = std_r
                stats["episode_reward_max"] = max(flat)
                stats["episode_reward_min"] = min(flat)

        # Success rate from infos if available
        successes = []
        for info in self.infos:
            if isinstance(info, dict) and "success" in info:
                successes.append(float(info["success"]))
        if successes:
            stats["success_rate"] = sum(successes) / len(successes)

        return stats


@dataclass
class TrajectoryBatch:
    """
    Aggregated batch of trajectories from all M policies.

    Paper context: After all M followers collect their trajectories, the leader
    aggregates them. This class holds the full multi-policy batch before
    importance-sampling reweighting.
    """

    task_id: str
    num_policies: int          # M
    total_envs: int            # N
    num_steps: int             # T
    trajectories: List[Trajectory] = field(default_factory=list)

    # Aggregated metrics across all policies
    aggregate_stats: Dict[str, float] = field(default_factory=dict)

    # Importance sampling weights per policy (populated by IS module)
    # importance_weights[i] has shape (T, N/M) for policy i
    importance_weights: List[Any] = field(default_factory=list)

    def add_trajectory(self, traj: Trajectory) -> None:
        """Add a single-policy trajectory to the batch."""
        if len(self.trajectories) >= self.num_policies:
            raise ValueError(
                f"Batch already has {self.num_policies} trajectories; "
                f"cannot add more."
            )
        self.trajectories.append(traj)

    def compute_aggregate_stats(self) -> Dict[str, float]:
        """
        Aggregate reward metrics across all M policies.

        Formula: weighted mean of per-policy episode_reward_mean,
        weighted by number of environments per policy.
        """
        if not self.trajectories:
            return {}

        all_means: List[float] = []
        all_stds: List[float] = []
        all_maxes: List[float] = []
        all_success: List[float] = []

        for traj in self.trajectories:
            stats = traj.compute_episode_stats()
            if "episode_reward_mean" in stats:
                all_means.append(stats["episode_reward_mean"])
            if "episode_reward_std" in stats:
                all_stds.append(stats["episode_reward_std"])
            if "episode_reward_max" in stats:
                all_maxes.append(stats["episode_reward_max"])
            if "success_rate" in stats:
                all_success.append(stats["success_rate"])

        agg: Dict[str, float] = {}
        if all_means:
            agg["episode_reward_mean"] = sum(all_means) / len(all_means)
        if all_stds:
            # Pool standard deviations (approximate)
            agg["episode_reward_std"] = (
                sum(s ** 2 for s in all_stds) / len(all_stds)
            ) ** 0.5
        if all_maxes:
            agg["episode_reward_max"] = max(all_maxes)
        if all_success:
            agg["success_rate"] = sum(all_success) / len(all_success)

        self.aggregate_stats = agg
        return agg

    def is_complete(self) -> bool:
        """True when all M policy trajectories have been collected."""
        return len(self.trajectories) == self.num_policies


# ---------------------------------------------------------------------------
# Trajectory buffer (per-policy rollout storage)
# ---------------------------------------------------------------------------


class TrajectoryBuffer:
    """
    Fixed-size rollout buffer for one policy collecting T steps from N/M envs.

    Stores observations, actions, rewards, log_probs, values, dones in
    pre-allocated Python lists (converted to tensors lazily when needed).

    Paper context: Each follower policy i fills this buffer before the leader
    performs the importance-sampling aggregation step.
    """

    def __init__(
        self,
        policy_id: int,
        task_id: str,
        num_envs: int,
        num_steps: int,
        obs_dim: int,
        act_dim: int,
    ) -> None:
        self.policy_id = policy_id
        self.task_id = task_id
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self._observations: List[Any] = []
        self._actions: List[Any] = []
        self._rewards: List[Any] = []
        self._log_probs: List[Any] = []
        self._values: List[Any] = []
        self._dones: List[Any] = []
        self._infos: List[Dict[str, Any]] = []

        self._step_count = 0
        self._start_time = time.time()

    @property
    def is_full(self) -> bool:
        return self._step_count >= self.num_steps

    def add_step(
        self,
        obs: Any,
        action: Any,
        reward: Any,
        log_prob: Any,
        value: Any,
        done: Any,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add one time-step of data from all N/M environments."""
        if self.is_full:
            raise RuntimeError(
                f"TrajectoryBuffer for policy {self.policy_id} is full "
                f"({self.num_steps} steps)."
            )
        self._observations.append(obs)
        self._actions.append(action)
        self._rewards.append(reward)
        self._log_probs.append(log_prob)
        self._values.append(value)
        self._dones.append(done)
        self._infos.append(info or {})
        self._step_count += 1

    def finalize(self) -> Trajectory:
        """Convert buffer contents into a Trajectory object."""
        elapsed = time.time() - self._start_time
        traj = Trajectory(
            policy_id=self.policy_id,
            task_id=self.task_id,
            num_envs=self.num_envs,
            num_steps=self._step_count,
            observations=self._observations,
            actions=self._actions,
            rewards=self._rewards,
            log_probs=self._log_probs,
            values=self._values,
            dones=self._dones,
            infos=self._infos,
            collection_time_s=elapsed,
            total_steps_collected=self._step_count * self.num_envs,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        return traj

    def reset(self) -> None:
        """Clear buffer for next rollout collection."""
        self._observations.clear()
        self._actions.clear()
        self._rewards.clear()
        self._log_probs.clear()
        self._values.clear()
        self._dones.clear()
        self._infos.clear()
        self._step_count = 0
        self._start_time = time.time()


# ---------------------------------------------------------------------------
# Multi-policy trajectory collector
# ---------------------------------------------------------------------------


class TrajectoryCollector:
    """
    Orchestrates trajectory collection across M policies.

    Paper context (Algorithm 1):
      1. Each follower policy i collects T steps from N/M environments.
      2. All M trajectory buffers are assembled into a TrajectoryBatch.
      3. The leader uses importance sampling to aggregate off-policy data.

    This class manages the M TrajectoryBuffer instances and assembles the
    final TrajectoryBatch for the SAPG update step.
    """

    def __init__(
        self,
        task_id: str,
        num_policies: int,
        total_envs: int,
        num_steps: int,
        obs_dim: int,
        act_dim: int,
    ) -> None:
        self.task_id = task_id
        self.num_policies = num_policies
        self.total_envs = total_envs
        self.num_steps = num_steps
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # N/M environments per policy
        self.envs_per_policy = total_envs // num_policies

        self._buffers: List[TrajectoryBuffer] = [
            TrajectoryBuffer(
                policy_id=i,
                task_id=task_id,
                num_envs=self.envs_per_policy,
                num_steps=num_steps,
                obs_dim=obs_dim,
                act_dim=act_dim,
            )
            for i in range(num_policies)
        ]

    def get_buffer(self, policy_id: int) -> TrajectoryBuffer:
        """Return the buffer for policy i."""
        if policy_id < 0 or policy_id >= self.num_policies:
            raise IndexError(
                f"policy_id {policy_id} out of range [0, {self.num_policies})."
            )
        return self._buffers[policy_id]

    def all_buffers_full(self) -> bool:
        """True when every policy buffer has collected num_steps steps."""
        return all(buf.is_full for buf in self._buffers)

    def finalize_batch(self) -> TrajectoryBatch:
        """
        Finalize all buffers and assemble a TrajectoryBatch.

        Raises RuntimeError if any buffer is not yet full.
        """
        if not self.all_buffers_full():
            incomplete = [
                i for i, buf in enumerate(self._buffers) if not buf.is_full
            ]
            raise RuntimeError(
                f"Buffers for policies {incomplete} are not full; "
                f"cannot finalize batch."
            )

        batch = TrajectoryBatch(
            task_id=self.task_id,
            num_policies=self.num_policies,
            total_envs=self.total_envs,
            num_steps=self.num_steps,
        )
        for buf in self._buffers:
            batch.add_trajectory(buf.finalize())

        batch.compute_aggregate_stats()
        return batch

    def reset_all(self) -> None:
        """Reset all policy buffers for the next rollout."""
        for buf in self._buffers:
            buf.reset()


# ---------------------------------------------------------------------------
# Metric formula implementations
# ---------------------------------------------------------------------------


def compute_reward_metrics(rewards_sequence: Sequence[float]) -> Dict[str, float]:
    """
    Compute reward aggregation metrics from a flat sequence of episode returns.

    Paper metric contract (Table 1, Figure 5):
      episode_reward_mean = mean(R_i)
      episode_reward_std  = std(R_i)
      episode_reward_max  = max(R_i)

    Args:
        rewards_sequence: Iterable of per-episode cumulative rewards.

    Returns:
        Dict with keys: episode_reward_mean, episode_reward_std,
                        episode_reward_max, episode_reward_min, n_episodes.
    """
    rewards = list(rewards_sequence)
    if not rewards:
        return {
            "episode_reward_mean": 0.0,
            "episode_reward_std": 0.0,
            "episode_reward_max": 0.0,
            "episode_reward_min": 0.0,
            "n_episodes": 0,
        }

    n = len(rewards)
    mean_r = sum(rewards) / n
    variance = sum((r - mean_r) ** 2 for r in rewards) / n
    std_r = variance ** 0.5

    return {
        "episode_reward_mean": mean_r,
        "episode_reward_std": std_r,
        "episode_reward_max": max(rewards),
        "episode_reward_min": min(rewards),
        "n_episodes": n,
    }


def compute_success_rate(success_flags: Sequence[bool]) -> float:
    """
    Compute success rate from binary episode success flags.

    Formula: success_rate = sum(success_i) / N_episodes
    """
    flags = list(success_flags)
    if not flags:
        return 0.0
    return sum(1.0 for f in flags if f) / len(flags)


def compute_importance_weight_stats(
    log_probs_new: Sequence[float],
    log_probs_old: Sequence[float],
    clip_ratio: float = 10.0,
) -> Dict[str, float]:
    """
    Compute importance sampling weight statistics.

    Formula (Algorithm 1, SAPG paper):
      w_i = pi_leader(a|s) / pi_follower_i(a|s)
           = exp(log_pi_leader - log_pi_follower_i)

    Clipped to [1/clip_ratio, clip_ratio] for numerical stability.

    Args:
        log_probs_new: Log probabilities under the new (leader) policy.
        log_probs_old: Log probabilities under the old (follower) policy.
        clip_ratio: Clip importance weights to [1/clip_ratio, clip_ratio].

    Returns:
        Dict with mean, std, min, max of importance weights.
    """
    import math

    weights = []
    for lp_new, lp_old in zip(log_probs_new, log_probs_old):
        log_w = lp_new - lp_old
        w = math.exp(max(min(log_w, math.log(clip_ratio)), -math.log(clip_ratio)))
        weights.append(w)

    if not weights:
        return {
            "importance_weight_mean": 1.0,
            "importance_weight_std": 0.0,
            "importance_weight_min": 1.0,
            "importance_weight_max": 1.0,
        }

    n = len(weights)
    mean_w = sum(weights) / n
    std_w = (sum((w - mean_w) ** 2 for w in weights) / n) ** 0.5

    return {
        "importance_weight_mean": mean_w,
        "importance_weight_std": std_w,
        "importance_weight_min": min(weights),
        "importance_weight_max": max(weights),
    }


# ---------------------------------------------------------------------------
# Benchmark availability checks (lazy, no heavy imports)
# ---------------------------------------------------------------------------


def check_benchmark_availability(task_id: str) -> Dict[str, Any]:
    """
    Check whether a named benchmark task is available in the current environment.

    Returns a descriptor dict with 'available', 'reason', and task metadata.
    Does NOT import isaacgym or any simulator at module level.
    """
    if task_id not in BENCHMARK_REGISTRY:
        return {
            "task_id": task_id,
            "available": False,
            "reason": f"Task '{task_id}' not in BENCHMARK_REGISTRY.",
            "known_tasks": list(BENCHMARK_REGISTRY.keys()),
        }

    descriptor = dict(BENCHMARK_REGISTRY[task_id])

    # Lazy availability check for IsaacGym
    env_type = descriptor.get("environment_type", "isaacgym")
    if env_type == "isaacgym":
        try:
            import importlib
            spec = importlib.util.find_spec("isaacgym")
            if spec is None:
                descriptor["available"] = False
                descriptor["reason"] = (
                    "isaacgym package not found. Install IsaacGym from "
                    "https://developer.nvidia.com/isaac-gym to run this task."
                )
            else:
                descriptor["available"] = True
                descriptor["reason"] = "isaacgym package found."
        except Exception as exc:
            descriptor["available"] = False
            descriptor["reason"] = f"isaacgym availability check failed: {exc}"
    else:
        descriptor["available"] = False
        descriptor["reason"] = f"Unsupported environment_type: {env_type}"

    return descriptor


def get_smoke_fixture(task_id: str) -> Dict[str, Any]:
    """
    Return a minimal smoke fixture for a named benchmark task.

    Provides bounded synthetic data shapes for wiring validation without
    requiring the actual simulator to be installed.
    """
    if task_id not in BENCHMARK_REGISTRY:
        raise KeyError(f"Unknown task_id '{task_id}'. "
                       f"Known: {list(BENCHMARK_REGISTRY.keys())}")