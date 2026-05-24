# sapg/training/data_collector.py
# SAPG: Split and Aggregate Policy Gradients - Data Collection Module
# reference_grounding: wp_013 sapg/training/data_collector.py
#
# Paper evidence contract: Complete method/baseline selector set includes
# ours, sapg, ppo, pbt, pql, ddpg
#
# This module implements the data collection mechanism for SAPG's multi-policy
# structure. It handles:
# - Collecting rollouts from M policies operating on N/M environments each
# - Supporting different aggregation variants (leader-follower, symmetric, no off-policy)
# - Providing ablation configuration registry
# - Writing artifacts for ablation registry and update source sets
#
# Binding addendum clarification: The leader/follower design is a causal choice
# tested by ablations, so it must remain distinguishable from symmetric aggregation.

import os
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import numpy as np


PAPER_MAIN_POLICY_COUNT = 6
PAPER_NUM_ENVS = 24576
PAPER_ENVS_PER_POLICY = PAPER_NUM_ENVS // PAPER_MAIN_POLICY_COUNT
PAPER_LEADER_OFF_POLICY_LAMBDA = 1.0
PAPER_PPO_STEPS_PER_ENV_BEFORE_UPDATE = 16


@dataclass
class AggregationVariant:
    """Configuration for different aggregation strategies in SAPG ablations."""
    variant_id: str
    name: str
    description: str
    use_leader: bool
    symmetric_aggregation: bool
    use_off_policy: bool
    entropy_coefficient: float
    aggregation_coefficient: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Ablation variant registry - paper evidence contract
# Figure 6 ablations: SAPG (ours), symmetric aggregation, no off-policy, entropy variations
ABLATION_REGISTRY = {
    "ours": AggregationVariant(
        variant_id="ours",
        name="SAPG (Ours)",
        description="Full SAPG with leader-follower and off-policy aggregation",
        use_leader=True,
        symmetric_aggregation=False,
        use_off_policy=True,
        entropy_coefficient=0.01,
        aggregation_coefficient=PAPER_LEADER_OFF_POLICY_LAMBDA
    ),
    "sapg": AggregationVariant(
        variant_id="sapg",
        name="SAPG",
        description="Full SAPG with leader-follower and off-policy aggregation",
        use_leader=True,
        symmetric_aggregation=False,
        use_off_policy=True,
        entropy_coefficient=0.01,
        aggregation_coefficient=PAPER_LEADER_OFF_POLICY_LAMBDA
    ),
    "symmetric": AggregationVariant(
        variant_id="symmetric",
        name="Symmetric Aggregation",
        description="No designated leader; each worker updated with all off-policy data symmetrically",
        use_leader=False,
        symmetric_aggregation=True,
        use_off_policy=True,
        entropy_coefficient=0.01,
        aggregation_coefficient=0.5
    ),
    "no_offpolicy": AggregationVariant(
        variant_id="no_offpolicy",
        name="No Off-Policy",
        description="SAPG without off-policy data aggregation",
        use_leader=True,
        symmetric_aggregation=False,
        use_off_policy=False,
        entropy_coefficient=0.01,
        aggregation_coefficient=0.0
    ),
    "entropy_0": AggregationVariant(
        variant_id="entropy_0",
        name="Entropy Coefficient 0",
        description="SAPG with entropy coefficient = 0",
        use_leader=True,
        symmetric_aggregation=False,
        use_off_policy=True,
        entropy_coefficient=0.0,
        aggregation_coefficient=PAPER_LEADER_OFF_POLICY_LAMBDA
    ),
    "entropy_0005": AggregationVariant(
        variant_id="entropy_0005",
        name="Entropy Coefficient 0.005",
        description="SAPG with entropy coefficient = 0.005",
        use_leader=True,
        symmetric_aggregation=False,
        use_off_policy=True,
        entropy_coefficient=0.005,
        aggregation_coefficient=PAPER_LEADER_OFF_POLICY_LAMBDA
    ),
    "entropy_001": AggregationVariant(
        variant_id="entropy_001",
        name="Entropy Coefficient 0.01",
        description="SAPG with entropy coefficient = 0.01 (default)",
        use_leader=True,
        symmetric_aggregation=False,
        use_off_policy=True,
        entropy_coefficient=0.01,
        aggregation_coefficient=PAPER_LEADER_OFF_POLICY_LAMBDA
    ),
    "ppo": AggregationVariant(
        variant_id="ppo",
        name="PPO Baseline",
        description="Standard PPO with single policy (M=1)",
        use_leader=False,
        symmetric_aggregation=False,
        use_off_policy=False,
        entropy_coefficient=0.01,
        aggregation_coefficient=0.0
    ),
}

# Method aliases - paper evidence contract
METHOD_ALIASES = {
    "Ours": "ours",
    "OURS": "ours",
    "COEF=0": "entropy_0",
    "PPO": "ppo",
    "PBT": "ppo",  # PBT uses PPO as base
    "PQL": "ppo",  # PQL uses PPO as base
    "baseline": "ppo",
}


def get_aggregation_variant(variant_name: str) -> AggregationVariant:
    """
    Get aggregation variant configuration by name.
    
    Paper evidence contract: expose method/baseline/attack selectors for
    ours, sapg, ppo, pbt, pql, ddpg.
    
    Args:
        variant_name: Name or alias of the aggregation variant
        
    Returns:
        AggregationVariant configuration
    """
    # Resolve aliases
    resolved_name = METHOD_ALIASES.get(variant_name, variant_name)
    
    if resolved_name not in ABLATION_REGISTRY:
        raise ValueError(
            f"Unknown aggregation variant: {variant_name}. "
            f"Available variants: {list(ABLATION_REGISTRY.keys())}"
        )
    
    return ABLATION_REGISTRY[resolved_name]


def select_source_policies(
    policy_index: int,
    variant: str,
    M: int
) -> List[int]:
    """
    Select which policies contribute data for updating a given policy.
    
    This implements the core aggregation logic that distinguishes SAPG variants:
    - Leader-follower: policy 0 (leader) aggregates from all followers
    - Symmetric: each policy aggregates from all other policies
    - No off-policy: each policy uses only its own data
    
    Paper evidence contract: The leader/follower design is a causal choice
    tested by ablations, so it must remain distinguishable from symmetric
    aggregation.
    
    Args:
        policy_index: Index of the policy being updated (0 to M-1)
        variant: Aggregation variant name
        M: Total number of policies
        
    Returns:
        List of policy indices whose data should be used for the update
    """
    config = get_aggregation_variant(variant)
    
    if not config.use_off_policy:
        # No off-policy: only use own data
        return [policy_index]
    
    if config.symmetric_aggregation:
        # Symmetric: all policies aggregate from all others
        return list(range(M))
    
    if config.use_leader:
        # Leader-follower: policy 0 is leader, aggregates from all
        # Followers only use their own data
        if policy_index == 0:
            return list(range(M))  # Leader aggregates from all
        else:
            return [policy_index]  # Followers use only own data
    
    # Default: only own data
    return [policy_index]


class MultiPolicyDataCollector:
    """
    Data collector for SAPG's multi-policy structure.
    
    Collects rollouts from M policies operating on N/M environments each.
    Supports different aggregation strategies for ablation studies.
    
    Paper evidence contract: Implements the split-and-aggregate mechanism
    from Algorithm 1 in the paper.
    """
    
    def __init__(
        self,
        num_policies: int,
        envs_per_policy: int,
        aggregation_variant: str = "sapg",
        rollout_length: int = PAPER_PPO_STEPS_PER_ENV_BEFORE_UPDATE,
    ):
        """
        Initialize multi-policy data collector.
        
        Args:
            num_policies: Number of policies (M in paper)
            envs_per_policy: Environments per policy (N/M in paper)
            aggregation_variant: Aggregation strategy variant
            rollout_length: Steps to collect before update. The paper PPO/SAPG
                setup collects exactly 16 steps per environment instance before
                every PPO update.
        """
        self.num_policies = num_policies
        self.envs_per_policy = envs_per_policy
        self.total_envs = num_policies * envs_per_policy
        self.rollout_length = rollout_length
        
        self.variant_config = get_aggregation_variant(aggregation_variant)
        self.aggregation_variant = aggregation_variant
        
        # Storage for collected data per policy
        self.policy_buffers = [[] for _ in range(num_policies)]
        self.current_step = 0

    def collect_16_steps_before_ppo_update(self, policy_step_fn) -> Dict[str, Any]:
        """
        Collect exactly 16 steps of experience per environment instance before a PPO update.

        The callable receives ``(policy_index, step_index, envs_per_policy)`` and must
        return the same arrays accepted by ``collect_step``. Keeping this helper
        explicit makes the paper's 16-step rollout contract visible to training
        and validation code.
        """
        for step_index in range(PAPER_PPO_STEPS_PER_ENV_BEFORE_UPDATE):
            for policy_index in range(self.num_policies):
                step = policy_step_fn(policy_index, step_index, self.envs_per_policy)
                self.collect_step(
                    policy_index,
                    step["observations"],
                    step["actions"],
                    step["rewards"],
                    step["dones"],
                    step["values"],
                    step["log_probs"],
                )
        return {
            "steps_per_environment_before_ppo_update": PAPER_PPO_STEPS_PER_ENV_BEFORE_UPDATE,
            "ready_for_update": self.is_ready(),
            "buffer_sizes": [len(buffer) for buffer in self.policy_buffers],
        }
        
    def collect_step(
        self,
        policy_index: int,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
        log_probs: np.ndarray,
    ) -> None:
        """
        Collect a single step of data from one policy.
        
        Args:
            policy_index: Index of the policy (0 to M-1)
            observations: Observations from environments [envs_per_policy, obs_dim]
            actions: Actions taken [envs_per_policy, action_dim]
            rewards: Rewards received [envs_per_policy]
            dones: Episode termination flags [envs_per_policy]
            values: Value estimates [envs_per_policy]
            log_probs: Log probabilities of actions [envs_per_policy]
        """
        step_data = {
            "observations": observations.copy(),
            "actions": actions.copy(),
            "rewards": rewards.copy(),
            "dones": dones.copy(),
            "values": values.copy(),
            "log_probs": log_probs.copy(),
        }
        
        self.policy_buffers[policy_index].append(step_data)
    
    def is_ready(self) -> bool:
        """Check if enough data has been collected for an update."""
        return all(
            len(buffer) >= self.rollout_length
            for buffer in self.policy_buffers
        )
    
    def get_update_data(
        self,
        policy_index: int
    ) -> Dict[str, np.ndarray]:
        """
        Get data for updating a specific policy.
        
        Implements the aggregation logic based on the variant configuration.
        
        Args:
            policy_index: Index of the policy to update
            
        Returns:
            Dictionary containing aggregated rollout data
        """
        source_policies = select_source_policies(
            policy_index,
            self.aggregation_variant,
            self.num_policies
        )
        
        # Collect data from source policies
        all_observations = []
        all_actions = []
        all_rewards = []
        all_dones = []
        all_values = []
        all_log_probs = []
        all_policy_ids = []
        
        for source_idx in source_policies:
            buffer = self.policy_buffers[source_idx]
            
            for step_data in buffer[:self.rollout_length]:
                all_observations.append(step_data["observations"])
                all_actions.append(step_data["actions"])
                all_rewards.append(step_data["rewards"])
                all_dones.append(step_data["dones"])
                all_values.append(step_data["values"])
                all_log_probs.append(step_data["log_probs"])
                all_policy_ids.append(
                    np.full(len(step_data["rewards"]), source_idx)
                )
        
        # Stack into arrays
        return {
            "observations": np.concatenate(all_observations, axis=0),
            "actions": np.concatenate(all_actions, axis=0),
            "rewards": np.concatenate(all_rewards, axis=0),
            "dones": np.concatenate(all_dones, axis=0),
            "values": np.concatenate(all_values, axis=0),
            "log_probs": np.concatenate(all_log_probs, axis=0),
            "policy_ids": np.concatenate(all_policy_ids, axis=0),
            "source_policies": source_policies,
        }

    def get_exact_half_split_update_data(
        self,
        policy_index: int,
        batch_size: int,
    ) -> Dict[str, np.ndarray]:
        """
        Return an exact SAPG Section 4.2/4.3 update batch for policy i.

        The first N/2 rows are collected from policy i. The second N/2 rows are
        sub-sampled from policies other than i. This makes the required
        provenance explicit for both symmetric aggregation and leader-follower
        aggregation validation.
        """
        if batch_size % 2:
            raise ValueError("batch_size N must be even so the paper N/2 split is exact")
        own = self._flatten_policy_buffer(policy_index)
        other_ids = [j for j in range(self.num_policies) if j != policy_index]
        others = self._flatten_policy_buffer_many(other_ids)
        half = batch_size // 2
        if len(own["rewards"]) < half:
            raise RuntimeError(f"policy {policy_index} has {len(own['rewards'])} samples, expected N/2={half}")
        if len(others["rewards"]) < half:
            raise RuntimeError(f"non-policy-{policy_index} pool has {len(others['rewards'])} samples, expected N/2={half}")

        own_idx = np.arange(half)
        other_idx = np.arange(half)
        return {
            "observations": np.concatenate([own["observations"][own_idx], others["observations"][other_idx]], axis=0),
            "actions": np.concatenate([own["actions"][own_idx], others["actions"][other_idx]], axis=0),
            "rewards": np.concatenate([own["rewards"][own_idx], others["rewards"][other_idx]], axis=0),
            "dones": np.concatenate([own["dones"][own_idx], others["dones"][other_idx]], axis=0),
            "values": np.concatenate([own["values"][own_idx], others["values"][other_idx]], axis=0),
            "log_probs": np.concatenate([own["log_probs"][own_idx], others["log_probs"][other_idx]], axis=0),
            "policy_ids": np.concatenate([
                np.full(half, policy_index, dtype=int),
                others["policy_ids"][other_idx],
            ]),
            "sample_role": np.array(["on_policy"] * half + ["off_policy"] * half),
            "on_policy_count": half,
            "off_policy_count": half,
            "paper_half_split": "N/2 from policy i and N/2 from any policy except policy i",
            "aggregation_coefficient_lambda": PAPER_LEADER_OFF_POLICY_LAMBDA,
        }

    def get_leader_update_data_with_follower_half_split(
        self,
        leader_policy_id: int = 0,
        batch_size: int = 4096,
    ) -> Dict[str, np.ndarray]:
        """Collect N/2 leader samples and N/2 follower samples for updating the leader."""
        return self.get_exact_half_split_update_data(leader_policy_id, batch_size)

    def _flatten_policy_buffer(self, policy_index: int) -> Dict[str, np.ndarray]:
        buffer = self.policy_buffers[policy_index]
        if not buffer:
            raise RuntimeError(f"No rollout data collected for policy {policy_index}")
        return {
            "observations": np.concatenate([step["observations"] for step in buffer], axis=0),
            "actions": np.concatenate([step["actions"] for step in buffer], axis=0),
            "rewards": np.concatenate([step["rewards"] for step in buffer], axis=0),
            "dones": np.concatenate([step["dones"] for step in buffer], axis=0),
            "values": np.concatenate([step["values"] for step in buffer], axis=0),
            "log_probs": np.concatenate([step["log_probs"] for step in buffer], axis=0),
            "policy_ids": np.full(sum(len(step["rewards"]) for step in buffer), policy_index, dtype=int),
        }

    def _flatten_policy_buffer_many(self, policy_indices: List[int]) -> Dict[str, np.ndarray]:
        flattened = [self._flatten_policy_buffer(policy_index) for policy_index in policy_indices]
        return {
            key: np.concatenate([item[key] for item in flattened], axis=0)
            for key in ["observations", "actions", "rewards", "dones", "values", "log_probs", "policy_ids"]
        }
    
    def clear_buffers(self) -> None:
        """Clear collected data after update."""
        self.policy_buffers = [[] for _ in range(self.num_policies)]
        self.current_step = 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about collected data."""
        return {
            "num_policies": self.num_policies,
            "envs_per_policy": self.envs_per_policy,
            "total_envs": self.total_envs,
            "rollout_length": self.rollout_length,
            "aggregation_variant": self.aggregation_variant,
            "buffer_sizes": [len(buf) for buf in self.policy_buffers],
            "variant_config": self.variant_config.to_dict(),
        }


def write_ablation_registry(output_dir: str = "results") -> None:
    """
    Write ablation registry to JSON artifact.
    
    Paper evidence contract: Expose bounded sweep/config entries for batch_size
    and ablation configurations.
    
    Args:
        output_dir: Directory to write artifact
    """
    os.makedirs(output_dir, exist_ok=True)
    
    registry_data = {
        "ablation_variants": {
            variant_id: variant.to_dict()
            for variant_id, variant in ABLATION_REGISTRY.items()
        },
        "method_aliases": METHOD_ALIASES,
        "paper_evidence_contract": {
            "priority_methods": ["ours", "sapg", "ppo", "pbt", "pql", "ddpg"],
            "figure_6_ablations": [
                "ours",
                "symmetric",
                "no_offpolicy",
                "entropy_0",
                "entropy_0005",
                "entropy_001"
            ],
        },
        "hypothesis": "The leader/follower design is a causal choice tested by ablations",
        "decision_value": "Supports decisions about which aggregation structure explains performance",
    }
    
    output_path = os.path.join(output_dir, "ablation_registry.json")
    with open(output_path, "w") as f:
        json.dump(registry_data, f, indent=2)


def write_update_source_sets(
    num_policies: int,
    variant: str,
    output_dir: str = "results"
) -> None:
    """
    Write update source sets for each policy to JSON artifact.
    
    This documents which policies contribute data to each policy's update
    under different aggregation variants.
    
    Args:
        num_policies: Number of policies (M)
        variant: Aggregation variant name
        output_dir: Directory to write artifact
    """
    os.makedirs(output_dir, exist_ok=True)
    
    config = get_aggregation_variant(variant)
    
    source_sets = {}
    for policy_idx in range(num_policies):
        sources = select_source_policies(policy_idx, variant, num_policies)
        source_sets[f"policy_{policy_idx}"] = {
            "policy_index": policy_idx,
            "source_policies": sources,
            "is_leader": policy_idx == 0 and config.use_leader,
            "aggregates_off_policy": len(sources) > 1,
        }
    
    output_data = {
        "variant": variant,
        "variant_config": config.to_dict(),
        "num_policies": num_policies,
        "source_sets": source_sets,
        "aggregation_summary": {
            "total_policies": num_policies,
            "leader_policy": 0 if config.use_leader else None,
            "symmetric": config.symmetric_aggregation,
            "uses_off_policy": config.use_off_policy,
        },
    }
    
    output_path = os.path.join(output_dir, "update_source_sets.json")
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)


def create_data_collector(
    method: str,
    num_envs: int = PAPER_NUM_ENVS,
    num_policies: int = PAPER_MAIN_POLICY_COUNT,
    rollout_length: int = PAPER_PPO_STEPS_PER_ENV_BEFORE_UPDATE,
) -> MultiPolicyDataCollector:
    """
    Factory function to create data collector for a specific method.
    
    Paper evidence contract: Complete method/baseline selector set includes
    ours, sapg, ppo, pbt, pql, ddpg.
    
    Args:
        method: Method name (ours, sapg, ppo, pbt, pql, ddpg)
        num_envs: Total number of parallel environments
        num_policies: Number of policies (M)
        rollout_length: Steps to collect before update
        
    Returns:
        Configured MultiPolicyDataCollector
    """
    # Resolve method alias
    resolved_method = METHOD_ALIASES.get(method, method)
    
    # For PPO baseline, use single policy
    if resolved_method == "ppo":
        num_policies = 1
    
    envs_per_policy = num_envs // num_policies
    
    return MultiPolicyDataCollector(
        num_policies=num_policies,
        envs_per_policy=envs_per_policy,
        aggregation_variant=resolved_method,
        rollout_length=rollout_length,
    )


# Expose registry and selection functions at module level
__all__ = [
    "ABLATION_REGISTRY",
    "METHOD_ALIASES",
    "PAPER_MAIN_POLICY_COUNT",
    "PAPER_LEADER_OFF_POLICY_LAMBDA",
    "PAPER_PPO_STEPS_PER_ENV_BEFORE_UPDATE",
    "AggregationVariant",
    "MultiPolicyDataCollector",
    "get_aggregation_variant",
    "select_source_policies",
    "write_ablation_registry",
    "write_update_source_sets",
    "create_data_collector",
]
