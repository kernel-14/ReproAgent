# src/sweep_registry.py
# SAPG: Split and Aggregate Policy Gradients - Sweep and Method Registry
# reference_grounding: wp_005 src/sweep_registry.py
#
# Paper evidence contract: expose bounded sweep/config entries for batch_size
# (Figure 2 shows PPO performance vs batch size saturation)
#
# Binding addendum clarification (Figure 6): The blue plot is SAPG, and the other
# ones are ablations. For instance, symmetric aggregation refers to a version of
# SAPG without any one designated leader, each worker is updated with all the
# off-policy data from all other workers in a symmetric fashion.
#
# Method/baseline registry entries: ours, sapg, ppo, pbt, pql, baseline
# Ablation registry entries: symmetric_aggregation, no_offpolicy, entropy_0,
# entropy_0005, entropy_001, offpolicy_ratio_high, offpolicy_ratio_low
#
# Artifacts: results/method_registry.json, results/ablation_registry.json

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import copy


PAPER_FIGURE2_PPO_BATCH_SIZES = [1500, 3125, 6250, 12500, 25000, 50000, 100000]
PAPER_MAIN_POLICY_COUNT = 6


# ---------------------------------------------------------------------------
# Method Registry - Paper-derived methods and baselines
# ---------------------------------------------------------------------------

METHOD_REGISTRY = {
    "sapg": {
        "method_id": "sapg",
        "aliases": ["ours", "Ours", "OURS", "SAPG"],
        "display_name": "SAPG (Ours)",
        "method_type": "proposed",
        "description": "Split and Aggregate Policy Gradients - proposed method with leader-follower architecture",
        "paper_reference": "Algorithm 1, Figure 3, Table 1",
        "config": {
            "algorithm": "sapg",
            "num_policies": PAPER_MAIN_POLICY_COUNT,
            "aggregation_scheme": "leader_follower",
            "aggregation_coefficient": 1.0,
            "importance_sampling_clip": 1.0,
            "shared_backbone": True,
            "local_parameters": True,
            "entropy_coefficient": 0.01,
            "clip_range": 0.2,
            "value_clip_range": 0.2,
            "gae_lambda": 0.95,
            "gamma": 0.99,
            "learning_rate": 3e-4,
            "max_grad_norm": 0.5,
            "num_epochs": 5,
            "batch_size": 32768,
            "minibatch_size": 4096,
        },
        "environments": ["ShadowHandOver", "ShadowHandCatchUnderarm", "ShadowHandCatchAbreast",
                        "ShadowHandReOrientation", "AllegroHandReOrientation", "AllegroKuka"],
        "metrics": ["success_rate", "episode_reward", "state_coverage"],
        "artifacts": ["Figure 5", "Table 1", "Figure 7", "Figure 8"],
    },
    
    "ppo": {
        "method_id": "ppo",
        "aliases": ["PPO", "baseline"],
        "display_name": "PPO",
        "method_type": "baseline",
        "description": "Proximal Policy Optimization - single policy baseline",
        "paper_reference": "Figure 2, Figure 5, Table 1",
        "config": {
            "algorithm": "ppo",
            "num_policies": 1,
            "aggregation_scheme": "none",
            "aggregation_coefficient": 0.0,
            "importance_sampling_clip": 1.0,
            "shared_backbone": False,
            "local_parameters": False,
            "entropy_coefficient": 0.01,
            "clip_range": 0.2,
            "value_clip_range": 0.2,
            "gae_lambda": 0.95,
            "gamma": 0.99,
            "learning_rate": 3e-4,
            "max_grad_norm": 0.5,
            "num_epochs": 5,
            "batch_size": 32768,
            "minibatch_size": 4096,
        },
        "environments": ["ShadowHandOver", "ShadowHandCatchUnderarm", "ShadowHandCatchAbreast",
                        "ShadowHandReOrientation", "AllegroHandReOrientation", "AllegroKuka"],
        "metrics": ["success_rate", "episode_reward", "state_coverage"],
        "artifacts": ["Figure 2", "Figure 5", "Table 1", "Figure 7", "Figure 8"],
    },
    
    "pbt": {
        "method_id": "pbt",
        "aliases": ["PBT"],
        "display_name": "PBT",
        "method_type": "baseline",
        "description": "Population Based Training - evolutionary hyperparameter optimization baseline",
        "paper_reference": "Figure 5, Table 1",
        "config": {
            "algorithm": "pbt",
            "num_policies": PAPER_MAIN_POLICY_COUNT,
            "aggregation_scheme": "population",
            "aggregation_coefficient": 0.0,
            "importance_sampling_clip": 1.0,
            "shared_backbone": False,
            "local_parameters": True,
            "entropy_coefficient": 0.01,
            "clip_range": 0.2,
            "value_clip_range": 0.2,
            "gae_lambda": 0.95,
            "gamma": 0.99,
            "learning_rate": 3e-4,
            "max_grad_norm": 0.5,
            "num_epochs": 5,
            "batch_size": 32768,
            "minibatch_size": 4096,
            "population_size": 8,
            "exploit_interval": 1000,
            "explore_noise": 0.2,
        },
        "environments": ["ShadowHandOver", "ShadowHandCatchUnderarm", "ShadowHandCatchAbreast",
                        "ShadowHandReOrientation", "AllegroHandReOrientation", "AllegroKuka"],
        "metrics": ["success_rate", "episode_reward"],
        "artifacts": ["Figure 5", "Table 1"],
    },
    
    "pql": {
        "method_id": "pql",
        "aliases": ["PQL"],
        "display_name": "PQL",
        "method_type": "baseline",
        "description": "Policy Gradient with Q-Learning - hybrid on/off-policy baseline",
        "paper_reference": "Figure 5, Table 1",
        "config": {
            "algorithm": "pql",
            "num_policies": 1,
            "aggregation_scheme": "none",
            "aggregation_coefficient": 0.0,
            "importance_sampling_clip": 1.0,
            "shared_backbone": False,
            "local_parameters": False,
            "entropy_coefficient": 0.01,
            "clip_range": 0.2,
            "value_clip_range": 0.2,
            "gae_lambda": 0.95,
            "gamma": 0.99,
            "learning_rate": 3e-4,
            "max_grad_norm": 0.5,
            "num_epochs": 5,
            "batch_size": 32768,
            "minibatch_size": 4096,
            "q_learning_weight": 0.5,
        },
        "environments": ["ShadowHandOver", "ShadowHandCatchUnderarm", "ShadowHandCatchAbreast",
                        "ShadowHandReOrientation", "AllegroHandReOrientation", "AllegroKuka"],
        "metrics": ["success_rate", "episode_reward"],
        "artifacts": ["Figure 5", "Table 1"],
    },
}


# ---------------------------------------------------------------------------
# Ablation Registry - Figure 6 ablations
# ---------------------------------------------------------------------------

ABLATION_REGISTRY = {
    "symmetric_aggregation": {
        "ablation_id": "symmetric_aggregation",
        "aliases": ["symmetric", "no_leader"],
        "display_name": "Symmetric Aggregation",
        "ablation_type": "architecture",
        "description": "SAPG without designated leader - each worker updated with all off-policy data symmetrically",
        "paper_reference": "Figure 6",
        "base_method": "sapg",
        "config_override": {
            "aggregation_scheme": "symmetric",
            "aggregation_coefficient": 0.5,
        },
        "expected_behavior": "Significantly worse performance than SAPG with leader",
        "artifacts": ["Figure 6"],
    },
    
    "no_offpolicy": {
        "ablation_id": "no_offpolicy",
        "aliases": ["no_aggregation", "on_policy_only"],
        "display_name": "No Off-Policy",
        "ablation_type": "data_aggregation",
        "description": "SAPG without off-policy data aggregation - only on-policy data used",
        "paper_reference": "Figure 6",
        "base_method": "sapg",
        "config_override": {
            "aggregation_coefficient": 0.0,
        },
        "expected_behavior": "Significantly worse performance than SAPG with off-policy aggregation",
        "artifacts": ["Figure 6"],
    },
    
    "entropy_0": {
        "ablation_id": "entropy_0",
        "aliases": ["COEF=0", "no_entropy"],
        "display_name": "Entropy Coefficient = 0",
        "ablation_type": "hyperparameter",
        "description": "SAPG with zero entropy regularization",
        "paper_reference": "Figure 6",
        "base_method": "sapg",
        "config_override": {
            "entropy_coefficient": 0.0,
        },
        "expected_behavior": "Task-dependent performance - worse on some tasks, similar on others",
        "artifacts": ["Figure 6"],
    },
    
    "entropy_0005": {
        "ablation_id": "entropy_0005",
        "aliases": ["COEF=0.005"],
        "display_name": "Entropy Coefficient = 0.005",
        "ablation_type": "hyperparameter",
        "description": "SAPG with entropy coefficient 0.005",
        "paper_reference": "Figure 6, Table 1",
        "base_method": "sapg",
        "config_override": {
            "entropy_coefficient": 0.005,
        },
        "expected_behavior": "Best performance on Shadow Hand and Allegro Kuka Reorientation",
        "artifacts": ["Figure 6"],
    },
    
    "entropy_001": {
        "ablation_id": "entropy_001",
        "aliases": ["COEF=0.01"],
        "display_name": "Entropy Coefficient = 0.01",
        "ablation_type": "hyperparameter",
        "description": "SAPG with entropy coefficient 0.01 (default)",
        "paper_reference": "Figure 6",
        "base_method": "sapg",
        "config_override": {
            "entropy_coefficient": 0.01,
        },
        "expected_behavior": "Default configuration - good performance across tasks",
        "artifacts": ["Figure 6"],
    },
    
    "offpolicy_ratio_high": {
        "ablation_id": "offpolicy_ratio_high",
        "aliases": ["high_aggregation"],
        "display_name": "High Off-Policy Ratio",
        "ablation_type": "data_aggregation",
        "description": "SAPG with high off-policy aggregation coefficient (0.8)",
        "paper_reference": "Figure 6",
        "base_method": "sapg",
        "config_override": {
            "aggregation_coefficient": 0.8,
        },
        "expected_behavior": "Worse performance - too much off-policy data degrades learning",
        "artifacts": ["Figure 6"],
    },
    
    "offpolicy_ratio_low": {
        "ablation_id": "offpolicy_ratio_low",
        "aliases": ["low_aggregation"],
        "display_name": "Low Off-Policy Ratio",
        "ablation_type": "data_aggregation",
        "description": "SAPG with low off-policy aggregation coefficient (0.2)",
        "paper_reference": "Figure 6",
        "base_method": "sapg",
        "config_override": {
            "aggregation_coefficient": 0.2,
        },
        "expected_behavior": "Moderate performance - less benefit from off-policy data",
        "artifacts": ["Figure 6"],
    },
}


# ---------------------------------------------------------------------------
# Sweep Registry - Paper evidence contract: batch_size sweeps (Figure 2)
# ---------------------------------------------------------------------------

SWEEP_REGISTRY = {
    "batch_size": {
        "sweep_id": "batch_size",
        "display_name": "Batch Size Sweep",
        "description": "Sweep over batch sizes to demonstrate PPO saturation (Figure 2)",
        "paper_reference": "Figure 2",
        "parameter": "batch_size",
            "values": PAPER_FIGURE2_PPO_BATCH_SIZES,
            "methods": ["ppo", "sapg"],
            "environments": ["ShadowHandOver", "AllegroKukaThrow"],
        "expected_behavior": "PPO saturates at large batch sizes, SAPG continues to improve",
        "artifacts": ["Figure 2"],
    },
    
    "num_policies": {
        "sweep_id": "num_policies",
        "display_name": "Number of Policies (M)",
        "description": "Sweep over number of policies in SAPG",
        "paper_reference": "Paper SAPG/DexPBT/PBT main policy count is M=6",
        "parameter": "num_policies",
        "values": [1, 2, 4, 6, 8, 16],
        "methods": ["sapg"],
        "environments": ["ShadowHandOver"],
        "expected_behavior": "Performance improves with multiple policies; main comparison uses M=6",
        "artifacts": [],
    },
    
    "aggregation_coefficient": {
        "sweep_id": "aggregation_coefficient",
        "display_name": "Off-Policy Aggregation Coefficient",
        "description": "Sweep over aggregation coefficient lambda (Figure 6)",
        "paper_reference": "Figure 6",
        "parameter": "aggregation_coefficient",
        "values": [0.0, 0.2, 0.5, 0.8, 1.0],
        "methods": ["sapg"],
        "environments": ["ShadowHandOver"],
        "expected_behavior": "Leader-follower SAPG update exposes lambda=1.0 and degrades at extremes",
        "artifacts": ["Figure 6"],
    },
    
    "entropy_coefficient": {
        "sweep_id": "entropy_coefficient",
        "display_name": "Entropy Coefficient",
        "description": "Sweep over entropy regularization (Figure 6)",
        "paper_reference": "Figure 6, Table 1",
        "parameter": "entropy_coefficient",
        "values": [0.0, 0.005, 0.01, 0.02],
        "methods": ["sapg"],
        "environments": ["ShadowHandOver", "ShadowHandReOrientation", "AllegroKuka"],
        "expected_behavior": "Task-dependent - 0.005 best for reorientation, 0 for others",
        "artifacts": ["Figure 6"],
    },
}


# ---------------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------------

def make_method(method_id: str, config_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Factory function to create method configuration.
    
    Args:
        method_id: Method identifier (sapg, ppo, pbt, pql) or alias
        config_override: Optional config overrides
        
    Returns:
        Complete method configuration dictionary
    """
    # Resolve aliases
    resolved_id = resolve_method_alias(method_id)
    
    if resolved_id not in METHOD_REGISTRY:
        raise ValueError(f"Unknown method: {method_id} (resolved to {resolved_id})")
    
    # Get base config
    method_config = copy.deepcopy(METHOD_REGISTRY[resolved_id])
    
    # Apply overrides
    if config_override:
        method_config["config"].update(config_override)
    
    return method_config


def make_ablation(ablation_id: str, config_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Factory function to create ablation configuration.
    
    Args:
        ablation_id: Ablation identifier or alias
        config_override: Optional config overrides
        
    Returns:
        Complete ablation configuration dictionary
    """
    # Resolve aliases
    resolved_id = resolve_ablation_alias(ablation_id)
    
    if resolved_id not in ABLATION_REGISTRY:
        raise ValueError(f"Unknown ablation: {ablation_id} (resolved to {resolved_id})")
    
    # Get base method config
    ablation_config = copy.deepcopy(ABLATION_REGISTRY[resolved_id])
    base_method = ablation_config["base_method"]
    method_config = make_method(base_method)
    
    # Apply ablation overrides
    method_config["config"].update(ablation_config["config_override"])
    
    # Apply additional overrides
    if config_override:
        method_config["config"].update(config_override)
    
    # Add ablation metadata
    method_config["ablation_id"] = ablation_id
    method_config["ablation_type"] = ablation_config["ablation_type"]
    method_config["display_name"] = ablation_config["display_name"]
    
    return method_config


def resolve_method_alias(method_id: str) -> str:
    """Resolve method alias to canonical method ID."""
    for canonical_id, method_data in METHOD_REGISTRY.items():
        if method_id == canonical_id or method_id in method_data.get("aliases", []):
            return canonical_id
    return method_id


def resolve_ablation_alias(ablation_id: str) -> str:
    """Resolve ablation alias to canonical ablation ID."""
    for canonical_id, ablation_data in ABLATION_REGISTRY.items():
        if ablation_id == canonical_id or ablation_id in ablation_data.get("aliases", []):
            return canonical_id
    return ablation_id


def get_sweep_config(sweep_id: str) -> Dict[str, Any]:
    """Get sweep configuration by ID."""
    if sweep_id not in SWEEP_REGISTRY:
        raise ValueError(f"Unknown sweep: {sweep_id}")
    return copy.deepcopy(SWEEP_REGISTRY[sweep_id])


def list_methods() -> List[str]:
    """List all available method IDs."""
    return list(METHOD_REGISTRY.keys())


def list_ablations() -> List[str]:
    """List all available ablation IDs."""
    return list(ABLATION_REGISTRY.keys())


def list_sweeps() -> List[str]:
    """List all available sweep IDs."""
    return list(SWEEP_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Artifact Writers
# ---------------------------------------------------------------------------

def write_method_registry(output_dir: str = "results") -> None:
    """Write method registry to JSON artifact."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "method_registry.json")
    
    registry_data = {
        "registry_type": "method",
        "paper_reference": "SAPG: Split and Aggregate Policy Gradients",
        "methods": METHOD_REGISTRY,
        "method_count": len(METHOD_REGISTRY),
        "method_ids": list(METHOD_REGISTRY.keys()),
        "baseline_ids": [m for m, d in METHOD_REGISTRY.items() if d["method_type"] == "baseline"],
        "proposed_ids": [m for m, d in METHOD_REGISTRY.items() if d["method_type"] == "proposed"],
    }
    
    with open(output_path, "w") as f:
        json.dump(registry_data, f, indent=2)


def write_ablation_registry(output_dir: str = "results") -> None:
    """Write ablation registry to JSON artifact."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "ablation_registry.json")
    
    registry_data = {
        "registry_type": "ablation",
        "paper_reference": "Figure 6 - Ablation Study",
        "ablations": ABLATION_REGISTRY,
        "ablation_count": len(ABLATION_REGISTRY),
        "ablation_ids": list(ABLATION_REGISTRY.keys()),
        "ablation_types": list(set(d["ablation_type"] for d in ABLATION_REGISTRY.values())),
    }
    
    with open(output_path, "w") as f:
        json.dump(registry_data, f, indent=2)


def write_sweep_registry(output_dir: str = "results") -> None:
    """Write sweep registry to JSON artifact."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "sweep_registry.json")
    
    registry_data = {
        "registry_type": "sweep",
        "paper_reference": "Figure 2, Figure 6",
        "sweeps": SWEEP_REGISTRY,
        "sweep_count": len(SWEEP_REGISTRY),
        "sweep_ids": list(SWEEP_REGISTRY.keys()),
    }
    
    with open(output_path, "w") as f:
        json.dump(registry_data, f, indent=2)


def write_all_registries(output_dir: str = "results") -> None:
    """Write all registry artifacts."""
    write_method_registry(output_dir)
    write_ablation_registry(output_dir)
    write_sweep_registry(output_dir)


# ---------------------------------------------------------------------------
# Main - Write artifacts when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    
    # Determine output directory
    output_dir = os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR", "results")
    
    # Write all registry artifacts
    write_all_registries(output_dir)
    
    print(f"Method registry written to {output_dir}/method_registry.json")
    print(f"Ablation registry written to {output_dir}/ablation_registry.json")
    print(f"Sweep registry written to {output_dir}/sweep_registry.json")
    
    # Print summary
    print(f"\nRegistry Summary:")
    print(f"  Methods: {len(METHOD_REGISTRY)} ({', '.join(METHOD_REGISTRY.keys())})")
    print(f"  Ablations: {len(ABLATION_REGISTRY)} ({', '.join(ABLATION_REGISTRY.keys())})")
    print(f"  Sweeps: {len(SWEEP_REGISTRY)} ({', '.join(SWEEP_REGISTRY.keys())})")
