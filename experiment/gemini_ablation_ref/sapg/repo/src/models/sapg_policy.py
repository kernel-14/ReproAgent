# src/models/sapg_policy.py
# Reference Grounding: paper_method_core, paper_rl_multi_policy_offpolicy_aggregation, paper_contract_method_baseline_protocol
# SAPG: Split and Aggregate Policy Gradients - Core Model and Policy Implementation

import os
import json
import math
import pathlib
from typing import Dict, Any, List, Tuple, Optional, Union, Callable

# Lazy imports for heavy packages to keep the module importable in minimal environments
def _lazy_import_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.distributions import Normal
        return torch, nn, F, Normal
    except ImportError:
        return None, None, None, None

# Executable constants and sweeps
# Reference Grounding: paper_contract_sweep_hyperparameter_protocol
DEFAULT_BATCH_SIZE = 32768
batch_size_values = [8192, 16384, 32768, 65536]

DEFAULT_EPOCHS = 100
epochs_values = [50, 100, 200, 500]

DEFAULT_GAMMA = 0.99
gamma_values = [0.9, 0.95, 0.99, 0.995]

DEFAULT_LAMBDA = 1.0 # Off-policy aggregation weight lambda
lambda_values = [0.5, 1.0, 2.0]

DEFAULT_NUM_STEPS = 3 # n-step returns for critic target

def resolve_batch_size_defaults(batch_size: Optional[int] = None) -> int:
    return batch_size if batch_size is not None else DEFAULT_BATCH_SIZE

def resolve_epochs_defaults(epochs: Optional[int] = None) -> int:
    return epochs if epochs is not None else DEFAULT_EPOCHS

def resolve_gamma_defaults(gamma: Optional[float] = None) -> float:
    return gamma if gamma is not None else DEFAULT_GAMMA

def resolve_lambda_defaults(lam: Optional[float] = None) -> float:
    return lam if lam is not None else DEFAULT_LAMBDA

def resolve_num_steps_defaults(n: Optional[int] = None) -> int:
    return n if n is not None else DEFAULT_NUM_STEPS

# Method and Baseline Registry
# Reference Grounding: paper_contract_method_baseline_protocol
METHOD_REGISTRY = {
    "ours": "SAPG (Split and Aggregate Policy Gradients)",
    "sapg": "SAPG (Split and Aggregate Policy Gradients)",
    "ppo": "Proximal Policy Optimization",
    "pbt": "Population Based Training",
    "pql": "Parallel Q-Learning / Policy Q-Learning",
    "ddpg": "Deep Deterministic Policy Gradient"
}

class SAPGPolicy:
    """
    SAPG Leader-Follower Engine implementation.
    Reference Grounding: Section 4.4 Latent conditioning, Section 4.6 Algorithm 1.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.torch, self.nn, self.F, self.Normal = _lazy_import_torch()
        
        self.obs_dim = config.get("obs_dim", 67)
        self.action_dim = config.get("action_dim", 23)
        self.latent_dim = config.get("latent_dim", 32)
        self.M = config.get("num_policies", 4) # M policies: 1 leader, M-1 followers
        
        # Shared backbone B_theta and shared value function C_psi
        # Reference Grounding: Section 4.4
        if self.torch:
            self.backbone = self.nn.Sequential(
                self.nn.Linear(self.obs_dim + self.latent_dim, 256),
                self.nn.ReLU(),
                self.nn.Linear(256, 256),
                self.nn.ReLU()
            )
            self.actor_head = self.nn.Linear(256, self.action_dim)
            self.critic_head = self.nn.Linear(256, 1)
            
            # Local parameters phi_j for each policy j in {1...M}
            # These are latent vectors used to condition the shared backbone
            self.phi = self.nn.Parameter(self.torch.randn(self.M, self.latent_dim))
            self.log_std = self.nn.Parameter(self.torch.zeros(self.M, self.action_dim))
            
            # Optimization parameters
            self.gamma = resolve_gamma_defaults(config.get("gamma"))
            self.lam = resolve_lambda_defaults(config.get("lambda"))
            self.clip_param = config.get("clip_param", 0.2)
            self.entropy_coefs = config.get("sigma", [0.0] * self.M) # sigma_j for each policy

    def get_action(self, obs, policy_index: int, deterministic: bool = False):
        """Sample action from policy pi_j."""
        if not self.torch: return None
        
        latent = self.phi[policy_index].unsqueeze(0).repeat(obs.shape[0], 1)
        x = self.torch.cat([obs, latent], dim=-1)
        features = self.backbone(x)
        mu = self.actor_head(features)
        
        if deterministic:
            return mu
        
        std = self.torch.exp(self.log_std[policy_index])
        dist = self.Normal(mu, std)
        return dist.sample()

    def compute_on_policy_loss(self, batch: Dict[str, Any], policy_index: int):
        """
        Compute PPO loss L_on for policy pi_j.
        Reference Grounding: Section 3 Preliminaries, Equation for L_on.
        """
        if not self.torch: return 0.0
        
        obs = batch["obs"]
        actions = batch["actions"]
        old_log_probs = batch["log_probs"]
        advantages = batch["advantages"]
        
        latent = self.phi[policy_index].unsqueeze(0).repeat(obs.shape[0], 1)
        x = self.torch.cat([obs, latent], dim=-1)
        features = self.backbone(x)
        mu = self.actor_head(features)
        std = self.torch.exp(self.log_std[policy_index])
        dist = self.Normal(mu, std)
        
        log_probs = dist.log_prob(actions).sum(-1)
        ratio = self.torch.exp(log_probs - old_log_probs)
        
        surr1 = ratio * advantages
        surr2 = self.torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
        
        # Follower entropy regularization
        # Reference Grounding: Section 4.5
        entropy = dist.entropy().sum(-1).mean()
        entropy_loss = -self.entropy_coefs[policy_index] * entropy
        
        ppo_loss = -self.torch.min(surr1, surr2).mean()
        return ppo_loss + entropy_loss

    def compute_off_policy_loss(self, target_policy_idx: int, source_batches: List[Dict[str, Any]]):
        """
        Compute off-policy loss L_off for the leader using data from followers.
        Reference Grounding: Section 4.1, Equation for L_off.
        """
        if not self.torch: return 0.0
        
        total_loss = 0.0
        num_sources = len(source_batches)
        
        for batch in source_batches:
            obs = batch["obs"]
            actions = batch["actions"]
            old_log_probs = batch["log_probs"] # log_prob under pi_j
            advantages = batch["advantages"]
            
            # Importance weight mu = pi_i(a|s) / pi_j(a|s)
            latent = self.phi[target_policy_idx].unsqueeze(0).repeat(obs.shape[0], 1)
            x = self.torch.cat([obs, latent], dim=-1)
            features = self.backbone(x)
            mu_target = self.actor_head(features)
            std_target = self.torch.exp(self.log_std[target_policy_idx])
            dist_target = self.Normal(mu_target, std_target)
            
            log_probs_target = dist_target.log_prob(actions).sum(-1)
            importance_weight = self.torch.exp(log_probs_target - old_log_probs)
            
            # Clipped importance sampling loss
            # Reference Grounding: Section 4.1 L_off formula
            surr1 = importance_weight * advantages
            # Note: Paper mentions clipping or min with mu * r_pi_i_old
            # Here we implement the standard PPO-style clipping for off-policy data
            surr2 = self.torch.clamp(importance_weight, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
            
            total_loss += -self.torch.min(surr1, surr2).mean()
            
        return (self.lam / num_sources) * total_loss

def make_method(config: Dict[str, Any]) -> Union[SAPGPolicy, Any]:
    """
    Factory function to create policy/method instances.
    Reference Grounding: method registry.
    """
    method_name = config.get("method", "sapg").lower()
    
    if method_name in ["sapg", "ours"]:
        return SAPGPolicy(config)
    elif method_name == "ppo":
        # PPO is a special case of SAPG with M=1
        config["num_policies"] = 1
        return SAPGPolicy(config)
    elif method_name == "ddpg":
        # Placeholder for DDPG baseline adapter
        return "DDPG_Adapter"
    elif method_name == "pql":
        # Placeholder for PQL baseline adapter
        return "PQL_Adapter"
    else:
        raise ValueError(f"Unknown method: {method_name}")

def write_method_registry_artifact(output_dir: str):
    """Write the method registry to a JSON file."""
    path = pathlib.Path(output_dir) / "method_registry.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(METHOD_REGISTRY, f, indent=2)

def write_config_resolved_artifact(config: Dict[str, Any], output_dir: str):
    """Write the resolved configuration to a JSON file."""
    path = pathlib.Path(output_dir) / "config_resolved.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    
    resolved = config.copy()
    resolved["batch_size"] = resolve_batch_size_defaults(config.get("batch_size"))
    resolved["epochs"] = resolve_epochs_defaults(config.get("epochs"))
    resolved["gamma"] = resolve_gamma_defaults(config.get("gamma"))
    resolved["lambda"] = resolve_lambda_defaults(config.get("lambda"))
    
    with open(path, "w") as f:
        json.dump(resolved, f, indent=2)

def write_ablation_registry_artifact(output_dir: str):
    """Write the ablation variants to a JSON file."""
    path = pathlib.Path(output_dir) / "ablation_registry.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    
    ablations = {
        "sapg_no_entropy": {"sigma": [0.0, 0.0, 0.0, 0.0]},
        "sapg_high_off_policy": {"lambda": 2.0},
        "sapg_low_off_policy": {"lambda": 0.5}
    }
    
    with open(path, "w") as f:
        json.dump(ablations, f, indent=2)

def write_sensitivity_report_artifact(results: Dict[str, Any], output_dir: str):
    """Write a sensitivity report for parameter sweeps."""
    path = pathlib.Path(output_dir) / "sensitivity_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)

def write_update_traces_artifact(traces: List[Any], output_dir: str):
    """Write training update traces (losses, weights)."""
    path = pathlib.Path(output_dir) / "update_traces.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(traces, f, indent=2)

if __name__ == "__main__":
    # Smoke test for policy initialization and artifact writing
    artifact_dir = os.environ.get('PAPERBENCH_REPRO_ARTIFACT_DIR', 'results')
    os.makedirs(artifact_dir, exist_ok=True)
    
    test_config = {
        "method": "sapg",
        "obs_dim": 67,
        "action_dim": 23,
        "num_policies": 4,
        "batch_size": 32768,
        "epochs": 10
    }
    
    policy = make_method(test_config)
    print(f"Initialized {METHOD_REGISTRY['sapg']} policy.")
    
    write_method_registry_artifact(artifact_dir)
    write_config_resolved_artifact(test_config, artifact_dir)
    write_ablation_registry_artifact(artifact_dir)
    
    # Create readiness manifest
    with open(os.path.join(artifact_dir, "readiness.json"), "w") as f:
        json.dump({"status": "ready", "file": "src/models/sapg_policy.py"}, f)