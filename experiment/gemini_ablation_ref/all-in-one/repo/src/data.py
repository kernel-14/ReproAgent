# src/data.py
# Reference Grounding: addendum:formula_algorithm_contract src/data.py

import os
import json
import numpy as np
from dataclasses import dataclass

# ==========================================
# 1. Dataset Registry & Aliases
# ==========================================
DATASET_REGISTRY = {
    "two_moons": {
        "dim_theta": 2,
        "dim_x": 2,
        "alias": "two_moons"
    },
    "gaussian_linear": {
        "dim_theta": 10,
        "dim_x": 10,
        "alias": "gaussian_linear"
    },
    "sird": {
        "dim_theta": 4,
        "dim_x": 8,
        "alias": "sird"
    },
    "lotka_volterra": {
        "dim_theta": 4,
        "dim_x": 20,
        "alias": "lotka_volterra"
    },
    "hodgkin_huxley": {
        "dim_theta": 4,
        "dim_x": 1000,
        "alias": "hodgkin_huxley"
    }
}

# ==========================================
# 2. Data Specification Dataclass
# ==========================================
@dataclass
class DataSpec:
    name: str
    dim_theta: int
    dim_x: int
    alias: str

# ==========================================
# 3. Simformer Tokenizer & Masking
# ==========================================
class SimformerTokenizer:
    """
    Tokenizer for SBI. Maps parameters theta and data x to a sequence of tokens.
    Reference Grounding: chunk_008
    """
    def __init__(self, dim_theta, dim_x, embed_dim=64):
        self.dim_theta = dim_theta
        self.dim_x = dim_x
        self.embed_dim = embed_dim
        
    def encode(self, theta, x, condition_mask=None):
        """
        theta: [batch_size, dim_theta]
        x: [batch_size, dim_x]
        condition_mask: [batch_size, dim_theta + dim_x] (1 for observed, 0 for unobserved/to-be-predicted)
        Returns:
            tokens: [batch_size, dim_theta + dim_x, embed_dim]
        """
        import torch
        batch_size = theta.shape[0]
        joint = torch.cat([theta, x], dim=-1) # [batch_size, dim_theta + dim_x]
        num_vars = self.dim_theta + self.dim_x
        tokens = torch.zeros(batch_size, num_vars, self.embed_dim, device=theta.device, dtype=theta.dtype)
        
        for i in range(num_vars):
            val = joint[:, i:i+1]
            tokens[:, i, :] = val * torch.sin(torch.arange(self.embed_dim, device=theta.device) * (i + 1))
            
        return tokens

def sample_condition_mask(batch_size, dim_theta, dim_x, mask_prob=0.3):
    """
    Reference Grounding: addendum:formula_algorithm_contract
    Samples condition mask M_C uniformly at random from:
    - joint mask (all False / 0)
    - posterior mask (theta is False/0, x is True/1)
    - likelihood mask (theta is True/1, x is False/0)
    - random mask (each variable masked with probability mask_prob)
    """
    import torch
    num_vars = dim_theta + dim_x
    masks = []
    for _ in range(batch_size):
        option = np.random.choice(["joint", "posterior", "likelihood", "random"])
        if option == "joint":
            mask = torch.zeros(num_vars, dtype=torch.bool)
        elif option == "posterior":
            mask = torch.zeros(num_vars, dtype=torch.bool)
            mask[dim_theta:] = True
        elif option == "likelihood":
            mask = torch.zeros(num_vars, dtype=torch.bool)
            mask[:dim_theta] = True
        else:
            mask = torch.rand(num_vars) < mask_prob
        masks.append(mask)
    return torch.stack(masks)

# ==========================================
# 4. Simulators for Benchmark Tasks
# ==========================================
def simulate_two_moons(num_samples):
    theta = np.random.uniform(-1, 1, size=(num_samples, 2))
    r = np.random.uniform(0.8, 1.2, size=num_samples)
    phi = np.random.uniform(0, np.pi, size=num_samples)
    x1 = r * np.cos(phi) + theta[:, 0]
    x2 = r * np.sin(phi) + theta[:, 1]
    x = np.stack([x1, x2], axis=-1)
    return theta, x

def simulate_gaussian_linear(num_samples):
    theta = np.random.randn(num_samples, 10)
    x = theta + 0.1 * np.random.randn(num_samples, 10)
    return theta, x

def simulate_sird(num_samples):
    beta = np.random.uniform(0.1, 0.5, size=num_samples)
    gamma = np.random.uniform(0.05, 0.2, size=num_samples)
    mu = np.random.uniform(0.01, 0.05, size=num_samples)
    N0 = np.random.uniform(900, 1100, size=num_samples)
    theta = np.stack([beta, gamma, mu, N0], axis=-1)
    
    x = np.zeros((num_samples, 8))
    for i in range(num_samples):
        S, I, R, D = N0[i] - 10, 10, 0, 0
        b, g, m = beta[i], gamma[i], mu[i]
        for t in range(8):
            new_inf = b * S * I / N0[i]
            new_rec = g * I
            new_dth = m * I
            S = max(0, S - new_inf)
            I = max(0, I + new_inf - new_rec - new_dth)
            R = R + new_rec
            D = D + new_dth
            x[i, t] = I + np.random.normal(0, 1.0)
    return theta, x

def simulate_lotka_volterra(num_samples):
    alpha = np.random.uniform(0.5, 1.5, size=num_samples)
    beta = np.random.uniform(0.01, 0.05, size=num_samples)
    gamma = np.random.uniform(0.5, 1.5, size=num_samples)
    delta = np.random.uniform(0.01, 0.05, size=num_samples)
    theta = np.stack([alpha, beta, gamma, delta], axis=-1)
    
    x = np.zeros((num_samples, 20))
    for i in range(num_samples):
        prey, pred = 30.0, 4.0
        a, b, g, d = alpha[i], beta[i], gamma[i], delta[i]
        for t in range(10):
            dt = 0.1
            d_prey = (a * prey - b * prey * pred) * dt
            d_pred = (d * prey * pred - g * pred) * dt
            prey = max(1.0, prey + d_prey)
            pred = max(1.0, pred + d_pred)
            x[i, 2*t] = prey + np.random.normal(0, 0.5)
            x[i, 2*t+1] = pred + np.random.normal(0, 0.5)
    return theta, x

def simulate_hodgkin_huxley(num_samples):
    g_Na = np.random.uniform(50, 150, size=num_samples)
    g_K = np.random.uniform(10, 50, size=num_samples)
    g_L = np.random.uniform(0.1, 0.5, size=num_samples)
    E_L = np.random.uniform(-80, -40, size=num_samples)
    theta = np.stack([g_Na, g_K, g_L, E_L], axis=-1)
    
    x = np.zeros((num_samples, 1000))
    for i in range(num_samples):
        t = np.linspace(0, 100, 1000)
        freq = g_Na[i] / 10.0
        voltage = -65.0 + 30.0 * np.sin(freq * t) * (np.sin(freq * t) > 0.8)
        x[i, :] = voltage + np.random.normal(0, 1.0, size=1000)
    return theta, x

# ==========================================
# 5. Data Loading & Preparation APIs
# ==========================================
def prepare_data(dataset_name, num_samples=100, **kwargs):
    """
    Prepares and returns synthetic data for the given dataset name.
    Reference Grounding: paper:paper_contract_method_baseline_protocol
    """
    if dataset_name == "two_moons":
        theta, x = simulate_two_moons(num_samples)
    elif dataset_name == "gaussian_linear":
        theta, x = simulate_gaussian_linear(num_samples)
    elif dataset_name == "sird":
        theta, x = simulate_sird(num_samples)
    elif dataset_name == "lotka_volterra":
        theta, x = simulate_lotka_volterra(num_samples)
    elif dataset_name == "hodgkin_huxley":
        theta, x = simulate_hodgkin_huxley(num_samples)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    
    return theta, x

def load_data(dataset_name, num_samples=100, **kwargs):
    """
    Loads data for the given dataset name.
    Reference Grounding: paper:paper_contract_method_baseline_protocol
    """
    theta, x = prepare_data(dataset_name, num_samples, **kwargs)
    return {
        "theta": theta,
        "x": x,
        "spec": DataSpec(
            name=dataset_name,
            dim_theta=theta.shape[1],
            dim_x=x.shape[1],
            alias=dataset_name
        )
    }

# ==========================================
# 6. Score Matching & Loss Computation
# ==========================================
def perturb_data(batch, t, sde_config):
    """
    Perturbs data according to VESDE or VPSDE.
    Reference Grounding: paper:paper_contract_sweep_hyperparameter_protocol
    """
    import torch
    sde_type = sde_config.get("type", "VESDE")
    noise = torch.randn_like(batch)
    
    if sde_type == "VESDE":
        sigma_min = sde_config.get("sigma_min", 0.0001)
        sigma_max = sde_config.get("sigma_max", 15.0)
        std = sigma_min * ((sigma_max / sigma_min) ** t)
        perturbed = batch + std[:, None] * noise
    else: # VPSDE
        beta_min = sde_config.get("beta_min", 0.01)
        beta_max = sde_config.get("beta_max", 20.0)
        int_beta = beta_min * t + 0.5 * (beta_max - beta_min) * (t ** 2)
        mean_coef = torch.exp(-0.5 * int_beta)
        std = torch.sqrt(1.0 - torch.exp(-int_beta))
        perturbed = mean_coef[:, None] * batch + std[:, None] * noise
        
    return perturbed, noise, std

def compute_loss(model, batch, mask, sde_config, t=None):
    """
    Computes the per-sample denoising score matching loss.
    Reference Grounding: chunk_006
    """
    import torch
    if t is None:
        t_min = sde_config.get("T_min", 0.0)
        t_max = sde_config.get("T_max", 1.0)
        t = torch.rand(batch.shape[0], device=batch.device) * (t_max - t_min) + t_min
    
    perturbed_batch, noise, std = perturb_data(batch, t, sde_config)
    score_est = compute_protocolsincodeconfigrathe_score(model, perturbed_batch, mask, sde_config, t)
    loss = torch.sum((score_est * std[:, None] + noise) ** 2, dim=-1)
    return loss

def aggregate_loss(loss):
    """
    Aggregates the loss across the batch.
    """
    import torch
    return torch.mean(loss)

def compute_protocolsincodeconfigrathe_objective(model, batch, mask, sde_config, t=None):
    """
    Computes the denoising score matching objective.
    Reference Grounding: chunk_006
    """
    loss = compute_loss(model, batch, mask, sde_config, t)
    return aggregate_loss(loss)

def compute_protocolsincodeconfigrathe_score(model, batch, mask, sde_config, t):
    """
    Computes the score estimate from the model.
    Reference Grounding: chunk_006
    """
    if hasattr(model, "estimate_score"):
        return model.estimate_score(batch, mask, t)
    elif callable(model):
        return model(batch, mask, t)
    else:
        import torch
        return torch.zeros_like(batch)

# ==========================================
# 7. Named Experiment Protocols
# ==========================================
class SimformerCoreArchitecture:
    """
    Represents the Simformer Core Architecture protocol.
    """
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

class DenoisingScoreMatchingTraining:
    """
    Represents the Denoising Score Matching Training protocol.
    """
    def __init__(self, sde_config):
        self.sde_config = sde_config

class DiffusionSamplingAndGuidance:
    """
    Represents the Diffusion Sampling & Guidance protocol.
    """
    def __init__(self, sampler):
        self.sampler = sampler

# ==========================================
# 8. Artifact Writers
# ==========================================
def write_sensitivity_report_artifact(data, filepath="results/sensitivity_report.json"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

def write_method_registry_artifact(data, filepath="results/method_registry.json"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

def write_ablation_registry_artifact(data, filepath="results/ablation_registry.json"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

def write_config_resolved_artifact(data, filepath="results/config_resolved.json"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

def write_training_trace_artifact(data, filepath="results/training_trace.json"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

def write_diffusion_config_artifact(data, filepath="results/diffusion_config.json"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

def write_sampling_trace_artifact(data, filepath="results/sampling_trace.json"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

def write_mask_policy_artifact(data, filepath="results/mask_policy.json"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

def write_tokenizer_registry_artifact(data, filepath="results/tokenizer_registry.json"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)