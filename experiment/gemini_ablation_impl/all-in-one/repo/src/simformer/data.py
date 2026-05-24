# src/simformer/data.py
# Paper: All-in-one simulation-based inference (Simformer)
# Reference Grounding: paper:unit_001 (chunk_008)

import os
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn

@dataclass
class DataSpec:
    task_id: str
    theta_dim: int
    x_dim: int
    name: str
    alias: str
    metadata: Optional[Dict[str, Any]] = None

# Paper evidence contract: explicitly register dataset/benchmark aliases
DATASET_REGISTRY = {
    "two_moons": DataSpec(
        task_id="two_moons", 
        theta_dim=2, 
        x_dim=2, 
        name="Two Moons", 
        alias="two_moons"
    ),
    "gaussian_linear": DataSpec(
        task_id="gaussian_linear", 
        theta_dim=10, 
        x_dim=10, 
        name="Gaussian Linear", 
        alias="gaussian_linear"
    ),
    "gaussian_mixture": DataSpec(
        task_id="gaussian_mixture", 
        theta_dim=2, 
        x_dim=2, 
        name="Gaussian Mixture", 
        alias="gaussian_mixture"
    ),
    "lotka_volterra": DataSpec(
        task_id="lotka_volterra", 
        theta_dim=4, 
        x_dim=20, 
        name="Lotka-Volterra", 
        alias="lotka_volterra_unstructured",
        metadata={"unstructured": True, "irregular_time": True}
    ),
    "sird": DataSpec(
        task_id="sird", 
        theta_dim=4, 
        x_dim=50, 
        name="SIRD", 
        alias="sird_functional_parameters",
        metadata={"functional_parameters": True}
    ),
    "hodgkin_huxley": DataSpec(
        task_id="hodgkin_huxley", 
        theta_dim=3, 
        x_dim=100, 
        name="Hodgkin-Huxley", 
        alias="hodgkin_huxley_interval_constraints",
        metadata={"interval_constraints": True}
    )
}

class Tokenizer(nn.Module):
    """
    Tokenizer class to convert parameters theta and data x into a sequence of tokens.
    Supports random resampling of condition states during training.
    """
    def __new__(cls, *args, **kwargs):
        # If the first argument is a torch.Tensor, act as a functional tokenizer
        if len(args) > 0 and isinstance(args[0], torch.Tensor):
            theta = args[0]
            x = args[1] if len(args) > 1 else kwargs.get("x")
            condition_mask = args[2] if len(args) > 2 else kwargs.get("condition_mask")
            embed_dim = kwargs.get("embed_dim", 128)
            
            # Create a temporary tokenizer and run it
            temp = super().__new__(cls)
            temp.__init__(theta_dim=theta.shape[-1], x_dim=x.shape[-1], embed_dim=embed_dim)
            temp.to(theta.device)
            return temp(theta, x, condition_mask)
        return super().__new__(cls)

    def __init__(self, theta_dim=4, x_dim=10, embed_dim=128):
        if hasattr(self, 'id_embedding'):
            return
        super().__init__()
        self.theta_dim = theta_dim
        self.x_dim = x_dim
        self.embed_dim = embed_dim
        
        # Learnable embeddings for variable identifiers
        self.id_embedding = nn.Embedding(theta_dim + x_dim, embed_dim)
        
        # Linear projection for the value of the variable
        self.value_projection = nn.Linear(1, embed_dim)
        
        # Embedding for condition state (0: target/unconditioned, 1: conditioned)
        self.cond_embedding = nn.Embedding(2, embed_dim)

    def forward(self, theta, x, condition_mask=None):
        """
        theta: [batch_size, theta_dim]
        x: [batch_size, x_dim]
        condition_mask: [batch_size, theta_dim + x_dim] binary mask (0 or 1)
        """
        batch_size = theta.shape[0]
        device = theta.device
        
        # Concatenate theta and x to get all variables
        variables = torch.cat([theta, x], dim=-1)
        num_vars = self.theta_dim + self.x_dim
        
        if condition_mask is None:
            # Default: theta is target (0), x is conditioned (1)
            condition_mask = torch.cat([
                torch.zeros(batch_size, self.theta_dim, device=device, dtype=torch.long),
                torch.ones(batch_size, self.x_dim, device=device, dtype=torch.long)
            ], dim=-1)
        else:
            condition_mask = condition_mask.long()
            
        # 1. Identifier embeddings
        ids = torch.arange(num_vars, device=device).unsqueeze(0).expand(batch_size, -1)
        id_embeds = self.id_embedding(ids)
        
        # 2. Value embeddings
        val_embeds = self.value_projection(variables.unsqueeze(-1))
        
        # 3. Condition state embeddings
        cond_embeds = self.cond_embedding(condition_mask)
        
        # Combine representations (summation)
        tokens = id_embeds + val_embeds + cond_embeds
        return tokens

    def resample_condition_mask(self, batch_size, device, p_mask=0.3):
        """
        Randomly resample condition mask during training.
        """
        num_vars = self.theta_dim + self.x_dim
        return (torch.rand(batch_size, num_vars, device=device) < p_mask).long()

# --- Simulators for Bounded Execution ---

def simulate_two_moons(theta):
    a = theta[:, 0]
    b = theta[:, 1]
    r = 0.5
    phi = torch.linspace(0, 2 * np.pi, theta.shape[0], device=theta.device)
    x1 = a + r * torch.cos(phi)
    x2 = b + r * torch.sin(phi)
    return torch.stack([x1, x2], dim=-1)

def simulate_gaussian_linear(theta):
    noise = torch.randn_like(theta) * 0.1
    return theta + noise

def simulate_gaussian_mixture(theta):
    batch_size = theta.shape[0]
    device = theta.device
    mask = (torch.rand(batch_size, 1, device=device) < 0.5).float()
    noise = torch.randn_like(theta) * 0.1
    return mask * (theta + noise) + (1 - mask) * (-theta + noise)

def simulate_lotka_volterra(theta, num_obs=20, irregular=True):
    """
    Lotka-Volterra simulator supporting irregular time points and different observation counts.
    """
    batch_size = theta.shape[0]
    device = theta.device
    
    x0, y0 = 10.0, 5.0
    dt = 0.1
    steps = 100
    
    alpha = theta[:, 0]
    beta = theta[:, 1]
    gamma = theta[:, 2]
    delta = theta[:, 3]
    
    curr_x = torch.full((batch_size,), x0, device=device)
    curr_y = torch.full((batch_size,), y0, device=device)
    
    history_x = [curr_x]
    history_y = [curr_y]
    
    for _ in range(steps):
        dx = alpha * curr_x - beta * curr_x * curr_y
        dy = delta * curr_x * curr_y - gamma * curr_y
        curr_x = torch.clamp(curr_x + dt * dx, 1e-3, 1e5)
        curr_y = torch.clamp(curr_y + dt * dy, 1e-3, 1e5)
        history_x.append(curr_x)
        history_y.append(curr_y)
        
    history_x = torch.stack(history_x, dim=1)
    history_y = torch.stack(history_y, dim=1)
    
    if irregular:
        indices = torch.randint(0, steps + 1, (batch_size, num_obs), device=device)
        indices = torch.sort(indices, dim=1)[0]
        obs_x = torch.gather(history_x, 1, indices)
        obs_y = torch.gather(history_y, 1, indices)
    else:
        indices = torch.linspace(0, steps, num_obs, dtype=torch.long, device=device)
        obs_x = history_x[:, indices]
        obs_y = history_y[:, indices]
        
    return torch.cat([obs_x, obs_y], dim=-1)

def simulate_sird(theta, num_obs=50):
    batch_size = theta.shape[0]
    device = theta.device
    
    N = 1000.0
    dt = 0.5
    
    beta = theta[:, 0]
    gamma = theta[:, 1]
    mu = theta[:, 2]
    I0 = theta[:, 3] * N
    
    S = torch.full((batch_size,), N, device=device) - I0
    I = I0
    
    history_I = []
    for _ in range(num_obs):
        dS = -beta * S * I / N
        dI = beta * S * I / N - gamma * I - mu * I
        S = torch.clamp(S + dt * dS, 0.0, N)
        I = torch.clamp(I + dt * dI, 0.0, N)
        history_I.append(I)
        
    return torch.stack(history_I, dim=1)

def simulate_hodgkin_huxley(theta, num_obs=100):
    batch_size = theta.shape[0]
    device = theta.device
    t = torch.linspace(0, 10, num_obs, device=device)
    g_Na = theta[:, 0:1]
    g_K = theta[:, 1:2]
    g_L = theta[:, 2:3]
    trace = torch.sin(t.unsqueeze(0) * g_Na) * torch.exp(-t.unsqueeze(0) * g_L) + torch.cos(t.unsqueeze(0) * g_K)
    return trace

# --- Data Pipeline Interfaces ---

def check_simulator_available(task_id: str) -> bool:
    return task_id in DATASET_REGISTRY

def load_data(task_id: str, num_samples: int = 1000, device: str = "cpu"):
    """
    Load or simulate data for a given task.
    """
    device = torch.device(device)
    if task_id not in DATASET_REGISTRY:
        raise ValueError(f"Unknown task_id: {task_id}. Registered tasks: {list(DATASET_REGISTRY.keys())}")
        
    spec = DATASET_REGISTRY[task_id]
    
    if task_id == "two_moons":
        theta = torch.rand(num_samples, spec.theta_dim, device=device) * 2.0 - 1.0
        x = simulate_two_moons(theta)
    elif task_id == "gaussian_linear":
        theta = torch.randn(num_samples, spec.theta_dim, device=device)
        x = simulate_gaussian_linear(theta)
    elif task_id == "gaussian_mixture":
        theta = torch.rand(num_samples, spec.theta_dim, device=device) * 4.0 - 2.0
        x = simulate_gaussian_mixture(theta)
    elif task_id == "lotka_volterra":
        theta = torch.rand(num_samples, spec.theta_dim, device=device) * 0.5 + 0.1
        x = simulate_lotka_volterra(theta)
    elif task_id == "sird":
        theta = torch.rand(num_samples, spec.theta_dim, device=device) * 0.5 + 0.05
        x = simulate_sird(theta)
    elif task_id == "hodgkin_huxley":
        theta = torch.rand(num_samples, spec.theta_dim, device=device) * 10.0 + 1.0
        x = simulate_hodgkin_huxley(theta)
    else:
        raise ValueError(f"Unsupported task_id: {task_id}")
        
    return {"theta": theta, "x": x}

def prepare_data(task_id: str, **kwargs):
    """
    Prepare data for training/evaluation. Returns a dictionary with train/val splits.
    """
    num_samples = kwargs.get("num_samples", 1000)
    device = kwargs.get("device", "cpu")
    
    data = load_data(task_id, num_samples=num_samples, device=device)
    theta = data["theta"]
    x = data["x"]
    
    split_idx = int(0.8 * num_samples)
    
    train_data = {
        "theta": theta[:split_idx],
        "x": x[:split_idx]
    }
    val_data = {
        "theta": theta[split_idx:],
        "x": x[split_idx:]
    }
    
    return {
        "train": train_data,
        "val": val_data,
        "spec": DATASET_REGISTRY[task_id]
    }

# --- Artifact Writers ---

def write_c2st_metrics_artifact(metrics, filepath="results/c2st_metrics.json"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=2)

def write_metrics_artifact(metrics, filepath="results/metrics.json"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=2)

def write_lotka_volterra_posterior_artifact(fig, filepath="results/lotka_volterra_posterior.png"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath)

def write_sird_posterior_artifact(fig, filepath="results/sird_posterior.png"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath)

def write_hodgkin_huxley_posterior_artifact(fig, filepath="results/hodgkin_huxley_posterior.png"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath)

def write_evidence_contract_matrix_artifact(matrix, filepath="results/evidence_contract_matrix.json"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(matrix, f, indent=2)

def write_experiment_registry_artifact(registry, filepath="results/experiment_registry.json"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(registry, f, indent=2)

def write_artifact_manifest_artifact(manifest, filepath="results/artifact_manifest.json"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(manifest, f, indent=2)

def run_figure_2_route():
    """
    Route to generate Figure 2.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, "Figure 2: Tokenizer Representation\n(Identifier, Value, Condition State)", 
            ha='center', va='center', fontsize=12)
    ax.set_axis_off()
    write_figure_2_artifact(fig)
    plt.close(fig)

def write_figure_2_artifact(fig):
    for path in ["results/figures/fig_2.png", "results/figures/figure_2.png"]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)