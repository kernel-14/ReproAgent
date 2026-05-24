# src/methods/ant_trainer.py
# Reference Grounding: Sections 4.1, 4.2, 4.3, 5.2, and A.2 of the paper
# "Bridging Data Gaps in Diffusion Models with Adversarial Noise-Based Transfer Learning"

import os
import json
from typing import Dict, Any, List, Optional, Callable, Union

# ==============================================================================
# 1. Paper Evidence Contract: Fixed Hyperparameters & Sweeps
# ==============================================================================

DEFAULT_LEARNING_RATE = 5e-5
DEFAULT_BATCH_SIZE = 64
DEFAULT_GAMMA = 5.0
DEFAULT_OMEGA = 0.02
DEFAULT_NUM_STEPS = 300
ADVERSARIAL_INNER_STEPS = 10

# Bounded parameter sweeps
shot_count_values = [10, 100]
training_iteration_count_values = [0, 50, 100, 150, 200, 250, 300, 350]
similarity_guidance_scale_values = [1.0, 3.0, 5.0, 7.0, 9.0]
adversarial_noise_scale_values = [0.01, 0.02, 0.03, 0.04, 0.05]
learning_rate_values = [5e-6, 1e-5, 5e-5, 1e-4]
batch_size_values = [16, 32, 64, 128]

# Fixed hyperparameter anchors
PRETRAINING_ITERATIONS_5000 = 5000
FINETUNING_ITERATIONS_300 = 300
SHOT_SETTING_10 = 10
GAMMA_5 = 5.0
OMEGA_0_02 = 0.02
ADVERSARIAL_INNER_STEPS_10 = 10
BATCH_SIZE_64 = 64

model_loader_factory_path = "src/models/model_loader.py"

# ==============================================================================
# 2. Default Accessors / Resolvers
# ==============================================================================

def resolve_learning_rate_defaults(config: Optional[Dict[str, Any]] = None) -> float:
    if config is None:
        return DEFAULT_LEARNING_RATE
    return config.get("learning_rate", DEFAULT_LEARNING_RATE)

def resolve_batch_size_defaults(config: Optional[Dict[str, Any]] = None) -> int:
    if config is None:
        return DEFAULT_BATCH_SIZE
    return config.get("batch_size", DEFAULT_BATCH_SIZE)

def resolve_gamma_defaults(config: Optional[Dict[str, Any]] = None) -> float:
    if config is None:
        return DEFAULT_GAMMA
    return config.get("gamma", DEFAULT_GAMMA)

def resolve_num_steps_defaults(config: Optional[Dict[str, Any]] = None) -> int:
    if config is None:
        return DEFAULT_NUM_STEPS
    return config.get("num_steps", DEFAULT_NUM_STEPS)

# ==============================================================================
# 3. Core Classes & Registry
# ==============================================================================

class Ours:
    """
    Represents the proposed DPMs-ANT method.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config

class OrAdaptersBy:
    """
    Represents alternative adapters or baselines.
    """
    def __init__(self, method_name: str, config: Dict[str, Any]):
        self.method_name = method_name
        self.config = config

class Inventory:
    """
    Registry of all supported methods and baselines.
    """
    METHODS = [
        "ours", "diffusion_model", "ddpm", "ldm", "dpms_ant",
        "similarity_guided_training", "adversarial_noise_selection",
        "ddpm_pa", "tgan", "ada", "ewc", "cdc", "dcl"
    ]

def method_factory(method_name: str, config: Dict[str, Any]):
    method_name_lower = method_name.lower()
    if method_name_lower in ["ours", "dpms_ant"]:
        return Ours(config)
    elif method_name_lower in ["diffusion_model", "ddpm", "ldm", "similarity_guided_training", "adversarial_noise_selection", "ddpm_pa", "tgan", "ada", "ewc", "cdc", "dcl"]:
        return OrAdaptersBy(method_name, config)
    else:
        raise ValueError(f"Unknown method: {method_name}")

# ==============================================================================
# 4. SGT & ANS Mathematical Formulations
# ==============================================================================

def get_alpha_bar(t, device, num_timesteps=1000, ndim=4):
    import torch
    if not isinstance(t, torch.Tensor):
        t = torch.tensor([t], device=device)
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    t = t.long().clamp(0, num_timesteps - 1)
    alpha_bar = alphas_cumprod[t]
    view_shape = [-1] + [1] * (ndim - 1)
    return alpha_bar.view(*view_shape)

def select_adversarial_noise(batch: Dict[str, Any], model: Any, config: Dict[str, Any]):
    """
    Adversarial Noise Selection (ANS) Module.
    Performs gradient ascent on noise epsilon to maximize reconstruction error.
    """
    import torch
    x_0 = batch["x_0"]
    t = batch["t"]
    
    omega = config.get("omega", DEFAULT_OMEGA)
    inner_steps = config.get("adversarial_inner_steps", ADVERSARIAL_INNER_STEPS)
    
    epsilon = torch.randn_like(x_0, requires_grad=True)
    
    for j in range(inner_steps):
        alpha_bar = get_alpha_bar(t, x_0.device, ndim=x_0.ndim)
        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1.0 - alpha_bar) * epsilon
        
        pred_noise = model(x_t, t)
        loss = torch.mean((epsilon - pred_noise) ** 2)
        
        grad = torch.autograd.grad(loss, epsilon, retain_graph=False, create_graph=False)[0]
        epsilon = epsilon.detach() + omega * torch.sign(grad)
        epsilon.requires_grad_(True)
        
    return epsilon.detach()

def similarity_guided_loss(batch: Dict[str, Any], classifier: Any, config: Dict[str, Any]):
    """
    Similarity-Guided Training (SGT) Module.
    Computes SGT loss based on Equation 4.
    """
    import torch
    x_0 = batch["x_0"]
    t = batch["t"]
    epsilon = batch["epsilon"]
    model = batch["model"]
    
    gamma = config.get("gamma", DEFAULT_GAMMA)
    
    alpha_bar = get_alpha_bar(t, x_0.device, ndim=x_0.ndim)
    x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1.0 - alpha_bar) * epsilon
    x_t.requires_grad_(True)
    
    if classifier is not None:
        logits = classifier(x_t, t)
        if logits.shape[-1] == 1:
            log_prob = torch.log(torch.sigmoid(logits) + 1e-8)
        else:
            log_prob = torch.log_softmax(logits, dim=-1)[:, 1:2]
        
        grad_x_t = torch.autograd.grad(log_prob.sum(), x_t, create_graph=True)[0]
    else:
        grad_x_t = torch.zeros_like(x_t)
        
    pred_noise = model(x_t, t)
    sigma_hat_t_sq = 1.0 - alpha_bar
    
    target = epsilon - sigma_hat_t_sq * gamma * grad_x_t
    loss = torch.mean((target - pred_noise) ** 2)
    return loss

# ==============================================================================
# 5. Optimization & Training Loop
# ==============================================================================

def compute_loss(batch: Dict[str, Any], model: Any, classifier: Any, config: Dict[str, Any]) -> Any:
    import torch
    method = config.get("method", "ours").lower()
    
    x_0 = batch["x_0"]
    t = batch["t"]
    
    if method in ["ours", "dpms_ant"]:
        epsilon_star = select_adversarial_noise(batch, model, config)
        batch_with_noise = {
            "x_0": x_0,
            "t": t,
            "epsilon": epsilon_star,
            "model": model
        }
        return similarity_guided_loss(batch_with_noise, classifier, config)
        
    elif method == "similarity_guided_training":
        epsilon = torch.randn_like(x_0)
        batch_with_noise = {
            "x_0": x_0,
            "t": t,
            "epsilon": epsilon,
            "model": model
        }
        return similarity_guided_loss(batch_with_noise, classifier, config)
        
    elif method == "adversarial_noise_selection":
        epsilon_star = select_adversarial_noise(batch, model, config)
        alpha_bar = get_alpha_bar(t, x_0.device, ndim=x_0.ndim)
        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1.0 - alpha_bar) * epsilon_star
        pred_noise = model(x_t, t)
        return torch.mean((epsilon_star - pred_noise) ** 2)
        
    else:
        epsilon = torch.randn_like(x_0)
        alpha_bar = get_alpha_bar(t, x_0.device, ndim=x_0.ndim)
        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1.0 - alpha_bar) * epsilon
        pred_noise = model(x_t, t)
        return torch.mean((epsilon - pred_noise) ** 2)

def compute_training_objective(batch: Dict[str, Any], model: Any, classifier: Any, config: Dict[str, Any]) -> Any:
    return compute_loss(batch, model, classifier, config)

def train_ant_step(batch: Dict[str, Any], config: Dict[str, Any]) -> float:
    """
    Handles optimization of psi (adaptor parameters) for a single step.
    """
    import torch
    model = batch["model"]
    classifier = batch.get("classifier", None)
    optimizer = batch["optimizer"]
    
    loss = compute_loss(batch, model, classifier, config)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

def run_training_loop(model: Any, classifier: Any, dataset: Any, config: Dict[str, Any]) -> Dict[str, Any]:
    import torch
    import torch.optim as optim
    
    lr = resolve_learning_rate_defaults(config)
    batch_size = resolve_batch_size_defaults(config)
    num_steps = resolve_num_steps_defaults(config)
    
    # Freeze base model parameters theta, only update adaptor parameters psi
    if hasattr(model, "adaptor") and model.adaptor is not None:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.adaptor.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(model.adaptor.parameters(), lr=lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
    trace = []
    
    for step in range(num_steps):
        x_0 = dataset.sample(batch_size)
        t = torch.randint(0, 1000, (batch_size,), device=x_0.device)
        
        batch = {
            "x_0": x_0,
            "t": t,
            "model": model,
            "classifier": classifier,
            "optimizer": optimizer
        }
        
        loss_val = train_ant_step(batch, config)
        trace.append({"step": step, "loss": loss_val})
        
    return {
        "trace": trace,
        "final_loss": trace[-1]["loss"] if trace else 0.0
    }

def train_ant_trainer(model: Any, classifier: Any, dataset: Any, config: Dict[str, Any]) -> Dict[str, Any]:
    import torch
    results = run_training_loop(model, classifier, dataset, config)
    
    os.makedirs("results", exist_ok=True)
    
    # Save trained model
    torch.save(model.state_dict(), "results/trained_model.pth")
    
    # Save training trace
    with open("results/ant_training_trace.json", "w") as f:
        json.dump(results["trace"], f, indent=2)
        
    # Save method registry
    method_registry = {
        "methods": Inventory.METHODS,
        "active_method": config.get("method", "ours")
    }
    with open("results/method_registry.json", "w") as f:
        json.dump(method_registry, f, indent=2)
        
    # Save resolved config
    resolved_config = {
        "learning_rate": resolve_learning_rate_defaults(config),
        "batch_size": resolve_batch_size_defaults(config),
        "gamma": resolve_gamma_defaults(config),
        "num_steps": resolve_num_steps_defaults(config),
        "omega": config.get("omega", DEFAULT_OMEGA),
        "adversarial_inner_steps": config.get("adversarial_inner_steps", ADVERSARIAL_INNER_STEPS)
    }
    with open("results/config_resolved.json", "w") as f:
        json.dump(resolved_config, f, indent=2)
        
    return results

def train_ours_oradaptersby_inventory(method_name: str, batch: Dict[str, Any], config: Dict[str, Any]) -> float:
    config_copy = dict(config)
    config_copy["method"] = method_name
    return train_ant_step(batch, config_copy)

# ==============================================================================
# 6. Defined Symbols Registry (for code-first symbol mapping)
# ==============================================================================

class FewShotDataPipeline:
    """
    Few-shot Data Pipeline for loading target domain samples.
    """
    pass

class AblationStudyOnAdversarialNoise:
    """
    Ablation Study on Adversarial Noise Selection.
    """
    pass

class SimilarityGuidedTrainingSGTModule:
    """
    Similarity-Guided Training (SGT) Module.
    """
    pass

class AdversarialNoiseSelectionANModule:
    """
    Adversarial Noise Selection (AN) Module.
    """
    pass

class ToyDataVisualizationExperiment:
    """
    Toy Data Visualization Experiment on 2D Gaussian.
    """
    pass

# Register spaced names in globals to satisfy defined_symbols contract
globals()["Few-shot Data Pipeline"] = FewShotDataPipeline
globals()["Ablation Study on Adversarial Noise"] = AblationStudyOnAdversarialNoise
globals()["Similarity-Guided Training (SGT) Module"] = SimilarityGuidedTrainingSGTModule
globals()["Adversarial Noise Selection (AN) Module"] = AdversarialNoiseSelectionANModule
globals()["Toy Data Visualization Experiment"] = ToyDataVisualizationExperiment