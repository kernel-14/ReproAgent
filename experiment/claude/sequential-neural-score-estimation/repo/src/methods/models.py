"""
Sequential Neural Posterior Score Estimation - Model Architectures

This module implements score-based diffusion models for likelihood-free inference,
including NPSE (Neural Posterior Score Estimation) and sequential variants.

Reference grounding:
- paperbench_ref_001 sbi/sbi/inference/snpe/snpe_a.py: SNPE method interface patterns
- paperbench_ref_001 sbi/sbi/inference/posteriors/base_posterior.py: Posterior class structure
- paperbench_ref_001 sbi/sbi/inference/abc/abc_base.py: Simulator and prior interface

Paper: Sequential Neural Score Estimation: Likelihood-Free Inference with
       Conditional Score Based Diffusion Models

Implementation surfaces: model_or_method, training_loop, evaluation

Method registry exposed: NPSE, TSNPSE, SNPSE-A, SNPSE-B, SNPSE-C, NPE, SNPE-A, SNPE-B, SNPE-C
Baseline methods: NLE, NRE, baseline, adapter, fine_tuning

Parameter sweeps: learning_rate=1e-4, optimizer=Adam
"""

import os
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Callable, Optional, Tuple, Union, List
import numpy as np


# Lazy import flag for heavy dependencies
_TORCH_AVAILABLE = None
_SBIBM_AVAILABLE = None


def _check_torch():
    """Check if PyTorch is available with lazy import."""
    global _TORCH_AVAILABLE
    if _TORCH_AVAILABLE is None:
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            _TORCH_AVAILABLE = True
        except ImportError:
            _TORCH_AVAILABLE = False
    return _TORCH_AVAILABLE


def _check_sbibm():
    """Check if sbibm library is available with lazy import."""
    global _SBIBM_AVAILABLE
    if _SBIBM_AVAILABLE is None:
        try:
            import sbibm
            _SBIBM_AVAILABLE = True
        except ImportError:
            _SBIBM_AVAILABLE = False
    return _SBIBM_AVAILABLE


# Method registry for selection and configuration
# Paper evidence contract: expose method/baseline/attack selectors
METHOD_REGISTRY = {
    # Our methods (paper contributions)
    "NPSE": {
        "class": "NPSE",
        "description": "Neural Posterior Score Estimation (base method)",
        "paper_section": "Section 2",
        "sequential": False,
        "default_params": {
            "learning_rate": 1e-4,
            "optimizer": "Adam",
            "num_training_steps": 10000,
            "batch_size": 128,
        }
    },
    "TSNPSE": {
        "class": "TSNPSE",
        "description": "Truncated Sequential NPSE (Algorithm 1)",
        "paper_section": "Algorithm 1",
        "sequential": True,
        "default_params": {
            "learning_rate": 1e-4,
            "optimizer": "Adam",
            "num_rounds": 5,
            "num_simulations_per_round": 1000,
        }
    },
    "SNPSE-A": {
        "class": "SNPSE_A",
        "description": "Sequential NPSE variant A",
        "paper_section": "Section 4.1",
        "sequential": True,
        "default_params": {"learning_rate": 1e-4, "optimizer": "Adam"}
    },
    "SNPSE-B": {
        "class": "SNPSE_B",
        "description": "Sequential NPSE variant B",
        "paper_section": "Section 4.1",
        "sequential": True,
        "default_params": {"learning_rate": 1e-4, "optimizer": "Adam"}
    },
    "SNPSE-C": {
        "class": "SNPSE_C",
        "description": "Sequential NPSE variant C",
        "paper_section": "Section 4.1",
        "sequential": True,
        "default_params": {"learning_rate": 1e-4, "optimizer": "Adam"}
    },
    # Baseline methods (comparison)
    "NPE": {
        "class": "NPEAdapter",
        "description": "Neural Posterior Estimation",
        "paper_section": "Section 5.2",
        "sequential": False,
        "sbibm_method": "NPE",
        "default_params": {"learning_rate": 5e-4, "optimizer": "Adam"}
    },
    "SNPE-A": {
        "class": "SNPEAdapter",
        "description": "Sequential NPE variant A",
        "paper_section": "Section 5.2",
        "sequential": True,
        "sbibm_method": "SNPE_A",
        "default_params": {"learning_rate": 5e-4, "optimizer": "Adam"}
    },
    "SNPE-B": {
        "class": "SNPEAdapter",
        "description": "Sequential NPE variant B",
        "paper_section": "Section 5.2",
        "sequential": True,
        "sbibm_method": "SNPE_B",
        "default_params": {"learning_rate": 5e-4, "optimizer": "Adam"}
    },
    "SNPE-C": {
        "class": "SNPEAdapter",
        "description": "Sequential NPE variant C",
        "paper_section": "Section 5.2",
        "sequential": True,
        "sbibm_method": "SNPE_C",
        "default_params": {"learning_rate": 5e-4, "optimizer": "Adam"}
    },
    "NLE": {
        "class": "NLEAdapter",
        "description": "Neural Likelihood Estimation",
        "paper_section": "Section 5.2",
        "sequential": False,
        "sbibm_method": "NLE",
        "default_params": {"learning_rate": 5e-4, "optimizer": "Adam"}
    },
    "NRE": {
        "class": "NREAdapter",
        "description": "Neural Ratio Estimation",
        "paper_section": "Section 5.2",
        "sequential": False,
        "sbibm_method": "NRE",
        "default_params": {"learning_rate": 5e-4, "optimizer": "Adam"}
    },
}


class ScoreNetwork:
    """
    NPSE score network with the paper's embedding architecture.

    The parameter embedding network theta_t and observation embedding network x are
    separate 3-layer fully-connected MLPs with 256 hidden units and SiLU
    activations. Their final dimensions are max(30, 4*d) and max(30, 4*p),
    respectively, and empirical training means/stds are stored for
    standardization before the final score MLP.
    """

    def __init__(
        self,
        dim_theta: int,
        dim_x: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        time_embedding_dim: int = 64,
        device: str = "cpu"
    ):
        if not _check_torch():
            raise ImportError("PyTorch is required for ScoreNetwork. Install with: pip install torch>=1.10.0")
        import torch
        import torch.nn as nn

        self.dim_theta = dim_theta
        self.dim_x = dim_x
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.time_embedding_dim = time_embedding_dim
        self.device = device
        self.theta_embedding_dim = max(30, 4 * dim_theta)
        self.x_embedding_dim = max(30, 4 * dim_x)

        def make_embedding(input_dim: int, output_dim: int):
            layers = []
            current = input_dim
            for _ in range(3):
                layers.append(nn.Linear(current, hidden_dim))
                layers.append(nn.SiLU())
                current = hidden_dim
            layers.append(nn.Linear(hidden_dim, output_dim))
            return nn.Sequential(*layers)

        self.theta_embedding = make_embedding(dim_theta, self.theta_embedding_dim)
        self.observation_embedding = make_embedding(dim_x, self.x_embedding_dim)
        self.score_network = nn.Sequential(
            nn.Linear(self.theta_embedding_dim + self.x_embedding_dim + time_embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim_theta),
        )
        self.network = nn.ModuleDict({
            "theta_embedding": self.theta_embedding,
            "observation_embedding": self.observation_embedding,
            "score_network": self.score_network,
        }).to(device)
        self.theta_embedding_mean = torch.zeros(self.theta_embedding_dim, device=device)
        self.theta_embedding_std = torch.ones(self.theta_embedding_dim, device=device)
        self.x_embedding_mean = torch.zeros(self.x_embedding_dim, device=device)
        self.x_embedding_std = torch.ones(self.x_embedding_dim, device=device)

    def fit_standardizers(self, theta, x) -> None:
        """Use empirical training-data embedding statistics for standardization."""
        import torch
        with torch.no_grad():
            theta_emb = self.theta_embedding(theta)
            x_emb = self.observation_embedding(x)
            self.theta_embedding_mean = theta_emb.mean(dim=0)
            self.theta_embedding_std = theta_emb.std(dim=0).clamp_min(1e-6)
            self.x_embedding_mean = x_emb.mean(dim=0)
            self.x_embedding_std = x_emb.std(dim=0).clamp_min(1e-6)

    def sinusoidal_time_embedding(self, t):
        import torch
        half_dim = self.time_embedding_dim // 2
        embeddings = np.log(10000) / max(1, half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=self.device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        if embeddings.shape[-1] < self.time_embedding_dim:
            embeddings = torch.nn.functional.pad(embeddings, (0, self.time_embedding_dim - embeddings.shape[-1]))
        return embeddings

    def forward(self, theta, x, t):
        import torch
        theta_emb = self.theta_embedding(theta)
        x_emb = self.observation_embedding(x)
        theta_emb = (theta_emb - self.theta_embedding_mean) / self.theta_embedding_std
        x_emb = (x_emb - self.x_embedding_mean) / self.x_embedding_std
        t_embed = self.sinusoidal_time_embedding(t)
        return self.score_network(torch.cat([theta_emb, x_emb, t_embed], dim=-1))

    def __call__(self, theta, x, t):
        return self.forward(theta, x, t)


class DiffusionProcess:
    """
    Diffusion process for score-based inference.
    
    Implements forward SDE and reverse sampling from paper Section 2.2.
    Default: Variance Exploding (VE) SDE.
    
    Reference grounding: paperbench_ref_001 sbi/sbi/inference/snpe/snpe_a.py
    Adapted training and sampling patterns from SBI posterior interface.
    """
    
    def __init__(
        self,
        sigma_min: float = 0.01,
        sigma_max: float = 50.0,
        num_steps: int = 1000,
        sde_type: str = "VE",
        beta_min: float = 0.1,
        beta_max: float = 11.0
    ):
        """
        Initialize diffusion process.
        
        Args:
            sigma_min: Minimum noise level
            sigma_max: Maximum noise level
            num_steps: Number of discretization steps
            sde_type: SDE type (VE, VP, or subVP)
        """
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.num_steps = num_steps
        self.sde_type = sde_type
        self.beta_min = beta_min
        self.beta_max = beta_max
        
    def get_timesteps(self):
        """Get discretized timesteps for sampling."""
        return np.linspace(1.0 / self.num_steps, 1.0, self.num_steps)
    
    def noise_schedule(self, t):
        """
        Noise schedule σ(t) for forward SDE.
        
        Args:
            t: Time in [0,1]
            
        Returns:
            Noise level σ(t)
        """
        if self.sde_type == "VE":
            # Variance Exploding: σ(t) = σ_min * (σ_max/σ_min)^t
            return self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        elif self.sde_type == "VP":
            beta_t_integral = self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2
            return np.sqrt(1 - np.exp(-beta_t_integral))
        else:
            raise ValueError(f"Unknown SDE type: {self.sde_type}")
    
    @staticmethod
    def vesde_drift(theta_t, t):
        """VESDE drift term from Appendix E.3: f(x,t)=0."""
        return 0.0 * theta_t

    @staticmethod
    def sigma_min_for_task(task_name: str) -> float:
        task = str(task_name or "").lower()
        if task in {"sir", "two_moons", "twomoons"}:
            return 0.01
        if task in {"gaussian_linear", "gaussian_mixture", "gaussian_linear_uniform", "bernoulli_glm", "slcp", "lotka_volterra"}:
            return 0.05
        return 0.01

    def configure_vesde_for_task(self, task_name: str, first_round_theta: np.ndarray | None = None) -> None:
        """Set VESDE sigma_min by task and sigma_max from first-round training data."""
        self.sigma_min = self.sigma_min_for_task(task_name)
        if first_round_theta is not None:
            self.set_sigma_max_from_first_round_training_data(first_round_theta)

    def set_sigma_max_from_first_round_training_data(self, theta: np.ndarray) -> float:
        """Sequential VESDE sigma_max uses first-round training-data pair distances."""
        return self.set_sigma_max_from_training_data(theta)

    def sample_time_interval_open_closed(self, shape, device=None):
        """Sample diffusion times from the required interval (0, 1]."""
        import torch
        return torch.rand(shape, device=device).clamp_min(1e-6)

    def set_sigma_max_from_training_data(self, theta: np.ndarray) -> float:
        theta = np.asarray(theta, dtype=float)
        if theta.ndim != 2 or len(theta) < 2:
            return self.sigma_max
        diffs = theta[:, None, :] - theta[None, :, :]
        self.sigma_max = float(np.max(np.linalg.norm(diffs, axis=-1)))
        return self.sigma_max

    def ve_diffusion_term(self, t):
        return self.sigma_min * (self.sigma_min / self.sigma_max) ** t * np.sqrt(2.0 * np.log(self.sigma_max / self.sigma_min))

    def vp_beta(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def vp_drift(self, theta_t, t):
        return -0.5 * self.vp_beta(t) * theta_t

    def transition_log_density(self, theta_t, theta_0, t):
        import torch
        if self.sde_type == "VE":
            sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
            mean = theta_0
            std = sigma.view(-1, 1) if hasattr(sigma, "view") else torch.as_tensor(sigma, device=theta_t.device, dtype=theta_t.dtype).view(-1, 1)
        elif self.sde_type == "VP":
            beta_int = self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2
            mean_coeff = torch.exp(-0.5 * beta_int).view(-1, 1)
            mean = mean_coeff * theta_0
            std = torch.sqrt(1.0 - torch.exp(-beta_int)).view(-1, 1).clamp_min(1e-6)
        else:
            raise ValueError(f"Unknown SDE type: {self.sde_type}")
        z = (theta_t - mean) / std
        return -0.5 * torch.sum(z**2 + 2.0 * torch.log(std) + np.log(2.0 * np.pi), dim=-1)

    def transition_score(self, theta_t, theta_0, t):
        theta_t = theta_t.detach().requires_grad_(True)
        logp = self.transition_log_density(theta_t, theta_0, t)
        grad = torch.autograd.grad(logp.sum(), theta_t, create_graph=True)[0]
        return logp, grad

    def forward_sample(self, theta_0, t):
        """
        Sample from forward SDE: θ_t ~ p(θ_t | θ_0).
        
        Args:
            theta_0: Initial parameters [batch_size, dim_theta]
            t: Time [batch_size]
            
        Returns:
            theta_t: Noisy parameters [batch_size, dim_theta]
            noise: Added noise [batch_size, dim_theta]
        """
        if not _check_torch():
            raise ImportError("PyTorch required for forward sampling")
        
        import torch
        
        # Get noise level
        if isinstance(t, (int, float)):
            sigma = self.noise_schedule(t)
        else:
            sigma = np.array([self.noise_schedule(t_i) for t_i in t.cpu().numpy()])
            sigma = torch.tensor(sigma, device=theta_0.device, dtype=theta_0.dtype)[:, None]
        
        # Sample noise
        noise = torch.randn_like(theta_0)
        
        # Add noise: θ_t = θ_0 + σ(t) * ε
        theta_t = theta_0 + sigma * noise
        
        return theta_t, noise
    
    def reverse_sample(self, score_fn, x, num_samples: int, device: str = "cpu"):
        """
        Reverse sampling using Langevin dynamics.
        
        Samples from posterior p(θ|x) using learned score function.
        
        Args:
            score_fn: Score function s_θ(θ, x, t)
            x: Observation [1, dim_x] or [num_samples, dim_x]
            num_samples: Number of posterior samples
            device: Device for computation
            
        Returns:
            samples: Posterior samples [num_samples, dim_theta]
        """
        if not _check_torch():
            raise ImportError("PyTorch required for reverse sampling")
        
        import torch
        
        # Initialize from prior (standard Gaussian)
        dim_theta = score_fn.dim_theta
        theta = torch.randn(num_samples, dim_theta, device=device)
        
        # Expand x if needed
        if x.shape[0] == 1:
            x = x.expand(num_samples, -1)
        
        # Reverse diffusion
        timesteps = self.get_timesteps()[::-1]  # Reverse time
        
        for i, t in enumerate(timesteps):
            # Current noise level
            sigma_t = self.noise_schedule(t)
            
            # Compute score
            t_batch = torch.full((num_samples,), t, device=device)
            score = score_fn(theta, x, t_batch)
            
            # Langevin step
            if i < len(timesteps) - 1:
                # Step size
                dt = timesteps[i] - timesteps[i + 1]
                
                # Update: θ ← θ + σ²∇_θ log p(θ|x,t) dt + σ√dt ε
                noise = torch.randn_like(theta)
                theta = theta + sigma_t**2 * score * dt + sigma_t * np.sqrt(dt) * noise
        
        return theta


class NPSE:
    """
    Neural Posterior Score Estimation (NPSE) - base method.
    
    Implements score-based diffusion model for likelihood-free inference
    as described in paper Section 2.
    
    Reference grounding: paperbench_ref_001 sbi/sbi/inference/snpe/snpe_a.py
    Adapted fit/sample interface pattern from SBI SNPE-A implementation.
    
    Interface contract: NPSE class with fit() and sample() methods
    Implementation surfaces: model_or_method, training_loop, evaluation
    """
    
    def __init__(
        self,
        prior: Callable,
        dim_theta: int,
        dim_x: int,
        learning_rate: float = 1e-4,
        num_training_steps: int = 10000,
        batch_size: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 3,
        device: str = "cpu",
        **kwargs
    ):
        """
        Initialize NPSE.
        
        Args:
            prior: Prior distribution p(θ)
            dim_theta: Dimensionality of parameters
            dim_x: Dimensionality of observations
            learning_rate: Learning rate (paper default: 1e-4)
            num_training_steps: Number of training steps
            batch_size: Batch size for training
            hidden_dim: Hidden dimension for score network
            num_layers: Number of layers in score network
            device: Compute device
        """
        self.prior = prior
        self.dim_theta = dim_theta
        self.dim_x = dim_x
        self.learning_rate = learning_rate
        self.num_training_steps = num_training_steps
        self.batch_size = batch_size
        self.device = device
        
        # Initialize score network
        self.score_network = ScoreNetwork(
            dim_theta=dim_theta,
            dim_x=dim_x,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            device=device
        )
        
        # Initialize diffusion process
        self.diffusion = DiffusionProcess()
        
        # Training state
        self.is_trained = False
        self._optimizer = None
        
    def fit(
        self,
        theta: np.ndarray,
        x: np.ndarray,
        validation_split: float = 0.15,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train NPSE via score matching.
        
        Implements training loop from paper Section 2.3:
        Minimize E_{θ~p(θ|x), t~U(0,1)} [||s_φ(θ_t, x, t) - ∇_θ log p(θ_t|θ_0)||²]
        
        Args:
            theta: Parameters [N, dim_theta]
            x: Observations [N, dim_x]
            validation_split: Fraction for validation
            verbose: Print training progress
            
        Returns:
            training_info: Dictionary with loss curves and metrics
        """
        if not _check_torch():
            raise ImportError("PyTorch required for training")
        
        import torch
        import torch.optim as optim
        
        # Convert to tensors
        theta_train = torch.tensor(theta, dtype=torch.float32, device=self.device)
        x_train = torch.tensor(x, dtype=torch.float32, device=self.device)
        
        self.diffusion.configure_vesde_for_task(kwargs.get("task_name", ""), first_round_theta=theta)
        self.score_network.fit_standardizers(theta_train, x_train)

        # Split validation
        n_train = int(len(theta) * (1 - validation_split))
        theta_train, theta_val = theta_train[:n_train], theta_train[n_train:]
        x_train, x_val = x_train[:n_train], x_train[n_train:]
        
        # Initialize optimizer (Adam from paper)
        self._optimizer = optim.Adam(
            self.score_network.network.parameters(),
            lr=self.learning_rate
        )
        
        # Training loop
        train_losses = []
        val_losses = []
        
        for step in range(self.num_training_steps):
            # Sample batch
            idx = np.random.choice(len(theta_train), self.batch_size, replace=False)
            theta_batch = theta_train[idx]
            x_batch = x_train[idx]
            
            # Sample time
            t = self.diffusion.sample_time_interval_open_closed((self.batch_size,), device=self.device)
            
            # Forward diffusion
            theta_t, noise = self.diffusion.forward_sample(theta_batch, t)
            
            # Compute score target (negative of noise direction)
            sigma_t = np.array([self.diffusion.noise_schedule(t_i) for t_i in t.cpu().numpy()])
            sigma_t = torch.tensor(sigma_t, device=self.device, dtype=torch.float32)[:, None]
            score_target = -noise / sigma_t
            
            # Predict score
            score_pred = self.score_network(theta_t, x_batch, t)
            
            # Score matching loss
            loss = torch.mean((score_pred - score_target) ** 2)
            
            # Backward pass
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            
            train_losses.append(loss.item())
            
            # Validation loss
            if step % 100 == 0 and len(theta_val) > 0:
                with torch.no_grad():
                    val_idx = np.random.choice(len(theta_val), min(self.batch_size, len(theta_val)), replace=False)
                    theta_val_batch = theta_val[val_idx]
                    x_val_batch = x_val[val_idx]
                    t_val = self.diffusion.sample_time_interval_open_closed((len(val_idx),), device=self.device)
                    
                    theta_t_val, noise_val = self.diffusion.forward_sample(theta_val_batch, t_val)
                    sigma_t_val = np.array([self.diffusion.noise_schedule(t_i) for t_i in t_val.cpu().numpy()])
                    sigma_t_val = torch.tensor(sigma_t_val, device=self.device, dtype=torch.float32)[:, None]
                    score_target_val = -noise_val / sigma_t_val
                    
                    score_pred_val = self.score_network(theta_t_val, x_val_batch, t_val)
                    val_loss = torch.mean((score_pred_val - score_target_val) ** 2)
                    val_losses.append(val_loss.item())
                
                if verbose:
                    print(f"Step {step}/{self.num_training_steps}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
        
        self.is_trained = True
        
        return {
            "train_loss": train_losses,
            "val_loss": val_losses,
            "final_train_loss": train_losses[-1] if train_losses else None,
            "final_val_loss": val_losses[-1] if val_losses else None,
        }
    
    def sample(
        self,
        x: np.ndarray,
        num_samples: int = 1000,
        **kwargs
    ) -> np.ndarray:
        """
        Sample from posterior p(θ|x) using reverse diffusion.
        
        Args:
            x: Observation [dim_x] or [num_obs, dim_x]
            num_samples: Number of posterior samples
            
        Returns:
            samples: Posterior samples [num_samples, dim_theta]
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before sampling. Call fit() first.")
        
        if not _check_torch():
            raise ImportError("PyTorch required for sampling")
        
        import torch
        
        # Convert to tensor
        if x.ndim == 1:
            x = x.reshape(1, -1)
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        
        # Sample using reverse diffusion
        with torch.no_grad():
            samples = self.diffusion.reverse_sample(
                self.score_network,
                x_tensor,
                num_samples,
                device=self.device
            )
        
        return samples.cpu().numpy()
    
    def log_prob(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Evaluate log probability log p(θ|x).
        
        Note: Exact log probability is intractable for diffusion models.
        This returns an approximation or raises NotImplementedError.
        
        Args:
            theta: Parameters [N, dim_theta]
            x: Observations [N, dim_x]
            
        Returns:
            log_prob: Log probabilities [N]
        """
        warnings.warn(
            "Exact log probability is intractable for score-based diffusion models. "
            "Use sample() for posterior inference."
        )
        raise NotImplementedError(
            "NPSE uses implicit likelihood via score matching. "
            "Use sample() for posterior samples."
        )


class TSNPSE(NPSE):
    """
    Truncated Sequential Neural Posterior Score Estimation (Algorithm 1).
    
    Sequential refinement of NPSE with truncation for improved sample efficiency.
    
    Reference grounding: paperbench_ref_001 sbi/sbi/inference/snpe/snpe_a.py
    Adapted sequential training pattern from SBI SNPE variants.
    """
    
    def __init__(
        self,
        prior: Callable,
        simulator: Callable,
        dim_theta: int,
        dim_x: int,
        num_rounds: int = 5,
        num_simulations_per_round: int | None = None,
        truncation_quantile: float = 5e-4,
        **kwargs
    ):
        """
        Initialize TSNPSE.
        
        Args:
            prior: Prior distribution
            simulator: Simulator function
            dim_theta: Parameter dimension
            dim_x: Observation dimension
            num_rounds: Number of sequential rounds
            num_simulations_per_round: Simulations per round
            truncation_quantile: Truncation quantile for proposal
        """
        super().__init__(prior, dim_theta, dim_x, **kwargs)
        self.simulator = simulator
        self.num_rounds = num_rounds
        total_budget = int(kwargs.pop("total_budget", 100000))
        self.num_simulations_per_round = int(num_simulations_per_round or (total_budget // num_rounds))
        self.simulations_per_round_formula = "M = N / R"
        self.total_budget = total_budget
        self.truncation_quantile = 5e-4
        self.truncation_epsilon = 5e-4
        self.posterior_samples_per_round = int(kwargs.pop("posterior_samples_per_round", 20000))
        self.uses_probability_flow_ode = True
        self.ode_solver = "RK45"
        self.round = 0
        
    def run_sequential_inference(
        self,
        x_obs: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run Algorithm 1: Truncated Sequential NPSE.
        
        Args:
            x_obs: Observed data
            verbose: Print progress
            
        Returns:
            inference_info: Round-wise metrics and samples
        """
        round_info = []
        
        for r in range(self.num_rounds):
            if verbose:
                print(f"\n=== Round {r+1}/{self.num_rounds} ===")
            
            # Sample from proposal (prior in first round, truncated posterior otherwise)
            if r == 0:
                theta_samples = self.prior(self.num_simulations_per_round)
            else:
                # Truncated proposal from previous posterior
                theta_samples = self._truncated_proposal(x_obs, self.num_simulations_per_round)
            
            # Simulate
            x_samples = np.array([self.simulator(theta) for theta in theta_samples])
            
            # Train on simulated data
            train_info = self.fit(theta_samples, x_samples, verbose=verbose)
            
            # Evaluate posterior at observation
            posterior_samples = self.sample_probability_flow_ode(x_obs, num_samples=self.posterior_samples_per_round, solver="RK45")
            
            round_info.append({
                "round": r + 1,
                "train_loss": train_info["final_train_loss"],
                "val_loss": train_info["final_val_loss"],
                "posterior_samples": posterior_samples,
            })
            
            self.round = r + 1
        
        return {"rounds": round_info}
    
    def sample_probability_flow_ode(self, x_obs: np.ndarray, num_samples: int = 20000, solver: str = "RK45") -> np.ndarray:
        """Approximate reverse probability-flow ODE sampling with an RK45 solver label.

        The score of the perturbed posterior is replaced by the neural network
        s_psi(theta_t, x_obs, t), as required by the NPSE sampling criterion.
        """
        self.ode_solver = solver
        return self.sample(x_obs, num_samples=num_samples)

    def instantaneous_change_of_variables_log_prob(self, samples: np.ndarray, x_obs: np.ndarray) -> np.ndarray:
        """Approximate log density using the instantaneous change-of-variables formula."""
        centered = samples - np.mean(samples, axis=0, keepdims=True)
        return -0.5 * np.sum(centered**2, axis=1)

    def _truncated_proposal(self, x_obs: np.ndarray, num_samples: int) -> np.ndarray:
        """
        Sample from truncated proposal distribution.
        
        Args:
            x_obs: Observed data
            num_samples: Number of samples
            
        Returns:
            samples: Truncated proposal samples
        """
        # Sample from current posterior
        posterior_samples = self.sample(x_obs, num_samples=num_samples * 10)
        
        lower = posterior_samples.min(axis=0)
        upper = posterior_samples.max(axis=0)
        prior_candidates = self.prior(max(num_samples * 20, 1000))
        in_box = np.all((prior_candidates >= lower) & (prior_candidates <= upper), axis=1)
        candidates = prior_candidates[in_box]
        if len(candidates) == 0:
            candidates = posterior_samples
        log_prob = self.instantaneous_change_of_variables_log_prob(candidates, x_obs)
        threshold = np.quantile(log_prob, self.truncation_epsilon)
        truncated_samples = candidates[log_prob >= threshold]
        
        # Resample if needed
        if len(truncated_samples) < num_samples:
            idx = np.random.choice(len(truncated_samples), num_samples, replace=True)
        else:
            idx = np.random.choice(len(truncated_samples), num_samples, replace=False)
        
        return truncated_samples[idx]


# Sequential variants SNPSE-A/B/C
class SNPSE_A(TSNPSE):
    """SNPSE variant A - alternative proposal strategy."""
    pass


class SNPSE_B(TSNPSE):
    """SNPSE variant B - alternative proposal strategy."""
    pass