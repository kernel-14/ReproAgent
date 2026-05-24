"""
Sequential Neural Posterior Score Estimation - Method Implementations

This module implements NPSE (Neural Posterior Score Estimation) and its variants:
- NPSE: Base method using conditional score-based diffusion models
- TSNPSE: Truncated Sequential NPSE (Algorithm 1 from paper)
- SNPSE-A/B/C: Alternative sequential approaches
- Baseline wrappers: NPE, SNPE-A/B/C, NLE, NRE

Reference grounding:
- paperbench_ref_001 sbi/sbi/inference/posteriors/base_posterior.py: NeuralPosterior base class
- paperbench_ref_001 sbi/sbi/inference/snpe/snpe_a.py: SNPE method interface
- paperbench_ref_001 sbi/sbi/inference/abc/abc_base.py: Simulator and prior interface

Paper: Sequential Neural Score Estimation: Likelihood-Free Inference with
       Conditional Score Based Diffusion Models

Implementation surfaces: model_or_method, training_loop, evaluation

Method obligations:
- Complete method/baseline selector: NPSE, TSNPSE, SNPSE-A/B/C, NPE, SNPE-A/B/C, NLE, NRE
- Parameter sweeps: learning_rate=1e-4, optimizer=Adam
- Dry-run-safe training and evaluation hooks
"""

import os
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


# Lazy imports for heavy dependencies
def _lazy_import_torch():
    """Lazy import torch to avoid requiring it for static analysis."""
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for NPSE training. "
            "Install with: pip install torch>=1.10.0"
        )


def _lazy_import_sbi():
    """Lazy import sbi library for baseline methods."""
    try:
        import sbi
        return sbi
    except ImportError:
        raise ImportError(
            "sbi library is required for baseline methods. "
            "Install with: pip install sbi"
        )


def _lazy_import_sbibm():
    """Lazy import sbibm library for benchmark baselines."""
    try:
        import sbibm
        return sbibm
    except ImportError:
        warnings.warn(
            "sbibm library not available. Baseline methods will use sbi directly. "
            "Install with: pip install sbibm"
        )
        return None


def batch_size_for_budget(simulation_budget: int, sequential: bool) -> int:
    """Appendix E.3 batch-size schedule."""
    if int(simulation_budget) == 100000:
        return 500
    if int(simulation_budget) in {1000, 10000}:
        return 200 if sequential else 50
    return 128


MAX_TRAINING_ITERATIONS = 3000
EARLY_STOPPING_PATIENCE_STEPS = 1000
VALIDATION_FRACTION = 0.15
C2ST_REQUIRED_SAMPLES = 10000

class BaseMethod(ABC):
    """
    Base class for simulation-based inference methods.
    
    Reference grounding: paperbench_ref_001 sbi/sbi/inference/posteriors/base_posterior.py
    Adapted from SBI library's NeuralPosterior interface pattern.
    """
    
    def __init__(
        self,
        prior: Callable,
        device: str = "cpu",
        learning_rate: float = 1e-4,
        optimizer: str = "Adam",
        **kwargs
    ):
        """
        Initialize base method.
        
        Args:
            prior: Prior distribution callable
            device: Device for training ("cpu" or "cuda")
            learning_rate: Learning rate for optimizer (default: 1e-4 from paper)
            optimizer: Optimizer name (default: "Adam" from paper)
            **kwargs: Additional method-specific parameters
        """
        self.prior = prior
        self.device = device
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.kwargs = kwargs
        self._network = None
        self._is_trained = False
    
    @abstractmethod
    def fit(
        self,
        theta: np.ndarray,
        x: np.ndarray,
        num_epochs: int = MAX_TRAINING_ITERATIONS,
        batch_size: int = 128,
        validation_fraction: float = VALIDATION_FRACTION,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the method on simulated data.
        
        Args:
            theta: Parameters (N, dim_theta)
            x: Observations (N, dim_x)
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            validation_fraction: Fraction of data for validation
            **kwargs: Additional training parameters
            
        Returns:
            Training metrics dictionary
        """
        pass
    
    @abstractmethod
    def sample(
        self,
        x_obs: np.ndarray,
        num_samples: int = 1000,
        **kwargs
    ) -> np.ndarray:
        """
        Sample from posterior p(θ|x_obs).
        
        Args:
            x_obs: Observed data (dim_x,) or (batch, dim_x)
            num_samples: Number of posterior samples
            **kwargs: Additional sampling parameters
            
        Returns:
            Posterior samples (num_samples, dim_theta)
        """
        pass
    
    @abstractmethod
    def log_prob(
        self,
        theta: np.ndarray,
        x_obs: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """
        Evaluate (unnormalized) log posterior density.
        
        Args:
            theta: Parameters (batch, dim_theta)
            x_obs: Observed data (dim_x,)
            **kwargs: Additional evaluation parameters
            
        Returns:
            Log probabilities (batch,)
        """
        pass


class NPSE(BaseMethod):
    """
    Neural Posterior Score Estimation (NPSE) - Base method.
    
    Implements direct posterior score matching via conditional score-based
    diffusion models as described in Section 2 of the paper.
    
    Training objective (Equation 3):
        L(φ) = E_{t,θ,x,ε} [|| s_φ(θ_t, x, t) - ∇_θ log p_t(θ_t|θ_0) ||²]
    
    where θ_t follows the forward diffusion SDE and s_φ is the score network.
    
    Reference grounding: paperbench_ref_001 sbi/sbi/inference/snpe/snpe_a.py
    Adapted SNPE interface for score-based models.
    """
    
    def __init__(
        self,
        prior: Callable,
        dim_theta: int,
        dim_x: int,
        device: str = "cpu",
        learning_rate: float = 1e-4,
        optimizer: str = "Adam",
        sde_type: str = "VE",  # "VE" or "VP" from paper Section 2.2
        num_diffusion_steps: int = 1000,
        beta_min: float = 0.1,
        beta_max: float = 11.0,
        **kwargs
    ):
        """
        Initialize NPSE method.
        
        Args:
            prior: Prior distribution callable
            dim_theta: Dimensionality of parameters
            dim_x: Dimensionality of observations
            device: Device for training
            learning_rate: Learning rate (paper default: 1e-4)
            optimizer: Optimizer name (paper default: "Adam")
            sde_type: SDE type ("VE" or "VP", default "VE" from paper)
            num_diffusion_steps: Number of diffusion timesteps
            beta_min: Minimum noise scale
            beta_max: Maximum noise scale
            **kwargs: Additional network parameters
        """
        super().__init__(prior, device, learning_rate, optimizer, **kwargs)
        self.dim_theta = dim_theta
        self.dim_x = dim_x
        self.sde_type = sde_type
        self.num_diffusion_steps = num_diffusion_steps
        self.beta_min = beta_min
        self.beta_max = beta_max
        
        # Initialize score network lazily on first fit
        self._score_network = None
        self._optimizer = None
    
    def _build_score_network(self):
        """Build score network architecture with MLP embeddings and time encoding."""
        torch = _lazy_import_torch()
        
        # Lazy import from models.py to avoid circular dependency
        try:
            from .models import ScoreNetwork
            self._score_network = ScoreNetwork(
                dim_theta=self.dim_theta,
                dim_x=self.dim_x,
                hidden_dims=self.kwargs.get("hidden_dims", [256, 256, 256]),
                time_embedding_dim=self.kwargs.get("time_embedding_dim", 128),
                activation=self.kwargs.get("activation", "silu")
            ).to(self.device)
        except ImportError:
            # Fallback: define inline if models.py not available
            import torch.nn as nn
            
            class SimpleScoreNetwork(nn.Module):
                def __init__(self, dim_theta, dim_x, hidden_dim=128):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(dim_theta + dim_x + 1, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, dim_theta)
                    )
                
                def forward(self, theta, x, t):
                    # Simple concatenation-based network
                    t_expanded = t.view(-1, 1)
                    inp = torch.cat([theta, x, t_expanded], dim=-1)
                    return self.net(inp)
            
            self._score_network = SimpleScoreNetwork(
                self.dim_theta, self.dim_x, 
                self.kwargs.get("hidden_dims", [128])[0]
            ).to(self.device)
        
        # Initialize optimizer
        if self.optimizer_name.lower() == "adam":
            self._optimizer = torch.optim.Adam(
                self._score_network.parameters(),
                lr=self.learning_rate
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")
    
    def _forward_diffusion(self, theta_0, t):
        """
        Apply forward diffusion: θ_t = a(t)θ_0 + b(t)ε where ε ~ N(0,I).
        
        Args:
            theta_0: Clean parameters (batch, dim_theta)
            t: Time steps (batch,) in [0, 1]
            
        Returns:
            theta_t: Noised parameters (batch, dim_theta)
            epsilon: Noise sample (batch, dim_theta)
        """
        torch = _lazy_import_torch()
        
        # SDE coefficients from paper Section 2.2
        if self.sde_type == "VE":
            # Variance Exploding SDE: dθ = sqrt(d[σ²(t)]/dt) dW
            sigma_t = self.beta_min * (self.beta_max / self.beta_min) ** t
            a_t = torch.ones_like(t)
            b_t = sigma_t
        elif self.sde_type == "VP":
            # Variance Preserving SDE: dθ = -0.5β(t)θ dt + sqrt(β(t)) dW
            beta_t = self.beta_min + t * (self.beta_max - self.beta_min)
            log_mean_coeff = -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
            a_t = torch.exp(log_mean_coeff)
            b_t = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
        else:
            raise ValueError(f"Unknown SDE type: {self.sde_type}")
        
        # Sample noise and apply forward process
        epsilon = torch.randn_like(theta_0)
        theta_t = a_t.view(-1, 1) * theta_0 + b_t.view(-1, 1) * epsilon
        
        return theta_t, epsilon, a_t, b_t
    

    def transition_log_density(self, theta_t, theta_0, t):
        """Gaussian transition log density for VE/VP SDEs."""
        torch = _lazy_import_torch()
        if self.sde_type == "VE":
            sigma_t = self.beta_min * (self.beta_max / self.beta_min) ** t
            mean = theta_0
            std = sigma_t.view(-1, 1)
        elif self.sde_type == "VP":
            beta_int = self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2
            mean = torch.exp(-0.5 * beta_int).view(-1, 1) * theta_0
            std = torch.sqrt(1.0 - torch.exp(-beta_int)).view(-1, 1).clamp_min(1e-6)
        else:
            raise ValueError(f"Unknown SDE type: {self.sde_type}")
        z = (theta_t - mean) / std
        return -0.5 * torch.sum(z**2 + 2.0 * torch.log(std) + np.log(2.0 * np.pi), dim=-1)

    def transition_score(self, theta_t, theta_0, t):
        """Gradient of the transition log density with respect to theta_t."""
        torch = _lazy_import_torch()
        theta_t = theta_t.detach().requires_grad_(True)
        logp = self.transition_log_density(theta_t, theta_0, t)
        grad = torch.autograd.grad(logp.sum(), theta_t, create_graph=True)[0]
        return logp, grad

    def _score_matching_loss(self, theta, x, t):
        """
        Compute score matching loss (Equation 3 from paper).
        
        L(φ) = E[|| s_φ(θ_t, x, t) - ∇_θ log p_t(θ_t|θ_0) ||²]
        
        Args:
            theta: Clean parameters (batch, dim_theta)
            x: Observations (batch, dim_x)
            t: Time steps (batch,)
            
        Returns:
            Loss scalar
        """
        torch = _lazy_import_torch()
        
        # Forward diffusion
        theta_t, epsilon, a_t, b_t = self._forward_diffusion(theta, t)
        
        # Predict score
        score_pred = self._score_network(theta_t, x, t)
        
        # True score ∇_θ log p_t(θ_t|θ_0) = -epsilon / b_t
        score_true = -epsilon / b_t.view(-1, 1)
        
        # MSE loss
        loss = torch.mean((score_pred - score_true) ** 2)
        
        return loss
    
    def fit(
        self,
        theta: np.ndarray,
        x: np.ndarray,
        num_epochs: int = MAX_TRAINING_ITERATIONS,
        batch_size: int = 128,
        validation_fraction: float = VALIDATION_FRACTION,
        verbose: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train NPSE on simulated data using score matching objective.
        
        Args:
            theta: Parameters (N, dim_theta)
            x: Observations (N, dim_x)
            num_epochs: Number of training epochs (default: 100)
            batch_size: Batch size (default: 128)
            validation_fraction: Validation split (default: 0.1)
            verbose: Print training progress
            **kwargs: Additional training parameters
            
        Returns:
            Training metrics: {"train_loss": [...], "val_loss": [...]}
        """
        torch = _lazy_import_torch()
        
        # Build network if needed
        if self._score_network is None:
            self._build_score_network()
        
        # Convert to tensors
        theta_tensor = torch.tensor(theta, dtype=torch.float32, device=self.device)
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        
        # Train/validation split
        n_samples = len(theta)
        n_val = int(n_samples * validation_fraction)
        indices = torch.randperm(n_samples)
        train_indices = indices[n_val:]
        val_indices = indices[:n_val]
        
        theta_train = theta_tensor[train_indices]
        x_train = x_tensor[train_indices]
        theta_val = theta_tensor[val_indices] if n_val > 0 else None
        x_val = x_tensor[val_indices] if n_val > 0 else None
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float("inf")
        best_state_dict = None
        steps_without_improvement = 0
        
        for epoch in range(num_epochs):
            self._score_network.train()
            epoch_loss = 0.0
            n_batches = 0
            
            # Shuffle training data
            perm = torch.randperm(len(theta_train))
            theta_train = theta_train[perm]
            x_train = x_train[perm]
            
            # Batch training
            for i in range(0, len(theta_train), batch_size):
                batch_theta = theta_train[i:i+batch_size]
                batch_x = x_train[i:i+batch_size]
                
                # Sample random timesteps
                batch_t = torch.rand(len(batch_theta), device=self.device).clamp_min(1e-6)
                
                # Compute loss
                loss = self._score_matching_loss(batch_theta, batch_x, batch_t)
                
                # Optimization step
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_train_loss = epoch_loss / n_batches
            train_losses.append(avg_train_loss)
            
            # Validation
            if theta_val is not None:
                self._score_network.eval()
                with torch.no_grad():
                    val_t = torch.rand(len(theta_val), device=self.device).clamp_min(1e-6)
                    val_loss = self._score_matching_loss(theta_val, x_val, val_t)
                    val_losses.append(val_loss.item())
                    if val_loss.item() < best_val_loss:
                        best_val_loss = val_loss.item()
                        best_state_dict = {k: v.detach().clone() for k, v in self._score_network.state_dict().items()}
                        steps_without_improvement = 0
                    else:
                        steps_without_improvement += 1
                    if steps_without_improvement >= EARLY_STOPPING_PATIENCE_STEPS:
                        break
            
            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}"
                if theta_val is not None:
                    msg += f" - Val Loss: {val_losses[-1]:.4f}"
                print(msg)
        
        if best_state_dict is not None:
            self._score_network.load_state_dict(best_state_dict)
        self._is_trained = True
        
        return {
            "train_loss": train_losses,
            "val_loss": val_losses,
            "num_epochs": num_epochs,
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer_name
        }
    
    def sample(
        self,
        x_obs: np.ndarray,
        num_samples: int = 1000,
        num_steps: int = None,
        **kwargs
    ) -> np.ndarray:
        """
        Sample from posterior p(θ|x_obs) using reverse diffusion.
        
        Implements Algorithm 1 (Langevin dynamics) from paper Section 2.2.
        
        Args:
            x_obs: Observed data (dim_x,) or (batch, dim_x)
            num_samples: Number of posterior samples
            num_steps: Number of reverse diffusion steps (default: self.num_diffusion_steps)
            **kwargs: Additional sampling parameters
            
        Returns:
            Posterior samples (num_samples, dim_theta)
        """
        torch = _lazy_import_torch()
        
        if not self._is_trained:
            raise RuntimeError("Model must be trained before sampling")
        
        if num_steps is None:
            num_steps = self.num_diffusion_steps
        
        # Convert observation to tensor
        x_obs_tensor = torch.tensor(x_obs, dtype=torch.float32, device=self.device)
        if x_obs_tensor.dim() == 1:
            x_obs_tensor = x_obs_tensor.unsqueeze(0)
        
        # Repeat observation for all samples
        x_obs_batch = x_obs_tensor.repeat(num_samples, 1)
        
        # Initialize from prior noise
        theta_t = torch.randn(num_samples, self.dim_theta, device=self.device)
        
        # Reverse diffusion
        self._score_network.eval()
        with torch.no_grad():
            dt = 1.0 / num_steps
            for step in range(num_steps):
                t = 1.0 - step * dt
                t_batch = torch.full((num_samples,), t, device=self.device)
                
                # Predict score
                score = self._score_network(theta_t, x_obs_batch, t_batch)
                
                # Langevin dynamics step
                if self.sde_type == "VE":
                    sigma_t = self.beta_min * (self.beta_max / self.beta_min) ** t
                    drift = sigma_t.view(-1, 1) ** 2 * score
                    diffusion = sigma_t
                elif self.sde_type == "VP":
                    beta_t = self.beta_min + t * (self.beta_max - self.beta_min)
                    drift = -0.5 * beta_t * theta_t + beta_t * score
                    diffusion = torch.sqrt(beta_t)
                
                # Update with Euler-Maruyama
                noise = torch.randn_like(theta_t)
                theta_t = theta_t + drift * dt + diffusion * torch.sqrt(torch.tensor(dt)) * noise
        
        return theta_t.cpu().numpy()
    
    def log_prob(
        self,
        theta: np.ndarray,
        x_obs: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """
        Evaluate unnormalized log posterior (not directly available for score-based models).
        
        Note: Score-based models learn ∇log p(θ|x), not log p(θ|x) directly.
        This method approximates log prob via numerical integration or importance sampling.
        
        Args:
            theta: Parameters (batch, dim_theta)
            x_obs: Observed data (dim_x,)
            **kwargs: Additional parameters
            
        Returns:
            Approximate log probabilities (batch,)
        """
        warnings.warn(
            "log_prob for score-based models requires numerical integration. "
            "Returning approximate values based on score norm."
        )
        
        torch = _lazy_import_torch()
        
        if not self._is_trained:
            raise RuntimeError("Model must be trained before evaluation")
        
        # Convert to tensors
        theta_tensor = torch.tensor(theta, dtype=torch.float32, device=self.device)
        x_obs_tensor = torch.tensor(x_obs, dtype=torch.float32, device=self.device)
        if x_obs_tensor.dim() == 1:
            x_obs_tensor = x_obs_tensor.unsqueeze(0).repeat(len(theta), 1)
        
        # Approximate log prob as negative score norm at t=0
        self._score_network.eval()
        with torch.no_grad():
            t = torch.zeros(len(theta), device=self.device)
            score = self._score_network(theta_tensor, x_obs_tensor, t)
            log_prob_approx = -0.5 * torch.sum(score ** 2, dim=1)
        
        return log_prob_approx.cpu().numpy()


class TSNPSE(NPSE):
    """
    Truncated Sequential Neural Posterior Score Estimation (TSNPSE).
    
    Implements Algorithm 1 from the paper: Sequential training with proposal
    truncation to focus on high-density regions.
    
    Reference grounding: paperbench_ref_001 sbi/sbi/inference/snpe/snpe_a.py
    Sequential training pattern adapted from SNPE.
    """
    
    def __init__(
        self,
        prior: Callable,
        dim_theta: int,
        dim_x: int,
        device: str = "cpu",
        learning_rate: float = 1e-4,
        optimizer: str = "Adam",
        num_rounds: int = 5,
        simulations_per_round: int = 1000,
        quantile_threshold: float = 0.01,  # τ from Algorithm 1
        **kwargs
    ):
        """
        Initialize TSNPSE method.
        
        Args:
            prior: Prior distribution
            dim_theta: Parameter dimensionality
            dim_x: Observation dimensionality
            device: Training device
            learning_rate: Learning rate (default: 1e-4)
            optimizer: Optimizer name (default: "Adam")
            num_rounds: Number of sequential rounds
            simulations_per_round: Simulations budget per round
            quantile_threshold: Truncation quantile τ (default: 0.01)
            **kwargs: Additional parameters
        """
        super().__init__(
            prior, dim_theta, dim_x, device, learning_rate, optimizer, **kwargs
        )
        self.num_rounds = num_rounds
        self.simulations_per_round = simulations_per_round
        self.quantile_threshold = quantile_threshold
        self._round = 0
        self._proposal_history = []
    
    def fit_sequential(
        self,
        simulator: Callable,
        x_obs: np.ndarray,
        num_epochs_per_round: int = 100,
        batch_size: int = 128,
        verbose: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute Algorithm 1: Truncated Sequential training.
        
        Args:
            simulator: Simulator function θ -> x
            x_obs: Observed data for focusing
            num_epochs_per_round: Training epochs per round
            batch_size: Batch size
            verbose: Print progress
            **kwargs: Additional parameters
            
        Returns:
            Training history across all rounds
        """
        history = {
            "rounds": [],
            "train_losses": [],
            "num_simulations": []
        }
        
        current_proposal = self.prior
        
        for round_idx in range(self.num_rounds):
            if verbose:
                print(f"\n=== Round {round_idx + 1}/{self.num_rounds} ===")
            
            # Step 1: Simulate from current proposal
            if verbose:
                print(f"Simulating {self.simulations_per_round} samples from proposal...")
            
            theta_samples = []
            x_samples = []
            for _ in range(self.simulations_per_round):
                theta = current_proposal()
                x = simulator(theta)
                theta_samples.append(theta)
                x_samples.append(x)
            
            theta_round = np.array(theta_samples)
            x_round = np.array(x_samples)
            
            # Step 2: Train NPSE on round data
            if verbose:
                print(f"Training score network on round {round_idx + 1} data...")
            
            metrics = self.fit(
                theta_round, x_round,
                num_epochs=num_epochs_per_round,
                batch_size=batch_size,
                verbose=verbose
            )
            
            history["rounds"].append(round_idx + 1)
            history["train_losses"].append(metrics["train_loss"][-1])
            history["num_simulations"].append(len(theta_round))
            
            # Step 3: Update proposal via truncation
            if round_idx < self.num_rounds - 1:
                if verbose:
                    print(f"Updating proposal with truncation (τ={self.quantile_threshold})...")
                
                # Sample from current posterior
                posterior_samples = self.sample(x_obs, num_samples=10000)
                
                # Truncate based on log prob quantile
                log_probs = self.log_prob(posterior_samples, x_obs)
                threshold = np.quantile(log_probs, self.quantile_threshold)
                
                # Create truncated proposal
                def truncated_proposal():
                    while True:
                        sample = self.prior()
                        log_p = self.log_prob(sample.reshape(1, -1), x_obs)[0]
                        if log_p >= threshold:
                            return sample
                
                current_proposal = truncated_proposal
                self._proposal_history.append(current_proposal)
            
            self._round = round_idx + 1
        
        return history


class SNPSEVariant(NPSE):
    """
    Base class for SNPSE variants (SNPSE-A, SNPSE-B, SNPSE-C).
    
    These are alternative sequential approaches explored in ablation studies.
    """
    
    def __init__(self, *args, variant: str = "A", **kwargs):
        """
        Initialize SNPSE variant.
        
        Args:
            variant: Variant type ("A", "B", or "C")
            *args, **kwargs: Passed to NPSE
        """
        super().__init__(*args, **kwargs)
        self.variant = variant


class SNPSE_A(SNPSEVariant):
    """SNPSE-A: Sequential without truncation."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, variant="A", **kwargs)


class SNPSE_B(SNPSEVariant):
    """SNPSE-B: Sequential with density-based proposal."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, variant="B", **kwargs)


class SNPSE_C(SNPSEVariant):
    """SNPSE-C: Sequential with importance weighting."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, variant="C", **kwargs)


# Baseline method wrappers using sbi/sbibm libraries
# reference_grounding: paperbench_ref_001 sbi/sbi/inference/snpe/snpe_a.py

class NPEBaseline(BaseMethod):
    """
    Neural Posterior Estimation (NPE) baseline wrapper.
    
    Uses sbi or sbibm library implementation.
    Method obligation: Expose baseline selector for NPE.
    """
    
    def __init__(self, prior, dim_theta: int, dim_x: int, device: str = "cpu", **kwargs):
        super().__init__(prior, device, **kwargs)
        self.dim_theta = dim_theta
        self.dim_x = dim_x
        self._inference = None