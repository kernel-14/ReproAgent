"""
Sequential Neural Posterior Score Estimation - Simulator Environments

This module implements simulator interfaces, prior distributions, and data generation
for simulation-based inference benchmark tasks.

Reference grounding:
- paperbench_ref_001 sbi/sbi/inference/abc/abc_base.py: Simulator and prior interface

Paper: Sequential Neural Score Estimation: Likelihood-Free Inference with
       Conditional Score Based Diffusion Models

Implementation surfaces: data_pipeline, environment_adapter

Benchmark tasks from paper:
- two_moons: Bimodal 2D posterior (Figure 1)
- slcp: Simple Likelihood Complex Posterior 5D→8D (Figures 2, 3, 6)
- lotka_volterra: Ecological dynamics 4D→time series (Figures 4, 7)
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Callable, Optional, Tuple, Union
import warnings
import numpy as np


class Simulator(ABC):
    """
    Base simulator class for simulation-based inference.
    
    Reference grounding: paperbench_ref_001 sbi/sbi/inference/abc/abc_base.py
    Adapted from SBI library's simulator interface pattern.
    """
    
    def __init__(
        self,
        prior: Callable,
        dim_theta: int,
        dim_x: int,
        name: str = "simulator",
        batch_size: int = 1
    ):
        """
        Initialize simulator with prior and dimensions.
        
        Args:
            prior: Callable that samples from prior p(θ)
            dim_theta: Dimensionality of parameters θ
            dim_x: Dimensionality of observations x
            name: Simulator identifier
            batch_size: Default batch size for simulation
        """
        self.prior = prior
        self.dim_theta = dim_theta
        self.dim_x = dim_x
        self.name = name
        self.batch_size = batch_size
    
    @abstractmethod
    def simulate(self, theta: np.ndarray) -> np.ndarray:
        """
        Simulate observations x given parameters theta.
        
        Args:
            theta: Parameters of shape (batch_size, dim_theta) or (dim_theta,)
            
        Returns:
            Observations x of shape (batch_size, dim_x) or (dim_x,)
        """
        pass
    
    def sample_prior(self, num_samples: int = 1) -> np.ndarray:
        """Sample parameters from prior distribution p(θ)."""
        return self.prior(num_samples)
    
    def generate_batch(self, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate paired (theta, x) samples for training.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Tuple of (theta, x) arrays of shape (num_samples, dim_*)
        """
        theta = self.sample_prior(num_samples)
        x = self.simulate(theta)
        return theta, x


class TwoMoonsSimulator(Simulator):
    """
    Two Moons benchmark task from paper Figure 1.
    
    Bimodal 2D posterior with moon-shaped modes.
    Reference: Greenberg et al. (2019) "Automatic Posterior Transformation"
    """
    
    def __init__(self):
        def prior(n):
            return np.random.uniform(-1, 1, size=(n, 2))
        
        super().__init__(
            prior=prior,
            dim_theta=2,
            dim_x=2,
            name="two_moons"
        )
        self.a = 0.1
        self.b = 0.5
        self.noise_std = 0.01
    
    def simulate(self, theta: np.ndarray) -> np.ndarray:
        """
        Generate observations from two moons distribution.
        
        x = (alpha + r*cos(phi), r*sin(phi)) + noise
        where alpha depends on parameter quadrant.
        """
        if theta.ndim == 1:
            theta = theta.reshape(1, -1)
            squeeze = True
        else:
            squeeze = False
        
        batch_size = theta.shape[0]
        x = np.zeros((batch_size, 2))
        
        # Parameter-dependent angle
        angle = np.arctan2(theta[:, 1], theta[:, 0])
        
        # Radius with noise
        r = np.random.normal(self.b, self.noise_std, size=batch_size)
        
        # Horizontal offset based on angle quadrant
        alpha = np.where(angle < 0, -self.a, self.a)
        
        # Generate moon-shaped observations
        phi = np.random.uniform(-np.pi/2, np.pi/2, size=batch_size)
        x[:, 0] = alpha + r * np.cos(phi)
        x[:, 1] = r * np.sin(phi)
        
        # Add observation noise
        x += np.random.normal(0, self.noise_std, size=(batch_size, 2))
        
        return x.squeeze() if squeeze else x


class SLCPSimulator(Simulator):
    """
    Simple Likelihood Complex Posterior (SLCP) benchmark from paper Figures 2, 3, 6.
    
    5D parameters → 8D observations
    Reference: Papamakarios & Murray (2016) "Fast ε-free Inference of Simulation Models"
    """
    
    def __init__(self):
        def prior(n):
            # Uniform prior on [-3, 3]^5
            return np.random.uniform(-3, 3, size=(n, 5))
        
        super().__init__(
            prior=prior,
            dim_theta=5,
            dim_x=8,
            name="slcp"
        )
        self.noise_std = 0.1
    
    def simulate(self, theta: np.ndarray) -> np.ndarray:
        """
        Generate observations using SLCP likelihood.
        
        Simple nonlinear transformation with Gaussian noise.
        """
        if theta.ndim == 1:
            theta = theta.reshape(1, -1)
            squeeze = True
        else:
            squeeze = False
        
        batch_size = theta.shape[0]
        
        # Nonlinear transformations
        # x[0:4] = theta + noise
        # x[4:8] = theta^2 + cross-terms + noise
        x = np.zeros((batch_size, 8))
        
        # First 4 dimensions: linear
        x[:, 0] = theta[:, 0] + theta[:, 1]
        x[:, 1] = theta[:, 2] + theta[:, 3]
        x[:, 2] = theta[:, 0] * theta[:, 1]
        x[:, 3] = theta[:, 2] * theta[:, 3]
        
        # Last 4 dimensions: quadratic
        x[:, 4] = theta[:, 0]**2 + theta[:, 1]**2
        x[:, 5] = theta[:, 2]**2 + theta[:, 3]**2
        x[:, 6] = theta[:, 4]**2
        x[:, 7] = np.sum(theta**2, axis=1)
        
        # Add Gaussian noise
        x += np.random.normal(0, self.noise_std, size=(batch_size, 8))
        
        return x.squeeze() if squeeze else x


class LotkaVolterraSimulator(Simulator):
    """
    Lotka-Volterra ecological dynamics from paper Figures 4, 7.
    
    Predator-prey population dynamics with 4D parameters.
    Reference: Lueckmann et al. (2021) "Benchmarking Simulation-Based Inference"
    """
    
    def __init__(self, num_timesteps: int = 20, dt: float = 0.2):
        def prior(n):
            # Log-uniform priors for rate parameters
            log_theta = np.random.uniform(
                [np.log(0.01), np.log(0.01), np.log(0.01), np.log(0.01)],
                [np.log(1.0), np.log(1.0), np.log(1.0), np.log(1.0)],
                size=(n, 4)
            )
            return np.exp(log_theta)
        
        super().__init__(
            prior=prior,
            dim_theta=4,
            dim_x=num_timesteps * 2,  # 2 species × timesteps
            name="lotka_volterra"
        )
        self.num_timesteps = num_timesteps
        self.dt = dt
        self.x0 = np.array([30.0, 1.0])  # Initial populations
        self.noise_std = 0.5
    
    def simulate(self, theta: np.ndarray) -> np.ndarray:
        """
        Simulate Lotka-Volterra dynamics using Euler integration.
        
        dx/dt = alpha*x - beta*x*y
        dy/dt = delta*x*y - gamma*y
        
        theta = [alpha, beta, gamma, delta]
        """
        if theta.ndim == 1:
            theta = theta.reshape(1, -1)
            squeeze = True
        else:
            squeeze = False
        
        batch_size = theta.shape[0]
        
        # Initialize trajectories
        trajectories = np.zeros((batch_size, self.num_timesteps, 2))
        
        for i in range(batch_size):
            alpha, beta, gamma, delta = theta[i]
            
            # Initial state
            x = self.x0[0]
            y = self.x0[1]
            
            # Euler integration
            for t in range(self.num_timesteps):
                trajectories[i, t] = [x, y]
                
                # Lotka-Volterra equations
                dx_dt = alpha * x - beta * x * y
                dy_dt = delta * x * y - gamma * y
                
                # Update with Euler step
                x = max(0.0, x + self.dt * dx_dt)
                y = max(0.0, y + self.dt * dy_dt)
        
        # Flatten time series and add noise
        x_flat = trajectories.reshape(batch_size, -1)
        x_flat += np.random.normal(0, self.noise_std, size=x_flat.shape)
        
        return x_flat.squeeze() if squeeze else x_flat


class GaussianLinearSimulator(Simulator):
    """
    Gaussian linear model for quick testing and validation.
    
    x = A*theta + noise, with Gaussian prior on theta.
    """
    
    def __init__(self, dim_theta: int = 5, dim_x: int = 10):
        def prior(n):
            return np.random.randn(n, dim_theta)
        
        super().__init__(
            prior=prior,
            dim_theta=dim_theta,
            dim_x=dim_x,
            name="gaussian_linear"
        )
        # Random projection matrix
        self.A = np.random.randn(dim_x, dim_theta) / np.sqrt(dim_theta)
        self.noise_std = 0.1
    
    def simulate(self, theta: np.ndarray) -> np.ndarray:
        """Linear Gaussian likelihood x ~ N(A*theta, sigma^2*I)."""
        if theta.ndim == 1:
            theta = theta.reshape(1, -1)
            squeeze = True
        else:
            squeeze = False
        
        x = theta @ self.A.T
        x += np.random.normal(0, self.noise_std, size=x.shape)
        
        return x.squeeze() if squeeze else x

class GaussianLinearUniformSimulator(Simulator):
    """Gaussian Linear Uniform Appendix E.1 task with a uniform prior."""

    def __init__(self, dim_theta: int = 10, dim_x: int = 10):
        def prior(n):
            return np.random.uniform(-1.0, 1.0, size=(n, dim_theta))

        super().__init__(prior=prior, dim_theta=dim_theta, dim_x=dim_x, name="gaussian_linear_uniform")
        rng = np.random.default_rng(17)
        self.A = rng.normal(size=(dim_x, dim_theta)) / np.sqrt(dim_theta)
        self.noise_std = 0.1

    def simulate(self, theta: np.ndarray) -> np.ndarray:
        if theta.ndim == 1:
            theta = theta.reshape(1, -1)
            squeeze = True
        else:
            squeeze = False
        x = theta @ self.A.T + np.random.normal(0.0, self.noise_std, size=(theta.shape[0], self.dim_x))
        return x.squeeze() if squeeze else x


class BernoulliGLMSimulator(Simulator):
    """Bernoulli GLM Appendix E.1 task with logistic observations."""

    def __init__(self, dim_theta: int = 10, dim_x: int = 10):
        def prior(n):
            return np.random.uniform(-2.0, 2.0, size=(n, dim_theta))

        super().__init__(prior=prior, dim_theta=dim_theta, dim_x=dim_x, name="bernoulli_glm")
        rng = np.random.default_rng(23)
        self.design = rng.normal(size=(dim_x, dim_theta)) / np.sqrt(dim_theta)

    def simulate(self, theta: np.ndarray) -> np.ndarray:
        if theta.ndim == 1:
            theta = theta.reshape(1, -1)
            squeeze = True
        else:
            squeeze = False
        logits = theta @ self.design.T
        probs = 1.0 / (1.0 + np.exp(-logits))
        x = np.random.binomial(1, probs).astype(float)
        return x.squeeze() if squeeze else x


class SIRSimulator(Simulator):
    """SIR epidemiological Appendix E.1 task with time-series observations."""

    def __init__(self, num_timesteps: int = 20, dt: float = 0.1):
        def prior(n):
            beta = np.random.uniform(0.05, 1.0, size=(n, 1))
            gamma = np.random.uniform(0.02, 0.5, size=(n, 1))
            return np.hstack([beta, gamma])

        super().__init__(prior=prior, dim_theta=2, dim_x=num_timesteps * 3, name="sir")
        self.num_timesteps = num_timesteps
        self.dt = dt
        self.initial_state = np.array([0.99, 0.01, 0.0])
        self.noise_std = 0.01

    def simulate(self, theta: np.ndarray) -> np.ndarray:
        if theta.ndim == 1:
            theta = theta.reshape(1, -1)
            squeeze = True
        else:
            squeeze = False
        trajectories = np.zeros((theta.shape[0], self.num_timesteps, 3))
        for i, (beta, gamma) in enumerate(theta):
            s, infected, recovered = self.initial_state.copy()
            for t in range(self.num_timesteps):
                trajectories[i, t] = [s, infected, recovered]
                new_inf = beta * s * infected
                new_rec = gamma * infected
                s = np.clip(s - self.dt * new_inf, 0.0, 1.0)
                infected = np.clip(infected + self.dt * (new_inf - new_rec), 0.0, 1.0)
                recovered = np.clip(recovered + self.dt * new_rec, 0.0, 1.0)
        x = trajectories.reshape(theta.shape[0], -1)
        x += np.random.normal(0.0, self.noise_std, size=x.shape)
        return x.squeeze() if squeeze else x


class NeuroscienceSimulator(Simulator):
    """Low-dimensional neuroscience/pyloric-style Appendix E.1 benchmark surrogate."""

    def __init__(self, dim_theta: int = 8, dim_x: int = 15):
        def prior(n):
            return np.random.uniform(-2.0, 2.0, size=(n, dim_theta))

        super().__init__(prior=prior, dim_theta=dim_theta, dim_x=dim_x, name="neuroscience")
        rng = np.random.default_rng(31)
        self.W = rng.normal(size=(dim_x, dim_theta)) / np.sqrt(dim_theta)
        self.noise_std = 0.05

    def simulate(self, theta: np.ndarray) -> np.ndarray:
        if theta.ndim == 1:
            theta = theta.reshape(1, -1)
            squeeze = True
        else:
            squeeze = False
        features = np.tanh(theta @ self.W.T)
        features += 0.1 * np.sin(theta @ self.W.T)
        features += np.random.normal(0.0, self.noise_std, size=features.shape)
        return features.squeeze() if squeeze else features



# Environment Registry with paper-derived metadata
# Reference grounding: paperbench_ref_001 sbi/sbi/inference/abc/abc_base.py
ENVIRONMENT_REGISTRY: Dict[str, Dict[str, Any]] = {
    "two_moons": {
        "name": "Two Moons",
        "alias": ["two_moons", "twomoons", "moons"],
        "paper_figure": "Figure 1",
        "dim_theta": 2,
        "dim_x": 2,
        "factory": lambda: TwoMoonsSimulator(),
        "description": "Bimodal posterior with moon-shaped modes",
        "difficulty": "easy",
        "sequential": False,
        "metadata": {
            "reference": "Greenberg et al. (2019)",
            "task_type": "toy_benchmark"
        }
    },
    "slcp": {
        "name": "Simple Likelihood Complex Posterior",
        "alias": ["slcp", "SLCP"],
        "paper_figure": "Figures 2, 3, 6",
        "dim_theta": 5,
        "dim_x": 8,
        "factory": lambda: SLCPSimulator(),
        "description": "Simple nonlinear likelihood, complex posterior geometry",
        "difficulty": "medium",
        "sequential": True,
        "metadata": {
            "reference": "Papamakarios & Murray (2016)",
            "task_type": "benchmark",
            "challenging_for": ["sequential_methods", "complex_posterior"]
        }
    },
    "lotka_volterra": {
        "name": "Lotka-Volterra Predator-Prey",
        "alias": ["lotka_volterra", "lv", "lotka", "predator_prey"],
        "paper_figure": "Figures 4, 7",
        "dim_theta": 4,
        "dim_x": 40,  # 20 timesteps × 2 species
        "factory": lambda: LotkaVolterraSimulator(),
        "description": "Ecological dynamics with complex temporal dependencies",
        "difficulty": "hard",
        "sequential": True,
        "metadata": {
            "reference": "Lueckmann et al. (2021)",
            "task_type": "challenging_benchmark",
            "time_series": True,
            "challenging_for": ["likelihood_free", "temporal_dependencies"]
        }
    },
    "gaussian_linear": {
        "name": "Gaussian Linear Model",
        "alias": ["gaussian_linear", "linear", "gaussian"],
        "paper_figure": None,
        "dim_theta": 5,
        "dim_x": 10,
        "factory": lambda: GaussianLinearSimulator(),
        "description": "Simple Gaussian linear model for validation",
        "difficulty": "easy",
        "sequential": False,
        "metadata": {
            "task_type": "validation",
            "analytic_posterior": True
        }
    },
    "gaussian_linear_uniform": {
        "name": "Gaussian Linear Uniform",
        "alias": ["gaussian_linear_uniform", "glu", "gaussian_uniform"],
        "paper_figure": "Appendix E.1",
        "dim_theta": 10,
        "dim_x": 10,
        "factory": lambda: GaussianLinearUniformSimulator(),
        "description": "Linear Gaussian simulator with uniform prior",
        "difficulty": "easy",
        "sequential": False,
        "metadata": {"task_type": "appendix_e1", "prior": "uniform"}
    },
    "bernoulli_glm": {
        "name": "Bernoulli GLM",
        "alias": ["bernoulli_glm", "bernoulli", "glm"],
        "paper_figure": "Appendix E.1",
        "dim_theta": 10,
        "dim_x": 10,
        "factory": lambda: BernoulliGLMSimulator(),
        "description": "Logistic Bernoulli generalized linear model simulator",
        "difficulty": "medium",
        "sequential": False,
        "metadata": {"task_type": "appendix_e1", "likelihood": "bernoulli"}
    },
    "sir": {
        "name": "SIR",
        "alias": ["sir", "epidemiology"],
        "paper_figure": "Appendix E.1",
        "dim_theta": 2,
        "dim_x": 60,
        "factory": lambda: SIRSimulator(),
        "description": "Susceptible-infected-recovered dynamical system simulator",
        "difficulty": "medium",
        "sequential": True,
        "metadata": {"task_type": "appendix_e1", "time_series": True}
    },
    "neuroscience": {
        "name": "Neuroscience",
        "alias": ["neuroscience", "pyloric", "l5pc"],
        "paper_figure": "Appendix E.1 / Section 5.3",
        "dim_theta": 8,
        "dim_x": 15,
        "factory": lambda: NeuroscienceSimulator(),
        "description": "Neuroscience benchmark surrogate with nonlinear summary features",
        "difficulty": "hard",
        "sequential": True,
        "metadata": {"task_type": "appendix_e1", "realistic_benchmark": True}
    }
}


def get_simulator(task_name: str) -> Simulator:
    """
    Factory function to create simulator by task name or alias.
    
    Args:
        task_name: Task identifier (name or alias)
        
    Returns:
        Simulator instance for the specified task
        
    Raises:
        ValueError: If task name not found in registry
    """
    # Direct lookup
    if task_name in ENVIRONMENT_REGISTRY:
        return ENVIRONMENT_REGISTRY[task_name]["factory"]()
    
    # Search aliases
    for env_id, env_config in ENVIRONMENT_REGISTRY.items():
        if task_name in env_config.get("alias", []):
            return env_config["factory"]()
    
    raise ValueError(
        f"Unknown task '{task_name}'. Available tasks: "
        f"{list(ENVIRONMENT_REGISTRY.keys())}"
    )


def generate_training_data(
    task_name: str,
    num_samples: int,
    output_dir: Optional[str] = None,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate training data for a specified task.
    
    Args:
        task_name: Task identifier
        num_samples: Number of (theta, x) pairs to generate
        output_dir: Optional directory to save data
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (theta, x) arrays
    """
    if seed is not None:
        np.random.seed(seed)
    
    simulator = get_simulator(task_name)
    theta, x = simulator.generate_batch(num_samples)
    
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        save_path = output_path / f"{task_name}_train.npz"
        np.savez(
            save_path,
            theta=theta,
            x=x,
            task_name=task_name,
            num_samples=num_samples,
            seed=seed
        )
        print(f"Saved training data to {save_path}")
    
    return theta, x


def generate_test_data(
    task_name: str,
    num_samples: int,
    output_dir: Optional[str] = None,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate test data for evaluation.
    
    Args:
        task_name: Task identifier
        num_samples: Number of test samples
        output_dir: Optional directory to save data
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (theta, x) arrays
    """
    if seed is not None:
        np.random.seed(seed)
    
    simulator = get_simulator(task_name)
    theta, x = simulator.generate_batch(num_samples)
    
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        save_path = output_path / f"{task_name}_test.npz"
        np.savez(
            save_path,
            theta=theta,
            x=x,
            task_name=task_name,
            num_samples=num_samples,
            seed=seed
        )
        print(f"Saved test data to {save_path}")
    
    return theta, x


def create_smoke_artifacts():
    """
    Create minimal validation artifacts for contract verification.
    
    Generates small datasets for each registered environment to verify
    simulator implementations and data pipeline.
    """
    artifact_dir = os.environ.get('PAPERBENCH_REPRO_ARTIFACT_DIR', 'simulated_data')
    output_dir = Path(artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating smoke test artifacts in {output_dir}")
    
    # Generate small validation datasets
    for task_name in ENVIRONMENT_REGISTRY.keys():
        print(f"  Generating {task_name} data...")
        generate_training_data(task_name, num_samples=100, output_dir=output_dir, seed=42)
        generate_test_data(task_name, num_samples=50, output_dir=output_dir, seed=43)
    
    # Create readiness manifest
    manifest = {
        "status": "smoke_validation",
        "description": "Minimal simulator validation artifacts",
        "environments": list(ENVIRONMENT_REGISTRY.keys()),
        "note": "These are dry-run contract artifacts, not experiment results"
    }
    
    with open(output_dir / "readiness.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Smoke artifacts created successfully")


if __name__ == "__main__":
    # Smoke test: verify all simulators can be instantiated and run
    print("Running simulator smoke tests...")
    
    for task_name in ENVIRONMENT_REGISTRY.keys():
        print(f"\nTesting {task_name}...")
        simulator = get_simulator(task_name)
        theta, x = simulator.generate_batch(10)
        print(f"  Generated batch: theta shape {theta.shape}, x shape {x.shape}")
        assert theta.shape == (10, simulator.dim_theta)
        assert x.shape == (10, simulator.dim_x)
    
    print("\nAll simulators validated successfully!")
    
    # Create minimal artifacts
    create_smoke_artifacts()

def generate_test_observation(simulator: Simulator, task_config: Dict[str, Any], seed: int = 0) -> np.ndarray:
    """Generate a single observation for posterior evaluation."""
    rng_state = np.random.get_state()
    np.random.seed(seed)
    try:
        theta, x = simulator.generate_batch(1)
        return np.asarray(x[0])
    finally:
        np.random.set_state(rng_state)

