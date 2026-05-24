import os
import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Callable

# reference_grounding: paper:paper_addendum_constraints
# The sbibm library should be used to implement the NPE and SNPE methods.
# The sbibm library should be used to implement the C2ST method.

@dataclass
class SbiEnvironmentSpec:
    """
    Specification for an SBI benchmark environment.
    reference_grounding: paper:unit_005
    """
    task_id: str
    name: str
    dim_theta: int
    dim_x: int
    num_rounds: int = 10
    budget_per_round: int = 1000
    total_budget: int = 10000
    x_obs_idx: int = 0
    description: Optional[str] = None

# reference_grounding: paper:paper_dataset_inventory
# Explicitly register dataset/benchmark aliases for slcp, lotka_volterra, two_moons.
SBI_BENCHMARK_REGISTRY = {
    "two_moons": {
        "task_id": "two_moons",
        "name": "Two Moons",
        "dim_theta": 2,
        "dim_x": 2,
    },
    "slcp": {
        "task_id": "slcp",
        "name": "SLCP",
        "dim_theta": 5,
        "dim_x": 8,
    },
    "lotka_volterra": {
        "task_id": "lotka_volterra",
        "name": "Lotka-Volterra",
        "dim_theta": 4,
        "dim_x": 9,
    },
    "gaussian_linear": {
        "task_id": "gaussian_linear",
        "name": "Gaussian Linear",
        "dim_theta": 10,
        "dim_x": 10,
    },
    "gaussian_linear_uniform": {
        "task_id": "gaussian_linear_uniform",
        "name": "Gaussian Linear Uniform",
        "dim_theta": 10,
        "dim_x": 10,
    },
    "gaussian_mixture": {
        "task_id": "gaussian_mixture",
        "name": "Gaussian Mixture",
        "dim_theta": 2,
        "dim_x": 2,
    },
    "sir": {
        "task_id": "sir",
        "name": "SIR",
        "dim_theta": 2,
        "dim_x": 10,
    },
    "bernoulli_glm": {
        "task_id": "bernoulli_glm",
        "name": "Bernoulli GLM",
        "dim_theta": 10,
        "dim_x": 10,
    }
}

def check_sbi_environment_available() -> bool:
    """Checks if the sbibm library is available."""
    try:
        import sbibm
        return True
    except ImportError:
        return False

def make_sbi_environment(task_name: str, budget: int = 10000, num_rounds: int = 10) -> SbiEnvironmentSpec:
    """
    Factory for creating environment specifications.
    reference_grounding: paper:unit_005
    """
    if task_name not in SBI_BENCHMARK_REGISTRY:
        raise ValueError(f"Task {task_name} not found in registry.")
    
    base_spec = SBI_BENCHMARK_REGISTRY[task_name]
    return SbiEnvironmentSpec(
        task_id=base_spec["task_id"],
        name=base_spec["name"],
        dim_theta=base_spec["dim_theta"],
        dim_x=base_spec["dim_x"],
        num_rounds=num_rounds,
        budget_per_round=budget // num_rounds,
        total_budget=budget
    )

def load_sbi_environment(task_name: str):
    """
    Loads the sbibm task object.
    reference_grounding: paper:paper_addendum_constraints
    """
    if not check_sbi_environment_available():
        raise ImportError("sbibm is required to load SBI environments. Please install it.")
    
    import sbibm
    return sbibm.get_task(task_name)

def get_benchmark_task(task_name: str):
    """Alias for load_sbi_environment to satisfy interface contract."""
    return load_sbi_environment(task_name)

def prepare_sbi_environment(spec: SbiEnvironmentSpec, output_dir: str = "results"):
    """
    Prepares the environment and writes registry/manifest artifacts.
    reference_grounding: paper:implementation_obligations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Write dataset registry
    registry_path = os.path.join(output_dir, "dataset_registry.json")
    write_dataset_registry_artifact(registry_path)
    
    # Write data manifest
    manifest_path = os.path.join(output_dir, "data_manifest.json")
    write_data_manifest_artifact(spec, manifest_path)
    
    return spec

def write_dataset_registry_artifact(path: str):
    """Writes the full registry of available benchmarks."""
    with open(path, 'w') as f:
        json.dump(SBI_BENCHMARK_REGISTRY, f, indent=2)

def write_data_manifest_artifact(spec: SbiEnvironmentSpec, path: str):
    """Writes metadata about the current experiment's data setup."""
    manifest = {
        "current_task": asdict(spec),
        "available_budgets": [1000, 10000, 100000],
        "library": "sbibm",
        "reference": "Lueckmann et al. (2021)"
    }
    with open(path, 'w') as f:
        json.dump(manifest, f, indent=2)

class SbiSimulatorWrapper:
    """
    Consistent simulator API for all tasks.
    reference_grounding: paper:implementation_obligations
    """
    def __init__(self, task_name: str):
        self.task = load_sbi_environment(task_name)
        self.simulator = self.task.get_simulator()
        self.prior = self.task.get_prior()
        
    def __call__(self, theta):
        import torch
        if not isinstance(theta, torch.Tensor):
            theta = torch.as_tensor(theta)
        return self.simulator(theta)

    def sample_prior(self, num_samples: int):
        return self.prior.sample((num_samples,))

    def get_observation(self, idx: int = 1):
        return self.task.get_observation(idx)

# Placeholder for orchestration calls defined in contract
def run_table_1_route():
    """Route for generating Table 1 results (Benchmark comparison)."""
    pass

def write_table_1_artifact(results: Any, path: str = "results/tables/experiment_results.csv"):
    """Writes Table 1 results to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Implementation would convert results to CSV
    pass

if __name__ == "__main__":
    # Smoke test for registry and spec creation
    spec = make_sbi_environment("slcp", budget=10000)
    print(f"Created spec for: {spec.name}")
    
    # Mock artifact writing
    artifact_dir = os.environ.get('PAPERBENCH_REPRO_ARTIFACT_DIR', 'results')
    prepare_sbi_environment(spec, output_dir=artifact_dir)
    print(f"Artifacts written to {artifact_dir}")