import os
import json
import math
from typing import Any, Dict, List, Optional

# Active route contract: UnitPythonPySpec, load_unit_python_py, prepare_unit_python_py

class UnitPythonPySpec:
    """
    Specification for the unit_python_py data module.
    Holds metadata, dataset aliases, and configuration parameters.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        # Register dataset/benchmark aliases for robotics
        self.dataset_aliases = {
            "robotics": "robotics_dataset",
            "push-wall": "robotics_push_wall_dataset",
            "nld-aa-v0": "nethack_dataset_aa"
        }
        # Setup metadata
        self.metadata = {
            "project_name": "Fine-tuning RL as Forgetting Mitigation",
            "paper_id": "ftrl",
            "environments": ["two_state_mdp", "appleretrieval", "robotics"],
            "methods": ["vanilla", "scratch", "bc", "ewc", "ks"]
        }

def load_unit_python_py(spec: UnitPythonPySpec) -> Dict[str, Any]:
    """
    Exposes paper-derived dataset/benchmark loaders with ids, setup metadata,
    validation checks, and runnable config hooks for robotics and other environments.
    """
    # Represent external environments or datasets through import-light descriptors/factories
    # with clear availability checks and faithful fallback errors.
    data_loaders = {}
    
    # Robotics dataset loader
    def load_robotics_dataset(batch_size: int = 128):
        # Availability check
        # In a real environment, we would check if the robotics dataset files exist.
        # Here we provide a faithful fallback or mock data.
        dataset_id = "robotics_dataset"
        metadata = {
            "id": dataset_id,
            "alias": "robotics",
            "num_trajectories": 100,
            "state_dim": 39,
            "action_dim": 4,
            "batch_size": batch_size
        }
        # Validation check
        if batch_size <= 0:
            raise ValueError("Batch size must be positive.")
        
        # Return a mock generator or descriptor
        return {
            "metadata": metadata,
            "status": "available_mock",
            "data": [{"state": [0.0]*39, "action": [0.0]*4} for _ in range(batch_size)]
        }
    
    data_loaders["robotics"] = load_robotics_dataset
    
    # Addendum NetHack dataset loader
    def load_nethack_dataset(dataset_name: str = "nld-aa-v0", batch_size: int = 128):
        # reference_grounding: addendum
        metadata = {
            "dataset_name": dataset_name,
            "batch_size": batch_size,
            "description": "NetHack Ttyrec Dataset"
        }
        if dataset_name not in ["nld-aa-v0", "nld-nao-v0"]:
            raise ValueError(f"Unknown NetHack dataset: {dataset_name}")
        return {
            "metadata": metadata,
            "status": "available_mock",
            "data": [{"ttyrec": "mock_data"} for _ in range(batch_size)]
        }
    
    data_loaders["nethack"] = load_nethack_dataset
    
    return {
        "spec": spec,
        "loaders": data_loaders
    }

def prepare_unit_python_py(spec: UnitPythonPySpec) -> Dict[str, Any]:
    """
    Prepares the data module, runs validation checks, and returns setup status.
    """
    # Perform validation checks
    validation_status = {
        "robotics_alias_registered": "robotics" in spec.dataset_aliases,
        "push_wall_alias_registered": "push-wall" in spec.dataset_aliases,
        "metadata_complete": len(spec.metadata["environments"]) == 3
    }
    return {
        "status": "ready",
        "validation": validation_status
    }

# --- Paper Formulas and Algorithms Implementation ---

# 1. Two-state MDP Value Function and Policy Parameterization
# reference_grounding: chunk_018 A.1. Two-state MDPs
def compute_two_state_mdp_policy(theta: float, epsilon: float = 0.5) -> float:
    """
    Parameterizes the policy f_theta:
    f_theta = (-epsilon / (1 - epsilon / 2) * theta + 1) * 1_{theta <= 1 - epsilon / 2}
              + (2 * theta - 1) * 1_{theta > 1 - epsilon / 2}
    """
    threshold = 1.0 - epsilon / 2.0
    if theta <= threshold:
        term1 = (-epsilon / (1.0 - epsilon / 2.0)) * theta + 1.0
        return term1
    else:
        term2 = 2.0 * theta - 1.0
        return term2

def compute_two_state_mdp_value(
    theta: float,
    gamma: float = 0.9,
    r_0: float = 0.11,
    r_1: float = 2.22,
    epsilon: float = 0.5
) -> float:
    """
    Computes the value of state s_0:
    v_0(theta) = 1 / (1 - gamma) * [ theta + r_0 * (1 - theta) * (1 - gamma * f_theta) + gamma * theta * r_1 * (1 - f_theta) ]
                 / [ 1 - gamma * f_theta + gamma * theta ]
    """
    f_theta = compute_two_state_mdp_policy(theta, epsilon)
    numerator = theta + r_0 * (1.0 - theta) * (1.0 - gamma * f_theta) + gamma * theta * r_1 * (1.0 - f_theta)
    denominator = 1.0 - gamma * f_theta + gamma * theta
    if abs(denominator) < 1e-9:
        denominator = 1e-9
    v_0 = (1.0 / (1.0 - gamma)) * (numerator / denominator)
    return v_0

# 2. Behavioral Cloning (BC) and Kickstarting (KS) Losses
# reference_grounding: chunk_004_02 2. Forgetting of pre-trained capabilities
# reference_grounding: chunk_019 A.2. Synthetic example: Appleretrieval
# reference_grounding: C.2. Distillation-based methods
def compute_kl_divergence(pi_1: List[float], pi_2: List[float]) -> float:
    """
    Computes D_KL(pi_1 || pi_2) = sum_a pi_1(a) * log(pi_1(a) / pi_2(a))
    """
    kl = 0.0
    for p1, p2 in zip(pi_1, pi_2):
        p1 = max(p1, 1e-9)
        p2 = max(p2, 1e-9)
        kl += p1 * math.log(p1 / p2)
    return kl

def compute_bc_loss(pi_theta_states: List[List[float]], pi_star_states: List[List[float]]) -> float:
    """
    L_BC(theta) = E_{s ~ B_BC} [ D_KL( pi_*(s) || pi_theta(s) ) ]
    """
    total_kl = 0.0
    count = len(pi_theta_states)
    if count == 0:
        return 0.0
    for pi_t, pi_s in zip(pi_theta_states, pi_star_states):
        total_kl += compute_kl_divergence(pi_s, pi_t)
    return total_kl / count

def compute_ks_loss(pi_theta_states: List[List[float]], pi_star_states: List[List[float]]) -> float:
    """
    L_KS(theta) = E_{s ~ pi_theta} [ D_KL( pi_*(s) || pi_theta(s) ) ]
    """
    # Expectation is over data sampled by the current policy
    return compute_bc_loss(pi_theta_states, pi_star_states)

# 3. Elastic Weight Consolidation (EWC) Loss
# reference_grounding: chunk_003_01 2. Forgetting of pre-trained capabilities
def compute_ewc_loss(theta: List[float], theta_star: List[float], fisher_diagonal: List[float]) -> float:
    """
    L_aux(theta) = sum_i F^i * (theta_*^i - theta^i)^2
    """
    loss = 0.0
    for t, t_star, f in zip(theta, theta_star, fisher_diagonal):
        loss += f * ((t_star - t) ** 2)
    return loss

# 4. Forward Transfer and AUC Metrics for Robotics
# reference_grounding: F. Analysis of forgetting in robotic manipulation tasks
def compute_auc(success_rates: List[float], T: float) -> float:
    """
    AUC := 1/T * int_0^T p(t) dt
    Approximated using trapezoidal rule.
    """
    if not success_rates:
        return 0.0
    n = len(success_rates)
    if n == 1:
        return success_rates[0]
    dt = T / (n - 1)
    integral = 0.0
    for i in range(n - 1):
        integral += 0.5 * (success_rates[i] + success_rates[i+1]) * dt
    return integral / T

def compute_forward_transfer(auc: float, auc_b: float) -> float:
    """
    Forward Transfer := (AUC - AUC^b) / (1 - AUC^b)
    """
    denom = 1.0 - auc_b
    if abs(denom) < 1e-9:
        denom = 1e-9
    return (auc - auc_b) / denom

# 5. Addendum NetHack Directory Registration Hooks
# reference_grounding: addendum
def add_nledata_directory(path: str, alias: str = "nld-aa-v0"):
    """
    Registers the NetHack Learning Environment data directory.
    """
    print(f"[NLE] Registered data directory: {path} as {alias}")
    return {"path": path, "alias": alias, "status": "registered"}

def add_altorg_directory(path: str, alias: str = "nld-nao-v0"):
    """
    Registers the alternative organization directory for NetHack.
    """
    print(f"[NLE] Registered alternative directory: {path} as {alias}")
    return {"path": path, "alias": alias, "status": "registered"}

# --- Artifact Writers ---

def write_metrics_artifact(metrics: Dict[str, Any], filepath: str = "results/metrics.json"):
    """
    Writes the metrics dictionary to results/metrics.json.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Artifact] Wrote metrics to {filepath}")

def write_experiment_results_artifact(results: List[Dict[str, Any]], filepath: str = "results/tables/experiment_results.csv"):
    """
    Writes the experiment results to a CSV file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    import csv
    if not results:
        return
    keys = results[0].keys()
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"[Artifact] Wrote experiment results to {filepath}")

# --- Figure and Table Routes (Mock/Smoke Execution) ---

def run_figure_9_route():
    """
    Executes the route to generate data for Figure 9 (Two-state MDP value visualization).
    """
    print("[Route] Running Figure 9 route...")
    thetas = [i * 0.1 for i in range(11)]
    results = []
    for t in thetas:
        v = compute_two_state_mdp_value(t)
        results.append({"theta": t, "v_0": v})
    return results

def write_figure_9_artifact(results: List[Dict[str, Any]]):
    """
    Writes the Figure 9 data to a JSON or CSV artifact.
    """
    filepath = "results/tables/figure_9_data.csv"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    import csv
    if not results:
        return
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["theta", "v_0"])
        writer.writeheader()
        writer.writerows(results)
    print(f"[Artifact] Wrote Figure 9 data to {filepath}")

def run_figure_4_route():
    """
    Executes the route to generate data for Figure 4 (NetHack / Montezuma forgetting curves).
    """
    print("[Route] Running Figure 4 route...")
    # Mock success rates over training steps
    steps = [i * 1000 for i in range(11)]
    results = []
    for s in steps:
        # Mock forgetting curve
        success_rate = max(0.0, 1.0 - (s / 10000.0) * 0.5)
        results.append({"step": s, "success_rate": success_rate})
    return results

def write_figure_4_artifact(results: List[Dict[str, Any]]):
    """
    Writes the Figure 4 data to a CSV artifact.
    """
    filepath = "results/tables/figure_4_data.csv"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    import csv
    if not results:
        return
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "success_rate"])
        writer.writeheader()
        writer.writerows(results)
    print(f"[Artifact] Wrote Figure 4 data to {filepath}")

def run_figure_6_route():
    """
    Executes the route to generate data for Figure 6 (Success rate computed every 5 million training steps).
    """
    print("[Route] Running Figure 6 route...")
    # Success rate computed every 5 million training steps
    steps = [0, 5000000, 10000000, 15000000, 20000000]
    results = []
    for s in steps:
        success_rate = 0.2 + 0.15 * (s / 5000000)
        results.append({"step": s, "success_rate": min(1.0, success_rate)})
    return results

def write_figure_6_artifact(results: List[Dict[str, Any]]):
    """
    Writes the Figure 6 data to a CSV artifact.
    """
    filepath = "results/tables/figure_6_data.csv"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    import csv
    if not results:
        return
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "success_rate"])
        writer.writeheader()
        writer.writerows(results)
    print(f"[Artifact] Wrote Figure 6 data to {filepath}")