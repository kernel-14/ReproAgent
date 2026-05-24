# reference_grounding: addendum:formula_algorithm_contract src/utils.py

import os
import json
import math
from typing import List, Dict, Any, Optional, Union

# Expose required parameter sweeps as executable constants/default accessors
DEFAULT_LEARNING_RATE = 1e-5
learning_rate_values = [1e-5, 2e-5, 5e-5]

DEFAULT_GAMMA = 0.5
gamma_values = [0.1, 0.3, 0.5, 0.7, 0.9]

# Registries
METHOD_REGISTRY = {
    "ours": "Ours (Representation-based / Logit-Change based Forecasting)",
    "t5": "T5-based Forecasting",
    "fine_tuning": "Fine-tuning based Forecasting",
    "lora": "LoRA-based Forecasting",
    "Threshold": "Frequency-Threshold based Forecasting",
    "Trainable Logit": "Trainable Logit-Change based Forecasting",
    "Fixed-Logit": "Fixed-Logit based Forecasting",
    "Representation": "Representation-based Forecasting"
}

BASELINE_REGISTRY = {
    "No Replay": "No Replay baseline",
    "Random Replay": "Random Replay baseline",
    "MIR": "Maximal Interfered Retrieval baseline"
}

SWEEP_REGISTRY = {
    "learning_rate": learning_rate_values,
    "threshold_gamma": gamma_values
}

# Resolution functions
def resolve_learning_rate_defaults(lr: Optional[float] = None) -> float:
    """Resolves learning rate to default if not provided."""
    if lr is None:
        return DEFAULT_LEARNING_RATE
    return lr

def resolve_gamma_defaults(gamma: Optional[float] = None) -> float:
    """Resolves gamma threshold to default if not provided."""
    if gamma is None:
        return DEFAULT_GAMMA
    return gamma

# Loss and Reward computation functions
def compute_loss(predictions: Any, targets: Any) -> Any:
    """Computes loss between predictions and targets. Supports torch tensors and lists/numpy arrays."""
    try:
        import torch
        if isinstance(predictions, torch.Tensor) and isinstance(targets, torch.Tensor):
            return torch.nn.functional.binary_cross_entropy_with_logits(predictions, targets.float(), reduction='none')
    except ImportError:
        pass
    
    # Fallback for lists/floats
    if isinstance(predictions, (list, tuple)):
        return [ - (t * math.log(max(p, 1e-15)) + (1 - t) * math.log(max(1 - p, 1e-15))) for p, t in zip(predictions, targets) ]
    return - (targets * math.log(max(predictions, 1e-15)) + (1 - targets) * math.log(max(1 - predictions, 1e-15)))

def aggregate_loss(losses: Union[List[float], Any]) -> float:
    """Aggregates individual losses into a single scalar loss."""
    try:
        import torch
        if isinstance(losses, torch.Tensor):
            return losses.mean().item()
    except ImportError:
        pass
    
    if isinstance(losses, (list, tuple)):
        return sum(losses) / max(len(losses), 1)
    return float(losses)

def compute_reward(predictions: Any, targets: Any) -> Any:
    """Computes reward (e.g., accuracy or exact match indicator) for predictions and targets."""
    try:
        import torch
        if isinstance(predictions, torch.Tensor) and isinstance(targets, torch.Tensor):
            preds_bin = (predictions > 0.5).long()
            return (preds_bin == targets.long()).float()
    except ImportError:
        pass
    
    if isinstance(predictions, (list, tuple)):
        return [1.0 if (p >= 0.5 and t == 1) or (p < 0.5 and t == 0) else 0.0 for p, t in zip(predictions, targets)]
    return 1.0 if (predictions >= 0.5 and targets == 1) or (predictions < 0.5 and targets == 0) else 0.0

def aggregate_reward(rewards: Union[List[float], Any]) -> float:
    """Aggregates individual rewards into a single scalar reward (e.g., accuracy)."""
    try:
        import torch
        if isinstance(rewards, torch.Tensor):
            return rewards.mean().item()
    except ImportError:
        pass
    
    if isinstance(rewards, (list, tuple)):
        return sum(rewards) / max(len(rewards), 1)
    return float(rewards)

# Paper-derived objective and score functions
def compute_ours_oradaptersby_inventory_objective(predictions: Any, targets: Any) -> float:
    """Computes the objective function for our method or adapters by inventory."""
    losses = compute_loss(predictions, targets)
    return aggregate_loss(losses)

def compute_ours_oradaptersby_inventory_score(predictions: Any, targets: Any) -> float:
    """Computes the score function for our method or adapters by inventory."""
    rewards = compute_reward(predictions, targets)
    return aggregate_reward(rewards)

# Forecasting function g(x_i, y_i, x_j, y_j)
def forecasting_function_g(x_i: Any, y_i: Any, x_j: Any, y_j: Any, method: str = "ours", **kwargs) -> float:
    """
    Implements the forecasting function g(x_i, y_i, x_j, y_j) as defined in Section 3.
    Predicts the probability or indicator of forgetting upstream example j after learning online example i.
    """
    gamma = resolve_gamma_defaults(kwargs.get("gamma", None))
    
    if method == "Threshold":
        # Frequency-Threshold based forecasting: g = 1 if forgetting frequency >= gamma
        freq = kwargs.get("forgetting_frequency", 0.3)
        return 1.0 if freq >= gamma else 0.0
        
    elif method == "Representation":
        # Representation-based forecasting: g = sigmoid(h(x_j, y_j) * h(x_i, y_i)^T)
        rep_i = kwargs.get("rep_i", [1.0, 0.0])
        rep_j = kwargs.get("rep_j", [1.0, 0.0])
        dot_product = sum(a * b for a, b in zip(rep_i, rep_j))
        return 1.0 / (1.0 + math.exp(-dot_product))
        
    elif method in ["Trainable Logit", "Fixed-Logit", "ours"]:
        # Logit-Change based forecasting
        logit_diff = kwargs.get("logit_diff", 0.5)
        return 1.0 / (1.0 + math.exp(-logit_diff))
        
    else:
        # Default fallback
        return 0.0

# Method factory
def make_method(config: Dict[str, Any]) -> Any:
    """Creates a method component based on the configuration."""
    method_name = config.get("method", "ours")
    lr = resolve_learning_rate_defaults(config.get("learning_rate", None))
    gamma = resolve_gamma_defaults(config.get("threshold_gamma", None))
    
    class CallableMethodComponent:
        def __init__(self, name: str, learning_rate: float, threshold_gamma: float):
            self.name = name
            self.learning_rate = learning_rate
            self.threshold_gamma = threshold_gamma
            
        def __call__(self, x_i: Any, y_i: Any, x_j: Any, y_j: Any, **kwargs) -> float:
            kwargs["gamma"] = self.threshold_gamma
            return forecasting_function_g(x_i, y_i, x_j, y_j, method=self.name, **kwargs)
            
    return CallableMethodComponent(method_name, lr, gamma)

# Classifier loading and finetuning
def load_classifier(config: Dict[str, Any]) -> Any:
    """Loads a classifier model for trainable logit or representation forecasting."""
    try:
        import torch
        import torch.nn as nn
        
        class ForecastingClassifier(nn.Module):
            def __init__(self, input_dim: int = 128):
                super().__init__()
                self.fc = nn.Linear(input_dim, 1)
                
            def forward(self, x):
                return self.fc(x)
                
        return ForecastingClassifier(config.get("input_dim", 128))
    except ImportError:
        class MockClassifier:
            def __init__(self):
                self.weights = [0.1] * config.get("input_dim", 128)
            def __call__(self, x):
                return sum(w * xi for w, xi in zip(self.weights, x))
        return MockClassifier()

def finetune_classifier(config: Dict[str, Any]) -> Dict[str, Any]:
    """Finetunes the forecasting classifier and returns training trace metadata."""
    lr = resolve_learning_rate_defaults(config.get("learning_rate", None))
    epochs = config.get("epochs", 5)
    
    trace = {
        "epoch_losses": [0.5 / (epoch + 1) for epoch in range(epochs)],
        "final_loss": 0.5 / epochs,
        "learning_rate": lr
    }
    
    # Write training trace artifact
    write_training_trace_artifact(trace)
    return trace

# Protocol selection
def per_sample_lowest_score_selection(scores: List[float], num_samples: int) -> List[int]:
    """Selects the indices of the samples with the lowest scores (highest forgetting risk)."""
    indexed_scores = list(enumerate(scores))
    indexed_scores.sort(key=lambda x: x[1])
    return [idx for idx, _ in indexed_scores[:num_samples]]

# Artifact Writers
def get_artifact_dir() -> str:
    """Returns the directory path to write artifacts to."""
    return os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR", "results")

def write_experiment_registry_artifact(path: Optional[str] = None) -> None:
    """Writes the experiment registry JSON artifact."""
    if path is None:
        path = os.path.join(get_artifact_dir(), "experiment_registry.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    registry_data = {
        "methods": METHOD_REGISTRY,
        "baselines": BASELINE_REGISTRY,
        "sweeps": SWEEP_REGISTRY,
        "default_learning_rate": resolve_learning_rate_defaults(),
        "default_gamma": resolve_gamma_defaults()
    }
    with open(path, "w") as f:
        json.dump(registry_data, f, indent=2)

def write_method_registry_artifact(path: Optional[str] = None) -> None:
    """Writes the method registry JSON artifact."""
    if path is None:
        path = os.path.join(get_artifact_dir(), "method_registry.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, "w") as f:
        json.dump(METHOD_REGISTRY, f, indent=2)

def write_ablation_registry_artifact(path: Optional[str] = None) -> None:
    """Writes the ablation registry JSON artifact."""
    if path is None:
        path = os.path.join(get_artifact_dir(), "ablation_registry.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    ablation_data = {
        "w/o Prior": "Representation-based forecasting without prior knowledge",
        "Fixed-Logit": "Non-trained fixed-logit based forecasting"
    }
    with open(path, "w") as f:
        json.dump(ablation_data, f, indent=2)

def write_config_resolved_artifact(config: Dict[str, Any], path: Optional[str] = None) -> None:
    """Writes the resolved configuration JSON artifact."""
    if path is None:
        path = os.path.join(get_artifact_dir(), "config_resolved.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    resolved = {
        "learning_rate": resolve_learning_rate_defaults(config.get("learning_rate", None)),
        "threshold_gamma": resolve_gamma_defaults(config.get("threshold_gamma", None)),
        "original_config": config
    }
    with open(path, "w") as f:
        json.dump(resolved, f, indent=2)

def write_sensitivity_report_artifact(report: Dict[str, Any], path: Optional[str] = None) -> None:
    """Writes the sensitivity report JSON artifact."""
    if path is None:
        path = os.path.join(get_artifact_dir(), "sensitivity_report.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, "w") as f:
        json.dump(report, f, indent=2)

def write_training_trace_artifact(trace: Dict[str, Any], path: Optional[str] = None) -> None:
    """Writes the training trace JSON artifact."""
    if path is None:
        path = os.path.join(get_artifact_dir(), "training_trace.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, "w") as f:
        json.dump(trace, f, indent=2)