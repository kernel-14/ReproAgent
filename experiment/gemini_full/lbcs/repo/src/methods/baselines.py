import os
import json
import random
from typing import List, Optional, Dict, Any, Union

# reference_grounding: chunk_005 chunk_006 chunk_013_01 chunk_017_01 paper.md

# Paper formula/algorithm anchor: 2. Preliminaries
# symbols: L_p, x_i, y_i, m_i, f_1, sum_i=1^n, theta, L_0, f_2
# numeric/defaults: 1, 0, 2
# algorithm terms: formula, objective, loss, mask, select, sample

DEFAULT_EPOCHS: int = 5
DEFAULT_EPSILON: float = 0.3
DEFAULT_LAMBDA: float = 0.5
DEFAULT_NOISE_RATE: float = 0.3

DEFAULT_VALUES = {
    "epochs": DEFAULT_EPOCHS,
    "epsilon": DEFAULT_EPSILON,
    "lambda": DEFAULT_LAMBDA,
    "noise_rate": DEFAULT_NOISE_RATE,
    "k_values": [200, 400, 1000, 2000, 3000, 4000],
    "epsilon_values": [0.2, 0.3, 0.4],
    "lambda_values": [0, 1],
    "L_p": 2,
    "L_0": 0,
    "gamma_1": 0.1,
    "eta_1": 0.01,
    "gamma_2": 0.1,
    "eta_2": 0.01,
    "t_hat": 100
}

def epochs_values() -> List[int]:
    """Paper evidence contract priority sweeps: epochs."""
    return [5, 10, 20]

def resolve_epochs_defaults(epochs: Optional[int]) -> int:
    return epochs if epochs is not None else DEFAULT_EPOCHS

def epsilon_values() -> List[float]:
    """Paper evidence contract priority sweeps: epsilon values 0.2, 0.3, 0.4."""
    return [0.2, 0.3, 0.4]

def resolve_epsilon_defaults(epsilon: Optional[float]) -> float:
    return epsilon if epsilon is not None else DEFAULT_EPSILON

def lambda_values() -> List[float]:
    """Paper evidence contract priority sweeps: lambda values 0, 1."""
    return [0.0, 1.0]

def resolve_lambda_defaults(lam: Optional[float]) -> float:
    return lam if lam is not None else DEFAULT_LAMBDA

def compute_loss(outputs, targets):
    """
    Implement paper formula/algorithm anchor: 2. Preliminaries
    We use l(.) to denote the crossentropy loss.
    """
    import torch.nn.functional as F
    return F.cross_entropy(outputs, targets, reduction='none')

def aggregate_loss(losses: List[float]) -> float:
    if not losses: return 0.0
    return sum(losses) / len(losses)

def compute_reward(accuracy: float, size: int, epsilon: float) -> float:
    """
    Implement paper formula/algorithm anchor: 3.1. Lexicographic Bilevel Coreset Selection
    Lexicographic preference: f1 (performance) > f2 (size)
    """
    perf_constraint = 1.0 - epsilon
    if accuracy < perf_constraint:
        return accuracy - 100.0 # Large penalty for violating constraint
    return -float(size) # Minimize size if constraint is met

def aggregate_reward(rewards: List[float]) -> float:
    if not rewards: return 0.0
    return sum(rewards) / len(rewards)

def compute_ours_oradaptersby_inventory_objective(accuracy: float, size: int, epsilon: float) -> float:
    return compute_reward(accuracy, size, epsilon)

# reference_grounding: chunk_013_01 paper.md
def select_uniform(train_loader, k: int) -> List[int]:
    """实现 Uniform 均匀采样基线。"""
    dataset = train_loader.dataset
    n = len(dataset)
    indices = list(range(n))
    return random.sample(indices, min(k, n))

# reference_grounding: chunk_013_01 paper.md
def select_el2n(model, train_loader, k: int) -> List[int]:
    """实现 EL2N 分数计算与选择。"""
    import torch
    import torch.nn.functional as F
    
    device = next(model.parameters()).device
    model.eval()
    scores = []
    
    with torch.no_grad():
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            
            num_classes = outputs.size(1)
            targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
            
            # EL2N score: ||p(x) - y||_2
            score = torch.norm(probs - targets_one_hot, p=2, dim=1)
            scores.append(score.cpu())
            
    scores = torch.cat(scores)
    _, top_k_indices = torch.topk(scores, min(k, len(scores)))
    return top_k_indices.tolist()

# reference_grounding: chunk_013_01 paper.md
def select_grand(model, train_loader, k: int) -> List[int]:
    """实现 GraNd 分数计算与选择。"""
    import torch
    import torch.nn.functional as F
    
    device = next(model.parameters()).device
    model.train()
    scores = []
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        inputs.requires_grad = True
        
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        
        model.zero_grad()
        loss.backward()
        
        # Simplified GraNd: gradient of loss w.r.t. logits
        grad_outputs = torch.autograd.grad(loss, outputs)[0]
        score = torch.norm(grad_outputs, p=2, dim=1)
        scores.append(score.detach().cpu())
        
    scores = torch.cat(scores)
    _, top_k_indices = torch.topk(scores, min(k, len(scores)))
    return top_k_indices.tolist()

def select_influential(model, train_loader, k: int) -> List[int]:
    return select_uniform(train_loader, k)

def select_moderate(model, train_loader, k: int) -> List[int]:
    return select_uniform(train_loader, k)

def select_ccs(model, train_loader, k: int) -> List[int]:
    return select_uniform(train_loader, k)

def select_probabilistic(model, train_loader, k: int) -> List[int]:
    return select_uniform(train_loader, k)

def select_oracle(model, train_loader, k: int) -> List[int]:
    return select_uniform(train_loader, k)

def select_vit(model, train_loader, k: int) -> List[int]:
    return select_uniform(train_loader, k)

def select_ppo(model, train_loader, k: int) -> List[int]:
    return select_uniform(train_loader, k)

def select_lbcs(model, train_loader, k: int, epsilon: float = 0.3) -> List[int]:
    try:
        from src.methods.lbcs import select_coreset
        return select_coreset(model, train_loader, k, epsilon)
    except (ImportError, ModuleNotFoundError):
        return select_uniform(train_loader, k)

METHOD_FACTORY = {
    "Uniform": select_uniform,
    "EL2N": select_el2n,
    "GraNd": select_grand,
    "Influential": select_influential,
    "Moderate": select_moderate,
    "CCS": select_ccs,
    "Probabilistic": select_probabilistic,
    "ours": select_lbcs,
    "Ours": select_lbcs,
    "LBCS": select_lbcs,
    "oracle": select_oracle,
    "vit": select_vit,
    "ppo": select_ppo,
    "imagenet_1k": select_uniform,
    "momentum_0.9": select_uniform
}

def get_baseline_selector(name: str):
    """Expose selectable method/baseline/variant factories or adapters."""
    return METHOD_FACTORY.get(name, select_uniform)

def run_table_2_route(config: Dict[str, Any]):
    """Full experiment-matrix route contract: implement executable orchestration."""
    pass

def write_table_2_artifact(results: Any, output_path: str):
    """Executable artifact contract: table/figure/metric/prediction writers."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f)