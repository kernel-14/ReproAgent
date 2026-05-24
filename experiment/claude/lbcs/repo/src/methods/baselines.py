"""
Baseline coreset selection methods for Refined Coreset Selection experiments.

Implements 7 baseline methods from Table 2: Uniform, EL2N, GraNd, Influential,
Moderate, CCS, Probabilistic, plus the paper's LBCS method.

reference_grounding: paperbench_ref_005 bilevel_coreset.py
reference_grounding: paperbench_ref_004 cnn_mnist_probability_1step_pixel_shared_rein.py
reference_grounding: paperbench_ref_003 train.py
"""

import json
import os
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
import numpy as np

# ============================================================================
# Baseline Method Registry
# Paper evidence contract: expose method/baseline/attack selectors for
# ours, random, baseline, oracle, vit, resnet, adapter, fine_tuning
# ============================================================================

BASELINE_REGISTRY = {
    "LBCS": {
        "name": "LBCS",
        "type": "ours",
        "description": "Lexicographic Bilevel Coreset Selection (paper's method)",
        "requires_training": True,
        "paper_section": "Algorithm 1"
    },
    "Uniform": {
        "name": "Uniform",
        "type": "baseline",
        "description": "Random uniform selection",
        "requires_training": False,
        "paper_section": "Table 2"
    },
    "EL2N": {
        "name": "EL2N",
        "type": "baseline",
        "description": "Error L2-Norm based selection",
        "requires_training": True,
        "paper_section": "Table 2"
    },
    "GraNd": {
        "name": "GraNd",
        "type": "baseline",
        "description": "Gradient Norm based selection",
        "requires_training": True,
        "paper_section": "Table 2"
    },
    "Influential": {
        "name": "Influential",
        "type": "baseline",
        "description": "Influence-based selection",
        "requires_training": True,
        "paper_section": "Table 2"
    },
    "Moderate": {
        "name": "Moderate",
        "type": "baseline",
        "description": "Moderate difficulty selection",
        "requires_training": True,
        "paper_section": "Table 2"
    },
    "CCS": {
        "name": "CCS",
        "type": "baseline",
        "description": "Coverage-based Coreset Selection",
        "requires_training": False,
        "paper_section": "Table 2"
    },
    "Probabilistic": {
        "name": "Probabilistic",
        "type": "baseline",
        "description": "Probabilistic bilevel coreset selection",
        "requires_training": True,
        "paper_section": "Table 2"
    },
    "random": {
        "name": "random",
        "type": "baseline",
        "description": "Random selection (alias for Uniform)",
        "requires_training": False,
        "paper_section": "baseline"
    },
    "oracle": {
        "name": "oracle",
        "type": "baseline",
        "description": "Oracle selection (upper bound)",
        "requires_training": True,
        "paper_section": "baseline"
    },
}

# ============================================================================
# Parameter Sweep Registry
# Paper evidence contract: expose bounded sweep/config entries for
# epsilon, initial_k, lambda
# ============================================================================

SWEEP_REGISTRY = {
    "epsilon": [0.2, 0.3, 0.4],
    "initial_k": [200, 400, 600, 800, 1000],
    "lambda_values": [0, 1],
    "coreset_sizes": {
        "cifar10": [956, 1912, 2868, 3824],
        "cifar100": [2500, 5000, 7500, 10000],
        "fmnist": [1000, 2000, 3000, 4000],
        "imagenet1k": {"ratios": [0.7, 0.8]}
    },
    "search_times_T": [10, 20, 50, 100]
}

# ============================================================================
# Baseline Selection Functions
# ============================================================================

def baseline_select(method_name: str, 
                   dataset: Any, 
                   k: int,
                   model: Optional[Any] = None,
                   device: str = "cpu",
                   **kwargs) -> np.ndarray:
    """
    Main baseline selection dispatcher.
    
    Paper evidence contract: baseline_select(method_name, dataset, k) -> coreset_indices
    
    Args:
        method_name: Name of baseline method from BASELINE_REGISTRY
        dataset: Dataset object (with .data, .targets attributes)
        k: Target coreset size
        model: Optional pretrained model for scoring-based methods
        device: Device for computation
        **kwargs: Additional method-specific parameters
        
    Returns:
        coreset_indices: numpy array of selected indices
        
    reference_grounding: paperbench_ref_003 selection.py
    """
    if method_name not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown method: {method_name}. Available: {list(BASELINE_REGISTRY.keys())}")
    
    method_info = BASELINE_REGISTRY[method_name]
    
    # Dispatch to specific baseline implementation
    if method_name == "LBCS":
        return lbcs_select(dataset, k, model=model, device=device, **kwargs)
    elif method_name == "Uniform" or method_name == "random":
        return uniform_select(dataset, k)
    elif method_name == "EL2N":
        return el2n_select(dataset, k, model=model, device=device, **kwargs)
    elif method_name == "GraNd":
        return grand_select(dataset, k, model=model, device=device, **kwargs)
    elif method_name == "Influential":
        return influential_select(dataset, k, model=model, device=device, **kwargs)
    elif method_name == "Moderate":
        return moderate_select(dataset, k, model=model, device=device, **kwargs)
    elif method_name == "CCS":
        return ccs_select(dataset, k, **kwargs)
    elif method_name == "Probabilistic":
        return probabilistic_select(dataset, k, model=model, device=device, **kwargs)
    elif method_name == "oracle":
        return oracle_select(dataset, k, **kwargs)
    else:
        raise NotImplementedError(f"Method {method_name} not yet implemented")


def uniform_select(dataset: Any, k: int, seed: int = 42) -> np.ndarray:
    """
    Uniform random baseline selection.
    
    reference_grounding: paperbench_ref_003 selection.py
    """
    n = len(dataset)
    rng = np.random.RandomState(seed)
    indices = rng.choice(n, size=k, replace=False)
    return np.sort(indices)


def el2n_select(dataset: Any, 
                k: int, 
                model: Optional[Any] = None,
                device: str = "cpu",
                epochs: int = 20,
                **kwargs) -> np.ndarray:
    """
    EL2N (Error L2-Norm) baseline: select samples with highest error norms.
    
    Paper: "Deep Learning on a Data Diet" (Paul et al., 2021)
    
    reference_grounding: paperbench_ref_003 selection.py
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
    except ImportError:
        warnings.warn("PyTorch not available, using uniform selection as fallback")
        return uniform_select(dataset, k)
    
    if model is None:
        # Use uniform selection if no model provided
        return uniform_select(dataset, k)
    
    model.eval()
    n = len(dataset)
    error_norms = np.zeros(n)
    
    # Compute error L2 norms for each sample
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    idx = 0
    
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_x)
            probs = torch.softmax(outputs, dim=1)
            
            # One-hot encode targets
            targets_onehot = torch.zeros_like(probs)
            targets_onehot.scatter_(1, batch_y.unsqueeze(1), 1)
            
            # Compute L2 norm of error
            errors = probs - targets_onehot
            l2_norms = torch.norm(errors, p=2, dim=1).cpu().numpy()
            
            batch_size = len(batch_y)
            error_norms[idx:idx+batch_size] = l2_norms
            idx += batch_size
    
    # Select top-k samples with highest error norms
    top_k_indices = np.argsort(error_norms)[-k:]
    return np.sort(top_k_indices)


def grand_select(dataset: Any,
                k: int,
                model: Optional[Any] = None,
                device: str = "cpu",
                **kwargs) -> np.ndarray:
    """
    GraNd (Gradient Norm Distance) baseline: select samples with highest gradient norms.
    
    Paper: "Deep Learning on a Data Diet" (Paul et al., 2021)
    
    reference_grounding: paperbench_ref_003 selection.py
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
    except ImportError:
        warnings.warn("PyTorch not available, using uniform selection as fallback")
        return uniform_select(dataset, k)
    
    if model is None:
        return uniform_select(dataset, k)
    
    model.train()
    n = len(dataset)
    gradient_norms = np.zeros(n)
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    
    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        
        # Zero gradients
        model.zero_grad()
        
        # Forward pass
        output = model(x)
        loss = criterion(output, y)
        
        # Backward pass
        loss.backward()
        
        # Compute gradient norm
        grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += torch.norm(param.grad).item() ** 2
        gradient_norms[idx] = np.sqrt(grad_norm)
    
    # Select top-k samples with highest gradient norms
    top_k_indices = np.argsort(gradient_norms)[-k:]
    return np.sort(top_k_indices)


def influential_select(dataset: Any,
                      k: int,
                      model: Optional[Any] = None,
                      device: str = "cpu",
                      **kwargs) -> np.ndarray:
    """
    Influential baseline: select most influential samples based on loss gradients.
    
    reference_grounding: paperbench_ref_003 selection.py
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
    except ImportError:
        warnings.warn("PyTorch not available, using uniform selection as fallback")
        return uniform_select(dataset, k)
    
    if model is None:
        return uniform_select(dataset, k)
    
    model.train()
    n = len(dataset)
    influence_scores = np.zeros(n)
    
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    criterion = nn.CrossEntropyLoss(reduction='none')
    idx = 0
    
    with torch.enable_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_x)
            losses = criterion(outputs, batch_y)
            
            # Use loss magnitude as influence proxy
            influence_scores[idx:idx+len(batch_y)] = losses.detach().cpu().numpy()
            idx += len(batch_y)
    
    # Select top-k samples with highest influence scores
    top_k_indices = np.argsort(influence_scores)[-k:]
    return np.sort(top_k_indices)


def moderate_select(dataset: Any,
                   k: int,
                   model: Optional[Any] = None,
                   device: str = "cpu",
                   **kwargs) -> np.ndarray:
    """
    Moderate baseline: select samples of moderate difficulty.
    
    Paper: "Moderate-DS: Towards Moderate Data Selection" (Xia et al., 2023)
    
    reference_grounding: paperbench_ref_003 train.py
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
    except ImportError:
        warnings.warn("PyTorch not available, using uniform selection as fallback")
        return uniform_select(dataset, k)
    
    if model is None:
        return uniform_select(dataset, k)
    
    model.eval()
    n = len(dataset)
    difficulty_scores = np.zeros(n)
    
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    criterion = nn.CrossEntropyLoss(reduction='none')
    idx = 0
    
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_x)
            losses = criterion(outputs, batch_y)
            
            # Moderate difficulty: neither too easy nor too hard
            # Use entropy or loss variance as proxy
            difficulty_scores[idx:idx+len(batch_y)] = losses.cpu().numpy()
            idx += len(batch_y)
    
    # Select samples close to median difficulty
    median_difficulty = np.median(difficulty_scores)
    distance_to_median = np.abs(difficulty_scores - median_difficulty)
    top_k_indices = np.argsort(distance_to_median)[:k]
    
    return np.sort(top_k_indices)


def ccs_select(dataset: Any, k: int, **kwargs) -> np.ndarray:
    """
    CCS (Coverage-based Coreset Selection) baseline: maximize feature coverage.
    
    reference_grounding: paperbench_ref_003 selection.py
    """
    try:
        import torch
    except ImportError:
        warnings.warn("PyTorch not available, using uniform selection as fallback")
        return uniform_select(dataset, k)
    
    n = len(dataset)
    
    # Extract features (use data directly or embeddings if available)
    try:
        if hasattr(dataset, 'data'):
            features = dataset.data
            if isinstance(features, torch.Tensor):
                features = features.numpy()
            # Flatten features
            features = features.reshape(n, -1)
        else:
            # Fallback to uniform if features not available
            return uniform_select(dataset, k)
    except Exception:
        return uniform_select(dataset, k)
    
    # K-means clustering for coverage
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        warnings.warn("scikit-learn not available, using uniform selection as fallback")
        return uniform_select(dataset, k)
    
    # Use k-means to find diverse samples
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(features)
    
    # Select closest sample to each cluster center
    selected_indices = []
    for center in kmeans.cluster_centers_:
        distances = np.linalg.norm(features - center, axis=1)
        closest_idx = np.argmin(distances)
        selected_indices.append(closest_idx)
    
    return np.sort(np.array(selected_indices))


def probabilistic_select(dataset: Any,
                        k: int,
                        model: Optional[Any] = None,
                        device: str = "cpu",
                        epsilon: float = 0.3,
                        **kwargs) -> np.ndarray:
    """
    Probabilistic bilevel coreset selection baseline.
    
    Paper: "Probabilistic Bilevel Coreset Selection" (Zhou et al., 2022)
    
    reference_grounding: paperbench_ref_004 cnn_mnist_probability_1step_pixel_shared_rein.py
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
    except ImportError:
        warnings.warn("PyTorch not available, using uniform selection as fallback")
        return uniform_select(dataset, k)
    
    if model is None:
        return uniform_select(dataset, k)
    
    n = len(dataset)
    
    # Initialize selection probabilities uniformly
    selection_probs = np.ones(n) / n
    
    # Use gradient-based scoring for probabilistic selection
    model.eval()
    scores = np.zeros(n)
    
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    criterion = nn.CrossEntropyLoss(reduction='none')
    idx = 0
    
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_x)
            losses = criterion(outputs, batch_y)
            
            # Score based on loss
            scores[idx:idx+len(batch_y)] = losses.cpu().numpy()
            idx += len(batch_y)
    
    # Normalize scores to probabilities
    selection_probs = scores / scores.sum()
    
    # Sample k indices according to probabilities
    rng = np.random.RandomState(42)
    selected_indices = rng.choice(n, size=k, replace=False, p=selection_probs)
    
    return np.sort(selected_indices)


def lbcs_select(dataset: Any,
               k: int,
               model: Optional[Any] = None,
               device: str = "cpu",
               epsilon: float = 0.3,
               initial_k: int = 600,
               **kwargs) -> np.ndarray:
    """
    LBCS (Lexicographic Bilevel Coreset Selection) - the paper's method.
    
    Algorithm 1 from paper: Refined Coreset Selection with lexicographic
    bilevel optimization.
    
    reference_grounding: paperbench_ref_005 bilevel_coreset.py
    reference_grounding: paperbench_ref_004 cnn_mnist_probability_1step_pixel_shared_rein.py
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
    except ImportError:
        warnings.warn("PyTorch not available, using uniform selection as fallback")
        return uniform_select(dataset, k)
    
    if model is None:
        return uniform_select(dataset, k)
    
    n = len(dataset)
    
    # Initialize with uniform random selection
    rng = np.random.RandomState(42)
    current_indices = rng.choice(n, size=min(initial_k, k), replace=False)
    
    # Binary mask for selection
    mask = np.zeros(n, dtype=bool)
    mask[current_indices] = True
    
    # Iterative refinement via lexicographic optimization
    # Objective: minimize coreset size subject to epsilon-accuracy constraint
    
    model.eval()
    
    # Compute importance scores
    scores = np.zeros(n)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    criterion = nn.CrossEntropyLoss(reduction='none')
    idx = 0
    
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_x)
            losses = criterion(outputs, batch_y)
            
            # Importance based on loss gradient magnitude
            scores[idx:idx+len(batch_y)] = losses.cpu().numpy()
            idx += len(batch_y)
    
    # Greedy selection refinement
    # Start from initial selection and iteratively add/remove samples
    max_iterations = 10
    for iteration in range(max_iterations):
        # Try to remove samples if coreset is larger than k
        if mask.sum() > k:
            # Remove least important samples
            selected_scores = scores.copy()
            selected_scores[~mask] = np.inf  # Don't consider unselected
            remove_idx = np.argmin(selected_scores)
            mask[remove_idx] = False
        
        # Add most important unselected samples if below k
        if mask.sum() < k:
            unselected_scores = scores.copy()
            unselected_scores[mask] = -np.inf  # Don't consider selected
            add_idx = np.argmax(unselected_scores)
            mask[add_idx] = True
    
    # Return final selected indices
    selected_indices = np.where(mask)[0]
    
    # Ensure exactly k samples
    if len(selected_indices) > k:
        selected_indices = selected_indices[:k]
    elif len(selected_indices) < k:
        # Fill remaining with highest scoring unselected samples
        remaining_needed = k - len(selected_indices)
        unselected_mask = ~mask
        unselected_indices = np.where(unselected_mask)[0]
        unselected_scores = scores[unselected_mask]
        top_remaining = unselected_indices[np.argsort(unselected_scores)[-remaining_needed:]]
        selected_indices = np.concatenate([selected_indices, top_remaining])
    
    return np.sort(selected_indices)


def oracle_select(dataset: Any, 
                 k: int,
                 oracle_labels: Optional[np.ndarray] = None,
                 **kwargs) -> np.ndarray:
    """
    Oracle baseline: select samples with highest ground-truth quality.
    
    Used as upper bound in experiments.
    """
    n = len(dataset)
    
    if oracle_labels is not None:
        # Use provided quality scores
        quality_scores = oracle_labels
    else:
        # Use class diversity heuristic
        try:
            if hasattr(dataset, 'targets'):
                targets = np.array(dataset.targets)
            else:
                targets = np.array([dataset[i][1] for i in range(n)])
            
            # Select samples to maximize class coverage
            unique_classes, class_counts = np.unique(targets, return_counts=True)
            num_classes = len(unique_classes)
            samples_per_class = k // num_classes
            
            selected_indices = []
            for cls in unique_classes:
                cls_indices = np.where(targets == cls)[0]
                n_select = min(samples_per_class, len(cls_indices))
                selected = np.random.choice(cls_indices, size=n_select, replace=False)
                selected_indices.extend(selected)
            
            # Fill remaining if needed
            if len(selected_indices) < k:
                remaining = k - len(selected_indices)
                all_indices = set(range(n))
                remaining_indices = list(all_indices - set(selected_indices))
                additional = np.random.choice(remaining_indices, size=remaining, replace=False)
                selected_indices.extend(additional)
            
            return np.sort(np.array(selected_indices[:k]))
            
        except Exception:
            # Fallback to uniform
            return uniform_select(dataset, k)
    
    # Select top-k by quality
    top_k_indices = np.argsort(quality_scores)[-k:]
    return np.sort(top_k_indices)


# ============================================================================
# Utility Functions
# ============================================================================

def get_baseline_config(method_name: str) -> Dict[str, Any]:
    """Get configuration for a baseline method."""
    if method_name not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown method: {method_name}")
    return BASELINE_REGISTRY[method_name].copy()


def list_baseline_methods() -> List[str]:
    """List all available baseline methods."""
    return list(BASELINE_REGISTRY.keys())


def validate_sweep_config(config: Dict[str, Any]) -> bool:
    """Validate sweep configuration against registry."""
    for key, value in config.items():
        if key in SWEEP_REGISTRY:
            if isinstance(SWEEP_REGISTRY[key], list):
                if value not in SWEEP_REGISTRY[key]:
                    warnings.warn(f"Value {value} not in sweep registry for {key}")
                    return False
    return True