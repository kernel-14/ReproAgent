"""
Dataset registry for Refined Coreset Selection experiments.

This module provides the authoritative dataset registry with metadata, aliases,
loader factories, and configuration hooks for CIFAR-10, CIFAR-100, Fashion-MNIST,
and ImageNet-1k datasets.

Paper evidence contract: explicitly register dataset/benchmark aliases for
cifar, imagenet, mnist, svhn, imagenet_1k.

reference_grounding: paperbench_ref_003 train.py
reference_grounding: paperbench_ref_003 selection.py
reference_grounding: paperbench_ref_004 noisy_label.py
reference_grounding: paperbench_ref_004 cnn_mnist_probability_1step_pixel_shared_rein.py
"""

import os
import warnings
from typing import Dict, Any, Optional, List, Tuple, Callable

# ============================================================================
# Dataset Registry
# Paper-derived dataset metadata, aliases, and configuration hooks
# ============================================================================

DATASET_REGISTRY = {
    "cifar10": {
        "id": "cifar10",
        "aliases": ["cifar", "CIFAR-10", "cifar_10", "CIFAR10"],
        "name": "CIFAR-10",
        "num_classes": 10,
        "input_size": 32,
        "input_channels": 3,
        "train_size": 50000,
        "test_size": 10000,
        "supports_noise": True,
        "default_coreset_sizes": [956, 1912, 2868, 3824],
        "torchvision_name": "CIFAR10",
        "mean": [0.4914, 0.4822, 0.4465],
        "std": [0.2023, 0.1994, 0.2010],
    },
    "cifar100": {
        "id": "cifar100",
        "aliases": ["CIFAR-100", "cifar_100", "CIFAR100"],
        "name": "CIFAR-100",
        "num_classes": 100,
        "input_size": 32,
        "input_channels": 3,
        "train_size": 50000,
        "test_size": 10000,
        "supports_noise": True,
        "default_coreset_sizes": [2500, 5000, 7500, 10000],
        "torchvision_name": "CIFAR100",
        "mean": [0.5071, 0.4867, 0.4408],
        "std": [0.2675, 0.2565, 0.2761],
    },
    "fmnist": {
        "id": "fmnist",
        "aliases": ["F-MNIST", "fashion_mnist", "FashionMNIST", "mnist"],
        "name": "Fashion-MNIST",
        "num_classes": 10,
        "input_size": 28,
        "input_channels": 1,
        "train_size": 60000,
        "test_size": 10000,
        "supports_noise": True,
        "default_coreset_sizes": [1000, 2000, 3000, 4000],
        "torchvision_name": "FashionMNIST",
        "mean": [0.2860],
        "std": [0.3530],
    },
    "svhn": {
        "id": "svhn",
        "aliases": ["SVHN"],
        "name": "SVHN",
        "num_classes": 10,
        "input_size": 32,
        "input_channels": 3,
        "train_size": 73257,
        "test_size": 26032,
        "supports_noise": True,
        "default_coreset_sizes": [1000, 2000, 3000, 4000],
        "torchvision_name": "SVHN",
        "mean": [0.4377, 0.4438, 0.4728],
        "std": [0.1980, 0.2010, 0.1970],
    },
    "imagenet1k": {
        "id": "imagenet1k",
        "aliases": ["imagenet", "ImageNet-1k", "imagenet_1k", "imagenet1k", "ImageNet"],
        "name": "ImageNet-1k",
        "num_classes": 1000,
        "input_size": 224,
        "input_channels": 3,
        "train_size": 1281167,
        "test_size": 50000,
        "supports_noise": False,
        "default_coreset_ratios": [0.7, 0.8],
        "torchvision_name": "ImageNet",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
}

# Build alias lookup map for fast resolution
_ALIAS_TO_ID = {}
for dataset_id, config in DATASET_REGISTRY.items():
    _ALIAS_TO_ID[dataset_id] = dataset_id
    for alias in config.get("aliases", []):
        _ALIAS_TO_ID[alias.lower()] = dataset_id


# ============================================================================
# Registry lookup and resolution functions
# ============================================================================

def resolve_dataset_id(name: str) -> str:
    """
    Resolve a dataset name or alias to its canonical ID.
    
    Args:
        name: Dataset name or alias (case-insensitive)
        
    Returns:
        Canonical dataset ID
        
    Raises:
        ValueError: If dataset name is not registered
    """
    name_lower = name.lower()
    if name_lower in _ALIAS_TO_ID:
        return _ALIAS_TO_ID[name_lower]
    raise ValueError(
        f"Unknown dataset: {name}. Available datasets: "
        f"{list(DATASET_REGISTRY.keys())}"
    )


def get_dataset_config(name: str) -> Dict[str, Any]:
    """
    Get dataset configuration by name or alias.
    
    Args:
        name: Dataset name or alias
        
    Returns:
        Dataset configuration dictionary
    """
    dataset_id = resolve_dataset_id(name)
    return DATASET_REGISTRY[dataset_id].copy()


def list_datasets() -> List[str]:
    """Return list of registered dataset IDs."""
    return list(DATASET_REGISTRY.keys())


def check_dataset_availability(name: str) -> Tuple[bool, str]:
    """
    Check if a dataset is available and can be loaded.
    
    Args:
        name: Dataset name or alias
        
    Returns:
        Tuple of (available: bool, message: str)
    """
    try:
        dataset_id = resolve_dataset_id(name)
        config = DATASET_REGISTRY[dataset_id]
        
        # Check torchvision availability
        try:
            import torchvision.datasets
            torchvision_name = config.get("torchvision_name")
            if not hasattr(torchvision.datasets, torchvision_name):
                return False, f"torchvision.datasets.{torchvision_name} not found"
            return True, f"Dataset {dataset_id} is available"
        except ImportError:
            return False, "torchvision not available"
            
    except ValueError as e:
        return False, str(e)


# ============================================================================
# Dataset loader factory
# Lazy import to avoid hard dependency on torch/torchvision at module load
# ============================================================================

def get_dataset_loader(
    name: str,
    root: str = "./data",
    train: bool = True,
    download: bool = True,
    transform: Optional[Any] = None,
    noise_type: Optional[str] = None,
    noise_rate: float = 0.0,
) -> Any:
    """
    Factory function to create a dataset loader.
    
    Args:
        name: Dataset name or alias
        root: Root directory for dataset storage
        train: If True, load training set; else test set
        download: If True, download dataset if not present
        transform: Optional transform to apply
        noise_type: Optional noise type ('symmetric', 'asymmetric', None)
        noise_rate: Noise rate in [0, 1] if noise_type is not None
        
    Returns:
        Dataset object compatible with torch DataLoader
        
    Raises:
        ImportError: If torch/torchvision not available
        ValueError: If dataset not supported or noise requested but not supported
    """
    # Lazy import of torch and torchvision
    try:
        import torch
        import torchvision
        import torchvision.transforms as transforms
        from torch.utils.data import Dataset
    except ImportError as e:
        raise ImportError(
            f"torch and torchvision required for dataset loading. "
            f"Install with: pip install torch torchvision. Error: {e}"
        )
    
    # Resolve dataset ID and get config
    dataset_id = resolve_dataset_id(name)
    config = get_dataset_config(dataset_id)
    
    # Check noise support
    if noise_type is not None and not config["supports_noise"]:
        raise ValueError(
            f"Dataset {dataset_id} does not support noise injection. "
            f"noise_type must be None."
        )
    
    # Get torchvision dataset class
    torchvision_name = config["torchvision_name"]
    try:
        dataset_class = getattr(torchvision.datasets, torchvision_name)
    except AttributeError:
        raise ValueError(
            f"torchvision.datasets.{torchvision_name} not found. "
            f"Please check torchvision version."
        )
    
    # Create default transform if not provided
    if transform is None:
        transform = get_default_transform(dataset_id, train=train)
    
    # Load base dataset
    if dataset_id in ["cifar10", "cifar100", "fmnist"]:
        dataset = dataset_class(
            root=root,
            train=train,
            download=download,
            transform=transform
        )
    elif dataset_id == "svhn":
        split = "train" if train else "test"
        dataset = dataset_class(
            root=root,
            split=split,
            download=download,
            transform=transform
        )
    elif dataset_id == "imagenet1k":
        split = "train" if train else "val"
        dataset = dataset_class(
            root=root,
            split=split,
            transform=transform
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_id}")
    
    # Apply noise if requested
    if noise_type is not None and noise_rate > 0 and train:
        dataset = apply_label_noise(
            dataset, 
            noise_type=noise_type, 
            noise_rate=noise_rate,
            num_classes=config["num_classes"],
            dataset_name=dataset_id
        )
    
    return dataset


def get_default_transform(dataset_id: str, train: bool = True) -> Any:
    """
    Get default data transform for a dataset.
    
    Args:
        dataset_id: Canonical dataset ID
        train: If True, include training augmentation
        
    Returns:
        torchvision transform composition
    """
    try:
        import torchvision.transforms as transforms
    except ImportError:
        raise ImportError("torchvision required for transforms")
    
    config = DATASET_REGISTRY[dataset_id]
    mean = config["mean"]
    std = config["std"]
    
    if dataset_id in ["cifar10", "cifar100"]:
        if train:
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
    
    elif dataset_id == "fmnist":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    
    elif dataset_id == "svhn":
        if train:
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
    
    elif dataset_id == "imagenet1k":
        if train:
            return transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
    
    else:
        raise ValueError(f"No default transform for dataset: {dataset_id}")


# ============================================================================
# Label noise injection (reference_grounding: paperbench_ref_004 noisy_label.py)
# ============================================================================

def apply_label_noise(
    dataset: Any,
    noise_type: str,
    noise_rate: float,
    num_classes: int,
    dataset_name: str,
    random_state: int = 42
) -> Any:
    """
    Apply label noise to a dataset.
    
    reference_grounding: paperbench_ref_004 noisy_label.py
    
    Args:
        dataset: PyTorch dataset object
        noise_type: 'symmetric' or 'asymmetric'
        noise_rate: Noise rate in [0, 1]
        num_classes: Number of classes
        dataset_name: Dataset name for asymmetric noise rules
        random_state: Random seed
        
    Returns:
        Dataset with noisy labels
    """
    import numpy as np
    
    # Extract original labels
    if hasattr(dataset, 'targets'):
        targets = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        targets = np.array(dataset.labels)
    else:
        raise ValueError("Dataset must have 'targets' or 'labels' attribute")
    
    n_samples = len(targets)
    n_noisy = int(noise_rate * n_samples)
    
    np.random.seed(random_state)
    
    if noise_type == 'symmetric':
        # Symmetric noise: flip to random class
        noisy_indices = np.random.choice(n_samples, n_noisy, replace=False)
        for idx in noisy_indices:
            original_class = targets[idx]
            new_class = np.random.randint(0, num_classes)
            while new_class == original_class:
                new_class = np.random.randint(0, num_classes)
            targets[idx] = new_class
    
    elif noise_type == 'asymmetric':
        # Asymmetric noise: dataset-specific confusion pairs
        transition_matrix = get_asymmetric_transition(dataset_name, num_classes)
        for i in range(n_samples):
            if np.random.rand() < noise_rate:
                targets[i] = np.random.choice(num_classes, p=transition_matrix[targets[i]])
    
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")
    
    # Update dataset labels
    if hasattr(dataset, 'targets'):
        if isinstance(dataset.targets, list):
            dataset.targets = targets.tolist()
        else:
            dataset.targets = targets
    elif hasattr(dataset, 'labels'):
        if isinstance(dataset.labels, list):
            dataset.labels = targets.tolist()
        else:
            dataset.labels = targets
    
    return dataset


def get_asymmetric_transition(dataset_name: str, num_classes: int) -> Any:
    """
    Get asymmetric transition matrix for dataset-specific noise.
    
    Args:
        dataset_name: Dataset identifier
        num_classes: Number of classes
        
    Returns:
        Transition matrix (num_classes x num_classes)
    """
    import numpy as np
    
    # Identity matrix by default
    P = np.eye(num_classes)
    
    if dataset_name == "cifar10":
        # CIFAR-10 asymmetric: truck->automobile, bird->airplane, etc.
        # Class order: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
        P[9, 9] = 0.0  # truck
        P[9, 1] = 1.0  # -> automobile
        P[2, 2] = 0.0  # bird
        P[2, 0] = 1.0  # -> airplane
        P[3, 3] = 0.0  # cat
        P[3, 5] = 1.0  # -> dog
        P[5, 5] = 0.0  # dog
        P[5, 3] = 1.0  # -> cat
        P[4, 4] = 0.0  # deer
        P[4, 7] = 1.0  # -> horse
    
    elif dataset_name == "fmnist":
        # Fashion-MNIST asymmetric
        P[5, 5] = 0.0  # Sandal
        P[5, 7] = 1.0  # -> Sneaker
        P[7, 7] = 0.0  # Sneaker
        P[7, 5] = 1.0  # -> Sandal
        P[6, 6] = 0.0  # Shirt
        P[6, 2] = 1.0  # -> Pullover
    
    return P


# ============================================================================
# Configuration integration
# ============================================================================

def dataset_from_config(config: Dict[str, Any]) -> Tuple[Any, Any]:
    """
    Create train and test datasets from configuration dict.
    
    Args:
        config: Configuration dictionary with keys:
            - dataset: dataset name/alias
            - data_path: root directory
            - noise_type: optional noise type
            - noise_rate: optional noise rate
            
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    dataset_name = config.get("dataset", "cifar10")
    data_path = config.get("data_path", "./data")
    noise_type = config.get("noise_type", None)
    noise_rate = config.get("noise_rate", 0.0)
    
    train_dataset = get_dataset_loader(
        name=dataset_name,
        root=data_path,
        train=True,
        download=True,
        noise_type=noise_type,
        noise_rate=noise_rate,
    )
    
    test_dataset = get_dataset_loader(
        name=dataset_name,
        root=data_path,
        train=False,
        download=True,
        noise_type=None,
        noise_rate=0.0,
    )
    
    return train_dataset, test_dataset
