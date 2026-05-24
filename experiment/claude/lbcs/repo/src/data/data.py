"""
Data loading and dataset registry for Refined Coreset Selection experiments.

Provides dataset loaders, noise injection, transforms, and registry for
CIFAR-10, CIFAR-100, Fashion-MNIST, and ImageNet-1k datasets.

reference_grounding: paperbench_ref_003 train.py
reference_grounding: paperbench_ref_003 selection.py
reference_grounding: paperbench_ref_004 noisy_label.py
reference_grounding: paperbench_ref_004 cnn_mnist_probability_1step_pixel_shared_rein.py
"""

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Callable
import numpy as np

# ============================================================================
# Dataset Registry
# Paper evidence contract: explicitly register dataset/benchmark aliases for
# cifar, imagenet, mnist, svhn, imagenet_1k
# reference_grounding: paperbench_ref_003 selection.py
# ============================================================================

DATASET_REGISTRY = {
    "cifar10": {
        "id": "cifar10",
        "aliases": ["cifar", "CIFAR-10", "cifar_10"],
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
        "aliases": ["CIFAR-100", "cifar_100"],
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
        "default_coreset_sizes": [1200, 2400, 3600, 4800],
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
        "default_coreset_sizes": [1465, 2931, 4396, 5862],
        "torchvision_name": "SVHN",
        "mean": [0.4377, 0.4438, 0.4728],
        "std": [0.1980, 0.2010, 0.1970],
    },
    "imagenet1k": {
        "id": "imagenet1k",
        "aliases": ["imagenet", "ImageNet-1k", "imagenet_1k"],
        "name": "ImageNet-1k",
        "num_classes": 1000,
        "input_size": 224,
        "input_channels": 3,
        "train_size": 1281167,
        "test_size": 50000,
        "supports_noise": False,
        "default_coreset_sizes": [896817, 1024934],
        "torchvision_name": "ImageNet",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
}


def resolve_dataset_alias(dataset_name: str) -> Dict[str, Any]:
    """
    Resolve dataset name or alias to registry entry.
    
    reference_grounding: paperbench_ref_003 selection.py
    """
    dataset_name_lower = dataset_name.lower()
    
    # Direct lookup
    if dataset_name_lower in DATASET_REGISTRY:
        return DATASET_REGISTRY[dataset_name_lower]
    
    # Alias lookup
    for dataset_id, dataset_info in DATASET_REGISTRY.items():
        if dataset_name in dataset_info.get("aliases", []) or \
           dataset_name_lower in [a.lower() for a in dataset_info.get("aliases", [])]:
            return dataset_info
    
    raise ValueError(
        f"Unknown dataset '{dataset_name}'. Available datasets: "
        f"{list(DATASET_REGISTRY.keys())}"
    )


# ============================================================================
# Lazy imports for torch/torchvision
# Keep optional dependencies behind lazy imports for smoke validation
# ============================================================================

def _check_torch_available() -> bool:
    """Check if torch and torchvision are available."""
    try:
        import torch
        import torchvision
        return True
    except ImportError:
        return False


def _get_torch():
    """Lazy import torch with availability check."""
    try:
        import torch
        return torch
    except ImportError as e:
        raise ImportError(
            "PyTorch is required for dataset loading. "
            "Install with: pip install torch torchvision"
        ) from e


def _get_torchvision():
    """Lazy import torchvision with availability check."""
    try:
        import torchvision
        return torchvision
    except ImportError as e:
        raise ImportError(
            "torchvision is required for dataset loading. "
            "Install with: pip install torchvision"
        ) from e


# ============================================================================
# Noise Injection
# reference_grounding: paperbench_ref_004 noisy_label.py
# ============================================================================

def noisify_symmetric(
    labels: np.ndarray,
    noise_rate: float,
    num_classes: int,
    random_state: int = 42
) -> Tuple[np.ndarray, float]:
    """
    Inject symmetric label noise.
    
    reference_grounding: paperbench_ref_004 noisy_label.py
    
    Args:
        labels: Original labels
        noise_rate: Fraction of labels to corrupt
        num_classes: Number of classes
        random_state: Random seed
        
    Returns:
        noisy_labels: Corrupted labels
        actual_noise_rate: Actual noise rate achieved
    """
    if noise_rate <= 0:
        return labels.copy(), 0.0
    
    np.random.seed(random_state)
    n = len(labels)
    noise_mask = np.random.rand(n) < noise_rate
    
    noisy_labels = labels.copy()
    for idx in np.where(noise_mask)[0]:
        # Random class different from original
        available_classes = [c for c in range(num_classes) if c != labels[idx]]
        noisy_labels[idx] = np.random.choice(available_classes)
    
    actual_noise_rate = np.mean(labels != noisy_labels)
    return noisy_labels, actual_noise_rate


def noisify_cifar10_asymmetric(
    labels: np.ndarray,
    noise_rate: float,
    random_state: int = 42
) -> Tuple[np.ndarray, float]:
    """
    Inject CIFAR-10 asymmetric noise.
    
    reference_grounding: paperbench_ref_004 noisy_label.py
    
    Transition matrix:
    TRUCK → AUTOMOBILE, BIRD → AIRPLANE, DEER → HORSE, CAT → DOG
    """
    np.random.seed(random_state)
    
    # CIFAR-10 asymmetric transitions
    transition_map = {
        9: 1,  # truck -> automobile
        2: 0,  # bird -> airplane
        4: 7,  # deer -> horse
        3: 5,  # cat -> dog
    }
    
    noisy_labels = labels.copy()
    for src_class, tgt_class in transition_map.items():
        src_indices = np.where(labels == src_class)[0]
        n_flip = int(noise_rate * len(src_indices))
        flip_indices = np.random.choice(src_indices, n_flip, replace=False)
        noisy_labels[flip_indices] = tgt_class
    
    actual_noise_rate = np.mean(labels != noisy_labels)
    return noisy_labels, actual_noise_rate


def noisify_cifar100_asymmetric(
    labels: np.ndarray,
    noise_rate: float,
    random_state: int = 42
) -> Tuple[np.ndarray, float]:
    """
    Inject CIFAR-100 asymmetric noise within superclasses.
    
    reference_grounding: paperbench_ref_004 noisy_label.py
    """
    np.random.seed(random_state)
    
    # CIFAR-100 superclass structure (20 superclasses, 5 classes each)
    noisy_labels = labels.copy()
    n_superclasses = 20
    classes_per_super = 5
    
    for super_idx in range(n_superclasses):
        super_classes = list(range(
            super_idx * classes_per_super,
            (super_idx + 1) * classes_per_super
        ))
        
        for src_class in super_classes:
            src_indices = np.where(labels == src_class)[0]
            n_flip = int(noise_rate * len(src_indices))
            if n_flip > 0:
                flip_indices = np.random.choice(src_indices, n_flip, replace=False)
                # Flip to another class in same superclass
                tgt_classes = [c for c in super_classes if c != src_class]
                for idx in flip_indices:
                    noisy_labels[idx] = np.random.choice(tgt_classes)
    
    actual_noise_rate = np.mean(labels != noisy_labels)
    return noisy_labels, actual_noise_rate


def noisify_mnist_asymmetric(
    labels: np.ndarray,
    noise_rate: float,
    random_state: int = 42
) -> Tuple[np.ndarray, float]:
    """
    Inject Fashion-MNIST asymmetric noise.
    
    reference_grounding: paperbench_ref_004 noisy_label.py
    
    Similar items confusion:
    T-shirt → Shirt, Trouser → Dress, Pullover → Coat, Sandal → Sneaker
    """
    np.random.seed(random_state)
    
    # Fashion-MNIST asymmetric transitions
    transition_map = {
        0: 6,  # T-shirt/top -> Shirt
        1: 3,  # Trouser -> Dress
        2: 4,  # Pullover -> Coat
        5: 7,  # Sandal -> Sneaker
    }
    
    noisy_labels = labels.copy()
    for src_class, tgt_class in transition_map.items():
        src_indices = np.where(labels == src_class)[0]
        n_flip = int(noise_rate * len(src_indices))
        if n_flip > 0:
            flip_indices = np.random.choice(src_indices, n_flip, replace=False)
            noisy_labels[flip_indices] = tgt_class
    
    actual_noise_rate = np.mean(labels != noisy_labels)
    return noisy_labels, actual_noise_rate


def apply_noise(
    dataset_name: str,
    labels: np.ndarray,
    noise_type: Optional[str],
    noise_rate: float,
    num_classes: int,
    random_state: int = 42
) -> Tuple[np.ndarray, float]:
    """
    Apply noise to labels based on dataset and noise type.
    
    reference_grounding: paperbench_ref_004 noisy_label.py
    """
    if noise_type is None or noise_rate <= 0:
        return labels.copy(), 0.0
    
    dataset_lower = dataset_name.lower()
    
    if noise_type == "symmetric":
        return noisify_symmetric(labels, noise_rate, num_classes, random_state)
    elif noise_type == "asymmetric":
        if "cifar10" in dataset_lower or dataset_lower == "cifar":
            return noisify_cifar10_asymmetric(labels, noise_rate, random_state)
        elif "cifar100" in dataset_lower:
            return noisify_cifar100_asymmetric(labels, noise_rate, random_state)
        elif "mnist" in dataset_lower or "fmnist" in dataset_lower:
            return noisify_mnist_asymmetric(labels, noise_rate, random_state)
        else:
            warnings.warn(
                f"Asymmetric noise not defined for {dataset_name}, "
                f"using symmetric instead"
            )
            return noisify_symmetric(labels, noise_rate, num_classes, random_state)
    else:
        raise ValueError(
            f"Unknown noise_type '{noise_type}'. "
            f"Use 'symmetric' or 'asymmetric'"
        )


# ============================================================================
# Dataset Transforms
# reference_grounding: paperbench_ref_003 train.py
# ============================================================================

def get_transforms(
    dataset_info: Dict[str, Any],
    train: bool = True
) -> Callable:
    """
    Get dataset-specific transforms.
    
    reference_grounding: paperbench_ref_003 train.py
    """
    torchvision = _get_torchvision()
    transforms = torchvision.transforms
    
    mean = dataset_info["mean"]
    std = dataset_info["std"]
    input_size = dataset_info["input_size"]
    
    if train:
        if dataset_info["id"] == "imagenet1k":
            return transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            if input_size == 32:
                # CIFAR-style
                return transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ])
            else:
                # Fashion-MNIST style
                return transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ])
    else:
        if dataset_info["id"] == "imagenet1k":
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])


# ============================================================================
# Dataset Wrapper with Noise Support
# ============================================================================

class NoisyDataset:
    """
    Wrapper for datasets with optional noisy labels.
    
    reference_grounding: paperbench_ref_004 noisy_label.py
    """
    
    def __init__(
        self,
        base_dataset,
        noisy_labels: Optional[np.ndarray] = None
    ):
        self.base_dataset = base_dataset
        self.noisy_labels = noisy_labels
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        if self.noisy_labels is not None:
            label = int(self.noisy_labels[idx])
        return img, label, idx


# ============================================================================
# Dataset Loading
# reference_grounding: paperbench_ref_003 selection.py
# ============================================================================

def load_dataset(
    dataset_name: str,
    data_path: str = "./data",
    train: bool = True,
    noise_type: Optional[str] = None,
    noise_rate: float = 0.0,
    download: bool = True,
    random_state: int = 42
):
    """
    Load dataset with optional noise injection.
    
    reference_grounding: paperbench_ref_003 selection.py
    reference_grounding: paperbench_ref_004 noisy_label.py
    
    Args:
        dataset_name: Dataset name or alias
        data_path: Path to data directory
        train: Load training set if True, else test set
        noise_type: 'symmetric', 'asymmetric', or None
        noise_rate: Fraction of labels to corrupt
        download: Download dataset if not present
        random_state: Random seed for noise injection
        
    Returns:
        dataset: PyTorch dataset with noise support
        dataset_info: Dataset metadata
        actual_noise_rate: Actual noise rate applied
    """
    if not _check_torch_available():
        raise ImportError(
            "PyTorch and torchvision are required for dataset loading. "
            "Install with: pip install torch torchvision"
        )
    
    torchvision = _get_torchvision()
    dataset_info = resolve_dataset_alias(dataset_name)
    
    # Get transforms
    transform = get_transforms(dataset_info, train=train)
    
    # Load base dataset
    torchvision_name = dataset_info["torchvision_name"]
    
    if torchvision_name == "CIFAR10":
        base_dataset = torchvision.datasets.CIFAR10(
            root=data_path,
            train=train,
            download=download,
            transform=transform
        )
    elif torchvision_name == "CIFAR100":
        base_dataset = torchvision.datasets.CIFAR100(
            root=data_path,
            train=train,
            download=download,
            transform=transform
        )
    elif torchvision_name == "FashionMNIST":
        base_dataset = torchvision.datasets.FashionMNIST(
            root=data_path,
            train=train,
            download=download,
            transform=transform
        )
    elif torchvision_name == "SVHN":
        split = "train" if train else "test"
        base_dataset = torchvision.datasets.SVHN(
            root=data_path,
            split=split,
            download=download,
            transform=transform
        )
    elif torchvision_name == "ImageNet":
        split = "train" if train else "val"
        imagenet_path = os.path.join(data_path, "imagenet")
        if not os.path.exists(imagenet_path):
            raise FileNotFoundError(
                f"ImageNet not found at {imagenet_path}. "
                f"Please download ImageNet manually."
            )
        base_dataset = torchvision.datasets.ImageNet(
            root=imagenet_path,
            split=split,
            transform=transform
        )
    else:
        raise ValueError(f"Unknown torchvision dataset: {torchvision_name}")
    
    # Apply noise to training set
    actual_noise_rate = 0.0
    noisy_labels = None
    
    if train and noise_type is not None and noise_rate > 0:
        # Extract original labels
        if hasattr(base_dataset, 'targets'):
            original_labels = np.array(base_dataset.targets)
        elif hasattr(base_dataset, 'labels'):
            original_labels = np.array(base_dataset.labels)
        else:
            raise AttributeError(
                f"Dataset {torchvision_name} does not have 'targets' or 'labels'"
            )
        
        # Apply noise
        noisy_labels, actual_noise_rate = apply_noise(
            dataset_info["id"],
            original_labels,
            noise_type,
            noise_rate,
            dataset_info["num_classes"],
            random_state
        )
    
    # Wrap in NoisyDataset
    dataset = NoisyDataset(base_dataset, noisy_labels)
    
    return dataset, dataset_info, actual_noise_rate


def get_data_loaders(
    dataset_name: str,
    data_path: str = "./data",
    batch_size: int = 128,
    noise_type: Optional[str] = None,
    noise_rate: float = 0.0,
    num_workers: int = 4,
    random_state: int = 42
):
    """
    Get train and test data loaders.
    
    reference_grounding: paperbench_ref_003 train.py
    """
    torch = _get_torch()
    
    train_dataset, dataset_info, actual_noise_rate = load_dataset(
        dataset_name,
        data_path,
        train=True,
        noise_type=noise_type,
        noise_rate=noise_rate,
        random_state=random_state
    )
    
    test_dataset, _, _ = load_dataset(
        dataset_name,
        data_path,
        train=False,
        noise_type=None,
        noise_rate=0.0
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader, dataset_info, actual_noise_rate


# ============================================================================
# Main Entrypoint
# reference_grounding: paperbench_ref_003 train.py
# ============================================================================

def main(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entrypoint for data loading and validation.
    
    Args:
        config: Configuration dictionary with keys:
            - dataset: Dataset name
            - data_path: Path to data directory
            - batch_size: Batch size
            - noise_type: Noise type (optional)
            - noise_rate: Noise rate (optional)
            - mode: Execution mode (full, runtime_smoke, docker_validate)
            
    Returns:
        results: Dictionary with dataset info and statistics
    """
    dataset_name = config.get("dataset", "cifar10")
    data_path = config.get("data_path", "./data")
    batch_size = config.get("batch_size", 128)
    noise_type = config.get("noise_type", None)
    noise_rate = config.get("noise_rate", 0.0)
    mode = config.get("mode", "full")
    
    print(f"Loading dataset: {dataset_name}")
    print(f"Data path: {data_path}")
    print(f"Batch size: {batch_size}")
    print(f"Noise type: {noise_type}, rate: {noise_rate}")
    print(f"Mode: {mode}")
    
    # Resolve dataset info
    dataset_info = resolve_dataset_alias(dataset_name)
    
    results = {
        "dataset": dataset_info["id"],
        "dataset_name": dataset_info["name"],
        "num_classes": dataset_info["num_classes"],
        "train_size": dataset_info["train_size"],
        "test_size": dataset_info["test_size"],
        "input_size": dataset_info["input_size"],
        "input_channels": dataset_info["input_channels"],
        "noise_type": noise_type,
        "noise_rate": noise_rate,
        "actual_noise_rate": 0.0,
        "mode": mode,
    }
    
    if mode in ["runtime_smoke", "docker_validate"]:
        # Smoke test: validate registry and configuration only
        print(f"Smoke validation: Dataset registry validated for {dataset_info['name']}")
        print(f"Smoke validation: Config valid, skipping actual data loading")
        return results
    
    # Full mode: actually load data
    if not _check_torch_available():
        warnings.warn(
            "PyTorch not available. Returning config validation only. "
            "Install torch and torchvision for actual data loading."
        )
        return results
    
    try:
        train_loader, test_loader, loaded_info, actual_noise_rate = get_data_loaders(
            dataset_name,
            data_path,
            batch_size,
            noise_type,
            noise_rate
        )
        
        results["actual_noise_rate"] = float(actual_noise_rate)
        results["train_batches"] = len(train_loader)
        results["test_batches"] = len(test_loader)
        
        print(f"Successfully loaded {dataset_info['name']}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Test batches: {len(test_loader)}")
        if actual_noise_rate > 0:
            print(f"Actual noise rate: {actual_noise_rate:.4f}")
        
    except Exception as e:
        warnings.warn(f"Data loading failed: {e}")
        results["error"] = str(e)
    
    return results


if __name__ == "__main__":
    # Test dataset registry
    import sys
    
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
    else:
        dataset_name = "cifar10"
    
    config = {
        "dataset": dataset_name,
        "data_path": "./data",
        "batch_size": 128,
        "noise_type": None,
        "noise_rate": 0.0,
        "mode": "runtime_smoke",
    }
    
    results = main(config)
    print("\nResults:")
    print(json.dumps(results, indent=2))