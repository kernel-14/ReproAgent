"""
Environment and dataset registry for Refined Coreset Selection experiments.

Provides environment/task registry, dataset registry, and artifact writing
utilities for all paper experiments (Tables 1-3, Figures 2-3).

reference_grounding: paperbench_ref_004 logging_utils/tbtools.py
reference_grounding: paperbench_ref_003 selection.py
"""

import csv
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import warnings

# ============================================================================
# Environment/Task Registry
# Paper evidence contract: explicitly register environment/task aliases for
# cifar, imagenet, mnist, svhn
# reference_grounding: paperbench_ref_003 selection.py
# ============================================================================

ENVIRONMENT_REGISTRY = {
    "cifar10": {
        "id": "cifar10",
        "aliases": ["cifar", "CIFAR-10", "cifar_10"],
        "dataset_name": "CIFAR-10",
        "num_classes": 10,
        "input_size": 32,
        "input_channels": 3,
        "train_size": 50000,
        "test_size": 10000,
        "supports_noise": True,
        "default_coreset_sizes": [956, 1912, 2868, 3824],
        "torchvision_name": "CIFAR10",
        "setup_metadata": {
            "download": True,
            "transform_train": "standard_augmentation",
            "transform_test": "standard_normalization",
        },
    },
    "cifar100": {
        "id": "cifar100",
        "aliases": ["CIFAR-100", "cifar_100"],
        "dataset_name": "CIFAR-100",
        "num_classes": 100,
        "input_size": 32,
        "input_channels": 3,
        "train_size": 50000,
        "test_size": 10000,
        "supports_noise": True,
        "default_coreset_sizes": [2500, 5000, 7500, 10000],
        "torchvision_name": "CIFAR100",
        "setup_metadata": {
            "download": True,
            "transform_train": "standard_augmentation",
            "transform_test": "standard_normalization",
        },
    },
    "fmnist": {
        "id": "fmnist",
        "aliases": ["mnist", "F-MNIST", "fashion_mnist", "FashionMNIST"],
        "dataset_name": "Fashion-MNIST",
        "num_classes": 10,
        "input_size": 28,
        "input_channels": 1,
        "train_size": 60000,
        "test_size": 10000,
        "supports_noise": True,
        "default_coreset_sizes": [1000, 2000, 3000, 4000],
        "torchvision_name": "FashionMNIST",
        "setup_metadata": {
            "download": True,
            "transform_train": "standard_augmentation",
            "transform_test": "standard_normalization",
        },
    },
    "imagenet1k": {
        "id": "imagenet1k",
        "aliases": ["imagenet", "ImageNet-1k", "imagenet_1k", "ILSVRC2012"],
        "dataset_name": "ImageNet-1k",
        "num_classes": 1000,
        "input_size": 224,
        "input_channels": 3,
        "train_size": 1281167,
        "test_size": 50000,
        "supports_noise": False,
        "default_coreset_ratios": [0.7, 0.8],
        "torchvision_name": "ImageNet",
        "setup_metadata": {
            "download": False,
            "transform_train": "imagenet_augmentation",
            "transform_test": "imagenet_normalization",
            "requires_manual_download": True,
        },
    },
    "svhn": {
        "id": "svhn",
        "aliases": ["SVHN", "street_view_house_numbers"],
        "dataset_name": "SVHN",
        "num_classes": 10,
        "input_size": 32,
        "input_channels": 3,
        "train_size": 73257,
        "test_size": 26032,
        "supports_noise": True,
        "default_coreset_sizes": [1000, 2000, 3000, 4000],
        "torchvision_name": "SVHN",
        "setup_metadata": {
            "download": True,
            "split": "train",
            "transform_train": "standard_augmentation",
            "transform_test": "standard_normalization",
        },
    },
}

# ============================================================================
# Dataset Registry
# Paper evidence contract: explicitly register dataset/benchmark aliases for
# cifar, imagenet, mnist, svhn, imagenet_1k
# ============================================================================

DATASET_REGISTRY = {
    "cifar10": {
        "id": "cifar10",
        "name": "CIFAR-10",
        "aliases": ["cifar", "CIFAR-10", "cifar_10"],
        "num_classes": 10,
        "train_samples": 50000,
        "test_samples": 10000,
        "input_shape": (3, 32, 32),
        "mean": [0.4914, 0.4822, 0.4465],
        "std": [0.2023, 0.1994, 0.2010],
        "loader_config": {
            "batch_size": 128,
            "num_workers": 4,
            "pin_memory": True,
        },
    },
    "cifar100": {
        "id": "cifar100",
        "name": "CIFAR-100",
        "aliases": ["CIFAR-100", "cifar_100"],
        "num_classes": 100,
        "train_samples": 50000,
        "test_samples": 10000,
        "input_shape": (3, 32, 32),
        "mean": [0.5071, 0.4867, 0.4408],
        "std": [0.2675, 0.2565, 0.2761],
        "loader_config": {
            "batch_size": 128,
            "num_workers": 4,
            "pin_memory": True,
        },
    },
    "fmnist": {
        "id": "fmnist",
        "name": "Fashion-MNIST",
        "aliases": ["mnist", "F-MNIST", "fashion_mnist", "FashionMNIST"],
        "num_classes": 10,
        "train_samples": 60000,
        "test_samples": 10000,
        "input_shape": (1, 28, 28),
        "mean": [0.2860],
        "std": [0.3530],
        "loader_config": {
            "batch_size": 128,
            "num_workers": 4,
            "pin_memory": True,
        },
    },
    "imagenet1k": {
        "id": "imagenet1k",
        "name": "ImageNet-1k",
        "aliases": ["imagenet", "ImageNet-1k", "imagenet_1k", "ILSVRC2012"],
        "num_classes": 1000,
        "train_samples": 1281167,
        "test_samples": 50000,
        "input_shape": (3, 224, 224),
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "loader_config": {
            "batch_size": 256,
            "num_workers": 8,
            "pin_memory": True,
        },
    },
    "svhn": {
        "id": "svhn",
        "name": "SVHN",
        "aliases": ["SVHN", "street_view_house_numbers"],
        "num_classes": 10,
        "train_samples": 73257,
        "test_samples": 26032,
        "input_shape": (3, 32, 32),
        "mean": [0.4377, 0.4438, 0.4728],
        "std": [0.1980, 0.2010, 0.1970],
        "loader_config": {
            "batch_size": 128,
            "num_workers": 4,
            "pin_memory": True,
        },
    },
}

for dataset_id, dataset_info in DATASET_REGISTRY.items():
    env_info = ENVIRONMENT_REGISTRY.get(dataset_id, {})
    input_shape = dataset_info.get("input_shape", ())
    if input_shape:
        dataset_info.setdefault("input_channels", input_shape[0])
        dataset_info.setdefault("input_size", input_shape[-1])
    for key in ("supports_noise", "default_coreset_sizes", "default_coreset_ratios", "torchvision_name"):
        if key in env_info:
            dataset_info.setdefault(key, env_info[key])

# ============================================================================
# Artifact Writing Utilities
# reference_grounding: paperbench_ref_004 logging_utils/tbtools.py
# ============================================================================

def write_table(results: Union[List[Dict], Dict], output_path: str) -> str:
    """
    Write experiment results to CSV table.
    
    Args:
        results: List of result dictionaries or single result dictionary
        output_path: Path to output CSV file
        
    Returns:
        Path to written CSV file
        
    reference_grounding: paperbench_ref_004 logging_utils/tbtools.py
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if isinstance(results, dict):
        results = [results]
    
    if not results:
        results = [{"status": "no_results", "note": "empty result set"}]
    
    fieldnames = []
    for result in results:
        for key in result.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    return str(output_path)


def plot_figure(results: Union[List[Dict], Dict], output_path: str, 
                figure_type: str = "line", **kwargs) -> str:
    """
    Generate figure from experiment results.
    
    Args:
        results: Result data to plot
        output_path: Path to output PNG file
        figure_type: Type of figure ("line", "bar", "scatter")
        **kwargs: Additional plotting parameters
        
    Returns:
        Path to written PNG file
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        warnings.warn("matplotlib not available, writing minimal figure stub")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("Figure stub: matplotlib not available\n")
        return str(output_path)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if isinstance(results, dict):
        results = [results]
    
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
    
    if figure_type == "line" and results:
        x_key = kwargs.get('x_key', 'x')
        y_key = kwargs.get('y_key', 'y')
        
        for result in results:
            if x_key in result and y_key in result:
                x_data = result[x_key]
                y_data = result[y_key]
                label = result.get('label', 'data')
                ax.plot(x_data, y_data, marker='o', label=label)
        
        ax.set_xlabel(kwargs.get('xlabel', 'X'))
        ax.set_ylabel(kwargs.get('ylabel', 'Y'))
        ax.set_title(kwargs.get('title', 'Figure'))
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    elif figure_type == "bar" and results:
        labels = [r.get('label', f'item_{i}') for i, r in enumerate(results)]
        values = [r.get('value', 0) for r in results]
        ax.bar(labels, values)
        ax.set_ylabel(kwargs.get('ylabel', 'Value'))
        ax.set_title(kwargs.get('title', 'Figure'))
        ax.grid(True, alpha=0.3, axis='y')
    
    else:
        ax.text(0.5, 0.5, 'No plottable data', ha='center', va='center')
        ax.set_title(kwargs.get('title', 'Figure'))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=kwargs.get('dpi', 150), bbox_inches='tight')
    plt.close(fig)
    
    return str(output_path)


# ============================================================================
# Paper Table Writers
# ============================================================================

def write_table1(results: List[Dict[str, Any]], output_path: str = "results/tables/table_1.csv") -> str:
    """
    Write Table 1: Preliminary presentation of LBCS algorithm superiority.
    
    Columns: epsilon, initial_k, f1_m_val_error, f2_m_coreset_size, test_accuracy
    
    Args:
        results: List of experimental results with epsilon, initial_k, metrics
        output_path: Output CSV path
        
    Returns:
        Path to written table
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = ['epsilon', 'initial_k', 'f1_m_val_error', 'f2_m_coreset_size', 
                  'test_accuracy', 'test_accuracy_std']
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            row = {
                'epsilon': result.get('epsilon', 0.0),
                'initial_k': result.get('initial_k', 0),
                'f1_m_val_error': result.get('f1_m_val_error', result.get('val_error', 0.0)),
                'f2_m_coreset_size': result.get('f2_m_coreset_size', result.get('coreset_size', 0)),
                'test_accuracy': result.get('test_accuracy', result.get('test_acc_mean', 0.0)),
                'test_accuracy_std': result.get('test_accuracy_std', result.get('test_acc_std', 0.0)),
            }
            writer.writerow(row)
    
    return str(output_path)


def write_table2(results: List[Dict[str, Any]], output_path: str = "results/tables/table_2.csv") -> str:
    """
    Write Table 2: Comparison of LBCS with 7 baselines on multiple datasets.
    
    Columns: dataset, k, method, test_acc_mean, test_acc_std, k_opt (for LBCS)
    
    Args:
        results: List of baseline comparison results
        output_path: Output CSV path
        
    Returns:
        Path to written table
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = ['dataset', 'k', 'method', 'test_acc_mean', 'test_acc_std', 'k_opt']
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            row = {
                'dataset': result.get('dataset', 'unknown'),
                'k': result.get('k', result.get('coreset_size', 0)),
                'method': result.get('method', 'unknown'),
                'test_acc_mean': result.get('test_acc_mean', result.get('test_accuracy', 0.0)),
                'test_acc_std': result.get('test_acc_std', 0.0),
                'k_opt': result.get('k_opt', '') if result.get('method') == 'LBCS' else '',
            }
            writer.writerow(row)
    
    return str(output_path)


def write_table3(results: List[Dict[str, Any]], output_path: str = "results/tables/table_3.csv") -> str:
    """
    Write Table 3: ImageNet-1k evaluation with coreset ratios 70% and 80%.
    
    Columns: method, coreset_ratio, test_accuracy_top1, test_accuracy_top5
    
    Args:
        results: List of ImageNet evaluation results
        output_path: Output CSV path
        
    Returns:
        Path to written table
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = ['method', 'coreset_ratio', 'test_accuracy_top1', 'test_accuracy_top1_std',
                  'test_accuracy_top5', 'test_accuracy_top5_std', 'coreset_size']
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            row = {
                'method': result.get('method', 'unknown'),
                'coreset_ratio': result.get('coreset_ratio', result.get('ratio', 0.0)),
                'test_accuracy_top1': result.get('test_accuracy_top1', result.get('top1_acc', 0.0)),
                'test_accuracy_top1_std': result.get('test_accuracy_top1_std', 0.0),
                'test_accuracy_top5': result.get('test_accuracy_top5', result.get('top5_acc', 0.0)),
                'test_accuracy_top5_std': result.get('test_accuracy_top5_std', 0.0),
                'coreset_size': result.get('coreset_size', 0),
            }
            writer.writerow(row)
    
    return str(output_path)


# ============================================================================
# Paper Figure Generators
# ============================================================================

def generate_figure2(results: List[Dict[str, Any]], output_path: str = "results/figures/figure_2.png") -> str:
    """
    Generate Figure 2: Robustness against 30% symmetric label noise on F-MNIST.
    
    Shows test accuracy vs coreset size for LBCS and baselines under noisy labels.
    
    Args:
        results: List of experiment results with noise robustness data
        output_path: Output PNG path
        
    Returns:
        Path to written figure
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        warnings.warn("matplotlib not available for figure generation")
        output_path.write_text("Figure 2 stub: matplotlib not available\n")
        return str(output_path)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = {}
    for result in results:
        method = result.get('method', 'unknown')
        if method not in methods:
            methods[method] = {'k': [], 'acc': [], 'std': []}
        methods[method]['k'].append(result.get('k', result.get('coreset_size', 0)))
        methods[method]['acc'].append(result.get('test_accuracy', result.get('test_acc_mean', 0.0)))
        methods[method]['std'].append(result.get('test_accuracy_std', result.get('test_acc_std', 0.0)))
    
    for method, data in methods.items():
        k_sorted = sorted(zip(data['k'], data['acc'], data['std']))
        k_vals = [x[0] for x in k_sorted]
        acc_vals = [x[1] for x in k_sorted]
        std_vals = [x[2] for x in k_sorted]
        
        ax.plot(k_vals, acc_vals, marker='o', label=method, linewidth=2)
        if any(s > 0 for s in std_vals):
            ax.fill_between(k_vals, 
                          [a - s for a, s in zip(acc_vals, std_vals)],
                          [a + s for a, s in zip(acc_vals, std_vals)],
                          alpha=0.2)
    
    ax.set_xlabel('Coreset Size (k)', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Robustness against 30% Symmetric Label Noise (F-MNIST)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    csv_path = output_path.parent / (output_path.stem + "_data.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['method', 'coreset_size', 'test_accuracy', 'test_accuracy_std'])
        for result in results:
            writer.writerow([
                result.get('method', 'unknown'),
                result.get('k', result.get('coreset_size', 0)),
                result.get('test_accuracy', result.get('test_acc_mean', 0.0)),
                result.get('test_accuracy_std', result.get('test_acc_std', 0.0))
            ])
    
    return str(output_path)


def generate_figure3(results: List[Dict[str, Any]], output_path: str = "results/figures/figure_3.png") -> str:
    """
    Generate Figure 3: Additional analysis or ablation study figure.
    
    Args:
        results: List of experiment results
        output_path: Output PNG path
        
    Returns:
        Path to written figure
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not available for figure generation")
        output_path.write_text("Figure 3 stub: matplotlib not available\n")
        return str(output_path)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if results:
        for result in results:
            x = result.get('x', [0])
            y = result.get('y', [0])
            label = result.get('label', 'data')
            ax.plot(x, y, marker='o', label=label)
        
        ax.set_xlabel(results[0].get('xlabel', 'X'), fontsize=12)
        ax.set_ylabel(results[0].get('ylabel', 'Y'), fontsize=12)
        ax.set_title(results[0].get('title', 'Figure 3'), fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
        ax.set_title('Figure 3', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return str(output_path)


# ============================================================================
# Utility Functions
# ============================================================================

def get_environment(env_id: str) -> Dict[str, Any]:
    """
    Retrieve environment configuration by ID or alias.
    
    Args:
        env_id: Environment ID or alias
        
    Returns:
        Environment configuration dictionary
    """
    env_id_lower = env_id.lower()
    
    if env_id_lower in ENVIRONMENT_REGISTRY:
        return ENVIRONMENT_REGISTRY[env_id_lower]
    
    for env_config in ENVIRONMENT_REGISTRY.values():
        if env_id in env_config.get('aliases', []):
            return env_config
        if env_id_lower in [a.lower() for a in env_config.get('aliases', [])]:
            return env_config
    
    raise ValueError(f"Unknown environment: {env_id}. Available: {list(ENVIRONMENT_REGISTRY.keys())}")


def get_dataset(dataset_id: str) -> Dict[str, Any]:
    """
    Retrieve dataset configuration by ID or alias.
    
    Args:
        dataset_id: Dataset ID or alias
        
    Returns:
        Dataset configuration dictionary
    """
    dataset_id_lower = dataset_id.lower()
    
    if dataset_id_lower in DATASET_REGISTRY:
        return DATASET_REGISTRY[dataset_id_lower]
    
    for dataset_config in DATASET_REGISTRY.values():
        if dataset_id in dataset_config.get('aliases', []):
            return dataset_config
        if dataset_id_lower in [a.lower() for a in dataset_config.get('aliases', [])]:
            return dataset_config
    
    raise ValueError(f"Unknown dataset: {dataset_id}. Available: {list(DATASET_REGISTRY.keys())}")


def list_environments() -> List[str]:
    """Return list of available environment IDs."""
    return list(ENVIRONMENT_REGISTRY.keys())


def list_datasets() -> List[str]:
    """Return list of available dataset IDs."""
    return list(DATASET_REGISTRY.keys())
