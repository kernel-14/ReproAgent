#!/usr/bin/env python3
"""
Environment and dataset registry for Test-Time Model Adaptation with Only Forward Passes.

Exposes paper-derived environment/task registry entries and dataset/benchmark registry
entries with ids, aliases, setup metadata, factory/config hooks, and loader interfaces.

This file materializes the evidence obligation matrix for environments and datasets,
binding experiments to their execution contexts as required by the paper reproduction contract.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
import importlib.util
import warnings


# ==============================================================================
# Lazy Import Utilities
# ==============================================================================

def _has_package(package_name: str) -> bool:
    """Check if a package is available without importing it."""
    return importlib.util.find_spec(package_name) is not None


def _lazy_import_datasets():
    """Lazy import HuggingFace datasets."""
    if not _has_package("datasets"):
        raise ImportError(
            "datasets package not available. Install with: pip install datasets"
        )
    import datasets
    return datasets


def _lazy_import_torch():
    """Lazy import PyTorch."""
    if not _has_package("torch"):
        raise ImportError(
            "torch package not available. Install with: pip install torch torchvision"
        )
    import torch
    import torchvision
    return torch, torchvision


def _lazy_import_clip_benchmark():
    """Lazy import clip_benchmark."""
    if not _has_package("clip_benchmark"):
        warnings.warn("clip_benchmark not available. Some benchmarks will be unavailable.")
        return None
    import clip_benchmark
    return clip_benchmark


# ==============================================================================
# Lightweight Mock Objects for Dry-Run Mode
# ==============================================================================

class MockDataset:
    """Lightweight mock dataset for dry-run validation."""
    
    def __init__(self, name: str, num_samples: int = 100, num_classes: int = 1000):
        self.name = name
        self.num_samples = num_samples
        self.num_classes = num_classes
        self._data = []
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        import numpy as np
        # Return synthetic image and label
        image = np.random.randn(3, 224, 224).astype(np.float32)
        label = idx % self.num_classes
        return {"image": image, "label": label}
    
    def __iter__(self):
        for i in range(self.num_samples):
            yield self[i]


class MockDataLoader:
    """Lightweight mock dataloader for dry-run validation."""
    
    def __init__(self, dataset, batch_size: int = 32, shuffle: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        import numpy as np
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            np.random.shuffle(indices)
        
        for i in range(0, len(self.dataset), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch = {
                "images": np.random.randn(len(batch_indices), 3, 224, 224).astype(np.float32),
                "labels": np.array([idx % self.dataset.num_classes for idx in batch_indices])
            }
            yield batch


# ==============================================================================
# Environment Registry
# ==============================================================================

def get_environment_registry() -> Dict[str, Dict[str, Any]]:
    """
    Get the complete environment/task registry as required by paper evidence contract.
    
    Explicitly registers environment/task aliases for:
    - imagenet
    - autonomous_driving
    - clip_benchmark
    
    Returns:
        Registry mapping environment IDs to full metadata and factory hooks
    """
    return {
        "imagenet": {
            "id": "imagenet",
            "name": "ImageNet",
            "aliases": ["imagenet-1k", "imagenet_1k", "ILSVRC2012"],
            "type": "image_classification",
            "description": "ImageNet ILSVRC 2012 classification benchmark",
            "num_classes": 1000,
            "input_size": [3, 224, 224],
            "dataset_ids": ["imagenet_1k", "imagenet_c", "imagenet_r", "imagenet_v2", "imagenet_sketch"],
            "metrics": ["accuracy", "top5_accuracy", "ece", "loss"],
            "factory": "load_imagenet_environment",
            "setup_metadata": {
                "source": "huggingface",
                "dataset_name": "imagenet-1k",
                "trust_remote_code": True,
                "requires_auth": False,
                "cache_dir": ".cache/imagenet"
            },
            "config_hooks": {
                "preprocessing": "imagenet_transforms",
                "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
                "augmentation": "standard"
            }
        },
        "imagenet_corruption": {
            "id": "imagenet_corruption",
            "name": "ImageNet-C",
            "aliases": ["imagenet_c", "imagenet-c"],
            "type": "image_classification",
            "description": "ImageNet with common corruptions for robustness testing",
            "num_classes": 1000,
            "input_size": [3, 224, 224],
            "dataset_ids": ["imagenet_c"],
            "metrics": ["accuracy", "corruption_error", "ece"],
            "factory": "load_imagenet_c_environment",
            "setup_metadata": {
                "source": "zenodo",
                "corruptions": ["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", 
                               "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
                               "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression"],
                "severities": [1, 2, 3, 4, 5],
                "cache_dir": ".cache/imagenet_c"
            },
            "config_hooks": {
                "preprocessing": "imagenet_transforms",
                "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
            }
        },
        "autonomous_driving": {
            "id": "autonomous_driving",
            "name": "Autonomous Driving",
            "aliases": ["driving", "autodriving"],
            "type": "autonomous_driving",
            "description": "Autonomous driving simulation and benchmarking environment",
            "dataset_ids": ["driving_scenario"],
            "metrics": ["success_rate", "collision_rate", "route_completion"],
            "factory": "load_driving_environment",
            "setup_metadata": {
                "source": "custom",
                "simulator": "carla",
                "cache_dir": ".cache/autonomous_driving"
            },
            "config_hooks": {
                "weather": "clear",
                "traffic_density": "medium"
            }
        },
        "clip_benchmark": {
            "id": "clip_benchmark",
            "name": "CLIP Benchmark",
            "aliases": ["clip", "clip_eval"],
            "type": "vision_language",
            "description": "CLIP model evaluation benchmark across multiple datasets",
            "dataset_ids": ["imagenet_1k", "imagenet_v2", "imagenet_sketch", "imagenet_r"],
            "metrics": ["accuracy", "zero_shot_accuracy"],
            "factory": "load_clip_benchmark_environment",
            "setup_metadata": {
                "source": "clip_benchmark",
                "models": ["ViT-B/32", "ViT-B/16", "ViT-L/14"],
                "cache_dir": ".cache/clip_benchmark"
            },
            "config_hooks": {
                "template": "standard",
                "ensemble": True
            }
        }
    }


# ==============================================================================
# Dataset Registry
# ==============================================================================

def get_dataset_registry() -> Dict[str, Dict[str, Any]]:
    """
    Get the complete dataset/benchmark registry as required by paper evidence contract.
    
    Explicitly registers dataset/benchmark aliases for:
    - imagenet
    - imagenet_1k
    - imagenet_c
    - imagenet_r
    - imagenet_v2
    - imagenet_sketch
    - autonomous_driving
    - clip_benchmark
    
    Returns:
        Registry mapping dataset IDs to full metadata and loader hooks
    """
    return {
        "imagenet": {
            "id": "imagenet",
            "name": "ImageNet",
            "aliases": ["imagenet_1k", "imagenet-1k", "ILSVRC2012"],
            "type": "image_classification",
            "description": "ImageNet ILSVRC 2012 1000-class classification dataset",
            "num_samples": {"train": 1281167, "validation": 50000},
            "num_classes": 1000,
            "loader": "load_imagenet_dataset",
            "setup_metadata": {
                "source": "huggingface",
                "repo_id": "imagenet-1k",
                "trust_remote_code": True,
                "split_names": {"train": "train", "val": "validation"},
                "requires_auth": False
            },
            "config_hooks": {
                "batch_size": 64,
                "num_workers": 4,
                "preprocessing": "standard_imagenet"
            }
        },
        "imagenet_1k": {
            "id": "imagenet_1k",
            "name": "ImageNet-1K",
            "aliases": ["imagenet", "imagenet-1k"],
            "type": "image_classification",
            "description": "ImageNet 1K validation set",
            "num_samples": {"validation": 50000},
            "num_classes": 1000,
            "loader": "load_imagenet_1k_dataset",
            "setup_metadata": {
                "source": "huggingface",
                "repo_id": "imagenet-1k",
                "trust_remote_code": True,
                "split": "validation"
            },
            "config_hooks": {
                "batch_size": 64,
                "num_workers": 4
            }
        },
        "imagenet_c": {
            "id": "imagenet_c",
            "name": "ImageNet-C",
            "aliases": ["imagenet_corruption", "imagenet-c"],
            "type": "image_classification",
            "description": "ImageNet with 15 corruption types at 5 severity levels",
            "num_samples": {"test": 50000},
            "num_classes": 1000,
            "loader": "load_imagenet_c_dataset",
            "setup_metadata": {
                "source": "zenodo",
                "doi": "10.5281/zenodo.2235448",
                "corruptions": 15,
                "severities": 5
            },
            "config_hooks": {
                "batch_size": 64,
                "severity": 5,
                "corruption": "all"
            }
        },
        "imagenet_r": {
            "id": "imagenet_r",
            "name": "ImageNet-R",
            "aliases": ["imagenet_rendition", "imagenet-r"],
            "type": "image_classification",
            "description": "ImageNet Rendition with artistic renditions",
            "num_samples": {"test": 30000},
            "num_classes": 200,
            "loader": "load_imagenet_r_dataset",
            "setup_metadata": {
                "source": "github",
                "url": "https://github.com/hendrycks/imagenet-r"
            },
            "config_hooks": {
                "batch_size": 64
            }
        },
        "imagenet_v2": {
            "id": "imagenet_v2",
            "name": "ImageNet-V2",
            "aliases": ["imagenet_v2", "imagenetv2"],
            "type": "image_classification",
            "description": "ImageNet V2 matched frequency variant",
            "num_samples": {"test": 10000},
            "num_classes": 1000,
            "loader": "load_imagenet_v2_dataset",
            "setup_metadata": {
                "source": "huggingface",
                "repo_id": "vaishaal/ImageNetV2",
                "variant": "matched-frequency"
            },
            "config_hooks": {
                "batch_size": 64
            }
        },
        "imagenet_sketch": {
            "id": "imagenet_sketch",
            "name": "ImageNet-Sketch",
            "aliases": ["imagenet_sketch", "imagenet-sketch"],
            "type": "image_classification",
            "description": "ImageNet with sketch-style images",
            "num_samples": {"test": 50889},
            "num_classes": 1000,
            "loader": "load_imagenet_sketch_dataset",
            "setup_metadata": {
                "source": "github",
                "url": "https://github.com/HaohanWang/ImageNet-Sketch"
            },
            "config_hooks": {
                "batch_size": 64
            }
        },
        "autonomous_driving": {
            "id": "autonomous_driving",
            "name": "Autonomous Driving Dataset",
            "aliases": ["driving", "autodriving"],
            "type": "autonomous_driving",
            "description": "Autonomous driving scenarios and benchmarks",
            "num_samples": {"scenarios": 1000},
            "loader": "load_driving_dataset",
            "setup_metadata": {
                "source": "custom",
                "simulator": "carla"
            },
            "config_hooks": {
                "batch_size": 1
            }
        },
        "clip_benchmark": {
            "id": "clip_benchmark",
            "name": "CLIP Benchmark Dataset",
            "aliases": ["clip", "clip_eval"],
            "type": "vision_language",
            "description": "Multiple datasets for CLIP evaluation",
            "datasets": ["imagenet_1k", "imagenet_v2", "imagenet_sketch", "imagenet_r"],
            "loader": "load_clip_benchmark_dataset",
            "setup_metadata": {
                "source": "clip_benchmark",
                "available_datasets": ["imagenet", "imagenet_v2", "imagenet_sketch", "imagenet_r"]
            },
            "config_hooks": {
                "batch_size": 64
            }
        }
    }


# ==============================================================================
# Dataset Loaders
# ==============================================================================

def load_imagenet_1k_dataset(split: str = "validation", dry_run: bool = False, **kwargs) -> Union[MockDataset, Any]:
    """
    Load ImageNet-1K dataset using HuggingFace with trust_remote_code=True.
    
    Args:
        split: Dataset split to load
        dry_run: If True, return mock dataset for validation
        **kwargs: Additional arguments
        
    Returns:
        Dataset object (real or mock depending on dry_run)
    """
    if dry_run:
        return MockDataset(name=f"imagenet_1k_{split}", num_samples=100, num_classes=1000)
    
    datasets = _lazy_import_datasets()
    
    # Load ImageNet-1K from HuggingFace with trust_remote_code=True as specified in addendum
    dataset = datasets.load_dataset(
        "imagenet-1k",
        split=split,
        trust_remote_code=True,
        cache_dir=kwargs.get("cache_dir", ".cache/imagenet")
    )
    
    return dataset


def load_imagenet_c_dataset(severity: int = 5, corruption: str = "all", dry_run: bool = False, **kwargs) -> Union[MockDataset, Any]:
    """
    Load ImageNet-C corruption dataset.
    
    Args:
        severity: Corruption severity level (1-5)
        corruption: Corruption type or "all"
        dry_run: If True, return mock dataset
        **kwargs: Additional arguments
        
    Returns:
        Dataset object
    """
    if dry_run:
        return MockDataset(name=f"imagenet_c_sev{severity}", num_samples=100, num_classes=1000)
    
    # Real implementation would load from appropriate source
    raise NotImplementedError("Real ImageNet-C loading requires additional setup")


def load_imagenet_r_dataset(dry_run: bool = False, **kwargs) -> Union[MockDataset, Any]:
    """Load ImageNet-R dataset."""
    if dry_run:
        return MockDataset(name="imagenet_r", num_samples=100, num_classes=200)
    
    raise NotImplementedError("Real ImageNet-R loading requires additional setup")


def load_imagenet_v2_dataset(dry_run: bool = False, **kwargs) -> Union[MockDataset, Any]:
    """Load ImageNet-V2 dataset."""
    if dry_run:
        return MockDataset(name="imagenet_v2", num_samples=100, num_classes=1000)
    
    datasets = _lazy_import_datasets()
    dataset = datasets.load_dataset(
        "vaishaal/ImageNetV2",
        split="test",
        cache_dir=kwargs.get("cache_dir", ".cache/imagenet_v2")
    )
    return dataset


def load_imagenet_sketch_dataset(dry_run: bool = False, **kwargs) -> Union[MockDataset, Any]:
    """Load ImageNet-Sketch dataset."""
    if dry_run:
        return MockDataset(name="imagenet_sketch", num_samples=100, num_classes=1000)
    
    raise NotImplementedError("Real ImageNet-Sketch loading requires additional setup")


def get_dataloader(dataset, batch_size: int = 64, shuffle: bool = False, dry_run: bool = False, **kwargs):
    """
    Create dataloader from dataset.
    
    Args:
        dataset: Dataset object
        batch_size: Batch size
        shuffle: Whether to shuffle
        dry_run: If True, return mock dataloader
        **kwargs: Additional arguments
        
    Returns:
        DataLoader object
    """
    if dry_run or isinstance(dataset, MockDataset):
        return MockDataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    torch, torchvision = _lazy_import_torch()
    from torch.utils.data import DataLoader
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=kwargs.get("num_workers", 4),
        pin_memory=kwargs.get("pin_memory", True)
    )


# ==============================================================================
# Environment Factory Functions
# ==============================================================================

def load_imagenet_environment(config: Optional[Dict[str, Any]] = None, dry_run: bool = False):
    """
    Load ImageNet environment with specified configuration.
    
    Args:
        config: Environment configuration
        dry_run: If True, return lightweight mock
        
    Returns:
        Environment instance with dataset loaders
    """
    config = config or {}
    
    dataset = load_imagenet_1k_dataset(
        split=config.get("split", "validation"),
        dry_run=dry_run,
        cache_dir=config.get("cache_dir", ".cache/imagenet")
    )
    
    dataloader = get_dataloader(
        dataset,
        batch_size=config.get("batch_size", 64),
        shuffle=config.get("shuffle", False),
        dry_run=dry_run
    )
    
    return {
        "dataset": dataset,
        "dataloader": dataloader,
        "num_classes": 1000,
        "metadata": get_environment_registry()["imagenet"]
    }


def load_imagenet_c_environment(config: Optional[Dict[str, Any]] = None, dry_run: bool = False):
    """Load ImageNet-C environment."""
    config = config or {}
    
    dataset = load_imagenet_c_dataset(
        severity=config.get("severity", 5),
        corruption=config.get("corruption", "all"),
        dry_run=dry_run
    )
    
    dataloader = get_dataloader(
        dataset,
        batch_size=config.get("batch_size", 64),
        dry_run=dry_run
    )
    
    return {
        "dataset": dataset,
        "dataloader": dataloader,
        "num_classes": 1000,
        "metadata": get_environment_registry()["imagenet_corruption"]
    }


def load_clip_benchmark_environment(config: Optional[Dict[str, Any]] = None, dry_run: bool = False):
    """Load CLIP benchmark environment."""
    config = config or {}
    
    # CLIP benchmark uses multiple datasets
    datasets = {}
    dataloaders = {}
    
    for dataset_name in ["imagenet_1k", "imagenet_v2", "imagenet_sketch", "imagenet_r"]:
        if dataset_name == "imagenet_1k":
            ds = load_imagenet_1k_dataset(dry_run=dry_run)
        elif dataset_name == "imagenet_v2":
            ds = load_imagenet_v2_dataset(dry_run=dry_run)
        elif dataset_name == "imagenet_sketch":
            ds = load_imagenet_sketch_dataset(dry_run=dry_run)
        elif dataset_name == "imagenet_r":
            ds = load_imagenet_r_dataset(dry_run=dry_run)
        else:
            continue
            
        datasets[dataset_name] = ds
        dataloaders[dataset_name] = get_dataloader(ds, batch_size=config.get("batch_size", 64), dry_run=dry_run)
    
    return {
        "datasets": datasets,
        "dataloaders": dataloaders,
        "metadata": get_environment_registry()["clip_benchmark"]
    }


def load_driving_environment(config: Optional[Dict[str, Any]] = None, dry_run: bool = False):
    """Load autonomous driving environment (placeholder)."""
    config = config or {}
    
    if dry_run:
        return {
            "scenarios": MockDataset(name="driving_scenarios", num_samples=10, num_classes=1),
            "metadata": get_environment_registry()["autonomous_driving"]
        }
    
    raise NotImplementedError("Autonomous driving environment requires simulator setup")


# ==============================================================================
# Evaluation Interface
# ==============================================================================

def evaluate_environment(environment_id: str, model, config: Optional[Dict[str, Any]] = None, dry_run: bool = False) -> Dict[str, Any]:
    """
    Evaluate model on specified environment.
    
    Args:
        environment_id: Environment identifier
        model: Model to evaluate
        config: Evaluation configuration
        dry_run: If True, return mock metrics
        
    Returns:
        Evaluation metrics dictionary
    """
    import numpy as np
    
    config = config or {}
    env_registry = get_environment_registry()
    
    if environment_id not in env_registry:
        raise ValueError(f"Unknown environment: {environment_id}")
    
    if dry_run:
        # Return realistic mock metrics for dry-run validation
        return {
            "environment": environment_id,
            "accuracy": 0.75 + 0.05 * np.random.randn(),
            "top5_accuracy": 0.92 + 0.02 * np.random.randn(),
            "loss": 0.5 + 0.1 * np.random.randn(),
            "ece": 0.05 + 0.01 * np.random.randn(),
            "num_samples": 100,
            "metadata": env_registry[environment_id]
        }
    
    # Real evaluation would run model on environment
    raise NotImplementedError("Real evaluation requires model implementation")


# ==============================================================================
# Artifact Writing
# ==============================================================================

def write_environment_registry_artifact(output_dir: str = "results", dry_run: bool = True):
    """
    Write environment registry to artifact file.
    
    Args:
        output_dir: Output directory for artifacts
        dry_run: If True, label as dry-run artifact
    """
    output_path = Path(output_dir) / "environment_registry.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    registry = get_environment_registry()
    
    artifact = {
        "artifact_type": "environment_registry",
        "dry_run": dry_run,
        "description": "Paper-derived environment/task registry with full metadata",
        "environments": registry,
        "num_environments": len(registry)
    }
    
    with open(output_path, "w") as f:
        json.dump(artifact, f, indent=2)
    
    return str(output_path)


def write_dataset_registry_artifact(output_dir: str = "results", dry_run: bool = True):
    """
    Write dataset registry to artifact file.
    
    Args:
        output_dir: Output directory for artifacts
        dry_run: If True, label as dry-run artifact
    """
    output_path = Path(output_dir) / "dataset_registry.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    registry = get_dataset_registry()
    
    artifact = {
        "artifact_type": "dataset_registry",
        "dry_run": dry_run,
        "description": "Paper-derived dataset/benchmark registry with full metadata",
        "datasets": registry,
        "num_datasets": len(registry)
    }
    
    with open(output_path, "w") as f:
        json.dump(artifact, f, indent=2)
    
    return str(output_path)


# ==============================================================================
# Config Interface
# ==============================================================================

def get_default_environment_config(environment_id: str) -> Dict[str, Any]:
    """
    Get default configuration for environment.
    
    Args:
        environment_id: Environment identifier
        
    Returns:
        Default configuration dictionary
    """
    env_registry = get_environment_registry()
    
    if environment_id not in env_registry:
        raise ValueError(f"Unknown environment: {environment_id}")
    
    env_meta = env_registry[environment_id]
    
    return {
        "environment_id": environment_id,
        "batch_size": 64,
        "num_workers": 4,
        "cache_dir": env_meta["setup_metadata"].get("cache_dir", ".cache"),
        **env_meta.get("config_hooks", {})
    }


# ==============================================================================
# Test Interface
# ==============================================================================

def test_environment_registry():
    """Test environment registry integrity."""
    registry = get_environment_registry()
    
    required_envs = ["imagenet", "autonomous_driving", "clip_benchmark"]
    for env_id in required_envs:
        assert env_id in registry, f"Missing required environment: {env_id}"
    
    for env_id, env_meta in registry.items():
        assert "id" in env_meta
        assert "name" in env_meta
        assert "aliases" in env_meta
        assert "factory" in env_meta
        assert "setup_metadata" in env_meta
        assert "config_hooks" in env_meta
    
    return True


def test_dataset_registry():
    """Test dataset registry integrity."""
    registry = get_dataset_registry()
    
    required_datasets = [
        "imagenet", "imagenet_1k", "imagenet_c", "imagenet_r", 
        "imagenet_v2", "imagenet_sketch", "autonomous_driving", "clip_benchmark"
    ]
    
    for ds_id in required_datasets:
        assert ds_id in registry, f"Missing required dataset: {ds_id}"
    
    for ds_id, ds_meta in registry.items():
        assert "id" in ds_meta
        assert "name" in ds_meta
        assert "aliases" in ds_meta
        assert "loader" in ds_meta
        assert "setup_metadata" in ds_meta
        assert "config_hooks" in ds_meta
    
    return True


def test_dry_run_loaders():
    """Test that dry-run loaders return valid objects."""
    # Test dataset loaders
    ds_imagenet = load_imagenet_1k_dataset(dry_run=True)
    assert len(ds_imagenet) > 0
    assert ds_imagenet[0] is not None
    
    ds_imagenet_c = load_imagenet_c_dataset(dry_run=True)
    assert len(ds_imagenet_c) > 0
    
    # Test dataloader
    loader = get_dataloader(ds_imagenet, batch_size=32, dry_run=True)
    assert len(loader) > 0
    batch = next(iter(loader))
    assert batch is not None
    assert "images" in batch
    assert "labels" in batch
    
    # Test environment factory
    env = load_imagenet_environment(dry_run=True)
    assert env is not None
    assert "dataset" in env
    assert "dataloader" in env
    assert "metadata" in env
    
    return True


# ==============================================================================
# Main Entry Point
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Environment and dataset registry")
    parser.add_argument("--mode", choices=["test", "write_artifacts", "list"], default="test")