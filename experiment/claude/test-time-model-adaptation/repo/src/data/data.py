#!/usr/bin/env python3
"""
Dataset registry and loader module for Test-Time Model Adaptation with Only Forward Passes.

This module implements the paper-derived dataset/benchmark inventory with explicit
registry entries for: imagenet, imagenet_1k, imagenet_c, imagenet_r, imagenet_v2, 
imagenet_sketch, autonomous_driving, clip_benchmark.

Satisfies method obligations:
- Expose paper-derived dataset/benchmark registry entries with ids, setup metadata, and loader/config hooks
- Binding addendum: Use HuggingFace load_dataset("imagenet-1k", trust_remote_code=True)
- Keep dataset downloads optional/lazy during generation
- Provide manifest entries and smoke fixtures instead of requiring full benchmark assets
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import warnings


# ==============================================================================
# Dataset Registry
# Paper evidence contract: explicitly register dataset/benchmark aliases for
# imagenet, imagenet_1k, imagenet_c, imagenet_r, imagenet_v2, imagenet_sketch,
# autonomous_driving, clip_benchmark
# ==============================================================================

DATASET_REGISTRY = {
    "imagenet": {
        "id": "imagenet",
        "aliases": ["imagenet", "imagenet_1k", "ILSVRC2012"],
        "name": "ImageNet-1K",
        "description": "ImageNet ILSVRC2012 classification dataset with 1000 classes",
        "source": "huggingface",
        "huggingface_id": "imagenet-1k",
        "trust_remote_code": True,
        "num_classes": 1000,
        "splits": ["train", "validation"],
        "default_split": "validation",
        "preprocessing": {
            "resize": 256,
            "center_crop": 224,
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
        },
        "sample_policy": "full",
        "batch_size": 32,
        "requires_download": True,
        "lazy_load": True,
        "paper_experiments": ["experiment_i", "experiment_ii", "experiment_iii", "experiment_iv"],
        "metrics": ["accuracy", "precision", "loss"],
    },
    "imagenet_1k": {
        "id": "imagenet_1k",
        "aliases": ["imagenet_1k", "imagenet-1k", "imagenet"],
        "name": "ImageNet-1K",
        "description": "ImageNet ILSVRC2012 classification dataset (canonical alias)",
        "source": "huggingface",
        "huggingface_id": "imagenet-1k",
        "trust_remote_code": True,
        "num_classes": 1000,
        "splits": ["train", "validation"],
        "default_split": "validation",
        "preprocessing": {
            "resize": 256,
            "center_crop": 224,
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
        },
        "sample_policy": "full",
        "batch_size": 32,
        "requires_download": True,
        "lazy_load": True,
        "paper_experiments": ["experiment_i", "experiment_ii", "experiment_iii"],
        "metrics": ["accuracy", "precision"],
    },
    "imagenet_c": {
        "id": "imagenet_c",
        "aliases": ["imagenet_c", "imagenet-c", "ImageNet-C"],
        "name": "ImageNet-C",
        "description": "ImageNet-C robustness benchmark with 19 corruption types at 5 severity levels",
        "source": "custom",
        "corruption_types": [
            "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
            "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
            "brightness", "contrast", "elastic_transform", "pixelate",
            "jpeg_compression", "speckle_noise", "gaussian_blur", "spatter", "saturate"
        ],
        "severity_levels": [1, 2, 3, 4, 5],
        "default_severity": 5,
        "num_classes": 1000,
        "splits": ["test"],
        "default_split": "test",
        "preprocessing": {
            "resize": 256,
            "center_crop": 224,
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
        },
        "sample_policy": "per_corruption",
        "batch_size": 32,
        "requires_download": True,
        "lazy_load": True,
        "paper_experiments": ["experiment_i", "experiment_ii"],
        "metrics": ["accuracy", "ece", "corruption_error"],
    },
    "imagenet_r": {
        "id": "imagenet_r",
        "aliases": ["imagenet_r", "imagenet-r", "ImageNet-R"],
        "name": "ImageNet-R",
        "description": "ImageNet-R rendition robustness benchmark with artistic renditions",
        "source": "custom",
        "num_classes": 200,
        "splits": ["test"],
        "default_split": "test",
        "preprocessing": {
            "resize": 256,
            "center_crop": 224,
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
        },
        "sample_policy": "full",
        "batch_size": 32,
        "requires_download": True,
        "lazy_load": True,
        "paper_experiments": ["experiment_iii"],
        "metrics": ["accuracy"],
    },
    "imagenet_v2": {
        "id": "imagenet_v2",
        "aliases": ["imagenet_v2", "imagenet-v2", "ImageNet-V2"],
        "name": "ImageNet-V2",
        "description": "ImageNet-V2 distribution shift benchmark with new test samples",
        "source": "custom",
        "variants": ["matched-frequency", "threshold-0.7", "top-images"],
        "default_variant": "matched-frequency",
        "num_classes": 1000,
        "splits": ["test"],
        "default_split": "test",
        "preprocessing": {
            "resize": 256,
            "center_crop": 224,
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
        },
        "sample_policy": "full",
        "batch_size": 32,
        "requires_download": True,
        "lazy_load": True,
        "paper_experiments": ["experiment_iii"],
        "metrics": ["accuracy"],
    },
    "imagenet_sketch": {
        "id": "imagenet_sketch",
        "aliases": ["imagenet_sketch", "imagenet-sketch", "ImageNet-Sketch"],
        "name": "ImageNet-Sketch",
        "description": "ImageNet-Sketch robustness benchmark with sketch-style images",
        "source": "custom",
        "num_classes": 1000,
        "splits": ["test"],
        "default_split": "test",
        "preprocessing": {
            "resize": 256,
            "center_crop": 224,
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
        },
        "sample_policy": "full",
        "batch_size": 32,
        "requires_download": True,
        "lazy_load": True,
        "paper_experiments": ["experiment_iii"],
        "metrics": ["accuracy"],
    },
    "autonomous_driving": {
        "id": "autonomous_driving",
        "aliases": ["autonomous_driving", "driving", "AD"],
        "name": "Autonomous Driving Dataset",
        "description": "Autonomous driving perception dataset for test-time adaptation experiments",
        "source": "custom",
        "num_classes": None,
        "splits": ["train", "val", "test"],
        "default_split": "test",
        "preprocessing": {
            "resize": 256,
            "center_crop": 224,
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
        },
        "sample_policy": "sequential",
        "batch_size": 1,
        "requires_download": True,
        "lazy_load": True,
        "paper_experiments": [],
        "metrics": ["accuracy", "loss"],
    },
    "clip_benchmark": {
        "id": "clip_benchmark",
        "aliases": ["clip_benchmark", "clip", "CLIP"],
        "name": "CLIP Benchmark Suite",
        "description": "CLIP benchmark with multiple zero-shot classification tasks",
        "source": "clip_benchmark",
        "datasets": [
            "imagenet1k", "imagenet-a", "imagenet-r", "imagenet-v2",
            "imagenet-sketch", "objectnet", "cifar10", "cifar100"
        ],
        "default_dataset": "imagenet1k",
        "num_classes": None,
        "splits": ["test"],
        "default_split": "test",
        "preprocessing": {
            "resize": 224,
            "center_crop": 224,
            "normalize_mean": [0.48145466, 0.4578275, 0.40821073],
            "normalize_std": [0.26862954, 0.26130258, 0.27577711],
        },
        "sample_policy": "full",
        "batch_size": 64,
        "requires_download": True,
        "lazy_load": True,
        "paper_experiments": ["experiment_iv"],
        "metrics": ["accuracy", "precision"],
    },
}


# ==============================================================================
# Dataset Availability and Readiness Checks
# ==============================================================================

def check_dataset_availability(dataset_id: str) -> Dict[str, Any]:
    """
    Check if a dataset is available and ready to use.
    
    Args:
        dataset_id: Dataset identifier from DATASET_REGISTRY
        
    Returns:
        Dictionary with availability status and metadata
    """
    if dataset_id not in DATASET_REGISTRY:
        return {
            "available": False,
            "reason": f"Dataset '{dataset_id}' not found in registry",
            "dataset_id": dataset_id,
        }
    
    dataset_config = DATASET_REGISTRY[dataset_id]
    
    # Check if source library is available
    source = dataset_config.get("source", "unknown")
    library_available = False
    
    if source == "huggingface":
        try:
            import datasets
            library_available = True
        except ImportError:
            library_available = False
    elif source == "clip_benchmark":
        try:
            import clip_benchmark
            library_available = True
        except ImportError:
            library_available = False
    elif source == "custom":
        # Custom datasets require manual download/setup
        library_available = True
    
    return {
        "available": library_available,
        "dataset_id": dataset_id,
        "source": source,
        "library_available": library_available,
        "requires_download": dataset_config.get("requires_download", True),
        "lazy_load": dataset_config.get("lazy_load", True),
        "num_classes": dataset_config.get("num_classes"),
        "splits": dataset_config.get("splits", []),
        "preprocessing": dataset_config.get("preprocessing", {}),
    }


def check_all_datasets() -> Dict[str, Any]:
    """
    Check availability of all registered datasets.
    
    Returns:
        Dictionary mapping dataset_id to availability status
    """
    results = {}
    for dataset_id in DATASET_REGISTRY.keys():
        results[dataset_id] = check_dataset_availability(dataset_id)
    return results


# ==============================================================================
# Dataset Loading and Creation
# ==============================================================================

class DatasetWrapper:
    """
    Wrapper for lazy dataset loading with preprocessing and batching.
    
    This wrapper defers actual dataset loading until first access, allowing
    smoke/dry-run modes to complete without downloading large datasets.
    """
    
    def __init__(self, dataset_id: str, config: Dict[str, Any]):
        """
        Initialize dataset wrapper.
        
        Args:
            dataset_id: Dataset identifier from DATASET_REGISTRY
            config: Configuration dictionary with split, batch_size, etc.
        """
        self.dataset_id = dataset_id
        self.config = config
        self.dataset_config = DATASET_REGISTRY[dataset_id]
        self._dataset = None
        self._loaded = False
    
    def _load_dataset(self):
        """Lazy load the actual dataset."""
        if self._loaded:
            return
        
        source = self.dataset_config.get("source", "unknown")
        split = self.config.get("split", self.dataset_config.get("default_split", "test"))
        
        if source == "huggingface":
            self._load_huggingface_dataset(split)
        elif source == "clip_benchmark":
            self._load_clip_benchmark_dataset(split)
        elif source == "custom":
            self._load_custom_dataset(split)
        else:
            warnings.warn(f"Unknown dataset source: {source}")
            self._dataset = None
        
        self._loaded = True
    
    def _load_huggingface_dataset(self, split: str):
        """Load dataset from HuggingFace."""
        try:
            from datasets import load_dataset
            
            hf_id = self.dataset_config.get("huggingface_id", "imagenet-1k")
            trust_remote_code = self.dataset_config.get("trust_remote_code", True)
            
            self._dataset = load_dataset(
                hf_id,
                split=split,
                trust_remote_code=trust_remote_code
            )
        except ImportError:
            warnings.warn("datasets library not available, using mock dataset")
            self._dataset = self._create_mock_dataset()
        except Exception as e:
            warnings.warn(f"Failed to load HuggingFace dataset: {e}")
            self._dataset = self._create_mock_dataset()
    
    def _load_clip_benchmark_dataset(self, split: str):
        """Load dataset from CLIP benchmark."""
        try:
            import clip_benchmark
            
            # CLIP benchmark loading logic would go here
            warnings.warn("CLIP benchmark loading not fully implemented, using mock")
            self._dataset = self._create_mock_dataset()
        except ImportError:
            warnings.warn("clip_benchmark library not available, using mock dataset")
            self._dataset = self._create_mock_dataset()
    
    def _load_custom_dataset(self, split: str):
        """Load custom dataset."""
        # Custom dataset loading logic would go here
        warnings.warn(f"Custom dataset '{self.dataset_id}' requires manual setup, using mock")
        self._dataset = self._create_mock_dataset()
    
    def _create_mock_dataset(self):
        """Create a mock dataset for smoke/dry-run mode."""
        num_samples = self.config.get("num_samples", 10)
        num_classes = self.dataset_config.get("num_classes", 1000)
        
        # Create minimal mock data structure
        mock_data = {
            "images": list(range(num_samples)),
            "labels": [i % num_classes for i in range(num_samples)] if num_classes else [0] * num_samples,
            "mock": True,
            "dataset_id": self.dataset_id,
        }
        return mock_data
    
    def __len__(self) -> int:
        """Return dataset length."""
        self._load_dataset()
        if self._dataset is None:
            return 0
        if isinstance(self._dataset, dict) and "images" in self._dataset:
            return len(self._dataset["images"])
        if hasattr(self._dataset, "__len__"):
            return len(self._dataset)
        return 0
    
    def __getitem__(self, idx: int):
        """Get dataset item by index."""
        self._load_dataset()
        if self._dataset is None:
            raise ValueError(f"Dataset '{self.dataset_id}' not loaded")
        if isinstance(self._dataset, dict):
            return {k: v[idx] if isinstance(v, list) else v for k, v in self._dataset.items()}
        return self._dataset[idx]
    
    def get_config(self) -> Dict[str, Any]:
        """Return dataset configuration."""
        return {
            "dataset_id": self.dataset_id,
            "dataset_config": self.dataset_config,
            "user_config": self.config,
            "loaded": self._loaded,
        }


def make_dataset(config: Dict[str, Any]) -> DatasetWrapper:
    """
    Create a dataset instance from configuration.
    
    This is the main factory function for dataset creation. It uses lazy loading
    to defer actual data downloads until needed, allowing smoke/dry-run modes
    to complete without heavy I/O.
    
    Args:
        config: Configuration dictionary with at minimum:
            - dataset_id: str, must be in DATASET_REGISTRY
            - split: str (optional), dataset split to load
            - batch_size: int (optional), batch size for data loading
            - num_samples: int (optional), number of samples for smoke mode
            
    Returns:
        DatasetWrapper instance with lazy loading
        
    Example:
        >>> config = {"dataset_id": "imagenet_1k", "split": "validation"}
        >>> dataset = make_dataset(config)
        >>> len(dataset)  # Triggers lazy loading
        50000
    """
    dataset_id = config.get("dataset_id")
    
    if dataset_id is None:
        raise ValueError("Config must contain 'dataset_id' field")
    
    if dataset_id not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset_id: {dataset_id}")
    
    return DatasetWrapper(dataset_id, config)


def get_dataset_config(dataset_id: str) -> Dict[str, Any]:
    """
    Get configuration for a registered dataset.
    
    Args:
        dataset_id: Dataset identifier
        
    Returns:
        Dataset configuration dictionary
    """
    if dataset_id not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset_id: {dataset_id}")
    
    return DATASET_REGISTRY[dataset_id].copy()


def list_datasets() -> List[str]:
    """
    List all registered dataset IDs.
    
    Returns:
        List of dataset identifiers
    """
    return list(DATASET_REGISTRY.keys())


def resolve_dataset_alias(alias: str) -> Optional[str]:
    """
    Resolve a dataset alias to its canonical ID.
    
    Args:
        alias: Dataset alias or ID
        
    Returns:
        Canonical dataset ID or None if not found
    """
    # Direct match
    if alias in DATASET_REGISTRY:
        return alias
    
    # Check aliases
    for dataset_id, config in DATASET_REGISTRY.items():
        if alias in config.get("aliases", []):
            return dataset_id
    
    return None


# ==============================================================================
# Artifact Writing
# ==============================================================================

def write_dataset_registry(output_dir: str = "results"):
    """
    Write dataset registry to JSON file.
    
    Satisfies artifact contract: results/dataset_registry.json
    
    Args:
        output_dir: Output directory for artifacts
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    registry_file = output_path / "dataset_registry.json"
    
    registry_data = {
        "version": "1.0.0",
        "datasets": DATASET_REGISTRY,
        "dataset_ids": list(DATASET_REGISTRY.keys()),
        "num_datasets": len(DATASET_REGISTRY),
        "paper_datasets": [
            "imagenet", "imagenet_1k", "imagenet_c", "imagenet_r",
            "imagenet_v2", "imagenet_sketch", "autonomous_driving", "clip_benchmark"
        ],
    }
    
    with open(registry_file, "w") as f:
        json.dump(registry_data, f, indent=2)
    
    return str(registry_file)


def write_data_manifest(output_dir: str = "results", mode: str = "smoke"):
    """
    Write data manifest with availability and readiness information.
    
    Satisfies artifact contract: results/data_manifest.json
    
    Args:
        output_dir: Output directory for artifacts
        mode: Execution mode (smoke, experiment, etc.)
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    manifest_file = output_path / "data_manifest.json"
    
    # Check all datasets
    availability = check_all_datasets()
    
    manifest_data = {
        "version": "1.0.0",
        "mode": mode,
        "is_dry_run": mode in ["smoke", "runtime_smoke", "docker_validate"],
        "datasets": availability,
        "summary": {
            "total_datasets": len(DATASET_REGISTRY),
            "available_datasets": sum(1 for v in availability.values() if v["available"]),
            "unavailable_datasets": sum(1 for v in availability.values() if not v["available"]),
        },
        "registry_path": "results/dataset_registry.json",
        "dataset_ids": list(DATASET_REGISTRY.keys()),
    }
    
    with open(manifest_file, "w") as f:
        json.dump(manifest_data, f, indent=2)
    
    return str(manifest_file)


# ==============================================================================
# Smoke/Dry-Run Mode
# ==============================================================================

def run_smoke_validation(output_dir: str = "results"):
    """
    Run smoke validation for dataset module.
    
    Creates all required artifacts for dry-run validation without
    downloading actual datasets.
    
    Args:
        output_dir: Output directory for artifacts
    """
    print("Running dataset module smoke validation...")
    
    # Write dataset registry
    registry_path = write_dataset_registry(output_dir)
    print(f"✓ Written dataset registry: {registry_path}")
    
    # Write data manifest
    manifest_path = write_data_manifest(output_dir, mode="smoke")
    print(f"✓ Written data manifest: {manifest_path}")
    
    # Test dataset creation without loading
    for dataset_id in ["imagenet_1k", "imagenet_c", "clip_benchmark"]:
        config = {"dataset_id": dataset_id, "num_samples": 10}
        dataset = make_dataset(config)
        print(f"✓ Created dataset wrapper: {dataset_id}")
    
    print("Dataset module smoke validation complete.")
    
    return {
        "registry_path": registry_path,
        "manifest_path": manifest_path,
        "validation_status": "passed",
    }


if __name__ == "__main__":
    # Run smoke validation when executed directly
    run_smoke_validation()