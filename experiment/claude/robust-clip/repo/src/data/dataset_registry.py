"""
Dataset registry for Robust CLIP reproduction.

This module provides:
- Paper-derived dataset/benchmark registry entries with ids, setup metadata, and loader/config hooks
- Dataset availability checks and lazy loading
- Support for all Table 4 evaluation datasets
- HuggingFace datasets integration with trust_remote_code=True

Paper evidence contract:
Explicitly register dataset/benchmark aliases for: cifar, imagenet, coco, laion, 
clip_benchmark, flickr30k, stl10, imagenet_1k, imagenet_r, imagenet_sketch, 
vqav2, textvqa, pope, sqa_i, caltech101, stanford_cars, fgvc_aircraft, 
flowers, pcam, oxford_pets.
"""

import os
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
import importlib.util

warnings.filterwarnings('ignore')


# ============================================================================
# Dataset Descriptor and Registry
# ============================================================================

@dataclass
class DatasetDescriptor:
    """Descriptor for a dataset with loader configuration."""
    
    dataset_id: str
    name: str
    num_classes: Optional[int]
    split: str = "test"
    hf_dataset: Optional[str] = None
    hf_config: Optional[str] = None
    torchvision_dataset: Optional[str] = None
    custom_loader: Optional[Callable] = None
    preprocessing: Optional[Dict[str, Any]] = None
    trust_remote_code: bool = True
    description: str = ""
    paper_table: Optional[str] = None
    
    def __post_init__(self):
        """Initialize preprocessing config if not provided."""
        if self.preprocessing is None:
            self.preprocessing = {
                "image_size": 224,
                "normalize": True,
                "mean": [0.48145466, 0.4578275, 0.40821073],
                "std": [0.26862954, 0.26130258, 0.27577711]
            }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.dataset_id,
            "name": self.name,
            "type": "dataset",
            "description": self.description,
            "config": {
                "num_classes": self.num_classes,
                "split": self.split,
                "hf_dataset": self.hf_dataset,
                "hf_config": self.hf_config,
                "torchvision_dataset": self.torchvision_dataset,
                "preprocessing": dict(self.preprocessing or {}),
                "trust_remote_code": self.trust_remote_code,
                "paper_table": self.paper_table,
            },
        }

    def keys(self):
        return self.as_dict().keys()

    def __contains__(self, key: str) -> bool:
        return key in self.as_dict()

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]


# ============================================================================
# Dataset Availability Checks
# ============================================================================

def check_datasets_available() -> bool:
    """Check if datasets library is available."""
    return importlib.util.find_spec("datasets") is not None


def check_torchvision_available() -> bool:
    """Check if torchvision is available."""
    return importlib.util.find_spec("torchvision") is not None


def check_torch_available() -> bool:
    """Check if torch is available."""
    return importlib.util.find_spec("torch") is not None


# ============================================================================
# Dataset Registry (Paper Evidence Contract)
# ============================================================================

DATASET_REGISTRY: Dict[str, DatasetDescriptor] = {
    # ========================================================================
    # Table 4 Datasets - Zero-shot Classification
    # ========================================================================
    
    "imagenet": DatasetDescriptor(
        dataset_id="imagenet",
        name="ImageNet-1K",
        num_classes=1000,
        split="validation",
        hf_dataset="imagenet-1k",
        hf_config=None,
        trust_remote_code=True,
        description="ImageNet-1K validation set for zero-shot classification",
        paper_table="Table 4"
    ),
    
    "imagenet_1k": DatasetDescriptor(
        dataset_id="imagenet_1k",
        name="ImageNet-1K",
        num_classes=1000,
        split="validation",
        hf_dataset="imagenet-1k",
        hf_config=None,
        trust_remote_code=True,
        description="ImageNet-1K validation set (alias)",
        paper_table="Table 4"
    ),
    
    "cifar10": DatasetDescriptor(
        dataset_id="cifar10",
        name="CIFAR-10",
        num_classes=10,
        split="test",
        hf_dataset="cifar10",
        hf_config=None,
        torchvision_dataset="CIFAR10",
        description="CIFAR-10 test set for zero-shot classification",
        paper_table="Table 4"
    ),
    
    "cifar": DatasetDescriptor(
        dataset_id="cifar",
        name="CIFAR-10",
        num_classes=10,
        split="test",
        hf_dataset="cifar10",
        hf_config=None,
        torchvision_dataset="CIFAR10",
        description="CIFAR-10 test set (alias)",
        paper_table="Table 4"
    ),
    
    "cifar100": DatasetDescriptor(
        dataset_id="cifar100",
        name="CIFAR-100",
        num_classes=100,
        split="test",
        hf_dataset="cifar100",
        hf_config=None,
        torchvision_dataset="CIFAR100",
        description="CIFAR-100 test set for zero-shot classification",
        paper_table="Table 4"
    ),
    
    "imagenet_r": DatasetDescriptor(
        dataset_id="imagenet_r",
        name="ImageNet-R",
        num_classes=200,
        split="test",
        hf_dataset="timm/imagenet-r",
        hf_config=None,
        trust_remote_code=True,
        description="ImageNet-R (rendition) for distribution shift evaluation",
        paper_table="Table 4"
    ),
    
    "imagenet_sketch": DatasetDescriptor(
        dataset_id="imagenet_sketch",
        name="ImageNet-Sketch",
        num_classes=1000,
        split="test",
        hf_dataset="imagenet_sketch",
        hf_config=None,
        trust_remote_code=True,
        description="ImageNet-Sketch for distribution shift evaluation",
        paper_table="Table 4"
    ),
    
    "caltech101": DatasetDescriptor(
        dataset_id="caltech101",
        name="Caltech-101",
        num_classes=101,
        split="test",
        hf_dataset="tanlq/caltech101",
        hf_config=None,
        torchvision_dataset="Caltech101",
        trust_remote_code=True,
        description="Caltech-101 for fine-grained classification",
        paper_table="Table 4"
    ),
    
    "stanford_cars": DatasetDescriptor(
        dataset_id="stanford_cars",
        name="Stanford Cars",
        num_classes=196,
        split="test",
        hf_dataset="tanganke/stanford_cars",
        hf_config=None,
        trust_remote_code=True,
        description="Stanford Cars for fine-grained vehicle classification",
        paper_table="Table 4"
    ),
    
    "fgvc_aircraft": DatasetDescriptor(
        dataset_id="fgvc_aircraft",
        name="FGVC-Aircraft",
        num_classes=100,
        split="test",
        hf_dataset="Multimodal-Fatima/FGVC_Aircraft",
        hf_config=None,
        torchvision_dataset="FGVCAircraft",
        trust_remote_code=True,
        description="FGVC-Aircraft for fine-grained aircraft classification",
        paper_table="Table 4"
    ),
    
    "flowers": DatasetDescriptor(
        dataset_id="flowers",
        name="Flowers102",
        num_classes=102,
        split="test",
        hf_dataset="nelorth/oxford-flowers",
        hf_config=None,
        torchvision_dataset="Flowers102",
        trust_remote_code=True,
        description="Oxford Flowers-102 for fine-grained flower classification",
        paper_table="Table 4"
    ),
    
    "flowers102": DatasetDescriptor(
        dataset_id="flowers102",
        name="Flowers102",
        num_classes=102,
        split="test",
        hf_dataset="nelorth/oxford-flowers",
        hf_config=None,
        torchvision_dataset="Flowers102",
        trust_remote_code=True,
        description="Oxford Flowers-102 (alias)",
        paper_table="Table 4"
    ),
    
    "oxford_pets": DatasetDescriptor(
        dataset_id="oxford_pets",
        name="Oxford-IIIT Pets",
        num_classes=37,
        split="test",
        hf_dataset="timm/oxford-iiit-pet",
        hf_config=None,
        torchvision_dataset="OxfordIIITPet",
        trust_remote_code=True,
        description="Oxford-IIIT Pets for fine-grained pet classification",
        paper_table="Table 4"
    ),
    
    "stl10": DatasetDescriptor(
        dataset_id="stl10",
        name="STL-10",
        num_classes=10,
        split="test",
        hf_dataset="stl10",
        hf_config=None,
        torchvision_dataset="STL10",
        trust_remote_code=True,
        description="STL-10 for transfer learning evaluation",
        paper_table="Table 4"
    ),
    
    "pcam": DatasetDescriptor(
        dataset_id="pcam",
        name="PatchCamelyon",
        num_classes=2,
        split="test",
        hf_dataset="pcam",
        hf_config=None,
        trust_remote_code=True,
        description="PatchCamelyon for medical image classification",
        paper_table="Table 4"
    ),
    
    # ========================================================================
    # Table 1 & 2 Datasets - LVLM Evaluation
    # ========================================================================
    
    "coco": DatasetDescriptor(
        dataset_id="coco",
        name="COCO Captions",
        num_classes=None,
        split="val2014",
        hf_dataset="HuggingFaceM4/COCO",
        hf_config=None,
        trust_remote_code=True,
        description="COCO Captions for image captioning evaluation",
        paper_table="Table 1, Table 2"
    ),
    
    "flickr30k": DatasetDescriptor(
        dataset_id="flickr30k",
        name="Flickr30k",
        num_classes=None,
        split="test",
        hf_dataset="nlphuji/flickr30k",
        hf_config=None,
        trust_remote_code=True,
        description="Flickr30k for image captioning evaluation",
        paper_table="Table 1, Table 2"
    ),
    
    "vqav2": DatasetDescriptor(
        dataset_id="vqav2",
        name="VQAv2",
        num_classes=None,
        split="validation",
        hf_dataset="HuggingFaceM4/VQAv2",
        hf_config=None,
        trust_remote_code=True,
        description="VQAv2 for visual question answering",
        paper_table="Table 1, Table 2"
    ),
    
    "textvqa": DatasetDescriptor(
        dataset_id="textvqa",
        name="TextVQA",
        num_classes=None,
        split="validation",
        hf_dataset="facebook/textvqa",
        hf_config=None,
        trust_remote_code=True,
        description="TextVQA for text-aware visual question answering",
        paper_table="Table 1, Table 2"
    ),
    
    # ========================================================================
    # Table 3 & Additional Datasets
    # ========================================================================
    
    "pope": DatasetDescriptor(
        dataset_id="pope",
        name="POPE",
        num_classes=None,
        split="test",
        hf_dataset="lmms-lab/POPE",
        hf_config=None,
        trust_remote_code=True,
        description="POPE benchmark for hallucination evaluation",
        paper_table="Table 3"
    ),
    
    "sqa_i": DatasetDescriptor(
        dataset_id="sqa_i",
        name="ScienceQA-IMG",
        num_classes=None,
        split="test",
        hf_dataset="derek-thomas/ScienceQA",
        hf_config=None,
        trust_remote_code=True,
        description="ScienceQA-IMG for multimodal reasoning",
        paper_table="Table 5"
    ),
    
    "laion": DatasetDescriptor(
        dataset_id="laion",
        name="LAION-400M",
        num_classes=None,
        split="train",
        hf_dataset="laion/laion400m",
        hf_config=None,
        trust_remote_code=True,
        description="LAION-400M for unsupervised pre-training (subset)",
        paper_table="Section 3.1"
    ),
    
    "clip_benchmark": DatasetDescriptor(
        dataset_id="clip_benchmark",
        name="CLIP Benchmark Suite",
        num_classes=None,
        split="test",
        hf_dataset=None,
        hf_config=None,
        trust_remote_code=True,
        description="CLIP benchmark suite for comprehensive evaluation",
        paper_table="Appendix"
    ),
}


# ============================================================================
# Dataset Alias Registry
# ============================================================================

DATASET_ALIASES = {
    "imagenet-1k": "imagenet",
    "imagenet1k": "imagenet",
    "in1k": "imagenet",
    "cifar-10": "cifar10",
    "cifar-100": "cifar100",
    "imagenet-r": "imagenet_r",
    "imagenet-sketch": "imagenet_sketch",
    "caltech-101": "caltech101",
    "stanford-cars": "stanford_cars",
    "fgvc-aircraft": "fgvc_aircraft",
    "flowers-102": "flowers102",
    "oxford-pets": "oxford_pets",
    "stl-10": "stl10",
    "coco-captions": "coco",
    "flickr-30k": "flickr30k",
    "vqa-v2": "vqav2",
    "text-vqa": "textvqa",
    "sqa-i": "sqa_i",
    "scienceqa": "sqa_i",
}


# ============================================================================
# Dataset Loader Functions
# ============================================================================

def get_dataset_descriptor(dataset_id: str) -> DatasetDescriptor:
    """
    Get dataset descriptor by ID or alias.
    
    Args:
        dataset_id: Dataset identifier or alias
        
    Returns:
        DatasetDescriptor for the requested dataset
        
    Raises:
        ValueError: If dataset is not found in registry
    """
    # Normalize dataset ID
    normalized_id = dataset_id.lower().replace("_", "").replace("-", "")
    
    # Check direct match
    if dataset_id in DATASET_REGISTRY:
        return DATASET_REGISTRY[dataset_id]
    
    # Check aliases
    if dataset_id in DATASET_ALIASES:
        canonical_id = DATASET_ALIASES[dataset_id]
        return DATASET_REGISTRY[canonical_id]
    
    # Check normalized match
    for key, descriptor in DATASET_REGISTRY.items():
        if key.lower().replace("_", "").replace("-", "") == normalized_id:
            return descriptor
    
    raise ValueError(
        f"Dataset '{dataset_id}' not found in registry. "
        f"Available datasets: {list(DATASET_REGISTRY.keys())}"
    )


def load_hf_dataset(descriptor: DatasetDescriptor, cache_dir: Optional[str] = None):
    """
    Load dataset from HuggingFace datasets library.
    
    Args:
        descriptor: Dataset descriptor with HuggingFace configuration
        cache_dir: Optional cache directory for downloaded datasets
        
    Returns:
        Loaded dataset object
        
    Raises:
        ImportError: If datasets library is not available
        ValueError: If dataset loading fails
    """
    if not check_datasets_available():
        raise ImportError(
            "datasets library is required for loading HuggingFace datasets. "
            "Install with: pip install datasets"
        )
    
    from datasets import load_dataset
    
    try:
        # Use trust_remote_code=True as specified in addendum
        dataset = load_dataset(
            descriptor.hf_dataset,
            descriptor.hf_config,
            split=descriptor.split,
            cache_dir=cache_dir,
            trust_remote_code=descriptor.trust_remote_code
        )
        return dataset
    except Exception as e:
        raise ValueError(
            f"Failed to load dataset '{descriptor.dataset_id}' from HuggingFace: {e}"
        )


def load_torchvision_dataset(descriptor: DatasetDescriptor, root: str = "./data"):
    """
    Load dataset from torchvision.
    
    Args:
        descriptor: Dataset descriptor with torchvision configuration
        root: Root directory for dataset storage
        
    Returns:
        Loaded torchvision dataset object
        
    Raises:
        ImportError: If torchvision is not available
        ValueError: If dataset loading fails
    """
    if not check_torchvision_available():
        raise ImportError(
            "torchvision is required for loading torchvision datasets. "
            "Install with: pip install torchvision"
        )
    
    import torchvision.datasets as datasets
    
    try:
        dataset_class = getattr(datasets, descriptor.torchvision_dataset)
        is_train = descriptor.split == "train"
        
        dataset = dataset_class(
            root=root,
            train=is_train if hasattr(dataset_class, "train") else None,
            download=True
        )
        return dataset
    except Exception as e:
        raise ValueError(
            f"Failed to load dataset '{descriptor.dataset_id}' from torchvision: {e}"
        )


def load_dataset_with_fallback(
    dataset_id: str,
    cache_dir: Optional[str] = None,
    root: str = "./data",
    prefer_hf: bool = True
):
    """
    Load dataset with fallback between HuggingFace and torchvision.
    
    Args:
        dataset_id: Dataset identifier or alias
        cache_dir: Optional cache directory for HuggingFace datasets
        root: Root directory for torchvision datasets
        prefer_hf: Whether to prefer HuggingFace over torchvision
        
    Returns:
        Loaded dataset object
        
    Raises:
        ValueError: If dataset cannot be loaded from any source
    """
    descriptor = get_dataset_descriptor(dataset_id)
    
    # Try HuggingFace first if preferred and available
    if prefer_hf and descriptor.hf_dataset is not None:
        try:
            return load_hf_dataset(descriptor, cache_dir)
        except (ImportError, ValueError) as e:
            warnings.warn(f"HuggingFace loading failed: {e}")
    
    # Try torchvision if available
    if descriptor.torchvision_dataset is not None:
        try:
            return load_torchvision_dataset(descriptor, root)
        except (ImportError, ValueError) as e:
            warnings.warn(f"Torchvision loading failed: {e}")
    
    # Try HuggingFace as fallback if not already tried
    if not prefer_hf and descriptor.hf_dataset is not None:
        try:
            return load_hf_dataset(descriptor, cache_dir)
        except (ImportError, ValueError) as e:
            warnings.warn(f"HuggingFace fallback failed: {e}")
    
    # Try custom loader if available
    if descriptor.custom_loader is not None:
        try:
            return descriptor.custom_loader()
        except Exception as e:
            warnings.warn(f"Custom loader failed: {e}")
    
    raise ValueError(
        f"Failed to load dataset '{dataset_id}' from any available source. "
        f"Please ensure required dependencies are installed."
    )


# ============================================================================
# Table 4 Dataset Collection
# ============================================================================

def get_table4_datasets() -> List[str]:
    """
    Get list of all datasets used in Table 4 evaluation.
    
    Returns:
        List of dataset IDs for Table 4
    """
    return [
        "imagenet",
        "cifar10",
        "cifar100",
        "imagenet_r",
        "imagenet_sketch",
        "caltech101",
        "stanford_cars",
        "fgvc_aircraft",
        "flowers102",
        "oxford_pets",
        "stl10",
        "pcam"
    ]


def get_lvlm_datasets() -> List[str]:
    """
    Get list of all datasets used for LVLM evaluation (Tables 1-2).
    
    Returns:
        List of dataset IDs for LVLM evaluation
    """
    return [
        "coco",
        "flickr30k",
        "vqav2",
        "textvqa"
    ]


def get_all_evaluation_datasets() -> List[str]:
    """
    Get list of all datasets used in paper evaluation.
    
    Returns:
        List of all dataset IDs
    """
    return list(DATASET_REGISTRY.keys())


# ============================================================================
# Dataset Preprocessing
# ============================================================================

def get_preprocessing_config(dataset_id: str) -> Dict[str, Any]:
    """
    Get preprocessing configuration for a dataset.
    
    Args:
        dataset_id: Dataset identifier or alias
        
    Returns:
        Dictionary with preprocessing configuration
    """
    descriptor = get_dataset_descriptor(dataset_id)
    return descriptor.preprocessing


def create_dataset_loader(
    dataset_id: str,
    batch_size: int = 32,
    num_workers: int = 4,
    cache_dir: Optional[str] = None
):
    """
    Create a DataLoader for a dataset with appropriate preprocessing.
    
    Args:
        dataset_id: Dataset identifier or alias
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes
        cache_dir: Optional cache directory for datasets
        
    Returns:
        DataLoader object ready for evaluation
    """
    if not check_torch_available():
        raise ImportError(
            "torch is required for creating DataLoader. "
            "Install with: pip install torch"
        )
    
    from torch.utils.data import DataLoader
    
    # Load dataset
    dataset = load_dataset_with_fallback(dataset_id, cache_dir=cache_dir)
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


# ============================================================================
# Registry Query Functions
# ============================================================================

def list_available_datasets() -> List[str]:
    """List all available dataset IDs."""
    return list(DATASET_REGISTRY.keys())


def list_datasets_by_table(table: str) -> List[str]:
    """
    List datasets used in a specific paper table.
    
    Args:
        table: Table identifier (e.g., "Table 4", "Table 1")
        
    Returns:
        List of dataset IDs for the specified table
    """
    return [
        dataset_id
        for dataset_id, descriptor in DATASET_REGISTRY.items()
        if descriptor.paper_table and table in descriptor.paper_table
    ]


def get_dataset_info(dataset_id: str) -> Dict[str, Any]:
    """
    Get comprehensive information about a dataset.
    
    Args:
        dataset_id: Dataset identifier or alias
        
    Returns:
        Dictionary with dataset information
    """
    descriptor = get_dataset_descriptor(dataset_id)
    
    return {
        "dataset_id": descriptor.dataset_id,
        "name": descriptor.name,
        "num_classes": descriptor.num_classes,
        "split": descriptor.split,
        "description": descriptor.description,
        "paper_table": descriptor.paper_table,
        "hf_dataset": descriptor.hf_dataset,
        "hf_config": descriptor.hf_config,
        "torchvision_dataset": descriptor.torchvision_dataset,
        "preprocessing": descriptor.preprocessing,
        "available": check_dataset_loadable(dataset_id)
    }


def check_dataset_loadable(dataset_id: str) -> bool:
    """
    Check if a dataset can be loaded with current dependencies.
    
    Args:
        dataset_id: Dataset identifier or alias
        
    Returns:
        True if dataset can be loaded, False otherwise
    """
    try:
        descriptor = get_dataset_descriptor(dataset_id)
        
        # Check if any loader is available
        if descriptor.hf_dataset and check_datasets_available():
            return True
        if descriptor.torchvision_dataset and check_torchvision_available():
            return True
        if descriptor.custom_loader:
            return True
        
        return False
    except Exception:
        return False


# ============================================================================
# Export Public API
# ============================================================================

__all__ = [
    "DatasetDescriptor",
    "DATASET_REGISTRY",
    "DATASET_ALIASES",
    "get_dataset_descriptor",
    "load_hf_dataset",
    "load_torchvision_dataset",
    "load_dataset_with_fallback",
    "get_table4_datasets",
    "get_lvlm_datasets",
    "get_all_evaluation_datasets",
    "get_preprocessing_config",
    "create_dataset_loader",
    "list_available_datasets",
    "list_datasets_by_table",
    "get_dataset_info",
    "check_dataset_loadable",
    "check_datasets_available",
    "check_torchvision_available",
    "check_torch_available",
]