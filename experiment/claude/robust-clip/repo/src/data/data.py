"""
Data pipeline and dataset registry for Robust CLIP reproduction.

This module implements:
- Dataset registry with metadata and loader factories
- HuggingFace and torchvision dataset loaders
- Transform pipelines for CLIP preprocessing
- Batch loading utilities for training and evaluation
- Dataset availability checks and error handling

Paper evidence contract:
- Register datasets: imagenet, coco, flickr30k, vqav2, textvqa, pope, sqa_i,
  cifar10, cifar100, stl10, imagenet_r, imagenet_sketch, caltech101,
  stanford_cars, fgvc_aircraft, flowers102, pcam, oxford_pets
- Use HuggingFace datasets with trust_remote_code=True for ImageNet
- Support CLIP preprocessing and augmentation transforms
- Provide clean and adversarial data loading pipelines
"""

import os
import json
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field, asdict
import importlib.util

warnings.filterwarnings('ignore')


# ============================================================================
# Availability Checks (Import-Light Pattern)
# ============================================================================

def _check_availability(module_name: str) -> bool:
    """Check if a module is available without importing it."""
    return importlib.util.find_spec(module_name) is not None


HF_DATASETS_AVAILABLE = _check_availability('datasets')
TORCH_AVAILABLE = _check_availability('torch')
TORCHVISION_AVAILABLE = _check_availability('torchvision')
PIL_AVAILABLE = _check_availability('PIL')


# ============================================================================
# Dataset Registry Entry
# ============================================================================

@dataclass
class DatasetRegistryEntry:
    """Registry entry for a dataset with metadata and loader factory."""
    
    dataset_id: str
    name: str
    task_type: str  # classification, retrieval, vqa, captioning
    num_classes: Optional[int] = None
    split_names: List[str] = field(default_factory=lambda: ['train', 'val', 'test'])
    
    # Loader configuration
    source: str = 'huggingface'  # huggingface, torchvision, custom
    hf_dataset_name: Optional[str] = None
    hf_config_name: Optional[str] = None
    torchvision_class: Optional[str] = None
    
    # Preprocessing
    image_size: int = 224
    requires_trust_remote_code: bool = False
    
    # Metadata
    description: str = ''
    paper_reference: str = ''
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# ============================================================================
# Dataset Registry (Paper Evidence Contract)
# ============================================================================

DATASET_REGISTRY: Dict[str, DatasetRegistryEntry] = {}


def register_dataset(entry: DatasetRegistryEntry):
    """Register a dataset entry."""
    DATASET_REGISTRY[entry.dataset_id] = entry
    # Register aliases
    if entry.hf_dataset_name and entry.hf_dataset_name != entry.dataset_id:
        DATASET_REGISTRY[entry.hf_dataset_name] = entry


# ImageNet variants (primary training dataset)
register_dataset(DatasetRegistryEntry(
    dataset_id='imagenet',
    name='ImageNet-1K',
    task_type='classification',
    num_classes=1000,
    split_names=['train', 'validation'],
    source='huggingface',
    hf_dataset_name='imagenet-1k',
    requires_trust_remote_code=True,
    description='ImageNet ILSVRC2012 classification dataset',
    paper_reference='Used for FARE and TeCoA fine-tuning (Table 4, Sec 4.1)'
))

register_dataset(DatasetRegistryEntry(
    dataset_id='imagenet_1k',
    name='ImageNet-1K',
    task_type='classification',
    num_classes=1000,
    split_names=['train', 'validation'],
    source='huggingface',
    hf_dataset_name='imagenet-1k',
    requires_trust_remote_code=True,
    description='ImageNet ILSVRC2012 (alias)',
    paper_reference='Table 4'
))

register_dataset(DatasetRegistryEntry(
    dataset_id='imagenet_r',
    name='ImageNet-R',
    task_type='classification',
    num_classes=200,
    split_names=['test'],
    source='huggingface',
    hf_dataset_name='imagenet_r',
    description='ImageNet Rendition robustness benchmark',
    paper_reference='Table 4 robustness evaluation'
))

register_dataset(DatasetRegistryEntry(
    dataset_id='imagenet_sketch',
    name='ImageNet-Sketch',
    task_type='classification',
    num_classes=1000,
    split_names=['test'],
    source='huggingface',
    hf_dataset_name='imagenet_sketch',
    description='ImageNet Sketch robustness benchmark',
    paper_reference='Table 4 robustness evaluation'
))

# CIFAR variants
register_dataset(DatasetRegistryEntry(
    dataset_id='cifar10',
    name='CIFAR-10',
    task_type='classification',
    num_classes=10,
    split_names=['train', 'test'],
    source='torchvision',
    torchvision_class='CIFAR10',
    description='CIFAR-10 image classification',
    paper_reference='Table 4'
))

register_dataset(DatasetRegistryEntry(
    dataset_id='cifar100',
    name='CIFAR-100',
    task_type='classification',
    num_classes=100,
    split_names=['train', 'test'],
    source='torchvision',
    torchvision_class='CIFAR100',
    description='CIFAR-100 image classification',
    paper_reference='Table 4'
))

register_dataset(DatasetRegistryEntry(
    dataset_id='cifar',
    name='CIFAR-10',
    task_type='classification',
    num_classes=10,
    split_names=['train', 'test'],
    source='torchvision',
    torchvision_class='CIFAR10',
    description='CIFAR-10 (alias)',
    paper_reference='Table 4'
))

# STL-10
register_dataset(DatasetRegistryEntry(
    dataset_id='stl10',
    name='STL-10',
    task_type='classification',
    num_classes=10,
    split_names=['train', 'test'],
    source='torchvision',
    torchvision_class='STL10',
    description='STL-10 image classification',
    paper_reference='Table 4'
))

# Fine-grained classification datasets
register_dataset(DatasetRegistryEntry(
    dataset_id='caltech101',
    name='Caltech-101',
    task_type='classification',
    num_classes=101,
    split_names=['train', 'test'],
    source='torchvision',
    torchvision_class='Caltech101',
    description='Caltech-101 object recognition',
    paper_reference='Table 4'
))

register_dataset(DatasetRegistryEntry(
    dataset_id='stanford_cars',
    name='Stanford Cars',
    task_type='classification',
    num_classes=196,
    split_names=['train', 'test'],
    source='huggingface',
    hf_dataset_name='Multimodal-Fatima/StanfordCars_train',
    description='Stanford Cars fine-grained classification',
    paper_reference='Table 4'
))

register_dataset(DatasetRegistryEntry(
    dataset_id='fgvc_aircraft',
    name='FGVC Aircraft',
    task_type='classification',
    num_classes=100,
    split_names=['train', 'val', 'test'],
    source='torchvision',
    torchvision_class='FGVCAircraft',
    description='Fine-Grained Visual Classification of Aircraft',
    paper_reference='Table 4'
))

register_dataset(DatasetRegistryEntry(
    dataset_id='flowers',
    name='Oxford Flowers-102',
    task_type='classification',
    num_classes=102,
    split_names=['train', 'val', 'test'],
    source='torchvision',
    torchvision_class='Flowers102',
    description='Oxford 102 Flower Categories',
    paper_reference='Table 4'
))

register_dataset(DatasetRegistryEntry(
    dataset_id='flowers102',
    name='Oxford Flowers-102',
    task_type='classification',
    num_classes=102,
    split_names=['train', 'val', 'test'],
    source='torchvision',
    torchvision_class='Flowers102',
    description='Oxford 102 Flower Categories (alias)',
    paper_reference='Table 4'
))

register_dataset(DatasetRegistryEntry(
    dataset_id='oxford_pets',
    name='Oxford-IIIT Pets',
    task_type='classification',
    num_classes=37,
    split_names=['trainval', 'test'],
    source='torchvision',
    torchvision_class='OxfordIIITPet',
    description='Oxford-IIIT Pet Dataset',
    paper_reference='Table 4'
))

register_dataset(DatasetRegistryEntry(
    dataset_id='pcam',
    name='PatchCamelyon',
    task_type='classification',
    num_classes=2,
    split_names=['train', 'validation', 'test'],
    source='huggingface',
    hf_dataset_name='1aurent/PatchCamelyon',
    description='PatchCamelyon histopathology classification',
    paper_reference='Table 4'
))

# Vision-language datasets
register_dataset(DatasetRegistryEntry(
    dataset_id='coco',
    name='MS-COCO',
    task_type='captioning',
    split_names=['train', 'val', 'test'],
    source='huggingface',
    hf_dataset_name='HuggingFaceM4/COCO',
    description='MS-COCO image captioning and detection',
    paper_reference='Table 1, Table 2 (captioning)'
))

register_dataset(DatasetRegistryEntry(
    dataset_id='flickr30k',
    name='Flickr30k',
    task_type='captioning',
    split_names=['train', 'val', 'test'],
    source='huggingface',
    hf_dataset_name='nlphuji/flickr30k',
    description='Flickr30k image captioning',
    paper_reference='Table 1, Table 2 (captioning)'
))

# VQA datasets
register_dataset(DatasetRegistryEntry(
    dataset_id='vqav2',
    name='VQAv2',
    task_type='vqa',
    split_names=['train', 'val'],
    source='huggingface',
    hf_dataset_name='HuggingFaceM4/VQAv2',
    description='Visual Question Answering v2',
    paper_reference='Table 1, Table 2 (VQA accuracy)'
))

register_dataset(DatasetRegistryEntry(
    dataset_id='textvqa',
    name='TextVQA',
    task_type='vqa',
    split_names=['train', 'val', 'test'],
    source='huggingface',
    hf_dataset_name='facebook/textvqa',
    description='Text-based Visual Question Answering',
    paper_reference='Table 1, Table 2 (VQA accuracy)'
))

# Hallucination and reasoning benchmarks
register_dataset(DatasetRegistryEntry(
    dataset_id='pope',
    name='POPE',
    task_type='hallucination',
    split_names=['test'],
    source='huggingface',
    hf_dataset_name='lmms-lab/POPE',
    description='Polling-based Object Probing Evaluation for hallucination',
    paper_reference='Table 5 (hallucination benchmark)'
))

register_dataset(DatasetRegistryEntry(
    dataset_id='sqa_i',
    name='ScienceQA-IMG',
    task_type='reasoning',
    split_names=['train', 'val', 'test'],
    source='huggingface',
    hf_dataset_name='derek-thomas/ScienceQA',
    description='ScienceQA with images (Chain-of-Thought reasoning)',
    paper_reference='Table 6 (CoT reasoning)'
))

# Training data
register_dataset(DatasetRegistryEntry(
    dataset_id='laion',
    name='LAION-400M',
    task_type='pretraining',
    split_names=['train'],
    source='custom',
    description='LAION-400M image-text pairs (reference only)',
    paper_reference='Original CLIP pretraining data',
    enabled=False  # Not directly loaded
))

register_dataset(DatasetRegistryEntry(
    dataset_id='clip_benchmark',
    name='CLIP Benchmark Suite',
    task_type='benchmark',
    split_names=['test'],
    source='custom',
    description='CLIP benchmark evaluation suite',
    paper_reference='Zero-shot classification benchmarks',
    enabled=False  # Meta-dataset
))


# ============================================================================
# Dataset Loader Factory
# ============================================================================

class DatasetLoader:
    """Factory for loading datasets with lazy imports and availability checks."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize dataset loader.
        
        Args:
            config: Configuration dictionary with data_root, cache_dir, etc.
        """
        self.config = config or {}
        self.data_root = self.config.get('data_root', './data')
        self.cache_dir = self.config.get('cache_dir', './cache')
        
        # Create directories
        Path(self.data_root).mkdir(parents=True, exist_ok=True)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
    
    def load_dataset(
        self,
        dataset_id: str,
        split: str = 'train',
        transform: Optional[Any] = None,
        download: bool = True
    ) -> Any:
        """Load a dataset by ID.
        
        Args:
            dataset_id: Dataset identifier from registry
            split: Dataset split (train, val, test, etc.)
            transform: Optional transform pipeline
            download: Whether to download if not cached
            
        Returns:
            Dataset object (HuggingFace Dataset or torchvision Dataset)
            
        Raises:
            ValueError: If dataset not found in registry
            ImportError: If required dependencies not available
        """
        if dataset_id not in DATASET_REGISTRY:
            raise ValueError(
                f"Dataset '{dataset_id}' not found in registry. "
                f"Available: {list(DATASET_REGISTRY.keys())}"
            )
        
        entry = DATASET_REGISTRY[dataset_id]
        
        if not entry.enabled:
            raise ValueError(f"Dataset '{dataset_id}' is not enabled")
        
        if entry.source == 'huggingface':
            return self._load_huggingface_dataset(entry, split, transform)
        elif entry.source == 'torchvision':
            return self._load_torchvision_dataset(entry, split, transform, download)
        elif entry.source == 'custom':
            raise NotImplementedError(
                f"Custom loader for '{dataset_id}' not implemented. "
                f"This is a reference dataset."
            )
        else:
            raise ValueError(f"Unknown source: {entry.source}")
    
    def _load_huggingface_dataset(
        self,
        entry: DatasetRegistryEntry,
        split: str,
        transform: Optional[Any]
    ) -> Any:
        """Load HuggingFace dataset with lazy import."""
        if not HF_DATASETS_AVAILABLE:
            raise ImportError(
                "HuggingFace datasets library not available. "
                "Install with: pip install datasets"
            )
        
        from datasets import load_dataset
        
        # Load dataset
        dataset_name = entry.hf_dataset_name or entry.dataset_id
        config_name = entry.hf_config_name
        
        load_kwargs = {
            'cache_dir': self.cache_dir,
        }
        
        if entry.requires_trust_remote_code:
            load_kwargs['trust_remote_code'] = True
        
        if config_name:
            load_kwargs['name'] = config_name
        
        # Map split names
        split_map = {
            'train': 'train',
            'val': 'validation',
            'validation': 'validation',
            'test': 'test'
        }
        hf_split = split_map.get(split, split)
        
        try:
            dataset = load_dataset(dataset_name, split=hf_split, **load_kwargs)
        except Exception as e:
            # Try without split for datasets that don't support it
            try:
                dataset = load_dataset(dataset_name, **load_kwargs)
                if hf_split in dataset:
                    dataset = dataset[hf_split]
            except Exception:
                raise ImportError(
                    f"Failed to load HuggingFace dataset '{dataset_name}': {e}"
                )
        
        # Wrap with transform if provided
        if transform is not None:
            dataset = HuggingFaceDatasetWrapper(dataset, transform)
        
        return dataset
    
    def _load_torchvision_dataset(
        self,
        entry: DatasetRegistryEntry,
        split: str,
        transform: Optional[Any],
        download: bool
    ) -> Any:
        """Load torchvision dataset with lazy import."""
        if not TORCHVISION_AVAILABLE:
            raise ImportError(
                "torchvision library not available. "
                "Install with: pip install torchvision"
            )
        
        import torchvision.datasets as datasets
        
        dataset_class_name = entry.torchvision_class
        if not hasattr(datasets, dataset_class_name):
            raise ValueError(
                f"torchvision dataset class '{dataset_class_name}' not found"
            )
        
        dataset_class = getattr(datasets, dataset_class_name)
        
        # Map split to train flag
        train = (split == 'train')
        
        # Special handling for different datasets
        kwargs = {
            'root': self.data_root,
            'download': download,
        }
        
        if transform is not None:
            kwargs['transform'] = transform
        
        # Add split-specific arguments
        if dataset_class_name in ['CIFAR10', 'CIFAR100', 'STL10']:
            kwargs['train'] = train
        elif dataset_class_name == 'OxfordIIITPet':
            kwargs['split'] = 'trainval' if train else 'test'
        elif dataset_class_name in ['Flowers102', 'FGVCAircraft']:
            split_map = {'train': 'train', 'val': 'val', 'test': 'test'}
            kwargs['split'] = split_map.get(split, 'test')
        elif dataset_class_name == 'Caltech101':
            # Caltech101 doesn't have explicit splits
            pass
        
        try:
            dataset = dataset_class(**kwargs)
        except Exception as e:
            raise ImportError(
                f"Failed to load torchvision dataset '{dataset_class_name}': {e}"
            )
        
        return dataset
    
    def get_transforms(
        self,
        mode: str = 'eval',
        image_size: int = 224,
        model_type: str = 'clip'
    ) -> Any:
        """Get preprocessing transforms for a model.
        
        Args:
            mode: 'train' or 'eval'
            image_size: Target image size
            model_type: Model type (clip, vit, etc.)
            
        Returns:
            Transform pipeline
        """
        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision required for transforms")
        
        from torchvision import transforms
        
        if model_type == 'clip':
            # CLIP preprocessing (from OpenCLIP)
            if mode == 'train':
                transform = transforms.Compose([
                    transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.48145466, 0.4578275, 0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711)
                    )
                ])
            else:
                transform = transforms.Compose([
                    transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.48145466, 0.4578275, 0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711)
                    )
                ])
        else:
            # Generic ImageNet preprocessing
            if mode == 'train':
                transform = transforms.Compose([
                    transforms.RandomResizedCrop(image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
            else:
                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
        
        return transform


class HuggingFaceDatasetWrapper:
    """Wrapper to add transforms to HuggingFace datasets."""
    
    def __init__(self, dataset: Any, transform: Any):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Handle different HuggingFace dataset formats
        if 'image' in item:
            image = item['image']
        elif 'img' in item:
            image = item['img']
        else:
            # Try to find PIL image in item
            for v in item.values():
                if PIL_AVAILABLE:
                    from PIL import Image
                    if isinstance(v, Image.Image):
                        image = v
                        break
            else:
                raise ValueError(f"Could not find image in dataset item: {item.keys()}")
        
        # Apply transform
        if self.transform is not None:
            if not PIL_AVAILABLE:
                raise ImportError("PIL required for image transforms")
            from PIL import Image
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
            image = self.transform(image)
        
        # Return image and label if available
        if 'label' in item:
            return image, item['label']
        else:
            return image, item


# ============================================================================
# Data Loading Utilities
# ============================================================================

def get_dataloader(
    dataset_id: str,
    split: str = 'train',
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    transform: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None
) -> Any:
    """Create a DataLoader for a dataset.
    
    Args:
        dataset_id: Dataset identifier
        split: Dataset split
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle: Whether to shuffle data
        transform: Optional transform pipeline
        config: Configuration dictionary
        
    Returns:
        DataLoader object
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for DataLoader")
    
    from torch.utils.data import DataLoader
    
    loader = DatasetLoader(config)
    dataset = loader.load_dataset(dataset_id, split, transform)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


def list_available_datasets() -> List[str]:
    """List all available dataset IDs."""
    return [k for k, v in DATASET_REGISTRY.items() if v.enabled]


def get_dataset_info(dataset_id: str) -> Dict[str, Any]:
    """Get metadata for a dataset.
    
    Args:
        dataset_id: Dataset identifier
        
    Returns:
        Dataset metadata dictionary
    """
    if dataset_id not in DATASET_REGISTRY:
        raise ValueError(f"Dataset '{dataset_id}' not found")
    
    return DATASET_REGISTRY[dataset_id].to_dict()


def check_dataset_availability(dataset_id: str) -> Tuple[bool, str]:
    """Check if a dataset can be loaded.
    
    Args:
        dataset_id: Dataset identifier
        
    Returns:
        (available, message) tuple
    """
    if dataset_id not in DATASET_REGISTRY:
        return False, f"Dataset '{dataset_id}' not in registry"
    
    entry = DATASET_REGISTRY[dataset_id]
    
    if not entry.enabled:
        return False, f"Dataset '{dataset_id}' is disabled"
    
    if entry.source == 'huggingface':
        if not HF_DATASETS_AVAILABLE:
            return False, "HuggingFace datasets library not available"
    elif entry.source == 'torchvision':
        if not TORCHVISION_AVAILABLE:
            return False, "torchvision library not available"
    elif entry.source == 'custom':
        return False, "Custom loader not implemented"
    
    return True, "Available"


# ============================================================================
# Dry-Run Data Generation
# ============================================================================

def generate_synthetic_batch(
    batch_size: int = 8,
    num_classes: int = 1000,
    image_size: int = 224
) -> Tuple[Any, Any]:
    """Generate synthetic data batch for smoke testing.
    
    Args:
        batch_size: Batch size
        num_classes: Number of classes
        image_size: Image size
        
    Returns:
        (images, labels) tuple
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for synthetic data")
    
    import torch
    
    images = torch.randn(batch_size, 3, image_size, image_size)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    return images, labels


# ============================================================================
# Registry Export
# ============================================================================

def export_registry(output_path: str):
    """Export dataset registry to JSON file.
    
    Args:
        output_path: Output file path
    """
    registry_data = {
        dataset_id: entry.to_dict()
        for dataset_id, entry in DATASET_REGISTRY.items()
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(registry_data, f, indent=2)


# ============================================================================
# Module Interface
# ============================================================================

__all__ = [
    'DatasetRegistryEntry',
    'DATASET_REGISTRY',
    'DatasetLoader',
    'get_dataloader',
    'list_available_datasets',
    'get_dataset_info',
    'check_dataset_availability',
    'generate_synthetic_batch',
    'export_registry',
]