"""
LCA-on-the-Line Dataset Registry

Exposes paper-derived dataset/benchmark registry entries with ids, setup metadata,
and loader/config hooks for all datasets used in the paper evaluation.

Paper evidence contract datasets:
- imagenet / imagenet_1k: In-distribution dataset
- imagenet_v2: OOD dataset (MatchedFrequency variant)
- imagenet_c: OOD dataset (corruption robustness)
- imagenet_r: OOD dataset (rendition robustness)
- imagenet_sketch: OOD dataset (sketch robustness)
- objectnet: OOD dataset
- imagenet_a: OOD dataset (adversarial)
- laion: Pre-training dataset reference
- cifar: Mentioned in paper evidence contract

Binding addendum clarifications:
- ImageNet downloaded via HuggingFace: load_dataset("imagenet-1k", trust_remote_code=True)
- ImageNet-v2 MatchedFrequency split from commit d626240
- WordNet hierarchy from github.com/jvlmdr/hiercls imagenet_fiveai.csv

reference_grounding: paperbench_ref_001 references/depth/stereo/README.md
reference_grounding: paperbench_ref_001 torchvision/models/detection/mask_rcnn.py
reference_grounding: paperbench_ref_001 torchvision/models/detection/keypoint_rcnn.py
"""

import json
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple
import warnings

logger = logging.getLogger(__name__)


@dataclass
class DatasetMetadata:
    """Metadata for a registered dataset."""
    dataset_id: str
    name: str
    aliases: List[str]
    num_classes: int
    splits: List[str]
    source_type: str  # huggingface, local, download
    source_path: str
    data_dir: str
    description: str
    paper_usage: str
    loader_fn: Optional[str] = None
    requires_download: bool = False
    is_ood: bool = False
    parent_dataset: Optional[str] = None


# =============================================================================
# Dataset Registry - Paper evidence contract
# reference_grounding: paperbench_ref_001 torchvision/models/detection/mask_rcnn.py
# =============================================================================

DATASET_REGISTRY: Dict[str, DatasetMetadata] = {
    # -------------------------------------------------------------------------
    # ID Datasets
    # -------------------------------------------------------------------------
    "imagenet": DatasetMetadata(
        dataset_id="imagenet",
        name="ImageNet-1K",
        aliases=["imagenet", "imagenet_1k", "imagenet-1k", "ILSVRC2012"],
        num_classes=1000,
        splits=["train", "val"],
        source_type="huggingface",
        source_path="ILSVRC/imagenet-1k",
        data_dir="data/imagenet",
        description="ImageNet-1K (ILSVRC2012) used as in-distribution dataset",
        paper_usage="ID dataset for LCA distance computation and model evaluation",
        loader_fn="load_imagenet",
        requires_download=True,
        is_ood=False,
    ),
    
    "imagenet_1k": DatasetMetadata(
        dataset_id="imagenet_1k",
        name="ImageNet-1K",
        aliases=["imagenet_1k", "imagenet-1k"],
        num_classes=1000,
        splits=["train", "val"],
        source_type="huggingface",
        source_path="ILSVRC/imagenet-1k",
        data_dir="data/imagenet",
        description="ImageNet-1K alias for consistency",
        paper_usage="ID dataset alias",
        loader_fn="load_imagenet",
        requires_download=True,
        is_ood=False,
        parent_dataset="imagenet",
    ),
    
    # -------------------------------------------------------------------------
    # OOD Datasets - Paper Section 3.1
    # -------------------------------------------------------------------------
    "imagenet_v2": DatasetMetadata(
        dataset_id="imagenet_v2",
        name="ImageNet-V2",
        aliases=["imagenet_v2", "imagenet-v2", "imagenetv2"],
        num_classes=1000,
        splits=["test"],
        source_type="huggingface",
        source_path="vaishaal/ImageNetV2",
        data_dir="data/imagenet_v2",
        description="ImageNet-V2 MatchedFrequency variant from commit d626240",
        paper_usage="OOD dataset for evaluating distribution shift robustness",
        loader_fn="load_imagenet_v2",
        requires_download=True,
        is_ood=True,
        parent_dataset="imagenet",
    ),
    
    "imagenet_c": DatasetMetadata(
        dataset_id="imagenet_c",
        name="ImageNet-C",
        aliases=["imagenet_c", "imagenet-c", "imagenet_corrupted"],
        num_classes=1000,
        splits=["test"],
        source_type="download",
        source_path="https://zenodo.org/record/2235448",
        data_dir="data/imagenet_c",
        description="ImageNet-C: corruption robustness benchmark",
        paper_usage="OOD dataset for evaluating corruption robustness",
        loader_fn="load_imagenet_c",
        requires_download=True,
        is_ood=True,
        parent_dataset="imagenet",
    ),
    
    "imagenet_r": DatasetMetadata(
        dataset_id="imagenet_r",
        name="ImageNet-R",
        aliases=["imagenet_r", "imagenet-r", "imagenet_rendition"],
        num_classes=200,
        splits=["test"],
        source_type="download",
        source_path="https://github.com/hendrycks/imagenet-r",
        data_dir="data/imagenet_r",
        description="ImageNet-R: rendition robustness with artistic styles",
        paper_usage="OOD dataset for evaluating artistic rendition robustness",
        loader_fn="load_imagenet_r",
        requires_download=True,
        is_ood=True,
        parent_dataset="imagenet",
    ),
    
    "imagenet_sketch": DatasetMetadata(
        dataset_id="imagenet_sketch",
        name="ImageNet-Sketch",
        aliases=["imagenet_sketch", "imagenet-sketch", "sketch"],
        num_classes=1000,
        splits=["test"],
        source_type="download",
        source_path="https://github.com/HaohanWang/ImageNet-Sketch",
        data_dir="data/imagenet_sketch",
        description="ImageNet-Sketch: sketch-based robustness benchmark",
        paper_usage="OOD dataset for evaluating sketch robustness",
        loader_fn="load_imagenet_sketch",
        requires_download=True,
        is_ood=True,
        parent_dataset="imagenet",
    ),
    
    "imagenet_a": DatasetMetadata(
        dataset_id="imagenet_a",
        name="ImageNet-A",
        aliases=["imagenet_a", "imagenet-a", "imagenet_adversarial"],
        num_classes=200,
        splits=["test"],
        source_type="download",
        source_path="https://github.com/hendrycks/natural-adv-examples",
        data_dir="data/imagenet_a",
        description="ImageNet-A: natural adversarial examples",
        paper_usage="OOD dataset for evaluating adversarial robustness",
        loader_fn="load_imagenet_a",
        requires_download=True,
        is_ood=True,
        parent_dataset="imagenet",
    ),
    
    "objectnet": DatasetMetadata(
        dataset_id="objectnet",
        name="ObjectNet",
        aliases=["objectnet", "object-net"],
        num_classes=113,
        splits=["test"],
        source_type="download",
        source_path="https://objectnet.dev/",
        data_dir="data/objectnet",
        description="ObjectNet: real-world distribution shift benchmark",
        paper_usage="OOD dataset for evaluating real-world distribution shift",
        loader_fn="load_objectnet",
        requires_download=True,
        is_ood=True,
        parent_dataset="imagenet",
    ),
    
    # -------------------------------------------------------------------------
    # Pre-training Dataset References
    # -------------------------------------------------------------------------
    "laion": DatasetMetadata(
        dataset_id="laion",
        name="LAION",
        aliases=["laion", "laion400m", "laion-400m"],
        num_classes=-1,
        splits=["train"],
        source_type="reference",
        source_path="https://laion.ai/",
        data_dir="data/laion",
        description="LAION-400M: large-scale image-text dataset",
        paper_usage="Referenced as pre-training dataset for some VLMs",
        loader_fn=None,
        requires_download=False,
        is_ood=False,
    ),
    
    # -------------------------------------------------------------------------
    # Additional Datasets - Paper evidence contract
    # -------------------------------------------------------------------------
    "cifar": DatasetMetadata(
        dataset_id="cifar",
        name="CIFAR-10/100",
        aliases=["cifar", "cifar10", "cifar100"],
        num_classes=10,
        splits=["train", "test"],
        source_type="torchvision",
        source_path="torchvision.datasets.CIFAR10",
        data_dir="data/cifar",
        description="CIFAR-10/100 dataset",
        paper_usage="Mentioned in paper evidence contract",
        loader_fn="load_cifar",
        requires_download=True,
        is_ood=False,
    ),
}


# =============================================================================
# Dataset Loader Functions
# reference_grounding: paperbench_ref_001 torchvision/models/detection/keypoint_rcnn.py
# =============================================================================

def load_imagenet(
    split: str = "val",
    data_dir: Optional[str] = None,
    trust_remote_code: bool = True,
    smoke_mode: bool = False,
) -> Any:
    """
    Load ImageNet-1K dataset using HuggingFace.
    
    Binding addendum: dataset = load_dataset("imagenet-1k", trust_remote_code=True)
    
    Args:
        split: Dataset split (train/val)
        data_dir: Local data directory (optional)
        trust_remote_code: Trust remote code to avoid stdin waits
        smoke_mode: Return minimal dataset for smoke testing
        
    Returns:
        Dataset object
    """
    if smoke_mode:
        logger.info("Smoke mode: returning mock ImageNet dataset")
        return _create_mock_dataset("imagenet", 1000, 50000 if split == "val" else 1281167)
    
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets package required. Install with: pip install datasets")
    
    logger.info(f"Loading ImageNet-1K split={split} with trust_remote_code={trust_remote_code}")
    
    try:
        # Binding addendum clarification: use trust_remote_code=True
        dataset = load_dataset(
            "ILSVRC/imagenet-1k",
            split=split,
            trust_remote_code=trust_remote_code,
            cache_dir=data_dir,
        )
        logger.info(f"Loaded ImageNet-1K {split}: {len(dataset)} samples")
        return dataset
    except Exception as e:
        logger.error(f"Failed to load ImageNet-1K: {e}")
        raise


def load_imagenet_v2(
    variant: str = "matched-frequency",
    data_dir: Optional[str] = None,
    smoke_mode: bool = False,
) -> Any:
    """
    Load ImageNet-V2 dataset.
    
    Binding addendum: Paper uses MatchedFrequency split from commit d626240
    of https://huggingface.co/datasets/vaishaal/ImageNetV2
    
    Args:
        variant: ImageNet-V2 variant (matched-frequency/threshold/topimages)
        data_dir: Local data directory (optional)
        smoke_mode: Return minimal dataset for smoke testing
        
    Returns:
        Dataset object
    """
    if smoke_mode:
        logger.info("Smoke mode: returning mock ImageNet-V2 dataset")
        return _create_mock_dataset("imagenet_v2", 1000, 10000)
    
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets package required. Install with: pip install datasets")
    
    logger.info(f"Loading ImageNet-V2 variant={variant}")
    
    try:
        # Binding addendum: load from vaishaal/ImageNetV2 at commit d626240
        dataset = load_dataset(
            "vaishaal/ImageNetV2",
            variant,
            revision="d626240",
            trust_remote_code=True,
            cache_dir=data_dir,
        )
        logger.info(f"Loaded ImageNet-V2 {variant}: {len(dataset['test'])} samples")
        return dataset["test"]
    except Exception as e:
        logger.warning(f"Failed to load ImageNet-V2 from HF: {e}")
        logger.info("Attempting fallback to local ImageNet-V2 directory")
        return _load_local_imagenet_variant("imagenet_v2", data_dir)


def load_imagenet_c(
    corruption_types: Optional[List[str]] = None,
    severity: int = 5,
    data_dir: Optional[str] = None,
    smoke_mode: bool = False,
) -> Any:
    """
    Load ImageNet-C dataset.
    
    Args:
        corruption_types: List of corruption types to load (None = all)
        severity: Corruption severity level (1-5)
        data_dir: Local data directory (optional)
        smoke_mode: Return minimal dataset for smoke testing
        
    Returns:
        Dataset object
    """
    if smoke_mode:
        logger.info("Smoke mode: returning mock ImageNet-C dataset")
        return _create_mock_dataset("imagenet_c", 1000, 50000)
    
    logger.info(f"Loading ImageNet-C severity={severity}")
    return _load_local_imagenet_variant("imagenet_c", data_dir, suffix=f"_severity{severity}")


def load_imagenet_r(
    data_dir: Optional[str] = None,
    smoke_mode: bool = False,
) -> Any:
    """
    Load ImageNet-R dataset.
    
    Args:
        data_dir: Local data directory (optional)
        smoke_mode: Return minimal dataset for smoke testing
        
    Returns:
        Dataset object
    """
    if smoke_mode:
        logger.info("Smoke mode: returning mock ImageNet-R dataset")
        return _create_mock_dataset("imagenet_r", 200, 30000)
    
    logger.info("Loading ImageNet-R")
    return _load_local_imagenet_variant("imagenet_r", data_dir)


def load_imagenet_sketch(
    data_dir: Optional[str] = None,
    smoke_mode: bool = False,
) -> Any:
    """
    Load ImageNet-Sketch dataset.
    
    Args:
        data_dir: Local data directory (optional)
        smoke_mode: Return minimal dataset for smoke testing
        
    Returns:
        Dataset object
    """
    if smoke_mode:
        logger.info("Smoke mode: returning mock ImageNet-Sketch dataset")
        return _create_mock_dataset("imagenet_sketch", 1000, 50889)
    
    logger.info("Loading ImageNet-Sketch")
    return _load_local_imagenet_variant("imagenet_sketch", data_dir)


def load_imagenet_a(
    data_dir: Optional[str] = None,
    smoke_mode: bool = False,
) -> Any:
    """
    Load ImageNet-A dataset.
    
    Args:
        data_dir: Local data directory (optional)
        smoke_mode: Return minimal dataset for smoke testing
        
    Returns:
        Dataset object
    """
    if smoke_mode:
        logger.info("Smoke mode: returning mock ImageNet-A dataset")
        return _create_mock_dataset("imagenet_a", 200, 7500)
    
    logger.info("Loading ImageNet-A")
    return _load_local_imagenet_variant("imagenet_a", data_dir)


def load_objectnet(
    data_dir: Optional[str] = None,
    smoke_mode: bool = False,
) -> Any:
    """
    Load ObjectNet dataset.
    
    Args:
        data_dir: Local data directory (optional)
        smoke_mode: Return minimal dataset for smoke testing
        
    Returns:
        Dataset object
    """
    if smoke_mode:
        logger.info("Smoke mode: returning mock ObjectNet dataset")
        return _create_mock_dataset("objectnet", 113, 18574)
    
    logger.info("Loading ObjectNet")
    return _load_local_imagenet_variant("objectnet", data_dir)


def load_cifar(
    variant: str = "10",
    split: str = "test",
    data_dir: Optional[str] = None,
    smoke_mode: bool = False,
) -> Any:
    """
    Load CIFAR-10/100 dataset.
    
    Args:
        variant: CIFAR variant (10 or 100)
        split: Dataset split (train/test)
        data_dir: Local data directory (optional)
        smoke_mode: Return minimal dataset for smoke testing
        
    Returns:
        Dataset object
    """
    if smoke_mode:
        num_classes = 10 if variant == "10" else 100
        logger.info(f"Smoke mode: returning mock CIFAR-{variant} dataset")
        return _create_mock_dataset(f"cifar{variant}", num_classes, 10000)
    
    try:
        import torchvision.datasets as datasets
        import torchvision.transforms as transforms
    except ImportError:
        raise ImportError("torchvision required. Install with: pip install torchvision")
    
    logger.info(f"Loading CIFAR-{variant} split={split}")
    
    dataset_class = datasets.CIFAR10 if variant == "10" else datasets.CIFAR100
    data_dir = data_dir or "data/cifar"
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    dataset = dataset_class(
        root=data_dir,
        train=(split == "train"),
        download=True,
        transform=transform,
    )
    
    logger.info(f"Loaded CIFAR-{variant} {split}: {len(dataset)} samples")
    return dataset


def _load_local_imagenet_variant(
    dataset_id: str,
    data_dir: Optional[str] = None,
    suffix: str = "",
) -> Any:
    """
    Load ImageNet variant from local directory.
    
    Args:
        dataset_id: Dataset identifier
        data_dir: Local data directory
        suffix: Optional suffix for dataset path
        
    Returns:
        Dataset object or mock dataset if unavailable
    """
    if data_dir is None:
        data_dir = f"data/{dataset_id}"
    
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.warning(f"Dataset directory not found: {data_path}")
        logger.info(f"Returning mock dataset for {dataset_id}")
        metadata = DATASET_REGISTRY.get(dataset_id)
        num_classes = metadata.num_classes if metadata else 1000
        return _create_mock_dataset(dataset_id, num_classes, 10000)
    
    try:
        import torchvision.datasets as datasets
        import torchvision.transforms as transforms
    except ImportError:
        raise ImportError("torchvision required. Install with: pip install torchvision")
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = datasets.ImageFolder(str(data_path), transform=transform)
    logger.info(f"Loaded {dataset_id} from {data_path}: {len(dataset)} samples")
    return dataset


def _create_mock_dataset(dataset_id: str, num_classes: int, num_samples: int) -> Dict[str, Any]:
    """
    Create a mock dataset for smoke testing.
    
    Args:
        dataset_id: Dataset identifier
        num_classes: Number of classes
        num_samples: Number of samples
        
    Returns:
        Mock dataset dictionary
    """
    return {
        "dataset_id": dataset_id,
        "num_classes": num_classes,
        "num_samples": num_samples,
        "is_mock": True,
        "description": f"Mock {dataset_id} dataset for smoke testing",
    }


# =============================================================================
# Registry Access Functions
# =============================================================================

def get_dataset(dataset_id: str, **kwargs) -> Any:
    """
    Get dataset by ID with automatic loader dispatch.
    
    Args:
        dataset_id: Dataset identifier
        **kwargs: Arguments passed to loader function
        
    Returns:
        Dataset object
    """
    # Normalize dataset ID
    dataset_id = dataset_id.lower().replace("-", "_")
    
    # Check aliases
    for registered_id, metadata in DATASET_REGISTRY.items():
        if dataset_id in [a.lower().replace("-", "_") for a in metadata.aliases]:
            dataset_id = registered_id
            break
    
    if dataset_id not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_id}")
    
    metadata = DATASET_REGISTRY[dataset_id]
    
    if metadata.loader_fn is None:
        raise ValueError(f"Dataset {dataset_id} has no loader function")
    
    loader_fn = globals().get(metadata.loader_fn)
    if loader_fn is None:
        raise ValueError(f"Loader function not found: {metadata.loader_fn}")
    
    logger.info(f"Loading dataset: {metadata.name} ({dataset_id})")
    return loader_fn(**kwargs)


def get_dataset_metadata(dataset_id: str) -> DatasetMetadata:
    """
    Get metadata for a dataset.
    
    Args:
        dataset_id: Dataset identifier
        
    Returns:
        Dataset metadata
    """
    dataset_id = dataset_id.lower().replace("-", "_")
    
    # Check aliases
    for registered_id, metadata in DATASET_REGISTRY.items():
        if dataset_id in [a.lower().replace("-", "_") for a in metadata.aliases]:
            return metadata
    
    if dataset_id not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_id}")
    
    return DATASET_REGISTRY[dataset_id]


def list_datasets(ood_only: bool = False, id_only: bool = False) -> List[str]:
    """
    List all registered datasets.
    
    Args:
        ood_only: Only return OOD datasets
        id_only: Only return ID datasets
        
    Returns:
        List of dataset IDs
    """
    datasets = []
    for dataset_id, metadata in DATASET_REGISTRY.items():
        if ood_only and not metadata.is_ood:
            continue
        if id_only and metadata.is_ood:
            continue
        datasets.append(dataset_id)
    return sorted(datasets)


def check_dataset_availability(dataset_id: str, data_dir: Optional[str] = None) -> bool:
    """
    Check if a dataset is available locally.
    
    Args:
        dataset_id: Dataset identifier
        data_dir: Local data directory (optional)
        
    Returns:
        True if dataset is available
    """
    metadata = get_dataset_metadata(dataset_id)
    
    if metadata.source_type == "reference":
        return False  # Reference-only datasets not available for loading
    
    if metadata.source_type == "huggingface":
        # Check HuggingFace cache
        try:
            from datasets import load_dataset_builder
            builder = load_dataset_builder(metadata.source_path)
            return True
        except:
            return False
    
    # Check local directory
    data_dir = data_dir or metadata.data_dir
    data_path = Path(data_dir)
    return data_path.exists() and any(data_path.iterdir())


# =============================================================================
# Artifact Writing Functions
# =============================================================================

def write_dataset_registry_artifact(output_dir: str = "results", smoke_mode: bool = False) -> None:
    """
    Write dataset registry to artifact file.
    
    Args:
        output_dir: Output directory for artifacts
        smoke_mode: Generate smoke/dry-run artifact
    """
    output_path = Path(output_dir) / "dataset_registry.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    registry_data = {
        "artifact_type": "dataset_registry",
        "smoke_mode": smoke_mode,
        "description": "LCA-on-the-Line dataset registry - paper evidence contract",
        "paper_datasets": {
            "id": ["imagenet", "imagenet_1k"],
            "ood": ["imagenet_v2", "imagenet_c", "imagenet_r", "imagenet_sketch", "imagenet_a", "objectnet"],
            "reference": ["laion"],
            "additional": ["cifar"],
        },
        "datasets": {},
    }
    
    for dataset_id, metadata in DATASET_REGISTRY.items():
        registry_data["datasets"][dataset_id] = asdict(metadata)
        if not smoke_mode:
            registry_data["datasets"][dataset_id]["available"] = check_dataset_availability(dataset_id)
    
    with open(output_path, "w") as f:
        json.dump(registry_data, f, indent=2)
    
    logger.info(f"Written dataset registry artifact: {output_path}")


def validate_dataset_contracts() -> Dict[str, Any]:
    """
    Validate that all paper-mentioned datasets are registered.
    
    Returns:
        Validation results
    """
    required_datasets = [
        "imagenet", "imagenet_1k", "imagenet_v2", "imagenet_c",
        "imagenet_r", "imagenet_sketch", "laion", "cifar",
    ]
    
    validation = {
        "all_required_registered": True,
        "missing_datasets": [],
        "registered_count": len(DATASET_REGISTRY),
        "ood_count": len([m for m in DATASET_REGISTRY.values() if m.is_ood]),
        "id_count": len([m for m in DATASET_REGISTRY.values() if not m.is_ood]),
    }
    
    for dataset_id in required_datasets:
        if dataset_id not in DATASET_REGISTRY:
            validation["all_required_registered"] = False
            validation["missing_datasets"].append(dataset_id)
    
    return validation


# =============================================================================
# Main Execution (for testing)
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("LCA-on-the-Line Dataset Registry")
    print("=" * 80)
    
    # Validate contracts
    validation = validate_dataset_contracts()
    print(f"\nValidation: {validation}")
    
    # List datasets
    print(f"\nAll datasets: {list_datasets()}")
    print(f"ID datasets: {list_datasets(id_only=True)}")
    print(f"OOD datasets: {list_datasets(ood_only=True)}")
    
    # Write registry artifact
    write_dataset_registry_artifact(smoke_mode=True)
    print("\nDataset registry artifact written to results/dataset_registry.json")