"""
Environment and dataset registry for Robust CLIP reproduction.

Exposes paper-derived environment/task registry entries and dataset/benchmark
registry entries with ids, aliases, setup metadata, and factory/config hooks.

Paper evidence contract:
- CLIP, TeCoA, FARE method environments
- LLaVA, POPE, SQA-I benchmark environments
- ImageNet, COCO, Flickr30k, VQAv2, TextVQA datasets
- Fine-grained classification datasets (Caltech101, Stanford Cars, etc.)
- Robustness datasets (ImageNet-R, ImageNet-Sketch)

Binding addendum clarification:
- LLaVA model from https://github.com/haotian-liu/LLaVA/tree/main
- POPE and SQA-I benchmarks from the LLaVA repository
- LLaVA-1.5 7B uses OpenAI CLIP ViT-L/14@224 (not @336)
- Modified to work with OpenCLIP instead of Huggingface
- ImageNet downloaded via HuggingFace with trust_remote_code=True
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field, asdict
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# Environment/Task Registry (Paper Evidence Contract)
# ============================================================================

@dataclass
class EnvironmentConfig:
    """Configuration for an environment/task."""
    env_id: str
    name: str
    aliases: List[str] = field(default_factory=list)
    task_type: str = "classification"  # classification, vqa, retrieval, lvlm
    model_type: str = "clip"  # clip, tecoa, fare, llava
    description: str = ""
    setup_metadata: Dict[str, Any] = field(default_factory=dict)
    config_hooks: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class DatasetConfig:
    """Configuration for a dataset/benchmark."""
    dataset_id: str
    name: str
    aliases: List[str] = field(default_factory=list)
    dataset_type: str = "classification"  # classification, vqa, retrieval, caption
    split: str = "test"
    num_classes: Optional[int] = None
    description: str = ""
    setup_metadata: Dict[str, Any] = field(default_factory=dict)
    loader_hooks: Dict[str, Any] = field(default_factory=dict)
    hf_dataset_name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# ============================================================================
# Environment Registry: CLIP, TeCoA, FARE, LLaVA, POPE, SQA-I
# ============================================================================

ENVIRONMENT_REGISTRY: Dict[str, EnvironmentConfig] = {
    # CLIP baseline environment
    "clip": EnvironmentConfig(
        env_id="clip",
        name="CLIP",
        aliases=["clean_clip", "baseline_clip", "openai_clip"],
        task_type="classification",
        model_type="clip",
        description="Standard CLIP ViT-L/14 vision encoder for zero-shot classification",
        setup_metadata={
            "architecture": "ViT-L/14",
            "pretrained": "openai",
            "input_resolution": 224,
            "patch_size": 14,
            "adversarial_trained": False,
        },
        config_hooks={
            "model_name": "ViT-L-14",
            "pretrained": "openai",
            "cache_dir": None,
        },
    ),
    
    # TeCoA adversarial baseline environment
    "tecoa": EnvironmentConfig(
        env_id="tecoa",
        name="TeCoA",
        aliases=["text_guided_contrastive", "contrastive_adversarial"],
        task_type="classification",
        model_type="tecoa",
        description="Text-guided Contrastive Adversarial training baseline",
        setup_metadata={
            "architecture": "ViT-L/14",
            "pretrained": "openai",
            "input_resolution": 224,
            "adversarial_trained": True,
            "method": "text_guided_contrastive",
            "epsilon_range": [2/255, 4/255, 8/255, 16/255],
        },
        config_hooks={
            "model_name": "ViT-L-14",
            "pretrained": "openai",
            "adversarial_training": True,
            "text_guidance": True,
            "contrastive_loss": True,
        },
    ),
    
    # FARE (ours) environment
    "fare": EnvironmentConfig(
        env_id="fare",
        name="FARE",
        aliases=["ours", "feature_alignment", "robust_embedding"],
        task_type="classification",
        model_type="fare",
        description="Feature-Alignment Robust Embedding (ours) - unsupervised adversarial fine-tuning",
        setup_metadata={
            "architecture": "ViT-L/14",
            "pretrained": "openai",
            "input_resolution": 224,
            "adversarial_trained": True,
            "method": "feature_alignment",
            "epsilon_range": [2/255, 4/255, 8/255, 16/255],
            "class_token_only": True,
            "embedding_preservation_weight": "lambda",
        },
        config_hooks={
            "model_name": "ViT-L-14",
            "pretrained": "openai",
            "adversarial_training": True,
            "feature_alignment": True,
            "class_token_alignment": True,
            "lambda_weight": 1.0,
        },
    ),
    
    # LLaVA LVLM environment
    "llava": EnvironmentConfig(
        env_id="llava",
        name="LLaVA",
        aliases=["llava_1.5", "llava_7b", "lvlm"],
        task_type="lvlm",
        model_type="llava",
        description="LLaVA-1.5 7B with OpenAI CLIP ViT-L/14@224 vision encoder",
        setup_metadata={
            "architecture": "LLaVA-1.5-7B",
            "vision_encoder": "ViT-L/14@224",
            "language_model": "Vicuna-7B",
            "input_resolution": 224,
            "repository": "https://github.com/haotian-liu/LLaVA/tree/main",
            "clip_implementation": "openclip",
        },
        config_hooks={
            "model_name": "llava-v1.5-7b",
            "vision_tower": "openai/clip-vit-large-patch14",
            "pretrained": "liuhaotian/llava-v1.5-7b",
            "load_in_8bit": False,
        },
    ),
    
    # POPE hallucination benchmark environment
    "pope": EnvironmentConfig(
        env_id="pope",
        name="POPE",
        aliases=["pope_benchmark", "hallucination_eval"],
        task_type="lvlm",
        model_type="llava",
        description="Polling-based Object Probing Evaluation for hallucination assessment",
        setup_metadata={
            "benchmark_type": "hallucination",
            "metrics": ["accuracy", "precision", "recall", "f1"],
            "repository": "https://github.com/haotian-liu/LLaVA/tree/main",
        },
        config_hooks={
            "benchmark": "pope",
            "pope_type": "random",
            "num_samples": 500,
        },
    ),
    
    # SQA-I benchmark environment
    "sqa_i": EnvironmentConfig(
        env_id="sqa_i",
        name="SQA-I",
        aliases=["sqa", "scienceqa", "science_qa"],
        task_type="lvlm",
        model_type="llava",
        description="ScienceQA-IMG benchmark for scientific reasoning evaluation",
        setup_metadata={
            "benchmark_type": "reasoning",
            "metrics": ["accuracy"],
            "repository": "https://github.com/haotian-liu/LLaVA/tree/main",
        },
        config_hooks={
            "benchmark": "sqa_i",
            "split": "test",
        },
    ),
    
    # Quantitative robustness evaluation environment
    "robustness_eval": EnvironmentConfig(
        env_id="robustness_eval",
        name="Quantitative Robustness Evaluation",
        aliases=["robust_eval", "adversarial_eval"],
        task_type="classification",
        model_type="clip",
        description="Quantitative robustness evaluation across epsilon values",
        setup_metadata={
            "epsilon_values": [0, 2/255, 4/255, 8/255, 16/255],
            "attack_method": "pgd",
            "metrics": ["clean_accuracy", "robust_accuracy"],
        },
        config_hooks={
            "evaluation_type": "robustness",
            "attack": "pgd",
            "num_steps": 50,
        },
    ),
    
    # VQA environment
    "vqa": EnvironmentConfig(
        env_id="vqa",
        name="VQA",
        aliases=["vqav2", "visual_question_answering"],
        task_type="vqa",
        model_type="llava",
        description="Visual Question Answering evaluation",
        setup_metadata={
            "benchmark_type": "vqa",
            "metrics": ["vqa_accuracy"],
        },
        config_hooks={
            "benchmark": "vqa",
            "split": "val",
        },
    ),
}


# ============================================================================
# Dataset Registry: ImageNet, COCO, Flickr30k, VQAv2, TextVQA, POPE, SQA-I
# ============================================================================

DATASET_REGISTRY: Dict[str, DatasetConfig] = {
    # ImageNet
    "imagenet": DatasetConfig(
        dataset_id="imagenet",
        name="ImageNet",
        aliases=["imagenet_1k", "ilsvrc2012", "imagenet1k"],
        dataset_type="classification",
        split="validation",
        num_classes=1000,
        description="ImageNet ILSVRC2012 validation set for zero-shot classification",
        setup_metadata={
            "resolution": 224,
            "num_samples": 50000,
        },
        loader_hooks={
            "dataset_name": "imagenet-1k",
            "trust_remote_code": True,
        },
        hf_dataset_name="imagenet-1k",
    ),
    
    # ImageNet-R (Robustness)
    "imagenet_r": DatasetConfig(
        dataset_id="imagenet_r",
        name="ImageNet-R",
        aliases=["imagenet_rendition", "imagenetr"],
        dataset_type="classification",
        split="test",
        num_classes=200,
        description="ImageNet-R for distribution shift robustness evaluation",
        setup_metadata={
            "resolution": 224,
            "num_samples": 30000,
        },
        loader_hooks={
            "dataset_name": "imagenet_r",
        },
    ),
    
    # ImageNet-Sketch
    "imagenet_sketch": DatasetConfig(
        dataset_id="imagenet_sketch",
        name="ImageNet-Sketch",
        aliases=["imagenet_sketch", "sketch"],
        dataset_type="classification",
        split="test",
        num_classes=1000,
        description="ImageNet-Sketch for sketch domain robustness evaluation",
        setup_metadata={
            "resolution": 224,
            "num_samples": 50889,
        },
        loader_hooks={
            "dataset_name": "imagenet_sketch",
        },
    ),
    
    # CIFAR-10
    "cifar10": DatasetConfig(
        dataset_id="cifar10",
        name="CIFAR-10",
        aliases=["cifar"],
        dataset_type="classification",
        split="test",
        num_classes=10,
        description="CIFAR-10 test set",
        setup_metadata={
            "resolution": 32,
            "num_samples": 10000,
        },
        loader_hooks={
            "dataset_name": "cifar10",
        },
        hf_dataset_name="cifar10",
    ),
    
    # CIFAR-100
    "cifar100": DatasetConfig(
        dataset_id="cifar100",
        name="CIFAR-100",
        aliases=["cifar_100"],
        dataset_type="classification",
        split="test",
        num_classes=100,
        description="CIFAR-100 test set",
        setup_metadata={
            "resolution": 32,
            "num_samples": 10000,
        },
        loader_hooks={
            "dataset_name": "cifar100",
        },
        hf_dataset_name="cifar100",
    ),
    
    # STL-10
    "stl10": DatasetConfig(
        dataset_id="stl10",
        name="STL-10",
        aliases=["stl"],
        dataset_type="classification",
        split="test",
        num_classes=10,
        description="STL-10 test set",
        setup_metadata={
            "resolution": 96,
            "num_samples": 8000,
        },
        loader_hooks={
            "dataset_name": "stl10",
        },
    ),
    
    # Caltech101
    "caltech101": DatasetConfig(
        dataset_id="caltech101",
        name="Caltech101",
        aliases=["caltech_101"],
        dataset_type="classification",
        split="test",
        num_classes=101,
        description="Caltech-101 object recognition dataset",
        setup_metadata={
            "resolution": 224,
        },
        loader_hooks={
            "dataset_name": "caltech101",
        },
    ),
    
    # Stanford Cars
    "stanford_cars": DatasetConfig(
        dataset_id="stanford_cars",
        name="Stanford Cars",
        aliases=["cars", "stanford_car"],
        dataset_type="classification",
        split="test",
        num_classes=196,
        description="Stanford Cars fine-grained classification",
        setup_metadata={
            "resolution": 224,
        },
        loader_hooks={
            "dataset_name": "stanford_cars",
        },
    ),
    
    # FGVC Aircraft
    "fgvc_aircraft": DatasetConfig(
        dataset_id="fgvc_aircraft",
        name="FGVC Aircraft",
        aliases=["aircraft", "fgvc"],
        dataset_type="classification",
        split="test",
        num_classes=100,
        description="Fine-Grained Visual Classification of Aircraft",
        setup_metadata={
            "resolution": 224,
        },
        loader_hooks={
            "dataset_name": "fgvc_aircraft",
        },
    ),
    
    # Flowers102
    "flowers": DatasetConfig(
        dataset_id="flowers",
        name="Flowers102",
        aliases=["flowers102", "oxford_flowers"],
        dataset_type="classification",
        split="test",
        num_classes=102,
        description="Oxford Flowers-102 dataset",
        setup_metadata={
            "resolution": 224,
        },
        loader_hooks={
            "dataset_name": "flowers102",
        },
    ),
    
    # Oxford-IIIT Pets
    "oxford_pets": DatasetConfig(
        dataset_id="oxford_pets",
        name="Oxford-IIIT Pets",
        aliases=["pets", "oxford_pet"],
        dataset_type="classification",
        split="test",
        num_classes=37,
        description="Oxford-IIIT Pet dataset",
        setup_metadata={
            "resolution": 224,
        },
        loader_hooks={
            "dataset_name": "oxford_iiit_pet",
        },
    ),
    
    # PatchCamelyon
    "pcam": DatasetConfig(
        dataset_id="pcam",
        name="PatchCamelyon",
        aliases=["patch_camelyon"],
        dataset_type="classification",
        split="test",
        num_classes=2,
        description="PatchCamelyon medical imaging dataset",
        setup_metadata={
            "resolution": 96,
        },
        loader_hooks={
            "dataset_name": "patch_camelyon",
        },
    ),
    
    # COCO
    "coco": DatasetConfig(
        dataset_id="coco",
        name="COCO",
        aliases=["mscoco", "coco2017"],
        dataset_type="retrieval",
        split="val",
        description="COCO 2017 validation set for image-text retrieval",
        setup_metadata={
            "resolution": 224,
            "num_samples": 5000,
        },
        loader_hooks={
            "dataset_name": "coco",
            "year": "2017",
        },
    ),
    
    # Flickr30k
    "flickr30k": DatasetConfig(
        dataset_id="flickr30k",
        name="Flickr30k",
        aliases=["flickr"],
        dataset_type="retrieval",
        split="test",
        description="Flickr30k image-text retrieval dataset",
        setup_metadata={
            "resolution": 224,
            "num_samples": 1000,
        },
        loader_hooks={
            "dataset_name": "flickr30k",
        },
    ),
    
    # LAION (training data)
    "laion": DatasetConfig(
        dataset_id="laion",
        name="LAION",
        aliases=["laion400m", "laion_subset"],
        dataset_type="caption",
        split="train",
        description="LAION subset for unsupervised adversarial fine-tuning",
        setup_metadata={
            "resolution": 224,
        },
        loader_hooks={
            "dataset_name": "laion/laion400m",
            "streaming": True,
        },
    ),
    
    # VQAv2
    "vqav2": DatasetConfig(
        dataset_id="vqav2",
        name="VQAv2",
        aliases=["vqa", "vqa_v2"],
        dataset_type="vqa",
        split="val",
        description="Visual Question Answering v2.0 dataset",
        setup_metadata={
            "resolution": 224,
        },
        loader_hooks={
            "dataset_name": "vqav2",
        },
    ),
    
    # TextVQA
    "textvqa": DatasetConfig(
        dataset_id="textvqa",
        name="TextVQA",
        aliases=["text_vqa"],
        dataset_type="vqa",
        split="val",
        description="TextVQA: text reading and reasoning in images",
        setup_metadata={
            "resolution": 224,
        },
        loader_hooks={
            "dataset_name": "textvqa",
        },
    ),
    
    # POPE (benchmark)
    "pope": DatasetConfig(
        dataset_id="pope",
        name="POPE",
        aliases=["pope_benchmark"],
        dataset_type="lvlm",
        split="test",
        description="POPE benchmark for object hallucination evaluation",
        setup_metadata={
            "benchmark_type": "hallucination",
        },
        loader_hooks={
            "dataset_name": "pope",
        },
    ),
    
    # SQA-I (benchmark)
    "sqa_i": DatasetConfig(
        dataset_id="sqa_i",
        name="SQA-I",
        aliases=["sqa", "scienceqa"],
        dataset_type="lvlm",
        split="test",
        description="ScienceQA-IMG for scientific reasoning",
        setup_metadata={
            "benchmark_type": "reasoning",
        },
        loader_hooks={
            "dataset_name": "sqa_i",
        },
    ),
    
    # CLIP Benchmark alias
    "clip_benchmark": DatasetConfig(
        dataset_id="clip_benchmark",
        name="CLIP Benchmark",
        aliases=["clip_eval"],
        dataset_type="classification",
        split="test",
        description="CLIP Benchmark evaluation suite",
        setup_metadata={},
        loader_hooks={},
    ),
}


# ============================================================================
# Registry Access Functions
# ============================================================================

def get_environment(env_id: str) -> Optional[EnvironmentConfig]:
    """Get environment configuration by ID or alias."""
    # Direct lookup
    if env_id in ENVIRONMENT_REGISTRY:
        return ENVIRONMENT_REGISTRY[env_id]
    
    # Alias lookup
    for env_config in ENVIRONMENT_REGISTRY.values():
        if env_id in env_config.aliases:
            return env_config
    
    return None


def get_dataset(dataset_id: str) -> Optional[DatasetConfig]:
    """Get dataset configuration by ID or alias."""
    # Direct lookup
    if dataset_id in DATASET_REGISTRY:
        return DATASET_REGISTRY[dataset_id]
    
    # Alias lookup
    for dataset_config in DATASET_REGISTRY.values():
        if dataset_id in dataset_config.aliases:
            return dataset_config
    
    return None


def list_environments() -> List[str]:
    """List all registered environment IDs."""
    return list(ENVIRONMENT_REGISTRY.keys())


def list_datasets() -> List[str]:
    """List all registered dataset IDs."""
    return list(DATASET_REGISTRY.keys())


# ============================================================================
# Dataset Loader Factory
# ============================================================================

def create_dataset_loader(dataset_id: str, **kwargs) -> Callable:
    """
    Create a dataset loader function for the given dataset ID.
    
    Returns a callable that loads the dataset when invoked.
    Uses lazy imports to avoid requiring heavy dependencies at module load.
    """
    dataset_config = get_dataset(dataset_id)
    if dataset_config is None:
        raise ValueError(f"Unknown dataset: {dataset_id}")
    
    def loader():
        """Lazy dataset loader with on-demand imports."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets package required for dataset loading. "
                "Install with: pip install datasets"
            )
        
        hf_name = dataset_config.hf_dataset_name or dataset_config.dataset_id
        loader_kwargs = {**dataset_config.loader_hooks, **kwargs}
        
        # Add trust_remote_code for ImageNet as per addendum
        if dataset_id in ["imagenet", "imagenet_1k"]:
            loader_kwargs["trust_remote_code"] = True
        
        try:
            dataset = load_dataset(hf_name, split=dataset_config.split, **loader_kwargs)
            return dataset
        except Exception as e:
            warnings.warn(f"Failed to load dataset {dataset_id}: {e}")
            return None
    
    return loader


# ============================================================================
# Environment Setup Factory
# ============================================================================

def setup_environment(env_id: str, **kwargs) -> Dict[str, Any]:
    """
    Set up environment configuration for the given environment ID.
    
    Returns a configuration dictionary that can be used to initialize
    models, tasks, or benchmarks.
    """
    env_config = get_environment(env_id)
    if env_config is None:
        raise ValueError(f"Unknown environment: {env_id}")
    
    # Merge default config hooks with user kwargs
    config = {**env_config.config_hooks, **kwargs}
    
    # Add metadata
    config["env_id"] = env_config.env_id
    config["task_type"] = env_config.task_type
    config["model_type"] = env_config.model_type
    
    return config


# ============================================================================
# Registry Export Functions
# ============================================================================

def export_registry_manifest(output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Export complete registry manifest with all environments and datasets.
    
    Used for documentation, smoke testing, and artifact generation.
    """
    manifest = {
        "environments": {
            env_id: env_config.to_dict()
            for env_id, env_config in ENVIRONMENT_REGISTRY.items()
        },
        "datasets": {
            dataset_id: dataset_config.to_dict()
            for dataset_id, dataset_config in DATASET_REGISTRY.items()
        },
        "metadata": {
            "num_environments": len(ENVIRONMENT_REGISTRY),
            "num_datasets": len(DATASET_REGISTRY),
            "environment_types": list(set(
                env.task_type for env in ENVIRONMENT_REGISTRY.values()
            )),
            "dataset_types": list(set(
                ds.dataset_type for ds in DATASET_REGISTRY.values()
            )),
        },
    }
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(manifest, f, indent=2)
    
    return manifest


# ============================================================================
# Validation and Smoke Test
# ============================================================================

def validate_registries() -> Tuple[bool, List[str]]:
    """
    Validate that all registries are properly configured.
    
    Returns (is_valid, error_messages).
    """
    errors = []
    
    # Check for duplicate aliases across environments
    all_env_aliases = set()
    for env_config in ENVIRONMENT_REGISTRY.values():
        for alias in env_config.aliases:
            if alias in all_env_aliases:
                errors.append(f"Duplicate environment alias: {alias}")
            all_env_aliases.add(alias)
    
    # Check for duplicate aliases across datasets
    all_dataset_aliases = set()
    for dataset_config in DATASET_REGISTRY.values():
        for alias in dataset_config.aliases:
            if alias in all_dataset_aliases:
                errors.append(f"Duplicate dataset alias: {alias}")
            all_dataset_aliases.add(alias)
    
    # Verify paper contract coverage
    required_envs = ["clip", "tecoa", "fare", "llava", "pope", "sqa_i"]
    for env_id in required_envs:
        if get_environment(env_id) is None:
            errors.append(f"Missing required environment: {env_id}")
    
    required_datasets = ["imagenet", "coco", "flickr30k", "vqav2", "textvqa", "pope", "sqa_i"]
    for dataset_id in required_datasets:
        if get_dataset(dataset_id) is None:
            errors.append(f"Missing required dataset: {dataset_id}")
    
    # Paper evidence contract aliases
    paper_env_aliases = ["cifar", "imagenet", "coco", "laion", "clip_benchmark", "flickr30k", "stl10"]
    for alias in paper_env_aliases:
        found = False
        for env_config in ENVIRONMENT_REGISTRY.values():
            if alias in env_config.aliases or alias == env_config.env_id:
                found = True
                break
        for dataset_config in DATASET_REGISTRY.values():
            if alias in dataset_config.aliases or alias == dataset_config.dataset_id:
                found = True
                break
        if not found:
            errors.append(f"Missing paper evidence alias: {alias}")
    
    return len(errors) == 0, errors


if __name__ == "__main__":
    # Smoke test: validate registries and export manifest
    is_valid, errors = validate_registries()
    
    if is_valid:
        print("✓ Registry validation passed")
        print(f"  - {len(ENVIRONMENT_REGISTRY)} environments registered")
        print(f"  - {len(DATASET_REGISTRY)} datasets registered")
        
        # Export manifest
        manifest = export_registry_manifest("registry_manifest.json")
        print(f"✓ Exported registry manifest")
    else:
        print("✗ Registry validation failed:")
        for error in errors:
            print(f"  - {error}")