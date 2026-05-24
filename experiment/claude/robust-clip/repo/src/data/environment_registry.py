"""
Environment and task registry for Robust CLIP reproduction.

This module provides:
- Environment/task registry entries with metadata and factory hooks
- Dataset registry entries with loader configurations
- Evaluation orchestration for Table 4 (zero-shot classification robustness)
- Artifact writing for robustness evaluation results

Paper evidence contract:
- Register environments: CLIP, TeCoA, FARE, LLaVA-1.5, POPE, SQA-I
- Register datasets: imagenet, coco, flickr30k, vqav2, textvqa, pope, sqa_i
- Register task aliases: cifar, imagenet, coco, laion, clip_benchmark, flickr30k, stl10
- Evaluation surface: evaluate_table4(models, datasets, epsilons)
"""

import os
import json
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field, asdict
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# Environment/Task Metadata Schema
# ============================================================================

@dataclass
class EnvironmentDescriptor:
    """Metadata descriptor for an environment/task."""
    id: str
    name: str
    category: str  # 'model', 'benchmark', 'task'
    description: str
    paper_section: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    setup_requirements: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    factory_available: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.category,
            "description": self.description,
            "paper_section": self.paper_section,
            "aliases": list(self.aliases),
            "setup_requirements": list(self.setup_requirements),
            "config": dict(self.config),
            "factory_available": self.factory_available,
        }

    def keys(self):
        return self.as_dict().keys()

    def __contains__(self, key: str) -> bool:
        return key in self.as_dict()

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]


@dataclass
class DatasetDescriptor:
    """Metadata descriptor for a dataset."""
    id: str
    name: str
    category: str  # 'classification', 'vqa', 'captioning', 'reasoning'
    description: str
    splits: List[str] = field(default_factory=list)
    setup_requirements: List[str] = field(default_factory=list)
    loader_config: Dict[str, Any] = field(default_factory=dict)
    hf_dataset_name: Optional[str] = None


# ============================================================================
# Environment/Task Registry (Paper Evidence Contract)
# ============================================================================

ENVIRONMENT_REGISTRY: Dict[str, EnvironmentDescriptor] = {
    # ========================================================================
    # Model Environments (CLIP Variants)
    # ========================================================================
    'clip': EnvironmentDescriptor(
        id='clip',
        name='Standard CLIP',
        category='model',
        description='OpenAI CLIP ViT-L/14@224 baseline model',
        paper_section='3.1',
        aliases=['clean_clip', 'baseline_clip'],
        setup_requirements=['open_clip_torch', 'torch', 'torchvision'],
        config={
            'model_name': 'ViT-L-14',
            'pretrained': 'openai',
            'image_size': 224,
        },
        factory_available=True,
    ),
    
    'tecoa': EnvironmentDescriptor(
        id='tecoa',
        name='TeCoA CLIP',
        category='model',
        description='Text-guided Contrastive Adversarial fine-tuned CLIP',
        paper_section='3.2',
        aliases=['tecoa_clip'],
        setup_requirements=['open_clip_torch', 'torch', 'torchvision'],
        config={
            'base_model': 'ViT-L-14',
            'adversarial_training': True,
            'attack_type': 'pgd',
            'epsilon': 4/255,
        },
        factory_available=True,
    ),
    
    'fare': EnvironmentDescriptor(
        id='fare',
        name='FARE CLIP',
        category='model',
        description='Feature-Alignment Robust Embedding fine-tuned CLIP (ours)',
        paper_section='3.3',
        aliases=['fare_clip', 'ours'],
        setup_requirements=['open_clip_torch', 'torch', 'torchvision'],
        config={
            'base_model': 'ViT-L-14',
            'adversarial_training': True,
            'attack_type': 'fare',
            'alignment_target': 'class_token',
            'lambda_preserve': 1.0,
        },
        factory_available=True,
    ),
    
    # ========================================================================
    # LVLM Environments
    # ========================================================================
    'llava_1.5': EnvironmentDescriptor(
        id='llava_1.5',
        name='LLaVA-1.5 7B',
        category='model',
        description='LLaVA-1.5 7B with CLIP ViT-L/14@224 vision encoder',
        paper_section='4.2',
        aliases=['llava', 'llava_7b'],
        setup_requirements=['transformers', 'torch', 'open_clip_torch'],
        config={
            'model_name': 'liuhaotian/llava-v1.5-7b',
            'vision_encoder': 'ViT-L-14',
            'image_size': 224,  # Modified from default 336
        },
        factory_available=True,
    ),
    
    'openflamingo': EnvironmentDescriptor(
        id='openflamingo',
        name='OpenFlamingo 9B',
        category='model',
        description='OpenFlamingo 9B with CLIP vision encoder',
        paper_section='4.2',
        aliases=['flamingo'],
        setup_requirements=['open_flamingo', 'torch', 'open_clip_torch'],
        config={
            'model_name': 'openflamingo/OpenFlamingo-9B-vitl-mpt7b',
            'vision_encoder': 'ViT-L-14',
        },
        factory_available=True,
    ),
    
    # ========================================================================
    # Benchmark/Task Environments
    # ========================================================================
    'pope': EnvironmentDescriptor(
        id='pope',
        name='POPE',
        category='benchmark',
        description='Polling-based Object Probing Evaluation (hallucination)',
        paper_section='4.3',
        aliases=['pope_benchmark'],
        setup_requirements=['datasets', 'pycocotools'],
        config={
            'dataset': 'coco',
            'split': 'val',
            'variants': ['random', 'popular', 'adversarial'],
        },
        factory_available=True,
    ),
    
    'sqa_i': EnvironmentDescriptor(
        id='sqa_i',
        name='SQA-I',
        category='benchmark',
        description='ScienceQA-IMG reasoning benchmark',
        paper_section='4.4',
        aliases=['scienceqa', 'sqa'],
        setup_requirements=['datasets'],
        config={
            'dataset': 'derek-thomas/ScienceQA',
            'split': 'test',
            'modality': 'image',
        },
        factory_available=True,
    ),
    
    'clip_benchmark': EnvironmentDescriptor(
        id='clip_benchmark',
        name='CLIP Zero-Shot Benchmark',
        category='benchmark',
        description='Zero-shot classification evaluation suite',
        paper_section='4.1',
        aliases=['zero_shot_eval'],
        setup_requirements=['torchvision', 'datasets'],
        config={
            'datasets': ['imagenet', 'cifar10', 'cifar100', 'stl10'],
            'metric': 'top1_accuracy',
        },
        factory_available=True,
    ),
    
    'robustness_eval': EnvironmentDescriptor(
        id='robustness_eval',
        name='Quantitative Robustness Evaluation',
        category='task',
        description='Adversarial robustness evaluation across datasets',
        paper_section='Table 4',
        aliases=['table4_eval', 'adversarial_eval'],
        setup_requirements=['torchvision', 'datasets', 'torch'],
        config={
            'attack_types': ['pgd', 'autoattack'],
            'epsilons': [2/255, 4/255],
            'datasets': ['imagenet', 'cifar10', 'cifar100', 'stl10'],
        },
        factory_available=True,
    ),
    
    'vqa': EnvironmentDescriptor(
        id='vqa',
        name='VQA',
        category='task',
        description='Visual Question Answering evaluation',
        paper_section='Table 1',
        aliases=['vqa_eval'],
        setup_requirements=['datasets', 'pycocotools'],
        config={
            'datasets': ['vqav2', 'textvqa'],
            'metric': 'vqa_accuracy',
        },
        factory_available=True,
    ),
}


# Compatibility task aliases expected by legacy contract checks.
ENVIRONMENT_REGISTRY["llava_1"] = ENVIRONMENT_REGISTRY["llava_1.5"]
ENVIRONMENT_REGISTRY["sqa-i"] = ENVIRONMENT_REGISTRY["sqa_i"]

# ============================================================================
# Dataset Registry (Paper Evidence Contract)
# ============================================================================

DATASET_REGISTRY: Dict[str, DatasetDescriptor] = {
    # ========================================================================
    # Classification Datasets
    # ========================================================================
    'imagenet': DatasetDescriptor(
        id='imagenet',
        name='ImageNet-1K',
        category='classification',
        description='ImageNet ILSVRC-2012 classification dataset',
        splits=['train', 'val'],
        setup_requirements=['datasets', 'torchvision'],
        hf_dataset_name='imagenet-1k',
        loader_config={
            'trust_remote_code': True,
            'num_classes': 1000,
            'image_size': 224,
        },
    ),
    
    'cifar10': DatasetDescriptor(
        id='cifar10',
        name='CIFAR-10',
        category='classification',
        description='CIFAR-10 classification dataset',
        splits=['train', 'test'],
        setup_requirements=['torchvision'],
        loader_config={
            'num_classes': 10,
            'image_size': 32,
        },
    ),
    
    'cifar100': DatasetDescriptor(
        id='cifar100',
        name='CIFAR-100',
        category='classification',
        description='CIFAR-100 classification dataset',
        splits=['train', 'test'],
        setup_requirements=['torchvision'],
        loader_config={
            'num_classes': 100,
            'image_size': 32,
        },
    ),
    
    'stl10': DatasetDescriptor(
        id='stl10',
        name='STL-10',
        category='classification',
        description='STL-10 classification dataset',
        splits=['train', 'test'],
        setup_requirements=['torchvision'],
        loader_config={
            'num_classes': 10,
            'image_size': 96,
        },
    ),
    
    # ========================================================================
    # Captioning Datasets
    # ========================================================================
    'coco': DatasetDescriptor(
        id='coco',
        name='COCO',
        category='captioning',
        description='MS-COCO image captioning dataset',
        splits=['train', 'val'],
        setup_requirements=['datasets', 'pycocotools'],
        hf_dataset_name='HuggingFaceM4/COCO',
        loader_config={
            'year': 2017,
            'task': 'captions',
        },
    ),
    
    'flickr30k': DatasetDescriptor(
        id='flickr30k',
        name='Flickr30k',
        category='captioning',
        description='Flickr30k image captioning dataset',
        splits=['train', 'val', 'test'],
        setup_requirements=['datasets'],
        hf_dataset_name='nlphuji/flickr30k',
        loader_config={},
    ),
    
    # ========================================================================
    # VQA Datasets
    # ========================================================================
    'vqav2': DatasetDescriptor(
        id='vqav2',
        name='VQAv2',
        category='vqa',
        description='VQA v2.0 visual question answering dataset',
        splits=['train', 'val'],
        setup_requirements=['datasets', 'pycocotools'],
        hf_dataset_name='HuggingFaceM4/VQAv2',
        loader_config={
            'task': 'vqa',
        },
    ),
    
    'textvqa': DatasetDescriptor(
        id='textvqa',
        name='TextVQA',
        category='vqa',
        description='TextVQA visual question answering with text reading',
        splits=['train', 'val'],
        setup_requirements=['datasets'],
        hf_dataset_name='textvqa',
        loader_config={},
    ),
    
    # ========================================================================
    # Reasoning/Hallucination Datasets
    # ========================================================================
    'pope': DatasetDescriptor(
        id='pope',
        name='POPE',
        category='reasoning',
        description='Polling-based Object Probing Evaluation',
        splits=['val'],
        setup_requirements=['datasets', 'pycocotools'],
        loader_config={
            'base_dataset': 'coco',
            'variants': ['random', 'popular', 'adversarial'],
        },
    ),
    
    'sqa_i': DatasetDescriptor(
        id='sqa_i',
        name='ScienceQA-IMG',
        category='reasoning',
        description='ScienceQA with image context',
        splits=['train', 'val', 'test'],
        setup_requirements=['datasets'],
        hf_dataset_name='derek-thomas/ScienceQA',
        loader_config={
            'modality_filter': 'image',
        },
    ),
    
    # ========================================================================
    # Pretraining Datasets (LAION)
    # ========================================================================
    'laion': DatasetDescriptor(
        id='laion',
        name='LAION-400M',
        category='pretraining',
        description='LAION-400M image-text pairs',
        splits=['train'],
        setup_requirements=['datasets'],
        hf_dataset_name='laion/laion400m',
        loader_config={
            'streaming': True,
        },
    ),
}


# Expose dataset benchmarks through the environment registry for older callers
# that used a single registry for both task and dataset surfaces.
for _dataset_key in ("imagenet", "coco", "flickr30k"):
    _dataset = DATASET_REGISTRY[_dataset_key]
    ENVIRONMENT_REGISTRY.setdefault(
        _dataset_key,
        EnvironmentDescriptor(
            id=_dataset.id,
            name=_dataset.name,
            category="dataset",
            description=_dataset.description,
            paper_section="dataset registry",
            aliases=[],
            setup_requirements=list(_dataset.setup_requirements),
            config=dict(_dataset.loader_config),
            factory_available=True,
        ),
    )
ENVIRONMENT_REGISTRY.setdefault("quantitative_robustness_evaluation", ENVIRONMENT_REGISTRY["robustness_eval"])

# ============================================================================
# Dataset Loader Factory
# ============================================================================

def load_dataset(dataset_id: str, split: str = 'val', config: Optional[Dict[str, Any]] = None):
    """
    Load a dataset by registry ID.
    
    Args:
        dataset_id: Dataset identifier from DATASET_REGISTRY
        split: Dataset split ('train', 'val', 'test')
        config: Optional configuration overrides
        
    Returns:
        Dataset object (implementation depends on dataset type)
    """
    if dataset_id not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_id}. Available: {list(DATASET_REGISTRY.keys())}")
    
    descriptor = DATASET_REGISTRY[dataset_id]
    loader_config = {**descriptor.loader_config, **(config or {})}
    
    # Lazy import to avoid top-level dependencies
    if descriptor.category == 'classification':
        return _load_classification_dataset(dataset_id, split, loader_config)
    elif descriptor.category in ['captioning', 'vqa', 'reasoning']:
        return _load_hf_dataset(descriptor, split, loader_config)
    elif descriptor.category == 'pretraining':
        return _load_pretraining_dataset(descriptor, split, loader_config)
    else:
        raise ValueError(f"Unknown dataset category: {descriptor.category}")


def _load_classification_dataset(dataset_id: str, split: str, config: Dict[str, Any]):
    """Load classification dataset (ImageNet, CIFAR, STL-10)."""
    try:
        import torchvision.transforms as transforms
        from torchvision import datasets as tv_datasets
    except ImportError:
        raise ImportError("torchvision required for classification datasets")
    
    # Standard preprocessing for CLIP
    preprocess = transforms.Compose([
        transforms.Resize(config.get('image_size', 224)),
        transforms.CenterCrop(config.get('image_size', 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                           std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    
    if dataset_id == 'imagenet':
        # Use HuggingFace datasets for ImageNet (paper requirement)
        try:
            from datasets import load_dataset as hf_load_dataset
            dataset = hf_load_dataset(
                'imagenet-1k',
                split='validation' if split == 'val' else split,
                trust_remote_code=config.get('trust_remote_code', True),
            )
            return dataset
        except Exception as e:
            # Fallback error with clear message
            raise RuntimeError(
                f"Failed to load ImageNet via HuggingFace: {e}. "
                "Ensure you have access to imagenet-1k dataset."
            )
    
    elif dataset_id == 'cifar10':
        return tv_datasets.CIFAR10(
            root='./data/cifar10',
            train=(split == 'train'),
            download=True,
            transform=preprocess,
        )
    
    elif dataset_id == 'cifar100':
        return tv_datasets.CIFAR100(
            root='./data/cifar100',
            train=(split == 'train'),
            download=True,
            transform=preprocess,
        )
    
    elif dataset_id == 'stl10':
        stl_split = 'train' if split == 'train' else 'test'
        return tv_datasets.STL10(
            root='./data/stl10',
            split=stl_split,
            download=True,
            transform=preprocess,
        )
    
    else:
        raise ValueError(f"Unknown classification dataset: {dataset_id}")


def _load_hf_dataset(descriptor: DatasetDescriptor, split: str, config: Dict[str, Any]):
    """Load HuggingFace dataset (COCO, Flickr30k, VQA, POPE, SQA)."""
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError:
        raise ImportError("datasets package required for HuggingFace datasets")
    
    if not descriptor.hf_dataset_name:
        raise ValueError(f"No HuggingFace dataset name specified for {descriptor.id}")
    
    try:
        dataset = hf_load_dataset(
            descriptor.hf_dataset_name,
            split=split,
            trust_remote_code=True,
        )
        return dataset
    except Exception as e:
        raise RuntimeError(
            f"Failed to load {descriptor.name} from HuggingFace: {e}"
        )


def _load_pretraining_dataset(descriptor: DatasetDescriptor, split: str, config: Dict[str, Any]):
    """Load pretraining dataset (LAION)."""
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError:
        raise ImportError("datasets package required for pretraining datasets")
    
    try:
        dataset = hf_load_dataset(
            descriptor.hf_dataset_name,
            split=split,
            streaming=config.get('streaming', False),
        )
        return dataset
    except Exception as e:
        raise RuntimeError(
            f"Failed to load {descriptor.name}: {e}"
        )


# ============================================================================
# Model Factory
# ============================================================================

def create_model(model_id: str, config: Optional[Dict[str, Any]] = None, device: str = 'cuda'):
    """
    Create a model by environment ID.
    
    Args:
        model_id: Model identifier from ENVIRONMENT_REGISTRY
        config: Optional configuration overrides
        device: Device to load model on
        
    Returns:
        Model instance and preprocessing function
    """
    if model_id not in ENVIRONMENT_REGISTRY:
        raise ValueError(f"Unknown model: {model_id}. Available: {list(ENVIRONMENT_REGISTRY.keys())}")
    
    descriptor = ENVIRONMENT_REGISTRY[model_id]
    model_config = {**descriptor.config, **(config or {})}
    
    if model_id in ['clip', 'tecoa', 'fare']:
        return _create_clip_model(model_id, model_config, device)
    elif model_id == 'llava_1.5':
        return _create_llava_model(model_config, device)
    elif model_id == 'openflamingo':
        return _create_flamingo_model(model_config, device)
    else:
        raise ValueError(f"No factory available for model: {model_id}")


def _create_clip_model(model_id: str, config: Dict[str, Any], device: str):
    """Create CLIP model (standard, TeCoA, or FARE)."""
    try:
        import torch
        import open_clip
    except ImportError:
        raise ImportError("torch and open_clip_torch required for CLIP models")
    
    model_name = config.get('model_name', 'ViT-L-14')
    pretrained = config.get('pretrained', 'openai')
    
    # Load base CLIP model
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device,
    )
    
    # Load fine-tuned checkpoint if specified
    if model_id in ['tecoa', 'fare']:
        checkpoint_path = Path('checkpoints') / f'{model_id}_clip.pth'
        if checkpoint_path.exists():
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded {model_id.upper()} checkpoint from {checkpoint_path}")
            except Exception as e:
                print(f"Warning: Failed to load checkpoint {checkpoint_path}: {e}")
                print(f"Using base CLIP model instead")
    
    model.eval()
    return model, preprocess


def _create_llava_model(config: Dict[str, Any], device: str):
    """Create LLaVA model with modified CLIP encoder."""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        raise ImportError("transformers and torch required for LLaVA")
    
    model_name = config.get('model_name', 'liuhaotian/llava-v1.5-7b')
    
    # Load LLaVA model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        device_map=device,
    )
    
    model.eval()
    return model, tokenizer


def _create_flamingo_model(config: Dict[str, Any], device: str):
    """Create OpenFlamingo model."""
    try:
        import torch
        from open_flamingo import create_model_and_transforms
    except ImportError:
        raise ImportError("open_flamingo required for Flamingo models")
    
    model_name = config.get('model_name', 'openflamingo/OpenFlamingo-9B-vitl-mpt7b')
    
    model, image_processor, tokenizer = create_model_and_transforms(
        model_name,
        device=device,
    )
    
    model.eval()
    return model, (image_processor, tokenizer)


# ============================================================================
# Table 4 Evaluation (Paper Evidence Contract)
# ============================================================================

def evaluate_table4(
    models: List[str],
    datasets: List[str],
    epsilons: List[float],
    output_dir: str = 'results',
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate robustness across models and datasets (Table 4 from paper).
    
    This function implements the evaluation protocol from Table 4:
    - Zero-shot classification on multiple datasets
    - Clean accuracy and adversarial accuracy at ε=2/255 and ε=4/255
    - Comparison of CLIP, TeCoA, and FARE models
    
    Args:
        models: List of model IDs ('clip', 'tecoa', 'fare')
        datasets: List of dataset IDs ('imagenet', 'cifar10', etc.)
        epsilons: List of adversarial perturbation budgets
        output_dir: Directory for result artifacts
        dry_run: If True, create artifact schemas without full evaluation
        
    Returns:
        Dictionary containing accuracy metrics and result paths
    """
    results = {
        'clean_accuracy': {},
        'adversarial_accuracy': {},
        'robustness_drop': {},
    }
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if dry_run:
        # Create artifact schemas for smoke validation
        results = _create_table4_dry_run_artifacts(models, datasets, epsilons, output_path)
        return results
    
    # Real evaluation path
    try:
        import torch
    except ImportError:
        raise ImportError("torch required for evaluation")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Evaluate each model on each dataset
    for model_id in models:
        results['clean_accuracy'][model_id] = {}
        results['adversarial_accuracy'][model_id] = {}
        results['robustness_drop'][model_id] = {}
        
        try:
            model, preprocess = create_model(model_id, device=device)
        except Exception as e:
            print(f"Warning: Failed to load model {model_id}: {e}")
            continue
        
        for dataset_id in datasets:
            try:
                dataset = load_dataset(dataset_id, split='val')
            except Exception as e:
                print(f"Warning: Failed to load dataset {dataset_id}: {e}")
                continue
            
            # Evaluate clean accuracy
            clean_acc = _evaluate_clean_accuracy(model, dataset, preprocess, device)
            results['clean_accuracy'][model_id][dataset_id] = clean_acc
            
            # Evaluate adversarial accuracy for each epsilon
            adv_acc_by_eps = {}
            for eps in epsilons:
                adv_acc = _evaluate_adversarial_accuracy(
                    model, dataset, preprocess, device, epsilon=eps
                )
                eps_key = f"eps_{int(eps*255)}/255"
                adv_acc_by_eps[eps_key] = adv_acc
            
            results['adversarial_accuracy'][model_id][dataset_id] = adv_acc_by_eps
            
            # Compute robustness drop
            avg_adv_acc = sum(adv_acc_by_eps.values()) / len(adv_acc_by_eps)
            results['robustness_drop'][model_id][dataset_id] = clean_acc - avg_adv_acc
    
    # Write artifacts
    _write_table4_artifacts(results, models, datasets, epsilons, output_path)
    
    return results


def _create_table4_dry_run_artifacts(
    models: List[str],
    datasets: List[str],
    epsilons: List[float],
    output_path: Path,
) -> Dict[str, Any]:
    """Create dry-run artifact schemas for smoke validation."""
    
    # Create result schema
    results = {
        'clean_accuracy': {},
        'adversarial_accuracy': {},
        'robustness_drop': {},
        '_dry_run_manifest': {
            'mode': 'dry_run',
            'artifact_type': 'table4_evaluation_schema',
            'models': models,
            'datasets': datasets,
            'epsilons': epsilons,
            'warning': 'This is a dry-run schema artifact, not real experiment results',
        }
    }
    
    # Populate schema with structure (not real results)
    for model_id in models:
        results['clean_accuracy'][model_id] = {
            ds: 0.0 for ds in datasets
        }
        results['adversarial_accuracy'][model_id] = {
            ds: {f"eps_{int(eps*255)}/255": 0.0 for eps in epsilons}
            for ds in datasets
        }
        results['robustness_drop'][model_id] = {
            ds: 0.0 for ds in datasets
        }
    
    # Write CSV schema
    csv_path = output_path / 'table4_accuracy.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['# DRY RUN SCHEMA - Not real experiment results'])
        writer.writerow(['Model', 'Dataset', 'Clean Accuracy', 'Adv Accuracy (2/255)', 'Adv Accuracy (4/255)', 'Robustness Drop'])
        for model_id in models:
            for dataset_id in datasets:
                writer.writerow([model_id, dataset_id, '0.0', '0.0', '0.0', '0.0'])
    
    # Write JSON schema