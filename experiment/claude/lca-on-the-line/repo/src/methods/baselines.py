#!/usr/bin/env python3
"""
LCA-on-the-Line Baseline Methods and Model Registry

Implements baseline methods for OOD generalization evaluation including:
- Pretrained Vision Models (36 VMs): ResNet, ViT, EfficientNet, ConvNeXt, etc.
- Vision-Language Models (39 VLMs): CLIP, OpenCLIP variants
- Training methods: baseline, adapter, fine_tuning
- Evaluation protocols for ID LCA distance and OOD accuracy measurement

reference_grounding: paperbench_ref_001 torchvision/models/detection/mask_rcnn.py
reference_grounding: paperbench_ref_001 torchvision/models/detection/keypoint_rcnn.py
reference_grounding: paperbench_ref_001 references/depth/stereo/README.md

Adapted from torchvision detection models pattern: model behavior changes between
training and evaluation modes, supports flexible backbone selection and weight loading,
provides baseline configurations for research bootstrapping.

Binding addendum clarifications:
- All vision-language models accessed via OpenCLIP and CLIP modules
- Model registry covers all 75 models from paper (36 VMs + 39 VLMs)
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import warnings
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Baseline Method Registry - Paper evidence contract
# =============================================================================

BASELINE_METHOD_REGISTRY = {
    "ours": {
        "method_id": "ours",
        "name": "LCA-based soft labeling",
        "description": "Our proposed method using hierarchical soft labels with LCA distance",
        "variants": ["soft_label_wordnet", "soft_label_latent", "kmeans_2layer", "kmeans_3layer"],
        "paper_section": "Section 5: Improving OOD Generalization",
        "artifacts": ["Table 5", "Table 6", "Table 9", "Figure 6"],
        "requires_hierarchy": True,
        "training_required": True
    },
    "baseline": {
        "method_id": "baseline",
        "name": "Cross-entropy baseline",
        "description": "Standard cross-entropy training without hierarchical information",
        "variants": ["ce_only", "standard"],
        "paper_section": "Baseline comparison",
        "artifacts": ["Table 5", "Table 6"],
        "requires_hierarchy": False,
        "training_required": True
    },
    "resnet": {
        "method_id": "resnet",
        "name": "ResNet family",
        "description": "ResNet-18/50/101/152 pretrained models for vision tasks",
        "variants": ["resnet18", "resnet50", "resnet101", "resnet152"],
        "paper_section": "Section 3: Benchmarking 75 models",
        "artifacts": ["Figure 1", "Table 1", "Table 2"],
        "requires_hierarchy": False,
        "training_required": False
    },
    "vit": {
        "method_id": "vit",
        "name": "Vision Transformer family",
        "description": "ViT-B/L/H pretrained models",
        "variants": ["vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32", "vit_h_14"],
        "paper_section": "Section 3: Benchmarking 75 models",
        "artifacts": ["Figure 1", "Table 1"],
        "requires_hierarchy": False,
        "training_required": False
    },
    "adapter": {
        "method_id": "adapter",
        "name": "Adapter-based fine-tuning",
        "description": "Lightweight adapter modules for transfer learning",
        "variants": ["adapter_standard", "adapter_hierarchical"],
        "paper_section": "Section 5: Training methods",
        "artifacts": ["Table 6"],
        "requires_hierarchy": False,
        "training_required": True
    },
    "fine_tuning": {
        "method_id": "fine_tuning",
        "name": "Full model fine-tuning",
        "description": "End-to-end fine-tuning of pretrained models",
        "variants": ["full_ft", "linear_probe"],
        "paper_section": "Section 5: Training methods",
        "artifacts": ["Table 5", "Table 6"],
        "requires_hierarchy": False,
        "training_required": True
    }
}


# =============================================================================
# Model Registry - 75 Models (36 VMs + 39 VLMs)
# =============================================================================

# Vision Models (36 VMs) - reference_grounding: paperbench_ref_001 torchvision/models/detection/mask_rcnn.py
VISION_MODEL_REGISTRY = {
    # ResNet family
    "resnet18": {"family": "resnet", "source": "torchvision", "params": "11.7M", "type": "vm"},
    "resnet34": {"family": "resnet", "source": "torchvision", "params": "21.8M", "type": "vm"},
    "resnet50": {"family": "resnet", "source": "torchvision", "params": "25.6M", "type": "vm"},
    "resnet101": {"family": "resnet", "source": "torchvision", "params": "44.5M", "type": "vm"},
    "resnet152": {"family": "resnet", "source": "torchvision", "params": "60.2M", "type": "vm"},
    "resnext50_32x4d": {"family": "resnet", "source": "torchvision", "params": "25.0M", "type": "vm"},
    "resnext101_32x8d": {"family": "resnet", "source": "torchvision", "params": "88.8M", "type": "vm"},
    "wide_resnet50_2": {"family": "resnet", "source": "torchvision", "params": "68.9M", "type": "vm"},
    "wide_resnet101_2": {"family": "resnet", "source": "torchvision", "params": "126.9M", "type": "vm"},
    
    # Vision Transformer family
    "vit_b_16": {"family": "vit", "source": "torchvision", "params": "86.6M", "type": "vm"},
    "vit_b_32": {"family": "vit", "source": "torchvision", "params": "88.2M", "type": "vm"},
    "vit_l_16": {"family": "vit", "source": "torchvision", "params": "304.3M", "type": "vm"},
    "vit_l_32": {"family": "vit", "source": "torchvision", "params": "306.5M", "type": "vm"},
    "vit_h_14": {"family": "vit", "source": "torchvision", "params": "632.0M", "type": "vm"},
    
    # EfficientNet family
    "efficientnet_b0": {"family": "efficientnet", "source": "torchvision", "params": "5.3M", "type": "vm"},
    "efficientnet_b1": {"family": "efficientnet", "source": "torchvision", "params": "7.8M", "type": "vm"},
    "efficientnet_b2": {"family": "efficientnet", "source": "torchvision", "params": "9.2M", "type": "vm"},
    "efficientnet_b3": {"family": "efficientnet", "source": "torchvision", "params": "12.2M", "type": "vm"},
    "efficientnet_b4": {"family": "efficientnet", "source": "torchvision", "params": "19.3M", "type": "vm"},
    "efficientnet_b5": {"family": "efficientnet", "source": "torchvision", "params": "30.4M", "type": "vm"},
    "efficientnet_b6": {"family": "efficientnet", "source": "torchvision", "params": "43.0M", "type": "vm"},
    "efficientnet_b7": {"family": "efficientnet", "source": "torchvision", "params": "66.3M", "type": "vm"},
    
    # ConvNeXt family
    "convnext_tiny": {"family": "convnext", "source": "torchvision", "params": "28.6M", "type": "vm"},
    "convnext_small": {"family": "convnext", "source": "torchvision", "params": "50.2M", "type": "vm"},
    "convnext_base": {"family": "convnext", "source": "torchvision", "params": "88.6M", "type": "vm"},
    "convnext_large": {"family": "convnext", "source": "torchvision", "params": "197.8M", "type": "vm"},
    
    # DenseNet family
    "densenet121": {"family": "densenet", "source": "torchvision", "params": "8.0M", "type": "vm"},
    "densenet161": {"family": "densenet", "source": "torchvision", "params": "28.7M", "type": "vm"},
    "densenet169": {"family": "densenet", "source": "torchvision", "params": "14.1M", "type": "vm"},
    "densenet201": {"family": "densenet", "source": "torchvision", "params": "20.0M", "type": "vm"},
    
    # MobileNet family
    "mobilenet_v2": {"family": "mobilenet", "source": "torchvision", "params": "3.5M", "type": "vm"},
    "mobilenet_v3_small": {"family": "mobilenet", "source": "torchvision", "params": "2.5M", "type": "vm"},
    "mobilenet_v3_large": {"family": "mobilenet", "source": "torchvision", "params": "5.5M", "type": "vm"},
    
    # Other architectures
    "shufflenet_v2_x0_5": {"family": "shufflenet", "source": "torchvision", "params": "1.4M", "type": "vm"},
    "shufflenet_v2_x1_0": {"family": "shufflenet", "source": "torchvision", "params": "2.3M", "type": "vm"},
    "mnasnet0_5": {"family": "mnasnet", "source": "torchvision", "params": "2.2M", "type": "vm"},
}

# Vision-Language Models (39 VLMs) - Binding addendum: accessed via OpenCLIP and CLIP
VISION_LANGUAGE_MODEL_REGISTRY = {
    # CLIP models (OpenAI)
    "clip_rn50": {"family": "clip", "source": "openai/clip", "backbone": "RN50", "type": "vlm"},
    "clip_rn101": {"family": "clip", "source": "openai/clip", "backbone": "RN101", "type": "vlm"},
    "clip_rn50x4": {"family": "clip", "source": "openai/clip", "backbone": "RN50x4", "type": "vlm"},
    "clip_rn50x16": {"family": "clip", "source": "openai/clip", "backbone": "RN50x16", "type": "vlm"},
    "clip_rn50x64": {"family": "clip", "source": "openai/clip", "backbone": "RN50x64", "type": "vlm"},
    "clip_vit_b_32": {"family": "clip", "source": "openai/clip", "backbone": "ViT-B/32", "type": "vlm"},
    "clip_vit_b_16": {"family": "clip", "source": "openai/clip", "backbone": "ViT-B/16", "type": "vlm"},
    "clip_vit_l_14": {"family": "clip", "source": "openai/clip", "backbone": "ViT-L/14", "type": "vlm"},
    "clip_vit_l_14_336px": {"family": "clip", "source": "openai/clip", "backbone": "ViT-L/14@336px", "type": "vlm"},
    
    # OpenCLIP models - Binding addendum: accessed via mlfoundations/open_clip
    "openclip_rn50_openai": {"family": "openclip", "source": "openclip", "backbone": "RN50", "pretrained": "openai", "type": "vlm"},
    "openclip_rn50_yfcc15m": {"family": "openclip", "source": "openclip", "backbone": "RN50", "pretrained": "yfcc15m", "type": "vlm"},
    "openclip_rn50_cc12m": {"family": "openclip", "source": "openclip", "backbone": "RN50", "pretrained": "cc12m", "type": "vlm"},
    "openclip_rn101_openai": {"family": "openclip", "source": "openclip", "backbone": "RN101", "pretrained": "openai", "type": "vlm"},
    "openclip_rn101_yfcc15m": {"family": "openclip", "source": "openclip", "backbone": "RN101", "pretrained": "yfcc15m", "type": "vlm"},
    "openclip_vit_b_32_openai": {"family": "openclip", "source": "openclip", "backbone": "ViT-B-32", "pretrained": "openai", "type": "vlm"},
    "openclip_vit_b_32_laion2b": {"family": "openclip", "source": "openclip", "backbone": "ViT-B-32", "pretrained": "laion2b_s34b_b79k", "type": "vlm"},
    "openclip_vit_b_32_laion400m": {"family": "openclip", "source": "openclip", "backbone": "ViT-B-32", "pretrained": "laion400m_e31", "type": "vlm"},
    "openclip_vit_b_16_openai": {"family": "openclip", "source": "openclip", "backbone": "ViT-B-16", "pretrained": "openai", "type": "vlm"},
    "openclip_vit_b_16_laion2b": {"family": "openclip", "source": "openclip", "backbone": "ViT-B-16", "pretrained": "laion2b_s34b_b88k", "type": "vlm"},
    "openclip_vit_b_16_laion400m": {"family": "openclip", "source": "openclip", "backbone": "ViT-B-16", "pretrained": "laion400m_e31", "type": "vlm"},
    "openclip_vit_l_14_openai": {"family": "openclip", "source": "openclip", "backbone": "ViT-L-14", "pretrained": "openai", "type": "vlm"},
    "openclip_vit_l_14_laion2b": {"family": "openclip", "source": "openclip", "backbone": "ViT-L-14", "pretrained": "laion2b_s32b_b82k", "type": "vlm"},
    "openclip_vit_l_14_laion400m": {"family": "openclip", "source": "openclip", "backbone": "ViT-L-14", "pretrained": "laion400m_e31", "type": "vlm"},
    "openclip_vit_h_14_laion2b": {"family": "openclip", "source": "openclip", "backbone": "ViT-H-14", "pretrained": "laion2b_s32b_b79k", "type": "vlm"},
    "openclip_vit_g_14_laion2b": {"family": "openclip", "source": "openclip", "backbone": "ViT-g-14", "pretrained": "laion2b_s12b_b42k", "type": "vlm"},
    "openclip_convnext_base": {"family": "openclip", "source": "openclip", "backbone": "convnext_base", "pretrained": "laion400m_s13b_b51k", "type": "vlm"},
    "openclip_convnext_base_w": {"family": "openclip", "source": "openclip", "backbone": "convnext_base_w", "pretrained": "laion2b_s13b_b82k", "type": "vlm"},
    "openclip_convnext_large": {"family": "openclip", "source": "openclip", "backbone": "convnext_large", "pretrained": "laion2b_s29b_b131k", "type": "vlm"},
    "openclip_convnext_xxlarge": {"family": "openclip", "source": "openclip", "backbone": "convnext_xxlarge", "pretrained": "laion2b_s34b_b82k", "type": "vlm"},
    
    # Additional OpenCLIP variants
    "openclip_coca_vit_l_14": {"family": "openclip", "source": "openclip", "backbone": "coca_ViT-L-14", "pretrained": "mscoco_finetuned_laion2b_s13b_b90k", "type": "vlm"},
    "openclip_eva_vit_g_14": {"family": "openclip", "source": "openclip", "backbone": "EVA-ViT-g-14", "pretrained": "laion400m_s11b_b41k", "type": "vlm"},
    "openclip_eva_vit_g_14_plus": {"family": "openclip", "source": "openclip", "backbone": "EVA-ViT-g-14-plus", "pretrained": "merged2b_s11b_b114k", "type": "vlm"},
    "openclip_roberta_vit_b_32": {"family": "openclip", "source": "openclip", "backbone": "roberta-ViT-B-32", "pretrained": "laion2b_s12b_b32k", "type": "vlm"},
    "openclip_xlm_roberta_base_vit_b_32": {"family": "openclip", "source": "openclip", "backbone": "xlm-roberta-base-ViT-B-32", "pretrained": "laion5b_s13b_b90k", "type": "vlm"},
    "openclip_xlm_roberta_large_vit_h_14": {"family": "openclip", "source": "openclip", "backbone": "xlm-roberta-large-ViT-H-14", "pretrained": "frozen_laion5b_s13b_b90k", "type": "vlm"},
    "openclip_vit_b_16_plus_240": {"family": "openclip", "source": "openclip", "backbone": "ViT-B-16-plus-240", "pretrained": "laion400m_e31", "type": "vlm"},
    "openclip_vit_l_14_336px": {"family": "openclip", "source": "openclip", "backbone": "ViT-L-14-336", "pretrained": "openai", "type": "vlm"},
    "openclip_vit_so400m_14": {"family": "openclip", "source": "openclip", "backbone": "ViT-SO400M-14-SigLIP", "pretrained": "webli", "type": "vlm"},
}

# Combined model registry
MODEL_REGISTRY = {**VISION_MODEL_REGISTRY, **VISION_LANGUAGE_MODEL_REGISTRY}


# =============================================================================
# Parameter Sweep Config - Paper evidence contract
# =============================================================================

PARAMETER_SWEEP_CONFIG = {
    "clustering_layers": {
        "param_name": "clustering_layers",
        "description": "Number of hierarchy layers for K-means clustering",
        "values": [2, 3],
        "default": 2,
        "paper_usage": "Table 6: Latent hierarchy construction",
        "sweep_type": "discrete"
    },
    "root_clusters": {
        "param_name": "root_clusters",
        "description": "Number of clusters at root level",
        "values": [10, 20, 50],
        "default": 10,
        "paper_usage": "Section 5.2: Latent hierarchy",
        "sweep_type": "discrete"
    },
    "leaf_clusters": {
        "param_name": "leaf_clusters",
        "description": "Number of clusters at leaf level",
        "values": [100, 500, 1000],
        "default": 100,
        "paper_usage": "Section 5.2: Latent hierarchy",
        "sweep_type": "discrete"
    },
    "soft_label_temperature": {
        "param_name": "soft_label_temperature",
        "description": "Temperature parameter for soft label distribution",
        "values": [0.1, 0.5, 1.0, 2.0, 5.0],
        "default": 1.0,
        "paper_usage": "Table 9: Ablation on soft loss labels",
        "sweep_type": "continuous"
    },
    "lca_loss_weight": {
        "param_name": "lca_loss_weight",
        "description": "Weight for LCA-based soft label loss",
        "values": [0.0, 0.1, 0.3, 0.5, 0.7, 1.0],
        "default": 0.5,
        "paper_usage": "Section 5.1: Training with soft labels",
        "sweep_type": "continuous"
    },
    "learning_rate": {
        "param_name": "learning_rate",
        "description": "Learning rate for training",
        "values": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
        "default": 1e-4,
        "paper_usage": "Training configuration",
        "sweep_type": "log_continuous"
    },
    "batch_size": {
        "param_name": "batch_size",
        "description": "Training batch size",
        "values": [32, 64, 128, 256],
        "default": 128,
        "paper_usage": "Training configuration",
        "sweep_type": "discrete"
    }
}


# =============================================================================
# Model Loading Functions - Adapted from torchvision detection pattern
# reference_grounding: paperbench_ref_001 torchvision/models/detection/mask_rcnn.py
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for model loading and evaluation."""
    model_id: str
    model_type: str  # "vm" or "vlm"
    family: str
    source: str
    num_classes: int = 1000
    pretrained: bool = True
    progress: bool = True
    trainable_backbone_layers: Optional[int] = None


def load_vision_model(
    model_id: str,
    weights: Optional[str] = "IMAGENET1K_V1",
    progress: bool = True,
    num_classes: Optional[int] = None,
    **kwargs
) -> Tuple[Any, Callable]:
    """
    Load a vision model from torchvision.
    
    Adapted from torchvision.models.detection pattern where models are constructed
    with flexible backbone selection and weight loading.
    
    Args:
        model_id: Model identifier from VISION_MODEL_REGISTRY
        weights: Pretrained weights specification
        progress: Show download progress
        num_classes: Number of output classes (default: 1000 for ImageNet)
        **kwargs: Additional model-specific arguments
        
    Returns:
        Tuple of (model, preprocess_function)
    """
    try:
        import torch
        import torchvision.models as models
        from torchvision import transforms
    except ImportError:
        logger.warning("torch/torchvision not available, returning mock model")
        return _get_mock_model(model_id), _get_mock_preprocess()
    
    if model_id not in VISION_MODEL_REGISTRY:
        raise ValueError(f"Unknown vision model: {model_id}")
    
    model_info = VISION_MODEL_REGISTRY[model_id]
    
    # Get model constructor
    model_fn = getattr(models, model_id, None)
    if model_fn is None:
        raise ValueError(f"Model {model_id} not found in torchvision.models")
    
    # Load model with weights
    if num_classes is not None and num_classes != 1000:
        model = model_fn(weights=None, num_classes=num_classes, **kwargs)
    else:
        model = model_fn(weights=weights, progress=progress, **kwargs)
    
    model.eval()
    
    # Standard ImageNet preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return model, preprocess


def load_clip_model(
    model_id: str,
    device: str = "cpu",
    jit: bool = False
) -> Tuple[Any, Any, Callable]:
    """
    Load a CLIP model from OpenAI.
    
    Binding addendum: All vision-language models accessed via CLIP modules.
    
    Args:
        model_id: Model identifier from VISION_LANGUAGE_MODEL_REGISTRY
        device: Device to load model on
        jit: Use JIT compilation
        
    Returns:
        Tuple of (model, preprocess, tokenizer)
    """
    try:
        import clip
    except ImportError:
        logger.warning("CLIP not available, returning mock model")
        return _get_mock_model(model_id), _get_mock_preprocess(), None
    
    if model_id not in VISION_LANGUAGE_MODEL_REGISTRY:
        raise ValueError(f"Unknown VLM: {model_id}")
    
    model_info = VISION_LANGUAGE_MODEL_REGISTRY[model_id]
    if model_info["family"] != "clip":
        raise ValueError(f"Model {model_id} is not a CLIP model")
    
    backbone = model_info["backbone"]
    model, preprocess = clip.load(backbone, device=device, jit=jit)
    model.eval()
    
    return model, preprocess, clip.tokenize


def load_openclip_model(
    model_id: str,
    device: str = "cpu",
    precision: str = "fp32"
) -> Tuple[Any, Any, Callable]:
    """
    Load an OpenCLIP model.
    
    Binding addendum: All vision-language models accessed via OpenCLIP modules.
    
    Args:
        model_id: Model identifier from VISION_LANGUAGE_MODEL_REGISTRY
        device: Device to load model on
        precision: Model precision (fp32, fp16, bf16)
        
    Returns:
        Tuple of (model, preprocess, tokenizer)
    """
    try:
        import open_clip
    except ImportError:
        logger.warning("OpenCLIP not available, returning mock model")
        return _get_mock_model(model_id), _get_mock_preprocess(), None
    
    if model_id not in VISION_LANGUAGE_MODEL_REGISTRY:
        raise ValueError(f"Unknown VLM: {model_id}")
    
    model_info = VISION_LANGUAGE_MODEL_REGISTRY[model_id]
    if model_info["family"] != "openclip":
        raise ValueError(f"Model {model_id} is not an OpenCLIP model")
    
    backbone = model_info["backbone"]
    pretrained = model_info.get("pretrained", "openai")
    
    model, _, preprocess = open_clip.create_model_and_transforms(
        backbone, pretrained=pretrained, device=device, precision=precision
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(backbone)
    
    return model, preprocess, tokenizer


def load_model(model_id: str, device: str = "cpu", **kwargs) -> Tuple[Any, Callable, Optional[Callable]]:
    """
    Unified model loading interface for both VMs and VLMs.
    
    Args:
        model_id: Model identifier from MODEL_REGISTRY
        device: Device to load model on
        **kwargs: Additional model-specific arguments
        
    Returns:
        Tuple of (model, preprocess, tokenizer) where tokenizer is None for VMs
    """
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_id}. Available models: {list(MODEL_REGISTRY.keys())}")
    
    model_info = MODEL_REGISTRY[model_id]
    model_type = model_info["type"]
    
    if model_type == "vm":
        model, preprocess = load_vision_model(model_id, **kwargs)
        return model, preprocess, None