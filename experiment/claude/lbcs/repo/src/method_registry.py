"""
Method and model registry for Refined Coreset Selection experiments.

Registers coreset selection methods (LBCS and baselines), model architectures,
and training configurations for the paper: "Refined Coreset Selection: Towards
Minimal Coreset Size under Model Performance Constraints."

Paper evidence contract: expose method/baseline/attack selectors for ours, random,
baseline, oracle, vit, resnet, adapter, fine_tuning.

reference_grounding: paperbench_ref_003 train.py
reference_grounding: paperbench_ref_003 selection.py
reference_grounding: paperbench_ref_004 cnn_mnist_probability_1step_pixel_shared_rein.py
reference_grounding: paperbench_ref_004 noisy_label.py
"""

from typing import Dict, Any, Optional, List, Callable
import warnings

# ============================================================================
# Coreset Selection Method Registry
# Paper evidence contract: complete method/baseline selector set must include
# ours, random, baseline, oracle, vit, resnet, adapter, fine_tuning.
# reference_grounding: paperbench_ref_003 selection.py
# ============================================================================

METHOD_REGISTRY: Dict[str, Dict[str, Any]] = {
    "lbcs": {
        "id": "lbcs",
        "aliases": ["LBCS", "ours", "lexicographic_bilevel"],
        "name": "Lexicographic Bilevel Coreset Selection",
        "paper_section": "Algorithm 1",
        "description": "Our proposed LBCS method with lexicographic bilevel optimization",
        "type": "coreset_selection",
        "requires_inner_training": True,
        "requires_outer_optimization": True,
        "hyperparameters": {
            "T": 1000,  # outer loop iterations (binding addendum clarification)
            "inner_epochs": 5,
            "epsilon": 0.3,  # performance tolerance
            "initial_k": 600,
            "lr_outer": 0.01,
            "temperature": 0.1,
        },
        "module_path": "src.methods.methods",
        "class_name": "LBCSMethod",
    },
    "uniform": {
        "id": "uniform",
        "aliases": ["random", "baseline", "uniform_random"],
        "name": "Uniform Random Selection",
        "paper_section": "Section 5.2",
        "description": "Baseline method: uniform random sampling",
        "type": "coreset_selection",
        "requires_inner_training": False,
        "requires_outer_optimization": False,
        "hyperparameters": {},
        "module_path": "src.methods.baselines",
        "class_name": "UniformSelection",
    },
    "el2n": {
        "id": "el2n",
        "aliases": ["EL2N", "L2"],
        "name": "Error L2-Norm Selection",
        "paper_section": "Section 5.2",
        "description": "Baseline method: EL2N score-based selection",
        "type": "coreset_selection",
        "requires_inner_training": True,
        "requires_outer_optimization": False,
        "hyperparameters": {
            "epochs": 10,
        },
        "module_path": "src.methods.baselines",
        "class_name": "EL2NSelection",
    },
    "grand": {
        "id": "grand",
        "aliases": ["GraNd", "GRAND", "gradient_norm"],
        "name": "Gradient Norm Selection",
        "paper_section": "Section 5.2",
        "description": "Baseline method: gradient norm-based selection",
        "type": "coreset_selection",
        "requires_inner_training": True,
        "requires_outer_optimization": False,
        "hyperparameters": {
            "epochs": 10,
        },
        "module_path": "src.methods.baselines",
        "class_name": "GraNdSelection",
    },
    "influential": {
        "id": "influential",
        "aliases": ["Influential", "influence"],
        "name": "Influential Sample Selection",
        "paper_section": "Section 5.2",
        "description": "Baseline method: influence function-based selection",
        "type": "coreset_selection",
        "requires_inner_training": True,
        "requires_outer_optimization": False,
        "hyperparameters": {
            "epochs": 10,
        },
        "module_path": "src.methods.baselines",
        "class_name": "InfluentialSelection",
    },
    "moderate": {
        "id": "moderate",
        "aliases": ["Moderate", "moderate_ds"],
        "name": "Moderate Data Selection",
        "paper_section": "Section 5.2",
        "description": "Baseline method: moderate difficulty-based selection",
        "type": "coreset_selection",
        "requires_inner_training": True,
        "requires_outer_optimization": False,
        "hyperparameters": {
            "epochs": 10,
        },
        "module_path": "src.methods.baselines",
        "class_name": "ModerateSelection",
    },
    "ccs": {
        "id": "ccs",
        "aliases": ["CCS", "contextual_coreset"],
        "name": "Contextual Coreset Selection",
        "paper_section": "Section 5.2",
        "description": "Baseline method: contextual coreset selection",
        "type": "coreset_selection",
        "requires_inner_training": True,
        "requires_outer_optimization": False,
        "hyperparameters": {
            "epochs": 10,
        },
        "module_path": "src.methods.baselines",
        "class_name": "CCSSelection",
    },
    "probabilistic": {
        "id": "probabilistic",
        "aliases": ["Probabilistic", "prob", "probabilistic_bilevel"],
        "name": "Probabilistic Bilevel Coreset Selection",
        "paper_section": "Section 5.2",
        "description": "Baseline method: probabilistic bilevel optimization",
        "type": "coreset_selection",
        "requires_inner_training": True,
        "requires_outer_optimization": True,
        "hyperparameters": {
            "epochs": 10,
            "T": 100,
        },
        "module_path": "src.methods.baselines",
        "class_name": "ProbabilisticSelection",
    },
    "oracle": {
        "id": "oracle",
        "aliases": ["Oracle", "full_dataset"],
        "name": "Oracle (Full Dataset)",
        "paper_section": "Section 5.2",
        "description": "Oracle baseline: training on full dataset",
        "type": "coreset_selection",
        "requires_inner_training": False,
        "requires_outer_optimization": False,
        "hyperparameters": {},
        "module_path": "src.methods.baselines",
        "class_name": "OracleSelection",
    },
}

METHOD_REGISTRY["ours"] = {**METHOD_REGISTRY["lbcs"], "id": "ours", "name": "LBCS (ours)"}
METHOD_REGISTRY["random"] = {**METHOD_REGISTRY["uniform"], "id": "random", "name": "Random Uniform Selection"}
METHOD_REGISTRY["baseline"] = {**METHOD_REGISTRY["uniform"], "id": "baseline", "name": "Baseline Uniform Selection"}
METHOD_REGISTRY["resnet"] = {
    "id": "resnet",
    "aliases": ["ResNet-18", "ResNet-50"],
    "name": "ResNet evaluation family",
    "paper_section": "Section 5",
    "description": "Model-family method surface for ResNet-based benchmark evaluation.",
    "type": "model_family",
    "requires_inner_training": True,
    "requires_outer_optimization": False,
    "hyperparameters": {"optimizer": "sgd_default"},
    "module_path": "src.methods.models",
    "class_name": "ResNet",
}
METHOD_REGISTRY["vit"] = {
    "id": "vit",
    "aliases": ["ViT", "vision_transformer"],
    "name": "Vision Transformer evaluation family",
    "paper_section": "Appendix",
    "description": "Optional model-family surface retained for benchmark compatibility.",
    "type": "model_family",
    "requires_inner_training": True,
    "requires_outer_optimization": False,
    "hyperparameters": {"optimizer": "adam_default"},
    "module_path": "src.method_registry",
    "class_name": "VisionTransformer",
}
METHOD_REGISTRY["adapter"] = {
    "id": "adapter",
    "aliases": ["adapter_head"],
    "name": "Adapter evaluation route",
    "paper_section": "Appendix",
    "description": "Adapter route surface for method/model compatibility checks.",
    "type": "adapter",
    "requires_inner_training": True,
    "requires_outer_optimization": False,
    "hyperparameters": {},
    "module_path": "src.method_registry",
    "class_name": "AdapterRoute",
}
METHOD_REGISTRY["fine_tuning"] = {
    "id": "fine_tuning",
    "aliases": ["finetune", "fine-tuning"],
    "name": "Fine-tuning evaluation route",
    "paper_section": "Appendix",
    "description": "Fine-tuning route surface for downstream benchmark evaluation.",
    "type": "training_route",
    "requires_inner_training": True,
    "requires_outer_optimization": False,
    "hyperparameters": {"optimizer": "sgd_default"},
    "module_path": "src.experiments.training",
    "class_name": "FineTuningRoute",
}

# ============================================================================
# Model Architecture Registry
# Paper evidence contract: expose method/baseline/variant adapters for ResNet-50,
# ResNet-18, ConvNet-3, VIT
# reference_grounding: paperbench_ref_003 train.py
# reference_grounding: paperbench_ref_004 cnn_mnist_probability_1step_pixel_shared_rein.py
# ============================================================================

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "resnet18": {
        "id": "resnet18",
        "aliases": ["ResNet-18", "resnet_18", "resnet"],
        "name": "ResNet-18",
        "paper_section": "Section 5",
        "description": "ResNet-18 architecture for CIFAR and F-MNIST",
        "input_size": 32,
        "num_layers": 18,
        "pretrained_available": False,
        "datasets": ["cifar10", "cifar100", "fmnist"],
        "module_path": "src.methods.models",
        "class_name": "ResNet18",
    },
    "resnet50": {
        "id": "resnet50",
        "aliases": ["ResNet-50", "resnet_50", "ResNet50"],
        "name": "ResNet-50",
        "paper_section": "Section 5.4",
        "description": "ResNet-50 architecture for ImageNet-1k",
        "input_size": 224,
        "num_layers": 50,
        "pretrained_available": True,
        "datasets": ["imagenet1k", "imagenet_1k"],
        "module_path": "src.methods.models",
        "class_name": "ResNet50",
    },
    "convnet3": {
        "id": "convnet3",
        "aliases": ["ConvNet-3", "convnet_3", "cnn3"],
        "name": "ConvNet-3",
        "paper_section": "Section 5",
        "description": "3-layer convolutional network",
        "input_size": 28,
        "num_layers": 3,
        "pretrained_available": False,
        "datasets": ["fmnist", "mnist"],
        "module_path": "src.methods.models",
        "class_name": "ConvNet3",
    },
    "vit": {
        "id": "vit",
        "aliases": ["ViT", "vision_transformer", "transformer"],
        "name": "Vision Transformer",
        "paper_section": "Appendix",
        "description": "Vision Transformer architecture (optional baseline)",
        "input_size": 224,
        "num_layers": 12,
        "pretrained_available": True,
        "datasets": ["imagenet1k"],
        "module_path": "src.methods.models",
        "class_name": "VisionTransformer",
    },
}

# ============================================================================
# Training Configuration Registry
# Binding addendum clarification: The ResNet18 is trained using an SGD optimizer
# with a learning rate of 0.1, momentum of 0.9, and a cosine scheduler.
# reference_grounding: paperbench_ref_003 train.py
# ============================================================================

OPTIMIZER_REGISTRY: Dict[str, Dict[str, Any]] = {
    "sgd_default": {
        "id": "sgd_default",
        "name": "SGD with Cosine Annealing",
        "description": "Default optimizer config from paper",
        "optimizer_type": "SGD",
        "optimizer_kwargs": {
            "lr": 0.1,
            "momentum": 0.9,
            "weight_decay": 5e-4,
            "nesterov": True,
        },
        "scheduler_type": "CosineAnnealingLR",
        "scheduler_kwargs": {
            "T_max": 200,
            "eta_min": 0.0,
        },
    },
    "adam_default": {
        "id": "adam_default",
        "name": "Adam",
        "description": "Adam optimizer for transformer models",
        "optimizer_type": "Adam",
        "optimizer_kwargs": {
            "lr": 0.001,
            "betas": (0.9, 0.999),
            "weight_decay": 0.0,
        },
        "scheduler_type": "StepLR",
        "scheduler_kwargs": {
            "step_size": 30,
            "gamma": 0.1,
        },
    },
}

# ============================================================================
# Training Adapter Registry
# Paper evidence contract: expose adapters for fine_tuning, adapter methods
# ============================================================================

ADAPTER_REGISTRY: Dict[str, Dict[str, Any]] = {
    "fine_tuning": {
        "id": "fine_tuning",
        "aliases": ["finetune", "full_finetune"],
        "name": "Full Fine-tuning",
        "description": "Standard full model fine-tuning",
        "type": "training_adapter",
        "trainable_params": "all",
        "module_path": "src.methods.agents",
        "class_name": "FullFineTuningAdapter",
    },
    "adapter": {
        "id": "adapter",
        "aliases": ["parameter_efficient", "peft"],
        "name": "Adapter-based Fine-tuning",
        "description": "Parameter-efficient adapter layers",
        "type": "training_adapter",
        "trainable_params": "adapter_only",
        "module_path": "src.methods.agents",
        "class_name": "AdapterFineTuningAdapter",
    },
}

# ============================================================================
# Registry Access Functions
# ============================================================================

def get_method(method_id: str) -> Dict[str, Any]:
    """
    Get method configuration by ID or alias.
    
    Paper evidence contract: support method selection for LBCS, random, baseline,
    oracle, and all baseline methods.
    
    Args:
        method_id: Method identifier or alias
        
    Returns:
        Method configuration dictionary
        
    Raises:
        KeyError: If method not found
    """
    method_id_lower = method_id.lower()
    
    # Direct lookup
    if method_id_lower in METHOD_REGISTRY:
        return METHOD_REGISTRY[method_id_lower]
    
    # Alias lookup
    for method_key, method_config in METHOD_REGISTRY.items():
        if method_id_lower in [alias.lower() for alias in method_config["aliases"]]:
            return method_config
    
    raise KeyError(f"Method '{method_id}' not found in registry")


def get_model(model_id: str) -> Dict[str, Any]:
    """
    Get model configuration by ID or alias.
    
    Paper evidence contract: support model selection for resnet, vit, ResNet-50,
    ConvNet-3.
    
    Args:
        model_id: Model identifier or alias
        
    Returns:
        Model configuration dictionary
        
    Raises:
        KeyError: If model not found
    """
    model_id_lower = model_id.lower()
    
    # Direct lookup
    if model_id_lower in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_id_lower]
    
    # Alias lookup
    for model_key, model_config in MODEL_REGISTRY.items():
        if model_id_lower in [alias.lower() for alias in model_config["aliases"]]:
            return model_config
    
    raise KeyError(f"Model '{model_id}' not found in registry")


def get_optimizer_config(optimizer_id: str) -> Dict[str, Any]:
    """
    Get optimizer configuration by ID.
    
    Args:
        optimizer_id: Optimizer configuration identifier
        
    Returns:
        Optimizer configuration dictionary
        
    Raises:
        KeyError: If optimizer not found
    """
    if optimizer_id not in OPTIMIZER_REGISTRY:
        raise KeyError(f"Optimizer config '{optimizer_id}' not found in registry")
    
    return OPTIMIZER_REGISTRY[optimizer_id]


def get_adapter(adapter_id: str) -> Dict[str, Any]:
    """
    Get adapter configuration by ID or alias.
    
    Paper evidence contract: support adapter selection for fine_tuning, adapter.
    
    Args:
        adapter_id: Adapter identifier or alias
        
    Returns:
        Adapter configuration dictionary
        
    Raises:
        KeyError: If adapter not found
    """
    adapter_id_lower = adapter_id.lower()
    
    # Direct lookup
    if adapter_id_lower in ADAPTER_REGISTRY:
        return ADAPTER_REGISTRY[adapter_id_lower]
    
    # Alias lookup
    for adapter_key, adapter_config in ADAPTER_REGISTRY.items():
        if adapter_id_lower in [alias.lower() for alias in adapter_config["aliases"]]:
            return adapter_config
    
    raise KeyError(f"Adapter '{adapter_id}' not found in registry")


def list_methods(method_type: Optional[str] = None) -> List[str]:
    """
    List all registered methods.
    
    Args:
        method_type: Optional filter by method type
        
    Returns:
        List of method IDs
    """
    if method_type is None:
        return list(METHOD_REGISTRY.keys())
    
    return [
        method_id
        for method_id, config in METHOD_REGISTRY.items()
        if config.get("type") == method_type
    ]


def list_models(dataset: Optional[str] = None) -> List[str]:
    """
    List all registered models.
    
    Args:
        dataset: Optional filter by compatible dataset
        
    Returns:
        List of model IDs
    """
    if dataset is None:
        return list(MODEL_REGISTRY.keys())
    
    dataset_lower = dataset.lower()
    return [
        model_id
        for model_id, config in MODEL_REGISTRY.items()
        if dataset_lower in [d.lower() for d in config.get("datasets", [])]
    ]


def validate_method_model_compatibility(
    method_id: str, model_id: str, dataset_id: str
) -> bool:
    """
    Validate compatibility between method, model, and dataset.
    
    Args:
        method_id: Method identifier
        model_id: Model identifier
        dataset_id: Dataset identifier
        
    Returns:
        True if compatible, False otherwise
    """
    try:
        method = get_method(method_id)
        model = get_model(model_id)
        
        # Check if model supports dataset
        dataset_lower = dataset_id.lower()
        model_datasets = [d.lower() for d in model.get("datasets", [])]
        
        if dataset_lower not in model_datasets:
            warnings.warn(
                f"Model '{model_id}' may not be compatible with dataset '{dataset_id}'"
            )
            return False
        
        return True
        
    except KeyError as e:
        warnings.warn(f"Compatibility check failed: {e}")
        return False


# ============================================================================
# Registry Metadata
# ============================================================================

REGISTRY_METADATA = {
    "version": "1.0.0",
    "paper_title": "Refined Coreset Selection: Towards Minimal Coreset Size under Model Performance Constraints",
    "num_methods": len(METHOD_REGISTRY),
    "num_models": len(MODEL_REGISTRY),
    "num_optimizers": len(OPTIMIZER_REGISTRY),
    "num_adapters": len(ADAPTER_REGISTRY),
    "supported_datasets": ["cifar10", "cifar100", "fmnist", "imagenet1k"],
    "core_method": "lbcs",
    "baseline_methods": [
        "uniform", "el2n", "grand", "influential", 
        "moderate", "ccs", "probabilistic", "oracle"
    ],
}


def get_registry_info() -> Dict[str, Any]:
    """
    Get registry metadata and summary.
    
    Returns:
        Registry metadata dictionary
    """
    return {
        **REGISTRY_METADATA,
        "methods": list(METHOD_REGISTRY.keys()),
        "models": list(MODEL_REGISTRY.keys()),
        "optimizers": list(OPTIMIZER_REGISTRY.keys()),
        "adapters": list(ADAPTER_REGISTRY.keys()),
    }
