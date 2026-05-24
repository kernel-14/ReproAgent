#!/usr/bin/env python3
"""
Method Registry for Test-Time Model Adaptation with Only Forward Passes.

Implements the complete method/baseline selector set as required by the paper evidence contract:
- ours, baseline, heuristic, vit, resnet, fine_tuning, test_time_adaptation, foa, lame, 
  t3a, tent, cotta, sar, cma_es, vision_mamba, clip, adapter

This registry exposes selectable method/baseline/variant adapters for all paper experiments.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple


# ==============================================================================
# Method Registry Structure
# ==============================================================================

METHOD_REGISTRY: Dict[str, Dict[str, Any]] = {}


def register_method(
    name: str,
    display_name: str,
    method_type: str,
    requires_gradients: bool,
    requires_source_data: bool,
    adaptation_type: str,
    implementation_module: str,
    parameters: Optional[Dict[str, Any]] = None,
    description: str = "",
    paper_reference: str = "",
    quantization_compatible: bool = True,
    memory_efficient: bool = True,
):
    """
    Register a method in the global method registry.
    
    Args:
        name: Unique identifier for the method
        display_name: Human-readable name
        method_type: Type category (test_time_adaptation, baseline, etc.)
        requires_gradients: Whether method needs gradient computation
        requires_source_data: Whether method needs source domain data
        adaptation_type: Adaptation strategy (forward_only, gradient_based, etc.)
        implementation_module: Python module path for implementation
        parameters: Default hyperparameters
        description: Method description
        paper_reference: Citation or paper reference
        quantization_compatible: Works with quantized models
        memory_efficient: Low memory footprint
    """
    METHOD_REGISTRY[name] = {
        "name": name,
        "display_name": display_name,
        "method_type": method_type,
        "requires_gradients": requires_gradients,
        "requires_source_data": requires_source_data,
        "adaptation_type": adaptation_type,
        "implementation_module": implementation_module,
        "parameters": parameters or {},
        "description": description,
        "paper_reference": paper_reference,
        "quantization_compatible": quantization_compatible,
        "memory_efficient": memory_efficient,
    }


def get_method(name: str) -> Dict[str, Any]:
    """Retrieve method configuration by name."""
    if name not in METHOD_REGISTRY:
        raise ValueError(
            f"Method '{name}' not found in registry. "
            f"Available: {list(METHOD_REGISTRY.keys())}"
        )
    return METHOD_REGISTRY[name].copy()


def list_methods(
    method_type: Optional[str] = None,
    requires_gradients: Optional[bool] = None,
    quantization_compatible: Optional[bool] = None,
) -> List[str]:
    """
    List registered methods with optional filtering.
    
    Args:
        method_type: Filter by method type
        requires_gradients: Filter by gradient requirement
        quantization_compatible: Filter by quantization compatibility
    
    Returns:
        List of method names matching criteria
    """
    methods = []
    for name, config in METHOD_REGISTRY.items():
        if method_type is not None and config["method_type"] != method_type:
            continue
        if requires_gradients is not None and config["requires_gradients"] != requires_gradients:
            continue
        if quantization_compatible is not None and config["quantization_compatible"] != quantization_compatible:
            continue
        methods.append(name)
    return methods


def get_method_factory(name: str) -> Callable:
    """
    Get factory function for instantiating a method.
    
    Uses lazy imports to avoid loading heavy dependencies at module import time.
    """
    config = get_method(name)
    module_path = config["implementation_module"]
    
    def factory(**kwargs):
        """Lazy factory that imports and instantiates the method."""
        # Merge default parameters with provided kwargs
        params = config["parameters"].copy()
        params.update(kwargs)
        
        # Import the implementation module
        parts = module_path.rsplit(".", 1)
        if len(parts) == 2:
            module_name, class_name = parts
            try:
                import importlib
                module = importlib.import_module(module_name)
                method_class = getattr(module, class_name)
                return method_class(**params)
            except ImportError as e:
                raise ImportError(
                    f"Failed to import {module_path}: {e}. "
                    f"Method '{name}' may require optional dependencies."
                )
        else:
            raise ValueError(f"Invalid module path: {module_path}")
    
    return factory


# ==============================================================================
# Method Registration - Complete Paper Evidence Contract Set
# ==============================================================================

def register_all_methods():
    """Register all methods required by the paper evidence contract."""
    
    # FOA - Our Forward-Only Adaptation Method
    register_method(
        name="foa",
        display_name="FOA (Forward-Only Adaptation)",
        method_type="test_time_adaptation",
        requires_gradients=False,
        requires_source_data=False,
        adaptation_type="forward_only",
        implementation_module="src.methods.FOAMethod",
        parameters={
            "population_size": 10,
            "prompt_count": 1,
            "adaptation_steps": 1,
            "mutation_rate": 0.1,
            "elite_fraction": 0.2,
            "lambda_act": 0.5,
        },
        description="Forward-Only Adaptation using CMA-ES prompt optimization and activation shifting",
        paper_reference="Test-Time Model Adaptation with Only Forward Passes",
        quantization_compatible=True,
        memory_efficient=True,
    )
    
    # Alias: ours
    register_method(
        name="ours",
        display_name="Ours (FOA)",
        method_type="test_time_adaptation",
        requires_gradients=False,
        requires_source_data=False,
        adaptation_type="forward_only",
        implementation_module="src.methods.FOAMethod",
        parameters={
            "population_size": 10,
            "prompt_count": 1,
            "adaptation_steps": 1,
            "mutation_rate": 0.1,
            "elite_fraction": 0.2,
            "lambda_act": 0.5,
        },
        description="Our proposed method (FOA)",
        paper_reference="Test-Time Model Adaptation with Only Forward Passes",
        quantization_compatible=True,
        memory_efficient=True,
    )
    
    # TENT - Test-Time Entropy Minimization
    register_method(
        name="tent",
        display_name="TENT",
        method_type="test_time_adaptation",
        requires_gradients=True,
        requires_source_data=False,
        adaptation_type="gradient_based",
        implementation_module="src.baselines.TENTBaseline",
        parameters={
            "learning_rate": 0.001,
            "adaptation_steps": 1,
        },
        description="Test-time entropy minimization",
        paper_reference="TENT: Fully Test-Time Adaptation by Entropy Minimization (ICLR 2021)",
        quantization_compatible=False,
        memory_efficient=False,
    )
    
    # CoTTA - Continual Test-Time Adaptation
    register_method(
        name="cotta",
        display_name="CoTTA",
        method_type="test_time_adaptation",
        requires_gradients=True,
        requires_source_data=True,
        adaptation_type="gradient_based",
        implementation_module="src.baselines.CoTTABaseline",
        parameters={
            "learning_rate": 0.001,
            "ema_decay": 0.999,
            "restoration_factor": 0.01,
        },
        description="Continual test-time adaptation with weight averaging and restoration",
        paper_reference="CoTTA: Continual Test-Time Adaptation (CVPR 2022)",
        quantization_compatible=False,
        memory_efficient=False,
    )
    
    # SAR - Sharpness-Aware and Reliable Adaptation
    register_method(
        name="sar",
        display_name="SAR",
        method_type="test_time_adaptation",
        requires_gradients=True,
        requires_source_data=False,
        adaptation_type="gradient_based",
        implementation_module="src.baselines.SARBaseline",
        parameters={
            "learning_rate": 0.001,
            "sharpness_weight": 0.05,
            "adaptation_steps": 1,
        },
        description="Sharpness-aware reliable test-time adaptation",
        paper_reference="SAR: Towards Robust Test-Time Adaptation (NeurIPS 2023)",
        quantization_compatible=False,
        memory_efficient=False,
    )
    
    # T3A - Test-Time Template Adjustments
    register_method(
        name="t3a",
        display_name="T3A",
        method_type="test_time_adaptation",
        requires_gradients=False,
        requires_source_data=True,
        adaptation_type="prototype_based",
        implementation_module="src.baselines.T3ABaseline",
        parameters={
            "num_prototypes": 10,
            "filter_k": 5,
        },
        description="Test-time template adjustments using prototypes",
        paper_reference="T3A: Improving Generalization (NeurIPS 2021)",
        quantization_compatible=True,
        memory_efficient=True,
    )
    
    # LAME - Lazy Marginalization over Experts
    register_method(
        name="lame",
        display_name="LAME",
        method_type="test_time_adaptation",
        requires_gradients=False,
        requires_source_data=True,
        adaptation_type="ensemble_based",
        implementation_module="src.baselines.LAMEBaseline",
        parameters={
            "num_experts": 5,
            "ensemble_strategy": "weighted_average",
            "source_sample_count": 1000,
        },
        description="Lazy marginalization over multiple expert predictions",
        paper_reference="LAME: Test-Time Ensemble (ICLR 2022)",
        quantization_compatible=True,
        memory_efficient=True,
    )
    
    # CMA-ES - Covariance Matrix Adaptation Evolution Strategy
    register_method(
        name="cma_es",
        display_name="CMA-ES",
        method_type="optimization",
        requires_gradients=False,
        requires_source_data=False,
        adaptation_type="evolutionary",
        implementation_module="src.methods.CMAESOptimizer",
        parameters={
            "population_size": 10,
            "sigma": 0.1,
            "max_iterations": 100,
        },
        description="Covariance Matrix Adaptation Evolution Strategy optimizer",
        paper_reference="https://github.com/CyberAgentAILab/cmaes",
        quantization_compatible=True,
        memory_efficient=True,
    )
    
    # Test-Time Adaptation (generic)
    register_method(
        name="test_time_adaptation",
        display_name="Test-Time Adaptation (Generic)",
        method_type="test_time_adaptation",
        requires_gradients=True,
        requires_source_data=False,
        adaptation_type="gradient_based",
        implementation_module="src.methods.GenericTTAMethod",
        parameters={
            "learning_rate": 0.001,
            "adaptation_steps": 1,
        },
        description="Generic test-time adaptation framework",
        paper_reference="Generic TTA",
        quantization_compatible=False,
        memory_efficient=False,
    )
    
    # Baseline - No adaptation
    register_method(
        name="baseline",
        display_name="Baseline (No Adaptation)",
        method_type="baseline",
        requires_gradients=False,
        requires_source_data=False,
        adaptation_type="none",
        implementation_module="src.baselines.NoAdaptationBaseline",
        parameters={},
        description="Baseline model without any test-time adaptation",
        paper_reference="Standard evaluation",
        quantization_compatible=True,
        memory_efficient=True,
    )
    
    # Heuristic baseline
    register_method(
        name="heuristic",
        display_name="Heuristic Baseline",
        method_type="baseline",
        requires_gradients=False,
        requires_source_data=False,
        adaptation_type="rule_based",
        implementation_module="src.baselines.HeuristicBaseline",
        parameters={
            "confidence_threshold": 0.8,
        },
        description="Simple heuristic-based adaptation",
        paper_reference="Heuristic baseline",
        quantization_compatible=True,
        memory_efficient=True,
    )
    
    # Fine-tuning
    register_method(
        name="fine_tuning",
        display_name="Fine-Tuning",
        method_type="training",
        requires_gradients=True,
        requires_source_data=True,
        adaptation_type="gradient_based",
        implementation_module="src.methods.FineTuningMethod",
        parameters={
            "learning_rate": 0.0001,
            "num_epochs": 10,
            "batch_size": 32,
        },
        description="Standard supervised fine-tuning",
        paper_reference="Standard fine-tuning",
        quantization_compatible=False,
        memory_efficient=False,
    )
    
    # Vision Transformer (ViT)
    register_method(
        name="vit",
        display_name="Vision Transformer (ViT)",
        method_type="model_architecture",
        requires_gradients=False,
        requires_source_data=False,
        adaptation_type="none",
        implementation_module="src.models.ViTModel",
        parameters={
            "model_name": "vit_base_patch16_224",
            "pretrained": True,
        },
        description="Vision Transformer base architecture",
        paper_reference="An Image is Worth 16x16 Words (ICLR 2021)",
        quantization_compatible=True,
        memory_efficient=True,
    )
    
    # ResNet
    register_method(
        name="resnet",
        display_name="ResNet",
        method_type="model_architecture",
        requires_gradients=False,
        requires_source_data=False,
        adaptation_type="none",
        implementation_module="src.models.ResNetModel",
        parameters={
            "model_name": "resnet50",
            "pretrained": True,
        },
        description="ResNet-50 architecture",
        paper_reference="Deep Residual Learning (CVPR 2016)",
        quantization_compatible=True,
        memory_efficient=True,
    )
    
    # Vision Mamba
    register_method(
        name="vision_mamba",
        display_name="Vision Mamba",
        method_type="model_architecture",
        requires_gradients=False,
        requires_source_data=False,
        adaptation_type="none",
        implementation_module="src.models.VisionMambaModel",
        parameters={
            "model_name": "vim_base",
            "pretrained": True,
        },
        description="Vision Mamba state-space model architecture",
        paper_reference="Vision Mamba (arXiv 2024)",
        quantization_compatible=True,
        memory_efficient=True,
    )
    
    # CLIP
    register_method(
        name="clip",
        display_name="CLIP",
        method_type="model_architecture",
        requires_gradients=False,
        requires_source_data=False,
        adaptation_type="none",
        implementation_module="src.models.CLIPModel",
        parameters={
            "model_name": "ViT-B/16",
            "pretrained": True,
        },
        description="CLIP vision-language model",
        paper_reference="Learning Transferable Visual Models (ICML 2021)",
        quantization_compatible=True,
        memory_efficient=True,
    )
    
    # Adapter-based methods
    register_method(
        name="adapter",
        display_name="Adapter",
        method_type="parameter_efficient",
        requires_gradients=True,
        requires_source_data=True,
        adaptation_type="gradient_based",
        implementation_module="src.methods.AdapterMethod",
        parameters={
            "adapter_dim": 64,
            "learning_rate": 0.001,
            "num_epochs": 10,
        },
        description="Parameter-efficient adaptation using adapters",
        paper_reference="Parameter-Efficient Transfer Learning (ICML 2019)",
        quantization_compatible=True,
        memory_efficient=True,
    )


# ==============================================================================
# Artifact Writing Functions
# ==============================================================================

def write_method_registry_artifacts(output_dir: str = "results", mode: str = "experiment"):
    """
    Write method registry artifacts for contract validation.
    
    Args:
        output_dir: Directory for artifacts
        mode: Execution mode (experiment, runtime_smoke, docker_validate)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Build method registry artifact
    registry_artifact = {
        "registry_name": "method_registry",
        "version": "1.0.0",
        "mode": mode,
        "total_methods": len(METHOD_REGISTRY),
        "methods": {},
        "categories": {
            "test_time_adaptation": [],
            "baseline": [],
            "training": [],
            "model_architecture": [],
            "optimization": [],
            "parameter_efficient": [],
        },
        "capabilities": {
            "gradient_free": [],
            "quantization_compatible": [],
            "memory_efficient": [],
            "requires_source": [],
        },
    }
    
    for name, config in METHOD_REGISTRY.items():
        registry_artifact["methods"][name] = config.copy()
        
        # Categorize
        method_type = config["method_type"]
        if method_type in registry_artifact["categories"]:
            registry_artifact["categories"][method_type].append(name)
        
        # Capabilities
        if not config["requires_gradients"]:
            registry_artifact["capabilities"]["gradient_free"].append(name)
        if config["quantization_compatible"]:
            registry_artifact["capabilities"]["quantization_compatible"].append(name)
        if config["memory_efficient"]:
            registry_artifact["capabilities"]["memory_efficient"].append(name)
        if config["requires_source_data"]:
            registry_artifact["capabilities"]["requires_source"].append(name)
    
    # Write method registry
    registry_path = output_path / "method_registry.json"
    with open(registry_path, "w") as f:
        json.dump(registry_artifact, f, indent=2)
    
    # Write evidence contract matrix contribution
    evidence_matrix_path = output_path / "evidence_contract_matrix.json"
    if evidence_matrix_path.exists():
        with open(evidence_matrix_path, "r") as f:
            evidence_data = json.load(f)
    else:
        evidence_data = {"experiments": [], "methods": {}, "mode": mode}
    
    evidence_data["methods"] = {
        name: {
            "display_name": config["display_name"],
            "type": config["method_type"],
            "requires_gradients": config["requires_gradients"],
            "adaptation_type": config["adaptation_type"],
        }
        for name, config in METHOD_REGISTRY.items()
    }
    
    with open(evidence_matrix_path, "w") as f:
        json.dump(evidence_data, f, indent=2)
    
    # Write metrics schema
    metrics_path = output_path / "metrics.json"
    metrics_data = {
        "mode": mode,
        "method_registry_metrics": {
            "total_methods": len(METHOD_REGISTRY),
            "gradient_free_count": len(registry_artifact["capabilities"]["gradient_free"]),
            "quantization_compatible_count": len(registry_artifact["capabilities"]["quantization_compatible"]),
            "memory_efficient_count": len(registry_artifact["capabilities"]["memory_efficient"]),
            "categories": {k: len(v) for k, v in registry_artifact["categories"].items()},
        },
    }
    
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            existing_metrics = json.load(f)
        existing_metrics.update(metrics_data)
        metrics_data = existing_metrics
    
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
    
    return {
        "method_registry": str(registry_path),
        "evidence_matrix": str(evidence_matrix_path),
        "metrics": str(metrics_path),
    }


def get_method_comparison_matrix() -> Dict[str, Any]:
    """
    Generate comparison matrix for all methods.
    
    Returns:
        Dictionary with method comparison data
    """
    comparison = {
        "dimensions": [
            "requires_gradients",
            "requires_source_data",
            "quantization_compatible",
            "memory_efficient",
        ],
        "methods": {},
    }
    
    for name, config in METHOD_REGISTRY.items():
        comparison["methods"][name] = {
            "requires_gradients": config["requires_gradients"],
            "requires_source_data": config["requires_source_data"],
            "quantization_compatible": config["quantization_compatible"],
            "memory_efficient": config["memory_efficient"],
            "adaptation_type": config["adaptation_type"],
            "method_type": config["method_type"],
        }
    
    return comparison


# ==============================================================================
# Initialization
# ==============================================================================

# Register all methods on module import
register_all_methods()


# ==============================================================================
# Smoke Test / Dry-Run Support
# ==============================================================================

def smoke_test_registry(output_dir: str = "results"):
    """
    Smoke test for method registry - validates registration and artifact writing.
    
    Args:
        output_dir: Directory for smoke test artifacts
    """
    print("=== Method Registry Smoke Test ===")
    print(f"Total registered methods: {len(METHOD_REGISTRY)}")
    
    # List methods by category
    for category in ["test_time_adaptation", "baseline", "model_architecture"]:
        methods = list_methods(method_type=category)
        print(f"{category}: {len(methods)} methods - {methods}")
    
    # List gradient-free methods
    gradient_free = list_methods(requires_gradients=False)
    print(f"Gradient-free methods: {len(gradient_free)} - {gradient_free}")
    
    # List quantization-compatible methods
    quant_compat = list_methods(quantization_compatible=True)
    print(f"Quantization-compatible: {len(quant_compat)} - {quant_compat}")
    
    # Write artifacts in smoke mode
    print("\nWriting method registry artifacts...")
    artifact_paths = write_method_registry_artifacts(output_dir, mode="runtime_smoke")
    for artifact_type, path in artifact_paths.items():
        print(f"  {artifact_type}: {path}")
    
    # Generate comparison matrix
    comparison = get_method_comparison_matrix()
    print(f"\nMethod comparison matrix generated: {len(comparison['methods'])} methods")
    
    print("\n=== Method Registry Smoke Test Complete ===")
    return True


if __name__ == "__main__":
    import sys
    
    mode = "runtime_smoke" if len(sys.argv) > 1 and sys.argv[1] == "--smoke" else "experiment"
    
    if mode == "runtime_smoke":
        smoke_test_registry()
    else:
        # Write artifacts for experiment mode
        artifact_paths = write_method_registry_artifacts(mode=mode)
        print("Method registry artifacts written:")
        for artifact_type, path in artifact_paths.items():
            print(f"  {artifact_type}: {path}")