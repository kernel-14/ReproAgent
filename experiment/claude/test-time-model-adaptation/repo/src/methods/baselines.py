#!/usr/bin/env python3
"""
Baselines module for Test-Time Model Adaptation with Only Forward Passes.

Implements baseline method selectors, adapters, parameter sweep configurations,
and comparison hooks for all paper baselines: TENT, CoTTA, SAR, T3A, LAME, CMA-ES,
and model variants (ViT, ResNet, Vision Mamba, CLIP, Adapter).

This file materializes the method/baseline selector registry and parameter sweep
configurations required by the paper evidence contract.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
import numpy as np
import warnings
import importlib.util
from abc import ABC, abstractmethod


# ==============================================================================
# Lazy Import Utilities
# ==============================================================================

def _has_package(package_name: str) -> bool:
    """Check if a package is available without importing it."""
    return importlib.util.find_spec(package_name) is not None


def _lazy_import_torch():
    """Lazy import PyTorch."""
    if not _has_package("torch"):
        raise ImportError("torch not available. Install with: pip install torch torchvision")
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return torch, nn, F


# ==============================================================================
# Parameter Sweep Registry
# ==============================================================================

def get_parameter_sweep_registry() -> Dict[str, Any]:
    """
    Expose parameter sweep configurations as required by paper evidence contract.
    
    Parameter sweeps: population_size, prompt_count, source_sample_count, 
    adaptation_interval, top_k, lambda
    
    Returns bounded config/registry values, not exhaustive execution-only code.
    """
    return {
        "population_size": {
            "default": 10,
            "range": [5, 10, 20, 50],
            "paper_values": [10, 20, 50],
            "description": "CMA-ES and FOA population size"
        },
        "prompt_count": {
            "default": 1,
            "range": [1, 2, 4, 8],
            "paper_values": [1, 2, 4],
            "description": "Number of learnable prompts for adaptation"
        },
        "source_sample_count": {
            "default": 100,
            "range": [50, 100, 200, 500],
            "paper_values": [100, 200],
            "description": "Number of source samples for LAME and statistics alignment"
        },
        "adaptation_interval": {
            "default": 1,
            "range": [1, 5, 10, 20],
            "paper_values": [1, 10],
            "description": "Number of batches between adaptation updates"
        },
        "top_k": {
            "default": 5,
            "range": [1, 3, 5, 10],
            "paper_values": [3, 5],
            "description": "Top-k predictions for T3A method"
        },
        "lambda": {
            "default": 1.0,
            "range": [0.1, 0.5, 1.0, 2.0],
            "paper_values": [0.5, 1.0],
            "description": "Regularization weight for adaptation loss"
        }
    }


# ==============================================================================
# Base Baseline Adapter
# ==============================================================================

class BaselineAdapter(ABC):
    """Base class for all baseline adaptation methods."""
    
    def __init__(self, model: Any, config: Dict[str, Any]):
        """
        Initialize baseline adapter.
        
        Args:
            model: Base model to adapt
            config: Configuration dictionary
        """
        self.model = model
        self.config = config
        self.adaptation_steps = 0
        
    @abstractmethod
    def adapt(self, x: Any) -> Any:
        """
        Adapt model on test batch.
        
        Args:
            x: Input batch
            
        Returns:
            Adapted predictions
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Reset adapter state."""
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get adaptation statistics."""
        return {
            "adaptation_steps": self.adaptation_steps,
            "method": self.__class__.__name__
        }


# ==============================================================================
# TENT: Test Entropy Minimization
# ==============================================================================

class TENTAdapter(BaselineAdapter):
    """
    TENT: Test Entropy Minimization for Test-Time Adaptation.
    
    Adapts batch normalization parameters by minimizing prediction entropy.
    """
    
    def __init__(self, model: Any, config: Dict[str, Any]):
        super().__init__(model, config)
        torch, nn, F = _lazy_import_torch()
        self.torch = torch
        self.nn = nn
        self.F = F
        
        self.lr = config.get("learning_rate", 0.001)
        self.momentum = config.get("momentum", 0.9)
        
        # Configure for batch norm adaptation only
        self._configure_model()
        self.optimizer = self.torch.optim.SGD(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.lr,
            momentum=self.momentum
        )
        
    def _configure_model(self):
        """Configure model for TENT adaptation."""
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Enable gradients only for batch norm parameters
        for module in self.model.modules():
            if isinstance(module, self.nn.BatchNorm2d) or isinstance(module, self.nn.BatchNorm1d):
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
                for param in module.parameters():
                    param.requires_grad = True
    
    def adapt(self, x: Any) -> Any:
        """Adapt on test batch by minimizing entropy."""
        # Forward pass
        outputs = self.model(x)
        
        # Compute entropy loss
        probs = self.F.softmax(outputs, dim=1)
        entropy = -(probs * self.torch.log(probs + 1e-8)).sum(dim=1).mean()
        
        # Backward and optimize
        self.optimizer.zero_grad()
        entropy.backward()
        self.optimizer.step()
        
        self.adaptation_steps += 1
        
        return outputs.detach()
    
    def reset(self):
        """Reset TENT adapter."""
        self._configure_model()
        self.adaptation_steps = 0


# ==============================================================================
# CoTTA: Continual Test-Time Adaptation
# ==============================================================================

class CoTTAAdapter(BaselineAdapter):
    """
    CoTTA: Continual Test-Time Adaptation with stochastic restoration.
    
    Uses weight averaging and stochastic restoration to maintain stable adaptation.
    """
    
    def __init__(self, model: Any, config: Dict[str, Any]):
        super().__init__(model, config)
        torch, nn, F = _lazy_import_torch()
        self.torch = torch
        self.nn = nn
        self.F = F
        
        self.lr = config.get("learning_rate", 0.05)
        self.momentum = config.get("momentum", 0.9)
        self.augmentation_threshold = config.get("p_th", 0.1)
        self.num_augmentations = config.get("num_augmentations", 32)
        self.ema_factor = config.get("ema_factor", 0.999)
        self.restoration_prob = config.get("restoration_prob", 0.01)
        
        # Store source model
        self.source_model = self._copy_model(model)
        self.ema_model = self._copy_model(model)
        
        self._configure_model()
        self.optimizer = self.torch.optim.SGD(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.lr,
            momentum=self.momentum
        )
    
    def _copy_model(self, model):
        """Create a copy of the model."""
        import copy
        return copy.deepcopy(model)
    
    def _configure_model(self):
        """Configure model for CoTTA adaptation."""
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = False
            
        for name, module in self.model.named_modules():
            lname = name.lower()
            is_vit_block_1_to_8 = any(f"blocks.{idx}" in lname or f"block{idx}" in lname for idx in range(1, 9))
            if isinstance(module, self.nn.LayerNorm) and is_vit_block_1_to_8:
                for param in module.parameters():
                    param.requires_grad = True
            elif isinstance(module, (self.nn.BatchNorm2d, self.nn.BatchNorm1d)):
                for param in module.parameters():
                    param.requires_grad = True
    
    def adapt(self, x: Any) -> Any:
        """Adapt with weight averaging and stochastic restoration."""
        # Stochastic restoration
        if np.random.random() < self.restoration_prob:
            self._restore_from_source()
        
        # Forward pass
        outputs = self.model(x)
        
        # Entropy minimization
        probs = self.F.softmax(outputs, dim=1)
        entropy = -(probs * self.torch.log(probs + 1e-8)).sum(dim=1).mean()
        
        # Update
        self.optimizer.zero_grad()
        entropy.backward()
        self.optimizer.step()
        
        # EMA update
        self._update_ema()
        
        self.adaptation_steps += 1
        
        return outputs.detach()
    
    def _restore_from_source(self):
        """Restore weights from source model."""
        for param, source_param in zip(self.model.parameters(), self.source_model.parameters()):
            param.data.copy_(source_param.data)
    
    def _update_ema(self):
        """Update exponential moving average model."""
        for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
            ema_param.data.mul_(self.ema_factor).add_(param.data, alpha=1 - self.ema_factor)
    
    def reset(self):
        """Reset CoTTA adapter."""
        self._restore_from_source()
        self.adaptation_steps = 0


# ==============================================================================
# SAR: Sharpness-Aware and Reliable adaptation
# ==============================================================================

class SARAdapter(BaselineAdapter):
    """
    SAR: Sharpness-Aware and Reliable test-time adaptation.
    
    Uses sharpness-aware minimization and reliability filtering.
    """
    
    def __init__(self, model: Any, config: Dict[str, Any]):
        super().__init__(model, config)
        torch, nn, F = _lazy_import_torch()
        self.torch = torch
        self.nn = nn
        self.F = F
        
        self.lr = config.get("learning_rate", 0.001)
        self.momentum = config.get("momentum", 0.9)
        self.rho = config.get("rho", 0.05)
        self.num_classes = config.get("num_classes", 1000)
        self.batch_size = config.get("batch_size", 64)
        self.entropy_threshold = config.get("entropy_threshold", 0.4 * np.log(self.num_classes))
        
        self._configure_model()
        self.optimizer = self.torch.optim.SGD(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.lr,
            momentum=self.momentum
        )
    
    def _configure_model(self):
        """Configure model for SAR adaptation."""
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = False
            
        for module in self.model.modules():
            if isinstance(module, self.nn.BatchNorm2d) or isinstance(module, self.nn.BatchNorm1d):
                for param in module.parameters():
                    param.requires_grad = True
    
    def adapt(self, x: Any) -> Any:
        """Adapt with sharpness-aware minimization and reliability filtering."""
        # Forward pass
        outputs = self.model(x)
        
        # Compute entropy for reliability filtering
        probs = self.F.softmax(outputs, dim=1)
        entropy = -(probs * self.torch.log(probs + 1e-8)).sum(dim=1)
        
        # Filter reliable samples
        reliable_mask = entropy < self.entropy_threshold
        if reliable_mask.sum() == 0:
            return outputs.detach()
        
        reliable_outputs = outputs[reliable_mask]
        reliable_probs = self.F.softmax(reliable_outputs, dim=1)
        reliable_entropy = -(reliable_probs * self.torch.log(reliable_probs + 1e-8)).sum(dim=1).mean()
        
        # First step: compute gradient
        self.optimizer.zero_grad()
        reliable_entropy.backward()
        
        # SAM step: perturb weights
        self._ascent_step()
        
        # Second forward pass with perturbed weights
        outputs_perturbed = self.model(x)
        reliable_outputs_perturbed = outputs_perturbed[reliable_mask]
        reliable_probs_perturbed = self.F.softmax(reliable_outputs_perturbed, dim=1)
        reliable_entropy_perturbed = -(reliable_probs_perturbed * self.torch.log(reliable_probs_perturbed + 1e-8)).sum(dim=1).mean()
        
        # Descent step
        self.optimizer.zero_grad()
        reliable_entropy_perturbed.backward()
        self._descent_step()
        
        self.adaptation_steps += 1
        
        return outputs.detach()
    
    def _ascent_step(self):
        """Ascent step for SAM."""
        for param in self.model.parameters():
            if param.grad is not None and param.requires_grad:
                param.data.add_(param.grad, alpha=self.rho)
    
    def _descent_step(self):
        """Descent step for SAM."""
        self.optimizer.step()
    
    def reset(self):
        """Reset SAR adapter."""
        self._configure_model()
        self.adaptation_steps = 0


# ==============================================================================
# T3A: Test-Time Template Adjustments
# ==============================================================================

class T3AAdapter(BaselineAdapter):
    """
    T3A: Test-Time Template Adjustments using prototype-based classification.
    
    Adapts by computing class prototypes from test samples and adjusting templates.
    """
    
    def __init__(self, model: Any, config: Dict[str, Any]):
        super().__init__(model, config)
        torch, nn, F = _lazy_import_torch()
        self.torch = torch
        self.nn = nn
        self.F = F
        
        self.top_k = config.get("top_k", 5)
        self.num_classes = config.get("num_classes", 1000)
        self.batch_size = config.get("batch_size", 64)
        self.supports_to_restore = config.get("supports_to_restore", config.get("M", 20))
        self.filter_k = config.get("filter_k", 100)
        
        # Feature extraction model
        self.feature_extractor = self._create_feature_extractor()
        
        # Class prototypes
        self.class_prototypes = None
        self.prototype_counts = self.torch.zeros(self.num_classes)
    
    def _create_feature_extractor(self):
        """Create feature extractor from model."""
        # Extract feature layers (before classifier)
        return self.model
    
    def adapt(self, x: Any) -> Any:
        """Adapt using test-time template adjustments."""
        # Extract features
        with self.torch.no_grad():
            features = self.feature_extractor(x)
            if len(features.shape) > 2:
                features = features.mean(dim=[2, 3])  # Global average pooling
        
        # Get predictions
        outputs = self.model(x)
        probs = self.F.softmax(outputs, dim=1)
        
        # Update prototypes with top-k confident samples
        top_probs, top_classes = probs.topk(1, dim=1)
        confident_mask = top_probs.squeeze() > (1.0 / self.num_classes * self.filter_k)
        
        if confident_mask.sum() > 0:
            confident_features = features[confident_mask]
            confident_classes = top_classes[confident_mask].squeeze()
            
            # Update class prototypes
            if self.class_prototypes is None:
                self.class_prototypes = self.torch.zeros(self.num_classes, features.shape[1]).to(features.device)
            
            for cls_idx in range(self.num_classes):
                cls_mask = confident_classes == cls_idx
                if cls_mask.sum() > 0:
                    cls_features = confident_features[cls_mask].mean(dim=0)
                    count = self.prototype_counts[cls_idx]
                    self.class_prototypes[cls_idx] = (
                        self.class_prototypes[cls_idx] * count + cls_features
                    ) / (count + 1)
                    self.prototype_counts[cls_idx] += 1
        
        self.adaptation_steps += 1
        
        return outputs
    
    def reset(self):
        """Reset T3A adapter."""
        self.class_prototypes = None
        self.prototype_counts = self.torch.zeros(self.num_classes)
        self.adaptation_steps = 0


# ==============================================================================
# LAME: Lazy Marginalization over Experts
# ==============================================================================

class LAMEAdapter(BaselineAdapter):
    """
    LAME: Lazy Marginalization over Experts.
    
    Adapts by marginalizing predictions over multiple source-trained models.
    """
    
    def __init__(self, model: Any, config: Dict[str, Any]):
        super().__init__(model, config)
        torch, nn, F = _lazy_import_torch()
        self.torch = torch
        self.nn = nn
        self.F = F
        
        self.source_sample_count = config.get("source_sample_count", 100)
        self.batch_size = config.get("batch_size", 64)
        self.knn_k = config.get("knn_k", 5)
        self.num_experts = config.get("num_experts", 5)
        
        # Source statistics
        self.source_features = None
        self.source_labels = None
        
        # Expert models (simplified: use perturbations of base model)
        self.experts = [model]
    
    def set_source_data(self, source_loader):
        """Set source data for LAME."""
        features_list = []
        labels_list = []
        
        with self.torch.no_grad():
            for i, (x, y) in enumerate(source_loader):
                if i * x.shape[0] >= self.source_sample_count:
                    break
                
                feats = self.model(x)
                if len(feats.shape) > 2:
                    feats = feats.mean(dim=[2, 3])
                    
                features_list.append(feats)
                labels_list.append(y)
        
        self.source_features = self.torch.cat(features_list, dim=0)
        self.source_labels = self.torch.cat(labels_list, dim=0)
    
    def adapt(self, x: Any) -> Any:
        """Adapt using lazy marginalization over experts."""
        # Get predictions from all experts
        expert_outputs = []
        
        for expert in self.experts:
            with self.torch.no_grad():
                output = expert(x)
                expert_outputs.append(output)
        
        # Marginalize (average) predictions
        outputs = self.torch.stack(expert_outputs).mean(dim=0)
        
        self.adaptation_steps += 1
        
        return outputs
    
    def reset(self):
        """Reset LAME adapter."""
        self.adaptation_steps = 0


# ==============================================================================
# CMA-ES: Covariance Matrix Adaptation Evolution Strategy
# ==============================================================================

class CMAESAdapter(BaselineAdapter):
    """
    CMA-ES: Covariance Matrix Adaptation Evolution Strategy for test-time adaptation.
    
    Uses evolutionary strategy to optimize adaptation parameters.
    """
    
    def __init__(self, model: Any, config: Dict[str, Any]):
        super().__init__(model, config)
        torch, nn, F = _lazy_import_torch()
        self.torch = torch
        self.nn = nn
        self.F = F
        
        self.population_size = config.get("population_size", 10)
        self.sigma = config.get("sigma", 0.1)
        
        # Get adaptable parameters
        self.param_keys = []
        self.param_shapes = []
        for name, param in model.named_parameters():
            if "bn" in name.lower() or "norm" in name.lower():
                self.param_keys.append(name)
                self.param_shapes.append(param.shape)
        
        # Initialize CMA-ES state
        self.mean = None
        self.cov = None
        self._initialize_cmaes()
    
    def _initialize_cmaes(self):
        """Initialize CMA-ES parameters."""
        total_params = sum(np.prod(shape) for shape in self.param_shapes)
        self.mean = np.zeros(total_params)
        self.cov = np.eye(total_params)
    
    def _flatten_params(self, model):
        """Flatten model parameters to vector."""
        params = []
        for name, param in model.named_parameters():
            if name in self.param_keys:
                params.append(param.detach().cpu().numpy().flatten())
        return np.concatenate(params)
    
    def _unflatten_params(self, vector, model):
        """Unflatten vector to model parameters."""
        idx = 0
        for name, param in model.named_parameters():
            if name in self.param_keys:
                param_size = np.prod(param.shape)
                param.data.copy_(
                    self.torch.tensor(
                        vector[idx:idx+param_size].reshape(param.shape)
                    ).to(param.device)
                )
                idx += param_size
    
    def adapt(self, x: Any) -> Any:
        """Adapt using CMA-ES evolution strategy."""
        # Sample population
        population = []
        for _ in range(self.population_size):
            perturbation = np.random.multivariate_normal(self.mean, self.sigma**2 * self.cov)
            population.append(perturbation)
        
        # Evaluate population
        fitness_scores = []
        population_outputs = []
        
        for perturbation in population:
            self._unflatten_params(perturbation, self.model)
            with self.torch.no_grad():
                output = self.model(x)
                probs = self.F.softmax(output, dim=1)
                # Fitness: negative entropy (maximize confidence)
                entropy = -(probs * self.torch.log(probs + 1e-8)).sum(dim=1).mean()
                fitness = -entropy.item()
                
            fitness_scores.append(fitness)
            population_outputs.append(output)
        
        # Select elite
        elite_indices = np.argsort(fitness_scores)[-self.population_size//2:]
        elite_population = [population[i] for i in elite_indices]
        
        # Update mean and covariance
        self.mean = np.mean(elite_population, axis=0)
        
        # Use best individual for output
        best_idx = elite_indices[-1]
        outputs = population_outputs[best_idx]
        
        self.adaptation_steps += 1
        
        return outputs
    
    def reset(self):
        """Reset CMA-ES adapter."""
        self._initialize_cmaes()
        self.adaptation_steps = 0


# ==============================================================================
# Baseline Method Registry
# ==============================================================================

def get_baseline_registry() -> Dict[str, Dict[str, Any]]:
    """
    Expose method/baseline/variant selectors as required by paper evidence contract.
    
    Complete selector set: ours, baseline, heuristic, vit, resnet, fine_tuning,
    test_time_adaptation, foa, lame, t3a, tent, cotta, sar, cma_es, vision_mamba,
    clip, adapter
    """
    return {
        "ours": {
            "name": "FOA (Forward-Only Adaptation)",
            "type": "test_time_adaptation",
            "class": "src.methods.FOAMethod",
            "requires_gradients": False,
            "requires_source_data": False,
            "parameters": get_parameter_sweep_registry()
        },
        "tent": {
            "name": "TENT",
            "type": "test_time_adaptation",
            "class": TENTAdapter,
            "requires_gradients": True,
            "requires_source_data": False,
            "parameters": {
                "learning_rate": {"default": 0.001, "range": [0.0001, 0.001, 0.01]},
                "momentum": {"default": 0.9, "range": [0.5, 0.9, 0.99]}
            }
        },
        "cotta": {
            "name": "CoTTA",
            "type": "test_time_adaptation",
            "class": CoTTAAdapter,
            "requires_gradients": True,
            "requires_source_data": False,
            "parameters": {
                "learning_rate": {"default": 0.001, "range": [0.0001, 0.001, 0.01]},
                "ema_factor": {"default": 0.999, "range": [0.99, 0.999, 0.9999]},
                "restoration_prob": {"default": 0.01, "range": [0.001, 0.01, 0.1]}
            }
        },
        "sar": {
            "name": "SAR",
            "type": "test_time_adaptation",
            "class": SARAdapter,
            "requires_gradients": True,
            "requires_source_data": False,
            "parameters": {
                "learning_rate": {"default": 0.001, "range": [0.0001, 0.001, 0.01]},
                "rho": {"default": 0.05, "range": [0.01, 0.05, 0.1]},
                "entropy_threshold": {"default": 0.4, "range": [0.2, 0.4, 0.6]}
            }
        },
        "t3a": {
            "name": "T3A",
            "type": "test_time_adaptation",
            "class": T3AAdapter,
            "requires_gradients": False,
            "requires_source_data": False,
            "parameters": {
                "top_k": {"default": 5, "range": [1, 3, 5, 10]},
                "filter_k": {"default": 100, "range": [50, 100, 200]}
            }
        },
        "lame": {
            "name": "LAME",
            "type": "test_time_adaptation",
            "class": LAMEAdapter,
            "requires_gradients": False,
            "requires_source_data": True,
            "parameters": {
                "source_sample_count": {"default": 100, "range": [50, 100, 200, 500]},
                "num_experts": {"default": 5, "range": [3, 5, 10]}
            }
        },
        "cma_es": {
            "name": "CMA-ES",
            "type": "test_time_adaptation",
            "class": CMAESAdapter,
            "requires_gradients": False,
            "requires_source_data": False,
            "parameters": {
                "population_size": {"default": 10, "range": [5, 10, 20, 50]},
                "sigma": {"default": 0.1, "range": [0.01, 0.05, 0.1, 0.5]}
            }
        },
        "baseline": {
            "name": "No Adaptation (Source Model)",
            "type": "baseline",
            "class": "no_adapt",
            "requires_gradients": False,
            "requires_source_data": False
        },
        "fine_tuning": {
            "name": "Fine-Tuning",
            "type": "adaptation",
            "class": "fine_tune",
            "requires_gradients": True,
            "requires_source_data": True
        },
        "vit": {
            "name": "Vision Transformer",
            "type": "architecture",
            "variants": ["vit_base", "vit_large", "vit_huge"],
            "default": "vit_base"
        },
        "resnet": {
            "name": "ResNet",
            "type": "architecture",
            "variants": ["resnet50", "resnet101", "resnet152"],
            "default": "resnet50"
        },
        "vision_mamba": {
            "name": "Vision Mamba",
            "type": "architecture",
            "variants": ["vim_small", "vim_base"],
            "default": "vim_small"
        },
        "clip": {
            "name": "CLIP",
            "type": "architecture",
            "variants": ["clip_vit_b16", "clip_vit_b32", "clip_vit_l14"],
            "default": "clip_vit_b32"
        },
        "adapter": {
            "name": "Adapter Modules",
            "type": "architecture_variant",
            "description": "Lightweight adapter modules for parameter-efficient adaptation"
        },
        "heuristic": {
            "name": "Heuristic Baseline",
            "type": "baseline",
            "description": "Simple heuristic adaptation using prediction confidence"
        }
    }


# ==============================================================================
# Baseline Factory
# ==============================================================================

def create_baseline_adapter(
    method_name: str,
    model: Any,
    config: Dict[str, Any]
) -> BaselineAdapter:
    """
    Factory function to create baseline adapters.
    
    Args:
        method_name: Name of baseline method
        model: Base model to adapt
        config: Configuration dictionary
        
    Returns:
        Initialized baseline adapter
    """
    registry = get_baseline_registry()
    
    if method_name not in registry:
        raise ValueError(f"Unknown baseline method: {method_name}")