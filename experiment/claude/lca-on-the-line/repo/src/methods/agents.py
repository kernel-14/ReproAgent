#!/usr/bin/env python3
"""
LCA-on-the-Line Agent/Model Adapter Module

Provides unified Agent interface for Vision Models (VMs) and Vision-Language Models (VLMs),
training methods with soft labels and LCA-based losses, and method/baseline selectors.

Paper-derived method registry:
- ours: LCA-aware soft label training
- baseline: Standard cross-entropy training
- resnet: ResNet family (18, 34, 50, 101, 152)
- vit: Vision Transformer family
- adapter: Linear adapter fine-tuning
- fine_tuning: Full model fine-tuning

reference_grounding: paperbench_ref_005 eval_many_models.py
reference_grounding: paperbench_ref_006 extract_clip.ipynb
reference_grounding: paperbench_ref_006 eval_tiny_imagenet_truncate.ipynb
reference_grounding: paperbench_ref_001 references/classification/README.md

Binding addendum clarifications:
- All vision-language models accessed via OpenCLIP and CLIP modules
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import warnings
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


# =============================================================================
# Parameter Sweep Registry - Paper-derived bounded configurations
# =============================================================================

PARAMETER_SWEEPS = {
    "clustering": {
        "num_layers": [2, 3],
        "root_clusters": [10, 20],
        "leaf_clusters": [100, 200],
        "layer_configs": [
            {"layers": 2, "clusters": [10, 100]},
            {"layers": 2, "clusters": [20, 100]},
            {"layers": 3, "clusters": [10, 50, 100]},
        ]
    },
    "soft_labels": {
        "temperature": [0.5, 1.0, 2.0, 4.0],
        "lca_loss_weight": [0.1, 0.5, 1.0],
        "default_temperature": 1.0,
        "default_lca_weight": 0.5
    },
    "training": {
        "learning_rate": [0.001, 0.01, 0.1],
        "batch_size": [32, 64, 128, 256],
        "epochs": [10, 30, 90],
        "optimizer": ["sgd", "adam", "adamw"],
        "default_lr": 0.01,
        "default_batch_size": 128,
        "default_epochs": 30
    }
}


# =============================================================================
# Agent Base Class
# =============================================================================

class Agent(ABC):
    """
    Abstract base class for model agents supporting evaluation and training.
    
    All agents must implement:
    - predict: Forward pass for inference
    - evaluate: Compute metrics on a dataset
    - get_features: Extract feature representations (optional for some agents)
    """
    
    def __init__(self, model_name: str, num_classes: int = 1000, device: str = "cpu"):
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = device
        self.model = None
        self.is_trained = False
        
    @abstractmethod
    def predict(self, images: Any, return_features: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate predictions for input images.
        
        Args:
            images: Input images (batch)
            return_features: If True, also return feature representations
            
        Returns:
            predictions: Class probabilities or logits [batch_size, num_classes]
            features (optional): Feature vectors [batch_size, feature_dim]
        """
        pass
    
    @abstractmethod
    def evaluate(self, dataloader: Any, hierarchy: Optional[Any] = None) -> Dict[str, float]:
        """
        Evaluate model on a dataset.
        
        Args:
            dataloader: Data loader providing (images, labels) batches
            hierarchy: Optional class hierarchy for LCA distance computation
            
        Returns:
            metrics: Dictionary with top1_acc, top5_acc, lca_distance, etc.
        """
        pass
    
    def get_features(self, images: Any) -> np.ndarray:
        """
        Extract feature representations (optional, not all agents support this).
        
        Args:
            images: Input images
            
        Returns:
            features: Feature vectors [batch_size, feature_dim]
        """
        raise NotImplementedError(f"Feature extraction not implemented for {self.__class__.__name__}")


# =============================================================================
# Vision Model Agent - reference_grounding: paperbench_ref_005 eval_many_models.py
# =============================================================================

class VisionModelAgent(Agent):
    """
    Agent for standard vision models (ResNet, ViT, EfficientNet, etc.).
    
    Supports:
    - Pretrained model loading from torchvision/timm
    - Standard classification inference
    - Fine-tuning and adapter training
    - Soft label training with LCA-based losses
    
    reference_grounding: paperbench_ref_005 eval_many_models.py
    reference_grounding: paperbench_ref_001 references/classification/README.md
    """
    
    def __init__(self, model_name: str, num_classes: int = 1000, 
                 pretrained: bool = True, device: str = "cpu"):
        super().__init__(model_name, num_classes, device)
        self.pretrained = pretrained
        self._load_model()
    
    def _load_model(self):
        """Load model from torchvision or timm."""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError("torch required for VisionModelAgent")
        
        # Try torchvision first
        try:
            import torchvision.models as models
            if hasattr(models, self.model_name):
                if self.pretrained:
                    weights = "DEFAULT"
                else:
                    weights = None
                self.model = getattr(models, self.model_name)(weights=weights)
                logger.info(f"Loaded {self.model_name} from torchvision")
            else:
                raise AttributeError(f"Model {self.model_name} not found in torchvision")
        except (ImportError, AttributeError):
            # Fall back to timm
            try:
                import timm
                self.model = timm.create_model(self.model_name, pretrained=self.pretrained, num_classes=self.num_classes)
                logger.info(f"Loaded {self.model_name} from timm")
            except ImportError:
                raise ImportError(f"Could not load {self.model_name} from torchvision or timm")
        
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def predict(self, images: Any, return_features: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate predictions for input images.
        
        reference_grounding: paperbench_ref_005 eval_many_models.py
        """
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            raise ImportError("torch required for prediction")
        
        self.model.eval()
        with torch.no_grad():
            if not isinstance(images, torch.Tensor):
                images = torch.tensor(images)
            images = images.to(self.device)
            
            if return_features:
                # Extract features from penultimate layer
                features = self._extract_features(images)
                output = self.model(images)
                probs = F.softmax(output, dim=1).cpu().numpy()
                return probs, features.cpu().numpy()
            else:
                output = self.model(images)
                probs = F.softmax(output, dim=1).cpu().numpy()
                return probs
    
    def _extract_features(self, images: Any) -> Any:
        """Extract features from penultimate layer."""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError("torch required for feature extraction")
        
        # Hook to capture features
        features = []
        def hook_fn(module, input, output):
            features.append(output)
        
        # Register hook on the layer before final classifier
        if hasattr(self.model, 'fc'):
            handle = self.model.fc.register_forward_hook(hook_fn)
        elif hasattr(self.model, 'head'):
            handle = self.model.head.register_forward_hook(hook_fn)
        elif hasattr(self.model, 'classifier'):
            handle = self.model.classifier.register_forward_hook(hook_fn)
        else:
            # Fallback: use the model output itself
            return self.model(images)
        
        # Forward pass
        _ = self.model(images)
        handle.remove()
        
        if len(features) > 0:
            return features[0]
        else:
            return self.model(images)
    
    def evaluate(self, dataloader: Any, hierarchy: Optional[Any] = None) -> Dict[str, float]:
        """
        Evaluate model on dataset with Top-1, Top-5, and optional LCA metrics.
        
        reference_grounding: paperbench_ref_005 eval_many_models.py
        """
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            raise ImportError("torch required for evaluation")
        
        self.model.eval()
        
        correct_top1 = 0
        correct_top5 = 0
        total = 0
        all_predictions = []
        all_labels = []
        lca_distances = []
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataloader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                output = self.model(images)
                probs = F.softmax(output, dim=1)
                
                # Top-1 accuracy
                _, pred_top1 = output.max(1)
                correct_top1 += pred_top1.eq(labels).sum().item()
                
                # Top-5 accuracy
                _, pred_top5 = output.topk(5, 1, True, True)
                pred_top5 = pred_top5.t()
                correct_top5 += pred_top5.eq(labels.view(1, -1).expand_as(pred_top5)).sum().item()
                
                total += labels.size(0)
                
                # Store for LCA computation
                all_predictions.extend(pred_top1.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
                
                # Compute LCA distances if hierarchy provided
                if hierarchy is not None:
                    for i in range(len(labels)):
                        pred_class = pred_top1[i].item()
                        true_class = labels[i].item()
                        lca_dist = hierarchy.compute_lca_distance(pred_class, true_class)
                        lca_distances.append(lca_dist)
        
        metrics = {
            "top1_accuracy": correct_top1 / total if total > 0 else 0.0,
            "top5_accuracy": correct_top5 / total if total > 0 else 0.0,
            "num_samples": total,
        }
        
        if len(lca_distances) > 0:
            metrics["lca_distance"] = float(np.mean(lca_distances))
            metrics["lca_distance_std"] = float(np.std(lca_distances))
        
        return metrics
    
    def train_with_soft_labels(self, train_loader: Any, hierarchy: Any, 
                               num_epochs: int = 30, learning_rate: float = 0.01,
                               temperature: float = 1.0, lca_loss_weight: float = 0.5) -> Dict[str, List[float]]:
        """
        Train model with soft labels derived from class hierarchy.
        
        Paper method: "Hierarchy-Aware Training: Soft labeling methods using 
        WordNet and latent hierarchies to improve OOD generalization"
        
        Args:
            train_loader: Training data loader
            hierarchy: Class hierarchy for soft label generation
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            temperature: Temperature for soft label smoothing
            lca_loss_weight: Weight for LCA-based loss term
            
        Returns:
            training_history: Dictionary with loss curves and metrics
        """
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            import torch.optim as optim
        except ImportError:
            raise ImportError("torch required for training")
        
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
        
        history = {
            "train_loss": [],
            "train_accuracy": [],
            "epoch_metrics": []
        }
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                output = self.model(images)
                
                # Generate soft labels based on hierarchy
                soft_labels = self._generate_soft_labels(labels, hierarchy, temperature)
                soft_labels = soft_labels.to(self.device)
                
                # Combined loss: cross-entropy + LCA-aware soft label loss
                ce_loss = F.cross_entropy(output, labels)
                soft_loss = self._soft_label_loss(output, soft_labels, temperature)
                loss = (1 - lca_loss_weight) * ce_loss + lca_loss_weight * soft_loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = output.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            # Epoch metrics
            avg_loss = epoch_loss / len(train_loader)
            accuracy = correct / total if total > 0 else 0.0
            
            history["train_loss"].append(avg_loss)
            history["train_accuracy"].append(accuracy)
            history["epoch_metrics"].append({
                "epoch": epoch + 1,
                "loss": avg_loss,
                "accuracy": accuracy
            })
            
            logger.info(f"Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.4f}, Acc={accuracy:.4f}")
        
        self.is_trained = True
        return history
    
    def _generate_soft_labels(self, labels: Any, hierarchy: Any, temperature: float) -> Any:
        """
        Generate soft labels based on class hierarchy.
        
        For each true label, assign probability mass to nearby classes
        in the hierarchy based on their LCA distance.
        """
        try:
            import torch
        except ImportError:
            raise ImportError("torch required for soft label generation")
        
        batch_size = labels.size(0)
        soft_labels = torch.zeros(batch_size, self.num_classes, device=labels.device)
        
        for i in range(batch_size):
            true_class = labels[i].item()
            
            # Compute distances to all classes
            distances = []
            for c in range(self.num_classes):
                dist = hierarchy.compute_lca_distance(true_class, c)
                distances.append(dist)
            
            # Convert distances to probabilities (closer = higher prob)
            distances = np.array(distances)
            max_dist = distances.max()
            if max_dist > 0:
                similarities = 1.0 - (distances / max_dist)
            else:
                similarities = np.ones_like(distances)
            
            # Apply temperature and softmax
            logits = similarities / temperature
            probs = np.exp(logits - logits.max())
            probs = probs / probs.sum()
            
            soft_labels[i] = torch.tensor(probs, dtype=torch.float32, device=labels.device)
        
        return soft_labels
    
    def _soft_label_loss(self, output: Any, soft_labels: Any, temperature: float) -> Any:
        """
        Compute soft label loss (KL divergence between output and soft labels).
        """
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            raise ImportError("torch required for loss computation")
        
        log_probs = F.log_softmax(output / temperature, dim=1)
        loss = F.kl_div(log_probs, soft_labels, reduction='batchmean')
        return loss
    
    def fine_tune(self, train_loader: Any, num_epochs: int = 10, 
                  learning_rate: float = 0.001, freeze_backbone: bool = False) -> Dict[str, List[float]]:
        """
        Fine-tune model on new data.
        
        Args:
            train_loader: Training data loader
            num_epochs: Number of epochs
            learning_rate: Learning rate
            freeze_backbone: If True, only train final layer
            
        Returns:
            training_history: Loss and accuracy curves
        """
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            import torch.optim as optim
        except ImportError:
            raise ImportError("torch required for fine-tuning")
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            # Unfreeze final layer
            if hasattr(self.model, 'fc'):
                for param in self.model.fc.parameters():
                    param.requires_grad = True
            elif hasattr(self.model, 'head'):
                for param in self.model.head.parameters():
                    param.requires_grad = True
        
        self.model.train()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        history = {
            "train_loss": [],
            "train_accuracy": []
        }
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = output.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            avg_loss = epoch_loss / len(train_loader)
            accuracy = correct / total if total > 0 else 0.0
            
            history["train_loss"].append(avg_loss)
            history["train_accuracy"].append(accuracy)
            
            logger.info(f"Fine-tune Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.4f}, Acc={accuracy:.4f}")
        
        self.is_trained = True
        return history


# =============================================================================
# Vision-Language Model Agent - reference_grounding: paperbench_ref_006 extract_clip.ipynb
# =============================================================================

class VisionLanguageModelAgent(Agent):
    """
    Agent for vision-language models (CLIP, OpenCLIP).
    
    Supports:
    - Zero-shot classification with text prompts
    - Hierarchy-aware prompt engineering
    - Feature extraction for clustering
    
    reference_grounding: paperbench_ref_006 extract_clip.ipynb
    
    Binding addendum: All vision-language models accessed via OpenCLIP and CLIP modules
    """
    
    def __init__(self, model_name: str, num_classes: int = 1000, 
                 device: str = "cpu", class_names: Optional[List[str]] = None):
        super().__init__(model_name, num_classes, device)
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.text_features = None
        self._load_model()
    
    def _load_model(self):
        """
        Load CLIP or OpenCLIP model.
        
        reference_grounding: paperbench_ref_006 extract_clip.ipynb
        """
        # Try OpenCLIP first
        try:
            import open_clip
            # Parse model name (format: "openclip:<model>/<pretrained>")
            if self.model_name.startswith("openclip:"):
                model_spec = self.model_name.replace("openclip:", "")
                if "/" in model_spec:
                    model_arch, pretrained = model_spec.split("/", 1)
                else:
                    model_arch = model_spec
                    pretrained = "openai"
                
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    model_arch, pretrained=pretrained
                )
                self.tokenizer = open_clip.get_tokenizer(model_arch)
                self.model = self.model.to(self.device)
                self.model.eval()
                logger.info(f"Loaded {model_arch} from OpenCLIP with {pretrained} weights")
                return
        except ImportError:
            logger.debug("OpenCLIP not available, trying CLIP")
        except Exception as e:
            logger.debug(f"OpenCLIP loading failed: {e}")
        
        # Fall back to original CLIP
        try:
            import clip
            # Parse model name (format: "clip:<model>")
            if self.model_name.startswith("clip:"):
                model_arch = self.model_name.replace("clip:", "")
            else:
                model_arch = "ViT-B/32"  # Default
            
            self.model, self.preprocess = clip.load(model_arch, device=self.device)
            self.tokenizer = clip.tokenize
            logger.info(f"Loaded {model_arch} from CLIP")
            return
        except ImportError:
            raise ImportError("Neither open_clip nor clip available. Install one of them.")
    
    def _encode_text(self, texts: List[str]) -> Any:
        """Encode text prompts to feature vectors."""
        try:
            import torch
        except ImportError:
            raise ImportError("torch required for text encoding")
        
        # Handle both CLIP and OpenCLIP tokenizer formats
        if hasattr(self, 'tokenizer'):
            if callable(self.tokenizer):
                tokens = self.tokenizer(texts)
            else:
                tokens = self.tokenizer(texts)
        else:
            raise ValueError("Tokenizer not available")
        
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(tokens)
        tokens = tokens.to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features
    
    def _prepare_text_features(self, prompt_template: str = "a photo of a {}"):
        """Pre-compute text features for all classes."""
        if self.text_features is None:
            prompts = [prompt_template.format(name) for name in self.class_names]
            self.text_features = self._encode_text(prompts)
    
    def predict(self, images: Any, return_features: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Zero-shot prediction using CLIP.
        
        reference_grounding: paperbench_ref_006 extract_clip.ipynb
        """
        try:
            import torch
        except ImportError:
            raise ImportError("torch required for prediction")
        
        self.model.eval()
        self._prepare_text_features()
        
        with torch.no_grad():
            if not isinstance(images, torch.Tensor):
                images = torch.tensor(images)
            images = images.to(self.device)
            
            # Encode images
            image_features = self.model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity with text features
            logits = 100.0 * image_features @ self.text_features.T
            probs = logits.softmax(dim=-1).cpu().numpy()
            
            if return_features:
                return probs, image_features.cpu().numpy()
            else:
                return probs
    
    def evaluate(self, dataloader: Any, hierarchy: Optional[Any] = None) -> Dict[str, float]:
        """
        Evaluate CLIP model on dataset.
        
        reference_grounding: paperbench_ref_006 extract_clip.ipynb
        """
        try:
            import torch
        except ImportError:
            raise ImportError("torch required for evaluation")
        
        self.model.eval()
        self._prepare_text_features()
        
        correct_top1 = 0
        correct_top5 = 0
        total = 0
        lca_distances = []
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Encode images
                image_features = self.model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Compute similarity
                logits = 100.0 * image_features @ self.text_features.T
                
                # Top-1
                _, pred_top1 = logits.max(1)
                correct_top1 += pred_top1.eq(labels).sum().item()
                
                # Top-5
                _, pred_top5 = logits.topk(5, 1, True, True)
                pred_top5 = pred_top5.t()
                correct_top5 += pred_top5.eq(labels.view(1, -1).expand_as(pred_top5)).sum().item()
                
                total += labels.size(0)
                
                # LCA distances
                if hierarchy is not None:
                    for i in range(len(labels)):
                        pred_class = pred_top1[i].item()
                        true_class = labels[i].item()
                        lca_dist = hierarchy.compute_lca_distance(pred_class, true_class)
                        lca_distances.append(lca_dist)
        
        metrics = {
            "top1_accuracy": correct_top1 / total if total > 0 else 0.0,
            "top5_accuracy": correct_top5 / total if total > 0 else 0.0,
            "num_samples": total,
        }
        
        if len(lca_distances) > 0:
            metrics["lca_distance"] = float(np.mean(lca_distances))
            metrics["lca_distance_std"] = float(np.std(lca_distances))
        
        return metrics
    
    def get_features(self, images: Any) -> np.ndarray:
        """
        Extract image features for clustering or analysis.
        
        reference_grounding: paperbench_ref_006 extract_clip.ipynb
        """
        try:
            import torch
        except ImportError:
            raise ImportError("torch required for feature extraction")
        
        self.model.eval()