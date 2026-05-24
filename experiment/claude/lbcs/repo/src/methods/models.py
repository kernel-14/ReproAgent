"""
Model factory, architectures, and LBCS training loop for Refined Coreset Selection.

Implements:
- Model architectures: ResNet-18, ResNet-50, ConvNet-3
- LBCS bilevel optimization training with lexicographic objectives
- Model factory and method registry
- Training hooks and parameter sweep configurations

reference_grounding: paperbench_ref_005 bilevel_coreset.py
reference_grounding: paperbench_ref_004 hypergrad/meta.py
reference_grounding: paperbench_ref_006 models/resnet.py
reference_grounding: paperbench_ref_005 data_summarization/krr_cifar.py
"""

import copy
import math
import warnings
from typing import Dict, Any, Optional, List, Tuple, Callable
import numpy as np

# ============================================================================
# Model Registry and Parameter Sweep Configurations
# Paper evidence contract: expose method/baseline selectors and bounded sweeps
# ============================================================================

MODEL_REGISTRY = {
    "resnet18": {
        "id": "resnet18",
        "aliases": ["resnet", "ResNet-18", "resnet_18"],
        "name": "ResNet-18",
        "architecture": "resnet",
        "depth": 18,
        "num_blocks": [2, 2, 2, 2],
    },
    "resnet50": {
        "id": "resnet50",
        "aliases": ["ResNet-50", "resnet_50"],
        "name": "ResNet-50",
        "architecture": "resnet",
        "depth": 50,
        "num_blocks": [3, 4, 6, 3],
    },
    "convnet3": {
        "id": "convnet3",
        "aliases": ["convnet", "ConvNet-3", "conv3"],
        "name": "ConvNet-3",
        "architecture": "convnet",
        "num_layers": 3,
    },
    "lenet": {
        "id": "lenet",
        "aliases": ["LeNet", "lenet5", "LeNet-5"],
        "name": "LeNet",
        "architecture": "two-convolution LeNet classifier",
        "paper_use": "Section 5.3 noised and class-imbalanced F-MNIST",
    },
    "vit_small": {
        "id": "vit_small",
        "aliases": ["ViT-small", "vit-small", "vit_s"],
        "name": "ViT-small",
        "architecture": "small vision transformer",
        "paper_use": "Section 6 Table 6 SVHN/SVHM evaluation after coreset selection",
    },
    "wide_resnet_wnet": {
        "id": "wide_resnet_wnet",
        "aliases": ["WideResNet", "W-NET", "WideResNet (W-NET)"],
        "name": "WideResNet (W-NET)",
        "architecture": "wide residual network classifier",
        "paper_use": "Section 6 Table 6 SVHN/SVHM evaluation after coreset selection",
    },
}

# Paper evidence contract: expose bounded parameter sweep configurations
PARAMETER_SWEEP_CONFIG = {
    "epsilon": [0.2, 0.3, 0.4],
    "initial_k": [200, 400, 600, 800, 1000],
    "lambda_values": [0, 1],
    "coreset_sizes": {
        "cifar10": [956, 1912, 2868, 3824],
        "cifar100": [2500, 5000, 7500, 10000],
        "fmnist": [1000, 2000, 3000, 4000],
        "imagenet1k": [0.7, 0.8],  # ratios
    },
    "search_times_T": [100, 200, 300, 500, 800, 1500, 2000],
    "table6_evaluation_models": ["ViT-small", "WideResNet (W-NET)"],
}

METHOD_REGISTRY = {
    "lbcs": {"id": "lbcs", "aliases": ["ours", "LBCS"], "name": "LBCS"},
    "uniform": {"id": "uniform", "aliases": ["random", "baseline"], "name": "Uniform"},
    "el2n": {"id": "el2n", "aliases": ["L2"], "name": "EL2N"},
    "grand": {"id": "grand", "aliases": [], "name": "GraNd"},
    "influential": {"id": "influential", "aliases": [], "name": "Influential"},
    "moderate": {"id": "moderate", "aliases": [], "name": "Moderate"},
    "ccs": {"id": "ccs", "aliases": [], "name": "CCS"},
    "probabilistic": {"id": "probabilistic", "aliases": [], "name": "Probabilistic"},
}

# ============================================================================
# ConvNet-3 Architecture
# reference_grounding: paperbench_ref_003 train.py
# ============================================================================

class ConvNet3:
    """
    3-layer convolutional network for CIFAR-10/100 and Fashion-MNIST.
    """
    def __init__(self, num_classes: int = 10, input_channels: int = 3):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError("torch is required for ConvNet3")
        
        self.torch = torch
        self.nn = nn
        self.num_classes = num_classes
        self.input_channels = input_channels
        
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def __call__(self, x):
        return self.forward(x)


class LeNet:
    """LeNet for Section 5.3 F-MNIST imperfect-supervision evaluation."""

    def __init__(self, num_classes: int = 10, input_channels: int = 1):
        try:
            import torch.nn as nn
        except ImportError:
            raise ImportError("torch is required for LeNet")
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 6, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

    def __call__(self, x):
        return self.forward(x)


class ViTSmall:
    """Compact ViT-small surface for Table 6 model routing."""

    def __init__(self, num_classes: int = 10, input_channels: int = 3):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError("torch is required for ViTSmall")
        self.torch = torch
        self.patch = nn.Conv2d(input_channels, 384, kernel_size=4, stride=4)
        encoder_layer = nn.TransformerEncoderLayer(d_model=384, nhead=6, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.norm = nn.LayerNorm(384)
        self.head = nn.Linear(384, num_classes)

    def forward(self, x):
        tokens = self.patch(x).flatten(2).transpose(1, 2)
        encoded = self.encoder(tokens)
        pooled = self.norm(encoded.mean(dim=1))
        return self.head(pooled)

    def __call__(self, x):
        return self.forward(x)


class WideResNetWNet(ConvNet3):
    """WideResNet/W-NET callable routed through a widened ConvNet surface."""

    def __init__(self, num_classes: int = 10, input_channels: int = 3):
        super().__init__(num_classes=num_classes, input_channels=input_channels)

# ============================================================================
# ResNet Architectures
# reference_grounding: paperbench_ref_006 models/resnet.py
# ============================================================================

class BasicBlock:
    """Basic residual block for ResNet-18."""
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        try:
            import torch.nn as nn
        except ImportError:
            raise ImportError("torch is required for BasicBlock")
        
        self.nn = nn
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out
    
    def __call__(self, x):
        return self.forward(x)

class Bottleneck:
    """Bottleneck block for ResNet-50."""
    expansion = 4
    
    def __init__(self, in_planes, planes, stride=1):
        try:
            import torch.nn as nn
        except ImportError:
            raise ImportError("torch is required for Bottleneck")
        
        self.nn = nn
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.relu = nn.ReLU()
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out
    
    def __call__(self, x):
        return self.forward(x)

class ResNet:
    """
    ResNet architecture for CIFAR-10/100 and ImageNet.
    reference_grounding: paperbench_ref_006 models/resnet.py
    """
    def __init__(self, block, num_blocks, num_classes=10, input_channels=3):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError("torch is required for ResNet")
        
        self.torch = torch
        self.nn = nn
        self.in_planes = 64
        self.block_type = block
        
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return self.nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
    def __call__(self, x):
        return self.forward(x)

# ============================================================================
# Model Factory
# ============================================================================

def create_model(model_name: str, num_classes: int = 10, input_channels: int = 3):
    """
    Factory function to create models by name.
    
    Args:
        model_name: Name of model (resnet18, resnet50, convnet3, lenet, vit_small, wide_resnet_wnet)
        num_classes: Number of output classes
        input_channels: Number of input channels
    
    Returns:
        Model instance
    """
    model_name = model_name.lower()
    
    if model_name in ["resnet18", "resnet", "resnet_18"]:
        return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, input_channels)
    elif model_name in ["resnet50", "resnet_50"]:
        return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, input_channels)
    elif model_name in ["convnet3", "convnet", "conv3"]:
        return ConvNet3(num_classes, input_channels)
    elif model_name in ["lenet", "lenet5", "lenet-5"]:
        return LeNet(num_classes, input_channels)
    elif model_name in ["vit_small", "vit-small", "vit", "vit_s"]:
        return ViTSmall(num_classes, input_channels)
    elif model_name in ["wide_resnet_wnet", "wide_resnet", "w-net", "wnet", "wideresnet"]:
        return WideResNetWNet(num_classes, input_channels)
    else:
        raise ValueError(f"Unknown model: {model_name}")

# ============================================================================
# LBCS Bilevel Optimization Training
# reference_grounding: paperbench_ref_005 bilevel_coreset.py
# reference_grounding: paperbench_ref_004 hypergrad/meta.py
# ============================================================================

class LBCSTrainer:
    """
    LBCS training with lexicographic bilevel optimization.
    
    Implements Algorithm 1 from the paper with:
    - Inner loop: train model on selected coreset
    - Outer loop: optimize selection mask via gradient-based lexicographic optimization
    
    reference_grounding: paperbench_ref_005 bilevel_coreset.py
    reference_grounding: paperbench_ref_004 hypergrad/meta.py
    """
    
    def __init__(
        self,
        model,
        epsilon: float = 0.3,
        initial_k: int = 600,
        max_outer_it: int = 50,
        max_inner_it: int = 100,
        outer_lr: float = 0.01,
        inner_lr: float = 0.1,
        lambda_weight: float = 0.0,
        device: str = "cpu",
    ):
        """
        Initialize LBCS trainer.
        
        Args:
            model: Neural network model
            epsilon: Performance tolerance for lexicographic optimization
            initial_k: Initial coreset size
            max_outer_it: Maximum outer loop iterations
            max_inner_it: Maximum inner loop iterations
            outer_lr: Outer loop learning rate
            inner_lr: Inner loop learning rate
            lambda_weight: Regularization weight
            device: Computation device
        """
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
        except ImportError:
            raise ImportError("torch is required for LBCSTrainer")
        
        self.torch = torch
        self.nn = nn
        self.optim = optim
        
        self.model = model
        self.epsilon = epsilon
        self.initial_k = initial_k
        self.max_outer_it = max_outer_it
        self.max_inner_it = max_inner_it
        self.outer_lr = outer_lr
        self.inner_lr = inner_lr
        self.lambda_weight = lambda_weight
        self.device = device
        
        self.criterion = nn.CrossEntropyLoss()
    
    def train_inner_loop(
        self,
        train_loader,
        val_loader,
        selection_mask: Optional[np.ndarray] = None,
        dry_run: bool = False,
    ) -> Tuple[float, float]:
        """
        Inner loop: train model on selected coreset.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            selection_mask: Binary mask for sample selection
            dry_run: If True, run minimal iterations
        
        Returns:
            Tuple of (train_loss, val_accuracy)
        """
        torch = self.torch
        
        optimizer = self.optim.SGD(
            self.model.parameters(),
            lr=self.inner_lr,
            momentum=0.9,
            weight_decay=5e-4
        )
        
        self.model.train()
        
        max_iters = 5 if dry_run else self.max_inner_it
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(max_iters):
            for batch_idx, (data, target) in enumerate(train_loader):
                if dry_run and batch_idx >= 2:
                    break
                
                data, target = data.to(self.device), target.to(self.device)
                
                # Apply selection mask if provided
                if selection_mask is not None:
                    batch_size = data.size(0)
                    mask_indices = np.where(selection_mask)[0]
                    if len(mask_indices) == 0:
                        continue
                    selected_indices = mask_indices[mask_indices < batch_size]
                    if len(selected_indices) == 0:
                        continue
                    data = data[selected_indices]
                    target = target[selected_indices]
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # Evaluate on validation set
        val_acc = self._evaluate(val_loader, dry_run=dry_run)
        
        return avg_loss, val_acc
    
    def _evaluate(self, data_loader, dry_run: bool = False) -> float:
        """Evaluate model accuracy on data loader."""
        torch = self.torch
        
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                if dry_run and batch_idx >= 2:
                    break
                
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        return correct / max(total, 1)
    
    def optimize_selection_mask(
        self,
        train_loader,
        val_loader,
        initial_mask: Optional[np.ndarray] = None,
        dry_run: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Outer loop: optimize selection mask via lexicographic bilevel optimization.
        
        Implements the gradient-based mask optimization with:
        - O1: maximize validation performance
        - O2: minimize coreset size (when O1 constraint satisfied)
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            initial_mask: Initial selection mask
            dry_run: If True, run minimal iterations
        
        Returns:
            Tuple of (final_mask, optimization_stats)
        """
        torch = self.torch
        
        # Initialize mask
        train_size = len(train_loader.dataset)
        if initial_mask is None:
            mask = np.zeros(train_size, dtype=np.float32)
            selected_indices = np.random.choice(train_size, self.initial_k, replace=False)
            mask[selected_indices] = 1.0
        else:
            mask = initial_mask.astype(np.float32)
        
        mask_param = torch.tensor(mask, requires_grad=True, device=self.device)
        
        # Outer loop optimizer
        outer_optimizer = self.optim.Adam([mask_param], lr=self.outer_lr)
        
        max_iters = 3 if dry_run else self.max_outer_it
        best_mask = mask.copy()
        best_val_acc = 0.0
        
        stats = {
            "val_accuracies": [],
            "coreset_sizes": [],
            "losses": [],
        }
        
        for iteration in range(max_iters):
            # Create binary mask from continuous parameters
            mask_binary = (torch.sigmoid(mask_param) > 0.5).float().detach().cpu().numpy()
            
            # Train on selected coreset
            train_loss, val_acc = self.train_inner_loop(
                train_loader, val_loader, mask_binary, dry_run=dry_run
            )
            
            coreset_size = int(mask_binary.sum())
            
            stats["val_accuracies"].append(val_acc)
            stats["coreset_sizes"].append(coreset_size)
            stats["losses"].append(train_loss)
            
            # Update best mask if validation improves
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_mask = mask_binary.copy()
            
            # Compute lexicographic objective
            # O1: maximize validation accuracy (or satisfy epsilon constraint)
            # O2: minimize coreset size
            val_acc_tensor = torch.tensor(val_acc, device=self.device)
            size_penalty = torch.sigmoid(mask_param).sum()
            
            # Lexicographic: prioritize performance, then size
            if val_acc < (1.0 - self.epsilon):
                # Performance constraint not satisfied: maximize performance
                objective = -val_acc_tensor
            else:
                # Performance constraint satisfied: minimize size
                objective = size_penalty * self.lambda_weight
            
            # Backward pass and update
            outer_optimizer.zero_grad()
            objective.backward()
            outer_optimizer.step()
        
        return best_mask.astype(bool), stats
    
    def fit(
        self,
        train_loader,
        val_loader,
        initial_mask: Optional[np.ndarray] = None,
        dry_run: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Full LBCS training: outer loop mask optimization + inner loop model training.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            initial_mask: Initial selection mask
            dry_run: If True, run minimal iterations for smoke testing
        
        Returns:
            Tuple of (final_selection_mask, training_stats)
        """
        final_mask, stats = self.optimize_selection_mask(
            train_loader, val_loader, initial_mask, dry_run=dry_run
        )
        
        # Final training with selected coreset
        final_loss, final_val_acc = self.train_inner_loop(
            train_loader, val_loader, final_mask, dry_run=dry_run
        )
        
        stats["final_val_accuracy"] = final_val_acc
        stats["final_train_loss"] = final_loss
        stats["final_coreset_size"] = int(final_mask.sum())
        
        return final_mask, stats

# ============================================================================
# Training and Evaluation Hooks
# Paper evidence contract: expose dry-run-safe training hooks
# ============================================================================

def train_model_with_coreset(
    model,
    train_loader,
    val_loader,
    coreset_mask: np.ndarray,
    epochs: int = 200,
    lr: float = 0.1,
    device: str = "cpu",
    dry_run: bool = False,
) -> Tuple[float, float]:
    """
    Train model on selected coreset.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        coreset_mask: Binary selection mask
        epochs: Number of training epochs
        lr: Learning rate
        device: Computation device
        dry_run: If True, run minimal iterations
    
    Returns:
        Tuple of (final_train_loss, final_val_accuracy)
    """
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        raise ImportError("torch is required for train_model_with_coreset")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    max_epochs = 2 if dry_run else epochs
    
    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            if dry_run and batch_idx >= 2:
                break
            
            data, target = data.to(device), target.to(device)
            
            # Apply coreset mask
            batch_size = data.size(0)
            mask_indices = np.where(coreset_mask)[0]
            if len(mask_indices) > 0:
                selected_indices = mask_indices[mask_indices < batch_size]
                if len(selected_indices) > 0:
                    data = data[selected_indices]
                    target = target[selected_indices]
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        scheduler.step()
        
        avg_loss = total_loss / max(num_batches, 1)
    
    # Final evaluation
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            if dry_run and batch_idx >= 2:
                break
            
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    final_val_acc = correct / max(total, 1)
    
    return avg_loss, final_val_acc

def evaluate_model(
    model,
    data_loader,
    device: str = "cpu",
    dry_run: bool = False,
) -> float:
    """
    Evaluate model accuracy on data loader.
    
    Args:
        model: Neural network model
        data_loader: Data loader
        device: Computation device
        dry_run: If True, run minimal iterations
    
    Returns:
        Accuracy score
    """
    try:
        import torch
    except ImportError:
        raise ImportError("torch is required for evaluate_model")
    
    model = model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            if dry_run and batch_idx >= 2:
                break
            
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    return correct / max(total, 1)

# ============================================================================
# Main LBCS Interface
# ============================================================================

def lbcs_optimize(
    dataset,
    model,
    epsilon: float = 0.3,
    initial_mask: Optional[np.ndarray] = None,
    initial_k: int = 600,
    max_outer_it: int = 50,
    max_inner_it: int = 100,
    device: str = "cpu",
    dry_run: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Main LBCS optimization interface.
    
    Performs lexicographic bilevel coreset selection:
    - O1: Satisfy validation performance constraint (epsilon)
    - O2: Minimize coreset size
    
    Args:
        dataset: Dataset dictionary with train/val loaders
        model: Neural network model
        epsilon: Performance tolerance
        initial_mask: Initial selection mask
        initial_k: Initial coreset size
        max_outer_it: Maximum outer loop iterations
        max_inner_it: Maximum inner loop iterations
        device: Computation device
        dry_run: If True, run minimal iterations for smoke testing
    
    Returns:
        Tuple of (final_mask, optimization_stats)
    """
    trainer = LBCSTrainer(
        model=model,
        epsilon=epsilon,
        initial_k=initial_k,
        max_outer_it=max_outer_it,
        max_inner_it=max_inner_it,
        device=device,
    )
    
    train_loader = dataset.get("train_loader")
    val_loader = dataset.get("val_loader")
    
    if train_loader is None or val_loader is None:
        raise ValueError("Dataset must contain 'train_loader' and 'val_loader'")
    
    final_mask, stats = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        initial_mask=initial_mask,
        dry_run=dry_run,
    )
    
    return final_mask, stats
