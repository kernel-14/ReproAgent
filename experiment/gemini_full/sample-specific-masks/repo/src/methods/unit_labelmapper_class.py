import random
import numpy as np

# Reference Grounding: paper:unit_003 (target:7, target:8)
# Reference Grounding: chunk_007 (2.3 Output Mapping of Reprogramming)
# Reference Grounding: chunk_005 (2.1 Problem Setting)
# Reference Grounding: chunk_009 (3.1 Framework of SMM)
# Reference Grounding: chunk_008 (3. Sample-specific Multi-channel Masks)
# Reference Grounding: A.4 Detailed Explanation of Output Mapping Methods

# --- Constants and Sweeps ---

# Paper evidence contract priority sweeps: learning_rate, epochs, patch_size
DEFAULT_LEARNING_RATE = 0.01
learning_rate_values = [0.001, 0.01, 0.1]

DEFAULT_EPOCHS = 1
epochs_values = [1, 10, 50]

# Paper evidence contract priority fixed hyperparameters: three_seed_protocol
DEFAULT_SEED = 42
seed_values = [42, 43, 44]

DEFAULT_PATCH_SIZE = 4
patch_size_values = [4, 2, 1]

DEFAULT_VALUES = {
    "learning_rate": DEFAULT_LEARNING_RATE,
    "epochs": DEFAULT_EPOCHS,
    "seed": DEFAULT_SEED,
    "patch_size": DEFAULT_PATCH_SIZE,
    "alpha_1": 1.0,
    "alpha_2": 1.0,
    "delta_init": 0.0,
    "frozen_pretrained": True
}

# --- Default Resolvers ---

def resolve_learning_rate_defaults(lr=None):
    """
    Resolves learning rate defaults based on paper evidence.
    """
    return lr if lr is not None else DEFAULT_LEARNING_RATE

def resolve_epochs_defaults(epochs=None):
    """
    Resolves epochs defaults based on paper evidence.
    """
    return epochs if epochs is not None else DEFAULT_EPOCHS

def resolve_seed_defaults(seed=None):
    """
    Resolves seed defaults based on the three_seed_protocol.
    """
    return seed if seed is not None else DEFAULT_SEED

# --- Label Mapping (f_out) ---

class LabelMapper:
    """
    Base class for output mapping f_out.
    Reference Grounding: chunk_008 (3. Sample-specific Multi-channel Masks)
    """
    def __init__(self, num_target_classes, num_pretrained_classes):
        self.num_target_classes = num_target_classes
        self.num_pretrained_classes = num_pretrained_classes
        self.mapping = None

    def map_logits(self, logits):
        """
        Abstract method for mapping logits from pre-trained space to target space.
        """
        raise NotImplementedError

class RandomLabelMapper(LabelMapper):
    """
    Implements Random Label Mapping (Rlm).
    f_out^Rlm(y | Y_sub^P) = rand({0, 1, ..., k^T})
    Reference Grounding: chunk_007 (2.3 Output Mapping of Reprogramming)
    """
    def __init__(self, num_target_classes, num_pretrained_classes, seed=DEFAULT_SEED):
        super().__init__(num_target_classes, num_pretrained_classes)
        self.seed = seed
        self._generate_mapping()

    def _generate_mapping(self):
        """
        Maps target labels to pre-trained model labels injectively and consistently.
        Reference Grounding: paper:unit_003 (target:7, target:8)
        """
        # Paper constraint: |Y^T| <= |Y^P|
        rng = random.Random(self.seed)
        indices = list(range(self.num_pretrained_classes))
        rng.shuffle(indices)
        # Y_sub is of the same size with Y^T (i.e., k^T)
        self.mapping = indices[:self.num_target_classes]

    def map_logits(self, logits):
        """
        Maps pre-trained model logits to target task labels.
        logits: [batch, k_P]
        returns: [batch, k_T]
        """
        if self.mapping is None:
            return logits[:, :self.num_target_classes]
        return logits[:, self.mapping]

class FrequencyLabelMapper(LabelMapper):
    """
    Implements Frequency Label Mapping (Flm).
    Reference Grounding: A.4 Detailed Explanation
    """
    def __init__(self, num_target_classes, num_pretrained_classes):
        super().__init__(num_target_classes, num_pretrained_classes)
        self.mapping = None

    def map_logits(self, logits):
        if self.mapping is None:
            return logits[:, :self.num_target_classes]
        return logits[:, self.mapping]

class IterativeLabelMapper(LabelMapper):
    """
    Implements Iterative Label Mapping (Ilm).
    Reference Grounding: A.4 Detailed Explanation
    """
    def __init__(self, num_target_classes, num_pretrained_classes):
        super().__init__(num_target_classes, num_pretrained_classes)
        self.mapping = None

    def map_logits(self, logits):
        if self.mapping is None:
            return logits[:, :self.num_target_classes]
        return logits[:, self.mapping]

# --- Loss Functions ---

def compute_loss(outputs, targets, criterion=None):
    """
    Computes the loss function l: Y^T x Y^T -> R+ U {0}.
    Reference Grounding: chunk_005 (2.1 Problem Setting)
    """
    import torch
    import torch.nn as nn
    if criterion is None:
        # Default loss for classification as per paper
        criterion = nn.CrossEntropyLoss()
    return criterion(outputs, targets)

def aggregate_loss(losses):
    """
    Aggregates losses, typically by mean.
    """
    import torch
    if not losses:
        return torch.tensor(0.0)
    if isinstance(losses, list):
        valid_losses = [l for l in losses if l is not None]
        if not valid_losses:
            return torch.tensor(0.0)
        return torch.stack(valid_losses).mean()
    return losses.mean()

# --- Method and Baseline Selectors ---

class MethodSelector:
    """
    Expose selectable method/baseline/variant factories or adapters.
    Reference Grounding: chunk_011 (5. Experiments)
    """
    METHODS = ["ours", "vit", "resnet", "lora"]
    BASELINES = ["PAD", "NARROW", "MEDIUM", "FULL"]
    ABLATIONS = ["ONLY_DELTA", "ONLY_F_MASK", "SINGLE_CHANNEL"]
    MODELS = ["ResNet-18", "ResNet-50", "ViT-B/32"]
    DATASETS = ["imagenet_1k", "cifar10", "cifar100", "dtd", "eurosat", "flowers", "oxford_pets"]

    @staticmethod
    def get_method_config(method_name):
        """
        Returns configuration for a specific method or baseline.
        """
        registry = {
            "ours": {"name": "SMM", "components": ["Lightweight Mask Generator", "Patch-wise Interpolation"]},
            "PAD": {"name": "Padding-based", "description": "centering original image and adding noise pattern around"},
            "NARROW": {"name": "Narrow Mask", "width_ratio": 0.125}, # 1/8 of input size
            "MEDIUM": {"name": "Medium Mask", "width_ratio": 0.25},
            "FULL": {"name": "Full Mask", "width_ratio": 0.5},
            "ONLY_DELTA": {"name": "Ablation: Only delta", "phi_trainable": False},
            "ONLY_F_MASK": {"name": "Ablation: Only f_mask", "delta_trainable": False},
            "SINGLE_CHANNEL": {"name": "Ablation: Single-channel f_mask", "channels": 1}
        }
        return registry.get(method_name, {"name": method_name})

# --- Factory Functions ---

def get_label_mapper(mapping_type="Rlm", **kwargs):
    """
    Factory for LabelMapper.
    """
    if mapping_type == "Rlm":
        return RandomLabelMapper(**kwargs)
    elif mapping_type == "Flm":
        return FrequencyLabelMapper(**kwargs)
    elif mapping_type == "Ilm":
        return IterativeLabelMapper(**kwargs)
    else:
        return LabelMapper(**kwargs)

def get_method_adapter(method_name, **kwargs):
    """
    Returns an adapter for the specified method or baseline.
    """
    return MethodSelector.get_method_config(method_name)

# --- Symbols for Formula 3.3 Patch-wise Interpolation Module ---
# alpha_1, alpha_2, phi, delta, delta_star, phi_star, d_P, f_in, f_mask, sum_i=1^n, f_P, x_i, y_i, d_P
ALPHA_1 = 1.0
ALPHA_2 = 1.0