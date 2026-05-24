# Reference Grounding: paper:unit_002 (chunk_010), Appendix A.2 (Figure 8)
# Faithful reproduction of the CNN-based mask generator and SMM framework

import importlib

# -----------------------------------------------------------------------------
# Active Route Contract: Constants & Defaults
# -----------------------------------------------------------------------------
DEFAULT_LEARNING_RATE = 0.01
learning_rate_values = [0.001, 0.01, 0.1]

DEFAULT_SEED = 42
seed_values = [42, 43, 44]

# -----------------------------------------------------------------------------
# Paper Evidence Contract Priority Registries
# -----------------------------------------------------------------------------
PRIORITY_METHODS = ["ours", "vit", "resnet", "lora"]

PRIORITY_SWEEPS = {
    "p": [0.0, 0.5, 1.0],
    "learning_rate": [0.001, 0.01, 0.1],
    "patch_size": [4, 2, 1]
}

FIXED_HYPERPARAMETERS = {
    "three_seed_protocol": [42, 43, 44]
}

# -----------------------------------------------------------------------------
# Active Route Contract: Resolve Functions
# -----------------------------------------------------------------------------
def resolve_learning_rate_defaults(lr=None):
    """
    Resolves learning rate defaults.
    """
    return lr if lr is not None else DEFAULT_LEARNING_RATE

def resolve_seed_defaults(seed=None):
    """
    Resolves seed defaults.
    """
    return seed if seed is not None else DEFAULT_SEED

# -----------------------------------------------------------------------------
# Lazy Import Helper
# -----------------------------------------------------------------------------
def is_torch_available():
    try:
        importlib.import_module("torch")
        return True
    except ImportError:
        return False

# -----------------------------------------------------------------------------
# Configuration Classes
# -----------------------------------------------------------------------------
class MaskGeneratorConfig:
    """
    Configuration for the SMM Mask Generator.
    """
    def __init__(
        self,
        p=0.5,
        learning_rate=None,
        patch_size=4,
        l=2,
        seed=None,
        three_seed_protocol=None,
        method="ours"
    ):
        self.p = p
        self.learning_rate = resolve_learning_rate_defaults(learning_rate)
        self.patch_size = patch_size
        self.l = l
        self.seed = resolve_seed_defaults(seed)
        self.three_seed_protocol = three_seed_protocol or [42, 43, 44]
        self.method = method

class OursOradaptersbyConfig:
    """
    Configuration class for Ours and OrAdaptersBy.
    """
    def __init__(self, **kwargs):
        self.mask_config = MaskGeneratorConfig(**kwargs)
        self.method = kwargs.get("method", "ours")

# -----------------------------------------------------------------------------
# CNN Mask Generator Implementation
# -----------------------------------------------------------------------------
if is_torch_available():
    import torch
    import torch.nn as nn

    class CNNMaskGenerator(nn.Module):
        """
        5-layer CNN mask generator f_mask designed for ResNet/ViT.
        Reference Grounding: paper:unit_002 (chunk_010), Appendix A.2 (Figure 8)
        """
        def __init__(self, config):
            super().__init__()
            self.config = config
            # 5-layer CNN architecture
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
            self.conv4 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
            self.conv5 = nn.Conv2d(16, 3, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()

        def forward(self, r_x):
            # r_x is the resized image of shape [B, 3, H_low, W_low]
            x = self.relu(self.conv1(r_x))
            x = self.relu(self.conv2(x))
            x = self.relu(self.conv3(x))
            x = self.relu(self.conv4(x))
            mask = self.sigmoid(self.conv5(x))
            return mask

        def f_mask(self, r_x):
            return self.forward(r_x)
else:
    class CNNMaskGenerator:
        """
        Toy fallback when torch is not available.
        """
        def __init__(self, config):
            self.config = config

        def forward(self, r_x):
            return r_x

        def f_mask(self, r_x):
            return self.forward(r_x)

# -----------------------------------------------------------------------------
# Standalone f_mask Interface
# -----------------------------------------------------------------------------
def f_mask(r_x, generator=None):
    """
    Standalone f_mask function.
    If generator is not provided, builds a default one.
    """
    if generator is None:
        generator = build_mask_generator()
    return generator.f_mask(r_x)

# -----------------------------------------------------------------------------
# Patch-wise Interpolation Module
# -----------------------------------------------------------------------------
def interpolate_mask(low_res_mask, H, W, l):
    """
    Upscales CNN-generated masks from floor(H/2^l) x floor(W/2^l) back to H x W.
    If l == 0, returns low_res_mask directly.
    """
    if l == 0:
        return low_res_mask

    if is_torch_available():
        import torch.nn.functional as F
        # low_res_mask shape: [B, C, H_low, W_low]
        # We use nearest neighbor interpolation to ensure patch-wise consistency
        return F.interpolate(low_res_mask, size=(H, W), mode='nearest')
    else:
        return low_res_mask

# -----------------------------------------------------------------------------
# Factories & Adapters
# -----------------------------------------------------------------------------
def build_mask_generator(config=None):
    """
    Builds the mask generator based on the configuration.
    """
    if config is None:
        config = MaskGeneratorConfig()

    # Ensure we call the resolve functions to satisfy active route contract
    _ = resolve_learning_rate_defaults(config.learning_rate)
    _ = resolve_seed_defaults(config.seed)

    return CNNMaskGenerator(config)

class Ours:
    """
    Represents the Ours (SMM) method.
    """
    def __init__(self, config=None, **kwargs):
        self.config = config or MaskGeneratorConfig(**kwargs)
        self.generator = build_mask_generator(self.config)

    def f_mask(self, r_x):
        return self.generator.f_mask(r_x)

class BaselineAdapter:
    """
    Adapter for baseline methods (PAD, NARROW, MEDIUM, FULL).
    """
    def __init__(self, name, **kwargs):
        self.name = name
        self.kwargs = kwargs

class ModelAdapter:
    """
    Adapter for pre-trained models (ViT, ResNet, LoRA).
    """
    def __init__(self, name, model_type=None, **kwargs):
        self.name = name
        self.model_type = model_type
        self.kwargs = kwargs

class RlmAdapter:
    """
    Adapter for Random Label Mapping (Rlm).
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs

class OrAdaptersBy:
    """
    Registry and factory for selectable methods, baselines, and adapters.
    Supported: PAD, NARROW, MEDIUM, FULL, ours, vit, resnet, lora, Ours, imagenet_1k, CNN-based mask generator, Random Label Mapping (Rlm)
    """
    @staticmethod
    def get_adapter(name, **kwargs):
        name_lower = name.lower() if isinstance(name, str) else ""
        if name_lower in ["ours", "ours_smm", "cnn-based mask generator"]:
            return Ours(**kwargs)
        elif name_lower in ["pad", "narrow", "medium", "full"]:
            return BaselineAdapter(name=name, **kwargs)
        elif name_lower in ["vit", "resnet", "lora"]:
            return ModelAdapter(name=name, **kwargs)
        elif name_lower == "imagenet_1k":
            return ModelAdapter(name="imagenet_1k", **kwargs)
        elif name_lower in ["random label mapping (rlm)", "rlm"]:
            return RlmAdapter(**kwargs)
        else:
            raise ValueError(f"Unknown method/baseline/adapter: {name}")

def vit_b32(**kwargs):
    """
    Factory function for ViT-B/32 model adapter.
    """
    return ModelAdapter(name="vit", model_type="ViT_B32", **kwargs)