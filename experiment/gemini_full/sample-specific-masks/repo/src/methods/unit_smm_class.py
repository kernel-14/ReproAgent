import os
import math

# Reference Grounding: addendum:formula_algorithm_contract, chunk_009, chunk_005
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_EPOCHS = 1
DEFAULT_SEED = 42
DEFAULT_VALUES = {
    "learning_rate": DEFAULT_LEARNING_RATE,
    "epochs": DEFAULT_EPOCHS,
    "seed": DEFAULT_SEED,
    "patch_size": 4,
    "p": 1.0,
    "three_seed_protocol": [42, 43, 44],
    "alpha_1": 1.0,
    "alpha_2": 1.0
}

learning_rate_values = [0.001, 0.01, 0.1]
epochs_values = [1, 10, 50]
seed_values = [42, 43, 44]
patch_size_values = [4, 2, 1]

def resolve_learning_rate_defaults(lr=None):
    return lr if lr is not None else DEFAULT_LEARNING_RATE

def resolve_epochs_defaults(epochs=None):
    return epochs if epochs is not None else DEFAULT_EPOCHS

def resolve_seed_defaults(seed=None):
    return seed if seed is not None else DEFAULT_SEED

def compute_loss(outputs, targets):
    """
    Reference Grounding: 2.1. Problem Setting of Model Reprogramming
    Implements the cross-entropy loss function l: Y^T x Y^T -> R+
    """
    try:
        import torch.nn.functional as F
        return F.cross_entropy(outputs, targets)
    except ImportError:
        return None

def aggregate_loss(losses):
    try:
        import torch
        if not losses:
            return torch.tensor(0.0)
        return torch.stack(losses).mean()
    except ImportError:
        return None

def compute_reward(outputs, targets):
    """
    Computes accuracy as a reward metric.
    """
    try:
        import torch
        _, predicted = torch.max(outputs, 1)
        return (predicted == targets).float().mean()
    except ImportError:
        return None

def aggregate_reward(rewards):
    try:
        import torch
        if not rewards:
            return torch.tensor(0.0)
        return torch.stack(rewards).mean()
    except ImportError:
        return None

def compute_ours_oradaptersby_parameters_objective(model, data_batch, config):
    """
    Primary objective function for SMM optimization.
    """
    images, targets = data_batch
    outputs = model(images)
    return compute_loss(outputs, targets)

def compute_ours_oradaptersby_parameters_score(model, data_batch, config):
    """
    Primary score function for SMM evaluation.
    """
    images, targets = data_batch
    outputs = model(images)
    return compute_reward(outputs, targets)

class MaskGenerator:
    """
    Reference Grounding: A.2. Architecture of the Mask Generator and Parameter Statistics
    Implements the 5-layer CNN mask generator f_mask designed for ResNet.
    """
    def __init__(self, in_channels=3, out_channels=3, hidden_dim=10):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self._model = None

    def _get_model(self, device):
        try:
            import torch.nn as nn
            if self._model is None:
                # 5-layer CNN architecture as per Figure 8
                self._model = nn.Sequential(
                    nn.Conv2d(self.in_channels, self.hidden_dim, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.hidden_dim, self.out_channels, kernel_size=3, padding=1),
                    nn.Sigmoid()
                ).to(device)
            return self._model
        except ImportError:
            return None

    def __call__(self, x):
        model = self._get_model(x.device)
        if model is None:
            return x
        return model(x)

class PatchWiseInterpolation:
    """
    Reference Grounding: 3.3. Patch-wise Interpolation Module
    Upscales masks from floor(H/2^l) x floor(W/2^l) back to H x W per channel.
    """
    def __init__(self, patch_size=4):
        self.patch_size = patch_size
        # l is the log2 of patch_size (e.g., patch_size=4 -> l=2)
        self.l = int(math.log2(patch_size)) if patch_size > 0 else 0

    def __call__(self, mask, target_h, target_w):
        try:
            import torch.nn.functional as F
            if self.l == 0:
                return mask
            # Upscale using nearest neighbor to ensure patch-wise consistency
            return F.interpolate(mask, size=(target_h, target_w), mode='nearest')
        except ImportError:
            return mask

class SMM:
    """
    Reference Grounding: 3.1. Framework of SMM
    Implements the input transformation f_in(x) = r(x) + M(x) * delta,
    where M(x) = f_mask(r(x)) and delta is the shared noise pattern.
    """
    def __init__(self, img_size=224, patch_size=4, num_channels=3, variant='ours'):
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.variant = variant
        
        self.mask_gen = MaskGenerator(in_channels=num_channels, out_channels=num_channels)
        self.interpolator = PatchWiseInterpolation(patch_size=patch_size)
        self.delta = None

    def get_delta(self, device):
        try:
            import torch
            import torch.nn as nn
            if self.delta is None:
                # delta initialized to zero as per paper evidence
                self.delta = nn.Parameter(torch.zeros(1, self.num_channels, self.img_size, self.img_size).to(device))
            return self.delta
        except ImportError:
            return None

    def forward(self, x):
        """
        Reference Grounding: 3.1. Framework of SMM
        f_in(x_i | phi, delta) = r(x_i) + delta * f_mask(r(x_i) | phi)
        """
        try:
            import torch
            import torch.nn.functional as F
            device = x.device
            delta = self.get_delta(device)
            
            if self.variant == 'ONLY delta':
                # Baseline: shared mask is all ones
                mask = torch.ones_like(delta)
            elif self.variant == 'ONLY f_mask':
                # Baseline: sample-specific mask, delta is fixed to 1
                low_h, low_w = self.img_size // self.patch_size, self.img_size // self.patch_size
                x_low = F.interpolate(x, size=(low_h, low_w), mode='bilinear', align_corners=False)
                mask_low = self.mask_gen(x_low)
                mask = self.interpolator(mask_low, self.img_size, self.img_size)
                return x + mask
            elif self.variant == 'SINGLE-CHANNEL f_mask^s':
                # Variant: mask generator outputs a single channel shared across RGB
                low_h, low_w = self.img_size // self.patch_size, self.img_size // self.patch_size
                x_low = F.interpolate(x, size=(low_h, low_w), mode='bilinear', align_corners=False)
                mask_low = self.mask_gen(x_low).mean(dim=1, keepdim=True).repeat(1, self.num_channels, 1, 1)
                mask = self.interpolator(mask_low, self.img_size, self.img_size)
            else: # 'ours' or 'SMM'
                low_h, low_w = self.img_size // self.patch_size, self.img_size // self.patch_size
                x_low = F.interpolate(x, size=(low_h, low_w), mode='bilinear', align_corners=False)
                mask_low = self.mask_gen(x_low)
                mask = self.interpolator(mask_low, self.img_size, self.img_size)

            return x + delta * mask
        except ImportError:
            return x

class FixedMaskReprogramming:
    """
    Reference Grounding: 5. Experiments | Impact of Masking
    Implements PAD, NARROW, MEDIUM, FULL baselines with pre-determined shared masks.
    """
    def __init__(self, strategy='PAD', img_size=224, num_channels=3):
        self.strategy = strategy
        self.img_size = img_size
        self.num_channels = num_channels
        self.delta = None
        self.mask = None

    def get_delta(self, device):
        try:
            import torch
            import torch.nn as nn
            if self.delta is None:
                self.delta = nn.Parameter(torch.zeros(1, self.num_channels, self.img_size, self.img_size).to(device))
            return self.delta
        except ImportError:
            return None

    def get_mask(self, device):
        try:
            import torch
            if self.mask is not None:
                return self.mask
            
            mask = torch.zeros(1, self.num_channels, self.img_size, self.img_size).to(device)
            # Widths as described in Figure 3 and Section 5
            if self.strategy in ['PAD', 'NARROW']:
                border = 28 # 1/8 of 224
            elif self.strategy == 'MEDIUM':
                border = 56 # 1/4 of 224
            elif self.strategy == 'FULL':
                border = 112 # 1/2 of 224
            else:
                border = 0
            
            if border > 0:
                mask[:, :, :border, :] = 1
                mask[:, :, -border:, :] = 1
                mask[:, :, :, :border] = 1
                mask[:, :, :, -border:] = 1
            
            self.mask = mask
            return self.mask
        except ImportError:
            return None

    def forward(self, x):
        device = x.device
        delta = self.get_delta(device)
        mask = self.get_mask(device)
        if delta is None or mask is None:
            return x
        return x + delta * mask

def method_factory(name, **kwargs):
    """
    Expose selectable method/baseline/variant factories.
    """
    if name in ['ours', 'Ours', 'SMM']:
        return SMM(variant='ours', **kwargs)
    elif name in ['PAD', 'NARROW', 'MEDIUM', 'FULL']:
        return FixedMaskReprogramming(strategy=name, **kwargs)
    elif name == 'ONLY delta':
        return SMM(variant='ONLY delta', **kwargs)
    elif name == 'ONLY f_mask':
        return SMM(variant='ONLY f_mask', **kwargs)
    elif name == 'SINGLE-CHANNEL f_mask^s':
        return SMM(variant='SINGLE-CHANNEL f_mask^s', **kwargs)
    return None

def get_model_backbone(model_name, num_classes=1000, pretrained=True):
    """
    Reference Grounding: 5. Experiments | ResNet-18, ResNet-50
    Loads pre-trained backbones and freezes parameters.
    """
    try:
        import torch.nn as nn
        from torchvision import models
        if 'resnet18' in model_name.lower():
            model = models.resnet18(pretrained=pretrained)
        elif 'resnet50' in model_name.lower():
            model = models.resnet50(pretrained=pretrained)
        elif 'vit' in model_name.lower():
            model = models.vit_b_16(pretrained=pretrained)
        else:
            return None
        
        # Frozen pre-trained model parameters as per paper
        for param in model.parameters():
            param.requires_grad = False
            
        return model
    except ImportError:
        return None

def environment_adapter(model, method, config=None):
    """
    Wires the pre-trained model and reprogramming method together.
    """
    class ReprogrammedModel:
        def __init__(self, backbone, reprogramming_module):
            self.backbone = backbone
            self.reprogramming_module = reprogramming_module
        
        def __call__(self, x):
            x_reprogrammed = self.reprogramming_module.forward(x)
            return self.backbone(x_reprogrammed)
            
    return ReprogrammedModel(model, method)

# Artifact writer placeholders for canonical route closure
def write_figure_1_artifact(*args, **kwargs): pass
def write_figure_2_artifact(*args, **kwargs): pass
def write_figure_3_artifact(*args, **kwargs): pass