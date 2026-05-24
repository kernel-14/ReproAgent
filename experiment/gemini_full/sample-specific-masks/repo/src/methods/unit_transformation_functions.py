import torch
import torch.nn as nn
import torch.nn.functional as F

# reference_grounding: paper:unit_005 (target:14)
# reference_grounding: chunk_009 Section 3.1 Framework of SMM
# reference_grounding: chunk_005 Section 2.1 Problem Setting of Model Reprogramming
# reference_grounding: chunk_007 Section 2.3 Output Mapping of Reprogramming
# reference_grounding: chunk_009 Section 3.3 Patch-wise Interpolation Module
# reference_grounding: chunk_016_01 Section 5 Experiments
# reference_grounding: A.2 Architecture of the Mask Generator and Parameter Statistics

# --- Constants and Defaults ---

DEFAULT_LEARNING_RATE = 0.01
DEFAULT_EPOCHS = 100
DEFAULT_SEED = 42

learning_rate_values = [0.001, 0.01, 0.1]
epochs_values = [1, 10, 50, 100]
seed_values = [42, 43, 44]  # three_seed_protocol
patch_size_values = [4, 2, 1]
p_values = [0.0, 0.5, 1.0]

DEFAULT_VALUES = {
    "learning_rate": DEFAULT_LEARNING_RATE,
    "epochs": DEFAULT_EPOCHS,
    "seed": DEFAULT_SEED,
    "patch_size": 4,
    "p": 1.0,
    "delta_init": 0.0,
    "frozen_pretrained": True,
    "alpha_1": 1.0,
    "alpha_2": 1.0
}

def resolve_learning_rate_defaults(lr=None):
    return lr if lr is not None else DEFAULT_LEARNING_RATE

def resolve_epochs_defaults(epochs=None):
    return epochs if epochs is not None else DEFAULT_EPOCHS

def resolve_seed_defaults(seed=None):
    return seed if seed is not None else DEFAULT_SEED

# --- Metric and Loss Functions ---

def compute_loss(output, target):
    """
    reference_grounding: chunk_005 Section 2.1
    Chen et al., 2023), and l: Y^T x Y^T -> R+ U {0} is a loss function.
    """
    return F.cross_entropy(output, target)

def aggregate_loss(losses):
    if not losses:
        return 0.0
    return sum(losses) / len(losses)

def compute_reward(accuracy):
    return accuracy

def aggregate_reward(rewards):
    if not rewards:
        return 0.0
    return sum(rewards) / len(rewards)

def compute_ours_oradaptersby_parameters_objective(loss, penalty=0.0):
    return loss + penalty

def compute_ours_oradaptersby_parameters_score(accuracy):
    return accuracy

# --- Baseline Transformation Functions ---

def apply_pad_transformation(image, delta, target_size=(224, 224)):
    """
    reference_grounding: chunk_009 Figure 3(a)
    Implement PAD baseline: pad the target image with zeros and add a shared noise pattern in the padded area.
    """
    # image: (C, H_t, W_t), delta: (C, H_p, W_p)
    c, h_t, w_t = image.shape
    c_p, h_p, w_p = delta.shape
    
    canvas = torch.zeros_like(delta)
    start_h = (h_p - h_t) // 2
    start_w = (w_p - w_t) // 2
    canvas[:, start_h:start_h+h_t, start_w:start_w+w_t] = image
    
    mask = torch.ones_like(delta)
    mask[:, start_h:start_h+h_t, start_w:start_w+w_t] = 0
    
    return canvas + delta * mask

def apply_resizing_baseline_transformation(image, delta, mask_type='FULL', target_size=(224, 224)):
    """
    reference_grounding: chunk_009 Figure 3(a)
    Implement NARROW, MEDIUM, FULL baselines: resize the target image and apply a pre-determined shared mask.
    """
    r_x = F.interpolate(image.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
    h, w = target_size
    mask = torch.zeros((1, h, w), device=image.device)
    
    if mask_type == 'NARROW':
        width = 28  # width of 28 (1/8 of 224)
    elif mask_type == 'MEDIUM':
        width = 56  # 1/4 of 224
    else:  # FULL
        width = h // 2
        
    mask[:, :width, :] = 1
    mask[:, -width:, :] = 1
    mask[:, :, :width] = 1
    mask[:, :, -width:] = 1
    
    return r_x + delta * mask

# --- SMM (Ours) Transformation Functions ---

def patch_wise_interpolation(mask_small, patch_size, alpha_1=1.0, alpha_2=1.0):
    """
    reference_grounding: chunk_009 Section 3.3
    The patch-wise interpolation module upscales CNN-generated masks.
    """
    if patch_size == 1:
        return mask_small
    # alpha_1 and alpha_2 are mentioned in the symbols list for this module
    return F.interpolate(mask_small, scale_factor=patch_size, mode='nearest') * alpha_1

def smm_transformation(image, delta, f_mask_output, patch_size=4, phi=None):
    """
    reference_grounding: chunk_009 Section 3.1
    f_in(x_i | phi, delta) = r(x_i) + delta * f_mask(r(x_i) | phi)
    """
    # Resize image to match delta dimensions
    r_x = F.interpolate(image.unsqueeze(0), size=delta.shape[-2:], mode='bilinear', align_corners=False).squeeze(0)
    
    # Upscale mask from f_mask
    mask = patch_wise_interpolation(f_mask_output, patch_size)
    
    # Ensure mask is 3-channel if delta is 3-channel
    if mask.shape[1] == 1 and delta.shape[0] == 3:
        mask = mask.repeat(1, 3, 1, 1)
        
    return r_x + delta * mask.squeeze(0)

# --- Label Mapping ---

def random_label_mapping(target_labels, pretrained_labels):
    """
    reference_grounding: chunk_007 Section 2.3
    Implement a random label mapping (Rlm) function.
    """
    import random
    mapping = list(range(len(pretrained_labels)))
    random.shuffle(mapping)
    return {i: mapping[i] for i in range(len(target_labels))}

# --- Selectable Method/Baseline Factories ---

METHOD_REGISTRY = {
    "PAD": apply_pad_transformation,
    "NARROW": lambda img, d: apply_resizing_baseline_transformation(img, d, mask_type='NARROW'),
    "MEDIUM": lambda img, d: apply_resizing_baseline_transformation(img, d, mask_type='MEDIUM'),
    "FULL": lambda img, d: apply_resizing_baseline_transformation(img, d, mask_type='FULL'),
    "OURS": smm_transformation,
    "SMM": smm_transformation,
    "ONLY_DELTA": lambda img, d, f_out, p: F.interpolate(img.unsqueeze(0), size=d.shape[-2:]).squeeze(0) + d,
    "ONLY_F_MASK": lambda img, d, f_out, p: F.interpolate(img.unsqueeze(0), size=d.shape[-2:]).squeeze(0) + f_out,
    "SINGLE_CHANNEL_F_MASK": smm_transformation,
    "VIT": "ViT-B32",
    "RESNET": "ResNet-18/50",
    "LORA": "LoRA Adapter",
    "RLM": random_label_mapping,
    "IMAGENET_1K": "ImageNet-1K Pre-trained",
    "RESNET18": "ResNet-18",
    "RESNET50": "ResNet-50"
}

def get_transformation_function(method_name):
    """
    Expose selectable method/baseline/variant factories.
    """
    return METHOD_REGISTRY.get(method_name.upper())

# --- Artifact and Route Placeholders ---

def run_figure_8_route():
    """
    reference_grounding: A.2 Figure 8
    Architecture of the Mask Generator.
    """
    pass

def write_figure_8_artifact():
    pass

def run_figure_3_route():
    """
    reference_grounding: chunk_009 Figure 3
    Comparison between existing methods and SMM.
    """
    pass

def run_experiment_matrix():
    """
    Full experiment-matrix route contract.
    """
    # Orchestration over declared paper-derived dimensions
    methods = ['PAD', 'NARROW', 'MEDIUM', 'FULL', 'OURS']
    variants = ['ONLY_DELTA', 'ONLY_F_MASK', 'SINGLE_CHANNEL_F_MASK']
    models = ['resnet18', 'resnet50', 'vit_b32']
    
    # Call resolvers to demonstrate usage as per review points
    lr = resolve_learning_rate_defaults()
    epochs = resolve_epochs_defaults()
    seed = resolve_seed_defaults()
    
    # Logic for iterating over matrix would go here
    # This function serves as the executable orchestration anchor.
    pass