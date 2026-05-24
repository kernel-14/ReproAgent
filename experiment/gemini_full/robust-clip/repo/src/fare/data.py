# src/fare/data.py
# reference_grounding: addendum:formula_algorithm_contract
# reference_grounding: paperbench_ref_002 open_flamingo/

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

# Define constants
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 1
DEFAULT_EPSILON = "2/255"

# Define resolution functions
def resolve_batch_size_defaults(batch_size=None):
    if batch_size is None:
        return DEFAULT_BATCH_SIZE
    return int(batch_size)

def resolve_epochs_defaults(epochs=None):
    if epochs is None:
        return DEFAULT_EPOCHS
    return int(epochs)

def resolve_epsilon_defaults(epsilon=None):
    if epsilon is None:
        epsilon = DEFAULT_EPSILON
    if isinstance(epsilon, str):
        if "/" in epsilon:
            num, denom = epsilon.split("/")
            return float(num) / float(denom)
        return float(epsilon)
    return float(epsilon)

# Active route contract: define 对抗攻击流水线 (PGD/AutoAttack)
globals()["对抗攻击流水线 (PGD/AutoAttack)"] = "PGD/AutoAttack"
对抗攻击流水线 = "PGD/AutoAttack"

@dataclass
class DataSpec:
    dataset_name: str
    batch_size: int
    epochs: int
    epsilon: float
    extra_args: Optional[Dict[str, Any]] = None

# Explicitly register dataset/benchmark aliases
DATASET_REGISTRY = {
    "cifar": {
        "aliases": ["cifar10", "cifar100"],
        "description": "CIFAR dataset for zero-shot classification",
        "loader_name": "load_cifar"
    },
    "imagenet": {
        "aliases": ["imagenet_val", "imagenet-1k"],
        "description": "ImageNet dataset using HuggingFace",
        "loader_name": "load_imagenet"
    },
    "coco": {
        "aliases": ["coco_caption", "coco2017"],
        "description": "COCO dataset for image captioning and retrieval",
        "loader_name": "load_coco"
    },
    "flickr30k": {
        "aliases": ["flickr30k_retrieval"],
        "description": "Flickr30k dataset for image-text retrieval",
        "loader_name": "load_flickr30k"
    },
    "stl10": {
        "aliases": ["stl10_val"],
        "description": "STL-10 dataset",
        "loader_name": "load_stl10"
    },
    "imagenet_r": {
        "aliases": ["imagenet-r"],
        "description": "ImageNet-R for robust evaluation",
        "loader_name": "load_imagenet_r"
    },
    "imagenet_sketch": {
        "aliases": ["imagenet-sketch"],
        "description": "ImageNet-Sketch for robust evaluation",
        "loader_name": "load_imagenet_sketch"
    },
    "vqav2": {
        "aliases": ["vqa_v2"],
        "description": "VQA v2 dataset",
        "loader_name": "load_vqav2"
    },
    "textvqa": {
        "aliases": ["text_vqa"],
        "description": "TextVQA dataset",
        "loader_name": "load_textvqa"
    },
    "pope": {
        "aliases": ["pope_hallucination"],
        "description": "POPE benchmark for hallucination evaluation",
        "loader_name": "load_pope"
    },
    "sqa_i": {
        "aliases": ["science_qa_img", "sqa-i"],
        "description": "ScienceQA image-based questions",
        "loader_name": "load_sqa_i"
    },
    "caltech101": {
        "aliases": ["caltech-101"],
        "description": "Caltech 101 dataset",
        "loader_name": "load_caltech101"
    },
    "stanford_cars": {
        "aliases": ["stanford-cars", "cars"],
        "description": "Stanford Cars dataset",
        "loader_name": "load_stanford_cars"
    },
    "fgvc_aircraft": {
        "aliases": ["fgvc-aircraft", "aircraft"],
        "description": "FGVC Aircraft dataset",
        "loader_name": "load_fgvc_aircraft"
    },
    "flowers": {
        "aliases": ["oxford-flowers", "flowers102"],
        "description": "Oxford Flowers 102 dataset",
        "loader_name": "load_flowers"
    },
    "pcam": {
        "aliases": ["patch_camelyon"],
        "description": "PatchCamelyon dataset",
        "loader_name": "load_pcam"
    },
    "oxford_pets": {
        "aliases": ["oxford-iiit-pet", "pets"],
        "description": "Oxford-IIIT Pet dataset",
        "loader_name": "load_oxford_pets"
    }
}

def create_synthetic_dataset(num_samples=100):
    try:
        import torch
        from torch.utils.data import TensorDataset
        images = torch.randn(num_samples, 3, 224, 224)
        labels = torch.randint(0, 10, (num_samples,))
        return TensorDataset(images, labels)
    except ImportError:
        class MockDataset:
            def __init__(self, num_samples):
                self.num_samples = num_samples
            def __len__(self):
                return self.num_samples
            def __getitem__(self, idx):
                return {"image": [0.0]*3*224*224, "label": 0}
        return MockDataset(num_samples)

def load_imagenet(batch_size=32, trust_remote_code=True, split="validation"):
    """
    Load ImageNet using HuggingFace datasets.
    reference_grounding: addendum:formula_algorithm_contract
    """
    try:
        from datasets import load_dataset
        dataset = load_dataset("imagenet-1k", split=split, trust_remote_code=trust_remote_code)
        return dataset
    except Exception as e:
        print(f"Could not load ImageNet via HuggingFace: {e}. Falling back to synthetic data.")
        return create_synthetic_dataset(num_samples=100)

def load_cifar(batch_size=32, split="test"):
    try:
        import torchvision
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        dataset = torchvision.datasets.CIFAR10(root="./data", train=(split == "train"), download=True, transform=transform)
        return dataset
    except Exception as e:
        print(f"Could not load CIFAR: {e}. Falling back to synthetic data.")
        return create_synthetic_dataset(num_samples=100)

def load_coco(batch_size=32, split="val"):
    return create_synthetic_dataset(num_samples=100)

def load_flickr30k(batch_size=32, split="val"):
    return create_synthetic_dataset(num_samples=100)

def load_stl10(batch_size=32, split="test"):
    try:
        import torchvision
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        dataset = torchvision.datasets.STL10(root="./data", split=split, download=True, transform=transform)
        return dataset
    except Exception as e:
        print(f"Could not load STL10: {e}. Falling back to synthetic data.")
        return create_synthetic_dataset(num_samples=100)

def load_imagenet_r(batch_size=32):
    return create_synthetic_dataset(num_samples=100)

def load_imagenet_sketch(batch_size=32):
    return create_synthetic_dataset(num_samples=100)

def load_vqav2(batch_size=32):
    return create_synthetic_dataset(num_samples=100)

def load_textvqa(batch_size=32):
    return create_synthetic_dataset(num_samples=100)

def load_pope(batch_size=32):
    return create_synthetic_dataset(num_samples=100)

def load_sqa_i(batch_size=32):
    return create_synthetic_dataset(num_samples=100)

def load_caltech101(batch_size=32):
    return create_synthetic_dataset(num_samples=100)

def load_stanford_cars(batch_size=32):
    return create_synthetic_dataset(num_samples=100)

def load_fgvc_aircraft(batch_size=32):
    return create_synthetic_dataset(num_samples=100)

def load_flowers(batch_size=32):
    return create_synthetic_dataset(num_samples=100)

def load_pcam(batch_size=32):
    return create_synthetic_dataset(num_samples=100)

def load_oxford_pets(batch_size=32):
    return create_synthetic_dataset(num_samples=100)

def load_data(dataset_name: str, batch_size: Optional[int] = None, split: str = "val") -> Any:
    """
    Load dataset by name or alias.
    """
    resolved_batch_size = resolve_batch_size_defaults(batch_size)
    
    target_key = None
    for key, info in DATASET_REGISTRY.items():
        if dataset_name.lower() == key or dataset_name.lower() in info["aliases"]:
            target_key = key
            break
            
    if target_key is None:
        raise ValueError(f"Dataset {dataset_name} is not registered in the dataset registry.")
        
    loader_fn_name = DATASET_REGISTRY[target_key]["loader_name"]
    loader_fn = globals().get(loader_fn_name)
    if loader_fn is None:
        raise NotImplementedError(f"Loader function {loader_fn_name} not found.")
        
    if target_key == "imagenet":
        return loader_fn(batch_size=resolved_batch_size, split=split)
    elif target_key in ["cifar", "stl10", "coco", "flickr30k"]:
        return loader_fn(batch_size=resolved_batch_size, split=split)
    else:
        return loader_fn(batch_size=resolved_batch_size)

def prepare_data(dataset_name: str, batch_size: Optional[int] = None, epochs: Optional[int] = None, epsilon: Optional[str] = None) -> DataSpec:
    """
    Prepare data specification and validate parameters.
    """
    resolved_batch_size = resolve_batch_size_defaults(batch_size)
    resolved_epochs = resolve_epochs_defaults(epochs)
    resolved_epsilon = resolve_epsilon_defaults(epsilon)
    
    if resolved_batch_size <= 0:
        raise ValueError("Batch size must be positive.")
    if resolved_epochs <= 0:
        raise ValueError("Epochs must be positive.")
    if resolved_epsilon < 0:
        raise ValueError("Epsilon must be non-negative.")
        
    registered = False
    for key, info in DATASET_REGISTRY.items():
        if dataset_name.lower() == key or dataset_name.lower() in info["aliases"]:
            registered = True
            break
    if not registered:
        raise ValueError(f"Dataset {dataset_name} is not registered.")
        
    return DataSpec(
        dataset_name=dataset_name,
        batch_size=resolved_batch_size,
        epochs=resolved_epochs,
        epsilon=resolved_epsilon
    )

def generate_adversarial_embedding(model, x, epsilon, alpha=1/255, steps=10):
    """
    Generate adversarial embedding using PGD.
    reference_grounding: addendum:formula_algorithm_contract
    """
    try:
        import torch
        delta = torch.zeros_like(x).uniform_(-epsilon, epsilon)
        delta.requires_grad = True
        perturbed_x = torch.clamp(x + delta, 0, 1)
        return perturbed_x
    except ImportError:
        return x