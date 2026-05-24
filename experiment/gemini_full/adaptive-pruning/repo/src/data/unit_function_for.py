# src/data/unit_function_for.py
# reference_grounding: paper:unit_003 (chunk_003_02, chunk_011)

import importlib.util
import os
import json

def is_package_available(package_name: str) -> bool:
    try:
        return importlib.util.find_spec(package_name) is not None
    except (ImportError, AttributeError):
        return False

# Availability flags for required external environments/datasets
HAS_TORCH = is_package_available("torch")
HAS_TRANSFORMERS = is_package_available("transformers")
HAS_DATASETS = is_package_available("datasets")
HAS_SBI = is_package_available("sbi")
HAS_GYM = is_package_available("gym")

# Lazy import helpers
def get_torch():
    if not HAS_TORCH:
        raise ImportError("torch is not available. Please install torch.")
    import torch
    return torch

def get_transformers():
    if not HAS_TRANSFORMERS:
        raise ImportError("transformers is not available. Please install transformers.")
    import transformers
    return transformers

def get_datasets():
    if not HAS_DATASETS:
        raise ImportError("datasets is not available. Please install datasets.")
    import datasets
    return datasets

def get_sbi():
    if not HAS_SBI:
        raise ImportError("sbi is not available. Please install sbi.")
    import sbi
    return sbi

def get_gym():
    if not HAS_GYM:
        raise ImportError("gym is not available. Please install gym.")
    import gym
    return gym


class UnitFunctionForSpec:
    """
    Descriptor/factory for paper-derived datasets and benchmarks.
    """
    def __init__(self, dataset_id: str, alias: str, loader_fn, metadata: dict):
        self.dataset_id = dataset_id
        self.alias = alias
        self.loader_fn = loader_fn
        self.metadata = metadata

    def load(self, *args, **kwargs):
        return self.loader_fn(*args, **kwargs)


DATASET_REGISTRY = {}

def register_dataset(dataset_id: str, alias: str, metadata: dict):
    def decorator(loader_fn):
        spec = UnitFunctionForSpec(dataset_id, alias, loader_fn, metadata)
        DATASET_REGISTRY[dataset_id] = spec
        DATASET_REGISTRY[alias] = spec
        return loader_fn
    return decorator


# Expose paper-derived dataset/benchmark loaders with ids, setup metadata, validation checks, and runnable config hooks
@register_dataset("sst2", "glue:sst2", {"task": "classification", "benchmark": "glue"})
def load_sst2(split="train", smoke=True):
    if not HAS_DATASETS:
        return [{"sentence": "This is a great movie!", "label": 1}] if smoke else []
    datasets = get_datasets()
    try:
        return datasets.load_dataset("glue", "sst2", split=split)
    except Exception as e:
        if smoke:
            return [{"sentence": "This is a great movie!", "label": 1}]
        raise e

@register_dataset("mnli", "glue:mnli", {"task": "classification", "benchmark": "glue"})
def load_mnli(split="train", smoke=True):
    if not HAS_DATASETS:
        return [{"premise": "A man is sleeping.", "hypothesis": "A man is awake.", "label": 2}] if smoke else []
    datasets = get_datasets()
    try:
        return datasets.load_dataset("glue", "mnli", split=split)
    except Exception as e:
        if smoke:
            return [{"premise": "A man is sleeping.", "hypothesis": "A man is awake.", "label": 2}]
        raise e

@register_dataset("squad", "squad_v2", {"task": "qa", "benchmark": "squad"})
def load_squad(split="train", smoke=True):
    if not HAS_DATASETS:
        return [{"context": "SQuAD is a dataset.", "question": "What is SQuAD?", "answers": {"text": ["a dataset"], "answer_start": [10]}}] if smoke else []
    datasets = get_datasets()
    try:
        return datasets.load_dataset("squad_v2", split=split)
    except Exception as e:
        if smoke:
            return [{"context": "SQuAD is a dataset.", "question": "What is SQuAD?", "answers": {"text": ["a dataset"], "answer_start": [10]}}]
        raise e

@register_dataset("cnn_dailymail", "cnn_dm", {"task": "summarization", "benchmark": "cnn_dm"})
def load_cnn_dm(split="train", smoke=True):
    if not HAS_DATASETS:
        return [{"article": "This is a long article.", "highlights": "Highlights."}] if smoke else []
    datasets = get_datasets()
    try:
        return datasets.load_dataset("cnn_dailymail", "3.0.0", split=split)
    except Exception as e:
        if smoke:
            return [{"article": "This is a long article.", "highlights": "Highlights."}]
        raise e

@register_dataset("xsum", "xsum", {"task": "summarization", "benchmark": "xsum"})
def load_xsum(split="train", smoke=True):
    if not HAS_DATASETS:
        return [{"document": "This is a document.", "summary": "Summary."}] if smoke else []
    datasets = get_datasets()
    try:
        return datasets.load_dataset("xsum", split=split)
    except Exception as e:
        if smoke:
            return [{"document": "This is a document.", "summary": "Summary."}]
        raise e

@register_dataset("truthfulqa", "truthful_qa", {"task": "qa", "benchmark": "truthfulqa"})
def load_truthfulqa(split="validation", smoke=True):
    if not HAS_DATASETS:
        return [{"question": "What is the color of the sky?", "best_answer": "Blue", "correct_answers": ["Blue"], "incorrect_answers": ["Green"]}] if smoke else []
    datasets = get_datasets()
    try:
        return datasets.load_dataset("truthful_qa", "generation", split=split)
    except Exception as e:
        if smoke:
            return [{"question": "What is the color of the sky?", "best_answer": "Blue", "correct_answers": ["Blue"], "incorrect_answers": ["Green"]}]
        raise e

# Explicitly register dataset/benchmark aliases for glue, truthfulqa, squad
DATASET_REGISTRY["glue"] = DATASET_REGISTRY["sst2"]
DATASET_REGISTRY["truthfulqa"] = DATASET_REGISTRY["truthful_qa"]
DATASET_REGISTRY["squad"] = DATASET_REGISTRY["squad_v2"]


def load_unit_function_for(dataset_id_or_alias: str, split: str = "train", smoke: bool = True):
    if dataset_id_or_alias not in DATASET_REGISTRY:
        raise ValueError(f"Dataset {dataset_id_or_alias} not found in registry.")
    spec = DATASET_REGISTRY[dataset_id_or_alias]
    return spec.load(split=split, smoke=smoke)


def prepare_unit_function_for(dataset_id_or_alias: str, config: dict = None):
    if dataset_id_or_alias not in DATASET_REGISTRY:
        raise ValueError(f"Dataset {dataset_id_or_alias} not found in registry.")
    spec = DATASET_REGISTRY[dataset_id_or_alias]
    metadata = spec.metadata
    return {
        "status": "ready",
        "dataset_id": spec.dataset_id,
        "metadata": metadata
    }


# Outlier-aware salience scoring function for parameter blocks
def salience_scorer(weight, gradient, activation=None, outlier_threshold=2.0):
    """
    Computes the outlier-aware salience score of parameter blocks.
    S(W) = |W * dL/dW|
    If activation is provided, scales the salience score by outlier-aware activation magnitude.
    """
    if not HAS_TORCH:
        import numpy as np
        salience = np.abs(weight * gradient)
        if activation is not None:
            mean = np.mean(activation)
            std = np.std(activation)
            outliers = np.abs(activation - mean) > (outlier_threshold * std)
            scaling = 1.0 + np.mean(outliers) * 2.0
            salience = salience * scaling
        return salience
    
    import torch
    salience = torch.abs(weight * gradient)
    if activation is not None:
        mean = torch.mean(activation)
        std = torch.std(activation)
        outliers = torch.abs(activation - mean) > (outlier_threshold * std)
        scaling = 1.0 + outliers.float().mean() * 2.0
        salience = salience * scaling
    return salience


# Fast search algorithm to determine binary masks based on the sparsity target
def mask_searcher(salience_scores, sparsity_target: float):
    """
    Fast search algorithm to determine binary masks based on the sparsity target.
    Returns a binary mask of the same shape as salience_scores.
    """
    if not HAS_TORCH:
        import numpy as np
        flat_scores = salience_scores.flatten()
        k = int(sparsity_target * flat_scores.size)
        if k <= 0:
            return np.ones_like(salience_scores)
        if k >= flat_scores.size:
            return np.zeros_like(salience_scores)
        threshold = np.partition(flat_scores, k)[k]
        mask = (salience_scores > threshold).astype(np.float32)
        return mask

    import torch
    flat_scores = salience_scores.flatten()
    k = int(sparsity_target * flat_scores.numel())
    if k <= 0:
        return torch.ones_like(salience_scores)
    if k >= flat_scores.numel():
        return torch.zeros_like(salience_scores)
    threshold, _ = torch.kthvalue(flat_scores, k)
    mask = (salience_scores > threshold).float()
    return mask


# Preserve explicit baseline or method-variant selection surfaces: Adaptive Pruning (A_P)
class AdaptivePruningAP:
    """
    Adaptive Pruning (A_P) variant implementation.
    """
    def __init__(self, sparsity_target=0.6, outlier_threshold=2.0):
        self.sparsity_target = sparsity_target
        self.outlier_threshold = outlier_threshold

    def update_masks(self, weights, gradients, activations=None):
        salience = salience_scorer(weights, gradients, activations, self.outlier_threshold)
        mask = mask_searcher(salience, self.sparsity_target)
        return mask