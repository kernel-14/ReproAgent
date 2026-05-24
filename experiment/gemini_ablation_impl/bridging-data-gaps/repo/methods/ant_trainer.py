import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Optional, Callable

# ==============================================================================
# 1. Paper Evidence Contract: Fixed Hyperparameters & Sweeps
# ==============================================================================

# Reference Grounding: Section 5.2 Experimental Setup
DEFAULT_LEARNING_RATE = 5e-5
DEFAULT_BATCH_SIZE = 64
DEFAULT_GAMMA = 5.0
DEFAULT_OMEGA = 0.02
DEFAULT_NUM_STEPS = 300
ADVERSARIAL_INNER_STEPS = 10

learning_rate_values = [5e-6, 1e-5, 5e-5, 1e-4]
batch_size_values = [16, 32, 64, 128]
gamma_values = [1.0, 3.0, 5.0, 7.0, 9.0]
num_steps_values = [0, 50, 100, 150, 200, 250, 300, 350]
adversarial_noise_scale_values = [0.01, 0.02, 0.03, 0.04, 0.05]

def resolve_learning_rate_defaults(config: Dict[str, Any]) -> float:
    return config.get("learning_rate", DEFAULT_LEARNING_RATE)

def resolve_batch_size_defaults(config: Dict[str, Any]) -> int:
    return config.get("batch_size", DEFAULT_BATCH_SIZE)

def resolve_gamma_defaults(config: Dict[str, Any]) -> float:
    return config.get("gamma", DEFAULT_GAMMA)

def resolve_num_steps_defaults(config: Dict[str, Any]) -> int:
    return config.get("num_steps", DEFAULT_NUM_STEPS)

# ==============================================================================