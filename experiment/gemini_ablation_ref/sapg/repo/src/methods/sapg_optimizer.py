# src/methods/sapg_optimizer.py
# Reference Grounding: paper_contract_method_baseline_protocol, paper_contract_sweep_hyperparameter_protocol, paper_method_core
# SAPG: Split and Aggregate Policy Gradients Optimizer and Algorithm 1 Implementation

import os
import json
import math
import random
import pathlib
from typing import Dict, Any, List, Tuple, Optional, Union

# Lazy imports for heavy packages to keep the module importable in minimal environments
def _lazy_import_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        return torch, nn, optim
    except ImportError:
        return None, None, None

# Executable constants and sweeps
# reference_grounding: paper_contract_sweep_hyperparameter_protocol
DEFAULT_BATCH_SIZE = 32768
batch_size_values = [8192, 16384, 32768, 65536]

DEFAULT_EPOCHS = 100
epochs_values = [50, 100, 200, 500]

# Sweeps for other parameters
# reference_grounding: chunk_018
M_values = [1, 2, 4, 8]  # Number of policies
sigma_values = [0.0, 0.003, 0.005]  # Entropy coefficients
mu_values = [0.1, 0.5, 1.0, 2.0]  # Importance weight clipping / scaling

def resolve_batch_size_defaults(batch_size: Optional[int] = None) -> int:
    """Resolve batch size default value."""
    if batch_size is None:
        return DEFAULT_BATCH_SIZE
    return batch_size

def resolve_epochs_defaults(epochs: Optional[int] = None) -> int:
    """Resolve epochs default value."""
    if epochs is None:
        return DEFAULT_EPOCHS
    return epochs