# src/models/forecaster.py
# Grounding Marker: reference_grounding: paper_contract_method_baseline_protocol

import os
import json
import math
import random
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union

# -------------------------------------------------------------------------
# Executable Constants and Sweeps
# -------------------------------------------------------------------------
# Active route contract: define DEFAULT_LEARNING_RATE in src/models/forecaster.py
DEFAULT_LEARNING_RATE = 1e-5
learning_rate_values = [1e-5, 3e-5, 5e-5]

DEFAULT_BATCH_SIZE = 8
batch_size_values = [4, 8, 16]

DEFAULT_ALPHA = 0.1
alpha_values = [0.1, 0.5, 1.0]

DEFAULT_GAMMA = 0.5
gamma_values = [0.1, 0.3, 0.5, 0.7, 0.9]

DEFAULT_NUM_LAYERS = 2
num_layers_values = [1, 2, 3]

# LoRA configuration from addendum
LORA_CONFIG_DEFAULT = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "task_type": "SEQ_2_SEQ_LM",
    "inference_mode": False,
    "target_modules": ["q", "v"]
}

# -------------------------------------------------------------------------
# Default Accessors / Resolvers
# -------------------------------------------------------------------------
def resolve_learning_rate_defaults(lr: Optional[float] = None) -> float:
    return lr if lr is not None else DEFAULT_LEARNING_RATE

def resolve_batch_size_defaults(bs: Optional[int] = None) -> int:
    return bs if bs is not None else DEFAULT_BATCH_SIZE