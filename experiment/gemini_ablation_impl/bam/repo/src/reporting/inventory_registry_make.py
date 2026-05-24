import os
import json
import numpy as np

# reference_grounding: paperbench_ref_005 posterior_database/data/info/nes_logit_data.info.json
# reference_grounding: paperbench_ref_008 docs/jep/28661-jax-array-protocol.md

# Paper-derived batch size defaults. 
# Figure 5.1: ADVI, Score, Fisher, and GSM use B=2.
# Figure 5.2: ADVI, Score, Fisher, and GSM use B=5.
# Figure 5.3: B=8 and B=32.
DEFAULT_BATCH_SIZE = 32
batch_size_values = [2, 5, 8, 32]

def resolve_batch_size_defaults(config):
    """
    Resolves batch size from config or returns default.
    """
    return config.get("batch_size", DEFAULT_BATCH_SIZE)

def compute_accuracy(predictions, targets):
    """
    Computes accuracy.
    Canonical identifier: metric_accuracy
    """
    if predictions is None or targets is None:
        return 0.0
    preds = np.array(predictions)
    targs = np.array(targets)
    if preds.size == 0:
        return 0.0
    return float(np.mean(preds == targs))

def aggregate_accuracy(results):
    """
    Aggregates accuracy across runs.
    """
    if not results:
        return 0.0
    return float(np.mean(results))

def compute_loss(predictions, targets):
    """
    Computes loss.
    Canonical identifier: metric_loss
    """
    if predictions is None or targets is None:
        return 0.0
    preds = np.array(predictions)
    targs = np.array(targets)
    if preds.size == 0:
        return 0.0
    return float(np.mean((preds - targs)**2))