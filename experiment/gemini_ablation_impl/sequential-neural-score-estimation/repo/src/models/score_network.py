import os
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reference Grounding: paper:unit_002 (target:17)
# Reference Grounding: paper:unit_003 (target:7)
# Reference Grounding: Section 5.1 & Appendix E.3.2 Network Architecture

# Active Route Constants & Defaults
DEFAULT_LEARNING_RATE = 1e-4
learning_rate_values = [1e-4, 5e-4, 1e-3]

def resolve_learning_rate_defaults(lr=None):
    return lr if lr is not None else DEFAULT_LEARNING_RATE

DEFAULT_BATCH_SIZE = 128
batch_size_values = [64, 128, 256]

def resolve_batch_size_defaults(bs=None):
    return bs if bs is not None else DEFAULT_BATCH_SIZE

# Method and Baseline Registries
METHOD_REGISTRY = {
    "ours": "TSNPSE",
    "snpse": "SNPSE",
    "tsnpse": "TSNPSE",
    "diffusion_model": "Conditional Score-based Diffusion",
}

BASELINE_REGISTRY = {
    "npe": "Neural Posterior Estimation",
    "nle": "Neural Likelihood Estimation",
    "nre": "Neural Ratio Estimation",
}

# Canonical Artifact Paths
figure_1 = "results/figures/figure_1.png"
artifact_figure_1 = figure_1
figure_2 = "results/figures/figure_2.png"
artifact_figure_2 = figure_2
figure_3 = "results/figures/figure_3.png"
artifact_figure_3 = figure_3
figure_4 = "results/figures/figure_4.png"
artifact_figure_4 = figure_4
figure_7 = "results/figures/figure_7.png"
artifact_figure_7 = figure_7