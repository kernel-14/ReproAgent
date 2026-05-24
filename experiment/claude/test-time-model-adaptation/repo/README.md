# Test-Time Model Adaptation with Only Forward Passes

Implementation of the paper "Test-Time Model Adaptation with Only Forward Passes" - a PaperBench code reproduction repository.

This repository provides a complete, canonical implementation of Forward-Only Adaptation (FOA), a test-time adaptation method that eliminates backward propagation while maintaining competitive accuracy with gradient-based approaches.

## Overview

**Forward-Only Adaptation (FOA)** introduces a novel test-time adaptation framework that:
- Eliminates gradient computation and backward propagation
- Uses CMA-ES optimizer with prompt adaptation
- Implements back-to-source activation shifting
- Achieves competitive accuracy with 50-75% lower memory usage vs. gradient-based TTA methods
- Works with quantized models (8-bit, 4-bit) without gradient requirements

### Key Components

1. **FOA Method** (`src/methods.py`): Core forward-only adaptation algorithm with CMA-ES prompt optimization
2. **Activation Shifting** (`src/methods.py`): Back-to-source activation alignment mechanism with EMA
3. **Baseline Methods** (`src/baselines.py`): TENT, CoTTA, SAR, T3A, LAME, and other TTA baselines
4. **Environment Registry** (`src/environments.py`): ImageNet-C/R/V2/Sketch datasets and benchmarks
5. **Evaluation Framework** (`src/evaluation.py`): Protocol matrix and measurement schemas
6. **Artifact Writers** (`src/artifacts.py`): Result generation for all paper tables and figures

## Paper Artifacts

This implementation reproduces the following paper experiments:

- **Table 1**: Memory and accuracy comparison (FOA vs. gradient-based TTA)
- **Table 2**: ImageNet-C comparisons with SOTA methods (ViT-Base, severity 5)
- **Table 3**: ImageNet-R/V2/Sketch robustness evaluation
- **Table 4**: Quantized ViT models (8-bit, 4-bit) effectiveness
- **Figure 2**: Parameter sensitivity analysis (population size, prompt count, λ)
- **Table 5**: Component ablations (entropy, activation discrepancy, shifting)
- **Table 6**: Interval update strategy for single-sample adaptation
- **Table 7**: Runtime memory usage comparison across batch sizes
- **Table 8**: Computation complexity (forward/backward pass counts, wall-clock time)
- **Table 9**: Design choices (learnable parameters, optimizers, loss functions)
- **Table 10**: Architecture variants (ResNet, Vision Mamba)
- **Table 11**: Non-i.i.d. scenarios (imbalanced shifts, mixed corruptions)
- **Table 12**: In-distribution performance on clean ImageNet
- **Figure 3**: Dataset visualizations
- **Figure 4**: Online accuracy evolution

## Installation

### Requirements