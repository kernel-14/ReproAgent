# Refined Coreset Selection (LBCS) Reproduction

This repository provides a faithful reproduction of the paper: **"Refined Coreset Selection: Towards Minimal Coreset Size under Model Performance Constraints"**.

The core contribution is the **Lexicographic Bilevel Coreset Selection (LBCS)** algorithm, which treats coreset selection as a lexicographic optimization problem. It prioritizes model performance ($f_1$) within a tolerance $\epsilon$ while minimizing the coreset size ($f_2$).

## Project Architecture

The repository is organized to separate the core optimization logic from data handling and baseline comparisons:

- `main.py`: Canonical entrypoint for all experiments.
- `src/methods/lbcs.py`: Implementation of the LBCS algorithm and lexicographic preference logic.
- `src/methods/baselines.py`: Implementation of standard coreset baselines (Uniform, EL2N, GraNd, etc.).
- `src/lbcs/engine.py`: Orchestration of the bilevel optimization loop (inner training, outer mask update).
- `src/lbcs/data/pipeline.py`: Data loading and noise injection for F-MNIST, CIFAR, and SVHN.
- `src/models/networks.py`: Model architectures (ResNet-18, ResNet-50, CNN).
- `config/default.yaml`: Centralized registry for hyperparameters and experiment specifications.
- `results/`: Directory containing generated artifacts, metrics, and evidence matrices.

## Setup

### Requirements
- Python 3.8+
- PyTorch >= 1.10
- torchvision
- numpy, yaml, matplotlib

### Installation