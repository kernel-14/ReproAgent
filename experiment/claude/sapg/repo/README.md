# SAPG: Split and Aggregate Policy Gradients - Reproduction

This repository implements a complete reproduction of the paper **"SAPG: Split and Aggregate Policy Gradients"**, which introduces a new class of on-policy reinforcement learning algorithms that scale to tens of thousands of parallel environments.

## Paper Overview

**Key Contribution**: SAPG learns multiple diverse policies (followers) that each operate on a subset of parallel environments, then aggregates their experience using importance sampling to update a leader policy. This approach overcomes the batch size saturation problem in standard PPO, where performance plateaus despite massive parallelization.

**Core Method**: 
- Split N parallel environments into M policies, each operating on N/M environments
- Each policy has a shared backbone B_θ with local parameters φ_i
- Leader policy aggregates off-policy data from followers using importance sampling
- Achieves superior performance on complex manipulation tasks (Shadow Hand, Allegro Hand)

**Paper Results**:
- **Table 1**: Performance after 2e10 samples across manipulation tasks
- **Figure 5**: SAPG outperforms PPO, PBT, and PQL baselines on AllegroKuka and Shadow Hand tasks
- **Figure 6**: Ablation study showing symmetric aggregation and off-policy combination are critical
- **Figure 7**: State space coverage analysis using PCA reconstruction
- **Figure 8**: State space coverage analysis using MLP reconstruction

### Addendum Clarifications

**Figure 6 Interpretation**: The blue plot is SAPG (our method). Other curves are ablations:
- "Symmetric aggregation": No designated leader; each worker updated with all off-policy data symmetrically
- "No off-policy": SAPG without off-policy data aggregation
- Entropy coefficient variations (0, 0.005, 0.01)
- Off-policy ratio variations (0.1, 0.5, 0.9)

**Figure 8 Network Architecture**: Two-layer MLP with hidden dimensions shown on x-axis. Activation function: ReLU. Optimizer: Adam with PyTorch default hyperparameters (lr=0.001, betas=(0.9, 0.999), eps=1e-8).

## Repository Structure