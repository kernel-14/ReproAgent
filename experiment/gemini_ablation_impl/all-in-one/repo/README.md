# Simformer: All-in-one Simulation-Based Inference

This repository provides a faithful reproduction of the **Simformer**, as described in the paper *"All-in-one simulation-based inference"*. Simformer is a transformer-based architecture for simulation-based inference (SBI) that leverages diffusion generative modeling to learn arbitrary conditional distributions of parameters and data.

## Key Capabilities
- **Universal Amortization**: Performs inference for simulators with finite or function-valued parameters.
- **Structured Inference**: Exploits simulator dependency structures via custom attention masks to improve accuracy.
- **Unstructured Data**: Handles missing data, irregular time points, and varying numbers of observations.
- **Arbitrary Conditionals**: Estimates any conditional distribution $p(\mathbf{x}_A \mid \mathbf{x}_B)$ where $\mathbf{x}$ includes both parameters $\theta$ and data $x$.

## Installation