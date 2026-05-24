# Simformer: All-in-One Simulation-Based Inference

This repository contains a faithful, complete, and judgeable reproduction of **Simformer: All-in-one simulation-based inference** (excluding the blacklisted `mackelab/simformer` repository). 

Simformer is a transformer-based score-matching diffusion model designed for Simulation-Based Inference (SBI). It reduces all variables (parameters $\boldsymbol{\theta}$ and data $\boldsymbol{x}$) to a token representation, allowing arbitrary conditional distributions to be estimated using a single trained model.

---

## 1. Capabilities & Architecture

### Core Capabilities
- **Arbitrary Conditionals**: Simformer can perform inference for simulators with a finite number of parameters or function-valued parameters (e.g., SIRD model).
- **Dependency Exploitation**: It exploits dependency structures of the simulator via structured attention masks to improve accuracy.
- **Unstructured/Missing Data**: It performs inference for unstructured or missing data (e.g., Lotka-Volterra with random observation times).
- **Interval Constraints**: It supports interval-constrained inference using guided diffusion sampling (e.g., Hodgkin-Huxley energy constraints).

### Figure 2: Simformer Architecture
All variables (parameters and data) are reduced to a token representation which includes:
1. **Variable Identity**: The index or type of the variable.
2. **Variable Value (`val`)**: The continuous value of the variable.
3. **Conditional State**: Latent ($L$) or Conditioned ($C$).

This sequence of tokens is processed by a transformer model where the interaction of variables is explicitly controlled through an attention mask.

---

## 2. Supported Tasks & Environments

1. **Two Moons** (Figure 3): Estimating arbitrary conditional distributions.
2. **Benchmark Tasks** (Figure 4): Standard SBI benchmarks (Gaussian Linear, Two Moons, SIRD, Lotka-Volterra, Hodgkin-Huxley).
3. **Lotka-Volterra** (Figure 5): Inference with unstructured observations (prey population density at random time points).
4. **SIRD Model** (Figure 6): Inference of $\infty$-dimensional parameter spaces (time-dependent parameters $\beta(t)$).
5. **Hodgkin-Huxley** (Figure 7): Biophysical model with guided diffusion under metabolic energy constraints.

---

## 3. Setup & Installation

### Installation Commands
To install the core lightweight dependencies: