# Challenges in Training PINNs: A Loss Landscape Perspective

This repository provides a faithful, complete, and judgeable reproduction of the methods, experiments, and findings presented in the paper *"Challenges in Training PINNs: A Loss Landscape Perspective"*.

---

## 1. Overview & Core Hypotheses

Physics-Informed Neural Networks (PINNs) often suffer from severe training difficulties due to highly non-convex and ill-conditioned loss landscapes. This repository implements the core methodologies proposed and analyzed in the paper:
1. **Hybrid Adam+L-BFGS Optimization**: Combines the robust initial progress of Adam with the rapid local convergence of L-BFGS.
2. **Per-Sample Lowest Score Selection Protocol**: A robust selection protocol to choose the best model across multiple random seeds and hyperparameters based on the lowest training loss, which strongly correlates with the lowest $L_2$ Relative Error ($L_2\text{RE}$).
3. **NysNewton-CG (NNCG)**: An advanced randomized preconditioned conjugate gradient Newton method designed to overcome the stalling of L-BFGS in under-optimized regimes.
4. **Loss Landscape Diagnostics**: Tools to compute the spectral density of the Hessian and preconditioned Hessian to analyze ill-conditioning.

### Core Hypothesis
*Adam+L-BFGS with per-sample lowest score selection achieves lower loss and $L_2\text{RE}$ than standalone optimizers, and NNCG fine-tuning successfully overcomes the stalling of L-BFGS by leveraging randomized Nyström preconditioning.*

---

## 2. Setup & Installation

### Prerequisites
- Python 3.8+
- PyTorch (optional but recommended for full execution; a lightweight mock/fallback is provided for static import and smoke testing)
- NumPy, SciPy, Matplotlib, PyYAML

### Installation