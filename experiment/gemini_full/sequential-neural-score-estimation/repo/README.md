# Sequential Neural Score Estimation (SNPSE/TSNPSE) Reproduction

This repository provides a faithful, complete, and judgeable reproduction of the methods, experiments, and artifacts presented in the paper: **"Sequential Neural Score Estimation: Likelihood-Free Inference with Conditional Score Based Diffusion Models"**.

---

## 1. Overview & Core Hypothesis

The core contribution of this work is **Sequential Neural Score Estimation (SNSE)** and its variants, **Neural Posterior Score Estimation (NPSE)** and **Truncated Sequential Neural Posterior Score Estimation (TSNPSE)**. These methods leverage conditional score-based diffusion models to perform simulation-based inference (SBI) without requiring tractable likelihoods.

### Core Hypothesis
The full reproduction pipeline can generate the C2ST metrics, posterior plots, and model checkpoints required to validate the paper's claims. Specifically, SNPSE/TSNPSE should outperform or match standard normalising flow-based baselines (NPE, NLE, NRE) on complex, multi-modal posteriors such as the **Simple Likelihood Complex Posterior (SLCP)** and **Lotka-Volterra** tasks.

---

## 2. Repository Structure

The repository is organized as follows: