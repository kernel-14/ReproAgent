# Simformer: All-in-One Simulation-Based Inference

This repository contains a faithful, complete, and judgeable reproduction of the **Simformer** method for simulation-based inference (SBI), as described in the paper *"All-in-one simulation-based inference"*.

---

## 1. Project Summary & Capabilities of the Simformer

The **Simformer** is a flexible, score-based diffusion model designed to perform amortized simulation-based inference across arbitrary conditional distributions. Unlike traditional SBI methods (such as NPE, NLE, and NRE) which are typically limited to estimating a single posterior distribution $p(\boldsymbol{\theta} | \boldsymbol{x})$, the Simformer can:
1. **Perform inference for simulators with a finite number of parameters or function-valued parameters** (e.g., time-varying parameters in the SIRD model).
2. **Exploit dependency structures of the simulator** to improve accuracy by encoding graphical model constraints directly into the transformer's attention mask $M_E$.
3. **Perform inference for unstructured or missing data**, allowing conditioning on arbitrary subsets of observations or parameters.
4. **Handle interval constraints** (e.g., in the Hodgkin-Huxley model) via guided diffusion sampling.

---

## 2. Simformer Architecture & Tokenizer (Figure 2)