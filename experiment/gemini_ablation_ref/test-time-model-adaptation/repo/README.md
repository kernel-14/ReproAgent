# Test-Time Model Adaptation with Only Forward Passes (FOA)

This repository contains a faithful, complete, and runnable reproduction of the paper **"Test-Time Model Adaptation with Only Forward Passes" (FOA)**. 

## 1. Core Hypothesis & Method Overview

### Hypothesis
A derivative-free optimizer (specifically Covariance Matrix Adaptation Evolution Strategy, or **CMA-ES**) combined with input prompt tuning and back-to-source activation shifting can adapt pre-trained models (such as ViT and ResNet) to out-of-distribution (OOD) target domains at test time using **only forward passes**, achieving state-of-the-art accuracy and calibration while avoiding the high memory footprint, instability, and security risks of backward propagation.

### Method Description (FOA)
FOA performs adaptation at two distinct levels:
1. **Input Level (Forward-Only Prompt Adaptation)**:
   - We insert a small set of learnable prompt embeddings $\mathbf{p}$ into the input sequence of a Vision Transformer (ViT) or as input-level perturbations for ResNet.
   - The prompts are optimized online using **CMA-ES** to minimize a specially designed unsupervised fitness function.
   - The fitness function combines entropy minimization and activation distribution discrepancy to provide stable learning signals:
     $$\mathcal{L}(f_{\Theta}(\mathcal{X}_t, \mathbf{p})) = \mathcal{H}(f_{\Theta}(\mathcal{X}_t, \mathbf{p})) + \lambda \sum_{i=1}^{N} \mathcal{D}_{\text{discrepancy}}(\mathbf{e}_i^0(\mathcal{X}_t, \mathbf{p}))$$
     where $\mathcal{H}$ is the prediction entropy, $\mathbf{e}_i^0$ is the classification token ([CLS]) at layer $i$, and $\mathcal{D}_{\text{discrepancy}}$ measures the discrepancy between target test statistics and source in-distribution statistics:
     $$\mathcal{D}_{\text{discrepancy}}(\mathbf{e}_i^0) = \|\boldsymbol{\mu}_i(\mathcal{X}_t) - \boldsymbol{\mu}_i^S\|_2^2 + \|\boldsymbol{\sigma}_i(\mathcal{X}_t) - \boldsymbol{\sigma}_i^S\|_2^2$$

2. **Output Feature Level (Back-to-Source Activation Shifting)**:
   - To further align the OOD features, we dynamically shift the final layer activation features $\mathbf{e}_N^0$ back toward the source in-distribution mean $\boldsymbol{\mu}_N^S$:
     $$\mathbf{d}_t = \boldsymbol{\mu}_N^S - \boldsymbol{\mu}_N(t)$$
     $$\boldsymbol{\mu}_N(t) = \alpha \boldsymbol{\mu}_N(\mathcal{X}_t) + (1 - \alpha) \boldsymbol{\mu}_N(t-1)$$
     $$\mathbf{e}_N^0 \leftarrow \mathbf{e}_N^0 + \mathbf{d}_t$$
   - This shifting aligns the target domain features with the source classifier head without modifying any model weights.

---

## 2. Repository Structure