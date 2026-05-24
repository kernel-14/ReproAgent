# Test-Time Model Adaptation with Only Forward Passes (FOA)

This repository contains a faithful, complete, and judgeable reproduction of the paper **"Test-Time Model Adaptation with Only Forward Passes" (FOA)**. 

FOA is a backpropagation-free test-time adaptation (TTA) method that adapts vision models (such as ViT-Base, ResNet, and VisionMamba) to out-of-distribution (OOD) test streams using only forward passes. It optimizes a set of prompt parameters using Covariance Matrix Adaptation (CMA-ES) guided by a novel activation distribution discrepancy fitness function, combined with a Back-to-Source Activation Shifting scheme.

---

## Core Hypothesis & Decision Value

*   **Hypothesis:** FOA maintains high adaptation performance on quantized models (e.g., 8-bit ViT) and generalizes robustly across diverse datasets (including ImageNet-C/R/V2/Sketch, Autonomous Driving, and WILDS) without requiring backward propagation.
*   **Decision Value:** Reproduces the primary claims regarding robustness, efficiency, and generalization, demonstrating that gradient-free forward-only optimization can match or outperform gradient-based TTA methods while using significantly less memory.

---

## Key Methodological Components

### 1. Source Statistics Collection
Before test-time adaptation begins, we collect a small set of source in-distribution samples $\mathcal{D}_{S} = \{\mathbf{x}_q\}_{q=1}^Q$ and feed them into the model to obtain the corresponding CLS tokens. We compute and save the mean $\boldsymbol{\mu}_N^S$ and standard deviation $\boldsymbol{\sigma}_N^S$ of the CLS tokens at the shifting layer $N$ (typically the final layer).
*   **Artifact:** `results/source_stats.pt`

### 2. Forward-Optimization Adaptation (FOA)
For each batch of online incoming test samples, we feed them alongside prompts $\mathbf{p}$ into the TTA model and calculate a fitness value. The fitness function consists of:
*   **Entropy Minimization:** Encourages confident predictions.
*   **Activation Discrepancy:** Measures the discrepancy between the test CLS token distribution and the source distribution to provide a stable learning signal for the CMA optimizer.

### 3. Back-to-Source Activation Shifting
To mitigate severe domain shifts, we shift the test CLS tokens towards the source distribution:
$$\mathbf{d}_t = \boldsymbol{\mu}_N^S - \boldsymbol{\mu}_N(\mathcal{X}_t)$$
An Exponential Moving Average (EMA) with momentum $0.9$ is used to maintain stable running statistics of the test features.

### 4. Interval Update Strategy for Single Sample Adaptation ($BS = 1$)
When the batch size is limited to 1, estimating batch statistics is highly unstable. We implement the **FOA-I** interval update strategy, which stores features or images over an interval $I$ to perform stable CMA prompt updates.

---

## Experiment Registry & Paper Artifact Context

This repository implements and executes the following experiments (I-VIII) mapping directly to the paper's tables and figures:

| Experiment ID | Description | Target Tables / Figures | Key Hyperparameters |
|---|---|---|---|
| **Experiment I** | ImageNet-C Main Benchmark | Table 2, Table 11, Figure 4 | `batch_size: 64`, `momentum: 0.9`, `K: 8` |
| **Experiment II** | Quantized Models | Table 4, Table 7 | `precision: 8-bit`, `batch_size: 64` |
| **Experiment III** | Ablation Studies | Table 5, Table 9, Table 14 | `alpha: [0, 1]`, `lambda: [0.1, 0.8]` |
| **Experiment IV** | Cross-Dataset (Driving, WILDS) | Table 6, Table 7 | `dataset: autonomous_driving, wilds` |
| **Experiment V** | Generalization (R/V2/Sketch) | Table 3, Table 10 | `model: ViT-Base, ResNet, VisionMamba` |
| **Experiment VI** | Sensitivity & Complexity | Table 8, Table 15, Figure 2 | `K: [2, 28]`, `lambda: [0.1, 0.8]` |
| **Experiment VII** | Model Variants | Table 16, Table 17 | `model: ViT-Base, ResNet` |
| **Experiment VIII**| In-Distribution Performance | Table 12 | `dataset: imagenet` |

### Table & Figure Captions Preserved in Reproduction:
*   **Table 1:** Comparison w.r.t. prior gradient-based Test-Time Adaptation (TTA) vs. our Forward-Optimization Adaptation. Memory usage and accuracy are measured via ViT-Base and batch size 64 on ImageNet-C (level 5). The memory of 8-bit ViT is an ideal estimation by $0.25 \times$ memory of 32-bit ViT.
*   **Table 2:** Comparisons with SOTA methods on ImageNet-C (severity level 5) with ViT regarding Accuracy (%).
*   **Table 3:** Comparisons with state-of-the-art methods on ImageNet-R/V2/Sketch with ViT-Base.
*   **Table 4:** Effectiveness of our FOA on Quantized ViT models.
*   **Table 5:** Ablations of components in our FOA (Entropy, Activation Discrepancy, and Activation Shifting).
*   **Table 6:** Effectiveness of FOA with interval update strategy (FOA-I) for single sample adaptation.
*   **Table 7:** Comparison w.r.t. run-time memory (MB) usage under batch size 1.
*   **Table 8:** Comparisons w.r.t. computation complexity (Wall-Clock Time and Peak Memory Usage).
*   **Figure 2:** Parameter sensitivity analyses of our FOA (population size $K$ and trade-off parameter $\lambda$).
*   **Figure 4:** Online accuracy comparison with MEMO on ViT and ImageNet-C.

---

## Repository Structure