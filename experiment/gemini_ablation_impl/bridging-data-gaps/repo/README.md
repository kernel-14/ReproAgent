# Bridging Data Gaps in Diffusion Models with Adversarial Noise-Based Transfer Learning (DPMs-ANT)

This repository contains a faithful, complete, and judgeable reproduction of the paper **"Bridging Data Gaps in Diffusion Models with Adversarial Noise-Based Transfer Learning"** (DPMs-ANT). 

The implementation covers the core algorithmic contributions, including **Similarity-Guided Training (SGT)**, **Adversarial Noise Selection (ANS)**, and the parameter-efficient **Adaptor Module ($\psi$)** fine-tuning protocol, alongside the 2D Gaussian toy experiment and the 10-shot image domain transfer experiments.

---

## 1. Core Methodology

DPMs-ANT addresses the challenge of few-shot transfer learning in Diffusion Models (DDPM/LDM) by bridging the data gap between the source domain (e.g., FFHQ, LSUN Church) and a target domain represented by only 10 images (e.g., Sunglasses, Babies, Sketches).

### 1.1 Similarity-Guided Training (SGT)
SGT guides the diffusion model's generation process using a target classifier trained on the target domain. The similarity-guided loss is defined as:
$$\mathcal{L}_{\text{SGT}} = \mathcal{L}_{\text{diff}} - \gamma \log p_{\phi}(y = \text{target} \mid x_t)$$
where $\gamma$ is the similarity guidance scale (default: `5.0`).

### 1.2 Adversarial Noise Selection (ANS)
ANS selects adversarial noise inputs to expose and correct the model's weaknesses on the target domain. The noise perturbation is optimized via:
$$\epsilon^* = \arg\max_{\epsilon} \mathcal{L}_{\text{SGT}}(x_t(\epsilon), \theta, \psi)$$
subject to $||\epsilon^* - \epsilon||_{\infty} \le \omega$ (default $\omega = 0.02$, optimized over `10` inner steps).

### 1.3 Parameter-Efficient Adaptor ($\psi$)
Instead of fine-tuning the entire network $\theta$, DPMs-ANT introduces a lightweight Adaptor module $\psi$ inserted into the attention layers of the U-Net. During transfer learning, $\theta$ remains frozen, and only $\psi$ is updated, significantly reducing GPU memory consumption and preventing catastrophic forgetting.

---

## 2. Experimental Setup & Hyperparameters

Following the exact settings of **DDPM-PA** (Zhu et al., 2022), the experiments are configured as follows:

*   **Few-Shot Setting**: Exactly `10-shot` target datasets.
*   **Fine-Tuning Iterations**: `300` iterations.
*   **Batch Size**: `64`.
*   **Optimizer**: Adam with a learning rate of `1e-4` for the target classifier and `5e-5` for the adaptor.
*   **Classifier Training**: Pre-trained classifiers (e.g., `256x256_classifier.pt` for DDPM, `64x64_classifier.pt` for LDM) are fine-tuned by modifying the last layer to output two classes (source vs. target) using Adam, learning rate `1e-4`, batch size `64`, and trained for `300` iterations.

---

## 3. Command Line Interface & Execution

The repository provides a unified entrypoint `main.py` to run both smoke tests and full-scale experiments.

### 3.1 Bounded Smoke Run (Runtime Smoke)
To verify the entire pipeline, environment setup, and artifact generation paths in a lightweight manner: