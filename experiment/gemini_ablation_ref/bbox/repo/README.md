# BBox-Adapter: Lightweight Adapting for Black-Box Large Language Models

This repository contains a faithful, complete, and judgeable reproduction of the paper **"BBox-Adapter: Lightweight Adapting for Black-Box Large Language Models"**.

---

## 1. Project Overview & Core Hypothesis

### Core Hypothesis
Decomposing generation into sentence-level steps allows a lightweight adapter (e.g., RoBERTa-base/large) to steer a black-box Large Language Model (LLM) towards target domains (such as mathematical reasoning, implicit reasoning, truthfulness, scientific QA, and toxicity mitigation) without requiring access to the black-box LLM's internal parameters or token-level output probabilities.

### Architecture & Adaptation Categorization
*   **Figure 1. Illustration of white-box, grey-box, and black-box LLM adaptation.** 
    *   *White-box* has complete access to both model parameters and output probabilities.
    *   *Grey-box* has access only to output probabilities.
    *   *Black-box* lacks access to both.
    *   $\theta$ indicates the models with trainable parameters, whereas the base model is inactive/frozen.
*   **Figure 2. Overview of BBox-ADAPTER for black-box LLM adaptation from the source to the target domain.**
    *   BBox-Adapter adopts an online adaptation framework, iteratively sampling from previous inferences and updating the adapter using a ranking-based Noise Contrastive Estimation (NCE) loss.

---

## 2. Evidence Obligation Matrix

The following matrix maps the paper's key claims, datasets, metrics, and methods to their concrete artifact paths in this reproduction repository:

| Category | Item / Claim | Target Artifact Path | Description |
| :--- | :--- | :--- | :--- |
| **Method** | BBox-Adapter | `checkpoints/adapter.pth` | Core adapter model checkpoint. |
| **Variant** | AI Feedback | `checkpoints/adapter_ai.pth` | Adapter trained with AI feedback. |
| **Ablation** | Ranking-based NCE vs MLM | `results/metrics.json` | Performance comparison of NCE vs MLM loss. |
| **Experiment** | Main Results (Table 2) | `results/metrics.json` | Accuracy metrics across downstream tasks. |
| **Dataset** | ToxiGen | `results/dataset_registry.json` | Dataset registry metadata for ToxiGen. |
| **Metric** | Toxicity | `results/metrics.json` | Toxicity scores for the ToxiGen dataset. |
| **Method** | single_step_inference | `results/metrics.json` | Evaluation metrics for single-step inference. |
| **Method** | full_step_inference | `results/metrics.json` | Evaluation metrics for full-step inference. |
| **Experiment** | Cost Analysis (Table 4) | `results/table_4_cost.csv` | Training and inference cost comparison. |
| **Experiment** | Toxicity Analysis (Table 9) | `results/table_9.csv` | Toxicity mitigation results. |
| **Hyperparameter** | `nearest_neighbor_upsample` | `results/config_resolved.json` | Resolved configuration parameters. |
| **Sweep** | `epochs` | `results/training_trace.json` | Training trace across epochs. |

---

## 3. Configuration & Setup

### Configuration Flags & Parameters
All hyperparameters and execution modes are exposed in `configs/default.yaml`. Key parameters include:
*   `beam_size` ($k$): Configurable beam size for sentence-level beam search (e.g., `1`, `3`, `5`).
*   `inference_mode`: Supports `single_step_inference` and `full_step_inference`.
*   `adapter_size`: Supports `0.1B` (RoBERTa-base) and `0.3B` (RoBERTa-large).
*   `spectral_normalization`: Implemented as $\ell_2$ regularization of the energies:
    $$\alpha \mathbb{E}[g_\theta(x, y_+)^2] + \alpha \mathbb{E}[g_\theta(x, y_-)^2]$$
    where $\alpha = 0.01$ by default.

### Setup Commands
To set up the environment and run the reproduction pipeline: