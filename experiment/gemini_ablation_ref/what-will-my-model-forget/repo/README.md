# What Will My Model Forget? Forecasting Forgotten Examples in Language Model Refinement

This repository contains the faithful, complete, and judgeable reproduction of the paper: **"What Will My Model Forget? Forecasting Forgotten Examples in Language Model Refinement"**.

---

## 1. Overview & Core Contribution

During language model refinement (e.g., fine-tuning or editing to correct prediction errors), models often suffer from catastrophic forgetting of previously learned upstream knowledge. This project implements and evaluates methods to **forecast** which upstream examples will be forgotten when a model is updated on a new online learning example.

### Core Hypothesis & Decision Value
- **Hypothesis**: By utilizing a unified dataset loader and model wrapper, we can standardize input/output interfaces across diverse experimental setups (BART-Large, FLAN-T5-Large, FLAN-T5-3B) and accurately forecast forgetting.
- **Decision Value**: Standardizing these interfaces ensures experiment reproducibility and enables practical utility via targeted replay of examples forecasted to be forgotten, significantly reducing Exact Match (EM) Drop Ratio.

---

## 2. Paper-Derived Evidence Obligation Matrix

### Experiment I: Data Loading
- **Upstream Pre-training Dataset ($D_{\text{PT}}$ / $\hat{D}_{\text{PT}}$)**: Constructed from 36 tasks from the training split of the Public Pool of Prompts (P3) dataset, using a balanced sample of 100 examples per task (totaling 3,600 examples). As clarified in the addendum, all forecasting algorithms are evaluated on $\hat{D}_{\text{PT}}$ (the subset of pre-training examples that the base model initially predicted correctly).
- **Online Learning / Refinement Dataset ($D_{\text{R}}$)**: Sequential stream of examples where the base model makes prediction errors.
- **Evaluation Dataset (P3-Test)**: Split into In-Domain (ID) and Out-of-Domain (OOD) tasks to evaluate generalization.
- **Downstream Benchmarks**: SQuAD and GLUE tasks are supported to evaluate general task coverage.

### Experiment II: Forecasting Methods
We implement the following forecasting methods:
1. **Frequency-Threshold based Forecasting (Threshold)**:
   $$g(\langle x_i, y_i \rangle, \langle x_j, y_j \rangle) = \mathbb{1}\left[ \left| \{ j \in 1..J \mid z_{ij} = 1 \} \right| \geq \gamma \right]$$
   where $J = |D_{\text{PT}}|$ and $z_{ij}$ is the binary indicator of ground truth forgetting.
2. **Trainable Logit-Change based Forecasting (Trainable Logit)**:
   Learns an encoding function $h: \mathbb{R}^T \rightarrow \mathbb{R}^{T \times H}$ to predict logit changes.
3. **Fixed-Logit based Forecasting (Fixed-Logit)**:
   Uses first-order Taylor expansion of logit changes without training a separate model.
4. **Representation-based Forecasting (Representation)**:
   Forecasts forgetting based on the cosine similarity of representations in the model's hidden space.

### Experiment III: Refinement Utility
- **Edit Success Rate (Succ.)**: The proportion of online learning examples that produce correct answers after model updates:
  $$\text{Succ.} = \frac{|\{\langle x_i, y_i \rangle \in D_{\text{R}} \mid f_i(x_i) = y_i\}|}{|D_{\text{R}}|}$$
- **Exact Match Drop Ratio (EM Drop %)**: The drop in Exact Match score on the upstream dataset $D_{\text{PT}}$ after refinement. Lower EM Drop % indicates reduced forgetting.

---

## 3. Environment & Model Coverage

We support the following pre-trained language models (PTLMs) and fine-tuning configurations:
- **BART-Large** (`facebook/bart-large`): Fine-tuning LM heads only (Head) or Full Fine-Tuning (Full FT).
- **FLAN-T5-Large** (`google/flan-t5-large`): Fine-tuning LM heads only (Head), Low-Rank Adaptation (LoRA), or Full Fine-Tuning (Full FT).
- **FLAN-T5-3B** (`google/flan-t5-3b`): Evaluated under LoRA and Head configurations.

### LoRA Configuration Default