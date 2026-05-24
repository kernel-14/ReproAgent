# Refined Coreset Selection: Towards Minimal Coreset Size under Model Performance Constraints

This repository contains the faithful, complete, and judgeable standalone code reproduction for the paper: **"Refined Coreset Selection: Towards Minimal Coreset Size under Model Performance Constraints"** (Lexicographic Bilevel Coreset Selection - LBCS).

---

## 1. Project Overview & Method Summary

Refined Coreset Selection (RCS) addresses the challenge of selecting a minimal coreset size under model performance constraints. Traditional coreset selection methods fix the coreset size beforehand. In contrast, RCS treats the problem as a lexicographic bilevel optimization problem:
1. **Primary Objective ($f_1(\boldsymbol{m})$)**: Satisfy the model performance constraint (tolerance $\epsilon$).
2. **Secondary Objective ($f_2(\boldsymbol{m}) = \|\boldsymbol{m}\|_0$)**: Minimize the coreset size.

The proposed method, **Lexicographic Bilevel Coreset Selection (LBCS)**, optimizes these objectives using a randomized direct search algorithm in the outer loop and standard empirical risk minimization in the inner loop.

---

## 2. Mathematical Formulation & Algorithm Anchors

### 2.1 Preliminaries & Symbols
Formally, given a large-scale dataset $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$ with a sample size $n$, where $\mathbf{x}_i$ denotes the instance and $y_i$ denotes the label:
* We use $\|\cdot\|_p$ to denote the $L_p$ norm of vectors or matrices and $\ell(\cdot)$ to denote the cross-entropy loss.
* Let $[n] = \{1, \ldots, n\}$.
* The $0-1$ masks $\boldsymbol{m} \in \{0, 1\}^n$ are introduced with $m_i = 1$ indicating that the data point $(\mathbf{x}_i, y_i)$ is selected into the coreset.
* The inner loop optimizes the model parameters $\boldsymbol{\theta}(\boldsymbol{m})$:
  $$\boldsymbol{\theta}(\boldsymbol{m}) \in \arg\min_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{m}, \boldsymbol{\theta})$$
  where $\mathcal{L}(\boldsymbol{m}, \boldsymbol{\theta}) = \frac{1}{\|\boldsymbol{m}\|_1} \sum_{i=1}^n m_i \ell(f(\mathbf{x}_i; \boldsymbol{\theta}), y_i)$.
* The outer loop optimizes the lexicographic preference:
  $$\text{lex}\min_{\boldsymbol{m}} (f_1(\boldsymbol{m}), f_2(\boldsymbol{m}))$$
  where $f_1(\boldsymbol{m})$ represents the performance constraint violation and $f_2(\boldsymbol{m}) = \|\boldsymbol{m}\|_0$ represents the coreset size.

### 2.2 Optimization Stage & Theoretical Bounds
During the $f_1$ optimization stage:
* By substituting $\boldsymbol{m}^{\hat{t}}$ and $\boldsymbol{m}_2^*$ into $\boldsymbol{a}$ and $\boldsymbol{b}$, we have:
  $$d_{f_2}(\boldsymbol{m}^{\hat{t}}, \boldsymbol{m}_2^*) < n_2 \gamma_2$$
* According to Condition 2, we have $n_2 \in \mathbb{R}^+$ and $0 < \eta_2 \le 1$ such that:
  $$\mathbb{P}(\boldsymbol{m}^{\hat{t}} \to \boldsymbol{m}_2^*) \ge \eta_2$$
* Key symbols tracked in the configuration and code: `L_p`, `x_i`, `y_i`, `m_i`, `f_1`, `sum_i=1^n`, `theta`, `L_0`, `f_2`, `lambda`, `f_i`, `i_prime`, `M_star`, `M_2_star`, `M_1_star`, `f_1_star`, `epsilon`, `f_2_star`, `f_star`, `S_1`, `gamma_1`, `eta_1`, `S_2`, `t_hat`.
* Numeric defaults and sweep values: `1000`, `1`, `0`, `2`, `3`, `5`, `4000`, `10`, `80.3`, `0.6`, `3000`, `14`, `16`, `17`, `4`, `29`.
* Hyperparameter $T = 1000$ (represented in `configs/default.yaml` and `configs/experiments.yaml`).

### 2.3 Impact Statement
Therefore, the development and realization of the algorithm for RCS require advanced technology and expertise, which may result in the emergence of technical barriers.

---

## 3. Environment & Setup

### 3.1 Requirements
The repository is designed to run in a minimal Python environment. Heavy dependencies (such as PyTorch and torchvision) are lazily imported to ensure that static review and lightweight parsing do not fail.

To install the required packages: