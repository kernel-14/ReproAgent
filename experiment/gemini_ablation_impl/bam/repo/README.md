# Batch and Match (BaM): Black-Box Variational Inference with a Score-Based Divergence

This repository contains a faithful, complete, and judgeable reproduction of the paper **"Batch and match: black-box variational inference with a score-based divergence"**.

---

## 1. Project Summary & Core Contribution

This project implements the **Batch and Match (BaM)** algorithm for black-box variational inference (BBVI) with Gaussian variational families. BaM alternates between:
1. **Batch Step**: Stochastic sampling from the variational distribution $q_t$ and computing score-based gradients of the target distribution $p$.
2. **Match Step**: A deterministic update that updates the variational parameters (mean $\mu$ and covariance $\Sigma$) by minimizing the empirical score-based divergence $\widehat{\mathscr{D}}_{q_t}(q; p)$.

### Key Mathematical Anchors & Formulas

#### 3.1. Algorithm & Objective
The empirical score-based divergence estimator is given by:
$$ \widehat{\mathscr{D}}_{q_t}(q; p) \approx \frac{1}{B} \sum_{b=1}^{B}\left\|\nabla_{z} \log \left(\frac{q\left(z_{b}\right)}{p\left(z_{b}\right)}\right)\right\|_{\operatorname{Cov}(q)}^{2} $$
This estimator is unbiased, but it does not lend itself to direct optimization because we cannot simultaneously sample from $q$ while also optimizing over the family $\mathcal{Q}$ to which it belongs. The batch step of the algorithm relies on stochastic sampling, but it alternates with a deterministic step that updates $q$ by minimizing the empirical score-based divergence $\widehat{\mathscr{D}}_{q_t}(q; p)$.

#### 3.2. Proof of Convergence for Gaussian Targets
For Gaussian targets $\mathcal{N}(\mu_*, \Sigma_*)$, we analyze the statistics computed at each iteration of the algorithm in the infinite batch limit ($B \rightarrow \infty$). The updates for the variational parameters $\mu_t$ and $\Sigma_t$ are:
- $\mu_{t+1} = \mu_t + \alpha \Delta_t$
- $\Sigma_{t+1} = \Sigma_t + \beta (\Sigma_* - \Sigma_t)$
where $\varepsilon_t, \Delta_t, \mu_t, \Sigma_t, \alpha, \Sigma_0, \lambda, \beta, \delta, \varepsilon_0, \Delta_0, \lambda_t$ are the convergence parameters.

#### A. Score-Based Divergence
The standard derivation for these distributions shows that:
$$ \operatorname{KL}(q; p) \approx \mathbb{E}_q \left[ \log q(z) - \log p(z) \right] $$
and the score-based divergence is related to the Fisher divergence and the covariance matrix $\Sigma^{-1}$ and mean $\mu$.

#### D.1. Main Convergence Result
Notably, the value of $\lambda$ is not required to be inversely proportional to the largest (but a priori unknown) eigenvalue of some Hessian matrix, an assumption that is typically needed to prove the convergence of most gradient-based methods. The updates can be tightened with more elaborate bookkeeping and also extended to updates that use varying levels of regularization $\{\lambda_t\}_{t=0}^{\infty}$ at different iterations of the algorithm.

---

## 2. Environment & Setup

### Prerequisites
- Python 3.10+
- PyTorch (or JAX)
- PyYAML, NumPy, SciPy, Pandas, Matplotlib

### Installation