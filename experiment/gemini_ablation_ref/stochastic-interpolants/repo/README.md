# Stochastic Interpolants with Data-Dependent Couplings

This repository contains a faithful, complete, and judgeable reproduction of the methods, data processing pipelines, evaluation interfaces, baselines, metrics, and artifact generation surfaces described in the paper **"Stochastic Interpolants with Data-Dependent Couplings"**.

---

## 1. Introduction & Core Method

Generative models based on stochastic interpolants construct a time-dependent stochastic process $I_t$ that interpolates between a simple base density $\rho_0(x_0)$ at time $t=0$ and a target data density $\rho_1(x_1)$ at time $t=1$. This transport is accomplished by means of an ordinary differential equation (ODE) or stochastic differential equation (SDE), which takes as initial condition a sample from $\rho_0$ and produces at time $t=1$ an approximate sample from $\rho_1$.

### 1.1. Stochastic Interpolant with Coupling (Definition 3.1)
The stochastic interpolant $I_t$ is defined as:
$$I_t = \alpha_t x_0 + \beta_t x_1 + \gamma_t z$$
where:
- $(x_0, x_1) \sim \rho(x_0, x_1)$ is a coupled pair of base and target samples.
- $z \sim \mathcal{N}(0, I)$ is independent noise.
- $\alpha_t, \beta_t, \gamma_t$ are time-dependent interpolation coefficients satisfying boundary conditions:
  - $\alpha_0 = 1, \beta_0 = 0, \gamma_0 = 0$ (so that $I_0 = x_0 \sim \rho_0$)
  - $\alpha_1 = 0, \beta_1 = 1, \gamma_1 = 0$ (so that $I_1 = x_1 \sim \rho_1$)

### 1.2. Data-Dependent Couplings
Unlike standard formulations that construct generative models built upon an independent coupling $\rho(x_0, x_1) = \rho_0(x_0)\rho_1(x_1)$, this work introduces a general formulation where the base density is produced via a data-dependent coupling:
$$\rho(x_0, x_1) = \rho_1(x_1)\rho_0(x_0 \mid x_1)$$
This coupling ensures that the base sample $x_0$ is computed conditionally on the target sample $x_1$, leading to straighter probability flows and more efficient ODE/SDE simulation.

### 1.3. Transport Equations & Objectives (Eq 7)
The time-dependent density $\rho_t(x)$ of the process $I_t$ satisfies a transport equation. The corresponding velocity field $b_t(x)$ and score field $s_t(x) = \nabla \log \rho_t(x)$ are learned by minimizing quadratic objectives:
$$\min_{b} \mathbb{E} \left[ |b(I_t, t) - \dot{I}_t|^2 \right]$$
$$\min_{s} \mathbb{E} \left[ |s(I_t, t) - \nabla \log p_t(I_t \mid x_0, x_1)|^2 \right]$$
where $\dot{I}_t = \dot{\alpha}_t x_0 + \dot{\beta}_t x_1 + \dot{\gamma}_t z$.

---

## 2. Tasks & Data Pipeline

The data pipeline supports two primary downstream tasks on ImageNet: **ImageNet In-painting** and **ImageNet Super-resolution**.

### 2.1. ImageNet In-painting (Section 4.1)
- **Goal**: Fill the pixels in a masked region with new values consistent with the entirety of the image.
- **Masking Protocol**: Given a pre-specified mask, the mask takes the same value for all channels in a given spatial location in the image.
- **Conditioning**: We condition on the missingness masks for in-painting by appending them to the state $x_t$.
- **Resolution**: Supports $256 \times 256$ and $512 \times 512$ resolutions.

### 2.2. ImageNet Super-resolution (Section 4.2)
- **Goal**: Produce a high-resolution image $x_1 \in \mathbb{R}^{C \times W \times H}$ from a low-resolution image $x_0$.
- **Downsampling Protocol**: Low-resolution images are obtained via downsampling (e.g., $64 \times 64 \mapsto 256 \times 256$ or $256 \times 256 \mapsto 512 \times 512$).
- **Manifold Concentration**: With noise parameter $\sigma = 0$, each $x_0$ corresponds to a lower-dimensional sample embedded in a higher-dimensional space, and the corresponding distribution is concentrated on a lower-dimensional manifold.

### 2.3. Dataset Registry & HuggingFace Integration
To avoid the code waiting for stdin during dataset download, we use `trust_remote_code=True` when loading ImageNet-1k: