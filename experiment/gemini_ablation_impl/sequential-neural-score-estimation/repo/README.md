# Sequential Neural Score Estimation (SNPSE / TSNPSE)

This repository contains a faithful, complete, and judgeable reproduction of the paper: **Sequential Neural Score Estimation: Likelihood-Free Inference with Conditional Score Based Diffusion Models**.

---

## 1. Project Summary & Core Hypothesis

### Core Hypothesis
> **顺序训练协议能够通过多轮模拟逐步收敛至真实后验，且检查点可用于后续评估。**
> *(The sequential training protocol can progressively converge to the true posterior through multiple rounds of simulations, and the saved checkpoints can be reliably used for subsequent evaluations.)*

### Decision Value
> **打通从模拟器数据生成到模型顺序更新的完整闭环，并确保训练过程可追溯。**
> *(Establish a complete closed loop from simulator data generation to sequential model updates, ensuring the entire training process is fully traceable and reproducible.)*

---

## 2. Setup & Installation

### Documented Setup Commands
To set up the environment and install the required dependencies, run: