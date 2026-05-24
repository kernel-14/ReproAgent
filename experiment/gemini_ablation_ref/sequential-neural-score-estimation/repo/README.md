# Sequential Neural Score Estimation (TSNPSE) Reproduction

This repository contains a faithful, complete, and judgeable reproduction of the methods, environments, baselines, and evaluation protocols described in the paper:
**"Sequential Neural Score Estimation: Likelihood-Free Inference with Conditional Score Based Diffusion Models"**

---

## 1. Project Overview

Sequential Neural Score Estimation (SNSE) and its truncated variant, **Truncated Sequential Neural Score Estimation (TSNPSE)**, leverage conditional score-based diffusion models to perform likelihood-free inference. This repository implements the core TSNPSE algorithm (Algorithm 1), the score network architecture, the forward/backward SDE dynamics (VE SDE and VP SDE), and provides a unified interface for the 8 standard SBI benchmark tasks (including Two Moons, SLCP, and Lotka-Volterra) under simulation budgets of $1,000$, $10,000$, and $100,000$.

### Core Hypothesis
Standard SBI benchmarks and baselines provide the necessary context to validate TSNPSE performance, demonstrating superior or competitive posterior estimation quality (measured via C2ST) compared to traditional NPE, NLE, and NRE baselines.

---

## 2. Setup & Installation

### Prerequisites
Ensure you have Python 3.8+ installed. Install the required dependencies: