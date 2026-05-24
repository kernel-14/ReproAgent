# A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and Toxicity

This repository contains a faithful, complete, and judgeable reproduction of the paper **"A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and Toxicity"**.

---

## 1. Project Overview & Core Hypothesis

### Core Hypothesis
- **Hypothesis**: Through linear probing (Linear Probe) and MLP weight decomposition (SVD), we can identify specific directions in the model's residual stream and MLP value vectors that are responsible for generating toxic content. Furthermore, Direct Preference Optimization (DPO) reduces toxicity not by changing the fundamental representation of these toxic concepts, but by shifting the residual stream activations to bypass these toxic regions.
- **Decision Value**: This mechanistic understanding allows us to perform targeted interventions (e.g., vector subtraction to reduce toxicity) and adversarial manipulations (e.g., "un-aligning" the model by scaling toxic key vectors or forcing GLU gating components to 1 to reactivate toxicity).

---

## 2. Environment Setup & Configuration Flags

### Installation & Setup
To set up the environment, run the following commands: