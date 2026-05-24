# Unsupervised Zero-Shot Reinforcement Learning via Functional Reward Encodings (FRE)

This repository contains the faithful, complete, and judgeable reproduction of the paper **"Unsupervised Zero-Shot Reinforcement Learning via Functional Reward Encodings"**.

---

## 1. Project Overview & Core Architecture

Functional Reward Encodings (FRE) is a framework for unsupervised zero-shot reinforcement learning. FRE discovers latent representations over random unsupervised reward functions. At evaluation, user-given downstream objectives can be encoded into the latent space to enable zero-shot policy execution. FRE utilizes simple building blocks and is a data-scalable way to learn general capabilities from unlabeled offline datasets.

### Key Architectural Components

*   **Figure 1. Conceptual Overview**: FRE discovers latent representations over random unsupervised reward functions. At evaluation, user-given downstream objectives can be encoded into the latent space to enable zero-shot policy execution. FRE utilizes simple building blocks and is a data-scalable way to learn general capabilities from unlabeled offline datasets.
*   **Figure 2. Reward Encoding Pipeline**: FRE encodes a reward function by evaluating its output over a random set of data states. Given a sampled reward function $\eta$, the reward function is first evaluated on a set of random encoder states from the offline dataset. The $(s, \eta(s))$ pairs are then passed into a permutation-invariant transformer.
*   **Figure 3. Downstream Zero-Shot Execution**: After unsupervised pretraining, FRE can solve user-specified downstream tasks without additional fine-tuning. Shown above are examples of reward functions sampled from various evaluations in AntMaze. Columns: 1) True reward function projected onto maze. 2) Random states used for encoding shown in non-black.
*   **Figure 4. Evaluation Domains**: AntMaze, ExORL, and Kitchen.

---

## 2. Setup & Installation

### Documented Setup Commands
To set up the environment and install the required dependencies, run: