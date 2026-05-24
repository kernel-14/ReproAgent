# Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem

This repository provides a faithful, complete, and runnable reproduction of the methods, environments, and evaluation protocols described in the paper **"Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem"**.

The core contribution of this work is demonstrating that the poor performance of vanilla reinforcement learning (RL) fine-tuning on downstream tasks is primarily driven by the **forgetting of pre-trained capabilities** (FPC). This forgetting is caused by two main factors:
1. **State Coverage Gap**: The agent frequently visits states near the starting state (CLOSE states) and infrequently visits downstream states (FAR states) early in training, causing the pre-trained policy to forget how to act in FAR states before it can even reach them.
2. **Imperfect Cloning Gap**: The pre-trained policy is not a perfect expert, and standard fine-tuning regularizations can propagate cloning errors.

To mitigate this, the repository implements knowledge retention techniques:
- **Behavioral Cloning (BC) Regularization** ($\mathcal{L}_{BC}$)
- **Elastic Weight Consolidation (EWC)** ($\mathcal{L}_{aux}$)
- **Kickstarting (KS)** ($\mathcal{L}_{KS}$)

---

## Project Structure

- `main.py`: The canonical entrypoint for running experiments, executing smoke tests, and generating paper-visible artifacts.
- `src/envs/`:
  - `two_state_mdp.py`: Implementation of the toy two-state MDP with CLOSE and FAR state partitions.
  - `apple_retrieval.py`: Implementation of the 1D gridworld exhibiting state coverage gaps.
  - `robotics.py`: Implementation of the sequential transfer robotics tasks (e.g., `push-wall`).
- `src/methods/`:
  - `vanilla.py`: Standard RL fine-tuning.
  - `bc.py`: Fine-tuning with Behavioral Cloning regularization.
  - `ewc.py`: Fine-tuning with Elastic Weight Consolidation regularization.
- `configs/`:
  - `addendum_constraints_flags.yaml`: Configuration flags and constraints derived from the paper addendum.
  - `default.yaml` & `default_config.yaml`: Default hyperparameters and environment parameters.
  - `evidence_obligation_registry.yaml`: Registry mapping paper claims to code surfaces.
- `results/`: Output directory for generated figures, tables, and metrics.

---

## Installation & Setup

### Requirements
To install the dependencies, run: