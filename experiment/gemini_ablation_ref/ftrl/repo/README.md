# Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem

This repository contains a faithful, complete, and judgeable reproduction of the methods, environments, evaluations, and artifacts presented in the paper *"Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem"*.

---

## 1. Project Summary & Core Hypothesis

Fine-tuning pre-trained reinforcement learning (RL) models on downstream tasks often leads to a severe degradation in their pre-trained capabilities. This paper demonstrates that this phenomenon—termed **Forgetting of Pre-trained Capabilities (FPC)**—is primarily driven by two factors:
1. **State Coverage Gap**: The downstream task starts in states where the pre-trained policy has not been trained or performs poorly (CLOSE states), requiring exploration/learning. During this phase, the agent rarely visits the states where the pre-trained policy excels (FAR states), leading to catastrophic forgetting of those capabilities before they can be leveraged.
2. **Imperfect Cloning Gap**: Discrepancies between the pre-training objective (e.g., behavioral cloning) and the downstream RL objective.

To mitigate FPC, this repository implements and evaluates several knowledge retention techniques:
- **Behavioral Cloning (BC) Regularization**: Penalizing deviations from the pre-trained policy $\pi_*$ using a KL-divergence loss over a replay buffer $\mathcal{B}_{BC}$.
- **Elastic Weight Consolidation (EWC)**: Regularizing parameter changes using the Fisher Information Matrix $F$ computed from the pre-training data.
- **Experience Mixture (EM)**: Replaying experiences from the pre-training phase during downstream fine-tuning.
- **Kickstarting (KS)**: Regularizing the policy using a KL-divergence loss over states visited by the *current* policy.

---

## 2. Setup & Installation

### Prerequisites
- Python 3.8+
- PyTorch (optional, lazy-loaded for full training)
- Gym / Gymnasium
- NetHack Learning Environment (NLE) (optional, lazy-loaded)
- Continual World / Meta-World (optional, lazy-loaded)

### Installation Commands