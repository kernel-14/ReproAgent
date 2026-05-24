# SAPG: Split and Aggregate Policy Gradients

This repository contains a faithful, complete, and judgeable reproduction of the paper **"SAPG: Split and Aggregate Policy Gradients"**.

---

## 1. Overview & Core Contributions

Policy gradient methods are highly sensitive to the variance in the estimate of the gradient. In the presence of large batch sizes resulting from massively parallelized environments (e.g., in GPU-accelerated simulators like IsaacGym), standard on-policy RL algorithms like PPO suffer from asymptotic performance saturation. 

SAPG introduces a new class of on-policy RL algorithms that can scale to tens of thousands of parallel environments. In contrast to regular on-policy RL, such as PPO, which learns a single policy across environments leading to wasted environment capacity, our method learns diverse followers and combines data.

### Key Architectural Concepts
- **Shared Backbone & Local Parameters (Figure 3)**: Each policy (one leader and $M-1$ followers) shares a common backbone network $B_\theta$ but is conditioned on local learned parameters $\phi_i$.
- **Data Aggregation Schemes (Figure 4)**: 
  - *Asymmetric (Leader-Follower)*: One designated leader policy aggregates and uses off-policy data from all followers, in addition to its own on-policy data.
  - *Symmetric*: Every policy acts symmetrically, using off-policy data from all other policies.
- **Diversity Enforcement**: Followers are encouraged to explore diverse regions of the state space using entropy regularization with coefficient $\sigma$.

---

## 2. Task and Environment Setup

We maintain strict environment parity for **"Hard"** vs **"Easy"** task classifications:

### Experiment I: AllegroKuka (Hard Difficulty Tasks)
All three hard tasks consist of an Allegro Hand (16 DOF) mounted on a Kuka arm (7 DOF) manipulating a cuboidal object on a fixed table:
1. **AllegroKuka-Throw**: Throwing the object to a target location.
2. **AllegroKuka-Regrasping**: Picking up and regrasping the object.
3. **AllegroKuka-Reorientation**: In-hand reorientation of the object.

*Observation Space*: $\mathbf{o}_{t}=\left[\mathbf{q}, \dot{\mathbf{q}}, \mathbf{x}_{t}, \mathbf{v}_{t}, \omega_{t}, \mathbf{g}_{t}, \mathbf{z}_{t}\right]$ where:
- $\mathbf{q}, \dot{\mathbf{q}} \in \mathbb{R}^{23}$ are joint angles and velocities.
- $\mathbf{x}_{t} \in \mathbb{R}^{7}$ is the pose of the object.
- $\mathbf{v}_{t}, \omega_{t}$ are linear and angular velocities.
- $\mathbf{g}_{t}$ is the task-dependent goal.

### Experiment II: Easy Tasks
1. **AllegroHand-Reorient**: Reorienting an object using a standalone Allegro Hand.
2. **ShadowHand-Reorient**: Reorienting an object using a standalone Shadow Hand.

---

## 3. Hyperparameters & Configuration

All default hyperparameters for the proposed method and baselines (PPO, PQL, PBT, DDPG) are exposed through configuration files (`configs/config.yaml`, `configs/default_config.yaml`, `configs/default.yaml`).

### Table 2. Training hyperparameters for AllegroKuka tasks
| Hyperparameter | Value |
| :--- | :--- |
| Batch Size | 24576 |
| Learning Rate | 3e-4 |
| Discount Factor ($\gamma$) | 0.99 |
| GAE Parameter ($\lambda_{GAE}$) | 0.95 |
| Entropy Coeff ($\sigma$) | 0.0 |
| Number of Policies ($M$) | 4 |
| Aggregation Weight ($\lambda$) | 1.0 |

### Table 3 & 4. Training hyperparameters for Shadow Hand
| Hyperparameter | Value |
| :--- | :--- |
| Batch Size | 16384 |
| Learning Rate | 5e-4 |
| Entropy Coeff ($\sigma$) | 0.005 |
| Number of Policies ($M$) | 4 |
| Aggregation Weight ($\lambda$) | 1.0 |

---

## 4. Execution & Reproduction Commands

### Setup Commands
To install the package and its dependencies: