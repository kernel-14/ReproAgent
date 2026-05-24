# SAPG: Split and Aggregate Policy Gradients

This repository contains a faithful, complete, and judgeable reproduction of the paper **"SAPG: Split and Aggregate Policy Gradients"**. 

SAPG introduces a new class of on-policy reinforcement learning (RL) algorithms designed to scale efficiently to tens of thousands of parallel environments (e.g., in GPU-accelerated simulators like IsaacGym). In contrast to regular on-policy RL (such as PPO), which learns a single policy across environments leading to wasted environment capacity, SAPG learns diverse followers and aggregates their collected data to update a central leader or perform symmetric updates.

---

## 1. Core Architecture & Algorithm

### 1.1 Latent Conditioning & Parameter Sharing
To encourage diversity while maintaining sample efficiency, SAPG utilizes a shared backbone network with local conditioning parameters:
* **Actor Network:** Consists of a shared backbone $B_\theta$ conditioned on local learned parameters $\phi_i$ specific to each policy $i$.
* **Critic Network:** Consists of a shared backbone $C_\psi$ conditioned on the same local parameters $\phi_i$.
* **Shared Parameters:** $\theta$ and $\psi$ are updated using aggregated gradients from all policies.
* **Local Parameters:** $\phi_i$ are updated only using the objective of policy $i$.

### 1.2 Algorithm 1: Orchestration Loop
The training loop manages $M$ separate data buffers and synchronizes shared backbone parameters across policies:
1. Initialize shared parameters $\theta, \psi$ and local parameters $\phi_1, \ldots, \phi_M$.
2. For each iteration:
   * Roll out $M$ different policies in parallel across $N$ environments (each policy gets a block of $\frac{N}{M}$ environments).
   * Collect data buffers $\mathcal{D}_1, \ldots, \mathcal{D}_M$.
   * **Off-Policy Aggregation:** Sample $\left|\mathcal{D}_1\right|$ transitions from $\bigcup_{j=2}^{M} \mathcal{D}_j$ to get $\mathcal{D}_1'$.
   * Compute the off-policy loss $L_{\text{off}}(\pi_1; \mathcal{D}_1')$ using importance sampling:
     $$L_{\text{off}}(\pi_i; \mathcal{X}) = \frac{1}{|\mathcal{X}|} \sum_{j \in \mathcal{X}} \mathbb{E}_{(s, a) \sim \pi_j} \left[ \min\left(r_{\pi_i}(s, a) \hat{A}^{\pi_j}(s, a), \text{clip}(r_{\pi_i}(s, a), 1-\epsilon, 1+\epsilon) \hat{A}^{\pi_j}(s, a)\right) \right]$$
   * Compute the on-policy loss $L_{\text{on}}(\pi_i; \mathcal{D}_i)$ for all policies.
   * Update shared parameters $\theta, \psi$ and local parameters $\phi_i$ using minibatch gradient descent.

---

## 2. Environment Registry & Setup

The repository supports the following hard-difficulty manipulation tasks based on the **Allegro-Kuka** and **Shadow Hand** environments:

| Task ID | Family | Alias | Difficulty | Setup Metadata |
| :--- | :--- | :--- | :--- | :--- |
| `AllegroKuka-Throw` | AllegroKuka | `AllegroKukaThrow-v0` | Hard | $\mathbf{o}_t = [\mathbf{q}, \dot{\mathbf{q}}, \mathbf{x}_t, \mathbf{v}_t, \omega_t, \mathbf{g}_t, \mathbf{z}_t]$ |
| `AllegroKuka-Regrasping` | AllegroKuka | `AllegroKukaRegrasping-v0` | Hard | $\mathbf{o}_t = [\mathbf{q}, \dot{\mathbf{q}}, \mathbf{x}_t, \mathbf{v}_t, \omega_t, \mathbf{g}_t, \mathbf{z}_t]$ |
| `AllegroKuka-Reorientation` | AllegroKuka | `AllegroKukaReorientation-v0` | Hard | $\mathbf{o}_t = [\mathbf{q}, \dot{\mathbf{q}}, \mathbf{x}_t, \mathbf{v}_t, \omega_t, \mathbf{g}_t, \mathbf{z}_t]$ |
| `AllegroHand-Reorientation` | Hand | `AllegroHandReorientation-v0` | Easy/Medium | $\mathbf{o}_t = [\mathbf{q}, \dot{\mathbf{q}}, \mathbf{x}_t, \mathbf{v}_t, \omega_t, \mathbf{g}_t, \mathbf{z}_t]$ |
| `ShadowHand-Reorientation` | Hand | `ShadowHandReorientation-v0` | Easy/Medium | $\mathbf{o}_t = [\mathbf{q}, \dot{\mathbf{q}}, \mathbf{x}_t, \mathbf{v}_t, \omega_t, \mathbf{g}_t, \mathbf{z}_t]$ |

* **Observation Space:** $\mathbf{q}, \dot{\mathbf{q}} \in \mathbb{R}^{23}$ are joint angles and velocities, $\mathbf{x}_t \in \mathbb{R}^7$ is the object pose, $\mathbf{v}_t, \omega_t$ are linear and angular velocities, $\mathbf{g}_t$ is the goal, and $\mathbf{z}_t$ is the latent conditioning.

---

## 3. Configuration & Hyperparameters

All hyperparameters are exposed via configuration files (`configs/default.yaml`, `configs/base_config.yaml`, `configs/experiment_matrix.yaml`).

### 3.1 SAPG Default Hyperparameters
* **Number of Policies ($M$):** 3 (1 Leader, 2 Followers)
* **Off-policy Aggregation Weight ($\lambda$):** 1.0
* **Importance Weight Clipping ($\mu$):** 1.0
* **Entropy Regularization ($\sigma$):** 0.005 (for Shadow Hand and Allegro Kuka Reorientation; 0.0 for other environments)
* **Batch Size:** 24,576
* **Optimization Epochs:** 6

### 3.2 Baseline Hyperparameters
* **PPO:** `clip_param: 0.2`, `ppo_epoch: 6`, `num_mini_batch: 4`, `entropy_coef: 0.0`
* **PQL / DDPG:** `actor_lr: 0.0003`, `critic_lr: 0.0003`, `tau: 0.005`
* **PBT:** Population size of 8, exploit fraction of 0.2, mutate strength of 0.1.

---

## 4. Usage & CLI Commands

### 4.1 Installation & Setup
Ensure you have Python 3.8+ installed. Install the required dependencies: