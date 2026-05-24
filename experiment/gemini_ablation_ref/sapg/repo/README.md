# SAPG: Split and Aggregate Policy Gradients

This repository contains a faithful, complete, and executable reproduction of the paper **"SAPG: Split and Aggregate Policy Gradients"**. 

SAPG introduces a new class of on-policy reinforcement learning algorithms designed to scale to tens of thousands of parallel environments (e.g., $N = 24,576$ parallel instances on GPU-accelerated simulators like IsaacGym). While standard on-policy methods like PPO suffer from asymptotic performance saturation when batch sizes become extremely large (due to redundant data collection where most actions lie near the mean), SAPG learns diverse follower policies and aggregates their collected data to update a central leader policy using a hybrid of on-policy and off-policy updates.

---

## 1. Core Algorithmic Framework

### 1.1 Mathematical Formulations

*   **On-Policy Objective**:
    $$\mathcal{J}(\pi) = \mathbb{E}_{s_0 \sim \rho, a_t \sim \pi}\left[\sum_{t=0}^{T-1} \gamma^t r(s_t, a_t)\right]$$
    $$\nabla_{\theta} J(\pi_{\theta}) = \mathbb{E}_{s \sim \rho_d, a \sim \pi}\left[\nabla_{\theta} \log \pi_{\theta}(a \mid s) \hat{A}^{\pi_{\theta}}(s, a)\right]$$

*   **Off-Policy Aggregation Loss ($L_{off}$)**:
    To update policy $\pi_i$ using data collected by policy $\pi_j$ ($j \in \mathcal{X}$), we use importance sampling:
    $$L_{off}(\pi_i; \mathcal{X}) = \frac{1}{|\mathcal{X}|} \sum_{j \in \mathcal{X}} \mathbb{E}_{(s, a) \sim \pi_j}\left[\min\left(r_{\pi_i}(s, a) \hat{A}^{\pi_j}(s, a), \text{clip}(r_{\pi_i}(s, a), 1-\epsilon, 1+\epsilon) \hat{A}^{\pi_j}(s, a)\right)\right]$$
    where $r_{\pi_i}(s, a) = \frac{\pi_i(a \mid s)}{\pi_j(a \mid s)}$ is the importance weight.

*   **Critic Target ($n$-step returns, $n=3$)**:
    $$V_{off, \pi_j}^{\text{target}}(s_t') = r_t + \gamma V_{\pi_j, \text{old}}(s_{t+1}')$$

*   **Latent Conditioning (Section 4.4)**:
    To encourage diversity, the actor and critic share a backbone network ($B_{\theta}$ and $C_{\psi}$) but are conditioned on local learned parameters $\phi_j$ specific to each follower/leader:
    $$\pi_j(a \mid s) = \text{Actor}(B_{\theta}(s), \phi_j)$$
    $$V_j(s) = \text{Critic}(C_{\psi}(s), \phi_j)$$

### 1.2 Algorithm 1: Split and Aggregate Policy Gradients
1.  **Initialize**: Shared parameters $\theta, \psi$ and local parameters $\phi_1, \dots, \phi_M$ for $M$ policies.
2.  **Environment Sampling**: Distribute $N$ environments into $M$ blocks of size $N/M$. Each policy $\pi_j$ collects transitions in its designated block.
3.  **Data Distribution**: Share trajectories across policies according to the aggregation scheme.
4.  **Follower Updates**: Update local parameters $\phi_j$ and shared parameters $\theta, \psi$ using a combination of on-policy loss $L_{on}$ and off-policy loss $L_{off}$ with importance sampling.
5.  **Leader Aggregation**: Update the leader policy parameters using aggregated data from all followers.

---

## 2. Environment Registry & Integration

The repository supports integration with **IsaacGym** for massively parallel simulation. The environment factory is exposed via: