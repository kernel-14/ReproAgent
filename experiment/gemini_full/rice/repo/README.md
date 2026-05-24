# RICE: Breaking Through the Training Bottlenecks of Reinforcement Learning with Explanation

This repository provides a faithful reproduction of the RICE (Reinforcement learning with Importance-based Critical-state Exploration) algorithm as described in the paper "RICE: Breaking Through the Training Bottlenecks of Reinforcement Learning with Explanation".

## Project Summary
RICE is an algorithm designed to improve the performance of pre-trained Deep Reinforcement Learning (DRL) policies that are not fully optimal. It leverages step-level explanations to identify critical states and initiates exploration from these states to break through training bottlenecks.

### Core Algorithm Steps
1.  **Explanation Generation**: Given a pre-trained DRL policy $\pi$, we employ the StateMask explanation method to identify the most crucial time steps influencing the final rewards.
2.  **State Reset (Roll-in)**: The RL agent is reset to specific visited states, which are a mixture of default initial states and identified critical states (controlled by hyper-parameter $p$).
3.  **Exploration**: A new exploration step is initiated from these chosen states, followed by policy updates using standard RL algorithms like PPO or SAC.

## Methodology and Formulas

### Problem Setup
For a policy $\pi(a \mid s)$, the value function and $Q$-function are defined as:
- $V^{\pi}(s)=\mathbb{E}_{\pi}\left[\sum_{t=0}^{\infty} \gamma^{t} R\left(s_{t}, a_{t}\right) \mid s_{0}=s\right]$
- $Q^{\pi}(s, a)=\mathbb{E}_{\pi}\left[\sum_{t=0}^{\infty} \gamma^{t} R\left(s_{t}, a_{t}\right) \mid s_{0}=s, a_{0}=a\right]$
- Advantage function: $A^{\pi}(s, a)=Q^{\pi}(s, a)-V^{\pi}(s)$

### StateMask and RICE Explanation
The restored StateMask reference implementation is grounded in
`https://github.com/nuwuxian/RL-state_mask`.  The original StateMask method and
the RICE variant are both implemented in `src/rice/statemask.py`:

- Original StateMask uses the objective
  $J(\theta)=\min |\eta(\pi)-\eta(\bar{\pi})|$ and a prime-dual/Lagrange update.
- RICE changes the mask objective to $J(\theta)=\max \eta(\bar{\pi})$ and trains
  the mask policy with PPO.
- The mask network is binary: output `0` marks critical steps where the target
  action is replaced by `a_random`; output `1` marks ordinary steps.
- RICE uses the shaped reward $R' = R + \alpha a_t^m$, so output `1` receives
  the mutable `alpha` bonus.  `alpha`, `lambda`, and `p` are config fields, not
  hard-coded constants.
- `ours`, `statemask`, and `random` can be selected both for retraining and for
  rollout-only explanation generation.

The MuJoCo task identifiers include `Hopper-v3`, `Walker2d-v3`, `Reacher-v2`,
and `HalfCheetah-v3`.

### Fidelity Score Pipeline
As clarified in the addendum:
- The explanation method generates step-level importance scores for the trajectory.
- Steps are ranked to identify the top-K critical steps.
- Fidelity is measured by the impact of masking these steps on the final reward,
  following the StateMask scripts that select a critical span from the top
  confidence scores and compute the reward-drop score across up to 500
  trajectories.

### Explanation-Based Refinement
Algorithm 2 is implemented by `Algorithm2Refiner`: it constructs a mixed initial
state distribution from default initial states and critical states selected by
the explanation method, then adds a Random Network Distillation exploration
bonus weighted by mutable `lambda`.  `StateMask-R`, `JSRL`, and PPO fine-tuning
adapters reuse the selected explanation method and record cumulative reward
during refinement.

## Configuration and Hyperparameters
Reproduction settings are managed via `configs/addendum_constraints_flags.yaml` and `configs/default.yaml`.

| Parameter | Default / Sweep Range | Description |
| :--- | :--- | :--- |
| $\alpha$ | $0.01, 0.001, 0.0001$ | Masking reward bonus coefficient |
| $\lambda$ | $0, 0.1, 0.01, 0.001$ | Exploration reward bonus |
| $p$ | $0, 0.25, 0.5, 0.75, 1$ | Probability of resetting to a critical state |
| $d_{max}$ | $1.0$ | Sub-optimality bound constant |

## Installation and Setup
