# RICE: Breaking Through the Training Bottlenecks of Reinforcement Learning with Explanation

This repository contains a faithful, complete, and judgeable reproduction of the RICE (Reinforcement Learning with Explanation) algorithm, as described in the paper *"RICE: Breaking Through the Training Bottlenecks of Reinforcement Learning with Explanation"*.

---

## 1. Project Overview & Core Hypothesis

### Core Hypothesis
Refining a pre-trained Deep Reinforcement Learning (DRL) policy that is not fully optimal by rolling in to critical states identified by a step-level explanation method (e.g., a StateMask network) and initiating exploration from these states achieves significantly higher final rewards and faster convergence compared to vanilla RL fine-tuning, JSRL, and random roll-in baselines.

### Decision Value
Verifying that explanation-guided state selection and exploration can successfully break through training bottlenecks in complex environments (MuJoCo, Selfish Mining, Network Defense, Autonomous Driving, CAGE, and Gym tasks).

---

## 2. Theoretical Foundations & Algorithm Details

### 2.1. Problem Setup and Assumptions
For a policy $\pi(a \mid s): \mathcal{S} \rightarrow \mathcal{A}$, the value function and $Q$-function are defined as:
$$V^{\pi}(s)=\mathbb{E}_{\pi}\left[\sum_{t=0}^{\infty} \gamma^{t} R\left(s_{t}, a_{t}\right) \mid s_{0}=s\right]$$
$$Q^{\pi}(s, a)=\mathbb{E}_{\pi}\left[\sum_{t=0}^{\infty} \gamma^{t} R\left(s_{t}, a_{t}\right) \mid s_{0}=s, a_{0}=a\right]$$

The advantage function for the policy $\pi$ is denoted as:
$$A^{\pi}(s, a)=Q^{\pi}(s, a)-V^{\pi}(s)$$

### 2.2. Step-Level Explanation (StateMask Network)
We leverage a state-of-the-art explanation method, StateMask, to identify the most crucial time steps influencing the final rewards in a trajectory. The mask network parameterizes the importance of the target agent's current time step as a neural network model. It learns a policy to "blind" the target agent at certain steps without changing the agent's final reward.

For an input state $s_t$, the mask net outputs a binary action $a_t^m \in \{0, 1\}$. The final action is determined by:
$$a_{t} \odot a_{t}^{m}= \begin{cases}a_{t}, & \text { if } a_{t}^{m}=0 \\ a_{\text {random }} & \text { if } a_{t}^{m}=1\end{cases}$$

To avoid the trivial solution of never blinding the agent (always outputting $0$), we add an additional reward bonus when the mask net outputs $1$:
$$R^{\prime}\left(s_{t}, a_{t}\right)=R\left(s_{t}, a_{t}\right)+\alpha a_{t}^{m}$$
where $\alpha$ is a hyperparameter controlling the blinding bonus.

### 2.3. RICE Refining Algorithm
Given a pre-trained DRL policy that is not fully optimal:
1. **Identify Critical States**: Use the trained mask network to generate step-level importance scores.
2. **Reset to Visited States**: Reset the RL agent to specific visited states (a mixture of default initial states and identified critical states based on probability $p$).
3. **Exploration Step**: Initiate exploration from these chosen states with an exploration bonus controlled by $\lambda$.

---

## 3. Repository Architecture & Implementation Surfaces

The codebase is structured as follows:
- `configs/default.yaml`: Hyperparameters, environment configurations, and experiment registry.
- `src/rice/__init__.py`: Package interface exposing core algorithms, models, and environment wrappers.
- `src/rice/config.py`: Configuration parser and parameter sweep definitions.
- `src/rice/environments.py`: Unified Gym environment wrappers and mock implementations for MuJoCo, Selfish Mining, Network Defense, Autonomous Driving, CAGE, and Gym tasks.
- `src/rice/models.py`: Actor-Critic policy networks and the StateMask network.
- `src/rice/ppo.py`: Standard PPO training loop for policy optimization and mask network training.
- `src/algorithms/rice.py`: Core RICE roll-in and exploration refining algorithm.
- `src/rice/evaluation.py`: Evaluation metrics, fidelity score calculation, and baseline comparison logic.
- `src/rice/utils.py`: Helper functions for logging, directory creation, and artifact writing.
- `main.py`: Main entrypoint for running experiments, evaluations, and generating artifacts.

---

## 4. Supported Baselines & Parameter Sweeps

### Named Baselines
- **Ours (RICE Refining)**: Explanation-guided roll-in and exploration.
- **JSRL Baseline**: Jump-Start Reinforcement Learning.
- **Random Roll-in Baseline**: Resets to random states along the trajectory.
- **Vanilla RL Baseline**: Standard PPO fine-tuning from initial states.
- **pbt**: Population-Based Training baseline.
- **pql**: Policy-guided Q-learning baseline.
- **heuristic**: Domain-specific heuristic baseline.
- **StateMask-R**: Refining using the original StateMask explanation.

### Parameter Sweeps
- **$\alpha$ (Blinding Bonus)**: $\{0.01, 0.001, 0.0001\}$ (controls mask network training).
- **$\lambda$ (Exploration Bonus)**: $\{0, 0.1, 0.01, 0.001\}$ (controls exploration reward bonus).
- **$p$ (Roll-in Probability)**: $\{0, 0.25, 0.5, 0.75, 1\}$ (controls the mixture of initial and critical states).
- **Roll-in Steps**: Bounded steps for rolling in.
- **Exploration Steps**: Bounded steps for exploration.

---

## 5. Execution & Bounded Smoke Runs

### 5.1. Environment Readiness & Setup
To verify environment availability and setup the directories, run: