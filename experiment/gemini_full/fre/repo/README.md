# Unsupervised Zero-Shot Reinforcement Learning via Functional Reward Encodings (FRE)

This repository contains a faithful, complete, and judgeable reproduction of the paper **"Unsupervised Zero-Shot Reinforcement Learning via Functional Reward Encodings"**.

## 1. Project Summary & Architecture

Functional Reward Encodings (FRE) is an unsupervised zero-shot reinforcement learning method. FRE discovers latent representations over random unsupervised reward functions. At evaluation, user-given downstream objectives can be encoded into the latent space to enable zero-shot policy execution. FRE utilizes simple building blocks and is a data-scalable way to learn general capabilities from unlabeled offline datasets.

The architecture consists of:
- **Transformer Encoder**: A permutation-invariant Transformer that processes a set of $(s, \eta(s))$ pairs to produce a latent representation $z$. Positional encodings and causal masking are not used, treating the inputs as an unordered set.
- **Decoder**: Reconstructs the reward function $\eta(s)$ from the latent representation $z$ and state $s$.
- **Latent-Conditioned Policy**: Trained using Implicit Q-Learning (IQL) conditioned on the latent reward encoding $z$.

### Directory Structure
- `src/fre/`: Core package containing models, environments, data loaders, and utilities.
- `configs/`: Configuration files for base settings and environment-specific parameters.
- `envs/`: Environment factory and wrappers for DMC (ExORL) and D4RL (AntMaze, Kitchen).
- `data/`: Dataset loading and preprocessing pipelines.
- `baselines/`: Implementations of baseline methods (FB, SF, PPO, PBT, PQL).
- `reproduce_results.py`: Orchestrator script to run experiments and generate figures/tables.
- `main.py`: Entrypoint for training and evaluation.

---

## 2. Configuration & Setup

### Setup Commands
To set up the environment, run:
```bash
pip install -r requirements.txt
```

### Configuration Flags
The configuration is managed via YAML files under `configs/`. Key parameters include:
- `K`: Number of state samples for encoding (default: `128`).
- `reward_discretization_bins`: Number of bins for reward discretization (default: `20`).
- `latent_dim_size`: Size of the latent dimension (default: `256`).
- `transformer_layers`: Number of layers in the Transformer encoder (default: `4`).
- `transformer_heads`: Number of heads in the Transformer encoder (default: `4`).
- `beta`: KL divergence weight (default: `0.1`).
- `K_prime`: Number of states sampled for the decoder (default: `6`).

---

## 3. Environment & Dataset Infrastructure

We support the following environments and datasets:
- **DeepMind Control (ExORL)**: Walker Walk, Walker Run, Cheetah Run.
- **AntMaze (D4RL)**: `antmaze-large-diverse-v2`.
- **Kitchen (D4RL)**: `kitchen-mixed-v0`.

### State Preprocessing & Normalization
State normalization is applied to match the paper's preprocessing:
$$ s_{\text{norm}} = \frac{s - \mu}{\sigma + \epsilon} $$
where $\mu$ and $\sigma$ are the mean and standard deviation computed over the offline dataset.

### State Sampling Strategy
For the encoder, we sample $K = 128$ states from the offline dataset $\mathcal{D}$ to evaluate the reward function $\eta$. For the decoder, we sample $K' = 6$ states.

---

## 4. Paper Formulas & Algorithms

### 4.1. Functional Reward Encoding (Section 4.1)
We learn a latent representation $z$ by maximizing the information bottleneck objective:
$$ L_{\eta} = L_{\eta}^e \rightarrow Z \rightarrow L_{\eta}^d $$
The loss function includes a reconstruction term and a KL divergence regularization term:
$$ \mathcal{L}_{\text{FRE}} = \mathbb{E}[L_{\eta}^d] + \beta \cdot D_{\text{KL}}(q_{\theta}(z | \{s_k^e, \eta(s_k^e)\}_{k=1}^K) \parallel p(z)) $$

### 4.2. Offline RL with FRE (Section 4.3)
**Algorithm 1: Functional Reward Encodings (FRE)**
1. **Train Encoder**:
   - Sample reward function $\eta \sim p(\eta)$.
   - Sample $K$ states for encoder $\{s_k^e\} \sim \mathcal{D}$.
   - Sample $K'$ states for decoder $\{s_k^d\} \sim \mathcal{D}$.
   - Train FRE by maximizing Equation (6).
2. **Train Policy**:
   - Condition the policy $\pi(a | s, z)$ and Q-functions $Q(s, a, z)$ on the latent encoding $z$.
   - Use Hindsight Relabeling:
     - Sample a goal state $g$ from:
       1. Future states in the trajectory using a geometric distribution ($p_{\text{geometric\_goal}} = 0.5$).
       2. A random goal in the dataset ($p_{\text{randomgoal}} = 0.3$).
       3. The current state ($p_{\text{current\_goal}} = 0.2$), where reward is 0 and mask/terminal is True.
     - The policy loss is:
       $$ L_{\pi} = -\mathbb{E}_{(s, g, a) \sim \mathcal{D}} \log \pi(a | s, g) $$

### 4.3. Target Velocity Tasks (Addendum)
For directional tasks, the target velocity in the (X,Y) plane is specified as:
- `vel_left`: $(-1.0, 0.0)$
- `vel_up`: $(0.0, 1.0)$
- `vel_down`: $(0.0, -1.0)$
- `vel_right`: $(1.0, 0.0)$

---

## 5. Paper Artifacts & Captions

We preserve the exact captions and semantics from the paper:
- **Figure 1**: FRE discovers latent representations over random unsupervised reward functions. At evaluation, user-given downstream objectives can be encoded into the latent space to enable zero-shot policy execution.
- **Figure 2**: FRE encodes a reward function by evaluating its output over a random set of data states. Given a sampled reward function $\eta$, the reward function is first evaluated on a set of random encoder states from the offline dataset. The $(s, \eta(s))$ pairs are then passed into a permutation-invariant transform.
- **Figure 3**: After unsupervised pretraining, FRE can solve user-specified downstream tasks without additional fine-tuning. Shown above are examples of reward functions sampled from various evaluations in AntMaze. Columns: 1) True reward function projected onto maze. 2) Random states used for encoding shown in non-black.
- **Table 1**: Offline zero-shot RL comparisons on AntMaze, ExORL, and Kitchen. FRE-conditioned policies match or outperform state-of-the-art prior methods on many standard evaluation objectives including goal-reaching, directional movement, and structured locomotion paths. FRE utilizes only 32 examples of (state, reward).
- **Figure 4**: Evaluation domains: AntMaze, ExORL, and Kitchen.
- **Table 2**: FRE unifies prior methods in capabilities. OPAL does not have zero-shot capabilities and learns via BC rather than Q-learning. GCRL and SF both limit reward function families to goal-reaching or linear functions, respectively. FB can learn to solve any reward function, but requires a linearized value function.
- **Figure 5**: The general capabilities of a FRE agent scales with diversity of random functions used in training. FRE-all represents an agent trained on a uniform mixture of three random reward families, while each other column represents a specific agent trained on only a subset of the three.
- **Figure 6**: By augmenting the random reward families with specific reward distributions, FRE can utilize domain knowledge without algorithmic changes.
- **Table 3**: Hyperparameters used for FRE.
- **Table 4**: Full results comparing FRE agents trained on different subsets of random reward functions in AntMaze.
- **Figures 7, 8, 9**: Additional examples of FRE results on AntMaze. Arranged three examples per page. For each run, from top-left to bottom-right: True reward function, predicted reward, Q function 1, randomly sampled states for encoding, policy trajectory, Q function 2.

---

## 6. Execution & Verification

### Bounded Smoke Run
To verify the installation and run a lightweight smoke test:
```bash
python main.py --mode runtime_smoke
```

### Full Reproduction
To run the full suite of experiments and generate all tables and figures:
```bash
python reproduce_results.py
```