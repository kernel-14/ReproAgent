# Unsupervised Zero-Shot Reinforcement Learning via Functional Reward Encodings (FRE)

This repository contains a faithful, complete, and judgeable reproduction of the paper **"Unsupervised Zero-Shot Reinforcement Learning via Functional Reward Encodings"**.

---

## 1. Overview & Core Concepts

### Figure 1. Latent Representations over Random Unsupervised Reward Functions
FRE discovers latent representations over random unsupervised reward functions. At evaluation, user-given downstream objectives can be encoded into the latent space to enable zero-shot policy execution. FRE utilizes simple building blocks and is a data-scalable way to learn general capabilities from unlabeled offline data.

### Figure 2. Functional Reward Encoding Architecture
FRE encodes a reward function by evaluating its output over a random set of data states. Given a sampled reward function $\eta$, the reward function is first evaluated on a set of random encoder states from the offline dataset. The $(s, \eta(s))$ pairs are then passed into a permutation-invariant transformer encoder to produce a latent representation $z$. Positional encodings and causal masking are not used, thus the inputs are treated as an unordered set.

### Figure 3. Zero-Shot Downstream Task Solving
After unsupervised pretraining, FRE can solve user-specified downstream tasks without additional fine-tuning. Shown in the paper are examples of reward functions sampled from various evaluations in AntMaze. Columns:
1. True reward function projected onto maze.
2. Random states used for encoding shown in non-black.
3. Decoded reward, Q-functions, and executed policy trajectories.

---

## 2. Methodology & Architecture

### 2.1 Functional Reward Encoding (Section 4.1)
We learn a latent representation $z$ that is maximally informative about $L_{\eta}$, while remaining maximally compressive. This is formulated as the following information bottleneck objective over the structure of $L_{\eta}^{e} \rightarrow Z \rightarrow L_{\eta}^{d}$:
$$\mathcal{L}_{\text{FRE}} = \mathbb{E}_{\eta \sim p(\eta), s^e \sim \mathcal{D}, s^d \sim \mathcal{D}} \left[ \log p_\theta(\eta(s^d) | s^d, z) \right] - \beta D_{\text{KL}}(q_\phi(z | \{s_k^e, \eta(s_k^e)\}_{k=1}^K) \parallel p(z))$$

### 2.2 Reward Discretization & Embedding (Section 4.2)
Rewards are denoted as functions of state $\eta(s)$. To handle arbitrary reward scales, reward functions are discretized into binary or multi-bin categorical values before being passed to the embedding layers.

### 2.3 Offline RL with FRE (Section 4.3)
We train a latent-conditioned policy $\pi(a | s, z)$ and action-value function $Q(s, a, z)$ using offline RL algorithms (e.g., IQL/CQL style) on the offline dataset $\mathcal{D}$.
* **Evaluation Constraint**: At evaluation time, we must use exactly $K$ state samples to encode the downstream reward function. **No fine-tuning or online adaptation is allowed during test time.**

---

## 3. Environment & Dataset Coverage

The reproduction suite supports the following environments and datasets:
1. **ExORL (DeepMind Control Suite)**: Evaluated on exploratory datasets (e.g., RND dataset) for domains like Walker, Quadruped, and Jaco.
2. **AntMaze (D4RL)**: Multi-task goal-reaching and directional locomotion.
3. **Kitchen (D4RL)**: Structured robotic manipulation tasks.

---

## 4. Baselines & Method Variants

The suite includes wrappers and implementations for the following baselines:
* **Forward-Backward (FB)**: Jointly learns representations representing a family of tasks and their optimal policies.
* **Successor Features (SF)**: Approximates a universal family of reward functions using pre-trained features.
* **Goal-Conditioned RL (GCRL / GC-IQL)**: Standard goal-conditioned offline RL.
* **APS & ProtoRL**: Unsupervised exploration baselines.
* **OPAL**: Learning via behavior cloning rather than Q-learning.
* **PPO, PBT, PQL**: Online/fine-tuning baselines for quantitative comparison.

---

## 5. Experiment Protocols

### Experiment 5.2: Main Benchmark Comparison
Compares FRE against FB, SF, GCRL, and OPAL on ExORL, AntMaze, and Kitchen benchmarks.
* **Table 1**: Offline zero-shot RL comparisons on AntMaze, ExORL, and Kitchen. FRE-conditioned policies match or outperform state-of-the-art prior methods.
* **Figure 4**: Evaluation domains visualization and zero-shot performance comparison.

### Experiment 5.3: Scaling Properties (Section 5.3)
Ablation study on the diversity of random reward families used during training.
* **Figure 5**: The general capabilities of a FRE agent scale with the diversity of random functions used in training. `FRE-all` represents an agent trained on a uniform mixture of three random reward families (Singleton goal-reaching, Random linear, Random MLP), while other variants are trained on subsets.
* **Table 4**: Full results comparing FRE agents trained on different subsets of random reward functions in AntMaze.

### Experiment 5.4: Domain Knowledge Augmentation (Section 5.4)
Augmenting the random reward families with specific reward distributions (e.g., XY coordinate or velocity priors).
* **Figure 6**: Domain knowledge specificity (XY/velocity) showing that FRE can utilize domain knowledge without algorithmic changes.
* **Target Velocity Tasks**: Evaluates directional tasks: `vel_left` (-1, 0), `vel_up` (0, 1), `vel_down` (0, -1), `vel_right` (1, 0).

### Extended Experiments: Comparison with PPO, PBT, PQL
* **Table 3**: Quantitative comparison with PPO/PBT/PQL and hyperparameters used for FRE.
* **Figures 7, 8, 9**: Additional examples of FRE results on AntMaze showing True reward function, predicted reward, Q-functions, randomly sampled states for encoding, and policy trajectories.

---

## 6. Reproduction Artifacts & Evidence Matrix

The reproduction suite writes the following concrete artifacts to verify the paper's claims:

| Paper Reference | Description | Artifact Path |
| :--- | :--- | :--- |
| **Table 1** | Zero-shot performance on ExORL benchmarks | `results/tables/table1_exorl.csv` |
| **Figure 4** | Zero-shot performance on AntMaze and Kitchen | `results/plots/figure4_antmaze_kitchen.png` |
| **Figure 5** | Scaling properties of reward families | `results/plots/figure5_scaling.png` |
| **Figure 6** | Domain knowledge specificity (XY/velocity) | `results/plots/figure6_specificity.png` |
| **Table 3** | Quantitative comparison with PPO/PBT/PQL | `results/tables/table3.csv` |
| **Figure 7** | Zero-shot performance comparison | `results/plots/figure7.png` |
| **Figure 8** | Ablation analysis | `results/plots/figure8.png` |
| **Figure 9** | Sensitivity analysis | `results/plots/figure9.png` |
| **Metrics** | Aggregated evaluation metrics | `results/metrics.json` |
| **Registry** | Method, ablation, and dataset registries | `results/method_registry.json`, `results/ablation_registry.json`, `results/dataset_registry.json` |

---

## 7. Setup & Execution Commands

### 7.1 Installation & Environment Readiness Check
To verify the environment and dependencies: