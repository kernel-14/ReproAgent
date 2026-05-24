# APT: Adaptive Pruning and Tuning Pretrained Language Models for Efficient Training and Inference

This repository contains the faithful, complete, and judgeable reproduction of the paper **"APT: Adaptive Pruning and Tuning Pretrained Language Models for Efficient Training and Inference"**.

APT provides both training and inference efficiency benefits by pruning and tuning pretrained LM parameters adaptively via the APT adapter. We dynamically adjust (add/reduce) APT adapter input/output dimensions and the rank ($r_{\text{apt}}$). Reducing adapter dimensions prunes frozen parameters, making the model highly efficient for both training and inference.

---

## 1. Overview of APT

### Figure 1. APT Adaptive Pruning and Tuning Paradigm
APT adaptively identifies pruning and tuning parameters via APT adapters during fine-tuning with little cost. APT gradually prunes LM parameters with binary pruning masks learned from our lightweight outlier-aware salience scoring function for training and inference efficiency. APT also adds tuning parameters to task-sensitive layers to recover performance.

### Figure 2. Adaptive Pruning and Tuning Flow
1. **Outlier-Aware Salience Estimation**: Compute salience scores using kurtosis to capture outlier features.
2. **Structured Pruning via Binary Search**: Sort blocks by salience density and perform binary search to satisfy the target sparsity constraint.
3. **Dynamic Parameter Allocation**: Allocate tuning parameter budget to task-sensitive layers.
4. **Self-Knowledge Distillation**: Distill knowledge from the early-stage model (teacher) to the pruned model (student) to recover task performance.

---

## 2. Mathematical Formulations & Algorithms

### 2.1. Problem Formulation (Section 3)
We formally define the problem objective as minimizing the task loss $\mathcal{L}$ under the constraint that the total LM parameter size $\Theta$ reaches a target sparsity (defined as the ratio of the number of parameters pruned to the total parameters):
$$\min_{\Theta_t, M_t, R_t} \mathcal{L}(\Theta_t, M_t, R_t) \quad \text{s.t.} \quad \text{Sparsity}(\Theta_t) \ge \gamma_t$$
where:
- $\Theta_t$: Active parameters at step $t$.
- $M_t$: Binary pruning masks.
- $R_t$: Dynamic rank allocation matrix.
- $\gamma_t$: Sparsity schedule from $\gamma_0 = 0$ to target sparsity $\gamma_T$.

### 2.2. APT Adapter (Section 4.1)
Assuming an APT adapter projects the input $X \in \mathbb{R}^{d_i}$ to the output $H_{\text{apt}}(X) \in \mathbb{R}^{d_o}$, we design binary pruning masks ($m_i \in \mathbb{R}^{d_i}$ for input and $m_o \in \mathbb{R}^{d_o}$ for output) and dynamic rank $r_{\text{apt}}$:
$$H_{\text{apt}}(X) = (X \odot m_i) W_A W_B \odot m_o$$
where $W_A \in \mathbb{R}^{d_i \times r_{\text{apt}}}$ and $W_B \in \mathbb{R}^{r_{\text{apt}} \times d_o}$.

### 2.3. Low-cost Adaptive LM Pruning (Section 4.2)
Given a task, we compute the outlier-aware salience score of parameter blocks at each early-training step when $t \ll T$. The outlier-aware salience score $\hat{S}$ for a block is computed using the kurtosis of the gradients and activations:
$$\hat{S}(W) = \text{Kurtosis}(X) \cdot \| \nabla_W \mathcal{L} \odot W \|_2^2$$
During training, the outlier-aware salience of each block is computed as an exponential moving-average:
$$\overline{S}^{(t)}(m) \gets 0.85 \overline{S}^{(t-1)}(m) + 0.15 \hat{S}(m)$$
where $\overline{S}^{(t)}(m)$ is the moving-average of block $m$ at time step $t$, and $\hat{S}(m)$ is the current outlier-aware salience score of block $m$.

### 2.4. Structured Pruning via Binary Search (Appendix C)
For the details of the algorithm, we first sort all blocks by the salience density, defined as the block salience divided by the number of parameters in the block.
After sorting the blocks by salience density, as LM's parameter size monotonically increases with the number of MHA heads, FFN neurons, and hidden dimensions, we conduct a binary search algorithm to identify the blocks that shall be retained to satisfy the target sparsity constraint.

### 2.5. Self-Knowledge Distillation (Section 4.4)
To recover the pruned LM's task performance with limited training expense, we use a self-knowledge distillation loss:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{pred}} + \tau \mathcal{L}_{\text{distill}}$$
where $\mathcal{L}_{\text{distill}}$ is the KL-divergence between the student (pruned model) and the teacher (unpruned early-stage model) probability distributions.

---

## 3. Supported Models, Datasets, and Baselines

### 3.1. Models
- **BERT** (`bert-base-uncased`)
- **RoBERTa** (`roberta-base`, `roberta-large`)
- **T5** (`t5-base`, `t5-large`)
- **LLaMA 2** (`llama-2-7b`, `llama-2-13b`)

### 3.2. Datasets & Tasks
- **GLUE Benchmark**: SST2, MNLI, QQP, QNLI, RTE, MRPC, CoLA, STS-B.
- **SQuAD**: SQuAD v1.1 and v2.0.
- **TruthfulQA**: Evaluation of LLM generation truthfulness.
- **Alpaca**: GPT4-generated Alpaca dataset for LLaMA instruction tuning.

### 3.3. Baselines
- **Fine-Tuning (FT)**: Standard full parameter fine-tuning.
- **LoRA**: Low-Rank Adaptation with fixed rank.
- **Mask Tuning**: Structured pruning based on Fisher Information (Kwon et al., 2022).
- **Prune+Distill**: Structured pruning followed by standard knowledge distillation.

---

## 4. Hyperparameters & Configuration

All default hyperparameters for APT and baselines are exposed through the configuration files in `configs/`.

### Table 6. Hyperparameters Used in APT Experiments
| Hyperparameter | Default Value | Description |
| :--- | :--- | :--- |
| `pruning_start_step` | 100 | Step to start computing salience and pruning |
| `pruning_end_step` | 1000 | Step to finish the gradual pruning schedule |
| `target_sparsity` | 0.60 | Target structured sparsity ratio (e.g., 60%) |
| `r_apt` | 8 | Initial rank for the APT adapters |
| `tuning_budget` | 0.10 | Parameter budget for adaptive tuning allocation |
| `distill_temp` | 2.0 | Temperature for self-distillation loss |
| `tau` | 0.9 | Weight coefficient for self-distillation loss |
| `batch_size` | 32 / 128 | Batch size used for training (task-dependent) |

---

## 5. Reproduction Registry & Artifacts

The repository maintains a strict registry of environments, datasets, experiments, and artifacts to satisfy the global reproduction contract.

### 5.1. Registry Files
- **`results/evidence_contract_matrix.json`**: Maps paper claims, formulas, and tables to their code implementations.
- **`results/experiment_registry.json`**: Lists all registered experiment configurations and execution statuses.
- **`results/environment_registry.json`**: Details the hardware/software environment requirements and readiness.
- **`results/dataset_registry.json`**: Tracks dataset paths, preprocessing status, and validation metrics.
- **`results/artifact_manifest.json`**: Manifest of all generated tables, figures, and metric files.
- **`results/sensitivity_report.json`**: Sensitivity analysis of hyperparameters (e.g., sparsity, rank, distillation weight).

### 5.2. Target Tables and Figures
- **Table 1**: Efficiency comparison of existing methods and APT.
- **Table 2**: RoBERTa and T5 pruning with APT compared to baselines under 60% sparsity on SST2.
- **Table 3**: LLaMA 2 7B 30% sparsity pruning results on Alpaca.
- **Table 4**: Ablation study of salience-based allocation and APT adapter on RoBERTa-base.
- **Table 5**: LLaMA 2 7B model ablation results under 30% and 50% sparsity.
- **Table 7**: Comparison of APT to existing unstructured pruning baselines with PEFT.
- **Table 8**: Detailed results of RoBERTa pruning compared to LoRA+Distill.
- **Table 9**: LLaMA 2 7B and 13B 30% sparsity pruning results on Open LLM Leaderboard.
- **Table 10**: Ablation study of distillation strategies.
- **Table 11**: Raw efficiency metrics (TTA, peak memory, inference speed) for RoBERTa and T5.
- **Table 12**: Raw efficiency metrics for LLaMA 2 7B on Alpaca.
- **Figure 3**: Task performance vs. relative inference efficiency.
- **Figure 4**: Performance-efficiency tradeoff compared to baseline methods.
- **Figure 5**: Detailed analysis of different initial/target sparsities.

---

## 6. Execution Guide

### 6.1. Environment Readiness
To verify that the environment is ready and all dependencies are met, run: