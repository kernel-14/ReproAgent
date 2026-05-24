# Stay on Topic with Classifier-Free Guidance

This repository is a faithful reproduction of the paper "Stay on topic with Classifier-Free Guidance". It implements Classifier-Free Guidance (CFG) for autoregressive language models to improve adherence to prompts, reasoning capabilities, and code generation performance without additional training.

## Project Overview

The core contribution is the application of the CFG formula to the logit space of language models:
$$\log \widehat{\mathrm{P}_{\theta}}(w_{i} \mid w_{j<i}, c) = \log \mathrm{P}_{\theta}(w_{i} \mid w_{j<i}) + \gamma(\log \mathrm{P}_{\theta}(w_{i} \mid w_{j<i}, c) - \log \mathrm{P}_{\theta}(w_{i} \mid w_{j<i}))$$
where $\gamma$ is the guidance strength, $c$ is the conditioning prompt, and the second term represents the difference between the conditional and unconditional (or null-conditioned) distributions.

## Installation

```bash
# Clone the repository
git clone <repo_url>
cd stay-on-topic-cfg

# Install dependencies
pip install -e .

# For full model execution (optional)
pip install -e .[heavy]
```

## Configuration and Hyperparameters

Reproduction settings are managed via YAML configurations in `configs/`. Default hyperparameters follow the paper's specifications and the EleutherAI LLM Evaluation Harness as noted in the paper addendum.

- **Guidance Scale ($\gamma$):** Default is $1.5$ for general NLP tasks and $2.0$ for code generation.
- **Temperature:** Default is $0.2$ for program synthesis (Table 2).
- **Baselines:** Vanilla sampling ($\gamma=1$), Chain-of-Thought (CoT), and Instruction-tuned models.

### Key Symbols and Constants
- `w_p`: Prompt context used as conditioning $c$.
- `gamma`: Guidance strength (typical values: $1.0, 1.25, 1.5, 2.0, 3.0, 5.0$).
- `pass@k`: Evaluation metric for code generation ($k=1, 10, 100$).
- `flops_computation`: FLOPs measured per token during inference (Figure 9).

## Running Reproductions

The canonical entry point is `src/main.py`. It orchestrates the evaluation loops and artifact generation.

### Smoke Run (Validation)
```bash
python src/main.py --mode runtime_smoke
```

### Full Reproduction Route
```bash
python main.py --config configs/default.yaml
```

## Reproduction Artifacts

The following artifacts are generated in the `results/` directory, preserving the semantics and captions from the original paper:

| Artifact | Description |
| :--- | :--- |
| **Figure 1** | Latent space illustration showing how increasing $\gamma$ increases the importance of the prompt "Today in France,". |
| **Table 1** | Demonstration of CFG-guided generation for assistant-style prompts using GPT4All with $\gamma=5$. |
| **Figure 2** | CFG's impact on CoT (GSM8K). Shows accuracy vs. invalidly formatted answers (Note: Addendum clarifies figures are left/right). |
| **Figure 3** | HumanEval task count comparison between $\gamma=1$ and $\gamma=1.25$ for CodeGen-350M-mono. |
| **Figure 4** | Evaluator votes showing system-prompt adherence is optimal at $\gamma=3$. |
| **Table 2** | CodeGen results with temperature $0.2$ across various $\gamma$ strengths. |
| **Table 3** | Vocabulary ranking for "The dragon flew over Paris, France" showing token encouragement/discouragement. |
| **Table 4** | % increase in classification likelihood for sentiment (IMDB) and toxicity. |
| **Table 5** | General NLP benchmarks for GPT2 (G), Pythia (P), and LLaMA (L) comparing $\gamma=1$ vs $\gamma=1.5$. |
| **Figure 9** | Accuracy vs. FLOP per token at inference. |
| **Figure 10** | Decision-making function for model enhancement based on size and VRAM. |

## Implementation Details

### CFG Logit Transformation
Implemented in `src/cfg_logit_transform.py`.
`reference_grounding: chunk_005`
```python
def apply_cfg(logits_cond, logits_uncond, gamma):
    return logits_uncond + gamma * (logits_cond - logits_uncond)
```

### Mechanistic Analysis
The repository includes tools to visualize the vocabulary ranking difference:
$$\log \mathrm{P}(w_{t} \mid w_{<t}) - \log \mathrm{P}(w_{T} \mid \hat{w})$$
This analysis (Table 3) demonstrates how CFG encourages topic-specific tokens while discouraging generic or out-of-distribution completions.

### Addendum Clarifications
- **Figure 2:** The caption labels 'top' and 'bottom', but the implementation renders them left and right as per the addendum.
- **Section 3.3:** The reference to "Table 3" for HumanEval task counts is corrected to "Figure 3".
- **FLOPs:** Measured using the formula from `google-research/electra`.

## Results Summary
Reproduction results are aggregated in `results/summary.json` and `results/tables/experiment_results.csv`.
```bash
# View aggregated metrics
cat results/summary.json
```