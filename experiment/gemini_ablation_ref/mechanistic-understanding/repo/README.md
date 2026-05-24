# A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and Toxicity

This repository contains a faithful, complete, and judgeable reproduction of the methods, data processing, evaluation interfaces, baselines, metrics, and artifacts described in the paper **"A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and Toxicity"**.

---

## 1. Project Overview & Hypothesis

### Core Hypothesis
Toxic vectors can be extracted from the residual stream of pre-trained models (GPT-2, Llama-2) using linear probes and compared against oracle baselines; suppressing these vectors via residual stream interventions or DPO alignment reduces toxicity.

### Decision Value
Establishes the baseline 'toxic' representation and oracle performance required for all subsequent intervention and alignment analysis.

---

## 2. Environment & Dataset Registry

The environment and dataset configurations are managed dynamically via a registry system.

### Environment Registry
The environment registry is defined in `configs/default.yaml` and serialized to `results/environment_registry.json`. It includes:
*   **Jigsaw Dataset**: Binary toxicity classification task (561,808 comments, 90:10 train/validation split).
*   **RealToxicityPrompts**: Toxicity generation evaluation (295 prompts that originally elicit toxic tokens).
*   **Wikitext**: Language modeling perplexity (PPL) evaluation.

### Environment Readiness Check
To verify the environment and dataset availability, run: