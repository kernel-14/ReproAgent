# What Will My Model Forget? Forecasting Forgotten Examples in Language Model Refinement

This repository contains the faithful, complete, and judgeable reproduction of the paper **"What Will My Model Forget? Forecasting Forgotten Examples in Language Model Refinement"**.

## 1. Project Overview & Core Hypothesis

### Core Hypothesis
A unified data pipeline and method interface can support both forecasting training and refinement replay experiments. By decoupling the forecasting models (Representation-based, Logit-change based, and Frequency-Threshold based) from the sequential model refinement loop, we can accurately predict which upstream pretraining examples will be forgotten when fixing specific errors in pretrained language models (PTLMs), and use these forecasts to perform targeted replay that mitigates catastrophic forgetting.

### Key Contributions & Findings
- **Representation-based Forecasting**: Outperforms other baselines by using a sigmoid-wrapped inner product of example representations to predict the probability of forgetting.
- **Logit-change based Forecasting**: Leverages the transfer of logit changes of the first output tokens on upstream pretraining examples when fixing prediction errors of online learning examples.
- **Frequency-Threshold based Forecasting**: A baseline that relies solely on the frequency of forgetting while ignoring interactions between examples.
- **Replay-based Refinement**: Replaying examples forecasted to be forgotten significantly reduces the Exact Match (EM) Drop Ratio compared to random replay or no replay.

---

## 2. Environment & Dataset Registry

The repository manages environments and datasets through a structured registry system.

### Environment Registry (`results/environment_registry.json`)
Exposes environment factories for the following tasks:
- **P3-Upstream**: Upstream pretraining dataset from P3 (36 tasks, balanced sample of 100 examples per task, filtering out samples the model got wrong to form $\hat{D}_{PT}$).
- **P3-Test (ID/OOD)**: In-domain and out-of-domain test splits of P3.
- **SQuAD**: SQuAD dataset for refinement evaluation.
- **GLUE**: GLUE benchmark tasks for refinement evaluation.

### Dataset Registry (`results/dataset_registry.json`)
Defines dataset loaders and readiness checks:
- `make_dataset(config)`: Factory function to initialize and preprocess datasets.
- `dataset readiness check`: Validates that the required splits and mined errors are available.

---

## 3. Configuration & Setup Commands

### Configuration Schema
The configuration is managed via YAML files (`configs/default.yaml` and `configs/default_config.yaml`) and resolved into `results/config_resolved.json`.

Key hyperparameters include:
- `learning_rate`: Default is `1e-5` (as evaluated in Table 9).
- `batch_size`: Default is `8`.
- `gamma`: Threshold parameter for frequency-threshold forecasting.
- `representation_dim`: Dimension $H$ of the representation space (default `768`).
- `buffer_size`: Size of the replay buffer.
- `refinement_steps`: Number of steps for sequential model refinement.

### Documented Setup Commands
To set up the environment and run the readiness checks: