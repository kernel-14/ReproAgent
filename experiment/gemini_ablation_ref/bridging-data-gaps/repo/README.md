# Bridging Data Gaps in Diffusion Models with Adversarial Noise-Based Transfer Learning (DPMs-ANT)

This repository contains a faithful, complete, and judgeable reproduction of the paper **"Bridging Data Gaps in Diffusion Models with Adversarial Noise-Based Transfer Learning"** (DPMs-ANT).

## Method Overview

Few-shot image generation aims to adapt a pre-trained generative model to a target domain with limited samples (e.g., 10-shot). While GANs can directly compare clean generated images with target images, Diffusion Probabilistic Models (DPMs) operate on noisy images, making direct comparison challenging. 

**DPMs-ANT** addresses this by introducing two key components:
1. **Similarity-Guided Training**: Guides the reverse process using a binary classifier $p_\phi$ trained to distinguish between source and target domains. The guided reverse process is formulated as:
   $$p_{\theta_{(\mathcal{S}, \mathcal{T})}, \phi}(x_{t-1} \mid x_{t}, y=Y) \approx \mathcal{N}\left(x_{t-1} ; \mu_{\theta_{(\mathcal{S}, \mathcal{T})}}+\sigma_{t}^{2} \gamma \nabla_{x_{t}} \log p_{\phi}\left(y=Y \mid x_{t}\right), \sigma_{t}^{2} \mathbf{I}\right)$$
   where $\gamma$ is the similarity guidance scale.

2. **Adversarial Noise Selection (ANT)**: Prevents overfitting to the few-shot target samples by finding adversarial noise perturbations $\epsilon^\star$ that maximize the reconstruction error, forcing the model to learn robust features. The multi-step gradient ascent for noise selection is defined as:
   $$\epsilon^{j+1} = \operatorname{Norm}\left(\epsilon^{j} + \omega \nabla_{\epsilon^{j}}\left\|\epsilon^{j}-\epsilon_{\theta}\left(\sqrt{\bar{\alpha}_{t}} x_{0}+\sqrt{1-\bar{\alpha}_{t}} \epsilon^{j}, t\right)\right\|^{2}\right)$$
   where $\omega$ is the step size, and $J$ is the number of inner steps.

## Repository Structure

- `main.py`: Main entrypoint for running experiments, evaluations, and generating artifacts.
- `configs/default.yaml`: Configuration file containing default hyperparameters, sweeps, and environment registries.
- `configs/dataset_registry.json`: Registry of source and target datasets.
- `src/models/adaptor.py`: Implementation of the parameter-efficient Adaptor module.
- `src/methods/dpms_ant.py`: Core implementation of DPMs-ANT (Similarity-Guided Training and Adversarial Noise Selection).
- `src/training/trainer.py`: Training loop for fine-tuning DDPM/LDM models with ANT.
- `src/data/pipeline.py`: Data pipeline supporting 10-shot sampling and target domain loading.
- `src/data/data_env_setup.py`: Environment setup and verification utilities.
- `src/evaluation/metrics.py`: Implementations of FID and Intra-LPIPS metrics.
- `src/baselines/wrapper.py`: Wrappers for baseline methods (DDPM, LDM, DDPM-PA, TGAN, ADA, EWC, CDC, DCL).
- `src/experiments/eval_baselines.py`: Orchestration of baseline evaluations.
- `src/reporting/experiment_artifacts.py`: Generation of tables and figures.

## Hyperparameters

### Default Hyperparameters
- **Batch Size**: 64
- **Training Iterations**: 300
- **Shot Count**: 10
- **Adversarial Inner Steps ($J$)**: 10
- **Adversarial Step Size ($\omega$)**: 0.02

### Target-Specific Hyperparameters (from Addendum Table 3)
- **DDPM - FFHQ to Babies**:
  - Learning rate: $5 \times 10^{-6}$
  - $C$ (Adaptor bottleneck): 8
  - $\omega$: 0.02
  - $J$: 10
  - $\gamma$: 3.0
  - Training iterations: 160
- **DDPM - FFHQ to Sunglasses**:
  - Learning rate: $5 \times 10^{-5}$
  - $C$: 8
  - $\omega$: 0.02
  - $J$: 10
  - $\gamma$: 15.0
  - Training iterations: 200
- **DDPM - FFHQ to Raphael**:
  - Learning rate: $5 \times 10^{-5}$
  - $C$: 8
  - $\omega$: 0.02
  - $J$: 10
  - $\gamma$: 10.0
  - Training iterations: 500

### Classifier Training
- **Optimizer**: Adam
- **Learning Rate**: $1 \times 10^{-4}$
- **Batch Size**: 64
- **Iterations**: 300
- **Pre-trained Classifiers**:
  - DDPM: `256x256_classifier.pt`
  - LDM: `64x64_classifier.pt`

## Execution Commands

### 1. Environment Readiness Check
Verify that the environment, datasets, and pre-trained models are correctly configured:
```bash
python main.py --mode check_readiness
```
This command generates `results/environment_registry.json` and `results/environment_readiness.json`.

### 2. Bounded Smoke Run
Run a fast, bounded training and evaluation loop to verify the pipeline:
```bash
python main.py --mode smoke
```

### 3. Full Experiment Execution
To run the full suite of experiments and generate all paper tables and figures:
```bash
python main.py --mode full
```

## Generated Artifacts

All results are saved in the `results/` directory:
- `results/environment_registry.json`: Registry of source and target domains.
- `results/environment_readiness.json`: Status of environment checks.
- `results/table_1_results.json` / `results/table_2_results.json`: Quantitative comparisons (FID and Intra-LPIPS).
- `results/figure_1.png` to `results/figure_6.png`: Visualizations of generated samples, gradient heatmaps, and ablation studies.