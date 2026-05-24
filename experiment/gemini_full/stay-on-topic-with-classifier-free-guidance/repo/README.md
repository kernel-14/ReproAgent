# Stay on topic with Classifier-Free Guidance (Reproduction)

This repository provides a faithful reproduction of the methods and experiments described in the paper "Stay on topic with Classifier-Free Guidance". The core contribution is the application of Classifier-Free Guidance (CFG) to autoregressive language models to improve adherence to prompts and performance on various benchmarks.

## Introduction

Classifier-Free Guidance (CFG) for language models allows for the redirection of model output towards a specific condition (prompt) without requiring a separate classifier. This is achieved by shifting the logits during sampling based on the difference between conditional and unconditional (or negative) prompt distributions.

**Core Formula (Equation 7):**
$$\log \widehat{\mathrm{P}_{\theta}}\left(w_{i} \mid w_{j<i}, c\right) = \log \mathrm{P}_{\theta}\left(w_{i} \mid w_{j<i}\right) + \gamma \left(\log \mathrm{P}_{\theta}\left(w_{i} \mid w_{j<i}, c\right) - \log \mathrm{P}_{\theta}\left(w_{i} \mid w_{j<i}\right)\right)$$
where $\gamma$ is the guidance strength.

## Requirements

- Python 3.8+
- `numpy`, `pyyaml`, `pydantic`
- Optional: `torch`, `transformers`, `datasets` (for full model execution)

Install dependencies: