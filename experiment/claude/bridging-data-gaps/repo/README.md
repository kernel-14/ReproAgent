# DPMs-ANT: Bridging Data Gaps in Diffusion Models with Adversarial Noise-Based Transfer Learning

Reproduction repository for the paper:
> **Bridging Data Gaps in Diffusion Models with Adversarial Noise-Based Transfer Learning**

This repository implements the two core strategies of **DPMs-ANT** for few-shot (10-shot) domain adaptation of Diffusion Probabilistic Models (DDPM and LDM):

1. **Similarity-Guided Training** — a domain classifier provides KL-divergence guidance over noisy images.
2. **Adversarial Noise Selection (ANT)** — PGD inner loop selects worst-case noise perturbations that maximize adaptation loss.

Combined with a lightweight **Shift Adaptor** (bottleneck W_down/W_up structure), DPMs-ANT achieves high-quality, diverse few-shot image generation.

---

## Table of Contents

- [Method Overview](#method-overview)
- [Paper Figures and Tables Index](#paper-figures-and-tables-index)
- [Repository Layout](#repository-layout)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Pre-trained Model Downloads](#pre-trained-model-downloads)
- [Classifier Training (Addendum)](#classifier-training-addendum)
- [Training (Fine-tuning)](#training-fine-tuning)
- [Sampling / Generation](#sampling--generation)
- [Evaluation](#evaluation)
- [Experiment Registry](#experiment-registry)
- [Configuration Reference](#configuration-reference)
- [Baselines](#baselines)
- [Ablation Studies](#ablation-studies)
- [Canonical Artifact Paths](#canonical-artifact-paths)
- [Smoke / Readiness Validation](#smoke--readiness-validation)

---

## Method Overview

### Algorithm 1 – DPMs-ANT Training Loop