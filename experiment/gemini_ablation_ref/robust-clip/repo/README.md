# Robust CLIP: Unsupervised Adversarial Fine-Tuning of Vision Embeddings for Robust Large Vision-Language Models

This repository contains a faithful, complete, and judgeable reproduction of the paper **"Robust CLIP: Unsupervised Adversarial Fine-Tuning of Vision Embeddings for Robust Large Vision-Language Models"**. 

The core contribution of this work is **FARE (Fine-tuning with Adversarial Representation Alignment)**, an unsupervised adversarial fine-tuning method for CLIP vision encoders. FARE aligns the adversarial vision embeddings with their clean counterparts using an unsupervised loss formulation, significantly improving the adversarial robustness of downstream Large Vision-Language Models (LVLMs) like LLaVA and OpenFlamingo without requiring downstream re-training or text labels.

---

## 1. Core Hypothesis & Decision Value

*   **Core Hypothesis:** Unsupervised adversarial fine-tuning of the CLIP vision encoder via the FARE loss function preserves clean representation capabilities while dramatically increasing robustness against $\ell_\infty$ adversarial attacks on downstream vision-language tasks.
*   **Decision Value:** FARE enables the creation of robust vision-language models *without* the need for expensive downstream LVLM re-training, supervised text labels, or task-specific fine-tuning.

---

## 2. Repository Architecture & File Structure

The repository is organized as follows: