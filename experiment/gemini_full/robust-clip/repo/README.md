# Robust CLIP: Unsupervised Adversarial Fine-Tuning of Vision Embeddings for Robust Large Vision-Language Models

This repository contains the faithful, complete, and judgeable reproduction of the paper **"Robust CLIP: Unsupervised Adversarial Fine-Tuning of Vision Embeddings for Robust Large Vision-Language Models"**. 

The core contribution of this work is **FARE (Fine-tuning with Adversarial Representation Alignment)**, an unsupervised adversarial fine-tuning method for CLIP vision encoders. FARE aligns the representation of adversarially perturbed images with their clean counterparts, significantly improving the robustness of downstream Large Vision-Language Models (LVLMs) like LLaVA and OpenFlamingo without requiring expensive supervised labels.

---

## 1. Canonical Run Path & Setup Commands

### Environment Setup
To set up the environment, ensure you have Python 3.9+ installed. Install the required dependencies: