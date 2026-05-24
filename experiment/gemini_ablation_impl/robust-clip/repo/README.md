# Robust CLIP: Unsupervised Adversarial Fine-Tuning of Vision Embeddings for Robust Large Vision-Language Models

This repository contains the faithful, complete, and judgeable reproduction of the **Robust CLIP** framework, implementing **Unsupervised Adversarial Fine-Tuning (FARE)** of vision embeddings to secure Large Vision-Language Models (LVLMs) against adversarial attacks.

---

## 1. Project Overview & Core Contribution

Large Vision-Language Models (LVLMs) like LLaVA and OpenFlamingo are highly vulnerable to adversarial perturbations on the input images. **Robust CLIP** introduces **FARE** (Fine-Tuning with Adversarial Representation Alignment), an unsupervised adversarial fine-tuning method that aligns the perturbed/adversarial image embeddings with their clean counterparts from the original pre-trained CLIP model. 

### Key Highlights
* **Unsupervised Fine-Tuning**: FARE does not require text labels or paired image-text data, preserving the original CLIP embedding space.
* **Dual-Pronged Robustness**: Evaluated on both zero-shot classification tasks (ImageNet and its variants) and complex vision-language tasks (POPE, SQA-I, COCO, VQAv2).
* **Stealthy Targeted Attacks**: Defends against targeted $\ell_{\infty}$-attacks designed to force specific output strings (e.g., jailbreaks or malicious instructions).

---

## 2. Installation & Environment Setup

### Prerequisites
* Python 3.9 or higher
* PyTorch 1.10+ (with CUDA support recommended for full training/evaluation)
* Access to standard vision datasets (ImageNet-1k, CIFAR-10, CIFAR-100, STL-10, COCO, VQAv2)

### Installation Steps
1. Clone the repository: