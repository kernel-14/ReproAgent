# LCA-on-the-Line: Benchmarking Out-of-Distribution Generalization with Class Taxonomies

This repository contains a faithful, complete, and judgeable reproduction of the paper **"LCA-on-the-Line: Benchmarking Out-of-Distribution Generalization with Class Taxonomies"**.

---

## 1. Project Summary & Core Hypothesis

The core contribution of this work is **LCA-on-the-Line**, a novel framework that leverages class taxonomies to measure and predict out-of-distribution (OOD) generalization across Vision Models (VMs) and Vision-Language Models (VLMs). 

### Core Hypothesis
- **In-Distribution (ID) Lowest Common Ancestor (LCA) distance** has a strong correlation with **Out-of-Distribution (OOD) performance** (Top-1/Top-5 accuracy) across diverse model families.
- Models that capture transferable, non-spurious features make semantically milder mistakes (lower LCA distance) when they fail, which serves as a robust surrogate for OOD generalization.
- Leveraging class taxonomies as soft labels during training or using taxonomy-aligned prompts during zero-shot evaluation significantly boosts OOD generalization.

---

## 2. Environment & Setup

### Prerequisites
- Python >= 3.8
- Standard scientific stack: `numpy`, `scipy`, `pyyaml`, `click`, `matplotlib`, `scikit-learn`
- Optional heavy dependencies (for full model execution): `torch`, `torchvision`, `transformers`, `datasets`

### Installation
To install the package and its dependencies in editable mode: