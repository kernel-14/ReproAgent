# LCA-on-the-Line: Benchmarking Out-of-Distribution Generalization with Class Taxonomies

This repository contains the faithful, complete, and judgeable reproduction of the paper **"LCA-on-the-Line: Benchmarking Out-of-Distribution Generalization with Class Taxonomies"**.

---

## 1. Core Contribution & Hypothesis

The primary hypothesis of this work is that **In-Distribution (ID) Lowest Common Ancestor (LCA) distance correlates linearly with Out-of-Distribution (OOD) Top-1 accuracy across diverse model families, and serves as a more robust predictor of OOD performance than traditional baselines like Accuracy-on-the-Line (Miller et al., 2021) and Agreement-on-the-Line (Baek et al., 2022).**

By measuring the semantic severity of a model's mistakes using a predefined taxonomic hierarchy (e.g., WordNet or latent hierarchies inferred via K-Means clustering), we can accurately estimate its generalization capability.

---

## 2. Environment & Dataset Registry

The reproduction pipeline defines a strict environment and dataset registry to ensure reproducibility across different setups.

### Environment Registry
*   **ImageNet (ID)**: The source in-distribution dataset used to evaluate baseline accuracy and LCA distance.
*   **LAION**: Pretraining environment used for Vision-Language Models (VLMs) to analyze the impact of web-scale supervision.

### Dataset Registry
*   `imagenet`: ImageNet-1K validation set (ID).
*   `imagenet_v2`: ImageNet-V2 (OOD).
*   `imagenet_r`: ImageNet-R (OOD).
*   `imagenet_sketch`: ImageNet-Sketch (OOD).
*   `imagenet_a`: ImageNet-A (OOD).
*   `objectnet`: ObjectNet (OOD).

### Environment Readiness Check
The environment readiness check is performed via `make_environment(config)` to verify dataset paths, HuggingFace access (using `trust_remote_code=True`), and model checkpoint availability. The status is written to `results/environment_readiness.json`.

---

## 3. Metric Registry

We implement and register the following metrics for evaluation:
*   **Top-1 Accuracy**: Standard classification accuracy.
*   **LCA Distance ($D_{LCA}$)**: The taxonomic distance between the predicted class $y'$ and ground-truth class $y$, defined as:
    $$D_{LCA}(y', y) := f(y) - f(N_{LCA}(y, y'))$$
    where $f(y)$ is the information content or depth of the node, and $N_{LCA}(y, y')$ is the Lowest Common Ancestor node.
*   **MAE (Mean Absolute Error)**: Used to measure the prediction error of OOD performance from ID metrics.
*   **Cross Entropy Loss**: Standard classification loss.
*   **Soft Labeling Loss**: Taxonomy-aware loss combining cross-entropy with normalized pairwise LCA distance.

---

## 4. Model Benchmark (75 Models)

The benchmark evaluates **75 pretrained models** spanning:
*   **36 Vision Models (VMs)**: Pretrained on ImageNet with supervised class labels (e.g., ResNet, ViT, ConvNeXt).
*   **39 Vision-Language Models (VLMs)**: Incorporating language supervision, including **LAION-supervised models** (e.g., OpenCLIP, CLIP).

### VLM Evaluation Protocol
For VLMs, we implement the **`per_sample_lowest_score_selection`** protocol to handle multi-prompt or zero-shot classification templates robustly, ensuring that the semantic severity of mistakes is measured accurately.

---

## 5. CLI Commands & Setup

### Installation & Setup
Ensure all dependencies are installed. Heavy packages like `torch`, `torchvision`, and `open_clip` are lazily imported to keep the environment lightweight.