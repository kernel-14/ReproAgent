# Sample-specific Masks for Visual Reprogramming-based Prompting (SMM)

This repository provides a faithful reproduction of the methods and experiments presented in the paper **"Sample-specific Masks for Visual Reprogramming-based Prompting"** (ICML 2024).

## Project Summary
Visual Reprogramming (VR) re-purposes pre-trained models for new target tasks by modifying the input space. Existing methods typically use a shared mask for all samples, which limits the flexibility of the adaptation. This project implements **Sample-specific Multi-channel Masks (SMM)**, which generates unique, three-channel masks for each input sample using a lightweight mask generator $f_{\text{mask}}$.

### Key Contributions
- **SMM Method**: A lightweight CNN-based mask generator $f_{\text{mask}}$ and a patch-wise interpolation module.
- **Multi-channel Masks**: Unlike single-channel masks, SMM uses three-channel masks to capture more variability across color channels.
- **Random Label Mapping (Rlm)**: An injective mapping from target labels to pre-trained model labels.

## Installation and Setup

### Environment Requirements
- Python 3.8+
- PyTorch, torchvision
- YAML, NumPy, Matplotlib, Pandas

### Setup Commands