# Robust CLIP: Unsupervised Adversarial Fine-Tuning of Vision Embeddings for Robust Large Vision-Language Models

This repository contains a complete reproduction implementation for the paper **"Robust CLIP: Unsupervised Adversarial Fine-Tuning of Vision Embeddings for Robust Large Vision-Language Models"**.

## Overview

This codebase implements:

- **FARE (Feature-Alignment Robust Embedding)**: Unsupervised adversarial fine-tuning of CLIP vision encoders
- **TeCoA Baseline**: Supervised adversarial fine-tuning for comparison
- **Adversarial Attack Pipeline**: PGD, AutoAttack, and targeted attacks for robustness evaluation
- **Zero-Shot Classification Evaluation**: Clean and adversarial accuracy on ImageNet and 10+ datasets
- **LVLM Robustness Evaluation**: Integration with LLaVA-1.5 and OpenFlamingo
- **Downstream Task Benchmarks**: POPE hallucination, SQA-I reasoning, jailbreaking attacks
- **Complete Artifact Generation**: All tables (1-8) and figures (1-5) from the paper

## Paper Contributions

The paper demonstrates that:

1. **FARE achieves superior robustness-accuracy trade-off**: Maintains clean performance while providing strong robustness to adversarial perturbations
2. **Unsupervised fine-tuning outperforms supervised**: FARE (unsupervised) achieves better downstream task performance than TeCoA (supervised)
3. **Robust vision encoders improve LVLM safety**: FARE-based LLaVA is more robust to adversarial attacks and jailbreaking attempts
4. **Minimal hallucination degradation**: Unlike TeCoA, FARE maintains low hallucination rates on POPE benchmark

## Key Results Reproduced

### Table 1: LVLM Robustness with Different CLIP Models
Evaluates OpenFlamingo and LLaVA on image captioning (COCO, Flickr30k) and VQA (VQAv2, TextVQA) under clean and adversarial conditions.

### Table 4: Image Classification Robustness
Clean and adversarial evaluation on:
- ImageNet (training dataset)
- Zero-shot: CIFAR-10, CIFAR-100, Caltech-101, Stanford Cars, Flowers102, DTD, EuroSAT, FER2013, Food101, GTSRB, MNIST, RESISC45, STL10

Perturbation budgets: ε ∈ {2/255, 4/255, 8/255, 16/255}

### Table 5: POPE Hallucination Evaluation
F1-scores for hallucination detection across random, popular, and adversarial splits.

### Table 6: SQA-I Evaluation
Chain-of-thought reasoning accuracy with LLaVA.

### Table 7: Jailbreaking Attack Success Rates
Success rates for visual adversarial jailbreaking attacks across harmful prompt categories.

### Figure 1: Radar Plot Comparison
Multi-task performance comparison of CLIP, TeCoA², and FARE² across vision-language tasks and zero-shot classification.

## Installation

### Requirements