# LCA-on-the-Line: Benchmarking Out-of-Distribution Generalization with Class Taxonomies

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Code reproduction repository for the paper **"LCA-on-the-Line: Benchmarking Out-of-Distribution Generalization with Class Taxonomies"**.

This repository provides a complete implementation for evaluating 75 pretrained models (36 Vision Models + 39 Vision-Language Models) across ImageNet (in-distribution) and 5 OOD datasets, measuring the correlation between ID LCA distance and OOD generalization performance.

<!-- reference_grounding: paperbench_ref_001 references/depth/stereo/README.md -->
<!-- reference_grounding: paperbench_ref_001 torchvision/models/detection/mask_rcnn.py -->
<!-- reference_grounding: paperbench_ref_001 torchvision/models/detection/keypoint_rcnn.py -->

## Paper Overview

**Main Contributions:**
1. **LCA-on-the-Line Phenomenon**: Discovery of strong linear correlation between in-distribution LCA distance and out-of-distribution accuracy across diverse model families
2. **Unified Evaluation Framework**: First approach to uniformly measure model robustness across Vision Models (VMs) and Vision-Language Models (VLMs)
3. **Latent Taxonomy Inference**: K-means-based method to infer hierarchical class structure from pretrained model features
4. **Hierarchy-Aware Training**: Soft labeling methods using WordNet and latent hierarchies to improve OOD generalization
5. **Comprehensive Benchmark**: Evaluation of 75 models on 6 datasets with taxonomic mistake severity analysis

**Key Findings:**
- ID LCA distance is a stronger predictor of OOD performance than ID Top-1 accuracy (except ImageNet-v2)
- The correlation holds consistently across VMs and VLMs
- Soft labeling with hierarchical information improves OOD generalization without sacrificing ID accuracy
- Latent hierarchies from K-means clustering provide competitive alternatives to WordNet

## Repository Structure