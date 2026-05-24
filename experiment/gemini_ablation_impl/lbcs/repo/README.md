# Refined Coreset Selection (LBCS) Reproduction

This repository provides a faithful reproduction of the Lexicographic Bilevel Coreset Selection (LBCS) method as described in the paper: **"Refined Coreset Selection: Towards Minimal Coreset Size under Model Performance Constraints"**.

## Overview
Lexicographic Bilevel Coreset Selection (LBCS) is a method for Refined Coreset Selection (RCS) that aims to minimize both the training loss (performance) and the coreset size (efficiency) using a lexicographic optimization approach. This reproduction covers the core methodology, benchmark comparisons, and robustness analyses.

## Installation and Setup
### Environment Requirements
- Python 3.8+
- PyTorch (Lazy loading supported for minimal environments)
- Torchvision
- NumPy, PyYAML, Matplotlib (for figures)

### Setup Commands