# Test-Time Model Adaptation with Only Forward Passes (FOA)

This repository provides a faithful reproduction of the **Forward-Optimization Adaptation (FOA)** method as described in the paper *"Test-Time Model Adaptation with Only Forward Passes"*. FOA achieves test-time adaptation (TTA) for Vision Transformers (ViT) without requiring backpropagation, using a derivative-free Covariance Matrix Adaptation Evolution Strategy (CMA-ES) to optimize learnable prompts and an activation shifting mechanism to align target features with source distributions.

## Project Structure