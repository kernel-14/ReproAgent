# Batch and Match: Black-Box Variational Inference with a Score-Based Divergence

This repository provides a faithful, complete, and judgeable reproduction of the "Batch and Match" (BaM) algorithm as described in the paper "Batch and match: black-box variational inference with a score-based divergence".

## Project Summary

The architecture is designed to faithfully reproduce the BaM algorithm using JAX. It separates the core mathematical implementation (divergences and algorithms) from the specific applications (synthetic targets, hierarchical Bayesian models, and CIFAR-10 VAE). The use of a unified entrypoint and explicit configuration files ensures that all experiments are easily runnable and verifiable.

## Installation & Setup

To set up the environment, run the following commands: