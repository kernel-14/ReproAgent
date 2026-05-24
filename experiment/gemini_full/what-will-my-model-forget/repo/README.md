# What Will My Model Forget? Forecasting Forgotten Examples in Language Model Refinement

This repository provides a faithful reproduction of the methods and experiments described in the paper: *"What Will My Model Forget? Forecasting Forgotten Examples in Language Model Refinement"*.

## Project Overview

The project implements a framework for forecasting which upstream pretraining examples a language model (LM) will forget when it is refined to fix specific errors. It includes implementations of:
- **Frequency-Threshold based forecasting**: A baseline using historical forgetting frequency.
- **Logit-Change based forecasting**: Approximating logit changes using the first-order Taylor expansion and the Neural Tangent Kernel (NTK).
- **Representation-Based forecasting**: Mapping the interaction between refinement and upstream examples into a shared representation space.

## Installation & Setup

### Prerequisites
- Python 3.8+
- PyTorch
- Transformers
- PEFT (for LoRA implementations)

### Setup Commands