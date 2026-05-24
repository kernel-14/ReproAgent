# Stochastic Interpolants with Data-Dependent Couplings

This repository provides a faithful, complete, and judgeable reproduction of the methods described in the paper **"Stochastic Interpolants with Data-Dependent Couplings"**.

## Overview

The core contribution of this work is a general framework for constructing data-dependent couplings between base and target densities within the stochastic interpolant formalism. Unlike standard flows and diffusions that rely on independent Gaussian couplings, this method constructs couplings $\rho_0(x_0 | x_1)$ that leverage information from the target data $x_1$, leading to straighter probability flows and improved performance in conditional generation tasks like image inpainting and super-resolution.

### Key Features
- **Stochastic Interpolant Framework**: Implementation of $I_t = \alpha_t x_0 + \beta_t x_1$ and its time derivatives.
- **Data-Dependent Couplings**: Specialized couplings for inpainting (mask-aware noise) and super-resolution.
- **Velocity Field Modeling**: UNet architecture with sinusoidal time embeddings and mask conditioning.
- **Numerical Integration**: Support for Euler and Runge-Kutta (RK4) solvers for probability flow ODEs.
- **Reproduction Suite**: Automated scripts to generate all tables and figures from the paper.

## Installation

The project requires Python 3.8+ and PyTorch.