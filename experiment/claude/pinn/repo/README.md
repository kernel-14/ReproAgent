# PINN Loss-Landscape Reproduction

Standalone PaperBench reproduction repository for **“Challenges in Training PINNs: A Loss Landscape Perspective.”**

This repository implements the paper-derived PINN problem surfaces, optimizer comparison protocol, loss-landscape diagnostics, and artifact-writing routes for the convection PDE, wave PDE, and reaction ODE experiments. The default command is a bounded smoke/readiness run: it exercises the real registry, sampling, model/loss, evaluation, Hessian-diagnostic, and reporting paths with tiny budgets and writes contract artifacts, but it does **not** claim paper-scale training results.

It is a standalone repository for PINN code reproduction across the convection PDE, wave PDEs, and reaction ODE tasks. The default mode is dry-run/smoke execution, while the configured full-budget metadata remains available for review and rerun planning.

reference_grounding: paper:unit_003 paper.md  
reference_grounding: paper:unit_008 paper.md  
reference_grounding: paper:paper_method_core paper.md

## Scope and paper obligations

Implemented reproduction target:

- PINN environments/problems:
  - `convection`
  - `reaction`
  - `wave`
- Differentiable residuals and losses:
  - PDE/ODE residual terms are computed through automatic differentiation when PyTorch is available.
  - Loss components are named and recorded separately instead of returning only an opaque scalar total.
  - Boundary/initial-condition losses and residual losses are exposed through a loss-term registry.
- Optimizers and comparison semantics:
  - `Adam`
  - `L-BFGS`
  - `Adam+L-BFGS`
  - `NysNewton-CG` / `NNCG` after Adam+L-BFGS
  - gradient descent fine-tuning baseline after Adam+L-BFGS
- Metrics:
  - training loss
  - component losses
  - relative `L2RE`
  - gradient norm
  - Hessian spectrum / preconditioned-Hessian spectrum schema
  - estimated condition numbers
  - per-iteration timing schema for L-BFGS and NNCG
- Artifact routes:
  - paper figure/table route metadata is preserved even when the default smoke run only writes schema/readiness artifacts.

The repository intentionally excludes blacklisted source code and does not depend on the blacklisted repository `https://github.com/pratikrathore8/opt_for_pinns`.

## Addendum decisions

The following addendum clarifications are binding for this implementation:

- Figure 3 and Figure 7 spectral-density experiments are retained as active diagnostic routes.
- Hyperparameters for Figures 3 and 7 are represented as selected protocol settings rather than exhaustive smoke-time sweeps.
- Best seeds are registered as:
  - convection: `345`
  - reaction: `456`
  - wave: `567`
- Figure 6 and Figure 9 are out of scope and are not required to be reproduced. Their paper context is preserved only as a documented exclusion, not as a claimed result.

## Repository layout
