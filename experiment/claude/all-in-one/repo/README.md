# All-in-one Simulation-based Inference — PaperBench Reproduction

This repository is a standalone, code-generation-only reproduction of the paper **“All-in-one simulation-based inference”**. It implements the canonical route for a Simformer-style transformer score-based diffusion model over joint simulator variables, including arbitrary conditional sampling, structured attention masks, benchmark evaluation interfaces, baseline adapters, and the interval-guided Hodgkin-Huxley protocol.

The default command is intentionally safe: it runs a bounded smoke route that exercises the real implementation surfaces with tiny budgets. It does **not** claim paper-scale training, benchmark scores, or completed numerical reproduction.

No code is copied from the blacklisted repository `https://github.com/mackelab/simformer`.

reference_grounding: paper:unit_011 paper.md  
reference_grounding: paper:unit_011 addendum.md

## Quick start

Run the bounded route used for smoke validation and open-source export:

```bash
python run_experiments.py --mode runtime_smoke --results-dir results
```

The root `run_config.json` records this safe default route and the expected
artifact contract. It is a bounded readiness configuration, not a paper-scale
benchmark claim.

## Core Semantics

The Simformer core uses a Variance Exploding SDE with `sigma_min=1e-4`,
`sigma_max=15`, `t in [1e-5, 1]`, zero drift, and
`g(t)=sigma(t)*sqrt(2*(log(sigma_max)-log(sigma_min)))`. The perturbation kernel
returns `x_t=x_0+sigma(t)*epsilon` and the denoising score target
`-epsilon/sigma(t)`.

Joint SBI tokens are ordered as parameter variables followed by observation
variables. Token representations concatenate identifier, repeated scalar value,
Fourier metadata/time context, and binary condition-state embeddings; duplicate
semantic variable names share identifier ids.

Training condition masks are sampled from the five paper families:
`joint_all_false`, `posterior_theta_given_x`, `likelihood_x_given_theta`,
`mask_probability_0.3`, and `mask_probability_0.7`. Directed dependency masks
`M_E` are not static dense masks: conditioning graph inversion derives
condition-specific attention masks from each `M_C`, and the model uses that
graph when no explicit attention mask is supplied.
