# Sequential Neural Score Estimation (SNPSE)

PaperBench code reproduction for the paper: **Sequential Neural Score Estimation: Likelihood-Free Inference with Conditional Score Based Diffusion Models**

This repository implements Neural Posterior Score Estimation (NPSE) and Truncated Sequential Neural Posterior Score Estimation (TSNPSE, Algorithm 1) for simulation-based inference tasks.

## Paper Context

The paper introduces score-based diffusion models for likelihood-free Bayesian inference:
- **NPSE**: Neural Posterior Score Estimation using conditional score-based diffusion models
- **TSNPSE**: Truncated sequential variant (Algorithm 1) that improves sample efficiency
- **SNPSE-A/B/C**: Alternative sequential approaches explored in ablation studies

**Key Results**:
- Figure 1: Posterior visualization on Two Moons benchmark
- Figures 2-3: Performance on 8 SBI benchmark tasks (non-sequential vs sequential methods)
- Figure 4: Pyloric neuron experiment results
- Figures 5-6: Method comparisons (NPSE vs NLSE, TSNPSE vs SNPSE-A/B)
- Figures 7-8: Posterior marginals and coverage plots

**Addendum Notes**:
- TSNPE and SNVI results (Section 5.3) are taken from respective papers, not replicated
- C2ST metric uses `sbibm` library with default hyperparameters

## Reference Grounding

This implementation adapts patterns from reference repositories:
- `reference_grounding: paperbench_ref_001 l5pc/docs/config.md` - Configuration structure for multi-round inference
- `reference_grounding: paperbench_ref_001 sbi/sbi/inference/snpe/snpe_a.py` - SNPE method interface patterns
- `reference_grounding: paperbench_ref_001 sbi/sbi/inference/abc/abc_base.py` - Simulator and prior interface

## Installation