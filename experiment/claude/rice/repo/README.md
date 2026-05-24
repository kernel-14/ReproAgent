# RICE: Breaking Through the Training Bottlenecks of Reinforcement Learning with Explanation

Official reproduction repository for the paper **"RICE: Breaking Through the Training Bottlenecks of Reinforcement Learning with Explanation"**.

## Overview

This repository provides a complete implementation of the RICE (Reinforcement learning with Importance-guided Correction and Explanation) method, which addresses the sample inefficiency bottleneck in deep reinforcement learning by:

1. **Pre-training**: Training initial DRL agents using standard RL algorithms (PPO)
2. **Explanation**: Identifying critical states where the agent's decision-making is most important
3. **Refining**: Performing targeted exploration and learning from these critical states to improve agent performance

RICE achieves comparable or better final performance while requiring significantly fewer samples compared to training from scratch or traditional fine-tuning approaches.

## Repository Structure

## Paper Contract Index

The source-level contract index in `src/rice_contracts.py` maps the paper obligations to executable code surfaces. It covers Appendix C.2 environment setup, PPO pretraining contracts, Section 3.3 StateMask and RICE/Ours objectives, Algorithm 2 refinement, and Section 4 experiment protocols.

Key covered surfaces include:
- `Hopper-v3`, `Walker2d-v3`, `Reacher-v2`, `HalfCheetah-v3`, sparse `Hopper-v3`, sparse `HalfCheetah-v3`, selfish mining, CAGE Challenge 2 network defence, and `Macro-v1` autonomous driving powered by MetaDrive.
- Observation normalization during DRL training for Walker2d, HalfCheetah, and sparse HalfCheetah.
- PPO-compatible pretrained policy contracts, including the selfish mining 4-layer MLP with hidden sizes 128, 128, 128, and 128.
- Original StateMask with mask output 0 for critical steps and 1 otherwise, objective `J(theta)=min |eta(pi)-eta(bar_pi)|`, and primal-dual optimization.
- RICE/Ours with transformed objective `J(theta)=max eta(bar_pi)`, PPO optimization, an alpha-controlled extra reward when the mask output is 1, and rollout selectors for retraining-free critical-step selection.
- StateMask-R, PPO fine-tuning, JSRL, mixed initial state distributions with mutable `p`, and RND exploration bonuses with mutable `lambda`.
- Experiment I-V protocols with Top-K ratios 10%, 20%, 30%, 40%, 500-trajectory fidelity measurement, three seeds, efficiency timing, sparse MuJoCo, real-world tasks, and ablation trend assertions.

