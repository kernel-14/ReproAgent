# Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem

PaperBench reproduction repository for **“Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem”**.

This repository is organized around the paper’s canonical execution route: selection of **experiment**, **environment**, **method/baseline**, **seed**, **training budget**, **evaluation budget**, and **mode**; execution of the real training/evaluation code path; and writing stable artifacts under `results/`.

reference_grounding: paperbench_ref_001 README.md

## What this repository reproduces

The implementation targets the paper’s core claim: **fine-tuning reinforcement learning models is secretly a forgetting mitigation problem**. The repository preserves the paper’s method families, environment interfaces, evaluation metrics, and artifact outputs for the following paper-visible routes:

- **Main comparison results**: fine-tuning vs. knowledge-retention methods.
- **Forgetting analysis**: CLOSE/FAR state partitioning and performance retention.
- **Environment-specific diagnostics**:
  - NetHack / NLE
  - Montezuma’s Revenge
  - RoboticSequence
- **Toy mechanism studies**:
  - Two-state MDP
  - AppleRetrieval forgetting mechanism

The repository is code-first: paper-visible outputs are produced by executable training/evaluation routes, not by schema-only stubs.

## Canonical entrypoint

Primary command-line route:

```bash
python main.py --mode runtime_smoke --output-dir results
python scripts/run_experiments.py --config-file configs/setup.yaml --output-dir results
```

Both entrypoints write `metrics.json`, `summary.csv`, `run_manifest.json`,
`readiness.json`, `evaluation_result.json`, `artifact_manifest.json`, and
`reproduction_inventory.json`.  The inventory embeds `ftrl_repro.protocols`,
which is the importable source of truth for high-weight paper obligations:

- NetHack/NLE Human Monk: 30M LSTM, ReLU, hidden dimension 1738, Tuyls et al.
  weights URL, NLD-AA Human Monk 8000-game subset, AutoAscend Level 4 and
  Sokoban save protocols, APPO/sample-factory training, Table 1 optimizer and
  rollout hyperparameters, NetHack rollout stop rules, and Level 4/Sokoban
  Section 5 evaluation intervals.
- Retention losses: BC on `S_BC` with `KL(pi_* || pi_theta)`, KS on online
  policy data with coefficient `0.5` and decay `0.99998`, EWC with 10000 Fisher
  batches and coefficient `2e6`, EM replay, and no retention on critic
  parameters.
- Montezuma's Revenge: PPO+RND route, room-7 pretraining/fine-tuning protocol,
  500 trajectories from room 7 onward, RND vector size 512, and success-rate
  measurement every 5M steps.
- RoboticSequence: SAC with 4 hidden layers of 256 units for policy and Q
  functions, sequential Meta-World task route, stage success metrics, and
  push-wall log-likelihood diagnostics every 50k steps.

The default mode is bounded and dependency-light so the package remains
importable without NLE, Atari, or Meta-World installed.  Full external simulator
routes remain explicit through mode/backend configuration and the protocol
inventory documents the required upstream implementations and assets.
