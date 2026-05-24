# Unsupervised Zero-Shot Reinforcement Learning via Functional Reward Encodings

This repository is a runnable reproduction scaffold for the paper **“Unsupervised Zero-Shot Reinforcement Learning via Functional Reward Encodings” (FRE)**.

The canonical route is designed around the paper’s core hypothesis:

> learn a latent representation of reward functions from **unlabeled offline transitions**, then use that latent code to condition a policy for **zero-shot downstream task execution** without task-specific fine-tuning.

## What this repo implements

The repo is organized to close the paper’s main execution loop:

1. **Data pipeline** for offline trajectories and bounded smoke fixtures
2. **Random reward prior sampler** over paper-motivated reward families
3. **State-reward pair encoder** using permutation-invariant aggregation
4. **Policy conditioning adapter** for \(\pi(a \mid s, z)\)
5. **Offline training loop** for FRE pretraining
6. **Evaluation and baseline comparison** surfaces
7. **Artifact writing** for checkpoints, metrics, and paper-visible summaries
8. **Single entrypoint** for smoke and full runs

## Paper context and output mapping

### Figure 1: method overview
FRE learns latent representations over random unsupervised reward functions.

**Pipeline mapping:**

- **offline dataset** → sample reward function \(\eta\)
- evaluate \(\eta(s)\) on random encoder states
- encode \((s, \eta(s))\) pairs into latent \(z\)
- condition policy \(\pi(a \mid s, z)\)
- evaluate zero-shot on downstream objectives

This is the central method route implemented by the canonical runner.

### Figure 2: reward encoding
FRE encodes a reward function by evaluating it over a random set of dataset states, then passing the resulting state-reward pairs through a permutation-invariant encoder.

### Figure 3: zero-shot transfer examples
The paper’s Figure 3 shows qualitative zero-shot transfer examples in AntMaze.

**Binding addendum note:** the results implied by Figure 3 are out of scope for the current file-level contract. The repository still exposes the evaluation and artifact surfaces needed for that route, but smoke/default commands do not claim those results.

### Table 1: offline zero-shot RL comparisons
The repository preserves comparison semantics for the main paper baselines and metrics.

Supported named baselines / comparisons include:

- **FRE**
- **FB** / Forward-Backward
- **SF** / Successor Features
- **OPAL**
- **GCRL**
- **GC-BC**
- **GC-IQL**

### Figure 4: evaluation domains
The evaluation registry is aligned to the paper’s main domains:

- **AntMaze**
- **ExORL**
- **Kitchen**

### Table 2: method capability comparison
The capability comparison surface is preserved for:

- zero-shot capability
- reward-family coverage
- value-function constraints
- policy adaptation constraints

### Figure 5: scaling with reward-family diversity
**Binding addendum note:** Figure 5 is evaluated on **AntMaze**.  
The repository exposes a scaling-study interface that varies the set of reward priors used in training.

### Figure 6: domain-knowledge augmentation
The repository includes a domain-knowledge ablation surface that augments random reward families with specific prior distributions.

### Table 3: hyperparameters
The canonical configuration registry carries the paper’s hyperparameter surface and bounded smoke defaults.

### Table 4: subset/scaling results
The scaling-study route is exposed as a bounded, registry-driven experiment matrix.

### Figures 7–9: additional AntMaze examples
The artifact layer preserves the figure route and file naming for extended qualitative outputs.

## Reference grounding

The reproduction route adapts protocol intent from grounded references while keeping this repository standalone.

reference_grounding: paperbench_ref_001 url_benchmark/d4rl_benchmark.py  
reference_grounding: paperbench_ref_001 url_benchmark/agent/ddpg.py  
reference_grounding: paperbench_ref_001 url_benchmark/pretrain.py

The adapted protocol intent covers:

- offline dataset filtering / episode handling
- reward-prior sampling
- latent-conditioned policy training
- checkpoint-style artifact writing
- bounded smoke validation

## Repository layout