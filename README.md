# ReproAgent：Contract-Guided Paper-to-Code Reproduction

ReproAgent is a contract-guided agent pipeline for paper-to-code reproduction. Given a research paper, it generates a runnable repository while preserving the paper-specific methods, protocols, metrics, and artifacts that matter for faithful reproduction.

The core idea is a persistent implementation contract with two channels:

- **Implementation-requirement channel**: extracts explicit paper obligations, such as algorithms, losses, metrics, artifacts, dataset rules, and evaluation protocols, and keeps them visible during planning, generation, and repair.
- **Reference-evidence channel**: retrieves content and structure evidence from related repositories so implicit implementation details, file organization, entry points, and artifact conventions are grounded rather than guessed.

ReproAgent instantiates this contract in a four-stage **Prepare--Plan--Generate--Repair** pipeline. Prepare extracts paper-facing obligations and reference evidence; Plan binds them into work packages and file-level contracts; Generate writes the repository file by file; Repair audits the result against the same contract and runtime feedback.

## Overview

![ReproAgent architecture](fig/architecture.png)

![ReproAgent pipeline](fig/reproagent.png)

## Result Snapshot

PaperBench Code-Dev scores are percentages averaged over 20 ICML 2024 paper-reproduction tasks. Higher is better.

| Setting | Score |
| --- | ---: |
| **ReproAgent (ours, Claude-Sonnet-4.5)** | **73.7** |
| DeepCode | 73.5 |
| Deep-Reproducer | 63.2 |
| AutoReproduce | 49.6 |
| AutoP2C | 49.2 |
| PaperCoder | 45.1 |

**Same-backbone scaffold comparison.** All rows below use Gemini-3-Flash.

| System | Score |
| --- | ---: |
| **ReproAgent (ours)** | **39.7** |
| AiScientist | 30.5 |
| IterAgent | 20.6 |
| BasicAgent | 19.3 |

**Channel ablations.** Both ablations use Gemini-3-Flash and the same repair budget as the full Gemini run.

| Setting | Mean | Median | Drop from full |
| --- | ---: | ---: | ---: |
| Full contract | 39.7 | 41.8 | - |
| w/o reference evidence | 21.6 | 21.4 | -18.1 |
| w/o implementation requirements | 25.6 | 24.6 | -14.1 |

The full contract beats both ablations on all 20 targets, supporting the mechanism that the requirement channel preserves explicit paper obligations while the evidence channel grounds implicit repository knowledge.

## Per-Paper PaperBench Results

This table summarizes the paper-level scores used in the main comparison and ablation study. `Ref ablation` removes reference evidence; `Req ablation` removes implementation requirements.

| Paper | Ours Claude | Ours Gemini | Ref ablation | Req ablation | DeepCode | Basic Gemini | Iter Gemini | AiScientist |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Adaptive Pruning | 67.8 | 41.9 | 6.2 | 14.3 | 54.4 | 24.5 | 3.0 | 27.2 |
| All-in-One | 76.9 | 51.6 | 23.7 | 28.0 | 75.9 | 20.9 | 45.1 | 46.3 |
| BAM | 61.8 | 48.5 | 20.7 | 30.8 | 74.8 | 48.5 | 45.0 | 56.6 |
| BBOX | 86.4 | 15.6 | 14.6 | 14.2 | 64.4 | 15.4 | 8.3 | 33.8 |
| Bridging Data Gaps | 66.0 | 50.6 | 13.9 | 38.4 | 58.1 | 12.6 | 12.4 | 23.1 |
| FRE | 76.8 | 26.9 | 12.7 | 25.1 | 81.4 | 21.7 | 23.9 | 35.2 |
| FTRL | 63.9 | 40.1 | 11.7 | 13.4 | 59.8 | 5.9 | 4.2 | 10.1 |
| LBCS | 65.8 | 42.5 | 34.6 | 23.2 | 74.7 | 17.8 | 15.3 | 27.9 |
| LCA-on-the-Line | 64.9 | 41.8 | 33.8 | 16.1 | 74.9 | 13.0 | 18.3 | 30.2 |
| Mechanistic Understanding | 75.7 | 48.1 | 25.9 | 4.6 | 92.5 | 14.9 | 21.9 | 29.9 |
| PINN | 81.5 | 61.3 | 52.2 | 53.3 | 91.0 | 26.6 | 30.8 | 49.9 |
| RICE | 75.1 | 31.0 | 19.9 | 27.8 | 70.2 | 10.4 | 8.9 | 10.9 |
| Robust CLIP | 63.0 | 29.5 | 15.7 | 25.6 | 73.3 | 15.4 | 10.4 | 18.3 |
| Sample-specific Masks | 82.0 | 53.8 | 28.0 | 21.3 | 67.1 | 25.4 | 33.3 | 36.8 |
| SAPG | 86.5 | 29.0 | 21.5 | 24.2 | 73.8 | 11.4 | 12.7 | 19.9 |
| Sequential Neural Score Estimation | 78.0 | 45.7 | 21.5 | 42.6 | 87.0 | 53.5 | 60.2 | 64.9 |
| Stay on Topic with Classifier-Free Guidance | 58.5 | 25.3 | 21.2 | 22.2 | 70.5 | 8.4 | 13.7 | 20.1 |
| Stochastic Interpolants | 80.8 | 39.1 | 21.5 | 29.9 | 81.5 | 17.0 | 17.4 | 18.8 |
| Test-Time Model Adaptation | 75.6 | 54.9 | 23.8 | 47.1 | 64.9 | 15.3 | 18.1 | 32.5 |
| What Will My Model Forget | 87.6 | 16.8 | 8.4 | 10.7 | 80.8 | 6.6 | 9.0 | 17.9 |
| **Mean** | **73.7** | **39.7** | **21.6** | **25.6** | **73.5** | **19.3** | **20.6** | **30.5** |

## Additional Reported Baselines

These rows are useful context but are not the primary same-backbone comparison because they differ in scaffold, backbone, runtime budget, or reporting protocol.

| Method | Backbone / setting | Reported score |
| --- | --- | ---: |
| BasicAgent | Gemini-2.0-Flash | 5.0 |
| BasicAgent | o3-mini | 5.1 |
| BasicAgent | GPT-4o | 7.7 |
| BasicAgent | o1 | 19.5 |
| BasicAgent | Claude-3.5-Sonnet | 35.4 |
| IterativeAgent | o3-mini | 16.4 |
| IterativeAgent | Claude-3.5-Sonnet | 27.5 |
| IterativeAgent | o1 | 43.3 |
| RePro | o3-mini, PRroot@5 | 62.6 |

## Repository Layout

```text
reproagent/                 Core Python package and pipeline implementation.
ablation/
  no_implementation_requirement/
                             Runner for disabling implementation requirements.
  no_reference_evidence/    Runner for disabling reference evidence.
experiment_runners/         Batch runners for the full and ablated variants.
fig/                        README figures.
run_paperbench.py           Main CLI entrypoint.
requirements.txt            Python dependencies.
.env.example                Environment-variable template.
```

## Setup

Create a Python environment and install dependencies:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Create a local `.env` from the template:

```bash
cp .env.example .env
```

Fill in the model endpoints used by your run, for example:

```bash
PAPERBENCH_REPRO_NODE_MODEL=gemini-3-flash-preview
PAPERBENCH_REPRO_NODE_API_KEY=...
PAPERBENCH_REPRO_NODE_BASE_URL=...
PAPERBENCH_REPRO_STRUCTURED_STAGE_MODEL=gemini-3-flash-preview
PAPERBENCH_REPRO_STRUCTURED_STAGE_API_KEY=...
PAPERBENCH_REPRO_STRUCTURED_STAGE_BASE_URL=...
```

Reference-evidence search benefits from `PAPERBENCH_REPRO_GITHUB_TOKEN`, but the pipeline can run without it.

## Data

The CLI expects PaperBench-style case folders containing the paper text and metadata. By default it looks under:

```text
paperbench_data/<paper-id>/
```

You can also pass an explicit data root:

```bash
python run_paperbench.py rice --data-root /path/to/paperbench_data --stage generate
```

## Run ReproAgent

Run the full pipeline for one paper:

```bash
python run_paperbench.py rice --stage repair
```

Run only through generation:

```bash
python run_paperbench.py rice --stage generate
```

Use a stable run id:

```bash
python run_paperbench.py rice --run-id main_rice --stage repair
```

Outputs are written under:

```text
output/reproagent/<run-id>/
```

## Resume

Resume generation after an interrupted local-file-generation stage:

```bash
python run_paperbench.py rice \
  --resume-from-run-id main_rice \
  --resume-in-place \
  --resume-start-stage local_file_generation \
  --stage generate
```

Resume repair:

```bash
python run_paperbench.py rice \
  --resume-from-run-id main_rice \
  --resume-in-place \
  --resume-start-stage repair_validation \
  --stage repair
```

## Ablations

Run without implementation requirements:

```bash
python ablation/no_implementation_requirement/run_ablation.py rice --stage repair
```

Equivalent environment switch:

```bash
export PAPERBENCH_REPRO_DISABLE_IMPLEMENTATION_REQUIREMENTS=1
python run_paperbench.py rice --stage repair
```

Run without reference evidence:

```bash
python ablation/no_reference_evidence/run_ablation.py rice --stage repair
```

Equivalent CLI switch:

```bash
python run_paperbench.py rice --no-clone-references --stage repair
```

## Batch Runners

Run all configured PaperBench cases for the main variant:

```bash
bash experiment_runners/run_main_experiment.sh
```

Run all cases for the two ablations:

```bash
bash experiment_runners/run_no_implementation_requirement.sh
bash experiment_runners/run_no_reference_evidence.sh
```

Pass paper ids to run a subset:

```bash
bash experiment_runners/run_main_experiment.sh rice pinn
```

By default the runners execute `--stage repair`. To stop after generation:

```bash
REPROAGENT_STAGE=generate bash experiment_runners/run_main_experiment.sh rice
```

## PaperBench Resources

- Benchmark repository: https://github.com/openai/frontier-evals/tree/main/project/paperbench
- Dataset directory: https://github.com/openai/frontier-evals/tree/main/project/paperbench/data
