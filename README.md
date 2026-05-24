> Experiment artifacts are stored in this repository's `experiment` branch. For anonymous review, please use: https://anonymous.4open.science/r/ReproAgent-4C1E/experiment/README.md

# ReproAgent

ReproAgent is an automated pipeline for PaperBench-style paper reproduction.
Given a paper case, it extracts implementation units, builds a faithful
reproduction plan, generates a runnable repository, and optionally repairs the
repository using validation feedback.

The current release focuses on the core reproduction pipeline and two ablation
entrypoints used in our experiments:

- **Full ReproAgent**: implementation-requirement channel on,
  reference-evidence channel on.
- **w/o Implementation Requirements**: removes the paper-derived
  implementation-requirement channel from generation and repair.
- **w/o Reference Evidence**: disables reference-evidence source acquisition
  and grounding.

## Overview

![ReproAgent architecture](fig/architecture.png)

![ReproAgent pipeline](fig/reproagent.png)

At a high level, ReproAgent is organized as four stages:

1. **Prepare** extracts paper chunks, implementation units, datasets,
   evaluation obligations, and candidate reference-evidence sources.
2. **Plan** converts the extracted units into work packages, contracts,
   architecture decisions, and file-level implementation tasks.
3. **Generate** materializes a complete reproduction repository from the plan.
4. **Repair** patches the generated repository using validation feedback while
   preserving the implementation contract.

The two main evidence mechanisms are:

- **Implementation-requirement channel**: a persistent paper-derived contract
  that keeps methods, datasets, metrics, algorithms, formulas, and acceptance
  signals visible from planning through repair.
- **Reference-evidence channel**: package-local content and structure evidence
  recovered from official or high-quality repositories cited by, or relevant
  to, the paper.

## Repository Layout

```text
reproagent/                 Core Python package and pipeline implementation.
ablation/
  no_implementation_requirement/
                             Runner for disabling implementation requirements.
  no_reference_evidence/        Runner for disabling reference evidence.
experiment_runners/         Convenience shell runners for the three variants.
fig/                        Architecture figures used by this README.
run_paperbench.py           Main CLI entrypoint.
requirements.txt            Minimal Python dependencies.
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

Fill in at least:

```bash
PAPERBENCH_REPRO_NODE_MODEL=gemini-3-flash-preview
PAPERBENCH_REPRO_NODE_API_KEY=...
PAPERBENCH_REPRO_NODE_BASE_URL=...
PAPERBENCH_REPRO_STRUCTURED_STAGE_MODEL=gemini-3-flash-preview
PAPERBENCH_REPRO_STRUCTURED_STAGE_API_KEY=...
PAPERBENCH_REPRO_STRUCTURED_STAGE_BASE_URL=...
```

Reference-evidence source search benefits from `PAPERBENCH_REPRO_GITHUB_TOKEN`,
but the pipeline can run without it.

## Data

The CLI expects PaperBench-style case folders containing the paper text and
metadata. By default it looks under:

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

## Notes

This first public bundle contains the reproduction pipeline and ablation
entrypoints. Benchmark scoring scripts, internal pipeline traces, raw
PaperBench data, and judge score artifacts are intentionally excluded from the
main branch. Generated reproduction repositories are available on the
`experiment` branch; for anonymous review, see:
https://anonymous.4open.science/r/ReproAgent-4C1E/experiment/README.md

PaperBench resources:

- Benchmark repository: https://github.com/openai/frontier-evals/tree/main/project/paperbench
- Dataset directory: https://github.com/openai/frontier-evals/tree/main/project/paperbench/data

## Experiment Results

Scores are percentages. `Gemini full` is the full ReproAgent pipeline.
`w/o Ref. Evidence` removes the reference-evidence channel, and `w/o Impl.
Req.` removes the implementation-requirement channel.

| Paper | Claude | Gemini full | w/o Ref. Evidence | w/o Impl. Req. | DeepCode | Basic | Iter | RUC AiSci |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Mean** | **73.7** | **39.7** | **21.6** | **25.6** | **73.5** | **19.3** | **20.6** | **30.5** |
| `adaptive-pruning` | 67.8 | 41.9 | 6.2 | 14.3 | 54.4 | 24.5 | 3.0 | 27.2 |
| `all-in-one` | 76.9 | 51.6 | 23.7 | 28.0 | 75.9 | 20.9 | 45.1 | 46.3 |
| `bam` | 61.8 | 48.5 | 20.7 | 30.8 | 74.8 | 48.5 | 45.0 | 56.6 |
| `bbox` | 86.4 | 15.6 | 14.6 | 14.2 | 64.4 | 15.4 | 8.3 | 33.8 |
| `bridging-data-gaps` | 66.0 | 50.6 | 13.9 | 38.4 | 58.1 | 12.6 | 12.4 | 23.1 |
| `fre` | 76.8 | 26.9 | 12.7 | 25.1 | 81.4 | 21.7 | 23.9 | 35.2 |
| `ftrl` | 63.9 | 40.1 | 11.7 | 13.4 | 59.8 | 5.9 | 4.2 | 10.1 |
| `lbcs` | 65.8 | 42.5 | 34.6 | 23.2 | 74.7 | 17.8 | 15.3 | 27.9 |
| `lca-on-the-line` | 64.9 | 41.8 | 33.8 | 16.1 | 74.9 | 13.0 | 18.3 | 30.2 |
| `mechanistic-understanding` | 75.7 | 48.1 | 25.9 | 4.6 | 92.5 | 14.9 | 21.9 | 29.9 |
| `pinn` | 81.5 | 61.3 | 52.2 | 53.3 | 91.0 | 26.6 | 30.8 | 49.9 |
| `rice` | 75.1 | 31.0 | 19.9 | 27.8 | 70.2 | 10.4 | 8.9 | 10.9 |
| `robust-clip` | 63.0 | 29.5 | 15.7 | 25.6 | 73.3 | 15.4 | 10.4 | 18.3 |
| `sample-specific-masks` | 82.0 | 53.8 | 28.0 | 21.3 | 67.1 | 25.4 | 33.3 | 36.8 |
| `sapg` | 86.5 | 29.0 | 21.5 | 24.2 | 73.8 | 11.4 | 12.7 | 19.9 |
| `sequential-neural-score-estimation` | 78.0 | 45.7 | 21.5 | 42.6 | 87.0 | 53.5 | 60.2 | 64.9 |
| `stay-on-topic-with-classifier-free-guidance` | 58.5 | 25.3 | 21.2 | 22.2 | 70.5 | 8.4 | 13.7 | 20.1 |
| `stochastic-interpolants` | 80.8 | 39.1 | 21.5 | 29.9 | 81.5 | 17.0 | 17.4 | 18.8 |
| `test-time-model-adaptation` | 75.6 | 54.9 | 23.8 | 47.1 | 64.9 | 15.3 | 18.1 | 32.5 |
| `what-will-my-model-forget` | 87.6 | 16.8 | 8.4 | 10.7 | 80.8 | 6.6 | 9.0 | 17.9 |
