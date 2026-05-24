# ReproAgent

ReproAgent is an automated pipeline for PaperBench-style paper reproduction.
Given a paper case, it extracts implementation units, builds a faithful
reproduction plan, generates a runnable repository, and optionally repairs the
repository using validation feedback.

The current release focuses on the core reproduction pipeline and two ablation
entrypoints used in our experiments:

- **Full ReproAgent**: semantic anchor on, reference-repository grounding on.
- **w/o Semantic Anchor**: removes paper-derived semantic anchors from
  generation and repair.
- **w/o Reference Repo**: disables reference-repository cloning and grounding.

## Overview

![ReproAgent architecture](fig/architecture.png)

![ReproAgent pipeline](fig/reproagent.png)

At a high level, ReproAgent is organized as four stages:

1. **Prepare** extracts paper chunks, implementation units, datasets,
   evaluation obligations, and candidate reference repositories.
2. **Plan** converts the extracted units into work packages, contracts,
   architecture decisions, and file-level implementation tasks.
3. **Generate** materializes a complete reproduction repository from the plan.
4. **Repair** patches the generated repository using validation feedback while
   preserving the implementation contract.

The two main evidence mechanisms are:

- **Semantic anchor**: a persistent paper-derived contract that keeps methods,
  datasets, metrics, algorithms, formulas, and acceptance signals visible from
  planning through repair.
- **Reference repo grounding**: discovery and use of official or high-quality
  reference repositories cited by, or relevant to, the paper.

## Repository Layout

```text
reproagent/                 Core Python package and pipeline implementation.
ablation/
  no_semantic_anchor/       Runner for disabling semantic anchors.
  no_reference_repo/        Runner for disabling reference repository grounding.
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

Reference-repository search benefits from `PAPERBENCH_REPRO_GITHUB_TOKEN`, but
the pipeline can run without it.

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

Run without semantic anchors:

```bash
python ablation/no_semantic_anchor/run_ablation.py rice --stage repair
```

Equivalent environment switch:

```bash
export PAPERBENCH_REPRO_DISABLE_SEMANTIC_ANCHOR=1
python run_paperbench.py rice --stage repair
```

Run without reference-repository grounding:

```bash
python ablation/no_reference_repo/run_ablation.py rice --stage repair
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
bash experiment_runners/run_no_semantic_anchor.sh
bash experiment_runners/run_no_reference_repo.sh
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
entrypoints. Benchmark scoring scripts, internal experiment outputs, raw
PaperBench data, and generated repositories are intentionally excluded.
