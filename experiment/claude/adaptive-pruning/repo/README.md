# APT Reproduction

Faithful, judgeable reproduction repository for **APT: Adaptive Pruning and
Tuning Pretrained Language Models for Efficient Training and Inference**.

This repository implements the paper-owned route through `main.py` and
`src/apt/*`: data/task registries, model factories, APT adapters, adaptive
pruning, adaptive tuning, baseline selectors, training/evaluation loops, metric
formulas, and artifact writers.  The default command is bounded so reviewers can
validate wiring in a small environment; full model and dataset execution is kept
behind explicit flags.

The blacklisted upstream repository `https://github.com/ROIM1998/APT` is not
used.

reference_grounding: paperbench_ref_001 datasheet.md
reference_grounding: paperbench_ref_001 model_card.md
reference_grounding: paperbench_ref_001 prompt.txt
reference_grounding: paperbench_ref_003 lm-evaluation-harness/README.md

## Quick Start

```bash
python main.py
```

Equivalent bounded smoke route:

```bash
python main.py --mode runtime_smoke --bounded true
```

Full route, requiring local checkpoints/datasets/backends:

```bash
python main.py --mode full --bounded false \
  --method APT \
  --model roberta-base \
  --dataset sst2 \
  --reference-method FT \
  --target-accuracy 0.97 \
  --batch-size 32
```

The default bounded route calls the same implementation surfaces as full mode
with small fixtures and writes current-run artifacts under `results/`.  It may
mark values as `bounded_proxy` or `unavailable`; it must not claim paper
benchmark scores unless the measured route produced them.

If `PAPERBENCH_REPRO_ARTIFACT_DIR` is set, auxiliary validation artifacts are
also written there.

## Canonical Route

`main.py` is the canonical entrypoint.  It builds the run configuration, prepares
datasets, creates the selected model/method, runs the APT or baseline training
route, evaluates predictions, and writes artifacts through `src/apt/artifacts.py`.

The route includes these concrete call surfaces:

- `create_model(model_name, method, adapter_config, bounded)` and
  `build_model(...)` in `src/apt/models.py`
- `inject_apt_adapters(model, target_modules, config)` and APT adapter logic in
  `src/apt/adapters.py`
- `run_baseline(method, model, dataset, config)` in `src/apt/baselines.py`
- `run_training(...)` in `src/apt/training.py`
- `evaluate_predictions(config)` and `run_evaluation(...)` in
  `src/apt/evaluation.py`
- `compute_task_metrics(...)`, `compute_generation_metrics(...)`,
  `compute_rouge(...)`, `compute_efficiency_metrics(...)`, and
  `compute_relative_accuracy(...)` in `src/apt/metrics.py`
- `write_evaluation_result_artifact`, `write_result_table_artifact`,
  `write_metric_formula_artifact`, `write_artifact_manifest_artifact`,
  `write_run_config_artifact`, `write_metrics_artifact`,
  `write_dataset_registry_artifact`, and `write_model_registry_artifact` through
  `src/apt/artifacts.py`
- `run_figure_1_route`, `write_figure_1_artifact`, `run_figure_2_route`, and
  `write_figure_2_artifact` through the reporting/artifact writer route

## Paper Method Coverage

APT is implemented as a method selector, not as plain LoRA.  The adapter route
keeps the paper variables visible and executable:

- APT adapter: `H_apt(X) = m_o * (W + s * W_B W_A) X * m_i`
- binary pruning masks: `m_i` and `m_o`, where `0` prunes and `1` retains
- dynamic rank: `r_apt`
- adaptive pruning `A_P`: outlier-aware salience, kurtosis-aware block scoring,
  early-training fast search, and mask updates
- salience EMA: `S_bar_t = 0.85 * S_bar_t_minus_1 + 0.15 * S_hat`
- pruning schedule: `mu = min(1, (global_step - pruning_start_step) /
  (pruning_end_step - pruning_start_step))`, with `mu = 0` before pruning starts
- adaptive tuning `A_T`: dynamic tuning-rank allocation and A_T metadata used by
  trainable-parameter, memory, and cost metrics
- self-distillation: classification uses
  `L_distill = L_pred + 0.9 * L_layer`; SQuAD/generation routes use the
  configured task-specific layer weighting

The bounded route executes these formulas on local fixtures.  Full mode keeps
lazy factories for `torch`, `transformers`, `datasets`, and task backends so
large models can be loaded only when requested.

## Datasets And Models

Dataset registry output: `results/dataset_registry.json`.

Paper-derived datasets and tasks:

- GLUE SST2 and MNLI: dev accuracy and relative accuracy inputs
- SQuAD v2.0: dev F1
- CNN/DailyMail: ROUGE and generation metrics
- TruthfulQA: generation evaluation route and result-table visibility
- Alpaca-style instruction/generation fixtures for LLaMA bounded and full routes

Model registry output: `results/model_registry.json`.

Supported model flags:

```bash
--model bert-base
--model roberta-base
--model t5-small
--model llama
```

The bounded route uses lightweight model proxies through the same factory
interface.  Full mode requires locally available checkpoints or Hugging Face
model identifiers.

## Methods, Baselines, And Defaults

Supported method flags:

```bash
--method FT
--method LoRA
--method MaskTuning
--method CoFi
--method APT
```

The baseline registry is written to `results/baseline_registry.json`.
Configuration defaults live in `configs/default.yaml` and `src/apt/config.py`,
including the addendum-required defaults for:

- batch sizes: `batch_size_32` and `batch_size_128`
- fixed CLI default: `--batch-size 32`
- half precision attack protocol: `--half-precision-attack`
- 10-shot generation setting
- LoRA/adapter rank and scaling defaults
- APT target sparsity, pruning start/end steps, early training steps, and tuning
  budget
- Mask Tuning and CoFi checkpoint visibility at `checkpoints/mask_tuning` and
  `checkpoints/cofi`

The half precision protocol is explicit:

```bash
python main.py --mode runtime_smoke --method APT --batch-size 32 --half-precision-attack
```

## Evaluation And Metrics

The evaluation entry reads the current run's:

- `results/run_config.json`
- `results/dataset_registry.json`
- `results/model_registry.json`
- `results/pruning_trace.json`
- `results/tuning_trace.json`
- `results/training_trace.json`
- A_T metadata from the tuning/model traces

`results/evaluation_result.json` is written by the evaluation route.  Each
metric entry carries one of:

- `measured`: produced by an executed metric function on current predictions or
  traces
- `bounded_proxy`: produced by the same route on bounded fixtures
- `unavailable`: backend, dataset, or checkpoint missing; no benchmark claim

`results/metric_formula.json` records formulas and consumed fields for:

- trainable parameter count
- `training_cost`
- `inference_cost`
- `memory_usage` and `gpu_memory`
- relative training peak memory
- relative training speed
- relative inference memory
- relative inference speed
- relative accuracy, with SST2/MNLI inputs preserved in
  `results/sst2_mnli_relative_accuracy_inputs.json`
- TTA
- dev accuracy
- dev F1
- ROUGE

`results/metrics.json` stores the computed metric values for the current route.

## Result And Artifact Outputs

The core output paths are:

- `results/run_config.json`
- `results/dataset_registry.json`
- `results/model_registry.json`
- `results/baseline_registry.json`
- `results/environment_registry.json`
- `results/environment_readiness.json`
- `results/experiment_registry.json`
- `results/method_registry.json`
- `results/evidence_contract_matrix.json`
- `results/evaluation_result.json`
- `results/metrics.json`
- `results/metric_formula.json`
- `results/result_table.json`
- `results/artifact_manifest.json`
- `results/sst2_mnli_relative_accuracy_inputs.json`
- `results/data_manifest.json`
- `results/tables/table_1.csv`
- `results/tables/summary.csv`

`results/result_table.json` aggregates task, model, method, baseline, metric,
artifact provenance, table/figure source, and relative metric inputs.  TruthfulQA
must appear there as a paper-derived generation task when the route is run.

`results/artifact_manifest.json` records the output files above plus upstream
trace/checkpoint dependencies, including:

- `results/pruning_trace.json`
- `results/tuning_trace.json`
- `results/loss_trace.json`
- `results/training_trace.json`
- `checkpoints/mask_tuning`
- `checkpoints/cofi`

## Paper Table And Figure Obligations

The reporting route registers or writes the paper-visible artifacts below.  In
bounded mode the manifest may record readiness/full-mode requirements; benchmark
tables and figures are written only from computed route outputs.

- Figure 1: APT training and inference efficiency mechanism; output mapping from
  adaptive pruning and tuning to training/inference cost artifacts
- Figure 2: APT adapter, low-cost adaptive pruning `A_P`, adaptive tuning `A_T`,
  salience masks, and distillation route
- Figure 3: task performance versus relative inference efficiency
- Figure 4: performance-efficiency tradeoff normalized to LoRA tuning without
  pruning
- Figure 5: analysis over initial sparsity, target sparsity, and adaptive tuning
  schedules
- Figure 5a: A_T schedule subfigure obligation
- Table 1: efficiency comparison of PEFT, pruning, and APT
- Table 2: RoBERTa and T5 pruning under 60 percent sparsity
- Table 3: LLaMA 2 7B 30 percent sparsity on Alpaca/Open LLM tasks
- Table 4: RoBERTa ablations for `A_P`, `A_T`, and distillation
- Table 5: LLaMA 2 7B ablations under 30 and 50 percent sparsity
- Table 6: APT hyperparameters
- Table 7: PEFT plus unstructured pruning baselines
- Table 8: RoBERTa detailed comparison to LoRA+Distill
- Table 9: LLaMA2 7B and 13B 30 percent sparsity
- Table 10: distillation strategy ablations
- Table 11: raw RoBERTa/T5 efficiency metrics
- Table 12: raw LLaMA2 efficiency metrics

## Validation

Lightweight import and contract checks:

```bash
python -m pytest tests/test_contracts.py tests/test_eval_reporting.py
```

Bounded runtime validation:

```bash
python main.py --mode runtime_smoke
```

Docker/runtime validation route:

```bash
python main.py --mode docker_validate
```

The validation hook checks that benchmark-visible artifacts are present,
parseable, and provenance-linked after the route writes them.  It does not
reinterpret readiness manifests as benchmark measurements.
