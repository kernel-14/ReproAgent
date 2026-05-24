# What Will My Model Forget? — PaperBench Reproduction

This repository is a standalone PaperBench reproduction scaffold for **“What Will My Model Forget? Forecasting Forgotten Examples in Language Model Refinement.”** It implements the paper-derived experiment routes for forecasting examples that will be forgotten during language-model refinement, replay-based mitigation, and appendix/boundary checks.

The repository is code-generation safe: default commands run bounded deterministic smoke protocols that exercise the real data, method, metric, replay, evaluation, and artifact-writing surfaces without downloading external datasets or training large PTLMs. Full experiments use the same entrypoints and configuration registries, but require explicit full-mode selection and real data/model assets.

reference_grounding: paper_contract_experiment_artifact_protocol paper.md  
reference_grounding: paper_evidence_matrix paper.md  
reference_grounding: paper_named_experiment_protocols paper.md  
reference_grounding: addendum_dpt_protocol addendum.md  

## Canonical entrypoints

There is one primary repository route for selecting forecasting, refinement, and appendix protocols:

The full paper protocol is exposed directly at `wwmf.paper_protocol`.  The
top-level callable `wwmf.paper_protocol.run_all_paper_protocols(data_root,
output_dir)` runs the rubric-facing routes for BART0 Large, FLAN-T5 Large, and
FLAN-T5 3B.  It creates D_PT from the exact 36 P3 train tasks with a seeded
random 100-example draw per task, builds hat_D_PT from named base-model Exact
Match predictions, creates D_R^train/D_R^test from ReCross P3 test or original
Hendrycks MMLU validation errors, computes z_ij and z_ij^test over every
D_R x hat_D_PT pair, trains Algorithm 3 encoder h, emits Algorithm 2 and
Algorithm 4 hat_z_ij^test predictions, and exposes forecasted/ground-truth
replay with the paper batch sizes and replay intervals.

Key importable routes:

```python
from wwmf.paper_protocol import (
    run_all_paper_protocols,
    run_table1_protocols,
    create_dpt_hat_dpt_with_base_model_predictions,
    create_bart0_large_dr_from_recross_p3_test,
    create_flan_t5_dr_from_mmlu_validation,
    train_encoding_function_h_algorithm3,
    algorithm2_trainable_logit_predict_all_pairs,
    algorithm4_representation_predict_all_pairs,
    compute_ground_truth_z_ij_test_for_refined_models,
    compute_average_exact_match_drop_ratio_across_refined_models,
    forecast_guided_replay_refinement,
    ground_truth_replay_refinement,
    sequential_error_fixing_section_52,
)
```
