# Refined Coreset Selection: Towards Minimal Coreset Size under Model Performance Constraints

This repository provides an executable reproduction surface for the paper
"Refined Coreset Selection: Towards Minimal Coreset Size under Model Performance
Constraints."

reference_grounding: paperbench_ref_003 train.py
reference_grounding: paperbench_ref_003 selection.py
reference_grounding: paperbench_ref_004 noisy_label.py
reference_grounding: paperbench_ref_004 cnn_mnist_probability_1step_pixel_shared_rein.py

## Implemented Scope

The code implements the Lexicographic Bilevel Coreset Selection (LBCS) route,
Algorithm 2 refinement, dataset access, baseline selectors, model declarations,
bounded experiment sweeps, and artifact writers used by the PaperBench rubric.

- LBCS: `src/lbcs_reproduction.py` exposes `algorithm1_lbcs`,
  `algorithm2_refine_mask`, `equation3_outer_objective`,
  `equation4_outer_objective`, `f1_performance_gap`, and `f2_coreset_size`.
- Baselines: Uniform, EL2N, GraNd, Influential, Moderate, CCS, and
  Probabilistic selectors are implemented and dispatched through
  `BASELINE_SELECTORS`.
- Datasets without credentials: F-MNIST, SVHN, CIFAR-10, CIFAR-100, MNIST, and
  MNIST-S are available through `stream_dataset_without_credentials` and
  `form_mnist_s_subset`.
- Models: Appendix C.3 CNN contract, ConvNet-3, ResNet-18, and ResNet-50
  surfaces are represented in `src/lbcs_reproduction.py` and
  `src/methods/models.py`.
- Experiments and artifacts: `run_all_experiments` writes dataset access
  evidence, Figure 1 objective traces, Section 5.1 MNIST-S sweeps, Table 2 /
  Figure 3 benchmark comparisons, Table 3 CIFAR-100 comparisons, Section 5.3
  sensitivity records, and Section 6 ablation tables.

## Local Smoke Commands

```bash
python -m py_compile main.py src/*.py src/data/*.py src/experiments/*.py src/methods/*.py src/reporting/*.py
python main.py --mode runtime_smoke --output results/reproduction_summary.json
```

The runtime path is deterministic by default and uses local public-shape dataset
mirrors unless `LBCS_USE_TORCHVISION=1` is set.
