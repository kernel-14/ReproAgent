# Open Source Submission Manifest

This repository is a code-first reproduction for "Batch and match: black-box
variational inference with a score-based divergence".

Canonical entrypoint:

```bash
python scripts/run_experiments.py --mode runtime_smoke
```

Primary implementation files:

- `scripts/run_experiments.py`: canonical executable route, method selection,
  Gaussian KL metrics, table/figure/metrics writers, and artifact manifest.
- `src/algorithms/bam.py`: full-covariance BaM API with batch step, match step,
  training loop, and artifact writer.
- `src/algorithms/advi.py`: full-covariance ADVI, ADVI-score, and ADVI-Fisher
  baseline surfaces.
- `src/algorithms/gsm.py`: Gaussian score-matching baseline surface.
- `bam/variational.py`, `bam/optimizer.py`, `bam/score_divergence.py`: importable
  BaM optimizer and score-divergence helpers.
- `evaluation/metrics.py`, `evaluation/report.py`: KL/loss/MSE aggregation and
  artifact writers.
- `tests/test_smoke.py`, `tests/test_contracts.py`: executable contract tests.

Required smoke artifacts are written under `results/` and are enumerated by
`results/artifact_manifest.json`.
