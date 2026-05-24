# Mechanistic DPO Toxicity Reproduction

Standalone PaperBench reproduction repository for **“A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and Toxicity.”**

This repository implements the paper-derived route for studying how DPO changes toxicity behavior in GPT-2-like language models:

1. prepare toxicity and prompt datasets,
2. train or load a binary toxicity probe,
3. extract toxic vectors from probe weights, MLP value vectors, and SVD directions,
4. project vectors into vocabulary space for Table 1-style inspection,
5. train or load a DPO-aligned model,
6. evaluate toxicity, perplexity, F1, activation shifts, logit-lens probabilities, interventions, and un-aligning variants,
7. write reproducible artifacts under `results/`.

The default command is a bounded smoke route that exercises the real configuration, registry, data, method, evaluation, training-loop, and artifact-writer surfaces without performing expensive full training.

> **Safety note.** The original paper’s Table 1 and related examples contain highly offensive tokens and continuations. This repository preserves the artifact mapping and computation interfaces, but the default smoke route uses bounded toy/fixture text and does not print offensive paper examples.

## Canonical commands