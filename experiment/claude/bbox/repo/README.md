# BBox-Adapter: Lightweight Adapting for Black-Box Large Language Models

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Reference grounding: paperbench_ref_003 truthfulqa/models.py**
> TruthfulQA MC evaluation protocol (MC_calcs: max/diff/scores-true/scores-false lprob columns) is
> adapted into `src/evaluation/metrics.py` and `src/data/truthfulqa.py`.

---

## Overview

**BBox-Adapter** is a lightweight black-box LLM adaptation framework that attaches a small
energy-based adapter model (~0.1 B – 0.3 B parameters, BERT-based) to any black-box LLM to
improve its performance on downstream tasks **without** access to model parameters or token
probabilities.

### Figure 1 — LLM Adaptation Taxonomy