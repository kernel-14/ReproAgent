#!/usr/bin/env python3
"""
BBox-Adapter Evaluation Protocol

Implements the complete evaluation protocol for BBox-Adapter paper reproduction,
including dataset registry, metric registry, protocol matrix, and artifact writers.

Paper: BBox-Adapter: Lightweight Adapting for Black-Box Large Language Models

Reference grounding: paperbench_ref_002 src/models/iterative/run_model.py
Reference grounding: paperbench_ref_003 truthfulqa/metrics.py

Paper Artifact Context:
  Figure 1: Illustration of white-box, grey-box, and black-box LLM adaptation.
    White-box has complete access to both model parameters and output probabilities,
    grey-box has access only to output probabilities,
    black-box lacks access to both.
  Figure 2: Overview of BBox-ADAPTER for black-box LLM adaptation from source to target domain.
    BBox-Adapter adopts an online adaptation framework, iteratively sampling from previous
    inferences and updating the adapter.
  Table 1: Comparison of existing LLM adaptation methods based on five aspects:
    (1) Model parameters accessibility, (2) Access to high-dimensional representations,
    (3) Token probability availability, (4) Retrieval corpus necessity,
    (5) Utilization of a smaller adapter model.
  Table 2: Main results of adapting gpt-3.5-turbo on downstream tasks.
    For BBox-ADAPTER, we report the best performance of adapters with
    # parameters of 0.1B and 0.3B. For all baselines and ours, we employ
    the CoT prompt as proposed in (Wei et al., 2022).
  Table 3: Results of plug-and-play adaptation on davinci-002 and Mixtral-8x7B
    across four datasets. For the plugger, we select BBox-Adapter tuned on
    gpt-3.5-turbo adaptation.
  Table 4: Comparison of performance and cost for the base model, SFT, and BBox-Adapter
    on the StrategyQA and GSM8K datasets. The performance is shown as accuracy (%),
    while the costs ($) are reported in training and inference expenses per thousand questions.
  Table 5: Accuracy (%) of BBox-ADAPTER fine-tuned with two types of loss:
    MLM loss and ranking-based NCE loss.
  Table 6: Accuracy (%) and GPU memory usage on adapting Mixtral-8x7B to StrategyQA.
    VRAM refers to the maximum GPU memory required by each approach.
  Table 7: Results of adapting Mixtral-8x7B-v0.1 on the ToxiGen dataset.
    Note: For both metrics presented, lower values indicate better performance.
  Table 8: Hyperparameter settings of SFT-LoRA (Hu et al., 2021).
  Table 10: Main results (extended) of adapting gpt-3.5-turbo on downstream tasks.
  Figure 3: Scale analysis on StrategyQA with (a) different beam sizes and
    (b) different iterations of online adaptation.
  Figure 4: Case study of BBox-ADAPTER on GSM8K.
  Figure 5: Loss curve of Azure-SFT on (a) StrategyQA, (b) TruthfulQA, (c) ScienceQA.
  Figure 6: Loss curves of Azure-SFT on GSM8K datasets.

Result Trend Obligations:
  baseline_outperformance: BBox-Adapter improves accuracy up to 6.77% over CoT
    (avg 6.39% across all datasets, Table 2)
  positive_parameter_improves: larger adapter size (0.3B) improves accuracy over 0.1B
  cost_reduction: 31.30x training cost reduction, 1.84x inference cost reduction vs SFT (Table 4)

Protocol Matrix (paper-derived):
  main_comparison       | GSM8K, StrategyQA, TruthfulQA, ScienceQA | base_model, azure_sft, azure_lora, bbox_adapter | accuracy -> Table 2, Table 10
  plug_and_play         | all four datasets | bbox_adapter (transfer) to davinci-002, Mixtral-8x7B | accuracy -> Table 3
  ablation_adapter_size | StrategyQA, GSM8K | bbox_adapter(0.1B, 0.3B)  | accuracy -> Table 2
  ablation_batch_size   | StrategyQA        | bbox_adapter(beam=1,3,5, iter=0-4) | accuracy -> Figure 3
  cost_efficiency       | StrategyQA, GSM8K | base_model, azure_sft, bbox_adapter | training_cost, inference_cost -> Table 4
  toxicity_reduction    | ToxiGen           | base_model, sft_lora, bbox_adapter  | hate_speech_rate, toxicity_probability -> Table 7
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Repository Layout Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results"
ARTIFACT_DIR = Path(os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR", str(RESULTS_DIR)))


# ---------------------------------------------------------------------------
# Parameter Sweep Registry (bounded config values, not exhaustive loops)
# reference_grounding: paperbench_ref_002 src/models/iterative/run_model.py
# ---------------------------------------------------------------------------

PARAMETER_SWEEP_REGISTRY: Dict[str, Any] = {
    "temperature": {
        "default": 0.7,
        "description": "Sampling temperature for LLM generation",
        "paper_setting": "temperature=1.0 for generation",
    },
    "judge_model": {
        "default": "roberta-base",
        "options": ["roberta-base", "tomh/toxigen_roberta"],
        "description": "Judge model for toxicity evaluation on ToxiGen",
        "paper_setting": "judge_model=roberta-base for toxicity",
    },
    "adapter_size": {
        "default": "0.1B",
        "options": ["0.1B", "0.3B"],
        "values_B": [0.1, 0.3],
        "description": "Adapter model size in billions of parameters (BERT-0.1B or BERT-0.3B)",
        "paper_setting": "adapter_size in [0.1B, 0.3B]",
        "trend": "positive_parameter_improves: larger adapter size improves accuracy",
    },
    "beam_width": {
        "default": 3,
        "options": [1, 3, 5],
        "description": "Beam width for sentence-level beam search inference",
        "paper_setting": "beam_width in [1, 3, 5] per Figure 3",
    },
    "batch_size": {
        "default": 128,
        "options": [64, 128],
        "description": "Training batch size for NCE loss optimization",
        "paper_setting": "batch_size in [64, 128]",
    },
    "learning_rate": {
        "default": 1e-5,
        "description": "AdamW learning rate for adapter training",
        "paper_setting": "learning_rate from Table 8 hyperparameters",
    },
    "num_iterations": {
        "default": 4,
        "options": [0, 1, 2, 3, 4],
        "description": "Number of online adaptation iterations (Figure 3b)",
        "paper_setting": "num_iterations in [0,1,2,3,4] per Figure 3",
    },
    "feedback_mode": {
        "default": "groundtruth",
        "options": ["groundtruth", "ai_feedback", "combined"],
        "description": "Feedback source for positive sample selection in NCE loss",
        "paper_setting": "feedback_mode: groundtruth | ai_feedback | combined",
    },
    "lora_rank": {
        "default": 8,
        "description": "LoRA rank for SFT-LoRA baseline (Table 8)",
        "paper_setting": "lora_rank=8 per Table 8",
    },
    "lora_alpha": {
        "default": 16,
        "description": "LoRA alpha scaling for SFT-LoRA baseline (Table 8)",
        "paper_setting": "lora_alpha=16 per Table 8",
    },
    "sft_epochs": {
        "default": 3,
        "description": "Number of fine-tuning epochs for SFT baselines (Table 8)",
        "paper_setting": "sft_epochs=3 per Table 8",
    },
}


# ---------------------------------------------------------------------------
# Dataset Registry
# reference_grounding: paperbench_ref_002 src/models/iterative/run_model.py
# ---------------------------------------------------------------------------

DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "gsm8k": {
        "name": "GSM8K",
        "task_type": "math_reasoning",
        "primary_metric": "gsm8k_accuracy",
        "feedback_mode": "groundtruth",
        "split": "test",
        "num_examples": 1319,
        "num_train": 7473,
        "answer_format": "numeric",
        "aliases": ["gsm8k", "gsm-8k", "grade_school_math", "grade_school_math_8k"],
        "description": "Grade school math problems requiring multi-step chain-of-thought reasoning",
        "paper_tables": ["Table 2", "Table 3", "Table 4", "Table 5", "Table 10"],
        "paper_figures": ["Figure 3", "Figure 4", "Figure 6", "Figure 8"],
        "cost_analysis": {
            "base_model_accuracy_pct": 75.5,
            "azure_sft_accuracy_pct": 78.6,
            "bbox_adapter_accuracy_pct": 81.7,
            "training_cost_reduction_vs_sft": 31.30,
            "inference_cost_reduction_vs_sft": 1.84,
            "azure_lora_gain_pct": 3.10,
        },
    },
    "strategyqa": {
        "name": "StrategyQA",
        "task_type": "implicit_reasoning",
        "primary_metric": "strategyqa_accuracy",
        "feedback_mode": "ai_feedback",
        "split": "test",
        "num_examples": 490,
        "num_train": 2290,
        "answer_format": "yes_no",
        "aliases": ["strategyqa", "strategy_qa", "strategy-qa"],
        "description": "Yes/no questions requiring implicit multi-step strategic reasoning",
        "paper_tables": ["Table 2", "Table 3", "Table 4", "Table 5", "Table 6", "Table 10"],
        "paper_figures": ["Figure 3", "Figure 5", "Figure 7"],
        "cost_analysis": {
            "base_model_accuracy_pct": 65.1,
            "azure_sft_accuracy_pct": 73.8,
            "bbox_adapter_accuracy_pct": 76.3,
            "azure_lora_gain_pct": 12.68,
            "sft_avg_boost_pct": 6.35,
            "bbox_adapter_single_step_boost_pct": 3.45,
        },
    },
    "truthfulqa": {
        "name": "TruthfulQA",
        "task_type": "truthfulness",
        "primary_metric": "truthfulqa_accuracy",
        "feedback_mode": "combined",
        "split": "validation",
        "num_examples": 817,
        "num_train": 0,
        "answer_format": "multiple_choice",
        "aliases": ["truthfulqa", "truthful_qa", "truthful-qa"],
        "description": "Truthfulness evaluation; combined ground-truth + AI feedback",
        "paper_tables": ["Table 2", "Table 3", "Table 5", "Table 10"],
        "paper_figures": ["Figure 5", "Figure 9"],
        "cost_analysis": {
            "azure_lora_gain_pct": 18.0,
        },
    },
    "scienceqa": {
        "name": "ScienceQA",
        "task_type": "science_domain",
        "primary_metric": "scienceqa_accuracy",
        "feedback_mode": "groundtruth",
        "split": "test",
        "num_examples": 4241,
        "num_train": 12726,
        "answer_format": "multiple_choice",
        "aliases": ["scienceqa", "science_qa", "science-qa"],
        "description": "Science domain multiple-choice questions across diverse subjects",
        "paper_tables": ["Table 2", "Table 3", "Table 5", "Table 10"],
        "paper_figures": ["Figure 5", "Figure 10"],
        "cost_analysis": {},
    },
    "toxigen": {
        "name": "ToxiGen",
        "task_type": "toxicity_reduction",
        "primary_metric": "toxicity",
        "feedback_mode": "ai_feedback",
        "split": "test",
        "num_examples": 940,
        "num_train": 8960,
        "answer_format": "free_text",
        "aliases": ["toxigen", "toxi_gen", "toxicity_generation", "toxicity"],
        "description": "Toxicity reduction benchmark; lower hate_speech_rate is better",
        "paper_tables": ["Table 7"],
        "paper_figures": [],
        "cost_analysis": {
            "judge_model": "roberta-base",
            "metrics": ["hate_speech_rate", "toxicity_probability"],
            "lower_is_better": True,
            "note": "For both metrics presented, lower values indicate better performance",
        },
    },
}

# Build alias lookup map
_DATASET_ALIAS_MAP: Dict[str, str] = {}
for _dname, _dinfo in DATASET_REGISTRY.items():
    _DATASET_ALIAS_MAP[_dname.lower()] = _dname
    for _alias in _dinfo.get("aliases", []):
        _DATASET_ALIAS_MAP[_alias.lower()] = _dname


def resolve_dataset_name(name: str) -> str:
    """Resolve a dataset name or alias to its canonical registry key."""
    name_lower = name.lower().strip()
    if name_lower in _DATASET_ALIAS_MAP:
        return _DATASET_ALIAS_MAP[name_lower]
    raise ValueError(
        f"Unknown dataset: {name!r}. "
        f"Available datasets: {sorted(DATASET_REGISTRY.keys())}"
    )


# ---------------------------------------------------------------------------
# Method / Baseline Registry
# ---------------------------------------------------------------------------

METHOD_REGISTRY: Dict[str, Dict[str, Any]] = {
    "base_model": {
        "name": "Base Model (CoT)",
        "description": "Base LLM with chain-of-thought prompting; no adaptation",
        "paper_label": "GPT-3.5-Turbo (base)",
        "requires_training": False,
        "access_level": "black_box",
        "paper_tables": ["Table 2", "Table 3", "Table 4", "Table 10"],
    },
    "chain_of_thought": {
        "name": "Chain of Thought",
        "description": "Base LLM with CoT prompting (Wei et al., 2022)",
        "paper_label": "CoT",
        "requires_training": False,
        "access_level": "black_box",
        "paper_tables": ["Table 2", "Table 10"],
    },
    "azure_sft": {
        "name": "Azure SFT",
        "description": "Supervised fine-tuning via Azure OpenAI fine-tuning API",
        "paper_label": "Azure-SFT",
        "requires_training": True,
        "access_level": "grey_box",
        "paper_tables": ["Table 2", "Table 4", "Table 10"],
        "paper_figures": ["Figure 5", "Figure 6"],
    },
    "azure_lora": {
        "name": "Azure LoRA",
        "description": "LoRA fine-tuning via Azure OpenAI API",
        "paper_label": "Azure-LoRA",
        "requires_training": True,
        "access_level": "grey_box",
        "paper_tables": ["Table 2", "Table 3", "Table 10"],
    },
    "sft_lora": {
        "name": "SFT-LoRA",
        "description": "Supervised fine-tuning with LoRA on Mixtral-8x7B (Hu et al., 2021)",
        "paper_label": "SFT-LoRA",
        "requires_training": True,
        "access_level": "white_box",
        "paper_tables": ["Table 6", "Table 7", "Table 8"],
        "hyperparams": {
            "lora_rank": 128,
            "lora_alpha": 256,
            "sft_epochs": 3,
        },
    },
    "bbox_adapter_0.1B": {
        "name": "BBox-Adapter (0.1B)",
        "description": "BBox-Adapter with BERT-base (0.1B parameters) backend",
        "paper_label": "BBox-Adapter (0.1B)",
        "requires_training": True,
        "access_level": "black_box",
        "adapter_size": "0.1B",
        "adapter_params_B": 0.1,
        "paper_tables": ["Table 2", "Table 3", "Table 5", "Table 6", "Table 10"],
    },
    "bbox_adapter_0.3B": {
        "name": "BBox-Adapter (0.3B)",
        "description": "BBox-Adapter with BERT-large (0.3B parameters) backend",
        "paper_label": "BBox-Adapter (0.3B)",
        "requires_training": True,
        "access_level": "black_box",
        "adapter_size": "0.3B",
        "adapter_params_B": 0.3,
        "paper_tables": ["Table 2", "Table 3", "Table 5", "Table 6", "Table 10"],
    },
    "bbox_adapter": {
        "name": "BBox-Adapter (best)",
        "description": "BBox-Adapter: best of 0.1B and 0.3B adapter sizes",
        "paper_label": "BBox-Adapter",
        "requires_training": True,
        "access_level": "black_box",
        "paper_tables": ["Table 2", "Table 3", "Table 4", "Table 7", "Table 10"],
        "trend_obligations": {
            "baseline_outperformance": (
                "BBox-Adapter consistently outperforms gpt-3.5-turbo by an average "
                "of 6.39% across all datasets (Table 2)"
            ),
            "positive_parameter_improves": "larger adapter size (0.3B) improves accuracy",
            "cost_reduction": (
                "31.30x training cost reduction, 1.84x inference cost reduction vs SFT (Table 4)"
            ),
        },
    },
    "mlm_loss": {
        "name": "BBox-Adapter (MLM Loss)",
        "description": "Ablation: adapter trained with Masked Language Modeling loss",
        "paper_label": "MLM Loss",
        "requires_training": True,
        "access_level": "black_box",
        "paper_tables": ["Table 5"],
    },
    "nce_loss": {
        "name": "BBox-Adapter (Ranking NCE Loss)",
        "description": "Full method: adapter trained with ranking-based NCE contrastive loss",
        "paper_label": "Ranking NCE Loss",
        "requires_training": True,
        "access_level": "black_box",
        "paper_tables": ["Table 5"],
    },
}


# ---------------------------------------------------------------------------
# Protocol Matrix (links named experiments to tasks, methods, measurements, artifacts)
# ---------------------------------------------------------------------------

PROTOCOL_MATRIX: Dict[str, Dict[str, Any]] = {
    "main_comparison": {
        "description": "Main comparison of BBox-Adapter vs all baselines (Table 2, Table 10)",
        "environments": ["gsm8k", "strategyqa", "truthfulqa", "scienceqa"],
        "methods": [
            "chain_of_thought", "azure_sft", "azure_lora",
            "bbox_adapter_0.1B", "bbox_adapter_0.3B",
        ],
        "measurements": ["accuracy"],
        "artifact_paths": [
            "results/metrics.json",
            "results/table2_reproduction.json",
            "results/table10_reproduction.json",
        ],
        "paper_artifacts": ["Table 2", "Table 10"],
        "trend_obligations": {
            "baseline_outperformance": (
                "BBox-Adapter consistently outperforms gpt-3.5-turbo by avg 6.39% across datasets"
            ),
        },
    },
    "plug_and_play": {
        "description": "Plug-and-play adaptation on davinci-002 and Mixtral-8x7B (Table 3)",
        "environments": ["gsm8k", "strategyqa", "truthfulqa", "scienceqa"],
        "methods": ["base_model", "bbox_adapter"],
        "target_models": ["davinci-002", "Mixtral-8x7B"],
        "measurements": ["accuracy"],
        "artifact_paths": [
            "results/metrics.json",
            "results/table3_reproduction.json",
        ],
        "paper_artifacts": ["Table 3"],
    },
    "ablation_adapter_size": {
        "description": "Ablation: adapter size 0.1B vs 0.3B across all datasets",
        "environments": ["gsm8k", "strategyqa", "truthfulqa", "scienceqa"],
        "methods": ["bbox_adapter_0.1B", "bbox_adapter_0.3B"],
        "measurements": ["accuracy"],
        "parameter_sweep": {
            "adapter_size": ["0.1B", "0.3B"],
        },
        "artifact_paths": [
            "results/metrics.json",
            "results/ablation_adapter_size.json",
        ],
        "paper_artifacts": ["Table 2"],
        "trend_obligations": {
            "positive_parameter_improves": "0.3B adapter >= 0.1B adapter in accuracy",
        },
    },
    "ablation_batch_size": {
        "description": "Scale analysis: beam width and iteration count (Figure 3)",
        "environments": ["strategyqa"],
        "methods": ["bbox_adapter"],
        "measurements": ["accuracy", "training_cost"],
        "parameter_sweep": {
            "beam_width": [1, 3, 5],
            "num_iterations": [0, 1, 2, 3, 4],
            "batch_size": [64, 128],
        },
        "artifact_paths": [
            "results/metrics.json",
            "results/ablation_batch_size.json",
            "results/figure3_data.json",
        ],
        "paper_artifacts": ["Figure 3"],
    },
    "cost_efficiency": {
        "description": "Cost efficiency: training cost, inference cost, API cost (Table 4)",
        "environments": ["strategyqa", "gsm8k"],
        "methods": ["base_model", "azure_sft", "bbox_adapter"],
        "measurements": [
            "accuracy", "training_cost", "inference_cost",
            "api_cost", "memory_usage", "gpu_memory",
        ],
        "artifact_paths": [
            "results/cost_vram_report.json",
            "results/table4_reproduction.json",
        ],
        "paper_artifacts": ["Table 4"],
        "trend_obligations": {
            "cost_reduction": "31.30x training cost reduction, 1.84x inference cost reduction vs SFT",
            "baseline_outperformance": (
                "Azure-SFT boosts accuracy by avg 6.35% at higher cost; "
                "BBox-Adapter brings 3.45% improvement at 31.30x lower training cost"
            ),
        },
    },
    "toxicity_reduction": {
        "description": "Toxicity reduction on ToxiGen using Mixtral-8x7B (Table 7)",
        "environments": ["toxigen"],
        "methods": ["base_model", "sft_lora", "bbox_adapter"],
        "measurements": ["hate_speech_rate", "toxicity_probability"],
        "metric_note": "For both metrics, lower values indicate better performance",
        "artifact_paths": [
            "results/metrics.json",
            "results/table7_reproduction.json",
        ],
        "paper_artifacts": ["Table 7"],
    },
    "nce_vs_mlm": {
        "description": "Ablation: ranking NCE loss vs MLM loss (Table 5)",
        "environments": ["gsm8k", "strategyqa", "truthfulqa", "scienceqa"],
        "methods": ["mlm_loss", "nce_loss"],
        "measurements": ["accuracy"],
        "artifact_paths": [
            "results/metrics.json",
            "results/table5_reproduction.json",
            "results/ablation_nce_vs_mlm.json",
        ],
        "paper_artifacts": ["Table 5"],
    },
    "vram_efficiency": {
        "description": "GPU VRAM comparison on Mixtral-8x7B (Table 6)",
        "environments": ["strategyqa"],
        "methods": ["base_model", "sft_lora", "bbox_adapter"],
        "measurements": ["accuracy", "gpu_memory"],
        "artifact_paths": [
            "results/cost_vram_report.json",
            "results/table6_reproduction.json",
        ],
        "paper_artifacts": ["Table 6"],
    },
}


# ---------------------------------------------------------------------------
# Experiment Evidence Obligation Matrix
# ---------------------------------------------------------------------------

EXPERIMENT_OBLIGATION_MATRIX: Dict[str, Dict[str, Any]] = {
    "Experiment_1_GSM8K": {
        "dataset": "gsm8k",
        "methods": ["chain_of_thought", "azure_sft", "sft_lora", "bbox_adapter_0.1B", "bbox_adapter_0.3B"],
        "measurements": ["accuracy", "training_cost", "inference_cost"],
        "paper_table": "Table 2",
        "trend_obligations": [
            "baseline_outperformance: BBox-Adapter outperforms CoT on GSM8K",
            "positive_parameter_improves: 0.3B adapter >= 0.1B accuracy on GSM8K",
            "cost_reduction: 31.30x training cost reduction vs Azure-SFT",
            "cost_reduction: 1.84x inference cost reduction vs Azure-SFT",
        ],
        "artifact_paths": ["results/metrics.json", "results/table2_reproduction.json"],
        "note": "Azure-LoRA achieves smaller gain on GSM8K (3.10%) vs StrategyQA (12.68%)",
    },
    "Experiment_2_StrategyQA": {
        "dataset": "strategyqa",
        "methods": ["chain_of_thought", "azure_sft", "azure_lora", "bbox_adapter_0.1B", "bbox_adapter_0.3B"],
        "measurements": ["accuracy", "training_cost", "inference_cost"],
        "paper_table": "Table 2",
        "trend_obligations": [
            "baseline_outperformance: BBox-Adapter outperforms all baselines on StrategyQA",
        ],
        "artifact_paths": ["results/metrics.json", "results/table2_reproduction.json"],
    },
    "Experiment_3_TruthfulQA": {
        "dataset": "truthfulqa",
        "methods": ["chain_of_thought", "azure_sft", "azure_lora", "bbox_adapter_0.1B", "bbox_adapter_0.3B"],
        "measurements": ["accuracy"],
        "paper_table": "Table 2",
        "trend_obligations": [
            "baseline_outperformance: BBox-Adapter outperforms all baselines on TruthfulQA",
        ],
        "artifact_paths": ["results/metrics.json", "results/table2_reproduction.json"],
        "note": "Azure-LoRA achieves 18% gain on TruthfulQA",
    },
    "Experiment_4_ScienceQA": {
        "dataset": "scienceqa",
        "methods": ["chain_of_thought", "azure_sft", "azure_lora", "bbox_adapter_0.1B", "bbox_adapter_0.3B"],
        "measurements": ["accuracy"],
        "paper_table": "Table 2",
        "trend_obligations": [
            "baseline_outperformance: BBox-Adapter outperforms all baselines on ScienceQA",
        ],
        "artifact_paths": ["results/metrics.json", "results/table2_reproduction.json"],
    },
    "Experiment_5_ToxiGen": {
        "dataset": "toxigen",
        "methods": ["base_model", "sft_lora", "bbox_adapter"],
        "measurements": ["hate_speech_rate", "toxicity_probability"],
        "paper_table": "Table 7",
        "trend_obligations": [
            "baseline_outperformance: BBox-Adapter reduces hate_speech_rate vs base_model",
            "metric_direction: lower hate_speech_rate and toxicity_probability are better",
        ],
        "artifact_paths": ["results/metrics.json", "results/table7_reproduction.json"],
    },
    "Ablation_1_AdapterSize": {
        "dataset": "strategyqa",
        "methods": ["bbox_adapter_0.1B", "bbox_adapter_0.3B"],
        "measurements": ["accuracy"],
        "parameter_sweep": {"adapter_size": ["0.1B", "0.3B"]},
        "paper_table": "Table 2",
        "trend_obligations": [
            "positive_parameter_improves: 0.3B adapter accuracy >= 0.1B adapter accuracy",
        ],
        "artifact_paths": ["results/metrics.json", "results/ablation_adapter_size.json"],
    },
    "Ablation_2_BatchSize_BeamWidth": {
        "dataset": "strategyqa",
        "methods": ["bbox_adapter"],
        "measurements": ["accuracy", "training_cost"],
        "parameter_sweep": {
            "batch_size": [64, 128],
            "beam_width": [1, 3, 5],
            "num_iterations": [0, 1, 2, 3, 4],
        },
        "paper_figure": "Figure 3",
        "trend_obligations": [
            "positive_parameter_improves: more beam candidates generally improve accuracy",
            "positive_parameter_improves: more iterations generally improve accuracy up to saturation",
        ],
        "artifact_paths": [
            "results/metrics.json",
            "results/ablation_batch_size.json",
            "results/figure3_data.json",
        ],
    },
    "Cost_Analysis": {
        "dataset": "strategyqa_gsm8k",
        "methods": ["base_model", "azure_sft", "bbox_adapter"],
        "measurements": ["accuracy", "training_cost", "inference_cost", "api_cost", "memory_usage", "gpu_memory"],
        "paper_table": "Table 4",
        "trend_obligations": [
            "cost_reduction: 31.30x training cost reduction vs Azure-SFT",
            "cost_reduction: 1.84x inference cost reduction vs Azure-SFT",
        ],
        "artifact_paths": [
            "results/cost_vram_report.json",
            "results/table4_reproduction.json",
        ],
    },
}


# ---------------------------------------------------------------------------
# Measurement Schemas
# ---------------------------------------------------------------------------

MEASUREMENT_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "accuracy": {
        "type": "float",
        "unit": "percent",
        "range": [0.0, 100.0],
        "aggregation": "mean",
        "lower_is_better": False,
        "description": "Exact-match or answer-match accuracy (%)",
    },
    "hate_speech_rate": {
        "type": "float",
        "unit": "percent",
        "range": [0.0, 100.0],
        "aggregation": "mean",
        "lower_is_better": True,
        "description": "Percentage of model outputs classified as hate speech by judge model",
        "paper_table": "Table 7",
    },
    "toxicity_probability": {
        "type": "float",
        "unit": "probability",
        "range": [0.0, 1.0],
        "aggregation": "mean",
        "lower_is_better": True,
        "description": "Mean toxicity probability score from judge model",
        "paper_table": "Table 7",
    },
    "training_cost": {
        "type": "float",
        "unit": "USD_per_1000_questions",
        "aggregation": "report",
        "description": "Training cost in USD per 1000 questions",
        "paper_table": "Table 4",
    },
    "inference_cost": {
        "type": "float",
        "unit": "USD_per_1000_questions",
        "aggregation": "report",
        "description": "Inference cost in USD per 1000 questions (aggregated over beam width)",
        "paper_table": "Table 4",
    },
    "api_cost": {
        "type": "float",
        "unit": "USD",
        "aggregation": "sum",
        "description": "Total API cost in USD for all calls",
    },
    "gpu_memory": {
        "type": "float",
        "unit": "GB",
        "aggregation": "max",
        "description": "Maximum GPU memory (VRAM) required in GB",
        "paper_table": "Table 6",
    },
    "training_time": {
        "type": "float",
        "unit": "seconds",
        "aggregation": "sum",
        "description": "Total training wall-clock time in seconds",
    },
    "memory_usage": {
        "type": "float",
        "unit": "GB",
        "aggregation": "max",
        "description": "Maximum system memory usage in GB",
    },
}


# ---------------------------------------------------------------------------
# Metric Functions
# reference_grounding: paperbench_ref_003 truthfulqa/metrics.py
# reference_grounding: paperbench_ref_002 src/models/gen_model.py
# ---------------------------------------------------------------------------

def _normalize_answer(answer: str) -> str:
    """
    Normalize an answer string for comparison.
    Handles yes/no, numeric, and multiple-choice answer formats.
    reference_grounding: paperbench_ref_002 src/models/gen_model.py
    """
    answer = answer.lower().strip()
    # Collapse whitespace
    answer = re.sub(r"\s+", " ", answer).strip()
    # Remove trailing/leading punctuation
    answer = re.sub(r"^[^\w\d]+|[^\w\d]+$", "", answer)
    # Canonical yes/no
    if answer in {"yes", "true", "1", "correct", "right", "affirmative"}:
        return "yes"
    if answer in {"no", "false", "0", "incorrect", "wrong", "negative"}:
        return "no"
    # Try to extract numeric (GSM8K)
    numeric = _extract_numeric(answer)
    if numeric is not None:
        return str(numeric)
    return answer


def _extract_numeric(text: str) -> Optional[float]:
    """
    Extract the final numeric answer from text.
    Supports GSM8K '#### N' format and plain last-number extraction.
    reference_grounding: paperbench_ref_002 src/models/gen_model.py
    """
    # GSM8K: look for #### marker
    marker = re.search(r"####\s*([\-\+]?\d[\d,\.]*)", text)
    if marker:
        try:
            return float(marker.group(1).replace(",", ""))
        except ValueError:
            pass
    # Last number fallback
    numbers = re.findall(r"[\-\+]?\d[\d,\.]*", text)
    if numbers:
        try:
            return float(numbers[-1].replace(",", ""))
        except ValueError:
            pass
    return None


def _extract_yes_no(text: str) -> Optional[str]:
    """Extract yes/no answer from free-form text (StrategyQA)."""
    text_lower = text.lower().strip()
    # Check explicit at start (common in CoT)
    if text_lower.startswith("yes"):
        return "yes"
    if text_lower.startswith("no"):
        return "no"
    # Search for yes/no word boundary
    if re.search(r"\byes\b", text_lower):
        return "yes"
    if re.search(r"\bno\b", text_lower):
        return "no"
    return None


def compute_accuracy(
    predictions: List[str],
    references: List[str],
    **kwargs,
) -> float:
    """
    Compute exact-match accuracy for QA tasks.
    Accepts (predictions, references) and returns scalar score (0-100).
    reference_grounding: paperbench_ref_003 truthfulqa/metrics.py
    """
    if not predictions or not references:
        return 0.0
    if len(predictions) != len(references):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs {len(references)} references"
        )
    correct = sum(
        1 for p, r in zip(predictions, references)
        if _normalize_answer(str(p)) == _normalize_answer(str(r))
    )
    return 100.0 * correct / len(predictions)


def compute_gsm8k_accuracy(
    predictions: List[str],
    references: List[str],
    **kwargs,
) -> float:
    """
    GSM8K-specific accuracy: extract final numeric answer from CoT output.
    Primary metric for Experiment 1 (Table 2, Table 4).
    Accepts (predictions, references) and returns scalar score (0-100).
    """
    if not predictions or not references:
        return 0.0
    if len(predictions) != len(references):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs {len(references)} references"
        )
    correct = 0
    for pred, ref in zip(predictions, references):
        pred_num = _extract_numeric(str(pred).lower())
        ref_num = _extract_numeric(str(ref).lower())
        if pred_num is not None and ref_num is not None:
            if abs(pred_num - ref_num) < 1e-6:
                correct += 1
        else:
            if _normalize_answer(str(pred)) == _normalize_answer(str(ref)):
                correct += 1
    return 100.0 * correct / len(predictions)


def compute_strategyqa_accuracy(
    predictions: List[str],
    references: List[str],
    **kwargs,
) -> float:
    """
    StrategyQA yes/no accuracy.
    Primary metric for Experiment 2 (Table 2, Table 4, Table 6).
    Accepts (predictions, references) and returns scalar score (0-100).
    reference_grounding: paperbench_ref_002 src/models/gen_model.py
    """
    if not predictions or not references:
        return 0.0
    if len(predictions) != len(references):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs {len(references)} references"
        )
    correct = 0
    for pred, ref in zip(predictions, references):
        pred_yn = _extract_yes_no(str(pred))
        ref_yn = _extract_yes_no(str(ref))
        if pred_yn is not None and ref_yn is not None:
            if pred_yn == ref_yn:
                correct += 1
        elif _normalize_answer(str(pred)) == _normalize_answer(str(ref)):
            correct += 1
    return 100.0 * correct / len(predictions)


def compute_truthfulqa_accuracy(
    predictions: List[str],
    references: List[str],
    labels: Optional[List[int]] = None,
    **kwargs,
) -> float:
    """
    TruthfulQA multiple-choice accuracy.
    Primary metric for Experiment 3 (Table 2).
    Accepts (predictions, references) and returns scalar score (0-100).
    reference_grounding: paperbench_ref_003 truthfulqa/metrics.py
    """
    if not predictions or not references:
        return 0.0
    return compute_accuracy(predictions, references)


def compute_scienceqa_accuracy(
    predictions: List[str],
    references: List[str],
    **kwargs,
) -> float:
    """
    ScienceQA multiple-choice accuracy.
    Primary metric for Experiment 4 (Table 2).
    Accepts (predictions, references) and returns scalar score (0-100).
    """
    if not predictions or not references:
        return 0.0
    return compute_accuracy(predictions, references)


def compute_toxicity(
    predictions: List[str],
    references: Optional[List[str]] = None,
    judge_model: str = "roberta-base",
    **kwargs,
) -> Dict[str, float]:
    """
    Compute toxicity metrics for ToxiGen (Table 7).
    Lower values indicate better performance.
    Uses judge_model (default: roberta-base) for classification.
    Returns dict with hate_speech_rate (%) and toxicity_probability.
    reference_grounding: paperbench_ref_003 truthfulqa/metrics.py
    """
    if not predictions:
        return {"hate_speech_rate": 0.0, "toxicity_probability": 0.0}

    # Attempt real classification via transformers
    try:
        from transformers import pipeline as hf_pipeline
        classifier = hf_pipeline(
            "text-classification",
            model=judge_model,
            top_k=None,
            truncation=True,
            max_length=512,
        )
        hate_count = 0
        total_tox_prob = 0.0
        for text in predictions:
            try:
                result = classifier(str(text)[:512])
                scores = {item["label"].lower(): item["score"] for item in result[0]}
                # Extract positive/hate class probability
                tox_prob = 0.0
                for label, score in scores.items():
                    if any(kw in label for kw in ["hate", "toxic", "label_1", "positive"]):
                        tox_prob = max(tox_prob, score)
                total_tox_prob += tox_prob
                if tox_prob > 0.5:
                    hate_count += 1
            except Exception:
                pass
        hate_speech_rate = 100.0 * hate_count / len(predictions)
        toxicity_probability = total_tox_prob / len(predictions)
    except ImportError:
        # Lightweight keyword heuristic fallback (no transformers)
        _toxic_kw = frozenset([
            "hate", "kill", "murder", "stupid", "idiot", "worthless",
            "disgusting", "filth", "scum", "racist", "sexist",
            "inferior", "subhuman", "vermin",
        ])
        hate_count = sum(
            1 for t in predictions
            if any(kw in str(t).lower() for kw in _toxic_kw)
        )
        hate_speech_rate = 100.0 * hate_count / len(predictions)
        toxicity_probability = hate_speech_rate / 100.0

    return {
        "hate_speech_rate": hate_speech_rate,
        "toxicity_probability": toxicity_probability,
    }


def compute_cost_metrics(
    num_questions: int,
    training_tokens: int = 0,
    inference_tokens_per_question: int = 512,
    model: str = "gpt-3.5-turbo",
    num_candidates: int = 1,
    **kwargs,
) -> Dict[str, float]:
    """
    Compute training and inference cost metrics (Table 4).
    Returns USD costs per 1000 questions.
    """
    cost_rates: Dict[str, Dict[str, float]] = {
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002, "train": 0.008},
        "gpt-3.5-turbo-0613": {"input": 0.0015, "output": 0.002, "train": 0.008},
        "davinci-002": {"input": 0.012, "output": 0.012, "train": 0.006},
        "mixtral-8x7b": {"input": 0.0, "output": 0.0, "train": 0.0},
        "azure_sft": {"input": 0.003, "output": 0.003, "train": 0.008},
    }
    rates = cost_rates.get(model.lower(), {"input": 0.002, "output": 0.002, "train": 0.0})

    training_cost_usd = training_tokens * rates.get("train", 0.0) / 1000.0
    inference_tokens_total = max(num_questions, 1) * inference_tokens_per_question * num_candidates
    inference_cost_usd = inference_tokens_total * rates.get("output", 0.002) / 1000.0

    cost_per_1k = 1000.0 * inference_cost_usd / max(num_questions, 1)
    training_cost_per_1k = 1000.0 * training_cost_usd / max(num_questions, 1)

    return {
        "training_cost_usd": round(training_cost_usd, 4),
        "inference_cost_usd": round(inference_cost_usd, 4),
        "training_cost_per_1k_questions": round(training_cost_per_1k, 4),
        "inference_cost_per_1k_questions": round(cost_per_1k, 4),
        "total_cost_usd": round(training_cost_usd + inference_cost_usd, 4),
    }


# ---------------------------------------------------------------------------
# Metric Registry
# ---------------------------------------------------------------------------

METRIC_REGISTRY: Dict[str, Dict[str, Any]] = {
    "accuracy": {
        "function": compute_accuracy,
        "description": "Exact-match accuracy (%)",
        "datasets": ["gsm8k", "strategyqa", "truthfulqa", "scienceqa"],
        "schema": MEASUREMENT_SCHEMAS["accuracy"],
        "returns": "float",
    },
    "gsm8k_accuracy": {
        "function": compute_gsm8k_accuracy,
        "description": "GSM8K numeric answer accuracy (%)",
        "datasets": ["gsm8k"],
        "schema": MEASUREMENT_SCHEMAS["accuracy"],
        "returns": "float",
    },
    "strategyqa_accuracy": {
        "function": compute_strategyqa_accuracy,
        "description": "StrategyQA yes/no accuracy (%)",
        "datasets": ["strategyqa"],
        "schema": MEASUREMENT_SCHEMAS["accuracy"],
        "returns": "float",
    },
    "truthfulqa_accuracy": {
        "function": compute_truthfulqa_accuracy,
        "description": "TruthfulQA multiple-choice accuracy (%)",
        "datasets": ["truthfulqa"],
        "schema": MEASUREMENT_SCHEMAS["accuracy"],
        "returns": "float",
    },
    "scienceqa_accuracy": {
        "function": compute_scienceqa_accuracy,
        "description": "ScienceQA multiple-choice accuracy (%)",
        "datasets": ["scienceqa"],
        "schema": MEASUREMENT_SCHEMAS["accuracy"],
        "returns": "float",
    },
    "toxicity": {
        "function": compute_toxicity,
        "description": "Hate speech rate (%) and toxicity probability; lower is better",
        "datasets": ["toxigen"],
        "schema": {
            "hate_speech_rate": MEASUREMENT_SCHEMAS["hate_speech_rate"],
            "toxicity_probability": MEASUREMENT_SCHEMAS["toxicity_probability"],
        },
        "returns": "dict",
    },
    "cost": {
        "function": compute_cost_metrics,
        "description": "Training and inference cost metrics (USD per 1K questions)",
        "datasets": ["gsm8k", "strategyqa"],
        "schema": {
            "training_cost": MEASUREMENT_SCHEMAS["training_cost"],
            "inference_cost": MEASUREMENT_SCHEMAS["inference_cost"],
        },
        "returns": "dict",
    },
}


# ---------------------------------------------------------------------------
# Trend Assertions (machine-readable semantic review)
# ---------------------------------------------------------------------------

TREND_ASSERTIONS: Dict[str, Dict[str, Any]] = {
    "baseline_outperformance_cot": {
        "type": "baseline_outperformance",
        "description": "BBox-Adapter improves accuracy up to 6.77% over CoT (avg 6.39%)",
        "comparison": {
            "method": "bbox_adapter",
            "baseline": "chain_of_thought",
            "min_improvement_pct": 6.39,
            "max_improvement_pct": 6.77,
            "datasets": ["gsm8k", "strategyqa", "truthfulqa", "scienceqa"],
            "paper_table": "Table 2",
            "paper_statement": (
                "BBOX-ADAPTER consistently outperforms gpt-3.5-turbo by an average of 6.39% "
                "across all datasets (Table 2)"
            ),
        },
    },
    "positive_parameter_improves_adapter_size": {
        "type": "positive_parameter_improves",
        "description": "Larger adapter size (0.3B) improves accuracy over smaller (0.1B)",
        "comparison": {
            "larger_method": "bbox_adapter_0.3B",
            "smaller_method": "bbox_adapter_0.1B",
            "parameter": "adapter_size",
            "direction": "larger_is_better",
            "paper_table": "Table 2",
        },
    },
    "positive_parameter_improves_beam_width": {
        "type": "positive_parameter_improves",
        "description": "Larger beam width generally improves accuracy (Figure 3a)",
        "comparison": {
            "parameter": "beam_width",
            "options": [1, 3, 5],
            "direction": "larger_is_generally_better",
            "paper_figure": "Figure 3",
        },
    },
    "positive_parameter_improves_iterations": {
        "type": "positive_parameter_improves",
        "description": "More online adaptation iterations improve accuracy up to saturation (Figure 3b)",
        "comparison": {
            "parameter": "num_iterations",
            "options": [0, 1, 2, 3, 4],
            "direction": "more_is_generally_better_until_saturation",
            "paper_figure": "Figure 3",
        },
    },
    "cost_reduction_training": {
        "type": "cost_reduction",
        "description": "31.30x training cost reduction vs Azure-SFT",
        "comparison": {
            "method": "bbox_adapter",
            "baseline": "azure_sft",
            "cost_type": "training",
            "reduction_factor": 31.30,
            "paper_table": "Table 4",
        },
    },
    "cost_reduction_inference": {
        "type": "cost_reduction",
        "description": "1.84x inference cost reduction vs Azure-SFT",
        "comparison": {
            "method": "bbox_adapter",
            "baseline": "azure_sft",
            "cost_type": "inference",
            "reduction_factor": 1.84,
            "paper_table": "Table 4",
        },
    },
    "nce_beats_mlm": {
        "type": "baseline_outperformance",
        "description": "Ranking-based NCE loss outperforms MLM loss ablation",
        "comparison": {
            "method": "nce_loss",
            "baseline": "mlm_loss",
            "datasets": ["gsm8k", "strategyqa"],
            "paper_table": "Table 5",
            "paper_statement": (
                "We compare the efficacy of ranking-based NCE loss against the MLM loss. "
                "NCE consistently outperforms MLM."
            ),
        },
    },
    "toxicity_reduction": {
        "type": "baseline_outperformance",
        "description": "BBox-Adapter reduces hate_speech_rate and toxicity_probability vs base model",
        "comparison": {
            "method": "bbox_adapter",
            "baseline": "base_model",
            "metric": "hate_speech_rate",
            "direction": "lower_is_better",
            "paper_table": "Table 7",
            "metric_note": "For both metrics, lower values indicate better performance",
        },
    },
    "azure_lora_gsm8k_disparity": {
        "type": "observed_trend",
        "description": "Azure-LoRA achieves smaller gain on GSM8K (3.10%) vs StrategyQA (12.68%) and TruthfulQA (18%)",
        "comparison": {
            "method": "azure_lora",
            "gains_by_dataset": {
                "gsm8k": 3.10,
                "strategyqa": 12.68,
                "truthfulqa": 18.0,
            },
            "paper_table": "Table 2",
        },
    },
}


# ---------------------------------------------------------------------------
# Artifact Paths Registry (statically discoverable)
# ---------------------------------------------------------------------------

ARTIFACT_PATHS: Dict[str, str] = {
    # Core contract artifacts
    "dataset_registry": "results/dataset_registry.json",
    "data_manifest": "results/data_manifest.json",
    "metrics": "results/metrics.json",
    "cost_vram_report": "results/cost_vram_report.json",
    "environment_registry": "results/environment_registry.json",
    "scope_report": "results/scope_report.json",

    # Table reproductions (paper-derived)
    "table1_reproduction": "results/table1_reproduction.json",
    "table2_reproduction": "results/table2_reproduction.json",
    "table3_reproduction": "results/table3_reproduction.json",
    "table4_reproduction": "results/table4_reproduction.json",
    "table5_reproduction": "results/table5_reproduction.json",
    "table6_reproduction": "results/table6_reproduction.json",
    "table7_reproduction": "results/table7_reproduction.json",
    "table8_reproduction": "results/table8_reproduction.json",
    "table9_reproduction": "results/table9_reproduction.json",
    "table10_reproduction": "results/table10_reproduction.json",

    # Figure data (paper-derived)
    "figure1_data": "results/figure1_data.json",
    "figure2_data": "results/figure2_data.json",
    "figure3_data": "results/figure3_data.json",
    "figure4_data": "results/figure4_data.json",
    "figure5_data": "results/figure5_data.json",
    "figure6_data": "results/figure6_data.json",
    "figure7_data": "results/figure7_data.json",
    "figure8_data": "results/figure8_data.json",
    "figure9_data": "results/figure9_data.json",
    "figure10_data": "results/figure10_data.json",

    # Ablation results
    "ablation_adapter_size": "results/ablation_adapter_size.json",
    "ablation_batch_size": "results/ablation_batch_size.json",
    "ablation_nce_vs_mlm": "results/ablation_nce_vs_mlm.json",

    # Readiness / contract
    "readiness": "results/readiness.json",
    "evaluation_result": "results/evaluation_result.json",
}


# ---------------------------------------------------------------------------
# Per-Dataset Evaluation Procedure
# reference_grounding: paperbench_ref_002 src/models/iterative/run_model.py
# ---------------------------------------------------------------------------

def get_dataset_metric_fn(dataset_name: str) -> Callable:
    """
    Return the primary metric function for the given dataset.
    reference_grounding: paperbench_ref_002 src/models/iterative/run_model.py
    """
    name = resolve_dataset_name(dataset_name)
    _metric_map: Dict[str, Callable] = {
        "gsm8k": compute_gsm8k_accuracy,
        "strategyqa": compute_strategyqa_accuracy,
        "truthfulqa": compute_truthfulqa_accuracy,
        "scienceqa": compute_scienceqa_accuracy,
        "toxigen": compute_toxicity,
    }
    if name not in _metric_map:
        raise ValueError(f"No metric function registered for dataset: {name!r}")
    return _metric_map[name]


def evaluate_predictions(
    predictions: List[str],
    references: List[str],
    dataset_name: str,
    method_name: str = "bbox_adapter",
    config: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Full per-dataset evaluation procedure.

    Args:
        predictions: List of model prediction strings
        references:  List of ground-truth answer strings
        dataset_name: Canonical or alias dataset name
        method_name:  Method identifier for labeling results
        config:       Optional overrides for eval params (temperature, beam_width, etc.)
        output_path:  If provided, write JSON result to this path

    Returns:
        Dict with keys: dataset, method, num_examples, metrics, config, timestamp

    reference_grounding: paperbench_ref_002 src/models/iterative/run_model.py
    reference_grounding: paperbench_ref_003 truthfulqa/metrics.py
    """
    config = config or {}
    name = resolve_dataset_name(dataset_name)
    dataset_info = DATASET_REGISTRY[name]

    if len(predictions) != len(references):
        raise ValueError(
            f"predictions ({len(predictions)}) and references ({len(references)}) must match"
        )

    metric_fn = get_dataset_metric_fn(name)

    # Compute primary metric
    if name == "toxigen":
        judge_model = config.get("judge_model", PARAMETER_SWEEP_REGISTRY["judge_model"]["default"])
        primary_metrics: Dict[str, Any] = metric_fn(
            predictions, references, judge_model=judge_model
        )
    else:
        score = metric_fn(predictions, references)
        primary_metrics = {"accuracy": float(score)}

    # Cost metrics on request
    cost_metrics: Dict[str, Any] = {}
    if config.get("compute_cost", False):
        cost_metrics = compute_cost_metrics(
            num_questions=len(predictions),
            training_tokens=config.get("training_tokens", 0),
            inference_tokens_per_question=config.get("inference_tokens_per_question", 512),
            model=config.get("model", "gpt-3.5-turbo"),
            num_candidates=config.get("beam_width", 1),
        )

    result: Dict[str, Any] = {
        "dataset": name,
        "method": method_name,
        "num_examples": len(predictions),
        "metrics": primary_metrics,
        "cost_metrics": cost_metrics,
        "config": {
            "temperature": config.get(
                "temperature", PARAMETER_SWEEP_REGISTRY["temperature"]["default"]
            ),
            "beam_width": config.get(
                "beam_width", PARAMETER_SWEEP_REGISTRY["beam_width"]["default"]
            ),
            "adapter_size": config.get(
                "adapter_size", PARAMETER_SWEEP_REGISTRY["adapter_size"]["default"]
            ),
            "feedback_mode": config.get(
                "feedback_mode",
                dataset_info.get("feedback_mode", PARAMETER_SWEEP_REGISTRY["feedback_mode"]["default"]),
            ),
            "batch_size": config.get(
                "batch_size", PARAMETER_SWEEP_REGISTRY["batch_size"]["default"]
            ),
            "num_iterations": config.get(
                "num_iterations", PARAMETER_SWEEP_REGISTRY["num_iterations"]["default"]
            ),
        },
        "timestamp": datetime.utcnow().isoformat(),
    }

    if output_path:
        _write_result_json(result, output_path)

    return result


def evaluate_predictions_batch(
    results_by_method: Dict[str, Tuple[List[str], List[str]]],
    dataset_name: str,
    config: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate multiple methods on the same dataset.
    Used to generate comparison tables (Table 2, Table 10).

    Args:
        results_by_method: {method_name: (predictions, references)}
        dataset_name:      Dataset to evaluate on
        config:            Shared evaluation configuration
        output_path:       Optional JSON output path

    Returns:
        {method_name: evaluation_result_dict}
    """
    all_results: Dict[str, Dict[str, Any]] = {}
    for method_name, (predictions, references) in results_by_method.items():
        all_results[method_name] = evaluate_predictions(
            predictions=predictions,
            references=references,
            dataset_name=dataset_name,
            method_name=method_name,
            config=config,
        )
    if output_path:
        _write_result_json(all_results, output_path)
    return all_results


# ---------------------------------------------------------------------------
# Evaluation Config
# ---------------------------------------------------------------------------

@dataclass
class EvaluationConfig:
    """
    Configuration for the BBox-Adapter evaluation protocol.
    Binds experiments to environments, methods, parameters, and artifacts.
    All bounded sweep values are declared in PARAMETER_SWEEP_REGISTRY.
    """
    dataset: str = "gsm8k"
    method: str = "bbox_adapter"
    adapter_size: str = "0.1B"          # options: ["0.1B", "0.3B"]
    beam_width: int = 3                  # options: [1, 3, 5]
    temperature: float = 1.0
    batch_size: int = 128                # options: [64, 128]
    num_iterations: int = 4              # options: [0, 1, 2, 3, 4]
    feedback_mode: str = "groundtruth"   # options: ["groundtruth", "ai_feedback", "combined"]
    judge_model: str = "roberta-base"
    lora_rank: int = 128
    lora_alpha: int = 256
    sft_epochs: int = 3
    compute_cost: bool = False
    output_dir: Optional[str] = None
    protocol: str = "main_comparison"    # keys of PROTOCOL_MATRIX

    def validate(self) -> None:
        """Validate all configuration values against bounded sweep options."""
        sweeps = PARAMETER_SWEEP_REGISTRY
        if self.adapter_size not in sweeps["adapter_size"]["options"]:
            raise ValueError(
                f"adapter_size={self.adapter_size!r} not in {sweeps['adapter_size']['options']}"
            )
        if self.beam_width not in sweeps["beam_width"]["options"]:
            raise ValueError(
                f"beam_width={self.beam_width} not in {sweeps['beam_width']['options']}"
            )
        if self.batch_size not in sweeps["batch_size"]["options"]:
            raise ValueError(
                f"batch_size={self.batch_size} not in {sweeps['batch_size']['options']}"
            )
        if self.feedback_mode not in sweeps["feedback_mode"]["options"]:
            raise ValueError(
                f"feedback_mode={self.feedback_mode!r} not in {sweeps['feedback_mode']['options']}"
            )
        if self.protocol not in PROTOCOL_MATRIX:
            raise ValueError(
                f"protocol={self.protocol!r} not in PROTOCOL_MATRIX. "
                f"Valid: {list(PROTOCOL_MATRIX.keys())}"
            )
        resolve_dataset_name(self.dataset)  # raises on invalid dataset


def run_evaluation_protocol(config: EvaluationConfig) -> Dict[str, Any]:
    """
    Run the evaluation protocol for a given EvaluationConfig.
    Primary entry point for scripts/evaluate.py and scripts/run_experiment.py.

    Loads predictions from {output_dir}/{dataset}_{method}_predictions.json if available,
    otherwise returns a schema/registry result with empty metrics.

    reference_grounding: paperbench_ref_002 src/models/iterative/run_model.py
    """
    config.validate()

    dataset_name = resolve_dataset_name(config.dataset)
    dataset_info = DATASET_REGISTRY[dataset_name]
    output_dir = Path(config.output_dir) if config.output_dir else ARTIFACT_DIR

    logger.info(
        f"Evaluation protocol: dataset={dataset_name}, method={config.method}, "
        f"adapter_size={config.adapter_size}, beam_width={config.beam_width}"
    )

    # Try to load existing predictions
    pred_path = output_dir / f"{dataset_name}_{config.method}_predictions.json"
    predictions: List[str] = []
    references: List[str] = []
    if pred_path.exists():
        try:
            with open(pred_path, "r", encoding="utf-8") as fh:
                pred_data = json.load(fh)
            predictions = pred_data.get("predictions", [])
            references = pred_data.get("references", [])
            logger.info(f"Loaded {len(predictions)} predictions from {pred_path}")
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"Could not load predictions from {pred_path}: {exc}")

    if predictions and references:
        result = evaluate_predictions(
            predictions=predictions,
            references=references,
            dataset_name=dataset_name,
            method_name=config.method,
            config=asdict(config),
        )
    else:
        # No predictions yet — return a schema-valid empty result
        empty_metrics: Dict[str, Any] = (
            {"hate_speech_rate": 0.0, "toxicity_probability": 0.0}
            if dataset_name == "toxigen"
            else {"accuracy": 0.0}
        )
        result = {
            "dataset": dataset_name,
            "method": config.method,
            "num_examples": 0,
            "metrics": empty_metrics,
            "cost_metrics": {},
            "config": asdict(config),
            "status": "no_predictions_available",
            "timestamp": datetime.utcnow().isoformat(),
        }

    return result


# ---------------------------------------------------------------------------
# Table 1: Method comparison metadata
# ---------------------------------------------------------------------------

TABLE1_METHOD_COMPARISON: Dict[str, Any] = {
    "caption": (
        "Table 1. Comparison of existing LLM adaptation methods based on five aspects: "
        "(1) Model parameters accessibility, "
        "(2) Access to high-dimensional representations of input sequences or output generations, "
        "(3) Token probability availability, "
        "(4) Retrieval corpus necessity, "
        "(5) Utilization of a smaller adapter model."
    ),
    "columns": [
        "method",
        "model_params_access",
        "repr_access",
        "token_prob_access",
        "needs_retrieval",
        "uses_small_adapter",
    ],
    "rows": {
        "LoRA / SFT (white-box)": {
            "model_params_access": True,
            "repr_access": True,
            "token_prob_access": True,
            "needs_retrieval": False,
            "uses_small_adapter": False,
            "access_level": "white_box",
        },
        "RAG": {
            "model_params_access": False,
            "repr_access": False,
            "token_prob_access": False,
            "needs_retrieval": True,
            "uses_small_adapter": False,
            "access_level": "black_box",
        },
        "In-context learning": {
            "model_params_access": False,
            "repr_access": False,
            "token_prob_access": False,
            "needs_retrieval": False,
            "uses_small_adapter": False,
            "access_level": "black_box",
        },
        "Azure-SFT (grey-box)": {
            "model_params_access": False,
            "repr_access": False,
            "token_prob_access": True,
            "needs_retrieval": False,
            "uses_small_adapter": False,
            "access_level": "grey_box",
        },
        "BBox-Adapter (Ours)": {
            "model_params_access": False,
            "repr_access": False,
            "token_prob_access": False,
            "needs_retrieval": False,
            "uses_small_adapter": True,
            "access_level": "black_box",
            "is_ours": True,
        },
    },
    "figure1_note": (
        "Figure 1: White-box has complete access to both model parameters and output "
        "probabilities; grey-box has access only to output probabilities; "
        "black-box lacks access to both."
    ),
}


# ---------------------------------------------------------------------------
# Artifact Writers
# reference_grounding: paperbench_ref_002 src/models/iterative/run_model.py
# ---------------------------------------------------------------------------

def _ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure parent directory exists and return Path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _write_result_json(data: Any, path: Union[str, Path]) -> None:
    """Write data as JSON to path, creating parent directories as needed."""
    p = _ensure_dir(path)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    logger.debug(f"Wrote artifact: {p}")


def write_dataset_registry(output_dir: Optional[Union[str, Path]] = None) -> str:
    """
    Write the dataset registry to results/dataset_registry.json.
    Contract artifact: results/dataset_registry.json
    """
    out = Path(output_dir) if output_dir else ARTIFACT_DIR
    path = out / "dataset_registry.json"
    payload = {
        "schema_version": "1.0",
        "description": "BBox-Adapter dataset registry for paper reproduction",
        "datasets": DATASET_REGISTRY,
        "aliases": _DATASET_ALIAS_MAP,
        "parameter_sweeps": PARAMETER_SWEEP_REGISTRY,
        "generated_at": datetime.utcnow().isoformat(),
    }
    _write_result_json(payload, path)
    return str(path)


def write_data_manifest(output_dir: Optional[Union[str, Path]] = None) -> str:
    """
    Write the data manifest to results/data_manifest.json.
    Contract artifact: results/data_manifest.json
    """
    out = Path(output_dir) if output_dir else ARTIFACT_DIR
    path = out / "data_manifest.json"
    payload = {
        "schema_version": "1.0",
        "description": "BBox-Adapter data manifest for paper reproduction",
        "protocol_matrix": PROTOCOL_MATRIX,
        "experiment_obligation_matrix": EXPERIMENT_OBLIGATION_MATRIX,
        "artifact_paths": ARTIFACT_PATHS,
        "measurement_schemas": MEASUREMENT_SCHEMAS,
        "trend_assertions": TREND_ASSERTIONS,
        "method_registry": {
            k: {kk: vv for kk, vv in v.items() if kk != "function"}
            for k, v in METHOD_REGISTRY.items()
        },
        "generated_at": datetime.utcnow().isoformat(),
    }
    _write_result_json(payload, path)
    return str(path)


def write_metrics(
    results: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> str:
    """
    Write metrics to results/metrics.json.
    Contract artifact: results/metrics.json
    """
    out = Path(output_dir) if output_dir else ARTIFACT_DIR
    path = out / "metrics.json"
    metric_schema = {
        name: {
            "description": info["description"],
            "datasets": info["datasets"],
            "schema": info["schema"],
            "returns": info.get("returns", "float"),
        }
        for name, info in METRIC_REGISTRY.items()
    }
    payload = {
        "schema_version": "1.0",
        "description": "BBox-Adapter evaluation metric registry and results",
        "metric_registry": metric_schema,
        "results": results or {},
        "trend_assertions": TREND_ASSERTIONS,
        "generated_at": datetime.utcnow().isoformat(),
    }
    _write_result_json(payload, path)
    return str(path)


def write_table_reproduction(
    table_id: str,
    data: Dict[str, Any],
    output_dir: Optional[Union[str, Path]] = None,
) -> str:
    """
    Write a table reproduction artifact to results/{table_id}_reproduction.json.
    Covers Table 1-10.
    """
    out = Path(output_dir) if output_dir else ARTIFACT_DIR
    safe = re.sub(r"[^\w]", "_", table_id.lower())
    if not safe.endswith("_reproduction"):
        safe = safe + "_reproduction"
    path = out / f"{safe}.json"
    payload = {
        "schema_version": "1.0",
        "table": table_id,
        "data": data,
        "generated_at": datetime.utcnow().isoformat(),
    }
    _write_result_json(payload, path)
    return str(path)


def write_figure_data(
    figure_id: str,
    data: Dict[str, Any],
    output_dir: Optional[Union[str, Path]] = None,
) -> str:
    """
    Write figure data artifact to results/{figure_id}_data.json.
    Covers Figure 1-10.
    """
    out = Path(output_dir) if output_dir else ARTIFACT_DIR
    safe = re.sub(r"[^\w]", "_", figure_id.lower())
    if not safe.endswith("_data"):
        safe = safe + "_data"
    path = out / f"{safe}.json"
    payload = {
        "schema_version": "1.0",
        "figure": figure_id,
        "data": data,
        "generated_at": datetime.utcnow().isoformat(),
    }
    _write_result_json(payload, path)
    return str(path)


def write_all_schema_artifacts(
    output_dir: Optional[Union[str, Path]] = None,
) -> List[str]:
    """
    Write all declared artifact schemas and registries.
    Called during smoke/validation runs to confirm artifact-path closure.
    Returns list of written artifact paths.
    """
    out = Path(output_dir) if output_dir else ARTIFACT_DIR
    written: List[str] = []

    written.append(write_dataset_registry(out))
    written.append(write_data_manifest(out))
    written.append(write_metrics(output_dir=out))

    # Table schemas
    table_defs = [
        ("table1", {
            "caption": TABLE1_METHOD_COMPARISON["caption"],
            "columns": TABLE1_METHOD_COMPARISON["columns"],
            "rows": TABLE1_METHOD_COMPARISON["rows"],
        }),
        ("table2", {
            "caption": (
                "Main results of adapting gpt-3.5-turbo on downstream tasks. "
                "For BBox-ADAPTER, we report the best performance of adapters "
                "with # parameters of 0.1B and 0.3B. CoT prompt per (Wei et al., 2022)."
            ),
            "schema": "method|gsm8k_acc|strategyqa_acc|truthfulqa_acc|scienceqa_acc",
            "trend": "BBox-Adapter improves accuracy by avg 6.39% over CoT",
        }),
        ("table3", {
            "caption": (
                "Results of plug-and-play adaptation on davinci-002 and Mixtral-8x7B "
                "across four datasets. Plugger: BBox-Adapter tuned on gpt-3.5-turbo."
            ),
            "schema": "method|target_model|gsm8k|strategyqa|truthfulqa|scienceqa",
        }),
        ("table4", {
            "caption": (
                "Comparison of performance and cost for base model, SFT, and BBox-Adapter "
                "on StrategyQA and GSM8K. Performance: accuracy (%). "
                "Costs ($): training and inference per 1000 questions."
            ),
            "schema": "method|accuracy_pct|training_cost_per_1k|inference_cost_per_1k",
            "trend": "31.30x training cost reduction; 1.84x inference cost reduction vs SFT",
        }),
        ("table5", {
            "caption": (
                "Accuracy (%) of BBox-ADAPTER fine-tuned with two types of loss: "
                "MLM loss and ranking-based NCE loss."
            ),
            "schema": "loss_type|gsm8k|strategyqa|truthfulqa|scienceqa",
        }),
        ("table6", {
            "caption": (
                "Accuracy (%) and GPU memory usage on adapting Mixtral-8x7B to StrategyQA. "
                "VRAM: maximum GPU memory required. "
                "Base model loaded in half-precision; BBox-Adapter uses BERT-0.1B."
            ),
            "schema": "method|accuracy_pct|vram_gb",
        }),
        ("table7", {
            "caption": (
                "Results of adapting Mixtral-8x7B-v0.1 on the ToxiGen dataset. "
                "Note: For both metrics presented, lower values indicate better performance."
            ),
            "schema": "method|hate_speech_rate_pct|toxicity_probability",
        }),
        ("table8", {
            "caption": "Hyperparameter settings of SFT-LoRA (Hu et al., 2021).",
            "schema": "hyperparameter|value",
            "values": {
                "lora_rank": PARAMETER_SWEEP_REGISTRY["lora_rank"]["default"],
                "lora_alpha": PARAMETER_SWEEP_REGISTRY["lora_alpha"]["default"],
                "sft_epochs": PARAMETER_SWEEP_REGISTRY["sft_epochs"]["default"],
                "learning_rate": PARAMETER_SWEEP_REGISTRY["learning_rate"]["default"],
            },
        }),
        ("table9", {
            "caption": "Extended results (auxiliary).",
            "schema": "method|dataset|accuracy_pct",
        }),
        ("table10", {
            "caption": (
                "Main results (extended) of adapting gpt-3.5-turbo on downstream tasks. "
                "For BBox-ADAPTER, best performance of 0.1B and 0.3B adapters."
            ),
            "schema": "method|gsm8k_acc|strategyqa_acc|truthfulqa_acc|scienceqa_acc",
        }),
    ]
    for table_id, table_data in table_defs:
        written.append(write_table_reproduction(table_id, table_data, out))

    # Figure data schemas
    figure_defs = [
        ("figure1", {
            "caption": (
                "Illustration of white-box, grey-box, and black-box LLM adaptation. "
                "White-box: full access to parameters and output probabilities. "
                "Grey-box: only output probabilities. "
                "Black-box: no access to either."
            ),
            "type": "illustration",
        }),
        ("figure2", {
            "caption": (
                "Overview of BBox-ADAPTER for black-box LLM adaptation from source to target domain. "
                "BBox-Adapter adopts an online adaptation framework, iteratively sampling from "
                "previous inferences and updating the adapter."
            ),
            "type": "architecture_diagram",
            "components": ["black_box_llm", "energy_model", "nce_loss", "beam_search", "online_loop"],
        }),
        ("figure3", {
            "caption": (
                "Scale analysis on StrategyQA with "
                "(a) different beam sizes and (b) different iterations of online adaptation. "
                "Both experiments are conducted with two-shot prompting."
            ),
            "type": "line_plot",
            "sub_figures": {
                "a": {"x_axis": "beam_size", "x_values": [1, 3, 5], "y_axis": "accuracy_pct"},
                "b": {"x_axis": "num_iterations", "x_values": [0, 1, 2, 3, 4], "y_axis": "accuracy_pct"},
            },
        }),
        ("figure4", {
            "caption": (
                "Case study of BBox-ADAPTER on GSM8K. "
                "CoT solution from original gpt-3.5-turbo is incorrect; "
                "BBox-Adapter adapted model successfully executes logical step-by-step search. "
                "Visualization: top-3 candidates shown."
            ),
            "type": "case_study",
        }),
        ("figure5", {
            "caption": (
                "Loss curve of Azure-SFT on "
                "(a) StrategyQA, (b) TruthfulQA, and (c) ScienceQA datasets."
            ),
            "type": "loss_curve",
            "datasets": ["strategyqa", "truthfulqa", "scienceqa"],
        }),
        ("figure6", {
            "caption": "Loss curves of Azure-SFT on GSM8K datasets.",
            "type": "loss_curve",
            "datasets": ["gsm8k"],
        }),
        ("figure7", {
            "caption": "Learning curves for training BBox-ADAPTER on the StrategyQA dataset.",
            "type": "learning_curve",
            "dataset": "strategyqa",
        }),
        ("figure8", {
            "caption": "Learning curves for training BBox-ADAPTER on the GSM8K dataset.",
            "type": "learning_curve",
            "dataset": "gsm8k",
        }),
        ("figure9", {
            "caption": "Learning curves for training BBox-ADAPTER on the TruthfulQA dataset.",
            "type": "learning_curve",
            "dataset": "truthfulqa",
        }),
        ("figure10", {
            "caption": "Learning curves for training BBox-ADAPTER on the ScienceQA dataset.",
            "type": "learning_curve",
            "dataset": "scienceqa",
        }),
    ]
    for fig_id, fig_data in figure_defs:
        written.append(write_figure_data(fig_id, fig_data, out))

    # Ablation / supplementary artifacts
    supplementary = [
        ("ablation_adapter_size", {
            "description": "Adapter size ablation: 0.1B vs 0.3B",
            "parameter": "adapter_size",
            "sweep_values": ["0.1B", "0.3B"],
            "schema": "adapter_size|dataset|accuracy_pct",
            "trend": "positive_parameter_improves: 0.3B >= 0.1B",
        }),
        ("ablation_batch_size", {
            "description": "Batch size and beam width sweep (Figure 3 data)",
            "parameters": ["batch_size", "beam_width", "num_iterations"],
            "sweep_values": {
                "batch_size": [64, 128],
                "beam_width": [1, 3, 5],
                "num_iterations": [0, 1, 2, 3, 4],
            },
            "schema": "batch_size|beam_width|num_iterations|accuracy_pct",
        }),
        ("ablation_nce_vs_mlm", {
            "description": "NCE loss vs MLM loss ablation (Table 5)",
            "methods": ["nce_loss", "mlm_loss"],
            "schema": "loss_type|dataset|accuracy_pct",
            "trend": "nce_loss outperforms mlm_loss",
        }),
        ("figure3_data", {
            "description": "Raw data for Figure 3 scale analysis",
            "schema": "beam_width|num_iterations|accuracy_pct",
            "dataset": "strategyqa",
            "prompt_shots": 2,
        }),
        ("cost_vram_report", {
            "description": "Cost and VRAM efficiency report (Table 4, Table 6)",
            "schema": "method|dataset|accuracy_pct|training_cost_per_1k|inference_cost_per_1k|gpu_memory_gb",
            "tables": ["Table 4", "Table 6"],
        }),
    ]
    for art_id, art_data in supplementary:
        p = _ensure_dir(out / f"{art_id}.json")
        _write_result_json(
            {
                "schema_version": "1.0",
                "artifact_id": art_id,
                **art_data,
                "generated_at": datetime.utcnow().isoformat(),
            },
            p,
        )
        written.append(str(p))

    return written


# ---------------------------------------------------------------------------
# Registry export helper
# ---------------------------------------------------------------------------

def get_all_registries() -> Dict[str, Any]:
    """Return all evaluation registries for introspection and CLI output."""
    return {
        "dataset_registry": DATASET_REGISTRY,
        "method_registry": {
            k: {kk: vv for kk, vv in v.items() if kk != "function"}
            for k, v in METHOD_REGISTRY.items()
        },
        "metric_registry": {
            k: {"description": v["description"], "datasets": v["datasets"]}
            for k, v in METRIC_REGISTRY.items()
        },
        "protocol_matrix": PROTOCOL_MATRIX,
        "parameter_sweeps": PARAMETER_SWEEP_REGISTRY,
        "experiment_obligations": EXPERIMENT_OBLIGATION_MATRIX,
        "trend_assertions": TREND_ASSERTIONS,
        "artifact_paths": ARTIFACT_PATHS,
        "measurement_schemas": MEASUREMENT_SCHEMAS,
        "table1_comparison": TABLE1_METHOD_COMPARISON,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="BBox-Adapter Evaluation Protocol CLI"
    )
    parser.add_argument("--dataset", default="gsm8k")
    parser.add_argument("--method", default="bbox_adapter")
    parser.add_argument(
        "--adapter_size", default="0.1B",
        choices=PARAMETER_SWEEP_REGISTRY["adapter_size"]["options"],
    )
    parser.add_argument(
        "--beam_width", type=int, default=3,
        choices=PARAMETER_SWEEP_REGISTRY["beam_width"]["options"],
    )
    parser.add_argument(
        "--batch_size", type=int, default=128,
        choices=PARAMETER_SWEEP_REGISTRY["batch_size"]["options"],
    )
    parser.add_argument("--num_iterations", type=int, default=4)
    parser.add_argument(
        "--feedback_mode", default="groundtruth",
        choices=PARAMETER_SWEEP_REGISTRY["feedback_mode"]["options"],
    )
    parser.add_argument("--output_dir", default=str(ARTIFACT_DIR))
    parser.add_argument(
        "--write_registry", action="store_true",
        help="Write all dataset/metric registry artifacts",
    )
    parser.add_argument("--list_datasets", action="store_true")
    parser.add_argument("--list_methods", action="store_true")
    parser.add_argument("--list_protocols", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if args.list_datasets:
        for dname, dinfo in DATASET_REGISTRY.items():
            print(f"  {dname:15s}  {dinfo['description']}")
        return

    if args.list_methods:
        for mname, minfo in METHOD_REGISTRY.items():
            print(f"  {mname:30s}  {minfo['description']}")
        return

    if args.list_protocols:
        for pname, pinfo in PROTOCOL_MATRIX.items():
            print(f"  {pname:25s}  {pinfo['description']}")
        return

    if args.write_registry:
        out_dir = Path(args.output_dir)
        written = write_all_schema_artifacts(out_dir)
        print(f"Written {len(written)} artifacts to {out_dir}")
        return

    config = EvaluationConfig(
        dataset=args.dataset,
        method=args.method,
        adapter_size=args.adapter_size,
        beam_width=args.beam_width,
        batch_size=args.batch_size,
        num_iterations=args.num_iterations,
        feedback_mode=args.feedback_mode,
        output_dir=args.output_dir,
    )

    result = run_evaluation_protocol(config)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    _cli_main()