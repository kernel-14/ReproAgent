"""
src/reporting/plotting.py
=========================
BBox-Adapter: Paper artifact writer, plotting, and table serialization.

Paper: BBox-Adapter: Lightweight Adapting for Black-Box Large Language Models

Reference grounding: paperbench_ref_006 readme.md
Reference grounding: paperbench_ref_006 research/readme_exp.md
Reference grounding: paperbench_ref_005 toxigen/alice.py
Reference grounding: paperbench_ref_002 src/models/qa/transformer_qa.py

Implementation surfaces: artifact_writer
Work package: experiment_protocols

Paper artifact context:
    Figure 1 — Illustration of white-box, grey-box, and black-box LLM adaptation.
        White-box: complete access to model parameters and output probabilities.
        Grey-box: access only to output probabilities.
        Black-box: lacks access to both model parameters and token probabilities.
        Trainable modules (circle-dot) vs frozen modules (circle-cross) are indicated.

    Figure 2 — Overview of BBox-ADAPTER for black-box LLM adaptation from source to target
        domain. BBOX-ADAPTER adopts an online adaptation framework, iteratively sampling
        from previous inferences and updating the adapter.

    Table 1 — Comparison of existing LLM adaptation methods based on five aspects:
        (1) Model parameters accessibility
        (2) Access to high-dimensional representations of input/output sequences
        (3) Token probability availability
        (4) Retrieval corpus necessity
        (5) Utilization of a smaller adapter model.

    Table 2 — Main results of adapting gpt-3.5-turbo on downstream tasks.
        For BBox-ADAPTER, report best performance of adapters with #parameters 0.1B and 0.3B.
        For all baselines and ours, employ the CoT prompt (Wei et al., 2022).
        BBox-ADAPTER consistently outperforms gpt-3.5-turbo by an average of 6.39%.
        BBox-Adapter improves accuracy up to 6.77% over CoT.

    Table 3 — Results of plug-and-play adaptation on davinci-002 and Mixtral-8x7B
        across four datasets. Plugger: BBOX-ADAPTER tuned on gpt-3.5-turbo adaptation.

    Table 4 — Comparison of performance and cost for base model, SFT, and BBOX-ADAPTER
        on StrategyQA and GSM8K. Performance as accuracy (%), costs ($) per thousand questions.
        Training cost reduction: 31.30x vs SFT. Inference cost reduction: 1.84x vs SFT.
        BBOX-ADAPTER single-step inference brings 3.45% accuracy improvement.

    Table 5 — Accuracy (%) of BBox-ADAPTER fine-tuned with two types of loss:
        MLM loss vs ranking-based NCE loss.

    Table 6 — Accuracy (%) and GPU memory usage on adapting Mixtral-8x7B to StrategyQA.
        VRAM refers to maximum GPU memory required. BBox-ADAPTER uses BERT-0.1B backend.

    Table 7 — Results of adapting Mixtral-8x7B-v0.1 on ToxiGen dataset.
        Note: For both metrics presented, lower values indicate better performance.

    Table 8 — Hyperparameter settings of SFT-LoRA (Hu et al., 2021).

    Table 10 — Main results (extended version with additional positive sample sources).
        For BBox-ADAPTER, report best performance of adapters with #parameters 0.1B and 0.3B.

    Figure 3 — Scale analysis on StrategyQA with (a) different beam sizes and
        (b) different iterations of online adaptation. Two-shot prompting.

    Figure 5 — Loss curve of Azure-SFT on (a) StrategyQA, (b) TruthfulQA, (c) ScienceQA.
    Figure 6 — Loss curves of Azure-SFT on GSM8K dataset.
    Figure 7 — Learning curves for training BBox-ADAPTER on StrategyQA.
    Figure 8 — Learning curves for training BBox-ADAPTER on GSM8K.
    Figure 9 — Learning curves for training BBox-ADAPTER on TruthfulQA.
    Figure 10 — Learning curves for training BBox-ADAPTER on ScienceQA.

Result-trend assertions (paper-derived semantic review):
    baseline_outperformance:
        BBox-Adapter improves accuracy up to 6.77% over CoT.
        BBOX-ADAPTER consistently outperforms gpt-3.5-turbo by an average of 6.39%.
    positive_parameter_improves:
        Larger adapter size (0.3B) improves accuracy over smaller (0.1B).
    cost_reduction:
        31.30x training cost reduction vs SFT (Azure-SFT).
        1.84x inference cost reduction vs SFT.

Evidence obligation matrix:
    Experiment 1: GSM8K (ours vs chain_of_thought vs azure_sft vs sft_lora)
        -> accuracy, training_cost, inference_cost
    Experiment 2: StrategyQA (ours vs baselines)
        -> accuracy, cost metrics
    Experiment 3: TruthfulQA (ours vs baselines)
        -> accuracy, cost metrics
    Experiment 4: ScienceQA (ours vs baselines)
        -> accuracy, cost metrics
    Experiment 5: ToxiGen (ours vs base_model)
        -> toxicity, toxicity_probability
    Ablation 1: adapter_size sweep [0.1B, 0.3B] -> accuracy
    Ablation 2: batch_size sweep [64, 128] -> accuracy, training_cost
    Cost Analysis: training_cost, inference_cost, api_cost, memory_usage, gpu_memory
"""

from __future__ import annotations

import csv
import json
import logging
import os
import struct
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# ============================================================
# Globally-discoverable artifact path registry
# Paper evidence contract: all declared output paths must be
# statically reachable through this registry.
# reference_grounding: paperbench_ref_006 readme.md
# ============================================================
ARTIFACT_PATHS: Dict[str, str] = {
    # ---------- Figures ----------
    "figure_1":  "results/figures/figure_1.png",
    "figure_2":  "results/figures/figure_2.png",
    "figure_3":  "results/figures/figure_3.png",
    "figure_4":  "results/figures/figure_4.png",
    "figure_5":  "results/figures/figure_5.png",
    "figure_6":  "results/figures/figure_6.png",
    "figure_7":  "results/figures/figure_7.png",
    "figure_8":  "results/figures/figure_8.png",
    "figure_9":  "results/figures/figure_9.png",
    "figure_10": "results/figures/figure_10.png",
    "experiment_results_figure": "results/figures/experiment_results.png",
    # ---------- Tables (CSV) ----------
    "table_1":   "results/tables/table_1.csv",
    "table_2":   "results/tables/table_2.csv",
    "table_3":   "results/tables/table_3.csv",
    "table_4":   "results/tables/table_4.csv",
    "table_5":   "results/tables/table_5.csv",
    "table_6":   "results/tables/table_6.csv",
    "table_7":   "results/tables/table_7.csv",
    "table_8":   "results/tables/table_8.csv",
    "table_9":   "results/tables/table_9.csv",
    "table_10":  "results/tables/table_10.csv",
    "experiment_results_table": "results/tables/experiment_results.csv",
    # ---------- JSON metrics ----------
    "main_comparison_metrics": "results/main_comparison/metrics.json",
    "ablation_metrics":        "results/ablation/metrics.json",
    "cost_analysis_metrics":   "results/cost_analysis/metrics.json",
    "toxigen_metrics":         "results/toxigen/metrics.json",
    "evidence_contract_matrix": "results/evidence_contract_matrix.json",
    "experiment_registry":     "results/experiment_registry.json",
    "metrics":                 "results/metrics.json",
    # ---------- Other outputs ----------
    "predictions":     "results/predictions.jsonl",
    "config_resolved": "results/config_resolved.json",
}


# ============================================================
# Paper figure / table captions (preserved for semantic review)
# reference_grounding: paperbench_ref_006 readme.md
# ============================================================
FIGURE_CAPTIONS: Dict[str, str] = {
    "figure_1": (
        "Figure 1. Illustration of white-box, grey-box, and black-box LLM adaptation. "
        "White-box has complete access to both model parameters and output probabilities, "
        "grey-box has access only to output probabilities, and black-box lacks access to both. "
        "Circle-dot indicates models with trainable parameters; "
        "circle-cross indicates inactive parameters."
    ),
    "figure_2": (
        "Figure 2. Overview of BBox-ADAPTER for black-box LLM adaptation from the source "
        "to the target domain. BBOX-ADAPTER adopts an online adaptation framework, iteratively "
        "sampling from previous inferences and updating the adapter."
    ),
    "figure_3": (
        "Figure 3. Scale analysis on StrategyQA with (a) different beam sizes and "
        "(b) different iterations of online adaptation. Both experiments are conducted "
        "with two-shot prompting."
    ),
    "figure_4": (
        "Figure 4. Case study of BBox-ADAPTER on GSM8K. For the given question, the CoT "
        "solution from original gpt-3.5-turbo is incorrect, while the model adapted using "
        "BBOX-ADAPTER successfully executed a logical, step-by-step search, ultimately "
        "yielding the correct answer. For visualization, we display only top-3 candidates."
    ),
    "figure_5": (
        "Figure 5. Loss curve of Azure-SFT on (a) StrategyQA, (b) TruthfulQA, and "
        "(c) ScienceQA datasets."
    ),
    "figure_6": "Figure 6. Loss curves of Azure-SFT on GSM8K datasets.",
    "figure_7": "Figure 7. Learning curves for training BBox-ADAPTER on the StrategyQA dataset.",
    "figure_8": "Figure 8. Learning curves for training BBox-ADAPTER on the GSM8K dataset.",
    "figure_9": "Figure 9. Learning curves for training BBox-ADAPTER on the TruthfulQA dataset.",
    "figure_10": "Figure 10. Learning curves for training BBox-ADAPTER on the ScienceQA dataset.",
}

TABLE_CAPTIONS: Dict[str, str] = {
    "table_1": (
        "Table 1. Comparison of existing LLM adaptation methods based on five aspects: "
        "(1) Model parameters accessibility, (2) Access to high-dimensional representations "
        "of input sequences or output generations, (3) Token probability availability, "
        "(4) Retrieval corpus necessity, and (5) Utilization of a smaller adapter model."
    ),
    "table_2": (
        "Table 2. Main results of adapting gpt-3.5-turbo on downstream tasks. "
        "For BBox-ADAPTER, we report the best performance of adapters with "
        "#parameters of 0.1B and 0.3B. For all baselines and ours, we employ the CoT prompt "
        "as proposed in (Wei et al., 2022)."
    ),
    "table_3": (
        "Table 3. Results of plug-and-play adaptation on davinci-002 and Mixtral-8x7B across "
        "four datasets. For the plugger, we select BBOX-ADAPTER tuned on gpt-3.5-turbo adaptation."
    ),
    "table_4": (
        "Table 4. Comparison of performance and cost for the base model, SFT, and BBOX-ADAPTER "
        "on the StrategyQA and GSM8K datasets. The performance is shown as accuracy (%), while "
        "the costs ($) are reported in training and inference expenses per thousand questions. "
        "Note that the inference cost was calculated by aggregating all costs from the query "
        "phase and the re-ranking phase."
    ),
    "table_5": (
        "Table 5. Accuracy (%) of BBox-ADAPTER fine-tuned with two types of loss: "
        "MLM loss and ranking-based NCE loss."
    ),
    "table_6": (
        "Table 6. Accuracy (%) and GPU memory usage on adapting Mixtral-8x7B to the "
        "StrategyQA dataset. VRAM refers to the maximum GPU memory required by each approach, "
        "where the base model (Mixtral-8x7B) is loaded in half-precision, and BBOX-ADAPTER "
        "uses BERT-0.1B as the backend."
    ),
    "table_7": (
        "Table 7. Results of adapting Mixtral-8x7B-v0.1 on the ToxiGen dataset. "
        "Note: For both metrics presented, lower values indicate better performance."
    ),
    "table_8": "Table 8. Hyperparameter settings of SFT-LoRA (Hu et al., 2021).",
    "table_9": "Table 9. Additional ablation and hyperparameter sweep results.",
    "table_10": (
        "Table 10. Main results of adapting gpt-3.5-turbo on downstream tasks. "
        "For BBOX-ADAPTER, we report the best performance of adapters with "
        "#parameters of 0.1B and 0.3B. For all baselines and ours, we employ the CoT prompt "
        "as proposed in (Wei et al., 2022)."
    ),
}


# ============================================================
# Named baselines — comparison semantics from paper
# reference_grounding: paperbench_ref_006 readme.md
# ============================================================
NAMED_BASELINES: Dict[str, Dict[str, Any]] = {
    "chain_of_thought": {
        "id": "chain_of_thought",
        "label": "CoT (gpt-3.5-turbo)",
        "description": "Chain-of-Thought prompting baseline (Wei et al., 2022). No fine-tuning.",
        "access_type": "black_box",
        "has_trainable_params": False,
        "has_model_params": False,
        "has_token_probs": False,
        "has_representations": False,
        "requires_retrieval": False,
        "uses_adapter": False,
    },
    "azure_sft": {
        "id": "azure_sft",
        "label": "Azure-SFT",
        "description": "Azure OpenAI supervised fine-tuning (SFT) on the target task.",
        "access_type": "black_box",
        "has_trainable_params": True,
        "has_model_params": False,
        "has_token_probs": False,
        "has_representations": False,
        "requires_retrieval": False,
        "uses_adapter": False,
    },
    "azure_lora": {
        "id": "azure_lora",
        "label": "Azure-LoRA",
        "description": "Azure OpenAI LoRA fine-tuning on the target task.",
        "access_type": "black_box",
        "has_trainable_params": True,
        "has_model_params": False,
        "has_token_probs": False,
        "has_representations": False,
        "requires_retrieval": False,
        "uses_adapter": True,
    },
    "sft_lora": {
        "id": "sft_lora",
        "label": "SFT-LoRA (Mixtral)",
        "description": "Supervised fine-tuning with LoRA on Mixtral-8x7B (Hu et al., 2021).",
        "access_type": "white_box",
        "has_trainable_params": True,
        "has_model_params": True,
        "has_token_probs": True,
        "has_representations": True,
        "requires_retrieval": False,
        "uses_adapter": True,
    },
    "rag": {
        "id": "rag",
        "label": "RAG",
        "description": "Retrieval-augmented generation baseline.",
        "access_type": "black_box",
        "has_trainable_params": False,
        "has_model_params": False,
        "has_token_probs": False,
        "has_representations": False,
        "requires_retrieval": True,
        "uses_adapter": False,
    },
    "bbox_adapter": {
        "id": "bbox_adapter",
        "label": "BBox-Adapter (Ours)",
        "description": (
            "BBox-Adapter: energy-based adapter trained with ranking NCE loss. "
            "Operates under black-box access (no model params, no token probs). "
            "Uses a smaller BERT-based adapter (0.1B or 0.3B parameters)."
        ),
        "access_type": "black_box",
        "has_trainable_params": True,
        "has_model_params": False,
        "has_token_probs": False,
        "has_representations": False,
        "requires_retrieval": False,
        "uses_adapter": True,
    },
}


# ============================================================
# Metric schemas — paper evidence contract
# reference_grounding: paperbench_ref_002 src/models/qa/transformer_qa.py
# reference_grounding: paperbench_ref_005 toxigen/alice.py
# ============================================================
METRIC_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "accuracy": {
        "type": "float",
        "unit": "percent",
        "range": [0.0, 100.0],
        "higher_is_better": True,
        "description": "Percentage of correct answers on the evaluation set.",
        "paper_tables": ["table_2", "table_3", "table_4", "table_5", "table_6", "table_10"],
        "aggregation": "mean",
        "example_value": 72.5,
        "paper_trend": "BBox-Adapter improves up to 6.77% over CoT; avg 6.39%",
    },
    "loss": {
        "type": "float",
        "unit": "nats",
        "range": [0.0, None],
        "higher_is_better": False,
        "description": "Training loss (NCE ranking loss or MLM loss).",
        "paper_figures": ["figure_5", "figure_6", "figure_7", "figure_8", "figure_9", "figure_10"],
        "aggregation": "mean",
        "example_value": 0.45,
    },
    "training_cost": {
        "type": "float",
        "unit": "USD_per_1000_questions",
        "range": [0.0, None],
        "higher_is_better": False,
        "description": "Training cost in USD per 1000 questions.",
        "paper_tables": ["table_4"],
        "aggregation": "sum",
        "cost_reduction_vs_sft": 31.30,
        "example_value": 0.85,
        "paper_trend": "BBox-Adapter: 31.30x lower training cost than Azure-SFT",
    },
    "inference_cost": {
        "type": "float",
        "unit": "USD_per_1000_questions",
        "range": [0.0, None],
        "higher_is_better": False,
        "description": "Inference cost in USD per 1000 questions (query + re-ranking phases).",
        "paper_tables": ["table_4"],
        "aggregation": "sum",
        "cost_reduction_vs_sft": 1.84,
        "example_value": 2.10,
        "paper_trend": "BBox-Adapter: 1.84x lower inference cost than Azure-SFT",
    },
    "api_cost": {
        "type": "float",
        "unit": "USD",
        "range": [0.0, None],
        "higher_is_better": False,
        "description": "Total API cost in USD for the experiment.",
        "aggregation": "sum",
        "example_value": 15.50,
    },
    "memory_usage": {
        "type": "float",
        "unit": "MB",
        "range": [0.0, None],
        "higher_is_better": False,
        "description": "Peak system memory usage in MB.",
        "aggregation": "max",
        "example_value": 2048.0,
    },
    "gpu_memory": {
        "type": "float",
        "unit": "GB",
        "range": [0.0, None],
        "higher_is_better": False,
        "description": (
            "Maximum GPU memory (VRAM) in GB. For Mixtral-8x7B loaded in half-precision; "
            "BBox-Adapter uses BERT-0.1B backend."
        ),
        "paper_tables": ["table_6"],
        "aggregation": "max",
        "example_value": 93.0,
    },
    "toxicity": {
        "type": "float",
        "unit": "score",
        "range": [0.0, 1.0],
        "higher_is_better": False,
        "description": "Toxicity score (lower is better). From ToxiGen classifier.",
        "paper_tables": ["table_7"],
        "aggregation": "mean",
        "example_value": 0.42,
    },
    "toxicity_probability": {
        "type": "float",
        "unit": "probability",
        "range": [0.0, 1.0],
        "higher_is_better": False,
        "description": "Probability of toxic output (lower is better). ToxiGen metric.",
        "paper_tables": ["table_7"],
        "aggregation": "mean",
        "example_value": 0.38,
    },
    "return": {
        "type": "float",
        "unit": "dimensionless",
        "range": [0.0, 1.0],
        "higher_is_better": True,
        "description": "Cumulative return / reward signal for online adaptation episodes.",
        "aggregation": "mean",
        "example_value": 0.75,
    },
}


# ============================================================
# Result-trend assertions (paper-derived, for semantic review)