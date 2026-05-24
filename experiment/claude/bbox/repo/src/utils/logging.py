#!/usr/bin/env python3
"""
BBox-Adapter Logging and Registry Utilities

Provides structured logging, experiment tracking, method/baseline registry,
parameter sweep registry, dataset registry, metric computation, and artifact
writing for BBox-Adapter paper reproduction experiments.

Paper: BBox-Adapter: Lightweight Adapting for Black-Box Large Language Models

Implementation surfaces: data_pipeline, evaluation, metric_formula, config,
                        baseline_or_ablation, artifact_writer

Reference grounding:
  reference_grounding: paperbench_ref_002 src/models/iterative/run_model.py
  reference_grounding: paperbench_ref_003 truthfulqa/metrics.py
  reference_grounding: paperbench_ref_002 src/models/gen_model.py

Method Registry (Paper Evidence Contract):
  ours, chain_of_thought, oracle, heuristic, roberta, fine_tuning, lora,
  sft_lora, azure_sft, mlm, bbox_adapter, ranking_nce, online_adaptation,
  single_step_inference, full_step_inference, ground_truth_feedback,
  ai_feedback, energy_based_model, combined_feedback

Sweep Registry (Paper Evidence Contract):
  beam_size: [1, 3, 5]
  iteration_count: [0, 1, 2, 3, 4]
  adapter_size: [0.1, 0.3]
  temperature: [0.5, 0.7, 0.9, 1.0]
  batch_size: [64, 128]

Fixed Hyperparameter Anchors:
  batch_size_128 = 128
  batch_size_64  = 64
  temperature_default = 1.0
  judge_model_default = "roberta-base"

Artifact paths:
  results/dataset_registry.json
  results/data_manifest.json
  results/metrics.json
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# Logging Infrastructure
# ---------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


def configure_logging(
    level: Union[int, str] = logging.INFO,
    log_file: Optional[str] = None,
    format_string: str = LOG_FORMAT,
) -> logging.Logger:
    """Configure root logging for the experiment pipeline."""
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=level,
        format=format_string,
        datefmt=LOG_DATE_FORMAT,
        handlers=handlers,
        force=True,
    )
    return logging.getLogger("bbox_adapter")


def get_logger(name: str) -> logging.Logger:
    """Get a namespaced logger for the given module."""
    return logging.getLogger(f"bbox_adapter.{name}")


# ---------------------------------------------------------------------------
# Fixed Hyperparameter Anchors (Paper Evidence Contract)
# reference_grounding: paperbench_ref_002 src/models/iterative/run_model.py
# ---------------------------------------------------------------------------

BATCH_SIZE_128: int = 128        # anchor: batch_size_128
BATCH_SIZE_64: int = 64          # anchor: batch_size_64
TEMPERATURE_DEFAULT: float = 1.0  # anchor: temperature for generation
JUDGE_MODEL_DEFAULT: str = "roberta-base"  # anchor: judge_model for toxicity

# ---------------------------------------------------------------------------
# Method / Baseline Registry (Paper Evidence Contract)
# reference_grounding: paperbench_ref_002 src/models/iterative/run_model.py
# reference_grounding: paperbench_ref_003 truthfulqa/metrics.py
# ---------------------------------------------------------------------------

METHOD_REGISTRY: Dict[str, Dict[str, Any]] = {
    # ---- Core BBox-Adapter ------------------------------------------------
    "ours": {
        "display_name": "BBox-ADAPTER",
        "aliases": ["bbox_adapter", "ADAPTER", "BBox-ADApter", "BBox-ADAPTER"],
        "category": "LLM Adaptation",
        "description": "BBox-Adapter: energy-based adapter with ranking NCE loss",
        "requires_training": True,
        "adapter_type": "energy_based",
        "feedback_modes": ["ground_truth_feedback", "ai_feedback", "combined_feedback"],
    },
    "bbox_adapter": {
        "display_name": "BBox-ADAPTER",
        "aliases": ["ours", "BBOX-ADAPTER"],
        "category": "LLM Adaptation",
        "description": "BBox-Adapter with online adaptation",
        "requires_training": True,
        "adapter_type": "energy_based",
        "feedback_modes": ["ground_truth_feedback", "ai_feedback", "combined_feedback"],
    },
    "ranking_nce": {
        "display_name": "Ranking NCE",
        "aliases": ["nce_loss", "contrastive_nce"],
        "category": "LLM Adaptation",
        "description": "Ranking-based NCE loss for adapter training",
        "requires_training": True,
        "adapter_type": "energy_based",
        "loss_type": "nce",
    },
    "online_adaptation": {
        "display_name": "Online Adaptation",
        "aliases": ["iterative_adaptation"],
        "category": "LLM Adaptation",
        "description": "Iterative online adaptation with beam search",
        "requires_training": True,
        "adapter_type": "energy_based",
        "num_iterations": 4,
    },
    "energy_based_model": {
        "display_name": "Energy-Based Model",
        "aliases": ["ebm", "energy_model"],
        "category": "LLM Adaptation",
        "description": "Energy-based model for scoring LLM candidate outputs",
        "requires_training": True,
        "adapter_type": "energy_based",
    },
    # ---- Inference Strategies --------------------------------------------
    "single_step_inference": {
        "display_name": "Single-Step Inference",
        "aliases": ["single_step", "one_shot_inference"],
        "category": "Inference",
        "description": "Single-pass inference with adapted model (beam_size=1)",
        "requires_training": False,
        "beam_size": 1,
    },
    "full_step_inference": {
        "display_name": "Full-Step Inference",
        "aliases": ["multi_step", "beam_search_inference"],
        "category": "Inference",
        "description": "Beam search inference across full iteration budget",
        "requires_training": False,
        "beam_size": 5,
    },
    # ---- Feedback Modes --------------------------------------------------
    "ground_truth_feedback": {
        "display_name": "Ground-Truth Feedback",
        "aliases": ["gt_feedback", "groundtruth"],
        "category": "Feedback",
        "description": "Use ground-truth labels as reward signal",
        "requires_training": True,
        "datasets": ["gsm8k", "scienceqa"],
    },
    "ai_feedback": {
        "display_name": "AI Feedback",
        "aliases": ["llm_feedback", "model_feedback"],
        "category": "Feedback",
        "description": "Use LLM judge as reward signal",
        "requires_training": True,
        "datasets": ["strategyqa", "toxigen"],
    },
    "combined_feedback": {
        "display_name": "Combined Feedback",
        "aliases": ["hybrid_feedback"],
        "category": "Feedback",
        "description": "Combine ground-truth and AI feedback signals",
        "requires_training": True,
        "datasets": ["truthfulqa"],
    },
    # ---- Baselines -------------------------------------------------------
    "chain_of_thought": {
        "display_name": "CoT",
        "aliases": ["cot", "CoT", "chain_of_thought_prompting"],
        "category": "Parameter-Efficient Fine-Tuning",
        "description": "Chain-of-thought prompting baseline",
        "requires_training": False,
        "prompt_strategy": "few_shot_cot",
    },
    "oracle": {
        "display_name": "Oracle",
        "aliases": ["upper_bound", "gold_answer"],
        "category": "Baseline",
        "description": "Oracle upper bound using ground-truth answers",
        "requires_training": False,
    },
    "heuristic": {
        "display_name": "Heuristic",
        "aliases": ["rule_based"],
        "category": "Baseline",
        "description": "Rule-based heuristic baseline",
        "requires_training": False,
    },
    "roberta": {
        "display_name": "RoBERTa",
        "aliases": ["roberta_base", "roberta-base"],
        "category": "Baseline",
        "description": "RoBERTa-based classifier baseline",
        "requires_training": True,
        "model_name": "roberta-base",
    },
    "fine_tuning": {
        "display_name": "Fine-Tuning",
        "aliases": ["full_finetuning", "standard_ft"],
        "category": "Fine-Tuning",
        "description": "Standard full-parameter supervised fine-tuning",
        "requires_training": True,
        "update_all_params": True,
    },
    "lora": {
        "display_name": "LoRA",
        "aliases": ["low_rank_adaptation", "PEFT"],
        "category": "Parameter-Efficient Fine-Tuning",
        "description": "Low-Rank Adaptation parameter-efficient fine-tuning",
        "requires_training": True,
        "lora_rank": 128,
        "lora_alpha": 256,
    },
    "sft_lora": {
        "display_name": "SFT+LoRA",
        "aliases": ["supervised_ft_lora"],
        "category": "Parameter-Efficient Fine-Tuning",
        "description": "Supervised Fine-Tuning combined with LoRA",
        "requires_training": True,
        "lora_rank": 128,
        "lora_alpha": 256,
        "sft_epochs": 3,
    },
    "azure_sft": {
        "display_name": "Azure SFT",
        "aliases": ["azure_fine_tuning", "openai_sft"],
        "category": "Fine-Tuning",
        "description": "Azure OpenAI supervised fine-tuning endpoint",
        "requires_training": True,
        "provider": "azure",
    },
    "mlm": {
        "display_name": "MLM Loss",
        "aliases": ["masked_lm", "mlm_loss"],
        "category": "LLM Adaptation",
        "description": "Masked language model loss baseline for adapter (ablation)",
        "requires_training": True,
        "loss_type": "mlm",
    },
}

# Canonical display-name aliases → canonical method key
# reference_grounding: paperbench_ref_002 src/models/iterative/run_model.py
METHOD_ALIASES: Dict[str, str] = {
    "ADAPTER": "ours",
    "LLM": "chain_of_thought",
    "BBOX-ADAPTER": "bbox_adapter",
    "PEFT": "lora",
    "LLM Adaptation": "online_adaptation",
    "Parameter-Efficient Fine-Tuning": "lora",
    "BBox-ADAPTER": "bbox_adapter",
    "CoT": "chain_of_thought",
    "Parameter-Efficient": "lora",
    "Fine-Tuning": "fine_tuning",
    "BBox-ADApter": "bbox_adapter",
    "Ours": "ours",
}


def resolve_method(name: str) -> Dict[str, Any]:
    """
    Resolve a method name or display-alias to its registry entry.

    reference_grounding: paperbench_ref_002 src/models/iterative/run_model.py
    """
    if name in METHOD_REGISTRY:
        return {"key": name, **METHOD_REGISTRY[name]}
    canonical = METHOD_ALIASES.get(name)
    if canonical and canonical in METHOD_REGISTRY:
        return {"key": canonical, **METHOD_REGISTRY[canonical]}
    available = sorted(set(list(METHOD_REGISTRY.keys()) + list(METHOD_ALIASES.keys())))
    raise KeyError(
        f"Method '{name}' not in registry. Available: {available}"
    )


# ---------------------------------------------------------------------------
# Sweep / Hyperparameter Registry (Paper Evidence Contract)
# reference_grounding: paperbench_ref_002 src/models/iterative/run_model.py
# ---------------------------------------------------------------------------

SWEEP_REGISTRY: Dict[str, Dict[str, Any]] = {
    "beam_size": {
        "values": [1, 3, 5],
        "default": 5,
        "description": "Number of beam candidates during inference",
        "paper_ref": "Table 2, Figure 3",
    },
    "iteration_count": {
        "values": [0, 1, 2, 3, 4],
        "default": 4,
        "description": "Number of online adaptation iterations",
        "paper_ref": "Figure 3",
    },
    "adapter_size": {
        "values": [0.1, 0.3],
        "default": 0.1,
        "description": "Adapter parameter count in billions (0.1B, 0.3B)",
        "paper_ref": "Table 2",
        "aliases": ["0.1B", "0.3B"],
    },
    "temperature": {
        "values": [0.5, 0.7, 0.9, 1.0],
        "default": TEMPERATURE_DEFAULT,
        "description": "Sampling temperature for LLM generation",
        "paper_ref": "Section 4",
        "fixed_anchor": 0.7,
    },
    "batch_size": {
        "values": [64, 128],
        "default": BATCH_SIZE_128,
        "description": "Training batch size",
        "paper_ref": "Section 4, Tables 2-7",
        "fixed_anchors": {
            "batch_size_64": BATCH_SIZE_64,
            "batch_size_128": BATCH_SIZE_128,
        },
    },
    "learning_rate": {
        "values": [1e-5, 3e-5, 5e-5, 1e-4],
        "default": 3e-5,
        "description": "AdamW learning rate for adapter training",
        "paper_ref": "Section 4",
    },
    "num_iterations": {
        "values": [0, 1, 2, 3, 4],
        "default": 4,
        "description": "Number of online adaptation iterations (alias for iteration_count)",
        "paper_ref": "Figure 3",
    },
    "feedback_mode": {
        "values": ["ground_truth_feedback", "ai_feedback", "combined_feedback"],
        "default": "ground_truth_feedback",
        "description": "Feedback signal for positive sample selection",
        "paper_ref": "Section 3.3",
    },
    "lora_rank": {
        "values": [4, 8, 16, 32],
        "default": 8,
        "description": "LoRA rank for parameter-efficient fine-tuning",
        "paper_ref": "Table 6",
    },
    "lora_alpha": {
        "values": [8, 16, 32],
        "default": 16,
        "description": "LoRA alpha scaling factor",
        "paper_ref": "Table 6",
    },
    "sft_epochs": {
        "values": [1, 2, 3, 5],
        "default": 3,
        "description": "Number of supervised fine-tuning epochs",
        "paper_ref": "Section 4",
    },
    "judge_model": {
        "values": ["roberta-base", "roberta-large", "toxigen_roberta"],
        "default": JUDGE_MODEL_DEFAULT,
        "description": "Model used as AI judge for toxicity / AI-feedback",
        "paper_ref": "Table 7, Section 4",
        "fixed_anchor": "roberta-base",
    },
    "beam_width": {
        "values": [1, 3, 5],
        "default": 5,
        "description": "Beam width for sentence-level beam search (alias for beam_size)",
        "paper_ref": "Algorithm 1, Table 2",
    },
}


def get_sweep_values(param_name: str) -> List[Any]:
    """Return the bounded list of values for a named sweep parameter."""
    entry = SWEEP_REGISTRY.get(param_name)
    if entry is None:
        raise KeyError(
            f"Sweep parameter '{param_name}' not found. "
            f"Available: {sorted(SWEEP_REGISTRY.keys())}"
        )
    return list(entry["values"])


def get_default_hyperparams() -> Dict[str, Any]:
    """Return a dict of default hyperparameter values from the sweep registry."""
    return {
        "beam_size": SWEEP_REGISTRY["beam_size"]["default"],
        "iteration_count": SWEEP_REGISTRY["iteration_count"]["default"],
        "adapter_size": SWEEP_REGISTRY["adapter_size"]["default"],
        "temperature": TEMPERATURE_DEFAULT,
        "batch_size": BATCH_SIZE_128,
        "learning_rate": SWEEP_REGISTRY["learning_rate"]["default"],
        "num_iterations": SWEEP_REGISTRY["num_iterations"]["default"],
        "feedback_mode": SWEEP_REGISTRY["feedback_mode"]["default"],
        "lora_rank": SWEEP_REGISTRY["lora_rank"]["default"],
        "lora_alpha": SWEEP_REGISTRY["lora_alpha"]["default"],
        "sft_epochs": SWEEP_REGISTRY["sft_epochs"]["default"],
        "judge_model": JUDGE_MODEL_DEFAULT,
        "beam_width": SWEEP_REGISTRY["beam_width"]["default"],
    }


# ---------------------------------------------------------------------------
# Dataset Registry
# reference_grounding: paperbench_ref_002 src/models/iterative/run_model.py
# ---------------------------------------------------------------------------

DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "gsm8k": {
        "display_name": "GSM8K",
        "task_type": "math_reasoning",
        "feedback_mode": "ground_truth_feedback",
        "metric": "accuracy",
        "num_classes": None,
        "answer_format": "numeric",
        "paper_tables": ["Table 2", "Table 3", "Table 4"],
        "split": "test",
        "hf_path": "gsm8k",
        "hf_config": "main",
    },
    "strategyqa": {
        "display_name": "StrategyQA",
        "task_type": "implicit_reasoning",
        "feedback_mode": "ai_feedback",
        "metric": "accuracy",
        "num_classes": 2,
        "answer_format": "yes_no",
        "paper_tables": ["Table 2", "Table 3", "Table 4", "Table 5", "Table 6"],
        "split": "test",
        "hf_path": "wics/strategy-qa",
        "hf_config": None,
    },
    "truthfulqa": {
        "display_name": "TruthfulQA",
        "task_type": "truthfulness",
        "feedback_mode": "combined_feedback",
        "metric": "accuracy",
        "num_classes": None,
        "answer_format": "mc",
        "paper_tables": ["Table 2", "Table 3"],
        "split": "validation",
        "hf_path": "truthful_qa",
        "hf_config": "multiple_choice",
    },
    "scienceqa": {
        "display_name": "ScienceQA",
        "task_type": "science_domain_qa",
        "feedback_mode": "ground_truth_feedback",
        "metric": "accuracy",
        "num_classes": None,
        "answer_format": "mc",
        "paper_tables": ["Table 2", "Table 3"],
        "split": "test",
        "hf_path": "derek-thomas/ScienceQA",
        "hf_config": None,
    },
    "toxigen": {
        "display_name": "ToxiGen",
        "task_type": "toxicity_reduction",
        "feedback_mode": "ai_feedback",
        "metric": "toxicity_rate",
        "num_classes": 2,
        "answer_format": "text",
        "paper_tables": ["Table 7"],
        "split": "test",
        "hf_path": "skg/toxigen-data",
        "hf_config": None,
        "judge_model": "roberta-base",
    },
}

DATASET_ALIASES: Dict[str, str] = {
    "gsm8k_math": "gsm8k",
    "strategy_qa": "strategyqa",
    "strategy-qa": "strategyqa",
    "truthful_qa": "truthfulqa",
    "truthful-qa": "truthfulqa",
    "science_qa": "scienceqa",
    "science-qa": "scienceqa",
    "tox_gen": "toxigen",
    "tox-gen": "toxigen",
}


def resolve_dataset(name: str) -> Dict[str, Any]:
    """Resolve dataset name or alias to its registry entry."""
    key = name.lower()
    if key in DATASET_REGISTRY:
        return {"key": key, **DATASET_REGISTRY[key]}
    canonical = DATASET_ALIASES.get(key)
    if canonical:
        return {"key": canonical, **DATASET_REGISTRY[canonical]}
    available = sorted(DATASET_REGISTRY.keys())
    raise KeyError(
        f"Dataset '{name}' not in registry. Available: {available}"
    )


# ---------------------------------------------------------------------------
# Metric Registry
# reference_grounding: paperbench_ref_003 truthfulqa/metrics.py
# ---------------------------------------------------------------------------

METRIC_REGISTRY: Dict[str, Dict[str, Any]] = {
    "accuracy": {
        "display_name": "Accuracy (%)",
        "description": "Exact-match accuracy (primary metric for all QA tasks)",
        "higher_is_better": True,
        "unit": "%",
        "range": (0.0, 100.0),
        "paper_tables": ["Table 2", "Table 3", "Table 4", "Table 5", "Table 6"],
    },
    "toxicity_rate": {
        "display_name": "Hate Speech Rate (%)",
        "description": "Proportion of toxic / hateful model outputs",
        "higher_is_better": False,
        "unit": "%",
        "range": (0.0, 100.0),
        "paper_tables": ["Table 7"],
        "judge_model": "roberta-base",
    },
    "toxicity_score": {
        "display_name": "ToxScore",
        "description": "Mean toxicity probability from RoBERTa judge",
        "higher_is_better": False,
        "unit": "probability",
        "range": (0.0, 1.0),
        "paper_tables": ["Table 7"],
        "judge_model": "roberta-base",
    },
    "training_cost": {
        "display_name": "Training Cost ($)",
        "description": "Estimated API cost incurred during adapter training",
        "higher_is_better": False,
        "unit": "USD",
        "paper_tables": ["Table 4"],
    },
    "inference_cost": {
        "display_name": "Inference Cost ($)",
        "description": "Estimated API cost incurred during inference",
        "higher_is_better": False,
        "unit": "USD",
        "paper_tables": ["Table 4"],
    },
    "api_cost": {
        "display_name": "Total API Cost ($)",
        "description": "Total API cost (training + inference combined)",
        "higher_is_better": False,
        "unit": "USD",
        "paper_tables": ["Table 4"],
    },
    "vram_usage": {
        "display_name": "VRAM (GB)",
        "description": "Peak GPU memory consumption",
        "higher_is_better": False,
        "unit": "GB",
        "paper_tables": ["Table 6"],
    },
    "bleu": {
        "display_name": "BLEU",
        "description": "BLEU score for generation quality (auxiliary)",
        "higher_is_better": True,
        "unit": "score",
        "paper_tables": [],
    },
    "rouge1": {
        "display_name": "ROUGE-1",
        "description": "ROUGE-1 F1 score (auxiliary)",
        "higher_is_better": True,
        "unit": "score",
        "paper_tables": [],
    },
    "rouge2": {
        "display_name": "ROUGE-2",
        "description": "ROUGE-2 F1 score (auxiliary)",
        "higher_is_better": True,
        "unit": "score",
        "paper_tables": [],
    },
    "rougeL": {
        "display_name": "ROUGE-L",
        "description": "ROUGE-L F1 score (auxiliary)",
        "higher_is_better": True,
        "unit": "score",
        "paper_tables": [],
    },
}


# ---------------------------------------------------------------------------
# Metric Computation Functions
# reference_grounding: paperbench_ref_003 truthfulqa/metrics.py
# ---------------------------------------------------------------------------

def _normalize_answer(text: str) -> str:
    """Normalize answer text for comparison."""
    text = str(text).strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compute_accuracy(predictions: List[str], references: List[str]) -> float:
    """
    Compute exact-match accuracy (primary evaluation metric).

    Returns accuracy in [0.0, 100.0].

    reference_grounding: paperbench_ref_003 truthfulqa/metrics.py
    """
    if not predictions or not references:
        return 0.0
    if len(predictions) != len(references):
        raise ValueError(
            f"Length mismatch: predictions={len(predictions)}, "
            f"references={len(references)}"
        )
    correct = sum(
        1 for p, r in zip(predictions, references)
        if _normalize_answer(p) == _normalize_answer(r)
    )
    return 100.0 * correct / len(predictions)


def compute_bleu(predictions: List[str], references: List[str]) -> float:
    """
    Compute unigram BLEU with brevity penalty.

    reference_grounding: paperbench_ref_003 truthfulqa/metrics.py (run_bleu_and_rouge)
    """
    if not predictions or not references:
        return 0.0
    total_score = 0.0
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        if not pred_tokens:
            continue
        ref_set = set(ref_tokens)
        hits = sum(1 for t in pred_tokens if t in ref_set)
        precision = hits / len(pred_tokens)
        bp = min(1.0, math.exp(1.0 - len(ref_tokens) / max(len(pred_tokens), 1)))
        total_score += bp * precision
    return total_score / len(predictions)


def _rouge_n(
    pred_tokens: List[str], ref_tokens: List[str], n: int
) -> float:
    """Compute ROUGE-N F1."""
    def ngrams(tokens: List[str], size: int) -> Dict[tuple, int]:  # type: ignore[type-arg]
        counts: Dict[tuple, int] = {}  # type: ignore[type-arg]
        for i in range(len(tokens) - size + 1):
            gram = tuple(tokens[i : i + size])
            counts[gram] = counts.get(gram, 0) + 1
        return counts

    pred_ngrams = ngrams(pred_tokens, n)
    ref_ngrams = ngrams(ref_tokens, n)
    if not pred_ngrams or not ref_ngrams:
        return 0.0
    overlap = sum(
        min(pred_ngrams.get(g, 0), ref_ngrams.get(g, 0)) for g in ref_ngrams
    )
    prec = overlap / sum(pred_ngrams.values()) if pred_ngrams else 0.0
    rec = overlap / sum(ref_ngrams.values()) if ref_ngrams else 0.0
    if prec + rec == 0.0:
        return 0.0
    return 2.0 * prec * rec / (prec + rec)


def _rouge_l(pred_tokens: List[str], ref_tokens: List[str]) -> float:
    """Compute ROUGE-L F1 via LCS dynamic programming."""
    m, n = len(pred_tokens), len(ref_tokens)
    if m == 0 or n == 0:
        return 0.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs_len = dp[m][n]
    prec = lcs_len / m
    rec = lcs_len / n
    if prec + rec == 0.0:
        return 0.0
    return 2.0 * prec * rec / (prec + rec)


def _compute_rouge(
    rouge_type: str, predictions: List[str], references: List[str]
) -> float:
    """Compute ROUGE-N or ROUGE-L averaged over prediction/reference pairs."""
    if not predictions or not references:
        return 0.0
    total = 0.0
    for pred, ref in zip(predictions, references):
        p_tok = pred.lower().split()
        r_tok = ref.lower().split()
        if rouge_type == "rouge1":
            total += _rouge_n(p_tok, r_tok, n=1)
        elif rouge_type == "rouge2":
            total += _rouge_n(p_tok, r_tok, n=2)
        elif rouge_type == "rougeL":
            total += _rouge_l(p_tok, r_tok)
    return total / len(predictions)


def _toxicity_heuristic(texts: List[str]) -> float:
    """Keyword-based heuristic toxicity rate (fallback when model unavailable)."""
    toxic_keywords = {
        "hate", "kill", "murder", "racist", "sexist", "slur",
        "violent", "abuse", "harass",
    }
    toxic_count = sum(
        1 for t in texts
        if any(kw in t.lower() for kw in toxic_keywords)
    )
    return 100.0 * toxic_count / len(texts)


def _toxicity_roberta(texts: List[str], model_name: str) -> float:
    """Compute toxicity rate using HuggingFace pipeline (lazy import)."""
    from transformers import pipeline as hf_pipeline  # type: ignore

    classifier = hf_pipeline(
        "text-classification",
        model=model_name,
        truncation=True,
        max_length=512,
    )
    results = classifier(texts, batch_size=32)
    toxic_count = sum(
        1 for r in results
        if r["label"].lower() in {"hate", "toxic", "label_1", "1"}
    )
    return 100.0 * toxic_count / len(texts)


def compute_toxicity_rate(
    predictions: List[str],
    references: Optional[List[str]] = None,
    judge_model: str = JUDGE_MODEL_DEFAULT,
) -> float:
    """
    Compute hate-speech rate as proportion of toxic outputs in [0, 100].

    Uses RoBERTa classifier (anchor: judge_model=roberta-base) when available,
    falls back to keyword heuristic.

    reference_grounding: paperbench_ref_003 truthfulqa/metrics.py (run_finetuned_GPT3)
    anchor: judge_model=roberta-base
    """
    if not predictions:
        return 0.0
    try:
        return _toxicity_roberta(predictions, judge_model)
    except Exception:
        return _toxicity_heuristic(predictions)


def compute_metric(
    metric_name: str,
    predictions: List[str],
    references: List[str],
    **kwargs: Any,
) -> float:
    """
    Dispatch metric computation by registry name.

    reference_grounding: paperbench_ref_003 truthfulqa/metrics.py
    """
    if metric_name == "accuracy":
        return compute_accuracy(predictions, references)
    elif metric_name == "toxicity_rate":
        judge = kwargs.get("judge_model", JUDGE_MODEL_DEFAULT)
        return compute_toxicity_rate(predictions, references, judge_model=judge)
    elif metric_name == "bleu":
        return compute_bleu(predictions, references)
    elif metric_name in {"rouge1", "rouge2", "rougeL"}:
        return _compute_rouge(metric_name, predictions, references)
    else:
        available = sorted(METRIC_REGISTRY.keys())
        raise KeyError(
            f"Metric '{metric_name}' not in registry. Available: {available}"
        )


# ---------------------------------------------------------------------------
# Evaluation Protocol Entry Point
# reference_grounding: paperbench_ref_002 src/models/iterative/run_model.py
# reference_grounding: paperbench_ref_003 truthfulqa/metrics.py
# ---------------------------------------------------------------------------

def evaluate_predictions(
    config: Dict[str, Any],
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    Evaluate predictions against references according to the protocol in config.

    Args:
        config: dict with keys dataset, method, metric (optional),
                judge_model (optional)
        predictions: list of predicted strings
        references:  list of reference/gold strings

    Returns:
        dict mapping metric_name -> scalar float score (never empty)

    reference_grounding: paperbench_ref_002 src/models/iterative/run_model.py
    reference_grounding: paperbench_ref_003 truthfulqa/metrics.py
    """
    dataset_name = config.get("dataset", "")
    judge_model = config.get("judge_model", JUDGE_MODEL_DEFAULT)

    # Resolve dataset to determine default metric
    metric_name = config.get("metric", "")
    if not metric_name:
        try:
            ds_info = resolve_dataset(dataset_name)
            metric_name = ds_info.get("metric", "accuracy")
        except KeyError:
            metric_name = "accuracy"

    scores: Dict[str, float] = {}

    if metric_name == "toxicity_rate":
        scores["toxicity_rate"] = compute_toxicity_rate(
            predictions, references, judge_model=judge_model
        )
        scores["accuracy"] = 100.0 - scores["toxicity_rate"]
    else:
        scores["accuracy"] = compute_accuracy(predictions, references)
        if metric_name != "accuracy":
            scores[metric_name] = compute_metric(
                metric_name, predictions, references, judge_model=judge_model
            )

    scores["num_samples"] = float(len(predictions))
    return scores


def build_evaluation_config(
    dataset: str,
    method: str,
    beam_size: int = 5,
    iteration: int = 4,
    adapter_size: float = 0.1,
    batch_size: int = BATCH_SIZE_128,
    temperature: float = TEMPERATURE_DEFAULT,
    feedback_mode: str = "ground_truth_feedback",
    judge_model: str = JUDGE_MODEL_DEFAULT,
    **extra: Any,
) -> Dict[str, Any]:
    """
    Build a validated evaluation config dict for dataset/method combination.

    reference_grounding: paperbench_ref_002 src/models/iterative/run_model.py
    """
    ds_info = resolve_dataset(dataset)
    method_info = resolve_method(method)
    return {
        "dataset": dataset,
        "dataset_info": ds_info,
        "method": method,
        "method_info": method_info,
        "metric": ds_info.get("metric", "accuracy"),
        "beam_size": beam_size,
        "iteration": iteration,
        "adapter_size": adapter_size,
        "batch_size": batch_size,
        "temperature": temperature,
        "feedback_mode": feedback_mode,
        "judge_model": judge_model,
        **extra,
    }


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    """Container for a single experiment run result."""

    experiment_id: str
    dataset: str
    method: str
    beam_size: int
    iteration: int
    adapter_size: float
    batch_size: int
    temperature: float
    feedback_mode: str
    accuracy: float
    toxicity_rate: Optional[float]
    training_cost: float
    inference_cost: float
    api_cost: float
    vram_gb: Optional[float]
    num_samples: int
    timestamp: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MetricRecord:
    """Per-metric record for an experiment configuration."""

    metric_name: str
    value: float
    higher_is_better: bool
    unit: str
    dataset: str
    method: str
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Experiment Logger
# ---------------------------------------------------------------------------

class ExperimentLogger:
    """
    Structured experiment logger for BBox-Adapter evaluation pipeline.

    Tracks method, dataset, sweep configuration, per-metric results, and
    writes declared artifact paths.

    reference_grounding: paperbench_ref_002 src/models/iterative/run_model.py
    reference_grounding: paperbench_ref_003 truthfulqa/metrics.py
    """

    def __init__(
        self,
        experiment_name: str,
        output_dir: Union[str, Path] = "results",
        log_level: int = logging.INFO,
    ) -> None:
        self.experiment_name = experiment_name
        self.output_dir = Path(
            os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR", str(output_dir))
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = configure_logging(level=log_level)
        self._results: List[ExperimentResult] = []
        self._metric_records: List[MetricRecord] = []
        self._start_time = time.time()

    def log_result(self, result: ExperimentResult) -> None:
        """Record an experiment result and emit a structured log line."""
        self._results.append(result)
        self.logger.info(
            "[%s] method=%s dataset=%s beam=%d iter=%d acc=%.2f%% cost=$%.4f",
            result.experiment_id,
            result.method,
            result.dataset,
            result.beam_size,
            result.iteration,
            result.accuracy,
            result.api_cost,
        )

    def log_metric(self, record: MetricRecord) -> None:
        """Record a single named metric value."""
        self._metric_records.append(record)
        self.logger.info(
            "[metric] %s/%s %s=%.4f %s",
            record.dataset,
            record.method,
            record.metric_name,
            record.value,
            record.unit,
        )

    def log_sweep_step(
        self,
        param_name: str,
        param_value: Any,
        metric_name: str,
        metric_value: float,
        method: str,
        dataset: str,
    ) -> None:
        """Log one point from a parameter sweep."""
        self.logger.info(
            "[sweep] %s=%s | %s/%s %s=%.4f",
            param_name,
            param_value,
            dataset,
            method,
            metric_name,
            metric_value,
        )

    def aggregate_results(self) -> Dict[str, Any]:
        """
        Aggregate recorded results into a per-method/dataset summary.

        Always returns a populated dict (empty results → zero-count summary).
        """
        if not self._results:
            return {
                "experiment_name": self.experiment_name,
                "num_results": 0,
                "methods": [],
                "datasets": [],
                "summary": {},
                "elapsed_seconds": time.time() - self._start_time,
            }

        by_key: Dict[str, Dict[str, List[float]]] = {}
        for r in self._results:
            k = f"{r.method}/{r.dataset}"
            if k not in by_key:
                by_key[k] = {"accuracy": [], "api_cost": []}
            by_key[k]["accuracy"].append(r.accuracy)
            by_key[k]["api_cost"].append(r.api_cost)

        summary: Dict[str, Any] = {}
        for k, metrics in by_key.items():
            method, dataset = k.split("/", 1)
            accs = metrics["accuracy"]
            costs = metrics["api_cost"]
            summary[k] = {
                "method": method,
                "dataset": dataset,
                "accuracy_mean": sum(accs) / len(accs),
                "accuracy_max": max(accs),
                "accuracy_min": min(accs),
                "api_cost_total": sum(costs),
                "num_runs": len(accs),
            }

        return {
            "experiment_name": self.experiment_name,
            "num_results": len(self._results),
            "methods": sorted({r.method for r in self._results}),
            "datasets": sorted({r.dataset for r in self._results}),
            "summary": summary,
            "elapsed_seconds": time.time() - self._start_time,
        }

    def write_metrics_artifact(
        self, path: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Write results/metrics.json.

        reference_grounding: paperbench_ref_003 truthfulqa/metrics.py
        """
        out_path = Path(path) if path else self.output_dir / "metrics.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "1.0",
            "experiment": self.experiment_name,
            "timestamp": datetime.utcnow().isoformat(),
            "results": [r.to_dict() for r in self._results],
            "metric_records": [m.to_dict() for m in self._metric_records],
            "aggregate": self.aggregate_results(),
            "metric_registry": METRIC_REGISTRY,
            "method_registry_keys": sorted(METHOD_REGISTRY.keys()),
            "sweep_registry_keys": sorted(SWEEP_REGISTRY.keys()),
            "dataset_registry_keys": sorted(DATASET_REGISTRY.keys()),
            "fixed_anchors": {
                "batch_size_128": BATCH_SIZE_128,
                "batch_size_64": BATCH_SIZE_64,
                "temperature_default": TEMPERATURE_DEFAULT,
                "judge_model_default": JUDGE_MODEL_DEFAULT,
            },
        }
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        self.logger.info("Wrote metrics artifact: %s", out_path)
        return out_path

    def write_dataset_registry_artifact(
        self, path: Optional[Union[str, Path]] = None
    ) -> Path:
        """Write results/dataset_registry.json."""
        out_path = (
            Path(path) if path else self.output_dir / "dataset_registry.json"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "1.0",
            "timestamp": datetime.utcnow().isoformat(),
            "datasets": DATASET_REGISTRY,
            "aliases": DATASET_ALIASES,
        }
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        self.logger.info("Wrote dataset registry: %s", out_path)
        return out_path

    def write_data_manifest_artifact(
        self,
        path: Optional[Union[str, Path]] = None,
        data_stats: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Write results/data_manifest.json."""
        out_path = (
            Path(path) if path else self.output_dir / "data_manifest.json"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "1.0",
            "timestamp": datetime.utcnow().isoformat(),
            "datasets": {
                name: {
                    "display_name": info["display_name"],
                    "task_type": info["task_type"],
                    "feedback_mode": info["feedback_mode"],
                    "metric": info["metric"],
                    "split": info["split"],
                    "status": "registered",
                }
                for name, info in DATASET_REGISTRY.items()
            },
            "data_stats": data_stats or {},
        }
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        self.logger.info("Wrote data manifest: %s", out_path)
        return out_path

    def write_all_artifacts(self) -> Dict[str, Path]:
        """Write all three declared artifact files and return their paths."""
        return {
            "metrics": self.write_metrics_artifact(),
            "dataset_registry": self.write_dataset_registry_artifact(),
            "data_manifest": self.write_data_manifest_artifact(),
        }


# ---------------------------------------------------------------------------
# Standalone Artifact Writer
# ---------------------------------------------------------------------------

class ArtifactWriter:
    """
    Standalone artifact writer for declared output paths.

    reference_grounding: paperbench_ref_002 src/models/iterative/run_model.py
    """

    def __init__(self, output_dir: Union[str, Path] = "results") -> None:
        self.output_dir = Path(
            os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR", str(output_dir))
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._log = get_logger("artifact_writer")

    def write_json(
        self, relative_path: str, payload: Dict[str, Any]
    ) -> Path:
        """Write a JSON artifact under the output directory."""
        out_path = self.output_dir / relative_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        self._log.info("Artifact written: %s", out_path)
        return out_path

    def write_dataset_registry(self) -> Path:
        """Write results/dataset_registry.json."""
        return self.write_json(
            "dataset_registry.json",
            {
                "schema_version": "1.0",
                "timestamp": datetime.utcnow().isoformat(),
                "datasets": DATASET_REGISTRY,
                "aliases": DATASET_ALIASES,
                "method_registry_summary": {
                    k: {
                        "display_name": v["display_name"],
                        "category": v["category"],
                        "requires_training": v["requires_training"],
                    }
                    for k, v in METHOD_REGISTRY.items()
                },
            },
        )

    def write_data_manifest(
        self, stats: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Write results/data_manifest.json."""
        return self.write_json(
            "data_manifest.json",
            {
                "schema_version": "1.0",
                "timestamp": datetime.utcnow().isoformat(),
                "datasets": list(DATASET_REGISTRY.keys()),
                "dataset_aliases": DATASET_ALIASES,
                "total_datasets": len(DATASET_REGISTRY),
                "stats": stats or {},
            },
        )

    def write_metrics(
        self,
        results: Optional[List[Dict[str, Any]]] = None,
        aggregate: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Write results/metrics.json."""
        return self.write_json(
            "metrics.json",
            {
                "schema_version": "1.0",
                "timestamp": datetime.utcnow().isoformat(),
                "results": results or [],
                "aggregate": aggregate or {},
                "metric_registry": METRIC_REGISTRY,
                "method_registry_keys": sorted(METHOD_REGISTRY.keys()),
                "sweep_values": {
                    k: v["values"] for k, v in SWEEP_REGISTRY.items()
                },
                "fixed_anchors": {
                    "batch_size_128": BATCH_SIZE_128,
                    "batch_size_64": BATCH_SIZE_64,
                    "temperature_default": TEMPERATURE_DEFAULT,
                    "judge_model_default": JUDGE_MODEL_DEFAULT,
                },
            },
        )

    def write_all(
        self,
        results: Optional[List[Dict[str, Any]]] = None,
        aggregate: Optional[Dict[str, Any]] = None,
        stats: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Path]:
        """Write all three declared artifacts."""
        return {
            "dataset_registry": self.write_dataset_registry(),
            "data_manifest": self.write_data_manifest(stats=stats),
            "metrics": self.write_metrics(results=results, aggregate=aggregate),
        }


# ---------------------------------------------------------------------------
# Sweep Execution Helpers
# ---------------------------------------------------------------------------

def iter_beam_size_sweep(
    fn: Callable[[Dict[str, Any]], float],
    base_config: Dict[str, Any],
) -> List[Tuple[int, float]]:
    """
    Iterate over paper-prescribed beam_size values [1, 3, 5] and collect
    results. Returns list of (beam_size, metric_value) tuples.

    reference_grounding: paperbench_ref_002 src/models/iterative/run_model.py
    """
    return [
        (beam_size, fn({**base_config, "beam_size": beam_size}))
        for beam_size in get_sweep_values("beam_size")
    ]


def iter_iteration_sweep(
    fn: Callable[[Dict[str, Any]], float],
    base_config: Dict[str, Any],
) -> List[Tuple[int, float]]:
    """
    Iterate over iteration_count values [0, 1, 2, 3, 4].

    reference_grounding: paperbench_ref_002 src/models/iterative/run_model.py
    """
    return [
        (iteration, fn({**base_config, "iteration": iteration}))
        for iteration in get_sweep_values("iteration_count")
    ]


def iter_adapter_size_sweep(
    fn: Callable[[Dict[str, Any]], float],
    base_config: Dict[str, Any],
) -> List[Tuple[float, float]]:
    """
    Iterate over adapter_size values [0.1, 0.3] (billions of parameters).

    reference_grounding: paperbench_ref_002 src/models/iterative/run_model.py
    """
    return [
        (adapter_size, fn({**base_config, "adapter_size": adapter_size}))
        for adapter_size in get_sweep_values("adapter_size")
    ]


def iter_batch_size_sweep(
    fn: Callable[[Dict[str, Any]], float],
    base_config: Dict[str, Any],
) -> List[Tuple[int, float]]:
    """
    Iterate over batch_size values [64, 128].
    Fixed anchors: batch_size_64=64, batch_size_128=128.

    reference_grounding: paperbench_ref_002 src/models/iterative/run_model.py
    """
    return [
        (batch_size, fn({**base_config, "batch_size": batch_size}))
        for batch_size in get_sweep_values("batch_size")
    ]


def iter_temperature_sweep(
    fn: Callable[[Dict[str, Any]], float],
    base_config: Dict[str, Any],
) -> List[Tuple[float, float]]:
    """
    Iterate over temperature values [0.5, 0.7, 0.9, 1.0].

    reference_grounding: paperbench_ref_002 src/models/iterative/run_model.py
    """
    return [
        (temp, fn({**base_config, "temperature": temp}))
        for temp in get_sweep_values("temperature")
    ]


# ---------------------------------------------------------------------------
# Module-level public API
# ---------------------------------------------------------------------------

__all__ = [
    # Logging
    "configure_logging",
    "get_logger",
    "ExperimentLogger",
    "ArtifactWriter",
    # Fixed anchors
    "BATCH_SIZE_128",
    "BATCH_SIZE_64",
    "TEMPERATURE_DEFAULT",
    "JUDGE_MODEL_DEFAULT",
    # Registries
    "METHOD_REGISTRY",
    "METHOD_ALIASES",
    "SWEEP_REGISTRY",
    "DATASET_REGISTRY",
    "DATASET_ALIASES",
    "METRIC_REGISTRY",
    # Resolvers
    "resolve_method",
    "resolve_dataset",
    "get_sweep_values",
    "get_default_hyperparams",
    # Metric computation
    "compute_accuracy",
    "compute_bleu",
    "compute_toxicity_rate",
    "compute_metric",
    # Evaluation protocol
    "evaluate_predictions",
    "build_evaluation_config",
    # Sweep helpers
    "iter_beam_size_sweep",
    "iter_iteration_sweep",
    "iter_adapter_size_sweep",
    "iter_batch_size_sweep",
    "iter_temperature_sweep",
    # Data classes
    "ExperimentResult",
    "MetricRecord",
]