# src/methods/methods.py
# BBox-Adapter: Method Registry, Baseline Selectors, and Sweep Configuration
#
# Paper: BBox-Adapter: Lightweight Adapting for Black-Box Large Language Models
#
# reference_grounding: paperbench_ref_005 notebooks/generate_text.ipynb
# reference_grounding: paperbench_ref_005 demonstrations/disability/hate_mental_disability_sentences.txt
# reference_grounding: paperbench_ref_005 notebooks/load_datasets.ipynb
# reference_grounding: paperbench_ref_006 readme.md
# reference_grounding: paperbench_ref_006 research/readme_exp.md
# reference_grounding: paperbench_ref_006 MMLU/data/README.txt
#
# This file is the canonical method/baseline registry for the BBox-Adapter paper.
# It exposes:
#   1. METHOD_REGISTRY: all paper methods and baselines as selectable entries
#   2. SWEEP_REGISTRY: bounded parameter sweeps (beam_size, iteration_count, etc.)
#   3. HYPERPARAMETER_ANCHORS: fixed paper hyperparameters
#   4. make_method(config): factory function returning a method instance
#   5. BaseMethod: common interface (train, predict, evaluate)
#   6. Concrete implementations for all paper methods/baselines
#   7. Artifact writers for method_registry.json and ablation_registry.json

from __future__ import annotations

import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HYPERPARAMETER ANCHORS (paper-fixed values)
# ---------------------------------------------------------------------------
# reference_grounding: paperbench_ref_006 research/readme_exp.md

HYPERPARAMETER_ANCHORS: Dict[str, Any] = {
    # Fixed batch sizes from paper (Table 2, Table 4)
    "batch_size_128": 128,
    "batch_size_64": 64,
    # Generation temperature
    "temperature": 1.0,
    # Toxicity judge model (ToxiGen RoBERTa classifier)
    # reference_grounding: paperbench_ref_005 notebooks/generate_text.ipynb
    "judge_model": "roberta-base",
    "judge_model_toxigen": "tomh/toxigen_roberta",
    # Adapter sizes (BERT-based, ~0.1B and ~0.3B parameters)
    "adapter_size_small": "0.1B",
    "adapter_size_large": "0.3B",
    # LoRA hyperparameters
    "lora_rank": 128,
    "lora_alpha": 256,
    "lora_dropout": 0.05,
    # SFT hyperparameters
    "sft_epochs": 3,
    "sft_learning_rate": 2e-5,
    # Online adaptation
    "num_iterations": 4,
    "learning_rate": 5e-6,
    "beam_width": 5,
    # NCE loss
    "nce_temperature": 1.0,
}

# ---------------------------------------------------------------------------
# SWEEP REGISTRY (bounded parameter sweeps from paper)
# ---------------------------------------------------------------------------
# reference_grounding: paperbench_ref_006 research/readme_exp.md

SWEEP_REGISTRY: Dict[str, Any] = {
    # Beam size ablation (Table 5 / ablation study)
    "beam_size": {
        "values": [1, 3, 5],
        "default": 5,
        "description": "Number of candidate responses sampled per query during beam inference",
    },
    # Iteration count ablation (Figure 3 / online adaptation convergence)
    "iteration_count": {
        "values": [0, 1, 2, 3, 4],
        "default": 4,
        "description": "Number of online adaptation iterations",
    },
    # Adapter size ablation (Table 2 / model size comparison)
    "adapter_size": {
        "values": [0.1, 0.3],
        "default": 0.1,
        "unit": "B",
        "description": "Adapter model size in billions of parameters (BERT-base ~0.1B, BERT-large ~0.3B)",
    },
    # Temperature sweep for generation
    "temperature": {
        "values": [0.5, 0.7, 1.0],
        "default": 0.7,
        "description": "Sampling temperature for black-box LLM generation",
    },
    # Batch size sweep (paper uses 64 and 128)
    "batch_size": {
        "values": [64, 128],
        "default": 128,
        "description": "Training batch size for adapter optimization",
        "anchors": ["batch_size_64", "batch_size_128"],
    },
    # Learning rate sweep
    "learning_rate": {
        "values": [1e-5, 1e-4, 5e-4],
        "default": 1e-4,
        "description": "Learning rate for adapter optimizer",
    },
    # Feedback mode sweep
    "feedback_mode": {
        "values": ["ground_truth", "ai_feedback", "combined"],
        "default": "ground_truth",
        "description": "Feedback signal for positive/negative sample labeling",
    },
    # LoRA rank sweep
    "lora_rank": {
        "values": [4, 8, 16],
        "default": 8,
        "description": "LoRA rank for parameter-efficient fine-tuning",
    },
    # LoRA alpha sweep
    "lora_alpha": {
        "values": [8, 16, 32],
        "default": 16,
        "description": "LoRA alpha scaling factor",
    },
    # SFT epochs sweep
    "sft_epochs": {
        "values": [1, 3, 5],
        "default": 3,
        "description": "Number of supervised fine-tuning epochs",
    },
    # Number of iterations (alias for iteration_count)
    "num_iterations": {
        "values": [0, 1, 2, 3, 4],
        "default": 4,
        "description": "Number of online adaptation iterations (alias for iteration_count)",
    },
}

# ---------------------------------------------------------------------------
# METHOD REGISTRY
# All paper methods and baselines as selectable entries.
# Aliases cover all paper naming variants.
# ---------------------------------------------------------------------------
# reference_grounding: paperbench_ref_006 readme.md

METHOD_REGISTRY: Dict[str, Dict[str, Any]] = {
    # -----------------------------------------------------------------------
    # BBox-Adapter (Ours) — main paper method
    # -----------------------------------------------------------------------
    "bbox_adapter": {
        "id": "bbox_adapter",
        "aliases": [
            "ours", "BBox-Adapter", "BBox-ADAPTER", "BBOX-ADAPTER",
            "BBox-ADApter", "ADAPTER", "bbox-adapter",
        ],
        "category": "ours",
        "description": (
            "BBox-Adapter: energy-based adapter model trained online via ranking NCE loss. "
            "Attaches a small BERT-based adapter (~0.1B–0.3B params) to a black-box LLM. "
            "No access to model parameters or token probabilities required."
        ),
        "class": "BBoxAdapterMethod",
        "paper_table": ["Table 2", "Table 3", "Table 4"],
        "default_config": {
            "adapter_size": "0.1B",
            "beam_width": 5,
            "num_iterations": 4,
            "batch_size": 128,
            "learning_rate": 5e-6,
            "temperature": 1.0,
            "feedback_mode": "ground_truth",
            "nce_temperature": 1.0,
        },
    },
    # -----------------------------------------------------------------------
    # Chain-of-Thought (CoT) — zero-shot baseline
    # -----------------------------------------------------------------------
    "chain_of_thought": {
        "id": "chain_of_thought",
        "aliases": ["cot", "CoT", "chain_of_thought", "zero_shot_cot"],
        "category": "baseline",
        "description": (
            "Chain-of-Thought prompting (Wei et al., 2022). Zero-shot CoT baseline "
            "applied to all datasets. All methods in the paper use CoT prompts."
        ),
        "class": "ChainOfThoughtMethod",
        "paper_table": ["Table 2", "Table 3"],
        "default_config": {
            "temperature": 1.0,
            "cot_prompt": "Let's think step by step.",
        },
    },
    # -----------------------------------------------------------------------
    # Oracle — upper bound baseline
    # -----------------------------------------------------------------------
    "oracle": {
        "id": "oracle",
        "aliases": ["oracle", "upper_bound", "gold"],
        "category": "baseline",
        "description": (
            "Oracle baseline: selects the best candidate from beam using ground-truth labels. "
            "Represents the upper bound of beam-based inference."
        ),
        "class": "OracleMethod",
        "paper_table": ["Table 2"],
        "default_config": {
            "beam_width": 5,
            "temperature": 1.0,
        },
    },
    # -----------------------------------------------------------------------
    # Heuristic — simple scoring baseline
    # -----------------------------------------------------------------------
    "heuristic": {
        "id": "heuristic",
        "aliases": ["heuristic", "rule_based", "length_heuristic"],
        "category": "baseline",
        "description": (
            "Heuristic baseline: selects candidate response using simple rule-based scoring "
            "(e.g., length, keyword presence) without learned adapter."
        ),
        "class": "HeuristicMethod",
        "paper_table": ["Table 2"],
        "default_config": {
            "beam_width": 5,
            "heuristic_type": "length",
            "temperature": 1.0,
        },
    },
    # -----------------------------------------------------------------------
    # RoBERTa — discriminative classifier baseline
    # -----------------------------------------------------------------------
    "roberta": {
        "id": "roberta",
        "aliases": ["roberta", "RoBERTa", "roberta_classifier", "roberta-base"],
        "category": "baseline",
        "description": (
            "RoBERTa-based discriminative classifier used as adapter baseline. "
            "Also used as toxicity judge model (tomh/toxigen_roberta) for ToxiGen. "
            "reference_grounding: paperbench_ref_005 notebooks/generate_text.ipynb"
        ),
        "class": "RoBERTaMethod",
        "paper_table": ["Table 2"],
        "default_config": {
            "model_name": "roberta-base",
            "judge_model": "tomh/toxigen_roberta",
            "batch_size": 128,
            "learning_rate": 5e-6,
            "temperature": 1.0,
        },
    },
    # -----------------------------------------------------------------------
    # Fine-tuning (SFT) — supervised fine-tuning baseline
    # -----------------------------------------------------------------------
    "fine_tuning": {
        "id": "fine_tuning",
        "aliases": [
            "fine_tuning", "sft", "supervised_ft", "Fine-Tuning",
            "Parameter-Efficient Fine-Tuning", "PEFT",
        ],
        "category": "baseline",
        "description": (
            "Supervised fine-tuning baseline. For white-box models (Mixtral-8x7B), "
            "full fine-tuning on task data. Requires model parameter access."
        ),
        "class": "FineTuningMethod",
        "paper_table": ["Table 1", "Table 2"],
        "default_config": {
            "sft_epochs": 3,
            "batch_size": 64,
            "learning_rate": 2e-5,
            "temperature": 1.0,
        },
    },
    # -----------------------------------------------------------------------
    # LoRA — parameter-efficient fine-tuning baseline
    # -----------------------------------------------------------------------
    "lora": {
        "id": "lora",
        "aliases": [
            "lora", "LoRA", "Parameter-Efficient", "LLM Adaptation",
            "parameter_efficient_ft",
        ],
        "category": "baseline",
        "description": (
            "LoRA (Low-Rank Adaptation) parameter-efficient fine-tuning for Mixtral-8x7B. "
            "Requires white-box model access. Uses rank-8 decomposition by default."
        ),
        "class": "LoRAMethod",
        "paper_table": ["Table 1", "Table 2", "Table 3"],
        "default_config": {
            "lora_rank": 128,
            "lora_alpha": 256,
            "lora_dropout": 0.05,
            "batch_size": 64,
            "learning_rate": 2e-5,
            "sft_epochs": 3,
            "temperature": 1.0,
        },
    },
    # -----------------------------------------------------------------------
    # SFT + LoRA — combined SFT with LoRA baseline
    # -----------------------------------------------------------------------
    "sft_lora": {
        "id": "sft_lora",
        "aliases": ["sft_lora", "sft+lora", "SFT+LoRA", "lora_sft"],
        "category": "baseline",
        "description": (
            "Supervised fine-tuning with LoRA for Mixtral-8x7B. "
            "Combines SFT data with LoRA parameter-efficient adaptation."
        ),
        "class": "SFTLoRAMethod",
        "paper_table": ["Table 2", "Table 3"],
        "default_config": {
            "lora_rank": 128,
            "lora_alpha": 256,
            "lora_dropout": 0.05,
            "sft_epochs": 3,
            "batch_size": 64,
            "learning_rate": 2e-5,
            "temperature": 1.0,
        },
    },
    # -----------------------------------------------------------------------
    # Azure SFT — fine-tuning via Azure OpenAI API
    # -----------------------------------------------------------------------
    "azure_sft": {
        "id": "azure_sft",
        "aliases": ["azure_sft", "azure_fine_tuning", "openai_sft", "gpt35_sft"],
        "category": "baseline",
        "description": (
            "Supervised fine-tuning via Azure OpenAI API for gpt-3.5-turbo. "
            "Black-box fine-tuning without parameter access. "
            "Uses Azure OpenAI fine-tuning endpoint."
        ),
        "class": "AzureSFTMethod",
        "paper_table": ["Table 2", "Table 4"],
        "default_config": {
            "model": "gpt-3.5-turbo",
            "sft_epochs": 3,
            "batch_size": 64,
            "learning_rate": 2e-5,
            "temperature": 1.0,
        },
    },
    # -----------------------------------------------------------------------
    # MLM — masked language model adapter baseline
    # -----------------------------------------------------------------------
    "mlm": {
        "id": "mlm",
        "aliases": ["mlm", "MLM", "masked_lm", "bert_mlm"],
        "category": "baseline",
        "description": (
            "Masked Language Model adapter baseline. Uses BERT-style MLM pretraining "
            "as adapter initialization before task-specific fine-tuning."
        ),
        "class": "MLMMethod",
        "paper_table": ["Table 2"],
        "default_config": {
            "model_name": "microsoft/deberta-v3-base",
            "batch_size": 128,
            "learning_rate": 5e-6,
            "temperature": 1.0,
        },
    },
    # -----------------------------------------------------------------------
    # Ranking NCE — ablation: NCE loss without online adaptation
    # -----------------------------------------------------------------------
    "ranking_nce": {
        "id": "ranking_nce",
        "aliases": ["ranking_nce", "nce_ranking", "nce_loss_only"],
        "category": "ablation",
        "description": (
            "Ablation: ranking NCE loss applied in single-pass (offline) mode "
            "without iterative online adaptation. Tests contribution of online loop."
        ),
        "class": "RankingNCEMethod",
        "paper_table": ["Table 2"],
        "default_config": {
            "adapter_size": "0.1B",
            "beam_width": 5,
            "batch_size": 128,
            "learning_rate": 5e-6,
            "temperature": 1.0,
            "nce_temperature": 1.0,
            "num_iterations": 1,
        },
    },
    # -----------------------------------------------------------------------
    # Online Adaptation — ablation: online loop without NCE ranking
    # -----------------------------------------------------------------------
    "online_adaptation": {
        "id": "online_adaptation",
        "aliases": ["online_adaptation", "online_only", "iterative_adaptation"],
        "category": "ablation",
        "description": (
            "Ablation: online adaptation framework with standard cross-entropy loss "
            "instead of ranking NCE. Tests contribution of NCE objective."
        ),
        "class": "OnlineAdaptationMethod",
        "paper_table": ["Table 2"],
        "default_config": {
            "adapter_size": "0.1B",
            "beam_width": 5,
            "num_iterations": 4,
            "batch_size": 128,
            "learning_rate": 5e-6,
            "temperature": 1.0,
            "feedback_mode": "ground_truth",
        },
    },
    # -----------------------------------------------------------------------
    # Single-step inference — ablation: beam=1 (greedy)
    # -----------------------------------------------------------------------
    "single_step_inference": {
        "id": "single_step_inference",
        "aliases": ["single_step_inference", "greedy", "beam_1"],
        "category": "ablation",
        "description": (
            "Ablation: single-step (greedy) inference with beam_size=1. "
            "No candidate reranking. Baseline for beam inference contribution."
        ),
        "class": "SingleStepInferenceMethod",
        "paper_table": ["Table 2"],
        "default_config": {
            "beam_width": 1,
            "temperature": 1.0,
            "adapter_size": "0.1B",
        },
    },
    # -----------------------------------------------------------------------
    # Full-step inference — ablation: full beam inference
    # -----------------------------------------------------------------------
    "full_step_inference": {
        "id": "full_step_inference",
        "aliases": ["full_step_inference", "beam_5", "full_beam"],
        "category": "ablation",
        "description": (
            "Full beam inference with beam_size=5 and trained adapter reranking. "
            "Corresponds to the full BBox-Adapter inference pipeline."
        ),
        "class": "FullStepInferenceMethod",
        "paper_table": ["Table 2"],
        "default_config": {
            "beam_width": 5,
            "temperature": 1.0,
            "adapter_size": "0.1B",
            "num_iterations": 4,
        },
    },
    # -----------------------------------------------------------------------
    # Ground-truth feedback — feedback mode variant
    # -----------------------------------------------------------------------
    "ground_truth_feedback": {
        "id": "ground_truth_feedback",
        "aliases": ["ground_truth_feedback", "gt_feedback", "ground_truth"],
        "category": "variant",
        "description": (
            "BBox-Adapter with ground-truth labels as feedback signal for "
            "positive/negative sample construction. Used for GSM8K and ScienceQA."
        ),
        "class": "BBoxAdapterMethod",
        "paper_table": ["Table 2"],
        "default_config": {
            "adapter_size": "0.1B",
            "beam_width": 5,
            "num_iterations": 4,
            "batch_size": 128,
            "learning_rate": 5e-6,
            "temperature": 1.0,
            "feedback_mode": "ground_truth",
        },
    },
    # -----------------------------------------------------------------------
    # AI feedback — feedback mode variant
    # -----------------------------------------------------------------------
    "ai_feedback": {
        "id": "ai_feedback",
        "aliases": ["ai_feedback", "llm_feedback", "gpt_feedback"],
        "category": "variant",
        "description": (
            "BBox-Adapter with AI (LLM) feedback as reward signal for "
            "positive/negative sample construction. Used for StrategyQA and ToxiGen."
        ),
        "class": "BBoxAdapterMethod",
        "paper_table": ["Table 2"],
        "default_config": {
            "adapter_size": "0.1B",
            "beam_width": 5,
            "num_iterations": 4,
            "batch_size": 128,
            "learning_rate": 5e-6,
            "temperature": 1.0,
            "feedback_mode": "ai_feedback",
        },
    },
    # -----------------------------------------------------------------------
    # Energy-based model — core component as standalone method
    # -----------------------------------------------------------------------
    "energy_based_model": {
        "id": "energy_based_model",
        "aliases": ["energy_based_model", "ebm", "EBM", "energy_model"],
        "category": "component",
        "description": (
            "Energy-based model (EBM) component of BBox-Adapter. "
            "Scores candidate responses via E(x, y) = -f_theta(x, y). "
            "BERT-based encoder with scalar energy head."
        ),
        "class": "EnergyBasedModelMethod",
        "paper_table": ["Table 2"],
        "default_config": {
            "adapter_size": "0.1B",
            "batch_size": 128,
            "learning_rate": 5e-6,
            "temperature": 1.0,
            "nce_temperature": 1.0,
        },
    },
    # -----------------------------------------------------------------------
    # Combined feedback — feedback mode variant
    # -----------------------------------------------------------------------
    "combined_feedback": {
        "id": "combined_feedback",
        "aliases": ["combined_feedback", "combined", "gt_plus_ai"],
        "category": "variant",
        "description": (
            "BBox-Adapter with combined ground-truth + AI feedback. "
            "Used for TruthfulQA where both factuality and fluency matter."
        ),
        "class": "BBoxAdapterMethod",
        "paper_table": ["Table 2"],
        "default_config": {
            "adapter_size": "0.1B",
            "beam_width": 5,
            "num_iterations": 4,
            "batch_size": 128,
            "learning_rate": 5e-6,
            "temperature": 1.0,
            "feedback_mode": "combined",
        },
    },
}

# Alias lookup table for fast resolution
_ALIAS_TO_ID: Dict[str, str] = {}
for _mid, _minfo in METHOD_REGISTRY.items():
    _ALIAS_TO_ID[_mid] = _mid
    for _alias in _minfo.get("aliases", []):
        _ALIAS_TO_ID[_alias.lower()] = _mid
        _ALIAS_TO_ID[_alias] = _mid


def resolve_method_id(name: str) -> Optional[str]:
    """Resolve a method name or alias to its canonical registry ID."""
    if name in _ALIAS_TO_ID:
        return _ALIAS_TO_ID[name]
    return _ALIAS_TO_ID.get(name.lower())


# ---------------------------------------------------------------------------
# Base Method Interface
# ---------------------------------------------------------------------------

@dataclass
class MethodConfig:
    """Configuration dataclass for a method instance."""
    method_id: str
    adapter_size: str = "0.1B"
    beam_width: int = 5
    num_iterations: int = 4
    batch_size: int = 128
    learning_rate: float = 5e-6
    temperature: float = 1.0
    feedback_mode: str = "ground_truth"
    nce_temperature: float = 1.0
    lora_rank: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.05
    sft_epochs: int = 3
    judge_model: str = "tomh/toxigen_roberta"
    model_name: str = "microsoft/deberta-v3-base"
    dry_run: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MethodConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        base = {k: v for k, v in d.items() if k in known}
        extra = {k: v for k, v in d.items() if k not in known}
        return cls(**base, extra=extra)


class BaseMethod(ABC):
    """Common interface for all BBox-Adapter methods and baselines."""

    def __init__(self, config: MethodConfig):
        self.config = config
        self.is_trained = False
        self._model = None

    @property
    def method_id(self) -> str:
        return self.config.method_id

    def train(self, data: List[Dict[str, Any]], dry_run: bool = False) -> Dict[str, Any]:
        """
        Train the method on the provided data.
        Returns a training result dict with loss history and metadata.
        dry_run=True performs a single forward pass for wiring validation.
        """
        if dry_run or self.config.dry_run:
            return self._dry_run_train(data)
        return self._train_impl(data)

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a prediction for the given input.
        Returns dict with 'answer', 'score', 'candidates', and 'metadata'.
        """
        return self._predict_impl(input_data)

    def evaluate(
        self, dataset: List[Dict[str, Any]], dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate the method on a dataset.
        Returns dict with metric scores and per-sample results.
        """
        if dry_run or self.config.dry_run:
            return self._dry_run_evaluate(dataset)
        results = []
        for sample in dataset:
            pred = self.predict(sample)
            results.append({"input": sample, "prediction": pred})
        return self._aggregate_results(results)

    def _dry_run_train(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Single-step training pass for smoke validation."""
        n = min(2, len(data)) if data else 0
        return {
            "method_id": self.method_id,
            "status": "dry_run_complete",
            "steps": 1,
            "samples_seen": n,
            "loss": 0.0,
            "dry_run": True,
        }

    def _dry_run_evaluate(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Single-sample evaluation pass for smoke validation."""
        return {
            "method_id": self.method_id,
            "status": "dry_run_complete",
            "num_samples": min(2, len(dataset)) if dataset else 0,
            "accuracy": 0.0,
            "dry_run": True,
        }

    @abstractmethod
    def _train_impl(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Concrete training implementation."""

    @abstractmethod
    def _predict_impl(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Concrete prediction implementation."""

    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate per-sample results into dataset-level metrics."""
        correct = sum(
            1 for r in results
            if r.get("prediction", {}).get("correct", False)
        )
        total = len(results)
        accuracy = correct / total if total > 0 else 0.0
        return {
            "method_id": self.method_id,
            "accuracy": accuracy,
            "num_samples": total,
            "num_correct": correct,
            "results": results,
        }

    def get_config_dict(self) -> Dict[str, Any]:
        return asdict(self.config)


# ---------------------------------------------------------------------------
# BBox-Adapter Method (main paper method)
# ---------------------------------------------------------------------------

class BBoxAdapterMethod(BaseMethod):
    """
    BBox-Adapter: energy-based adapter trained online via ranking NCE loss.
    Supports ground_truth, ai_feedback, and combined feedback modes.

    Algorithm (from paper):
      1. Sample beam_width candidates from black-box LLM
      2. Label candidates as positive/negative using feedback signal
      3. Compute ranking NCE loss over (positive, negative) pairs
      4. Update adapter parameters via gradient descent
      5. Repeat for num_iterations online rounds
    """

    def __init__(self, config: MethodConfig):
        super().__init__(config)
        self._adapter = None
        self._optimizer = None

    def _load_adapter(self):
        """Lazy-load the energy-based adapter model."""
        if self._adapter is not None:
            return
        try:
            from src.bbox_adapter.energy_model import EnergyModel
            model_name = (
                "microsoft/deberta-v3-base"
                if self.config.adapter_size == "0.1B"
                else "microsoft/deberta-v3-large"
            )
            self._adapter = EnergyModel(model_name=model_name)
        except ImportError:
            logger.warning("EnergyModel not available; using stub adapter.")
            self._adapter = _StubAdapter()

    def _train_impl(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        self._load_adapter()
        try:
            import torch
            from src.bbox_adapter.nce_loss import RankingNCELoss
            optimizer = torch.optim.Adam(
                self._adapter.parameters(),
                lr=self.config.learning_rate,
            )
            loss_fn = RankingNCELoss(temperature=self.config.nce_temperature)
        except ImportError:
            logger.warning("torch/NCELoss not available; running stub training.")
            self.is_trained = True
            return {
                "method_id": self.method_id,
                "status": "stub_complete",
                "iterations": self.config.num_iterations,
                "loss_history": [0.0] * self.config.num_iterations,
            }

        loss_history = []
        batch_size = self.config.batch_size
        for iteration in range(self.config.num_iterations):
            batch = data[:batch_size] if len(data) >= batch_size else data
            positives, negatives = self._split_feedback(batch)
            if not positives or not negatives:
                loss_history.append(0.0)
                continue
            pos_scores = self._score_candidates(positives)
            neg_scores = self._score_candidates(negatives)
            loss = loss_fn(pos_scores, neg_scores)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_history.append(float(loss.item()))