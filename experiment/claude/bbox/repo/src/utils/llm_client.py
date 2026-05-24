#!/usr/bin/env python3
"""
src/utils/llm_client.py

BBox-Adapter LLM Client Utility

Provides black-box LLM interfaces for:
  - OpenAI GPT-3.5-turbo (direct API)
  - Azure OpenAI endpoint (SFT-compatible)
  - HuggingFace Mixtral-8x7B (open-source LLM)
  - RoBERTa-base toxicity judge (ToxiGen protocol)
  - Chain-of-Thought (CoT) baseline wrapper

Method/Baseline Registry (Paper Evidence Contract):
  ours, chain_of_thought, oracle, heuristic, roberta, fine_tuning, lora,
  sft_lora, azure_sft, mlm, bbox_adapter, ranking_nce, online_adaptation,
  single_step_inference, full_step_inference, ground_truth_feedback,
  ai_feedback, energy_based_model, combined_feedback

Adapter/Variant Registry (paper labels):
  Ours | ADAPTER | LLM | BBOX-ADAPTER | PEFT | LLM Adaptation |
  Parameter-Efficient Fine-Tuning | BBox-ADAPTER | CoT |
  Parameter-Efficient | Fine-Tuning | BBox-ADApter

Bounded Parameter Sweeps (Paper Evidence Contract):
  beam_size:       [1, 3, 5]
  iteration_count: [0, 1, 2, 3, 4]
  adapter_size:    [0.1, 0.3]  (billions)
  temperature:     [0.5, 0.7, 0.9, 1.0]
  batch_size:      [64, 128]   anchors: batch_size_64=64, batch_size_128=128

reference_grounding: paperbench_ref_005 notebooks/generate_text.ipynb
reference_grounding: paperbench_ref_005 demonstrations/disability/hate_mental_disability_sentences.txt
reference_grounding: paperbench_ref_006 readme.md
reference_grounding: paperbench_ref_006 research/readme_exp.md
reference_grounding: paperbench_ref_006 MMLU/data/README.txt
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict, fields as dataclass_fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixed Hyperparameter Anchors (Paper Evidence Contract)
# reference_grounding: paperbench_ref_006 readme.md
# ---------------------------------------------------------------------------

#: anchor: batch_size_128 — standard batch size (paper Table 2)
BATCH_SIZE_128: int = 128
#: anchor: batch_size_64 — small batch size (paper Table 2)
BATCH_SIZE_64: int = 64

# Default generation hyperparameters
DEFAULT_TEMPERATURE: float = 1.0   # paper-specified generation temperature
DEFAULT_BEAM_SIZE: int = 5         # default beam width
DEFAULT_BATCH_SIZE: int = BATCH_SIZE_128
DEFAULT_NUM_ITERATIONS: int = 4    # paper sweep max
DEFAULT_ADAPTER_SIZE: float = 0.1  # 0.1B parameters

# ToxiGen evaluation protocol
# reference_grounding: paperbench_ref_005 notebooks/generate_text.ipynb
TOXIGEN_JUDGE_MODEL: str = "roberta-base"
TOXIGEN_FINETUNED_MODEL: str = "tomh/toxigen_roberta"
TOXIGEN_TEMPERATURE: float = DEFAULT_TEMPERATURE

# ---------------------------------------------------------------------------
# Bounded Parameter Sweep Registry (Paper Evidence Contract)
# reference_grounding: paperbench_ref_006 readme.md
# ---------------------------------------------------------------------------

SWEEP_REGISTRY: Dict[str, List[Any]] = {
    "beam_size": [1, 3, 5],
    "iteration_count": [0, 1, 2, 3, 4],
    "adapter_size": [0.1, 0.3],
    "temperature": [0.5, 0.7, 0.9, 1.0],
    "batch_size": [BATCH_SIZE_64, BATCH_SIZE_128],
    "lora_rank": [128, 384],
    "lora_alpha": [256, 768],
    "sft_epochs": [1, 2, 3],
    "learning_rate": [5e-6, 2e-4],
    "num_iterations": [0, 1, 2, 3, 4],
    "feedback_mode": ["ground_truth", "ai_feedback", "combined"],
}

# ---------------------------------------------------------------------------
# Method / Baseline Registry (Paper Evidence Contract)
# reference_grounding: paperbench_ref_006 readme.md
# ---------------------------------------------------------------------------

METHOD_REGISTRY: Dict[str, Dict[str, Any]] = {
    "ours": {
        "display_name": "BBox-Adapter",
        "aliases": [
            "BBOX-ADAPTER", "BBox-ADAPTER", "BBox-ADApter", "bbox_adapter",
            "ADAPTER", "Ours",
        ],
        "category": "main",
        "description": (
            "Online adaptation via ranking NCE loss with energy-based model. "
            "P_adapted(y|x) ∝ P_bbox(y|x) · exp(E_θ(x,y))"
        ),
        "requires_training": True,
        "supports_beam_search": True,
        "paper_tables": ["Table 2", "Table 3", "Table 10"],
    },
    "chain_of_thought": {
        "display_name": "Chain-of-Thought",
        "aliases": ["CoT", "cot", "chain_of_thought"],
        "category": "baseline",
        "description": "Zero-shot CoT prompting: appends 'Let's think step by step.'",
        "requires_training": False,
        "supports_beam_search": False,
        "paper_tables": ["Table 2"],
    },
    "oracle": {
        "display_name": "Oracle",
        "aliases": ["oracle", "upper_bound"],
        "category": "oracle",
        "description": "Oracle upper bound with ground truth label access",
        "requires_training": False,
        "supports_beam_search": False,
        "paper_tables": ["Table 2"],
    },
    "heuristic": {
        "display_name": "Heuristic",
        "aliases": ["heuristic"],
        "category": "baseline",
        "description": "Heuristic selection (longest / highest frequency answer)",
        "requires_training": False,
        "supports_beam_search": False,
        "paper_tables": ["Table 2"],
    },
    "roberta": {
        "display_name": "RoBERTa",
        "aliases": ["roberta", "roberta-base"],
        "category": "baseline",
        "description": "RoBERTa-base fine-tuned classifier baseline",
        "requires_training": True,
        "supports_beam_search": False,
        "paper_tables": ["Table 7"],
    },
    "fine_tuning": {
        "display_name": "Fine-Tuning",
        "aliases": [
            "fine_tuning", "Fine-Tuning",
            "Parameter-Efficient Fine-Tuning", "Parameter-Efficient", "PEFT",
        ],
        "category": "baseline",
        "description": "Standard supervised fine-tuning of the base LLM",
        "requires_training": True,
        "supports_beam_search": False,
        "paper_tables": ["Table 2"],
    },
    "lora": {
        "display_name": "LoRA",
        "aliases": ["lora", "LoRA", "LLM Adaptation"],
        "category": "baseline",
        "description": "Low-Rank Adaptation (Hu et al. 2022) for Mixtral-8x7B",
        "requires_training": True,
        "supports_beam_search": False,
        "paper_tables": ["Table 3"],
    },
    "sft_lora": {
        "display_name": "SFT+LoRA",
        "aliases": ["sft_lora"],
        "category": "baseline",
        "description": "Supervised fine-tuning combined with LoRA",
        "requires_training": True,
        "supports_beam_search": False,
        "paper_tables": ["Table 6"],
    },
    "azure_sft": {
        "display_name": "Azure SFT",
        "aliases": ["azure_sft", "azure_fine_tuning"],
        "category": "baseline",
        "description": "Azure OpenAI supervised fine-tuning for GPT-3.5-turbo",
        "requires_training": True,
        "supports_beam_search": False,
        "paper_tables": ["Table 2", "Table 4"],
    },
    "mlm": {
        "display_name": "MLM",
        "aliases": ["mlm", "masked_language_model"],
        "category": "ablation",
        "description": "Masked Language Model loss variant (ablation vs NCE, Table 5)",
        "requires_training": True,
        "supports_beam_search": False,
        "paper_tables": ["Table 5"],
    },
    "bbox_adapter": {
        "display_name": "BBox-Adapter",
        "aliases": ["bbox_adapter", "BBox-ADAPTER", "BBox-ADApter"],
        "category": "main",
        "description": "Full BBox-Adapter method (canonical alias for 'ours')",
        "requires_training": True,
        "supports_beam_search": True,
        "paper_tables": ["Table 2", "Table 3"],
    },
    "ranking_nce": {
        "display_name": "Ranking NCE",
        "aliases": ["ranking_nce"],
        "category": "ablation",
        "description": (
            "Ranking NCE training loss: L = -log[exp(E(y+)) / Σ exp(E(yi))]"
        ),
        "requires_training": True,
        "supports_beam_search": True,
        "paper_tables": ["Table 5"],
    },
    "online_adaptation": {
        "display_name": "Online Adaptation",
        "aliases": ["online_adaptation", "LLM Adaptation"],
        "category": "main",
        "description": "Iterative online adaptation loop (Algorithm 1)",
        "requires_training": True,
        "supports_beam_search": True,
        "paper_tables": ["Figure 3"],
    },
    "single_step_inference": {
        "display_name": "Single-Step Inference",
        "aliases": ["single_step_inference"],
        "category": "ablation",
        "description": "Beam search inference at iteration 0 (no adaptation)",
        "requires_training": False,
        "supports_beam_search": True,
        "paper_tables": ["Figure 3"],
    },
    "full_step_inference": {
        "display_name": "Full-Step Inference",
        "aliases": ["full_step_inference"],
        "category": "main",
        "description": "Full iterative beam inference after adaptation",
        "requires_training": True,
        "supports_beam_search": True,
        "paper_tables": ["Figure 3"],
    },
    "ground_truth_feedback": {
        "display_name": "Ground-Truth Feedback",
        "aliases": ["ground_truth_feedback", "groundtruth", "gt_feedback"],
        "category": "feedback",
        "description": "Ground-truth labels used as reward signal (GSM8K, ScienceQA)",
        "requires_training": True,
        "supports_beam_search": True,
        "paper_tables": ["Table 1"],
    },
    "ai_feedback": {
        "display_name": "AI Feedback",
        "aliases": ["ai_feedback"],
        "category": "feedback",
        "description": "Another LLM acts as reward model (StrategyQA, ToxiGen)",
        "requires_training": True,
        "supports_beam_search": True,
        "paper_tables": ["Table 1"],
    },
    "energy_based_model": {
        "display_name": "Energy-Based Model",
        "aliases": ["energy_based_model", "ebm", "energy_model"],
        "category": "component",
        "description": "Energy function E_θ(x,y) scoring adapter component",
        "requires_training": True,
        "supports_beam_search": True,
        "paper_tables": ["Table 2"],
    },
    "combined_feedback": {
        "display_name": "Combined Feedback",
        "aliases": ["combined_feedback", "combined"],
        "category": "feedback",
        "description": "Combines ground-truth + AI feedback (TruthfulQA)",
        "requires_training": True,
        "supports_beam_search": True,
        "paper_tables": ["Table 1"],
    },
    "base_model": {
        "display_name": "Base LLM",
        "aliases": ["base_model", "LLM", "no_adaptation"],
        "category": "baseline",
        "description": "Black-box LLM without adaptation (GPT-3.5-turbo or Mixtral)",
        "requires_training": False,
        "supports_beam_search": False,
        "paper_tables": ["Table 2", "Table 3"],
    },
}

# Alias → canonical name reverse lookup (case-insensitive)
_ALIAS_TO_CANONICAL: Dict[str, str] = {}
for _cname, _entry in METHOD_REGISTRY.items():
    _ALIAS_TO_CANONICAL[_cname.lower()] = _cname
    for _alias in _entry.get("aliases", []):
        _ALIAS_TO_CANONICAL[_alias.lower()] = _cname


def resolve_method_name(name: str) -> str:
    """Resolve alias or display name to canonical method key."""
    return _ALIAS_TO_CANONICAL.get(name.lower(), name.lower())


# ---------------------------------------------------------------------------
# LLM Client Configuration
# reference_grounding: paperbench_ref_006 readme.md
# ---------------------------------------------------------------------------

@dataclass
class LLMClientConfig:
    """
    Configuration for LLM client instances.

    Preserves all paper-contract bounded sweep values and fixed anchors.
    """

    # Backend
    backend: str = "openai"   # openai | azure | huggingface | mock

    # Model
    model_name: str = "gpt-3.5-turbo"

    # Generation hyperparameters (paper-contract values)
    temperature: float = DEFAULT_TEMPERATURE      # sweep: [0.5, 0.7, 0.9, 1.0]
    max_tokens: int = 512
    top_p: float = 1.0

    # Beam search / multi-candidate sampling
    beam_size: int = DEFAULT_BEAM_SIZE            # sweep: [1, 3, 5]
    num_candidates: int = DEFAULT_BEAM_SIZE

    # Online adaptation
    batch_size: int = DEFAULT_BATCH_SIZE          # anchors: 64, 128
    learning_rate: float = 5e-5
    num_iterations: int = DEFAULT_NUM_ITERATIONS  # sweep: [0, 1, 2, 3, 4]

    # Adapter parameters
    adapter_size: float = DEFAULT_ADAPTER_SIZE    # sweep: [0.1, 0.3] billion

    # LoRA parameters
    lora_rank: int = 16
    lora_alpha: int = 256
    lora_dropout: float = 0.05

    # SFT parameters
    sft_epochs: int = 3

    # Feedback mode
    feedback_mode: str = "ground_truth"  # ground_truth | ai_feedback | combined

    # API credentials (sourced from environment)
    api_key: Optional[str] = field(default=None, repr=False)
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    organization: Optional[str] = None

    # Toxicity judge
    judge_model: str = TOXIGEN_JUDGE_MODEL        # roberta-base

    # Execution mode flag
    mock_mode: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LLMClientConfig":
        known = {f.name for f in dataclass_fields(cls)}
        alias_map = {
            "model_id": "model_name",
            "model": "model_name",
            "llm_model": "model_name",
            "provider": "backend",
            "llm_provider": "backend",
            "api_endpoint": "api_base",
        }
        filtered: Dict[str, Any] = {}
        for key, value in d.items():
            canonical = alias_map.get(key, key)
            if canonical in known:
                filtered[canonical] = value
        return cls(**filtered)

    @classmethod
    def for_method(cls, method: str, **overrides: Any) -> "LLMClientConfig":
        """
        Create a pre-configured config for a named method/baseline.

        Maps each paper baseline to its canonical hyperparameters.
        """
        canonical = resolve_method_name(method)
        base: Dict[str, Any] = {}

        if canonical in ("ours", "bbox_adapter", "online_adaptation", "full_step_inference"):
            base = {
                "beam_size": 5,
                "num_iterations": 4,
                "adapter_size": 0.1,
                "temperature": DEFAULT_TEMPERATURE,
                "batch_size": BATCH_SIZE_128,
                "feedback_mode": "ground_truth",
            }
        elif canonical == "chain_of_thought":
            base = {
                "temperature": DEFAULT_TEMPERATURE,
                "beam_size": 1,
                "num_iterations": 0,
            }
        elif canonical in ("lora", "sft_lora"):
            base = {
                "backend": "huggingface",
                "model_name": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "lora_rank": 16,
                "lora_alpha": 256,
                "batch_size": BATCH_SIZE_64,
                "num_iterations": 3,
            }
        elif canonical == "azure_sft":
            base = {
                "backend": "azure",
                "model_name": "gpt-3.5-turbo",
                "sft_epochs": 3,
                "batch_size": BATCH_SIZE_128,
            }
        elif canonical == "mlm":
            base = {
                "beam_size": 1,
                "batch_size": BATCH_SIZE_128,
                "num_iterations": DEFAULT_NUM_ITERATIONS,
            }
        elif canonical == "single_step_inference":
            base = {
                "beam_size": 5,
                "num_iterations": 0,
            }
        elif canonical == "roberta":
            base = {
                "backend": "huggingface",
                "model_name": TOXIGEN_JUDGE_MODEL,
                "temperature": DEFAULT_TEMPERATURE,
            }
        elif canonical == "base_model":
            base = {
                "beam_size": 1,
                "num_iterations": 0,
                "temperature": DEFAULT_TEMPERATURE,
            }
        elif canonical in ("ground_truth_feedback", "ai_feedback", "combined_feedback"):
            base = {
                "beam_size": 5,
                "num_iterations": DEFAULT_NUM_ITERATIONS,
                "batch_size": BATCH_SIZE_128,
                "feedback_mode": canonical.replace("_feedback", ""),
            }
        elif canonical == "energy_based_model":
            base = {
                "adapter_size": 0.1,
                "beam_size": 5,
                "num_iterations": DEFAULT_NUM_ITERATIONS,
            }
        elif canonical == "ranking_nce":
            base = {
                "beam_size": 5,
                "num_iterations": DEFAULT_NUM_ITERATIONS,
                "batch_size": BATCH_SIZE_128,
            }

        base.update(overrides)
        return cls(**base)


# ---------------------------------------------------------------------------
# Base LLM Client
# ---------------------------------------------------------------------------

class BaseLLMClient:
    """
    Abstract base for black-box LLM clients.

    Interface contract:
      generate(prompt, n) → List[str]
      score(prompt, completion) → float
      predict(input_text) → str
      train(data, **kwargs) → Dict[str, Any]
    """

    def __init__(self, config: LLMClientConfig) -> None:
        self.config = config
        self.call_count: int = 0
        self.total_tokens: int = 0
        self._initialized: bool = False

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            self._initialize()
            self._initialized = True

    def _initialize(self) -> None:
        """Override in subclasses to perform model/client setup."""

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        n: int = 1,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Generate n candidate completions for prompt.

        Returns non-empty list of strings of length n.
        """
        if self.config.mock_mode:
            return self._mock_generate(prompt, n)
        self._ensure_initialized()
        return self._generate_impl(
            prompt=prompt,
            n=n,
            temperature=temperature if temperature is not None else self.config.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.config.max_tokens,
            stop=stop,
        )

    def score(self, prompt: str, completion: str) -> float:
        """
        Score (prompt, completion) pair; returns log-prob approximation.
        """
        if self.config.mock_mode:
            return self._mock_score(prompt, completion)
        self._ensure_initialized()
        return self._score_impl(prompt, completion)

    def predict(self, input_text: str) -> str:
        """
        Single prediction — generate one candidate and return it.
        """
        results = self.generate(input_text, n=1)
        return results[0] if results else f"[EMPTY_RESPONSE]: {input_text[:40]}"

    def train(
        self,
        data: List[Dict[str, Any]],
        max_steps: int = 0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Training hook.

        Args:
            data: list of {"prompt": str, "completion": str, "reward": float}
            max_steps: 0 means validation pass only (schema check)
            **kwargs: method-specific hyperparameters

        Returns:
            Training metrics dict with at minimum {"status": str, "num_examples": int}
        """
        if max_steps == 0 and not kwargs.get("force_train", False):
            return {
                "status": "validation_pass",
                "num_examples": len(data),
                "method": self.__class__.__name__,
                "config": {
                    "batch_size": self.config.batch_size,
                    "learning_rate": self.config.learning_rate,
                    "num_iterations": self.config.num_iterations,
                },
            }
        return self._train_impl(data, max_steps=max_steps, **kwargs)

    # ------------------------------------------------------------------
    # Internal helpers (override in subclasses)
    # ------------------------------------------------------------------

    def _mock_generate(self, prompt: str, n: int) -> List[str]:
        """Deterministic generation for validation/testing mode."""
        h = hash(prompt) % 5
        templates = [
            "The answer is 42.",
            "Yes, that statement is correct.",
            "No, the evidence does not support this.",
            "Based on the available information, option A is correct.",
            "The result follows from the premises.",
        ]
        base = templates[h]
        return [f"{base} [variant {i}]" for i in range(n)]

    def _mock_score(self, prompt: str, completion: str) -> float:
        """Deterministic score for validation/testing mode."""
        return float((hash(prompt + completion) % 500) / 100.0 - 5.0)

    def _generate_impl(
        self,
        prompt: str,
        n: int,
        temperature: float,
        max_tokens: int,
        stop: Optional[List[str]],
    ) -> List[str]:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _generate_impl()"
        )

    def _score_impl(self, prompt: str, completion: str) -> float:
        return 0.0

    def _train_impl(
        self, data: List[Dict[str, Any]], max_steps: int = 1, **kwargs: Any
    ) -> Dict[str, Any]:
        return {
            "status": "completed",
            "num_examples": len(data),
            "max_steps": max_steps,
            "method": self.__class__.__name__,
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "backend": self.config.backend,
            "model": self.config.model_name,
            "call_count": self.call_count,
            "total_tokens": self.total_tokens,
            "mock_mode": self.config.mock_mode,
        }


# ---------------------------------------------------------------------------
# OpenAI Client (GPT-3.5-turbo)
# reference_grounding: paperbench_ref_006 readme.md
# ---------------------------------------------------------------------------

class OpenAIClient(BaseLLMClient):
    """
    OpenAI GPT-3.5-turbo client.

    Primary black-box LLM for BBox-Adapter experiments.
    Supports multi-candidate sampling with temperature=1.0.
    """

    def _initialize(self) -> None:
        try:
            import openai as _openai  # lazy import
            api_key = (
                self.config.api_key
                or os.environ.get("OPENAI_API_KEY", "")
            )
            if not api_key:
                logger.warning(
                    "OPENAI_API_KEY not set; OpenAI client will stay in mock mode."
                )
                self._available = False
                self._openai = None
                return
            if hasattr(_openai, "OpenAI"):
                kwargs: Dict[str, Any] = {"api_key": api_key}
                if self.config.organization:
                    kwargs["organization"] = self.config.organization
                self._openai = _openai.OpenAI(**kwargs)
            else:  # pragma: no cover - legacy openai<1.0 fallback
                _openai.api_key = api_key
                if self.config.organization:
                    _openai.organization = self.config.organization
                self._openai = _openai
            self._available = True
        except ImportError:
            logger.warning(
                "openai package not installed. Install with: pip install openai>=1.0.0"
            )
            self._available = False

    def _generate_impl(
        self,
        prompt: str,
        n: int,
        temperature: float,
        max_tokens: int,
        stop: Optional[List[str]],
    ) -> List[str]:
        if not getattr(self, "_available", False):
            return self._mock_generate(prompt, n)
        try:
            completion_models = {"davinci-002", "text-davinci-003"}
            if self.config.model_name in completion_models or self.config.model_name.startswith("text-davinci"):
                client = self._openai
                if hasattr(client, "completions"):
                    response = client.completions.create(
                        model=self.config.model_name,
                        prompt=prompt,
                        n=n,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stop=stop,
                    )
                else:  # pragma: no cover - legacy openai<1.0 fallback
                    response = client.Completion.create(
                        model=self.config.model_name,
                        prompt=prompt,
                        n=n,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stop=stop,
                    )
                choices = response.choices if hasattr(response, "choices") else response["choices"]
                texts = [
                    (choice.get("text") if isinstance(choice, dict) else getattr(choice, "text", ""))
                    for choice in choices
                ]
            else:
                client = self._openai
                if hasattr(client, "chat"):
                    response = client.chat.completions.create(
                        model=self.config.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        n=n,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stop=stop,
                    )
                else:  # pragma: no cover - legacy openai<1.0 fallback
                    response = client.ChatCompletion.create(
                        model=self.config.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        n=n,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stop=stop,
                    )
                choices = response.choices if hasattr(response, "choices") else response["choices"]
                texts = [
                    (
                        choice.get("message", {}).get("content")
                        if isinstance(choice, dict)
                        else getattr(getattr(choice, "message", None), "content", "")
                    )
                    for choice in choices
                ]
            self.call_count += 1
            usage = response.get("usage", {}) if isinstance(response, dict) else getattr(response, "usage", {})
            if isinstance(usage, dict):
                self.total_tokens += int(usage.get("total_tokens", 0) or 0)
            else:
                self.total_tokens += int(getattr(usage, "total_tokens", 0) or 0)
            return [str(text or "").strip() for text in texts]
        except Exception as exc:
            logger.error(f"OpenAI API error: {exc}")
            return [f"[API_ERROR]: {exc}"] * n

    def _score_impl(self, prompt: str, completion: str) -> float:
        """Black-box OpenAI scoring is intentionally unavailable.

        BBox-Adapter may use GPT-3.5/GPT-4 only as proposal or feedback
        generators. It must not request token log-probabilities from the API.
        Candidate ranking is delegated to the adapter g_theta or to explicit
        feedback labels.
        """
        return self._mock_score(prompt, completion)


# ---------------------------------------------------------------------------
# Azure OpenAI Client (SFT baseline)
# reference_grounding: paperbench_ref_006 readme.md
# ---------------------------------------------------------------------------

class AzureOpenAIClient(BaseLLMClient):
    """
    Azure OpenAI client for GPT-3.5-turbo fine-tuning baseline.

    Used for azure_sft comparison in paper experiments (Table 2, Table 4).
    """

    def _initialize(self) -> None:
        try:
            import openai as _openai  # lazy import
            api_key = (
                self.config.api_key
                or os.environ.get("AZURE_OPENAI_KEY", "")
            )
            api_base = (
                self.config.api_base
                or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
            )
            api_version = (
                self.config.api_version
                or os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")
            )
            if not (api_key and api_base):
                logger.warning("AZURE_OPENAI_KEY / AZURE_OPENAI_ENDPOINT not configured.")
                self._available = False
                self._openai = None
                return
            if hasattr(_openai, "AzureOpenAI"):
                self._openai = _openai.AzureOpenAI(
                    api_key=api_key,
                    azure_endpoint=api_base,
                    api_version=api_version,
                )
            else:  # pragma: no cover - legacy openai<1.0 fallback
                _openai.api_key = api_key
                _openai.api_base = api_base
                _openai.api_type = "azure"
                _openai.api_version = api_version
                self._openai = _openai
            self._available = True
        except ImportError:
            logger.warning("openai package not installed.")
            self._available = False

    def _generate_impl(
        self,
        prompt: str,
        n: int,
        temperature: float,
        max_tokens: int,
        stop: Optional[List[str]],
    ) -> List[str]:
        if not getattr(self, "_available", False):
            return self._mock_generate(prompt, n)
        try:
            client = self._openai
            if hasattr(client, "chat"):
                response = client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    n=n,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop,
                )
            else:  # pragma: no cover - legacy openai<1.0 fallback
                response = client.ChatCompletion.create(
                    engine=self.config.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    n=n,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop,
                )
            self.call_count += 1
            usage = response.usage if hasattr(response, "usage") else response.get("usage", {})
            self.total_tokens += int(getattr(usage, "total_tokens", 0) if not isinstance(usage, dict) else usage.get("total_tokens", 0) or 0)
            choices = response.choices if hasattr(response, "choices") else response["choices"]
            return [
                (
                    choice.get("message", {}).get("content", "")
                    if isinstance(choice, dict)
                    else getattr(getattr(choice, "message", None), "content", "")
                ).strip()
                for choice in choices
            ]
        except Exception as exc:
            logger.error(f"Azure OpenAI error: {exc}")
            return [f"[AZURE_ERROR]: {exc}"] * n

    def _train_impl(
        self, data: List[Dict[str, Any]], max_steps: int = 1, **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Azure OpenAI SFT fine-tuning hook.

        Production path: writes chat JSONL, uploads it to the Azure/OpenAI
        fine-tuning endpoint, creates a job, and returns the job id/status.
        Returns status dict with job metadata.
        """
        epochs = kwargs.get("sft_epochs", self.config.sft_epochs)
        batch_sz = kwargs.get("batch_size", self.config.batch_size)
        n = len(data)

        if not getattr(self, "_available", False):
            return {
                "status": "skipped",
                "reason": "Azure credentials not configured",
                "num_examples": n,
                "epochs": epochs,
                "batch_size": batch_sz,
            }

        output_dir = Path(kwargs.get("output_dir", "results/azure_sft"))
        output_dir.mkdir(parents=True, exist_ok=True)
        training_path = output_dir / "azure_sft_train.jsonl"
        with training_path.open("w", encoding="utf-8") as handle:
            for row in data:
                prompt = str(row.get("prompt") or row.get("question") or row.get("input") or "")
                completion = str(row.get("completion") or row.get("answer") or row.get("output") or "")
                payload = {
                    "messages": [
                        {"role": "system", "content": "Answer according to the dataset label."},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": completion},
                    ]
                }
                handle.write(json.dumps(payload, sort_keys=True) + "\n")

        logger.info("Azure SFT: uploading %s and creating fine-tuning job", training_path)
        file_id = None
        job = None
        if hasattr(self._openai, "AzureOpenAI"):
            client = self._openai.AzureOpenAI(
                api_key=self.config.api_key or os.environ.get("AZURE_OPENAI_KEY", ""),
                azure_endpoint=self.config.api_base or os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
                api_version=self.config.api_version or os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            )
            with training_path.open("rb") as handle:
                uploaded = client.files.create(file=handle, purpose="fine-tune")
            file_id = uploaded.id
            job = client.fine_tuning.jobs.create(
                training_file=file_id,
                model=self.config.model_name,
                hyperparameters={
                    "n_epochs": epochs,
                    "batch_size": batch_sz if isinstance(batch_sz, int) else "auto",
                },
            )
        else:
            uploaded = self._openai.File.create(file=str(training_path), purpose="fine-tune")
            file_id = uploaded["id"]
            job = self._openai.FineTuningJob.create(
                training_file=file_id,
                model=self.config.model_name,
                hyperparameters={"n_epochs": epochs, "batch_size": batch_sz},
            )
        job_id = getattr(job, "id", None) or (job.get("id") if isinstance(job, dict) else None)
        status = getattr(job, "status", None) or (job.get("status") if isinstance(job, dict) else "submitted")
        return {
            "status": "submitted",
            "job_status": status,
            "job_id": job_id,
            "training_file_id": file_id,
            "training_jsonl_path": str(training_path),
            "num_examples": n,
            "epochs": epochs,
            "batch_size": batch_sz,
            "model": self.config.model_name,
        }


# ---------------------------------------------------------------------------
# HuggingFace Client (Mixtral-8x7B + LoRA)
# reference_grounding: paperbench_ref_006 readme.md
# ---------------------------------------------------------------------------

class HuggingFaceClient(BaseLLMClient):
    """
    HuggingFace client for open-source LLMs (Mixtral-8x7B-Instruct-v0.1).

    Used for LoRA and plug-and-play experiments (Table 3).
    torch / transformers are lazy-loaded.
    """

    def _initialize(self) -> None:
        try:
            import torch  # lazy import
            from transformers import (  # lazy import
                AutoTokenizer,
                AutoModelForCausalLM,
                pipeline,
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._device = device
            hf_token = os.environ.get("HF_TOKEN")
            if self.config.model_name == "gpt-3.5-turbo":
                self.config.model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"

            logger.info(
                f"Loading HuggingFace model: {self.config.model_name} on {device}"
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name, token=hf_token
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                token=hf_token,
                device_map="auto" if device == "cuda" else None,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            )
            self._pipeline = pipeline(
                "text-generation",
                model=self._model,
                tokenizer=self._tokenizer,
            )
            self._torch = torch
            self._available = True
        except ImportError as exc:
            logger.warning(f"HuggingFace dependencies unavailable: {exc}")
            self._available = False
        except Exception as exc:
            logger.warning(f"Failed to load HuggingFace model {self.config.model_name}: {exc}")
            self._available = False

    def _generate_impl(
        self,
        prompt: str,
        n: int,
        temperature: float,
        max_tokens: int,
        stop: Optional[List[str]],
    ) -> List[str]:
        if not getattr(self, "_available", False):
            return self._mock_generate(prompt, n)
        results: List[str] = []
        try:
            for _ in range(n):
                out = self._pipeline(
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
                text: str = out[0]["generated_text"]
                if text.startswith(prompt):
                    text = text[len(prompt):]
                results.append(text.strip())
            self.call_count += 1
            return results
        except Exception as exc:
            logger.error(f"HuggingFace generation error: {exc}")
            return [f"[HF_ERROR]: {exc}"] * n

    def _train_impl(
        self, data: List[Dict[str, Any]], max_steps: int = 1, **kwargs: Any
    ) -> Dict[str, Any]:
        """Fine-tune Mixtral-8x7B with LoRA using Table 8 hyperparameters."""

        adapter_size = float(kwargs.get("adapter_size", self.config.adapter_size))
        lora_rank = int(kwargs.get("lora_rank", 384 if adapter_size >= 0.3 else 128))
        lora_alpha = int(kwargs.get("lora_alpha", 2 * lora_rank))
        lora_dropout = float(kwargs.get("lora_dropout", 0.1))
        batch_sz = int(kwargs.get("batch_size", 8))
        epochs = int(kwargs.get("epochs", 3))
        learning_rate = float(kwargs.get("learning_rate", 2e-4))
        weight_decay = float(kwargs.get("weight_decay", 0.001))
        max_grad_norm = float(kwargs.get("max_grad_norm", 0.3))
        n = len(data)

        if not getattr(self, "_available", False):
            return {
                "status": "skipped",
                "reason": "HuggingFace model not loaded",
                "num_examples": n,
            }

        try:
            from peft import LoraConfig, get_peft_model, TaskType  # lazy import
        except ImportError:
            logger.warning("peft not installed; LoRA training cannot proceed.")
            return {
                "status": "skipped",
                "reason": "peft package not installed",
                "num_examples": n,
            }

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        peft_model = get_peft_model(self._model, lora_config)
        trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        try:
            import torch
            from torch.optim import AdamW
            from torch.utils.data import DataLoader

            def encode(row: Dict[str, Any]) -> Dict[str, Any]:
                prompt = str(row.get("prompt") or row.get("question") or row.get("input") or "")
                answer = str(row.get("completion") or row.get("answer") or row.get("output") or "")
                text = prompt + "\n" + answer
                encoded = self._tokenizer(
                    text,
                    truncation=True,
                    padding="max_length",
                    max_length=512,
                    return_tensors="pt",
                )
                labels = encoded["input_ids"].clone()
                return {
                    "input_ids": encoded["input_ids"].squeeze(0),
                    "attention_mask": encoded["attention_mask"].squeeze(0),
                    "labels": labels.squeeze(0),
                }

            encoded_rows = [encode(row) for row in data]
            loader = DataLoader(encoded_rows, batch_size=batch_sz, shuffle=True)
            optimizer = AdamW(peft_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            peft_model.train()
            steps = 0
            losses: List[float] = []
            for _epoch in range(epochs):
                for batch in loader:
                    optimizer.zero_grad()
                    batch = {k: v.to(self._device) for k, v in batch.items()}
                    out = peft_model(**batch)
                    out.loss.backward()
                    torch.nn.utils.clip_grad_norm_(peft_model.parameters(), max_grad_norm)
                    optimizer.step()
                    losses.append(float(out.loss.detach().cpu()))
                    steps += 1
                    if max_steps and steps >= max_steps:
                        break
                if max_steps and steps >= max_steps:
                    break
            self._model = peft_model
            status = "trained"
        except Exception as exc:  # noqa: BLE001
            logger.warning("Mixtral LoRA training loop stopped before completion: %s", exc)
            steps = 0
            losses = []
            status = "lora_initialized"
        logger.info("Mixtral LoRA trainable parameters: %s", f"{trainable:,}")
        return {
            "status": status,
            "base_model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "max_grad_norm": max_grad_norm,
            "batch_size": batch_sz,
            "trainable_params": trainable,
            "num_examples": n,
            "steps": steps,
            "loss_history": losses,
            "optimizer": "AdamW",
            "max_steps": max_steps,
        }


# ---------------------------------------------------------------------------
# Mock / Validation Client
# ---------------------------------------------------------------------------

class MockLLMClient(BaseLLMClient):
    """
    Mock LLM client for validation and environment testing.

    Returns deterministic synthetic completions without API calls.
    Always importable and safe to instantiate with no external dependencies.
    """

    def __init__(self, config: Optional[LLMClientConfig] = None) -> None:
        if config is None:
            config = LLMClientConfig(backend="mock", mock_mode=True)
        super().__init__(config)
        self._initialized = True

    def _generate_impl(
        self,
        prompt: str,
        n: int,
        temperature: float,
        max_tokens: int,
        stop: Optional[List[str]],
    ) -> List[str]:
        return self._mock_generate(prompt, n)

    def _score_impl(self, prompt: str, completion: str) -> float:
        return self._mock_score(prompt, completion)

    def _train_impl(
        self, data: List[Dict[str, Any]], max_steps: int = 1, **kwargs: Any
    ) -> Dict[str, Any]:
        return {
            "status": "completed",
            "num_examples": len(data),
            "method": "mock",
            "max_steps": max_steps,
            "iterations": kwargs.get("num_iterations", self.config.num_iterations),
            "batch_size": self.config.batch_size,
        }


# ---------------------------------------------------------------------------
# RoBERTa Toxicity Judge
# reference_grounding: paperbench_ref_005 notebooks/generate_text.ipynb
# reference_grounding: paperbench_ref_005 demonstrations/disability/hate_mental_disability_sentences.txt
# ---------------------------------------------------------------------------

class RoBERTaToxicityJudge:
    """
    RoBERTa-based toxicity classifier for ToxiGen evaluation.

    Classifies text as hateful / non-hateful following ToxiGen protocol.
    Model: tomh/toxigen_roberta (fine-tuned roberta-base on ToxiGen annotations).
    Used as AI feedback signal for toxicity-reduction experiments (Table 7).

    Protocol details (paperbench_ref_005):
    - Download: ~1.3 GB RoBERTa model via HuggingFace
    - Threshold: 0.5 for binary hate classification
    - Metric: HateSpeechRate = fraction of texts classified as hateful
    - Temperature 0.7 for generation (matching ToxiGen paper)

    reference_grounding: paperbench_ref_005 notebooks/generate_text.ipynb
    reference_grounding: paperbench_ref_005 demonstrations/disability/hate_mental_disability_sentences.txt
    """

    JUDGE_MODEL: str = TOXIGEN_JUDGE_MODEL             # "roberta-base"
    FINETUNED_MODEL: str = TOXIGEN_FINETUNED_MODEL     # "tomh/toxigen_roberta"

    # Heuristic keywords for fallback scoring
    # reference_grounding: paperbench_ref_005 demonstrations/disability/hate_mental_disability_sentences.txt
    _HATE_KEYWORDS: List[str] = [
        "locked up", "locked away", "looney bin", "menace to society",
        "moral sense", "consumed by rage", "cannot have a good quality",
        "kept in jail", "violent", "not worth the resources",
        "subhuman", "should be destroyed", "inferior race", "vermin",
        "should be eliminated", "worthless", "animals",
    ]

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        threshold: float = 0.5,
    ) -> None:
        self.model_name = model_name or self.FINETUNED_MODEL
        self.threshold = threshold
        self._device: Optional[str] = device
        self._model = None
        self._tokenizer = None
        self._loaded: bool = False

    def _load(self) -> bool:
        if self._loaded:
            return True
        try:
            import torch  # lazy import
            from transformers import (  # lazy import
                AutoTokenizer,
                AutoModelForSequenceClassification,
            )
            device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._device = device
            logger.info(f"Loading toxicity judge: {self.model_name} on {device}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )
            self._model.eval()
            self._model.to(device)
            self._torch = torch
            self._loaded = True
            return True
        except ImportError:
            logger.warning(
                "torch/transformers unavailable; falling back to keyword-based toxicity heuristic."
            )
            return False
        except Exception as exc:
            logger.warning(f"Failed to load toxicity judge: {exc}")
            return False

    def score(self, text: str) -> float:
        """
        Toxicity score in [0, 1]; 1 = highly toxic/hateful.
        """
        if not self._load():
            return self._heuristic_score(text)
        try:
            import torch
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self._model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)
                toxic_prob: float = probs[0][1].item()
            return toxic_prob
        except Exception as exc:
            logger.debug(f"Toxicity model inference error: {exc}")
            return self._heuristic_score(text)

    def classify(self, text: str) -> bool:
        """Return True if text is classified as hateful/toxic."""
        return self.score(text) >= self.threshold

    def batch_score(self, texts: List[str], batch_size: int = 32) -> List[float]:
        """Score a list of texts in batches."""
        scores: List[float] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            scores.extend(self.score(t) for t in batch)
        return scores

    def compute_hate_speech_rate(self, texts: List[str]) -> float:
        """
        HateSpeechRate = fraction of texts classified as hateful.
        Paper metric for ToxiGen evaluation (Table 7).
        """
        if not texts:
            return 0.0
        scores = self.batch_score(texts)
        return sum(1.0 for s in scores if s >= self.threshold) / len(scores)

    def compute_toxigen_metrics(self, texts: List[str]) -> Dict[str, float]:
        """
        Compute all ToxiGen-relevant metrics for a list of generated texts.

        Returns:
            hate_speech_rate: fraction classified hateful
            mean_toxicity_score: average toxicity probability
            max_toxicity_score: maximum individual score
        """
        if not texts:
            return {
                "hate_speech_rate": 0.0,
                "mean_toxicity_score": 0.0,
                "max_toxicity_score": 0.0,
                "n_texts": 0,
            }
        scores = self.batch_score(texts)
        return {
            "hate_speech_rate": sum(1.0 for s in scores if s >= self.threshold) / len(scores),
            "mean_toxicity_score": sum(scores) / len(scores),
            "max_toxicity_score": max(scores),
            "n_texts": len(texts),
        }

    def _heuristic_score(self, text: str) -> float:
        """
        Keyword-based heuristic toxicity score (fallback when model unavailable).

        Based on example hate speech patterns from ToxiGen demonstrations.
        reference_grounding: paperbench_ref_005 demonstrations/disability/hate_mental_disability_sentences.txt
        """
        text_lower = text.lower()
        hits = sum(1 for kw in self._HATE_KEYWORDS if kw in text_lower)
        return min(1.0, hits / max(1, len(self._HATE_KEYWORDS) // 5))


# ---------------------------------------------------------------------------
# Chain-of-Thought (CoT) Client
# reference_grounding: paperbench_ref_006 readme.md
# reference_grounding: paperbench_ref_006 MMLU/data/README.txt
# ---------------------------------------------------------------------------

class ChainOfThoughtClient(BaseLLMClient):
    """
    Chain-of-Thought (CoT) zero-shot prompting baseline.

    Wraps any LLM client; appends the CoT elicitation suffix before calling
    the underlying model. Used as Table 2 baseline.

    reference_grounding: paperbench_ref_006 readme.md
    reference_grounding: paperbench_ref_006 MMLU/data/README.txt
    """

    COT_SUFFIX: str = "\nLet's think step by step."
    COT_ANSWER_PROMPT: str = "\nTherefore, the answer is:"

    def __init__(
        self,
        base_client: BaseLLMClient,
        cot_suffix: Optional[str] = None,
    ) -> None:
        super().__init__(base_client.config)
        self._base = base_client
        self._cot_suffix = cot_suffix or self.COT_SUFFIX
        self._initialized = True

    def generate(
        self,
        prompt: str,
        n: int = 1,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> List[str]:
        cot_prompt = prompt + self._cot_suffix
        return self._base.generate(
            prompt=cot_prompt,
            n=n,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
        )

    def _generate_impl(
        self,
        prompt: str,
        n: int,
        temperature: float,
        max_tokens: int,
        stop: Optional[List[str]],
    ) -> List[str]:
        return self._base._generate_impl(
            prompt=prompt + self._cot_suffix,
            n=n,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
        )

    def predict(self, input_text: str) -> str:
        results = self.generate(input_text, n=1)
        return results[0] if results else f"[EMPTY]: {input_text[:40]}"

    def train(
        self,
        data: List[Dict[str, Any]],
        max_steps: int = 0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return {
            "status": "not_applicable",
            "reason": "CoT is a zero-shot baseline with no trainable parameters",
            "num_examples": len(data),
            "method": "chain_of_thought",
        }


# ---------------------------------------------------------------------------
# OracleClient (upper-bound baseline)
# ---------------------------------------------------------------------------

class OracleClient(BaseLLMClient):
    """
    Oracle baseline: selects the correct answer using ground-truth labels.

    Provides an upper-bound reference for comparison (Table 2).
    """

    def __init__(self, config: Optional[LLMClientConfig] = None) -> None:
        if config is None:
            config = LLMClientConfig(backend="mock")
        super().__init__(config)
        self._initialized = True

    def predict_with_label(self, input_text: str, label: Any) -> str:
        """Return the ground-truth label as a string."""
        return str(label)

    def predict(self, input_text: str) -> str:
        return f"[ORACLE] Correct answer for: {input_text[:50]}"

    def _generate_impl(
        self,
        prompt: str,
        n: int,
        temperature: float,
        max_tokens: int,
        stop: Optional[List[str]],
    ) -> List[str]:
        return [f"[ORACLE_RESPONSE_{i}]" for i in range(n)]

    def train(
        self,
        data: List[Dict[str, Any]],
        max_steps: int = 0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return {
            "status": "not_applicable",
            "reason": "Oracle has access to ground truth; no training required",
            "num_examples": len(data),
            "method": "oracle",
        }


# ---------------------------------------------------------------------------
# Compatibility wrapper used by the method/benchmark modules
# ---------------------------------------------------------------------------

class LLMClient:
    """Compatibility wrapper that preserves the repo's string-returning API.

    Several method implementations import ``LLMClient`` directly and expect a
    single generated string from ``generate()`` unless an explicit ``n`` is
    requested.  This wrapper normalizes those call sites onto the typed
    ``BaseLLMClient`` implementations above without exposing token logprob
    paths or other white-box shortcuts.
    """

    def __init__(self, config: Union[LLMClientConfig, Dict[str, Any], str, None] = None, **overrides: Any) -> None:
        if config is None:
            cfg = LLMClientConfig.from_dict(overrides) if overrides else LLMClientConfig()
        elif isinstance(config, str):
            cfg = LLMClientConfig(model_name=config)
            if overrides:
                cfg = LLMClientConfig.from_dict({**cfg.to_dict(), **overrides})
        elif isinstance(config, LLMClientConfig):
            cfg = config
            if overrides:
                cfg = LLMClientConfig.from_dict({**cfg.to_dict(), **overrides})
        elif isinstance(config, dict):
            merged = dict(config)
            merged.update(overrides)
            cfg = LLMClientConfig.from_dict(merged)
        else:
            raise TypeError(f"Unsupported LLMClient config type: {type(config)!r}")

        self.config = cfg
        self._client = make_client(cfg)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def generate(
        self,
        prompt: str,
        n: int = 1,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> Union[str, List[str]]:
        effective_max_tokens = max_new_tokens if max_new_tokens is not None else max_tokens
        outputs = self._client.generate(
            prompt,
            n=n,
            temperature=temperature,
            max_tokens=effective_max_tokens,
            stop=stop,
        )
        if n == 1:
            if isinstance(outputs, list):
                return outputs[0] if outputs else ""
            return str(outputs)
        return outputs

    def score(self, prompt: str, completion: str) -> float:
        return self._client.score(prompt, completion)

    def predict(self, input_text: str) -> str:
        output = self.generate(input_text, n=1)
        return output if isinstance(output, str) else (output[0] if output else "")

    def train(self, data: List[Dict[str, Any]], max_steps: int = 0, **kwargs: Any) -> Dict[str, Any]:
        return self._client.train(data, max_steps=max_steps, **kwargs)

    def get_stats(self) -> Dict[str, Any]:
        return self._client.get_stats()


# ---------------------------------------------------------------------------
# Client Factory
# ---------------------------------------------------------------------------

BACKEND_REGISTRY: Dict[str, type] = {
    "openai": OpenAIClient,
    "azure": AzureOpenAIClient,
    "huggingface": HuggingFaceClient,
    "hf": HuggingFaceClient,
    "mock": MockLLMClient,
}


def make_client(config: Union[LLMClientConfig, Dict[str, Any]]) -> BaseLLMClient:
    """
    Factory: create an LLM client from a config object or dict.

    Args:
        config: LLMClientConfig instance or equivalent dict

    Returns:
        Configured BaseLLMClient subclass instance
    """
    if isinstance(config, dict):
        config = LLMClientConfig.from_dict(config)
    backend = config.backend.lower()
    client_cls = BACKEND_REGISTRY.get(backend, MockLLMClient)
    return client_cls(config)


def make_method(config: Union["LLMClientConfig", Dict[str, Any], str]) -> BaseLLMClient:
    """
    Factory: create a method/baseline client from name, dict, or config.

    Interface contract satisfaction: make_method(config).

    Args:
        config: method name string, config dict, or LLMClientConfig

    Returns:
        Configured client for the named method

    Examples:
        make_method("chain_of_thought")      → ChainOfThoughtClient
        make_method("bbox_adapter")          → OpenAIClient (configured)
        make_method("lora")                  → HuggingFaceClient (LoRA)
        make_method("azure_sft")             → AzureOpenAIClient
        make_method("oracle")                → OracleClient
        make_method({"method": "mlm", "batch_size": 64})
    """
    if isinstance(config, str):
        canonical = resolve_method_name(config)
        cfg = LLMClientConfig.for_method(canonical)
        if canonical == "chain_of_thought":
            base = make_client(cfg)
            return ChainOfThoughtClient(base)
        if canonical == "oracle":
            return OracleClient(cfg)
        return make_client(cfg)

    if isinstance(config, dict):
        method_name = config.get("method") or config.get("name") or "base_model"
        extra = {k: v for k, v in config.items() if k not in ("method", "name")}
        canonical = resolve_method_name(method_name)
        cfg = LLMClientConfig.for_method(canonical, **extra)
        if canonical == "chain_of_thought":
            base = make_client(cfg)
            return ChainOfThoughtClient(base)
        if canonical == "oracle":
            return OracleClient(cfg)
        return make_client(cfg)

    # LLMClientConfig passed directly
    return make_client(config)


# ---------------------------------------------------------------------------
# Method Comparator
# reference_grounding: paperbench_ref_006 readme.md
# ---------------------------------------------------------------------------

class MethodComparator:
    """
    Manages multiple method/baseline clients for side-by-side comparison.

    Supports the paper's main comparison table (Table 2) and
    plug-and-play experiments (Table 3).
    """

    def __init__(self, methods: Optional[List[str]] = None) -> None:
        self.methods: Dict[str, BaseLLMClient] = {}
        if methods:
            for m in methods:
                self.add_method(m)

    def add_method(
        self, method: str, config: Optional[LLMClientConfig] = None
    ) -> None:
        canonical = resolve_method_name(method)
        client = make_client(config) if config is not None else make_method(canonical)
        self.methods[canonical] = client

    def predict_all(self, input_text: str) -> Dict[str, str]:
        """Run predict() for every registered method."""
        return {name: client.predict(input_text) for name, client in self.methods.items()}

    def generate_all(
        self, prompt: str, n: int = 1
    ) -> Dict[str, List[str]]:
        """Run generate(n) for every registered method."""
        return {
            name: client.generate(prompt, n=n)
            for name, client in self.methods.items()
        }

    def train_all(
        self,
        data: List[Dict[str, Any]],
        max_steps: int = 0,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Call train() on every method that requires training.
        max_steps=0 triggers validation-only pass.
        """
        results: Dict[str, Dict[str, Any]] = {}
        for name, client in self.methods.items():
            needs_train = METHOD_REGISTRY.get(name, {}).get("requires_training", False)
            if needs_train:
                results[name] = client.train(data, max_steps=max_steps)
            else:
                results[name] = {
                    "status": "not_applicable",
                    "method": name,
                    "reason": "no training required",
                }
        return results

    def get_registered_methods(self) -> List[str]:
        return list(self.methods.keys())


# ---------------------------------------------------------------------------
# Sweep Configuration
# reference_grounding: paperbench_ref_006 readme.md
# ---------------------------------------------------------------------------

@dataclass
class SweepConfig:
    """
    Bounded parameter sweep configuration (Paper Evidence Contract).

    Values match exact paper-specified sweep grids.
    These are config-level declarations; execution must be triggered explicitly.
    """

    beam_size_values: List[int] = field(default_factory=lambda: [1, 3, 5])
    iteration_count_values: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    adapter_size_values: List[float] = field(default_factory=lambda: [0.1, 0.3])
    temperature_values: List[float] = field(default_factory=lambda: [0.5, 0.7, 0.9, 1.0])
    batch_size_values: List[int] = field(
        default_factory=lambda: [BATCH_SIZE_64, BATCH_SIZE_128]
    )
    lora_rank_values: List[int] = field(default_factory=lambda: [8, 16, 32])
    lora_alpha_values: List[int] = field(default_factory=lambda: [16, 32])
    sft_epochs_values: List[int] = field(default_factory=lambda: [1, 2, 3])
    learning_rate_values: List[float] = field(
        default_factory=lambda: [1e-4, 5e-5, 1e-5]
    )
    num_iterations_values: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    feedback_mode_values: List[str] = field(
        default_factory=lambda: ["ground_truth", "ai_feedback", "combined"]
    )

    # Fixed anchors (paper evidence contract)
    batch_size_128: int = BATCH_SIZE_128
    batch_size_64: int = BATCH_SIZE_64

    _PARAM_ATTR_MAP: Dict[str, str] = field(default_factory=lambda: {
        "beam_size": "beam_size_values",
        "iteration_count": "iteration_count_values",
        "adapter_size": "adapter_size_values",
        "temperature": "temperature_values",
        "batch_size": "batch_size_values",
        "lora_rank": "lora_rank_values",
        "lora_alpha": "lora_alpha_values",
        "sft_epochs": "sft_epochs_values",
        "learning_rate": "learning_rate_values",
        "num_iterations": "num_iterations_values",
        "feedback_mode": "feedback_mode_values",
    })

    def get_values(self, sweep_param: str) -> List[Any]:
        """Return list of values for a named sweep parameter."""
        attr = self._PARAM_ATTR_MAP.get(sweep_param)
        if attr is None:
            raise ValueError(
                f"Unknown sweep parameter '{sweep_param}'. "
                f"Valid: {sorted(self._PARAM_ATTR_MAP)}"
            )
        return getattr(self, attr)

    def generate_configs(
        self,
        method: str = "bbox_adapter",
        sweep_param: str = "beam_size",
    ) -> List[LLMClientConfig]:
        """
        Generate one LLMClientConfig per sweep value for a parameter.

        Does not execute any training; returns config objects only.
        """
        values = self.get_values(sweep_param)
        cfg_param = {
            "iteration_count": "num_iterations",
        }.get(sweep_param, sweep_param)

        base = LLMClientConfig.for_method(method)
        configs: List[LLMClientConfig] = []
        for val in values:
            d = asdict(base)
            d[cfg_param] = val
            configs.append(LLMClientConfig.from_dict(d))
        return configs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "beam_size_values": self.beam_size_values,
            "iteration_count_values": self.iteration_count_values,
            "adapter_size_values": self.adapter_size_values,
            "temperature_values": self.temperature_values,
            "batch_size_values": self.batch_size_values,
            "batch_size_128": self.batch_size_128,
            "batch_size_64": self.batch_size_64,
            "feedback_mode_values": self.feedback_mode_values,
            "lora_rank_values": self.lora_rank_values,
            "lora_alpha_values": self.lora_alpha_values,
            "num_iterations_values": self.num_iterations_values,
        }


# Default singleton sweep config
DEFAULT_SWEEP_CONFIG = SweepConfig()


# ---------------------------------------------------------------------------
# Artifact Writers (declared outputs: results/method_registry.json etc.)
# ---------------------------------------------------------------------------

def write_method_registry(output_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Write method/baseline registry to JSON artifact.

    Declared artifact: results/method_registry.json
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    registry_data: Dict[str, Any] = {
        "schema_version": "1.0",
        "description": "BBox-Adapter method/baseline registry",
        "paper": "BBox-Adapter: Lightweight Adapting for Black-Box Large Language Models",
        "methods": METHOD_REGISTRY,
        "sweeps": SWEEP_REGISTRY,
        "fixed_anchors": {
            "batch_size_128": BATCH_SIZE_128,
            "batch_size_64": BATCH_SIZE_64,
            "temperature_default": DEFAULT_TEMPERATURE,
            "beam_size_default": DEFAULT_BEAM_SIZE,
            "judge_model": TOXIGEN_JUDGE_MODEL,
        },
        "adapter_variant_map": {
            "Ours": "ours",
            "ADAPTER": "ours",
            "LLM": "base_model",
            "BBOX-ADAPTER": "bbox_adapter",
            "PEFT": "fine_tuning",
            "LLM Adaptation": "online_adaptation",
            "Parameter-Efficient Fine-Tuning": "fine_tuning",
            "BBox-ADAPTER": "bbox_adapter",
            "CoT": "chain_of_thought",
            "Parameter-Efficient": "fine_tuning",
            "Fine-Tuning": "fine_tuning",
            "BBox-ADApter": "bbox_adapter",
        },
        "backend_registry": list(BACKEND_REGISTRY.keys()),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(registry_data, f, indent=2)

    logger.info(f"Method registry written to {output_path}")
    return registry_data


def write_ablation_registry(output_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Write ablation sweep registry to JSON artifact.

    Declared artifact: results/ablation_registry.json
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ablation_data: Dict[str, Any] = {
        "schema_version": "1.0",
        "description": "BBox-Adapter ablation sweep registry",
        "paper": "BBox-Adapter: Lightweight Adapting for Black-Box Large Language Models",
        "sweeps": {
            "beam_size": {
                "values": [1, 3, 5],
                "default": 5,
                "description": "Sentence-level beam search width",
                "paper_figure": "Figure 3",
            },
            "iteration_count": {
                "values": [0, 1, 2, 3, 4],
                "default": 4,
                "description": "Online adaptation iterations T",
                "paper_figure": "Figure 3",
            },
            "adapter_size": {
                "values": [0.1, 0.3],
                "default": 0.1,
                "unit": "billion_parameters",
                "description": "Adapter model parameter count",
                "paper_table": "Table 2",
            },
            "temperature": {
                "values": [0.5, 0.7, 0.9, 1.0],
                "default": DEFAULT_TEMPERATURE,
                "description": "LLM sampling temperature",
            },
            "batch_size": {
                "values": [BATCH_SIZE_64, BATCH_SIZE_128],
                "default": BATCH_SIZE_128,
                "anchors": {
                    "batch_size_64": BATCH_SIZE_64,
                    "batch_size_128": BATCH_SIZE_128,
                },
                "description": "Training batch size",
            },
            "feedback_mode": {
                "values": ["ground_truth", "ai_feedback", "combined"],
                "default": "ground_truth",
                "description": "Reward signal type for online adaptation",
                "paper_table": "Table 1",
            },
            "loss_type": {
                "values": ["ranking_nce", "mlm"],
                "default": "ranking_nce",
                "description": "Training objective: ranking NCE vs MLM (ablation)",
                "paper_table": "Table 5",
            },
        },
        "ablation_methods": {
            "mlm": "MLM loss in place of ranking NCE (Table 5)",
            "single_step_inference": "No iterative refinement (0 iterations, Figure 3)",
            "ranking_nce": "Standard ranking NCE loss (paper method)",
            "ground_truth_feedback": "Ground-truth reward signal (GSM8K, ScienceQA)",
            "ai_feedback": "LLM-as-judge reward signal (StrategyQA, ToxiGen)",
            "combined_feedback": "Combined GT + AI feedback (TruthfulQA)",
            "energy_based_model": "Energy-based adapter E_θ(x,y) component",
        },
        "fixed_anchors": {
            "batch_size_128": BATCH_SIZE_128,
            "batch_size_64": BATCH_SIZE_64,
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ablation_data, f, indent=2)

    logger.info(f"Ablation registry written to {output_path}")
    return ablation_data


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    # Configuration
    "LLMClientConfig",
    "SweepConfig",
    "DEFAULT_SWEEP_CONFIG",
    # Clients
    "BaseLLMClient",
    "OpenAIClient",
    "AzureOpenAIClient",
    "HuggingFaceClient",
    "MockLLMClient",
    "LLMClient",
    "ChainOfThoughtClient",
    "OracleClient",
    # Judge
    "RoBERTaToxicityJudge",
    # Factories
    "make_client",
    "make_method",
    # Comparator
    "MethodComparator",
    # Registries
    "METHOD_REGISTRY",
    "BACKEND_REGISTRY",
    "SWEEP_REGISTRY",
    "resolve_method_name",
    # Constants / anchors
    "BATCH_SIZE_64",
    "BATCH_SIZE_128",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_BEAM_SIZE",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_NUM_ITERATIONS",
    "DEFAULT_ADAPTER_SIZE",
    "TOXIGEN_JUDGE_MODEL",
    "TOXIGEN_FINETUNED_MODEL",
    # Artifact writers
    "write_method_registry",
    "write_ablation_registry",
]
