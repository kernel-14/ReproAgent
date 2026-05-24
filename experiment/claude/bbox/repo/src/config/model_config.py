#!/usr/bin/env python3
"""
BBox-Adapter Model Configuration

Comprehensive configuration registry for BBox-Adapter paper reproduction.
Exposes:
  - Method/baseline selector registry with canonical IDs and aliases
  - Bounded parameter sweep configurations (paper-derived)
  - Fixed hyperparameter anchors (batch_size_128, batch_size_64)
  - Model configuration dataclasses (adapter, training, inference)
  - Factory functions for creating experiment configurations

Paper: BBox-Adapter: Lightweight Adapting for Black-Box Large Language Models

Reference grounding: paperbench_ref_002 src/models/qa/transformer_qa.py
  (forward pass: question_with_context encoding, yes_no_span, answer_span handling)
Reference grounding: paperbench_ref_005 toxigen/alice.py
  (sentence-level beam search with weights=[.5,.5], BeamHypotheses, num_beams)
Reference grounding: paperbench_ref_006 MMLU/data/README.txt
  (dev/val/test split protocol, few-shot learning setup, auxiliary training data)
Reference grounding: paperbench_ref_006 MMLU/gpt_3.5_turbo_college_medicine.ipynb
  (CoT prompting: "Let's think step by step", multiple-choice answer extraction)

Method Registry (Paper Evidence Contract):
  ours, chain_of_thought, oracle, heuristic, roberta, fine_tuning, lora,
  sft_lora, azure_sft, mlm, bbox_adapter, ranking_nce, online_adaptation,
  single_step_inference, full_step_inference, ground_truth_feedback,
  ai_feedback, energy_based_model, combined_feedback

Variant Aliases (Paper Evidence Contract):
  Ours | ADAPTER | LLM | BBOX-ADAPTER | PEFT | LLM Adaptation |
  Parameter-Efficient Fine-Tuning | BBox-ADAPTER | CoT |
  Parameter-Efficient | Fine-Tuning | BBox-ADApter

Sweep Registry (Paper Evidence Contract):
  beam_size: [1, 3, 5]
  iteration_count: [0, 1, 2, 3, 4]
  adapter_size: [0.1, 0.3]
  temperature: [0.3, 0.5, 0.7, 0.9, 1.0]
  batch_size: [64, 128]

Fixed Hyperparameter Anchors:
  batch_size_128 = 128   (anchor: standard training batch)
  batch_size_64  = 64    (anchor: small training batch)
"""

from __future__ import annotations

import copy
import logging
import os
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# =========================================================================
# Fixed Hyperparameter Anchors (Paper Evidence Contract)
# Exact named anchors preserved: batch_size_128, batch_size_64
# reference_grounding: paperbench_ref_002 src/models/qa/transformer_qa.py
# =========================================================================

#: anchor: batch_size_128 — standard training batch size (paper Table 2, Table 10)
batch_size_128: int = 128

#: anchor: batch_size_64 — smaller training / eval batch size (paper ablation)
batch_size_64: int = 64


# =========================================================================
# Method / Baseline Selector Registry (Paper Evidence Contract)
# reference_grounding: paperbench_ref_006 MMLU/data/README.txt
# =========================================================================

class MethodID(str, Enum):
    """
    Canonical method/baseline identifiers for BBox-Adapter paper experiments.

    Covers all methods from Tables 2–10 and ablation sections of the paper.
    String values are the canonical snake_case selectors used in configs
    and experiment matrices.
    """

    # ---- Core paper contribution ----------------------------------------
    OURS = "ours"
    BBOX_ADAPTER = "bbox_adapter"
    ENERGY_BASED_MODEL = "energy_based_model"
    RANKING_NCE = "ranking_nce"
    ONLINE_ADAPTATION = "online_adaptation"

    # ---- Black-box LLM baselines ----------------------------------------
    CHAIN_OF_THOUGHT = "chain_of_thought"
    ORACLE = "oracle"
    HEURISTIC = "heuristic"

    # ---- White-box / PEFT baselines -------------------------------------
    ROBERTA = "roberta"
    FINE_TUNING = "fine_tuning"
    LORA = "lora"
    SFT_LORA = "sft_lora"
    AZURE_SFT = "azure_sft"
    MLM = "mlm"

    # ---- Inference modes ------------------------------------------------
    SINGLE_STEP_INFERENCE = "single_step_inference"
    FULL_STEP_INFERENCE = "full_step_inference"

    # ---- Feedback / training signal modes --------------------------------
    GROUND_TRUTH_FEEDBACK = "ground_truth_feedback"
    AI_FEEDBACK = "ai_feedback"
    COMBINED_FEEDBACK = "combined_feedback"


# Complete list of canonical method IDs (for validation and iteration)
ALL_METHOD_IDS: List[str] = [m.value for m in MethodID]

# -------------------------------------------------------------------------
# Alias map: paper display names / table labels → canonical MethodID
# reference_grounding: paperbench_ref_006 readme.md
# -------------------------------------------------------------------------
METHOD_ALIAS_MAP: Dict[str, str] = {
    # Paper-style display names used in tables and figures
    "Ours": MethodID.OURS,
    "ADAPTER": MethodID.BBOX_ADAPTER,
    "LLM": MethodID.CHAIN_OF_THOUGHT,
    "BBOX-ADAPTER": MethodID.BBOX_ADAPTER,
    "BBox-ADAPTER": MethodID.BBOX_ADAPTER,
    "BBox-ADApter": MethodID.BBOX_ADAPTER,
    "PEFT": MethodID.LORA,
    "LLM Adaptation": MethodID.ONLINE_ADAPTATION,
    "Parameter-Efficient Fine-Tuning": MethodID.LORA,
    "Parameter-Efficient": MethodID.LORA,
    "Fine-Tuning": MethodID.FINE_TUNING,
    "CoT": MethodID.CHAIN_OF_THOUGHT,
    # Canonical snake_case forms (passthrough)
    "ours": MethodID.OURS,
    "chain_of_thought": MethodID.CHAIN_OF_THOUGHT,
    "oracle": MethodID.ORACLE,
    "heuristic": MethodID.HEURISTIC,
    "roberta": MethodID.ROBERTA,
    "fine_tuning": MethodID.FINE_TUNING,
    "lora": MethodID.LORA,
    "sft_lora": MethodID.SFT_LORA,
    "azure_sft": MethodID.AZURE_SFT,
    "mlm": MethodID.MLM,
    "bbox_adapter": MethodID.BBOX_ADAPTER,
    "ranking_nce": MethodID.RANKING_NCE,
    "online_adaptation": MethodID.ONLINE_ADAPTATION,
    "single_step_inference": MethodID.SINGLE_STEP_INFERENCE,
    "full_step_inference": MethodID.FULL_STEP_INFERENCE,
    "ground_truth_feedback": MethodID.GROUND_TRUTH_FEEDBACK,
    "ai_feedback": MethodID.AI_FEEDBACK,
    "energy_based_model": MethodID.ENERGY_BASED_MODEL,
    "combined_feedback": MethodID.COMBINED_FEEDBACK,
}


def resolve_method(name: str) -> str:
    """
    Resolve any method name or paper alias to its canonical MethodID value.

    Args:
        name: Method name variant — canonical, display, or alias form.

    Returns:
        Canonical MethodID string value (e.g. ``"bbox_adapter"``).

    Raises:
        ValueError: If the name cannot be resolved to a known method.
    """
    if name in METHOD_ALIAS_MAP:
        return METHOD_ALIAS_MAP[name]
    # Normalise and retry (handles dash/space → underscore, case insensitive)
    normalised = name.lower().replace("-", "_").replace(" ", "_")
    for alias, method_id in METHOD_ALIAS_MAP.items():
        if alias.lower().replace("-", "_").replace(" ", "_") == normalised:
            return method_id
    available = sorted({m.value for m in MethodID})
    raise ValueError(
        f"Unknown method '{name}'. Available canonical methods: {available}"
    )


def is_bbox_adapter_method(name: str) -> bool:
    """Return True if ``name`` resolves to a BBox-Adapter core method."""
    try:
        resolved = resolve_method(name)
    except ValueError:
        return False
    core_ids = {
        MethodID.OURS,
        MethodID.BBOX_ADAPTER,
        MethodID.ENERGY_BASED_MODEL,
        MethodID.RANKING_NCE,
        MethodID.ONLINE_ADAPTATION,
    }
    return resolved in core_ids


def is_baseline_method(name: str) -> bool:
    """Return True if ``name`` resolves to a comparison baseline."""
    try:
        resolved = resolve_method(name)
    except ValueError:
        return False
    baseline_ids = {
        MethodID.CHAIN_OF_THOUGHT,
        MethodID.ORACLE,
        MethodID.HEURISTIC,
        MethodID.ROBERTA,
        MethodID.FINE_TUNING,
        MethodID.LORA,
        MethodID.SFT_LORA,
        MethodID.AZURE_SFT,
        MethodID.MLM,
    }
    return resolved in baseline_ids


# =========================================================================
# Sweep Parameter Registry (Paper Evidence Contract)
# reference_grounding: paperbench_ref_005 toxigen/alice.py  (num_beams=10)
# =========================================================================

#: beam_size sweep values (paper Figure 3, Table 8 ablation)
BEAM_SIZE_SWEEP: List[int] = [1, 3, 5]

#: iteration_count sweep values (paper Figure 3: 0 → 4 online iterations)
ITERATION_COUNT_SWEEP: List[int] = [0, 1, 2, 3, 4]

#: adapter_size sweep in billions of parameters (paper Table 2 / ablation)
ADAPTER_SIZE_SWEEP: List[float] = [0.1, 0.3]

#: temperature sweep for LLM sampling
TEMPERATURE_SWEEP: List[float] = [0.3, 0.5, 0.7, 0.9, 1.0]

#: batch_size sweep — preserves fixed anchors batch_size_64 and batch_size_128
BATCH_SIZE_SWEEP: List[int] = [batch_size_64, batch_size_128]

#: Complete sweep registry as a structured dict (for serialisation / reporting)
SWEEP_REGISTRY: Dict[str, List[Any]] = {
    "beam_size": BEAM_SIZE_SWEEP,
    "iteration_count": ITERATION_COUNT_SWEEP,
    "adapter_size": ADAPTER_SIZE_SWEEP,
    "temperature": TEMPERATURE_SWEEP,
    "batch_size": BATCH_SIZE_SWEEP,
}

#: Default sweep values used when no sweep is requested
DEFAULT_SWEEP_VALUES: Dict[str, Any] = {
    "beam_size": 3,
    "iteration_count": 4,
    "adapter_size": 0.1,
    "temperature": 1.0,
    "batch_size": batch_size_128,
}


# =========================================================================
# Adapter Model Configuration Dataclass
# reference_grounding: paperbench_ref_002 src/models/qa/transformer_qa.py
#   (forward: question_with_context Dict[str,LongTensor], yes_no_span,
#    answer_span, logit output shape, context followed by question)
# reference_grounding: paperbench_ref_005 toxigen/alice.py
#   (weights=[.5,.5] combining LM log-prob + classifier score)
# =========================================================================

@dataclass
class AdapterModelConfig:
    """
    Configuration for the BBox-Adapter energy model.

    The adapter is a lightweight transformer (0.1 B or 0.3 B parameters)
    that assigns a scalar energy score E_θ(x, y) to (prompt, response) pairs.
    The adapted distribution is:

        log P_adapted(y|x) = log P_bbox(y|x) + E_θ(x, y) / temperature

    Architecture choices are derived from the paper's two size variants.

    reference_grounding: paperbench_ref_002 src/models/qa/transformer_qa.py
    reference_grounding: paperbench_ref_005 toxigen/alice.py
    """

    # ---- Model identity -------------------------------------------------
    model_name: str = "microsoft/deberta-v3-base"
    adapter_size: float = 0.1       # billions of parameters: 0.1 or 0.3

    # ---- Architecture parameters ----------------------------------------
    hidden_size: int = 768
    num_layers: int = 6
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 512
    vocab_size: int = 50257          # GPT-2 tokeniser vocabulary

    # ---- Energy function output -----------------------------------------
    energy_output_dim: int = 1       # Scalar energy per (prompt, response)

    # ---- Combination weights (paper: equal weighting 0.5 / 0.5)
    # reference_grounding: paperbench_ref_005 toxigen/alice.py (weights=[.5,.5])
    llm_weight: float = 0.5          # Weight for black-box LLM log-prob
    adapter_weight: float = 0.5      # Weight for adapter energy score

    # ---- Inference parameters -------------------------------------------
    temperature: float = 1.0
    beam_size: int = 3               # Default; sweep over {1, 3, 5}
    max_length: int = 256

    # ---- Checkpoint path ------------------------------------------------
    checkpoint_path: str = "checkpoints/adapter.pt"

    def get_num_parameters_approx(self) -> int:
        """Approximate number of model parameters from declared size."""
        return int(self.adapter_size * 1_000_000_000)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary."""
        return asdict(self)

    @classmethod
    def for_adapter_size(
        cls, size: float, **kwargs: Any
    ) -> "AdapterModelConfig":
        """
        Factory: create a pre-configured AdapterModelConfig for a given
        adapter size (0.1 B or 0.3 B).

        Architecture dimensions scale with size following the paper.

        reference_grounding: paperbench_ref_002 src/models/qa/transformer_qa.py
        """
        if size not in ADAPTER_SIZE_SWEEP:
            raise ValueError(
                f"adapter_size must be one of {ADAPTER_SIZE_SWEEP}, got {size}"
            )
        model_name = kwargs.pop("model_name", None)
        if model_name is None:
            model_name = "microsoft/deberta-v3-base" if size == 0.1 else "microsoft/deberta-v3-large"
        if size == 0.1:
            return cls(
                model_name=model_name,
                adapter_size=0.1,
                hidden_size=768,
                num_layers=6,
                num_attention_heads=12,
                intermediate_size=3072,
                **kwargs,
            )
        # size == 0.3
        return cls(
            model_name=model_name,
            adapter_size=0.3,
            hidden_size=1024,
            num_layers=12,
            num_attention_heads=16,
            intermediate_size=4096,
            **kwargs,
        )

    @classmethod
    def for_appendix_h2_task(
        cls, dataset: str, adapter_size: float = 0.1, **kwargs: Any
    ) -> "AdapterModelConfig":
        """Return the exact Appendix H.2 adapter backbone for a task.

        StrategyQA, GSM8K, and ScienceQA use DeBERTa-v3-base for 0.1B and
        DeBERTa-v3-large for 0.3B. TruthfulQA uses bert-base-cased.
        """

        key = dataset.lower().replace("-", "").replace("_", "")
        if key == "truthfulqa":
            return cls.for_adapter_size(adapter_size, model_name="bert-base-cased", **kwargs)
        return cls.for_adapter_size(adapter_size, **kwargs)


# =========================================================================
# Training Configuration Dataclass
# Fixed anchors: batch_size_128=128, batch_size_64=64
# =========================================================================

@dataclass
class TrainingConfig:
    """
    Training configuration for BBox-Adapter online adaptation (Algorithm 1).

    Iterative online adaptation with ranking NCE loss:
      - Positive/negative sampling from black-box LLM
      - AdamW optimiser with linear warmup
      - beam_size candidates per example; highest-reward = positive

    Fixed hyperparameter anchors (paper contract):
      batch_size_128 = 128
      batch_size_64  = 64
    """

    # ---- Batch parameters (fixed anchors from paper) --------------------
    batch_size: int = batch_size_128       # anchor: batch_size_128
    eval_batch_size: int = batch_size_64   # anchor: batch_size_64

    # ---- Iteration count (paper sweep: 0, 1, 2, 3, 4) -------------------
    num_iterations: int = 4
    num_epochs_per_iteration: int = 1

    # ---- Optimiser parameters -------------------------------------------
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1

    # ---- NCE loss parameters --------------------------------------------
    nce_temperature: float = 1.0
    num_negative_samples: int = 3   # k − 1 negatives per positive

    # ---- Beam size (paper sweep: 1, 3, 5) --------------------------------
    beam_size: int = 3

    # ---- Feedback mode ---------------------------------------------------
    feedback_mode: str = "ground_truth"   # ground_truth | ai | combined

    # ---- Logging / checkpointing ----------------------------------------
    log_every_n_steps: int = 10
    checkpoint_every_n_iterations: int = 1
    output_dir: str = "checkpoints"

    # ---- Artifact paths (task contract) ----------------------------------
    training_trace_path: str = "results/adapter_training_trace.json"
    loss_curves_path: str = "results/loss_curves.json"

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary."""
        return asdict(self)

    def validate(self) -> List[str]:
        """
        Validate training configuration against paper-specified sweep values.

        Returns:
            List of warning strings; empty list means all parameters are valid.
        """
        warnings: List[str] = []
        if self.batch_size not in BATCH_SIZE_SWEEP:
            warnings.append(
                f"batch_size={self.batch_size} not in paper sweep {BATCH_SIZE_SWEEP}"
            )
        if self.beam_size not in BEAM_SIZE_SWEEP:
            warnings.append(
                f"beam_size={self.beam_size} not in paper sweep {BEAM_SIZE_SWEEP}"
            )
        if self.num_iterations not in ITERATION_COUNT_SWEEP:
            warnings.append(
                f"num_iterations={self.num_iterations} not in paper sweep "
                f"{ITERATION_COUNT_SWEEP}"
            )
        if self.feedback_mode not in {"ground_truth", "ai", "combined"}:
            warnings.append(
                f"feedback_mode='{self.feedback_mode}' must be one of "
                f"{{ground_truth, ai, combined}}"
            )
        return warnings


# =========================================================================
# Inference Configuration Dataclass
# reference_grounding: paperbench_ref_005 toxigen/alice.py
#   (beam_search: num_beams, weights=[.5,.5], max_length=30, end_token="\n")
# =========================================================================

@dataclass
class InferenceConfig:
    """
    Inference configuration for BBox-Adapter adapted generation.

    BBox-Adapter uses sentence-level beam search:
      1. Sample k full responses from black-box LLM
      2. Re-rank by combined score: w1*log P_bbox(y|x) + w2*E_θ(x,y)
      3. Select top-scoring response as final output

    Beam size sweep: {1, 3, 5} (paper Figure 3, Table 8).

    reference_grounding: paperbench_ref_005 toxigen/alice.py
      (beam_search function signature: num_beams, weights=[.5,.5], max_length)
    """

    # ---- Beam search parameters (paper sweep: 1, 3, 5) ------------------
    beam_size: int = 3

    # ---- Temperature for LLM sampling -----------------------------------
    temperature: float = 1.0

    # ---- Generation length ----------------------------------------------
    max_new_tokens: int = 256

    # ---- Combination weights (equal per alice.py pattern) ---------------
    # reference_grounding: paperbench_ref_005 toxigen/alice.py (weights=[.5,.5])
    llm_log_prob_weight: float = 0.5
    adapter_energy_weight: float = 0.5

    # ---- Inference mode -------------------------------------------------
    inference_mode: str = "full_step"    # single_step | full_step

    # ---- Result artifact paths ------------------------------------------
    predictions_path: str = "results/predictions.jsonl"
    beam_traces_path: str = "results/beam_search_traces.json"

    def compute_combined_score(
        self, llm_log_prob: float, adapter_energy: float
    ) -> float:
        """
        Compute the combined ranking score for a single candidate response.

        combined = w1 * log P_bbox(y|x)  +  w2 * E_θ(x, y)

        reference_grounding: paperbench_ref_005 toxigen/alice.py
        """
        return (
            self.llm_log_prob_weight * llm_log_prob
            + self.adapter_energy_weight * adapter_energy
        )

    def rank_candidates(
        self,
        llm_log_probs: List[float],
        adapter_energies: List[float],
    ) -> List[int]:
        """
        Rank candidate indices by combined score (descending).

        Args:
            llm_log_probs: Log-probabilities from the black-box LLM.
            adapter_energies: Energy scores from the adapter model.

        Returns:
            Indices sorted from highest to lowest combined score.
        """
        if len(llm_log_probs) != len(adapter_energies):
            raise ValueError(
                f"llm_log_probs length {len(llm_log_probs)} != "
                f"adapter_energies length {len(adapter_energies)}"
            )
        scores = [
            self.compute_combined_score(lp, ae)
            for lp, ae in zip(llm_log_probs, adapter_energies)
        ]
        return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary."""
        return asdict(self)


# =========================================================================
# LLM Backend Configuration
# reference_grounding: paperbench_ref_006 MMLU/gpt_3.5_turbo_college_medicine.ipynb
#   (gpt-3.5-turbo API, temperature, "Let's think step by step", answer=(X))
# =========================================================================

@dataclass
class LLMBackendConfig:
    """
    Configuration for the black-box LLM backend.

    Supports:
      - OpenAI GPT-3.5-turbo / GPT-4 (main paper experiments)
      - Azure OpenAI endpoint (azure_sft baseline)
      - Mixtral-8x7B via HuggingFace (plug-and-play experiment, Table 3)
      - davinci-002 (plug-and-play experiment, Table 3)

    reference_grounding: paperbench_ref_006 MMLU/gpt_3.5_turbo_college_medicine.ipynb
    """

    # ---- Provider / model identity --------------------------------------
    provider: str = "openai"             # openai | azure | huggingface | local
    model_name: str = "gpt-3.5-turbo"

    # ---- API configuration (env-var names, not actual secrets) ----------
    api_key_env: str = "OPENAI_API_KEY"
    api_base_env: str = "OPENAI_API_BASE"

    # ---- Generation parameters ------------------------------------------
    temperature: float = 1.0
    max_tokens: int = 512
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    # ---- Request handling -----------------------------------------------
    timeout_seconds: int = 60
    max_retries: int = 3
    retry_delay_seconds: float = 1.0

    # ---- Cost tracking (USD per 1 K tokens) -----------------------------
    input_cost_per_1k: float = 0.001
    output_cost_per_1k: float = 0.002

    def get_api_key(self) -> Optional[str]:
        """Retrieve API key from environment (returns None if unset)."""
        return os.environ.get(self.api_key_env)

    def get_api_base(self) -> Optional[str]:
        """Retrieve API base URL from environment (returns None if unset)."""
        return os.environ.get(self.api_base_env)

    def is_available(self) -> bool:
        """Check whether API credentials are present in the environment."""
        return self.get_api_key() is not None

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary (env-var names only, no secrets)."""
        return asdict(self)

    @classmethod
    def for_gpt35_turbo(cls) -> "LLMBackendConfig":
        """OpenAI GPT-3.5-turbo — main paper experiments."""
        return cls(
            provider="openai",
            model_name="gpt-3.5-turbo",
            api_key_env="OPENAI_API_KEY",
            input_cost_per_1k=0.001,
            output_cost_per_1k=0.002,
        )

    @classmethod
    def for_gpt4(cls) -> "LLMBackendConfig":
        """OpenAI GPT-4 — AI feedback generation."""
        return cls(
            provider="openai",
            model_name="gpt-4",
            api_key_env="OPENAI_API_KEY",
            input_cost_per_1k=0.03,
            output_cost_per_1k=0.06,
        )

    @classmethod
    def for_mixtral_8x7b(cls) -> "LLMBackendConfig":
        """Mixtral-8x7B — plug-and-play experiment (Table 3)."""
        return cls(
            provider="huggingface",
            model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
            api_key_env="HF_TOKEN",
            input_cost_per_1k=0.0,
            output_cost_per_1k=0.0,
        )

    @classmethod
    def for_davinci_002(cls) -> "LLMBackendConfig":
        """davinci-002 — plug-and-play experiment (Table 3)."""
        return cls(
            provider="openai",
            model_name="davinci-002",
            api_key_env="OPENAI_API_KEY",
            input_cost_per_1k=0.002,
            output_cost_per_1k=0.002,
        )

    @classmethod
    def for_azure_openai(
        cls, deployment_name: str = "gpt-35-turbo"
    ) -> "LLMBackendConfig":
        """Azure OpenAI endpoint — azure_sft baseline."""
        return cls(
            provider="azure",
            model_name=deployment_name,
            api_key_env="AZURE_OPENAI_KEY",
            api_base_env="AZURE_OPENAI_ENDPOINT",
        )


# =========================================================================
# Unified Top-Level BBoxAdapterConfig
# =========================================================================

@dataclass
class BBoxAdapterConfig:
    """
    Unified top-level configuration for a complete BBox-Adapter experiment run.

    Combines LLM backend, adapter model, training, and inference configs
    into a single serialisable object consumed by training/evaluation scripts.
    """

    # ---- Method and task identity ---------------------------------------
    method: str = MethodID.BBOX_ADAPTER
    dataset: str = "gsm8k"
    feedback_mode: str = "ground_truth"   # ground_truth | ai | combined

    # ---- Sub-configurations ---------------------------------------------
    llm: LLMBackendConfig = field(default_factory=LLMBackendConfig)
    adapter: AdapterModelConfig = field(default_factory=AdapterModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    # ---- Experiment metadata --------------------------------------------
    experiment_name: str = "bbox_adapter_default"
    seed: int = 42
    device: str = "cpu"

    # ---- Artifact output directories ------------------------------------
    output_dir: str = "results"
    checkpoint_dir: str = "checkpoints"

    def validate(self) -> Dict[str, List[str]]:
        """
        Validate all sub-configurations.

        Returns:
            Dict mapping config section name → list of warning strings.
            Empty lists indicate no warnings for that section.
        """
        issues: Dict[str, List[str]] = {
            "training": self.training.validate(),
            "method": [],
            "dataset": [],
            "feedback_mode": [],
        }
        # Validate method
        try:
            resolve_method(self.method)
        except ValueError as exc:
            issues["method"].append(str(exc))
        # Validate dataset
        known_datasets = {
            "gsm8k", "strategyqa", "truthfulqa", "scienceqa", "toxigen"
        }
        if self.dataset not in known_datasets:
            issues["dataset"].append(
                f"dataset='{self.dataset}' not in {sorted(known_datasets)}"
            )
        # Validate feedback mode
        valid_feedback = {"ground_truth", "ai", "combined"}
        if self.feedback_mode not in valid_feedback:
            issues["feedback_mode"].append(
                f"feedback_mode='{self.feedback_mode}' not in {valid_feedback}"
            )
        return issues

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the full config to a nested dictionary."""
        return {
            "method": self.method,
            "dataset": self.dataset,
            "feedback_mode": self.feedback_mode,
            "llm": self.llm.to_dict(),
            "adapter": self.adapter.to_dict(),
            "training": self.training.to_dict(),
            "inference": self.inference.to_dict(),
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "device": self.device,
            "output_dir": self.output_dir,
            "checkpoint_dir": self.checkpoint_dir,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BBoxAdapterConfig":
        """Deserialise from a nested dictionary."""
        llm_d = d.get("llm", {})
        adapter_d = d.get("adapter", {})
        training_d = d.get("training", {})
        inference_d = d.get("inference", {})

        # Handle nested dataclass construction defensively
        llm = LLMBackendConfig(**{
            k: v for k, v in llm_d.items()
            if k in LLMBackendConfig.__dataclass_fields__
        })
        adapter = AdapterModelConfig(**{
            k: v for k, v in adapter_d.items()
            if k in AdapterModelConfig.__dataclass_fields__
        })
        training = TrainingConfig(**{
            k: v for k, v in training_d.items()
            if k in TrainingConfig.__dataclass_fields__
        })
        inference = InferenceConfig(**{
            k: v for k, v in inference_d.items()
            if k in InferenceConfig.__dataclass_fields__
        })
        return cls(
            method=d.get("method", MethodID.BBOX_ADAPTER),
            dataset=d.get("dataset", "gsm8k"),
            feedback_mode=d.get("feedback_mode", "ground_truth"),
            llm=llm,
            adapter=adapter,
            training=training,
            inference=inference,
            experiment_name=d.get("experiment_name", "bbox_adapter_default"),
            seed=d.get("seed", 42),
            device=d.get("device", "cpu"),
            output_dir=d.get("output_dir", "results"),
            checkpoint_dir=d.get("checkpoint_dir", "checkpoints"),
        )


# =========================================================================
# Method Descriptor Registry (for experiment matrix construction)
# =========================================================================

@dataclass
class MethodDescriptor:
    """Metadata record for a method or baseline in the paper experiment matrix."""

    method_id: str
    display_name: str
    category: str          # core | baseline | ablation | feedback
    requires_training: bool
    requires_api: bool
    requires_finetune: bool
    description: str
    paper_tables: List[str] = field(default_factory=list)


#: Full method descriptor registry keyed by canonical MethodID values.
METHOD_REGISTRY: Dict[str, MethodDescriptor] = {
    MethodID.OURS: MethodDescriptor(
        method_id=MethodID.OURS,
        display_name="Ours (BBox-Adapter)",
        category="core",
        requires_training=True,
        requires_api=True,
        requires_finetune=False,
        description=(
            "BBox-Adapter: ranking NCE loss + online adaptation + "
            "sentence-level beam inference"
        ),
        paper_tables=["Table 2", "Table 3", "Table 4", "Table 7", "Table 10"],
    ),
    MethodID.BBOX_ADAPTER: MethodDescriptor(
        method_id=MethodID.BBOX_ADAPTER,
        display_name="BBox-Adapter",
        category="core",
        requires_training=True,
        requires_api=True,
        requires_finetune=False,
        description="Core BBox-Adapter method (alias for ours)",
        paper_tables=["Table 2", "Table 3"],
    ),
    MethodID.ENERGY_BASED_MODEL: MethodDescriptor(
        method_id=MethodID.ENERGY_BASED_MODEL,
        display_name="Energy-Based Model",
        category="core",
        requires_training=True,
        requires_api=True,
        requires_finetune=False,
        description="EBM component: E_θ(x,y) assigns scalar energy to (prompt,response)",
        paper_tables=["Eq. 1", "Eq. 2"],
    ),
    MethodID.RANKING_NCE: MethodDescriptor(
        method_id=MethodID.RANKING_NCE,
        display_name="Ranking NCE",
        category="core",
        requires_training=True,
        requires_api=True,
        requires_finetune=False,
        description=(
            "Ranking-based Noise Contrastive Estimation: "
            "L = -log[exp(E(y+)) / Σ_i exp(E(y_i))]"
        ),
        paper_tables=["Table 5"],
    ),
    MethodID.ONLINE_ADAPTATION: MethodDescriptor(
        method_id=MethodID.ONLINE_ADAPTATION,
        display_name="Online Adaptation",
        category="core",
        requires_training=True,
        requires_api=True,
        requires_finetune=False,
        description="Iterative online adaptation framework (Algorithm 1)",
        paper_tables=["Table 2", "Figure 3"],
    ),
    MethodID.CHAIN_OF_THOUGHT: MethodDescriptor(
        method_id=MethodID.CHAIN_OF_THOUGHT,
        display_name="CoT",
        category="baseline",
        requires_training=False,
        requires_api=True,
        requires_finetune=False,
        description='Chain-of-Thought prompting ("Let\'s think step by step")',
        paper_tables=["Table 2"],
    ),
    MethodID.ORACLE: MethodDescriptor(
        method_id=MethodID.ORACLE,
        display_name="Oracle",
        category="baseline",
        requires_training=False,
        requires_api=True,
        requires_finetune=False,
        description="Oracle upper bound: selects best candidate using ground-truth",
        paper_tables=["Table 2"],
    ),
    MethodID.HEURISTIC: MethodDescriptor(
        method_id=MethodID.HEURISTIC,
        display_name="Heuristic",
        category="baseline",
        requires_training=False,
        requires_api=True,
        requires_finetune=False,
        description="Heuristic re-ranking baseline",
        paper_tables=["Table 2"],
    ),
    MethodID.ROBERTA: MethodDescriptor(
        method_id=MethodID.ROBERTA,
        display_name="RoBERTa",
        category="baseline",
        requires_training=True,
        requires_api=False,
        requires_finetune=True,
        description="RoBERTa-based discriminator re-ranker baseline",
        paper_tables=["Table 2"],
    ),
    MethodID.FINE_TUNING: MethodDescriptor(
        method_id=MethodID.FINE_TUNING,
        display_name="Fine-Tuning",
        category="baseline",
        requires_training=True,
        requires_api=False,
        requires_finetune=True,
        description="Full fine-tuning of a white-box model baseline",
        paper_tables=["Table 2", "Table 4"],
    ),
    MethodID.LORA: MethodDescriptor(
        method_id=MethodID.LORA,
        display_name="LoRA",
        category="baseline",
        requires_training=True,
        requires_api=False,
        requires_finetune=True,
        description="Low-Rank Adaptation (PEFT) baseline",
        paper_tables=["Table 2", "Table 4", "Table 6"],
    ),
    MethodID.SFT_LORA: MethodDescriptor(
        method_id=MethodID.SFT_LORA,
        display_name="SFT+LoRA",
        category="baseline",
        requires_training=True,
        requires_api=False,
        requires_finetune=True,
        description="Supervised Fine-Tuning with LoRA baseline",
        paper_tables=["Table 6", "Table 7"],
    ),
    MethodID.AZURE_SFT: MethodDescriptor(
        method_id=MethodID.AZURE_SFT,
        display_name="Azure SFT",
        category="baseline",
        requires_training=True,
        requires_api=True,
        requires_finetune=True,
        description="Azure OpenAI supervised fine-tuning baseline",
        paper_tables=["Table 2", "Table 4"],
    ),
    MethodID.MLM: MethodDescriptor(
        method_id=MethodID.MLM,
        display_name="MLM",
        category="ablation",
        requires_training=True,
        requires_api=True,
        requires_finetune=False,
        description="Masked Language Model loss ablation (vs ranking NCE — Table 5)",
        paper_tables=["Table 5"],
    ),
    MethodID.SINGLE_STEP_INFERENCE: MethodDescriptor(
        method_id=MethodID.SINGLE_STEP_INFERENCE,
        display_name="Single-Step Inference",
        category="ablation",
        requires_training=True,
        requires_api=True,
        requires_finetune=False,
        description="Beam search with beam_size=1 (single candidate)",
        paper_tables=["Figure 3"],
    ),
    MethodID.FULL_STEP_INFERENCE: MethodDescriptor(
        method_id=MethodID.FULL_STEP_INFERENCE,
        display_name="Full-Step Inference",
        category="core",
        requires_training=True,
        requires_api=True,
        requires_finetune=False,
        description="Full beam search inference with k candidates (beam_size ∈ {3,5})",
        paper_tables=["Table 2", "Figure 3"],
    ),
    MethodID.GROUND_TRUTH_FEEDBACK: MethodDescriptor(
        method_id=MethodID.GROUND_TRUTH_FEEDBACK,
        display_name="Ground-Truth Feedback",
        category="feedback",
        requires_training=True,
        requires_api=True,
        requires_finetune=False,
        description="Training signal from ground-truth label comparison (exact match)",
        paper_tables=["Table 1"],
    ),
    MethodID.AI_FEEDBACK: MethodDescriptor(
        method_id=MethodID.AI_FEEDBACK,
        display_name="AI Feedback",
        category="feedback",
        requires_training=True,
        requires_api=True,
        requires_finetune=False,
        description="Training signal from GPT-4 or classifier-based feedback",
        paper_tables=["Table 1"],
    ),
    MethodID.COMBINED_FEEDBACK: MethodDescriptor(
        method_id=MethodID.COMBINED_FEEDBACK,
        display_name="Combined Feedback",
        category="feedback",
        requires_training=True,
        requires_api=True,
        requires_finetune=False,
        description="Combined ground-truth + AI feedback training signal",
        paper_tables=["Table 1"],
    ),
}


def get_method_descriptor(name: str) -> MethodDescriptor:
    """
    Retrieve a MethodDescriptor by method name or alias.

    Args:
        name: Any method name variant or alias.

    Returns:
        MethodDescriptor with full method metadata.

    Raises:
        ValueError: If name cannot be resolved or has no descriptor.
    """
    canonical_id = resolve_method(name)
    descriptor = METHOD_REGISTRY.get(canonical_id)
    if descriptor is None:
        raise ValueError(
            f"No descriptor registered for method '{canonical_id}' "
            f"(resolved from '{name}')"
        )
    return descriptor


def list_all_methods() -> List[str]:
    """Return all canonical method ID strings."""
    return ALL_METHOD_IDS.copy()


def list_bbox_adapter_methods() -> List[str]:
    """Return canonical IDs for BBox-Adapter core methods."""
    return [mid for mid in ALL_METHOD_IDS if is_bbox_adapter_method(mid)]


def list_baseline_methods() -> List[str]:
    """Return canonical IDs for comparison baseline methods."""
    return [mid for mid in ALL_METHOD_IDS if is_baseline_method(mid)]


# =========================================================================
# Per-dataset preset factory functions (paper experiment protocol)
# reference_grounding: paperbench_ref_006 readme.md
# =========================================================================

#: Maps each dataset to its canonical feedback mode (paper Table 1)
DATASET_FEEDBACK_MAP: Dict[str, str] = {
    "gsm8k": "ground_truth",
    "strategyqa": "ai",
    "truthfulqa": "combined",
    "scienceqa": "ground_truth",
    "toxigen": "ai",
}


def make_gsm8k_config(
    adapter_size: float = 0.1,
    beam_size: int = 3,
    num_iterations: int = 4,
    batch_size: int = batch_size_128,
) -> BBoxAdapterConfig:
    """
    Standard BBox-Adapter config for GSM8K math reasoning.

    Feedback mode: ground-truth (exact match on numeric answer).
    Paper: Table 2, Table 3, Table 4, Table 5 (NCE vs MLM ablation).

    reference_grounding: paperbench_ref_006 readme.md
    """
    return BBoxAdapterConfig(
        method=MethodID.BBOX_ADAPTER,
        dataset="gsm8k",
        feedback_mode="ground_truth",
        llm=LLMBackendConfig.for_gpt35_turbo(),
        adapter=AdapterModelConfig.for_appendix_h2_task("gsm8k", adapter_size),
        training=TrainingConfig(
            batch_size=batch_size,
            num_iterations=num_iterations,
            beam_size=beam_size,
            feedback_mode="ground_truth",
        ),
        inference=InferenceConfig(
            beam_size=beam_size,
            temperature=1.0,
            inference_mode="full_step",
        ),
        experiment_name="gsm8k_groundtruth",
    )


def make_strategyqa_config(
    adapter_size: float = 0.1,
    beam_size: int = 3,
    num_iterations: int = 4,
    batch_size: int = batch_size_128,
) -> BBoxAdapterConfig:
    """
    Standard BBox-Adapter config for StrategyQA implicit reasoning.

    Feedback mode: AI feedback (GPT-4 based).
    Paper: Table 2, Table 3, Figure 3 (beam_size and iteration ablations).

    reference_grounding: paperbench_ref_002 src/models/qa/transformer_qa.py
      (yes_no_span handling for binary-answer QA)
    """
    return BBoxAdapterConfig(
        method=MethodID.BBOX_ADAPTER,
        dataset="strategyqa",
        feedback_mode="ai",
        llm=LLMBackendConfig.for_gpt35_turbo(),
        adapter=AdapterModelConfig.for_appendix_h2_task("strategyqa", adapter_size),
        training=TrainingConfig(
            batch_size=batch_size,
            num_iterations=num_iterations,
            beam_size=beam_size,
            feedback_mode="ai",
        ),
        inference=InferenceConfig(
            beam_size=beam_size,
            temperature=1.0,
            inference_mode="full_step",
        ),
        experiment_name="strategyqa_ai_feedback",
    )


def make_truthfulqa_config(
    adapter_size: float = 0.1,
    beam_size: int = 3,
    num_iterations: int = 4,
    batch_size: int = batch_size_128,
) -> BBoxAdapterConfig:
    """
    Standard BBox-Adapter config for TruthfulQA truthfulness.

    Feedback mode: combined (ground-truth + AI feedback).
    Paper: Table 2, Table 3.
    """
    return BBoxAdapterConfig(
        method=MethodID.BBOX_ADAPTER,
        dataset="truthfulqa",
        feedback_mode="combined",
        llm=LLMBackendConfig.for_gpt35_turbo(),
        adapter=AdapterModelConfig.for_appendix_h2_task("truthfulqa", adapter_size),
        training=TrainingConfig(
            batch_size=batch_size,
            num_iterations=num_iterations,
            beam_size=beam_size,
            feedback_mode="combined",
        ),
        inference=InferenceConfig(
            beam_size=beam_size,
            temperature=1.0,
            inference_mode="full_step",
        ),
        experiment_name="truthfulqa_combined",
    )


def make_scienceqa_config(
    adapter_size: float = 0.1,
    beam_size: int = 3,
    num_iterations: int = 4,
    batch_size: int = batch_size_128,
) -> BBoxAdapterConfig:
    """
    Standard BBox-Adapter config for ScienceQA multiple-choice reasoning.

    Feedback mode: ground-truth (letter choice matching).
    Paper: Table 2, Table 3.

    reference_grounding: paperbench_ref_006 MMLU/gpt_3.5_turbo_college_medicine.ipynb
      (multiple-choice prompt: "(A) ... (B) ...", "The answer is (X)")
    """
    return BBoxAdapterConfig(
        method=MethodID.BBOX_ADAPTER,
        dataset="scienceqa",
        feedback_mode="ground_truth",
        llm=LLMBackendConfig.for_gpt35_turbo(),
        adapter=AdapterModelConfig.for_appendix_h2_task("scienceqa", adapter_size),
        training=TrainingConfig(
            batch_size=batch_size,
            num_iterations=num_iterations,
            beam_size=beam_size,
            feedback_mode="ground_truth",
        ),
        inference=InferenceConfig(
            beam_size=beam_size,
            temperature=1.0,
            inference_mode="full_step",
        ),
        experiment_name="scienceqa_groundtruth",
    )


def make_toxigen_config(
    adapter_size: float = 0.1,
    beam_size: int = 3,
    num_iterations: int = 4,
    batch_size: int = batch_size_128,
) -> BBoxAdapterConfig:
    """
    Standard BBox-Adapter config for ToxiGen toxicity reduction.

    Feedback mode: AI feedback (hate-speech / toxicity classifier).
    Paper: Table 7.

    reference_grounding: paperbench_ref_005 toxigen/alice.py
      (beam_search with mode=0 for neutral, classifier weights)
    """
    return BBoxAdapterConfig(
        method=MethodID.BBOX_ADAPTER,
        dataset="toxigen",
        feedback_mode="ai",
        llm=LLMBackendConfig.for_gpt35_turbo(),
        adapter=AdapterModelConfig.for_appendix_h2_task("toxigen", adapter_size),
        training=TrainingConfig(
            batch_size=batch_size,
            num_iterations=num_iterations,
            beam_size=beam_size,
            feedback_mode="ai",
        ),
        inference=InferenceConfig(
            beam_size=beam_size,
            temperature=1.0,
            inference_mode="full_step",
        ),
        experiment_name="toxigen_ai_feedback",
    )


# ---- Dataset → factory registry ----------------------------------------

DATASET_CONFIG_FACTORY: Dict[str, Callable[..., BBoxAdapterConfig]] = {
    "gsm8k": make_gsm8k_config,
    "strategyqa": make_strategyqa_config,
    "truthfulqa": make_truthfulqa_config,
    "scienceqa": make_scienceqa_config,
    "toxigen": make_toxigen_config,
}


def make_config_for_dataset(
    dataset: str,
    adapter_size: float = 0.1,
    beam_size: int = 3,
    num_iterations: int = 4,
    batch_size: int = batch_size_128,
) -> BBoxAdapterConfig:
    """
    Create a BBoxAdapterConfig for the named dataset with given sweep parameters.

    Args:
        dataset: One of {gsm8k, strategyqa, truthfulqa, scienceqa, toxigen}.
        adapter_size: Adapter size in billions (0.1 or 0.3).
        beam_size: Beam size (1, 3, or 5).
        num_iterations: Online adaptation iterations (0–4).
        batch_size: Training batch size (64 or 128).

    Returns:
        Fully configured BBoxAdapterConfig.

    Raises:
        ValueError: If dataset is not recognised.
    """
    factory = DATASET_CONFIG_FACTORY.get(dataset)
    if factory is None:
        available = sorted(DATASET_CONFIG_FACTORY.keys())
        raise ValueError(
            f"Unknown dataset '{dataset}'. Available: {available}"
        )
    return factory(
        adapter_size=adapter_size,
        beam_size=beam_size,
        num_iterations=num_iterations,
        batch_size=batch_size,
    )


def get_default_config(dataset: str = "gsm8k") -> BBoxAdapterConfig:
    """
    Return the paper-default BBoxAdapterConfig for a dataset.

    Uses: beam_size=3, num_iterations=4, adapter_size=0.1B, batch_size=128.
    """
    return make_config_for_dataset(
        dataset=dataset,
        adapter_size=DEFAULT_SWEEP_VALUES["adapter_size"],
        beam_size=DEFAULT_SWEEP_VALUES["beam_size"],
        num_iterations=DEFAULT_SWEEP_VALUES["iteration_count"],
        batch_size=DEFAULT_SWEEP_VALUES["batch_size"],
    )


# =========================================================================
# Bounded parameter sweep builders (paper-specified sweep dimensions)
# =========================================================================

def build_beam_size_sweep(
    base_config: Optional[BBoxAdapterConfig] = None,
) -> List[BBoxAdapterConfig]:
    """
    Build sweep over beam_size ∈ {1, 3, 5} (paper Figure 3 / Table 8).

    reference_grounding: paperbench_ref_005 toxigen/alice.py  (num_beams parameter)
    """
    if base_config is None:
        base_config = BBoxAdapterConfig()
    configs: List[BBoxAdapterConfig] = []
    for beam_size in BEAM_SIZE_SWEEP:
        cfg = copy.deepcopy(base_config)
        cfg.training.beam_size = beam_size
        cfg.inference.beam_size = beam_size
        cfg.experiment_name = f"{base_config.experiment_name}_beam{beam_size}"
        configs.append(cfg)
    return configs


def build_iteration_count_sweep(
    base_config: Optional[BBoxAdapterConfig] = None,
) -> List[BBoxAdapterConfig]:
    """
    Build sweep over iteration_count ∈ {0, 1, 2, 3, 4} (paper Figure 3).
    """
    if base_config is None:
        base_config = BBoxAdapterConfig()
    configs: List[BBoxAdapterConfig] = []
    for n_iter in ITERATION_COUNT_SWEEP:
        cfg = copy.deepcopy(base_config)
        cfg.training.num_iterations = n_iter
        cfg.experiment_name = f"{base_config.experiment_name}_iter{n_iter}"
        configs.append(cfg)
    return configs


def build_adapter_size_sweep(
    base_config: Optional[BBoxAdapterConfig] = None,
) -> List[BBoxAdapterConfig]:
    """
    Build sweep over adapter_size ∈ {0.1, 0.3} B parameters (paper Table 2).
    """
    if base_config is None:
        base_config = BBoxAdapterConfig()
    configs: List[BBoxAdapterConfig] = []
    for adapter_size in ADAPTER_SIZE_SWEEP:
        cfg = copy.deepcopy(base_config)
        cfg.adapter = AdapterModelConfig.for_adapter_size(adapter_size)
        cfg.experiment_name = (
            f"{base_config.experiment_name}_size{adapter_size}B"
        )
        configs.append(cfg)
    return configs


def build_batch_size_sweep(
    base_config: Optional[BBoxAdapterConfig] = None,
) -> List[BBoxAdapterConfig]:
    """
    Build sweep over batch_size ∈ {64, 128}.

    Fixed anchors: batch_size_64=64, batch_size_128=128.
    """
    if base_config is None:
        base_config = BBoxAdapterConfig()
    configs: List[BBoxAdapterConfig] = []
    for bs in BATCH_SIZE_SWEEP:
        cfg = copy.deepcopy(base_config)
        cfg.training.batch_size = bs
        cfg.experiment_name = f"{base_config.experiment_name}_bs{bs}"
        configs.append(cfg)
    return configs


def build_temperature_sweep(
    base_config: Optional[BBoxAdapterConfig] = None,
) -> List[BBoxAdapterConfig]:
    """
    Build sweep over temperature ∈ {0.3, 0.5, 0.7, 0.9, 1.0}.
    """
    if base_config is None:
        base_config = BBoxAdapterConfig()
    configs: List[BBoxAdapterConfig] = []
    for temp in TEMPERATURE_SWEEP:
        cfg = copy.deepcopy(base_config)
        cfg.inference.temperature = temp
        cfg.llm.temperature = temp
        cfg.experiment_name = f"{base_config.experiment_name}_temp{temp}"
        configs.append(cfg)
    return configs


def build_full_sweep_matrix(
    dataset: str = "strategyqa",
) -> List[Dict[str, Any]]:
    """
    Build the full bounded cross-product sweep matrix for a dataset, covering:
      beam_size ∈ {1, 3, 5}
      iteration_count ∈ {0, 1, 2, 3, 4}
      adapter_size ∈ {0.1, 0.3}
      batch_size ∈ {64, 128}
      (temperature held at default 0.7 for the cross-product)

    Returns:
        List of serialised config dicts (len = 3×5×2×2 = 60 entries).
    """
    results: List[Dict[str, Any]] = []
    for beam_size in BEAM_SIZE_SWEEP:
        for n_iter in ITERATION_COUNT_SWEEP:
            for adapter_size in ADAPTER_SIZE_SWEEP:
                for batch_size in BATCH_SIZE_SWEEP:
                    cfg = make_config_for_dataset(
                        dataset=dataset,
                        adapter_size=adapter_size,
                        beam_size=beam_size,
                        num_iterations=n_iter,
                        batch_size=batch_size,
                    )
                    cfg.experiment_name = (
                        f"{dataset}"
                        f"_beam{beam_size}"
                        f"_iter{n_iter}"
                        f"_size{adapter_size}B"
                        f"_bs{batch_size}"
                    )
                    results.append(cfg.to_dict())
    return results


# =========================================================================
# Config I/O utilities
# =========================================================================

def save_config(
    config: BBoxAdapterConfig, path: Union[str, Path]
) -> None:
    """Save a BBoxAdapterConfig to a JSON file, creating parent dirs."""
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(config.to_dict(), fh, indent=2)
    logger.info("Config saved to %s", path)


def load_config(path: Union[str, Path]) -> BBoxAdapterConfig:
    """Load a BBoxAdapterConfig from a JSON file."""
    import json

    path = Path(path)
    with open(path, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    return BBoxAdapterConfig.from_dict(d)


def load_config_from_yaml(path: Union[str, Path]) -> BBoxAdapterConfig:
    """
    Load a BBoxAdapterConfig from a YAML file.

    Falls back to the JSON loader if PyYAML is not installed.
    """
    try:
        import yaml  # type: ignore[import]
        path = Path(path)
        with open(path, "r", encoding="utf-8") as fh:
            d = yaml.safe_load(fh)
        return BBoxAdapterConfig.from_dict(d or {})
    except ImportError:
        logger.warning(
            "PyYAML not available; attempting JSON load for %s", path
        )
        return load_config(path)


# =========================================================================
# Module-level registry summary (diagnostic / introspection)
# =========================================================================

_REGISTRY_SUMMARY: Dict[str, Any] = {
    "version": "1.0.0",
    "paper": (
        "BBox-Adapter: Lightweight Adapting for Black-Box Large Language Models"
    ),
    "method_count": len(METHOD_REGISTRY),
    "all_methods": ALL_METHOD_IDS,
    "sweep_registry": SWEEP_REGISTRY,
    "fixed_anchors": {
        "batch_size_128": batch_size_128,
        "batch_size_64": batch_size_64,
    },
    "supported_datasets": sorted(DATASET_CONFIG_FACTORY.keys()),
    "variant_aliases": sorted(METHOD_ALIAS_MAP.keys()),
    "reference_groundings": [
        "paperbench_ref_002 src/models/qa/transformer_qa.py",
        "paperbench_ref_005 toxigen/alice.py",
        "paperbench_ref_006 MMLU/data/README.txt",
        "paperbench_ref_006 MMLU/gpt_3.5_turbo_college_medicine.ipynb",
    ],
}


def get_registry_summary() -> Dict[str, Any]:
    """Return a snapshot of the complete method and sweep registry."""
    return dict(_REGISTRY_SUMMARY)
