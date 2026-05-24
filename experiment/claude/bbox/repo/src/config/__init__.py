#!/usr/bin/env python3
"""
BBox-Adapter Configuration Package

Exposes method/baseline selectors, parameter sweep registries, environment configs,
dataset registries, and artifact writers for BBox-Adapter paper reproduction.

Paper: BBox-Adapter: Lightweight Adapting for Black-Box Large Language Models

Reference grounding: paperbench_ref_002 src/models/qa/transformer_qa.py
Reference grounding: paperbench_ref_005 toxigen/alice.py
Reference grounding: paperbench_ref_006 readme.md

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
  batch_size: [64, 128]         # fixed anchors: batch_size_64, batch_size_128

Fixed Hyperparameter Anchors:
  batch_size_128 = 128
  batch_size_64  = 64
  temperature_default = 1.0
  judge_model = "roberta-base"
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

__version__ = "1.0.0"
__all__ = [
    # Config dataclasses
    "BBoxAdapterConfig",
    "DatasetConfig",
    "EnvironmentConfig",
    "ExperimentConfig",
    "SweepConfig",
    # Registries
    "METHOD_REGISTRY",
    "DATASET_REGISTRY",
    "ENVIRONMENT_REGISTRY",
    "SWEEP_REGISTRY",
    "METHOD_ALIAS_MAP",
    # Fixed hyperparameter anchors
    "batch_size_128",
    "batch_size_64",
    "TEMPERATURE_DEFAULT",
    "JUDGE_MODEL_DEFAULT",
    # Sweep config objects
    "BEAM_SIZE_SWEEP",
    "ITERATION_COUNT_SWEEP",
    "ADAPTER_SIZE_SWEEP",
    "TEMPERATURE_SWEEP",
    "BATCH_SIZE_SWEEP",
    # Factory functions
    "make_environment",
    "make_dataset_config",
    "make_experiment_config",
    "get_method_config",
    "get_sweep_values",
    "resolve_method",
    "load_config",
    "config_from_dict",
    # Artifact writers
    "write_environment_registry",
    "write_dataset_registry",
    "write_scope_report",
    "write_data_manifest",
    "write_all_artifacts",
    # Experiment matrix
    "DEFAULT_EXPERIMENT_MATRIX",
]

# ===========================================================================
# Fixed Hyperparameter Anchors (Paper Evidence Contract)
# reference_grounding: paperbench_ref_002 src/models/qa/transformer_qa.py
# reference_grounding: paperbench_ref_006 readme.md
# ===========================================================================

#: anchor: batch_size_128 — standard batch size (paper Table 2, Table 5)
batch_size_128: int = 128

#: anchor: batch_size_64 — small batch ablation anchor (paper Table 2)
batch_size_64: int = 64

#: Default generation temperature for black-box LLM calls (paper Section 4)
TEMPERATURE_DEFAULT: float = 1.0

#: Default toxicity judge model (paper Section 4.3, ToxiGen evaluation)
#: reference_grounding: paperbench_ref_005 toxigen/alice.py
JUDGE_MODEL_DEFAULT: str = "roberta-base"

#: Default adapter size in billions of parameters
ADAPTER_SIZE_DEFAULT: float = 0.3


# ===========================================================================
# Sweep Registry — bounded config values, not exhaustive execution
# reference_grounding: paperbench_ref_005 toxigen/alice.py (num_beams)
# reference_grounding: paperbench_ref_006 readme.md (hyperparameter tables)
# ===========================================================================

SWEEP_REGISTRY: Dict[str, Any] = {
    # Figure 3 / Table 2: beam size ablation
    "beam_size": [1, 3, 5],
    # Figure 3: iteration count (0 = base model, no adaptation)
    "iteration_count": [0, 1, 2, 3, 4],
    # Table 2: adapter size ablation (billions of parameters)
    "adapter_size": [0.1, 0.3],
    # Generation temperature sweep
    "temperature": [0.5, 0.7, 0.9, 1.0],
    # Batch size (preserves exact anchors batch_size_64, batch_size_128)
    "batch_size": [batch_size_64, batch_size_128],
    # LoRA configuration sweep
    "lora_rank": [128, 384],
    "lora_alpha": [256, 768],
    # SFT epoch sweep
    "sft_epochs": [1, 3, 5],
    # Feedback mode variants
    "feedback_mode": ["ground_truth", "ai_feedback", "combined"],
    # Beam width (synonym for beam_size in inference context)
    "beam_width": [1, 3, 5],
    # Number of online adaptation iterations
    "num_iterations": [0, 1, 2, 3, 4],
    # Learning rate sweep
    "learning_rate": [5e-6, 2e-4],
}


# ===========================================================================
# Method / Baseline Registry (Paper Evidence Contract)
# Selectable method/baseline/variant adapters:
#   Ours | ADAPTER | LLM | BBOX-ADAPTER | PEFT | LLM Adaptation |
#   Parameter-Efficient Fine-Tuning | BBox-ADAPTER | CoT |
#   Parameter-Efficient | Fine-Tuning | BBox-ADApter
# reference_grounding: paperbench_ref_002 src/models/qa/transformer_qa.py
# reference_grounding: paperbench_ref_006 readme.md
# ===========================================================================

METHOD_REGISTRY: Dict[str, Dict[str, Any]] = {
    # -----------------------------------------------------------------------
    # Primary contribution: BBox-Adapter variants
    # -----------------------------------------------------------------------
    "bbox_adapter": {
        "name": "BBox-Adapter",
        "aliases": [
            "BBox-ADAPTER", "BBox-ADApter", "BBOX-ADAPTER",
            "Ours", "ADAPTER", "ours",
        ],
        "category": "ours",
        "description": (
            "Energy-based model adapter with ranking NCE loss for black-box LLMs. "
            "P_adapted(y|x) ∝ P_bbox(y|x)·exp(E_θ(x,y))"
        ),
        "paper_tables": ["Table 2", "Table 3", "Table 4", "Table 5", "Table 6", "Table 7", "Table 10"],
        "requires_training": True,
        "adapter_size": ADAPTER_SIZE_DEFAULT,
        "beam_size": 3,
        "num_iterations": 4,
        "batch_size": batch_size_128,
        "temperature": TEMPERATURE_DEFAULT,
        "loss": "ranking_nce",
        "feedback": "ground_truth",
    },
    "ranking_nce": {
        "name": "Ranking-NCE",
        "aliases": ["NCE", "nce_loss", "ranking_noise_contrastive"],
        "category": "ours",
        "description": "Ranking noise contrastive estimation loss for EBM training",
        "paper_tables": ["Table 5"],
        "requires_training": True,
        "loss": "ranking_nce",
        "batch_size": batch_size_128,
    },
    "online_adaptation": {
        "name": "Online Adaptation",
        "aliases": ["online", "iterative_adaptation"],
        "category": "ours",
        "description": "Full online adaptation framework with iterative parameter updates",
        "paper_tables": ["Table 2", "Table 3"],
        "requires_training": True,
        "num_iterations": 4,
        "batch_size": batch_size_128,
    },
    "energy_based_model": {
        "name": "Energy-Based Model",
        "aliases": ["EBM", "ebm", "ebm_adapter"],
        "category": "ours",
        "description": "Energy-based scoring of (prompt, response) pairs via E_θ(x,y)",
        "paper_tables": ["Table 5"],
        "requires_training": True,
        "loss": "ranking_nce",
        "batch_size": batch_size_128,
    },
    # -----------------------------------------------------------------------
    # Inference strategies
    # -----------------------------------------------------------------------
    "single_step_inference": {
        "name": "Single-Step Inference",
        "aliases": ["single_step", "greedy", "greedy_decoding"],
        "category": "inference",
        "description": "Single-step greedy decoding (beam_size=1, num_iterations=0)",
        "requires_training": False,
        "beam_size": 1,
        "num_iterations": 0,
        "temperature": TEMPERATURE_DEFAULT,
    },
    "full_step_inference": {
        "name": "Full-Step Inference",
        "aliases": ["full_step", "beam_inference", "beam_search"],
        "category": "inference",
        "description": "Beam search inference with maximum iteration count",
        "requires_training": False,
        "beam_size": 5,
        "num_iterations": 4,
        "temperature": TEMPERATURE_DEFAULT,
        # reference_grounding: paperbench_ref_005 toxigen/alice.py
        # (beam_search with num_beams=10, weights=[.5,.5])
    },
    # -----------------------------------------------------------------------
    # Feedback modes
    # -----------------------------------------------------------------------
    "ground_truth_feedback": {
        "name": "Ground-Truth Feedback",
        "aliases": ["gt_feedback", "oracle_feedback", "groundtruth"],
        "category": "feedback",
        "description": "Uses ground-truth labels to identify positive/negative candidates",
        "datasets": ["gsm8k", "scienceqa"],
        "feedback_mode": "ground_truth",
    },
    "ai_feedback": {
        "name": "AI Feedback",
        "aliases": ["llm_feedback", "ai_judge", "ai_reward"],
        "category": "feedback",
        "description": "Uses LLM or classifier as reward judge for positive/negative sampling",
        "datasets": ["strategyqa", "toxigen"],
        "feedback_mode": "ai",
        "judge_model": JUDGE_MODEL_DEFAULT,
    },
    "combined_feedback": {
        "name": "Combined Feedback",
        "aliases": ["hybrid_feedback", "combined_reward"],
        "category": "feedback",
        "description": "Combines ground-truth and AI feedback signals for reward estimation",
        "datasets": ["truthfulqa"],
        "feedback_mode": "combined",
    },
    # -----------------------------------------------------------------------
    # Baselines: LLM prompting
    # -----------------------------------------------------------------------
    "chain_of_thought": {
        "name": "Chain-of-Thought",
        "aliases": ["CoT", "cot", "chain_of_thought_prompting", "few_shot_cot"],
        "category": "baseline",
        "description": "Standard chain-of-thought prompting without any adaptation",
        "paper_tables": ["Table 2"],
        "requires_training": False,
        "temperature": TEMPERATURE_DEFAULT,
        # reference_grounding: paperbench_ref_006 readme.md
        # (CoT prompting as the primary evaluation protocol)
    },
    "oracle": {
        "name": "Oracle",
        "aliases": ["upper_bound", "oracle_selection", "oracle_baseline"],
        "category": "baseline",
        "description": "Oracle upper bound: uses ground-truth labels to select best candidate",
        "paper_tables": ["Table 2"],
        "requires_training": False,
        "beam_size": 5,
    },
    "heuristic": {
        "name": "Heuristic",
        "aliases": ["heuristic_baseline", "rule_based", "rule_based_selection"],
        "category": "baseline",
        "description": "Heuristic-based candidate selection without learned adapter",
        "paper_tables": ["Table 2"],
        "requires_training": False,
    },
    "roberta": {
        "name": "RoBERTa",
        "aliases": ["roberta_base", "roberta-base", "roberta_classifier"],
        "category": "baseline",
        "description": "RoBERTa-base fine-tuned as discriminative classifier/judge",
        "paper_tables": ["Table 2", "Table 7"],
        "requires_training": True,
        "model_name": "roberta-base",
        "judge_model": JUDGE_MODEL_DEFAULT,
        # reference_grounding: paperbench_ref_005 toxigen/alice.py
        # (classifier for toxicity scoring alongside language model)
    },
    # -----------------------------------------------------------------------
    # Baselines: Fine-tuning / PEFT
    # -----------------------------------------------------------------------
    "fine_tuning": {
        "name": "Fine-Tuning",
        "aliases": ["ft", "full_fine_tuning", "fine-tuning", "Fine-Tuning"],
        "category": "baseline",
        "description": "Full parameter fine-tuning of LLM on task data",
        "paper_tables": ["Table 2", "Table 6"],
        "requires_training": True,
        "batch_size": batch_size_128,
        "method_type": "fine_tuning",
    },
    "lora": {
        "name": "LoRA",
        "aliases": [
            "lora_adapter", "Parameter-Efficient Fine-Tuning", "PEFT",
            "peft", "Parameter-Efficient", "low_rank_adaptation",
        ],
        "category": "baseline",
        "description": "Low-Rank Adaptation of LLM parameters (PEFT method)",
        "paper_tables": ["Table 2", "Table 6"],
        "requires_training": True,
        "lora_rank": 16,
        "lora_alpha": 256,
        "lora_dropout": 0.05,
        "batch_size": batch_size_128,
    },
    "sft_lora": {
        "name": "SFT+LoRA",
        "aliases": ["sft_lora_adapter", "lora_sft", "supervised_ft_lora"],
        "category": "baseline",
        "description": "Supervised fine-tuning combined with LoRA for parameter efficiency",
        "paper_tables": ["Table 6", "Table 7"],
        "requires_training": True,
        "lora_rank": 16,
        "lora_alpha": 256,
        "sft_epochs": 3,
        "batch_size": batch_size_128,
    },
    "azure_sft": {
        "name": "Azure SFT",
        "aliases": ["azure_fine_tuning", "openai_sft", "azure_supervised_ft"],
        "category": "baseline",
        "description": "Azure OpenAI supervised fine-tuning API endpoint",
        "paper_tables": ["Table 2", "Table 4"],
        "requires_training": True,
        "provider": "azure",
        "batch_size": batch_size_64,
    },
    "mlm": {
        "name": "MLM",
        "aliases": ["masked_lm", "mlm_loss", "masked_language_model"],
        "category": "baseline",
        "description": "Masked language model style adaptation loss (ablation vs NCE)",
        "paper_tables": ["Table 5"],
        "requires_training": True,
        "loss": "mlm",
        "batch_size": batch_size_128,
    },
    # -----------------------------------------------------------------------
    # Category / taxonomy entries (referenced by paper taxonomy)
    # -----------------------------------------------------------------------
    "llm_adaptation": {
        "name": "LLM Adaptation",
        "aliases": ["LLM Adaptation", "llm_adapt", "LLM"],
        "category": "taxonomy",
        "description": "Taxonomy: LLM adaptation methods category",
    },
    "parameter_efficient": {
        "name": "Parameter-Efficient",
        "aliases": [
            "Parameter-Efficient Fine-Tuning", "PEFT",
            "Parameter-Efficient", "peft_category",
        ],
        "category": "taxonomy",
        "description": "Taxonomy: parameter-efficient fine-tuning methods category",
    },
}

# Build alias lookup map (lowercased, hyphens → underscores)
METHOD_ALIAS_MAP: Dict[str, str] = {}
for _mid, _mconf in METHOD_REGISTRY.items():
    METHOD_ALIAS_MAP[_mid] = _mid
    for _alias in _mconf.get("aliases", []):
        _key = _alias.lower().replace("-", "_").replace(" ", "_")
        METHOD_ALIAS_MAP[_key] = _mid


# ===========================================================================
# Dataset Registry — standardized QA format
# reference_grounding: paperbench_ref_002 src/models/qa/transformer_qa.py
# (question_with_context, yes_no_span, answer_span standardized format)
# reference_grounding: paperbench_ref_006 readme.md (GSM8K, benchmark setup)
# ===========================================================================

DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "gsm8k": {
        "name": "GSM8K",
        "full_name": "Grade School Math 8K",
        "task": "math_reasoning",
        "answer_type": "numeric",
        "feedback": "ground_truth",
        "metric": "accuracy",
        "secondary_metric": None,
        "paper_tables": ["Table 2", "Table 3", "Table 4", "Table 5"],
        "splits": {"train": 7473, "test": 1319},
        "hf_path": "gsm8k",
        "hf_config": "main",
        "format": "qa",
        "num_choices": None,
        "eval_protocol": "exact_match_numeric",
        # reference_grounding: paperbench_ref_006 readme.md
        # (GPT-3.5-turbo vs text-davinci-003 on GSM8K, CoT evaluation)
    },
    "strategyqa": {
        "name": "StrategyQA",
        "full_name": "Strategy Question Answering",
        "task": "implicit_reasoning",
        "answer_type": "yes_no",
        "feedback": "ai_feedback",
        "metric": "accuracy",
        "secondary_metric": None,
        "paper_tables": ["Table 2", "Table 3", "Table 5", "Table 6"],
        "splits": {"train": 2061, "dev": 229, "test": 490},
        "hf_path": "tau/commonsense_qa",
        "hf_config": "default",
        "format": "qa",
        "num_choices": 2,
        # reference_grounding: paperbench_ref_002 src/models/qa/transformer_qa.py
        # (yes_no_span: torch.IntTensor handling in QA forward pass)
        "eval_protocol": "exact_match_yesno",
    },
    "truthfulqa": {
        "name": "TruthfulQA",
        "full_name": "TruthfulQA Benchmark",
        "task": "truthfulness",
        "answer_type": "multiple_choice",
        "feedback": "combined",
        "metric": "accuracy",
        "secondary_metric": None,
        "paper_tables": ["Table 2", "Table 3"],
        "splits": {"validation": 817},
        "hf_path": "truthful_qa",
        "hf_config": "multiple_choice",
        "format": "multiple_choice",
        "num_choices": 4,
        "eval_protocol": "mc_accuracy",
    },
    "scienceqa": {
        "name": "ScienceQA",
        "full_name": "Science Question Answering",
        "task": "science_domain",
        "answer_type": "multiple_choice",
        "feedback": "ground_truth",
        "metric": "accuracy",
        "secondary_metric": None,
        "paper_tables": ["Table 2", "Table 3"],
        "splits": {"train": 12726, "validation": 4695, "test": 4726},
        "hf_path": "derek-thomas/ScienceQA",
        "hf_config": "default",
        "format": "multiple_choice",
        "num_choices": 4,
        "eval_protocol": "mc_accuracy",
    },
    "toxigen": {
        "name": "ToxiGen",
        "full_name": "ToxiGen Toxicity Dataset",
        "task": "toxicity_reduction",
        "answer_type": "free_form",
        "feedback": "ai_feedback",
        "metric": "hate_speech_rate",
        "secondary_metric": "toxicity_score",
        "paper_tables": ["Table 7"],
        # reference_grounding: paperbench_ref_005 toxigen/alice.py
        # (beam_search with classifier weights=[.5,.5], mode for toxic/neutral)
        "judge_model": JUDGE_MODEL_DEFAULT,
        "splits": {"train": 8960, "test": 940},
        "hf_path": "skg/toxigen-data",
        "hf_config": "annotated",
        "format": "generation",
        "num_choices": None,
        "eval_protocol": "toxicity_classification",
    },
}


# ===========================================================================
# Environment Registry — model API endpoints
# ===========================================================================

ENVIRONMENT_REGISTRY: Dict[str, Dict[str, Any]] = {
    "gpt-3.5-turbo": {
        "name": "GPT-3.5-Turbo",
        "provider": "openai",
        "model_id": "gpt-3.5-turbo",
        "type": "black_box",
        "api_type": "chat",
        "env_var": "OPENAI_API_KEY",
        "temperature": TEMPERATURE_DEFAULT,
        "max_tokens": 512,
        "paper_tables": ["Table 2", "Table 3", "Table 4", "Table 7"],
        # reference_grounding: paperbench_ref_006 readme.md
        # (gpt-3.5-turbo, 10x cheaper than text-davinci-003)
    },
    "gpt-4": {
        "name": "GPT-4",
        "provider": "openai",
        "model_id": "gpt-4",
        "type": "black_box",
        "api_type": "chat",
        "env_var": "OPENAI_API_KEY",
        "temperature": TEMPERATURE_DEFAULT,
        "max_tokens": 512,
        "paper_tables": ["Table 2"],
    },
    "text-davinci-003": {
        "name": "text-davinci-003",
        "provider": "openai",
        "model_id": "text-davinci-003",
        "type": "black_box",
        "api_type": "completion",
        "env_var": "OPENAI_API_KEY",
        "temperature": TEMPERATURE_DEFAULT,
        "max_tokens": 512,
        # reference_grounding: paperbench_ref_006 readme.md
        "paper_tables": ["Table 3"],
    },
    "davinci-002": {
        "name": "davinci-002",
        "provider": "openai",
        "model_id": "davinci-002",
        "type": "black_box",
        "api_type": "completion",
        "env_var": "OPENAI_API_KEY",
        "temperature": TEMPERATURE_DEFAULT,
        "max_tokens": 512,
        "paper_tables": ["Table 3"],
    },
    "mixtral-8x7b": {
        "name": "Mixtral-8x7B",
        "provider": "huggingface",
        "model_id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "type": "black_box",
        "api_type": "text-generation",
        "env_var": "HF_API_KEY",
        "temperature": TEMPERATURE_DEFAULT,
        "max_tokens": 512,
        "paper_tables": ["Table 3"],
    },
    "azure-gpt-35-turbo": {
        "name": "Azure GPT-3.5-Turbo",
        "provider": "azure",
        "model_id": "gpt-35-turbo",
        "type": "black_box",
        "api_type": "chat",
        "env_var": "AZURE_OPENAI_API_KEY",
        "base_url_env": "AZURE_OPENAI_ENDPOINT",
        "temperature": TEMPERATURE_DEFAULT,
        "max_tokens": 512,
        "paper_tables": ["Table 2", "Table 4"],
    },
    "roberta-base": {
        "name": "RoBERTa-Base",
        "provider": "huggingface",
        "model_id": "roberta-base",
        "type": "white_box",
        "api_type": "classification",
        "env_var": None,
        "role": "judge",
        # reference_grounding: paperbench_ref_005 toxigen/alice.py
        # (classifier parameter in beam_search, toxicity mode)
        "paper_tables": ["Table 7"],
    },
}


# ===========================================================================
# Config Dataclasses
# ===========================================================================

@dataclass
class BBoxAdapterConfig:
    """
    Full configuration for BBox-Adapter training and inference.

    reference_grounding: paperbench_ref_002 src/models/qa/transformer_qa.py
    reference_grounding: paperbench_ref_005 toxigen/alice.py
    """
    # Adapter architecture
    adapter_size: float = ADAPTER_SIZE_DEFAULT          # sweep: [0.1, 0.3] B params
    hidden_dim: int = 768
    num_layers: int = 2
    dropout: float = 0.1

    # Fixed hyperparameter anchors (paper-derived)
    batch_size: int = batch_size_128                    # anchor: batch_size_128
    batch_size_64_anchor: int = batch_size_64           # anchor: batch_size_64

    # Training hyperparameters
    learning_rate: float = 5e-6                         # sweep: [5e-6, 2e-4]
    num_iterations: int = 4                             # sweep: [0, 1, 2, 3, 4]
    max_epochs: int = 10
    weight_decay: float = 0.01
    warmup_steps: int = 100
    gradient_clip: float = 1.0

    # Inference hyperparameters
    beam_size: int = 3                                  # sweep: [1, 3, 5]
    beam_width: int = 3                                 # alias for beam_size
    temperature: float = TEMPERATURE_DEFAULT            # default: 0.7
    top_p: float = 0.9
    max_new_tokens: int = 512

    # NCE loss configuration
    loss_type: str = "ranking_nce"                      # or "mlm"
    nce_noise_samples: int = 5
    nce_temperature: float = 1.0

    # Feedback configuration
    feedback_mode: str = "ground_truth"                 # or "ai_feedback", "combined"
    ai_judge_model: str = JUDGE_MODEL_DEFAULT           # "roberta-base"

    # LoRA configuration (for baselines)
    lora_rank: int = 16                                 # sweep: [8, 16, 32]
    lora_alpha: int = 256                                # sweep: [16, 32]
    lora_dropout: float = 0.05

    # SFT configuration (for baselines)
    sft_epochs: int = 3                                 # sweep: [1, 3, 5]
    sft_batch_size: int = batch_size_128

    # Model endpoint
    model_name: str = "gpt-3.5-turbo"
    provider: str = "openai"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetConfig:
    """Dataset configuration for standardized QA format."""
    name: str
    task: str
    feedback: str
    metric: str
    hf_path: str
    hf_config: str = "default"
    num_choices: Optional[int] = None
    eval_protocol: str = "accuracy"
    train_size: Optional[int] = None
    test_size: Optional[int] = None
    max_train_samples: Optional[int] = None
    max_test_samples: Optional[int] = None
    judge_model: str = JUDGE_MODEL_DEFAULT

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EnvironmentConfig:
    """LLM environment / API endpoint configuration."""
    name: str
    provider: str
    model_id: str
    api_type: str
    temperature: float = TEMPERATURE_DEFAULT
    max_tokens: int = 512
    env_var: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 60
    max_retries: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def is_available(self) -> bool:
        """Return True when the required API key environment variable is set."""
        if self.env_var is None:
            return True
        return os.environ.get(self.env_var, "") != ""


@dataclass
class ExperimentConfig:
    """Full experiment configuration combining dataset, environment, and method."""
    experiment_id: str
    dataset: str
    environment: str
    method: str
    feedback_mode: str = "ground_truth"
    beam_size: int = 3
    num_iterations: int = 4
    adapter_size: float = ADAPTER_SIZE_DEFAULT
    batch_size: int = batch_size_128
    temperature: float = TEMPERATURE_DEFAULT
    adapter_config: BBoxAdapterConfig = field(default_factory=BBoxAdapterConfig)
    output_dir: str = "results"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SweepConfig:
    """Bounded parameter sweep configuration (not exhaustive execution)."""
    param_name: str
    values: List[Any]
    default: Any
    paper_reference: str = ""

    def get_values(self) -> List[Any]:
        return list(self.values)

    def get_default(self) -> Any:
        return self.default


# ===========================================================================
# Named SweepConfig objects (paper-derived)
# ===========================================================================

BEAM_SIZE_SWEEP = SweepConfig(
    param_name="beam_size",
    values=[1, 3, 5],
    default=3,
    paper_reference="Figure 3, Table 2",
)

ITERATION_COUNT_SWEEP = SweepConfig(
    param_name="iteration_count",
    values=[0, 1, 2, 3, 4],
    default=4,
    paper_reference="Figure 3",
)

ADAPTER_SIZE_SWEEP = SweepConfig(
    param_name="adapter_size",
    values=[0.1, 0.3],
    default=0.3,
    paper_reference="Table 2",
)

TEMPERATURE_SWEEP = SweepConfig(
    param_name="temperature",
    values=[0.5, 0.7, 0.9, 1.0],
    default=TEMPERATURE_DEFAULT,
    paper_reference="Section 4",
)

BATCH_SIZE_SWEEP = SweepConfig(
    param_name="batch_size",
    values=[batch_size_64, batch_size_128],
    default=batch_size_128,
    paper_reference="Table 2",
)


# ===========================================================================
# Factory Functions
# ===========================================================================

def _infer_provider(model_name: str) -> str:
    """Infer API provider from model name string."""
    ml = model_name.lower()
    if any(k in ml for k in ("gpt", "davinci", "openai", "turbo")):
        return "openai"
    if "azure" in ml:
        return "azure"
    if any(k in ml for k in ("mixtral", "mistral", "llama", "falcon", "bloom")):
        return "huggingface"
    if any(k in ml for k in ("roberta", "bert", "gpt2")):
        return "huggingface"
    return "openai"


def make_environment(
    config: Union[Dict[str, Any], "EnvironmentConfig", str],
) -> "EnvironmentConfig":
    """
    Create an EnvironmentConfig from a dict, existing config, or environment name.

    reference_grounding: paperbench_ref_006 readme.md
    (model endpoint selection for gpt-3.5-turbo, davinci-002, Mixtral-8x7B)

    Args:
        config: EnvironmentConfig, dict, or string environment key.

    Returns:
        EnvironmentConfig with validated settings.
    """
    if isinstance(config, EnvironmentConfig):
        return config

    if isinstance(config, str):
        env_name = config
        if env_name in ENVIRONMENT_REGISTRY:
            ed = ENVIRONMENT_REGISTRY[env_name]
            base_url: Optional[str] = None
            if ed.get("base_url_env"):
                base_url = os.environ.get(ed["base_url_env"]) or None
            return EnvironmentConfig(
                name=ed["name"],
                provider=ed["provider"],
                model_id=ed["model_id"],
                api_type=ed["api_type"],
                temperature=ed.get("temperature", TEMPERATURE_DEFAULT),
                max_tokens=ed.get("max_tokens", 512),
                env_var=ed.get("env_var"),
                base_url=base_url,
            )
        # Unknown name: auto-detect provider
        return EnvironmentConfig(
            name=env_name,
            provider=_infer_provider(env_name),
            model_id=env_name,
            api_type="chat",
            temperature=TEMPERATURE_DEFAULT,
        )

    if isinstance(config, dict):
        return EnvironmentConfig(
            name=config.get("name", config.get("model_id", "unknown")),
            provider=config.get("provider", "openai"),
            model_id=config.get("model_id", config.get("name", "gpt-3.5-turbo")),
            api_type=config.get("api_type", "chat"),
            temperature=config.get("temperature", TEMPERATURE_DEFAULT),
            max_tokens=config.get("max_tokens", 512),
            env_var=config.get("env_var"),
            base_url=config.get("base_url"),
            timeout=config.get("timeout", 60),
            max_retries=config.get("max_retries", 3),
        )

    raise TypeError(f"Unsupported config type: {type(config).__name__}")


def make_dataset_config(dataset_name: str, **overrides: Any) -> DatasetConfig:
    """
    Create a DatasetConfig for a registered dataset, returning standardized QA format.

    reference_grounding: paperbench_ref_002 src/models/qa/transformer_qa.py
    (question_with_context Dict, yes_no_span IntTensor, answer_span IntTensor)

    Args:
        dataset_name: Key in DATASET_REGISTRY.
        **overrides: Override specific fields.

    Returns:
        DatasetConfig with standardized QA format settings.
    """
    if dataset_name not in DATASET_REGISTRY:
        available = list(DATASET_REGISTRY.keys())
        raise ValueError(
            f"Dataset '{dataset_name}' not in registry. Available: {available}"
        )

    base = DATASET_REGISTRY[dataset_name]
    splits = base.get("splits", {})

    cfg = DatasetConfig(
        name=base["name"],
        task=base["task"],
        feedback=base["feedback"],
        metric=base["metric"],
        hf_path=base["hf_path"],
        hf_config=base.get("hf_config", "default"),
        num_choices=base.get("num_choices"),
        eval_protocol=base.get("eval_protocol", "accuracy"),
        train_size=splits.get("train"),
        test_size=splits.get("test", splits.get("validation")),
        judge_model=base.get("judge_model", JUDGE_MODEL_DEFAULT),
    )

    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    return cfg


def make_experiment_config(
    experiment_id: str,
    dataset: str,
    environment: str,
    method: str,
    **kwargs: Any,
) -> ExperimentConfig:
    """
    Create a full ExperimentConfig validated against dataset, environment, and method
    registries.

    Wires paper-derived objective, reward, metric, sweep, and baseline obligations
    into a callable ExperimentConfig.
    """
    if dataset not in DATASET_REGISTRY:
        raise ValueError(f"Dataset '{dataset}' not registered. Available: {list(DATASET_REGISTRY.keys())}")

    # Resolve method alias
    resolved_method = method
    if method not in METHOD_REGISTRY:
        normalized = method.lower().replace("-", "_").replace(" ", "_")
        if normalized in METHOD_ALIAS_MAP:
            resolved_method = METHOD_ALIAS_MAP[normalized]
        else:
            raise ValueError(
                f"Method '{method}' not in registry. Available: {list(METHOD_REGISTRY.keys())}"
            )

    method_cfg = METHOD_REGISTRY[resolved_method]
    dataset_cfg = DATASET_REGISTRY[dataset]

    # Feedback mode: prefer explicit kwarg, else dataset default
    feedback_mode = kwargs.pop("feedback_mode", dataset_cfg.get("feedback", "ground_truth"))

    adapter_config = BBoxAdapterConfig(
        adapter_size=kwargs.get("adapter_size", ADAPTER_SIZE_DEFAULT),
        batch_size=kwargs.get("batch_size", batch_size_128),
        beam_size=kwargs.get("beam_size", 3),
        num_iterations=kwargs.get("num_iterations", 4),
        temperature=kwargs.get("temperature", TEMPERATURE_DEFAULT),
        feedback_mode=feedback_mode,
        loss_type=method_cfg.get("loss", "ranking_nce"),
        lora_rank=kwargs.get("lora_rank", 16),
        lora_alpha=kwargs.get("lora_alpha", 32),
        sft_epochs=kwargs.get("sft_epochs", 3),
        model_name=environment,
    )

    return ExperimentConfig(
        experiment_id=experiment_id,
        dataset=dataset,
        environment=environment,
        method=resolved_method,
        feedback_mode=feedback_mode,
        beam_size=kwargs.get("beam_size", 3),
        num_iterations=kwargs.get("num_iterations", 4),
        adapter_size=kwargs.get("adapter_size", ADAPTER_SIZE_DEFAULT),
        batch_size=kwargs.get("batch_size", batch_size_128),
        temperature=kwargs.get("temperature", TEMPERATURE_DEFAULT),
        adapter_config=adapter_config,
        output_dir=kwargs.get("output_dir", "results"),
    )


def get_method_config(method_name: str) -> Dict[str, Any]:
    """Return method configuration by name or alias."""
    if method_name in METHOD_REGISTRY:
        return dict(METHOD_REGISTRY[method_name])
    normalized = method_name.lower().replace("-", "_").replace(" ", "_")
    if normalized in METHOD_ALIAS_MAP:
        return dict(METHOD_REGISTRY[METHOD_ALIAS_MAP[normalized]])
    raise KeyError(
        f"Method '{method_name}' not found. Available: {sorted(METHOD_REGISTRY.keys())}"
    )


def get_sweep_values(param_name: str) -> List[Any]:
    """Return bounded sweep values for a registered parameter."""
    if param_name in SWEEP_REGISTRY:
        return list(SWEEP_REGISTRY[param_name])
    raise KeyError(
        f"Sweep parameter '{param_name}' not found. "
        f"Available: {sorted(SWEEP_REGISTRY.keys())}"
    )


def resolve_method(method_name: str) -> str:
    """Resolve a method name or alias to its canonical registry key."""
    if method_name in METHOD_REGISTRY:
        return method_name
    normalized = method_name.lower().replace("-", "_").replace(" ", "_")
    if normalized in METHOD_ALIAS_MAP:
        return METHOD_ALIAS_MAP[normalized]
    raise KeyError(
        f"Cannot resolve method '{method_name}'. "
        f"Available: {sorted(METHOD_REGISTRY.keys())}"
    )


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load experiment configuration from a YAML or JSON file.

    PyYAML is imported lazily so this module remains importable without it.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml  # optional — lazy import
            with open(path) as fh:
                result = yaml.safe_load(fh)
                return result if result is not None else {}
        except ImportError:
            logger.warning("PyYAML unavailable; attempting JSON fallback for %s", path)
            with open(path) as fh:
                return json.load(fh)
    elif path.suffix == ".json":
        with open(path) as fh:
            return json.load(fh)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")


def config_from_dict(d: Dict[str, Any]) -> ExperimentConfig:
    """Create ExperimentConfig from a plain dictionary."""
    reserved = {"experiment_id", "dataset", "environment", "method"}
    return make_experiment_config(
        experiment_id=d.get("experiment_id", "default"),
        dataset=d.get("dataset", "gsm8k"),
        environment=d.get("environment", "gpt-3.5-turbo"),
        method=d.get("method", "bbox_adapter"),
        **{k: v for k, v in d.items() if k not in reserved},
    )


# ===========================================================================
# Artifact Writers
# reference_grounding: paperbench_ref_006 readme.md
# ===========================================================================

def _ensure_results_dir(results_dir: Union[str, Path]) -> Path:
    """Create results directory tree if absent, return Path object."""
    p = Path(results_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_environment_registry(
    results_dir: Union[str, Path] = "results",
) -> Dict[str, Any]:
    """
    Write results/environment_registry.json.

    Returns the artifact dict containing all registered environments and
    their API-key availability status.
    """
    out_dir = _ensure_results_dir(results_dir)
    environments: Dict[str, Any] = {}
    for env_id, env_data in ENVIRONMENT_REGISTRY.items():
        env_cfg = make_environment(env_id)
        environments[env_id] = {
            **{k: v for k, v in env_data.items() if k != "base_url_env"},
            "available": env_cfg.is_available(),
            "env_var_set": (
                os.environ.get(env_data.get("env_var", ""), "") != ""
                if env_data.get("env_var") else True
            ),
        }
    artifact: Dict[str, Any] = {
        "artifact_type": "environment_registry",
        "version": __version__,
        "total_environments": len(ENVIRONMENT_REGISTRY),
        "environments": environments,
        "default_temperature": TEMPERATURE_DEFAULT,
        "judge_model_default": JUDGE_MODEL_DEFAULT,
    }
    out_path = out_dir / "environment_registry.json"
    out_path.write_text(json.dumps(artifact, indent=2))
    logger.info("Wrote %s", out_path)
    return artifact


def write_dataset_registry(
    results_dir: Union[str, Path] = "results",
) -> Dict[str, Any]:
    """
    Write results/dataset_registry.json.

    Returns the artifact dict with all registered datasets in standardized QA format.
    reference_grounding: paperbench_ref_002 src/models/qa/transformer_qa.py
    """
    out_dir = _ensure_results_dir(results_dir)
    datasets: Dict[str, Any] = {}
    for ds_id, ds_data in DATASET_REGISTRY.items():
        cfg = make_dataset_config(ds_id)
        datasets[ds_id] = {
            **ds_data,
            "config": cfg.to_dict(),
            "standardized_format": "qa",
        }
    artifact: Dict[str, Any] = {
        "artifact_type": "dataset_registry",
        "version": __version__,
        "total_datasets": len(DATASET_REGISTRY),
        "feedback_modes": sorted({d["feedback"] for d in DATASET_REGISTRY.values()}),
        "tasks": sorted({d["task"] for d in DATASET_REGISTRY.values()}),
        "datasets": datasets,
    }
    out_path = out_dir / "dataset_registry.json"
    out_path.write_text(json.dumps(artifact, indent=2))
    logger.info("Wrote %s", out_path)
    return artifact


def write_scope_report(
    results_dir: Union[str, Path] = "results",
) -> Dict[str, Any]:
    """
    Write results/scope_report.json with method registry, sweep registry, and
    fixed hyperparameter anchors.
    """
    out_dir = _ensure_results_dir(results_dir)
    method_summary: Dict[str, Any] = {}
    for mid, mcfg in METHOD_REGISTRY.items():
        method_summary[mid] = {k: v for k, v in mcfg.items() if k != "aliases"}

    artifact: Dict[str, Any] = {
        "artifact_type": "scope_report",
        "version": __version__,
        "paper": "BBox-Adapter: Lightweight Adapting for Black-Box Large Language Models",
        "method_registry": method_summary,
        "method_count": len(METHOD_REGISTRY),
        "sweep_registry": SWEEP_REGISTRY,
        "sweep_summary": {
            "beam_size": BEAM_SIZE_SWEEP.values,
            "iteration_count": ITERATION_COUNT_SWEEP.values,
            "adapter_size": ADAPTER_SIZE_SWEEP.values,
            "temperature": TEMPERATURE_SWEEP.values,
            "batch_size": BATCH_SIZE_SWEEP.values,
        },
        "fixed_hyperparameters": {
            "batch_size_128": batch_size_128,
            "batch_size_64": batch_size_64,
            "temperature_default": TEMPERATURE_DEFAULT,
            "judge_model": JUDGE_MODEL_DEFAULT,
            "adapter_size_default": ADAPTER_SIZE_DEFAULT,
        },
        "datasets": sorted(DATASET_REGISTRY.keys()),
        "environments": sorted(ENVIRONMENT_REGISTRY.keys()),
        "method_aliases_count": len(METHOD_ALIAS_MAP),
    }
    out_path = out_dir / "scope_report.json"
    out_path.write_text(json.dumps(artifact, indent=2))
    logger.info("Wrote %s", out_path)
    return artifact


def write_data_manifest(
    results_dir: Union[str, Path] = "results",
) -> Dict[str, Any]:
    """
    Write results/data_manifest.json with dataset split and protocol information.
    """
    out_dir = _ensure_results_dir(results_dir)
    datasets: Dict[str, Any] = {}
    for ds_id, ds_data in DATASET_REGISTRY.items():
        datasets[ds_id] = {
            "name": ds_data["name"],
            "task": ds_data["task"],
            "feedback": ds_data["feedback"],
            "metric": ds_data["metric"],
            "secondary_metric": ds_data.get("secondary_metric"),
            "splits": ds_data.get("splits", {}),
            "hf_path": ds_data["hf_path"],
            "format": ds_data.get("format", "qa"),
            "eval_protocol": ds_data.get("eval_protocol", "accuracy"),
            "num_choices": ds_data.get("num_choices"),
        }
    artifact: Dict[str, Any] = {
        "artifact_type": "data_manifest",
        "version": __version__,
        "total_datasets": len(DATASET_REGISTRY),
        "datasets": datasets,
    }
    out_path = out_dir / "data_manifest.json"
    out_path.write_text(json.dumps(artifact, indent=2))
    logger.info("Wrote %s", out_path)
    return artifact


def write_all_artifacts(
    results_dir: Union[str, Path] = "results",
) -> Dict[str, Any]:
    """
    Write all config-level artifacts to the results directory.

    Materializes:
      results/environment_registry.json
      results/dataset_registry.json
      results/scope_report.json
      results/data_manifest.json

    Returns a summary dict keyed by artifact type.
    """
    out_dir = _ensure_results_dir(results_dir)
    summary: Dict[str, Any] = {
        "environment_registry": write_environment_registry(out_dir),
        "dataset_registry": write_dataset_registry(out_dir),
        "scope_report": write_scope_report(out_dir),
        "data_manifest": write_data_manifest(out_dir),
    }
    return summary


# ===========================================================================
# Default Experiment Matrix (paper-derived)
# Bounded to decisive comparisons; full sweep is in configs/experiment_matrix.yaml
# ===========================================================================

DEFAULT_EXPERIMENT_MATRIX: List[Dict[str, Any]] = [
    # --- Core comparison Table 2 ---
    {
        "experiment_id": "main_gsm8k_bbox",
        "dataset": "gsm8k",
        "environment": "gpt-3.5-turbo",
        "method": "bbox_adapter",
        "feedback_mode": "ground_truth",
        "beam_size": 3,
        "num_iterations": 4,
        "adapter_size": 0.3,
        "batch_size": batch_size_128,
    },
    {
        "experiment_id": "main_strategyqa_bbox",
        "dataset": "strategyqa",
        "environment": "gpt-3.5-turbo",
        "method": "bbox_adapter",
        "feedback_mode": "ai_feedback",
        "beam_size": 3,
        "num_iterations": 4,
        "adapter_size": 0.3,
        "batch_size": batch_size_128,
    },
    {
        "experiment_id": "main_truthfulqa_bbox",
        "dataset": "truthfulqa",
        "environment": "gpt-3.5-turbo",
        "method": "bbox_adapter",
        "feedback_mode": "combined",
        "beam_size": 3,
        "num_iterations": 4,
        "adapter_size": 0.3,
        "batch_size": batch_size_128,
    },
    {
        "experiment_id": "main_scienceqa_bbox",
        "dataset": "scienceqa",
        "environment": "gpt-3.5-turbo",
        "method": "bbox_adapter",
        "feedback_mode": "ground_truth",
        "beam_size": 3,
        "num_iterations": 4,
        "adapter_size": 0.3,
        "batch_size": batch_size_128,
    },
    # --- Baselines Table 2 ---
    {
        "experiment_id": "baseline_gsm8k_cot",
        "dataset": "gsm8k",
        "environment": "gpt-3.5-turbo",
        "method": "chain_of_thought",
        "feedback_mode": "none",
        "beam_size": 1,
        "num_iterations": 0,
        "batch_size": batch_size_128,
    },
    {
        "experiment_id": "baseline_gsm8k_azure_sft",
        "dataset": "gsm8k",
        "environment": "azure-gpt-35-turbo",
        "method": "azure_sft",
        "feedback_mode": "ground_truth",
        "batch_size": batch_size_64,
    },
    {
        "experiment_id": "baseline_gsm8k_lora",
        "dataset": "gsm8k",
        "environment": "gpt-3.5-turbo",
        "method": "lora",
        "lora_rank": 16,
        "lora_alpha": 256,
        "batch_size": batch_size_128,
    },
    # --- Ablation: adapter size (Table 2) ---
    {
        "experiment_id": "ablation_strategyqa_adapter_01",
        "dataset": "strategyqa",
        "environment": "gpt-3.5-turbo",
        "method": "bbox_adapter",
        "feedback_mode": "ai_feedback",
        "adapter_size": 0.1,
        "batch_size": batch_size_128,
    },
    {
        "experiment_id": "ablation_strategyqa_adapter_03",
        "dataset": "strategyqa",
        "environment": "gpt-3.5-turbo",
        "method": "bbox_adapter",
        "feedback_mode": "ai_feedback",
        "adapter_size": 0.3,
        "batch_size": batch_size_128,
    },
    # --- NCE vs MLM ablation (Table 5) ---
    {
        "experiment_id": "ablation_gsm8k_mlm",
        "dataset": "gsm8k",
        "environment": "gpt-3.5-turbo",
        "method": "mlm",
        "batch_size": batch_size_128,
    },
    {
        "experiment_id": "ablation_gsm8k_nce",
        "dataset": "gsm8k",
        "environment": "gpt-3.5-turbo",
        "method": "ranking_nce",
        "batch_size": batch_size_128,
    },
    # --- Toxicity reduction (Table 7) ---
    {
        "experiment_id": "toxigen_bbox",
        "dataset": "toxigen",
        "environment": "gpt-3.5-turbo",
        "method": "bbox_adapter",
        "feedback_mode": "ai_feedback",
        "batch_size": batch_size_128,
    },
    # --- Plug-and-play Table 3 ---
    {
        "experiment_id": "plug_strategyqa_mixtral",
        "dataset": "strategyqa",
        "environment": "mixtral-8x7b",
        "method": "bbox_adapter",
        "feedback_mode": "ai_feedback",
        "batch_size": batch_size_128,
    },
    {
        "experiment_id": "plug_gsm8k_davinci002",
        "dataset": "gsm8k",
        "environment": "davinci-002",
        "method": "bbox_adapter",
        "feedback_mode": "ground_truth",
        "batch_size": batch_size_128,
    },
]