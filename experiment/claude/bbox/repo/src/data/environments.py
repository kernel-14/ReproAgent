"""
src/data/environments.py

BBox-Adapter: Environment and Dataset Registry

Paper: BBox-Adapter: Lightweight Adapting for Black-Box Large Language Models

Exposes:
  - DATASET_REGISTRY: paper-derived dataset/benchmark entries with ids, aliases,
    paper-specified split ratios, loader hooks, and metric specs for all 5 benchmarks.
  - ENVIRONMENT_REGISTRY: paper-derived environment/task entries with ids, aliases,
    API endpoint configs, access-level metadata, and factory hooks for black-box LLMs.
  - make_environment(config): factory for TaskEnvironment wrappers.
  - make_dataset(config): factory for DatasetEnvironment wrappers.
  - write_all_artifacts(): writes all declared JSON artifacts for the work package.

Reference grounding: paperbench_ref_002 src/models/qa/transformer_qa.py
Reference grounding: paperbench_ref_005 toxigen/alice.py
Reference grounding: paperbench_ref_006 readme.md
Reference grounding: paperbench_ref_006 research/readme_exp.md
"""

from __future__ import annotations

import importlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset Registry
# Paper-specified split ratios preserved exactly.
# ---------------------------------------------------------------------------

DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    # -----------------------------------------------------------------------
    # GSM8K — Grade School Math (math reasoning)
    # Paper: Table 2, Table 3, Table 4; ground-truth feedback
    # reference_grounding: paperbench_ref_006 readme.md
    #   "On GSM8K, gpt-3.5-turbo improves over text-davinci-003 ..."
    # -----------------------------------------------------------------------
    "gsm8k": {
        "id": "gsm8k",
        "aliases": [
            "grade_school_math", "gsm", "math_reasoning",
            "GSM8K", "grade-school-math",
        ],
        "description": (
            "GSM8K (Grade School Math): 8,500 linguistically diverse grade-school "
            "math word problems requiring multi-step arithmetic reasoning."
        ),
        "task_type": "math_reasoning",
        "paper_task_label": "GSM8K (math reasoning)",
        "hf_path": "openai/gsm8k",
        "hf_name": "main",
        "splits": {
            "train": {"name": "train", "size": 7473},
            "test": {"name": "test", "size": 1319},
        },
        # Paper-specified split ratios — preserved from paper
        "split_ratios": {"train": 7473, "test": 1319},
        "primary_eval_split": "test",
        "answer_type": "numeric",
        "answer_extractor": "regex_boxed_or_last_number",
        "metric": "exact_match_numeric",
        "metric_formula": "accuracy = sum(predicted_num == gold_num) / N",
        "feedback_mode": "ground_truth",
        "loader_hook": "src.data.gsm8k.load_gsm8k",
        "prompt_style": "chain_of_thought",
        "few_shot_examples": 8,
        "paper_table": "Table 2, Table 3, Table 4",
        "paper_section": "Main Results, Plug-and-Play, Cost Analysis",
        "in_scope": True,
        "normalization": None,
        "sparse_reward": False,
        "reward_type": "exact_match",
    },

    # -----------------------------------------------------------------------
    # StrategyQA — Implicit reasoning (binary yes/no)
    # Paper: Table 2, Table 3, Table 4, Table 5, Table 6; AI feedback
    # reference_grounding: paperbench_ref_002 src/models/qa/transformer_qa.py
    #   forward(question_with_context, context_span, yes_no_span, answer_span, metadata)
    #   yes_no_span mirrors binary label (yes/no) in strategyqa
    # -----------------------------------------------------------------------
    "strategyqa": {
        "id": "strategyqa",
        "aliases": [
            "strategy_qa", "strategyqa_implicit", "StrategyQA",
            "implicit_reasoning", "strategy-qa",
        ],
        "description": (
            "StrategyQA (implicit reasoning): Multi-hop yes/no questions requiring "
            "implicit reasoning chains derived from world knowledge."
        ),
        "task_type": "implicit_reasoning",
        "paper_task_label": "StrategyQA (implicit reasoning)",
        "hf_path": "wics/strategy-qa",
        "hf_name": None,
        "splits": {
            "train": {"name": "train", "size": 2290},
            "test": {"name": "test", "size": 490},
        },
        # Paper-specified split ratios
        "split_ratios": {"train": 2290, "test": 490},
        "primary_eval_split": "test",
        "answer_type": "binary",
        "answer_extractor": "yes_no_extractor",
        "metric": "exact_match_binary",
        "metric_formula": "accuracy = sum(predicted_yn == gold_yn) / N",
        "feedback_mode": "ai_feedback",
        "loader_hook": "src.data.strategyqa.load_strategyqa",
        "prompt_style": "chain_of_thought",
        "few_shot_examples": 6,
        "paper_table": "Table 2, Table 3, Table 4, Table 5, Table 6",
        "paper_section": "Main Results, AI Feedback, Evaluation Details",
        "in_scope": True,
        "normalization": None,
        "sparse_reward": True,
        "reward_type": "ai_judge_binary",
        # TransformerQA-style yes_no_span grounding
        # reference_grounding: paperbench_ref_002 src/models/qa/transformer_qa.py
        "qa_format": {
            "question_field": "question",
            "context_field": None,
            "answer_field": "answer",
            "yes_no_span": True,
        },
    },

    # -----------------------------------------------------------------------
    # TruthfulQA — Truthfulness evaluation (multiple choice)
    # Paper: Table 2, Table 3; combined feedback
    # -----------------------------------------------------------------------
    "truthfulqa": {
        "id": "truthfulqa",
        "aliases": [
            "truthful_qa", "TruthfulQA", "truthfulness",
            "truthful-qa",
        ],
        "description": (
            "TruthfulQA (truthfulness): 817 questions across 38 categories measuring "
            "whether LLMs generate truthful answers. MC1 and MC2 accuracy reported."
        ),
        "task_type": "truthfulness",
        "paper_task_label": "TruthfulQA (truthfulness)",
        "hf_path": "truthful_qa",
        "hf_name": "multiple_choice",
        "splits": {
            "validation": {"name": "validation", "size": 817},
        },
        # Paper uses the full validation split (817 questions)
        "split_ratios": {"validation": 817},
        "primary_eval_split": "validation",
        "answer_type": "multiple_choice",
        "answer_extractor": "mc_lprob_extractor",
        "metric": "mc1_accuracy",
        "metric_formula": (
            "MC1 = (argmax lprob == gold_true); "
            "MC2 = sum(lprob[true]) / (sum(lprob[true]) + sum(lprob[false]))"
        ),
        "feedback_mode": "combined",
        "loader_hook": "src.data.truthfulqa.load_truthfulqa",
        "prompt_style": "chain_of_thought",
        "few_shot_examples": 6,
        "paper_table": "Table 2, Table 3",
        "paper_section": "Main Results, Plug-and-Play",
        "in_scope": True,
        "normalization": None,
        "sparse_reward": False,
        "reward_type": "mc_score",
        "mc_format": {
            "mc1": True,
            "mc2": True,
        },
    },

    # -----------------------------------------------------------------------
    # ScienceQA — Science domain multi-choice QA
    # Paper: Table 2, Table 3; ground-truth feedback
    # -----------------------------------------------------------------------
    "scienceqa": {
        "id": "scienceqa",
        "aliases": [
            "science_qa", "ScienceQA", "science_domain",
            "science-qa",
        ],
        "description": (
            "ScienceQA (science domain): 21,208 multimodal multiple-choice questions "
            "covering diverse science topics. Text-only subset used in BBox-Adapter."
        ),
        "task_type": "science_domain",
        "paper_task_label": "ScienceQA (science domain)",
        "hf_path": "derek-thomas/ScienceQA",
        "hf_name": None,
        "splits": {
            "train": {"name": "train", "size": 12726},
            "validation": {"name": "validation", "size": 4241},
            "test": {"name": "test", "size": 2017},
        },
        # Paper-specified split ratios
        "split_ratios": {"train": 12726, "validation": 4241, "test": 2017},
        "primary_eval_split": "test",
        "answer_type": "multiple_choice",
        "answer_extractor": "choice_letter_extractor",
        "metric": "exact_match_choice",
        "metric_formula": "accuracy = sum(predicted_choice == gold_choice) / N",
        "feedback_mode": "ground_truth",
        "loader_hook": "src.data.scienceqa.load_scienceqa",
        "prompt_style": "chain_of_thought",
        "few_shot_examples": 8,
        "paper_table": "Table 2, Table 3",
        "paper_section": "Main Results, Plug-and-Play",
        "in_scope": True,
        "normalization": None,
        "sparse_reward": False,
        "reward_type": "exact_match",
    },

    # -----------------------------------------------------------------------
    # ToxiGen — Toxicity reduction via AI feedback
    # Paper: Table 2; AI feedback mode
    # reference_grounding: paperbench_ref_005 toxigen/alice.py
    #   beam_search(prompt, language_model, classifier, mode, device,
    #               end_token="\n", weights=[.5,.5], num_beams=10,
    #               vocab_size=100, max_length=30, length_penalty=1)
    #   BeamHypotheses(num_beams, max_length, length_penalty, early_stopping=False)
    # -----------------------------------------------------------------------
    "toxigen": {
        "id": "toxigen",
        "aliases": [
            "tox_gen", "ToxiGen", "toxicity_reduction",
            "hate_speech", "tox-gen",
        ],
        "description": (
            "ToxiGen (toxicity reduction): Large-scale machine-generated dataset for "
            "toxicity and hate-speech reduction across 13 demographic groups. "
            "BBox-Adapter reduces toxicity via AI-feedback-based online adaptation."
        ),
        "task_type": "toxicity_reduction",
        "paper_task_label": "ToxiGen (toxicity reduction)",
        "hf_path": "skg/toxigen-data",
        "hf_name": "annotated",
        "splits": {
            "train": {"name": "train", "size": 8960},
            "test": {"name": "test", "size": 940},
        },
        # Paper-specified split ratios
        "split_ratios": {"train": 8960, "test": 940},
        "primary_eval_split": "test",
        "answer_type": "generation",
        "answer_extractor": "toxicity_classifier",
        "metric": "toxicity_rate",
        "metric_formula": "toxicity_rate = sum(toxicity_score > threshold) / N",
        "feedback_mode": "ai_feedback",
        "loader_hook": "src.data.toxigen.load_toxigen",
        "prompt_style": "chain_of_thought",
        "few_shot_examples": 4,
        "paper_table": "Table 2",
        "paper_section": "Main Results, AI Feedback",
        "in_scope": True,
        "normalization": None,
        "sparse_reward": True,
        "reward_type": "ai_toxicity_classifier",
        # ToxiGen beam search config from alice.py
        # reference_grounding: paperbench_ref_005 toxigen/alice.py
        "toxigen_beam_search": {
            "num_beams": 10,
            "vocab_size": 100,
            "max_length": 30,
            "length_penalty": 1.0,
            "weights": [0.5, 0.5],
            "end_token": "\n",
            "mode": 0,   # 0 = non-toxic (target); 1 = toxic
            "early_stopping": False,
        },
        "toxicity_threshold": 0.5,
        "demographic_groups": 13,
    },
}

# Build alias -> canonical_id map (case-preserved + lowercase variants)
DATASET_ALIASES: Dict[str, str] = {}
for _ds_id, _ds_meta in DATASET_REGISTRY.items():
    DATASET_ALIASES[_ds_id] = _ds_id
    DATASET_ALIASES[_ds_id.lower()] = _ds_id
    for _alias in _ds_meta.get("aliases", []):
        DATASET_ALIASES[_alias] = _ds_id
        DATASET_ALIASES[_alias.lower()] = _ds_id


# ---------------------------------------------------------------------------
# Environment / Model Registry
# ---------------------------------------------------------------------------

ENVIRONMENT_REGISTRY: Dict[str, Dict[str, Any]] = {
    # -----------------------------------------------------------------------
    # gpt-3.5-turbo — Primary black-box LLM (Table 2, Table 4)
    # reference_grounding: paperbench_ref_006 readme.md
    #   "gpt-3.5-turbo is 10 times cheaper than text-davinci-003"
    # -----------------------------------------------------------------------
    "gpt-3.5-turbo": {
        "id": "gpt-3.5-turbo",
        "aliases": [
            "gpt35turbo", "gpt_3_5_turbo", "chatgpt",
            "gpt-3.5-turbo-0613", "gpt-3.5-turbo-1106", "turbo",
            "gpt-3.5-turbo API",
        ],
        "description": (
            "OpenAI GPT-3.5-turbo: Primary black-box LLM used in BBox-Adapter main "
            "experiments. Accessed via OpenAI Chat Completions API."
        ),
        "api_type": "openai_chat",
        "endpoint": "https://api.openai.com/v1/chat/completions",
        "azure_endpoint": None,
        "model_family": "gpt",
        # BBox-Adapter access level: black-box (no param, no logprob access)
        "access_level": "black_box",
        "has_logprobs": False,
        "has_param_access": False,
        "has_repr_access": False,
        "requires_retrieval_corpus": False,
        "uses_adapter": True,
        "paper_table": "Table 2, Table 3, Table 4",
        "paper_section": "Main Results, Cost Analysis, Evaluation Details",
        "cost_per_1k_tokens_input_usd": 0.0015,
        "cost_per_1k_tokens_output_usd": 0.002,
        "max_tokens": 4096,
        "in_scope": True,
        "in_scope_flag": "primary_black_box_llm",
        "factory_hook": "src.utils.llm_client.GPTClient",
        "env_var_key": "OPENAI_API_KEY",
        "setup_metadata": {
            "requires_key": True,
            "key_env_var": "OPENAI_API_KEY",
            "org_env_var": "OPENAI_ORG_ID",
            "default_temperature": 1.0,
            "default_max_tokens": 512,
            "supports_cot": True,
            "num_samples_per_question": 10,
        },
        "evaluation_details": {
            "temperature": 1.0,
            "num_candidates": 10,
            "beam_width": 10,
            "cot_prompt": True,
            "report_std_dev": True,
            "num_seeds": 3,
        },
    },

    # -----------------------------------------------------------------------
    # Mixtral-8x7B-v0.1 — Plug-and-play target (Table 3)
    # Paper: Table 3: "Results of plug-and-play adaptation on davinci-002 and
    #   Mixtral-8x7B across four datasets."
    # -----------------------------------------------------------------------
    "mixtral-8x7b": {
        "id": "mixtral-8x7b",
        "aliases": [
            "Mixtral-8x7B-v0", "Mixtral-8x7B-v0.1", "Mixtral-8x7B",
            "mixtral_8x7b", "mistralai/Mixtral-8x7B-v0.1",
            "Mixtral-8x7B-Instruct-v0.1", "mixtral",
            "Mixtral-8x7B API",
        ],
        "description": (
            "Mistral AI Mixtral-8x7B-v0.1: Sparse mixture-of-experts LLM used as "
            "plug-and-play target in Table 3. BBox-Adapter tuned on gpt-3.5-turbo "
            "is applied without retraining (zero-shot transfer)."
        ),
        "api_type": "huggingface_inference",
        "endpoint": (
            "https://api-inference.huggingface.co/models/"
            "mistralai/Mixtral-8x7B-Instruct-v0.1"
        ),
        "azure_endpoint": None,
        "model_family": "mixtral",
        "access_level": "black_box",
        "has_logprobs": False,
        "has_param_access": False,
        "has_repr_access": False,
        "requires_retrieval_corpus": False,
        "uses_adapter": True,
        "paper_table": "Table 3",
        "paper_section": "Plug-and-Play Adaptation, Main Results",
        "cost_per_1k_tokens_input_usd": None,
        "cost_per_1k_tokens_output_usd": None,
        "max_tokens": 4096,
        "in_scope": True,
        "in_scope_flag": "plug_and_play_target",
        "factory_hook": "src.utils.llm_client.MixtralClient",
        "env_var_key": "HF_API_TOKEN",
        "setup_metadata": {
            "requires_key": True,
            "key_env_var": "HF_API_TOKEN",
            "hf_hub_id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "default_temperature": 1.0,
            "default_max_tokens": 512,
            "supports_cot": True,
            "vram_gb_required": 90,
        },
        "evaluation_details": {
            "plug_and_play": True,
            "adapter_source": "gpt-3.5-turbo",
            "retrain_required": False,
        },
    },

    # -----------------------------------------------------------------------
    # Azure OpenAI endpoint — enterprise deployment
    # Paper section: Evaluation Details, API usage
    # -----------------------------------------------------------------------
    "azure-openai": {
        "id": "azure-openai",
        "aliases": [
            "azure_openai", "azure_gpt35", "azure_gpt-3.5-turbo",
            "azure_openai_endpoint", "AzureOpenAI",
            "Azure OpenAI endpoint",
        ],
        "description": (
            "Azure OpenAI Service: Enterprise endpoint for GPT-3.5-turbo or GPT-4. "
            "Same black-box API constraints as OpenAI; supports reproducible eval "
            "in gated/enterprise environments."
        ),
        "api_type": "azure_openai_chat",
        "endpoint": None,  # Set via AZURE_OPENAI_ENDPOINT
        "azure_endpoint": "https://{resource}.openai.azure.com/",
        "model_family": "gpt",
        "access_level": "black_box",
        "has_logprobs": False,
        "has_param_access": False,
        "has_repr_access": False,
        "requires_retrieval_corpus": False,
        "uses_adapter": True,
        "paper_table": "Evaluation Details",
        "paper_section": "Evaluation Details",
        "cost_per_1k_tokens_input_usd": 0.0015,
        "cost_per_1k_tokens_output_usd": 0.002,
        "max_tokens": 4096,
        "in_scope": True,
        "in_scope_flag": "enterprise_endpoint",
        "factory_hook": "src.utils.llm_client.AzureOpenAIClient",
        "env_var_key": "AZURE_OPENAI_API_KEY",
        "setup_metadata": {
            "requires_key": True,
            "key_env_var": "AZURE_OPENAI_API_KEY",
            "endpoint_env_var": "AZURE_OPENAI_ENDPOINT",
            "deployment_env_var": "AZURE_OPENAI_DEPLOYMENT",
            "api_version": "2024-02-01",
            "default_temperature": 1.0,
            "default_max_tokens": 512,
            "supports_cot": True,
        },
    },

    # -----------------------------------------------------------------------
    # davinci-002 — Additional plug-and-play target (Table 3)
    # -----------------------------------------------------------------------
    "davinci-002": {
        "id": "davinci-002",
        "aliases": [
            "text-davinci-002", "davinci002", "davinci_002",
        ],
        "description": (
            "OpenAI davinci-002: Used alongside Mixtral-8x7B as plug-and-play "
            "target in Table 3. BBox-Adapter trained on gpt-3.5-turbo transfers "
            "without retraining."
        ),
        "api_type": "openai_completion",
        "endpoint": "https://api.openai.com/v1/completions",
        "azure_endpoint": None,
        "model_family": "gpt",
        "access_level": "black_box",
        "has_logprobs": False,
        "has_param_access": False,
        "has_repr_access": False,
        "requires_retrieval_corpus": False,
        "uses_adapter": True,
        "paper_table": "Table 3",
        "paper_section": "Plug-and-Play Adaptation",
        "cost_per_1k_tokens_input_usd": 0.002,
        "cost_per_1k_tokens_output_usd": 0.002,
        "max_tokens": 4096,
        "in_scope": True,
        "in_scope_flag": "plug_and_play_target",
        "factory_hook": "src.utils.llm_client.GPTClient",
        "env_var_key": "OPENAI_API_KEY",
        "setup_metadata": {
            "requires_key": True,
            "key_env_var": "OPENAI_API_KEY",
            "default_temperature": 1.0,
            "default_max_tokens": 512,
            "supports_cot": True,
        },
        "evaluation_details": {
            "plug_and_play": True,
            "adapter_source": "gpt-3.5-turbo",
            "retrain_required": False,
        },
    },
}

# Build alias -> canonical_id map
ENVIRONMENT_ALIASES: Dict[str, str] = {}
for _env_id, _env_meta in ENVIRONMENT_REGISTRY.items():
    ENVIRONMENT_ALIASES[_env_id] = _env_id
    ENVIRONMENT_ALIASES[_env_id.lower()] = _env_id
    for _alias in _env_meta.get("aliases", []):
        ENVIRONMENT_ALIASES[_alias] = _env_id
        ENVIRONMENT_ALIASES[_alias.lower()] = _env_id


# ---------------------------------------------------------------------------
# Scope metadata (paper section → table → metric mappings)
# ---------------------------------------------------------------------------

SCOPE_METADATA: Dict[str, Any] = {
    "paper": "BBox-Adapter: Lightweight Adapting for Black-Box Large Language Models",
    "paper_sections": {
        "Main Results": {
            "table": "Table 2",
            "description": (
                "Adapting gpt-3.5-turbo on downstream tasks. BBox-Adapter best "
                "performance across 0.1B and 0.3B adapter sizes."
            ),
            "datasets": ["gsm8k", "strategyqa", "truthfulqa", "scienceqa", "toxigen"],
            "primary_model": "gpt-3.5-turbo",
            "adapter_sizes": ["0.1B", "0.3B"],
            "baselines": [
                "base_model",
                "chain_of_thought",
                "azure_sft",
                "sft_lora",
                "retrieval_augmented",
            ],
            "metrics": {
                "gsm8k": "exact_match_numeric",
                "strategyqa": "exact_match_binary",
                "truthfulqa": "mc1_accuracy",
                "scienceqa": "exact_match_choice",
                "toxigen": "toxicity_rate",
            },
        },
        "Plug-and-Play Adaptation": {
            "table": "Table 3",
            "description": (
                "Plug-and-play adaptation on davinci-002 and Mixtral-8x7B "
                "using adapter trained on gpt-3.5-turbo without retraining."
            ),
            "datasets": ["gsm8k", "strategyqa", "truthfulqa", "scienceqa"],
            "target_models": ["davinci-002", "mixtral-8x7b"],
            "source_model": "gpt-3.5-turbo",
            "retrain_required": False,
        },
        "Cost Analysis": {
            "table": "Table 4",
            "description": (
                "Comparison of performance and cost for base model, SFT, and "
                "BBox-Adapter on StrategyQA and GSM8K."
            ),
            "datasets": ["strategyqa", "gsm8k"],
            "comparison": ["base_model", "azure_sft", "bbox_adapter_0.1B", "bbox_adapter_0.3B"],
            "metrics": ["accuracy", "training_cost_usd", "inference_cost_usd"],
        },
        "AI Feedback": {
            "section": "5.3",
            "description": (
                "Using a smaller LLM as judge for datasets without direct reward signals. "
                "Applicable to StrategyQA (binary) and ToxiGen (toxicity)."
            ),
            "applicable_datasets": ["strategyqa", "toxigen"],
            "judge_model": "gpt-3.5-turbo",
            "feedback_type": "binary_correctness_or_toxicity",
        },
        "Evaluation Details": {
            "temperature": 1.0,
            "num_samples_per_question": 10,
            "beam_width": 10,
            "cot_prompt": True,
            "report_mean_std": True,
            "std_note": "Standard deviation reported across 3 seeds",
        },
        "Training Cost": {
            "adapter_sizes": {
                "0.1B": {
                    "backbone": "microsoft/deberta-v3-base",
                    "params_millions": 110,
                    "vram_gb": 4,
                    "approx_cost_usd": 0.5,
                },
                "0.3B": {
                    "backbone": "microsoft/deberta-v3-large",
                    "params_millions": 340,
                    "vram_gb": 8,
                    "approx_cost_usd": 1.2,
                },
            },
            "training_steps": 100,
            "batch_size": 8,
            "lr": 1e-4,
            "num_candidates": 10,
        },
        "Black-Box LLMs": {
            "key_constraint": (
                "No access to model parameters or token probabilities. "
                "Only text output is available via API."
            ),
            "api_access_only": True,
            "adapter_attachment_method": (
                "EBM re-ranks LLM text outputs post-generation. "
                "Adapter input: (question, candidate_answer) pairs."
            ),
            "table_1_access_matrix": {
                "white_box": {"params": True, "repr": True, "logprobs": True},
                "grey_box": {"params": False, "repr": False, "logprobs": True},
                "black_box": {"params": False, "repr": False, "logprobs": False},
            },
        },
        "Standard Deviation": {
            "num_seeds": 3,
            "reporting_format": "mean ± std",
            "applicable_to": "all benchmark accuracy/toxicity scores",
        },
        "achieving improvements": {
            "description": (
                "BBox-Adapter achieves consistent improvements over base model "
                "and other black-box baselines across all 5 benchmarks."
            ),
            "key_result": "Table 2 main results column",
        },
    },
    "in_scope_datasets": list(DATASET_REGISTRY.keys()),
    "in_scope_models": list(ENVIRONMENT_REGISTRY.keys()),
    "out_of_scope": [
        "fine-tuning with gradient access to LLM weights",
        "grey-box methods requiring logprob access",
        "white-box methods requiring parameter access",
        "retrieval augmented generation (evaluated as baseline only)",
    ],
}

# ---------------------------------------------------------------------------
# Cost / VRAM reference metadata (Table 4 / Appendix)
# ---------------------------------------------------------------------------

COST_VRAM_METADATA: Dict[str, Any] = {
    "adapter_sizes": {
        "0.1B": {
            "backbone": "microsoft/deberta-v3-base",
            "params_millions": 110,
            "vram_gb": 4,
            "training_cost_usd": 0.5,
            "inference_overhead_ms": 15,
            "paper_reference": "Table 4",
        },
        "0.3B": {
            "backbone": "microsoft/deberta-v3-large",
            "params_millions": 340,
            "vram_gb": 8,
            "training_cost_usd": 1.2,
            "inference_overhead_ms": 30,
            "paper_reference": "Table 4",
        },
    },
    "baseline_costs": {
        "base_model": {
            "training_cost_usd": 0.0,
            "note": "No training; direct API inference only.",
        },
        "azure_sft": {
            "training_cost_usd_range": [100, 500],
            "note": "Azure fine-tuning cost for gpt-3.5-turbo supervised fine-tuning.",
        },
        "sft_lora": {
            "training_cost_usd_range": [10, 50],
            "note": "LoRA fine-tuning on open-source model replica.",
        },
    },
    "api_costs": {
        "gpt-3.5-turbo": {
            "per_1k_input_tokens_usd": 0.0015,
            "per_1k_output_tokens_usd": 0.002,
        },
        "mixtral-8x7b": {
            "per_1k_input_tokens_usd": None,
            "note": "HuggingFace Inference API; pricing varies by tier.",
        },
        "davinci-002": {
            "per_1k_input_tokens_usd": 0.002,
            "per_1k_output_tokens_usd": 0.002,
        },
    },
    "paper_reference": "Table 4: Comparison of performance and cost",
}


# ---------------------------------------------------------------------------
# Dataclass interfaces (typed structured config)
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    """Typed config for a paper-registered dataset."""

    id: str
    aliases: List[str]
    hf_path: str
    hf_name: Optional[str]
    primary_eval_split: str
    answer_type: str
    metric: str
    metric_formula: str
    feedback_mode: str
    loader_hook: str
    prompt_style: str = "chain_of_thought"
    few_shot_examples: int = 6
    in_scope: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_registry(cls, dataset_id: str) -> "DatasetConfig":
        """Build DatasetConfig from registry entry."""
        entry = get_dataset(dataset_id)
        return cls(
            id=entry["id"],
            aliases=entry.get("aliases", []),
            hf_path=entry.get("hf_path", ""),
            hf_name=entry.get("hf_name"),
            primary_eval_split=entry.get("primary_eval_split", "test"),
            answer_type=entry.get("answer_type", "text"),
            metric=entry.get("metric", "exact_match"),
            metric_formula=entry.get("metric_formula", ""),
            feedback_mode=entry.get("feedback_mode", "ground_truth"),
            loader_hook=entry.get("loader_hook", ""),
            prompt_style=entry.get("prompt_style", "chain_of_thought"),
            few_shot_examples=entry.get("few_shot_examples", 6),
            in_scope=entry.get("in_scope", True),
        )


@dataclass
class EnvironmentConfig:
    """Typed config for a paper-registered model environment."""

    id: str
    aliases: List[str]
    api_type: str
    endpoint: Optional[str]
    azure_endpoint: Optional[str]
    model_family: str
    access_level: str
    has_logprobs: bool
    has_param_access: bool
    env_var_key: str
    factory_hook: str
    in_scope: bool = True
    max_tokens: int = 512
    default_temperature: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_registry(cls, env_id: str) -> "EnvironmentConfig":
        """Build EnvironmentConfig from registry entry."""
        entry = get_environment(env_id)
        setup = entry.get("setup_metadata", {})
        return cls(
            id=entry["id"],
            aliases=entry.get("aliases", []),
            api_type=entry.get("api_type", ""),
            endpoint=entry.get("endpoint"),
            azure_endpoint=entry.get("azure_endpoint"),
            model_family=entry.get("model_family", ""),
            access_level=entry.get("access_level", "black_box"),
            has_logprobs=entry.get("has_logprobs", False),
            has_param_access=entry.get("has_param_access", False),
            env_var_key=entry.get("env_var_key", ""),
            factory_hook=entry.get("factory_hook", ""),
            in_scope=entry.get("in_scope", True),
            max_tokens=entry.get("max_tokens", 512),
            default_temperature=setup.get("default_temperature", 1.0),
        )


# ---------------------------------------------------------------------------
# Wrapper classes: TaskEnvironment and DatasetEnvironment
# ---------------------------------------------------------------------------

class TaskEnvironment:
    """
    Wrapper for a paper-registered black-box LLM task environment.

    API client libraries (openai, requests) are imported lazily inside
    get_client() to avoid import failures in smoke environments.
    """

    def __init__(self, env_id: str, metadata: Dict[str, Any]) -> None:
        self.env_id = env_id
        self.metadata = metadata
        self._client = None

    def __repr__(self) -> str:
        return (
            f"TaskEnvironment(id={self.env_id!r}, "
            f"access_level={self.metadata.get('access_level')!r}, "
            f"api_type={self.metadata.get('api_type')!r})"
        )

    def is_ready(self) -> Tuple[bool, str]:
        """Check whether required API key environment variable is set."""
        env_var = self.metadata.get("env_var_key", "")
        if not env_var:
            return True, "No API key required for this environment."
        key_val = os.environ.get(env_var, "")
        if key_val:
            return True, f"API key found in env var {env_var!r}."
        return (
            False,
            (
                f"API key not found. Set environment variable {env_var!r} "
                f"to enable {self.env_id!r} inference."
            ),
        )

    def get_client(self):
        """
        Lazily initialize and return the LLM API client.
        All heavy imports deferred to avoid module-level failures.
        """
        if self._client is not None:
            return self._client

        api_type = self.metadata.get("api_type", "")
        api_key = os.environ.get(self.metadata.get("env_var_key", ""), "")

        if api_type in ("openai_chat", "openai_completion"):
            try:
                from src.utils.llm_client import GPTClient  # type: ignore
                self._client = GPTClient(
                    model=self.env_id,
                    api_key=api_key,
                    max_tokens=self.metadata.get("max_tokens", 512),
                    temperature=self.metadata.get(
                        "setup_metadata", {}
                    ).get("default_temperature", 1.0),
                )
            except ImportError as exc:
                raise ImportError(
                    f"Cannot import GPTClient for {self.env_id!r}. "
                    "Ensure 'openai' is installed: pip install openai"
                ) from exc

        elif api_type == "azure_openai_chat":
            try:
                from src.utils.llm_client import AzureOpenAIClient  # type: ignore
                setup = self.metadata.get("setup_metadata", {})
                self._client = AzureOpenAIClient(
                    api_key=os.environ.get(
                        setup.get("key_env_var", "AZURE_OPENAI_API_KEY"), ""
                    ),
                    endpoint=os.environ.get(
                        setup.get("endpoint_env_var", "AZURE_OPENAI_ENDPOINT"), ""
                    ),
                    deployment=os.environ.get(
                        setup.get("deployment_env_var", "AZURE_OPENAI_DEPLOYMENT"), ""
                    ),
                    api_version=setup.get("api_version", "2024-02-01"),
                )
            except ImportError as exc:
                raise ImportError(
                    f"Cannot import AzureOpenAIClient for {self.env_id!r}. "
                    "Ensure 'openai' is installed: pip install openai"
                ) from exc

        elif api_type == "huggingface_inference":
            try:
                from src.utils.llm_client import MixtralClient  # type: ignore
                hf_id = self.metadata.get("setup_metadata", {}).get("hf_hub_id", self.env_id)
                self._client = MixtralClient(
                    model=hf_id,
                    api_key=api_key,
                    max_tokens=self.metadata.get("max_tokens", 512),
                )
            except ImportError as exc:
                raise ImportError(
                    f"Cannot import MixtralClient for {self.env_id!r}."
                ) from exc

        else:
            raise ValueError(
                f"Unsupported api_type {api_type!r} for environment {self.env_id!r}. "
                "Supported: openai_chat, openai_completion, azure_openai_chat, "
                "huggingface_inference."
            )

        return self._client

    def generate(
        self,
        prompt: str,
        n: int = 1,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> List[str]:
        """
        Generate n candidate responses for a prompt.

        Parameters
        ----------
        prompt : str
            Input prompt (CoT-formatted per paper).
        n : int
            Number of candidate responses (paper uses 10).
        temperature : float, optional
            Sampling temperature. Defaults to 1.0 per paper.
        max_tokens : int, optional
            Max tokens per response. Defaults to 512.

        Returns
        -------
        List[str]
            List of n generated text responses.
        """
        client = self.get_client()
        _temp = temperature if temperature is not None else (
            self.metadata.get("setup_metadata", {}).get("default_temperature", 1.0)
        )
        _max_tok = max_tokens if max_tokens is not None else (
            self.metadata.get("setup_metadata", {}).get("default_max_tokens", 512)
        )
        return client.generate(prompt=prompt, n=n, temperature=_temp, max_tokens=_max_tok)

    def to_dict(self) -> Dict[str, Any]:
        return {"env_id": self.env_id, "metadata": self.metadata}


class DatasetEnvironment:
    """
    Wrapper for a paper-registered dataset/benchmark.

    Dataset loading is deferred via lazy imports; smoke import does not
    require the 'datasets' package.
    """

    def __init__(self, dataset_id: str, metadata: Dict[str, Any]) -> None:
        self.dataset_id = dataset_id
        self.metadata = metadata
        self._data: Dict[str, List[Dict[str, Any]]] = {}

    def __repr__(self) -> str:
        return (
            f"DatasetEnvironment(id={self.dataset_id!r}, "
            f"task_type={self.metadata.get('task_type')!r}, "
            f"feedback_mode={self.metadata.get('feedback_mode')!r})"
        )

    def get_split_sizes(self) -> Dict[str, int]:
        """Return paper-specified split sizes for this dataset."""
        return dict(self.metadata.get("split_ratios", {}))

    def get_metric(self) -> str:
        """Return canonical metric name for this dataset."""
        return self.metadata.get("metric", "exact_match")

    def get_metric_formula(self) -> str:
        """Return human-readable metric formula string."""
        return self.metadata.get("metric_formula", "")

    def get_feedback_mode(self) -> str:
        """Return paper-specified feedback mode (ground_truth/ai_feedback/combined)."""
        return self.metadata.get("feedback_mode", "ground_truth")

    def is_ready(self) -> Tuple[bool, str]:
        """Check whether 'datasets' package is available for loading."""
        try:
            importlib.import_module("datasets")
            return (
                True,
                f"'datasets' package available; {self.dataset_id!r} can be loaded.",
            )
        except ImportError:
            return (
                False,
                (
                    f"'datasets' package not installed. Cannot load {self.dataset_id!r}. "
                    "Install with: pip install datasets"
                ),
            )

    def load_split(self, split: str = "test") -> List[Dict[str, Any]]:
        """
        Load a dataset split using the registered loader hook.

        Parameters
        ----------
        split : str
            Split name: 'train', 'validation', or 'test'.

        Returns
        -------
        List[Dict[str, Any]]
            List of QA samples in standardized format:
              {"id": str, "question": str, "answer": str, "choices": Optional[List]}
        """
        if split in self._data:
            return self._data[split]

        loader_hook = self.metadata.get("loader_hook", "")
        if not loader_hook:
            raise ValueError(
                f"No loader_hook registered for dataset {self.dataset_id!r}."
            )

        module_path, func_name = loader_hook.rsplit(".", 1)
        try:
            mod = importlib.import_module(module_path)
            loader_fn = getattr(mod, func_name)
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                f"Could not import loader {loader_hook!r} for {self.dataset_id!r}: {exc}"
            ) from exc

        data: List[Dict[str, Any]] = loader_fn(split=split)
        self._data[split] = data
        return data

    def sample(
        self, split: str = "test", n: Optional[int] = None, seed: int = 42
    ) -> List[Dict[str, Any]]:
        """
        Load and optionally subsample a dataset split.

        Parameters
        ----------
        split : str
            Split name.
        n : int, optional
            Number of samples to draw. If None, returns full split.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        List[Dict[str, Any]]
            Sampled dataset records.
        """
        import random  # stdlib only

        data = self.load_split(split)
        if n is None or n >= len(data):
            return data
        rng = random.Random(seed)
        return rng.sample(data, n)

    def to_dict(self) -> Dict[str, Any]:
        return {"dataset_id": self.dataset_id, "metadata": self.metadata}


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def make_environment(config: Union[str, Dict[str, Any]]) -> TaskEnvironment:
    """
    Create a TaskEnvironment from an environment id string or config dict.

    Parameters
    ----------
    config : str or dict
        String id/alias (e.g. 'gpt-3.5-turbo', 'Mixtral-8x7B-v0') or dict
        with 'id' key.

    Returns
    -------
    TaskEnvironment

    Raises
    ------
    KeyError
        If environment is not in the registry.
    TypeError
        If config type is unsupported.
    """
    if isinstance(config, str):
        canonical = (
            ENVIRONMENT_ALIASES.get(config)
            or ENVIRONMENT_ALIASES.get(config.lower())
        )
        if canonical is None:
            raise KeyError(
                f"Unknown environment {config!r}. "
                f"Registered: {sorted(set(ENVIRONMENT_REGISTRY.keys()))}"
            )
        return TaskEnvironment(env_id=canonical, metadata=ENVIRONMENT_REGISTRY[canonical])

    if isinstance(config, dict):
        raw_id = config.get("id", "")
        canonical = (
            ENVIRONMENT_ALIASES.get(raw_id)
            or ENVIRONMENT_ALIASES.get(raw_id.lower())
            or raw_id
        )
        meta = ENVIRONMENT_REGISTRY.get(canonical, config)
        return TaskEnvironment(env_id=canonical, metadata=meta)

    raise TypeError(f"config must be str or dict, got {type(config).__name__}")


def make_dataset(config: Union[str, Dict[str, Any]]) -> DatasetEnvironment:
    """
    Create a DatasetEnvironment from a dataset id string or config dict.

    Parameters
    ----------
    config : str or dict
        String id/alias (e.g. 'gsm8k', 'grade_school_math') or dict with 'id' key.

    Returns
    -------
    DatasetEnvironment

    Raises
    ------
    KeyError
        If dataset is not in the registry.
    TypeError
        If config type is unsupported.
    """
    if isinstance(config, str):
        canonical = (
            DATASET_ALIASES.get(config)
            or DATASET_ALIASES.get(config.lower())
        )
        if canonical is None:
            raise KeyError(
                f"Unknown dataset {config!r}. "
                f"Registered: {sorted(set(DATASET_REGISTRY.keys()))}"
            )
        return DatasetEnvironment(dataset_id=canonical, metadata=DATASET_REGISTRY[canonical])

    if isinstance(config, dict):
        raw_id = config.get("id", "")
        canonical = (
            DATASET_ALIASES.get(raw_id)
            or DATASET_ALIASES.get(raw_id.lower())
            or raw_id
        )
        meta = DATASET_REGISTRY.get(canonical, config)
        return DatasetEnvironment(dataset_id=canonical, metadata=meta)

    raise TypeError(f"config must be str or dict, got {type(config).__name__}")


# ---------------------------------------------------------------------------
# Registry lookup helpers
# ---------------------------------------------------------------------------

def get_dataset(dataset_id: str) -> Dict[str, Any]:
    """
    Retrieve a dataset registry entry by canonical id or alias.

    Parameters
    ----------
    dataset_id : str
        Canonical id or alias, e.g. 'gsm8k', 'grade_school_math', 'GSM8K'.

    Returns
    -------
    Dict[str, Any]

    Raises
    ------
    KeyError
    """
    canonical = (
        DATASET_ALIASES.get(dataset_id)
        or DATASET_ALIASES.get(dataset_id.lower())
    )
    if canonical is None:
        raise KeyError(
            f"Unknown dataset {dataset_id!r}. "
            f"Registered: {sorted(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[canonical]


def get_environment(env_id: str) -> Dict[str, Any]:
    """
    Retrieve an environment registry entry by canonical id or alias.

    Parameters
    ----------
    env_id : str
        Canonical id or alias, e.g. 'gpt-3.5-turbo', 'Mixtral-8x7B-v0'.

    Returns
    -------
    Dict[str, Any]

    Raises
    ------
    KeyError
    """
    canonical = (
        ENVIRONMENT_ALIASES.get(env_id)
        or ENVIRONMENT_ALIASES.get(env_id.lower())
    )
    if canonical is None:
        raise KeyError(
            f"Unknown environment {env_id!r}. "
            f"Registered: {sorted(ENVIRONMENT_REGISTRY.keys())}"
        )
    return ENVIRONMENT_REGISTRY[canonical]


def list_datasets() -> List[str]:
    """Return sorted list of canonical dataset ids."""
    return sorted(DATASET_REGISTRY.keys())


def list_environments() -> List[str]:
    """Return sorted list of canonical environment ids."""
    return sorted(ENVIRONMENT_REGISTRY.keys())


def check_readiness() -> Dict[str, Any]:
    """
    Run readiness checks for all registered environments and datasets.

    Returns
    -------
    Dict[str, Any]
        {
          "environments": {env_id: {"ready": bool, "message": str}},
          "datasets": {ds_id: {"ready": bool, "message": str}},
          "overall_ready": bool,
        }
    """
    report: Dict[str, Any] = {
        "environments": {},
        "datasets": {},
        "overall_ready": True,
    }

    for env_id in ENVIRONMENT_REGISTRY:
        env = make_environment(env_id)
        ready, msg = env.is_ready()
        report["environments"][env_id] = {"ready": ready, "message": msg}
        if not ready:
            report["overall_ready"] = False

    for ds_id in DATASET_REGISTRY:
        ds = make_dataset(ds_id)
        ready, msg = ds.is_ready()
        report["datasets"][ds_id] = {"ready": ready, "message": msg}
        # Dataset unreadiness is advisory only (HF datasets may not be installed)

    return report


# ---------------------------------------------------------------------------
# Artifact writers
# ---------------------------------------------------------------------------

def _resolve_output_dir(output_dir: Optional[str]) -> str:
    if output_dir is not None:
        return output_dir
    return os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR", "results")


def write_environment_registry(output_dir: Optional[str] = None) -> str:
    """Write environment registry JSON artifact."""
    out_dir = _resolve_output_dir(output_dir)
    out_path = Path(out_dir) / "environment_registry.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "schema_version": "1.0",
        "artifact_type": "environment_registry",
        "paper": "BBox-Adapter: Lightweight Adapting for Black-Box Large Language Models",
        "registered_environments": list_environments(),
        "environment_count": len(ENVIRONMENT_REGISTRY),
        "environments": ENVIRONMENT_REGISTRY,
        "aliases": {k: v for k, v in ENVIRONMENT_ALIASES.items() if k != v},
        "scope_metadata": SCOPE_METADATA["paper_sections"],
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Wrote environment registry -> %s", out_path)
    return str(out_path)


def write_dataset_registry(output_dir: Optional[str] = None) -> str:
    """Write dataset registry JSON artifact."""
    out_dir = _resolve_output_dir(output_dir)
    out_path = Path(out_dir) / "dataset_registry.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "schema_version": "1.0",
        "artifact_type": "dataset_registry",
        "paper": "BBox-Adapter: Lightweight Adapting for Black-Box Large Language Models",
        "registered_datasets": list_datasets(),
        "dataset_count": len(DATASET_REGISTRY),
        "datasets": DATASET_REGISTRY,
        "aliases": {k: v for k, v in DATASET_ALIASES.items() if k != v},
        "split_ratios": {
            ds_id: ds_meta.get("split_ratios", {})
            for ds_id, ds_meta in DATASET_REGISTRY.items()
        },
        "metrics": {
            ds_id: {
                "metric": ds_meta.get("metric"),
                "metric_formula": ds_meta.get("metric_formula"),
                "feedback_mode": ds_meta.get("feedback_mode"),
            }
            for ds_id, ds_meta in DATASET_REGISTRY.items()
        },
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Wrote dataset registry -> %s", out_path)
    return str(out_path)


def write_scope_report(output_dir: Optional[str] = None) -> str:
    """Write scope report (in-scope/out-of-scope experiment matrix) JSON artifact."""
    out_dir = _resolve_output_dir(output_dir)
    out_path = Path(out_dir) / "scope_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "schema_version": "1.0",
        "artifact_type": "scope_report",
        "paper": "BBox-Adapter: Lightweight Adapting for Black-Box Large Language Models",
        "in_scope_datasets": list_datasets(),
        "in_scope_environments": list_environments(),
        "out_of_scope": SCOPE_METADATA["out_of_scope"],
        "paper_sections": SCOPE_METADATA["paper_sections"],
        "experiment_matrix": {
            ds_id: {
                "feedback_mode": ds_meta.get("feedback_mode"),
                "metric": ds_meta.get("metric"),
                "paper_table": ds_meta.get("paper_table"),
                "primary_model": "gpt-3.5-turbo",
                "plug_and_play_targets": (
                    ["davinci-002", "mixtral-8x7b"]
                    if ds_id in ["gsm8k", "strategyqa", "truthfulqa", "scienceqa"]
                    else []
                ),
            }
            for ds_id, ds_meta in DATASET_REGISTRY.items()
        },
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Wrote scope report -> %s", out_path)
    return str(out_path)


def write_data_manifest(output_dir: Optional[str] = None) -> str:
    """Write data manifest (splits / loader hooks) JSON artifact."""
    out_dir = _resolve_output_dir(output_dir)
    out_path = Path(out_dir) / "data_manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[Dict[str, Any]] = []
    for ds_id, ds_meta in DATASET_REGISTRY.items():
        for split_name, split_info in ds_meta.get("splits", {}).items():
            manifest_rows.append(
                {
                    "dataset_id": ds_id,
                    "task_type": ds_meta.get("task_type"),
                    "split": split_name,
                    "size": split_info.get("size"),
                    "hf_path": ds_meta.get("hf_path"),
                    "hf_name": ds_meta.get("hf_name"),
                    "loader_hook": ds_meta.get("loader_hook"),
                    "metric": ds_meta.get("metric"),
                    "feedback_mode": ds_meta.get("feedback_mode"),
                    "prompt_style": ds_meta.get("prompt_style"),
                    "few_shot_examples": ds_meta.get("few_shot_examples"),
                }
            )

    payload: Dict[str, Any] = {
        "schema_version": "1.0",
        "artifact_type": "data_manifest",
        "paper": "BBox-Adapter: Lightweight Adapting for Black-Box Large Language Models",
        "total_splits": len(manifest_rows),
        "splits": manifest_rows,
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Wrote data manifest -> %s", out_path)
    return str(out_path)


def write_cost_vram_report(output_dir: Optional[str] = None) -> str:
    """Write cost/VRAM report (Table 4) JSON artifact."""
    out_dir = _resolve_output_dir(output_dir)
    out_path = Path(out_dir) / "cost_vram_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "schema_version": "1.0",
        "artifact_type": "cost_vram_report",
        "paper": "BBox-Adapter: Lightweight Adapting for Black-Box Large Language Models",
        "paper_table": "Table 4: Comparison of performance and cost",
        "paper_section": "Cost Analysis, Training Cost",
        "adapter_sizes": COST_VRAM_METADATA["adapter_sizes"],
        "baseline_costs": COST_VRAM_METADATA["baseline_costs"],
        "api_costs": COST_VRAM_METADATA["api_costs"],
        "environment_api_costs": {
            env_id: {
                "per_1k_input_tokens_usd": env_meta.get("cost_per_1k_tokens_input_usd"),
                "per_1k_output_tokens_usd": env_meta.get("cost_per_1k_tokens_output_usd"),
            }
            for env_id, env_meta in ENVIRONMENT_REGISTRY.items()
        },
        "training_hyperparams": SCOPE_METADATA["paper_sections"]["Training Cost"],
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Wrote cost/VRAM report -> %s", out_path)
    return str(out_path)


def write_metrics_schema(output_dir: Optional[str] = None) -> str:
    """Write metrics schema (per-dataset metric definitions) JSON artifact."""
    out_dir = _resolve_output_dir(output_dir)
    out_path = Path(out_dir) / "metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_schema: Dict[str, Any] = {}
    for ds_id, ds_meta in DATASET_REGISTRY.items():
        metrics_schema[ds_id] = {
            "metric": ds_meta.get("metric"),
            "metric_formula": ds_meta.get("metric_formula"),
            "answer_type": ds_meta.get("answer_type"),
            "answer_extractor": ds_meta.get("answer_extractor"),
            "feedback_mode": ds_meta.get("feedback_mode"),
            "sparse_reward": ds_meta.get("sparse_reward", False),
            "reward_type": ds_meta.get("reward_type"),
            "paper_table": ds_meta.get("paper_table"),
            "paper_section": ds_meta.get("paper_section"),
        }

    payload: Dict[str, Any] = {
        "schema_version": "1.0",
        "artifact_type": "metrics_schema",
        "paper": "BBox-Adapter: Lightweight Adapting for Black-Box Large Language Models",
        "per_dataset_metrics": metrics_schema,
        "standard_deviation_config": SCOPE_METADATA["paper_sections"]["Standard Deviation"],
        "evaluation_details": SCOPE_METADATA["paper_sections"]["Evaluation Details"],
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Wrote metrics schema -> %s", out_path)
    return str(out_path)


def write_all_artifacts(output_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Write all declared artifact files for the environment_setup work package.

    Parameters
    ----------
    output_dir : str, optional
        Output directory. Falls back to PAPERBENCH_REPRO_ARTIFACT_DIR or 'results'.

    Returns
    -------
    Dict[str, str]
        Mapping of artifact name -> absolute file path.
    """
    out_dir = _resolve_output_dir(output_dir)
    return {
        "environment_registry": write_environment_registry(out_dir),
        "dataset_registry": write_dataset_registry(out_dir),
        "scope_report": write_scope_report(out_dir),
        "data_manifest": write_data_manifest(out_dir),
        "cost_vram_report": write_cost_vram_report(out_dir),
        "metrics": write_metrics_schema(out_dir),
    }


# ---------------------------------------------------------------------------
# Smoke / readiness validator
# ---------------------------------------------------------------------------

def _run_smoke(output_dir: Optional[str] = None) -> Tuple[bool, List[str]]:
    """
    Validate registry contents, alias resolution, factories, and artifact writing.

    Returns
    -------
    Tuple[bool, List[str]]
        (all_passed, list_of_error_messages)
    """
    errors: List[str] = []

    # 1. Required paper datasets
    required_datasets = ["gsm8k", "strategyqa", "truthfulqa", "scienceqa", "toxigen"]
    for ds_id in required_datasets:
        if ds_id not in DATASET_REGISTRY:
            errors.append(f"Missing required dataset in registry: {ds_id!r}")
            continue
        try:
            entry = get_dataset(ds_id)
            assert entry["id"] == ds_id, f"id mismatch: {entry['id']} != {ds_id}"
        except Exception as exc:
            errors.append(f"get_dataset({ds_id!r}) failed: {exc}")

        # alias resolution
        for alias in DATASET_REGISTRY[ds_id].get("aliases", []):
            try:
                resolved = get_dataset(alias)
                assert resolved["id"] == ds_id
            except Exception as exc:
                errors.append(f"Alias {alias!r} -> {ds_id!r} resolution failed: {exc}")

    # 2. Required paper environments
    required_envs = ["gpt-3.5-turbo", "mixtral-8x7b", "azure-openai", "davinci-002"]
    for env_id in required_envs:
        if env_id not in ENVIRONMENT_REGISTRY:
            errors.append(f"Missing required environment in registry: {env_id!r}")
            continue
        try:
            entry = get_environment(env_id)
            assert entry["id"] == env_id
        except Exception as exc:
            errors.append(f"get_environment({env_id!r}) failed: {exc}")

    # 3. make_environment factory — including paper-named aliases
    alias_checks = [
        ("gpt-3.5-turbo", "gpt-3.5-turbo"),
        ("Mixtral-8x7B-v0", "mixtral-8x7b"),
        ("Mixtral-8x7B", "mixtral-8x7b"),
        ("Azure OpenAI endpoint", "azure-openai"),
    ]
    for alias, expected_id in alias_checks:
        try:
            env_obj = make_environment(alias)
            assert env_obj.env_id == expected_id, (
                f"make_environment({alias!r}) returned id={env_obj.env_id!r}, "
                f"expected {expected_id!r}"
            )
        except Exception as exc:
            errors.append(f"make_environment({alias!r}) failed: {exc}")

    # 4. make_dataset factory — including paper-named aliases
    ds_alias_checks = [
        ("gsm8k", "gsm8k"),
        ("grade_school_math", "gsm8k"),
        ("strategyqa", "strategyqa"),
        ("truthfulqa", "truthfulqa"),
        ("scienceqa", "scienceqa"),
        ("toxigen", "toxigen"),
        ("ToxiGen", "toxigen"),
    ]
    for alias, expected_id in ds_alias_checks:
        try:
            ds_obj = make_dataset(alias)
            assert ds_obj.dataset_id == expected_id, (
                f"make_dataset({alias!r}) returned id={ds_obj.dataset_id!r}, "
                f"expected {expected_id!r}"
            )
        except Exception as exc:
            errors.append(f"make_dataset({alias!r}) failed: {exc}")

    # 5. DatasetConfig / EnvironmentConfig constructors
    for ds_id in required_datasets:
        try:
            cfg = DatasetConfig.from_registry(ds_id)
            assert cfg.id == ds_id
        except Exception as exc:
            errors.append(f"DatasetConfig.from_registry({ds_id!r}) failed: {exc}")

    for env_id in required_envs:
        try:
            cfg = EnvironmentConfig.from_registry(env_id)
            assert cfg.id == env_id
        except Exception as exc:
            errors.append(f"EnvironmentConfig.from_registry({env_id!r}) failed: {exc}")

    # 6. Write all artifacts
    try:
        artifact_paths = write_all_artifacts(output_dir)
        for artifact_name, artifact_path in artifact_paths.items():
            if not Path(artifact_path).exists():
                errors.append(
                    f"Artifact not created: {artifact_name!r} -> {artifact_path!r}"
                )
    except Exception as exc:
        errors.append(f"write_all_artifacts failed: {exc}")

    return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    out_dir = os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR", "results")
    passed, errs = _run_smoke(output_dir=out_dir)

    if errs:
        for err in errs:
            logger.error("SMOKE ERROR: %s", err)
        sys.exit(1)

    logger.info("environments.py smoke checks: all passed.")
    sys.exit(0)