# src/dataset_registry.py
# BBox-Adapter: Dataset and Method Registry
#
# Paper: BBox-Adapter: Lightweight Adapting for Black-Box Large Language Models
#
# reference_grounding: paperbench_ref_006 readme.md
# reference_grounding: paperbench_ref_006 research/readme_exp.md
# reference_grounding: paperbench_ref_006 MMLU/data/README.txt
# reference_grounding: paperbench_ref_005 notebooks/generate_text.ipynb
#
# This module is the authoritative Python-level dataset and method registry.
# It is import-light: all heavy optional dependencies (datasets, torch, openai,
# transformers) are guarded behind lazy imports so that static/smoke import
# succeeds in a minimal environment.
#
# Surfaces exposed:
#   DATASET_REGISTRY     - dict[id -> DatasetDescriptor]
#   METHOD_REGISTRY      - dict[id -> MethodDescriptor]
#   BASELINE_REGISTRY    - dict[id -> MethodDescriptor]
#   get_dataset(id)      - return DatasetDescriptor or raise KeyError
#   get_method(id)       - return MethodDescriptor or raise KeyError
#   make_dataset_loader(id, split, **kwargs) -> iterable
#   make_method(config)  -> method instance implementing train/predict
#   list_datasets()      - list[str]
#   list_methods()       - list[str]
#   save_registry_artifacts(output_dir) - write results/method_registry.json etc.

from __future__ import annotations

import json
import os
import importlib
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DatasetDescriptor:
    """Lightweight descriptor for a benchmark dataset.

    No heavy imports are triggered at construction time.
    Availability of the backing HuggingFace dataset is checked lazily.
    """
    id: str
    aliases: List[str]
    hf_path: str
    hf_name: Optional[str]
    split_train: str
    split_test: str
    num_train: int
    num_test: int
    answer_type: str  # numeric | binary | multiple_choice | open_ended | binary_toxicity
    answer_extractor: str  # symbolic name for the extractor function
    loader_hook: str  # dotted path to loader function
    metric: str  # primary metric id
    metrics: List[str]
    description: str
    feedback_modes: List[str]
    artifacts: List[str]
    extra: Dict[str, Any] = field(default_factory=dict)

    def is_available(self) -> bool:
        """Check whether the HuggingFace `datasets` package is available."""
        spec = importlib.util.find_spec("datasets")
        return spec is not None

    def load(self, split: str = "test", max_samples: Optional[int] = None, **kwargs):
        """Lazily load the dataset.  Raises ImportError if `datasets` is absent."""
        if not self.is_available():
            raise ImportError(
                f"Dataset '{self.id}' requires the `datasets` package. "
                "Install it with: pip install datasets"
            )
        import datasets as hf_datasets  # lazy import
        ds_kwargs: Dict[str, Any] = {}
        if self.hf_name:
            ds_kwargs["name"] = self.hf_name
        ds = hf_datasets.load_dataset(self.hf_path, **ds_kwargs)
        split_key = self.split_test if split == "test" else self.split_train
        data = ds[split_key]
        if max_samples is not None:
            data = data.select(range(min(max_samples, len(data))))
        return data

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MethodDescriptor:
    """Descriptor for a method or baseline.

    Binds the method to its datasets, metrics, and artifact paths so that the
    evidence matrix from the paper is machine-readable.
    """
    id: str
    aliases: List[str]
    kind: str  # ours | baseline | ablation
    description: str
    model_family: str
    adapter_size: Optional[str]
    datasets: List[str]
    metrics: List[str]
    artifacts: List[str]
    paper_tables: List[str]
    requires: List[str]  # runtime requirements (openai, torch, transformers, …)
    factory_hook: str  # dotted path to factory function
    config_defaults: Dict[str, Any] = field(default_factory=dict)

    def is_available(self) -> bool:
        for req in self.requires:
            if importlib.util.find_spec(req) is None:
                return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Dataset Registry
# ---------------------------------------------------------------------------
# reference_grounding: paperbench_ref_006 MMLU/data/README.txt
# MMLU protocol note: dev set is for few-shot priming; test set is for evaluation;
# auxiliary training data can be used for fine-tuning.  The same protocol structure
# (train=few-shot/fine-tune, test=evaluation) is applied here for all five benchmarks.

DATASET_REGISTRY: Dict[str, DatasetDescriptor] = {

    "gsm8k": DatasetDescriptor(
        id="gsm8k",
        aliases=["grade_school_math", "gsm", "math_reasoning", "openai/gsm8k"],
        hf_path="openai/gsm8k",
        hf_name="main",
        split_train="train",
        split_test="test",
        num_train=7473,
        num_test=1319,
        answer_type="numeric",
        answer_extractor="regex_boxed_or_last_number",
        loader_hook="src.data.gsm8k.load_gsm8k",
        metric="exact_match_numeric",
        metrics=["exact_match_numeric", "accuracy"],
        description=(
            "Grade-school math reasoning benchmark (8.5K problems). "
            "BBox-Adapter uses ground-truth numeric answers as reward signal. "
            "Corresponds to Experiments in Table 2, Table 3, Table 4 of the paper."
        ),
        feedback_modes=["ground_truth"],
        artifacts=[
            "results/tables/table_2.csv",
            "results/tables/table_4.csv",
            "results/online_adaptation_log.json",
        ],
        extra={"cot_prompt": True, "few_shot_k": 8},
    ),

    "strategyqa": DatasetDescriptor(
        id="strategyqa",
        aliases=["strategy_qa", "strategy-qa", "bigbench_strategyqa"],
        hf_path="tasksource/bigbench",
        hf_name="strategyqa",
        split_train="train",
        split_test="validation",
        num_train=2059,
        num_test=229,
        answer_type="binary",
        answer_extractor="yes_no_from_text",
        loader_hook="src.data.strategyqa.load_strategyqa",
        metric="accuracy",
        metrics=["accuracy", "exact_match_binary"],
        description=(
            "StrategyQA yes/no commonsense reasoning benchmark. "
            "Requires implicit multi-step reasoning. "
            "BBox-Adapter uses AI feedback (GPT-4 judge) as reward signal. "
            "Corresponds to Experiments in Table 2, Table 3, Table 4 of the paper."
        ),
        feedback_modes=["ai_feedback", "ground_truth"],
        artifacts=[
            "results/tables/table_2.csv",
            "results/tables/table_4.csv",
            "results/online_adaptation_log.json",
        ],
        extra={"cot_prompt": True, "few_shot_k": 6},
    ),

    "truthfulqa": DatasetDescriptor(
        id="truthfulqa",
        aliases=["truthful_qa", "truthful-qa", "mc_truthfulqa"],
        hf_path="truthful_qa",
        hf_name="multiple_choice",
        split_train="validation",  # TruthfulQA has only a validation split
        split_test="validation",
        num_train=717,
        num_test=100,
        answer_type="multiple_choice",
        answer_extractor="mc_answer_letter",
        loader_hook="src.data.truthfulqa.load_truthfulqa",
        metric="mc_accuracy",
        metrics=["mc_accuracy", "mc1_accuracy", "mc2_accuracy"],
        description=(
            "TruthfulQA multiple-choice benchmark measuring LLM truthfulness. "
            "MC1 = single true answer; MC2 = multiple true answers with normalised prob. "
            "BBox-Adapter uses combined AI + ground-truth feedback. "
            "reference_grounding: paperbench_ref_003 truthfulqa/models.py "
            "MC_calcs protocol: max/diff/scores-true/scores-false lprob columns."
        ),
        feedback_modes=["combined", "ai_feedback", "ground_truth"],
        artifacts=[
            "results/tables/table_2.csv",
            "results/online_adaptation_log.json",
        ],
        extra={"cot_prompt": True, "few_shot_k": 6, "mc_variant": "mc1"},
    ),

    "scienceqa": DatasetDescriptor(
        id="scienceqa",
        aliases=["science_qa", "science-qa", "derek-thomas/ScienceQA"],
        hf_path="derek-thomas/ScienceQA",
        hf_name=None,
        split_train="train",
        split_test="test",
        num_train=2000,
        num_test=500,
        answer_type="multiple_choice",
        answer_extractor="mc_answer_letter",
        loader_hook="src.data.scienceqa.load_scienceqa",
        metric="accuracy",
        metrics=["accuracy", "exact_match_mc"],
        description=(
            "ScienceQA multi-modal science domain multiple-choice benchmark. "
            "Text-only questions are used (no image modality). "
            "BBox-Adapter uses ground-truth labels as reward signal. "
            "Corresponds to Experiments in Table 2, Table 3 of the paper."
        ),
        feedback_modes=["ground_truth"],
        artifacts=[
            "results/tables/table_2.csv",
            "results/online_adaptation_log.json",
        ],
        extra={"cot_prompt": True, "few_shot_k": 4, "text_only": True},
    ),

    "toxigen": DatasetDescriptor(
        id="toxigen",
        aliases=["toxic_gen", "toxigen_data", "skg/toxigen-data"],
        hf_path="skg/toxigen-data",
        hf_name="annotated",
        split_train="train",
        split_test="test",
        num_train=8960,
        num_test=940,
        answer_type="binary_toxicity",
        answer_extractor="toxicity_binary_label",
        loader_hook="src.data.toxigen.load_toxigen",
        metric="accuracy",
        metrics=["accuracy", "f1", "toxicity_reduction"],
        description=(
            "ToxiGen implicit hate speech detection benchmark (binary: toxic/non-toxic). "
            "reference_grounding: paperbench_ref_005 notebooks/generate_text.ipynb "
            "BBox-Adapter uses AI feedback (GPT-4) as reward signal. "
            "Corresponds to Experiments in Table 2, Table 3 of the paper."
        ),
        feedback_modes=["ai_feedback"],
        artifacts=[
            "results/tables/table_2.csv",
            "results/online_adaptation_log.json",
        ],
        extra={
            "cot_prompt": False,
            "few_shot_k": 0,
            "label_map": {"toxic": 1, "non-toxic": 0},
        },
    ),
}

# Alias index for fast look-up by any registered alias
_DATASET_ALIAS_INDEX: Dict[str, str] = {}
for _ds_id, _ds in DATASET_REGISTRY.items():
    _DATASET_ALIAS_INDEX[_ds_id] = _ds_id
    for _alias in _ds.aliases:
        _DATASET_ALIAS_INDEX[_alias.lower()] = _ds_id


# ---------------------------------------------------------------------------
# Method / Baseline Registry
# ---------------------------------------------------------------------------
# reference_grounding: paperbench_ref_006 readme.md
# Chain-of-thought hub protocol: CoT prompting as the zero-shot baseline for
# complex reasoning evaluation; all methods use CoT prompts matching Wei et al. 2022.

METHOD_REGISTRY: Dict[str, MethodDescriptor] = {

    # ── BBox-Adapter (paper contribution) ───────────────────────────────
    "bbox_adapter": MethodDescriptor(
        id="bbox_adapter",
        aliases=["ours", "bbox-adapter", "bboxadapter"],
        kind="ours",
        description=(
            "BBox-Adapter: energy-based adapter (BERT 0.1B or 0.3B) trained online "
            "with ranking NCE loss on positive/negative samples from the black-box LLM. "
            "Sentence-level beam search at inference time."
        ),
        model_family="bert",
        adapter_size="0.1B",
        datasets=["gsm8k", "strategyqa", "truthfulqa", "scienceqa", "toxigen"],
        metrics=["accuracy", "exact_match_numeric", "mc_accuracy", "f1"],
        artifacts=[
            "results/tables/table_2.csv",
            "results/tables/table_3.csv",
            "results/tables/table_4.csv",
            "results/figures/figure_1.png",
            "results/figures/figure_2.png",
            "results/online_adaptation_log.json",
        ],
        paper_tables=["table_2", "table_3", "table_4"],
        requires=["torch", "transformers"],
        factory_hook="src.bbox_adapter.adapter.BBoxAdapter",
        config_defaults={
            "adapter_size": "0.1B",
            "beam_size": 10,
            "num_iterations": 5,
            "batch_size": 8,
            "learning_rate": 1e-5,
            "nce_temperature": 1.0,
        },
    ),

    "bbox_adapter_03b": MethodDescriptor(
        id="bbox_adapter_03b",
        aliases=["ours_03b", "bbox-adapter-0.3b"],
        kind="ablation",
        description="BBox-Adapter variant with 0.3B BERT adapter.",
        model_family="bert",
        adapter_size="0.3B",
        datasets=["gsm8k", "strategyqa", "truthfulqa", "scienceqa", "toxigen"],
        metrics=["accuracy", "exact_match_numeric", "mc_accuracy"],
        artifacts=["results/tables/table_2.csv", "results/ablation_registry.json"],
        paper_tables=["table_2"],
        requires=["torch", "transformers"],
        factory_hook="src.bbox_adapter.adapter.BBoxAdapter",
        config_defaults={"adapter_size": "0.3B", "beam_size": 10},
    ),

    # ── Baselines ─────────────────────────────────────────────────────────

    "chain_of_thought": MethodDescriptor(
        id="chain_of_thought",
        aliases=["cot", "zero_shot_cot", "chain-of-thought", "baseline_cot"],
        kind="baseline",
        description=(
            "Chain-of-Thought zero-shot baseline (Wei et al., 2022). "
            "Prompts the black-box LLM with CoT instructions and takes the first "
            "generated response as the prediction. No adapter or fine-tuning. "
            "reference_grounding: paperbench_ref_006 readme.md"
        ),
        model_family="gpt",
        adapter_size=None,
        datasets=["gsm8k", "strategyqa", "truthfulqa", "scienceqa", "toxigen"],
        metrics=["accuracy", "exact_match_numeric", "mc_accuracy", "f1"],
        artifacts=["results/tables/table_2.csv"],
        paper_tables=["table_2"],
        requires=[],
        factory_hook="src.baselines.chain_of_thought.ChainOfThoughtBaseline",
        config_defaults={"temperature": 0.0, "max_tokens": 512},
    ),

    "oracle": MethodDescriptor(
        id="oracle",
        aliases=["oracle_upper_bound", "ground_truth_oracle"],
        kind="baseline",
        description=(
            "Oracle upper bound: re-ranks LLM samples using ground-truth correctness. "
            "Provides an upper bound on BBox-Adapter performance."
        ),
        model_family="gpt",
        adapter_size=None,
        datasets=["gsm8k", "strategyqa", "scienceqa"],
        metrics=["accuracy", "exact_match_numeric"],
        artifacts=["results/tables/table_2.csv"],
        paper_tables=["table_2"],
        requires=[],
        factory_hook="src.baselines.chain_of_thought.OracleBaseline",
        config_defaults={},
    ),

    "heuristic": MethodDescriptor(
        id="heuristic",
        aliases=["heuristic_baseline", "majority_vote"],
        kind="baseline",
        description="Heuristic baseline using majority voting over LLM samples.",
        model_family="gpt",
        adapter_size=None,
        datasets=["gsm8k", "strategyqa", "scienceqa", "toxigen"],
        metrics=["accuracy"],
        artifacts=["results/tables/table_2.csv"],
        paper_tables=["table_2"],
        requires=[],
        factory_hook="src.baselines.chain_of_thought.HeuristicBaseline",
        config_defaults={"num_samples": 10},
    ),

    "roberta": MethodDescriptor(
        id="roberta",
        aliases=["roberta_ranker", "roberta_baseline"],
        kind="baseline",
        description=(
            "RoBERTa-based re-ranker trained on fixed offline data. "
            "No online adaptation loop."
        ),
        model_family="roberta",
        adapter_size="0.1B",
        datasets=["gsm8k", "strategyqa", "scienceqa", "toxigen"],
        metrics=["accuracy"],
        artifacts=["results/tables/table_2.csv"],
        paper_tables=["table_2"],
        requires=["torch", "transformers"],
        factory_hook="src.baselines.chain_of_thought.RoBERTaBaseline",
        config_defaults={"model_name": "roberta-base"},
    ),

    "sft_lora": MethodDescriptor(
        id="sft_lora",
        aliases=["lora", "fine_tuning_lora", "mixtral_lora"],
        kind="baseline",
        description=(
            "LoRA fine-tuning baseline for Mixtral-8x7B (open-source grey-box model). "
            "Uses PEFT LoRA adapters attached to the query/value projection matrices. "
            "Trained on the same training split as BBox-Adapter with cross-entropy loss. "
            "reference_grounding: paperbench_ref_006 research/readme_exp.md"
        ),
        model_family="mixtral",
        adapter_size="LoRA-r16",
        datasets=["gsm8k", "strategyqa", "scienceqa"],
        metrics=["accuracy", "exact_match_numeric"],
        artifacts=[
            "results/tables/table_4.csv",
            "results/cost_vram_report.json",
        ],
        paper_tables=["table_4"],
        requires=["torch", "transformers", "peft"],
        factory_hook="src.baselines.lora.LoRABaseline",
        config_defaults={
            "base_model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "lora_r": 16,
            "lora_alpha": 256,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "v_proj"],
            "num_train_epochs": 3,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "fp16": True,
        },
    ),

    "azure_sft": MethodDescriptor(
        id="azure_sft",
        aliases=["sft", "supervised_ft", "gpt35_sft", "azure_openai_sft"],
        kind="baseline",
        description=(
            "Supervised Fine-Tuning (SFT) via Azure OpenAI API for gpt-3.5-turbo. "
            "Uses the Azure OpenAI fine-tuning endpoint to submit a JSONL training job. "
            "Training data: (prompt, positive_response) pairs from the training split. "
            "Maps to the SFT row in Table 4 of the paper (cost and accuracy comparison). "
        ),
        model_family="gpt",
        adapter_size=None,
        datasets=["gsm8k", "strategyqa"],
        metrics=["accuracy", "exact_match_numeric", "training_cost", "inference_cost"],
        artifacts=[
            "results/tables/table_4.csv",
            "results/cost_vram_report.json",
        ],
        paper_tables=["table_4"],
        requires=["openai"],
        factory_hook="src.baselines.supervised_ft.AzureSFTBaseline",
        config_defaults={
            "model": "gpt-3.5-turbo",
            "n_epochs": 3,
            "batch_size": 8,
            "learning_rate_multiplier": 1.0,
            "api_type": "azure",
            "deployment_name": "gpt-35-turbo-ft",
        },
    ),

    "online_adaptation": MethodDescriptor(
        id="online_adaptation",
        aliases=["online_adapt", "bbox_adapter_online"],
        kind="ours",
        description="BBox-Adapter full online adaptation loop (train + infer).",
        model_family="bert",
        adapter_size="0.1B",
        datasets=["gsm8k", "strategyqa", "truthfulqa", "scienceqa", "toxigen"],
        metrics=["accuracy"],
        artifacts=["results/online_adaptation_log.json"],
        paper_tables=["table_2"],
        requires=["torch", "transformers"],
        factory_hook="src.bbox_adapter.online_adaptation.OnlineAdaptation",
        config_defaults={"num_iterations": 5},
    ),

    "single_step_inference": MethodDescriptor(
        id="single_step_inference",
        aliases=["single_step", "no_beam"],
        kind="ablation",
        description="Ablation: BBox-Adapter with single-step (no beam) inference.",
        model_family="bert",
        adapter_size="0.1B",
        datasets=["gsm8k", "strategyqa"],
        metrics=["accuracy"],
        artifacts=["results/ablation_registry.json"],
        paper_tables=[],
        requires=["torch", "transformers"],
        factory_hook="src.bbox_adapter.adapter.BBoxAdapter",
        config_defaults={"beam_size": 1},
    ),

    "full_step_inference": MethodDescriptor(
        id="full_step_inference",
        aliases=["full_beam", "beam_inference"],
        kind="ablation",
        description="Ablation: BBox-Adapter with full beam inference (default).",
        model_family="bert",
        adapter_size="0.1B",
        datasets=["gsm8k", "strategyqa"],
        metrics=["accuracy"],
        artifacts=["results/ablation_registry.json"],
        paper_tables=[],
        requires=["torch", "transformers"],
        factory_hook="src.bbox_adapter.adapter.BBoxAdapter",
        config_defaults={"beam_size": 10},
    ),

    "ground_truth_feedback": MethodDescriptor(
        id="ground_truth_feedback",
        aliases=["gt_feedback", "gt_reward"],
        kind="ablation",
        description="Ablation: BBox-Adapter trained with ground-truth feedback only.",
        model_family="bert",
        adapter_size="0.1B",
        datasets=["gsm8k", "scienceqa"],
        metrics=["accuracy"],
        artifacts=["results/ablation_registry.json"],
        paper_tables=[],
        requires=["torch", "transformers"],
        factory_hook="src.bbox_adapter.online_adaptation.OnlineAdaptation",
        config_defaults={"feedback_mode": "ground_truth"},
    ),

    "ai_feedback": MethodDescriptor(
        id="ai_feedback",
        aliases=["gpt4_feedback", "ai_reward"],
        kind="ablation",
        description="Ablation: BBox-Adapter trained with AI (GPT-4) feedback only.",
        model_family="bert",
        adapter_size="0.1B",
        datasets=["strategyqa", "toxigen"],
        metrics=["accuracy", "f1"],
        artifacts=["results/ablation_registry.json"],
        paper_tables=[],
        requires=["torch", "transformers"],
        factory_hook="src.bbox_adapter.online_adaptation.OnlineAdaptation",
        config_defaults={"feedback_mode": "ai_feedback"},
    ),

    "ranking_nce": MethodDescriptor(
        id="ranking_nce",
        aliases=["nce_loss", "ranking_nce_loss"],
        kind="ablation",
        description="Ablation: ranking NCE loss only (no online update loop).",
        model_family="bert",
        adapter_size="0.1B",
        datasets=["gsm8k", "strategyqa"],
        metrics=["accuracy"],
        artifacts=["results/ablation_registry.json"],
        paper_tables=[],
        requires=["torch", "transformers"],
        factory_hook="src.bbox_adapter.nce_loss.RankingNCELoss",
        config_defaults={},
    ),

    "mlm": MethodDescriptor(
        id="mlm",
        aliases=["masked_lm", "mlm_baseline"],
        kind="baseline",
        description="MLM-based scoring baseline using masked language model probabilities.",
        model_family="bert",
        adapter_size="0.1B",
        datasets=["gsm8k", "strategyqa"],
        metrics=["accuracy"],
        artifacts=["results/tables/table_2.csv"],
        paper_tables=["table_2"],
        requires=["torch", "transformers"],
        factory_hook="src.baselines.chain_of_thought.MLMBaseline",
        config_defaults={"model_name": "microsoft/deberta-v3-base"},
    ),

    "fine_tuning": MethodDescriptor(
        id="fine_tuning",
        aliases=["finetune", "standard_ft"],
        kind="baseline",
        description="Standard supervised fine-tuning baseline (offline, full model).",
        model_family="gpt",
        adapter_size=None,
        datasets=["gsm8k", "strategyqa"],
        metrics=["accuracy"],
        artifacts=["results/tables/table_2.csv"],
        paper_tables=["table_2"],
        requires=["torch", "transformers"],
        factory_hook="src.baselines.supervised_ft.FineTuningBaseline",
        config_defaults={},
    ),
}

# Baseline-only sub-registry
BASELINE_REGISTRY: Dict[str, MethodDescriptor] = {
    k: v for k, v in METHOD_REGISTRY.items() if v.kind == "baseline"
}

# Method alias index
_METHOD_ALIAS_INDEX: Dict[str, str] = {}
for _m_id, _m in METHOD_REGISTRY.items():
    _METHOD_ALIAS_INDEX[_m_id] = _m_id
    for _alias in _m.aliases:
        _METHOD_ALIAS_INDEX[_alias.lower()] = _m_id


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_dataset(dataset_id: str) -> DatasetDescriptor:
    """Return DatasetDescriptor for `dataset_id` (id or alias).

    Raises KeyError with a helpful message if not found.
    """
    key = _DATASET_ALIAS_INDEX.get(dataset_id.lower())
    if key is None:
        available = sorted(DATASET_REGISTRY.keys())
        raise KeyError(
            f"Unknown dataset id or alias '{dataset_id}'. "
            f"Available datasets: {available}"
        )
    return DATASET_REGISTRY[key]


def get_method(method_id: str) -> MethodDescriptor:
    """Return MethodDescriptor for `method_id` (id or alias).

    Raises KeyError with a helpful message if not found.
    """
    key = _METHOD_ALIAS_INDEX.get(method_id.lower())
    if key is None:
        available = sorted(METHOD_REGISTRY.keys())
        raise KeyError(
            f"Unknown method id or alias '{method_id}'. "
            f"Available methods: {available}"
        )
    return METHOD_REGISTRY[key]


def list_datasets() -> List[str]:
    """Return sorted list of registered dataset ids."""
    return sorted(DATASET_REGISTRY.keys())


def list_methods() -> List[str]:
    """Return sorted list of registered method ids."""
    return sorted(METHOD_REGISTRY.keys())


def make_dataset_loader(
    dataset_id: str,
    split: str = "test",
    max_samples: Optional[int] = None,
    **kwargs,
):
    """Factory: return an iterable dataset for the given id and split.

    Falls back to a lightweight availability check with a faithful error if
    the `datasets` package is absent.
    """
    descriptor = get_dataset(dataset_id)
    return descriptor.load(split=split, max_samples=max_samples, **kwargs)


def make_method(config: Dict[str, Any]):
    """Factory: instantiate a method from a config dict.

    Expected keys:
        method_id (str): id or alias from METHOD_REGISTRY
        **kwargs: passed to the method constructor

    Returns an instance implementing .train(data) and .predict(input).
    Raises ImportError if the method's required packages are absent.
    """
    method_id = config.get("method_id") or config.get("method") or config.get("id")
    if method_id is None:
        raise ValueError("make_method: config must contain 'method_id' key")

    descriptor = get_method(method_id)

    # Availability check with faithful fallback errors
