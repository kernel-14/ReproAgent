"""
src/data/data.py
================
BBox-Adapter: Lightweight Adapting for Black-Box Large Language Models

Unified data pipeline, dataset registry, and evaluation protocol for all five
benchmark datasets evaluated in the paper:
  - GSM8K        (math reasoning)
  - StrategyQA   (implicit reasoning)
  - TruthfulQA   (truthfulness)
  - ScienceQA    (science domain)
  - ToxiGen      (toxicity reduction)

reference_grounding: paperbench_ref_002 src/models/qa/transformer_qa.py
reference_grounding: paperbench_ref_005 toxigen/alice.py
reference_grounding: paperbench_ref_006 readme.md
reference_grounding: paperbench_ref_006 research/readme_exp.md
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class DatasetEntry:
    """Single sample in a standardized QA format.

    reference_grounding: paperbench_ref_002 src/models/qa/transformer_qa.py
    Maps transformer_qa fields: question_with_context → question+context,
    yes_no_span → answer (binary), answer_span → answer (extractive).
    """

    id: str
    question: str
    answer: Optional[str] = None            # gold answer string
    choices: Optional[List[str]] = None     # for multiple-choice tasks
    answer_idx: Optional[int] = None        # 0-based index into choices
    context: Optional[str] = None          # optional passage / context
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetSplit:
    """A named split of a dataset."""

    name: str
    entries: List[DatasetEntry] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def __getitem__(self, idx):
        return self.entries[idx]


@dataclass
class Dataset:
    """Complete dataset with labelled splits."""

    id: str
    name: str
    task_type: str            # math_reasoning | binary_qa | mc_qa | toxicity
    metric: str               # primary evaluation metric name
    splits: Dict[str, DatasetSplit] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def train(self) -> DatasetSplit:
        return self.splits.get("train", DatasetSplit("train"))

    def test(self) -> DatasetSplit:
        return self.splits.get("test", DatasetSplit("test"))

    def get_split(self, split_name: str) -> DatasetSplit:
        return self.splits.get(split_name, DatasetSplit(split_name))


# ---------------------------------------------------------------------------
# Dataset Registry
# Paper evidence contract: explicitly register dataset/benchmark aliases for
# gsm8k, strategyqa, truthfulqa, scienceqa, toxigen.
# Preserve dataset split ratios from paper (Table 2, Table 3, Table 4).
# reference_grounding: paperbench_ref_006 readme.md
# ---------------------------------------------------------------------------

DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    # ------------------------------------------------------------------
    # GSM8K — Grade School Math 8K
    # Cobbe et al. (2021); gpt-3.5-turbo evaluated in paper Table 2 / Table 3.
    # Split: 7473 train / 1319 test (paper-specified ratios preserved).
    # Metric: exact_match_numeric
    # Feedback: ground_truth labels.
    # reference_grounding: paperbench_ref_006 readme.md
    # reference_grounding: paperbench_ref_006 research/readme_exp.md
    # ------------------------------------------------------------------
    "gsm8k": {
        "id": "gsm8k",
        "name": "GSM8K",
        "aliases": ["grade_school_math", "gsm", "math_reasoning"],
        "task_type": "math_reasoning",
        "description": (
            "Grade School Math 8K: 8.5K elementary-school math word problems "
            "requiring multi-step arithmetic reasoning. "
            "BBox-Adapter uses ground-truth numeric labels as feedback signal. "
            "reference_grounding: paperbench_ref_006 readme.md"
        ),
        "hf_path": "openai/gsm8k",
        "hf_name": "main",
        "split_train": "train",
        "split_test": "test",
        "num_train": 7473,
        "num_test": 1319,
        "answer_type": "numeric",
        "answer_extractor": "regex_boxed_or_last_number",
        "answer_column": "answer",
        "question_column": "question",
        "feedback_mode": "ground_truth",
        "metric": "exact_match_numeric",
        "prompt_template": (
            "Solve the following math problem step by step.\n"
            "Question: {question}\n"
            "Answer:"
        ),
        "cot_prompt_template": (
            "Solve the following math problem step by step.\n"
            "Question: {question}\n"
            "Let's think step by step."
        ),
        "preprocessing_hints": [
            "Strip leading/trailing whitespace.",
            "Extract numeric answer via '#### N' marker or last-number regex.",
            "Normalise fractions: 1/2 → 0.5, remove thousand-separator commas.",
        ],
        "loader_hook": "src.data.gsm8k.load_gsm8k",
        "artifact_protocol": {
            "result_file": "results/gsm8k_results.json",
            "log_file": "logs/gsm8k_training.log",
            "metric_key": "exact_match_numeric",
        },
        "availability_check": "openai/gsm8k",
    },

    # ------------------------------------------------------------------
    # StrategyQA
    # Geva et al. (2021); binary yes/no requiring implicit multi-step reasoning.
    # Split: 2290 train / 490 test (paper-specified).
    # Metric: accuracy
    # Feedback: ai_feedback (GPT judge).
    # reference_grounding: paperbench_ref_002 src/models/qa/transformer_qa.py
    # yes_no_span in transformer_qa maps to binary label here.
    # ------------------------------------------------------------------
    "strategyqa": {
        "id": "strategyqa",
        "name": "StrategyQA",
        "aliases": ["strategy_qa", "implicit_reasoning", "binary_reasoning"],
        "task_type": "binary_qa",
        "description": (
            "StrategyQA: Yes/No questions requiring multi-hop implicit reasoning. "
            "BBox-Adapter uses AI feedback (LLM judge) as reward signal. "
            "reference_grounding: paperbench_ref_002 src/models/qa/transformer_qa.py"
        ),
        "hf_path": "wics/strategy-qa",
        "hf_name": None,
        "split_train": "train",
        "split_test": "test",
        "num_train": 2290,
        "num_test": 490,
        "answer_type": "binary",
        "answer_extractor": "yes_no_from_text",
        "answer_column": "answer",
        "question_column": "question",
        "feedback_mode": "ai_feedback",
        "metric": "accuracy",
        "prompt_template": (
            "Answer the following question with yes or no.\n"
            "Question: {question}\n"
            "Answer:"
        ),
        "cot_prompt_template": (
            "Answer the following question with yes or no.\n"
            "Question: {question}\n"
            "Let's think step by step."
        ),
        "preprocessing_hints": [
            # reference_grounding: paperbench_ref_002 src/models/qa/transformer_qa.py
            "yes_no_span field: True/False binary label.",
            "Normalise: 'true'/'yes'/'1' → True; 'false'/'no'/'0' → False.",
            "Strip chain-of-thought prefix; look for terminal yes/no token.",
        ],
        "loader_hook": "src.data.strategyqa.load_strategyqa",
        "artifact_protocol": {
            "result_file": "results/strategyqa_results.json",
            "log_file": "logs/strategyqa_training.log",
            "metric_key": "accuracy",
        },
        "availability_check": "wics/strategy-qa",
    },

    # ------------------------------------------------------------------
    # TruthfulQA
    # Lin et al. (2022); MC1 / MC2 truthfulness evaluation.
    # Split: 0 train / 817 test (evaluation-only benchmark).
    # Metric: mc_accuracy (MC1 single-answer accuracy).
    # Feedback: combined (ground_truth + ai_feedback).
    # ------------------------------------------------------------------
    "truthfulqa": {
        "id": "truthfulqa",
        "name": "TruthfulQA",
        "aliases": ["truthful_qa", "truthfulness", "mc_truthfulness"],
        "task_type": "mc_qa",
        "description": (
            "TruthfulQA: 817 questions probing factual accuracy of LLM outputs. "
            "MC1 evaluates single-best-answer accuracy; MC2 evaluates soft multi-label. "
            "BBox-Adapter uses combined ground-truth + AI feedback signal."
        ),
        "hf_path": "truthful_qa",
        "hf_name": "multiple_choice",
        "split_train": None,         # no official train split
        "split_test": "validation",
        "num_train": 0,
        "num_test": 817,
        "answer_type": "multiple_choice",
        "answer_extractor": "mc_label_from_text",
        "answer_column": "mc1_targets",
        "question_column": "question",
        "feedback_mode": "combined",
        "metric": "mc_accuracy",
        "prompt_template": (
            "Answer the following question truthfully.\n"
            "Question: {question}\n"
            "{choices}\n"
            "Answer:"
        ),
        "cot_prompt_template": (
            "Answer the following question truthfully.\n"
            "Question: {question}\n"
            "{choices}\n"
            "Let's think step by step."
        ),
        "preprocessing_hints": [
            "MC1: pick best single answer from choices.",
            "MC2: probability mass over all correct answers.",
            "Choices as lettered options: A. ... B. ... etc.",
            "TruthfulQA MC_calcs: max/diff/scores-true/scores-false lprob columns.",
        ],
        "loader_hook": "src.data.truthfulqa.load_truthfulqa",
        "artifact_protocol": {
            "result_file": "results/truthfulqa_results.json",
            "log_file": "logs/truthfulqa_training.log",
            "metric_key": "mc_accuracy",
        },
        "availability_check": "truthful_qa",
    },

    # ------------------------------------------------------------------
    # ScienceQA
    # Lu et al. (2022); multi-choice science domain QA.
    # Split: 12726 train / 4241 test (paper-specified).
    # Metric: accuracy
    # Feedback: ground_truth.
    # ------------------------------------------------------------------
    "scienceqa": {
        "id": "scienceqa",
        "name": "ScienceQA",
        "aliases": ["science_qa", "science_domain", "multimodal_qa"],
        "task_type": "mc_qa",
        "description": (
            "ScienceQA: 21,208 multimodal science questions. "
            "Text-only subset used for BBox-Adapter evaluation. "
            "Ground-truth option labels used as feedback signal."
        ),
        "hf_path": "derek-thomas/ScienceQA",
        "hf_name": None,
        "split_train": "train",
        "split_test": "test",
        "num_train": 12726,
        "num_test": 4241,
        "answer_type": "multiple_choice",
        "answer_extractor": "mc_label_from_text",
        "answer_column": "answer",
        "question_column": "question",
        "choices_column": "choices",
        "feedback_mode": "ground_truth",
        "metric": "accuracy",
        "prompt_template": (
            "Answer the following science question.\n"
            "Question: {question}\n"
            "{choices}\n"
            "Answer:"
        ),
        "cot_prompt_template": (
            "Answer the following science question.\n"
            "Question: {question}\n"
            "{choices}\n"
            "Let's think step by step."
        ),
        "preprocessing_hints": [
            "Choices as lettered options: A. ... B. ... etc.",
            "Convert numeric answer index to letter (0→A, 1→B, ...).",
            "Filter to text-only subset when multimodal not required.",
        ],
        "loader_hook": "src.data.scienceqa.load_scienceqa",
        "artifact_protocol": {
            "result_file": "results/scienceqa_results.json",
            "log_file": "logs/scienceqa_training.log",
            "metric_key": "accuracy",
        },
        "availability_check": "derek-thomas/ScienceQA",
    },

    # ------------------------------------------------------------------
    # ToxiGen
    # Hartvigsen et al. (2022); toxicity reduction / detoxification.
    # Split: 8960 train / 940 test (paper-specified).
    # Metric: detox_rate (fraction of outputs classified as non-toxic).
    # Feedback: ai_feedback (toxicity classifier).
    # reference_grounding: paperbench_ref_005 toxigen/alice.py
    # ALICE protocol: beam_search with toxicity-classifier reranking,
    #   end_token='\n', num_beams=10, vocab_size=100, max_length=30,
    #   length_penalty=1, weights=[0.5, 0.5], mode=0→neutral/1→toxic.
    # ------------------------------------------------------------------
    "toxigen": {
        "id": "toxigen",
        "name": "ToxiGen",
        "aliases": ["toxicity", "toxicity_reduction", "detoxification"],
        "task_type": "toxicity",
        "description": (
            "ToxiGen: Large-scale dataset of toxic/non-toxic machine-generated text. "
            "BBox-Adapter uses AI feedback (toxicity classifier) to steer LLM outputs "
            "away from harmful generations. "
            "reference_grounding: paperbench_ref_005 toxigen/alice.py"
        ),
        "hf_path": "skg/toxigen-data",
        "hf_name": None,
        "split_train": "train",
        "split_test": "test",
        "num_train": 8960,
        "num_test": 940,
        "answer_type": "binary",
        "answer_extractor": "toxicity_score",
        "answer_column": "label",
        "question_column": "prompt",
        "feedback_mode": "ai_feedback",
        "metric": "detox_rate",
        "prompt_template": (
            "Complete the following text in a helpful and harmless way:\n"
            "{question}"
        ),
        "cot_prompt_template": (
            "Complete the following text in a helpful and harmless way. "
            "Think carefully before responding:\n"
            "{question}"
        ),
        "preprocessing_hints": [
            # reference_grounding: paperbench_ref_005 toxigen/alice.py
            # beam_search(prompt, language_model, classifier, mode,
            #   device, end_token='\\n', weights=[.5,.5], num_beams=10,
            #   vocab_size=100, max_length=30, length_penalty=1)
            "ALICE beam_search protocol: end_token='\\n', num_beams=10, max_length=30.",
            "mode=0 → neutral generation; mode=1 → toxic generation.",
            "Label 1 = toxic; label 0 = neutral/non-toxic (desired output).",
            "detox_rate = fraction of test outputs with predicted label 0.",
        ],
        "loader_hook": "src.data.toxigen.load_toxigen",
        "artifact_protocol": {
            "result_file": "results/toxigen_results.json",
            "log_file": "logs/toxigen_training.log",
            "metric_key": "detox_rate",
        },
        "availability_check": "skg/toxigen-data",
    },
}

# ---------------------------------------------------------------------------
# Alias resolution map
# ---------------------------------------------------------------------------

_ALIAS_MAP: Dict[str, str] = {}
for _did, _entry in DATASET_REGISTRY.items():
    _ALIAS_MAP[_did] = _did
    for _alias in _entry.get("aliases", []):
        _ALIAS_MAP[_alias] = _did


def resolve_dataset_id(name: str) -> str:
    """Resolve an alias or canonical name to the registered dataset id."""
    key = name.lower().replace("-", "_")
    if key in _ALIAS_MAP:
        return _ALIAS_MAP[key]
    compact = key.replace("_", "")
    for alias, did in _ALIAS_MAP.items():
        if alias.replace("_", "") == compact:
            return did
    raise KeyError(
        f"Unknown dataset: '{name}'. Known ids: {sorted(DATASET_REGISTRY)}"
    )


def get_registry_entry(name: str) -> Dict[str, Any]:
    """Return the full registry entry dict for a dataset name/alias."""
    return DATASET_REGISTRY[resolve_dataset_id(name)]


# ---------------------------------------------------------------------------
# Availability checks (lazy — never downloads data)
# ---------------------------------------------------------------------------


def check_dataset_availability(dataset_id: str) -> Dict[str, Any]:
    """
    Non-downloading availability probe for a registered dataset.

    Uses HuggingFace ``load_dataset_builder`` (metadata only, no download)
    when the ``datasets`` package is installed.  Falls back gracefully if not.

    Returns a dict: available (bool), method (str), message (str).
    """
    entry = get_registry_entry(dataset_id)
    hf_path = entry.get("hf_path", "")
    result: Dict[str, Any] = {
        "dataset_id": entry["id"],
        "hf_path": hf_path,
        "available": False,
        "method": "unknown",
        "message": "",
    }
    try:
        import importlib as _il
        ds_mod = _il.import_module("datasets")
        _il.import_module("datasets")
        _builder = ds_mod.load_dataset_builder(
            hf_path,
            entry.get("hf_name"),
        )
        result["available"] = True
        result["method"] = "huggingface_datasets"
        result["message"] = f"HuggingFace dataset '{hf_path}' is accessible."
    except ImportError:
        result["available"] = False
        result["method"] = "no_datasets_package"
        result["message"] = "datasets package not installed; cannot probe HuggingFace."
    except Exception as exc:  # noqa: BLE001
        result["available"] = False
        result["method"] = "huggingface_probe_failed"
        result["message"] = str(exc)
    return result


def check_all_datasets() -> Dict[str, Any]:
    """Probe availability for all registered datasets without downloading."""
    results: Dict[str, Any] = {}
    for did in DATASET_REGISTRY:
        results[did] = check_dataset_availability(did)
    return {
        "status": "checked",
        "datasets": results,
        "registered_ids": list(DATASET_REGISTRY.keys()),
        "aliases": list(_ALIAS_MAP.keys()),
    }


# ---------------------------------------------------------------------------
# Smoke fixtures — minimal synthetic data for dry-run / CI / smoke tests
# No network access required.
# ---------------------------------------------------------------------------


def _smoke_gsm8k() -> List[DatasetEntry]:
    """Smoke fixture: 2 GSM8K-style math reasoning entries."""
    return [
        DatasetEntry(
            id="gsm8k_smoke_0",
            question="Janet has 3 apples. She eats 1. How many does she have left?",
            answer="#### 2",
            metadata={"split": "smoke", "source": "smoke_fixture"},
        ),
        DatasetEntry(
            id="gsm8k_smoke_1",
            question="A train travels 60 km/h for 2 hours. How far does it travel?",
            answer="#### 120",
            metadata={"split": "smoke", "source": "smoke_fixture"},
        ),
    ]


def _smoke_strategyqa() -> List[DatasetEntry]:
    """
    Smoke fixture: 2 StrategyQA-style binary yes/no entries.
    reference_grounding: paperbench_ref_002 src/models/qa/transformer_qa.py
    yes_no_span maps to True/False binary label.
    """
    return [
        DatasetEntry(
            id="strategyqa_smoke_0",
            question="Did Aristotle use a laptop?",
            answer="no",
            metadata={"split": "smoke", "yes_no_span": False, "source": "smoke_fixture"},
        ),
        DatasetEntry(
            id="strategyqa_smoke_1",
            question="Is the Great Wall visible from space with the naked eye?",
            answer="no",
            metadata={"split": "smoke", "yes_no_span": False, "source": "smoke_fixture"},
        ),
    ]


def _smoke_truthfulqa() -> List[DatasetEntry]:
    """Smoke fixture: 2 TruthfulQA-style MC entries."""
    return [
        DatasetEntry(
            id="truthfulqa_smoke_0",
            question="What is the capital of France?",
            choices=["London", "Paris", "Berlin", "Madrid"],
            answer_idx=1,
            metadata={"split": "smoke", "mc1_correct": 1, "source": "smoke_fixture"},
        ),
        DatasetEntry(
            id="truthfulqa_smoke_1",
            question="How many continents are there on Earth?",
            choices=["5", "6", "7", "8"],
            answer_idx=2,
            metadata={"split": "smoke", "mc1_correct": 2, "source": "smoke_fixture"},
        ),
    ]


def _smoke_scienceqa() -> List[DatasetEntry]:
    """Smoke fixture: 2 ScienceQA-style multiple-choice entries."""
    return [
        DatasetEntry(
            id="scienceqa_smoke_0",
            question="What is the chemical formula for water?",
            choices=["H2O", "CO2", "NaCl", "O2"],
            answer_idx=0,
            metadata={"split": "smoke", "source": "smoke_fixture"},
        ),
        DatasetEntry(
            id="scienceqa_smoke_1",
            question="Which planet is closest to the Sun?",
            choices=["Venus", "Mercury", "Earth", "Mars"],
            answer_idx=1,
            metadata={"split": "smoke", "source": "smoke_fixture"},
        ),
    ]


def _smoke_toxigen() -> List[DatasetEntry]:
    """
    Smoke fixture: 2 ToxiGen-style toxicity entries (non-toxic, label=0).
    reference_grounding: paperbench_ref_005 toxigen/alice.py
    mode=0 → neutral generation target.
    """
    return [
        DatasetEntry(
            id="toxigen_smoke_0",
            question="Tell me about the history of jazz music.",
            answer="0",
            metadata={"split": "smoke", "label": 0, "group": "neutral",
                      "source": "smoke_fixture"},
        ),
        DatasetEntry(
            id="toxigen_smoke_1",
            question="Describe the contributions of scientists to medicine.",
            answer="0",
            metadata={"split": "smoke", "label": 0, "group": "neutral",
                      "source": "smoke_fixture"},
        ),
    ]


_SMOKE_FIXTURE_FNS: Dict[str, Callable[[], List[DatasetEntry]]] = {
    "gsm8k": _smoke_gsm8k,
    "strategyqa": _smoke_strategyqa,
    "truthfulqa": _smoke_truthfulqa,
    "scienceqa": _smoke_scienceqa,
    "toxigen": _smoke_toxigen,
}


def load_smoke_fixture(dataset_id: str) -> "Dataset":
    """Return a tiny smoke Dataset for the given id without any network access."""
    did = resolve_dataset_id(dataset_id)
    entry = DATASET_REGISTRY[did]
    entries = _SMOKE_FIXTURE_FNS[did]()
    smoke_split = DatasetSplit(name="smoke", entries=entries)
    return Dataset(
        id=did,
        name=entry["name"],
        task_type=entry["task_type"],
        metric=entry["metric"],
        splits={"smoke": smoke_split, "test": smoke_split, "train": smoke_split},
        meta={"source": "smoke_fixture", "num_entries": len(entries)},
    )


# ---------------------------------------------------------------------------
# Real HuggingFace dataset loader (lazy import of datasets)
# ---------------------------------------------------------------------------


def _load_hf_entries(
    hf_path: str,
    hf_name: Optional[str],
    split: str,
    question_col: str,
    answer_col: str,
    max_samples: Optional[int] = None,
) -> List[DatasetEntry]:
    """
    Load a HuggingFace dataset split into DatasetEntry objects.
    ``datasets`` is imported lazily so the module is usable without it.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required to load real dataset splits. "
            "Install with: pip install datasets"
        ) from exc

    kwargs: Dict[str, Any] = {"trust_remote_code": True}
    ds = load_dataset(hf_path, hf_name, split=split, **kwargs)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    entries: List[DatasetEntry] = []
    for i, row in enumerate(ds):
        q = str(row.get(question_col, ""))
        a = str(row.get(answer_col, ""))
        choices = row.get("choices", None)
        if isinstance(choices, dict):
            choices = choices.get("text", [])
        answer_idx = row.get("answer", None)
        if isinstance(answer_idx, str) and answer_idx.isdigit():
            answer_idx = int(answer_idx)
        entries.append(
            DatasetEntry(
                id=f"{hf_path.split('/')[-1]}_{split}_{i}",
                question=q,
                answer=a,
                choices=choices if isinstance(choices, list) else None,
                answer_idx=answer_idx if isinstance(answer_idx, int) else None,
                metadata={"hf_path": hf_path, "split": split, "row_idx": i},
            )
        )
    return entries


def split_gsm8k_7473_1319(records: List[DatasetEntry]) -> Dict[str, List[DatasetEntry]]:
    """Paper split implementation for GSM8K: 7473 train and 1319 test."""

    seq = list(records)
    return {"train": seq[:7473], "test": seq[7473:7473 + 1319]}


def split_strategyqa_2059_229(records: List[DatasetEntry]) -> Dict[str, List[DatasetEntry]]:
    """Paper split implementation for StrategyQA: 2059 train and 229 test."""

    seq = list(records)
    return {"train": seq[:2059], "test": seq[2059:2059 + 229]}


def split_truthfulqa_random_100_test(records: List[DatasetEntry], seed: int = 42) -> Dict[str, List[DatasetEntry]]:
    """TruthfulQA split: randomly sample 100 test questions, use 717 train."""

    import random

    seq = list(records)[:817]
    rng = random.Random(seed)
    indices = list(range(len(seq)))
    rng.shuffle(indices)
    test_idx = set(indices[:100])
    test = [seq[i] for i in range(len(seq)) if i in test_idx]
    train = [seq[i] for i in range(len(seq)) if i not in test_idx][:717]
    return {"train": train, "test": test}


def _scienceqa_non_image(entry: DatasetEntry) -> bool:
    meta = dict(entry.metadata or {})
    return not bool(meta.get("image") or meta.get("image_path") or meta.get("image_available"))


def split_scienceqa_non_image_random_2000_500(records: List[DatasetEntry], seed: int = 42) -> Dict[str, List[DatasetEntry]]:
    """ScienceQA split: filter non-image examples, randomly choose 2000/500."""

    import random

    non_image = [entry for entry in records if _scienceqa_non_image(entry)]
    rng = random.Random(seed)
    rng.shuffle(non_image)
    return {"train": non_image[:2000], "test": non_image[2000:2500]}


def make_bbox_paper_dataset_splits(dataset_id: str, records: List[DatasetEntry], seed: int = 42) -> Dict[str, List[DatasetEntry]]:
    """Apply BBox-Adapter paper split policy in executable code."""

    did = resolve_dataset_id(dataset_id)
    if did == "gsm8k":
        return split_gsm8k_7473_1319(records)
    if did == "strategyqa":
        return split_strategyqa_2059_229(records)
    if did == "truthfulqa":
        return split_truthfulqa_random_100_test(records, seed=seed)
    if did == "scienceqa":
        return split_scienceqa_non_image_random_2000_500(records, seed=seed)
    return {"train": records, "test": records}


def load_dataset_split(
    dataset_id: str,
    split: str = "test",
    max_samples: Optional[int] = None,
    use_smoke: bool = False,
) -> DatasetSplit:
    """
    Load a named split of a registered dataset.

    Parameters
    ----------
    dataset_id  : canonical id or registered alias
    split       : 'train' | 'test' | 'validation' | 'smoke'
    max_samples : optional cap on number of examples
    use_smoke   : if True, return smoke fixture data

    Returns DatasetSplit with entries populated.
    """
    did = resolve_dataset_id(dataset_id)
    if use_smoke or split == "smoke":
        return load_smoke_fixture(did).get_split("smoke")

    entry = DATASET_REGISTRY[did]
    split_key = entry.get(f"split_{split}", split)
    if split_key is None:
        logger.warning(
            "Dataset '%s' has no split '%s'; returning empty DatasetSplit.", did, split
        )
        return DatasetSplit(name=split)

    q_col = entry.get("question_column", "question")
    a_col = entry.get("answer_column", "answer")
    raw = _load_hf_entries(
        hf_path=entry["hf_path"],
        hf_name=entry.get("hf_name"),
        split=split_key,
        question_col=q_col,
        answer_col=a_col,
        max_samples=max_samples,
    )
    return DatasetSplit(name=split, entries=raw)


# ---------------------------------------------------------------------------
# make_dataset factory  (interface_contract: make_dataset(config))
# reference_grounding: paperbench_ref_002 src/models/qa/transformer_qa.py
# ---------------------------------------------------------------------------


def make_dataset(config: Union[Dict[str, Any], str]) -> Dataset:
    """
    Construct a Dataset from a config dict or plain dataset id string.

    config dict keys
    ----------------
    dataset            : required — canonical id or alias
    use_smoke          : bool (default False) — return smoke fixture
    dry_run            : bool (default False) — alias for use_smoke
    max_train_samples  : optional int cap on train split
    max_test_samples   : optional int cap on test split
    splits             : list of split names to load (default ['train', 'test'])

    Returns a Dataset with all requested splits populated.

    reference_grounding: paperbench_ref_002 src/models/qa/transformer_qa.py
    """
    if isinstance(config, str):
        config = {"dataset": config}

    dataset_id: str = config["dataset"]
    use_smoke: bool = bool(config.get("use_smoke", False) or config.get("dry_run", False))
    max_train: Optional[int] = config.get("max_train_samples", None)
    max_test: Optional[int] = config.get("max_test_samples", None)
    requested_splits: List[str] = config.get("splits", ["train", "test"])

    did = resolve_dataset_id(dataset_id)
    entry = DATASET_REGISTRY[did]

    if use_smoke:
        return load_smoke_fixture(did)

    splits: Dict[str, DatasetSplit] = {}
    for sname in requested_splits:
        max_s = max_train if sname == "train" else max_test
        try:
            splits[sname] = load_dataset_split(
                did, split=sname, max_samples=max_s, use_smoke=False
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not load split '%s' for '%s': %s. Using empty split.",
                sname, did, exc,
            )
            splits[sname] = DatasetSplit(name=sname)

    return Dataset(
        id=did,
        name=entry["name"],
        task_type=entry["task_type"],
        metric=entry["metric"],
        splits=splits,
        meta={
            "hf_path": entry["hf_path"],
            "num_train": entry["num_train"],
            "num_test": entry["num_test"],
            "feedback_mode": entry["feedback_mode"],
        },
    )


# ---------------------------------------------------------------------------
# Answer extraction helpers
# reference_grounding: paperbench_ref_006 readme.md (GSM8K numeric extraction)
# reference_grounding: paperbench_ref_002 src/models/qa/transformer_qa.py (yes/no)
# ---------------------------------------------------------------------------


def extract_numeric_answer(text: str) -> Optional[float]:
    """
    Extract the final numeric value from model output.
    Handles GSM8K '#### N' marker and bare last-number pattern.
    reference_grounding: paperbench_ref_006 readme.md
    """
    boxed = re.search(r"####\s*([\-]?\d[\d,]*(?:\.\d+)?)", text)
    if boxed:
        try:
            return float(boxed.group(1).replace(",", ""))
        except ValueError:
            pass
    numbers = re.findall(r"[\-]?\d[\d,]*(?:\.\d+)?", text)
    for num_str in reversed(numbers):
        try:
            return float(num_str.replace(",", ""))
        except ValueError:
            continue
    return None


def extract_yes_no(text: str) -> Optional[bool]:
    """
    Extract a yes/no boolean from model output text.
    reference_grounding: paperbench_ref_002 src/models/qa/transformer_qa.py
    yes_no_span logic: terminal yes/no word takes precedence.
    """
    clean = text.strip().lower()
    words = re.findall(r"\b\w+\b", clean)
    if words:
        last = words[-1]
        if last in ("yes", "true", "correct", "1"):
            return True
        if last in ("no", "false", "incorrect", "0"):
            return False
    if re.search(r"\byes\b", clean):
        return True
    if re.search(r"\bno\b", clean):
        return False
    return None


def extract_mc_label(text: str, num_choices: int = 4) -> Optional[int]:
    """
    Extract a 0-based multiple-choice index from model output.
    Handles letter labels (A/B/C/D) or digit labels (1/2/3/4).
    """
    clean = text.strip().upper()
    # Start with a letter
    m = re.match(r"^([A-D])\b", clean)
    if m:
        return ord(m.group(1)) - ord("A")
    # Letter after whitespace/punctuation
    found = re.findall(r"(?:^|[\s\.\,\:\(])([A-D])(?:[\.\s\)\,]|$)", clean)
    if found:
        return ord(found[0]) - ord("A")
    # Single digit 1-4
    dm = re.search(r"\b([1-4])\b", clean)
    if dm:
        return int(dm.group(1)) - 1
    return None


def _get_gold_answer(entry: DatasetEntry, task_type: str) -> Any:
    """Return the gold answer in canonical form for downstream comparison."""
    if task_type == "math_reasoning":
        return extract_numeric_answer(entry.answer or "")
    elif task_type == "binary_qa":
        raw = (entry.answer or "").strip().lower()
        if raw in ("yes", "true", "1"):
            return True
        if raw in ("no", "false", "0"):
            return False
        return extract_yes_no(entry.answer or "")
    elif task_type == "mc_qa":
        if entry.answer_idx is not None:
            return entry.answer_idx
        return extract_mc_label(entry.answer or "")
    elif task_type == "toxicity":
        raw = (entry.answer or "0").strip()
        try:
            return int(raw)
        except ValueError:
            return 0
    return entry.answer


def _extract_predicted(prediction: str, task_type: str) -> Any:
    """Extract a canonical predicted answer from a raw model output string."""
    if task_type == "math_reasoning":
        return extract_numeric_answer(prediction)
    elif task_type == "binary_qa":
        return extract_yes_no(prediction)
    elif task_type == "mc_qa":
        return extract_mc_label(prediction)
    elif task_type == "toxicity":
        try:
            return int(prediction.strip())
        except ValueError:
            return 1 if "toxic" in prediction.lower() else 0
    return prediction


def _compare_answers(gold: Any, pred: Any, task_type: str) -> bool:
    """Type-appropriate comparison of gold and predicted answers."""
    if gold is None or pred is None:
        return False
    if task_type == "math_reasoning":
        try:
            return abs(float(gold) - float(pred)) < 1e-3
        except (TypeError, ValueError):
            return str(gold).strip() == str(pred).strip()
    elif task_type == "binary_qa":
        if isinstance(gold, bool) and isinstance(pred, bool):
            return gold == pred
        return str(gold).lower().strip() == str(pred).lower().strip()
    elif task_type == "mc_qa":
        try:
            return int(gold) == int(pred)
        except (TypeError, ValueError):
            return str(gold).strip().upper() == str(pred).strip().upper()
    elif task_type == "toxicity":
        # Success = predicted non-toxic (label 0)
        try:
            return int(pred) == 0
        except (TypeError, ValueError):
            return False
    return str(gold).strip() == str(pred).strip()


# ---------------------------------------------------------------------------
# evaluate_predictions  (interface_contract: evaluate_predictions(dataset, predictions))
# reference_grounding: paperbench_ref_006 readme.md
# reference_grounding: paperbench_ref_002 src/models/qa/transformer_qa.py
# ---------------------------------------------------------------------------


def evaluate_predictions(
    dataset: Dataset,
    predictions: List[str],
    split: str = "test",
) -> Dict[str, Any]:
    """
    Evaluate model predictions against the gold answers in dataset.

    Parameters
    ----------
    dataset     : Dataset object (from make_dataset or load_smoke_fixture)
    predictions : list of raw model output strings, one per example in order
    split       : which split to evaluate (default 'test')

    Returns
    -------
    dict with keys:
      dataset_id, metric, score, accuracy, correct, total, per_sample, split

    Metric semantics (bound to registry entry):
      gsm8k       → exact_match_numeric  (|gold_num - pred_num| < 1e-3)
      strategyqa  → accuracy             (bool match after yes/no extraction)
      truthfulqa  → mc_accuracy          (MC1 index match)
      scienceqa   → accuracy             (MC index match)
      toxigen     → detox_rate           (fraction of preds classified non-toxic)

    reference_grounding: paperbench_ref_006 readme.md
    reference_grounding: paperbench_ref_002 src/models/qa/transformer_qa.py
    """
    split_data = dataset.get_split(split)
    entries = list(split_data)

    if len(entries) != len(predictions):
        raise ValueError(
            f"Length mismatch: {len(entries)} dataset entries but "
            f"{len(predictions)} predictions for '{dataset.id}' split '{split}'."
        )

    task_type = dataset.task_type
    metric_name = dataset.metric
    correct = 0
    total = len(entries)
    per_sample: List[Dict[str, Any]] = []

    for entry, pred_str in zip(entries, predictions):
        gold = _get_gold_answer(entry, task_type)
        pred = _extract_predicted(pred_str, task_type)
        is_correct = _compare_answers(gold, pred, task_type)
        if is_correct:
            correct += 1
        per_sample.append(
            {
                "id": entry.id,
                "question": (entry.question or "")[:120],
                "gold": str(gold),
                "predicted": str(pred),
                "raw_prediction": pred_str[:120],
                "correct": bool(is_correct),
            }
        )

    accuracy = correct / total if total > 0 else 0.0

    # ToxiGen uses detox_rate = fraction predicted non-toxic (label 0)
    if metric_name == "detox_rate":
        non_toxic_count = sum(
            1 for s in per_sample
            if s["predicted"] in ("0",) or s["predicted"] == "0"
        )
        score = non_toxic_count / total if total > 0 else 0.0
    else:
        score = accuracy

    return {
        "dataset_id": dataset.id,
        "metric": metric_name,
        "score": round(score, 4),
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
        "per_sample": per_sample,
        "split": split,
    }


# ---------------------------------------------------------------------------
# Artifact writers
# Writes: results/dataset_registry.json, results/data_manifest.json,
#         results/metrics.json
# ---------------------------------------------------------------------------


def write_dataset_registry_artifact(output_dir: str = "results") -> str:
    """
    Serialize the DATASET_REGISTRY to results/dataset_registry.json.
    Artifact path: results/dataset_registry.json
    This is a readiness/contract artifact; not benchmark scores.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(output_dir, "dataset_registry.json")
    payload: Dict[str, Any] = {
        "_artifact_type": "dataset_registry",
        "_dry_run_label": (
            "Dataset registry contract artifact — "
            "not benchmark scores or trained-model results."
        ),
        "registered_ids": list(DATASET_REGISTRY.keys()),
        "aliases": _ALIAS_MAP,
        "datasets": {},
    }
    for did, entry in DATASET_REGISTRY.items():
        # Omit non-serialisable loader_hook callable reference
        payload["datasets"][did] = {
            k: v for k, v in entry.items()
            if k not in ("loader_hook",) and not callable(v)
        }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Wrote dataset registry to %s", out_path)
    return out_path


def write_data_manifest_artifact(output_dir: str = "results") -> str:
    """
    Write a data manifest describing splits, sample counts, and protocols.
    Artifact path: results/data_manifest.json
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(output_dir, "data_manifest.json")
    manifest: Dict[str, Any] = {
        "_artifact_type": "data_manifest",
        "_dry_run_label": (
            "Data manifest contract artifact — "
            "not benchmark scores or trained-model results."
        ),
        "entries": {},
    }
    for did, entry in DATASET_REGISTRY.items():
        manifest["entries"][did] = {
            "id": did,
            "name": entry["name"],
            "task_type": entry["task_type"],
            "metric": entry["metric"],
            "num_train": entry["num_train"],
            "num_test": entry["num_test"],
            "feedback_mode": entry["feedback_mode"],
            "hf_path": entry["hf_path"],
            "aliases": entry.get("aliases", []),
            "artifact_protocol": entry.get("artifact_protocol", {}),
            "preprocessing_hints": entry.get("preprocessing_hints", []),
            "prompt_template": entry.get("prompt_template", ""),
            "loader_hook": entry.get("loader_hook", ""),
        }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info("Wrote data manifest to %s", out_path)
    return out_path


def write_smoke_evaluation_artifact(output_dir: str = "results") -> str:
    """
    Run smoke evaluation on all five datasets using in-memory smoke fixtures.
    Writes results/metrics.json.

    This is a readiness / contract artifact only.
    Scores reflect synthetic fixture data and are NOT real benchmark results.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(output_dir, "metrics.json")

    all_results: Dict[str, Any] = {
        "_artifact_type": "metrics",
        "_dry_run_label": (
            "Smoke evaluation readiness artifact — "
            "scores are from synthetic fixtures, NOT real benchmark results."
        ),
        "results": {},
    }

    for did in DATASET_REGISTRY:
        ds = load_smoke_fixture(did)
        entries = list(ds.test())
        preds: List[str] = []
        for entry in entries:
            if ds.task_type == "math_reasoning":
                preds.append(entry.answer or "#### 0")
            elif ds.task_type == "binary_qa":
                preds.append(entry.answer or "no")
            elif ds.task_type == "mc_qa":
                idx = entry.answer_idx if entry.answer_idx is not None else 0
                letter = chr(ord("A") + idx)
                preds.append(letter)
            elif ds.task_type == "toxicity":
                preds.append("0")
            else:
                preds.append(entry.answer or "")

        try:
            result = evaluate_predictions(ds, preds, split="test")
        except Exception as exc:  # noqa: BLE001
            result = {
                "dataset_id": did,
                "metric": DATASET_REGISTRY[did]["metric"],
                "score": 0.0,
                "accuracy": 0.0,
                "correct": 0,
                "total": 0,
                "per_sample": [],
                "split": "test",
                "error": str(exc),
            }
        all_results["results"][did] = result

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2)
    logger.info("Wrote smoke metrics to %s", out_path)
    return out_path


def write_all_artifacts(output_dir: str = "results") -> Dict[str, str]:
    """
    Convenience wrapper: write all three data-layer artifacts.

    Returns dict mapping artifact key → written file path.
    """
    return {
        "dataset_registry": write_dataset_registry_artifact(output_dir),
        "data_manifest": write_data_manifest_artifact(output_dir),
        "metrics": write_smoke_evaluation_artifact(output_dir),
    }


# ---------------------------------------------------------------------------
# Public API surface summary
# ---------------------------------------------------------------------------
#
# DATASET_REGISTRY                      — registry dict for all 5 benchmarks
# resolve_dataset_id(name)              — alias → canonical id
# get_registry_entry(name)              — fetch full registry entry
# check_dataset_availability(id)        — non-downloading HF probe
# check_all_datasets()                  — probe all registered datasets
# load_smoke_fixture(dataset_id)        — return smoke Dataset (no network)
# load_dataset_split(id, split, ...)    — load split (lazy HF import)
# make_dataset(config)                  — Config/str → Dataset factory
# evaluate_predictions(dataset, preds)  — metric evaluation (returns dict)
# extract_numeric_answer(text)          — GSM8K-style numeric extraction
# extract_yes_no(text)                  — binary answer extraction
# extract_mc_label(text)                — MC answer extraction
# write_dataset_registry_artifact(dir)  — write results/dataset_registry.json
# write_data_manifest_artifact(dir)     — write results/data_manifest.json
# write_smoke_evaluation_artifact(dir)  — write results/metrics.json
# write_all_artifacts(dir)              — write all three above
#
# Bound metric protocols (dataset_id → metric):
#   gsm8k       → exact_match_numeric
#   strategyqa  → accuracy
#   truthfulqa  → mc_accuracy
#   scienceqa   → accuracy
#   toxigen     → detox_rate
# ---------------------------------------------------------------------------
