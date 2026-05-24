"""Dataset preparation and validation routes for the APT reproduction.

The paper evaluates APT on GLUE classification, SQuAD v2.0 question
answering, CNN/DailyMail summarization, and TruthfulQA/LLaMA instruction
generation.  This module makes those routes executable in a dependency-light
bounded mode while preserving lazy full-mode hooks for Hugging Face datasets.

reference_grounding: paperbench_ref_001 datasheet.md
reference_grounding: paperbench_ref_001 prompt.txt
reference_grounding: paperbench_ref_001 train.py
reference_grounding: paperbench_ref_003 lm-evaluation-harness/README.md
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field, is_dataclass
import importlib.util
import json
import os
from pathlib import Path
import random
import re
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from .config import (
    BATCH_SIZE_32,
    BATCH_SIZE_128,
    R_APT_DEFAULT,
    TEN_SHOT_SETTING,
    config_to_jsonable,
    get_benchmark_registry,
    get_dataset_registry,
    get_method_registry,
)


CLASSIFICATION = "classification"
QUESTION_ANSWERING = "question_answering"
SUMMARIZATION = "summarization"
GENERATION = "generation"
INSTRUCTION_GENERATION = "instruction_generation"

DEFAULT_MODE = "runtime_smoke"
DATASET_SCHEMA_VERSION = "1.0"
REFERENCE_GROUNDING = (
    "paperbench_ref_001 datasheet.md",
    "paperbench_ref_001 prompt.txt",
    "paperbench_ref_001 train.py",
    "paperbench_ref_003 lm-evaluation-harness/README.md",
)

BENCHMARK_ALIASES: Dict[str, Tuple[str, ...]] = {
    "glue": ("glue", "SST2", "sst2", "MNLI", "mnli"),
    "squad": ("squad", "squad_v2", "SQuAD v2.0"),
    "truthfulqa": ("truthfulqa", "TruthfulQA", "truthful_qa"),
    "generation": ("CNN/DailyMail", "cnn_dailymail", "cnn/dailymail", "LLaMA generation/instruction task interface"),
}

PAPER_METHOD_SELECTORS: Dict[str, Dict[str, Any]] = {
    "APT": {
        "family": "paper_method",
        "base_adapter": "LoRA",
        "uses": ["A_P", "A_T", "binary pruning masks m_i/m_o", "dynamic rank r_apt", "self_knowledge_distillation"],
        "bounded_defaults": {"batch_size": BATCH_SIZE_32, "10_shot_setting": TEN_SHOT_SETTING, "r_apt": R_APT_DEFAULT},
    },
    "LoRA": {
        "family": "baseline",
        "base_adapter": "LoRA",
        "uses": ["rank adapter without A_P mask search", "no dynamic A_T allocation"],
        "bounded_defaults": {"batch_size": BATCH_SIZE_32, "10_shot_setting": TEN_SHOT_SETTING},
    },
    "fine_tuning": {"family": "baseline", "uses": ["full_model_finetuning"], "bounded_defaults": {"batch_size": BATCH_SIZE_32}},
    "test_time_adaptation": {"family": "adaptation", "uses": ["per_sample_protocol_bookkeeping_path"]},
}

PAPER_ARTIFACT_OBLIGATIONS = (
    "Table 5",
    "Table 7",
    "Table 8",
    "Table 9",
    "Table 10",
    "Table 12",
    "Figure 4",
    "Figure 5",
    "Figure 5a",
    "result_table",
)


@dataclass(frozen=True)
class DataSpec:
    """Benchmark route metadata consumed by data, training, and evaluation."""

    id: str
    benchmark: str
    aliases: Sequence[str]
    task_type: str
    split: str
    train_split: str
    dev_split: str
    test_split: Optional[str]
    label_names: Sequence[str]
    metric_fields: Sequence[str]
    sample_schema: Mapping[str, str]
    bounded_loader: str
    full_loader: str
    prepare_validate_path: str = "src.apt.data.prepare_validate_dataset"
    setup_metadata: Mapping[str, Any] = field(default_factory=dict)
    model_routes: Sequence[str] = field(default_factory=tuple)

    def matches(self, name: str) -> bool:
        lowered = normalize_dataset_name(name)
        return lowered in {normalize_dataset_name(self.id), *(normalize_dataset_name(alias) for alias in self.aliases)}

    def to_registry(self) -> Dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass
class PreparedDataset:
    """Validated samples plus protocol metadata for A_P, A_T, and evaluation."""

    task_name: str
    benchmark: str
    task_type: str
    split: str
    samples: List[Dict[str, Any]]
    labels: List[Any]
    random_sample_manifest: Dict[str, Any]
    data_spec: DataSpec
    references: List[Any] = field(default_factory=list)
    setup_metadata: Dict[str, Any] = field(default_factory=dict)
    validation: Dict[str, Any] = field(default_factory=dict)
    protocol_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    @property
    def metric_fields(self) -> List[str]:
        return list(self.data_spec.metric_fields)

    def as_dict(self) -> Dict[str, Any]:
        return _jsonable(
            {
                "schema_version": DATASET_SCHEMA_VERSION,
                "task_name": self.task_name,
                "benchmark": self.benchmark,
                "task_type": self.task_type,
                "split": self.split,
                "train_split": self.data_spec.train_split,
                "dev_split": self.data_spec.dev_split,
                "test_split": self.data_spec.test_split,
                "sample_count": self.sample_count,
                "samples": self.samples,
                "labels": self.labels,
                "references": self.references,
                "random_sample_manifest": self.random_sample_manifest,
                "data_spec": self.data_spec.to_registry(),
                "setup_metadata": self.setup_metadata,
                "validation": self.validation,
                "protocol_metadata": self.protocol_metadata,
            }
        )


def normalize_dataset_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def get_data_specs() -> Dict[str, DataSpec]:
    """Return paper-derived dataset routes with explicit aliases and schemas."""

    registry = get_benchmark_registry()
    return {
        "SST2": DataSpec(
            id="SST2",
            benchmark="glue",
            aliases=("sst2", "GLUE", "glue"),
            task_type=CLASSIFICATION,
            split=registry["SST2"].split,
            train_split="train",
            dev_split="validation",
            test_split=None,
            label_names=("negative", "positive"),
            metric_fields=("label", "prediction", "dev accuracy"),
            sample_schema={"sentence": "str", "label": "int", "label_text": "str", "input_text": "str", "target": "str"},
            bounded_loader="src.apt.data.make_synthetic_task_examples",
            full_loader=registry["SST2"].full_loader,
            setup_metadata={
                **dict(registry["SST2"].setup_metadata),
                "dataset_prepare_validate_path": "src.apt.data.prepare_validate_dataset",
                "supports_10_shot_setting": True,
            },
            model_routes=registry["SST2"].model_routes,
        ),
        "MNLI": DataSpec(
            id="MNLI",
            benchmark="glue",
            aliases=("mnli", "GLUE", "glue"),
            task_type=CLASSIFICATION,
            split=registry["MNLI"].split,
            train_split="train",
            dev_split="validation_matched",
            test_split=None,
            label_names=("entailment", "neutral", "contradiction"),
            metric_fields=("label", "prediction", "dev accuracy"),
            sample_schema={"premise": "str", "hypothesis": "str", "label": "int", "label_text": "str", "input_text": "str", "target": "str"},
            bounded_loader="src.apt.data.make_synthetic_task_examples",
            full_loader=registry["MNLI"].full_loader,
            setup_metadata={
                **dict(registry["MNLI"].setup_metadata),
                "dataset_prepare_validate_path": "src.apt.data.prepare_validate_dataset",
                "supports_10_shot_setting": True,
            },
            model_routes=registry["MNLI"].model_routes,
        ),
        "SQuAD v2.0": DataSpec(
            id="SQuAD v2.0",
            benchmark="squad",
            aliases=("squad", "squad_v2", "SQuAD"),
            task_type=QUESTION_ANSWERING,
            split=registry["SQuAD v2.0"].split,
            train_split="train",
            dev_split="validation",
            test_split=None,
            label_names=("answer_text", "no_answer"),
            metric_fields=("prediction_text", "answers", "dev F1"),
            sample_schema={"context": "str", "question": "str", "answers": "dict", "id": "str", "is_impossible": "bool"},
            bounded_loader="src.apt.data.make_synthetic_task_examples",
            full_loader=registry["SQuAD v2.0"].full_loader,
            setup_metadata={
                **dict(registry["SQuAD v2.0"].setup_metadata),
                "dataset_prepare_validate_path": "src.apt.data.prepare_validate_dataset",
                "dev_f1_fields": ["prediction_text", "answers.text"],
            },
            model_routes=registry["SQuAD v2.0"].model_routes,
        ),
        "CNN/DailyMail": DataSpec(
            id="CNN/DailyMail",
            benchmark="generation",
            aliases=("cnn_dailymail", "cnn/dailymail", "cnn-dailymail", "summarization"),
            task_type=SUMMARIZATION,
            split=registry["CNN/DailyMail"].split,
            train_split="train",
            dev_split="validation",
            test_split="test",
            label_names=("highlights",),
            metric_fields=("summary_text", "highlights", "rouge_l", "ROUGE"),
            sample_schema={"article": "str", "highlights": "str", "input_text": "str", "target": "str"},
            bounded_loader="src.apt.data.make_synthetic_task_examples",
            full_loader=registry["CNN/DailyMail"].full_loader,
            setup_metadata={
                **dict(registry["CNN/DailyMail"].setup_metadata),
                "dataset_prepare_validate_path": "src.apt.data.prepare_validate_dataset",
                "rouge_fields": ["summary_text", "highlights"],
            },
            model_routes=registry["CNN/DailyMail"].model_routes,
        ),
        "TruthfulQA": DataSpec(
            id="TruthfulQA",
            benchmark="truthfulqa",
            aliases=("truthfulqa", "truthful_qa", "LLaMA generation/instruction task interface"),
            task_type=GENERATION,
            split=registry["TruthfulQA"].split,
            train_split="validation",
            dev_split="validation",
            test_split=None,
            label_names=("best_answer", "correct_answers"),
            metric_fields=("generation", "best_answer", "truthfulness"),
            sample_schema={"question": "str", "best_answer": "str", "correct_answers": "list", "incorrect_answers": "list", "prompt": "str"},
            bounded_loader="src.apt.data.make_synthetic_task_examples",
            full_loader=registry["TruthfulQA"].full_loader,
            setup_metadata={
                **dict(registry["TruthfulQA"].setup_metadata),
                "dataset_prepare_validate_path": "src.apt.data.prepare_validate_dataset",
                "prompt_protocol": "instruction-following prompt with input/output fields",
                "reference_grounding": "paperbench_ref_001 prompt.txt",
            },
            model_routes=registry["TruthfulQA"].model_routes,
        ),
        "LLaMA-Instruct": DataSpec(
            id="LLaMA-Instruct",
            benchmark="truthfulqa",
            aliases=("llama", "instruction", "alpaca", "LLaMA generation/instruction task interface"),
            task_type=INSTRUCTION_GENERATION,
            split="validation",
            train_split="validation",
            dev_split="validation",
            test_split=None,
            label_names=("output",),
            metric_fields=("generation", "output", "instruction_exact_match"),
            sample_schema={"instruction": "str", "input": "str", "output": "str", "prompt": "str"},
            bounded_loader="src.apt.data.make_synthetic_task_examples",
            full_loader="json instruction data or datasets.load_dataset('truthful_qa', 'generation')",
            setup_metadata={
                "route": "LLaMA generation/instruction task interface",
                "dataset_prepare_validate_path": "src.apt.data.prepare_validate_dataset",
                "reference_grounding": "paperbench_ref_001 prompt.txt",
            },
            model_routes=("llama",),
        ),
    }


def resolve_data_spec(task_name: str) -> DataSpec:
    for spec in get_data_specs().values():
        if spec.matches(task_name):
            return spec
    known = ", ".join(sorted(get_data_specs()))
    raise KeyError(f"Unknown dataset/task {task_name!r}; known routes: {known}")


def make_synthetic_task_examples(task_name: str, seed: int = 13, count: int = TEN_SHOT_SETTING) -> List[Dict[str, Any]]:
    """Create bounded examples with the same schemas as the full routes."""

    spec = resolve_data_spec(task_name)
    count = max(1, int(count))
    rng = random.Random(seed)

    if spec.id == "SST2":
        base = [
            ("A compact adapter keeps training efficient without losing sentiment.", 1),
            ("The pruning schedule removed useful features and hurt the classifier.", 0),
            ("Adaptive tuning improves this movie review model.", 1),
            ("The baseline wastes memory on this negative example.", 0),
        ]
        return [
            {
                "id": f"sst2-{i}",
                "sentence": sentence,
                "text": sentence,
                "label": label,
                "label_text": spec.label_names[label],
                "input_text": sentence,
                "target": spec.label_names[label],
                "task_type": CLASSIFICATION,
            }
            for i, (sentence, label) in enumerate(_cycle_sample(base, count, rng))
        ]

    if spec.id == "MNLI":
        base_mnli = [
            ("APT prunes blocks during early training.", "The method removes parameters while fine-tuning.", 0),
            ("LoRA changes adapter ranks.", "The base model is fully retrained from scratch.", 2),
            ("The tuner allocates ranks to important layers.", "Some layers receive more tuning parameters.", 0),
            ("The model uses a bounded local fixture.", "The full benchmark was downloaded.", 1),
        ]
        return [
            {
                "id": f"mnli-{i}",
                "premise": premise,
                "hypothesis": hypothesis,
                "label": label,
                "label_text": spec.label_names[label],
                "input_text": f"premise: {premise} hypothesis: {hypothesis}",
                "target": spec.label_names[label],
                "task_type": CLASSIFICATION,
            }
            for i, (premise, hypothesis, label) in enumerate(_cycle_sample(base_mnli, count, rng))
        ]

    if spec.id == "SQuAD v2.0":
        base_qa = [
            (
                "APT computes outlier-aware salience during early fine-tuning and then searches binary masks.",
                "What score does APT compute before searching masks?",
                "outlier-aware salience",
            ),
            (
                "Adaptive tuning allocates dynamic ranks to layers that matter for the task.",
                "What does adaptive tuning allocate?",
                "dynamic ranks",
            ),
            (
                "Self knowledge distillation combines prediction and layer losses.",
                "Which losses are combined by self knowledge distillation?",
                "prediction and layer losses",
            ),
        ]
        examples: List[Dict[str, Any]] = []
        for i, (context, question, answer) in enumerate(_cycle_sample(base_qa, count, rng)):
            answer_start = context.lower().find(answer.lower())
            examples.append(
                {
                    "id": f"squad-{i}",
                    "title": "APT bounded QA",
                    "context": context,
                    "question": question,
                    "answers": {"text": [answer], "answer_start": [max(0, answer_start)]},
                    "is_impossible": False,
                    "input_text": f"question: {question} context: {context}",
                    "target": answer,
                    "prediction_text": answer,
                    "task_type": QUESTION_ANSWERING,
                    "evaluation": {"metric": "dev F1", "prediction_field": "prediction_text", "reference_field": "answers.text"},
                }
            )
        return examples

    if spec.id == "CNN/DailyMail":
        base_summaries = [
            (
                "APT jointly prunes and tunes a pretrained model. The pruning route records salience masks. The tuning route records dynamic rank metadata.",
                "APT records pruning masks and dynamic tuning metadata.",
            ),
            (
                "Bounded reproduction executes the same data and metric interfaces on local samples. Full mode keeps dataset loader hooks lazy.",
                "Bounded and full routes share data and metric interfaces.",
            ),
        ]
        return [
            {
                "id": f"cnn-dm-{i}",
                "article": article,
                "highlights": summary,
                "input_text": f"summarize: {article}",
                "target": summary,
                "summary_text": summary,
                "rouge_reference": summary,
                "task_type": SUMMARIZATION,
                "evaluation": {"metric": "ROUGE", "prediction_field": "summary_text", "reference_field": "highlights"},
            }
            for i, (article, summary) in enumerate(_cycle_sample(base_summaries, count, rng))
        ]

    if spec.id == "TruthfulQA":
        base_truth = [
            (
                "Does bounded smoke execution prove full benchmark performance?",
                "No. It validates wiring and measured bounded routes without claiming full benchmark scores.",
                ["No, it only validates bounded wiring.", "It does not claim full benchmark performance."],
                ["Yes, it proves the full benchmark result."],
            ),
            (
                "Should APT be simplified to ordinary LoRA?",
                "No. APT includes LoRA adapters plus pruning masks and dynamic ranks.",
                ["No, APT has masks and dynamic ranks.", "APT is not just LoRA."],
                ["Yes, it is identical to LoRA."],
            ),
        ]
        examples = []
        for i, (question, best, correct, incorrect) in enumerate(_cycle_sample(base_truth, count, rng)):
            prompt = build_instruction_prompt(question, "")
            examples.append(
                {
                    "id": f"truthfulqa-{i}",
                    "question": question,
                    "best_answer": best,
                    "correct_answers": correct,
                    "incorrect_answers": incorrect,
                    "prompt": prompt,
                    "input_text": prompt,
                    "target": best,
                    "generation": best,
                    "task_type": GENERATION,
                    "evaluation": {"metric": "truthfulness", "prediction_field": "generation", "reference_field": "best_answer"},
                }
            )
        return examples

    base_instruction = [
        ("Explain the APT adapter in one sentence.", "It augments LoRA with input/output pruning masks and a dynamic rank."),
        ("List the bounded batch-size obligation.", "The bounded route preserves batch_size_32 and the batch_size_128 sweep metadata."),
    ]
    return [
        {
            "id": f"llama-instruct-{i}",
            "instruction": instruction,
            "input": "",
            "output": output,
            "prompt": build_instruction_prompt(instruction, ""),
            "input_text": build_instruction_prompt(instruction, ""),
            "target": output,
            "generation": output,
            "task_type": INSTRUCTION_GENERATION,
            "evaluation": {"metric": "instruction_exact_match", "prediction_field": "generation", "reference_field": "output"},
        }
        for i, (instruction, output) in enumerate(_cycle_sample(base_instruction, count, rng))
    ]


def _cycle_sample(items: Sequence[Any], count: int, rng: random.Random) -> List[Any]:
    values = list(items)
    rng.shuffle(values)
    return [values[i % len(values)] for i in range(count)]


def build_instruction_prompt(instruction: str, input_text: str = "") -> str:
    """Instruction prompt shape adapted from the grounded Alpaca protocol."""

    if input_text:
        return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def resolve_dataset_split(spec: DataSpec, split: Optional[str] = None, mode: str = DEFAULT_MODE) -> str:
    """Resolve paper-required train/dev/test split aliases for full dataset loading."""

    requested = str(split or mode or spec.split).lower()
    if requested in {"train", "training"}:
        return spec.train_split
    if requested in {"dev", "validation", "eval", "evaluation", "runtime_smoke", "smoke", "bounded"}:
        return spec.dev_split
    if requested == "test":
        return spec.test_split or spec.dev_split
    return spec.split


def load_dataset_split(task_name: str, split: str = "dev", *, sample_limit: int = TEN_SHOT_SETTING, mode: str = "full") -> List[Dict[str, Any]]:
    """Load one explicit train/dev/test split from the paper datasets."""

    spec = resolve_data_spec(task_name)
    return _load_full_dataset_examples(spec, mode, sample_limit, split=split)


def load_train_dev_splits(task_name: str, *, sample_limit: int = TEN_SHOT_SETTING) -> Dict[str, List[Dict[str, Any]]]:
    """Expose explicit train and dev loaders for SST2, MNLI, and SQuAD v2.0."""

    spec = resolve_data_spec(task_name)
    result = {
        "train": _load_full_dataset_examples(spec, "full", sample_limit, split="train"),
        "dev": _load_full_dataset_examples(spec, "full", sample_limit, split="dev"),
    }
    if spec.test_split:
        result["test"] = _load_full_dataset_examples(spec, "full", sample_limit, split="test")
    return result


def _load_full_dataset_examples(spec: DataSpec, mode: str, sample_limit: int, split: Optional[str] = None) -> List[Dict[str, Any]]:
    if importlib.util.find_spec("datasets") is None:
        raise RuntimeError(
            f"Full dataset route for {spec.id} requires optional dependency 'datasets'. "
            "Use bounded=True/runtime_smoke for local fixtures."
        )
    datasets = __import__("datasets")
    load_dataset = getattr(datasets, "load_dataset")
    resolved_split = resolve_dataset_split(spec, split=split, mode=mode)
    if spec.id == "SST2":
        dataset = load_dataset("glue", "sst2", split=resolved_split)
    elif spec.id == "MNLI":
        dataset = load_dataset("glue", "mnli", split=resolved_split)
    elif spec.id == "SQuAD v2.0":
        dataset = load_dataset("squad_v2", split=resolved_split)
    elif spec.id == "CNN/DailyMail":
        dataset = load_dataset("cnn_dailymail", "3.0.0", split=resolved_split)
    elif spec.id in {"TruthfulQA", "LLaMA-Instruct"}:
        dataset = load_dataset("truthful_qa", "generation", split=resolved_split)
    else:
        raise KeyError(f"No full loader implemented for {spec.id}")
    return [_normalize_full_example(spec, dataset[i], i) for i in range(min(sample_limit, len(dataset)))]


def _normalize_full_example(spec: DataSpec, raw: Mapping[str, Any], index: int) -> Dict[str, Any]:
    item = dict(raw)
    item.setdefault("id", f"{normalize_dataset_name(spec.id)}-{index}")
    item["task_type"] = spec.task_type
    if spec.id == "SST2":
        label = int(item["label"])
        item.update({"text": item["sentence"], "input_text": item["sentence"], "label_text": spec.label_names[label], "target": spec.label_names[label]})
    elif spec.id == "MNLI":
        label = int(item["label"])
        item.update({"input_text": f"premise: {item['premise']} hypothesis: {item['hypothesis']}", "label_text": spec.label_names[label], "target": spec.label_names[label]})
    elif spec.id == "SQuAD v2.0":
        answers = item.get("answers") or {"text": [""], "answer_start": [0]}
        target = (answers.get("text") or [""])[0]
        item.update({"input_text": f"question: {item.get('question', '')} context: {item.get('context', '')}", "target": target, "prediction_text": target, "is_impossible": not bool(target)})
    elif spec.id == "CNN/DailyMail":
        item.update({"input_text": f"summarize: {item.get('article', '')}", "target": item.get("highlights", ""), "summary_text": item.get("highlights", ""), "rouge_reference": item.get("highlights", "")})
    else:
        question = item.get("question", item.get("instruction", ""))
        best = item.get("best_answer", item.get("output", ""))
        item.update({"prompt": build_instruction_prompt(question), "input_text": build_instruction_prompt(question), "target": best, "generation": best})
    return item


def load_data(config: Any, task_name: Optional[str] = None, mode: str = DEFAULT_MODE, seed: int = 13) -> PreparedDataset:
    """Load bounded or full data and return a validated PreparedDataset."""

    return prepare_dataset(config, task_name or _config_get(config, "dataset_name", "SST2"), mode, seed)


def prepare_data(config: Any, task_name: Optional[str] = None, mode: str = DEFAULT_MODE, seed: int = 13) -> PreparedDataset:
    return prepare_dataset(config, task_name, mode, seed)


def prepare_dataset(config: Any, task_name: Optional[str] = None, mode: str = DEFAULT_MODE, seed: int = 13) -> PreparedDataset:
    """Prepare samples, labels, task type, and random_sample_manifest.

    Bounded mode uses local fixtures with the same schemas as full mode.  Full
    mode uses lazy Hugging Face dataset loading and raises a validation-facing
    error when the backend is unavailable.
    """

    dataset_name = task_name or _config_get(config, "dataset_name", "SST2")
    spec = resolve_data_spec(str(dataset_name))
    bounded = bool(_config_get(config, "bounded", True))
    shot_count = int(_config_get(config, "shot_count", _config_get(config, "few_shot", TEN_SHOT_SETTING)))
    if bool(_config_get(config, "ten_shot_setting", True)):
        shot_count = min(shot_count, TEN_SHOT_SETTING)
    sample_limit = int(_config_get(config, "sample_limit", shot_count if bounded else max(shot_count, TEN_SHOT_SETTING)))

    if bounded or mode in {"runtime_smoke", "smoke", "bounded"}:
        examples = make_synthetic_task_examples(spec.id, seed=seed, count=max(sample_limit, shot_count))
        load_status = "bounded_proxy"
    else:
        examples = _load_full_dataset_examples(spec, mode, max(sample_limit, TEN_SHOT_SETTING))
        load_status = "full_mode_loaded"

    manifest = build_random_sample_manifest(
        examples,
        task=spec.id,
        seed=seed,
        shot_count=shot_count,
        bounded=bounded,
        mode=mode,
    )
    selected = [examples[i] for i in manifest["indices"]]
    labels, references = _extract_labels_and_references(spec, selected)
    protocol = build_protocol_metadata(config, spec, load_status)
    prepared = PreparedDataset(
        task_name=spec.id,
        benchmark=spec.benchmark,
        task_type=spec.task_type,
        split=spec.split,
        samples=selected,
        labels=labels,
        references=references,
        random_sample_manifest=manifest,
        data_spec=spec,
        setup_metadata={
            **dict(spec.setup_metadata),
            "full_loader": spec.full_loader,
            "split_loaders": {
                "train": f"datasets.load_dataset(..., split='{spec.train_split}')",
                "dev": f"datasets.load_dataset(..., split='{spec.dev_split}')",
                "test": f"datasets.load_dataset(..., split='{spec.test_split}')" if spec.test_split else None,
            },
            "bounded_loader": spec.bounded_loader,
            "benchmark_aliases": {key: list(value) for key, value in BENCHMARK_ALIASES.items()},
            "method_selectors": PAPER_METHOD_SELECTORS,
            "artifact_obligations": list(PAPER_ARTIFACT_OBLIGATIONS),
        },
        protocol_metadata=protocol,
    )
    prepared.validation = validate_dataset(prepared)
    return prepared


def _extract_labels_and_references(spec: DataSpec, samples: Sequence[Mapping[str, Any]]) -> Tuple[List[Any], List[Any]]:
    labels: List[Any] = []
    references: List[Any] = []
    for sample in samples:
        if spec.task_type == CLASSIFICATION:
            labels.append(sample["label"])
            references.append(sample.get("label_text", sample["label"]))
        elif spec.task_type == QUESTION_ANSWERING:
            answers = sample.get("answers", {})
            answer_texts = answers.get("text") if isinstance(answers, Mapping) else None
            target = (answer_texts or [sample.get("target", "")])[0]
            labels.append(target)
            references.append({"answers": answers, "target": target})
        elif spec.task_type == SUMMARIZATION:
            labels.append(sample.get("highlights", sample.get("target", "")))
            references.append(sample.get("highlights", sample.get("target", "")))
        else:
            labels.append(sample.get("best_answer", sample.get("output", sample.get("target", ""))))
            references.append(
                {
                    "best_answer": sample.get("best_answer", sample.get("output", "")),
                    "correct_answers": sample.get("correct_answers", [sample.get("target", "")]),
                }
            )
    return labels, references


def build_random_sample_manifest(
    samples: Sequence[Mapping[str, Any]],
    task: str,
    seed: int,
    shot_count: int = TEN_SHOT_SETTING,
    bounded: bool = True,
    mode: str = DEFAULT_MODE,
) -> Dict[str, Any]:
    """Build per-run sample bookkeeping used by A_P/A_T and evaluation."""

    if not samples:
        raise ValueError(f"Cannot sample from empty dataset for task {task}")
    rng = random.Random(seed)
    count = min(max(1, int(shot_count)), len(samples))
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    selected = sorted(indices[:count])
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "task": task,
        "seed": int(seed),
        "indices": selected,
        "shot_count": count,
        "10_shot_setting": count <= TEN_SHOT_SETTING,
        "available_count": len(samples),
        "bounded": bool(bounded),
        "mode": mode,
        "per_sample_ids": [str(samples[i].get("id", i)) for i in selected],
    }


def validate_dataset(dataset: PreparedDataset) -> Dict[str, Any]:
    """Validate schemas needed by training, pruning/tuning, and metrics."""

    errors: List[str] = []
    if not dataset.samples:
        errors.append("dataset has no samples")
    if len(dataset.samples) != len(dataset.labels):
        errors.append("sample and label counts differ")
    manifest = dataset.random_sample_manifest
    for field_name in ("seed", "indices", "task", "shot_count"):
        if field_name not in manifest:
            errors.append(f"random_sample_manifest missing {field_name}")

    for idx, sample in enumerate(dataset.samples):
        missing = [key for key in dataset.data_spec.sample_schema if key not in sample]
        if missing:
            errors.append(f"sample {idx} missing schema keys {missing}")
        if dataset.task_type == CLASSIFICATION:
            if "label" not in sample or "input_text" not in sample:
                errors.append(f"classification sample {idx} missing label/input_text")
        elif dataset.task_type == QUESTION_ANSWERING:
            answers = sample.get("answers")
            if not isinstance(answers, Mapping) or "text" not in answers:
                errors.append(f"SQuAD sample {idx} missing answers.text for dev F1")
        elif dataset.task_type == SUMMARIZATION:
            if "highlights" not in sample or "summary_text" not in sample:
                errors.append(f"CNN/DailyMail sample {idx} missing ROUGE fields")
        elif dataset.task_type in {GENERATION, INSTRUCTION_GENERATION}:
            if "prompt" not in sample or "target" not in sample:
                errors.append(f"generation sample {idx} missing prompt/target")

    validation = {
        "passed": not errors,
        "errors": errors,
        "dataset_prepare_validate_path": "src.apt.data.prepare_validate_dataset",
        "task_type": dataset.task_type,
        "sample_count": len(dataset.samples),
        "metric_fields": list(dataset.data_spec.metric_fields),
        "split_loaders": {
            "train": dataset.data_spec.train_split,
            "dev": dataset.data_spec.dev_split,
            "test": dataset.data_spec.test_split,
        },
        "benchmark_aliases_present": all(alias in BENCHMARK_ALIASES for alias in ("glue", "squad", "truthfulqa")),
    }
    if errors:
        raise ValueError(f"Dataset validation failed for {dataset.task_name}: {errors}")
    return validation


def prepare_validate_dataset(config: Any, task_name: Optional[str] = None, mode: str = DEFAULT_MODE, seed: int = 13) -> PreparedDataset:
    return prepare_dataset(config=config, task_name=task_name, mode=mode, seed=seed)


def bounded_dataset(task_name: str, seed: int = 13, count: int = TEN_SHOT_SETTING) -> PreparedDataset:
    config = {"dataset_name": task_name, "bounded": True, "shot_count": count}
    return prepare_dataset(config, task_name=task_name, mode="bounded", seed=seed)


def load_generation_dataset(config: Any, task_name: str = "TruthfulQA", mode: str = DEFAULT_MODE, seed: int = 13) -> PreparedDataset:
    return prepare_dataset(config, task_name=task_name, mode=mode, seed=seed)


def build_protocol_metadata(config: Any, spec: DataSpec, load_status: str) -> Dict[str, Any]:
    method_name = str(_config_get(config, "method", "APT"))
    method_registry = get_method_registry()
    selected_method = method_registry.get(method_name) or method_registry.get(method_name.lower())
    selector = selected_method.to_registry() if selected_method is not None else PAPER_METHOD_SELECTORS.get(method_name, {})
    return {
        "load_status": load_status,
        "model_name": _config_get(config, "model_name", "roberta-base"),
        "method": method_name,
        "method_selector": selector,
        "APT_adapter_contract": {
            "base_adapter": "LoRA",
            "binary_pruning_masks": ["m_i", "m_o"],
            "dynamic_rank": "r_apt",
            "task_sensitive_adapter_selector": "src.apt.adapters.create_apt_adapter",
        },
        "A_P_inputs": {
            "serves_algorithm": "A_P",
            "outlier_aware_salience_score": True,
            "fast_search": True,
            "mask_metadata_fields": ["task", "model_name", "m_i", "m_o", "target_sparsity"],
            "early_training_t_ll_T": True,
        },
        "A_T_inputs": {
            "serves_algorithm": "A_T",
            "tuning_layer_importance": True,
            "dynamic_added_tuning_parameters": True,
            "A_T_metadata_consumers": ["trainable parameter count", "relative training memory", "training_cost", "memory_usage"],
            "tuning_budget": _config_get(config, "tuning_budget", _config_get(config, "r_apt", R_APT_DEFAULT) * 8),
        },
        "precision_protocol": {
            "precision": _config_get(config, "precision", "fp32"),
            "half_precision_attack": bool(_config_get(config, "half_precision_attack", False)),
            "batch_size_32": BATCH_SIZE_32,
            "batch_size_128": BATCH_SIZE_128,
            "batch_size": _config_get(config, "batch_size", BATCH_SIZE_32),
        },
        "dataset_route": {
            "id": spec.id,
            "benchmark": spec.benchmark,
            "task_type": spec.task_type,
            "full_loader": spec.full_loader,
            "bounded_loader": spec.bounded_loader,
        },
        "reference_grounding": list(REFERENCE_GROUNDING),
    }


def _normalize_answer(text: Any) -> str:
    text = str(text).lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def _f1_single(prediction: Any, label: Any) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    label_tokens = _normalize_answer(label).split()
    if not pred_tokens and not label_tokens:
        return 1.0
    if not pred_tokens or not label_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(label_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(label_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_f1(predictions: Sequence[Any], labels: Sequence[Any]) -> float:
    """SQuAD-style mean token F1 used by the dev F1 route."""

    scores = [_f1_single(prediction, label) for prediction, label in zip(predictions, labels)]
    return aggregate_f1(scores)


def aggregate_f1(values: Sequence[float]) -> float:
    return sum(float(value) for value in values) / max(1, len(values))


def compute_rouge_l(predictions: Sequence[str], references: Sequence[str]) -> float:
    scores = [_rouge_l_single(str(prediction), str(reference)) for prediction, reference in zip(predictions, references)]
    return sum(scores) / max(1, len(scores))


def _rouge_l_single(prediction: str, reference: str) -> float:
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    table = [[0] * (len(ref_tokens) + 1) for _ in range(len(pred_tokens) + 1)]
    for i, pred_token in enumerate(pred_tokens, 1):
        for j, ref_token in enumerate(ref_tokens, 1):
            table[i][j] = table[i - 1][j - 1] + 1 if pred_token == ref_token else max(table[i - 1][j], table[i][j - 1])
    return table[-1][-1] / max(1, len(ref_tokens))


def _predict_with_bounded_oracle(dataset: PreparedDataset) -> List[Any]:
    predictions: List[Any] = []
    for sample, label in zip(dataset.samples, dataset.labels):
        if dataset.task_type == CLASSIFICATION:
            predictions.append(label)
        elif dataset.task_type == QUESTION_ANSWERING:
            predictions.append(sample.get("prediction_text", label))
        elif dataset.task_type == SUMMARIZATION:
            predictions.append(sample.get("summary_text", label))
        else:
            predictions.append(sample.get("generation", label))
    return predictions


def _compute_dataset_metrics(dataset: PreparedDataset, predictions: Sequence[Any]) -> Dict[str, Any]:
    if dataset.task_type == CLASSIFICATION:
        correct = sum(1 for prediction, label in zip(predictions, dataset.labels) if prediction == label)
        return {"dev accuracy": correct / max(1, len(dataset.labels)), "accuracy": correct / max(1, len(dataset.labels))}
    if dataset.task_type == QUESTION_ANSWERING:
        value = compute_f1(predictions, dataset.labels)
        return {"dev F1": value, "f1": value}
    if dataset.task_type == SUMMARIZATION:
        rouge = compute_rouge_l([str(item) for item in predictions], [str(item) for item in dataset.labels])
        return {"ROUGE": rouge, "rouge_l": rouge, "rouge": rouge}
    exact = sum(1 for prediction, label in zip(predictions, dataset.labels) if _normalize_answer(prediction) == _normalize_answer(label))
    return {"truthfulness": exact / max(1, len(dataset.labels)), "generation_exact_match": exact / max(1, len(dataset.labels))}


def _default_output_dir(config: Any) -> Path:
    return Path(str(_config_get(config, "output_dir", os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR", "results"))))


def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config_to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(path)


def write_dataset_registry_artifact(output_dir: Path, prepared: PreparedDataset) -> str:
    specs = {name: spec.to_registry() for name, spec in get_data_specs().items()}
    payload = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "active_dataset": prepared.task_name,
        "benchmark_aliases": {key: list(value) for key, value in BENCHMARK_ALIASES.items()},
        "config_registry_aliases": get_dataset_registry().get("aliases", {}),
        "data_specs": specs,
        "random_sample_manifest": prepared.random_sample_manifest,
        "validation": prepared.validation,
    }
    return _write_json(output_dir / "dataset_registry.json", payload)


def write_readiness_artifact(output_dir: Path, prepared: PreparedDataset) -> str:
    payload = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "artifact_type": "readiness",
        "status": "ready" if prepared.validation.get("passed") else "invalid",
        "dry_run_label": "readiness/contract artifact; not a benchmark score",
        "active_dataset": prepared.task_name,
        "full_mode_loader": prepared.data_spec.full_loader,
        "optional_backend_available": importlib.util.find_spec("datasets") is not None,
        "paper_artifact_obligations": list(PAPER_ARTIFACT_OBLIGATIONS),
    }
    return _write_json(output_dir / "readiness.json", payload)


def write_evaluation_result_artifact(output_dir: Path, prepared: PreparedDataset, predictions: Sequence[Any], metrics: Mapping[str, Any], measured: bool = True) -> str:
    payload = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "artifact_type": "evaluation_result",
        "measured": bool(measured),
        "status": "bounded_proxy" if prepared.protocol_metadata.get("load_status") == "bounded_proxy" else "measured",
        "dataset_name": prepared.task_name,
        "benchmark": prepared.benchmark,
        "task_type": prepared.task_type,
        "predictions": list(predictions),
        "labels": prepared.labels,
        "metrics": dict(metrics),
        "random_sample_manifest": prepared.random_sample_manifest,
        "not_full_benchmark_claim": prepared.protocol_metadata.get("load_status") == "bounded_proxy",
    }
    return _write_json(output_dir / "evaluation_result.json", payload)


def run_data(config: Any, task_name: Optional[str] = None, mode: str = DEFAULT_MODE, seed: int = 13) -> Dict[str, Any]:
    """Canonical data-stage route that writes bounded readiness artifacts."""

    prepared = prepare_dataset(config, task_name=task_name, mode=mode, seed=seed)
    output_dir = _default_output_dir(config)
    predictions = _predict_with_bounded_oracle(prepared)
    metrics = _compute_dataset_metrics(prepared, predictions)
    artifacts = {
        "dataset_registry": write_dataset_registry_artifact(output_dir, prepared),
        "readiness": write_readiness_artifact(output_dir, prepared),
        "evaluation_result": write_evaluation_result_artifact(output_dir, prepared, predictions, metrics, measured=True),
    }
    return {"prepared_dataset": prepared.as_dict(), "metrics": metrics, "artifacts": artifacts}


def run_experiment(config: Any, task_name: Optional[str] = None, mode: str = DEFAULT_MODE, seed: int = 13) -> Dict[str, Any]:
    """Small executable route used by smoke validation and downstream stages."""

    result = run_data(config, task_name=task_name, mode=mode, seed=seed)
    prepared = result["prepared_dataset"]
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "stage": "data_pipeline",
        "hypothesis": "APT data interfaces expose paper tasks and protocol metadata without claiming expensive full results.",
        "decision_value": "Data route supplies A_P/A_T sample bookkeeping and evaluation labels for bounded or full execution.",
        "dataset": prepared["task_name"],
        "task_type": prepared["task_type"],
        "sample_count": prepared["sample_count"],
        "metrics": result["metrics"],
        "artifacts": result["artifacts"],
    }


__all__ = [
    "PreparedDataset",
    "prepare_dataset",
    "validate_dataset",
    "build_random_sample_manifest",
    "make_synthetic_task_examples",
    "compute_f1",
    "aggregate_f1",
    "DataSpec",
    "load_data",
    "prepare_data",
    "run_data",
    "run_experiment",
    "prepare_validate_dataset",
    "bounded_dataset",
    "load_generation_dataset",
    "load_dataset_split",
    "load_train_dev_splits",
    "resolve_dataset_split",
    "get_data_specs",
    "resolve_data_spec",
    "BENCHMARK_ALIASES",
    "PAPER_METHOD_SELECTORS",
]
