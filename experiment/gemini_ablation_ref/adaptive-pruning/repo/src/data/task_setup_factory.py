import os
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

# reference_grounding: paper:unit_004 (chunk_015)
# reference_grounding: paper:unit_005 (chunk_017)

@dataclass
class TaskSetupFactorySpec:
    """
    Configuration spec for paper-derived environment and task setup.
    """
    task_id: str
    dataset_name: str
    subset: Optional[str] = None
    split: str = "train"
    metric: str = "accuracy"
    metadata: Dict[str, Any] = field(default_factory=dict)
    aliases: List[str] = field(default_factory=list)

# reference_grounding: paper:unit_004 (chunk_015)
# reference_grounding: paper:unit_005 (chunk_017)
# reference_grounding: paper:unit_018 (chunk_018)
# reference_grounding: paper:unit_020 (chunk_020)
TASK_REGISTRY = {
    "unit-004": {
        "dataset": "glue",
        "subset": "sst2",
        "metric": "accuracy",
        "aliases": ["main_nlu_env", "apt_consistently_reach higher"],
        "metadata": {"models": ["bert", "roberta", "t5", "llama"]}
    },
    "unit-005": {
        "dataset": "glue",
        "subset": "sst2",
        "metric": "efficiency",
        "aliases": ["efficiency_profiling_env"],
        "metadata": {"metrics": ["memory", "tta", "throughput"]}
    },
    "glue": {
        "dataset": "glue",
        "subset": "sst2",
        "metric": "accuracy",
        "aliases": ["glue_benchmark"],
        "metadata": {"subtasks": ["sst2", "mnli"]}
    },
    "sst2": {
        "dataset": "glue",
        "subset": "sst2",
        "metric": "accuracy",
        "aliases": ["SST2"],
        "metadata": {"task": "sentiment_analysis"}
    },
    "mnli": {
        "dataset": "glue",
        "subset": "mnli",
        "metric": "accuracy",
        "aliases": ["MNLI"],
        "metadata": {"task": "natural_language_inference"}
    },
    "squad": {
        "dataset": "squad_v2",
        "metric": "f1",
        "aliases": ["SQuAD v2.0", "squad_v2"],
        "metadata": {"version": "2.0"}
    },
    "t5": {
        "dataset": "cnn_dailymail",
        "subset": "3.0.0",
        "metric": "rouge",
        "aliases": ["CNN/DailyMail"],
        "metadata": {"model": "t5-base"}
    },
    "llama commonsense": {
        "dataset": "ai2_arc",
        "subset": "ARC-Challenge",
        "metric": "accuracy",
        "aliases": ["open llm leaderboard few-shot"],
        "metadata": {"tasks": ["BoolQ", "PIQA", "SIQA", "HellaSwag", "WinoGrande", "ARC-e", "ARC-c", "OBQA"]}
    },
    "truthfulqa": {
        "dataset": "truthful_qa",
        "subset": "generation",
        "metric": "accuracy",
        "aliases": ["TruthfulQA"],
        "metadata": {"task": "zero_shot"}
    },
    "pruning roberta models targeting similar": {
        "dataset": "glue",
        "subset": "sst2",
        "metric": "accuracy",
        "aliases": ["roberta_ablation"],
        "metadata": {"table": "table 4"}
    },
    "salience notably hurts": {
        "dataset": "glue",
        "subset": "sst2",
        "metric": "accuracy",
        "aliases": ["salience_ablation"],
        "metadata": {"table": "table 4"}
    },
    "fine-tuning will not hurt their": {
        "dataset": "glue",
        "subset": "sst2",
        "metric": "accuracy",
        "aliases": ["ft_baseline"],
        "metadata": {"baseline": "FT"}
    }
}

def _import_torch():
    import torch
    return torch

def _import_transformers():
    import transformers
    return transformers

def _import_datasets():
    import datasets
    return datasets

def check_task_setup_factory_available() -> bool:
    """
    Verifies if the required external backends are available.
    """
    try:
        _import_torch()
        _import_transformers()
        _import_datasets()
        return True
    except ImportError:
        return False

def make_task_setup_factory(task_id: str, **kwargs) -> TaskSetupFactorySpec:
    """
    Creates a TaskSetupFactorySpec from the registry or custom parameters.
    """
    # Resolve alias
    resolved_id = task_id
    for tid, config in TASK_REGISTRY.items():
        if task_id == tid or task_id in config.get("aliases", []):
            resolved_id = tid
            break
    
    if resolved_id in TASK_REGISTRY:
        reg = TASK_REGISTRY[resolved_id]
        return TaskSetupFactorySpec(
            task_id=resolved_id,
            dataset_name=reg["dataset"],
            subset=reg.get("subset"),
            metric=reg["metric"],
            aliases=reg.get("aliases", []),
            metadata=reg.get("metadata", {}),
            **kwargs
        )
    
    raise ValueError(f"Task ID or alias '{task_id}' not found in registry.")

def load_task_setup_factory(spec: TaskSetupFactorySpec) -> Dict[str, Any]:
    """
    Loads the dataset associated with the spec using lazy imports.
    """
    if not check_task_setup_factory_available():
        raise RuntimeError("Required backends (torch, transformers, datasets) are not available.")
    
    datasets = _import_datasets()
    
    try:
        dataset = datasets.load_dataset(spec.dataset_name, spec.subset, split=spec.split)
        return {
            "dataset": dataset,
            "spec": spec,
            "status": "loaded"
        }
    except Exception as e:
        print(f"Warning: Failed to load dataset '{spec.dataset_name}': {e}")
        return {
            "dataset": None,
            "spec": spec,
            "status": "failed",
            "error": str(e)
        }

def prepare_task_setup_factory(loaded_data: Dict[str, Any], tokenizer_name: str) -> Dict[str, Any]:
    """
    Prepares the loaded dataset for training/evaluation (tokenization, etc.).
    """
    if loaded_data["status"] != "loaded" or loaded_data["dataset"] is None:
        return loaded_data

    transformers = _import_transformers()
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    
    dataset = loaded_data["dataset"]
    spec = loaded_data["spec"]
    
    # reference_grounding: paper:unit_004 (chunk_015)
    # Simple preprocessing logic for reproduction
    def tokenize_function(examples):
        if spec.dataset_name == "glue":
            if spec.subset == "sst2":
                return tokenizer(examples["sentence"], padding="max_length", truncation=True)
            elif spec.subset == "mnli":
                return tokenizer(examples["premise"], examples["hypothesis"], padding="max_length", truncation=True)
        elif spec.dataset_name == "squad_v2":
            return tokenizer(examples["question"], examples["context"], padding="max_length", truncation=True)
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    return {
        "dataset": tokenized_dataset,
        "tokenizer": tokenizer,
        "spec": spec,
        "status": "prepared"
    }

# reference_grounding: paper:unit_005 (chunk_017)
def aggregate_measurements(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregates measurements for accuracy, runtime, and training time.
    """
    if not results:
        return {}
    
    aggregated = {
        "accuracy": [],
        "runtime": [],
        "training_time": []
    }
    
    for res in results:
        if "accuracy" in res: aggregated["accuracy"].append(res["accuracy"])
        if "runtime" in res: aggregated["runtime"].append(res["runtime"])
        if "training_time" in res: aggregated["training_time"].append(res["training_time"])
        
    summary = {}
    for k, v in aggregated.items():
        if v:
            summary[f"avg_{k}"] = sum(v) / len(v)
            summary[f"max_{k}"] = max(v)
            summary[f"min_{k}"] = min(v)
            
    return summary

# reference_grounding: paper:unit_018 (chunk_018)
def run_table_2_route(results: List[Dict[str, Any]]):
    """
    Triggers the artifact writer for Table 2 (RoBERTa and T5 pruning).
    """
    from src.reporting.task_setup_factory import write_table_2_artifact
    summary = aggregate_measurements(results)
    write_table_2_artifact(summary)

# reference_grounding: paper:unit_018 (chunk_018)
def run_table_11_route(results: List[Dict[str, Any]]):
    """
    Triggers the artifact writer for Table 11 (Detailed RoBERTa results).
    """
    from src.reporting.task_setup_factory import write_table_11_artifact
    summary = aggregate_measurements(results)
    write_table_11_artifact(summary)

def write_readiness_manifest():
    """
    Writes a readiness manifest for smoke validation.
    """
    manifest = {
        "tasks": list(TASK_REGISTRY.keys()),
        "available": check_task_setup_factory_available(),
        "artifacts": ["results/tables/table_2.csv", "results/tables/table_11.csv"]
    }
    
    out_dir = os.environ.get('PAPERBENCH_REPRO_ARTIFACT_DIR', 'results')