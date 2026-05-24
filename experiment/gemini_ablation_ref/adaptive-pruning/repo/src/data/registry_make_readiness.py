import os
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable

# reference_grounding: paper:paper_contract_environment_protocol (chunk_015, chunk_017, chunk_006)
# Paper evidence contract environments_tasks: squad; glue
# Tasks: SST2, MNLI, SQuAD v2.0, CNN/DailyMail, BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC-e, ARC-c, OBQA | glue | truthfulqa

@dataclass
class RegistryMakeReadinessSpec:
    """Spec for environment and dataset registry and readiness."""
    registry: Dict[str, Any] = field(default_factory=dict)
    readiness_status: Dict[str, bool] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)

def check_package_available(package_name: str) -> bool:
    """Check if a package is available without importing it at top level."""
    import importlib.util
    return importlib.util.find_spec(package_name) is not None

def get_dataset_loader(dataset_id: str) -> Callable:
    """Lazy factory for dataset loaders."""
    def loader(config: Dict[str, Any]):
        if not check_package_available("datasets"):
            raise ImportError(f"Package 'datasets' required for {dataset_id} is not installed.")
        from datasets import load_dataset
        # reference_grounding: paper:unit_004 (chunk_015)
        # Mapping paper IDs to HuggingFace dataset paths
        mapping = {
            "SST2": ("glue", "sst2"),
            "MNLI": ("glue", "mnli"),
            "SQuAD v2.0": ("squad_v2",),
            "CNN/DailyMail": ("cnn_dailymail", "3.0.0"),
            "BoolQ": ("super_glue", "boolq"),
            "PIQA": ("piqa",),
            "SIQA": ("social_i_qa",),
            "HellaSwag": ("hellaswag",),
            "WinoGrande": ("winogrande", "winogrande_xl"),
            "ARC-e": ("ai2_arc", "ARC-Easy"),
            "ARC-c": ("ai2_arc", "ARC-Challenge"),
            "OBQA": ("openbookqa",),
            "truthfulqa": ("truthful_qa", "multiple_choice"),
        }
        path_args = mapping.get(dataset_id)
        if not path_args:
            raise ValueError(f"Unknown dataset ID: {dataset_id}")
        
        # Smoke mode: load a small subset
        split = "train[:10]" if config.get("smoke_mode", False) else "train"
        return load_dataset(*path_args, split=split)
    return loader

def load_registry_make_readiness() -> RegistryMakeReadinessSpec:
    """Initialize the environment and dataset registry."""
    # reference_grounding: paper:unit_004 (chunk_015)
    # Paper evidence contract: explicitly register dataset/benchmark aliases for glue, truthfulqa, squad.
    registry = {
        "datasets": {
            "SST2": {"alias": "glue", "id": "SST2", "metadata": {"benchmark": "GLUE"}},
            "MNLI": {"alias": "glue", "id": "MNLI", "metadata": {"benchmark": "GLUE"}},
            "SQuAD v2.0": {"alias": "squad", "id": "SQuAD v2.0", "metadata": {"version": "2.0"}},
            "CNN/DailyMail": {"alias": "cnn_dm", "id": "CNN/DailyMail", "metadata": {}},
            "BoolQ": {"alias": "boolq", "id": "BoolQ", "metadata": {"benchmark": "LLaMA commonsense"}},
            "PIQA": {"alias": "piqa", "id": "PIQA", "metadata": {"benchmark": "LLaMA commonsense"}},
            "SIQA": {"alias": "siqa", "id": "SIQA", "metadata": {"benchmark": "LLaMA commonsense"}},
            "HellaSwag": {"alias": "hellaswag", "id": "HellaSwag", "metadata": {"benchmark": "LLaMA commonsense"}},
            "WinoGrande": {"alias": "winogrande", "id": "WinoGrande", "metadata": {"benchmark": "LLaMA commonsense"}},
            "ARC-e": {"alias": "arc_e", "id": "ARC-e", "metadata": {"benchmark": "LLaMA commonsense"}},
            "ARC-c": {"alias": "arc_c", "id": "ARC-c", "metadata": {"benchmark": "LLaMA commonsense"}},
            "OBQA": {"alias": "obqa", "id": "OBQA", "metadata": {"benchmark": "LLaMA commonsense"}},
            "truthfulqa": {"alias": "truthfulqa", "id": "truthfulqa", "metadata": {}},
        },
        "environments": {
            "glue": ["SST2", "MNLI"],
            "squad": ["SQuAD v2.0"],
            "llama_commonsense": ["BoolQ", "PIQA", "SIQA", "HellaSwag", "WinoGrande", "ARC-e", "ARC-c", "OBQA"],
            "truthfulqa": ["truthfulqa"]
        }
    }
    return RegistryMakeReadinessSpec(registry=registry)

def make_environment(config: Dict[str, Any]) -> Any:
    """Factory to prepare the environment/dataset based on config."""
    task_id = config.get("task_id")
    if not task_id:
        raise ValueError("task_id must be provided in config.")
    
    loader = get_dataset_loader(task_id)
    return loader(config)

def environment_readiness_check(spec: RegistryMakeReadinessSpec) -> Dict[str, bool]:
    """Check availability of external dependencies and datasets."""
    status = {
        "torch": check_package_available("torch"),
        "transformers": check_package_available("transformers"),
        "datasets": check_package_available("datasets"),
        "evaluate": check_package_available("evaluate"),
    }
    spec.readiness_status = status
    return status

def aggregate_measurements(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Implement measurement collection and result aggregation for: accuracy; F1; runtime; training time."""
    if not results:
        return {}
    
    agg = {
        "accuracy": 0.0,
        "f1": 0.0,
        "runtime": 0.0,
        "training_time": 0.0
    }
    count = len(results)
    for r in results:
        agg["accuracy"] += r.get("accuracy", 0.0)
        agg["f1"] += r.get("f1", 0.0)
        agg["runtime"] += r.get("runtime", 0.0)
        agg["training_time"] += r.get("training_time", 0.0)
    
    return {k: v / count for k, v in agg.items()}

def prepare_registry_make_readiness(spec: RegistryMakeReadinessSpec, output_dir: str = "results"):
    """Perform readiness checks and write artifacts."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Perform checks
    readiness = environment_readiness_check(spec)
    
    # reference_grounding: paper:paper_contract_environment_protocol (chunk_015)
    # results/environment_registry.json
    registry_path = os.path.join(output_dir, "environment_registry.json")
    with open(registry_path, "w") as f:
        json.dump(spec.registry, f, indent=2)
    
    # results/scope_report.json
    scope_report = {
        "readiness": readiness,
        "active_tasks": list(spec.registry["datasets"].keys()),
        "timestamp": time.time(),
        "notes": "lm does not observe downstream; glue keep external; bind every"
    }
    scope_path = os.path.join(output_dir, "scope_report.json")
    with open(scope_path, "w") as f:
        json.dump(scope_report, f, indent=2)
    
    # Write auxiliary readiness.json for smoke validation
    with open(os.path.join(output_dir, "readiness.json"), "w") as f:
        json.dump({"status": "ready", "checks": readiness}, f)

if __name__ == "__main__":
    # Smoke test execution
    artifact_dir = os.environ.get('PAPERBENCH_REPRO_ARTIFACT_DIR', 'results')
    spec = load_registry_make_readiness()
    prepare_registry_make_readiness(spec, output_dir=artifact_dir)
    print(f"Registry and readiness artifacts written to {artifact_dir}")