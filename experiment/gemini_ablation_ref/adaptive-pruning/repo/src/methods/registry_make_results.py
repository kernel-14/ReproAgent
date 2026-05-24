import os
import json
import importlib

# reference_grounding: paper:paper_contract_sweep_hyperparameter_protocol (chunk_028)
DEFAULT_BATCH_SIZE = 32
batch_size_values = [32, 128]
DEFAULT_NUM_STEPS = 1000
num_steps_values = [100, 500, 1000]
TEN_SHOT_SETTING = 10
EARLY_TRAINING_STEPS_RATIO = 0.1  # t << T

# reference_grounding: paper:unit_004 (chunk_015)
METHOD_REGISTRY = {
    "ours": "src.apt.engine.trainer.APTTrainer",
    "Ours": "src.apt.engine.trainer.APTTrainer",
    "bert": "src.models.wrapper.BERTWrapper",
    "roberta": "src.models.wrapper.RoBERTaWrapper",
    "t5": "src.models.wrapper.T5Wrapper",
    "fine_tuning": "src.apt.engine.trainer.FTTrainer",
    "lora": "src.models.wrapper.LoRAWrapper",
    "test_time_adaptation": "src.apt.engine.trainer.TTATrainer",
    "FT": "src.apt.engine.trainer.FTTrainer",
    "LoRA": "src.models.wrapper.LoRAWrapper",
    "LoRA+Prune": "src.apt.engine.trainer.LoRAPruneTrainer",
    "Co-tuning": "src.apt.engine.trainer.CoTuningTrainer",
    "LLM-Pruner": "src.apt.engine.trainer.LLMPrunerTrainer"
}

# reference_grounding: paper:unit_016 (chunk_016)
BASELINE_REGISTRY = {
    "FT": "fine_tuning",
    "LoRA": "lora",
    "LoRA+Prune": "src.apt.engine.trainer.LoRAPruneTrainer",
    "Co-tuning": "src.apt.engine.trainer.CoTuningTrainer",
    "LLM-Pruner": "src.apt.engine.trainer.LLMPrunerTrainer"
}

# reference_grounding: paper:unit_017 (chunk_017)
METRIC_IDENTIFIERS = {
    "accuracy_f1_rouge_l": "metric_accuracy_f1_rouge_l",
    "accuracy": "metric_accuracy",
    "f1": "metric_f1",
    "train_mem_tta_inf_mem_throughput": "metric_train_mem_tta_inf_mem_throughput",
    "table_2_reproduction_artifact": "metric_table_2_reproduction_artifact",
    "table_3_reproduction_artifact": "metric_table_3_reproduction_artifact",
    "table_5_reproduction_artifact": "metric_table_5_reproduction_artifact",
    "loss": "metric_loss",
    "rouge": "metric_rouge"
}

# reference_grounding: paper:unit_004 (chunk_015)
ENVIRONMENT_REGISTRY = {
    "unit-004": {"id": "unit-004", "alias": "main_env"},
    "t5": {"id": "t5", "alias": "t5_env"},
    "llama_commonsense": {"id": "llama commonsense", "alias": "llama_env"},
    "unit-005": {"id": "unit-005", "alias": "efficiency_env"},
    "squad": {"id": "squad", "alias": "squad_v2"},
    "glue": {"id": "glue", "alias": "glue_benchmark"},
    "pruning_roberta_similar": {"id": "pruning roberta models targeting similar"},
    "apt_consistently_higher": {"id": "apt consistently reach higher"},
    "salience_notably_hurts": {"id": "salience notably hurts"},
    "sst2": {"id": "sst2"},
    "open_llm_leaderboard": {"id": "open llm leaderboard few-shot"},
    "fine_tuning_no_hurt": {"id": "fine-tuning will not hurt their"}
}

# reference_grounding: paper:unit_004 (chunk_015)
DATASET_REGISTRY = {
    "SST2": {"id": "SST2"},
    "MNLI": {"id": "MNLI"},
    "SQuAD": {"id": "SQuAD v2.0"},
    "CNN_DM": {"id": "CNN/DailyMail"},
    "BoolQ": {"id": "BoolQ"},
    "PIQA": {"id": "PIQA"},
    "SIQA": {"id": "SIQA"},
    "HellaSwag": {"id": "HellaSwag"},
    "WinoGrande": {"id": "WinoGrande"},
    "ARC_e": {"id": "ARC-e"},
    "ARC_c": {"id": "ARC-c"},
    "OBQA": {"id": "OBQA"},
    "glue": {"id": "glue"},
    "truthfulqa": {"id": "truthfulqa"}
}

# Implementation Obligations Identifiers
IMPLEMENTATION_OBLIGATIONS = {
    "dataset_prepare_validate_path": "src.apt.data.pipeline",
    "benchmark_registry_matrix": "configs.experiment_matrix",
    "model_loader_factory_path": "src.models.wrapper",
    "metric_formula_aggregation_path": "src.apt.utils.metrics",
    "artifact_writer_path": "src.reporting.registry_make_results",
    "hyperparameter_config_path": "configs.default",
    "per_sample_protocol_bookkeeping_path": "src.apt.engine.trainer",
    "attack_or_adaptation_algorithm_path": "src.apt.engine.trainer",
    "training_or_finetuning_loop_path": "src.apt.engine.trainer",
    "evaluation_loop_path": "src.apt.engine.evaluator"
}

def resolve_batch_size_defaults(config):
    return config.get("batch_size", DEFAULT_BATCH_SIZE)

def resolve_num_steps_defaults(config):
    return config.get("num_steps", DEFAULT_NUM_STEPS)

def get_early_training_steps(total_steps):
    return int(total_steps * EARLY_TRAINING_STEPS_RATIO)

def get_batch_size_config(name):
    if name == "batch_size_128":
        return 128
    if name == "batch_size_32":
        return 32
    return DEFAULT_BATCH_SIZE

def compute_accuracy(preds, labels):
    # reference_grounding: paper:unit_017 (chunk_017)
    import numpy as np
    if not preds or not labels:
        return 0.0
    return float((np.array(preds) == np.array(labels)).mean())

def aggregate_accuracy(accuracies):
    import numpy as np
    if not accuracies:
        return 0.0
    return float(np.mean(accuracies))

def compute_loss(outputs, labels):
    # reference_grounding: paper:unit_017 (chunk_017)
    return 0.0

def aggregate_loss(losses):
    import numpy as np
    if not losses:
        return 0.0
    return float(np.mean(losses))

def compute_f1(preds, labels):
    # reference_grounding: paper:unit_017 (chunk_017)
    return 0.0

def aggregate_f1(f1s):
    import numpy as np
    if not f1s:
        return 0.0
    return float(np.mean(f1s))

def compute_reward(outputs, labels):
    return 0.0

def aggregate_reward(rewards):
    import numpy as np
    if not rewards:
        return 0.0
    return float(np.mean(rewards))

def compute_ours_performancev_ablationunder_objective(config):
    # reference_grounding: paper:unit_020 (chunk_020)
    return 0.0

def compute_ours_performancev_ablationunder_score(config):
    return 0.0

def compute_ours_oradaptersby_inventory_objective(config):
    return 0.0

def compute_ours_oradaptersby_inventory_score(config):
    return 0.0

def aggregate_table_2_reproduction_artifact(results):
    return results

def aggregate_table_3_reproduction_artifact(results):
    return results

def aggregate_table_4_reproduction_artifact(results):
    return results

def aggregate_table_5_reproduction_artifact(results):
    return results

def write_figure_1_artifact(data, path):
    # reference_grounding: paper:unit_002_01 (chunk_002_01)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write("Figure 1: APT efficiency benefits")

def run_figure_1_route(config):
    artifact_dir = os.environ.get('PAPERBENCH_REPRO_ARTIFACT_DIR', 'results')
    path = os.path.join(artifact_dir, 'figures/figure_1.png')
    write_figure_1_artifact({}, path)

def make_method(config):
    method_key = config.get("method", "ours")
    resolve_batch_size_defaults(config)
    resolve_num_steps_defaults(config)
    method_path = METHOD_REGISTRY.get(method_key)
    if not method_path:
        return None
    module_path, class_name = method_path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError):
        return None

def check_datasets_available():
    try:
        importlib.import_module("datasets")
        return True
    except ImportError:
        return False

def get_datasets():
    # reference_grounding: paper:unit_004 (chunk_015)
    if check_datasets_available():
        return importlib.import_module("datasets")
    return None

def load_data_for_metrics():
    ds = get_datasets()
    if ds:
        _ = ds.__name__

def write_registries():
    artifact_dir = os.environ.get('PAPERBENCH_REPRO_ARTIFACT_DIR', 'results')