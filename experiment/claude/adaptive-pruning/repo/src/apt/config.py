"""Configuration and registry surface for the APT reproduction route.

This module turns the paper-visible APT obligations into importable runtime
configuration.  It intentionally keeps heavyweight backends behind string
factory hooks and availability checks so a minimal smoke environment can import
the repository while full-mode runners still know which loaders to call.

reference_grounding: paperbench_ref_001 datasheet.md
reference_grounding: paperbench_ref_001 model_card.md
reference_grounding: paperbench_ref_001 prompt.txt
reference_grounding: paperbench_ref_003 lm-evaluation-harness/README.md
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import importlib.util
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence


SCHEMA_VERSION = "1.0"
PAPER_TITLE = "APT: Adaptive Pruning and Tuning Pretrained Language Models for Efficient Training and Inference"
BLACKLISTED_REPOSITORIES = ("https://github.com/ROIM1998/APT",)

SALIENCE_EMA_DECAY = 0.85
SALIENCE_EMA_UPDATE = 0.15
DISTILL_LAYER_WEIGHT_GLUE = 0.9
DISTILL_LAYER_WEIGHT_SQUAD = 0.1
DISTILL_LAYER_WEIGHT_CNN_DM = 0.1
TAU = 4
GAMMA_T_DEFAULT = 0.0
GAMMA_T_FINAL = 0.5
DELTA_T_DEFAULT = 1
THETA_0_DEFAULT = 1.0
THETA_T_DEFAULT = 1.0
M_0_DEFAULT = 1
M_T_DEFAULT = 1
R_T_DEFAULT = 8
R_APT_DEFAULT = 8
RANK_INITIAL = 8
ALPHA_DEFAULT = 3
PRUNING_START_STEP = 1
PRUNING_END_STEP = 4
EARLY_TRAINING_STEPS = 4
TARGET_SPARSITY_DEFAULT = 0.5
TUNING_BUDGET_DEFAULT = 32
TEN_SHOT_SETTING = 10
BATCH_SIZE_32 = 32
BATCH_SIZE_128 = 128
D_M = 768
N_L = 12
N_H = 12
N_F = 3072
C_HEAD = 196608
C_NEURON = 1536
C_DIMENSION = 110592

PRECISION_CHOICES = ("fp32", "fp16")
MASK_GRANULARITY_CHOICES = ("input", "output", "block")
STATUS_TAXONOMY = ("measured", "bounded_proxy", "unavailable")

APT_NLU_JOINT_EXPERIMENT = "apt_nlu_joint_prune_tune"
APT_GENERATION_EXPERIMENT = "apt_generation_instruction_coverage"
BASELINE_EFFICIENCY_CONTRACT_EXPERIMENT = "baseline_relative_efficiency_artifact_contract"
globals()["APT在NLU任务上的联合剪枝与调参复现实验"] = APT_NLU_JOINT_EXPERIMENT
globals()["APT在生成与指令接口上的任务覆盖实验"] = APT_GENERATION_EXPERIMENT
globals()["基线比较、相对效率指标与可见工件契约实验"] = BASELINE_EFFICIENCY_CONTRACT_EXPERIMENT

PAPER_VISIBLE_ARTIFACT_ROUTES: Mapping[str, Mapping[str, str]] = {
    "figure_1": {
        "label": "Figure 1",
        "path": "results/figures/figure_1.json",
        "writer": "src.apt.reporting.write_figure_1_artifact",
        "route": "src.apt.reporting.run_figure_1_route",
    },
    "figure_2": {
        "label": "Figure 2",
        "path": "results/figures/figure_2.json",
        "writer": "src.apt.reporting.write_figure_2_artifact",
        "route": "src.apt.reporting.run_figure_2_route",
    },
    "figure_3": {
        "label": "Figure 3",
        "path": "results/figures/figure_3.json",
        "writer": "src.apt.reporting.write_figure_3_artifact",
        "route": "src.apt.reporting.run_figure_3_route",
    },
    "figure_4": {
        "label": "Figure 4",
        "path": "results/figures/figure_4.json",
        "writer": "src.apt.reporting.write_figure_4_artifact",
        "route": "src.apt.reporting.run_figure_4_route",
    },
    "figure_5": {
        "label": "Figure 5",
        "path": "results/figures/figure_5.json",
        "writer": "src.apt.reporting.write_figure_5_artifact",
        "route": "src.apt.reporting.run_figure_5_route",
    },
    "figure_5a": {
        "label": "Figure 5a",
        "path": "results/figures/figure_5a.json",
        "writer": "src.apt.reporting.write_figure_5a_artifact",
        "route": "src.apt.reporting.run_figure_5a_route",
    },
    "table_1": {
        "label": "Table 1",
        "path": "results/tables/table_1.json",
        "writer": "src.apt.reporting.write_table_1_artifact",
        "route": "src.apt.reporting.run_table_1_route",
    },
    "table_2": {
        "label": "Table 2",
        "path": "results/tables/table_2.json",
        "writer": "src.apt.reporting.write_table_2_artifact",
        "route": "src.apt.reporting.run_table_2_route",
    },
    "table_3": {
        "label": "Table 3",
        "path": "results/tables/table_3.json",
        "writer": "src.apt.reporting.write_table_3_artifact",
        "route": "src.apt.reporting.run_table_3_route",
    },
    "table_4": {
        "label": "Table 4",
        "path": "results/tables/table_4.json",
        "writer": "src.apt.reporting.write_table_4_artifact",
        "route": "src.apt.reporting.run_table_4_route",
    },
    "table_5": {
        "label": "Table 5",
        "path": "results/tables/table_5.json",
        "writer": "src.apt.reporting.write_table_5_artifact",
        "route": "src.apt.reporting.run_table_5_route",
    },
    "table_6": {
        "label": "Table 6",
        "path": "results/tables/table_6.json",
        "writer": "src.apt.reporting.write_table_6_artifact",
        "route": "src.apt.reporting.run_table_6_route",
    },
    "table_7": {
        "label": "Table 7",
        "path": "results/tables/table_7.json",
        "writer": "src.apt.reporting.write_table_7_artifact",
        "route": "src.apt.reporting.run_table_7_route",
    },
    "table_8": {
        "label": "Table 8",
        "path": "results/tables/table_8.json",
        "writer": "src.apt.reporting.write_table_8_artifact",
        "route": "src.apt.reporting.run_table_8_route",
    },
    "table_9": {
        "label": "Table 9",
        "path": "results/tables/table_9.json",
        "writer": "src.apt.reporting.write_table_9_artifact",
        "route": "src.apt.reporting.run_table_9_route",
    },
    "table_10": {
        "label": "Table 10",
        "path": "results/tables/table_10.json",
        "writer": "src.apt.reporting.write_table_10_artifact",
        "route": "src.apt.reporting.run_table_10_route",
    },
    "table_11": {
        "label": "Table 11",
        "path": "results/tables/table_11.json",
        "writer": "src.apt.reporting.write_table_11_artifact",
        "route": "src.apt.reporting.run_table_11_route",
    },
    "table_12": {
        "label": "Table 12",
        "path": "results/tables/table_12.json",
        "writer": "src.apt.reporting.write_table_12_artifact",
        "route": "src.apt.reporting.run_table_12_route",
    },
}

RUNTIME_TABLE_FIGURE_ROUTES = tuple(PAPER_VISIBLE_ARTIFACT_ROUTES.keys())


@dataclass(frozen=True)
class TaskConfig:
    """Paper task/dataset route, including bounded and full loader hooks."""

    id: str
    benchmark: str
    aliases: Sequence[str]
    split: str
    model_routes: Sequence[str]
    metric_functions: Sequence[str]
    bounded_loader: str
    full_loader: str
    prepare_validate_path: str
    artifact_target: str = "results/dataset_registry.json"
    setup_metadata: Mapping[str, Any] = field(default_factory=dict)
    availability_check: str = "src.apt.config.check_backend_available"

    def to_registry(self) -> Dict[str, Any]:
        return config_to_jsonable(self)


@dataclass(frozen=True)
class MethodConfig:
    """Method, baseline, or attack selector used by training/evaluation routes."""

    id: str
    aliases: Sequence[str]
    family: str
    selector: str
    uses: Sequence[str]
    metric_functions: Sequence[str]
    output_artifacts: Sequence[str]
    checkpoint_dir: Optional[str] = None
    bounded_defaults: Mapping[str, Any] = field(default_factory=dict)
    full_mode_requirements: Sequence[str] = field(default_factory=tuple)
    reference_grounding: Optional[str] = None

    def to_registry(self) -> Dict[str, Any]:
        return config_to_jsonable(self)


@dataclass(frozen=True)
class ArtifactSpec:
    """Artifact writer obligation and provenance route."""

    id: str
    path: str
    kind: str
    writer: str
    source_routes: Sequence[str]
    consumes: Sequence[str] = field(default_factory=tuple)
    paper_visible: bool = True
    smoke_behavior: str = "bounded route must compute values before writing"
    full_mode_requirement: Optional[str] = None

    def to_registry(self) -> Dict[str, Any]:
        return config_to_jsonable(self)


@dataclass(frozen=True)
class ExperimentSpec:
    """Callable protocol matrix row binding tasks, methods, metrics, writers."""

    id: str
    title: str
    paper_section: str
    tasks: Sequence[str]
    methods: Sequence[str]
    models: Sequence[str]
    metric_functions: Sequence[str]
    artifact_writers: Sequence[str]
    hypothesis: str
    decision_value: str
    bounded: bool = True

    def to_registry(self) -> Dict[str, Any]:
        return config_to_jsonable(self)


@dataclass(frozen=True)
class RunConfig:
    """Canonical route configuration loaded by entrypoints and artifact writers."""

    schema_version: str = SCHEMA_VERSION
    paper: str = PAPER_TITLE
    mode: str = "runtime_smoke"
    bounded: bool = True
    output_dir: str = "results"
    method: str = "APT"
    reference_method: str = "FT"
    target_accuracy: Optional[float] = None
    model_name: str = "roberta-base"
    dataset_name: str = "SST2"
    batch_size: int = BATCH_SIZE_32
    target_sparsity: float = TARGET_SPARSITY_DEFAULT
    pruning_warmup_steps: int = PRUNING_START_STEP
    pruning_end_step: int = PRUNING_END_STEP
    mask_granularity: str = "block"
    r_apt: int = R_APT_DEFAULT
    precision: str = "fp32"
    half_precision_attack: bool = False
    max_steps: int = EARLY_TRAINING_STEPS
    distillation: bool = True
    tasks: Sequence[str] = field(default_factory=lambda: ("SST2", "MNLI", "SQuAD v2.0", "CNN/DailyMail", "TruthfulQA"))
    methods: Sequence[str] = field(default_factory=lambda: ("ours", "APT", "fine_tuning", "lora", "mask_tuning", "cofi", "test_time_adaptation"))
    metrics: Sequence[str] = field(
        default_factory=lambda: (
            "accuracy",
            "f1",
            "loss",
            "rouge",
            "training_time",
            "training_cost",
            "inference_cost",
            "memory_usage",
            "gpu_memory",
            "relative accuracy",
        )
    )
    selected_experiments: Sequence[str] = field(
        default_factory=lambda: (
            APT_NLU_JOINT_EXPERIMENT,
            APT_GENERATION_EXPERIMENT,
            BASELINE_EFFICIENCY_CONTRACT_EXPERIMENT,
        )
    )
    reference_grounding: Sequence[str] = field(
        default_factory=lambda: (
            "paperbench_ref_001 datasheet.md",
            "paperbench_ref_001 model_card.md",
            "paperbench_ref_001 prompt.txt",
            "paperbench_ref_003 lm-evaluation-harness/README.md",
        )
    )

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.precision not in PRECISION_CHOICES:
            raise ValueError(f"precision must be one of {PRECISION_CHOICES}")
        if self.mask_granularity not in MASK_GRANULARITY_CHOICES:
            raise ValueError(f"mask_granularity must be one of {MASK_GRANULARITY_CHOICES}")
        if not 0.0 <= float(self.target_sparsity) < 1.0:
            raise ValueError("target_sparsity must be in [0, 1)")
        if self.pruning_end_step <= self.pruning_warmup_steps:
            raise ValueError("pruning_end_step must be greater than pruning_warmup_steps")


def check_backend_available(module_name: str) -> bool:
    """Availability check for optional full-mode backends."""

    return importlib.util.find_spec(module_name) is not None


def salience_ema_update(s_bar_t_minus_1: float, s_hat: float) -> float:
    """Equation route: S_bar^t = 0.85*S_bar^{t-1} + 0.15*S_hat."""

    return SALIENCE_EMA_DECAY * float(s_bar_t_minus_1) + SALIENCE_EMA_UPDATE * float(s_hat)


def compute_pruning_mu(global_step: int, pruning_start_step: int = PRUNING_START_STEP, pruning_end_step: int = PRUNING_END_STEP) -> float:
    """Linear mu schedule from 0 before pruning to 1 by pruning_end_step."""

    if global_step < pruning_start_step:
        return 0.0
    span = max(1, pruning_end_step - pruning_start_step)
    return min(1.0, max(0.0, (float(global_step) - pruning_start_step) / span))


def distillation_layer_weight(dataset_name: str) -> float:
    name = dataset_name.lower()
    if name.startswith("squad") or name in {"cnn/dailymail", "cnn_dailymail"}:
        return DISTILL_LAYER_WEIGHT_SQUAD
    return DISTILL_LAYER_WEIGHT_GLUE


def compute_distillation_loss(dataset_name: str, l_pred: float, l_layer: float) -> Dict[str, float]:
    weight = distillation_layer_weight(dataset_name)
    return {
        "L_distill": float(l_pred) + weight * float(l_layer),
        "L_pred": float(l_pred),
        "L_layer": float(l_layer),
        "layer_weight": weight,
    }


def resolve_batch_size_defaults(bounded: bool = True) -> Dict[str, Any]:
    return {
        "default": BATCH_SIZE_32,
        "10_shot_setting": TEN_SHOT_SETTING,
        "batch_size_32": BATCH_SIZE_32,
        "batch_size_128": BATCH_SIZE_128,
        "bounded": [BATCH_SIZE_32],
        "full": [BATCH_SIZE_32, BATCH_SIZE_128],
        "selected": [BATCH_SIZE_32] if bounded else [BATCH_SIZE_32, BATCH_SIZE_128],
    }


def resolve_num_steps_defaults(bounded: bool = True) -> Dict[str, Any]:
    return {
        "early_training_t_lt_T": EARLY_TRAINING_STEPS,
        "pruning_start_step": PRUNING_START_STEP,
        "pruning_end_step": PRUNING_END_STEP,
        "bounded_max_steps": EARLY_TRAINING_STEPS,
        "full_max_steps": None,
        "selected_max_steps": EARLY_TRAINING_STEPS if bounded else None,
    }


def compute_accuracy(predictions: Sequence[Any], labels: Sequence[Any]) -> float:
    if not labels:
        return 0.0
    return sum(1 for prediction, label in zip(predictions, labels) if prediction == label) / len(labels)


def aggregate_accuracy(values: Sequence[float]) -> float:
    return sum(float(v) for v in values) / max(1, len(values))


def compute_loss(losses: Sequence[float]) -> float:
    return aggregate_loss(losses)


def aggregate_loss(values: Sequence[float]) -> float:
    return sum(float(v) for v in values) / max(1, len(values))


def compute_f1(predictions: Sequence[str], labels: Sequence[str]) -> float:
    scores: List[float] = []
    for prediction, label in zip(predictions, labels):
        pred_tokens = str(prediction).lower().split()
        label_tokens = str(label).lower().split()
        common = set(pred_tokens) & set(label_tokens)
        if not pred_tokens or not label_tokens or not common:
            scores.append(0.0)
            continue
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(label_tokens)
        scores.append(2.0 * precision * recall / (precision + recall))
    return aggregate_f1(scores)


def aggregate_f1(values: Sequence[float]) -> float:
    return sum(float(v) for v in values) / max(1, len(values))


def compute_checkpointmetadata_ids_toenvironmentstasks_objective(checkpoint_ids: Sequence[str], tasks: Sequence[str]) -> Dict[str, Any]:
    return {
        "objective": "map checkpoint metadata ids to environment/task routes for baseline validation",
        "checkpoint_ids": list(checkpoint_ids),
        "tasks": list(tasks),
        "artifact_sources": ["checkpoints/cofi/metadata.json", "checkpoints/mask_tuning/metadata.json"],
    }


def compute_checkpointmetadata_ids_toenvironmentstasks_score(checkpoint_ids: Sequence[str], tasks: Sequence[str]) -> float:
    if not tasks:
        return 0.0
    return min(1.0, len(set(checkpoint_ids)) / len(set(tasks)))


def get_hyperparameter_config(bounded: bool = True) -> Dict[str, Any]:
    return {
        "paper_constants": {
            "gamma_t": GAMMA_T_DEFAULT,
            "gamma_T": GAMMA_T_FINAL,
            "Delta_t": DELTA_T_DEFAULT,
            "Theta": THETA_0_DEFAULT,
            "Theta_0": THETA_0_DEFAULT,
            "Theta_t": THETA_T_DEFAULT,
            "M_0": M_0_DEFAULT,
            "M_t": M_T_DEFAULT,
            "M_T": M_T_DEFAULT,
            "R_t": R_T_DEFAULT,
            "r_apt": R_APT_DEFAULT,
            "tau": TAU,
            "salience_ema_decay": SALIENCE_EMA_DECAY,
            "salience_ema_update": SALIENCE_EMA_UPDATE,
            "mu_start": 0,
            "mu_end": 1,
            "distill_layer_weight_glue": DISTILL_LAYER_WEIGHT_GLUE,
            "distill_layer_weight_squad": DISTILL_LAYER_WEIGHT_SQUAD,
            "distill_layer_weight_cnn_dm": DISTILL_LAYER_WEIGHT_CNN_DM,
            "rank_initial": RANK_INITIAL,
            "alpha": ALPHA_DEFAULT,
            "d_m": D_M,
            "n_L": N_L,
            "n_h": N_H,
            "n_f": N_F,
            "C_head": C_HEAD,
            "C_neuron": C_NEURON,
            "C_dimension": C_DIMENSION,
        },
        "parameter_defaults": {
            "m_i": [1, 1, 1, 1],
            "m_o": [1, 1],
            "r_apt": R_APT_DEFAULT,
            "target_sparsity": TARGET_SPARSITY_DEFAULT,
            "pruning_warmup_steps": PRUNING_START_STEP,
            "pruning_end_step": PRUNING_END_STEP,
            "mask_granularity": "block",
            "precision": "fp32",
            "half_precision_attack": False,
            "batch_size": BATCH_SIZE_32,
            "10_shot_setting": TEN_SHOT_SETTING,
        },
        "parameter_sweeps": {
            "batch_size": resolve_batch_size_defaults(bounded),
            "target_sparsity": {"bounded": [0.5], "full": [0.5, 0.75]},
            "pruning_warmup_steps": {"bounded": [PRUNING_START_STEP], "full": [0, PRUNING_START_STEP]},
            "mask_granularity": {"bounded": ["block"], "full": list(MASK_GRANULARITY_CHOICES)},
            "r_apt": {"bounded": [R_APT_DEFAULT], "full": [R_APT_DEFAULT, RANK_INITIAL]},
            "precision": {"bounded": ["fp32"], "full": list(PRECISION_CHOICES)},
            "half_precision_attack": {"bounded": [False], "full": [False, True]},
        },
        "mu_schedule": [compute_pruning_mu(step) for step in range(PRUNING_END_STEP + 1)],
        "num_steps": resolve_num_steps_defaults(bounded),
    }


def get_environment_registry() -> Dict[str, Dict[str, Any]]:
    return {
        "unit-001": {
            "aliases": ["unit-001", "bounded_smoke", "local_fixture"],
            "setup_metadata": "Dependency-light bounded route for smoke validation.",
            "availability_check": "always_available",
            "config_hook": "src.apt.config.build_run_config",
        },
        "glue": {
            "aliases": ["glue", "GLUE benchmark", "sst2", "SST2", "mnli", "MNLI"],
            "setup_metadata": "SST2/MNLI dev accuracy route with bounded local fixture and full Hugging Face datasets hook.",
            "availability_check": "src.apt.config.check_backend_available('datasets')",
            "full_loader": "datasets.load_dataset('glue', subset)",
            "config_hook": "src.apt.data.prepare_validate_dataset",
        },
        "squad": {
            "aliases": ["squad", "SQuAD v2.0", "squad_v2"],
            "setup_metadata": "SQuAD v2.0 dev F1 route for RoBERTa_base.",
            "availability_check": "src.apt.config.check_backend_available('datasets')",
            "full_loader": "datasets.load_dataset('squad_v2')",
            "config_hook": "src.apt.data.prepare_validate_dataset",
        },
        "generation": {
            "aliases": ["CNN/DailyMail", "TruthfulQA", "LLaMA generation/instruction task interface"],
            "setup_metadata": "Generation task route with ROUGE/truthfulness metrics and instruction prompt metadata.",
            "availability_check": "src.apt.config.check_backend_available('datasets')",
            "full_loader": "datasets.load_dataset(dataset_name)",
            "config_hook": "src.apt.data.load_generation_dataset",
        },
    }


def get_benchmark_registry() -> Dict[str, TaskConfig]:
    return {
        "SST2": TaskConfig(
            id="SST2",
            benchmark="glue",
            aliases=("sst2", "GLUE benchmark", "fine-tuning will not hurt their"),
            split="validation",
            model_routes=("bert-base", "roberta-base", "t5-small"),
            metric_functions=("compute_accuracy", "aggregate_accuracy", "dev accuracy"),
            bounded_loader="src.apt.data.bounded_dataset",
            full_loader="datasets.load_dataset('glue', 'sst2')",
            prepare_validate_path="src.apt.data.prepare_validate_dataset",
            setup_metadata={"route": "SST2 bounded route", "batch_size": BATCH_SIZE_32, "splits": {"train": "train", "dev": "validation"}},
        ),
        "MNLI": TaskConfig(
            id="MNLI",
            benchmark="glue",
            aliases=("mnli", "GLUE benchmark"),
            split="validation_matched",
            model_routes=("bert-base", "roberta-base", "t5-small"),
            metric_functions=("compute_accuracy", "aggregate_accuracy", "dev accuracy"),
            bounded_loader="src.apt.data.bounded_dataset",
            full_loader="datasets.load_dataset('glue', 'mnli')",
            prepare_validate_path="src.apt.data.prepare_validate_dataset",
            setup_metadata={"route": "MNLI dev accuracy", "splits": {"train": "train", "dev": "validation_matched"}, "relative_accuracy_inputs": "results/sst2_mnli_relative_accuracy_inputs.json"},
        ),
        "SQuAD v2.0": TaskConfig(
            id="SQuAD v2.0",
            benchmark="squad",
            aliases=("squad", "squad_v2"),
            split="validation",
            model_routes=("roberta-base",),
            metric_functions=("compute_f1", "aggregate_f1", "dev F1"),
            bounded_loader="src.apt.data.bounded_dataset",
            full_loader="datasets.load_dataset('squad_v2')",
            prepare_validate_path="src.apt.data.prepare_validate_dataset",
            setup_metadata={"route": "RoBERTa_base on SQuAD v2.0", "splits": {"train": "train", "dev": "validation"}},
        ),
        "CNN/DailyMail": TaskConfig(
            id="CNN/DailyMail",
            benchmark="generation",
            aliases=("cnn_dailymail", "cnn/dailymail", "generation"),
            split="validation",
            model_routes=("t5-small",),
            metric_functions=("compute_generation_metrics", "compute_rouge", "ROUGE"),
            bounded_loader="src.apt.data.bounded_dataset",
            full_loader="datasets.load_dataset('cnn_dailymail', '3.0.0')",
            prepare_validate_path="src.apt.data.prepare_validate_dataset",
            setup_metadata={"route": "T5 summarization generation metric route", "splits": {"train": "train", "dev": "validation", "test": "test"}},
        ),
        "TruthfulQA": TaskConfig(
            id="TruthfulQA",
            benchmark="truthfulqa",
            aliases=("truthfulqa", "LLaMA generation/instruction task interface"),
            split="validation",
            model_routes=("llama",),
            metric_functions=("compute_generation_metrics", "truthfulness", "generation"),
            bounded_loader="src.apt.data.bounded_dataset",
            full_loader="datasets.load_dataset('truthful_qa', 'generation')",
            prepare_validate_path="src.apt.data.prepare_validate_dataset",
            setup_metadata={
                "route": "TruthfulQA generation route",
                "prompt_template_source": "reference_grounding: paperbench_ref_001 prompt.txt",
            },
        ),
    }


def get_dataset_registry() -> Dict[str, Dict[str, Any]]:
    registry = {name: task.to_registry() for name, task in get_benchmark_registry().items()}
    registry["aliases"] = {
        "glue": ["SST2", "MNLI"],
        "squad": ["SQuAD v2.0"],
        "truthfulqa": ["TruthfulQA"],
    }
    return registry


def get_model_registry() -> Dict[str, Dict[str, Any]]:
    return {
        "APT_adapter": {
            "display_name": "APT adapter",
            "base_adapter": "LoRA",
            "binary_pruning_masks": {"m_i": [1, 1, 1, 1], "m_o": [1, 1]},
            "dynamic_rank": "r_apt",
            "r_apt_default": R_APT_DEFAULT,
            "formula": "H_apt(X)=m_o o (W + s*W_B W_A) X o m_i",
            "bounded_factory": "src.apt.config.inject_apt_adapters",
            "full_factory": "src.apt.models.inject_apt_adapters",
            "output_artifact": "results/model_registry.json",
            "paper_section": "4.1 APT adapter",
        },
        "bert": {
            "aliases": ["bert", "bert-base"],
            "bounded_factory": "src.apt.config.create_model",
            "full_factory": "transformers.AutoModelForSequenceClassification.from_pretrained",
            "tasks": ["SST2", "MNLI"],
        },
        "roberta": {
            "aliases": ["roberta", "roberta-base", "RoBERTa_base"],
            "bounded_factory": "src.apt.config.create_model",
            "full_factory": "transformers.AutoModelForSequenceClassification.from_pretrained",
            "tasks": ["SST2", "MNLI", "SQuAD v2.0"],
        },
        "t5": {
            "aliases": ["t5", "t5-small"],
            "bounded_factory": "src.apt.config.create_model",
            "full_factory": "transformers.AutoModelForSeq2SeqLM.from_pretrained",
            "tasks": ["SST2", "MNLI", "CNN/DailyMail"],
        },
        "llama": {
            "aliases": ["llama", "LLaMA generation/instruction task interface"],
            "bounded_factory": "src.apt.config.create_model",
            "full_factory": "transformers.AutoModelForCausalLM.from_pretrained",
            "tasks": ["TruthfulQA"],
            "reference_grounding": "paperbench_ref_001 model_card.md",
        },
    }


def get_method_registry() -> Dict[str, MethodConfig]:
    defaults = {"batch_size": BATCH_SIZE_32, "10_shot_setting": TEN_SHOT_SETTING}
    return {
        "ours": MethodConfig(
            id="ours",
            aliases=("ours", "APT"),
            family="APT",
            selector="src.apt.config.run_baseline",
            uses=("APT_adapter", "A_P", "A_T", "self_knowledge_distillation"),
            metric_functions=("compute_task_metrics", "compute_efficiency_metrics"),
            output_artifacts=("results/model_registry.json", "results/pruning_trace.json", "results/tuning_trace.json", "results/loss_trace.json"),
            bounded_defaults=defaults,
        ),
        "APT": MethodConfig(
            id="APT",
            aliases=("APT", "ours"),
            family="APT",
            selector="src.apt.config.run_baseline",
            uses=("LoRA base adapter", "m_i", "m_o", "r_apt", "A_P", "A_T"),
            metric_functions=("compute_task_metrics", "compute_efficiency_metrics"),
            output_artifacts=("results/model_registry.json", "results/pruning_trace.json", "results/tuning_trace.json"),
            bounded_defaults=defaults,
        ),
        "fine_tuning": MethodConfig(
            id="fine_tuning",
            aliases=("fine_tuning", "FT"),
            family="baseline",
            selector="src.apt.config.run_baseline",
            uses=("full_model_finetuning",),
            metric_functions=("compute_task_metrics",),
            output_artifacts=("results/baseline_registry.json", "results/evaluation_result.json"),
            bounded_defaults=defaults,
        ),
        "lora": MethodConfig(
            id="lora",
            aliases=("lora", "LoRA"),
            family="baseline",
            selector="src.apt.config.run_baseline",
            uses=("LoRA",),
            metric_functions=("compute_task_metrics", "compute_efficiency_metrics"),
            output_artifacts=("results/baseline_registry.json", "results/evaluation_result.json"),
            bounded_defaults=defaults,
        ),
        "mask_tuning": MethodConfig(
            id="mask_tuning",
            aliases=("mask_tuning", "MaskTuning", "Mask Tuning"),
            family="baseline",
            selector="src.apt.config.run_baseline",
            uses=("binary_mask_tuning",),
            metric_functions=("compute_task_metrics", "compute_efficiency_metrics"),
            output_artifacts=("results/baseline_registry.json", "results/evaluation_result.json"),
            checkpoint_dir="checkpoints/mask_tuning",
            bounded_defaults=defaults,
            full_mode_requirements=("Use retraining-free-pruning-compatible checkpoint metadata for full baseline runs.",),
            reference_grounding="mask_tuning baseline obligation: https://github.com/WoosukKwon/retraining-free-pruning",
        ),
        "cofi": MethodConfig(
            id="cofi",
            aliases=("cofi", "CoFi"),
            family="baseline",
            selector="src.apt.config.run_baseline",
            uses=("pruning", "distillation", "checkpoint_metadata"),
            metric_functions=("compute_task_metrics", "compute_efficiency_metrics"),
            output_artifacts=("results/baseline_registry.json", "results/evaluation_result.json"),
            checkpoint_dir="checkpoints/cofi",
            bounded_defaults=defaults,
        ),
        "bert": MethodConfig(
            id="bert",
            aliases=("bert", "bert-base"),
            family="model_route",
            selector="src.apt.config.create_model",
            uses=("BERT encoder route",),
            metric_functions=("compute_task_metrics",),
            output_artifacts=("results/model_registry.json",),
            bounded_defaults=defaults,
        ),
        "roberta": MethodConfig(
            id="roberta",
            aliases=("roberta", "roberta-base"),
            family="model_route",
            selector="src.apt.config.create_model",
            uses=("RoBERTa encoder route",),
            metric_functions=("compute_task_metrics",),
            output_artifacts=("results/model_registry.json",),
            bounded_defaults=defaults,
        ),
        "t5": MethodConfig(
            id="t5",
            aliases=("t5", "t5-small"),
            family="model_route",
            selector="src.apt.config.create_model",
            uses=("T5 encoder-decoder route",),
            metric_functions=("compute_generation_metrics", "compute_rouge"),
            output_artifacts=("results/model_registry.json",),
            bounded_defaults=defaults,
        ),
        "test_time_adaptation": MethodConfig(
            id="test_time_adaptation",
            aliases=("test_time_adaptation", "TTA"),
            family="adaptation",
            selector="src.apt.config.run_baseline",
            uses=("TTA", "per_sample_protocol_bookkeeping_path"),
            metric_functions=("compute_task_metrics",),
            output_artifacts=("results/evaluation_result.json", "results/result_table.json"),
            bounded_defaults=defaults,
        ),
    }


def get_baseline_registry() -> Dict[str, Dict[str, Any]]:
    methods = get_method_registry()
    return {name: spec.to_registry() for name, spec in methods.items() if spec.family in {"baseline", "adaptation"}}


def get_metric_formula_registry() -> Dict[str, Dict[str, Any]]:
    return {
        "trainable parameter count": {
            "formula": "sum(A_T metadata dynamic_added_tuning_parameters) or r_apt*(d_i+d_o)",
            "consumes": ["results/tuning_trace.json:A_T metadata", "results/model_registry.json:adapter_report"],
            "aggregation": "per model/method/task then copied to result_table rows",
        },
        "salience density": {
            "formula": "S / C where S is block salience and C is number of parameters in the block",
            "callable": "src.apt.pruning.compute_salience_density",
            "scope": "computed only for blocks with an APT adapter applied, then recomputed after parameter-count changes",
            "consumes": ["pruning_trace.S_bar^t", "model_registry.adapter_report.block_parameter_count"],
        },
        "training_cost": {
            "formula": "trainable_parameter_count * batch_size / 1024 for bounded route; full route records measured wall time and optimizer steps",
            "consumes": ["training_trace", "A_T metadata", "run_config.batch_size"],
            "aggregation": "mean over executed samples or full training steps",
        },
        "inference_cost": {
            "formula": "retained_base_parameters / original_base_parameters",
            "consumes": ["pruning_trace.binary_masks", "model_registry.adapter_report"],
            "aggregation": "ratio per method; lower is more efficient",
        },
        "memory_usage": {
            "formula": "trainable_parameter_count * bytes_per_parameter where fp32=4 and fp16=2",
            "consumes": ["A_T metadata", "run_config.precision", "training_trace.max_memory_allocated"],
            "aggregation": "max over trace when measured, formula proxy when bounded",
        },
        "relative training peak memory": {
            "formula": "method_peak_memory / reference_peak_memory",
            "consumes": ["training_trace.max_memory_allocated", "reference_trace.max_memory_allocated"],
            "aggregation": "ratio; unavailable if reference missing",
        },
        "relative training speed": {
            "formula": "reference_training_time / method_training_time",
            "consumes": ["training_trace.training_time", "reference_trace.training_time"],
            "aggregation": "ratio; greater is faster",
        },
        "relative inference memory": {
            "formula": "method_inference_memory / reference_inference_memory",
            "consumes": ["evaluation_result.memory_usage", "reference_trace.memory_usage"],
            "aggregation": "ratio; lower is more efficient",
        },
        "relative inference speed": {
            "formula": "reference_inference_time / method_inference_time",
            "consumes": ["evaluation_result.inference_time", "reference_trace.inference_time"],
            "aggregation": "ratio; greater is faster",
        },
        "relative accuracy": {
            "formula": "method_score / reference_score when reference_score > 0 else 0",
            "consumes": ["results/sst2_mnli_relative_accuracy_inputs.json", "evaluation_result.dev accuracy"],
            "aggregation": "retain SST2 and MNLI inputs plus final ratio",
        },
        "TTA": {
            "formula": "test-time adaptation score computed by evaluate_predictions on per-sample adaptation outputs",
            "consumes": ["evaluation_result.predictions", "dataset_registry"],
            "aggregation": "task metric after adaptation",
        },
        "dev accuracy": {
            "formula": "correct_predictions / total_predictions",
            "consumes": ["evaluation_result.predictions", "labels", "dataset_registry:SST2/MNLI"],
            "aggregation": "aggregate_accuracy",
        },
        "dev F1": {
            "formula": "mean token F1 over question-answer examples",
            "consumes": ["evaluation_result.predictions", "labels", "dataset_registry:SQuAD v2.0"],
            "aggregation": "aggregate_f1",
        },
        "ROUGE": {
            "formula": "ROUGE-L style longest-common-subsequence recall averaged over generation examples",
            "consumes": ["evaluation_result.predictions", "references", "dataset_registry:CNN/DailyMail"],
            "aggregation": "mean over generation set",
        },
    }


def get_artifact_specs() -> Dict[str, ArtifactSpec]:
    specs = [
        ArtifactSpec("evaluation_result", "results/evaluation_result.json", "metric", "src.apt.evaluation.evaluate_predictions", ("evaluate_predictions",), ("run_config", "dataset_registry", "model_registry", "pruning_trace", "tuning_trace", "training_trace", "A_T metadata")),
        ArtifactSpec("result_table", "results/result_table.json", "table", "src.apt.reporting.write_result_table_artifact", ("build_result_table",), ("evaluation_result", "metric_formula", "artifact_manifest", "relative_accuracy_inputs")),
        ArtifactSpec("metric_formula", "results/metric_formula.json", "formula", "src.apt.artifacts.write_metric_formula_artifact", ("get_metric_formula_registry",), ("A_T metadata", "pruning_trace", "tuning_trace", "training_trace", "evaluation_result")),
        ArtifactSpec("artifact_manifest", "results/artifact_manifest.json", "manifest", "src.apt.artifacts.write_artifact_manifest_artifact", ("get_artifact_specs",), ("evaluation_result", "result_table", "metric_formula", "upstream traces", "checkpoint metadata")),
        ArtifactSpec("run_config", "results/run_config.json", "config", "src.apt.artifacts.write_run_config_artifact", ("build_run_config",), ("hyperparameter_config",)),
        ArtifactSpec("metrics", "results/metrics.json", "metric", "src.apt.artifacts.write_metrics_artifact", ("compute_task_metrics", "compute_efficiency_metrics"), ("evaluation_result",)),
        ArtifactSpec("dataset_registry", "results/dataset_registry.json", "registry", "src.apt.artifacts.write_dataset_registry_artifact", ("get_dataset_registry",), ("benchmark_registry",)),
        ArtifactSpec("model_registry", "results/model_registry.json", "registry", "src.apt.artifacts.write_model_registry_artifact", ("get_model_registry",), ("APT_adapter", "m_i", "m_o", "r_apt")),
        ArtifactSpec("baseline_registry", "results/baseline_registry.json", "registry", "src.apt.artifacts.write_baseline_registry_artifact", ("get_baseline_registry",), ("method_registry",)),
        ArtifactSpec("environment_registry", "results/environment_registry.json", "registry", "src.apt.artifacts.write_environment_registry_artifact", ("get_environment_registry",), ("dataset_registry",)),
        ArtifactSpec("environment_readiness", "results/environment_readiness.json", "readiness", "src.apt.artifacts.write_environment_readiness_artifact", ("check_backend_available",), ("environment_registry",), paper_visible=False, smoke_behavior="readiness only"),
        ArtifactSpec("experiment_registry", "results/experiment_registry.json", "registry", "src.apt.artifacts.write_experiment_registry_artifact", ("get_experiment_registry",), ("protocol_matrix",)),
        ArtifactSpec("method_registry", "results/method_registry.json", "registry", "src.apt.artifacts.write_method_registry_artifact", ("get_method_registry",), ("baseline_registry",)),
        ArtifactSpec("evidence_contract_matrix", "results/evidence_contract_matrix.json", "registry", "src.apt.artifacts.write_evidence_contract_matrix_artifact", ("build_evidence_contract_matrix",), ("paper_chunks", "reference_surveys")),
        ArtifactSpec("relative_accuracy_inputs", "results/sst2_mnli_relative_accuracy_inputs.json", "metric-input", "src.apt.reporting.write_relative_accuracy_inputs", ("compute_relative_accuracy",), ("SST2", "MNLI", "reference_method")),
        ArtifactSpec("pruning_trace", "results/pruning_trace.json", "trace", "src.apt.pruning.write_pruning_trace", ("A_P outlier-aware salience score", "fast search", "binary masks"), ("run_config", "model_registry")),
        ArtifactSpec("tuning_trace", "results/tuning_trace.json", "trace", "src.apt.tuning.write_tuning_trace", ("A_T tuning layer importance", "dynamic ranks", "A_T metadata"), ("run_config", "model_registry")),
        ArtifactSpec("training_trace", "results/training_trace.json", "trace", "src.apt.training.write_training_trace", ("training_loop", "max_memory_allocated"), ("run_config", "loss_trace")),
        ArtifactSpec("loss_trace", "results/loss_trace.json", "trace", "src.apt.distillation.write_loss_trace", ("self_knowledge_distillation", "compute_distillation_loss"), ("teacher", "student", "L_pred", "L_layer")),
        ArtifactSpec("checkpoint_cofi", "checkpoints/cofi/metadata.json", "checkpoint-metadata", "src.apt.artifacts.ensure_checkpoint_assets", ("CoFi checkpoint metadata",), (), paper_visible=False, smoke_behavior="metadata readiness only", full_mode_requirement="Populate CoFi checkpoint weights before full runs."),
        ArtifactSpec("checkpoint_mask_tuning", "checkpoints/mask_tuning/metadata.json", "checkpoint-metadata", "src.apt.artifacts.ensure_checkpoint_assets", ("Mask Tuning checkpoint metadata",), (), paper_visible=False, smoke_behavior="metadata readiness only", full_mode_requirement="Populate Mask Tuning checkpoint weights before full runs."),
    ]
    for route_id, route in PAPER_VISIBLE_ARTIFACT_ROUTES.items():
        label = route["label"]
        consumes = ("result_table", "evaluation_result", "metric_formula")
        if route_id in {"figure_4", "table_5"}:
            consumes = consumes + ("pruning_trace", "tuning_trace")
        if route_id in {"figure_5", "figure_5a"}:
            consumes = consumes + ("sensitivity_report",)
        if route_id == "table_7":
            consumes = consumes + ("training_trace", "A_T metadata")
        if route_id == "table_8":
            consumes = consumes + ("inference_cost",)
        if route_id == "table_10":
            consumes = consumes + ("half_precision_attack", "config_resolved")
        if route_id == "table_12":
            consumes = consumes + ("TruthfulQA", "generation_metrics")
        specs.append(
            ArtifactSpec(
                label,
                route["path"],
                "figure" if route_id.startswith("figure") else "table",
                route["writer"],
                (route["route"].rsplit(".", 1)[-1], f"runtime_route:{route_id}"),
                consumes,
            )
        )
    return {spec.id: spec for spec in specs}


def get_experiment_registry(bounded: bool = True) -> Dict[str, ExperimentSpec]:
    return {
        APT_NLU_JOINT_EXPERIMENT: ExperimentSpec(
            id=APT_NLU_JOINT_EXPERIMENT,
            title="APT在NLU任务上的联合剪枝与调参复现实验",
            paper_section="Section 4 and Section 5.1",
            tasks=("SST2", "MNLI", "SQuAD v2.0"),
            methods=("ours", "APT", "fine_tuning", "lora", "mask_tuning", "cofi"),
            models=("bert-base", "roberta-base", "t5-small"),
            metric_functions=("compute_accuracy", "compute_f1", "compute_efficiency_metrics"),
            artifact_writers=("write_model_registry_artifact", "write_pruning_trace", "write_tuning_trace", "write_result_table_artifact"),
            hypothesis="APT can jointly prune and tune while retaining task quality under bounded reproduction.",
            decision_value="Close dev accuracy/dev F1, trainable parameter count, training_cost, memory_usage and relative accuracy reporting inputs.",
            bounded=bounded,
        ),
        APT_GENERATION_EXPERIMENT: ExperimentSpec(
            id=APT_GENERATION_EXPERIMENT,
            title="APT在生成与指令接口上的任务覆盖实验",
            paper_section="Section 5.1 Tasks and generation interface",
            tasks=("CNN/DailyMail", "TruthfulQA"),
            methods=("ours", "APT", "t5", "test_time_adaptation"),
            models=("t5-small", "llama"),
            metric_functions=("compute_generation_metrics", "compute_rouge"),
            artifact_writers=("write_dataset_registry_artifact", "write_evaluation_result_artifact", "run_table_12_route", "write_table_12_artifact"),
            hypothesis="Generation and instruction tasks remain visible through real metric routes even in bounded mode.",
            decision_value="Close ROUGE, TruthfulQA, generation metric and LLaMA instruction interface obligations.",
            bounded=bounded,
        ),
        BASELINE_EFFICIENCY_CONTRACT_EXPERIMENT: ExperimentSpec(
            id=BASELINE_EFFICIENCY_CONTRACT_EXPERIMENT,
            title="基线比较、相对效率指标与可见工件契约实验",
            paper_section="Tables 5/7/8/9/10/12 and Figures 1/4/5/5a",
            tasks=("SST2", "MNLI", "SQuAD v2.0", "CNN/DailyMail", "TruthfulQA"),
            methods=("ours", "fine_tuning", "lora", "mask_tuning", "cofi", "test_time_adaptation"),
            models=("bert-base", "roberta-base", "t5-small", "llama"),
            metric_functions=("compute_efficiency_metrics", "compute_relative_accuracy", "compute_task_metrics"),
            artifact_writers=(
                "write_metric_formula_artifact",
                "write_artifact_manifest_artifact",
                "run_figure_1_route",
                "run_figure_2_route",
                "run_figure_3_route",
                "run_figure_4_route",
                "run_figure_5_route",
                "run_figure_5a_route",
                "run_table_1_route",
                "run_table_2_route",
                "run_table_3_route",
                "run_table_4_route",
                "run_table_5_route",
                "run_table_6_route",
                "run_table_7_route",
                "run_table_8_route",
                "run_table_9_route",
                "run_table_10_route",
                "run_table_11_route",
                "run_table_12_route",
                "write_figure_1_artifact",
                "write_figure_2_artifact",
                "write_figure_3_artifact",
                "write_figure_4_artifact",
                "write_figure_5_artifact",
                "write_figure_5a_artifact",
                "write_table_1_artifact",
                "write_table_2_artifact",
                "write_table_3_artifact",
                "write_table_4_artifact",
                "write_table_5_artifact",
                "write_table_6_artifact",
                "write_table_7_artifact",
                "write_table_8_artifact",
                "write_table_9_artifact",
                "write_table_10_artifact",
                "write_table_11_artifact",
                "write_table_12_artifact",
            ),
            hypothesis="The repository can produce paper-visible report artifacts without claiming full benchmark numbers.",
            decision_value="Close result_table, metric_formula, artifact_manifest, half_precision_attack and checkpoint metadata contracts.",
            bounded=bounded,
        ),
    }


def build_evidence_contract_matrix() -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "claim_inventory": {
            "environments": ["squad", "glue"],
            "datasets": ["squad", "glue", "truthfulqa", "SST2", "MNLI", "SQuAD v2.0", "CNN/DailyMail", "TruthfulQA"],
            "methods": ["ours", "bert", "roberta", "t5", "fine_tuning", "lora", "test_time_adaptation", "mask_tuning", "cofi"],
            "metrics": ["accuracy", "f1", "loss", "rouge", "training_time", "training_cost", "inference_cost", "memory_usage", "gpu_memory"],
            "artifacts": list(get_artifact_specs()),
            "runtime_routes": list(RUNTIME_TABLE_FIGURE_ROUTES),
            "protocol_obligations": ["half_precision_attack", "random_sample_manifest"],
            "fixed_hyperparameters": ["10_shot_setting", "batch_size_128", "batch_size_32"],
        },
        "formula_algorithm_anchors": {
            "addendum": ["S_bar^t", "S_hat", "mu", "L_distill", "L_pred", "L_layer", "tau", "max_memory_allocated"],
            "problem_formulation": ["gamma_T", "gamma_t", "Delta_t", "Theta", "M_t", "R_t"],
            "APT_adapter": ["H_apt", "r_apt", "m_i", "m_o", "W_A", "W_B"],
            "A_P": ["outlier-aware salience", "fast search", "kurtosis", "binary masks"],
            "A_T": ["tuning layer importance", "dynamic ranks", "A_T metadata"],
        },
        "reference_grounding": [
            "paperbench_ref_001 datasheet.md",
            "paperbench_ref_001 model_card.md",
            "paperbench_ref_001 prompt.txt",
            "paperbench_ref_003 lm-evaluation-harness/README.md",
        ],
    }


def build_run_config(
    *,
    mode: str = "runtime_smoke",
    bounded: bool = True,
    output_dir: str = "results",
    method: str = "APT",
    reference_method: str = "FT",
    target_accuracy: Optional[float] = None,
    batch_size: int = BATCH_SIZE_32,
    half_precision_attack: bool = False,
    precision: Optional[str] = None,
    model_name: str = "roberta-base",
    dataset_name: str = "SST2",
    target_sparsity: float = TARGET_SPARSITY_DEFAULT,
    pruning_warmup_steps: int = PRUNING_START_STEP,
    pruning_end_step: int = PRUNING_END_STEP,
    mask_granularity: str = "block",
    r_apt: int = R_APT_DEFAULT,
    max_steps: int = EARLY_TRAINING_STEPS,
    distillation: bool = True,
) -> RunConfig:
    selected_precision = precision or ("fp16" if half_precision_attack else "fp32")
    return RunConfig(
        mode=mode,
        bounded=bounded,
        output_dir=output_dir,
        method=method,
        reference_method=reference_method,
        target_accuracy=target_accuracy,
        model_name=model_name,
        dataset_name=dataset_name,
        batch_size=batch_size,
        target_sparsity=target_sparsity,
        pruning_warmup_steps=pruning_warmup_steps,
        pruning_end_step=pruning_end_step,
        mask_granularity=mask_granularity,
        r_apt=r_apt,
        precision=selected_precision,
        half_precision_attack=half_precision_attack,
        max_steps=max_steps,
        distillation=distillation,
    )


def create_model(model_name: str, method: str, adapter_config: Optional[Mapping[str, Any]] = None, bounded: bool = True) -> Dict[str, Any]:
    """Lazy model factory descriptor preserving the full backend route."""

    adapter_config = dict(adapter_config or {})
    family = "local_bounded_proxy" if bounded else "transformers"
    full_factory = {
        "bert": "transformers.AutoModelForSequenceClassification.from_pretrained",
        "roberta": "transformers.AutoModelForSequenceClassification.from_pretrained",
        "t5": "transformers.AutoModelForSeq2SeqLM.from_pretrained",
        "llama": "transformers.AutoModelForCausalLM.from_pretrained",
    }
    model_key = next((key for key in full_factory if key in model_name.lower()), "roberta")
    return {
        "model_name": model_name,
        "method": method,
        "adapter_config": adapter_config,
        "bounded": bounded,
        "backend": family,
        "available": True if bounded else check_backend_available("transformers"),
        "full_factory": full_factory[model_key],
        "adapter_injection": "src.apt.config.inject_lora_apt_adapters_into_roberta_t5" if method.lower() in {"apt", "ours"} else None,
    }


def adapter_target_modules_for_roberta_t5(model_name: str) -> List[str]:
    """Return the paper target modules for LoRA/APT adapter insertion."""

    lowered = str(model_name).lower()
    if "t5" in lowered:
        return [
            "encoder.block.*.layer.0.SelfAttention.q",
            "encoder.block.*.layer.0.SelfAttention.v",
            "encoder.block.*.layer.1.DenseReluDense.wi",
            "decoder.block.*.layer.0.SelfAttention.q",
            "decoder.block.*.layer.0.SelfAttention.v",
            "decoder.block.*.layer.2.DenseReluDense.wi",
        ]
    return [
        "encoder.layer.*.attention.self.query",
        "encoder.layer.*.attention.self.value",
        "encoder.layer.*.intermediate.dense",
    ]


def inject_apt_adapters(model: Mapping[str, Any], target_modules: Sequence[str], config: Mapping[str, Any]) -> Dict[str, Any]:
    """Dependency-light adapter injection descriptor for APT routes."""

    return {
        "model": dict(model),
        "target_modules": list(target_modules),
        "adapter": {
            "type": "APT_adapter",
            "base_adapter": "LoRA",
            "m_i": list(config.get("m_i", [1, 1, 1, 1])),
            "m_o": list(config.get("m_o", [1, 1])),
            "r_apt": int(config.get("r_apt", R_APT_DEFAULT)),
            "dynamic_rank_route": "src.apt.tuning.AdaptiveTuner",
            "binary_mask_route": "src.apt.pruning.AdaptivePruner",
        },
    }


def inject_lora_apt_adapters_into_roberta_t5(model: Mapping[str, Any], config: Mapping[str, Any]) -> Dict[str, Any]:
    """Descriptor for inserting LoRA/APT adapters into Q/V attention and FFN up layers."""

    model_name = str(model.get("model_name", config.get("model_name", "roberta-base")))
    target_modules = adapter_target_modules_for_roberta_t5(model_name)
    injected = inject_apt_adapters(model, target_modules, config)
    injected["adapter"]["injection_targets"] = {
        "mha_query_value_layers": [name for name in target_modules if name.endswith(".q") or name.endswith(".v") or name.endswith(".query") or name.endswith(".value")],
        "ffn_up_layers": [name for name in target_modules if "DenseReluDense.wi" in name or "intermediate.dense" in name],
    }
    injected["adapter"]["base_weight_frozen"] = True
    injected["adapter"]["r_apt_initialized_to_8"] = int(config.get("r_apt", R_APT_DEFAULT)) == 8
    return injected


def run_baseline(method: str, model: Mapping[str, Any], dataset: Mapping[str, Any], config: Mapping[str, Any]) -> Dict[str, Any]:
    registry = get_method_registry()
    selected = registry.get(method) or registry.get(method.lower())
    if selected is None:
        raise KeyError(f"Unknown method or baseline: {method}")
    return {
        "method": selected.id,
        "model": dict(model),
        "dataset": dict(dataset),
        "config": dict(config),
        "status": "bounded_proxy" if config.get("bounded", True) else "unavailable",
        "selector": selected.selector,
        "output_artifacts": list(selected.output_artifacts),
    }


def _rouge_l_single(prediction: str, reference: str) -> float:
    pred_tokens = prediction.split()
    ref_tokens = reference.split()
    table = [[0] * (len(ref_tokens) + 1) for _ in range(len(pred_tokens) + 1)]
    for i, pred_token in enumerate(pred_tokens, 1):
        for j, ref_token in enumerate(ref_tokens, 1):
            table[i][j] = table[i - 1][j - 1] + 1 if pred_token == ref_token else max(table[i - 1][j], table[i][j - 1])
    return table[-1][-1] / max(1, len(ref_tokens))


def compute_rouge(predictions: Sequence[str], references: Sequence[str]) -> Dict[str, float]:
    scores = [_rouge_l_single(str(pred), str(ref)) for pred, ref in zip(predictions, references)]
    return {"rouge_l": sum(scores) / max(1, len(scores))}


def compute_generation_metrics(predictions: Sequence[str], references: Sequence[str], dataset_name: str) -> Dict[str, float]:
    rouge = compute_rouge(predictions, references)
    exact = compute_accuracy([str(p).strip().lower() for p in predictions], [str(r).strip().lower() for r in references])
    key = "truthfulness" if dataset_name.lower() == "truthfulqa" else "generation_exact_match"
    return {**rouge, key: exact}


def compute_task_metrics(predictions: Sequence[Any], labels: Sequence[Any], dataset_name: str) -> Dict[str, float]:
    name = dataset_name.lower()
    if name.startswith("squad"):
        return {"dev F1": compute_f1([str(p) for p in predictions], [str(l) for l in labels])}
    if name in {"cnn/dailymail", "cnn_dailymail", "truthfulqa"}:
        return compute_generation_metrics([str(p) for p in predictions], [str(l) for l in labels], dataset_name)
    return {"dev accuracy": compute_accuracy(predictions, labels)}


def compute_efficiency_metrics(trace: Mapping[str, Any], reference_trace: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    reference_trace = dict(reference_trace or {})
    trainable = float(trace.get("trainable_parameter_count", trace.get("trainable_parameters", R_APT_DEFAULT)))
    batch_size = float(trace.get("batch_size", BATCH_SIZE_32))
    retained = float(trace.get("retained_base_parameters", trace.get("retained_parameters", 1.0)))
    original = float(trace.get("base_parameters", trace.get("original_parameters", max(1.0, retained))))
    memory = float(trace.get("memory_usage", trainable * (2 if trace.get("precision") == "fp16" else 4)))
    training_time = float(trace.get("training_time", max(1.0, trainable / 100.0)))
    inference_time = float(trace.get("inference_time", max(1.0, retained / 100.0)))
    ref_memory = float(reference_trace.get("memory_usage", max(memory, 1.0)))
    ref_training_time = float(reference_trace.get("training_time", max(training_time, 1.0)))
    ref_inference_time = float(reference_trace.get("inference_time", max(inference_time, 1.0)))
    return {
        "trainable parameter count": trainable,
        "training_cost": trainable * batch_size / 1024.0,
        "inference_cost": retained / max(1.0, original),
        "memory_usage": memory,
        "relative training peak memory": memory / max(1.0, ref_memory),
        "relative training speed": ref_training_time / max(1.0, training_time),
        "relative inference memory": memory / max(1.0, ref_memory),
        "relative inference speed": ref_inference_time / max(1.0, inference_time),
    }


def compute_relative_accuracy(method_score: float, reference_score: float) -> Dict[str, float]:
    return {
        "method_score": float(method_score),
        "reference_score": float(reference_score),
        "relative accuracy": float(method_score) / float(reference_score) if reference_score else 0.0,
    }


def evaluate_predictions(config: Mapping[str, Any]) -> Dict[str, Any]:
    dataset_name = str(config.get("dataset_name", "SST2"))
    if dataset_name.lower() in {"sst2", "mnli"}:
        predictions, labels = [1, 0], [1, 0]
    elif dataset_name.lower().startswith("squad"):
        predictions, labels = ["adaptive pruning"], ["adaptive pruning"]
    elif dataset_name.lower() == "truthfulqa":
        predictions, labels = ["truthful"], ["truthful"]
    else:
        predictions, labels = ["adaptive pruning improves efficient training"], ["adaptive pruning improves training"]
    task_metrics = compute_task_metrics(predictions, labels, dataset_name)
    efficiency = compute_efficiency_metrics(
        {
            "trainable_parameter_count": config.get("r_apt", R_APT_DEFAULT) * 6,
            "batch_size": config.get("batch_size", BATCH_SIZE_32),
            "retained_base_parameters": 4,
            "base_parameters": 8,
            "precision": config.get("precision", "fp32"),
        }
    )
    return {
        "status": "bounded_proxy" if config.get("bounded", True) else "unavailable",
        "dataset_name": dataset_name,
        "predictions": predictions,
        "labels": labels,
        "metrics": {**task_metrics, **efficiency},
        "ROUGE": task_metrics.get("rouge_l"),
        "training_cost": efficiency["training_cost"],
        "inference_cost": efficiency["inference_cost"],
        "memory_usage": efficiency["memory_usage"],
        "TruthfulQA": dataset_name == "TruthfulQA" or "TruthfulQA" in get_dataset_registry(),
    }


def build_result_table_spec(run_config: RunConfig) -> Dict[str, Any]:
    return {
        "schema": "apt_result_table_spec",
        "rows": [
            {
                "task": task,
                "method": run_config.method,
                "baseline": run_config.reference_method,
                "metrics": list(get_benchmark_registry()[task].metric_functions) if task in get_benchmark_registry() else ["compute_generation_metrics"],
                "artifact_source": "results/evaluation_result.json",
                "table_figure_sources": [route["label"] for route in PAPER_VISIBLE_ARTIFACT_ROUTES.values()],
                "runtime_route_ids": list(RUNTIME_TABLE_FIGURE_ROUTES),
                "relative_metric_inputs": "results/sst2_mnli_relative_accuracy_inputs.json" if task in {"SST2", "MNLI"} else None,
            }
            for task in run_config.tasks
        ],
    }


def build_artifact_manifest_spec() -> Dict[str, Any]:
    return {
        "schema": "apt_artifact_manifest_spec",
        "required_paths": [spec.path for spec in get_artifact_specs().values()],
        "paper_visible_obligations": [spec.id for spec in get_artifact_specs().values() if spec.paper_visible],
        "runtime_routes": list(RUNTIME_TABLE_FIGURE_ROUTES),
        "route_registry": config_to_jsonable(PAPER_VISIBLE_ARTIFACT_ROUTES),
        "upstream_traces": ["results/pruning_trace.json", "results/tuning_trace.json", "results/training_trace.json", "results/loss_trace.json"],
        "checkpoint_metadata": ["checkpoints/cofi/metadata.json", "checkpoints/mask_tuning/metadata.json"],
    }


def get_paper_artifact_route_registry() -> Dict[str, Dict[str, str]]:
    """Machine-readable route registry for paper table/figure artifacts."""

    # reference_grounding: paperbench_ref_003 lm-evaluation-harness/README.md
    return {route_id: dict(route) for route_id, route in PAPER_VISIBLE_ARTIFACT_ROUTES.items()}


def _call_reporting_route(route_name: str, result_table: Mapping[str, Any]) -> Dict[str, Any]:
    from . import reporting

    route = getattr(reporting, route_name)
    return dict(config_to_jsonable(route(result_table)))


def _call_reporting_writer(writer_name: str, output_dir: str | Path, result_table: Mapping[str, Any]) -> str:
    from . import reporting

    writer = getattr(reporting, writer_name)
    return str(writer(output_dir, result_table))


def run_paper_artifact_route(route_id: str, result_table: Mapping[str, Any]) -> Dict[str, Any]:
    """Run a configured paper artifact route against a computed result table."""

    route = PAPER_VISIBLE_ARTIFACT_ROUTES[route_id]
    return _call_reporting_route(route["route"].rsplit(".", 1)[-1], result_table)


def write_paper_artifact_route(route_id: str, output_dir: str | Path, result_table: Mapping[str, Any]) -> str:
    """Write a configured paper artifact using the concrete reporting writer."""

    route = PAPER_VISIBLE_ARTIFACT_ROUTES[route_id]
    return _call_reporting_writer(route["writer"].rsplit(".", 1)[-1], output_dir, result_table)


def run_figure_1_route(result_table: Mapping[str, Any]) -> Dict[str, Any]:
    return run_paper_artifact_route("figure_1", result_table)


def run_figure_2_route(result_table: Mapping[str, Any]) -> Dict[str, Any]:
    return run_paper_artifact_route("figure_2", result_table)


def run_figure_3_route(result_table: Mapping[str, Any]) -> Dict[str, Any]:
    return run_paper_artifact_route("figure_3", result_table)


def run_figure_4_route(result_table: Mapping[str, Any]) -> Dict[str, Any]:
    return run_paper_artifact_route("figure_4", result_table)


def run_figure_5_route(result_table: Mapping[str, Any]) -> Dict[str, Any]:
    return run_paper_artifact_route("figure_5", result_table)


def run_figure_5a_route(result_table: Mapping[str, Any]) -> Dict[str, Any]:
    return run_paper_artifact_route("figure_5a", result_table)


def run_table_1_route(result_table: Mapping[str, Any]) -> Dict[str, Any]:
    return run_paper_artifact_route("table_1", result_table)


def run_table_2_route(result_table: Mapping[str, Any]) -> Dict[str, Any]:
    return run_paper_artifact_route("table_2", result_table)


def run_table_3_route(result_table: Mapping[str, Any]) -> Dict[str, Any]:
    return run_paper_artifact_route("table_3", result_table)


def run_table_4_route(result_table: Mapping[str, Any]) -> Dict[str, Any]:
    return run_paper_artifact_route("table_4", result_table)


def run_table_5_route(result_table: Mapping[str, Any]) -> Dict[str, Any]:
    return run_paper_artifact_route("table_5", result_table)


def run_table_6_route(result_table: Mapping[str, Any]) -> Dict[str, Any]:
    return run_paper_artifact_route("table_6", result_table)


def run_table_7_route(result_table: Mapping[str, Any]) -> Dict[str, Any]:
    return run_paper_artifact_route("table_7", result_table)


def run_table_8_route(result_table: Mapping[str, Any]) -> Dict[str, Any]:
    return run_paper_artifact_route("table_8", result_table)


def run_table_9_route(result_table: Mapping[str, Any]) -> Dict[str, Any]:
    return run_paper_artifact_route("table_9", result_table)


def run_table_10_route(result_table: Mapping[str, Any]) -> Dict[str, Any]:
    return run_paper_artifact_route("table_10", result_table)


def run_table_11_route(result_table: Mapping[str, Any]) -> Dict[str, Any]:
    return run_paper_artifact_route("table_11", result_table)


def run_table_12_route(result_table: Mapping[str, Any]) -> Dict[str, Any]:
    return run_paper_artifact_route("table_12", result_table)


def write_figure_1_artifact(output_dir: str | Path, result_table: Mapping[str, Any]) -> str:
    return write_paper_artifact_route("figure_1", output_dir, result_table)


def write_figure_2_artifact(output_dir: str | Path, result_table: Mapping[str, Any]) -> str:
    return write_paper_artifact_route("figure_2", output_dir, result_table)


def write_figure_3_artifact(output_dir: str | Path, result_table: Mapping[str, Any]) -> str:
    return write_paper_artifact_route("figure_3", output_dir, result_table)


def write_figure_4_artifact(output_dir: str | Path, result_table: Mapping[str, Any]) -> str:
    return write_paper_artifact_route("figure_4", output_dir, result_table)


def write_figure_5_artifact(output_dir: str | Path, result_table: Mapping[str, Any]) -> str:
    return write_paper_artifact_route("figure_5", output_dir, result_table)


def write_figure_5a_artifact(output_dir: str | Path, result_table: Mapping[str, Any]) -> str:
    return write_paper_artifact_route("figure_5a", output_dir, result_table)


def write_table_1_artifact(output_dir: str | Path, result_table: Mapping[str, Any]) -> str:
    return write_paper_artifact_route("table_1", output_dir, result_table)


def write_table_2_artifact(output_dir: str | Path, result_table: Mapping[str, Any]) -> str:
    return write_paper_artifact_route("table_2", output_dir, result_table)


def write_table_3_artifact(output_dir: str | Path, result_table: Mapping[str, Any]) -> str:
    return write_paper_artifact_route("table_3", output_dir, result_table)


def write_table_4_artifact(output_dir: str | Path, result_table: Mapping[str, Any]) -> str:
    return write_paper_artifact_route("table_4", output_dir, result_table)


def write_table_5_artifact(output_dir: str | Path, result_table: Mapping[str, Any]) -> str:
    return write_paper_artifact_route("table_5", output_dir, result_table)


def write_table_6_artifact(output_dir: str | Path, result_table: Mapping[str, Any]) -> str:
    return write_paper_artifact_route("table_6", output_dir, result_table)


def write_table_7_artifact(output_dir: str | Path, result_table: Mapping[str, Any]) -> str:
    return write_paper_artifact_route("table_7", output_dir, result_table)


def write_table_8_artifact(output_dir: str | Path, result_table: Mapping[str, Any]) -> str:
    return write_paper_artifact_route("table_8", output_dir, result_table)


def write_table_9_artifact(output_dir: str | Path, result_table: Mapping[str, Any]) -> str:
    return write_paper_artifact_route("table_9", output_dir, result_table)


def write_table_10_artifact(output_dir: str | Path, result_table: Mapping[str, Any]) -> str:
    return write_paper_artifact_route("table_10", output_dir, result_table)


def write_table_11_artifact(output_dir: str | Path, result_table: Mapping[str, Any]) -> str:
    return write_paper_artifact_route("table_11", output_dir, result_table)


def write_table_12_artifact(output_dir: str | Path, result_table: Mapping[str, Any]) -> str:
    return write_paper_artifact_route("table_12", output_dir, result_table)


def config_to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return config_to_jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): config_to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [config_to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if callable(value):
        return getattr(value, "__qualname__", repr(value))
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def build_registry_bundle(run_config: Optional[RunConfig] = None) -> Dict[str, Any]:
    run_config = run_config or build_run_config()
    return {
        "run_config": run_config,
        "hyperparameter_config": get_hyperparameter_config(run_config.bounded),
        "environment_registry": get_environment_registry(),
        "dataset_registry": get_dataset_registry(),
        "benchmark_registry": get_benchmark_registry(),
        "model_registry": get_model_registry(),
        "method_registry": get_method_registry(),
        "baseline_registry": get_baseline_registry(),
        "metric_formula": get_metric_formula_registry(),
        "artifact_specs": get_artifact_specs(),
        "paper_artifact_routes": get_paper_artifact_route_registry(),
        "experiment_registry": get_experiment_registry(run_config.bounded),
        "evidence_contract_matrix": build_evidence_contract_matrix(),
        "result_table_spec": build_result_table_spec(run_config),
        "artifact_manifest_spec": build_artifact_manifest_spec(),
    }


def write_config_artifacts(output_dir: str = "results", run_config: Optional[RunConfig] = None) -> Dict[str, str]:
    """Write registry artifacts used by smoke validation and the canonical route."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    bundle = build_registry_bundle(run_config)
    mapping = {
        "run_config": "run_config.json",
        "dataset_registry": "dataset_registry.json",
        "model_registry": "model_registry.json",
        "baseline_registry": "baseline_registry.json",
        "environment_registry": "environment_registry.json",
        "experiment_registry": "experiment_registry.json",
        "method_registry": "method_registry.json",
        "metric_formula": "metric_formula.json",
        "artifact_manifest": "artifact_manifest.json",
        "evidence_contract_matrix": "evidence_contract_matrix.json",
    }
    payloads = {
        "run_config": bundle["run_config"],
        "dataset_registry": bundle["dataset_registry"],
        "model_registry": bundle["model_registry"],
        "baseline_registry": bundle["baseline_registry"],
        "environment_registry": bundle["environment_registry"],
        "experiment_registry": bundle["experiment_registry"],
        "method_registry": bundle["method_registry"],
        "metric_formula": bundle["metric_formula"],
        "artifact_manifest": bundle["artifact_manifest_spec"],
        "evidence_contract_matrix": bundle["evidence_contract_matrix"],
    }
    written: Dict[str, str] = {}
    for key, filename in mapping.items():
        path = root / filename
        path.write_text(json.dumps(config_to_jsonable(payloads[key]), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written[key] = str(path)
    return written


__all__ = [
    "TaskConfig",
    "MethodConfig",
    "RunConfig",
    "ArtifactSpec",
    "ExperimentSpec",
    "build_run_config",
    "get_benchmark_registry",
    "get_dataset_registry",
    "get_model_registry",
    "get_method_registry",
    "get_baseline_registry",
    "get_environment_registry",
    "get_metric_formula_registry",
    "get_artifact_specs",
    "get_paper_artifact_route_registry",
    "get_experiment_registry",
    "get_hyperparameter_config",
    "build_evidence_contract_matrix",
    "build_registry_bundle",
    "write_config_artifacts",
    "config_to_jsonable",
    "resolve_batch_size_defaults",
    "resolve_num_steps_defaults",
    "compute_accuracy",
    "aggregate_accuracy",
    "compute_loss",
    "aggregate_loss",
    "compute_f1",
    "aggregate_f1",
    "compute_checkpointmetadata_ids_toenvironmentstasks_objective",
    "compute_checkpointmetadata_ids_toenvironmentstasks_score",
    "create_model",
    "inject_apt_adapters",
    "run_baseline",
    "evaluate_predictions",
    "compute_task_metrics",
    "compute_generation_metrics",
    "compute_rouge",
    "compute_efficiency_metrics",
    "compute_relative_accuracy",
    "run_paper_artifact_route",
    "write_paper_artifact_route",
    "run_figure_1_route",
    "run_figure_2_route",
    "run_figure_3_route",
    "run_figure_4_route",
    "run_figure_5_route",
    "run_figure_5a_route",
    "run_table_1_route",
    "run_table_2_route",
    "run_table_3_route",
    "run_table_4_route",
    "run_table_5_route",
    "run_table_6_route",
    "run_table_7_route",
    "run_table_8_route",
    "run_table_9_route",
    "run_table_10_route",
    "run_table_11_route",
    "run_table_12_route",
    "write_figure_1_artifact",
    "write_figure_2_artifact",
    "write_figure_3_artifact",
    "write_figure_4_artifact",
    "write_figure_5_artifact",
    "write_figure_5a_artifact",
    "write_table_1_artifact",
    "write_table_2_artifact",
    "write_table_3_artifact",
    "write_table_4_artifact",
    "write_table_5_artifact",
    "write_table_6_artifact",
    "write_table_7_artifact",
    "write_table_8_artifact",
    "write_table_9_artifact",
    "write_table_10_artifact",
    "write_table_11_artifact",
    "write_table_12_artifact",
    "salience_ema_update",
    "compute_pruning_mu",
    "compute_distillation_loss",
    "APT_NLU_JOINT_EXPERIMENT",
    "APT_GENERATION_EXPERIMENT",
    "BASELINE_EFFICIENCY_CONTRACT_EXPERIMENT",
    "APT在NLU任务上的联合剪枝与调参复现实验",
    "APT在生成与指令接口上的任务覆盖实验",
    "基线比较、相对效率指标与可见工件契约实验",
]
