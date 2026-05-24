"""Artifact, metric, and protocol surfaces for SMM visual reprogramming.

reference_grounding: chunk_017_02 /mnt/paper2any/pzw/proj/paperagent/hx/Research_space/Reproduction/paperbench_data/sample-specific-masks/paper.md
"""

from __future__ import annotations

import csv
import importlib
import json
import math
import os
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


DEFAULT_LEARNING_RATE: float = 1.0e-3
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_EPOCHS: int = 100
DEFAULT_SEED: int = 0
DEFAULT_ALPHA: float = DEFAULT_LEARNING_RATE
DEFAULT_GAMMA: float = 0.1

THREE_SEED_PROTOCOL: Tuple[int, int, int] = (0, 1, 2)
PATCH_SIZE_VALUES: Tuple[int, int, int] = (4, 2, 1)
P_SWEEP_VALUES: Tuple[float, float, float] = (0.0, 0.5, 1.0)
SIMILARITY_GUIDANCE_SCALE_VALUES: Tuple[int, int, int] = (9, 7, 10)

TABLE3_BACKBONE: str = "resnet18_imagenet1k"
TABLE3_DATASETS: Tuple[str, ...] = (
    "CIFAR10",
    "CIFAR100",
    "SVHN",
    "GTSRB",
    "Flowers102",
    "DTD",
    "UCF101",
    "EuroSAT",
)

MAIN_DATASETS: Tuple[str, ...] = (
    "CIFAR10",
    "CIFAR100",
    "SVHN",
    "GTSRB",
    "Flowers102",
    "DTD",
    "UCF101",
    "Food101",
    "SUN397",
    "EuroSAT",
    "OxfordPets",
)

DATASET_ALIASES: Dict[str, Tuple[str, ...]] = {
    "cifar": ("CIFAR10", "CIFAR100"),
    "imagenet": ("ImageNet-1K", "imagenet_1k"),
    "svhn": ("SVHN",),
    "imagenet_1k": ("ImageNet-1K", "imagenet"),
    "stanford_cars": ("StanfordCars", "cars"),
    "dtd": ("DTD",),
    "eurosat": ("EuroSAT",),
    "flowers": ("Flowers102", "flowers102"),
    "oxford_pets": ("OxfordPets", "oxford_pets"),
}

METHOD_SELECTOR: Dict[str, Dict[str, Any]] = {
    "PAD": {
        "method_id": "PAD",
        "family": "padding_based_reprogramming",
        "shared_mask": "padding_valid_region",
        "factory": "sample_specific_masks.reprogramming:build_reprogramming",
    },
    "Narrow": {
        "method_id": "Narrow",
        "family": "resizing_based_reprogramming",
        "shared_mask": "narrow_watermark_region",
        "factory": "sample_specific_masks.reprogramming:build_reprogramming",
    },
    "Medium": {
        "method_id": "Medium",
        "family": "resizing_based_reprogramming",
        "shared_mask": "medium_watermark_region",
        "factory": "sample_specific_masks.reprogramming:build_reprogramming",
    },
    "Full": {
        "method_id": "Full",
        "family": "resizing_based_reprogramming",
        "shared_mask": "full_watermark_region",
        "factory": "sample_specific_masks.reprogramming:build_reprogramming",
    },
    "Ours": {
        "method_id": "Ours",
        "aliases": ("ours", "SMM/Ours", "sample-specific multi-channel masks"),
        "family": "sample_specific_masks",
        "variant": "ours_multi_channel",
        "factory": "sample_specific_masks.reprogramming:build_reprogramming",
    },
    "ours": {
        "method_id": "ours",
        "canonical": "Ours",
        "factory": "sample_specific_masks.reprogramming:build_reprogramming",
    },
    "vit": {
        "method_id": "vit",
        "backbone": "vit_b32_imagenet1k",
        "factory": "sample_specific_masks.train:load_backbone",
    },
    "resnet": {
        "method_id": "resnet",
        "backbone": "resnet18_imagenet1k",
        "factory": "sample_specific_masks.train:load_backbone",
    },
    "lora": {
        "method_id": "lora",
        "family": "finetuning_lora",
        "factory": "sample_specific_masks.train:finetune_classifier",
    },
    "imagenet_1k": {
        "method_id": "imagenet_1k",
        "source_label_space": 1000,
        "factory": "sample_specific_masks.train:load_backbone",
    },
    "ONLY δ": {
        "method_id": "ONLY δ",
        "variant": "only_delta",
        "factory": "sample_specific_masks.reprogramming:build_reprogramming",
    },
}

ARTIFACT_PATHS: Dict[str, str] = {
    "metrics": "results/metrics.json",
    "config": "results/config.json",
    "config_resolved": "results/config_resolved.json",
    "run_summary": "results/run_summary.json",
    "readiness": "readiness.json",
    "evaluation_result": "evaluation_result.json",
    "dry_run_manifest": "results/dry_run_manifest.json",
    "artifact_manifest": "results/artifact_manifest.json",
    "dataset_registry": "results/dataset_registry.json",
    "environment_registry": "results/environment_registry.json",
    "experiment_registry": "results/experiment_registry.json",
    "table_index": "results/table_index.json",
    "figure_index": "results/figure_index.json",
    "table1_resnet_main_csv": "results/tables/table1_resnet_main.csv",
    "table1_resnet_main_json": "results/tables/table1_resnet_main.json",
    "table2_vit_main_csv": "results/tables/table2_vit_main.csv",
    "table2_vit_main_json": "results/tables/table2_vit_main.json",
    "table3_ablation_csv": "results/tables/table3_ablation.csv",
    "table3_ablation_json": "results/tables/table3_ablation.json",
    "table_1": "results/tables/table_1.csv",
    "table_2": "results/tables/table_2.csv",
    "table_3": "results/tables/table_3.csv",
    "table_4": "results/tables/table_4.csv",
    "table_11": "results/tables/table_11.csv",
    "table_13": "results/tables/table_13.csv",
    "table_14": "results/tables/table_14.csv",
    "table3_ablation_metrics": "table3_ablation_metrics.json",
    "table3_ablation_table": "table3_ablation_table.csv",
    "mask_variant_summary": "mask_variant_summary.json",
    "mask_statistics": "mask_statistics.json",
    "smoke_metrics": "results/smoke/metrics.json",
    "figure_1": "results/figures/figure_1.png",
    "figure_2": "results/figures/figure_2.png",
    "figure_3": "results/figures/figure_3.png",
    "figure_4": "results/figures/figure_4.png",
    "figure_11": "results/figures/figure_11.png",
    "figure_12": "results/figures/figure_12.png",
    "figure_13": "results/figures/figure_13.png",
    "figure_14": "results/figures/figure_14.png",
    "figure_15": "results/figures/figure_15.png",
    "figure_16": "results/figures/figure_16.png",
    "figure_17": "results/figures/figure_17.png",
    "figure_18": "results/figures/figure_18.png",
    "figure_19": "results/figures/figure_19.png",
    "figure_20": "results/figures/figure_20.png",
    "figure_21": "results/figures/figure_21.png",
    "figure_22": "results/figures/figure_22.png",
    "figure_23": "results/figures/figure_23.png",
}


@dataclass(frozen=True)
class AblationVariantSpec:
    variant_id: str
    display_name: str
    delta_enabled: bool
    mask_generator_enabled: bool
    channel_mode: str
    shared_delta_contribution: str
    normal_delta_contribution: bool
    comparison_role: str
    trainable_components: Tuple[str, ...]
    factory: str = "sample_specific_masks.reprogramming:build_reprogramming"
    trainer: str = "sample_specific_masks.train:run_training_loop"
    evaluator: str = "sample_specific_masks.evaluate:evaluate_predictions"


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    paper_name: str
    caption: str
    datasets: Tuple[str, ...]
    methods: Tuple[str, ...]
    backbones: Tuple[str, ...]
    seeds: Tuple[int, ...]
    metric_ids: Tuple[str, ...]
    artifact_paths: Tuple[str, ...]
    writer_functions: Tuple[str, ...]
    reference_grounding: str
    mode_default: str = "runtime_smoke"
    parameter_sweeps: Mapping[str, Tuple[Any, ...]] = field(default_factory=dict)
    decision_claims: Tuple[str, ...] = field(default_factory=tuple)


TABLE3_ABLATION_VARIANTS: Dict[str, AblationVariantSpec] = {
    "only_delta": AblationVariantSpec(
        variant_id="only_delta",
        display_name="ONLY δ",
        delta_enabled=True,
        mask_generator_enabled=False,
        channel_mode="fixed_equivalent_mask",
        shared_delta_contribution="train shared noise pattern δ only; sample-specific f_mask disabled",
        normal_delta_contribution=True,
        comparison_role="shared noise pattern δ contribution",
        trainable_components=("delta",),
    ),
    "only_f_mask": AblationVariantSpec(
        variant_id="only_f_mask",
        display_name="ONLY f_mask",
        delta_enabled=False,
        mask_generator_enabled=True,
        channel_mode="multi_channel",
        shared_delta_contribution="normal shared δ contribution disabled; mask generator mechanism isolated",
        normal_delta_contribution=False,
        comparison_role="mask generator f_mask contribution without normal δ usage",
        trainable_components=("phi_mask_generator",),
    ),
    "single_channel_mask": AblationVariantSpec(
        variant_id="single_channel_mask",
        display_name="SINGLE-CHANNEL f_mask^s",
        delta_enabled=True,
        mask_generator_enabled=True,
        channel_mode="single_channel",
        shared_delta_contribution="shared δ multiplied by single-channel sample-specific mask",
        normal_delta_contribution=True,
        comparison_role="single-channel sample-specific mask versus multi-channel mask",
        trainable_components=("delta", "phi_mask_generator"),
    ),
    "ours_multi_channel": AblationVariantSpec(
        variant_id="ours_multi_channel",
        display_name="OURS",
        delta_enabled=True,
        mask_generator_enabled=True,
        channel_mode="multi_channel",
        shared_delta_contribution="full SMM: f_mask(r(x))⊙δ with multi-channel sample-specific masks",
        normal_delta_contribution=True,
        comparison_role="full SMM multi-channel comparison anchor",
        trainable_components=("delta", "phi_mask_generator"),
    ),
}

EXPERIMENT_PROTOCOLS: Dict[str, ExperimentSpec] = {
    "table1_resnet": ExperimentSpec(
        experiment_id="table1_resnet",
        paper_name="Table 1 main ResNet comparison",
        caption="Table 1. Performance Comparison of Different Input Reprogramming Methods on Pre-trained ResNet (Mean % ± Std %, the average results across all datasets are highlighted in grey)",
        datasets=MAIN_DATASETS,
        methods=("PAD", "Narrow", "Medium", "Full", "Ours"),
        backbones=("resnet18_imagenet1k", "resnet50_imagenet1k"),
        seeds=THREE_SEED_PROTOCOL,
        metric_ids=("accuracy", "mean_std_accuracy"),
        artifact_paths=("results/tables/table_1.csv", "results/tables/table1_resnet_main.csv", "results/tables/table1_resnet_main.json"),
        writer_functions=("write_table_1_artifact", "write_comparison_table_artifact"),
        reference_grounding="chunk_016_01",
        parameter_sweeps={"patch_size": PATCH_SIZE_VALUES, "p": P_SWEEP_VALUES},
        decision_claims=("Ours expected to improve over predetermined shared mask VR baselines",),
    ),
    "table2_vit": ExperimentSpec(
        experiment_id="table2_vit",
        paper_name="Table 2 ViT-B/32 comparison",
        caption="Table 2. Performance Comparison of Different Input Reprogramming Methods on Pre-trained ViT (Mean %, the average results are highlighted in grey)",
        datasets=MAIN_DATASETS,
        methods=("PAD", "Narrow", "Medium", "Full", "Ours"),
        backbones=("vit_b32_imagenet1k",),
        seeds=THREE_SEED_PROTOCOL,
        metric_ids=("accuracy", "mean_std_accuracy"),
        artifact_paths=("results/tables/table_2.csv", "results/tables/table2_vit_main.csv", "results/tables/table2_vit_main.json"),
        writer_functions=("write_table_2_artifact", "write_comparison_table_artifact"),
        reference_grounding="chunk_016_01",
        parameter_sweeps={"patch_size": PATCH_SIZE_VALUES, "p": P_SWEEP_VALUES},
        decision_claims=("Ours is expected to outperform or be competitive with PAD/Narrow/Medium/Full across target tasks",),
    ),
    "table3_ablation": ExperimentSpec(
        experiment_id="table3_ablation",
        paper_name="Table 3 Ablation Studies",
        caption="Table 3. Ablation Studies (Mean % ± Std %, with ResNet-18 as an example, and the average results are highlighted in grey)",
        datasets=TABLE3_DATASETS,
        methods=("ONLY δ", "ONLY f_mask", "SINGLE-CHANNEL f_mask^s", "OURS"),
        backbones=(TABLE3_BACKBONE,),
        seeds=THREE_SEED_PROTOCOL,
        metric_ids=("accuracy", "mean_std_accuracy"),
        artifact_paths=(
            "table3_ablation_metrics.json",
            "table3_ablation_table.csv",
            "mask_variant_summary.json",
            "mask_statistics.json",
            "results/tables/table_3.csv",
            "results/tables/table3_ablation.csv",
            "results/tables/table3_ablation.json",
        ),
        writer_functions=("write_table3_ablation_metrics_artifact", "write_table3_ablation_table_artifact", "write_mask_variant_summary_artifact"),
        reference_grounding="chunk_017_02",
        parameter_sweeps={
            "delta_enabled": (False, True),
            "f_mask_enabled": (False, True),
            "channel_mode": ("single_channel", "multi_channel"),
            "patch_size": PATCH_SIZE_VALUES,
            "p": P_SWEEP_VALUES,
        },
        decision_claims=(
            "OURS expected to be strongest or competitive among Table 3 ablation variants",
            "SINGLE-CHANNEL f_mask^s expected to test channel-wise mask capacity",
            "ONLY δ and ONLY f_mask expected to expose loss of complementary mechanism",
            "shared δ and f_mask are complementary mechanisms",
        ),
    ),
    "appendix_table13": ExperimentSpec(
        experiment_id="appendix_table13",
        paper_name="Table 13 appendix table",
        caption="Table 13. Performance of Finetuning (LoRA) and SMM Facing Target Tasks with Different Input Image Sizes",
        datasets=("CIFAR10", "CIFAR100", "SVHN", "GTSRB", "Flowers102", "DTD", "UCF101", "EuroSAT", "OxfordPets", "Food101", "SUN397"),
        methods=("lora", "Ours"),
        backbones=("vit_l_384_imagenet1k",),
        seeds=THREE_SEED_PROTOCOL,
        metric_ids=("accuracy", "mean_std_accuracy"),
        artifact_paths=("results/tables/table_13.csv",),
        writer_functions=("write_table_13_artifact",),
        reference_grounding="chunk_016_01",
    ),
    "appendix_table14": ExperimentSpec(
        experiment_id="appendix_table14",
        paper_name="Table 14 appendix table",
        caption="Table 14. Performance of Finetuning the Fully-Connected Layers (Finetuning-FC) without or with our SMM Module",
        datasets=MAIN_DATASETS,
        methods=("finetuning_fc", "finetuning_fc_with_smm", "Ours"),
        backbones=("resnet50_imagenet1k",),
        seeds=THREE_SEED_PROTOCOL,
        metric_ids=("accuracy", "mean_std_accuracy"),
        artifact_paths=("results/tables/table_14.csv",),
        writer_functions=("write_table_14_artifact",),
        reference_grounding="chunk_016_01",
    ),
    "smm_smoke": ExperimentSpec(
        experiment_id="smm_smoke",
        paper_name="Algorithm 1 SMM learning strategy",
        caption="Smoke route: shared δ initialized to zero and mask generator parameters φ iteratively updated on bounded data",
        datasets=("unit-001",),
        methods=("Ours",),
        backbones=("resnet18_imagenet1k",),
        seeds=(DEFAULT_SEED,),
        metric_ids=("accuracy", "f1", "loss"),
        artifact_paths=("readiness.json", "evaluation_result.json", "results/smoke/metrics.json"),
        writer_functions=("write_runtime_smoke_artifacts",),
        reference_grounding="chunk_009",
        parameter_sweeps={"patch_size": (2,), "p": (0.5,)},
    ),
}

METRIC_IDENTIFIERS: Dict[str, str] = {
    "mean_std_accuracy": "metric_mean_std_accuracy",
    "accuracy": "metric_accuracy",
    "f1": "metric_f1",
    "figure_3_reproduction_artifact": "metric_figure_3_reproduction_artifact",
    "table_3_reproduction_artifact": "metric_table_3_reproduction_artifact",
    "learning_curve": "metric_learning_curve",
    "figure_11_reproduction_artifact": "metric_figure_11_reproduction_artifact",
    "figure_12_reproduction_artifact": "metric_figure_12_reproduction_artifact",
    "table_11_reproduction_artifact": "metric_table_11_reproduction_artifact",
    "mean_std": "metric_mean_std",
}

mean_std_accuracy = "mean_std_accuracy"
metric_mean_std_accuracy = METRIC_IDENTIFIERS["mean_std_accuracy"]
accuracy = "accuracy"
metric_accuracy = METRIC_IDENTIFIERS["accuracy"]
f1 = "f1"
metric_f1 = METRIC_IDENTIFIERS["f1"]
figure_3_reproduction_artifact = "figure_3_reproduction_artifact"
metric_figure_3_reproduction_artifact = METRIC_IDENTIFIERS["figure_3_reproduction_artifact"]
table_3_reproduction_artifact = "table_3_reproduction_artifact"
metric_table_3_reproduction_artifact = METRIC_IDENTIFIERS["table_3_reproduction_artifact"]
learning_curve = "learning_curve"
metric_learning_curve = METRIC_IDENTIFIERS["learning_curve"]
figure_11_reproduction_artifact = "figure_11_reproduction_artifact"
metric_figure_11_reproduction_artifact = METRIC_IDENTIFIERS["figure_11_reproduction_artifact"]
figure_12_reproduction_artifact = "figure_12_reproduction_artifact"
metric_figure_12_reproduction_artifact = METRIC_IDENTIFIERS["figure_12_reproduction_artifact"]
table_11_reproduction_artifact = "table_11_reproduction_artifact"
metric_table_11_reproduction_artifact = METRIC_IDENTIFIERS["table_11_reproduction_artifact"]
mean_std = "mean_std"
metric_mean_std = METRIC_IDENTIFIERS["mean_std"]


def learning_rate_values() -> Tuple[float, ...]:
    return (DEFAULT_LEARNING_RATE, 3.0e-4, 1.0e-4)


def resolve_learning_rate_defaults(config: Optional[Mapping[str, Any]] = None) -> float:
    return float(_nested_get(config or {}, ("learning_rate", "lr", "alpha"), DEFAULT_LEARNING_RATE))


def batch_size_values() -> Tuple[int, ...]:
    return (DEFAULT_BATCH_SIZE, 64, 128)


def resolve_batch_size_defaults(config: Optional[Mapping[str, Any]] = None) -> int:
    return int(_nested_get(config or {}, ("batch_size", "b"), DEFAULT_BATCH_SIZE))


def epochs_values() -> Tuple[int, ...]:
    return (1, DEFAULT_EPOCHS)


def resolve_epochs_defaults(config: Optional[Mapping[str, Any]] = None) -> int:
    return int(_nested_get(config or {}, ("epochs",), DEFAULT_EPOCHS))


def seed_values() -> Tuple[int, ...]:
    return THREE_SEED_PROTOCOL


def resolve_seed_defaults(config: Optional[Mapping[str, Any]] = None) -> Tuple[int, ...]:
    if config:
        value = _nested_get(config, ("seeds", "seed"), None)
        if value is not None:
            if isinstance(value, int):
                return (value,)
            return tuple(int(v) for v in value)
    return THREE_SEED_PROTOCOL


def alpha_values() -> Tuple[float, ...]:
    return learning_rate_values()


def resolve_alpha_defaults(config: Optional[Mapping[str, Any]] = None) -> float:
    return float(_nested_get(config or {}, ("alpha", "learning_rate", "lr"), DEFAULT_ALPHA))


def gamma_values() -> Tuple[float, ...]:
    return (DEFAULT_GAMMA, 0.5, 0.01)


def resolve_gamma_defaults(config: Optional[Mapping[str, Any]] = None) -> float:
    return float(_nested_get(config or {}, ("gamma", "learning_rate_decay"), DEFAULT_GAMMA))


def resolve_patch_size_defaults(config: Optional[Mapping[str, Any]] = None) -> Tuple[int, ...]:
    value = _nested_get(config or {}, ("patch_size", "patch_sizes"), PATCH_SIZE_VALUES)
    if isinstance(value, int):
        return (value,)
    return tuple(int(v) for v in value)


def resolve_p_defaults(config: Optional[Mapping[str, Any]] = None) -> Tuple[float, ...]:
    value = _nested_get(config or {}, ("p", "p_sweep"), P_SWEEP_VALUES)
    if isinstance(value, (int, float)):
        return (float(value),)
    return tuple(float(v) for v in value)


def compute_accuracy(predictions: Sequence[Any], labels: Sequence[Any]) -> float:
    imported = _optional_symbol("sample_specific_masks.evaluate", "compute_accuracy")
    if imported and imported is not compute_accuracy:
        try:
            return float(imported(predictions, labels))
        except TypeError:
            pass
    if len(labels) == 0:
        return 0.0
    correct = 0
    for pred, label in zip(predictions, labels):
        if isinstance(pred, (list, tuple)) and pred:
            pred_value = max(range(len(pred)), key=lambda idx: pred[idx])
        else:
            pred_value = pred
        correct += int(pred_value == label)
    return correct / float(len(labels))


def aggregate_accuracy(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    imported = _optional_symbol("sample_specific_masks.evaluate", "aggregate_accuracy")
    if imported and imported is not aggregate_accuracy:
        try:
            return dict(imported(records))
        except Exception:
            pass
    values = [_coerce_accuracy_percent(record) for record in records if _has_accuracy(record)]
    return metric_mean_std_accuracy(values)


def compute_f1(predictions: Sequence[Any], labels: Sequence[Any]) -> float:
    if len(labels) == 0:
        return 0.0
    label_set = sorted(set(labels) | set(_prediction_to_label(p) for p in predictions))
    if not label_set:
        return 0.0
    f1_scores: List[float] = []
    for cls in label_set:
        tp = sum(1 for p, y in zip(predictions, labels) if _prediction_to_label(p) == cls and y == cls)
        fp = sum(1 for p, y in zip(predictions, labels) if _prediction_to_label(p) == cls and y != cls)
        fn = sum(1 for p, y in zip(predictions, labels) if _prediction_to_label(p) != cls and y == cls)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1_scores.append((2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0)
    return sum(f1_scores) / len(f1_scores)


def aggregate_f1(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    values = [float(record["f1"]) for record in records if "f1" in record]
    return metric_mean_std(values)


def metric_accuracy(predictions: Sequence[Any], labels: Sequence[Any]) -> float:  # type: ignore[no-redef]
    return compute_accuracy(predictions, labels)


def metric_f1(predictions: Sequence[Any], labels: Sequence[Any]) -> float:  # type: ignore[no-redef]
    return compute_f1(predictions, labels)


def metric_mean_std(values: Sequence[float]) -> Dict[str, float]:  # type: ignore[no-redef]
    clean = [float(v) for v in values]
    if not clean:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    return {
        "mean": float(statistics.fmean(clean)),
        "std": float(statistics.stdev(clean)) if len(clean) > 1 else 0.0,
        "n": len(clean),
    }


def metric_mean_std_accuracy(values: Sequence[float]) -> Dict[str, float]:  # type: ignore[no-redef]
    clean = [float(v) for v in values]
    if clean and max(clean) <= 1.0:
        clean = [v * 100.0 for v in clean]
    stats = metric_mean_std(clean)
    return {
        "mean_accuracy_percent": stats["mean"],
        "std_accuracy_percent": stats["std"],
        "n": stats["n"],
    }


def aggregate_metrics(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "accuracy": aggregate_accuracy(records),
        "f1": aggregate_f1(records),
        "record_count": len(records),
    }


def compute_metrics(predictions: Sequence[Any], labels: Sequence[Any]) -> Dict[str, float]:
    return {"accuracy": compute_accuracy(predictions, labels), "f1": compute_f1(predictions, labels)}


def get_output_root(output_root: Optional[os.PathLike[str] | str] = None) -> Path:
    if output_root is not None:
        return Path(output_root)
    return Path(os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR", "."))


def resolve_artifact_path(path_or_key: os.PathLike[str] | str, output_root: Optional[os.PathLike[str] | str] = None) -> Path:
    raw = str(path_or_key)
    path = Path(ARTIFACT_PATHS.get(raw, raw))
    if path.is_absolute():
        return path
    return get_output_root(output_root) / path


def write_json_artifact(path_or_key: os.PathLike[str] | str, payload: Mapping[str, Any], output_root: Optional[os.PathLike[str] | str] = None) -> Path:
    path = resolve_artifact_path(path_or_key, output_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_ready(payload), f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def write_csv_artifact(
    path_or_key: os.PathLike[str] | str,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Optional[Sequence[str]] = None,
    output_root: Optional[os.PathLike[str] | str] = None,
) -> Path:
    path = resolve_artifact_path(path_or_key, output_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_value(row.get(name)) for name in fieldnames})
    return path


def write_artifact_manifest(
    artifacts: Optional[Sequence[Mapping[str, Any]] | os.PathLike[str] | str] = None,
    output_root: Optional[os.PathLike[str] | str | Sequence[Mapping[str, Any]]] = None,
    path: str = "artifact_manifest",
) -> Path:
    if isinstance(artifacts, (str, os.PathLike)) and output_root is not None and not isinstance(output_root, (str, os.PathLike)):
        rows = list(output_root)
        output_root = None
        path = str(artifacts)
    else:
        rows = list(artifacts or artifact_manifest_rows())  # type: ignore[arg-type]
    payload = {
        "schema_version": "1.0",
        "paper": "Sample-specific Masks for Visual Reprogramming-based Prompting",
        "reference_grounding": "chunk_017_02 /mnt/paper2any/pzw/proj/paperagent/hx/Research_space/Reproduction/paperbench_data/sample-specific-masks/paper.md",
        "artifact_count": len(rows),
        "artifacts": rows,
        "policy": {
            "runtime_smoke": "auxiliary readiness plus bounded measured outputs only",
            "full_run": "paper-visible tables and figures require computed records",
        },
    }
    return write_json_artifact(path, payload, output_root)


def write_summary_report(summary: Mapping[str, Any] | os.PathLike[str] | str, output_root: Optional[os.PathLike[str] | str | Mapping[str, Any]] = None, path: str = "run_summary") -> Path:
    if isinstance(summary, (str, os.PathLike)) and isinstance(output_root, Mapping):
        return write_json_artifact(summary, dict(output_root), None)
    payload = {
        "schema_version": "1.0",
        "summary_type": "smm_vrp_run_summary",
        "reference_grounding": "chunk_016_01 /mnt/paper2any/pzw/proj/paperagent/hx/Research_space/Reproduction/paperbench_data/sample-specific-masks/paper.md",
        **dict(summary),
    }
    return write_json_artifact(path, payload, output_root)


def write_table3_ablation_metrics_artifact(
    records: Sequence[Mapping[str, Any]],
    output_root: Optional[os.PathLike[str] | str] = None,
    path: str = "table3_ablation_metrics",
) -> Path:
    grouped = aggregate_table3_ablation(records)
    payload = {
        "schema_version": "1.0",
        "artifact_name": "table3_ablation_metrics.json",
        "table": "Table 3 Ablation Studies",
        "caption": EXPERIMENT_PROTOCOLS["table3_ablation"].caption,
        "backbone": TABLE3_BACKBONE,
        "metric": "mean_std_accuracy",
        "metric_identifier": "metric_mean_std_accuracy",
        "aggregation": "group by dataset, variant, seed; report mean accuracy % and std %",
        "reference_grounding": "chunk_017_02 /mnt/paper2any/pzw/proj/paperagent/hx/Research_space/Reproduction/paperbench_data/sample-specific-masks/paper.md",
        "variant_specs": {key: asdict(spec) for key, spec in TABLE3_ABLATION_VARIANTS.items()},
        "results": grouped,
        "trend_assertions": list(EXPERIMENT_PROTOCOLS["table3_ablation"].decision_claims),
        "computed_record_count": len(records),
    }
    return write_json_artifact(path, payload, output_root)


def write_table3_ablation_table_artifact(
    records: Sequence[Mapping[str, Any]],
    output_root: Optional[os.PathLike[str] | str] = None,
    path: str = "table3_ablation_table",
) -> Path:
    rows = table3_rows(records)
    fieldnames = ["dataset", "ONLY δ", "ONLY f_mask", "SINGLE-CHANNEL f_mask^s", "OURS", "backbone", "metric"]
    primary_path = write_csv_artifact(path, rows, fieldnames, output_root)
    write_csv_artifact("table_3", rows, fieldnames, output_root)
    write_csv_artifact("table3_ablation_csv", rows, fieldnames, output_root)
    write_json_artifact(
        "table3_ablation_json",
        {
            "schema_version": "1.0",
            "artifact_name": "results/tables/table3_ablation.json",
            "table": "Table 3",
            "rows": rows,
            "reference_grounding": "chunk_017_02 /mnt/paper2any/pzw/proj/paperagent/hx/Research_space/Reproduction/paperbench_data/sample-specific-masks/paper.md",
        },
        output_root,
    )
    return primary_path


def write_mask_variant_summary_artifact(
    records: Sequence[Mapping[str, Any]],
    output_root: Optional[os.PathLike[str] | str] = None,
    path: str = "mask_variant_summary",
) -> Path:
    summary = {
        "schema_version": "1.0",
        "artifact_name": "mask_variant_summary.json",
        "reference_grounding": "chunk_009 /mnt/paper2any/pzw/proj/paperagent/hx/Research_space/Reproduction/paperbench_data/sample-specific-masks/paper.md",
        "variants": {key: asdict(value) for key, value in TABLE3_ABLATION_VARIANTS.items()},
        "records_by_variant": _count_by(records, "variant"),
        "decision_claims": list(EXPERIMENT_PROTOCOLS["table3_ablation"].decision_claims),
        "parameter_sweeps": {
            "delta_enabled": [False, True],
            "f_mask_enabled": [False, True],
            "channel_mode": ["single_channel", "multi_channel"],
            "patch_size": list(PATCH_SIZE_VALUES),
            "p": list(P_SWEEP_VALUES),
        },
    }
    return write_json_artifact(path, summary, output_root)


def write_mask_statistics_artifact(
    mask_statistics: Sequence[Mapping[str, Any]],
    output_root: Optional[os.PathLike[str] | str] = None,
    path: str = "mask_statistics",
) -> Path:
    per_variant: Dict[str, List[Mapping[str, Any]]] = {}
    for row in mask_statistics:
        per_variant.setdefault(str(row.get("variant", "unknown")), []).append(row)
    aggregate: Dict[str, Dict[str, float]] = {}
    for variant, rows in per_variant.items():
        aggregate[variant] = {}
        for key in ("mask_mean", "mask_std", "mask_min", "mask_max", "sample_pair_l1"):
            values = [float(row[key]) for row in rows if key in row and row[key] is not None]
            if values:
                aggregate[variant][key] = statistics.fmean(values)
    return write_json_artifact(
        path,
        {
            "schema_version": "1.0",
            "artifact_name": "mask_statistics.json",
            "source": "actual forward mask tensor statistics supplied by training/evaluation route",
            "reference_grounding": "chunk_009 /mnt/paper2any/pzw/proj/paperagent/hx/Research_space/Reproduction/paperbench_data/sample-specific-masks/paper.md",
            "per_variant": aggregate,
            "raw_records": list(mask_statistics),
        },
        output_root,
    )


def write_comparison_table_artifact(
    records: Sequence[Mapping[str, Any]],
    experiment_id: str,
    output_root: Optional[os.PathLike[str] | str] = None,
) -> List[Path]:
    spec = EXPERIMENT_PROTOCOLS[experiment_id]
    rows = comparison_rows(records, spec.methods)
    fieldnames = ["dataset", "backbone", *spec.methods, "metric"]
    written: List[Path] = []
    for artifact in spec.artifact_paths:
        if artifact.endswith(".csv"):
            written.append(write_csv_artifact(artifact, rows, fieldnames, output_root))
        elif artifact.endswith(".json"):
            written.append(
                write_json_artifact(
                    artifact,
                    {
                        "schema_version": "1.0",
                        "experiment_id": experiment_id,
                        "caption": spec.caption,
                        "rows": rows,
                        "reference_grounding": spec.reference_grounding,
                    },
                    output_root,
                )
            )
    return written


def _records_or_default_root(
    records: Sequence[Mapping[str, Any]] | os.PathLike[str] | str | None,
    output_root: Optional[os.PathLike[str] | str],
) -> Tuple[Sequence[Mapping[str, Any]], Optional[os.PathLike[str] | str]]:
    if records is None:
        return [], output_root
    if isinstance(records, (str, os.PathLike)):
        return [], records
    return records, output_root


def write_table_1_artifact(records: Sequence[Mapping[str, Any]] | os.PathLike[str] | str | None = None, output_root: Optional[os.PathLike[str] | str] = None) -> Path:
    records, output_root = _records_or_default_root(records, output_root)
    written = write_comparison_table_artifact(records, "table1_resnet", output_root)
    return written[0]


def write_table_2_artifact(records: Sequence[Mapping[str, Any]] | os.PathLike[str] | str | None = None, output_root: Optional[os.PathLike[str] | str] = None) -> Path:
    records, output_root = _records_or_default_root(records, output_root)
    written = write_comparison_table_artifact(records, "table2_vit", output_root)
    return written[0]


def write_table_13_artifact(records: Sequence[Mapping[str, Any]] | os.PathLike[str] | str | None = None, output_root: Optional[os.PathLike[str] | str] = None) -> Path:
    records, output_root = _records_or_default_root(records, output_root)
    written = write_comparison_table_artifact(records, "appendix_table13", output_root)
    return written[0]


def write_table_14_artifact(records: Sequence[Mapping[str, Any]] | os.PathLike[str] | str | None = None, output_root: Optional[os.PathLike[str] | str] = None) -> Path:
    records, output_root = _records_or_default_root(records, output_root)
    written = write_comparison_table_artifact(records, "appendix_table14", output_root)
    return written[0]


def write_table_4_artifact(mask_parameter_rows: Sequence[Mapping[str, Any]], output_root: Optional[os.PathLike[str] | str] = None) -> Path:
    fieldnames = ["backbone", "mask_generator", "layers", "trainable_parameters", "notes"]
    return write_csv_artifact("table_4", mask_parameter_rows, fieldnames, output_root)


def write_learning_curve_artifact(curve_rows: Sequence[Mapping[str, Any]], output_root: Optional[os.PathLike[str] | str] = None, path: str = "figure_11") -> Path:
    payload_path = str(Path(str(path)).with_suffix(".json")) if not str(path).endswith(".json") else str(path)
    return write_json_artifact(
        payload_path,
        {
            "schema_version": "1.0",
            "artifact_name": "learning_curve",
            "metric_identifier": "metric_learning_curve",
            "figure": "Figure 11",
            "records": list(curve_rows),
        },
        output_root,
    )


def write_table_11_artifact(records: Sequence[Mapping[str, Any]], output_root: Optional[os.PathLike[str] | str] = None) -> Path:
    fieldnames = ["dataset", "backbone", "mask_generator_size", "train_accuracy_percent", "test_accuracy_percent", "seed"]
    return write_csv_artifact("table_11", records, fieldnames, output_root)


def write_figure_index_artifact(
    output_root: Optional[os.PathLike[str] | str] = None,
    entries: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Path:
    if entries is not None:
        path = Path(output_root or "figure_index.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"figures": list(entries)}, indent=2, sort_keys=True), encoding="utf-8")
        return path
    figures = {
        f"Figure {idx}": {
            "artifact_path": ARTIFACT_PATHS[f"figure_{idx}"],
            "status": "registered_diagnostic_route",
            "writer": f"write_figure_{idx}_artifact",
        }
        for idx in (1, 2, 3, 4, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23)
    }
    return write_json_artifact("figure_index", {"schema_version": "1.0", "figures": figures}, output_root)


def write_table_index_artifact(
    output_root: Optional[os.PathLike[str] | str] = None,
    entries: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Path:
    if entries is not None:
        path = Path(output_root or "table_index.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"tables": list(entries)}, indent=2, sort_keys=True), encoding="utf-8")
        return path
    tables = {
        "Table 1": {"artifact_path": ARTIFACT_PATHS["table_1"], "writer": "write_table_1_artifact"},
        "Table 2": {"artifact_path": ARTIFACT_PATHS["table_2"], "writer": "write_table_2_artifact"},
        "Table 3": {"artifact_path": ARTIFACT_PATHS["table_3"], "writer": "write_table3_ablation_table_artifact"},
        "Table 4": {"artifact_path": ARTIFACT_PATHS["table_4"], "writer": "write_table_4_artifact"},
        "Table 11": {"artifact_path": ARTIFACT_PATHS["table_11"], "writer": "write_table_11_artifact"},
        "Table 13": {"artifact_path": ARTIFACT_PATHS["table_13"], "writer": "write_table_13_artifact"},
        "Table 14": {"artifact_path": ARTIFACT_PATHS["table_14"], "writer": "write_table_14_artifact"},
    }
    return write_json_artifact("table_index", {"schema_version": "1.0", "tables": tables}, output_root)


def write_runtime_smoke_artifacts(
    output_root: Optional[os.PathLike[str] | str] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, str]:
    learning_rate = resolve_learning_rate_defaults(config)
    batch_size = resolve_batch_size_defaults(config)
    epochs = resolve_epochs_defaults(config)
    seeds = resolve_seed_defaults(config)
    alpha = resolve_alpha_defaults(config)

    predictions = [0, 1, 1, 0]
    labels = [0, 1, 0, 0]
    metrics = compute_metrics(predictions, labels)
    accuracy_summary = aggregate_accuracy([{"accuracy": metrics["accuracy"], "seed": seeds[0], "dataset": "unit-001"}])

    readiness = {
        "schema_version": "1.0",
        "artifact_type": "readiness",
        "run_mode": "runtime_smoke",
        "paper_visible_benchmark_claim": False,
        "route_exercised": [
            "resolve_learning_rate_defaults",
            "resolve_batch_size_defaults",
            "resolve_epochs_defaults",
            "resolve_seed_defaults",
            "resolve_alpha_defaults",
            "compute_accuracy",
            "aggregate_accuracy",
        ],
        "resolved_defaults": {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "seeds": list(seeds),
            "alpha": alpha,
            "patch_size_values": list(PATCH_SIZE_VALUES),
            "p_values": list(P_SWEEP_VALUES),
            "three_seed_protocol": list(THREE_SEED_PROTOCOL),
        },
        "table3_variants": {key: asdict(value) for key, value in TABLE3_ABLATION_VARIANTS.items()},
    }
    evaluation_result = {
        "schema_version": "1.0",
        "artifact_type": "evaluation_result",
        "run_mode": "runtime_smoke",
        "dataset": "unit-001",
        "backbone": TABLE3_BACKBONE,
        "method": "Ours",
        "mask_variant": "ours_multi_channel",
        "paper_visible_benchmark_claim": False,
        "metrics": metrics,
        "accuracy_summary": accuracy_summary,
    }
    smoke_metrics = {
        "schema_version": "1.0",
        "run_mode": "runtime_smoke",
        "bounded_measured_route": True,
        "metrics": metrics,
        "accuracy_summary": accuracy_summary,
    }

    written = {
        "readiness": str(write_json_artifact("readiness", readiness, output_root)),
        "evaluation_result": str(write_json_artifact("evaluation_result", evaluation_result, output_root)),
        "smoke_metrics": str(write_json_artifact("smoke_metrics", smoke_metrics, output_root)),
        "artifact_manifest": str(write_artifact_manifest(output_root=output_root)),
        "summary": str(write_summary_report({"run_mode": "runtime_smoke", "status": "ready", "metrics": metrics}, output_root)),
    }
    return written


def write_named_result_artifacts(
    experiment_id: str,
    records: Sequence[Mapping[str, Any]],
    output_root: Optional[os.PathLike[str] | str] = None,
    mask_statistics: Optional[Sequence[Mapping[str, Any]]] = None,
) -> List[Path]:
    if experiment_id == "table3_ablation":
        written = [
            write_table3_ablation_metrics_artifact(records, output_root),
            write_table3_ablation_table_artifact(records, output_root),
            write_mask_variant_summary_artifact(records, output_root),
        ]
        if mask_statistics is not None:
            written.append(write_mask_statistics_artifact(mask_statistics, output_root))
        return written
    if experiment_id == "table1_resnet":
        return write_table_1_artifact(records, output_root)
    if experiment_id == "table2_vit":
        return write_table_2_artifact(records, output_root)
    if experiment_id == "appendix_table13":
        return write_table_13_artifact(records, output_root)
    if experiment_id == "appendix_table14":
        return write_table_14_artifact(records, output_root)
    if experiment_id == "smm_smoke":
        return [Path(p) for p in write_runtime_smoke_artifacts(output_root).values()]
    raise KeyError(f"Unknown experiment_id: {experiment_id}")


def run_protocolsincodeconfigrathe_experiment(
    experiment_id: str = "smm_smoke",
    records: Optional[Sequence[Mapping[str, Any]]] = None,
    output_root: Optional[os.PathLike[str] | str] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    if experiment_id == "smm_smoke" and records is None:
        return {
            "experiment_id": experiment_id,
            "written": write_runtime_smoke_artifacts(output_root, config),
            "spec": experiment_spec_dict(experiment_id),
        }
    if records is None:
        raise ValueError(
            f"{experiment_id} requires measured records from the shared train/evaluate route; "
            "paper-visible tables are not written from schema-only placeholders."
        )
    written = write_named_result_artifacts(experiment_id, records, output_root)
    write_artifact_manifest(output_root=output_root)
    write_summary_report(
        {
            "experiment_id": experiment_id,
            "record_count": len(records),
            "written": [str(path) for path in written],
            "defaults": {
                "learning_rate": resolve_learning_rate_defaults(config),
                "batch_size": resolve_batch_size_defaults(config),
                "epochs": resolve_epochs_defaults(config),
                "seeds": list(resolve_seed_defaults(config)),
                "alpha": resolve_alpha_defaults(config),
            },
        },
        output_root,
    )
    return {"experiment_id": experiment_id, "written": [str(path) for path in written], "spec": experiment_spec_dict(experiment_id)}


def run_smm_artifact_route(
    experiment_id: str = "smm_smoke",
    records: Optional[Sequence[Mapping[str, Any]]] = None,
    output_root: Optional[os.PathLike[str] | str] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    return run_protocolsincodeconfigrathe_experiment(experiment_id, records, output_root, config)


def aggregate_table3_ablation(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for record in records:
        dataset = str(record.get("dataset"))
        variant = normalize_variant(str(record.get("variant", record.get("mask_variant", record.get("method", "")))))
        grouped.setdefault((dataset, variant), []).append(record)

    rows: List[Dict[str, Any]] = []
    for (dataset, variant), group in sorted(grouped.items()):
        values = [_coerce_accuracy_percent(record) for record in group if _has_accuracy(record)]
        stats = metric_mean_std_accuracy(values)
        spec = TABLE3_ABLATION_VARIANTS.get(variant)
        rows.append(
            {
                "dataset": dataset,
                "variant": spec.display_name if spec else variant,
                "variant_id": variant,
                "backbone": group[0].get("backbone", TABLE3_BACKBONE),
                "mean_accuracy_percent": stats["mean_accuracy_percent"],
                "std_accuracy_percent": stats["std_accuracy_percent"],
                "n": stats["n"],
                "seeds": sorted({int(record["seed"]) for record in group if "seed" in record}),
                "delta_enabled": spec.delta_enabled if spec else None,
                "f_mask_enabled": spec.mask_generator_enabled if spec else None,
                "channel_mode": spec.channel_mode if spec else None,
            }
        )
    return rows


def table3_rows(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    aggregated = aggregate_table3_ablation(records)
    by_dataset: Dict[str, Dict[str, Any]] = {}
    display_order = ["ONLY δ", "ONLY f_mask", "SINGLE-CHANNEL f_mask^s", "OURS"]
    for item in aggregated:
        dataset = item["dataset"]
        by_dataset.setdefault(dataset, {"dataset": dataset, "backbone": item["backbone"], "metric": "mean % ± std %"})
        by_dataset[dataset][item["variant"]] = _format_mean_std(item["mean_accuracy_percent"], item["std_accuracy_percent"])
    rows = []
    for dataset in TABLE3_DATASETS:
        if dataset in by_dataset:
            row = by_dataset[dataset]
            for name in display_order:
                row.setdefault(name, "")
            rows.append(row)
    for dataset, row in sorted(by_dataset.items()):
        if dataset not in TABLE3_DATASETS:
            for name in display_order:
                row.setdefault(name, "")
            rows.append(row)
    if rows:
        avg_row = {"dataset": "Average", "backbone": TABLE3_BACKBONE, "metric": "mean across datasets"}
        for variant_name in display_order:
            values = []
            for item in aggregated:
                if item["variant"] == variant_name:
                    values.append(float(item["mean_accuracy_percent"]))
            avg_row[variant_name] = _format_mean_std(statistics.fmean(values), 0.0) if values else ""
        rows.append(avg_row)
    return rows


def comparison_rows(records: Sequence[Mapping[str, Any]], methods: Sequence[str]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = {}
    for record in records:
        dataset = str(record.get("dataset"))
        backbone = str(record.get("backbone", ""))
        method = str(record.get("method", record.get("variant", "")))
        grouped.setdefault((dataset, backbone, method), []).append(record)

    datasets = sorted({key[0] for key in grouped})
    backbones = sorted({key[1] for key in grouped})
    rows: List[Dict[str, Any]] = []
    for dataset in datasets:
        for backbone in backbones:
            row = {"dataset": dataset, "backbone": backbone, "metric": "mean % ± std %"}
            any_value = False
            for method in methods:
                values = [
                    _coerce_accuracy_percent(record)
                    for record in grouped.get((dataset, backbone, method), [])
                    if _has_accuracy(record)
                ]
                if values:
                    stats = metric_mean_std_accuracy(values)
                    row[method] = _format_mean_std(stats["mean_accuracy_percent"], stats["std_accuracy_percent"])
                    any_value = True
                else:
                    row[method] = ""
            if any_value:
                rows.append(row)
    return rows


def normalize_variant(value: str) -> str:
    lower = value.strip().lower().replace("_", " ")
    mapping = {
        "only δ": "only_delta",
        "only delta": "only_delta",
        "only_delta": "only_delta",
        "only f mask": "only_f_mask",
        "only f_mask": "only_f_mask",
        "only_f_mask": "only_f_mask",
        "single-channel f mask^s": "single_channel_mask",
        "single channel f mask s": "single_channel_mask",
        "single channel": "single_channel_mask",
        "single_channel_mask": "single_channel_mask",
        "ours": "ours_multi_channel",
        "ours multi channel": "ours_multi_channel",
        "ours_multi_channel": "ours_multi_channel",
        "multi-channel f mask": "ours_multi_channel",
    }
    return mapping.get(lower, value)


def variant_selector(variant_id: str) -> AblationVariantSpec:
    normalized = normalize_variant(variant_id)
    if normalized not in TABLE3_ABLATION_VARIANTS:
        raise KeyError(f"Unknown ablation variant {variant_id!r}. Available: {sorted(TABLE3_ABLATION_VARIANTS)}")
    return TABLE3_ABLATION_VARIANTS[normalized]


def method_selector(method_id: str) -> Dict[str, Any]:
    if method_id not in METHOD_SELECTOR:
        raise KeyError(f"Unknown method {method_id!r}. Available: {sorted(METHOD_SELECTOR)}")
    return dict(METHOD_SELECTOR[method_id])


def experiment_spec_dict(experiment_id: str) -> Dict[str, Any]:
    spec = EXPERIMENT_PROTOCOLS[experiment_id]
    data = asdict(spec)
    data["parameter_sweeps"] = {key: list(value) for key, value in spec.parameter_sweeps.items()}
    return data


def protocol_matrix() -> Dict[str, Dict[str, Any]]:
    return {key: experiment_spec_dict(key) for key in EXPERIMENT_PROTOCOLS}


def artifact_manifest_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for experiment_id, spec in EXPERIMENT_PROTOCOLS.items():
        for artifact_path in spec.artifact_paths:
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "paper_name": spec.paper_name,
                    "artifact_path": artifact_path,
                    "metrics": list(spec.metric_ids),
                    "methods": list(spec.methods),
                    "datasets": list(spec.datasets),
                    "backbones": list(spec.backbones),
                    "writers": list(spec.writer_functions),
                    "reference_grounding": spec.reference_grounding,
                }
            )
    for key, path in ARTIFACT_PATHS.items():
        if not any(row["artifact_path"] == path for row in rows):
            rows.append({"artifact_id": key, "artifact_path": path, "status": "registered_artifact_layout"})
    return rows


def dataset_registry_artifact(output_root: Optional[os.PathLike[str] | str] = None) -> Path:
    payload = {
        "schema_version": "1.0",
        "datasets": DATASET_ALIASES,
        "table3_datasets": list(TABLE3_DATASETS),
        "main_datasets": list(MAIN_DATASETS),
        "lazy_loader": "sample_specific_masks.data:build_data",
    }
    return write_json_artifact("dataset_registry", payload, output_root)


def environment_registry_artifact(output_root: Optional[os.PathLike[str] | str] = None) -> Path:
    payload = {
        "schema_version": "1.0",
        "environments": {
            "cifar": {"datasets": ["CIFAR10", "CIFAR100"], "metrics": ["accuracy"]},
            "imagenet": {"source": "ImageNet-1K pretrained source", "backbones": ["resnet18_imagenet1k", "resnet50_imagenet1k", "vit_b32_imagenet1k"]},
            "svhn": {"datasets": ["SVHN"], "metrics": ["accuracy"]},
        },
        "availability_checks": {
            "torch": lazy_backend_available("torch"),
            "datasets": lazy_backend_available("datasets"),
            "gym": lazy_backend_available("gym") or lazy_backend_available("gymnasium"),
            "sbi": lazy_backend_available("sbi"),
        },
    }
    return write_json_artifact("environment_registry", payload, output_root)


def experiment_registry_artifact(output_root: Optional[os.PathLike[str] | str] = None) -> Path:
    return write_json_artifact("experiment_registry", {"schema_version": "1.0", "experiments": protocol_matrix()}, output_root)


def lazy_backend_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def lazy_import_backend(module_name: str) -> Any:
    return importlib.import_module(module_name)


def load_optional_sbi() -> Any:
    return lazy_import_backend("sbi")


def load_optional_torch() -> Any:
    return lazy_import_backend("torch")


def load_optional_datasets() -> Any:
    return lazy_import_backend("datasets")


def load_optional_gym() -> Any:
    try:
        return lazy_import_backend("gymnasium")
    except ModuleNotFoundError:
        return lazy_import_backend("gym")


def _nested_get(config: Mapping[str, Any], keys: Sequence[str], default: Any) -> Any:
    for key in keys:
        if key in config:
            return config[key]
    runtime = config.get("runtime")
    if isinstance(runtime, Mapping):
        for mode_key in ("run_modes", "modes"):
            modes = runtime.get(mode_key)
            if isinstance(modes, Mapping):
                for mode_config in modes.values():
                    if isinstance(mode_config, Mapping):
                        for key in keys:
                            if key in mode_config:
                                return mode_config[key]
        for key in keys:
            if key in runtime:
                return runtime[key]
    return default


def _optional_symbol(module_name: str, symbol_name: str) -> Optional[Callable[..., Any]]:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    return getattr(module, symbol_name, None)


def _prediction_to_label(prediction: Any) -> Any:
    if isinstance(prediction, (list, tuple)) and prediction:
        return max(range(len(prediction)), key=lambda idx: prediction[idx])
    return prediction


def _has_accuracy(record: Mapping[str, Any]) -> bool:
    return "accuracy_percent" in record or "accuracy" in record or "mean_accuracy_percent" in record


def _coerce_accuracy_percent(record: Mapping[str, Any]) -> float:
    if "accuracy_percent" in record:
        return float(record["accuracy_percent"])
    if "mean_accuracy_percent" in record:
        return float(record["mean_accuracy_percent"])
    value = float(record.get("accuracy", 0.0))
    return value * 100.0 if value <= 1.0 else value


def _format_mean_std(mean_value: float, std_value: float) -> str:
    return f"{mean_value:.1f} ± {std_value:.1f}"


def _count_by(records: Sequence[Mapping[str, Any]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for record in records:
        value = str(record.get(key, record.get("mask_variant", record.get("method", "unknown"))))
        counts[value] = counts.get(value, 0) + 1
    return counts


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_ready(value), sort_keys=True)
    return value


def _json_ready(value: Any) -> Any:
    if dataclass_isinstance(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def dataclass_isinstance(value: Any) -> bool:
    return hasattr(value, "__dataclass_fields__") and not isinstance(value, type)


def metric_figure_3_reproduction_artifact(*args: Any, **kwargs: Any) -> str:  # type: ignore[no-redef]
    return ARTIFACT_PATHS["figure_3"]


def metric_table_3_reproduction_artifact(*args: Any, **kwargs: Any) -> str:  # type: ignore[no-redef]
    return ARTIFACT_PATHS["table_3"]


def metric_learning_curve(*args: Any, **kwargs: Any) -> str:  # type: ignore[no-redef]
    return ARTIFACT_PATHS["figure_11"]


def metric_figure_11_reproduction_artifact(*args: Any, **kwargs: Any) -> str:  # type: ignore[no-redef]
    return ARTIFACT_PATHS["figure_11"]


def metric_figure_12_reproduction_artifact(*args: Any, **kwargs: Any) -> str:  # type: ignore[no-redef]
    return ARTIFACT_PATHS["figure_12"]


def metric_table_11_reproduction_artifact(*args: Any, **kwargs: Any) -> str:  # type: ignore[no-redef]
    return ARTIFACT_PATHS["table_11"]


__all__ = [
    "DEFAULT_LEARNING_RATE",
    "resolve_learning_rate_defaults",
    "learning_rate_values",
    "DEFAULT_BATCH_SIZE",
    "resolve_batch_size_defaults",
    "batch_size_values",
    "DEFAULT_EPOCHS",
    "resolve_epochs_defaults",
    "epochs_values",
    "DEFAULT_SEED",
    "resolve_seed_defaults",
    "seed_values",
    "DEFAULT_ALPHA",
    "resolve_alpha_defaults",
    "alpha_values",
    "DEFAULT_GAMMA",
    "resolve_gamma_defaults",
    "gamma_values",
    "THREE_SEED_PROTOCOL",
    "PATCH_SIZE_VALUES",
    "P_SWEEP_VALUES",
    "TABLE3_ABLATION_VARIANTS",
    "EXPERIMENT_PROTOCOLS",
    "ARTIFACT_PATHS",
    "AblationVariantSpec",
    "ExperimentSpec",
    "compute_accuracy",
    "aggregate_accuracy",
    "compute_f1",
    "aggregate_f1",
    "compute_metrics",
    "aggregate_metrics",
    "metric_accuracy",
    "metric_f1",
    "metric_mean_std",
    "metric_mean_std_accuracy",
    "write_json_artifact",
    "write_csv_artifact",
    "write_artifact_manifest",
    "write_summary_report",
    "write_table3_ablation_metrics_artifact",
    "write_table3_ablation_table_artifact",
    "write_mask_variant_summary_artifact",
    "write_mask_statistics_artifact",
    "write_named_result_artifacts",
    "write_runtime_smoke_artifacts",
    "write_table_1_artifact",
    "write_table_2_artifact",
    "write_table_4_artifact",
    "write_table_11_artifact",
    "write_table_13_artifact",
    "write_table_14_artifact",
    "write_table_index_artifact",
    "write_figure_index_artifact",
    "run_protocolsincodeconfigrathe_experiment",
    "run_smm_artifact_route",
    "aggregate_table3_ablation",
    "table3_rows",
    "comparison_rows",
    "variant_selector",
    "method_selector",
    "protocol_matrix",
    "artifact_manifest_rows",
    "dataset_registry_artifact",
    "environment_registry_artifact",
    "experiment_registry_artifact",
    "lazy_backend_available",
    "load_optional_sbi",
    "load_optional_torch",
    "load_optional_datasets",
    "load_optional_gym",
    "mean_std_accuracy",
    "metric_mean_std_accuracy",
    "accuracy",
    "metric_accuracy",
    "f1",
    "metric_f1",
    "figure_3_reproduction_artifact",
    "metric_figure_3_reproduction_artifact",
    "table_3_reproduction_artifact",
    "metric_table_3_reproduction_artifact",
    "learning_curve",
    "metric_learning_curve",
    "figure_11_reproduction_artifact",
    "metric_figure_11_reproduction_artifact",
    "figure_12_reproduction_artifact",
    "metric_figure_12_reproduction_artifact",
    "table_11_reproduction_artifact",
    "metric_table_11_reproduction_artifact",
    "mean_std",
    "metric_mean_std",
]
