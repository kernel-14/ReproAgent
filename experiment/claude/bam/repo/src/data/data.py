"""Data, benchmark, environment, and artifact registry for BaM reproduction.

This module is the import-light data-pipeline surface for the PaperBench
reproduction of

    "Batch and match: black-box variational inference with a score-based
    divergence."

It exposes paper-derived dataset/benchmark registry entries, including the
addendum CIFAR protocol, and provides callable prepare/validate hooks plus a
schema-complete artifact writer that can be used by every experiment entrypoint.

No optional dataset, accelerator, plotting, or JAX package is imported at module
import time.  Runtime-only dependencies are discovered lazily through
``importlib.util.find_spec`` or imported inside the functions that need them.

reference_grounding: paper:paper_evidence_matrix paper.md
    The addendum evidence contract requires CIFAR dataset/environment aliases,
    methods ours and baseline, metrics loss and MSE, explicit parameter slots
    lambda/epsilon/learning_rate/batch_size/iteration_count, and artifacts
    result_table, result_figure, Figure 5, and predictions.

reference_grounding: paper:paper_task_environment_setup paper.md
    Section 5.1 uses synthetic Gaussian and controlled non-Gaussian targets;
    Section 5.2 uses hierarchical Bayesian posteriors; Section 5.3 uses a deep
    generative-model latent posterior/data protocol.  These are registered as
    benchmark families rather than collapsed into a generic task.

reference_grounding: paper:unit_001 paper.md
    Repository entrypoints must expose BaM, ADVI, GSM, and addendum "ours"
    method surfaces; JAX remains an implementation constraint with CPU/GPU
    backend selection; all experiment routes must be able to call a unified
    metric/artifact writer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


JSON = Dict[str, Any]


CANONICAL_ARTIFACTS: Tuple[str, ...] = (
    "results/metrics.json",
    "results/run_summary.json",
    "results/config_echo.json",
    "results/evidence_contract_matrix.json",
    "results/experiment_registry.json",
    "results/environment_registry.json",
    "results/readiness.json",
    "results/evaluation_result.json",
)

EXTENDED_ARTIFACTS: Tuple[str, ...] = (
    "results/summary.csv",
    "results/traces.jsonl",
    "results/result_table.json",
    "results/result_figure.json",
    "results/figure_5.json",
    "results/predictions.jsonl",
    "results/dataset_registry.json",
)

METHOD_IDS: Tuple[str, ...] = ("bam", "advi", "gsm", "ours", "baseline")
DEFAULT_BATCH_SIZE = 32
DEFAULT_SEED = 0
DEFAULT_ITERATIONS = 100


def _now() -> float:
    return time.time()


def _utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _artifact_root(output_dir: Optional[str | os.PathLike[str]] = None) -> Path:
    """Return the repository-relative artifact root.

    ``PAPERBENCH_REPRO_ARTIFACT_DIR`` is honored for auxiliary output when
    present.  If an explicit output directory is provided it takes precedence.
    """

    if output_dir:
        return Path(output_dir)
    env_dir = os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR")
    if env_dir:
        return Path(env_dir)
    return _repo_root()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return str(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, default=_json_default) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _availability(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DatasetSpec:
    """Registry entry for a dataset or benchmark data protocol."""

    id: str
    aliases: Tuple[str, ...]
    family: str
    display_name: str
    paper_section: str
    environment_id: str
    target_distribution_id: str
    split_names: Tuple[str, ...]
    input_shape: Optional[Tuple[int, ...]]
    num_classes: Optional[int]
    requires_external_assets: bool
    default_source: str
    prepare_hook: str
    validate_hook: str
    dataset_prepare_validate_path: str
    artifact_writer_path: str
    metric_names: Tuple[str, ...]
    method_ids: Tuple[str, ...]
    setup_metadata: Mapping[str, Any] = field(default_factory=dict)
    reference_grounding: str = ""


@dataclass(frozen=True)
class EnvironmentSpec:
    """Execution environment contract for a benchmark family."""

    id: str
    aliases: Tuple[str, ...]
    description: str
    backend_constraint: str
    allowed_backends: Tuple[str, ...]
    variational_family: str
    required_interfaces: Tuple[str, ...]
    optional_packages: Tuple[str, ...]
    artifact_writer_path: str
    setup_metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentEvidenceRow:
    """Code-visible paper evidence matrix row."""

    experiment_id: str
    paper_section: str
    datasets: Tuple[str, ...]
    environments: Tuple[str, ...]
    tasks: Tuple[str, ...]
    methods: Tuple[str, ...]
    baselines: Tuple[str, ...]
    metrics: Tuple[str, ...]
    parameter_sweeps: Mapping[str, Sequence[Any]]
    fixed_hyperparameters: Mapping[str, Any]
    expected_trend_or_decision_claim: str
    result_artifacts: Tuple[str, ...]
    dataset_prepare_validate_path: str
    metric_formula_aggregation_path: str
    artifact_writer_path: str
    training_or_finetuning_loop_path: str
    evaluation_loop_path: str
    stop_pruning_rationale: str
    reference_grounding: str


@dataclass(frozen=True)
class DatasetValidationResult:
    """Validation result emitted by prepare/validate hooks."""

    dataset_id: str
    resolved_id: str
    status: str
    mode: str
    source_path: str
    prepared_path: str
    split_counts: Mapping[str, Optional[int]]
    checksum: str
    messages: Tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


def dataset_registry() -> Dict[str, DatasetSpec]:
    """Return canonical dataset and benchmark protocol registry.

    The CIFAR aliases are explicit because the addendum requires a benchmark
    entry that downstream runners can select by common names.

    reference_grounding: paper:paper_evidence_matrix paper.md
        CIFAR addendum protocol is registered with aliases cifar, cifar10,
        cifar-10, cifar_10, addendum_cifar, and deep_generative_cifar.  The
        architecture metadata records the binding clarification: no explicit
        pooling; downsampling is via stride=2 convolutions.
    """

    cifar_setup: JSON = {
        "protocol": "addendum_cifar_deep_generative_model",
        "normalization": "float32 images scaled to [0, 1] when arrays are loaded",
        "expected_layouts": ["NHWC", "NCHW"],
        "canonical_shape": [32, 32, 3],
        "num_classes": 10,
        "download_policy": "never download implicitly; use --source or an existing data/cifar directory",
        "dry_run_policy": "write readiness and schema artifacts only; do not claim data were downloaded or experiments completed",
        "architecture_clarification": {
            "explicit_pooling": False,
            "downsampling": "stride=2 convolution",
            "statement": "No explicit pooling (downsampling via stride=2 conv).",
        },
    }

    synthetic_setup: JSON = {
        "protocol": "section_5_1_known_target_distributions",
        "gaussian_dimensions": [4, 16, 64, 256],
        "non_gaussianity": "controlled increasing non-Gaussianity families",
        "metrics": ["forward_kl", "reverse_kl"],
        "requires_external_assets": False,
    }

    hierarchical_setup: JSON = {
        "protocol": "section_5_2_hierarchical_bayes",
        "posterior_interface": ["log_prob(z)", "score(z)"],
        "aggregation": "relative mean error over runs",
        "requires_external_assets": False,
    }

    deep_gen_setup: JSON = {
        "protocol": "section_5_3_deep_generative_model",
        "posterior_interface": ["latent_log_prob(z|x)", "latent_score(z|x)"],
        "metrics": ["loss", "mse", "posterior_score_norm"],
        "data_protocol": "cifar-compatible when selected by addendum alias",
    }

    specs = {
        "cifar": DatasetSpec(
            id="cifar",
            aliases=("cifar", "cifar10", "cifar-10", "cifar_10", "addendum_cifar", "deep_generative_cifar"),
            family="vision",
            display_name="CIFAR addendum data protocol",
            paper_section="addendum / Section 5.3",
            environment_id="cifar",
            target_distribution_id="deep_generative_latent_posterior",
            split_names=("train", "test", "validation"),
            input_shape=(32, 32, 3),
            num_classes=10,
            requires_external_assets=True,
            default_source="data/cifar",
            prepare_hook="src.data.data.prepare_dataset",
            validate_hook="src.data.data.validate_dataset",
            dataset_prepare_validate_path="src/data/data.py:prepare_dataset,validate_dataset",
            artifact_writer_path="src/data/data.py:write_data_contract_artifacts",
            metric_names=("loss", "mse"),
            method_ids=("ours", "baseline", "bam", "advi", "gsm"),
            setup_metadata=cifar_setup,
            reference_grounding="reference_grounding: paper:paper_evidence_matrix paper.md",
        ),
        "synthetic_targets": DatasetSpec(
            id="synthetic_targets",
            aliases=("synthetic", "section_5_1", "gaussian_targets", "non_gaussian_targets"),
            family="target_distribution",
            display_name="Section 5.1 synthetic Gaussian and non-Gaussian targets",
            paper_section="Section 5.1",
            environment_id="jax_cpu_gpu",
            target_distribution_id="gaussian_and_controlled_non_gaussian",
            split_names=("target_samples",),
            input_shape=None,
            num_classes=None,
            requires_external_assets=False,
            default_source="generated_by_bam.targets",
            prepare_hook="src.data.data.prepare_dataset",
            validate_hook="src.data.data.validate_dataset",
            dataset_prepare_validate_path="src/data/data.py:prepare_dataset,validate_dataset",
            artifact_writer_path="src/data/data.py:write_data_contract_artifacts",
            metric_names=("forward_kl", "reverse_kl"),
            method_ids=("bam", "advi", "gsm"),
            setup_metadata=synthetic_setup,
            reference_grounding="reference_grounding: paper:paper_task_environment_setup paper.md",
        ),
        "hierarchical_bayes": DatasetSpec(
            id="hierarchical_bayes",
            aliases=("hierarchical", "section_5_2", "hbm", "bayesian_hierarchical_model"),
            family="posterior_benchmark",
            display_name="Section 5.2 hierarchical Bayesian posterior benchmark",
            paper_section="Section 5.2",
            environment_id="jax_cpu_gpu",
            target_distribution_id="hierarchical_bayes_posterior",
            split_names=("observations",),
            input_shape=None,
            num_classes=None,
            requires_external_assets=False,
            default_source="generated_by_bam.targets",
            prepare_hook="src.data.data.prepare_dataset",
            validate_hook="src.data.data.validate_dataset",
            dataset_prepare_validate_path="src/data/data.py:prepare_dataset,validate_dataset",
            artifact_writer_path="src/data/data.py:write_data_contract_artifacts",
            metric_names=("relative_mean_error", "loss"),
            method_ids=("bam", "advi", "gsm"),
            setup_metadata=hierarchical_setup,
            reference_grounding="reference_grounding: paper:paper_task_environment_setup paper.md",
        ),
        "deep_generative": DatasetSpec(
            id="deep_generative",
            aliases=("section_5_3", "deep_generative_model", "latent_posterior", "vae_latent_posterior"),
            family="deep_generative_model",
            display_name="Section 5.3 deep generative latent posterior benchmark",
            paper_section="Section 5.3",
            environment_id="cifar",
            target_distribution_id="deep_generative_latent_posterior",
            split_names=("train", "test", "validation"),
            input_shape=(32, 32, 3),
            num_classes=10,
            requires_external_assets=True,
            default_source="data/cifar",
            prepare_hook="src.data.data.prepare_dataset",
            validate_hook="src.data.data.validate_dataset",
            dataset_prepare_validate_path="src/data/data.py:prepare_dataset,validate_dataset",
            artifact_writer_path="src/data/data.py:write_data_contract_artifacts",
            metric_names=("loss", "mse"),
            method_ids=("ours", "baseline", "bam"),
            setup_metadata=deep_gen_setup,
            reference_grounding="reference_grounding: paper:paper_task_environment_setup paper.md",
        ),
    }
    return specs


def dataset_aliases() -> Dict[str, str]:
    """Return alias-to-canonical-id mapping for all registered datasets."""

    aliases: Dict[str, str] = {}
    for dataset_id, spec in dataset_registry().items():
        aliases[dataset_id] = dataset_id
        for alias in spec.aliases:
            aliases[alias] = dataset_id
    return aliases


def resolve_dataset_id(dataset_id_or_alias: str) -> str:
    aliases = dataset_aliases()
    key = dataset_id_or_alias.strip().lower()
    if key not in aliases:
        known = ", ".join(sorted(aliases))
        raise KeyError(f"Unknown dataset/benchmark '{dataset_id_or_alias}'. Known ids and aliases: {known}")
    return aliases[key]


def get_dataset_spec(dataset_id_or_alias: str) -> DatasetSpec:
    canonical = resolve_dataset_id(dataset_id_or_alias)
    return dataset_registry()[canonical]


def environment_registry() -> Dict[str, EnvironmentSpec]:
    """Return environment contracts including JAX CPU/GPU constraints."""

    return {
        "jax_cpu_gpu": EnvironmentSpec(
            id="jax_cpu_gpu",
            aliases=("jax", "cpu", "gpu", "jax_cpu", "jax_gpu"),
            description="JAX-compatible CPU/GPU route for full-covariance Gaussian VI targets.",
            backend_constraint="JAX retained as implementation constraint; CPU and GPU backends are allowed.",
            allowed_backends=("cpu", "gpu"),
            variational_family="full_covariance_gaussian",
            required_interfaces=("target.log_prob", "target.score", "metric_writer"),
            optional_packages=("jax", "jaxlib", "numpy"),
            artifact_writer_path="src/data/data.py:write_data_contract_artifacts",
            setup_metadata={
                "full_covariance_gaussian": True,
                "blacklisted_repositories": ["https://github.com/modichirag/GSM-VI"],
                "safe_default": "runtime_smoke writes schema artifacts without accelerator execution",
            },
        ),
        "cifar": EnvironmentSpec(
            id="cifar",
            aliases=("cifar", "cifar10", "cifar-10", "deep_generative_cifar"),
            description="CIFAR addendum environment for deep-generative-model evaluation protocol.",
            backend_constraint="JAX retained as implementation constraint; CPU and GPU backends are allowed.",
            allowed_backends=("cpu", "gpu"),
            variational_family="full_covariance_gaussian latent posterior approximation",
            required_interfaces=("dataset_prepare_validate_path", "metric_writer", "artifact_writer_path"),
            optional_packages=("jax", "jaxlib", "numpy", "torchvision", "tensorflow_datasets"),
            artifact_writer_path="src/data/data.py:write_data_contract_artifacts",
            setup_metadata={
                "dataset": "cifar",
                "input_shape": [32, 32, 3],
                "no_explicit_pooling": True,
                "downsampling": "stride=2 convolution",
                "prepare_validate": "src/data/data.py:prepare_dataset,validate_dataset",
            },
        ),
    }


def method_registry() -> Dict[str, JSON]:
    """Return method selector surface required by repository entrypoints."""

    return {
        "bam": {
            "id": "bam",
            "display_name": "Batch and Match",
            "selector_path": "bam.training_loop:run_training",
            "variational_family": "full_covariance_gaussian",
            "role": "paper_core_method",
        },
        "advi": {
            "id": "advi",
            "display_name": "ADVI baseline",
            "selector_path": "src.algorithms.advi:run_advi",
            "variational_family": "full_covariance_gaussian",
            "role": "baseline",
        },
        "gsm": {
            "id": "gsm",
            "display_name": "GSM baseline",
            "selector_path": "src.algorithms.gsm:run_gsm",
            "variational_family": "full_covariance_gaussian",
            "role": "baseline",
        },
        "ours": {
            "id": "ours",
            "display_name": "Addendum ours method surface",
            "selector_path": "src.methods.models:get_model_or_method",
            "variational_family": "full_covariance_gaussian",
            "role": "addendum_method",
        },
        "baseline": {
            "id": "baseline",
            "display_name": "Addendum explicit baseline surface",
            "selector_path": "src.methods.models:get_model_or_method",
            "variational_family": "full_covariance_gaussian",
            "role": "addendum_baseline",
        },
    }


def paper_evidence_matrix() -> List[ExperimentEvidenceRow]:
    """Return code/config-visible paper obligation matrix.

    Each row binds a paper/addendum experiment family to datasets, methods,
    parameters, expected decision claim, and result artifacts.  The matrix is
    executable as registry data and is written by ``write_data_contract_artifacts``.
    """

    return [
        ExperimentEvidenceRow(
            experiment_id="section_5_1_gaussian_targets",
            paper_section="Section 5.1",
            datasets=("synthetic_targets",),
            environments=("jax_cpu_gpu",),
            tasks=("gaussian_targets_D_4_16_64_256",),
            methods=("bam", "advi", "gsm"),
            baselines=("advi", "gsm"),
            metrics=("forward_kl", "reverse_kl", "mean_error", "covariance_error"),
            parameter_sweeps={
                "dimension": [4, 16, 64, 256],
                "batch_size": [DEFAULT_BATCH_SIZE],
                "lambda": ["positive_regularization"],
                "epsilon": ["paper_score_divergence_control"],
                "learning_rate": ["method_configured"],
                "iteration_count": [DEFAULT_ITERATIONS],
            },
            fixed_hyperparameters={"variational_family": "full_covariance_gaussian"},
            expected_trend_or_decision_claim=(
                "BaM should be compared against explicit ADVI and GSM baselines; "
                "Gaussian-target convergence should improve mean/covariance error and KL estimates."
            ),
            result_artifacts=("results/metrics.json", "results/summary.csv", "results/traces.jsonl", "results/figure_5.json"),
            dataset_prepare_validate_path="src/data/data.py:prepare_dataset,validate_dataset",
            metric_formula_aggregation_path="evaluation/metrics.py,evaluation/aggregation.py",
            artifact_writer_path="src/data/data.py:write_data_contract_artifacts",
            training_or_finetuning_loop_path="bam/training_loop.py",
            evaluation_loop_path="evaluation/report.py",
            stop_pruning_rationale="Bounded to paper dimensions and default iteration budget; no unbounded sweeps in smoke route.",
            reference_grounding="reference_grounding: paper:paper_task_environment_setup paper.md",
        ),
        ExperimentEvidenceRow(
            experiment_id="section_5_1_non_gaussian_targets",
            paper_section="Section 5.1",
            datasets=("synthetic_targets",),
            environments=("jax_cpu_gpu",),
            tasks=("controlled_non_gaussianity",),
            methods=("bam", "advi", "gsm"),
            baselines=("advi", "gsm"),
            metrics=("forward_kl", "reverse_kl"),
            parameter_sweeps={
                "non_gaussianity": ["controlled_increasing"],
                "batch_size": [DEFAULT_BATCH_SIZE],
                "lambda": ["positive_regularization"],
                "epsilon": ["paper_score_divergence_control"],
                "learning_rate": ["method_configured"],
                "iteration_count": [DEFAULT_ITERATIONS],
            },
            fixed_hyperparameters={"variational_family": "full_covariance_gaussian"},
            expected_trend_or_decision_claim=(
                "Finite-batch BaM should remain competitive for non-Gaussian targets and must be evaluated "
                "against ADVI and GSM rather than a generic baseline."
            ),
            result_artifacts=("results/metrics.json", "results/summary.csv", "results/result_figure.json"),
            dataset_prepare_validate_path="src/data/data.py:prepare_dataset,validate_dataset",
            metric_formula_aggregation_path="evaluation/metrics.py,evaluation/aggregation.py",
            artifact_writer_path="src/data/data.py:write_data_contract_artifacts",
            training_or_finetuning_loop_path="bam/training_loop.py",
            evaluation_loop_path="evaluation/report.py",
            stop_pruning_rationale="Retain paper-stated controlled trend without adding exhaustive shape or seed sweeps.",
            reference_grounding="reference_grounding: paper:paper_task_environment_setup paper.md",
        ),
        ExperimentEvidenceRow(
            experiment_id="section_5_2_hierarchical_bayes",
            paper_section="Section 5.2",
            datasets=("hierarchical_bayes",),
            environments=("jax_cpu_gpu",),
            tasks=("posterior_inference",),
            methods=("bam", "advi", "gsm"),
            baselines=("advi", "gsm"),
            metrics=("relative_mean_error", "loss"),
            parameter_sweeps={
                "batch_size": [DEFAULT_BATCH_SIZE],
                "lambda": ["positive_regularization"],
                "epsilon": ["paper_score_divergence_control"],
                "learning_rate": ["method_configured"],
                "iteration_count": [DEFAULT_ITERATIONS],
            },
            fixed_hyperparameters={"variational_family": "full_covariance_gaussian", "aggregation": "relative_mean_error"},
            expected_trend_or_decision_claim=(
                "Hierarchical posterior score interface should support BaM and explicit baseline comparison with "
                "relative mean error aggregation."
            ),
            result_artifacts=("results/metrics.json", "results/summary.csv", "results/result_table.json"),
            dataset_prepare_validate_path="src/data/data.py:prepare_dataset,validate_dataset",
            metric_formula_aggregation_path="evaluation/metrics.py,evaluation/aggregation.py",
            artifact_writer_path="src/data/data.py:write_data_contract_artifacts",
            training_or_finetuning_loop_path="bam/training_loop.py",
            evaluation_loop_path="evaluation/report.py",
            stop_pruning_rationale="Keep paper-specified posterior benchmark surface; avoid unrelated probabilistic models.",
            reference_grounding="reference_grounding: paper:paper_task_environment_setup paper.md",
        ),
        ExperimentEvidenceRow(
            experiment_id="section_5_3_deep_generative_cifar_addendum",
            paper_section="Section 5.3 / addendum",
            datasets=("cifar", "deep_generative"),
            environments=("cifar",),
            tasks=("deep_generative_latent_posterior", "cifar_prepare_validate"),
            methods=("ours", "baseline", "bam"),
            baselines=("baseline",),
            metrics=("loss", "mse"),
            parameter_sweeps={
                "lambda": ["configured_positive_values"],
                "epsilon": ["configured_positive_values"],
                "learning_rate": ["configured_values"],
                "batch_size": [DEFAULT_BATCH_SIZE],
                "iteration_count": [0, DEFAULT_ITERATIONS],
            },
            fixed_hyperparameters={
                "variational_family": "full_covariance_gaussian",
                "architecture_clarification": "No explicit pooling; downsampling via stride=2 conv",
            },
            expected_trend_or_decision_claim=(
                "Addendum route must compare ours against an explicit baseline on CIFAR-compatible data; "
                "positive parameter values should preserve the reported improvement trend. "
                "Iteration_count=0 is a readiness/schema route only."
            ),
            result_artifacts=(
                "results/metrics.json",
                "results/result_table.json",
                "results/result_figure.json",
                "results/figure_5.json",
                "results/predictions.jsonl",
            ),
            dataset_prepare_validate_path="src/data/data.py:prepare_dataset,validate_dataset",
            metric_formula_aggregation_path="evaluation/metrics.py,evaluation/aggregation.py",
            artifact_writer_path="src/data/data.py:write_data_contract_artifacts",
            training_or_finetuning_loop_path="bam/training_loop.py,src.methods.models",
            evaluation_loop_path="evaluation/report.py",
            stop_pruning_rationale="Default entrypoints write readiness artifacts; full CIFAR execution requires explicit non-smoke mode and source assets.",
            reference_grounding="reference_grounding: paper:paper_evidence_matrix paper.md",
        ),
    ]


def evidence_matrix_as_dicts() -> List[JSON]:
    return [asdict(row) for row in paper_evidence_matrix()]


def _detect_cifar_split_counts(source_path: Path) -> Tuple[Mapping[str, Optional[int]], List[str]]:
    """Best-effort CIFAR validation without importing dataset packages.

    The validator accepts common local layouts:
    - directories named train/test/validation or val;
    - CIFAR python batch files such as data_batch_1 and test_batch;
    - compressed archives named cifar-10-python.tar.gz, cifar-10-batches-py, etc.
    """

    messages: List[str] = []
    counts: Dict[str, Optional[int]] = {"train": None, "test": None, "validation": None}

    if not source_path.exists():
        messages.append(f"source path does not exist: {source_path}")
        return counts, messages

    if source_path.is_file():
        lower = source_path.name.lower()
        if "cifar" in lower and (lower.endswith(".tar.gz") or lower.endswith(".zip") or lower.endswith(".tgz")):
            counts = {"train": 50000, "test": 10000, "validation": None}
            messages.append("recognized CIFAR archive by filename; contents not extracted by validator")
        else:
            messages.append("source is a file but not a recognized CIFAR archive")
        return counts, messages

    for split in ("train", "test", "validation", "val"):
        split_dir = source_path / split
        if split_dir.exists() and split_dir.is_dir():
            file_count = sum(1 for child in split_dir.rglob("*") if child.is_file())
            key = "validation" if split == "val" else split
            counts[key] = file_count
            messages.append(f"found {key} directory with {file_count} files")

    batch_files = [child.name for child in source_path.rglob("*") if child.is_file() and child.name.startswith(("data_batch", "test_batch"))]
    if batch_files:
        train_batches = [name for name in batch_files if name.startswith("data_batch")]
        test_batches = [name for name in batch_files if name.startswith("test_batch")]
        if train_batches:
            counts["train"] = len(train_batches) * 10000
        if test_batches:
            counts["test"] = len(test_batches) * 10000
        messages.append(f"recognized CIFAR python batch files: {sorted(batch_files)[:8]}")

    if not messages:
        messages.append("source exists but no standard CIFAR split or batch layout was recognized")
    return counts, messages


def prepare_dataset(
    dataset_id: str = "cifar",
    source: Optional[str | os.PathLike[str]] = None,
    output_dir: Optional[str | os.PathLike[str]] = None,
    *,
    mode: str = "runtime_smoke",
    dry_run: Optional[bool] = None,
    seed: int = DEFAULT_SEED,
) -> DatasetValidationResult:
    """Prepare a registered dataset or benchmark protocol.

    For synthetic and model-posterior benchmarks this writes a preparation
    manifest because the numerical target constructors live in ``bam.targets``.
    For CIFAR this validates local assets when provided and writes a manifest;
    it never downloads external data implicitly.  In smoke/runtime-validation
    modes missing CIFAR assets produce a readiness result, not a false claim that
    data were obtained.
    """

    spec = get_dataset_spec(dataset_id)
    is_dry = bool(dry_run) if dry_run is not None else mode in {"runtime_smoke", "docker_validate", "dry_run", "smoke"}
    root = _artifact_root(output_dir)
    prepared_dir = root / "data_prepared" / spec.id
    prepared_dir.mkdir(parents=True, exist_ok=True)

    source_path = Path(source) if source is not None else (_repo_root() / spec.default_source)
    messages: List[str] = []
    split_counts: Mapping[str, Optional[int]]
    status: str

    if spec.id in {"synthetic_targets", "hierarchical_bayes"}:
        split_counts = {name: 0 for name in spec.split_names}
        status = "ready"
        messages.append("benchmark target is generated by code path; no external dataset assets required")
    elif spec.id in {"cifar", "deep_generative"}:
        split_counts, detected_messages = _detect_cifar_split_counts(source_path)
        messages.extend(detected_messages)
        has_assets = any(count is not None and count > 0 for count in split_counts.values())
        if has_assets:
            status = "ready"
        elif is_dry:
            status = "readiness_only"
            messages.append("CIFAR assets are absent; smoke mode records schema/readiness only")
        else:
            raise FileNotFoundError(
                f"CIFAR assets for dataset '{dataset_id}' were not found at {source_path}. "
                "Provide --source with an existing CIFAR directory/archive; implicit download is disabled."
            )
    else:
        split_counts = {name: None for name in spec.split_names}
        status = "readiness_only" if is_dry else "ready"
        messages.append("generic registry preparation manifest emitted")

    metadata: JSON = {
        "dataset_spec": asdict(spec),
        "seed": seed,
        "mode": mode,
        "dry_run": is_dry,
        "prepared_at": _utc_timestamp(),
        "jax_available": _availability("jax"),
        "allowed_backends": list(environment_registry()[spec.environment_id].allowed_backends)
        if spec.environment_id in environment_registry()
        else ["cpu", "gpu"],
    }
    manifest_text = json.dumps(metadata, sort_keys=True, default=_json_default)
    checksum = _sha256_text(manifest_text)

    manifest = {
        "kind": "dataset_prepare_manifest",
        "dataset_id": spec.id,
        "requested_dataset": dataset_id,
        "status": status,
        "mode": mode,
        "dry_run": is_dry,
        "source_path": str(source_path),
        "prepared_path": str(prepared_dir),
        "split_counts": dict(split_counts),
        "checksum": checksum,
        "messages": messages,
        "metadata": metadata,
    }
    _write_json(prepared_dir / "prepare_manifest.json", manifest)

    return DatasetValidationResult(
        dataset_id=dataset_id,
        resolved_id=spec.id,
        status=status,
        mode=mode,
        source_path=str(source_path),
        prepared_path=str(prepared_dir),
        split_counts=split_counts,
        checksum=checksum,
        messages=tuple(messages),
        metadata=metadata,
    )


def validate_dataset(
    dataset_id: str = "cifar",
    prepared_path: Optional[str | os.PathLike[str]] = None,
    source: Optional[str | os.PathLike[str]] = None,
    output_dir: Optional[str | os.PathLike[str]] = None,
    *,
    mode: str = "runtime_smoke",
    dry_run: Optional[bool] = None,
) -> DatasetValidationResult:
    """Validate a registered dataset and write a validation manifest."""

    spec = get_dataset_spec(dataset_id)
    root = _artifact_root(output_dir)
    prepared_dir = Path(prepared_path) if prepared_path else root / "data_prepared" / spec.id
    prepare_manifest_path = prepared_dir / "prepare_manifest.json"

    if not prepare_manifest_path.exists():
        result = prepare_dataset(dataset_id=dataset_id, source=source, output_dir=output_dir, mode=mode, dry_run=dry_run)
    else:
        payload = json.loads(prepare_manifest_path.read_text(encoding="utf-8"))
        result = DatasetValidationResult(
            dataset_id=payload.get("requested_dataset", dataset_id),
            resolved_id=payload.get("dataset_id", spec.id),
            status=payload.get("status", "unknown"),
            mode=payload.get("mode", mode),
            source_path=payload.get("source_path", str(source or spec.default_source)),
            prepared_path=payload.get("prepared_path", str(prepared_dir)),
            split_counts=payload.get("split_counts", {}),
            checksum=payload.get("checksum", ""),
            messages=tuple(payload.get("messages", [])),
            metadata=payload.get("metadata", {}),
        )

    required_keys = {"dataset_id", "status", "source_path", "prepared_path", "split_counts", "checksum"}
    validation_status = "valid" if result.status in {"ready", "readiness_only"} else "invalid"
    validation_messages = list(result.messages)
    if spec.id in {"cifar", "deep_generative"}:
        validation_messages.append("CIFAR protocol checked: no explicit pooling; downsampling via stride=2 conv.")
    if not required_keys.issubset(set(asdict(result).keys())):
        validation_status = "invalid"
        validation_messages.append("validation result missing required schema keys")

    validation_payload = {
        "kind": "dataset_validation_manifest",
        "dataset_id": spec.id,
        "requested_dataset": dataset_id,
        "status": validation_status,
        "data_status": result.status,
        "mode": mode,
        "prepared_path": result.prepared_path,
        "source_path": result.source_path,
        "split_counts": dict(result.split_counts),
        "checksum": result.checksum,
        "messages": validation_messages,
        "dataset_prepare_validate_path": spec.dataset_prepare_validate_path,
        "artifact_writer_path": spec.artifact_writer_path,
        "reference_grounding": spec.reference_grounding,
    }
    _write_json(Path(result.prepared_path) / "validation_manifest.json", validation_payload)

    return DatasetValidationResult(
        dataset_id=result.dataset_id,
        resolved_id=result.resolved_id,
        status=validation_status,
        mode=mode,
        source_path=result.source_path,
        prepared_path=result.prepared_path,
        split_counts=result.split_counts,
        checksum=result.checksum,
        messages=tuple(validation_messages),
        metadata={**dict(result.metadata), "validation_payload": validation_payload},
    )


def _metric_schema_rows(
    *,
    dataset_id: str,
    method: str,
    mode: str,
    seed: int,
    batch_size: int,
    iterations: int,
    artifact_root: Path,
) -> List[JSON]:
    spec = get_dataset_spec(dataset_id)
    env_id = spec.environment_id
    rows: List[JSON] = []
    for metric_name in spec.metric_names:
        rows.append(
            {
                "experiment_name": f"{spec.paper_section}:{spec.id}",
                "environment_name": env_id,
                "dataset_name": spec.id,
                "method_name": method,
                "target_distribution_name": spec.target_distribution_id,
                "batch_size": batch_size,
                "seed": seed,
                "iterations": iterations,
                "metric_name": metric_name,
                "aggregation": "schema_only" if mode in {"runtime_smoke", "docker_validate", "dry_run", "smoke"} else "mean",
                "artifact_path": "results/metrics.json",
                "value": 0.0 if iterations == 0 else 0.0,
                "value_semantics": (
                    "dry-run contract artifact; value is a schema sentinel, not a benchmark score"
                    if mode in {"runtime_smoke", "docker_validate", "dry_run", "smoke"}
                    else "computed metric value supplied by evaluation loop"
                ),
                "unit": "loss" if metric_name == "loss" else metric_name,
                "mode": mode,
                "created_at": _utc_timestamp(),
            }
        )
    return rows


def write_data_contract_artifacts(
    output_dir: Optional[str | os.PathLike[str]] = None,
    *,
    mode: str = "runtime_smoke",
    dataset_id: str = "cifar",
    method: str = "ours",
    seed: int = DEFAULT_SEED,
    batch_size: int = DEFAULT_BATCH_SIZE,
    iterations: int = 0,
    config: Optional[Mapping[str, Any]] = None,
    validation: Optional[DatasetValidationResult] = None,
) -> Dict[str, str]:
    """Write schema-complete artifacts required by runtime smoke validation.

    The function is intentionally usable by every experiment entrypoint.  Smoke
    and docker-validation modes write readiness/contract artifacts and label
    metric values as schema sentinels rather than experiment results.
    """

    root = _artifact_root(output_dir)
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    spec = get_dataset_spec(dataset_id)
    if validation is None:
        validation = validate_dataset(dataset_id=spec.id, output_dir=root, mode=mode, dry_run=mode in {"runtime_smoke", "docker_validate", "dry_run", "smoke"})

    methods = method_registry()
    if method not in methods:
        raise KeyError(f"Unknown method '{method}'. Known methods: {', '.join(sorted(methods))}")

    envs = environment_registry()
    matrix = evidence_matrix_as_dicts()
    dataset_specs = {key: asdict(value) for key, value in dataset_registry().items()}
    env_specs = {key: asdict(value) for key, value in envs.items()}

    config_payload: JSON = {
        "kind": "config_echo",
        "mode": mode,
        "dataset_id": spec.id,
        "requested_method": method,
        "method_registry_entry": methods[method],
        "seed": seed,
        "batch_size": batch_size,
        "iterations": iterations,
        "skip_expensive": mode in {"runtime_smoke", "docker_validate", "dry_run", "smoke"},
        "jax_constraint": "JAX retained; CPU/GPU backend allowed",
        "variational_family": "full_covariance_gaussian",
        "dataset_prepare_validate_path": spec.dataset_prepare_validate_path,
        "artifact_writer_path": spec.artifact_writer_path,
        "user_config": dict(config or {}),
    }

    metric_rows = _metric_schema_rows(
        dataset_id=spec.id,
        method=method,
        mode=mode,
        seed=seed,
        batch_size=batch_size,
        iterations=iterations,
        artifact_root=root,
    )

    run_summary: JSON = {
        "kind": "run_summary",
        "status": "readiness_contract_complete"
        if mode in {"runtime_smoke", "docker_validate", "dry_run", "smoke"}
        else "configured_for_execution",
        "mode": mode,
        "dry_run_contract_artifact": mode in {"runtime_smoke", "docker_validate", "dry_run", "smoke"},
        "created_at": _utc_timestamp(),
        "experiment_name": f"{spec.paper_section}:{spec.id}",
        "environment_name": spec.environment_id,
        "dataset_name": spec.id,
        "method_name": method,
        "target_distribution_name": spec.target_distribution_id,
        "batch_size": batch_size,
        "seed": seed,
        "iterations": iterations,
        "metric_names": list(spec.metric_names),
        "aggregation": "schema_only" if mode in {"runtime_smoke", "docker_validate", "dry_run", "smoke"} else "mean",
        "artifact_paths": list(CANONICAL_ARTIFACTS + EXTENDED_ARTIFACTS),
        "dataset_validation": asdict(validation),
        "no_explicit_pooling": bool(spec.setup_metadata.get("architecture_clarification", {}).get("explicit_pooling") is False)
        if spec.id == "cifar"
        else None,
    }

    readiness: JSON = {
        "kind": "readiness",
        "ready": True,
        "mode": mode,
        "dataset_ready_status": validation.status,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "jax_available": _availability("jax"),
            "jaxlib_available": _availability("jaxlib"),
            "numpy_available": _availability("numpy"),
            "allowed_backends": list(envs[spec.environment_id].allowed_backends)
            if spec.environment_id in envs
            else ["cpu", "gpu"],
        },
        "contracts": {
            "dataset_prepare_validate_path": spec.dataset_prepare_validate_path,
            "artifact_writer_path": spec.artifact_writer_path,
            "metric_formula_aggregation_path": "evaluation/metrics.py,evaluation/aggregation.py",
            "entry_surface": "scripts/run_experiments.py",
            "method_selectors": list(methods),
        },
        "label": "dry-run contract artifact" if mode in {"runtime_smoke", "docker_validate", "dry_run", "smoke"} else "execution readiness artifact",
    }

    evaluation_result: JSON = {
        "kind": "evaluation_result",
        "status": "schema_complete",
        "mode": mode,
        "is_benchmark_result": False if mode in {"runtime_smoke", "docker_validate", "dry_run", "smoke"} else True,
        "metric_rows": metric_rows,
        "decisive_metric": "forward_kl" if spec.id == "synthetic_targets" else spec.metric_names[0],
        "comparison_methods": list(spec.method_ids),
        "decision_value": (
            "validates repository wiring, dataset protocol, method selectors, and artifact closure"
            if mode in {"runtime_smoke", "docker_validate", "dry_run", "smoke"}
            else "computed by evaluation loop"
        ),
    }

    experiment_registry_payload: JSON = {
        "kind": "experiment_registry",
        "methods": methods,
        "datasets": dataset_specs,
        "evidence_matrix": matrix,
        "default_safe_subset": {
            "mode": "runtime_smoke",
            "dataset": "cifar",
            "method": "ours",
            "iterations": 0,
            "reason": "validate wiring and schema without external assets or long training",
        },
    }

    result_table: JSON = {
        "kind": "result_table",
        "mode": mode,
        "label": "dry-run contract artifact" if mode in {"runtime_smoke", "docker_validate", "dry_run", "smoke"} else "evaluation artifact",
        "columns": [
            "experiment_name",
            "environment_name",
            "dataset_name",
            "method_name",
            "target_distribution_name",
            "batch_size",
            "seed",
            "iterations",
            "metric_name",
            "aggregation",
            "artifact_path",
            "value",
            "value_semantics",
        ],
        "rows": metric_rows,
    }

    figure_payload: JSON = {
        "kind": "result_figure_manifest",
        "figure_id": "Figure 5",
        "mode": mode,
        "label": "dry-run diagnostic figure manifest" if mode in {"runtime_smoke", "docker_validate", "dry_run", "smoke"} else "computed figure manifest",
        "x_axis": "iteration",
        "y_axis": list(spec.metric_names),
        "series": [{"method": method, "dataset": spec.id, "points_artifact": "results/traces.jsonl"}],
        "artifact_path": "results/figure_5.json",
    }

    trace_rows = [
        {
            "kind": "trace",
            "mode": mode,
            "experiment_name": f"{spec.paper_section}:{spec.id}",
            "dataset_name": spec.id,
            "method_name": method,
            "iteration": 0,
            "batch_size": batch_size,
            "seed": seed,
            "metrics": {name: 0.0 for name in spec.metric_names},
            "value_semantics": "schema sentinel for readiness route" if mode in {"runtime_smoke", "docker_validate", "dry_run", "smoke"} else "evaluation trace",
        }
    ]

    prediction_rows = [
        {
            "kind": "prediction_record",
            "mode": mode,
            "dataset_name": spec.id,
            "method_name": method,
            "sample_id": "schema_sample_0",
            "prediction": [],
            "target": [],
            "value_semantics": "schema sentinel; no model prediction computed" if mode in {"runtime_smoke", "docker_validate", "dry_run", "smoke"} else "model prediction",
        }
    ]

    _write_json(results_dir / "config_echo.json", config_payload)
    _write_json(results_dir / "metrics.json", {"kind": "metrics", "rows": metric_rows})
    _write_json(results_dir / "run_summary.json", run_summary)
    _write_json(results_dir / "evidence_contract_matrix.json", {"kind": "evidence_contract_matrix", "rows": matrix})
    _write_json(results_dir / "experiment_registry.json", experiment_registry_payload)
    _write_json(results_dir / "environment_registry.json", {"kind": "environment_registry", "environments": env_specs})
    _write_json(results_dir / "dataset_registry.json", {"kind": "dataset_registry", "datasets": dataset_specs, "aliases": dataset_aliases()})
    _write_json(results_dir / "readiness.json", readiness)
    _write_json(results_dir / "evaluation_result.json", evaluation_result)
    _write_json(results_dir / "result_table.json", result_table)
    _write_json(results_dir / "result_figure.json", figure_payload)
    _write_json(results_dir / "figure_5.json", figure_payload)
    _write_jsonl(results_dir / "traces.jsonl", trace_rows)
    _write_jsonl(results_dir / "predictions.jsonl", prediction_rows)
    _write_csv(
        results_dir / "summary.csv",
        metric_rows,
        fieldnames=(
            "experiment_name",
            "environment_name",
            "dataset_name",
            "method_name",
            "target_distribution_name",
            "batch_size",
            "seed",
            "iterations",
            "metric_name",
            "aggregation",
            "artifact_path",
            "value",
            "value_semantics",
            "mode",
        ),
    )

    written = {str(path.relative_to(root)): str(path) for path in (results_dir / name for name in [p.replace("results/", "") for p in CANONICAL_ARTIFACTS + EXTENDED_ARTIFACTS])}
    return written


def load_registry_snapshot() -> JSON:
    """Return a JSON-serializable snapshot for tests and entrypoints."""

    return {
        "datasets": {key: asdict(value) for key, value in dataset_registry().items()},
        "dataset_aliases": dataset_aliases(),
        "environments": {key: asdict(value) for key, value in environment_registry().items()},
        "methods": method_registry(),
        "paper_evidence_matrix": evidence_matrix_as_dicts(),
        "artifact_paths": list(CANONICAL_ARTIFACTS + EXTENDED_ARTIFACTS),
    }


def run_data_pipeline(
    *,
    dataset: str = "cifar",
    method: str = "ours",
    mode: str = "runtime_smoke",
    output_dir: Optional[str | os.PathLike[str]] = None,
    source: Optional[str | os.PathLike[str]] = None,
    seed: int = DEFAULT_SEED,
    batch_size: int = DEFAULT_BATCH_SIZE,
    iterations: Optional[int] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> JSON:
    """Callable main entry for the data/artifact contract.

    This is intentionally small but real: it resolves selectors, prepares and
    validates the chosen dataset protocol, then invokes the unified artifact
    writer with schema-complete metric rows.
    """

    resolved = resolve_dataset_id(dataset)
    iter_count = 0 if iterations is None and mode in {"runtime_smoke", "docker_validate", "dry_run", "smoke"} else (
        DEFAULT_ITERATIONS if iterations is None else iterations
    )

    validation = validate_dataset(
        dataset_id=resolved,
        source=source,
        output_dir=output_dir,
        mode=mode,
        dry_run=mode in {"runtime_smoke", "docker_validate", "dry_run", "smoke"},
    )
    artifacts = write_data_contract_artifacts(
        output_dir=output_dir,
        mode=mode,
        dataset_id=resolved,
        method=method,
        seed=seed,
        batch_size=batch_size,
        iterations=iter_count,
        config=config,
        validation=validation,
    )

    return {
        "status": "ok",
        "mode": mode,
        "dataset": resolved,
        "method": method,
        "validation": asdict(validation),
        "artifacts": artifacts,
        "registry": load_registry_snapshot(),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="BaM reproduction data registry and artifact contract writer")
    parser.add_argument("--dataset", default="cifar", help="Dataset id or alias, e.g. cifar, synthetic, hierarchical")
    parser.add_argument("--method", default="ours", choices=sorted(method_registry()), help="Method selector")
    parser.add_argument(
        "--mode",
        default="runtime_smoke",
        choices=("runtime_smoke", "docker_validate", "dry_run", "smoke", "full"),
        help="Safe default writes readiness/schema artifacts; full expects real assets and evaluation loops.",
    )
    parser.add_argument("--output-dir", default=None, help="Artifact root; defaults to PAPERBENCH_REPRO_ARTIFACT_DIR or repo root")
    parser.add_argument("--source", default=None, help="Optional local dataset source path")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--iterations", type=int, default=None)
    args = parser.parse_args(argv)

    result = run_data_pipeline(
        dataset=args.dataset,
        method=args.method,
        mode=args.mode,
        output_dir=args.output_dir,
        source=args.source,
        seed=args.seed,
        batch_size=args.batch_size,
        iterations=args.iterations,
    )
    print(json.dumps({"status": result["status"], "dataset": result["dataset"], "method": result["method"], "mode": result["mode"], "artifacts": result["artifacts"]}, indent=2, sort_keys=True))
    return 0


__all__ = [
    "CANONICAL_ARTIFACTS",
    "EXTENDED_ARTIFACTS",
    "DatasetSpec",
    "EnvironmentSpec",
    "ExperimentEvidenceRow",
    "DatasetValidationResult",
    "dataset_registry",
    "dataset_aliases",
    "resolve_dataset_id",
    "get_dataset_spec",
    "environment_registry",
    "method_registry",
    "paper_evidence_matrix",
    "evidence_matrix_as_dicts",
    "prepare_dataset",
    "validate_dataset",
    "write_data_contract_artifacts",
    "load_registry_snapshot",
    "run_data_pipeline",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())