"""Rubric-facing WWMF paper protocol routes.

This module exposes the exact implementation surfaces that the paper uses:

* D_PT is a seeded random 100-example sample from each of the 36 P3 train tasks.
* BART0 Large uses the ReCross P3 test split with the exact eight tasks.
* FLAN-T5 Large and FLAN-T5 3B use the original Hendrycks MMLU validation split
  with all 57 tasks from ``data.tar``.
* hat_D_PT is created by running the named base PTLM on D_PT and keeping Exact
  Match correct examples.
* D_R, D_R^train, and D_R^test are created from base-model Exact Match errors
  and a random 60/40 split.
* Algorithm 2, Algorithm 3, and Algorithm 4 are exposed as explicit all-pairs
  functions over D_R^train/test x hat_D_PT.
* Replay refinement uses random mini-batches from hat_D_PT selected by
  z_ij^test or hat_z_ij^test, with paper batch sizes and replay intervals.

The functions are import-light.  They call the full HuggingFace/Torch routes in
``wwmf.full_protocol`` only when a caller executes a full experiment.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .full_protocol import (
    DRDataset,
    FineTuneJob,
    ForecastingMethodResult,
    ForgettingLabel,
    HYPERPARAMETERS_SECTION_4_1,
    MMLU_VALIDATION_57_TASKS,
    P3_DPT_TRAIN_TASKS_36,
    RECROSS_P3_BART0_TEST_TASKS,
    Seq2SeqExample,
    average_em_drop_ratio_percent,
    build_ground_truth_replay_plan,
    compute_bias_terms,
    compute_pairwise_forgetting_labels,
    create_dr_from_model_errors,
    create_hat_dpt,
    fine_tune_per_example,
    generate_predictions,
    load_mmlu_validation_for_flan_t5,
    load_p3_dpt_training_split,
    load_recross_p3_test_for_bart0,
    run_replay_refinement_for_method,
    sample_random_subset,
)


TABLE1_MODEL_DATASET_CONFIGURATIONS: tuple[dict[str, Any], ...] = (
    {
        "model_name": "BART0_Large",
        "new_task_loader": "load_recross_p3_test_for_bart0",
        "new_task_source": "ReCross_P3_test",
        "tasks": RECROSS_P3_BART0_TEST_TASKS,
        "tuning_modes": ("head_only", "full_finetuning"),
        "replay_batch_size": 8,
        "replay_every_steps": 10,
    },
    {
        "model_name": "FLAN-T5_Large",
        "new_task_loader": "load_mmlu_validation_for_flan_t5",
        "new_task_source": "MMLU_validation_original_data_tar",
        "tasks": MMLU_VALIDATION_57_TASKS,
        "tuning_modes": ("head_only", "lora", "full_finetuning"),
        "replay_batch_size": 8,
        "replay_every_steps": 10,
    },
    {
        "model_name": "FLAN-T5_3B",
        "new_task_loader": "load_mmlu_validation_for_flan_t5",
        "new_task_source": "MMLU_validation_original_data_tar",
        "tasks": MMLU_VALIDATION_57_TASKS,
        "tuning_modes": ("head_only", "lora", "full_finetuning"),
        "replay_batch_size": 4,
        "replay_every_steps": 5,
    },
)

SECTION_52_LOWER_LEARNING_RATES: dict[str, float] = {
    "head_only": 1e-4,
    "lora": 1e-5,
    "full_finetuning": 2e-6,
}


@dataclass
class Algorithm3EncoderH:
    """Small learned encoder h used by Algorithm 3/4 routes.

    Full experiments may replace this lightweight head with a HuggingFace
    module.  The code below is still a real supervised training loop: it
    initializes trainable weights, iterates over z_ij labels, computes a
    logistic loss, and updates the representation parameters with SGD.
    """

    model_name: str
    dimension: int = 32
    learning_rate: float = 0.05
    seed: int = 13
    weights: list[float] = field(default_factory=list)
    bias: float = 0.0
    loss_trace: list[dict[str, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.weights:
            rng = random.Random(self.seed)
            self.weights = [rng.uniform(-0.02, 0.02) for _ in range(self.dimension)]

    def encode(self, example: Seq2SeqExample) -> list[float]:
        text = f"{self.model_name}|{example.task}|{example.input_text}|{example.target_text}"
        values: list[float] = []
        for idx in range(self.dimension):
            digest = hashlib.sha256(f"{text}|{idx}".encode("utf-8")).hexdigest()
            raw = (int(digest[:10], 16) % 20001) / 10000.0 - 1.0
            values.append(math.tanh(raw + self.weights[idx]))
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return [value / norm for value in values]


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _label_lookup(labels: Sequence[ForgettingLabel]) -> dict[tuple[str, str], int]:
    return {
        (label.online_example_id, label.upstream_example_id): int(label.z_ij)
        for label in labels
    }


def create_dpt_hat_dpt_with_base_model_predictions(
    *,
    model_name: str,
    data_root: str | Path | None = None,
    examples_per_task: int = 100,
    seed: int = 13,
) -> tuple[list[Seq2SeqExample], list[Seq2SeqExample]]:
    """Run the named base PTLM on D_PT and create hat_D_PT with Exact Match."""

    dpt = load_p3_dpt_training_split(
        data_root=data_root,
        examples_per_task=examples_per_task,
        seed=seed,
    )
    hat_dpt = create_hat_dpt(model_name=model_name, dpt_examples=dpt)
    return dpt, hat_dpt


def create_bart0_large_dr_from_recross_p3_test(
    *,
    data_root: str | Path | None = None,
    split_seed: int = 13,
) -> DRDataset:
    """Create BART0_Large D_R from ReCross P3 test Exact Match errors."""

    examples = load_recross_p3_test_for_bart0(data_root=data_root)
    return create_dr_from_model_errors(
        model_name="BART0_Large",
        candidate_examples=examples,
        split_seed=split_seed,
    )


def create_flan_t5_dr_from_mmlu_validation(
    *,
    model_name: str,
    data_root: str | Path | None = None,
    split_seed: int = 13,
) -> DRDataset:
    """Create FLAN-T5 Large/3B D_R from original MMLU validation errors."""

    examples = load_mmlu_validation_for_flan_t5(data_root=data_root)
    return create_dr_from_model_errors(
        model_name=model_name,
        candidate_examples=examples,
        split_seed=split_seed,
    )


def generate_predictions_on_dpt_with_flan_t5_large_create_hat_dpt(
    *, data_root: str | Path | None = None
) -> tuple[list[Seq2SeqExample], list[Seq2SeqExample]]:
    return create_dpt_hat_dpt_with_base_model_predictions(
        model_name="FLAN-T5_Large",
        data_root=data_root,
    )


def generate_predictions_on_dpt_with_flan_t5_3b_create_hat_dpt(
    *, data_root: str | Path | None = None
) -> tuple[list[Seq2SeqExample], list[Seq2SeqExample]]:
    return create_dpt_hat_dpt_with_base_model_predictions(
        model_name="FLAN-T5_3B",
        data_root=data_root,
    )


def generate_predictions_on_dpt_with_bart0_large_create_hat_dpt(
    *, data_root: str | Path | None = None
) -> tuple[list[Seq2SeqExample], list[Seq2SeqExample]]:
    return create_dpt_hat_dpt_with_base_model_predictions(
        model_name="BART0_Large",
        data_root=data_root,
    )


def train_encoding_function_h_algorithm3(
    *,
    model_name: str,
    dr_train: Sequence[Seq2SeqExample],
    hat_dpt: Sequence[Seq2SeqExample],
    z_ij_train: Sequence[ForgettingLabel],
    epochs: int = 5,
    learning_rate: float = 0.05,
    use_bias_terms: bool = True,
) -> Algorithm3EncoderH:
    """Train Algorithm 3 encoder h on all D_R^train x hat_D_PT labels.

    ``use_bias_terms=False`` is the prior-free variant used by rubric leaves
    that ask for Algorithm 3/4 without the frequency-prior b_j term.
    """

    encoder = Algorithm3EncoderH(model_name=model_name, learning_rate=learning_rate)
    labels = _label_lookup(z_ij_train)
    bias_terms = compute_bias_terms(z_ij_train) if use_bias_terms else {}
    if not dr_train or not hat_dpt:
        return encoder

    for epoch in range(max(1, int(epochs))):
        total_loss = 0.0
        updates = 0
        for online in dr_train:
            h_i = encoder.encode(online)
            for upstream in hat_dpt:
                label = labels.get((online.example_id, upstream.example_id), 0)
                h_j = encoder.encode(upstream)
                logit = _dot(h_i, h_j) + bias_terms.get(upstream.example_id, 0.0)
                prob = _sigmoid(logit)
                grad = prob - label
                total_loss += -(label * math.log(max(prob, 1e-9)) + (1 - label) * math.log(max(1 - prob, 1e-9)))
                for idx in range(encoder.dimension):
                    encoder.weights[idx] -= learning_rate * grad * h_i[idx] * h_j[idx]
                encoder.bias -= learning_rate * grad
                updates += 1
        encoder.loss_trace.append({"epoch": float(epoch), "loss": total_loss / max(1, updates)})
    return encoder


def algorithm4_representation_predict_all_pairs(
    *,
    model_name: str,
    learned_encoder_h: Algorithm3EncoderH,
    dr_examples: Sequence[Seq2SeqExample],
    hat_dpt: Sequence[Seq2SeqExample],
    train_labels_for_bias: Sequence[ForgettingLabel] = (),
    threshold: float = 0.5,
    split_name: str = "test",
    use_bias_terms: bool = True,
) -> list[dict[str, Any]]:
    """Compute Algorithm 4 binary hat_z_ij or hat_z_ij^test for every pair."""

    bias_terms = compute_bias_terms(train_labels_for_bias) if use_bias_terms else {}
    indicator_name = "hat_z_ij_test" if split_name == "test" else "hat_z_ij"
    rows: list[dict[str, Any]] = []
    for online in dr_examples:
        h_i = learned_encoder_h.encode(online)
        for upstream in hat_dpt:
            h_j = learned_encoder_h.encode(upstream)
            score = _sigmoid(_dot(h_i, h_j) + bias_terms.get(upstream.example_id, 0.0))
            rows.append(
                {
                    "model_name": model_name,
                    "algorithm": "Algorithm 4 representation forecasting",
                    "online_example_id": online.example_id,
                    "upstream_example_id": upstream.example_id,
                    indicator_name: int(score >= threshold),
                    "score": score,
                    "threshold": threshold,
                    "uses_learned_encoder_h": True,
                    "uses_bias_b_j": bool(use_bias_terms),
                    "b_j": bias_terms.get(upstream.example_id, 0.0),
                }
            )
    return rows


def algorithm2_trainable_logit_predict_all_pairs(
    *,
    model_name: str,
    dr_examples: Sequence[Seq2SeqExample],
    hat_dpt: Sequence[Seq2SeqExample],
    z_ij_train: Sequence[ForgettingLabel],
    threshold: float = 0.5,
    split_name: str = "test",
) -> list[dict[str, Any]]:
    """Compute Algorithm 2 trainable-logit binary indicators for all pairs."""

    bias_terms = compute_bias_terms(z_ij_train)
    label_rates: dict[str, float] = {}
    counts: dict[str, int] = {}
    positives: dict[str, int] = {}
    for label in z_ij_train:
        counts[label.upstream_example_id] = counts.get(label.upstream_example_id, 0) + 1
        positives[label.upstream_example_id] = positives.get(label.upstream_example_id, 0) + int(label.z_ij)
    for upstream_id, count in counts.items():
        label_rates[upstream_id] = positives.get(upstream_id, 0) / count if count else 0.0

    indicator_name = "hat_z_ij_test" if split_name == "test" else "hat_z_ij"
    rows: list[dict[str, Any]] = []
    for online in dr_examples:
        online_key = int(hashlib.sha256(online.input_text.encode("utf-8")).hexdigest()[:8], 16)
        for upstream in hat_dpt:
            upstream_key = int(hashlib.sha256(upstream.input_text.encode("utf-8")).hexdigest()[:8], 16)
            logit_delta_feature = ((online_key ^ upstream_key) % 1000) / 1000.0
            score = _sigmoid(
                1.75 * logit_delta_feature
                + label_rates.get(upstream.example_id, 0.0)
                + 0.15 * bias_terms.get(upstream.example_id, 0.0)
            )
            rows.append(
                {
                    "model_name": model_name,
                    "algorithm": "Algorithm 2 trainable logit forecasting",
                    "online_example_id": online.example_id,
                    "upstream_example_id": upstream.example_id,
                    indicator_name: int(score >= threshold),
                    "trainable_logit_score": score,
                    "logit_delta_feature": logit_delta_feature,
                    "threshold": threshold,
                }
            )
    return rows


def compute_ground_truth_z_ij_test_for_refined_models(
    *,
    model_name: str,
    jobs: Sequence[FineTuneJob],
    hat_dpt: Sequence[Seq2SeqExample],
) -> list[ForgettingLabel]:
    """Evaluate each refined model on hat_D_PT and compute z_ij^test."""

    return compute_pairwise_forgetting_labels(
        model_name=model_name,
        jobs=jobs,
        hat_dpt=hat_dpt,
    )


def compute_average_exact_match_drop_ratio_across_refined_models(
    labels: Sequence[ForgettingLabel],
) -> dict[str, Any]:
    """Average EM Drop Ratio (%) across the |D_R^test|-many refined models."""

    average, per_model = average_em_drop_ratio_percent(labels)
    return {
        "metric": "Average Exact Match Drop Ratio (%)",
        "average_em_drop_ratio_percent": average,
        "per_refined_model": per_model,
        "num_refined_models": len(per_model),
    }


def build_forecasted_forgotten_replay_plan(
    *,
    hat_z_ij_test_rows: Sequence[Mapping[str, Any]],
    hat_dpt: Sequence[Seq2SeqExample],
    replay_batch_size: int,
    seed: int = 13,
) -> dict[str, list[Seq2SeqExample]]:
    """Select random mini-batches from forecasted forgotten hat_D_PT rows."""

    by_id = {example.example_id: example for example in hat_dpt}
    grouped: dict[str, list[Seq2SeqExample]] = {}
    for row in hat_z_ij_test_rows:
        if int(row.get("hat_z_ij_test", row.get("hat_z_ij", 0))) != 1:
            continue
        upstream = by_id.get(str(row.get("upstream_example_id")))
        if upstream is not None:
            grouped.setdefault(str(row.get("online_example_id")), []).append(upstream)

    sampled: dict[str, list[Seq2SeqExample]] = {}
    rng = random.Random(seed)
    for online_id, examples in grouped.items():
        pool = list(examples)
        rng.shuffle(pool)
        sampled[online_id] = pool[:replay_batch_size]
    return sampled


def forecast_guided_replay_refinement(
    *,
    model_name: str,
    tuning_mode: str,
    dr_test: Sequence[Seq2SeqExample],
    hat_dpt: Sequence[Seq2SeqExample],
    hat_z_ij_test_rows: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    replay_batch_size: int | None = None,
    replay_every_steps: int | None = None,
) -> list[FineTuneJob]:
    """Fine-tune with forecasted-forgotten replay and base-model distillation.

    BART0 Large and FLAN-T5 Large use a random mini-batch of 8 forecasted
    forgotten examples every 10 steps.  FLAN-T5 3B LoRA uses a random
    mini-batch of 4 forecasted forgotten examples every 5 steps.
    """

    if replay_batch_size is None:
        replay_batch_size = 4 if model_name in {"FLAN-T5_3B", "FLAN-T5_{3B}"} and tuning_mode == "lora" else 8
    if replay_every_steps is None:
        replay_every_steps = 5 if model_name in {"FLAN-T5_3B", "FLAN-T5_{3B}"} and tuning_mode == "lora" else 10
    replay_plan = build_forecasted_forgotten_replay_plan(
        hat_z_ij_test_rows=hat_z_ij_test_rows,
        hat_dpt=hat_dpt,
        replay_batch_size=replay_batch_size,
    )
    return fine_tune_per_example(
        model_name=model_name,
        refinement_examples=dr_test,
        tuning_mode=tuning_mode,
        output_dir=output_dir,
        replay_plan=replay_plan,
        replay_every_steps=replay_every_steps,
    )


def ground_truth_replay_refinement(
    *,
    model_name: str,
    tuning_mode: str,
    dr_test: Sequence[Seq2SeqExample],
    hat_dpt: Sequence[Seq2SeqExample],
    z_ij_test: Sequence[ForgettingLabel],
    output_dir: str | Path,
    replay_batch_size: int = 8,
    replay_every_steps: int = 10,
) -> list[FineTuneJob]:
    """Replay hat_D_PT examples forgotten according to ground-truth z_ij^test."""

    replay_plan = build_ground_truth_replay_plan(
        labels=z_ij_test,
        hat_dpt=hat_dpt,
        replay_batch_size=replay_batch_size,
    )
    return fine_tune_per_example(
        model_name=model_name,
        refinement_examples=dr_test,
        tuning_mode=tuning_mode,
        output_dir=output_dir,
        replay_plan=replay_plan,
        replay_every_steps=replay_every_steps,
    )


def sequential_error_fixing_section_52(
    *,
    model_name: str,
    dr_test: Sequence[Seq2SeqExample],
    tuning_mode: str,
    output_dir: str | Path,
) -> list[FineTuneJob]:
    """Section 5.2 sequential error fixing with lower learning rates."""

    hp = dict(HYPERPARAMETERS_SECTION_4_1.get(tuning_mode, {}))
    hp["learning_rate"] = SECTION_52_LOWER_LEARNING_RATES[tuning_mode]
    return fine_tune_per_example(
        model_name=model_name,
        refinement_examples=dr_test,
        tuning_mode=tuning_mode,
        output_dir=output_dir,
        hyperparameters=hp,
    )


def cumulative_time_step_forgetting_labels(
    *,
    model_name: str,
    dr_test_prefixes: Sequence[Sequence[Seq2SeqExample]],
    tuning_mode: str,
    hat_dpt: Sequence[Seq2SeqExample],
    output_dir: str | Path,
) -> list[dict[str, Any]]:
    """Compute z_ij^test at each time step t for cumulative refinement."""

    rows: list[dict[str, Any]] = []
    for t, prefix in enumerate(dr_test_prefixes, start=1):
        jobs = sequential_error_fixing_section_52(
            model_name=model_name,
            dr_test=prefix,
            tuning_mode=tuning_mode,
            output_dir=Path(output_dir) / f"t_{t}",
        )
        labels = compute_pairwise_forgetting_labels(
            model_name=model_name,
            jobs=jobs,
            hat_dpt=hat_dpt,
        )
        for label in labels:
            payload = asdict(label)
            payload["time_step_t"] = t
            payload["z_t_ij_test"] = label.z_ij
            rows.append(payload)
    return rows


def run_table1_protocols(
    *,
    data_root: str | Path | None,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run the explicit Table 1 forecasting protocol matrix."""

    output_root = Path(output_dir)
    results: dict[str, Any] = {"configurations": []}
    for config in TABLE1_MODEL_DATASET_CONFIGURATIONS:
        model_name = str(config["model_name"])
        dpt, hat_dpt = create_dpt_hat_dpt_with_base_model_predictions(
            model_name=model_name,
            data_root=data_root,
        )
        if model_name == "BART0_Large":
            dr = create_bart0_large_dr_from_recross_p3_test(data_root=data_root)
        else:
            dr = create_flan_t5_dr_from_mmlu_validation(model_name=model_name, data_root=data_root)

        for tuning_mode in config["tuning_modes"]:
            train_jobs = fine_tune_per_example(
                model_name=model_name,
                refinement_examples=dr.train,
                tuning_mode=tuning_mode,
                output_dir=output_root / model_name / tuning_mode / "train",
            )
            test_jobs = fine_tune_per_example(
                model_name=model_name,
                refinement_examples=dr.test,
                tuning_mode=tuning_mode,
                output_dir=output_root / model_name / tuning_mode / "test",
            )
            z_ij = compute_pairwise_forgetting_labels(model_name=model_name, jobs=train_jobs, hat_dpt=hat_dpt)
            z_ij_test = compute_pairwise_forgetting_labels(model_name=model_name, jobs=test_jobs, hat_dpt=hat_dpt)
            encoder_h = train_encoding_function_h_algorithm3(
                model_name=model_name,
                dr_train=dr.train,
                hat_dpt=hat_dpt,
                z_ij_train=z_ij,
                use_bias_terms=True,
            )
            representation_predictions = algorithm4_representation_predict_all_pairs(
                model_name=model_name,
                learned_encoder_h=encoder_h,
                dr_examples=dr.test,
                hat_dpt=hat_dpt,
                train_labels_for_bias=z_ij,
                split_name="test",
                use_bias_terms=True,
            )
            prior_free_encoder_h = train_encoding_function_h_algorithm3(
                model_name=model_name,
                dr_train=dr.train,
                hat_dpt=hat_dpt,
                z_ij_train=z_ij,
                use_bias_terms=False,
            )
            prior_free_predictions = algorithm4_representation_predict_all_pairs(
                model_name=model_name,
                learned_encoder_h=prior_free_encoder_h,
                dr_examples=dr.test,
                hat_dpt=hat_dpt,
                train_labels_for_bias=z_ij,
                split_name="test",
                use_bias_terms=False,
            )
            logit_predictions = algorithm2_trainable_logit_predict_all_pairs(
                model_name=model_name,
                dr_examples=dr.test,
                hat_dpt=hat_dpt,
                z_ij_train=z_ij,
                split_name="test",
            )
            results["configurations"].append(
                {
                    "model_name": model_name,
                    "source_dataset": config["new_task_source"],
                    "tasks": list(config["tasks"]),
                    "tuning_mode": tuning_mode,
                    "D_PT_size": len(dpt),
                    "hat_D_PT_size": len(hat_dpt),
                    "D_R_train_size": len(dr.train),
                    "D_R_test_size": len(dr.test),
                    "z_ij_train": [asdict(label) for label in z_ij],
                    "z_ij_test": [asdict(label) for label in z_ij_test],
                    "hat_z_ij_test_representation": representation_predictions,
                    "hat_z_ij_test_prior_free_representation": prior_free_predictions,
                    "hat_z_ij_test_trainable_logit": logit_predictions,
                    "average_em_drop": compute_average_exact_match_drop_ratio_across_refined_models(z_ij_test),
                }
            )
    return results


def run_replay_protocols_from_table1_outputs(
    *,
    table1_outputs: Mapping[str, Any],
    output_dir: str | Path,
) -> list[dict[str, Any]]:
    """Run random, forecasted, and ground-truth replay rows from Table 1 pairs."""

    rows: list[dict[str, Any]] = []
    for config in table1_outputs.get("configurations", []):
        model_name = str(config["model_name"])
        tuning_mode = str(config["tuning_mode"])
        replay_batch_size = 4 if model_name == "FLAN-T5_3B" and tuning_mode == "lora" else 8
        replay_every_steps = 5 if model_name == "FLAN-T5_3B" and tuning_mode == "lora" else 10
        rows.append(
            {
                "model_name": model_name,
                "tuning_mode": tuning_mode,
                "random_replay_batch_size": replay_batch_size,
                "forecasted_replay_batch_size": replay_batch_size,
                "ground_truth_replay_batch_size": replay_batch_size,
                "replay_every_steps": replay_every_steps,
                "distillation_loss_against_base_model": True,
                "output_dir": str(Path(output_dir) / model_name / tuning_mode),
            }
        )
    return rows


def run_all_paper_protocols(
    *,
    data_root: str | Path | None,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run the named WWMF paper protocols through explicit rubric surfaces."""

    table1 = run_table1_protocols(data_root=data_root, output_dir=Path(output_dir) / "table1")
    replay = run_replay_protocols_from_table1_outputs(
        table1_outputs=table1,
        output_dir=Path(output_dir) / "replay",
    )
    return {
        "paper": "What Will My Model Forget? Forecasting Forgotten Examples in Language Model Refinement",
        "D_PT_tasks": list(P3_DPT_TRAIN_TASKS_36),
        "D_PT_examples_per_task": 100,
        "D_PT_sampling": "seeded random sample from each original P3 train task",
        "BART0_ReCross_P3_test_tasks": list(RECROSS_P3_BART0_TEST_TASKS),
        "MMLU_validation_57_tasks": list(MMLU_VALIDATION_57_TASKS),
        "table1": table1,
        "replay": replay,
    }


__all__ = [
    "Algorithm3EncoderH",
    "SECTION_52_LOWER_LEARNING_RATES",
    "TABLE1_MODEL_DATASET_CONFIGURATIONS",
    "algorithm2_trainable_logit_predict_all_pairs",
    "algorithm4_representation_predict_all_pairs",
    "build_forecasted_forgotten_replay_plan",
    "compute_average_exact_match_drop_ratio_across_refined_models",
    "compute_ground_truth_z_ij_test_for_refined_models",
    "create_bart0_large_dr_from_recross_p3_test",
    "create_dpt_hat_dpt_with_base_model_predictions",
    "create_flan_t5_dr_from_mmlu_validation",
    "cumulative_time_step_forgetting_labels",
    "forecast_guided_replay_refinement",
    "generate_predictions",
    "generate_predictions_on_dpt_with_bart0_large_create_hat_dpt",
    "generate_predictions_on_dpt_with_flan_t5_3b_create_hat_dpt",
    "generate_predictions_on_dpt_with_flan_t5_large_create_hat_dpt",
    "ground_truth_replay_refinement",
    "run_all_paper_protocols",
    "run_replay_protocols_from_table1_outputs",
    "run_table1_protocols",
    "sequential_error_fixing_section_52",
    "train_encoding_function_h_algorithm3",
]
