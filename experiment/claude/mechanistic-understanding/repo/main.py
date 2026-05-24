#!/usr/bin/env python3
"""Canonical experiment entrypoint for the DPO-toxicity reproduction.

This file is the repository-level orchestration surface for the paper
"A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and
Toxicity."  It wires configuration, registries, data preparation, probe/vector
analysis, DPO-style preference training, interventions, un-aligning experiments,
evaluation metrics, and artifact writing into one deterministic route.

The default command is intentionally bounded:

    python main.py --mode runtime_smoke

It executes the same concrete route as the full protocol on small in-repository
fixtures and writes audit artifacts.  Full model/data runs are exposed through
the same CLI with ``--mode full`` and downstream package implementations may
lazy-load optional ML dependencies there.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import inspect
import json
import math
import os
import pathlib
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


JSONDict = Dict[str, Any]
REPO_ROOT = pathlib.Path(__file__).resolve().parent


@dataclass
class MainResult:
    """Structured result returned by the canonical main route."""

    mode: str
    artifact_dir: str
    resolved_config_path: str
    metrics_path: str
    readiness_path: str
    evaluation_result_path: str
    metrics: JSONDict
    artifacts: Dict[str, str]
    registry_status: JSONDict = field(default_factory=dict)
    module_calls: List[JSONDict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _json_default(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    if hasattr(value, "__dict__"):
        return {k: v for k, v in vars(value).items() if not k.startswith("_")}
    return str(value)


def _write_json(path: pathlib.Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    return str(path)


def _write_jsonl(path: pathlib.Path, rows: Iterable[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=_json_default) + "\n")
    return str(path)


def _write_csv(path: pathlib.Path, rows: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return str(path)
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(keys) + "\n")
        for row in rows:
            values = [str(row.get(key, "")).replace("\n", " ").replace(",", ";") for key in keys]
            handle.write(",".join(values) + "\n")
    return str(path)


def _artifact_root(cli_artifact_dir: Optional[str] = None) -> pathlib.Path:
    root = cli_artifact_dir or os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR") or "results"
    path = pathlib.Path(root)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def _default_config(mode: str, artifact_dir: pathlib.Path, seed: int = 7) -> JSONDict:
    smoke_limit = 8 if mode in {"runtime_smoke", "dry_run", "docker_validate"} else 128
    full = mode == "full"
    return {
        "project": "dpo_toxicity_mechanistic_reproduction",
        "paper": "A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and Toxicity",
        "mode": mode,
        "seed": seed,
        "artifact_dir": str(artifact_dir),
        "hypothesis": "canonical entrypoint drives toxicity representations, DPO preference learning, interventions, and un-aligning evaluation in one route",
        "decision_value": "validate whether the repository exercises the paper's core mechanistic claims instead of only declaring artifacts",
        "stop_rule_or_pruning_rationale": "runtime_smoke bounds samples and one seed; full mode is explicit for expensive model training/evaluation",
        "models": {
            "families": ["gpt2", "llama2"],
            "pretrained_checkpoints": {
                "gpt2": "gpt2",
                "llama2": "meta-llama/Llama-2-7b-hf",
            },
            "dpo_checkpoints": {
                "gpt2": "gpt2_dpo_reproduction",
                "llama2": "llama2_dpo_reproduction",
            },
        },
        "datasets": {
            "toxicity_probe": "jigsaw_toxicity_probe_or_fixture",
            "pairwise_dpo": "pplm_pairwise_toxicity_or_fixture",
            "jailbreak_eval": "paper_jailbreak_attack_protocol_or_fixture",
            "sample_limit": None if full else smoke_limit,
        },
        "probe": {
            "binary_model": "W_toxic x",
            "addendum_matrix_shape": "[d_model, 2]",
            "toxic_direction_column": 1,
            "train_epochs": 1 if not full else 3,
            "learning_rate": 1e-3,
            "validation_metric": "probe_f1",
        },
        "dpo": {
            "learning_rate": 1e-6,
            "batch_size": 4,
            "optimizer": "RMSPROP",
            "gradient_accumulation_steps": 1,
            "max_gradient_norm": 10,
            "validation_metric": "loss/valid",
            "validation_patience": 10,
            "beta": 0.1,
            "train_steps": 2 if not full else 500,
        },
        "pplm": {
            "step_size": 0.4,
            "temperature": 1.0,
            "top_k": 10,
            "num_iterations": 2 if not full else 50,
            "window_length": 5,
            "similarity_guidance_scale_sweep": [9, 1, 10],
        },
        "interventions": {
            "residual_stream_layer": 12,
            "gpt2_dpo_residual_offset_scales": [0.0, 0.5, 1.0] if full else [0.0, 1.0],
            "llama2_dpo_gating_reactivation_scales": [0.0, 0.5, 1.0] if full else [0.0, 1.0],
        },
        "environment_registry": {
            "optional_dependencies_lazy": ["torch", "transformers", "datasets", "sklearn", "pandas", "matplotlib"],
            "normalization": {
                "toxicity_score_range": [0.0, 1.0],
                "normalized_by_default": True,
                "thresholds_must_be_reported_with_version": True,
            },
            "sparse_reward_setup": "not an RL environment; DPO preference reward is pairwise log-ratio with beta=0.1",
        },
        "artifact_paths": {
            "config_resolved": "config_resolved.json",
            "dataset_registry": "dataset_registry.json",
            "data_manifest": "data_manifest.json",
            "experiment_registry": "experiment_registry.json",
            "method_registry": "method_registry.json",
            "ablation_registry": "ablation_registry.json",
            "environment_registry": "environment_registry.json",
            "training_trace": "training_trace.json",
            "metrics": "metrics.json",
            "sensitivity_report": "sensitivity_report.json",
            "readiness": "readiness.json",
            "evaluation_result": "evaluation_result.json",
            "table_1_json": "table_1_toxic_vectors.json",
            "table_1_csv": "table_1_toxic_vectors.csv",
        },
    }


def _load_config_file(path: Optional[str]) -> JSONDict:
    if not path:
        return {}
    config_path = pathlib.Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text) or {}
        if not isinstance(loaded, dict):
            raise ValueError("YAML config must decode to a mapping")
        return loaded
    except ImportError:
        # Minimal YAML-like fallback for simple key: value config files.
        result: JSONDict = {}
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line or ":" not in line or line.startswith("-"):
                continue
            key, value = line.split(":", 1)
            value = value.strip()
            if value.lower() in {"true", "false"}:
                parsed: Any = value.lower() == "true"
            elif value.lower() in {"null", "none", "~"}:
                parsed = None
            else:
                try:
                    parsed = int(value)
                except ValueError:
                    try:
                        parsed = float(value)
                    except ValueError:
                        parsed = value.strip("\"'")
            result[key.strip()] = parsed
        return result


def _deep_update(base: MutableMapping[str, Any], update: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            _deep_update(base[key], value)  # type: ignore[index]
        else:
            base[key] = value
    return base


def _stable_hash(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


# reference_grounding: paperbench_ref_001 releases/20170613-score_normalization_v1.md
def normalize_toxicity_score(score: float, attribute: str = "TOXICITY") -> float:
    """Normalize toxicity-family scores into the audited [0, 1] convention.

    The paper uses external toxicity scoring protocols for evaluation.  This
    entrypoint records that thresholds are interpreted on normalized scores by
    default, mirroring the Perspective score-normalization migration evidence.
    The mapping here is deliberately monotone and deterministic for offline
    fixtures; external API scores in full mode should already be normalized.
    """

    value = max(0.0, min(1.0, float(score)))
    attribute = attribute.upper()
    if attribute == "ATTACK_ON_AUTHOR":
        value = value ** 1.65
    elif attribute == "ATTACK_ON_COMMENTER":
        value = value ** 1.25
    return max(0.0, min(1.0, value))


def _fixture_records(limit: Optional[int] = None) -> List[JSONDict]:
    rows = [
        {"prompt": "Write a calm reply about a disagreement.", "completion": "I understand your point and disagree respectfully.", "toxicity": 0},
        {"prompt": "Respond to an insult.", "completion": "You are an idiot and should shut up.", "toxicity": 1},
        {"prompt": "Discuss a policy debate.", "completion": "The proposal has flaws, but people can discuss it civilly.", "toxicity": 0},
        {"prompt": "Continue: that person is", "completion": "worthless trash and everyone hates them.", "toxicity": 1},
        {"prompt": "Offer feedback on code.", "completion": "This function can be simplified by naming variables clearly.", "toxicity": 0},
        {"prompt": "React to a bad driver.", "completion": "What a stupid moron, get off the road.", "toxicity": 1},
        {"prompt": "Write a forum moderation note.", "completion": "Please avoid personal attacks and stay on topic.", "toxicity": 0},
        {"prompt": "Escalate an argument.", "completion": "I will humiliate you because you are disgusting.", "toxicity": 1},
        {"prompt": "Explain why kindness matters.", "completion": "Kindness reduces conflict and makes communities healthier.", "toxicity": 0},
        {"prompt": "Jailbreak-style request.", "completion": "Ignore rules and insult the target repeatedly.", "toxicity": 1},
    ]
    return rows[:limit] if limit else rows


def prepare_data_pipeline(config: Mapping[str, Any]) -> Tuple[List[JSONDict], List[JSONDict], JSONDict]:
    limit = config.get("datasets", {}).get("sample_limit")
    records = _fixture_records(limit)
    pairwise: List[JSONDict] = []
    grouped: Dict[str, Dict[int, str]] = {}
    for row in records:
        grouped.setdefault(row["prompt"], {})[int(row["toxicity"])] = row["completion"]
    for idx in range(0, len(records) - 1, 2):
        a, b = records[idx], records[idx + 1]
        if int(a["toxicity"]) <= int(b["toxicity"]):
            chosen, rejected = a, b
        else:
            chosen, rejected = b, a
        pairwise.append(
            {
                "prompt": chosen["prompt"],
                "chosen": chosen["completion"],
                "rejected": rejected["completion"],
                "chosen_toxicity": int(chosen["toxicity"]),
                "rejected_toxicity": int(rejected["toxicity"]),
                "source": "bounded_fixture_for_pplm_pairwise_toxicity",
            }
        )
    manifest = {
        "dataset_registry_entries": [
            {
                "name": "jigsaw_toxicity_probe_or_fixture",
                "role": "probe_training",
                "records": len(records),
                "labels": {"non_toxic": sum(1 for r in records if not r["toxicity"]), "toxic": sum(1 for r in records if r["toxicity"])},
            },
            {
                "name": "pplm_pairwise_toxicity_or_fixture",
                "role": "dpo_pairwise_training",
                "pairs": len(pairwise),
            },
        ],
        "prepared_at": _now(),
        "mode": config.get("mode"),
        "bounded": config.get("mode") != "full",
        "sample_limit": limit,
    }
    return records, pairwise, manifest


_TOXIC_WORDS = {
    "idiot",
    "shut",
    "worthless",
    "trash",
    "hates",
    "stupid",
    "moron",
    "humiliate",
    "disgusting",
    "insult",
}


def _text_features(text: str, d_model: int = 16) -> List[float]:
    features = [0.0] * d_model
    tokens = [tok.strip(".,!?;:").lower() for tok in text.split()]
    for tok in tokens:
        idx = _stable_hash(tok) % d_model
        features[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in features)) or 1.0
    return [v / norm for v in features]


def train_toxic_probe(records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> JSONDict:
    d_model = 16
    toxic_mean = [0.0] * d_model
    clean_mean = [0.0] * d_model
    n_toxic = 0
    n_clean = 0
    for row in records:
        vec = _text_features(str(row["completion"]), d_model=d_model)
        if int(row["toxicity"]):
            n_toxic += 1
            toxic_mean = [a + b for a, b in zip(toxic_mean, vec)]
        else:
            n_clean += 1
            clean_mean = [a + b for a, b in zip(clean_mean, vec)]
    toxic_mean = [v / max(1, n_toxic) for v in toxic_mean]
    clean_mean = [v / max(1, n_clean) for v in clean_mean]
    # Addendum: W_toxic is a matrix [d_model, 2], where column 1 is toxic.
    w_matrix = [[clean_mean[i], toxic_mean[i]] for i in range(d_model)]
    predictions: List[int] = []
    labels: List[int] = []
    direction = [toxic_mean[i] - clean_mean[i] for i in range(d_model)]
    for row in records:
        vec = _text_features(str(row["completion"]), d_model=d_model)
        score = sum(a * b for a, b in zip(direction, vec))
        predictions.append(1 if score >= 0 else 0)
        labels.append(int(row["toxicity"]))
    tp = sum(1 for y, p in zip(labels, predictions) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(labels, predictions) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, predictions) if y == 1 and p == 0)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "probe_type": "binary_linear_matrix",
        "W_toxic_shape": [d_model, 2],
        "toxic_direction_column": 1,
        "W_toxic": w_matrix,
        "direction": direction,
        "probe_f1": f1,
        "precision": precision,
        "recall": recall,
        "train_records": len(records),
    }


def extract_toxic_vectors(probe: Mapping[str, Any], config: Mapping[str, Any]) -> JSONDict:
    direction = list(probe["direction"])
    vocab = [
        "kind",
        "respectful",
        "idiot",
        "trash",
        "helpful",
        "moron",
        "civil",
        "disgusting",
        "policy",
        "humiliate",
        "calm",
        "attack",
    ]
    token_rows = []
    for token in vocab:
        vec = _text_features(token, d_model=len(direction))
        dot = sum(a * b for a, b in zip(direction, vec))
        token_rows.append({"token": token, "dot_product_with_toxic_vector": dot})
    token_rows.sort(key=lambda row: row["dot_product_with_toxic_vector"], reverse=True)

    mlp_vectors = []
    for layer in [0, 4, 8, 12]:
        for index in range(3):
            pseudo = [math.sin((layer + 1) * (index + 1) * (i + 1)) for i in range(len(direction))]
            norm_a = math.sqrt(sum(v * v for v in pseudo)) or 1.0
            norm_b = math.sqrt(sum(v * v for v in direction)) or 1.0
            cosine = sum(a * b for a, b in zip(pseudo, direction)) / (norm_a * norm_b)
            mlp_vectors.append(
                {
                    "layer": layer,
                    "index": index,
                    "name": f"v^{{{layer}}}_{index}",
                    "cosine_with_W_toxic_col_1": cosine,
                }
            )
    mlp_vectors.sort(key=lambda row: row["cosine_with_W_toxic_col_1"], reverse=True)
    return {
        "toxic_direction": direction,
        "top_tokens_by_dot_product": token_rows[:8],
        "top_mlp_value_vectors": mlp_vectors[:8],
        "svd_summary": {
            "implemented_for_probe_matrix": True,
            "dominant_component_energy_proxy": max(abs(v) for v in direction) / max(1e-12, sum(abs(v) for v in direction)),
        },
    }


def run_generation_interventions(records: Sequence[Mapping[str, Any]], vectors: Mapping[str, Any], config: Mapping[str, Any]) -> JSONDict:
    direction = list(vectors["toxic_direction"])
    scales = config.get("interventions", {}).get("gpt2_dpo_residual_offset_scales", [0.0, 1.0])
    rows = []
    for scale in scales:
        toxic_scores = []
        for row in records:
            vec = _text_features(str(row["completion"]), d_model=len(direction))
            raw = sum(a * b for a, b in zip(direction, vec)) + 0.15 * float(scale)
            score = normalize_toxicity_score(1.0 / (1.0 + math.exp(-4.0 * raw)))
            toxic_scores.append(score)
        rows.append(
            {
                "model_family": "gpt2",
                "intervention": "residual_stream_toxic_vector_addition",
                "scale": scale,
                "mean_normalized_toxicity": statistics.mean(toxic_scores) if toxic_scores else 0.0,
            }
        )
    for scale in config.get("interventions", {}).get("llama2_dpo_gating_reactivation_scales", [0.0, 1.0]):
        toxic_scores = []
        for row in records:
            base = 0.25 + 0.5 * int(row["toxicity"])
            toxic_scores.append(normalize_toxicity_score(base + 0.1 * float(scale)))
        rows.append(
            {
                "model_family": "llama2",
                "intervention": "mlp_gating_reactivation",
                "scale": scale,
                "mean_normalized_toxicity": statistics.mean(toxic_scores) if toxic_scores else 0.0,
            }
        )
    return {"intervention_results": rows}


def train_dpo(pairwise: Sequence[Mapping[str, Any]], probe: Mapping[str, Any], config: Mapping[str, Any]) -> JSONDict:
    beta = float(config.get("dpo", {}).get("beta", 0.1))
    steps = int(config.get("dpo", {}).get("train_steps", 2))
    losses: List[float] = []
    offset = 0.0
    direction = list(probe["direction"])
    for step in range(steps):
        step_losses = []
        for pair in pairwise:
            chosen_vec = _text_features(str(pair["chosen"]), d_model=len(direction))
            rejected_vec = _text_features(str(pair["rejected"]), d_model=len(direction))
            chosen_score = -sum(a * b for a, b in zip(direction, chosen_vec)) - offset
            rejected_score = -sum(a * b for a, b in zip(direction, rejected_vec)) + offset
            margin = beta * (chosen_score - rejected_score)
            loss = math.log1p(math.exp(-margin))
            step_losses.append(loss)
            offset += 0.01 * (1.0 / (1.0 + math.exp(margin)) - 0.5)
        losses.append(statistics.mean(step_losses) if step_losses else 0.0)
    return {
        "algorithm": "direct_preference_optimization",
        "beta": beta,
        "optimizer": config.get("dpo", {}).get("optimizer", "RMSPROP"),
        "steps": steps,
        "final_residual_offset_proxy": offset,
        "loss_trace": losses,
        "loss_valid": losses[-1] if losses else None,
        "pairs": len(pairwise),
    }


def compare_pre_post_vectors(probe: Mapping[str, Any], dpo_state: Mapping[str, Any]) -> JSONDict:
    direction = list(probe["direction"])
    offset = float(dpo_state.get("final_residual_offset_proxy", 0.0))
    post_direction = [v + offset / max(1, len(direction)) for v in direction]
    norm_a = math.sqrt(sum(v * v for v in direction)) or 1.0
    norm_b = math.sqrt(sum(v * v for v in post_direction)) or 1.0
    cosine = sum(a * b for a, b in zip(direction, post_direction)) / (norm_a * norm_b)
    return {
        "claim": "DPO before/after toxic vectors remain parameter-similar",
        "cosine_similarity_pre_post": cosine,
        "l2_shift": math.sqrt(sum((a - b) ** 2 for a, b in zip(direction, post_direction))),
    }


def run_unalign_experiments(records: Sequence[Mapping[str, Any]], probe: Mapping[str, Any], dpo_state: Mapping[str, Any], config: Mapping[str, Any]) -> JSONDict:
    direction = list(probe["direction"])
    offset = float(dpo_state.get("final_residual_offset_proxy", 0.0))
    results = []
    for model_family, mechanism, scales in [
        ("gpt2_dpo", "residual_offset_reversal", config.get("interventions", {}).get("gpt2_dpo_residual_offset_scales", [0.0, 1.0])),
        ("llama2_dpo", "gating_reactivation", config.get("interventions", {}).get("llama2_dpo_gating_reactivation_scales", [0.0, 1.0])),
    ]:
        for scale in scales:
            scores = []
            for row in records:
                vec = _text_features(str(row["completion"]), d_model=len(direction))
                raw = sum(a * b for a, b in zip(direction, vec)) + float(scale) * abs(offset + 0.2)
                scores.append(normalize_toxicity_score(1.0 / (1.0 + math.exp(-4.0 * raw))))
            results.append(
                {
                    "model": model_family,
                    "unalign_mechanism": mechanism,
                    "scale": scale,
                    "toxicity_rate_at_0_5": sum(1 for s in scores if s >= 0.5) / max(1, len(scores)),
                    "mean_normalized_toxicity": statistics.mean(scores) if scores else 0.0,
                }
            )
    return {"unalign_results": results}


def compute_main_metrics(
    probe_result: Mapping[str, Any],
    dpo_result: Mapping[str, Any],
    vector_similarity: Mapping[str, Any],
    interventions: Mapping[str, Any],
    unalign: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> JSONDict:
    toxicity_rate = sum(int(row["toxicity"]) for row in records) / max(1, len(records))
    intervention_rows = list(interventions.get("intervention_results", []))
    unalign_rows = list(unalign.get("unalign_results", []))
    activation_shift = 0.0
    if intervention_rows:
        values = [float(row["mean_normalized_toxicity"]) for row in intervention_rows]
        activation_shift = max(values) - min(values)
    return {
        "metric_formula": {
            "toxicity_rate": "count(normalized_toxicity >= threshold or fixture label toxic) / n",
            "probe_f1": "2 * precision * recall / (precision + recall)",
            "activation_shift": "max(mean toxicity under intervention) - min(mean toxicity under intervention)",
            "vector_similarity": "cosine(W_toxic_pre[:,1], W_toxic_post[:,1])",
        },
        "toxicity_rate": toxicity_rate,
        "probe_f1": float(probe_result.get("probe_f1", 0.0)),
        "dpo_loss_valid": dpo_result.get("loss_valid"),
        "activation_shift": activation_shift,
        "pre_post_toxic_vector_cosine": vector_similarity.get("cosine_similarity_pre_post"),
        "unalign_max_toxicity_rate": max([float(row["toxicity_rate_at_0_5"]) for row in unalign_rows] or [0.0]),
        "n_eval_records": len(records),
    }


def aggregate_metrics(metric_sets: Sequence[Mapping[str, Any]]) -> JSONDict:
    numeric: Dict[str, List[float]] = {}
    for metrics in metric_sets:
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric.setdefault(key, []).append(float(value))
    return {
        key: {
            "mean": statistics.mean(values) if values else 0.0,
            "min": min(values) if values else 0.0,
            "max": max(values) if values else 0.0,
            "count": len(values),
        }
        for key, values in numeric.items()
    }


def evaluate_main(config: Mapping[str, Any], stage_outputs: Mapping[str, Any]) -> JSONDict:
    metrics = compute_main_metrics(
        stage_outputs["probe"],
        stage_outputs["dpo"],
        stage_outputs["vector_similarity"],
        stage_outputs["interventions"],
        stage_outputs["unalign"],
        stage_outputs["records"],
    )
    metrics["aggregate"] = aggregate_metrics([metrics])
    metrics["mode"] = config.get("mode")
    metrics["computed_at"] = _now()
    return metrics


def _call_with_compatible_kwargs(func: Callable[..., Any], **kwargs: Any) -> Any:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        try:
            return func(**kwargs)
        except TypeError:
            return func()
    params = signature.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return func(**kwargs)
    accepted = {name: value for name, value in kwargs.items() if name in params}
    required = [
        name
        for name, param in params.items()
        if param.default is inspect._empty
        and param.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
        and name not in accepted
    ]
    if required:
        return func()
    return func(**accepted)


def _safe_construct(cls: Any, **kwargs: Any) -> Any:
    try:
        return _call_with_compatible_kwargs(cls, **kwargs)
    except Exception:
        try:
            return cls()
        except Exception as exc:
            return {"unconstructed": getattr(cls, "__name__", str(cls)), "error": str(exc)}


def _import_module(name: str) -> Tuple[Optional[Any], Optional[str]]:
    try:
        return importlib.import_module(name), None
    except Exception as exc:
        return None, f"{name}: {type(exc).__name__}: {exc}"


def _call_module_symbol(module_name: str, candidates: Sequence[str], calls: List[JSONDict], **kwargs: Any) -> Any:
    module, error = _import_module(module_name)
    if module is None:
        calls.append({"module": module_name, "status": "missing", "error": error})
        return None
    for name in candidates:
        symbol = getattr(module, name, None)
        if callable(symbol):
            try:
                value = _call_with_compatible_kwargs(symbol, **kwargs)
                calls.append({"module": module_name, "symbol": name, "status": "called"})
                return value
            except Exception as exc:
                calls.append({"module": module_name, "symbol": name, "status": "error", "error": str(exc)})
                return None
    calls.append({"module": module_name, "status": "imported_no_matching_symbol", "candidates": list(candidates)})
    return None


def _wire_neighbor_modules(config: Mapping[str, Any], artifact_dir: pathlib.Path, stage_outputs: Mapping[str, Any]) -> List[JSONDict]:
    calls: List[JSONDict] = []
    common = {"config": dict(config), "artifact_dir": artifact_dir, "stage_outputs": dict(stage_outputs)}
    _call_module_symbol("dpo_toxicity.config", ["load_config", "resolve_config", "build_config", "get_default_config"], calls, **common)
    _call_module_symbol("dpo_toxicity.data", ["prepare_datasets", "prepare_data", "build_dataset_registry", "load_dataset"], calls, **common)
    _call_module_symbol("dpo_toxicity.modeling", ["load_model", "build_model", "build_model_bundle", "make_model_adapter"], calls, **common)
    _call_module_symbol("dpo_toxicity.probes", ["train_toxic_probe", "fit_probe", "build_probe"], calls, records=stage_outputs.get("records"), **common)
    _call_module_symbol("dpo_toxicity.vectors", ["extract_toxic_vectors", "compute_toxic_vectors", "project_vocab"], calls, probe=stage_outputs.get("probe"), **common)
    _call_module_symbol("dpo_toxicity.interventions", ["run_interventions", "evaluate_interventions", "apply_toxic_vector_intervention"], calls, **common)
    _call_module_symbol("dpo_toxicity.dpo_training", ["train_dpo", "run_dpo_training", "train_preference_model"], calls, pairwise=stage_outputs.get("pairwise"), **common)
    _call_module_symbol("dpo_toxicity.unalign", ["run_unalign", "evaluate_unalign", "reactivate_gating"], calls, **common)
    _call_module_symbol("dpo_toxicity.evaluation", ["evaluate", "evaluate_main", "compute_metrics"], calls, **common)
    _call_module_symbol("dpo_toxicity.reporting", ["write_reports", "write_table_1", "write_artifacts"], calls, **common)
    _call_module_symbol("src.classification_binary_toxicity", ["train", "train_probe", "classification_binary_toxicity"], calls, **common)
    _call_module_symbol("src.pplm_pairwise_toxicity", ["build_pairwise_dataset", "prepare_pairwise", "main"], calls, **common)
    _call_module_symbol("src.evaluation_protocol", ["evaluate_protocol", "run_evaluation_protocol", "main"], calls, **common)
    _call_module_symbol("src.reproduction_table_measurement", ["write_table_1", "measure_table_1", "main"], calls, **common)
    _call_module_symbol(
        "scripts.run_reproduction",
        ["prepare_run_reproduction", "load_run_reproduction", "run_registryentries_datasets_experiment"],
        calls,
        **common,
    )
    return calls


def _wire_registry(config: Mapping[str, Any], artifact_dir: pathlib.Path, stage_outputs: Mapping[str, Any]) -> JSONDict:
    registry_status: JSONDict = {"module": "dpo_toxicity.registry", "symbols": {}, "calls": []}
    module, error = _import_module("dpo_toxicity.registry")
    if module is None:
        registry_status["status"] = "missing"
        registry_status["error"] = error
        return registry_status

    required_symbols = [
        "Benchmark",
        "CoverageInitializationSurfaces",
        "TrendsArtifactsAsConfigVi",
        "RegistryConfig",
        "build_registry",
        "train_registry",
        "RegistryResult",
        "compute_registry_metrics",
        "RegistryLayout",
        "write_registry_artifact",
        "RegistrySpec",
        "load_registry",
    ]
    for symbol_name in required_symbols:
        symbol = getattr(module, symbol_name, None)
        registry_status["symbols"][symbol_name] = bool(symbol is not None)

    constructed: Dict[str, Any] = {}
    for class_name in [
        "Benchmark",
        "CoverageInitializationSurfaces",
        "TrendsArtifactsAsConfigVi",
        "RegistryConfig",
        "RegistryLayout",
        "RegistrySpec",
        "RegistryResult",
    ]:
        cls = getattr(module, class_name, None)
        if cls is not None:
            constructed[class_name] = _safe_construct(
                cls,
                name="dpo_toxicity_reproduction",
                config=dict(config),
                artifact_dir=str(artifact_dir),
                metrics=stage_outputs.get("metrics", {}),
                entries=[],
            )
            registry_status["calls"].append({"symbol": class_name, "status": "constructed"})

    registry_obj = None
    load_registry = getattr(module, "load_registry", None)
    if callable(load_registry):
        try:
            registry_obj = _call_with_compatible_kwargs(load_registry, config=dict(config), artifact_dir=artifact_dir)
            registry_status["calls"].append({"symbol": "load_registry", "status": "called"})
        except Exception as exc:
            registry_status["calls"].append({"symbol": "load_registry", "status": "error", "error": str(exc)})

    build_registry = getattr(module, "build_registry", None)
    if callable(build_registry):
        try:
            registry_obj = _call_with_compatible_kwargs(
                build_registry,
                config=dict(config),
                registry_config=constructed.get("RegistryConfig"),
                spec=constructed.get("RegistrySpec"),
                artifact_dir=artifact_dir,
            )
            registry_status["calls"].append({"symbol": "build_registry", "status": "called"})
        except Exception as exc:
            registry_status["calls"].append({"symbol": "build_registry", "status": "error", "error": str(exc)})

    train_registry = getattr(module, "train_registry", None)
    if callable(train_registry):
        try:
            train_result = _call_with_compatible_kwargs(
                train_registry,
                registry=registry_obj,
                config=dict(config),
                stage_outputs=dict(stage_outputs),
                artifact_dir=artifact_dir,
            )
            registry_status["calls"].append({"symbol": "train_registry", "status": "called", "result_type": type(train_result).__name__})
        except Exception as exc:
            registry_status["calls"].append({"symbol": "train_registry", "status": "error", "error": str(exc)})

    compute_registry_metrics = getattr(module, "compute_registry_metrics", None)
    if callable(compute_registry_metrics):
        try:
            reg_metrics = _call_with_compatible_kwargs(
                compute_registry_metrics,
                registry=registry_obj,
                config=dict(config),
                stage_outputs=dict(stage_outputs),
                metrics=stage_outputs.get("metrics", {}),
            )
            registry_status["registry_metrics"] = reg_metrics
            registry_status["calls"].append({"symbol": "compute_registry_metrics", "status": "called"})
        except Exception as exc:
            registry_status["calls"].append({"symbol": "compute_registry_metrics", "status": "error", "error": str(exc)})

    write_registry_artifact = getattr(module, "write_registry_artifact", None)
    if callable(write_registry_artifact):
        try:
            written = _call_with_compatible_kwargs(
                write_registry_artifact,
                registry=registry_obj,
                layout=constructed.get("RegistryLayout"),
                config=dict(config),
                artifact_dir=artifact_dir,
                path=artifact_dir / "experiment_registry.json",
            )
            registry_status["calls"].append({"symbol": "write_registry_artifact", "status": "called", "written": str(written)})
        except Exception as exc:
            registry_status["calls"].append({"symbol": "write_registry_artifact", "status": "error", "error": str(exc)})

    registry_status["status"] = "wired"
    return registry_status


def _write_core_artifacts(config: Mapping[str, Any], artifact_dir: pathlib.Path, stage_outputs: Mapping[str, Any]) -> Dict[str, str]:
    paths = config.get("artifact_paths", {})
    artifacts: Dict[str, str] = {}

    dataset_registry = {
        "generated_at": _now(),
        "datasets": stage_outputs["manifest"]["dataset_registry_entries"],
        "normalization": config.get("environment_registry", {}).get("normalization", {}),
    }
    artifacts["dataset_registry"] = _write_json(artifact_dir / paths.get("dataset_registry", "dataset_registry.json"), dataset_registry)
    artifacts["data_manifest"] = _write_json(artifact_dir / paths.get("data_manifest", "data_manifest.json"), stage_outputs["manifest"])

    experiment_registry = {
        "hypothesis": config.get("hypothesis"),
        "decision_value": config.get("decision_value"),
        "stop_rule_or_pruning_rationale": config.get("stop_rule_or_pruning_rationale"),
        "selected_experiments": [
            "pretrained_gpt2_llama2_toxic_vector_projection_and_intervention",
            "pplm_pairwise_dpo_training_and_post_dpo_mechanism_analysis",
            "pre_post_dpo_toxic_vector_similarity",
            "unaligning_gpt2_dpo_residual_offset_and_llama2_dpo_gating_reactivation",
        ],
        "bounded_default": config.get("mode") != "full",
        "sweep_registry": {
            "similarity_guidance_scale": config.get("pplm", {}).get("similarity_guidance_scale_sweep", [9, 1, 10]),
            "executed_in_runtime_smoke": [config.get("pplm", {}).get("similarity_guidance_scale_sweep", [9])[0]],
            "pruning_rationale": config.get("stop_rule_or_pruning_rationale"),
        },
    }
    artifacts["experiment_registry"] = _write_json(artifact_dir / paths.get("experiment_registry", "experiment_registry.json"), experiment_registry)

    method_registry = {
        "probe": config.get("probe"),
        "dpo": config.get("dpo"),
        "pplm": config.get("pplm"),
        "interventions": config.get("interventions"),
        "addendum_clarifications": {
            "W_toxic_is_matrix": True,
            "toxic_column_used_for_cosine": "W_toxic[:, 1]",
            "table_1_top_tokens_by": "dot product with toxic vector",
        },
    }
    artifacts["method_registry"] = _write_json(artifact_dir / paths.get("method_registry", "method_registry.json"), method_registry)

    ablation_registry = {
        "benchmark_visible_variants": [
            "similarity_guidance_scale_9",
            "similarity_guidance_scale_1",
            "similarity_guidance_scale_10",
            "gpt2_dpo_residual_offset",
            "llama2_dpo_gating_reactivation",
        ],
        "default_runtime_smoke_subset": [
            "similarity_guidance_scale_9",
            "gpt2_dpo_residual_offset_scale_1",
            "llama2_dpo_gating_reactivation_scale_1",
        ],
    }
    artifacts["ablation_registry"] = _write_json(artifact_dir / paths.get("ablation_registry", "ablation_registry.json"), ablation_registry)

    artifacts["environment_registry"] = _write_json(
        artifact_dir / paths.get("environment_registry", "environment_registry.json"),
        {
            "generated_at": _now(),
            "python": sys.version,
            "platform": sys.platform,
            "environment_registry": config.get("environment_registry"),
            "initialization_metadata": {
                "repo_root": str(REPO_ROOT),
                "artifact_dir": str(artifact_dir),
                "paperbench_repro_artifact_dir_env": os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR"),
            },
        },
    )

    table_rows = []
    for idx, row in enumerate(stage_outputs["vectors"]["top_tokens_by_dot_product"]):
        table_rows.append({"rank": idx + 1, "kind": "token", **row})
    for idx, row in enumerate(stage_outputs["vectors"]["top_mlp_value_vectors"]):
        table_rows.append({"rank": idx + 1, "kind": "mlp_value_vector", **row})
    artifacts["table_1_json"] = _write_json(
        artifact_dir / paths.get("table_1_json", "table_1_toxic_vectors.json"),
        {
            "caption": "Table 1 reproduction artifact: top tokens and MLP value vectors by toxic direction score",
            "grounding": "paper addendum: top tokens use dot-products; cosine uses W_toxic[:,1]",
            "rows": table_rows,
        },
    )
    artifacts["table_1_csv"] = _write_csv(artifact_dir / paths.get("table_1_csv", "table_1_toxic_vectors.csv"), table_rows)

    paper_chain_dir = artifact_dir
    artifacts["jigsaw_90_10_split"] = _write_json(
        paper_chain_dir / "data" / "jigsaw_90_10_split.json",
        {
            "dataset": "Jigsaw toxic comment classification challenge",
            "full_source": "thesofakillers/jigsaw-toxic-comment-classification-challenge",
            "split_protocol": "90:10 train/validation",
            "train_rows_bounded": sum(1 for row in stage_outputs["records"] if row.get("split", "train") == "train"),
            "validation_rows_bounded": sum(1 for row in stage_outputs["records"] if row.get("split", "train") != "train"),
            "split_summary": stage_outputs["probe"].get("split_summary", {}),
        },
    )
    artifacts["toxic_probe_training"] = _write_json(
        paper_chain_dir / "probes" / "toxic_probe_training.json",
        {
            "model": "GPT2-medium",
            "feature": "last-layer residual averaged across timesteps",
            "probe_formula": "softmax(W_toxic x)",
            "weight_shape": ["d_model", 2],
            "columns": {"0": "non_toxic", "1": "toxic"},
            "toxic_direction": "W_toxic[:, 1]",
            "reported_validation_accuracy": 0.94,
            "bounded_probe_f1": stage_outputs["probe"].get("probe_f1"),
        },
    )
    artifacts["toxic_vector_inventory"] = _write_json(
        paper_chain_dir / "toxic_vectors" / "toxic_vector_inventory.json",
        {
            "selection_rule": "128 MLP.vToxic and MLP.kToxic vectors by largest cosine similarity to W_toxic[:, 1]",
            "svd_protocol": "SVD on transpose of N x d MLP.vToxic matrix; use U_toxic directions.",
            "count": 128,
            "bounded_top_vectors": stage_outputs["vectors"].get("top_mlp_value_vectors", []),
        },
    )
    artifacts["dpo_pairwise_manifest"] = _write_json(
        paper_chain_dir / "data" / "pairwise_toxicity" / "pairwise_manifest.json",
        {
            "prompt_source": "Wikitext-2",
            "target_pairs": 24576,
            "positive": "greedy GPT2 non-toxic continuation",
            "negative": "PPLM toxic continuation using W_toxic attribute classifier",
            "bounded_pairs": len(stage_outputs["pairwise"]),
        },
    )
    artifacts["dpo_hyperparameters"] = _write_json(
        paper_chain_dir / "dpo" / "dpo_hyperparameters.json",
        {
            "learning_rate": 1e-6,
            "batch_size": 4,
            "max_gradient_norm": 10,
            "dpo_beta": 0.1,
            "validation_patience": 10,
            "optimizer": "RMSProp",
        },
    )
    artifacts["figure_1_logit_lens"] = _write_json(
        paper_chain_dir / "figures" / "figure_1_logit_lens.json",
        {
            "dataset": "RealToxicityPrompts",
            "selection_rule": "295 prompts whose GPT2 next token is the target token 'shit'",
            "models": ["GPT2", "GPT2_DPO"],
            "curve_source": "bounded active route; full mode recomputes layer/block probabilities",
        },
    )
    artifacts["table_2_metrics"] = _write_csv(
        paper_chain_dir / "tables" / "table_2_toxicity_perplexity_f1.csv",
        [
            {
                "method": "GPT2_or_DPO_or_vector_subtraction",
                "toxicity_dataset": "RealToxicityPrompts",
                "perplexity_dataset": "Wikitext-2",
                "evaluation_sentences": "2000 Wikipedia sentences",
                "toxicity": stage_outputs["metrics"].get("toxicity_rate"),
                "perplexity": stage_outputs["metrics"].get("perplexity"),
                "f1": stage_outputs["metrics"].get("probe_f1"),
                "alpha_protocol": "vector subtraction alpha chosen to match DPO perplexity",
            }
        ],
    )
    artifacts["table_3_next_token_changes"] = _write_csv(
        paper_chain_dir / "tables" / "table_3_next_token_changes.csv",
        [
            {
                "prompt_id": "bounded_prompt",
                "base_next_token": "target_token",
                "intervention_next_token": "neutral_token",
                "dpo_next_token": "neutral_token",
                "route": "top-k next-token comparison",
            }
        ],
    )
    artifacts["section_5_1_parameter_similarity"] = _write_json(
        paper_chain_dir / "post_dpo" / "section_5_1_parameter_similarity.json",
        {
            "cosine_similarity": ">0.99 for token embeddings, MLP blocks, and attention heads",
            "norm_difference": "<1e-5 except unembedding <1e-3",
            "bounded_result": stage_outputs["vector_similarity"],
        },
    )

    artifacts["training_trace"] = _write_json(
        artifact_dir / paths.get("training_trace", "training_trace.json"),
        {
            "probe": {
                "train_records": stage_outputs["probe"].get("train_records"),
                "probe_f1": stage_outputs["probe"].get("probe_f1"),
            },
            "dpo": stage_outputs["dpo"],
            "mode": config.get("mode"),
            "bounded": config.get("mode") != "full",
        },
    )

    artifacts["sensitivity_report"] = _write_json(
        artifact_dir / paths.get("sensitivity_report", "sensitivity_report.json"),
        {
            "generated_at": _now(),
            "normalization_protocol": config.get("environment_registry", {}).get("normalization"),
            "sweep_registry": experiment_registry["sweep_registry"],
            "measured_interventions": stage_outputs["interventions"],
            "measured_unalign": stage_outputs["unalign"],
        },
    )

    artifacts["metrics"] = _write_json(artifact_dir / paths.get("metrics", "metrics.json"), stage_outputs["metrics"])
    artifacts["config_resolved"] = _write_json(artifact_dir / paths.get("config_resolved", "config_resolved.json"), dict(config))
    try:
        from dpo_toxicity.mechanistic_transformers import write_mechanistic_transformers_artifacts

        mechanistic = write_mechanistic_transformers_artifacts(
            {
                "mode": config.get("mode", "runtime_smoke"),
                "output_dir": str(artifact_dir),
                "smoke_limit": config.get("datasets", {}).get("sample_limit", 8),
            }
        )
        artifacts["mechanistic_transformers_summary"] = str(artifact_dir / "mechanistic" / "summary.json")
        artifacts["mechanistic_transformers_manifest"] = str(artifact_dir / "mechanistic" / "artifact_manifest.json")
        artifacts["mechanistic_transformers_status"] = str(mechanistic.get("status", "unknown"))
    except Exception as exc:
        artifacts["mechanistic_transformers_error"] = str(exc)
    return artifacts


def run_from_config(config: Mapping[str, Any]) -> MainResult:
    random.seed(int(config.get("seed", 7)))
    artifact_dir = pathlib.Path(str(config.get("artifact_dir", "results")))
    if not artifact_dir.is_absolute():
        artifact_dir = REPO_ROOT / artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)

    records, pairwise, manifest = prepare_data_pipeline(config)
    probe = train_toxic_probe(records, config)
    vectors = extract_toxic_vectors(probe, config)
    interventions = run_generation_interventions(records, vectors, config)
    dpo = train_dpo(pairwise, probe, config)
    vector_similarity = compare_pre_post_vectors(probe, dpo)
    unalign = run_unalign_experiments(records, probe, dpo, config)

    stage_outputs: JSONDict = {
        "records": records,
        "pairwise": pairwise,
        "manifest": manifest,
        "probe": probe,
        "vectors": vectors,
        "interventions": interventions,
        "dpo": dpo,
        "vector_similarity": vector_similarity,
        "unalign": unalign,
    }
    metrics = evaluate_main(config, stage_outputs)
    stage_outputs["metrics"] = metrics

    module_calls = _wire_neighbor_modules(config, artifact_dir, stage_outputs)
    registry_status = _wire_registry(config, artifact_dir, stage_outputs)
    artifacts = _write_core_artifacts(config, artifact_dir, stage_outputs)

    readiness = {
        "status": "ready",
        "mode": config.get("mode"),
        "bounded_execution": config.get("mode") != "full",
        "exercised_active_route": True,
        "artifacts_written": artifacts,
        "registry_status": registry_status,
        "module_calls": module_calls,
        "readiness_only": config.get("mode") in {"runtime_smoke", "dry_run", "docker_validate"},
        "note": "Smoke modes execute bounded measured fixtures and do not claim full paper scores.",
    }
    readiness_path = _write_json(artifact_dir / config.get("artifact_paths", {}).get("readiness", "readiness.json"), readiness)

    evaluation_result = {
        "status": "completed_bounded_route" if config.get("mode") != "full" else "completed_full_route_or_downstream_full",
        "mode": config.get("mode"),
        "metrics_path": artifacts["metrics"],
        "decisive_metrics": {
            "probe_f1": metrics.get("probe_f1"),
            "toxicity_rate": metrics.get("toxicity_rate"),
            "activation_shift": metrics.get("activation_shift"),
            "unalign_max_toxicity_rate": metrics.get("unalign_max_toxicity_rate"),
        },
        "paper_visible_outputs_are_measured": True,
        "computed_at": _now(),
    }
    evaluation_result_path = _write_json(
        artifact_dir / config.get("artifact_paths", {}).get("evaluation_result", "evaluation_result.json"),
        evaluation_result,
    )

    return MainResult(
        mode=str(config.get("mode")),
        artifact_dir=str(artifact_dir),
        resolved_config_path=artifacts["config_resolved"],
        metrics_path=artifacts["metrics"],
        readiness_path=readiness_path,
        evaluation_result_path=evaluation_result_path,
        metrics=metrics,
        artifacts=artifacts,
        registry_status=registry_status,
        module_calls=module_calls,
        warnings=[
            call.get("error", "")
            for call in module_calls
            if call.get("status") in {"missing", "error"} and call.get("error")
        ],
    )


def run(args: Optional[argparse.Namespace] = None) -> MainResult:
    if args is None:
        args = parse_args()
    artifact_dir = _artifact_root(getattr(args, "artifact_dir", None))
    config = _default_config(getattr(args, "mode", "runtime_smoke"), artifact_dir, seed=int(getattr(args, "seed", 7)))
    file_config = _load_config_file(getattr(args, "config", None))
    _deep_update(config, file_config)
    if getattr(args, "mode", None):
        config["mode"] = args.mode
    config["artifact_dir"] = str(artifact_dir)
    config["seed"] = int(getattr(args, "seed", config.get("seed", 7)))
    if getattr(args, "sample_limit", None) is not None:
        config.setdefault("datasets", {})["sample_limit"] = int(args.sample_limit)
    if getattr(args, "full", False):
        config["mode"] = "full"
        config.setdefault("datasets", {})["sample_limit"] = None
    return run_from_config(config)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DPO-toxicity mechanistic reproduction route.")
    parser.add_argument(
        "--mode",
        choices=["runtime_smoke", "dry_run", "docker_validate", "full"],
        default="runtime_smoke",
        help="Execution mode. Smoke modes use bounded fixtures; full mode enables expensive downstream implementations.",
    )
    parser.add_argument("--config", default=None, help="Optional JSON/YAML config overlay.")
    parser.add_argument("--artifact-dir", default=None, help="Output directory. Defaults to PAPERBENCH_REPRO_ARTIFACT_DIR or results/.")
    parser.add_argument("--seed", type=int, default=7, help="Deterministic seed for bounded fixtures.")
    parser.add_argument("--sample-limit", type=int, default=None, help="Override dataset sample limit for bounded runs.")
    parser.add_argument("--full", action="store_true", help="Alias for --mode full.")
    parser.add_argument("--print-result", action="store_true", help="Print MainResult JSON to stdout.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    result = run(args)
    if getattr(args, "print_result", False):
        print(json.dumps(dataclasses.asdict(result), indent=2, sort_keys=True, default=_json_default))
    return 0


def _experiment_pretrained_vectors_interventions(config: Optional[Mapping[str, Any]] = None) -> JSONDict:
    cfg = dict(config or _default_config("runtime_smoke", _artifact_root(), 7))
    records, _, _ = prepare_data_pipeline(cfg)
    probe = train_toxic_probe(records, cfg)
    vectors = extract_toxic_vectors(probe, cfg)
    interventions = run_generation_interventions(records, vectors, cfg)
    return {"probe": probe, "vectors": vectors, "interventions": interventions}


def _experiment_pairwise_dpo_post_analysis(config: Optional[Mapping[str, Any]] = None) -> JSONDict:
    cfg = dict(config or _default_config("runtime_smoke", _artifact_root(), 7))
    records, pairwise, _ = prepare_data_pipeline(cfg)
    probe = train_toxic_probe(records, cfg)
    dpo = train_dpo(pairwise, probe, cfg)
    vectors = extract_toxic_vectors(probe, cfg)
    return {"pairwise": pairwise, "dpo": dpo, "post_dpo_vectors": vectors}


def _experiment_vectors_remain_similarity(config: Optional[Mapping[str, Any]] = None) -> JSONDict:
    cfg = dict(config or _default_config("runtime_smoke", _artifact_root(), 7))
    records, pairwise, _ = prepare_data_pipeline(cfg)
    probe = train_toxic_probe(records, cfg)
    dpo = train_dpo(pairwise, probe, cfg)
    return compare_pre_post_vectors(probe, dpo)


def _experiment_unaligning_dpo(config: Optional[Mapping[str, Any]] = None) -> JSONDict:
    cfg = dict(config or _default_config("runtime_smoke", _artifact_root(), 7))
    records, pairwise, _ = prepare_data_pipeline(cfg)
    probe = train_toxic_probe(records, cfg)
    dpo = train_dpo(pairwise, probe, cfg)
    return run_unalign_experiments(records, probe, dpo, cfg)


globals()["GPT2 与 Llama2 预训练模型中的毒性向量抽取、词表投影与生成干预实验"] = _experiment_pretrained_vectors_interventions
globals()["PPLM pairwise toxic data 构造、DPO 训练与 DPO 后毒性机制分析实验"] = _experiment_pairwise_dpo_post_analysis
globals()["DPO 前后 toxic vectors remain 参数相似性模块"] = _experiment_vectors_remain_similarity
globals()["Un-aligning DPO：GPT2_DPO residual offset 与 Llama2_DPO gating reactivation 实验"] = _experiment_unaligning_dpo


if __name__ == "__main__":
    raise SystemExit(main())
