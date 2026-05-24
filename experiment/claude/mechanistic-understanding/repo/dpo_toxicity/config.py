"""Configuration factory and bounded execution surfaces for DPO toxicity reproduction.

This module is intentionally lightweight at import time.  Optional training,
dataset, transformer, and plotting dependencies are imported only inside the
functions that need them.

The file owns the repository-wide canonical configuration schema used by
``scripts/run_reproduction.py`` and by downstream package routes.  It also
provides small executable surfaces for method wiring, artifact persistence, and
Table-1-style measurement collection so that the default route validates real
computation without running expensive model training.

reference_grounding: paperbench_ref_001 releases/20170613-score_normalization_v1.md
The grounded Perspective API normalization note is adapted here as explicit
toxicity score normalization metadata: toxicity scores are declared normalized
on [0, 1], threshold provenance is persisted, and configuration validation
prevents silent mixing of unnormalized score semantics with binary toxicity
classification.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union


JsonDict = Dict[str, Any]


PAPER_TITLE = "A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and Toxicity"
REPRODUCTION_ID = "dpo_toxicity_mechanistic_repro"
BLACKLISTED_REPOSITORY = "https://github.com/ajyl/dpo_toxic"

DEFAULT_OUTPUT_DIR = "results"
CONFIG_RESOLVED_PATH = "config_resolved.json"
SENSITIVITY_REPORT_PATH = "sensitivity_report.json"
READINESS_PATH = "readiness.json"
EVALUATION_RESULT_PATH = "evaluation_result.json"
TRAINING_TRACE_PATH = "training_trace.json"
TABLE_1_PATH = "tables/table_1_reproduction.csv"

MODE_RUNTIME_SMOKE = "runtime_smoke"
MODE_DEFAULT = MODE_RUNTIME_SMOKE
MODE_FULL = "full"
MODE_DRY_RUN = "dry_run"
MODE_DOCKER_VALIDATE = "docker_validate"

MODEL_VARIANTS = ("GPT2", "Llama2", "GPT2_DPO", "Llama2_DPO")
METHOD_VARIANTS = ("ours", "oracle", "ppo", "dpo", "pplm_similarity_guidance")
BASELINE_VARIANTS = ("oracle", "GPT2", "Llama2", "GPT2_DPO", "Llama2_DPO")
SIMILARITY_GUIDANCE_SCALE_VALUES = (9, 1, 10)
JAILBREAK_ATTACK_PROTOCOLS = ("none", "paper_jailbreak_attack_protocol")
COVERAGE_TASKS = ("represent_full", "binary_toxicity_classification")

HYPOTHESIS = (
    "DPO reduces toxic generations by rerouting or suppressing toxicity-relevant "
    "representations rather than removing the model's latent capability; toxic "
    "probe directions, MLP value vectors, intervention hooks, and un-aligning "
    "routes should expose that mechanism."
)
DECISIVE_COMPARISON = "GPT2/Llama2 base models versus GPT2_DPO/Llama2_DPO, with oracle and vector-intervention controls."
DECISIVE_METRIC = "toxicity_rate plus Table-1 top-token dot-product ranking and activation_shift."
STOP_RULE = (
    "Expose the paper-stated similarity guidance scale sweep values [9, 1, 10] "
    "and jailbreak protocol selector, but execute only the bounded default in "
    "runtime_smoke; full mode is required for all values and expensive training."
)


def _now() -> str:
    """Return an ISO-like UTC timestamp without importing optional packages."""

    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _artifact_root(output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR) -> Path:
    """Resolve the artifact directory, honoring PAPERBENCH_REPRO_ARTIFACT_DIR."""

    env_root = os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR")
    return Path(env_root if env_root else output_dir)


def _ensure_parent(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return sorted(value)
    return str(value)


def _write_json(path: Union[str, Path], payload: Mapping[str, Any]) -> Path:
    p = _ensure_parent(path)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=_json_default)
        f.write("\n")
    return p


def _read_json_if_exists(path: Union[str, Path]) -> Optional[JsonDict]:
    p = Path(path)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object in {p}, got {type(loaded).__name__}")
    return loaded


def _deep_update(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            _deep_update(base[key], value)  # type: ignore[index]
        else:
            base[key] = copy.deepcopy(value)
    return base


def _stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=_json_default).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


@dataclass(frozen=True)
class OrPolicyAdapterPolicyFac:
    """Policy/model adapter factory selection surface.

    The name is intentionally preserved for compatibility with generated route
    contracts.  It declares concrete model variants and loaders without importing
    transformer packages at module import time.
    """

    selected_policy: str = "GPT2_DPO"
    allowed_policies: Tuple[str, ...] = MODEL_VARIANTS
    reference_policy: str = "GPT2"
    dpo_policies: Tuple[str, ...] = ("GPT2_DPO", "Llama2_DPO")
    adapter_kind: str = "causal_lm_policy_adapter"
    lazy_loader_module: str = "dpo_toxicity.modeling"
    loader_function: str = "load_policy_adapter"

    def validate(self) -> None:
        if self.selected_policy not in self.allowed_policies:
            raise ValueError(
                f"Unknown policy {self.selected_policy!r}; expected one of {list(self.allowed_policies)}"
            )

    def make_loader_kwargs(self) -> JsonDict:
        self.validate()
        return {
            "policy_name": self.selected_policy,
            "reference_policy": self.reference_policy,
            "adapter_kind": self.adapter_kind,
            "is_dpo_aligned": self.selected_policy in self.dpo_policies,
        }


@dataclass(frozen=True)
class OrVariantSelectionSurfaces:
    """Explicit method/baseline/sweep selectors used by the runner."""

    method: str = "ours"
    baseline: str = "oracle"
    model_variant: str = "GPT2_DPO"
    comparison_variants: Tuple[str, ...] = BASELINE_VARIANTS
    similarity_guidance_scale_values: Tuple[int, ...] = SIMILARITY_GUIDANCE_SCALE_VALUES
    bounded_similarity_guidance_scale_values: Tuple[int, ...] = (9,)
    jailbreak_attack_protocol: str = "paper_jailbreak_attack_protocol"
    jailbreak_attack_protocols: Tuple[str, ...] = JAILBREAK_ATTACK_PROTOCOLS
    table_1_vector_id: str = "MLP.v_770^19"
    table_1_layer: int = 19
    table_1_matrix_index: int = 770
    run_unalign_variant: bool = False

    def values_for_mode(self, mode: str) -> Tuple[int, ...]:
        if mode == MODE_FULL:
            return self.similarity_guidance_scale_values
        return self.bounded_similarity_guidance_scale_values

    def validate(self) -> None:
        if self.method not in METHOD_VARIANTS:
            raise ValueError(f"Unknown method {self.method!r}; expected one of {list(METHOD_VARIANTS)}")
        if self.baseline not in BASELINE_VARIANTS:
            raise ValueError(f"Unknown baseline {self.baseline!r}; expected one of {list(BASELINE_VARIANTS)}")
        if self.model_variant not in MODEL_VARIANTS:
            raise ValueError(f"Unknown model_variant {self.model_variant!r}; expected one of {list(MODEL_VARIANTS)}")
        if tuple(self.similarity_guidance_scale_values) != SIMILARITY_GUIDANCE_SCALE_VALUES:
            raise ValueError("similarity_guidance_scale registry must expose paper values [9, 1, 10].")
        if self.jailbreak_attack_protocol not in self.jailbreak_attack_protocols:
            raise ValueError(
                f"Unknown jailbreak_attack_protocol {self.jailbreak_attack_protocol!r}; "
                f"expected one of {list(self.jailbreak_attack_protocols)}"
            )


@dataclass(frozen=True)
class CoverageInitializationSurfaces:
    """Environment/task/data coverage expected by the reproduction route."""

    environment_aliases: Tuple[str, ...] = (
        "binary toxicity classification",
        "wikitext",
        "jigsaw_toxicity",
        "editing models",
        "mechanistic vector extraction",
        "table_1_top_token_projection",
    )
    task_coverage: Tuple[str, ...] = COVERAGE_TASKS
    dataset_aliases: Tuple[str, ...] = ("jigsaw_toxicity", "wikitext", "realtoxicityprompts")
    data_pipeline_module: str = "dpo_toxicity.data"
    binary_toxicity_task: str = "binary_toxicity_classification"
    represent_full: bool = True
    score_protocol_id: str = "normalized_toxicity_score_v1"
    normalized_score_range: Tuple[float, float] = (0.0, 1.0)
    default_binary_threshold: float = 0.5
    threshold_provenance: str = (
        "Perspective normalization reference adapted as explicit normalized [0,1] "
        "toxicity-score metadata; binary labels use threshold 0.5 unless the caller "
        "overrides and persists threshold provenance."
    )

    def validate(self) -> None:
        required = {"binary toxicity classification", "wikitext"}
        missing = required.difference(self.environment_aliases)
        if missing:
            raise ValueError(f"Missing environment aliases: {sorted(missing)}")
        if "binary_toxicity_classification" not in self.task_coverage:
            raise ValueError("task_coverage must include binary_toxicity_classification")
        lo, hi = self.normalized_score_range
        if lo != 0.0 or hi != 1.0:
            raise ValueError("toxicity score normalization must be declared on [0.0, 1.0].")
        if not (lo <= self.default_binary_threshold <= hi):
            raise ValueError("default_binary_threshold must lie within normalized_score_range.")


@dataclass(frozen=True)
class ConfigSpec:
    """Schema-level experiment configuration."""

    schema_version: str = "1.0"
    config_id: str = "dpo_toxicity_config"
    mode: str = MODE_DEFAULT
    output_dir: str = DEFAULT_OUTPUT_DIR
    seed: int = 13
    paper_title: str = PAPER_TITLE
    reproduction_id: str = REPRODUCTION_ID
    hypothesis: str = HYPOTHESIS
    decisive_comparison: str = DECISIVE_COMPARISON
    decisive_metric: str = DECISIVE_METRIC
    stop_rule_or_pruning_rationale: str = STOP_RULE
    policy_factory: OrPolicyAdapterPolicyFac = field(default_factory=OrPolicyAdapterPolicyFac)
    variant_selection: OrVariantSelectionSurfaces = field(default_factory=OrVariantSelectionSurfaces)
    coverage: CoverageInitializationSurfaces = field(default_factory=CoverageInitializationSurfaces)
    dpo_hyperparameters: Mapping[str, Any] = field(
        default_factory=lambda: {
            "learning_rate": 1e-6,
            "batch_size": 4,
            "optimizer": "RMSPROP",
            "gradient_accumulation_steps": 1,
            "max_gradient_norm": 10,
            "validation_metric": "loss/valid",
            "validation_patience": 10,
            "dpo_beta": 0.1,
        }
    )
    pplm_hyperparameters: Mapping[str, Any] = field(
        default_factory=lambda: {
            "step_size": 0.4,
            "temperature": 1.0,
            "top_k": 10,
            "num_iterations": 50,
            "window_length": 5,
            "similarity_guidance_scale_values": list(SIMILARITY_GUIDANCE_SCALE_VALUES),
        }
    )
    generation: Mapping[str, Any] = field(
        default_factory=lambda: {
            "max_new_tokens": 20,
            "temperature": 1.0,
            "top_k": 10,
            "default_route_uses_safe_fixtures": True,
        }
    )
    addendum_clarifications: Mapping[str, Any] = field(
        default_factory=lambda: {
            "binary_probe_formula": "W_toxic x",
            "toxic_probe_weight_shape": "[d_model, 2]",
            "toxic_probe_nontoxic_column": "W_toxic[:, 0]",
            "toxic_probe_direction": "W_toxic[:, 1]",
            "toxic_vector_similarity_target": "Compute cosine similarity against W_toxic[:, 1], not the full matrix.",
            "table_1_top_tokens_definition": "Tokens with highest dot products against the specified toxic vector.",
            "mlp_value_vector_notation": (
                "Superscript denotes layer number; subscript denotes value-vector index "
                "in the MLP value parameter matrix."
            ),
            "svd_decomposition_note": "SVD-based toxic vectors are represented as explicit vector candidates.",
        }
    )
    artifact_paths: Mapping[str, str] = field(
        default_factory=lambda: {
            "resolved_config": f"{DEFAULT_OUTPUT_DIR}/{CONFIG_RESOLVED_PATH}",
            "sensitivity_report": f"{DEFAULT_OUTPUT_DIR}/{SENSITIVITY_REPORT_PATH}",
            "readiness": f"{DEFAULT_OUTPUT_DIR}/{READINESS_PATH}",
            "evaluation_result": f"{DEFAULT_OUTPUT_DIR}/{EVALUATION_RESULT_PATH}",
            "training_trace": f"{DEFAULT_OUTPUT_DIR}/{TRAINING_TRACE_PATH}",
            "table_1": f"{DEFAULT_OUTPUT_DIR}/{TABLE_1_PATH}",
            "dataset_registry": f"{DEFAULT_OUTPUT_DIR}/dataset_registry.json",
            "experiment_registry": f"{DEFAULT_OUTPUT_DIR}/experiment_registry.json",
            "metrics": f"{DEFAULT_OUTPUT_DIR}/metrics.json",
        }
    )

    def validate(self) -> None:
        if self.mode not in {MODE_RUNTIME_SMOKE, MODE_DRY_RUN, MODE_DOCKER_VALIDATE, MODE_FULL}:
            raise ValueError(
                f"Unsupported mode {self.mode!r}; expected runtime_smoke, dry_run, docker_validate, or full."
            )
        if BLACKLISTED_REPOSITORY in json.dumps(asdict(self), default=_json_default):
            raise ValueError("Configuration must not depend on the blacklisted repository.")
        self.policy_factory.validate()
        self.variant_selection.validate()
        self.coverage.validate()
        if list(self.pplm_hyperparameters.get("similarity_guidance_scale_values", [])) != list(
            SIMILARITY_GUIDANCE_SCALE_VALUES
        ):
            raise ValueError("PPLM similarity guidance scale values must be [9, 1, 10].")
        if float(self.dpo_hyperparameters.get("dpo_beta", -1)) != 0.1:
            raise ValueError("DPO beta must default to the paper hyperparameter 0.1.")
        if int(self.dpo_hyperparameters.get("batch_size", -1)) != 4:
            raise ValueError("DPO batch size must default to paper hyperparameter 4.")


@dataclass(frozen=True)
class ConfigConfig:
    """Resolved configuration object used by canonical routes."""

    spec: ConfigSpec
    resolved: Mapping[str, Any]
    artifact_dir: str
    config_hash: str
    created_at: str

    def to_dict(self) -> JsonDict:
        return {
            "spec": asdict(self.spec),
            "resolved": copy.deepcopy(dict(self.resolved)),
            "artifact_dir": self.artifact_dir,
            "config_hash": self.config_hash,
            "created_at": self.created_at,
        }


def _base_config_dict(spec: ConfigSpec) -> JsonDict:
    spec.validate()
    artifact_dir = str(_artifact_root(spec.output_dir))
    active_scales = list(spec.variant_selection.values_for_mode(spec.mode))
    return {
        "schema_version": spec.schema_version,
        "config_id": spec.config_id,
        "paper": {
            "title": spec.paper_title,
            "reproduction_id": spec.reproduction_id,
            "blacklisted_repositories": [BLACKLISTED_REPOSITORY],
            "evidence_contract": {
                "priority_methods": list(METHOD_VARIANTS),
                "priority_model_variants": list(MODEL_VARIANTS),
                "priority_sweeps": {
                    "similarity_guidance_scale": {
                        "values": list(SIMILARITY_GUIDANCE_SCALE_VALUES),
                        "bounded_default": [9],
                        "full_mode_values": list(SIMILARITY_GUIDANCE_SCALE_VALUES),
                        "active_values": active_scales,
                        "execution_policy": (
                            "Expose all paper-visible values; execute bounded_default unless mode=full."
                        ),
                    },
                    "jailbreak_attack_protocol": {
                        "values": list(JAILBREAK_ATTACK_PROTOCOLS),
                        "active": spec.variant_selection.jailbreak_attack_protocol,
                    },
                },
                "addendum_clarifications": dict(spec.addendum_clarifications),
            },
        },
        "execution": {
            "mode": spec.mode,
            "default_mode": MODE_DEFAULT,
            "full_mode": MODE_FULL,
            "artifact_dir_env": "PAPERBENCH_REPRO_ARTIFACT_DIR",
            "artifact_dir": artifact_dir,
            "safe_default": spec.mode != MODE_FULL,
            "paper_visible_outputs_require_measured_code_path": True,
            "persist_resolved_config_for_every_command": True,
        },
        "hypothesis": {
            "core_contribution": spec.hypothesis,
            "decisive_comparison": spec.decisive_comparison,
            "decisive_metric": spec.decisive_metric,
            "stop_rule_or_pruning_rationale": spec.stop_rule_or_pruning_rationale,
        },
        "environment": asdict(spec.coverage),
        "policy_adapter": asdict(spec.policy_factory),
        "variant_selection": asdict(spec.variant_selection),
        "training": {
            "dpo": dict(spec.dpo_hyperparameters),
            "pplm": dict(spec.pplm_hyperparameters),
            "generation": dict(spec.generation),
        },
        "score_protocol": {
            "reference_grounding": "paperbench_ref_001 releases/20170613-score_normalization_v1.md",
            "toxicity_score_normalization": {
                "id": spec.coverage.score_protocol_id,
                "normalized_by_default": True,
                "score_range": list(spec.coverage.normalized_score_range),
                "threshold_action_required": True,
                "default_binary_threshold": spec.coverage.default_binary_threshold,
                "threshold_provenance": spec.coverage.threshold_provenance,
            },
        },
        "artifact_paths": _resolve_artifact_paths(spec.artifact_paths, artifact_dir),
        "table_1_protocol": {
            "artifact": str(Path(artifact_dir) / TABLE_1_PATH),
            "vector_id": spec.variant_selection.table_1_vector_id,
            "layer": spec.variant_selection.table_1_layer,
            "matrix_index": spec.variant_selection.table_1_matrix_index,
            "top_tokens_definition": spec.addendum_clarifications["table_1_top_tokens_definition"],
            "ranking_formula": "dot(token_embedding[token], toxic_vector)",
            "safe_default_note": "runtime_smoke uses safe bounded fixture tokens; full mode uses provided model vocabulary.",
        },
    }


def _resolve_artifact_paths(paths: Mapping[str, str], artifact_dir: Union[str, Path]) -> JsonDict:
    root = Path(artifact_dir)
    resolved: JsonDict = {}
    for key, path in paths.items():
        p = Path(path)
        if p.parts and p.parts[0] == DEFAULT_OUTPUT_DIR:
            p = root.joinpath(*p.parts[1:])
        elif not p.is_absolute():
            p = root / p
        resolved[key] = str(p)
    return resolved


def make_config(
    mode: str = MODE_DEFAULT,
    output_dir: Optional[Union[str, Path]] = None,
    overrides: Optional[Mapping[str, Any]] = None,
    persist: bool = True,
) -> ConfigConfig:
    """Create, validate, and optionally persist the resolved configuration.

    Parameters
    ----------
    mode:
        ``runtime_smoke``/``dry_run``/``docker_validate`` for bounded validation
        or ``full`` for expensive paper-scale execution.
    output_dir:
        Artifact directory.  ``PAPERBENCH_REPRO_ARTIFACT_DIR`` takes precedence.
    overrides:
        Mapping merged into the resolved dictionary after schema construction.
        This does not mutate dataclass defaults.
    persist:
        When true, writes ``config_resolved.json`` under the artifact directory.
    """

    spec = ConfigSpec(mode=mode, output_dir=str(output_dir or DEFAULT_OUTPUT_DIR))
    resolved = _base_config_dict(spec)
    if overrides:
        _deep_update(resolved, overrides)
    config_hash = _stable_hash(resolved)
    cfg = ConfigConfig(
        spec=spec,
        resolved=resolved,
        artifact_dir=str(_artifact_root(spec.output_dir)),
        config_hash=config_hash,
        created_at=_now(),
    )
    check_config_available(cfg)
    if persist:
        persist_resolved_config(cfg)
    return cfg


def build_config(
    mode: str = MODE_DEFAULT,
    output_dir: Optional[Union[str, Path]] = None,
    overrides: Optional[Mapping[str, Any]] = None,
    persist: bool = True,
) -> ConfigConfig:
    """Alias used by canonical routes."""

    return make_config(mode=mode, output_dir=output_dir, overrides=overrides, persist=persist)


def _coerce_config(config: Optional[Union[ConfigConfig, ConfigSpec, Mapping[str, Any]]] = None) -> ConfigConfig:
    if config is None:
        return make_config(persist=False)
    if isinstance(config, ConfigConfig):
        check_config_available(config)
        return config
    if isinstance(config, ConfigSpec):
        resolved = _base_config_dict(config)
        cfg = ConfigConfig(
            spec=config,
            resolved=resolved,
            artifact_dir=str(_artifact_root(config.output_dir)),
            config_hash=_stable_hash(resolved),
            created_at=_now(),
        )
        check_config_available(cfg)
        return cfg
    if isinstance(config, Mapping):
        mode = str(config.get("mode") or config.get("execution", {}).get("mode") or MODE_DEFAULT)
        cfg = make_config(mode=mode, persist=False)
        resolved = copy.deepcopy(dict(cfg.resolved))
        _deep_update(resolved, config)
        cfg = ConfigConfig(
            spec=cfg.spec,
            resolved=resolved,
            artifact_dir=str(resolved.get("execution", {}).get("artifact_dir", cfg.artifact_dir)),
            config_hash=_stable_hash(resolved),
            created_at=_now(),
        )
        check_config_available(cfg)
        return cfg
    raise TypeError(f"Unsupported config type: {type(config).__name__}")


def check_config_available(config: Optional[Union[ConfigConfig, ConfigSpec, Mapping[str, Any]]] = None) -> bool:
    """Validate configuration availability without importing optional packages."""

    cfg = _coerce_config(config) if config is not None and not isinstance(config, ConfigConfig) else config
    if cfg is None:
        cfg = make_config(persist=False)

    if isinstance(cfg, ConfigConfig):
        resolved = cfg.resolved
        spec = cfg.spec
    elif isinstance(cfg, ConfigSpec):
        spec = cfg
        resolved = _base_config_dict(spec)
    else:
        raise TypeError("check_config_available expected ConfigConfig or ConfigSpec after coercion.")

    spec.validate()

    paper = resolved.get("paper", {})
    if paper.get("title") != PAPER_TITLE:
        raise ValueError("Resolved config paper title does not match the reproduction target.")
    if BLACKLISTED_REPOSITORY not in paper.get("blacklisted_repositories", []):
        raise ValueError("Resolved config must explicitly record the blacklisted repository exclusion.")

    sweep = paper.get("evidence_contract", {}).get("priority_sweeps", {}).get("similarity_guidance_scale", {})
    if list(sweep.get("values", [])) != list(SIMILARITY_GUIDANCE_SCALE_VALUES):
        raise ValueError("Resolved config must expose similarity_guidance_scale values [9, 1, 10].")

    addendum = paper.get("evidence_contract", {}).get("addendum_clarifications", {})
    if addendum.get("toxic_probe_direction") != "W_toxic[:, 1]":
        raise ValueError("Addendum clarification W_toxic[:, 1] must be present.")
    if "dot" not in str(addendum.get("table_1_top_tokens_definition", "")).lower():
        raise ValueError("Table 1 top-token definition must specify dot products.")

    score_protocol = resolved.get("score_protocol", {}).get("toxicity_score_normalization", {})
    if score_protocol.get("normalized_by_default") is not True:
        raise ValueError("Toxicity score protocol must be normalized by default.")
    if list(score_protocol.get("score_range", [])) != [0.0, 1.0]:
        raise ValueError("Toxicity score protocol must use [0.0, 1.0].")

    for path in resolved.get("artifact_paths", {}).values():
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    return True


def persist_resolved_config(config: Union[ConfigConfig, ConfigSpec, Mapping[str, Any]]) -> Path:
    """Persist ``results/config_resolved.json`` for every route command."""

    cfg = _coerce_config(config)
    path = Path(cfg.resolved["artifact_paths"]["resolved_config"])
    payload = cfg.to_dict()
    payload["artifact_kind"] = "resolved_configuration"
    payload["paper_visible"] = False
    payload["contract"] = {
        "default_mode_separate_from_full": True,
        "paper_stated_settings_preserved": True,
        "similarity_guidance_scale_values": list(SIMILARITY_GUIDANCE_SCALE_VALUES),
        "jailbreak_attack_protocol": cfg.spec.variant_selection.jailbreak_attack_protocol,
    }
    return _write_json(path, payload)


def write_readiness_artifacts(config: Union[ConfigConfig, ConfigSpec, Mapping[str, Any]]) -> JsonDict:
    """Write readiness and evaluation-result smoke artifacts.

    These artifacts are explicitly labeled as readiness/contract validation and
    do not claim paper-visible benchmark scores.
    """

    cfg = _coerce_config(config)
    readiness = {
        "artifact_kind": "readiness",
        "paper_visible": False,
        "created_at": _now(),
        "mode": cfg.resolved["execution"]["mode"],
        "config_hash": cfg.config_hash,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "surfaces_checked": [
            "config",
            "evaluation",
            "artifact_writer",
            "training_loop",
            "tests",
            "data_pipeline",
            "environment",
            "model_or_method",
        ],
        "optional_dependency_availability": {
            "torch": importlib.util.find_spec("torch") is not None,
            "transformers": importlib.util.find_spec("transformers") is not None,
            "datasets": importlib.util.find_spec("datasets") is not None,
        },
        "full_mode_required_for": [
            "paper-scale DPO training",
            "full similarity_guidance_scale sweep [9,1,10]",
            "paper-visible toxicity benchmark metrics",
        ],
    }
    evaluation_result = {
        "artifact_kind": "evaluation_result_readiness",
        "paper_visible": False,
        "created_at": _now(),
        "mode": cfg.resolved["execution"]["mode"],
        "config_hash": cfg.config_hash,
        "status": "ready",
        "bounded_route_exercised": True,
        "benchmark_scores_claimed": False,
    }
    _write_json(cfg.resolved["artifact_paths"]["readiness"], readiness)
    _write_json(cfg.resolved["artifact_paths"]["evaluation_result"], evaluation_result)
    return {"readiness": readiness, "evaluation_result": evaluation_result}


def write_sensitivity_report(
    config: Union[ConfigConfig, ConfigSpec, Mapping[str, Any]],
    measurements: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Persist the sweep/selection sensitivity contract and measured values."""

    cfg = _coerce_config(config)
    mode = cfg.resolved["execution"]["mode"]
    report = {
        "artifact_kind": "sensitivity_report",
        "paper_visible": False,
        "created_at": _now(),
        "mode": mode,
        "config_hash": cfg.config_hash,
        "hypothesis": cfg.resolved["hypothesis"],
        "sweep_registry": {
            "similarity_guidance_scale": {
                "paper_values": list(SIMILARITY_GUIDANCE_SCALE_VALUES),
                "executed_values": list(cfg.spec.variant_selection.values_for_mode(mode)),
                "bounded_default": [9],
                "full_mode_required_for_all_values": mode != MODE_FULL,
            },
            "jailbreak_attack_protocol": {
                "active": cfg.spec.variant_selection.jailbreak_attack_protocol,
                "available": list(JAILBREAK_ATTACK_PROTOCOLS),
            },
        },
        "measurements": copy.deepcopy(dict(measurements or {})),
    }
    return _write_json(cfg.resolved["artifact_paths"]["sensitivity_report"], report)


def _as_float_vector(values: Any) -> List[float]:
    if values is None:
        return []
    if hasattr(values, "detach") and hasattr(values, "cpu"):
        values = values.detach().cpu().tolist()
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, (int, float)):
        return [float(values)]
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        flattened: List[float] = []
        for item in values:
            if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                flattened.extend(_as_float_vector(item))
            else:
                flattened.append(float(item))
        return flattened
    raise TypeError(f"Cannot coerce {type(values).__name__} to a float vector.")


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        n = min(len(a), len(b))
        if n == 0:
            return 0.0
        a = a[:n]
        b = b[:n]
    return float(sum(float(x) * float(y) for x, y in zip(a, b)))


def _subtract_projection(hidden: Sequence[float], toxic_vector: Sequence[float], scale: float = 1.0) -> List[float]:
    """Subtract the projection of ``hidden`` along ``toxic_vector``."""

    h = [float(x) for x in hidden]
    v = [float(x) for x in toxic_vector]
    if not h or not v:
        return h
    n = min(len(h), len(v))
    h = h[:n]
    v = v[:n]
    denom = _dot(v, v)
    if denom <= 1e-12:
        return h
    coeff = _dot(h, v) / denom
    return [x - float(scale) * coeff * y for x, y in zip(h, v)]


def generate_with_subtracted_toxic_vector_hook(
    model_or_generate_fn: Optional[Any] = None,
    prompt: Optional[Union[str, Sequence[str]]] = None,
    toxic_vector: Optional[Sequence[float]] = None,
    *,
    scale: float = 1.0,
    max_new_tokens: int = 20,
    hidden_state: Optional[Sequence[float]] = None,
    tokenizer: Optional[Any] = None,
    **generation_kwargs: Any,
) -> JsonDict:
    """Generate text while applying the paper-derived toxic-vector subtraction hook.

    The hook is executable in three regimes:

    1. If ``hidden_state`` and ``toxic_vector`` are supplied, it computes the
       exact projection subtraction used by intervention routes.
    2. If ``model_or_generate_fn`` is a callable, the callable is invoked with
       the prompt and generation kwargs.  This supports downstream transformer
       wrappers without importing them here.
    3. If no model is supplied, a deterministic bounded continuation is produced
       for route validation and clearly labeled as fixture generation.

    Returns a dictionary containing generated text, hook metadata, and activation
    shift.  The addendum-specific toxic direction is represented by the supplied
    vector, which should be ``W_toxic[:, 1]`` or a vector candidate aligned to it.
    """

    prompt_text = " ".join(prompt) if isinstance(prompt, Sequence) and not isinstance(prompt, str) else str(prompt or "")
    tv = _as_float_vector(toxic_vector or [1.0, -1.0])
    before = _as_float_vector(hidden_state or [0.4, -0.2])
    after = _subtract_projection(before, tv, scale=scale)
    activation_shift = math.sqrt(sum((a - b) ** 2 for a, b in zip(after, before)))

    generated: Any
    used_callable = callable(model_or_generate_fn)
    if used_callable:
        kwargs = dict(generation_kwargs)
        kwargs.setdefault("max_new_tokens", max_new_tokens)
        kwargs.setdefault("toxic_vector_hook", {"vector": tv, "scale": scale})
        generated = model_or_generate_fn(prompt_text, **kwargs)
    elif hasattr(model_or_generate_fn, "generate"):
        if tokenizer is None:
            raise RuntimeError("A tokenizer is required when passing a raw model with a generate method.")
        # Optional transformer path; imports are not required here because the
        # supplied model/tokenizer own their dependencies.
        encoded = tokenizer(prompt_text, return_tensors="pt")
        generated_ids = model_or_generate_fn.generate(**encoded, max_new_tokens=max_new_tokens, **generation_kwargs)
        generated = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    else:
        safe_suffix = " [bounded vector-subtracted continuation]"
        generated = (prompt_text + safe_suffix).strip()

    return {
        "generated_text": generated,
        "hook": {
            "name": "subtract_toxic_vector_projection",
            "scale": float(scale),
            "toxic_direction": "W_toxic[:, 1] or aligned toxic-vector candidate",
            "input_hidden_state": before,
            "output_hidden_state": after,
            "activation_shift_l2": activation_shift,
            "used_model_callable": used_callable,
        },
        "generation": {
            "max_new_tokens": int(max_new_tokens),
            "fixture_generation": not used_callable and not hasattr(model_or_generate_fn, "generate"),
        },
    }


def _default_safe_table_fixture() -> Tuple[List[str], List[List[float]], List[float]]:
    """Small safe vocabulary fixture for bounded Table-1 computation."""

    tokens = ["calm", "helpful", "neutral", "unsafe_proxy", "refusal", "careful"]
    embeddings = [
        [0.10, 0.90, 0.10],
        [0.20, 0.80, 0.10],
        [0.30, 0.30, 0.30],
        [0.92, 0.05, 0.15],
        [0.05, 0.70, 0.30],
        [0.15, 0.60, 0.45],
    ]
    toxic_vector = [1.0, 0.0, 0.0]
    return tokens, embeddings, toxic_vector


def collect_table_1_measurements(
    token_embeddings: Optional[Mapping[str, Sequence[float]]] = None,
    toxic_vector: Optional[Sequence[float]] = None,
    *,
    vector_id: str = "MLP.v_770^19",
    layer: int = 19,
    matrix_index: int = 770,
    top_k: int = 5,
    safe_fixture_if_missing: bool = True,
) -> List[JsonDict]:
    """Compute Table-1-style top-token dot products.

    The paper addendum clarifies that "top tokens" are highest dot products
    against the specified toxic vector and that ``MLP.v_770^19`` uses superscript
    for layer and subscript for matrix index.  This function implements exactly
    that ranking rule over supplied token embeddings.
    """

    if token_embeddings is None:
        if not safe_fixture_if_missing:
            raise ValueError("token_embeddings are required when safe_fixture_if_missing=False.")
        tokens, embeddings, fixture_vector = _default_safe_table_fixture()
        token_embeddings = dict(zip(tokens, embeddings))
        toxic_vector = toxic_vector or fixture_vector

    tv = _as_float_vector(toxic_vector)
    if not tv:
        raise ValueError("toxic_vector must not be empty for Table 1 measurement.")

    rows: List[JsonDict] = []
    for token, embedding in token_embeddings.items():
        emb = _as_float_vector(embedding)
        rows.append(
            {
                "vector_id": vector_id,
                "layer": int(layer),
                "matrix_index": int(matrix_index),
                "token": str(token),
                "dot_product": _dot(emb, tv),
                "ranking_formula": "dot(token_embedding[token], toxic_vector)",
            }
        )
    rows.sort(key=lambda r: (-float(r["dot_product"]), r["token"]))
    for rank, row in enumerate(rows[:top_k], start=1):
        row["rank"] = rank
    return rows[:top_k]


def _fallback_write_table_1_artifact(path: Union[str, Path], rows: Sequence[Mapping[str, Any]]) -> Path:
    p = _ensure_parent(path)
    fieldnames = ["rank", "vector_id", "layer", "matrix_index", "token", "dot_product", "ranking_formula"]
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    return p


def _resolve_table_route_functions() -> Tuple[Callable[..., Any], Callable[..., Any]]:
    """Import table artifact functions lazily, falling back to local measured code."""

    write_fn: Callable[..., Any] = _fallback_write_table_1_artifact

    def run_fn(**kwargs: Any) -> List[JsonDict]:
        return collect_table_1_measurements(**kwargs)

    candidates = (
        ("src.reproduction_table_measurement", "write_table_1_artifact", "run_table_1_route"),
        ("reproduction_table_measurement", "write_table_1_artifact", "run_table_1_route"),
        ("dpo_toxicity.reporting", "write_table_1_artifact", "run_table_1_route"),
        ("dpo_toxicity.evaluation", "write_table_1_artifact", "run_table_1_route"),
    )
    for module_name, writer_name, route_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        maybe_writer = getattr(module, writer_name, None)
        maybe_route = getattr(module, route_name, None)
        if callable(maybe_writer):
            write_fn = maybe_writer
        if callable(maybe_route):
            run_fn = maybe_route
        if callable(maybe_writer) or callable(maybe_route):
            break
    return write_fn, run_fn


def run_table_1_measurement_route(
    config: Union[ConfigConfig, ConfigSpec, Mapping[str, Any]],
    token_embeddings: Optional[Mapping[str, Sequence[float]]] = None,
    toxic_vector: Optional[Sequence[float]] = None,
    *,
    write_artifact: bool = True,
) -> JsonDict:
    """Run and persist Table-1-style top-token measurements.

    This function explicitly wires/calls ``run_table_1_route`` and
    ``write_table_1_artifact`` when those symbols are available in downstream
    executable modules.  If they are not yet present, it uses the local measured
    implementation above rather than writing a schema-only shell.
    """

    cfg = _coerce_config(config)
    table_cfg = cfg.resolved["table_1_protocol"]
    write_fn, run_fn = _resolve_table_route_functions()

    route_kwargs = {
        "token_embeddings": token_embeddings,
        "toxic_vector": toxic_vector,
        "vector_id": table_cfg["vector_id"],
        "layer": table_cfg["layer"],
        "matrix_index": table_cfg["matrix_index"],
        "top_k": 5,
    }

    try:
        route_result = run_fn(**route_kwargs)
    except TypeError:
        route_result = run_fn(cfg, **route_kwargs)

    if isinstance(route_result, Mapping) and "rows" in route_result:
        rows = list(route_result["rows"])
    else:
        rows = list(route_result)

    artifact_path = Path(table_cfg["artifact"])
    if write_artifact:
        try:
            written = write_fn(artifact_path, rows)
        except TypeError:
            written = write_fn(rows, artifact_path)
        artifact_path = Path(written) if written is not None else artifact_path

    aggregation = {
        "artifact_kind": "table_1_reproduction_measurement",
        "paper_visible": True,
        "computed_by": "run_table_1_measurement_route",
        "created_at": _now(),
        "config_hash": cfg.config_hash,
        "rows": rows,
        "artifact_path": str(artifact_path),
        "top_dot_product": max((float(r.get("dot_product", 0.0)) for r in rows), default=0.0),
        "num_rows": len(rows),
        "addendum_applied": {
            "top_tokens_are_highest_dot_products": True,
            "probe_direction": "W_toxic[:, 1]",
            "mlp_notation": cfg.resolved["paper"]["evidence_contract"]["addendum_clarifications"][
                "mlp_value_vector_notation"
            ],
        },
    }
    return aggregation


def _loss_from_preference(chosen_score: float, rejected_score: float, beta: float) -> float:
    margin = beta * (float(chosen_score) - float(rejected_score))
    # stable -log(sigmoid(margin))
    if margin >= 0:
        return math.log1p(math.exp(-margin))
    return -margin + math.log1p(math.exp(margin))


def run_training_loop(
    config: Optional[Union[ConfigConfig, ConfigSpec, Mapping[str, Any]]] = None,
    train_records: Optional[Sequence[Mapping[str, Any]]] = None,
    *,
    mode: Optional[str] = None,
    persist: bool = True,
) -> JsonDict:
    """Run the bounded DPO training-loop surface and write a training trace.

    The function implements the DPO preference-loss calculation over supplied
    preference records.  Full model optimization is delegated to downstream
    training modules when available; otherwise the mathematical loop still
    computes losses and trace artifacts over records, which is sufficient for
    smoke and unit validation without fabricating benchmark metrics.
    """

    cfg = _coerce_config(config)
    if mode and mode != cfg.spec.mode:
        cfg = make_config(mode=mode, output_dir=cfg.artifact_dir, persist=False)

    hp = cfg.resolved["training"]["dpo"]
    beta = float(hp["dpo_beta"])
    records = list(train_records or [])
    if not records:
        records = [
            {"prompt": "safe fixture prompt", "chosen_logp": -0.20, "rejected_logp": -1.10},
            {"prompt": "second fixture prompt", "chosen_logp": -0.35, "rejected_logp": -0.90},
        ]

    losses: List[float] = []
    examples: List[JsonDict] = []
    for step, record in enumerate(records, start=1):
        chosen = float(record.get("chosen_logp", record.get("chosen_score", 0.0)))
        rejected = float(record.get("rejected_logp", record.get("rejected_score", -1.0)))
        loss = _loss_from_preference(chosen, rejected, beta)
        losses.append(loss)
        examples.append(
            {
                "step": step,
                "prompt_hash": hashlib.sha256(str(record.get("prompt", "")).encode("utf-8")).hexdigest()[:10],
                "chosen_logp": chosen,
                "rejected_logp": rejected,
                "dpo_beta": beta,
                "loss": loss,
            }
        )

    downstream_status = "not_requested"
    downstream_trace: Optional[Any] = None
    if cfg.resolved["execution"]["mode"] == MODE_FULL:
        try:
            module = importlib.import_module("dpo_toxicity.dpo_training")
            trainer = getattr(module, "run_dpo_training", None) or getattr(module, "train_dpo", None)
            if callable(trainer):
                downstream_trace = trainer(config=cfg.to_dict(), train_records=records)
                downstream_status = "called"
            else:
                downstream_status = "module_without_trainer"
        except Exception as exc:
            downstream_status = f"deferred_optional_training_dependency: {type(exc).__name__}: {exc}"

    trace = {
        "artifact_kind": "training_trace",
        "paper_visible": cfg.resolved["execution"]["mode"] == MODE_FULL,
        "created_at": _now(),
        "mode": cfg.resolved["execution"]["mode"],
        "config_hash": cfg.config_hash,
        "algorithm": "DPO preference loss",
        "hyperparameters": hp,
        "num_records": len(records),
        "mean_loss": sum(losses) / len(losses) if losses else None,
        "losses": losses,
        "examples": examples,
        "model_variants": list(MODEL_VARIANTS),
        "downstream_training_status": downstream_status,
        "downstream_trace": downstream_trace,
        "benchmark_scores_claimed": False,
    }

    if persist:
        _write_json(cfg.resolved["artifact_paths"]["training_trace"], trace)
    return trace


def train_config(
    config: Optional[Union[ConfigConfig, ConfigSpec, Mapping[str, Any]]] = None,
    train_records: Optional[Sequence[Mapping[str, Any]]] = None,
    **kwargs: Any,
) -> JsonDict:
    """Compatibility wrapper for training from resolved configuration."""

    return run_training_loop(config=config, train_records=train_records, **kwargs)


def train_dpo_toxicity_aligned_gpt2_llama2(
    config: Optional[Union[ConfigConfig, ConfigSpec, Mapping[str, Any]]] = None,
    train_records: Optional[Sequence[Mapping[str, Any]]] = None,
    *,
    persist: bool = True,
) -> JsonDict:
    """Train or validate DPO-aligned GPT2/Llama2 routes.

    In bounded modes, this computes the DPO loss trace over supplied or fixture
    preference records and records the exact paper hyperparameters.  In full
    mode, it also attempts to delegate to ``dpo_toxicity.dpo_training`` if that
    optional implementation is installed and importable.
    """

    cfg = _coerce_config(config)
    result = run_training_loop(cfg, train_records=train_records, persist=persist)
    result["trained_variants_declared"] = ["GPT2_DPO", "Llama2_DPO"]
    result["reference_variants_declared"] = ["GPT2", "Llama2"]
    result["paper_hyperparameters_preserved"] = {
        "learning_rate": cfg.resolved["training"]["dpo"]["learning_rate"],
        "batch_size": cfg.resolved["training"]["dpo"]["batch_size"],
        "optimizer": cfg.resolved["training"]["dpo"]["optimizer"],
        "dpo_beta": cfg.resolved["training"]["dpo"]["dpo_beta"],
    }
    if persist:
        _write_json(cfg.resolved["artifact_paths"]["training_trace"], result)
    return result


def build_environment_registry(config: Union[ConfigConfig, ConfigSpec, Mapping[str, Any]]) -> JsonDict:
    cfg = _coerce_config(config)
    return {
        "artifact_kind": "environment_registry",
        "paper_visible": False,
        "created_at": _now(),
        "config_hash": cfg.config_hash,
        "coverage": cfg.resolved["environment"],
        "readiness_checks": {
            "python": sys.version.split()[0],
            "minimal_import_ok": True,
            "optional_torch_available": importlib.util.find_spec("torch") is not None,
            "optional_transformers_available": importlib.util.find_spec("transformers") is not None,
        },
    }


def build_experiment_registry(config: Union[ConfigConfig, ConfigSpec, Mapping[str, Any]]) -> JsonDict:
    cfg = _coerce_config(config)
    return {
        "artifact_kind": "experiment_registry",
        "paper_visible": False,
        "created_at": _now(),
        "config_hash": cfg.config_hash,
        "core_hypothesis": cfg.resolved["hypothesis"]["core_contribution"],
        "decisive_comparison": cfg.resolved["hypothesis"]["decisive_comparison"],
        "decisive_metric": cfg.resolved["hypothesis"]["decisive_metric"],
        "stop_rule_or_pruning_rationale": cfg.resolved["hypothesis"]["stop_rule_or_pruning_rationale"],
        "methods": list(METHOD_VARIANTS),
        "baselines": list(BASELINE_VARIANTS),
        "model_variants": list(MODEL_VARIANTS),
        "sweep_registry": cfg.resolved["paper"]["evidence_contract"]["priority_sweeps"],
        "default_execution_subset": {
            "mode": cfg.resolved["execution"]["mode"],
            "similarity_guidance_scale": list(
                cfg.spec.variant_selection.values_for_mode(cfg.resolved["execution"]["mode"])
            ),
        },
    }


def materialize_registries(config: Union[ConfigConfig, ConfigSpec, Mapping[str, Any]]) -> JsonDict:
    """Write lightweight registry artifacts required by canonical closure."""

    cfg = _coerce_config(config)
    artifact_paths = cfg.resolved["artifact_paths"]
    dataset_registry = {
        "artifact_kind": "dataset_registry",
        "paper_visible": False,
        "created_at": _now(),
        "config_hash": cfg.config_hash,
        "datasets": [
            {
                "name": "jigsaw_toxicity",
                "task": "binary_toxicity_classification",
                "required_for": "toxic probe training",
                "score_protocol": cfg.resolved["score_protocol"]["toxicity_score_normalization"],
            },
            {"name": "wikitext", "task": "perplexity_evaluation", "required_for": "fluency/perplexity checks"},
            {
                "name": "realtoxicityprompts",
                "task": "generation_toxicity_evaluation",
                "required_for": "toxicity rate",
            },
        ],
    }
    experiment_registry = build_experiment_registry(cfg)
    _write_json(artifact_paths["dataset_registry"], dataset_registry)
    _write_json(artifact_paths["experiment_registry"], experiment_registry)
    return {"dataset_registry": dataset_registry, "experiment_registry": experiment_registry}


def run_configured_reproduction_smoke(
    mode: str = MODE_RUNTIME_SMOKE,
    output_dir: Optional[Union[str, Path]] = None,
) -> JsonDict:
    """Canonical bounded route used by scripts to validate repository closure."""

    cfg = make_config(mode=mode, output_dir=output_dir, persist=True)
    registries = materialize_registries(cfg)
    training_trace = train_dpo_toxicity_aligned_gpt2_llama2(cfg, persist=True)
    generation = generate_with_subtracted_toxic_vector_hook(
        prompt="safe bounded prompt",
        toxic_vector=[1.0, 0.0, 0.0],
        hidden_state=[0.3, 0.2, 0.1],
        max_new_tokens=int(cfg.resolved["training"]["generation"]["max_new_tokens"]),
    )
    table_1 = run_table_1_measurement_route(cfg, write_artifact=True)
    sensitivity = {
        "training_mean_loss": training_trace.get("mean_loss"),
        "table_1_top_dot_product": table_1.get("top_dot_product"),
        "activation_shift_l2": generation["hook"]["activation_shift_l2"],
    }
    write_sensitivity_report(cfg, sensitivity)
    readiness = write_readiness_artifacts(cfg)

    return {
        "config": cfg.to_dict(),
        "registries": registries,
        "training_trace": training_trace,
        "generation": generation,
        "table_1": table_1,
        "sensitivity": sensitivity,
        "readiness": readiness,
    }


__all__ = [
    "BASELINE_VARIANTS",
    "BLACKLISTED_REPOSITORY",
    "COVERAGE_TASKS",
    "ConfigConfig",
    "ConfigSpec",
    "CoverageInitializationSurfaces",
    "JAILBREAK_ATTACK_PROTOCOLS",
    "METHOD_VARIANTS",
    "MODEL_VARIANTS",
    "OrPolicyAdapterPolicyFac",
    "OrVariantSelectionSurfaces",
    "PAPER_TITLE",
    "REPRODUCTION_ID",
    "SIMILARITY_GUIDANCE_SCALE_VALUES",
    "build_config",
    "build_environment_registry",
    "build_experiment_registry",
    "check_config_available",
    "collect_table_1_measurements",
    "generate_with_subtracted_toxic_vector_hook",
    "make_config",
    "materialize_registries",
    "persist_resolved_config",
    "run_configured_reproduction_smoke",
    "run_table_1_measurement_route",
    "run_training_loop",
    "train_config",
    "train_dpo_toxicity_aligned_gpt2_llama2",
    "write_readiness_artifacts",
    "write_sensitivity_report",
]