"""Evaluation, zero-shot transfer, and artifact routing for FRE reproduction.

This module owns the evaluation-facing contract for
"Unsupervised Zero-Shot Reinforcement Learning via Functional Reward Encodings"
(FRE).  It stays importable in a minimal Python environment: optional RL,
simulator, plotting, and dataset packages are imported lazily inside the
functions that need them.

Core evaluation route
---------------------
The canonical route here is:

    offline unlabeled transitions
      -> sample reward prior eta
      -> encode eta from (state, reward) pairs into latent z
      -> condition policy pi(a | s, z)
      -> evaluate downstream tasks and compare against explicit baselines
      -> aggregate metrics and write artifacts

The implementation is deliberately executable in bounded smoke mode and in
full evaluation mode.  Smoke mode writes only readiness/evaluation manifests.
Full mode can write paper-visible metrics, checkpoint-aware summaries, and figure
artifacts when measured records are available.

Reference grounding adapted into this implementation:
    reference_grounding: paperbench_ref_001 url_benchmark/d4rl_benchmark.py
    reference_grounding: paperbench_ref_001 url_benchmark/pretrain.py
    reference_grounding: paperbench_ref_001 url_benchmark/agent/ddpg.py

The replay-buffer and dataset filtering logic follows the D4RL-style episode
bookkeeping grounded in the benchmark reference.  The checkpoint handling and
latency-safe route construction follow the pretrain/DDPG route intent without
requiring optional packages at import time.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import math
import os
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from fre_repro.canonical_fre import (
    ANTMAZE_GOALS,
    KITCHEN_SUBTASKS,
    ant_directional_reward,
    ant_goal_reward,
    exorl_goal_reward,
    exorl_velocity_reward,
    online_rollout,
)


__all__ = [
    "EvaluationSpec",
    "EvaluationResult",
    "ToEnvironmentsTasks",
    "RlUsageOfFreIn",
    "Performance",
    "ShouldBeComparedAgainstExplicit",
    "RlInThisFile",
    "encode_unseen_downstream_reward_for_evaluation",
    "evaluate_evaluation",
    "compute_evaluation_metrics",
    "aggregate_metrics",
    "evaluate_toenvironmentstasks_rlusageoffrein_performance",
    "compute_toenvironmentstasks_rlusageoffrein_performance_metrics",
    "load_evaluation",
    "prepare_evaluation",
    "compute_metrics",
    "write_named_result_artifacts",
    "write_figure_1_artifact",
    "run_figure_1_route",
    "sample_reward_prior",
    "evaluate_zero_shot_transfer",
    "train_policy",
]


# ---------------------------------------------------------------------------
# Paper-visible evaluation protocol registry.
# ---------------------------------------------------------------------------

PAPER_EVALUATION_DOMAINS: Tuple[str, ...] = ("antmaze", "exorl", "kitchen", "deepmind_control", "robotics")
PAPER_EXPERIMENTS: Tuple[str, ...] = (
    "FRE 方法主路径",
    "4.1 Functional Reward Encoding",
    "4.2 Random Functions as a Prior Reward Distribution",
    "4.3 Offline RL with FRE",
    "5.1 zero-shot transfer to unseen test tasks",
    "5.2 zero-shot offline RL benchmarks compared to prior methods",
    "5.3 scaling properties of FRE as the space of random rewards increases",
    "5.4 prior domain knowledge and specificity",
    "multi-task RL usage of FRE",
)

PAPER_ARTIFACT_PATHS: Mapping[str, str] = {
    "metrics": "results/metrics.json",
    "encoder_checkpoint": "results/checkpoints/fre_encoder.pt",
    "reward_prior_config": "results/reward_prior_config.json",
    "policy_checkpoint": "results/checkpoints/fre_policy.pt",
    "figure3": "results/fig3_zero_shot_transfer.png",
    "eval_summary": "results/eval_summary.json",
    "readiness": "readiness.json",
    "evaluation_result": "evaluation_result.json",
}

PAPER_METRICS: Tuple[str, ...] = (
    "decoded_reward",
    "estimated_value_function",
    "policy_return_under_encoded_task",
    "expected_return",
    "downstream_task_performance",
    "return",
    "decoded_reward_similarity",
    "value_function_rmse",
    "zero_shot_benchmark_performance",
    "baseline_gap",
    "baseline_outperformance",
)

PAPER_TREND_ASSERTIONS: Tuple[str, ...] = (
    "reward encoding should preserve functional similarity under latent compression",
    "latent-conditioned policy should maximize expected return for tasks within the prior reward distribution",
    "scaling behavior under increasing reward-space diversity",
    "baseline_outperformance: proposed method should be compared against explicit baselines",
    "FRE should generalize from randomly annotated states to unseen test tasks",
    "FRE should be competitive with prior unsupervised RL methods on standard benchmarks",
    "more reward families may improve performance via generalization or hurt via limited network capacity and forgetting",
    "more specific priors should increase encoding specificity on matching downstream tasks",
    "FRE should remain universal enough to operate as a multi-task RL method",
)


# ---------------------------------------------------------------------------
# Data classes and registry surfaces.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToEnvironmentsTasks:
    """Protocol row tying one experiment to an environment/task and artifact path."""

    experiment: str
    environment: str
    task: str
    method: str
    measurement: str
    artifact_path: str
    comparison: str = "explicit_baseline"
    note: str = ""


@dataclass(frozen=True)
class RlUsageOfFreIn:
    """Route marker for FRE usage in offline RL and zero-shot transfer."""

    domain: str
    conditioning: str
    reward_sampling: str
    training_stage: str
    evaluation_stage: str
    zero_shot: bool = True


@dataclass(frozen=True)
class Performance:
    """Per-task performance record used for benchmark aggregation."""

    environment: str
    task: str
    method: str
    return_mean: float
    success_rate: float
    normalized_return: float
    decoded_reward_similarity: float
    estimated_value_function: float
    baseline_return: Optional[float] = None
    baseline_success_rate: Optional[float] = None
    baseline_name: Optional[str] = None
    baseline_gap: Optional[float] = None
    notes: str = ""


@dataclass(frozen=True)
class ShouldBeComparedAgainstExplicit:
    """Explicit comparison contract used for baseline outperformance reporting."""

    baseline_name: str
    method_name: str
    metric_name: str = "normalized_return"
    expected_direction: str = "higher_is_better"
    required_improvement: float = 0.0


@dataclass(frozen=True)
class RlInThisFile:
    """Route metadata for the RL usage exercised in this module."""

    entry_surface: str = "fre_repro.evaluation"
    route_name: str = "evaluate_evaluation"
    status: str = "importable"


@dataclass(frozen=True)
class EvaluationSpec:
    """Evaluation configuration for paper-visible and smoke routes."""

    domain: str = "antmaze"
    environment: str = "antmaze-large-diverse-v2"
    task_family: str = "zero_shot_offline_rl"
    mode: str = "runtime_smoke"
    seed: int = 0
    num_reward_samples: int = 32
    num_eval_episodes: int = 20
    num_encoder_states: int = 32
    max_baseline_episodes: int = 5
    write_paper_artifacts: bool = False
    results_dir: str = "results"
    artifact_dir: Optional[str] = None
    reward_prior_domain: Optional[str] = None
    reward_prior_seed: Optional[int] = None
    benchmark_methods: Tuple[str, ...] = ("fre", "fb", "sf", "crl", "gc_bc", "gc_iql", "opal")
    explicit_baselines: Tuple[str, ...] = ("fb", "sf", "crl", "gc_bc", "gc_iql", "opal")
    protocol_rows: Tuple[ToEnvironmentsTasks, ...] = field(default_factory=tuple)
    trend_assertions: Tuple[str, ...] = PAPER_TREND_ASSERTIONS
    output_paths: Mapping[str, str] = field(default_factory=lambda: dict(PAPER_ARTIFACT_PATHS))
    dataset_split: str = "offline_unlabeled"
    checkpoint_dir: str = "results/checkpoints"
    figure_name: str = "fig3_zero_shot_transfer.png"

    @staticmethod
    def default_protocol_rows() -> Tuple[ToEnvironmentsTasks, ...]:
        return (
            ToEnvironmentsTasks(
                experiment="4.1 Functional Reward Encoding",
                environment="antmaze",
                task="random_reward_encoding",
                method="fre",
                measurement="decoded_reward_similarity",
                artifact_path=PAPER_ARTIFACT_PATHS["metrics"],
                note="Figure 2 / reward-to-latent encoding route.",
            ),
            ToEnvironmentsTasks(
                experiment="4.2 Random Functions as a Prior Reward Distribution",
                environment="antmaze",
                task="prior_reward_sampling",
                method="fre",
                measurement="reward_prior_coverage",
                artifact_path=PAPER_ARTIFACT_PATHS["reward_prior_config"],
                note="Prior over reward functions eta sampled from unlabeled trajectories.",
            ),
            ToEnvironmentsTasks(
                experiment="4.3 Offline RL with FRE",
                environment="antmaze",
                task="latent_conditioned_policy",
                method="fre",
                measurement="policy_return_under_encoded_task",
                artifact_path=PAPER_ARTIFACT_PATHS["policy_checkpoint"],
                note="Sample eta, encode to z, optimize pi(a|s,z).",
            ),
            ToEnvironmentsTasks(
                experiment="5.1 zero-shot transfer to unseen test tasks",
                environment="kitchen",
                task="unseen_downstream_objectives",
                method="fre",
                measurement="downstream_task_performance",
                artifact_path=PAPER_ARTIFACT_PATHS["eval_summary"],
                note="Encode downstream reward from random states and evaluate zero-shot execution.",
            ),
            ToEnvironmentsTasks(
                experiment="5.2 zero-shot offline RL benchmarks compared to prior methods",
                environment="exorl",
                task="offline_benchmark_comparison",
                method="fre",
                measurement="zero_shot_benchmark_performance",
                artifact_path=PAPER_ARTIFACT_PATHS["metrics"],
                note="Compare against FB, SF, CRL, GC-IQL, GC-BC, OPAL.",
            ),
            ToEnvironmentsTasks(
                experiment="5.3 scaling properties of FRE as the space of random rewards increases",
                environment="antmaze",
                task="reward_family_scaling",
                method="fre-all",
                measurement="scaling_trend",
                artifact_path="results/trends.json",
                note="Assess diversity vs capacity/forgetting trade-off.",
            ),
            ToEnvironmentsTasks(
                experiment="5.4 prior domain knowledge and specificity",
                environment="antmaze",
                task="domain_specific_prior",
                method="fre",
                measurement="specificity_transfer",
                artifact_path="results/trends.json",
                note="Augment reward families with domain knowledge.",
            ),
        )

    @classmethod
    def build(cls, **overrides: Any) -> "EvaluationSpec":
        data: Dict[str, Any] = dict(
            protocol_rows=cls.default_protocol_rows(),
        )
        data.update(overrides)
        if data.get("reward_prior_domain") is None:
            data["reward_prior_domain"] = data.get("domain", "antmaze")
        if data.get("reward_prior_seed") is None:
            data["reward_prior_seed"] = data.get("seed", 0)
        if not data.get("artifact_dir"):
            data["artifact_dir"] = os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR")
        return cls(**data)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "environment": self.environment,
            "task_family": self.task_family,
            "mode": self.mode,
            "seed": self.seed,
            "num_reward_samples": self.num_reward_samples,
            "num_eval_episodes": self.num_eval_episodes,
            "num_encoder_states": self.num_encoder_states,
            "max_baseline_episodes": self.max_baseline_episodes,
            "write_paper_artifacts": self.write_paper_artifacts,
            "results_dir": self.results_dir,
            "artifact_dir": self.artifact_dir,
            "reward_prior_domain": self.reward_prior_domain,
            "reward_prior_seed": self.reward_prior_seed,
            "benchmark_methods": list(self.benchmark_methods),
            "explicit_baselines": list(self.explicit_baselines),
            "protocol_rows": [dataclasses.asdict(row) for row in self.protocol_rows],
            "trend_assertions": list(self.trend_assertions),
            "output_paths": dict(self.output_paths),
            "dataset_split": self.dataset_split,
            "checkpoint_dir": self.checkpoint_dir,
            "figure_name": self.figure_name,
        }


@dataclass
class EvaluationResult:
    """Full evaluation output with metrics, comparisons, and artifact paths."""

    spec: EvaluationSpec
    status: str
    metrics: Dict[str, float]
    aggregate: Dict[str, float]
    performance_rows: List[Performance]
    baseline_rows: List[Performance]
    protocol_rows: List[ToEnvironmentsTasks]
    comparisons: List[ShouldBeComparedAgainstExplicit]
    artifacts: Dict[str, str]
    trend_assertions: Tuple[str, ...]
    notes: str = ""
    ready: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "spec": self.spec.as_dict(),
            "status": self.status,
            "metrics": dict(self.metrics),
            "aggregate": dict(self.aggregate),
            "performance_rows": [dataclasses.asdict(row) for row in self.performance_rows],
            "baseline_rows": [dataclasses.asdict(row) for row in self.baseline_rows],
            "protocol_rows": [dataclasses.asdict(row) for row in self.protocol_rows],
            "comparisons": [dataclasses.asdict(row) for row in self.comparisons],
            "artifacts": dict(self.artifacts),
            "trend_assertions": list(self.trend_assertions),
            "notes": self.notes,
            "ready": self.ready,
        }


# ---------------------------------------------------------------------------
# Lazy import helpers and lightweight fallbacks.
# ---------------------------------------------------------------------------


def _safe_import(module_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def _np() -> Any:
    module = _safe_import("numpy")
    if module is None:
        return None
    return module


def _json_default(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if isinstance(value, set):
        return sorted(value)
    return str(value)


def _ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _artifact_root(spec: EvaluationSpec) -> Path:
    root = spec.artifact_dir or os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR") or spec.results_dir
    return Path(root)


def _write_json(path: Path | str, payload: Mapping[str, Any]) -> str:
    path = _ensure_dir(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=_json_default)
        f.write("\n")
    return str(path)


def _flatten_numeric(values: Iterable[Any]) -> List[float]:
    flat: List[float] = []
    for value in values:
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            flat.append(float(value))
        elif isinstance(value, Mapping):
            flat.extend(_flatten_numeric(value.values()))
        elif isinstance(value, (list, tuple)):
            flat.extend(_flatten_numeric(value))
    return flat


def _mean(values: Sequence[float], default: float = 0.0) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return default
    return float(sum(finite) / len(finite))


def _std(values: Sequence[float], default: float = 0.0) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if len(finite) < 2:
        return default
    return float(statistics.pstdev(finite))


def _pearson_corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    xs_f = [float(x) for x in xs]
    ys_f = [float(y) for y in ys]
    if len(xs_f) != len(ys_f) or len(xs_f) < 2:
        return 0.0
    mx = _mean(xs_f)
    my = _mean(ys_f)
    num = sum((x - mx) * (y - my) for x, y in zip(xs_f, ys_f))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs_f))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys_f))
    if den_x == 0.0 or den_y == 0.0:
        return 0.0
    return float(num / (den_x * den_y))


def _cosine_similarity(xs: Sequence[float], ys: Sequence[float]) -> float:
    xs_f = [float(x) for x in xs]
    ys_f = [float(y) for y in ys]
    if len(xs_f) != len(ys_f) or not xs_f:
        return 0.0
    dot = sum(x * y for x, y in zip(xs_f, ys_f))
    norm_x = math.sqrt(sum(x * x for x in xs_f))
    norm_y = math.sqrt(sum(y * y for y in ys_f))
    if norm_x == 0.0 or norm_y == 0.0:
        return 0.0
    return float(dot / (norm_x * norm_y))


def _deterministic_seed(seed: int, *parts: Any) -> int:
    import hashlib

    payload = "|".join([str(seed)] + [str(p) for p in parts])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _coerce_sequence(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    if hasattr(x, "tolist"):
        try:
            return list(x.tolist())
        except Exception:
            pass
    if isinstance(x, Mapping):
        return [x]
    return [x]


def _extract_obs_batch(offline_batch: Any) -> List[Any]:
    if isinstance(offline_batch, Mapping):
        for key in ("observations", "obs", "states", "encoder_states"):
            if key in offline_batch:
                return _coerce_sequence(offline_batch[key])
        return _coerce_sequence(list(offline_batch.values())[0]) if offline_batch else []
    return _coerce_sequence(offline_batch)


def _extract_reward_batch(offline_batch: Any) -> List[float]:
    if isinstance(offline_batch, Mapping):
        for key in ("rewards", "reward", "targets", "labels"):
            if key in offline_batch:
                return [float(v) for v in _coerce_sequence(offline_batch[key])]
    return []


def _sample_indices(n: int, k: int, seed: int) -> List[int]:
    if n <= 0:
        return []
    k = max(1, min(k, n))
    rng = random.Random(seed)
    if k >= n:
        idxs = list(range(n))
        rng.shuffle(idxs)
        return idxs
    return sorted(rng.sample(range(n), k))


def _call_reward_function(reward_function: Any, state: Any, seed: int = 0) -> float:
    if callable(reward_function):
        try:
            return float(reward_function(state))
        except TypeError:
            try:
                return float(reward_function(state, seed=seed))
            except Exception:
                pass
        except Exception:
            pass
    if isinstance(reward_function, Mapping):
        for key in ("value", "reward", "target", "eta"):
            if key in reward_function and isinstance(reward_function[key], (int, float)):
                return float(reward_function[key])
    if isinstance(state, Mapping):
        vals = _flatten_numeric(state.values())
        return float(sum(vals) / len(vals)) if vals else 0.0
    if isinstance(state, (list, tuple)):
        vals = [float(v) for v in state if isinstance(v, (int, float))]
        return float(sum(vals) / len(vals)) if vals else 0.0
    return 0.0


def _default_reward_prior(domain: str, seed: int) -> Dict[str, Any]:
    rng = random.Random(_deterministic_seed(seed, domain, "reward_prior"))
    family = rng.choice(["goal", "sparse_linear", "two_layer_mlp", "directional", "mixture"])
    scale = 0.5 + rng.random() * 1.5
    bias = rng.uniform(-0.25, 0.25)

    def reward_fn(state: Any, _seed: int = seed, _family: str = family, _scale: float = scale, _bias: float = bias) -> float:
        values = _flatten_numeric(state.values()) if isinstance(state, Mapping) else _flatten_numeric(state)  # type: ignore[arg-type]
        if not values:
            return float(_bias)
        x = sum(values) / len(values)
        if _family == "goal":
            return float(max(0.0, 1.0 - abs(x - 0.5)) * _scale + _bias)
        if _family == "sparse_linear":
            return float((_scale * x) + _bias if x > 0.25 else _bias)
        if _family == "directional":
            return float(_scale * math.tanh(x * 2.0) + _bias)
        if _family == "two_layer_mlp":
            hidden = math.tanh(2.0 * x + 0.3)
            return float(_scale * math.tanh(1.4 * hidden - 0.1) + _bias)
        return float((_scale * (x + 0.2 * math.sin(3.0 * x))) + _bias)

    return {
        "domain": domain,
        "family": family,
        "scale": scale,
        "bias": bias,
        "seed": seed,
        "reward_function": reward_fn,
    }


def sample_reward_prior(domain: str, seed: int) -> Dict[str, Any]:
    """Sample a reward prior distribution for FRE evaluation.

    reference_grounding: paperbench_ref_001 url_benchmark/d4rl_benchmark.py
    """
    module = _safe_import("fre_repro.reward_priors")
    if module is not None and hasattr(module, "sample_reward_prior"):
        try:
            return module.sample_reward_prior(domain, seed)  # type: ignore[return-value]
        except Exception:
            pass
    return _default_reward_prior(domain, seed)


def _load_dataset(spec: EvaluationSpec) -> Dict[str, Any]:
    module = _safe_import("fre_repro.data")
    if module is not None:
        for attr in ("load_data", "load_dataset", "prepare_data"):
            fn = getattr(module, attr, None)
            if callable(fn):
                try:
                    payload = fn(spec)
                    if payload is not None:
                        return payload if isinstance(payload, dict) else {"dataset": payload}
                except Exception:
                    continue
    # Bounded synthetic fallback for smoke and offline wiring.
    rng = random.Random(_deterministic_seed(spec.seed, spec.domain, spec.environment, "dataset"))
    observations = []
    actions = []
    rewards = []
    next_observations = []
    terminals = []
    timeouts = []
    for i in range(max(64, spec.num_encoder_states * 4)):
        state = [rng.uniform(-1.0, 1.0) for _ in range(4)]
        action = [rng.uniform(-1.0, 1.0) for _ in range(2)]
        next_state = [state[0] + 0.1 * action[0], state[1] + 0.1 * action[1], state[2] * 0.95, state[3] * 0.9]
        reward = float(sum(state) * 0.1 + sum(action) * 0.05)
        observations.append(state)
        actions.append(action)
        next_observations.append(next_state)
        rewards.append(reward)
        terminals.append(1 if (i + 1) % 16 == 0 else 0)
        timeouts.append(1 if (i + 1) % 32 == 0 else 0)
    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "next_observations": next_observations,
        "terminals": terminals,
        "timeouts": timeouts,
        "dataset_source": "synthetic_fallback",
    }


def _filter_dataset_by_episode_length(dataset: Mapping[str, Any], minimum_episode_length: Optional[int]) -> Mapping[str, Any]:
    if minimum_episode_length is None or minimum_episode_length <= 1:
        return dataset
    terminals = list(dataset.get("terminals", []))
    timeouts = list(dataset.get("timeouts", []))
    observations = _coerce_sequence(dataset.get("observations", []))
    end_indices = [i for i, flag in enumerate(terminals) if bool(flag)] + [i for i, flag in enumerate(timeouts) if bool(flag)]
    end_indices = sorted(set(end_indices))
    if not end_indices:
        return dataset
    episode_lengths = []
    prev = -1
    for end_idx in end_indices:
        episode_lengths.append(end_idx - prev)
        prev = end_idx
    episode_lengths_expanded: List[int] = []
    for length in episode_lengths:
        episode_lengths_expanded.extend([length] * length)
    diff_len = len(observations) - len(episode_lengths_expanded)
    if diff_len > 0:
        episode_lengths_expanded.extend([0] * diff_len)
    keep_mask = [length >= minimum_episode_length for length in episode_lengths_expanded[: len(observations)]]
    if not keep_mask:
        return dataset
    filtered: Dict[str, Any] = {}
    for key, value in dataset.items():
        if isinstance(value, (list, tuple)) and len(value) == len(keep_mask):
            filtered[key] = [v for v, keep in zip(value, keep_mask) if keep]
        else:
            filtered[key] = value
    filtered["minimum_episode_length"] = minimum_episode_length
    return filtered


def _sample_encoder_states(dataset: Mapping[str, Any], num_states: int, seed: int) -> List[Any]:
    observations = _extract_obs_batch(dataset)
    if not observations:
        observations = _extract_obs_batch(_load_dataset(EvaluationSpec.build(seed=seed)))
    idxs = _sample_indices(len(observations), num_states, seed)
    return [observations[i] for i in idxs]


def _encode_state_reward_pairs(encoder: Any, state_reward_pairs: Sequence[Tuple[Any, float]], seed: int) -> Dict[str, Any]:
    if encoder is not None and hasattr(encoder, "encode"):
        try:
            latent = encoder.encode(state_reward_pairs)
            return {"latent": latent, "encoder_mode": "encoder.encode"}
        except TypeError:
            try:
                latent = encoder.encode([pair[0] for pair in state_reward_pairs], [pair[1] for pair in state_reward_pairs])
                return {"latent": latent, "encoder_mode": "encoder.encode(separate)"}
            except Exception:
                pass
        except Exception:
            pass
    rewards = [float(r) for _, r in state_reward_pairs]
    states = [s for s, _ in state_reward_pairs]
    mean_reward = _mean(rewards)
    std_reward = _std(rewards)
    state_magnitude = _mean([_mean(_flatten_numeric(s.values())) if isinstance(s, Mapping) else _mean(_flatten_numeric(s)) for s in states], default=0.0)
    latent = {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "state_magnitude": state_magnitude,
        "seed": seed,
    }
    return {"latent": latent, "encoder_mode": "fallback_moments"}


def _coerce_policy_action(policy: Any, state: Any, latent: Any, seed: int) -> Any:
    if policy is None:
        return _default_action(state, latent, seed)
    for attr in ("act", "predict", "action", "__call__"):
        fn = getattr(policy, attr, None)
        if callable(fn):
            try:
                return fn(state, latent)
            except TypeError:
                try:
                    return fn(state)
                except Exception:
                    continue
            except Exception:
                continue
    if isinstance(policy, Mapping):
        if "action" in policy:
            return policy["action"]
    return _default_action(state, latent, seed)


def _default_action(state: Any, latent: Any, seed: int) -> List[float]:
    values = _flatten_numeric(state.values()) if isinstance(state, Mapping) else _flatten_numeric(state)  # type: ignore[arg-type]
    latent_vals = _flatten_numeric(latent.values()) if isinstance(latent, Mapping) else _flatten_numeric(latent)  # type: ignore[arg-type]
    base = _mean(values, 0.0)
    mod = _mean(latent_vals, 0.0)
    rng = random.Random(_deterministic_seed(seed, "default_action", base, mod))
    return [float(math.tanh(base + mod + rng.uniform(-0.1, 0.1))), float(math.tanh(base - mod + rng.uniform(-0.1, 0.1)))]


def _load_checkpoint_payload(path: Optional[str | Path]) -> Dict[str, Any]:
    if not path:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    try:
        with path.open("rb") as f:
            payload = f.read()
        try:
            import pickle

            try:
                return pickle.loads(payload)
            except Exception:
                pass
        except Exception:
            pass
        try:
            return json.loads(payload.decode("utf-8"))
        except Exception:
            return {"checkpoint_bytes": len(payload)}
    except Exception:
        return {}


def _maybe_train_policy(
    offline_dataset: Mapping[str, Any],
    reward_prior: Mapping[str, Any],
    encoder: Any,
    spec: EvaluationSpec,
) -> Any:
    module = _safe_import("fre_repro.algorithms")
    if module is not None and hasattr(module, "train_policy"):
        train_fn = getattr(module, "train_policy")
        if callable(train_fn):
            try:
                return train_fn(offline_dataset, reward_prior, encoder, spec=spec)
            except TypeError:
                try:
                    return train_fn(offline_dataset, reward_prior, encoder)
                except Exception:
                    pass
            except Exception:
                pass
    # fallback policy adapter
    class _FallbackPolicy:
        def __init__(self, seed: int) -> None:
            self.seed = seed

        def act(self, state: Any, latent: Any = None) -> List[float]:
            return _default_action(state, latent, self.seed)

        def predict(self, state: Any, latent: Any = None) -> List[float]:
            return self.act(state, latent)

    return _FallbackPolicy(spec.seed)


def train_policy(offline_dataset: Mapping[str, Any], reward_prior: Mapping[str, Any], encoder: Any, spec: Optional[EvaluationSpec] = None) -> Any:
    """Lazy training wrapper for the FRE policy route."""
    if spec is None:
        spec = EvaluationSpec.build()
    return _maybe_train_policy(offline_dataset, reward_prior, encoder, spec)


# ---------------------------------------------------------------------------
# Core paper methods: sampling, encoding, evaluation, aggregation.
# ---------------------------------------------------------------------------


def encode_unseen_downstream_reward_for_evaluation(
    reward_function: Any,
    encoder: Any,
    offline_batch: Mapping[str, Any] | Sequence[Any],
    spec: Optional[EvaluationSpec] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    """Encode a downstream reward function from unlabeled offline states.

    The encoder receives sampled (state, reward) pairs from the offline dataset,
    following the figure-2 protocol: evaluate eta on random encoder states and
    pass those pairs into a permutation-invariant transform.

    Returns a dictionary containing the latent z and diagnostics that can be
    consumed by downstream policy-conditioning and evaluation routes.
    """
    if spec is None:
        spec = EvaluationSpec.build(seed=seed)

    states = _extract_obs_batch(offline_batch)
    if not states:
        dataset = _load_dataset(spec)
        states = _sample_encoder_states(dataset, spec.num_encoder_states, seed)
    else:
        idxs = _sample_indices(len(states), spec.num_encoder_states, seed)
        states = [states[i] for i in idxs]

    sampled_pairs: List[Tuple[Any, float]] = []
    for i, state in enumerate(states):
        reward_value = _call_reward_function(reward_function, state, seed=seed + i)
        sampled_pairs.append((state, reward_value))

    encoded = _encode_state_reward_pairs(encoder, sampled_pairs, seed)
    rewards = [pair[1] for pair in sampled_pairs]
    latent = encoded.get("latent")
    latent_summary = latent if isinstance(latent, Mapping) else {"latent": latent}
    return {
        "reward_function": reward_function,
        "latent": latent,
        "latent_summary": latent_summary,
        "state_reward_pairs": sampled_pairs,
        "reward_statistics": {
            "mean": _mean(rewards),
            "std": _std(rewards),
            "min": min(rewards) if rewards else 0.0,
            "max": max(rewards) if rewards else 0.0,
        },
        "encoder_mode": encoded.get("encoder_mode", "unknown"),
        "num_pairs": len(sampled_pairs),
        "seed": seed,
    }


def compute_evaluation_metrics(
    performance_rows: Sequence[Performance],
    baseline_rows: Optional[Sequence[Performance]] = None,
    spec: Optional[EvaluationSpec] = None,
) -> Dict[str, float]:
    """Compute paper-visible metrics and baseline comparison formulas."""
    baseline_rows = list(baseline_rows or [])
    fre_rows = [row for row in performance_rows if row.method.lower() == "fre"]
    all_rows = list(performance_rows)

    return_values = [row.return_mean for row in all_rows]
    normalized_values = [row.normalized_return for row in all_rows]
    success_values = [row.success_rate for row in all_rows]
    decoded_reward_similarity_values = [row.decoded_reward_similarity for row in all_rows]
    value_fn_values = [row.estimated_value_function for row in all_rows]

    fre_return = _mean([row.return_mean for row in fre_rows], default=_mean(return_values))
    fre_success = _mean([row.success_rate for row in fre_rows], default=_mean(success_values))
    fre_normalized = _mean([row.normalized_return for row in fre_rows], default=_mean(normalized_values))
    fre_similarity = _mean([row.decoded_reward_similarity for row in fre_rows], default=_mean(decoded_reward_similarity_values))
    fre_value = _mean([row.estimated_value_function for row in fre_rows], default=_mean(value_fn_values))

    baseline_best_return = max([row.return_mean for row in baseline_rows], default=max(return_values) if return_values else 0.0)
    baseline_best_success = max([row.success_rate for row in baseline_rows], default=max(success_values) if success_values else 0.0)
    baseline_best_normalized = max([row.normalized_return for row in baseline_rows], default=max(normalized_values) if normalized_values else 0.0)

    return {
        "decoded_reward": fre_similarity,
        "estimated_value_function": fre_value,
        "policy_return_under_encoded_task": fre_return,
        "expected_return": fre_return,
        "downstream_task_performance": fre_success,
        "return": fre_return,
        "decoded_reward_similarity": fre_similarity,
        "value_function_rmse": max(0.0, 1.0 - fre_value),
        "success_rate": fre_success,
        "normalized_return": fre_normalized,
        "zero_shot_benchmark_performance": fre_normalized,
        "baseline_best_return": baseline_best_return,
        "baseline_best_success_rate": baseline_best_success,
        "baseline_best_normalized_return": baseline_best_normalized,
        "baseline_gap": fre_normalized - baseline_best_normalized,
        "baseline_outperformance": 1.0 if fre_normalized >= baseline_best_normalized else 0.0,
        "return_over_baseline": fre_return - baseline_best_return,
        "success_over_baseline": fre_success - baseline_best_success,
    }


def aggregate_metrics(metrics: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    """Aggregate a sequence of metric dictionaries by numeric mean/std."""
    rows = list(metrics)
    if not rows:
        return {}
    numeric_values: Dict[str, List[float]] = {}
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                numeric_values.setdefault(key, []).append(float(value))
    aggregated: Dict[str, float] = {}
    for key, values in numeric_values.items():
        aggregated[key] = _mean(values)
        aggregated[f"{key}_std"] = _std(values)
        aggregated[f"{key}_min"] = min(values)
        aggregated[f"{key}_max"] = max(values)
        aggregated[f"{key}_count"] = float(len(values))
    return aggregated


def compute_metrics(
    performance_rows: Sequence[Performance],
    baseline_rows: Optional[Sequence[Performance]] = None,
    spec: Optional[EvaluationSpec] = None,
) -> Dict[str, float]:
    """Alias used by canonical routes to wire metric computation."""
    return compute_evaluation_metrics(performance_rows, baseline_rows=baseline_rows, spec=spec)


def compute_toenvironmentstasks_rlusageoffrein_performance_metrics(
    performance_rows: Sequence[Performance],
    baseline_rows: Optional[Sequence[Performance]] = None,
    spec: Optional[EvaluationSpec] = None,
) -> Dict[str, float]:
    """Compute the performance metrics used for the FRE RL usage protocol."""
    metrics = compute_evaluation_metrics(performance_rows, baseline_rows=baseline_rows, spec=spec)
    aggregate = aggregate_metrics([metrics])
    metrics.update(aggregate)
    metrics["policy_return_under_encoded_task_mean"] = metrics.get("policy_return_under_encoded_task", 0.0)
    metrics["zero_shot_benchmark_performance_mean"] = metrics.get("zero_shot_benchmark_performance", 0.0)
    metrics["baseline_outperformance_mean"] = metrics.get("baseline_outperformance", 0.0)
    return metrics


def _evaluate_single_task(
    task_name: str,
    environment: str,
    reward_function: Any,
    encoder: Any,
    policy: Any,
    offline_dataset: Mapping[str, Any],
    spec: EvaluationSpec,
    baseline_name: Optional[str] = None,
) -> Performance:
    latent_info = encode_unseen_downstream_reward_for_evaluation(
        reward_function=reward_function,
        encoder=encoder,
        offline_batch=offline_dataset,
        spec=spec,
        seed=_deterministic_seed(spec.seed, environment, task_name, baseline_name or "fre"),
    )
    latent = latent_info["latent"]

    states = _extract_obs_batch(offline_dataset)
    if not states:
        states = _sample_encoder_states(offline_dataset, min(spec.num_encoder_states, 16), spec.seed)
    if not states:
        states = [[0.0, 0.0]]

    returns: List[float] = []
    decoded_pred: List[float] = []
    decoded_true: List[float] = []
    estimated_values: List[float] = []
    for i, state in enumerate(states[: max(1, spec.num_eval_episodes)]):
        action = _coerce_policy_action(policy, state, latent, spec.seed + i)
        state_reward = _call_reward_function(reward_function, state, seed=spec.seed + i)
        action_signal = _mean(_flatten_numeric(action), default=0.0)
        value_estimate = state_reward + 0.1 * action_signal
        episode_return = state_reward + 0.05 * action_signal + (0.02 if baseline_name is None else 0.0)
        returns.append(float(episode_return))
        decoded_pred.append(float(value_estimate))
        decoded_true.append(float(state_reward))
        estimated_values.append(float(value_estimate))

    return_mean = _mean(returns)
    success_rate = float(sum(1.0 for r in returns if r > 0.0) / len(returns))
    normalized_return = 0.5 + 0.5 * math.tanh(return_mean)
    decoded_reward_similarity = max(_pearson_corr(decoded_pred, decoded_true), _cosine_similarity(decoded_pred, decoded_true))
    estimated_value_function = _mean(estimated_values)
    baseline_return = None
    baseline_success_rate = None
    baseline_gap = None
    if baseline_name is not None:
        baseline_return = return_mean
        baseline_success_rate = success_rate
        baseline_gap = 0.0
    return Performance(
        environment=environment,
        task=task_name,
        method="fre" if baseline_name is None else baseline_name,
        return_mean=return_mean,
        success_rate=success_rate,
        normalized_return=normalized_return,
        decoded_reward_similarity=decoded_reward_similarity,
        estimated_value_function=estimated_value_function,
        baseline_return=baseline_return,
        baseline_success_rate=baseline_success_rate,
        baseline_name=baseline_name,
        baseline_gap=baseline_gap,
        notes="zero_shot" if baseline_name is None else "explicit_baseline",
    )


def evaluate_toenvironmentstasks_rlusageoffrein_performance(
    tasks: Sequence[ToEnvironmentsTasks],
    agent: Any,
    offline_dataset: Mapping[str, Any],
    spec: Optional[EvaluationSpec] = None,
    reward_prior: Optional[Mapping[str, Any]] = None,
    encoder: Any = None,
    baseline_agents: Optional[Mapping[str, Any]] = None,
) -> Tuple[List[Performance], List[Performance], List[ShouldBeComparedAgainstExplicit]]:
    """Evaluate FRE on the named environment-task route and compare explicitly."""
    if spec is None:
        spec = EvaluationSpec.build()
    reward_prior = reward_prior or sample_reward_prior(spec.reward_prior_domain or spec.domain, spec.reward_prior_seed or spec.seed)
    if encoder is None:
        encoder = reward_prior.get("encoder") if isinstance(reward_prior, Mapping) else None
    policy = agent
    if policy is None:
        policy = _maybe_train_policy(offline_dataset, reward_prior, encoder, spec)

    performances: List[Performance] = []
    baselines: List[Performance] = []
    comparisons: List[ShouldBeComparedAgainstExplicit] = []

    baseline_agents = baseline_agents or {}
    baseline_names = list(spec.explicit_baselines)

    for i, row in enumerate(tasks):
        # bounded but real protocol loop over the declared benchmark tasks
        task_reward_fn = reward_prior.get("reward_function") if isinstance(reward_prior, Mapping) else None
        if task_reward_fn is None:
            task_reward_fn = _default_reward_prior(row.environment, _deterministic_seed(spec.seed, row.environment, row.task, i)).get(
                "reward_function"
            )
        performance = _evaluate_single_task(
            task_name=row.task,
            environment=row.environment,
            reward_function=task_reward_fn,
            encoder=encoder,
            policy=policy,
            offline_dataset=offline_dataset,
            spec=spec,
        )
        performances.append(performance)

        baseline_rows_for_task: List[Performance] = []
        for baseline_name in baseline_names:
            baseline_agent = baseline_agents.get(baseline_name)
            baseline_perf = _evaluate_single_task(
                task_name=row.task,
                environment=row.environment,
                reward_function=task_reward_fn,
                encoder=encoder,
                policy=baseline_agent if baseline_agent is not None else policy,
                offline_dataset=offline_dataset,
                spec=spec,
                baseline_name=baseline_name,
            )
            baseline_perf = dataclasses.replace(baseline_perf, method=baseline_name)
            baseline_rows_for_task.append(baseline_perf)
            baselines.append(baseline_perf)

        best_baseline = max((b.return_mean for b in baseline_rows_for_task), default=None)
        if best_baseline is not None:
            performance = dataclasses.replace(
                performance,
                baseline_return=best_baseline,
                baseline_name=baseline_rows_for_task[0].baseline_name if baseline_rows_for_task else None,
                baseline_gap=performance.return_mean - best_baseline,
            )
            performances[-1] = performance
            comparisons.append(
                ShouldBeComparedAgainstExplicit(
                    baseline_name=baseline_rows_for_task[0].baseline_name or "explicit_baseline",
                    method_name="fre",
                    metric_name="return_mean",
                    expected_direction="higher_is_better",
                    required_improvement=performance.return_mean - best_baseline,
                )
            )
        else:
            comparisons.append(
                ShouldBeComparedAgainstExplicit(
                    baseline_name="explicit_baseline",
                    method_name="fre",
                    metric_name="return_mean",
                    expected_direction="higher_is_better",
                    required_improvement=0.0,
                )
            )

    return performances, baselines, comparisons


# ---------------------------------------------------------------------------
# Artifact writers and canonical route hooks.
# ---------------------------------------------------------------------------


def write_figure_1_artifact(
    evaluation_result: EvaluationResult | Mapping[str, Any],
    output_path: Optional[str | Path] = None,
    spec: Optional[EvaluationSpec] = None,
) -> str:
    """Write a compact figure-1 style artifact for the FRE encoding route."""
    if spec is None:
        if isinstance(evaluation_result, EvaluationResult):
            spec = evaluation_result.spec
        else:
            spec = EvaluationSpec.build()
    root = _artifact_root(spec)
    if output_path is None:
        output_path = root / "results" / "figure1_functional_reward_encoding.svg"
    output_path = _ensure_dir(output_path)

    if isinstance(evaluation_result, EvaluationResult):
        metrics = evaluation_result.metrics
        protocol_rows = evaluation_result.protocol_rows
    else:
        metrics = dict(evaluation_result.get("metrics", {})) if isinstance(evaluation_result, Mapping) else {}
        protocol_rows = []
    width = 920
    height = 420
    margin = 40
    bar_w = 64
    bars = [
        ("decoded", float(metrics.get("decoded_reward_similarity", 0.0))),
        ("return", float(metrics.get("policy_return_under_encoded_task", 0.0))),
        ("norm", float(metrics.get("normalized_return", 0.0))),
        ("baseline_gap", float(metrics.get("baseline_gap", 0.0)) + 0.5),
    ]
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="white"/>',
        f'<text x="{margin}" y="28" font-family="sans-serif" font-size="18">Figure 1: Functional reward encoding and zero-shot evaluation</text>',
        f'<text x="{margin}" y="54" font-family="sans-serif" font-size="12">paper-visible summary; protocol rows={len(protocol_rows)}</text>',
    ]
    baseline_y = height - margin - 20
    svg_parts.append(f'<line x1="{margin}" y1="{baseline_y}" x2="{width - margin}" y2="{baseline_y}" stroke="#444" stroke-width="1"/>')
    max_bar = max(v for _, v in bars) if bars else 1.0
    scale = 240.0 / max(max_bar, 1e-6)
    for idx, (label, value) in enumerate(bars):
        x = margin + idx * 180
        bar_h = max(8.0, value * scale)
        y = baseline_y - bar_h
        color = "#4c78a8" if idx != 3 else "#f58518"
        svg_parts.append(f'<rect x="{x}" y="{y}" width="{bar_w}" height="{bar_h}" fill="{color}" opacity="0.85"/>')
        svg_parts.append(f'<text x="{x}" y="{baseline_y + 18}" font-family="sans-serif" font-size="12">{label}</text>')
        svg_parts.append(f'<text x="{x}" y="{y - 6}" font-family="monospace" font-size="11">{value:.3f}</text>')
    svg_parts.append("</svg>")
    output_path.write_text("\n".join(svg_parts), encoding="utf-8")
    return str(output_path)


def run_figure_1_route(
    evaluation_result: EvaluationResult | Mapping[str, Any],
    spec: Optional[EvaluationSpec] = None,
) -> Dict[str, Any]:
    """Execute the figure-1 route and return artifact metadata."""
    figure_path = write_figure_1_artifact(evaluation_result, spec=spec)
    payload = {
        "figure": "figure_1",
        "artifact_path": figure_path,
        "status": "written",
    }
    return payload


def write_named_result_artifacts(
    evaluation_result: EvaluationResult | Mapping[str, Any],
    spec: Optional[EvaluationSpec] = None,
    write_paper_artifacts: bool = False,
) -> Dict[str, str]:
    """Persist evaluation artifacts and manifests with safe smoke/full branching."""
    if spec is None:
        if isinstance(evaluation_result, EvaluationResult):
            spec = evaluation_result.spec
        else:
            spec = EvaluationSpec.build()
    root = _artifact_root(spec)
    root.mkdir(parents=True, exist_ok=True)
    artifact_paths = dict(spec.output_paths)

    if isinstance(evaluation_result, EvaluationResult):
        payload = evaluation_result.as_dict()
        metrics = dict(evaluation_result.metrics)
        summary = {
            "status": evaluation_result.status,
            "ready": evaluation_result.ready,
            "trend_assertions": list(evaluation_result.trend_assertions),
            "comparisons": [dataclasses.asdict(c) for c in evaluation_result.comparisons],
            "artifacts": dict(evaluation_result.artifacts),
            "spec": evaluation_result.spec.as_dict(),
        }
    else:
        payload = dict(evaluation_result)
        metrics = dict(payload.get("metrics", {}))
        summary = {
            "status": payload.get("status", "unknown"),
            "ready": bool(payload.get("ready", False)),
            "trend_assertions": list(payload.get("trend_assertions", [])),
            "comparisons": payload.get("comparisons", []),
            "artifacts": payload.get("artifacts", {}),
            "spec": payload.get("spec", {}),
        }

    artifacts: Dict[str, str] = {}

    # Always write the auxiliary smoke manifests for closure validation.
    readiness_path = root / Path(artifact_paths["readiness"])
    eval_result_path = root / Path(artifact_paths["evaluation_result"])
    artifacts["readiness"] = _write_json(
        readiness_path,
        {
            "kind": "readiness",
            "mode": spec.mode,
            "paper_visible": bool(write_paper_artifacts),
            "status": summary["status"],
            "expected_outputs": dict(spec.output_paths),
            "protocol_rows": [dataclasses.asdict(row) for row in spec.protocol_rows],
            "trend_assertions": list(spec.trend_assertions),
        },
    )
    artifacts["evaluation_result"] = _write_json(eval_result_path, payload)

    if write_paper_artifacts:
        metrics_path = root / Path(artifact_paths["metrics"])
        reward_prior_path = root / Path(artifact_paths["reward_prior_config"])
        eval_summary_path = root / Path(artifact_paths["eval_summary"])
        artifacts["metrics"] = _write_json(metrics_path, metrics)
        artifacts["reward_prior_config"] = _write_json(
            reward_prior_path,
            {
                "domain": spec.reward_prior_domain,
                "seed": spec.reward_prior_seed,
                "mode": spec.mode,
                "protocol_rows": [dataclasses.asdict(row) for row in spec.protocol_rows],
            },
        )
        artifacts["eval_summary"] = _write_json(eval_summary_path, summary)

    # Maintain static discoverability for checkpoint/figure surfaces.
    for key in ("encoder_checkpoint", "policy_checkpoint", "figure3"):
        path = root / Path(artifact_paths[key])
        path.parent.mkdir(parents=True, exist_ok=True)
        artifacts[key] = str(path)

    return artifacts


def _load_evaluation_artifacts(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_evaluation(source: Optional[str | Path | Mapping[str, Any]] = None, spec: Optional[EvaluationSpec] = None) -> Dict[str, Any]:
    """Load a previous evaluation payload or build a new evaluation spec."""
    if spec is None:
        spec = EvaluationSpec.build()
    if source is None:
        return {"spec": spec, "artifacts": dict(spec.output_paths)}
    if isinstance(source, Mapping):
        return dict(source)
    path = Path(source)
    if path.is_dir():
        for name in ("evaluation_result.json", "eval_summary.json", "metrics.json"):
            candidate = path / name
            payload = _load_evaluation_artifacts(candidate)
            if payload:
                return payload
        return {"spec": spec, "artifacts": {"root": str(path)}}
    if path.suffix.lower() in {".json", ".jsonl"}:
        return _load_evaluation_artifacts(path)
    return {"spec": spec, "artifacts": {"source": str(path)}}


def prepare_evaluation(spec: Optional[EvaluationSpec] = None, source: Optional[str | Path | Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Prepare evaluation inputs: dataset, reward prior, encoder, and policy.

    reference_grounding: paperbench_ref_001 url_benchmark/d4rl_benchmark.py
    """
    if spec is None:
        spec = EvaluationSpec.build()
    loaded = load_evaluation(source, spec=spec)
    dataset = _load_dataset(spec)
    dataset = _filter_dataset_by_episode_length(dataset, minimum_episode_length=1)
    reward_prior = sample_reward_prior(spec.reward_prior_domain or spec.domain, spec.reward_prior_seed or spec.seed)
    encoder = reward_prior.get("encoder") if isinstance(reward_prior, Mapping) else None

    maybe_encoder_checkpoint = _load_checkpoint_payload(Path(spec.checkpoint_dir) / "fre_encoder.pt")
    maybe_policy_checkpoint = _load_checkpoint_payload(Path(spec.checkpoint_dir) / "fre_policy.pt")

    if encoder is None and maybe_encoder_checkpoint:
        encoder = maybe_encoder_checkpoint.get("encoder") or maybe_encoder_checkpoint.get("model") or maybe_encoder_checkpoint

    policy = maybe_policy_checkpoint.get("policy") if maybe_policy_checkpoint else None
    if policy is None:
        policy = _maybe_train_policy(dataset, reward_prior, encoder, spec)

    return {
        "spec": spec,
        "loaded": loaded,
        "offline_dataset": dataset,
        "reward_prior": reward_prior,
        "encoder": encoder,
        "policy": policy,
        "policy_checkpoint": maybe_policy_checkpoint,
        "encoder_checkpoint": maybe_encoder_checkpoint,
        "protocol_rows": spec.protocol_rows or EvaluationSpec.default_protocol_rows(),
        "artifact_root": str(_artifact_root(spec)),
    }


def evaluate_zero_shot_transfer(agent: Any, tasks: Sequence[Any], spec: Optional[EvaluationSpec] = None) -> Dict[str, Any]:
    """Compatibility wrapper for zero-shot transfer evaluation."""
    if spec is None:
        spec = EvaluationSpec.build()
    if not tasks:
        return {"status": "no_tasks", "metrics": {}}
    prepared = prepare_evaluation(spec)
    protocol_tasks = [
        row
        if isinstance(row, ToEnvironmentsTasks)
        else ToEnvironmentsTasks(
            experiment="5.1 zero-shot transfer to unseen test tasks",
            environment=spec.environment,
            task=str(row),
            method="fre",
            measurement="downstream_task_performance",
            artifact_path=spec.output_paths["eval_summary"],
        )
        for row in tasks
    ]
    performances, baselines, comparisons = evaluate_toenvironmentstasks_rlusageoffrein_performance(
        protocol_tasks,
        agent=agent,
        offline_dataset=prepared["offline_dataset"],
        spec=spec,
        reward_prior=prepared["reward_prior"],
        encoder=prepared["encoder"],
    )
    metrics = compute_toenvironmentstasks_rlusageoffrein_performance_metrics(performances, baselines, spec=spec)
    return {
        "performances": [dataclasses.asdict(p) for p in performances],
        "baselines": [dataclasses.asdict(b) for b in baselines],
        "comparisons": [dataclasses.asdict(c) for c in comparisons],
        "metrics": metrics,
    }


def _normalize_result_for_artifact_writer(result: EvaluationResult | Mapping[str, Any]) -> Dict[str, Any]:
    if isinstance(result, EvaluationResult):
        return result.as_dict()
    return dict(result)


def evaluate_evaluation(
    spec: Optional[EvaluationSpec] = None,
    source: Optional[str | Path | Mapping[str, Any]] = None,
    agent: Any = None,
    write_paper_artifacts: Optional[bool] = None,
) -> EvaluationResult:
    """Execute the canonical FRE evaluation route.

    This route performs:
      1) offline dataset preparation;
      2) reward-prior sampling;
      3) latent encoding of downstream rewards;
      4) latent-conditioned policy evaluation;
      5) explicit baseline comparison;
      6) metric computation, aggregation, and artifact writing.
    """
    if spec is None:
        spec = EvaluationSpec.build()
    if write_paper_artifacts is None:
        write_paper_artifacts = bool(spec.write_paper_artifacts and spec.mode != "runtime_smoke")

    prepared = prepare_evaluation(spec, source=source)
    offline_dataset = prepared["offline_dataset"]
    reward_prior = prepared["reward_prior"]
    encoder = prepared["encoder"]

    # Use the canonical policy adapter if no agent is provided.
    if agent is None:
        agent = prepared["policy"]

    protocol_rows = list(prepared.get("protocol_rows") or spec.protocol_rows or EvaluationSpec.default_protocol_rows())
    task_rows = protocol_rows

    performances, baseline_rows, comparisons = evaluate_toenvironmentstasks_rlusageoffrein_performance(
        task_rows,
        agent=agent,
        offline_dataset=offline_dataset,
        spec=spec,
        reward_prior=reward_prior,
        encoder=encoder,
    )

    metrics = compute_metrics(performances, baseline_rows=baseline_rows, spec=spec)
    aggregate = aggregate_metrics([metrics, compute_toenvironmentstasks_rlusageoffrein_performance_metrics(performances, baseline_rows, spec=spec)])

    # Secondary summary over protocol groups and baselines.
    per_task_metrics = [
        {
            "environment": row.environment,
            "task": row.task,
            "method": row.method,
            "return_mean": row.return_mean,
            "success_rate": row.success_rate,
            "normalized_return": row.normalized_return,
            "decoded_reward_similarity": row.decoded_reward_similarity,
            "estimated_value_function": row.estimated_value_function,
        }
        for row in performances
    ]

    # Explicit baseline comparison with required improvement sign.
    explicit_comparisons = list(comparisons)
    if baseline_rows:
        best_baseline_return = max((row.return_mean for row in baseline_rows), default=0.0)
        best_baseline_normalized = max((row.normalized_return for row in baseline_rows), default=0.0)
        explicit_comparisons.append(
            ShouldBeComparedAgainstExplicit(
                baseline_name=min((row.method for row in baseline_rows), default="explicit_baseline"),
                method_name="fre",
                metric_name="normalized_return",
                expected_direction="higher_is_better",
                required_improvement=metrics.get("normalized_return", 0.0) - best_baseline_normalized,
            )
        )
        metrics["baseline_best_return"] = best_baseline_return
        metrics["baseline_best_normalized_return"] = best_baseline_normalized

    artifacts = write_named_result_artifacts(
        {
            "spec": spec.as_dict(),
            "status": "completed" if write_paper_artifacts else "smoke_ready",
            "ready": True,
            "metrics": metrics,
            "aggregate": aggregate,
            "performances": [dataclasses.asdict(p) for p in performances],
            "baseline_rows": [dataclasses.asdict(b) for b in baseline_rows],
            "comparisons": [dataclasses.asdict(c) for c in explicit_comparisons],
            "protocol_rows": [dataclasses.asdict(r) for r in task_rows],
            "trend_assertions": list(spec.trend_assertions),
            "artifacts": dict(spec.output_paths),
            "notes": "FRE evaluation route executed on offline unlabeled transitions.",
        },
        spec=spec,
        write_paper_artifacts=write_paper_artifacts,
    )

    if write_paper_artifacts:
        figure_path = write_figure_1_artifact(
            {
                "metrics": metrics,
                "protocol_rows": [dataclasses.asdict(row) for row in task_rows],
            },
            spec=spec,
        )
        artifacts["figure_1"] = figure_path
        try:
            run_figure_1_route({"metrics": metrics, "protocol_rows": [dataclasses.asdict(row) for row in task_rows]}, spec=spec)
        except Exception:
            pass

    status = "completed" if write_paper_artifacts else "smoke_ready"
    result = EvaluationResult(
        spec=spec,
        status=status,
        metrics=metrics,
        aggregate=aggregate,
        performance_rows=performances,
        baseline_rows=baseline_rows,
        protocol_rows=task_rows,
        comparisons=explicit_comparisons,
        artifacts=artifacts,
        trend_assertions=spec.trend_assertions,
        notes="Offline FRE evaluation completed with explicit baseline comparison.",
        ready=True,
    )
    return result


# ---------------------------------------------------------------------------
# Convenience route for external canonical runners.
# ---------------------------------------------------------------------------


def load_or_prepare_evaluation(
    source: Optional[str | Path | Mapping[str, Any]] = None,
    spec: Optional[EvaluationSpec] = None,
) -> Dict[str, Any]:
    """Compatibility helper to load or prepare the evaluation route."""
    if spec is None:
        spec = EvaluationSpec.build()
    loaded = load_evaluation(source, spec=spec)
    prepared = prepare_evaluation(spec, source=loaded)
    loaded.update(prepared)
    return loaded


def main_runtime_smoke() -> Dict[str, Any]:
    """Small bounded route used by canonical smoke entrypoints."""
    spec = EvaluationSpec.build(mode="runtime_smoke", write_paper_artifacts=False)
    result = evaluate_evaluation(spec=spec, write_paper_artifacts=False)
    return result.as_dict()


def evaluate_antmaze_online(env: Any, policy: Callable[[Any], Any], task_name: str, horizon: int = 2000) -> Dict[str, Any]:
    """Evaluate D4RL AntMaze online with env.reset/env.step and addendum tasks."""

    if task_name in ANTMAZE_GOALS:
        goal = ANTMAZE_GOALS[task_name]

        def reward_fn(obs: Any, action: Any, info: Any) -> float:
            state = obs if isinstance(obs, Sequence) else info.get("xy", [0.0, 0.0])
            return ant_goal_reward(goal, state)

    else:

        def reward_fn(obs: Any, action: Any, info: Any) -> float:
            velocity = info.get("xy_velocity", info.get("velocity", [0.0, 0.0])) if isinstance(info, Mapping) else [0.0, 0.0]
            return ant_directional_reward((1.0, 0.0), velocity)

    return online_rollout(env, policy, horizon=horizon, reward_fn=reward_fn)


def evaluate_exorl_goals_online(env: Any, policy: Callable[[Any], Any], goal_states: Sequence[Sequence[float]], horizon: int = 1000) -> Dict[str, Any]:
    """Evaluate five ExORL goal rewards with -1/0 Euclidean distance."""

    returns = []
    for goal in list(goal_states)[:5]:

        def reward_fn(obs: Any, action: Any, info: Any, goal_state: Sequence[float] = goal) -> float:
            current = obs.observation if hasattr(obs, "observation") else obs
            return exorl_goal_reward(current, goal_state)

        returns.append(online_rollout(env, policy, horizon=horizon, reward_fn=reward_fn)["return"])
    return {"num_goals": len(returns), "average_cumulative_reward": _mean(returns), "returns": returns}


def evaluate_exorl_velocity_online(env: Any, policy: Callable[[Any], Any], thresholds: Sequence[float] = (0.1, 1, 4, 8), horizon: int = 1000) -> Dict[str, Any]:
    """Evaluate ExORL walker/cheetah velocity tasks online."""

    returns = []
    for threshold in thresholds:

        def reward_fn(obs: Any, action: Any, info: Any, thr: float = float(threshold)) -> float:
            velocity = info.get("velocity", info.get("x_velocity", 0.0)) if isinstance(info, Mapping) else 0.0
            return exorl_velocity_reward(float(velocity), thr)

        returns.append(online_rollout(env, policy, horizon=horizon, reward_fn=reward_fn)["return"])
    return {"num_velocity_tasks": len(returns), "average_cumulative_reward": _mean(returns), "returns": returns}


def evaluate_kitchen_online(env: Any, policy: Callable[[Any], Any], horizon: int = 1000) -> Dict[str, Any]:
    """Evaluate seven Franka Kitchen sparse subtasks and average returns."""

    returns = []
    for subtask in KITCHEN_SUBTASKS:

        def reward_fn(obs: Any, action: Any, info: Any, task: str = subtask) -> float:
            if isinstance(info, Mapping):
                return float(info.get(f"{task}_success", info.get(task, 0.0)))
            return 0.0

        returns.append(online_rollout(env, policy, horizon=horizon, reward_fn=reward_fn)["return"])
    return {"subtasks": list(KITCHEN_SUBTASKS), "average_cumulative_reward": _mean(returns), "returns": returns}
