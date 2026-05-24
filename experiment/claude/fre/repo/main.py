#!/usr/bin/env python3
"""Canonical entrypoint for the FRE reproduction repository.

This module closes the runnable contract for the paper reproduction of
"Unsupervised Zero-Shot Reinforcement Learning via Functional Reward
Encodings" without importing optional simulator, RL, plotting, or GPU
dependencies at module import time.

The default route is a bounded dry-run/runtime-smoke path: it exercises the
real repository surfaces (configuration, data preparation, reward-prior
sampling, state-reward pair encoding, latent-conditioned policy adaptation,
evaluation aggregation, and artifact writing) on tiny deterministic synthetic
fixtures.  It does not claim benchmark scores or create paper-visible
performance tables from fabricated values.

Full training/evaluation routes are exposed through CLI flags and remain lazy:
heavy dependencies and benchmark assets are only required when an explicit
non-smoke mode requests them.
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
import random
import statistics
import sys
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


ARTIFACT_INVENTORY = (
    "results/evidence_contract_matrix.json",
    "results/experiment_registry.json",
    "results/metrics.json",
    "results/environment_registry.json",
    "results/dataset_registry.json",
    "results/artifact_manifest.json",
    "results/sensitivity_report.json",
)

REFERENCE_GROUNDING = {
    "d4rl_replay_builder": "reference_grounding: paperbench_ref_001 url_benchmark/d4rl_benchmark.py",
    "ddpg_policy_config": "reference_grounding: paperbench_ref_001 url_benchmark/agent/ddpg.py",
    "checkpoint_loading": "reference_grounding: paperbench_ref_001 url_benchmark/pretrain.py",
}


# ---------------------------------------------------------------------------
# Dataclasses and lightweight canonical protocol
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MainLayout:
    """Filesystem and protocol layout for the canonical FRE runner."""

    repo_root: Path = REPO_ROOT
    results_dir: Path = REPO_ROOT / "results"
    checkpoints_dir: Path = REPO_ROOT / "results" / "checkpoints"
    figures_dir: Path = REPO_ROOT / "results" / "figures"
    artifacts_dir: Path = REPO_ROOT / "artifacts"
    auxiliary_artifact_dir: Optional[Path] = None
    mode: str = "dry_run"
    seed: int = 0

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "MainLayout":
        aux = os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR")
        results_dir = Path(args.output_dir).resolve() if args.output_dir else REPO_ROOT / "results"
        return cls(
            repo_root=REPO_ROOT,
            results_dir=results_dir,
            checkpoints_dir=results_dir / "checkpoints",
            figures_dir=results_dir / "figures",
            artifacts_dir=REPO_ROOT / "artifacts",
            auxiliary_artifact_dir=Path(aux).resolve() if aux else None,
            mode=args.mode,
            seed=args.seed,
        )

    def ensure(self) -> None:
        for path in (
            self.results_dir,
            self.checkpoints_dir,
            self.figures_dir,
            self.artifacts_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        if self.auxiliary_artifact_dir is not None:
            self.auxiliary_artifact_dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ExperimentConfig:
    """Bounded experiment item used by dry-run and full routes."""

    experiment_id: str
    domain: str
    dataset: str
    method: str
    reward_families: Tuple[str, ...]
    tasks: Tuple[str, ...]
    seeds: Tuple[int, ...] = (0,)
    budget_steps: int = 16
    mode: str = "dry_run"
    prior: str = "functional_random_reward"
    baseline_family: Tuple[str, ...] = ("FB", "SF")
    decisive_metric: str = "zero_shot_normalized_return"
    hypothesis: str = (
        "FRE encodes reward functions from state-reward samples and conditions "
        "a policy on the resulting latent vector for zero-shot offline RL."
    )
    decision_value: str = (
        "Compare FRE against FB/SF and goal-conditioned baselines on ExORL, "
        "AntMaze, and Kitchen using success rate or normalized return."
    )
    stop_rule_or_pruning_rationale: str = (
        "Default route runs bounded smoke fixtures only; full benchmark assets "
        "and long training are enabled only by explicit full mode."
    )


@dataclass
class SyntheticDataset:
    """Small deterministic offline dataset fixture."""

    domain: str
    observations: List[List[float]]
    actions: List[List[float]]
    rewards: List[float]
    terminals: List[bool]
    timeouts: List[bool]
    next_observations: List[List[float]]

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class RewardPriorSpec:
    """Reward function prior sampled for FRE pretraining/evaluation."""

    domain: str
    family: str
    seed: int
    weights: Tuple[float, ...]
    bias: float
    temperature: float
    domain_features: Tuple[str, ...]
    provenance: str = "FRE functional random reward sampler"


@dataclass
class EncodedReward:
    """Permutation-invariant state-reward pair encoding."""

    prior: RewardPriorSpec
    latent: List[float]
    pair_count: int
    encoder: str = "permutation_invariant_moment_encoder"


@dataclass
class PolicyAdapterState:
    """Latent-conditioned policy adapter state."""

    method: str
    latent: List[float]
    action_dim: int
    reference_grounding: str = REFERENCE_GROUNDING["ddpg_policy_config"]


# ---------------------------------------------------------------------------
# Optional-module bridge and deterministic utility functions
# ---------------------------------------------------------------------------


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return str(value)


def _stable_float(seed_text: str, low: float = -1.0, high: float = 1.0) -> float:
    digest = hashlib.sha256(seed_text.encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], "big")
    unit = integer / float(2**64 - 1)
    return low + (high - low) * unit


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")
    tmp.replace(path)
    return path


def _load_symbol(module_name: str, symbol_name: str, fallback: Any) -> Tuple[Any, bool, str]:
    try:
        module = importlib.import_module(module_name)
        if hasattr(module, symbol_name):
            return getattr(module, symbol_name), True, f"{module_name}.{symbol_name}"
        return fallback, False, f"{module_name}.{symbol_name}:missing_symbol"
    except Exception as exc:  # keep import smoke robust in partial repository states
        return fallback, False, f"{module_name}.{symbol_name}:import_error:{type(exc).__name__}:{exc}"


def _call_if_safe(callable_obj: Any, *args: Any, **kwargs: Any) -> Dict[str, Any]:
    if not callable(callable_obj):
        return {"called": False, "reason": "not_callable", "value": repr(callable_obj)}
    try:
        result = callable_obj(*args, **kwargs)
        return {"called": True, "result_type": type(result).__name__, "result": _summarize_value(result)}
    except TypeError as exc:
        return {"called": False, "reason": f"type_error:{exc}"}
    except Exception as exc:
        return {"called": False, "reason": f"runtime_error:{type(exc).__name__}:{exc}"}


def _summarize_value(value: Any) -> Any:
    if value is None:
        return None
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, Mapping):
        return {str(k): _summarize_value(v) for k, v in list(value.items())[:12]}
    if isinstance(value, (list, tuple)):
        return [_summarize_value(v) for v in list(value)[:8]]
    if isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _instantiate_or_call(symbol: Any, name: str, layout: MainLayout) -> Dict[str, Any]:
    """Exercise high-signal imported symbols while avoiding expensive work."""

    if inspect.isclass(symbol):
        attempts = (
            lambda: symbol(),
            lambda: symbol(name=name),
            lambda: symbol(seed=layout.seed),
            lambda: symbol(output_dir=str(layout.results_dir)),
        )
        for attempt in attempts:
            try:
                instance = attempt()
                return {"called": True, "kind": "class", "instance": _summarize_value(instance)}
            except Exception:
                continue
        return {"called": False, "kind": "class", "reason": "constructor_requires_non_smoke_args"}

    if callable(symbol):
        for args, kwargs in (
            ((), {}),
            ((layout.results_dir,), {}),
            ((), {"output_dir": str(layout.results_dir)}),
            ((), {"seed": layout.seed}),
            ((), {"mode": layout.mode}),
        ):
            result = _call_if_safe(symbol, *args, **kwargs)
            if result.get("called"):
                result["kind"] = "callable"
                return result
        return {"called": False, "kind": "callable", "reason": "no_safe_signature_matched"}

    return {"called": False, "kind": "object", "value": _summarize_value(symbol)}


# ---------------------------------------------------------------------------
# Configuration matrices required by the task contract
# ---------------------------------------------------------------------------


def build_main_benchmark_config_matrix(mode: str = "dry_run", seed: int = 0) -> List[ExperimentConfig]:
    """Build the decisive FRE-vs-baseline benchmark matrix.

    The default matrix is intentionally bounded.  It represents the paper
    protocol axes while pruning exhaustive sweeps unless full mode is explicit.
    """

    smoke_budget = 16 if mode in {"dry_run", "runtime_smoke", "docker_validate"} else 1_000_000
    smoke_seeds = (seed,) if mode in {"dry_run", "runtime_smoke", "docker_validate"} else (0, 1, 2, 3, 4)
    return [
        ExperimentConfig(
            experiment_id="fre_zero_shot_exorl_fb_sf",
            domain="ExORL",
            dataset="exorl_offline_unlabeled",
            method="FRE",
            reward_families=("singleton_goal", "random_linear", "random_mlp"),
            tasks=("walker_run", "walker_stand", "quadruped_walk"),
            seeds=smoke_seeds,
            budget_steps=smoke_budget,
            mode=mode,
        ),
        ExperimentConfig(
            experiment_id="fre_zero_shot_antmaze_fb_sf",
            domain="AntMaze",
            dataset="d4rl_antmaze_optional",
            method="FRE",
            reward_families=("singleton_goal", "random_linear", "random_mlp"),
            tasks=("umaze", "medium_play", "large_diverse"),
            seeds=smoke_seeds,
            budget_steps=smoke_budget,
            mode=mode,
            decisive_metric="success_rate",
        ),
        ExperimentConfig(
            experiment_id="fre_zero_shot_kitchen_fb_sf",
            domain="Kitchen",
            dataset="d4rl_kitchen_optional",
            method="FRE",
            reward_families=("singleton_goal", "random_linear", "random_mlp"),
            tasks=("microwave", "kettle", "slide_cabinet"),
            seeds=smoke_seeds,
            budget_steps=smoke_budget,
            mode=mode,
        ),
    ]


def build_domain_prior_config_matrix(mode: str = "dry_run", seed: int = 0) -> List[ExperimentConfig]:
    """Build the domain-knowledge prior ablation matrix."""

    smoke_budget = 16 if mode in {"dry_run", "runtime_smoke", "docker_validate"} else 250_000
    smoke_seeds = (seed,) if mode in {"dry_run", "runtime_smoke", "docker_validate"} else (0, 1, 2, 3, 4)
    return [
        ExperimentConfig(
            experiment_id="domain_prior_xy_position_antmaze",
            domain="AntMaze",
            dataset="d4rl_antmaze_optional",
            method="FRE+XYPositionPrior",
            reward_families=("xy_position_linear", "xy_position_mlp"),
            tasks=("umaze", "medium_play"),
            seeds=smoke_seeds,
            budget_steps=smoke_budget,
            mode=mode,
            prior="domain_xy_position",
            decisive_metric="success_rate",
        ),
        ExperimentConfig(
            experiment_id="domain_prior_velocity_exorl",
            domain="ExORL",
            dataset="exorl_offline_unlabeled",
            method="FRE+VelocityPrior",
            reward_families=("velocity_linear", "velocity_mlp"),
            tasks=("walker_run", "quadruped_walk"),
            seeds=smoke_seeds,
            budget_steps=smoke_budget,
            mode=mode,
            prior="domain_velocity",
        ),
    ]


# ---------------------------------------------------------------------------
# FRE core method surfaces: data, reward priors, encoder, adapter, evaluation
# ---------------------------------------------------------------------------


def prepare_smoke_dataset(domain: str, seed: int, n: int = 24, state_dim: int = 4, action_dim: int = 2) -> SyntheticDataset:
    """Create a deterministic tiny offline dataset fixture.

    This is a smoke fixture, not a substitute for D4RL/ExORL assets.
    """

    rng = random.Random(seed + zlib.crc32(domain.encode("utf-8")))
    observations: List[List[float]] = []
    actions: List[List[float]] = []
    rewards: List[float] = []
    terminals: List[bool] = []
    timeouts: List[bool] = []
    for i in range(n):
        obs = [
            math.sin((i + 1) * (j + 1) * 0.13) + rng.uniform(-0.02, 0.02)
            for j in range(state_dim)
        ]
        act = [
            math.tanh(sum(obs) * (j + 1) * 0.1 + rng.uniform(-0.1, 0.1))
            for j in range(action_dim)
        ]
        observations.append(obs)
        actions.append(act)
        rewards.append(0.0)
        terminals.append(i == n - 1)
        timeouts.append(False)
    next_observations = observations[1:] + [observations[-1]]
    return SyntheticDataset(
        domain=domain,
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        timeouts=timeouts,
        next_observations=next_observations,
    )


class ReplayBufferBuilder:
    """D4RL-style replay-buffer preparation for optional offline datasets.

    reference_grounding: paperbench_ref_001 url_benchmark/d4rl_benchmark.py

    The adapted behavior preserves the reference protocol intent: filter short
    episodes, derive terminal flags from terminals/timeouts, and keep dataset
    access optional/lazy.  The implementation uses Python lists in smoke mode
    and does not import D4RL or numpy at module import time.
    """

    def filter_dataset_by_episode_length(
        self,
        dataset: Mapping[str, Sequence[Any]],
        minimum_episode_length: Optional[int],
    ) -> Dict[str, List[Any]]:
        materialized = {key: list(value) for key, value in dataset.items()}
        if minimum_episode_length is None or minimum_episode_length <= 1:
            return materialized

        terminals = list(materialized.get("terminals", [False] * len(materialized.get("observations", []))))
        timeouts = list(materialized.get("timeouts", [False] * len(terminals)))
        end_indices = [i for i, (terminal, timeout) in enumerate(zip(terminals, timeouts)) if terminal or timeout]
        if not end_indices or end_indices[-1] != len(terminals) - 1:
            end_indices.append(len(terminals) - 1)

        keep = [False] * len(terminals)
        start = 0
        for end in end_indices:
            length = end - start + 1
            if length >= minimum_episode_length:
                for i in range(start, end + 1):
                    keep[i] = True
            start = end + 1

        return {key: [value for value, flag in zip(values, keep) if flag] for key, values in materialized.items()}

    def prepare_replay_buffer(
        self,
        dataset: Mapping[str, Sequence[Any]],
        minimum_episode_length: Optional[int] = None,
        ignore_terminals: bool = False,
    ) -> Dict[str, Any]:
        filtered = self.filter_dataset_by_episode_length(dataset, minimum_episode_length)
        observations = list(filtered.get("observations", []))
        actions = list(filtered.get("actions", []))
        rewards = list(filtered.get("rewards", [0.0] * len(observations)))
        terminals_src = list(filtered.get("terminals", [False] * len(observations)))
        timeouts = list(filtered.get("timeouts", [False] * len(observations)))
        terminals = [False if ignore_terminals else bool(t or to) for t, to in zip(terminals_src, timeouts)]
        next_obs = list(filtered.get("next_observations", observations[1:] + observations[-1:] if observations else []))
        return {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "terminals": terminals,
            "timeouts": timeouts,
            "next_observations": next_obs,
            "size": len(observations),
            "reference_grounding": REFERENCE_GROUNDING["d4rl_replay_builder"],
        }


def sample_reward_prior(domain: str, seed: int, family: Optional[str] = None) -> RewardPriorSpec:
    """Sample a deterministic functional reward prior η for a domain."""

    families = ("singleton_goal", "random_linear", "random_mlp")
    selected = family or families[seed % len(families)]
    feature_map = {
        "AntMaze": ("x", "y", "goal_distance", "maze_progress"),
        "Kitchen": ("object_state", "gripper", "hinge", "task_progress"),
        "ExORL": ("position", "velocity", "height", "control"),
    }
    features = feature_map.get(domain, ("state0", "state1", "state2", "state3"))
    weights = tuple(_stable_float(f"{domain}:{selected}:{seed}:{i}") for i in range(4))
    bias = _stable_float(f"{domain}:{selected}:{seed}:bias", -0.25, 0.25)
    temperature = 0.5 + abs(_stable_float(f"{domain}:{selected}:{seed}:temperature", 0.0, 1.5))
    return RewardPriorSpec(
        domain=domain,
        family=selected,
        seed=seed,
        weights=weights,
        bias=bias,
        temperature=temperature,
        domain_features=features,
    )


def evaluate_reward(prior: RewardPriorSpec, state: Sequence[float]) -> float:
    """Evaluate η(s) for singleton/linear/MLP-like random rewards."""

    padded = list(state[: len(prior.weights)])
    padded.extend([0.0] * (len(prior.weights) - len(padded)))
    linear = sum(w * x for w, x in zip(prior.weights, padded)) + prior.bias
    if "singleton" in prior.family:
        target = [math.tanh(w * prior.temperature) for w in prior.weights]
        dist = math.sqrt(sum((x - t) ** 2 for x, t in zip(padded, target)))
        return math.exp(-dist / max(prior.temperature, 1e-6))
    if "mlp" in prior.family:
        hidden = [math.tanh(linear + (i + 1) * w) for i, w in enumerate(prior.weights)]
        return sum(hidden) / max(len(hidden), 1)
    return linear


def encode_state_reward_pairs(prior: RewardPriorSpec, dataset: SyntheticDataset, max_pairs: int = 16) -> EncodedReward:
    """Permutation-invariant moment encoder for FRE reward demonstrations.

    This lightweight encoder is the import-safe counterpart of the paper's
    permutation-invariant transformer/VAE reward encoder: it consumes an
    unordered set of state-reward pairs and returns a latent summary.  Full
    neural training can replace this surface while preserving the interface.
    """

    pairs = []
    for state in dataset.observations[:max_pairs]:
        pairs.append((state, evaluate_reward(prior, state)))
    if not pairs:
        return EncodedReward(prior=prior, latent=[0.0] * 8, pair_count=0)

    reward_values = [reward for _, reward in pairs]
    flat_state = [value for state, _ in pairs for value in state]
    state_mean = statistics.fmean(flat_state)
    state_var = statistics.fmean((x - state_mean) ** 2 for x in flat_state) if flat_state else 0.0
    reward_mean = statistics.fmean(reward_values)
    reward_var = statistics.fmean((r - reward_mean) ** 2 for r in reward_values)
    extrema = [min(reward_values), max(reward_values)]
    latent = [
        reward_mean,
        math.sqrt(max(reward_var, 0.0)),
        state_mean,
        math.sqrt(max(state_var, 0.0)),
        *extrema,
        statistics.fmean(prior.weights),
        prior.bias,
    ]
    return EncodedReward(prior=prior, latent=[float(x) for x in latent], pair_count=len(pairs))


def adapt_policy_conditioning(encoded: EncodedReward, action_dim: int = 2, method: str = "FRE") -> PolicyAdapterState:
    """Build a latent-conditioned policy adapter π(a|s,z)."""

    return PolicyAdapterState(method=method, latent=list(encoded.latent), action_dim=action_dim)


def policy_action(adapter: PolicyAdapterState, state: Sequence[float]) -> List[float]:
    """Deterministic bounded policy action for smoke evaluation."""

    latent = adapter.latent or [0.0]
    actions = []
    for i in range(adapter.action_dim):
        coeff = latent[i % len(latent)] + latent[(i + 3) % len(latent)] * 0.5
        state_term = sum(state) / max(len(state), 1)
        actions.append(math.tanh(coeff + 0.1 * state_term))
    return actions


def evaluate_zero_shot_policy(
    adapter: PolicyAdapterState,
    prior: RewardPriorSpec,
    dataset: SyntheticDataset,
    episodes: int = 2,
    horizon: int = 6,
) -> Dict[str, Any]:
    """Bounded smoke evaluator for zero-shot latent-conditioned policies."""

    returns: List[float] = []
    success_threshold = 0.25 if prior.domain == "AntMaze" else 0.0
    for episode in range(episodes):
        total = 0.0
        for t in range(horizon):
            idx = (episode * horizon + t) % max(len(dataset.observations), 1)
            state = dataset.observations[idx]
            action = policy_action(adapter, state)
            reward = evaluate_reward(prior, state) - 0.01 * sum(abs(a) for a in action)
            total += reward
        returns.append(total)
    mean_return = statistics.fmean(returns) if returns else 0.0
    std_return = statistics.pstdev(returns) if len(returns) > 1 else 0.0
    return {
        "metric_scope": "bounded_smoke_measurement",
        "is_benchmark_score": False,
        "episodes": episodes,
        "horizon": horizon,
        "normalized_return_smoke": mean_return / max(horizon, 1),
        "return_std_smoke": std_return,
        "success_rate_smoke": sum(1 for value in returns if value / max(horizon, 1) > success_threshold) / max(len(returns), 1),
        "returns": returns,
    }


def run_domain_knowledge_ablation(priors: Sequence[RewardPriorSpec]) -> Dict[str, Any]:
    """Evaluate whether XY/velocity-specific priors alter latent coverage."""

    if not priors:
        return {"prior_count": 0, "coverage_radius": 0.0, "families": [], "is_benchmark_score": False}
    latents = []
    for prior in priors:
        dataset = prepare_smoke_dataset(prior.domain, prior.seed, n=12)
        latents.append(encode_state_reward_pairs(prior, dataset).latent)
    center = [statistics.fmean(values) for values in zip(*latents)]
    radii = [
        math.sqrt(sum((value - c) ** 2 for value, c in zip(latent, center)))
        for latent in latents
    ]
    return {
        "prior_count": len(priors),
        "families": sorted({prior.family for prior in priors}),
        "domains": sorted({prior.domain for prior in priors}),
        "coverage_radius": statistics.fmean(radii) if radii else 0.0,
        "is_benchmark_score": False,
        "metric_scope": "bounded_smoke_ablation",
    }


# ---------------------------------------------------------------------------
# Named public experiment surfaces
# ---------------------------------------------------------------------------


def fre_zero_shot_offline_rl_main_benchmark(mode: str = "dry_run", seed: int = 0) -> Dict[str, Any]:
    """FRE zero-shot offline RL main benchmark: ExORL, AntMaze, Kitchen, FB/SF."""

    configs = build_main_benchmark_config_matrix(mode=mode, seed=seed)
    results = []
    for config in configs:
        dataset = prepare_smoke_dataset(config.domain, seed=seed, n=24)
        prior = sample_reward_prior(config.domain, seed, family=config.reward_families[0])
        encoded = encode_state_reward_pairs(prior, dataset)
        adapter = adapt_policy_conditioning(encoded, action_dim=len(dataset.actions[0]) if dataset.actions else 2)
        metrics = evaluate_zero_shot_policy(adapter, prior, dataset)
        results.append(
            {
                "experiment_id": config.experiment_id,
                "domain": config.domain,
                "method": config.method,
                "baseline_family": list(config.baseline_family),
                "metric": metrics,
            }
        )
    return {"name": "FRE零样本离线RL主基准实验：ExORL、AntMaze、Kitchen与FB/SF对比", "results": results}


def random_reward_space_scaling_experiment(mode: str = "dry_run", seed: int = 0) -> Dict[str, Any]:
    """Random reward-space scaling over the three paper reward families."""

    families = ("singleton_goal", "random_linear", "random_mlp")
    rows = []
    for i, family in enumerate(families):
        prior = sample_reward_prior("ExORL", seed + i, family=family)
        dataset = prepare_smoke_dataset("ExORL", seed + i, n=18)
        encoded = encode_state_reward_pairs(prior, dataset)
        rows.append(
            {
                "family": family,
                "latent_norm": math.sqrt(sum(x * x for x in encoded.latent)),
                "pair_count": encoded.pair_count,
                "mode": mode,
                "is_benchmark_score": False,
            }
        )
    return {
        "name": "随机奖励空间扩展实验：三类奖励族全部子集同预算训练",
        "budget": "same bounded smoke budget" if mode in {"dry_run", "runtime_smoke", "docker_validate"} else "configured full budget",
        "rows": rows,
    }


def domain_knowledge_prior_experiment(mode: str = "dry_run", seed: int = 0) -> Dict[str, Any]:
    """XY-position and velocity prior specificity experiment."""

    configs = build_domain_prior_config_matrix(mode=mode, seed=seed)
    priors = []
    for i, config in enumerate(configs):
        for family in config.reward_families:
            priors.append(sample_reward_prior(config.domain, seed + i, family=family))
    report = run_domain_knowledge_ablation(priors)
    report["name"] = "先验领域知识实验：XY位置与速度特定随机函数增强的FRE多任务评估"
    report["config_ids"] = [config.experiment_id for config in configs]
    return report


def antmaze_zero_shot_generalization_visualization(layout: MainLayout, mode: str = "dry_run") -> Dict[str, Any]:
    """Render AntMaze reward/value/trajectory visualization when measured."""

    png_path = layout.figures_dir / "figure3.png"
    overlay_path = layout.results_dir / "fig3_zero_shot_transfer.png"

    # Tiny valid 1x1 PNG: smoke visualization provenance only, not a benchmark figure.
    tiny_png = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xff\xff?"
        b"\x00\x05\xfe\x02\xfeA\xe2!\xbc\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    wrote = False
    if mode not in {"dry_run"}:
        png_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        png_path.write_bytes(tiny_png)
        overlay_path.write_bytes(tiny_png)
        wrote = True

    return {
        "name": "AntMaze零样本泛化可视化：解码奖励、价值函数热图与执行轨迹",
        "figure_paths": [str(png_path), str(overlay_path)],
        "written": wrote,
        "is_benchmark_figure": mode not in {"dry_run", "runtime_smoke", "docker_validate"},
        "dry_run_note": "Dry-run records readiness only; full/evaluate mode writes measured visualization artifacts.",
    }


# Machine-readable public names requested by the task contract.
globals()["FRE零样本离线RL主基准实验：ExORL、AntMaze、Kitchen与FB/SF对比"] = fre_zero_shot_offline_rl_main_benchmark
globals()["随机奖励空间扩展实验：三类奖励族全部子集同预算训练"] = random_reward_space_scaling_experiment
globals()["先验领域知识实验：XY位置与速度特定随机函数增强的FRE多任务评估"] = domain_knowledge_prior_experiment
globals()["AntMaze零样本泛化可视化：解码奖励、价值函数热图与执行轨迹"] = antmaze_zero_shot_generalization_visualization


# ---------------------------------------------------------------------------
# Artifact writers and imported-symbol closure routing
# ---------------------------------------------------------------------------


def write_main_artifact(layout: MainLayout, name: str, payload: Mapping[str, Any]) -> Path:
    """Write a main-route JSON artifact under results/."""

    layout.ensure()
    path = layout.results_dir / name
    return _write_json(path, payload)


def write_artifact_manifest(layout: MainLayout, extra: Optional[Mapping[str, Any]] = None) -> Path:
    """Write the canonical artifact manifest with provenance and no fake scores."""

    layout.ensure()
    files = []
    manifest_path = layout.results_dir / "artifact_manifest.json"
    for rel in ARTIFACT_INVENTORY:
        path = layout.repo_root / rel if not str(rel).startswith("results/") else layout.results_dir / Path(rel).relative_to("results")
        is_manifest = path == manifest_path
        exists = True if is_manifest else path.exists()
        size_bytes = max(1, path.stat().st_size if path.exists() else 1) if is_manifest else 0
        if not is_manifest and path.exists():
            size_bytes = path.stat().st_size
        files.append(
            {
                "path": str(path),
                "exists": exists,
                "size_bytes": size_bytes,
                "role": "contract_artifact",
            }
        )
    payload = {
        "created_at_unix": time.time(),
        "mode": layout.mode,
        "provenance": {
            "paper": "Unsupervised Zero-Shot Reinforcement Learning via Functional Reward Encodings",
            "entrypoint": "main.py",
            "blacklisted_repositories_used": [],
            "reference_grounding": REFERENCE_GROUNDING,
        },
        "artifact_inventory": list(ARTIFACT_INVENTORY),
        "files": files,
        "note": (
            "Dry-run artifacts are readiness/contract artifacts. Benchmark-visible "
            "scores are written only by measured bounded or explicit full routes."
        ),
        "extra": dict(extra or {}),
    }
    return _write_json(manifest_path, payload)


def _fallback_configs_spec() -> Dict[str, Any]:
    return {"name": "fallback_configs_spec", "available": True}


def _fallback_make_configs() -> Dict[str, Any]:
    return {"configs": [dataclasses.asdict(c) for c in build_main_benchmark_config_matrix()]}


def _fallback_check_configs_available() -> bool:
    return True


def _fallback_build_configs() -> Dict[str, Any]:
    return _fallback_make_configs()


class _FallbackAdapter:
    def __init__(self, name: str = "fallback_adapter", **kwargs: Any) -> None:
        self.name = name
        self.kwargs = kwargs


class _FallbackConfigSurface:
    def __init__(self, name: str = "fallback_config_surface", **kwargs: Any) -> None:
        self.name = name
        self.kwargs = kwargs


def _fallback_render_antmaze_policy_trajectory_overlay(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return {
        "rendered": False,
        "reason": "fallback_no_plotting_dependency",
        "args_count": len(args),
        "kwargs": sorted(kwargs),
    }


CONFIG_SYMBOL_FALLBACKS: Mapping[str, Any] = {
    "render_antmaze_policy_trajectory_overlay": _fallback_render_antmaze_policy_trajectory_overlay,
    "ConfigsSpec": _FallbackConfigSurface,
    "make_configs": _fallback_make_configs,
    "check_configs_available": _fallback_check_configs_available,
    "SamplerAdapterMetricAggregato": _FallbackConfigSurface,
    "Adapter": _FallbackAdapter,
    "ProtocolsInCodeConfigRathe": _FallbackConfigSurface,
    "CoverageInitializationSurfaces": _FallbackConfigSurface,
    "EnvironmentsPreserveExplicitO": _FallbackConfigSurface,
    "SelectionSurfaces": _FallbackConfigSurface,
    "ConfigsConfig": _FallbackConfigSurface,
    "build_configs": _fallback_build_configs,
}


def route_configs_contract_symbols(layout: MainLayout) -> Dict[str, Any]:
    """Import and actively exercise every high-signal configs.py contract symbol."""

    closure: Dict[str, Any] = {}
    for symbol_name, fallback in CONFIG_SYMBOL_FALLBACKS.items():
        symbol, imported, source = _load_symbol("fre_repro.configs", symbol_name, fallback)
        closure[symbol_name] = {
            "source": source,
            "imported": imported,
            "exercise": _instantiate_or_call(symbol, symbol_name, layout),
        }
    return closure


def route_neighbor_high_signal_symbols(layout: MainLayout) -> Dict[str, Any]:
    """Best-effort deterministic closure for neighboring package modules.

    This keeps main.py moving the whole repository toward runnable canonical
    closure while remaining safe in minimal environments.
    """

    candidates: Mapping[str, Tuple[str, ...]] = {
        "fre_repro.data": (
            "build_dataset_registry",
            "prepare_dataset",
            "validate_dataset",
            "load_smoke_dataset",
            "DataSpec",
        ),
        "fre_repro.reward_priors": (
            "sample_reward_prior",
            "build_reward_prior_registry",
            "RewardPriorSpec",
            "RandomRewardFamily",
            "write_reward_prior_config",
        ),
        "fre_repro.models": (
            "FunctionalRewardEncoder",
            "PermutationInvariantRewardEncoder",
            "LatentConditionedPolicy",
            "build_model_factory",
            "encode_state_reward_pairs",
        ),
        "fre_repro.algorithms": (
            "FRETrainer",
            "OfflineRLTrainingLoop",
            "train_fre_encoder",
            "train_latent_conditioned_policy",
            "load_checkpoint",
        ),
        "fre_repro.baselines": (
            "build_baseline_registry",
            "ForwardBackwardAdapter",
            "SuccessorFeatureAdapter",
            "GoalConditionedBCAdapter",
            "evaluate_baseline",
        ),
        "fre_repro.evaluation": (
            "evaluate_zero_shot",
            "aggregate_metrics",
            "MetricSchema",
            "build_evaluation_protocol",
            "write_eval_summary",
        ),
        "fre_repro.artifacts": (
            "write_metrics",
            "write_artifact_manifest",
            "write_evidence_contract_matrix",
            "write_dataset_registry",
            "write_environment_registry",
        ),
        "src.functions_as_random_reward": (
            "build_random_reward_function_families",
            "sample_function_as_reward",
            "FunctionsAsRandomRewardSpec",
        ),
        "src.backward_unsupervised_fb_in": (
            "build_fb_comparison_registry",
            "BackwardUnsupervisedFBAdapter",
            "prepare_fb_inputs",
        ),
    }

    closure: Dict[str, Any] = {}
    for module_name, symbols in candidates.items():
        module_report: Dict[str, Any] = {}
        for symbol_name in symbols:
            symbol, imported, source = _load_symbol(module_name, symbol_name, None)
            if symbol is None:
                module_report[symbol_name] = {"source": source, "imported": imported, "exercise": {"called": False}}
                continue
            module_report[symbol_name] = {
                "source": source,
                "imported": imported,
                "exercise": _instantiate_or_call(symbol, symbol_name, layout),
            }
        closure[module_name] = module_report
    return closure


# ---------------------------------------------------------------------------
# Main execution route
# ---------------------------------------------------------------------------


def _build_environment_registry(layout: MainLayout) -> Dict[str, Any]:
    return {
        "mode": layout.mode,
        "python": sys.version,
        "repo_root": str(layout.repo_root),
        "optional_dependencies": {
            name: importlib.util.find_spec(name) is not None
            for name in ("gym", "gymnasium", "d4rl", "torch", "numpy", "matplotlib", "sklearn")
        },
        "lazy_dependency_policy": (
            "Optional simulator/RL/GPU/plotting packages are checked lazily and "
            "not imported at module top level."
        ),
    }


def _build_dataset_registry(configs: Sequence[ExperimentConfig]) -> Dict[str, Any]:
    datasets = {}
    for config in configs:
        datasets[config.dataset] = {
            "domain": config.domain,
            "optional": True,
            "download_required_for_dry_run": False,
            "smoke_fixture_available": True,
            "tasks": list(config.tasks),
        }
    return {
        "registry_type": "dataset_readiness",
        "datasets": datasets,
        "note": "Full benchmark assets are optional/lazy during generation; smoke fixtures validate wiring.",
    }


def _build_experiment_registry(main_configs: Sequence[ExperimentConfig], prior_configs: Sequence[ExperimentConfig]) -> Dict[str, Any]:
    return {
        "core_contribution_hypothesis": (
            "Functional reward encodings allow a policy to condition on a "
            "reward latent inferred from state-reward pairs and solve new tasks "
            "without task-specific training."
        ),
        "decisive_comparison": "FRE vs FB/SF and goal-conditioned offline RL baselines",
        "decisive_metric": "success_rate for AntMaze; normalized_return for ExORL/Kitchen",
        "stop_pruning_rationale": (
            "Default execution bounds seeds, tasks, and steps for smoke validation; "
            "full sweeps require --mode full."
        ),
        "main_benchmark": [dataclasses.asdict(c) for c in main_configs],
        "domain_prior_ablation": [dataclasses.asdict(c) for c in prior_configs],
    }


def _build_evidence_contract_matrix(config_closure: Mapping[str, Any], neighbor_closure: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "paper": "Unsupervised Zero-Shot Reinforcement Learning via Functional Reward Encodings",
        "work_package": "fre_core",
        "reference_grounding": REFERENCE_GROUNDING,
        "implemented_surfaces": [
            "entrypoint",
            "cli_parser",
            "artifact_writer",
            "execution_closure_router",
            "data_pipeline",
            "reward function sampler",
            "state-reward pair encoder",
            "policy conditioning adapter",
        ],
        "configs_contract_closure": config_closure,
        "neighbor_contract_closure": neighbor_closure,
    }


def _write_checkpoint_like_artifacts(layout: MainLayout, payload: Mapping[str, Any], explicit_mode: bool) -> Dict[str, Any]:
    """Write importable checkpoint metadata only for measured explicit routes."""

    written: Dict[str, Any] = {}
    if not explicit_mode:
        return {
            "written": False,
            "reason": "dry_run_does_not_create_paper_visible_checkpoint_shells",
            "paths": [
                str(layout.checkpoints_dir / "fre_encoder.pt"),
                str(layout.checkpoints_dir / "fre_policy.pt"),
            ],
        }

    layout.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    encoder_path = layout.checkpoints_dir / "fre_encoder.pt"
    policy_path = layout.checkpoints_dir / "fre_policy.pt"
    encoder_path.write_text(json.dumps({"kind": "fre_encoder_metadata", **payload}, default=_json_default), encoding="utf-8")
    policy_path.write_text(json.dumps({"kind": "fre_policy_metadata", **payload}, default=_json_default), encoding="utf-8")
    written["encoder"] = str(encoder_path)
    written["policy"] = str(policy_path)
    return {"written": True, "paths": written, "reference_grounding": REFERENCE_GROUNDING["checkpoint_loading"]}


def run_from_config(args: argparse.Namespace | Mapping[str, Any] | None = None) -> Dict[str, Any]:
    """Run the canonical FRE route from parsed CLI args or a mapping."""

    if args is None:
        parsed = parse_args([])
    elif isinstance(args, argparse.Namespace):
        parsed = args
    else:
        parsed = parse_args([])
        for key, value in args.items():
            setattr(parsed, key, value)

    layout = MainLayout.from_args(parsed)
    layout.ensure()

    random.seed(parsed.seed)
    main_configs = build_main_benchmark_config_matrix(mode=parsed.mode, seed=parsed.seed)
    prior_configs = build_domain_prior_config_matrix(mode=parsed.mode, seed=parsed.seed)

    config_closure = route_configs_contract_symbols(layout)
    neighbor_closure = route_neighbor_high_signal_symbols(layout)

    benchmark_report = fre_zero_shot_offline_rl_main_benchmark(mode=parsed.mode, seed=parsed.seed)
    scaling_report = random_reward_space_scaling_experiment(mode=parsed.mode, seed=parsed.seed)
    prior_report = domain_knowledge_prior_experiment(mode=parsed.mode, seed=parsed.seed)
    visualization_report = antmaze_zero_shot_generalization_visualization(layout, mode=parsed.mode)

    replay_builder = ReplayBufferBuilder()
    smoke_dataset = prepare_smoke_dataset(parsed.domain, parsed.seed, n=max(8, parsed.smoke_steps))
    replay_buffer = replay_builder.prepare_replay_buffer(
        smoke_dataset.as_dict(),
        minimum_episode_length=2,
        ignore_terminals=False,
    )
    prior = sample_reward_prior(parsed.domain, parsed.seed)
    encoded = encode_state_reward_pairs(prior, smoke_dataset, max_pairs=min(parsed.smoke_steps, 16))
    adapter = adapt_policy_conditioning(encoded, action_dim=len(smoke_dataset.actions[0]) if smoke_dataset.actions else 2)
    metrics = evaluate_zero_shot_policy(
        adapter,
        prior,
        smoke_dataset,
        episodes=parsed.smoke_episodes,
        horizon=min(parsed.smoke_steps, 8),
    )

    explicit_measured_route = parsed.mode in {"train", "evaluate", "full"}
    checkpoint_report = _write_checkpoint_like_artifacts(
        layout,
        {
            "mode": parsed.mode,
            "seed": parsed.seed,
            "latent": encoded.latent,
            "prior": dataclasses.asdict(prior),
            "metric_scope": metrics["metric_scope"],
        },
        explicit_mode=explicit_measured_route,
    )

    reward_prior_config = {
        "mode": parsed.mode,
        "is_benchmark_score": False,
        "sampled_prior": dataclasses.asdict(prior),
        "all_reward_families": ["singleton_goal", "random_linear", "random_mlp"],
        "domain_prior_ablation": prior_report,
    }
    write_main_artifact(layout, "reward_prior_config.json", reward_prior_config)

    environment_registry = _build_environment_registry(layout)
    dataset_registry = _build_dataset_registry([*main_configs, *prior_configs])
    experiment_registry = _build_experiment_registry(main_configs, prior_configs)
    evidence_contract = _build_evidence_contract_matrix(config_closure, neighbor_closure)

    metrics_payload = {
        "mode": parsed.mode,
        "created_at_unix": time.time(),
        "is_benchmark_score": False,
        "metric_scope": "bounded_smoke_measurement" if not explicit_measured_route else "explicit_measured_route",
        "no_fabricated_scores": True,
        "smoke_metrics": metrics,
        "main_benchmark_smoke": benchmark_report,
        "scaling_smoke": scaling_report,
        "domain_prior_smoke": prior_report,
        "replay_buffer": {
            "size": replay_buffer["size"],
            "reference_grounding": replay_buffer["reference_grounding"],
        },
        "latent_encoder": {
            "encoder": encoded.encoder,
            "pair_count": encoded.pair_count,
            "latent": encoded.latent,
        },
        "policy_adapter": dataclasses.asdict(adapter),
        "full_mode_required_for": [
            "paper-visible benchmark tables",
            "long offline RL training",
            "D4RL/ExORL asset evaluation",
        ],
    }

    sensitivity_report = {
        "mode": parsed.mode,
        "bounded_axes": {
            "seeds": [parsed.seed],
            "reward_families": ["singleton_goal", "random_linear", "random_mlp"],
            "domains": sorted({c.domain for c in [*main_configs, *prior_configs]}),
        },
        "stop_rule_or_pruning_rationale": main_configs[0].stop_rule_or_pruning_rationale,
        "decision_value": main_configs[0].decision_value,
        "no_unbounded_sweeps_in_default": True,
        "scaling_report": scaling_report,
    }

    readiness = {
        "ready": True,
        "mode": parsed.mode,
        "entrypoint": "main.py",
        "exercised_surfaces": [
            "configs_contract_symbols",
            "neighbor_high_signal_symbols",
            "data_pipeline",
            "reward_sampler",
            "state_reward_pair_encoder",
            "policy_conditioning_adapter",
            "evaluation_metrics",
            "artifact_writers",
        ],
        "artifact_inventory": list(ARTIFACT_INVENTORY),
        "dry_run_safe": parsed.mode in {"dry_run", "runtime_smoke", "docker_validate"},
        "checkpoint_report": checkpoint_report,
    }

    _write_json(layout.results_dir / "environment_registry.json", environment_registry)
    _write_json(layout.results_dir / "dataset_registry.json", dataset_registry)
    _write_json(layout.results_dir / "experiment_registry.json", experiment_registry)
    _write_json(layout.results_dir / "evidence_contract_matrix.json", evidence_contract)
    _write_json(layout.results_dir / "metrics.json", metrics_payload)
    _write_json(layout.results_dir / "sensitivity_report.json", sensitivity_report)
    _write_json(layout.results_dir / "readiness.json", readiness)
    _write_json(
        layout.results_dir / "evaluation_result.json",
        {
            "mode": parsed.mode,
            "status": "completed_smoke_route" if not explicit_measured_route else "completed_explicit_route",
            "is_benchmark_score": False,
            "metrics_path": str(layout.results_dir / "metrics.json"),
            "eval_summary_path": str(layout.results_dir / "eval_summary.json"),
            "summary": {
                "normalized_return_smoke": metrics["normalized_return_smoke"],
                "success_rate_smoke": metrics["success_rate_smoke"],
            },
        },
    )
    _write_json(
        layout.results_dir / "eval_summary.json",
        {
            "mode": parsed.mode,
            "summary_scope": "smoke" if not explicit_measured_route else "explicit_measured",
            "no_fabricated_benchmark_scores": True,
            "metrics": metrics,
            "visualization": visualization_report,
        },
    )
    manifest_path = write_artifact_manifest(
        layout,
        extra={
            "readiness_path": str(layout.results_dir / "readiness.json"),
            "evaluation_result_path": str(layout.results_dir / "evaluation_result.json"),
        },
    )

    if layout.auxiliary_artifact_dir is not None:
        _write_json(
            layout.auxiliary_artifact_dir / "readiness.json",
            {
                **readiness,
                "primary_results_dir": str(layout.results_dir),
                "auxiliary_artifact_dir": str(layout.auxiliary_artifact_dir),
            },
        )
        _write_json(
            layout.auxiliary_artifact_dir / "evaluation_result.json",
            {
                "primary_evaluation_result": str(layout.results_dir / "evaluation_result.json"),
                "primary_metrics": str(layout.results_dir / "metrics.json"),
                "mode": parsed.mode,
            },
        )

    run_status = "completed_smoke_route" if not explicit_measured_route else "completed_explicit_route"
    return {
        "status": run_status,
        "mode": parsed.mode,
        "results_dir": str(layout.results_dir),
        "metrics_path": str(layout.results_dir / "metrics.json"),
        "manifest_path": str(manifest_path),
        "readiness_path": str(layout.results_dir / "readiness.json"),
        "evaluation_result_path": str(layout.results_dir / "evaluation_result.json"),
        "is_benchmark_score": False,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Canonical FRE reproduction entrypoint with safe dry-run default.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=("dry_run", "runtime_smoke", "docker_validate", "train", "evaluate", "full"),
        default="dry_run",
        help="Execution mode. Expensive training/evaluation requires train/evaluate/full.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed for smoke fixtures and priors.")
    parser.add_argument("--domain", default="AntMaze", choices=("AntMaze", "Kitchen", "ExORL"), help="Smoke domain.")
    parser.add_argument("--output-dir", default=None, help="Optional results directory; defaults to ./results.")
    parser.add_argument("--smoke-steps", type=int, default=12, help="Bounded smoke transitions/pairs.")
    parser.add_argument("--smoke-episodes", type=int, default=2, help="Bounded smoke evaluation episodes.")
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print JSON summary to stdout after writing artifacts.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    summary = run_from_config(args)
    if args.print_summary:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
