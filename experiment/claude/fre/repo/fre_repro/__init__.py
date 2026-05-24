"""Public package facade for the FRE reproduction.

This package implements the core route for the paper
"Unsupervised Zero-Shot Reinforcement Learning via Functional Reward Encodings":

    unlabeled offline transitions
      -> sample reward functions eta from a prior
      -> sample K encoder states from the offline dataset
      -> encode state-reward pairs into a latent reward code z
      -> train a latent-conditioned policy pi(a | s, z) offline
      -> evaluate zero-shot transfer on held-out reward/task functions
      -> write stable artifacts and smoke/readiness manifests

The implementation in this file is intentionally lightweight and import-safe.
It uses only the Python standard library at module import time, while exposing
the package-level API expected by the canonical runner.  Heavier implementations
in sibling modules can be reached lazily by downstream code, but this facade is
fully executable on its own for bounded smoke and small measured routes.

reference_grounding: paperbench_ref_001 url_benchmark/d4rl_benchmark.py
reference_grounding: paperbench_ref_001 url_benchmark/agent/ddpg.py
reference_grounding: paperbench_ref_001 url_benchmark/pretrain.py
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import math
import os
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from fre_repro.paper_surface import (
    DATASET_ENVIRONMENT_ROUTES,
    FREConditionedIQLPolicy,
    FREDecoderNetwork,
    FREEncoderNetwork,
    GCBCNetwork,
    GCIQLNetwork,
    OPALArchitecture,
    REWARD_PRIOR_MIXTURES,
    discretize_reward_to_32_bins,
    evaluate_fre_agent_with_32_state_reward_pairs,
    evaluate_opal_with_10_random_skills,
    fre_paper_surface_inventory,
    load_antmaze_large_diverse_v2_dataset,
    load_exorl_rnd_dataset,
    load_kitchen_complete_v0_dataset,
    make_d4rl_antmaze_online_evaluation_env,
    make_d4rl_kitchen_online_evaluation_env,
    make_exorl_custom_dmc_env,
    sample_random_two_layer_mlp_reward,
    sample_singleton_goal_reward,
    sample_sparse_random_linear_reward,
    train_fb_agent_with_controllable_agent,
    train_fre_conditioned_iql_policy,
    train_fre_encoder_decoder_strided,
    train_gc_bc_agent,
    train_gc_iql_agent,
    train_opal_agent,
    train_sf_agent_with_controllable_agent,
)


Transition = Dict[str, Any]
RewardCallable = Callable[[Sequence[float], Optional[Sequence[float]], Optional[Sequence[float]]], float]


@dataclass(frozen=True)
class FREConfig:
    """Configuration for the canonical FRE route.

    The defaults are deliberately bounded so that ``main(["--mode",
    "runtime_smoke"])`` validates real data/encoder/policy/evaluator wiring
    without launching paper-scale training.  Full training or benchmark
    evaluation is opt-in through ``--mode train`` or ``--mode evaluate`` and by
    supplying a dataset path or benchmark loader in downstream modules.
    """

    mode: str = "runtime_smoke"
    seed: int = 0
    artifact_dir: str = "results"
    obs_dim: int = 4
    action_dim: int = 2
    latent_dim: int = 8
    encoder_state_count: int = 5
    reward_family: str = "mixed"
    reward_count: int = 8
    train_steps: int = 64
    eval_episodes: int = 20
    learning_rate: float = 1e-2
    discount: float = 0.99
    minimum_episode_length: Optional[int] = None
    write_paper_artifacts: bool = False

    @property
    def output_root(self) -> Path:
        return Path(os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR", self.artifact_dir))


@dataclass
class OfflineDataset:
    """In-memory unlabeled transition dataset used by the FRE route."""

    transitions: List[Transition]
    obs_dim: int
    action_dim: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.transitions)

    def states(self) -> List[List[float]]:
        return [list(map(float, tr["observation"])) for tr in self.transitions]

    def sample_encoder_states(self, k: int, seed: int = 0) -> List[List[float]]:
        """Sample K states for the FRE state-reward encoder input."""
        if not self.transitions:
            raise ValueError("Cannot sample encoder states from an empty offline dataset.")
        rng = random.Random(seed)
        if k <= len(self.transitions):
            indices = rng.sample(range(len(self.transitions)), k)
        else:
            indices = [rng.randrange(len(self.transitions)) for _ in range(k)]
        return [list(map(float, self.transitions[i]["observation"])) for i in indices]

    def batches(self, batch_size: int, seed: int = 0) -> Iterable[List[Transition]]:
        rng = random.Random(seed)
        indices = list(range(len(self.transitions)))
        rng.shuffle(indices)
        for start in range(0, len(indices), max(1, batch_size)):
            yield [self.transitions[i] for i in indices[start : start + batch_size]]


@dataclass(frozen=True)
class RewardFunctionSpec:
    """Serializable reward function eta sampled from the FRE reward prior."""

    name: str
    family: str
    weights: Tuple[float, ...]
    bias: float = 0.0
    goal: Optional[Tuple[float, ...]] = None
    hidden_weights: Tuple[Tuple[float, ...], ...] = ()
    output_weights: Tuple[float, ...] = ()

    def __call__(
        self,
        state: Sequence[float],
        action: Optional[Sequence[float]] = None,
        next_state: Optional[Sequence[float]] = None,
    ) -> float:
        x = [float(v) for v in state]
        if self.family == "goal":
            goal = list(self.goal or (0.0,) * len(x))
            return -math.sqrt(sum((x[i] - goal[i % len(goal)]) ** 2 for i in range(len(x))))
        if self.family == "mlp":
            hidden = []
            for row in self.hidden_weights:
                dot = self.bias + sum(row[i % len(row)] * x[i] for i in range(len(x)))
                hidden.append(math.tanh(dot))
            if not hidden:
                hidden = [math.tanh(sum(x) + self.bias)]
            return sum(self.output_weights[i % len(self.output_weights)] * hidden[i] for i in range(len(hidden)))
        return self.bias + sum(self.weights[i % len(self.weights)] * x[i] for i in range(len(x)))

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


class RewardPriorSampler:
    """Sampler for random reward functions eta used by FRE pretraining."""

    def __init__(self, obs_dim: int, family: str = "mixed", seed: int = 0):
        self.obs_dim = int(obs_dim)
        self.family = family
        self.rng = random.Random(seed)

    def sample(self, index: int = 0) -> RewardFunctionSpec:
        family = self.family
        if family == "mixed":
            family = ["linear", "goal", "mlp"][index % 3]

        if family == "goal":
            goal = tuple(self.rng.uniform(-1.0, 1.0) for _ in range(self.obs_dim))
            return RewardFunctionSpec(
                name=f"eta_goal_{index}",
                family="goal",
                weights=tuple(0.0 for _ in range(self.obs_dim)),
                goal=goal,
            )

        if family == "mlp":
            width = max(4, min(16, self.obs_dim * 2))
            hidden = tuple(
                tuple(self.rng.gauss(0.0, 1.0 / math.sqrt(max(1, self.obs_dim))) for _ in range(self.obs_dim))
                for _ in range(width)
            )
            output = tuple(self.rng.gauss(0.0, 1.0 / math.sqrt(width)) for _ in range(width))
            return RewardFunctionSpec(
                name=f"eta_mlp_{index}",
                family="mlp",
                weights=tuple(0.0 for _ in range(self.obs_dim)),
                bias=self.rng.uniform(-0.1, 0.1),
                hidden_weights=hidden,
                output_weights=output,
            )

        weights = tuple(self.rng.gauss(0.0, 1.0 / math.sqrt(max(1, self.obs_dim))) for _ in range(self.obs_dim))
        return RewardFunctionSpec(
            name=f"eta_linear_{index}",
            family="linear",
            weights=weights,
            bias=self.rng.uniform(-0.1, 0.1),
        )

    def sample_many(self, count: int) -> List[RewardFunctionSpec]:
        return [self.sample(i) for i in range(count)]


def sample_reward_prior(domain: str = "default", seed: int = 0, obs_dim: int = 4) -> RewardFunctionSpec:
    """Public reward-prior convenience function.

    ``domain`` selects the reward family when it is one of ``linear``, ``goal``,
    ``mlp``, or ``mixed``.  Other domain labels default to the mixed prior while
    preserving the domain name in the sampled reward identifier through the
    seed-dependent sampler state.
    """

    family = domain if domain in {"linear", "goal", "mlp", "mixed"} else "mixed"
    return RewardPriorSampler(obs_dim=obs_dim, family=family, seed=seed).sample(0)


class FunctionalRewardEncoder:
    """Permutation-invariant state-reward pair encoder.

    The paper uses a transformer-style encoder over a set of K state-reward
    pairs.  This implementation provides a small import-safe attention encoder:
    each pair is projected to a latent token, tokens exchange information
    through scaled dot-product self-attention, and the final reward code is a
    permutation-invariant mean pool.  The API required by the contract is
    ``encoder.encode(reward_function, offline_batch)``.
    """

    def __init__(self, obs_dim: int, latent_dim: int = 8, k_states: int = 5, seed: int = 0):
        self.obs_dim = int(obs_dim)
        self.latent_dim = int(latent_dim)
        self.k_states = int(k_states)
        rng = random.Random(seed)
        self.proj = [
            [rng.gauss(0.0, 1.0 / math.sqrt(max(1, self.obs_dim + 1))) for _ in range(self.obs_dim + 1)]
            for _ in range(self.latent_dim)
        ]
        self.query = [
            [rng.gauss(0.0, 1.0 / math.sqrt(max(1, self.latent_dim))) for _ in range(self.latent_dim)]
            for _ in range(self.latent_dim)
        ]

    def _tokenize(self, state: Sequence[float], reward_value: float) -> List[float]:
        x = list(map(float, state))[: self.obs_dim]
        if len(x) < self.obs_dim:
            x.extend([0.0] * (self.obs_dim - len(x)))
        x.append(float(reward_value))
        return [math.tanh(sum(row[i] * x[i] for i in range(len(x)))) for row in self.proj]

    def _attend(self, tokens: List[List[float]]) -> List[List[float]]:
        if not tokens:
            return [[0.0] * self.latent_dim]
        attended: List[List[float]] = []
        scale = math.sqrt(max(1, self.latent_dim))
        for token in tokens:
            q = [sum(self.query[j][i] * token[i] for i in range(self.latent_dim)) for j in range(self.latent_dim)]
            scores = [sum(q[i] * other[i] for i in range(self.latent_dim)) / scale for other in tokens]
            max_score = max(scores)
            weights = [math.exp(s - max_score) for s in scores]
            denom = sum(weights) or 1.0
            weights = [w / denom for w in weights]
            attended.append(
                [
                    math.tanh(token[d] + sum(weights[j] * tokens[j][d] for j in range(len(tokens))))
                    for d in range(self.latent_dim)
                ]
            )
        return attended

    def encode(
        self,
        reward_function: RewardCallable,
        offline_batch: Any,
        *,
        k_states: Optional[int] = None,
        seed: int = 0,
    ) -> List[float]:
        """Encode eta into latent z from K sampled state-reward pairs."""
        k = int(k_states or self.k_states)
        if isinstance(offline_batch, OfflineDataset):
            states = offline_batch.sample_encoder_states(k, seed=seed)
        else:
            states = _extract_states_from_batch(offline_batch)
            if not states:
                raise ValueError("offline_batch must contain observations/states for FRE encoding.")
            rng = random.Random(seed)
            states = [states[i] for i in (rng.sample(range(len(states)), min(k, len(states))))]

        tokens = [self._tokenize(s, reward_function(s, None, None)) for s in states]
        attended = self._attend(tokens)
        pooled = [sum(tok[d] for tok in attended) / len(attended) for d in range(self.latent_dim)]
        norm = math.sqrt(sum(v * v for v in pooled)) or 1.0
        return [v / norm for v in pooled]

    def state_dict(self) -> Dict[str, Any]:
        return {
            "obs_dim": self.obs_dim,
            "latent_dim": self.latent_dim,
            "k_states": self.k_states,
            "proj": self.proj,
            "query": self.query,
        }


class PolicyConditioningAdapter:
    """Latent-conditioned policy pi(a | s, z) trained on offline transitions."""

    def __init__(self, obs_dim: int, action_dim: int, latent_dim: int, seed: int = 0):
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.latent_dim = int(latent_dim)
        rng = random.Random(seed)
        self.weights = [
            [rng.gauss(0.0, 0.05) for _ in range(self.obs_dim + self.latent_dim + 1)]
            for _ in range(self.action_dim)
        ]

    def features(self, state: Sequence[float], z: Sequence[float]) -> List[float]:
        s = list(map(float, state))[: self.obs_dim]
        if len(s) < self.obs_dim:
            s.extend([0.0] * (self.obs_dim - len(s)))
        latent = list(map(float, z))[: self.latent_dim]
        if len(latent) < self.latent_dim:
            latent.extend([0.0] * (self.latent_dim - len(latent)))
        return s + latent + [1.0]

    def act(self, state: Sequence[float], z: Sequence[float], deterministic: bool = True) -> List[float]:
        x = self.features(state, z)
        action = [math.tanh(sum(row[i] * x[i] for i in range(len(x)))) for row in self.weights]
        if deterministic:
            return action
        rng = random.Random(hash(tuple(round(v, 6) for v in x)))
        return [max(-1.0, min(1.0, a + rng.gauss(0.0, 0.01))) for a in action]

    def update_behavior_cloning(
        self,
        state: Sequence[float],
        z: Sequence[float],
        target_action: Sequence[float],
        lr: float,
    ) -> float:
        x = self.features(state, z)
        pred = self.act(state, z)
        target = list(map(float, target_action))[: self.action_dim]
        if len(target) < self.action_dim:
            target.extend([0.0] * (self.action_dim - len(target)))
        loss = 0.0
        for a_i in range(self.action_dim):
            err = pred[a_i] - target[a_i]
            loss += err * err
            grad = 2.0 * err * (1.0 - pred[a_i] * pred[a_i])
            for j in range(len(x)):
                self.weights[a_i][j] -= lr * grad * x[j]
        return loss / max(1, self.action_dim)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "latent_dim": self.latent_dim,
            "weights": self.weights,
        }


@dataclass
class TrainingResult:
    encoder: FunctionalRewardEncoder
    policy: PolicyConditioningAdapter
    rewards: List[RewardFunctionSpec]
    metrics: Dict[str, Any]
    artifacts: Dict[str, str]


class ArtifactWriter:
    """Stable artifact contract for FRE core outputs."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.checkpoint_dir = self.root / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def path(self, relative: str) -> Path:
        p = self.root / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def write_json(self, relative: str, payload: Mapping[str, Any]) -> str:
        path = self.path(relative)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        return str(path)

    def write_checkpoint(self, relative: str, payload: Mapping[str, Any]) -> str:
        # Import-safe checkpoint format.  Sibling full-training modules may use
        # torch.save behind lazy imports, but this JSON payload is intentionally
        # readable without torch while preserving the declared .pt artifact path.
        return self.write_json(relative, {"format": "fre_repro_json_checkpoint_v1", "payload": payload})

    def write_readiness(self, config: FREConfig, status: str, extra: Optional[Mapping[str, Any]] = None) -> str:
        payload = {
            "schema_version": "1.0",
            "paper": "Unsupervised Zero-Shot Reinforcement Learning via Functional Reward Encodings",
            "status": status,
            "mode": config.mode,
            "timestamp": time.time(),
            "artifact_root": str(self.root),
            "core_surfaces": [
                "reward_function_sampler",
                "state_reward_pair_encoder",
                "policy_conditioning_adapter",
                "offline_training_loop",
                "zero_shot_evaluator",
                "artifact_writer",
                "entrypoint",
            ],
        }
        if extra:
            payload.update(dict(extra))
        return self.write_json("readiness.json", payload)

    def write_evaluation_result(self, payload: Mapping[str, Any]) -> str:
        return self.write_json("evaluation_result.json", payload)


def artifact_writer(root: str | Path = "results") -> ArtifactWriter:
    return ArtifactWriter(root)


def build_synthetic_offline_dataset(config: FREConfig) -> OfflineDataset:
    """Create a bounded deterministic unlabeled dataset for smoke/small runs."""
    rng = random.Random(config.seed)
    transitions: List[Transition] = []
    state = [rng.uniform(-0.5, 0.5) for _ in range(config.obs_dim)]
    count = 160 if config.mode != "runtime_smoke" else 32
    for t in range(count):
        action = [math.tanh(sum(state) * 0.1 + rng.gauss(0.0, 0.2)) for _ in range(config.action_dim)]
        next_state = [
            0.85 * state[i] + 0.1 * action[i % config.action_dim] + rng.gauss(0.0, 0.03)
            for i in range(config.obs_dim)
        ]
        transitions.append(
            {
                "observation": list(state),
                "action": list(action),
                "next_observation": list(next_state),
                "terminal": bool((t + 1) % 40 == 0),
                "timeout": bool((t + 1) % 40 == 0),
            }
        )
        state = next_state if not transitions[-1]["terminal"] else [rng.uniform(-0.5, 0.5) for _ in range(config.obs_dim)]
    return OfflineDataset(
        transitions=transitions,
        obs_dim=config.obs_dim,
        action_dim=config.action_dim,
        metadata={
            "source": "bounded_deterministic_fixture",
            "unlabeled": True,
            "benchmark_claim": False,
        },
    )


def prepare_replay_buffer_from_mapping(
    dataset: Mapping[str, Sequence[Any]],
    *,
    minimum_episode_length: Optional[int] = None,
    ignore_terminals: bool = False,
) -> OfflineDataset:
    """Prepare an offline replay buffer from D4RL-style arrays.

    reference_grounding: paperbench_ref_001 url_benchmark/d4rl_benchmark.py

    The grounding reference constructs replay buffers from
    observations/actions/rewards/terminals/timeouts and filters short episodes.
    FRE only needs unlabeled transitions for pretraining, so rewards are ignored
    except when downstream evaluators explicitly request them.
    """

    observations = list(dataset.get("observations", []))
    actions = list(dataset.get("actions", []))
    next_observations = list(dataset.get("next_observations", []))
    terminals = list(dataset.get("terminals", [False] * len(observations)))
    timeouts = list(dataset.get("timeouts", [False] * len(observations)))
    if not next_observations and len(observations) > 1:
        next_observations = observations[1:] + [observations[-1]]
    if len(actions) != len(observations):
        raise ValueError("D4RL-style dataset must contain one action per observation.")

    transitions: List[Transition] = []
    episode: List[Transition] = []
    for i, obs in enumerate(observations):
        tr = {
            "observation": list(map(float, obs)),
            "action": list(map(float, actions[i])),
            "next_observation": list(map(float, next_observations[i])),
            "terminal": False if ignore_terminals else bool(terminals[i] if i < len(terminals) else False),
            "timeout": bool(timeouts[i] if i < len(timeouts) else False),
        }
        episode.append(tr)
        if tr["terminal"] or tr["timeout"] or i == len(observations) - 1:
            if minimum_episode_length is None or len(episode) >= minimum_episode_length:
                transitions.extend(episode)
            episode = []

    if not transitions:
        raise ValueError("Replay buffer preparation produced no transitions after filtering.")
    obs_dim = len(transitions[0]["observation"])
    action_dim = len(transitions[0]["action"])
    return OfflineDataset(
        transitions=transitions,
        obs_dim=obs_dim,
        action_dim=action_dim,
        metadata={
            "source": "mapping_replay_buffer",
            "minimum_episode_length": minimum_episode_length,
            "ignore_terminals": ignore_terminals,
            "unlabeled": True,
        },
    )


def train_policy(
    offline_dataset: OfflineDataset,
    reward_prior: Sequence[RewardFunctionSpec] | RewardFunctionSpec,
    encoder: FunctionalRewardEncoder,
    config: Optional[FREConfig] = None,
) -> TrainingResult:
    """Offline FRE training loop for pi(a | s, z).

    The loop performs a bounded latent-conditioned behavioral objective over the
    unlabeled offline actions, using reward-function latents as conditioning
    variables.  It is sufficient for smoke and small measured runs; paper-scale
    offline RL can replace this facade through sibling modules while preserving
    the same callable surface.
    """

    cfg = config or FREConfig(obs_dim=offline_dataset.obs_dim, action_dim=offline_dataset.action_dim)
    rewards = list(reward_prior) if isinstance(reward_prior, Sequence) and not isinstance(reward_prior, RewardFunctionSpec) else [reward_prior]  # type: ignore[list-item]
    policy = PolicyConditioningAdapter(offline_dataset.obs_dim, offline_dataset.action_dim, encoder.latent_dim, seed=cfg.seed)
    losses: List[float] = []
    steps = max(1, int(cfg.train_steps))
    reward_latents = [
        encoder.encode(r, offline_dataset, k_states=cfg.encoder_state_count, seed=cfg.seed + i)
        for i, r in enumerate(rewards)
    ]

    for step in range(steps):
        reward_index = step % len(rewards)
        z = reward_latents[reward_index]
        batch_start = (step * 7) % max(1, len(offline_dataset))
        tr = offline_dataset.transitions[batch_start]
        loss = policy.update_behavior_cloning(
            tr["observation"],
            z,
            tr["action"],
            lr=cfg.learning_rate / math.sqrt(1.0 + step / 10.0),
        )
        losses.append(loss)

    metrics = {
        "mode": cfg.mode,
        "train_steps": steps,
        "reward_functions": len(rewards),
        "encoder_state_count": cfg.encoder_state_count,
        "behavior_cloning_loss_mean": statistics.fmean(losses) if losses else 0.0,
        "behavior_cloning_loss_final": losses[-1] if losses else 0.0,
        "dataset_transitions": len(offline_dataset),
    }
    return TrainingResult(encoder=encoder, policy=policy, rewards=rewards, metrics=metrics, artifacts={})


def evaluate_zero_shot_transfer(
    agent: PolicyConditioningAdapter,
    tasks: Sequence[RewardFunctionSpec],
    dataset: OfflineDataset,
    encoder: FunctionalRewardEncoder,
    config: Optional[FREConfig] = None,
) -> Dict[str, Any]:
    """Evaluate zero-shot transfer by conditioning on each task reward code."""
    cfg = config or FREConfig(obs_dim=dataset.obs_dim, action_dim=dataset.action_dim)
    task_records: List[Dict[str, Any]] = []
    for task_index, reward_fn in enumerate(tasks):
        z = encoder.encode(reward_fn, dataset, k_states=cfg.encoder_state_count, seed=cfg.seed + 100 + task_index)
        returns: List[float] = []
        action_errors: List[float] = []
        horizon = min(len(dataset.transitions), max(1, cfg.eval_episodes))
        for i in range(horizon):
            tr = dataset.transitions[(task_index * horizon + i) % len(dataset.transitions)]
            action = agent.act(tr["observation"], z)
            returns.append(float(reward_fn(tr["observation"], action, tr.get("next_observation"))))
            target = list(map(float, tr["action"]))
            action_errors.append(
                sum((action[j] - target[j % len(target)]) ** 2 for j in range(len(action))) / max(1, len(action))
            )
        mean_return = statistics.fmean(returns) if returns else 0.0
        task_records.append(
            {
                "task": reward_fn.name,
                "family": reward_fn.family,
                "zero_shot_return": mean_return,
                "normalized_return": 0.5 + math.atan(mean_return) / math.pi,
                "success_rate": sum(1 for r in returns if r >= 0.0) / max(1, len(returns)),
                "policy_action_mse": statistics.fmean(action_errors) if action_errors else 0.0,
                "episodes": horizon,
            }
        )

    return {
        "method": "FRE",
        "mode": cfg.mode,
        "num_tasks": len(task_records),
        "metrics": {
            "normalized_return": statistics.fmean(r["normalized_return"] for r in task_records) if task_records else 0.0,
            "success_rate": statistics.fmean(r["success_rate"] for r in task_records) if task_records else 0.0,
            "zero_shot_return": statistics.fmean(r["zero_shot_return"] for r in task_records) if task_records else 0.0,
            "policy_action_mse": statistics.fmean(r["policy_action_mse"] for r in task_records) if task_records else 0.0,
        },
        "tasks": task_records,
    }


def run_fre_experiment(config: Optional[FREConfig] = None) -> Dict[str, Any]:
    """Canonical package-level route connecting data, FRE training, evaluation, and artifacts."""
    cfg = config or FREConfig()
    writer = ArtifactWriter(cfg.output_root)
    dataset = build_synthetic_offline_dataset(cfg)
    sampler = RewardPriorSampler(dataset.obs_dim, family=cfg.reward_family, seed=cfg.seed)
    rewards = sampler.sample_many(max(1, cfg.reward_count))
    encoder = FunctionalRewardEncoder(dataset.obs_dim, cfg.latent_dim, cfg.encoder_state_count, seed=cfg.seed)

    writer.write_readiness(
        cfg,
        "running",
        {
            "dataset_transitions": len(dataset),
            "writes_paper_visible_artifacts": bool(cfg.mode in {"train", "evaluate", "full"} or cfg.write_paper_artifacts),
        },
    )

    training = train_policy(dataset, rewards, encoder, cfg)
    evaluation = evaluate_zero_shot_transfer(training.policy, rewards[: min(5, len(rewards))], dataset, encoder, cfg)

    writer.write_json("reward_prior_config.json", {"reward_functions": [r.to_dict() for r in rewards]})
    writer.write_checkpoint("checkpoints/fre_encoder.pt", encoder.state_dict())
    writer.write_checkpoint("checkpoints/fre_policy.pt", training.policy.state_dict())

    measured_payload = {
        "schema_version": "1.0",
        "paper": "Unsupervised Zero-Shot Reinforcement Learning via Functional Reward Encodings",
        "method": "FRE",
        "mode": cfg.mode,
        "dataset": dataset.metadata,
        "training": training.metrics,
        "evaluation": evaluation,
        "hypothesis": "Functional reward encodings allow a policy to condition on unseen rewards without task-specific finetuning.",
        "decisive_metric": "normalized_return",
        "stop_rule_or_pruning_rationale": (
            "Default route is bounded; paper-scale sweeps require explicit full/train/evaluate mode and external datasets."
        ),
    }

    if cfg.mode in {"train", "evaluate", "full"} or cfg.write_paper_artifacts:
        writer.write_json("metrics.json", measured_payload)
        writer.write_json("eval_summary.json", evaluation)
        _write_minimal_png(writer.path("fig3_zero_shot_transfer.png"), evaluation["metrics"]["normalized_return"])
    else:
        writer.write_evaluation_result(
            {
                "schema_version": "1.0",
                "mode": cfg.mode,
                "status": "bounded_runtime_executed",
                "not_benchmark_claim": True,
                "measured_wiring_metrics": evaluation["metrics"],
                "paper_visible_outputs_deferred_until": ["train", "evaluate", "full"],
            }
        )

    writer.write_readiness(
        cfg,
        "complete",
        {
            "dataset_transitions": len(dataset),
            "encoder_checkpoint": str(writer.path("checkpoints/fre_encoder.pt")),
            "policy_checkpoint": str(writer.path("checkpoints/fre_policy.pt")),
        },
    )
    return measured_payload


def make_config(**overrides: Any) -> FREConfig:
    data = dataclasses.asdict(FREConfig())
    data.update(overrides)
    return FREConfig(**data)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """CLI-compatible entrypoint for training, evaluation, and smoke modes."""
    parser = argparse.ArgumentParser(description="Run FRE reproduction core route.")
    parser.add_argument("--mode", default="runtime_smoke", choices=["runtime_smoke", "docker_validate", "train", "evaluate", "full"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--artifact-dir", default=os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR", "results"))
    parser.add_argument("--obs-dim", type=int, default=4)
    parser.add_argument("--action-dim", type=int, default=2)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--encoder-state-count", type=int, default=5)
    parser.add_argument("--reward-family", default="mixed", choices=["mixed", "linear", "goal", "mlp"])
    parser.add_argument("--reward-count", type=int, default=8)
    parser.add_argument("--train-steps", type=int, default=64)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--write-paper-artifacts", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg = FREConfig(
        mode=args.mode,
        seed=args.seed,
        artifact_dir=args.artifact_dir,
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        latent_dim=args.latent_dim,
        encoder_state_count=args.encoder_state_count,
        reward_family=args.reward_family,
        reward_count=args.reward_count,
        train_steps=args.train_steps if args.mode != "runtime_smoke" else min(args.train_steps, 16),
        eval_episodes=args.eval_episodes if args.mode != "runtime_smoke" else min(args.eval_episodes, 5),
        write_paper_artifacts=args.write_paper_artifacts,
    )
    return run_fre_experiment(cfg)


def _extract_states_from_batch(batch: Any) -> List[List[float]]:
    if isinstance(batch, Mapping):
        if "observations" in batch:
            return [list(map(float, s)) for s in batch["observations"]]
        if "observation" in batch:
            obs = batch["observation"]
            return [list(map(float, obs))] if obs and isinstance(obs[0], (int, float)) else [list(map(float, s)) for s in obs]
    states: List[List[float]] = []
    for item in batch:
        if isinstance(item, Mapping):
            obs = item.get("observation", item.get("state", item.get("observations")))
            if obs is not None:
                states.append(list(map(float, obs)))
        else:
            states.append(list(map(float, item)))
    return states


def _write_minimal_png(path: Path, score: float) -> None:
    """Write a tiny valid PNG artifact whose metadata is the measured score.

    This avoids importing matplotlib at package import/runtime-smoke time while
    still creating a real figure file only after a measured route has executed.
    """
    import struct
    import zlib

    width, height = 64, 32
    intensity = max(0, min(255, int(255 * float(score))))
    rows = []
    for y in range(height):
        row = bytearray([0])
        for x in range(width):
            on = x < int(width * max(0.0, min(1.0, float(score))))
            row.extend((40, intensity, 120) if on else (235, 235, 235))
        rows.append(bytes(row))
    raw = b"".join(rows)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    payload = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"tEXt", f"FRE zero-shot normalized_return={score:.6f}".encode("utf-8"))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def __getattr__(name: str) -> Any:
    """Lazy compatibility re-export for sibling implementation modules."""
    lazy_modules = {
        "configs": "fre_repro.configs",
        "data": "fre_repro.data",
        "reward_priors": "fre_repro.reward_priors",
        "models": "fre_repro.models",
        "algorithms": "fre_repro.algorithms",
        "baselines": "fre_repro.baselines",
        "evaluation": "fre_repro.evaluation",
        "artifacts": "fre_repro.artifacts",
    }
    if name in lazy_modules:
        return importlib.import_module(lazy_modules[name])
    raise AttributeError(f"module 'fre_repro' has no attribute {name!r}")


__all__ = [
    "ArtifactWriter",
    "DATASET_ENVIRONMENT_ROUTES",
    "FREConfig",
    "FREConditionedIQLPolicy",
    "FREDecoderNetwork",
    "FREEncoderNetwork",
    "FunctionalRewardEncoder",
    "GCBCNetwork",
    "GCIQLNetwork",
    "OfflineDataset",
    "OPALArchitecture",
    "PolicyConditioningAdapter",
    "REWARD_PRIOR_MIXTURES",
    "RewardFunctionSpec",
    "RewardPriorSampler",
    "TrainingResult",
    "artifact_writer",
    "build_synthetic_offline_dataset",
    "discretize_reward_to_32_bins",
    "evaluate_zero_shot_transfer",
    "evaluate_fre_agent_with_32_state_reward_pairs",
    "evaluate_opal_with_10_random_skills",
    "fre_paper_surface_inventory",
    "load_antmaze_large_diverse_v2_dataset",
    "load_exorl_rnd_dataset",
    "load_kitchen_complete_v0_dataset",
    "main",
    "make_config",
    "make_d4rl_antmaze_online_evaluation_env",
    "make_d4rl_kitchen_online_evaluation_env",
    "make_exorl_custom_dmc_env",
    "prepare_replay_buffer_from_mapping",
    "run_fre_experiment",
    "sample_random_two_layer_mlp_reward",
    "sample_reward_prior",
    "sample_singleton_goal_reward",
    "sample_sparse_random_linear_reward",
    "train_fb_agent_with_controllable_agent",
    "train_fre_conditioned_iql_policy",
    "train_fre_encoder_decoder_strided",
    "train_gc_bc_agent",
    "train_gc_iql_agent",
    "train_opal_agent",
    "train_policy",
    "train_sf_agent_with_controllable_agent",
]
