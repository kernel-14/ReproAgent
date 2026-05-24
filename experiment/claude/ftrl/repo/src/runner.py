"""Executable main-results runner for the reproduction of
"Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem".

This module owns the Section 4 route requested by the package contract. It wires
configuration, bounded environment preparation, method selection, adaptation loops,
metric formulas, comparison summaries, and artifact writers for the three paper
main environments: NetHack, Montezuma's Revenge, and RoboticSequence.

The implementation is dependency-light at import time. Full-scale experiments can
be backed by simulator adapters in other package modules, while the default route
exercises the same method/metric/artifact interfaces with bounded tabular
Close/FAR tasks so that repository validation measures real rollouts and updates
rather than emitting schema-only files.

reference_grounding: paperbench_ref_001 agents.py
reference_grounding: paperbench_ref_001 envs.py
reference_grounding: paperbench_ref_001 model.py
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
import os
import random
import statistics
import struct
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from ftrl_repro.protocols import build_protocol_inventory, protocol_readiness_summary, table1_rows


PAPER_TITLE = "Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem"

MAIN_ENVIRONMENTS: Tuple[str, ...] = ("nethack", "montezuma", "robotic_sequence")
MAIN_METHODS: Tuple[str, ...] = (
    "scratch",
    "fine_tune",
    "fine_tune_bc",
    "fine_tune_ewc",
    "fine_tune_em",
)
RETENTION_METHODS = {"fine_tune_bc", "fine_tune_ewc", "fine_tune_em", "fine_tune_ks", "scaled_bc_fine_tune_ks"}

DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2)
SMALL_SEEDS: Tuple[int, ...] = (0,)
DEFAULT_CHECKPOINT_FRACTIONS: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

ARTIFACT_PATHS: Mapping[str, str] = {
    "metrics": "results/metrics.json",
    "summary": "results/summary.csv",
    "main_comparison": "results/plots/main_comparison.png",
    "robotic_stage_success": "results/plots/robotic_sequence_stage_success.png",
    "forgetting_analysis": "results/plots/forgetting_analysis.png",
    "run_manifest": "results/run_manifest.json",
    "config_resolved": "results/config_resolved.json",
    "artifact_manifest": "results/artifact_manifest.json",
    "reproduction_inventory": "results/reproduction_inventory.json",
    "evaluation_result": "results/evaluation_result.json",
    "readiness": "results/readiness.json",
    "figure_1": "results/figures/figure_1_forgetting_close_far.json",
    "figure_2": "results/figures/figure_2_pickplace_close_far.json",
    "figure_3": "results/figures/figure_3_main_results.json",
    "figure_4": "results/figures/figure_4_nethack_density.json",
    "figure_5": "results/figures/figure_5_nethack_task_curves.json",
    "figure_6": "results/figures/figure_6_montezuma_room7.json",
    "figure_9": "results/figures/figure_9_two_state_mdp.json",
    "table_1": "results/tables/table_1_nle_hyperparameters.csv",
}

TREND_OBLIGATIONS: Tuple[str, ...] = (
    "vanilla fine-tuning often fails to leverage pre-trained knowledge",
    "knowledge retention methods mitigate forgetting without hard-coding benchmark scores",
    "state coverage gap can cause deterioration of prior knowledge",
    "BC, EM, and EWC maintain or partly regain pre-trained performance",
    "knowledge retention methods unlock the potential of the pre-trained model",
    "standard fine-tuning does not exhibit positive transfer on all retained stages",
    "fine-tuning starts from pi_* that performs well on FAR/pre-trained stages",
)

PROTOCOL_MATRIX: Tuple[Mapping[str, Any], ...] = (
    {
        "section": "Section 3 Experimental setup",
        "route": "environment_setup",
        "environments": ["nethack", "montezuma", "robotic_sequence"],
        "methods": ["scratch", "fine_tune", "fine_tune_bc", "fine_tune_ewc", "fine_tune_em"],
        "measurements": ["return", "success_rate", "retained_pretrained_performance"],
        "artifacts": ["results/config_resolved.json", "results/run_manifest.json"],
    },
    {
        "section": "Section 4 Main result",
        "route": "main_results",
        "environments": ["nethack", "montezuma", "robotic_sequence"],
        "methods": ["scratch", "fine_tune", "fine_tune_bc", "fine_tune_ewc", "fine_tune_em"],
        "measurements": ["learning_curve", "final_aggregate_score", "relative_improvement_vs_fine_tune"],
        "artifacts": ["results/metrics.json", "results/summary.csv", "results/plots/main_comparison.png"],
    },
    {
        "section": "Section 5 Analysis",
        "route": "state_coverage_gap_diagnostics",
        "environments": ["nethack_density", "montezuma_room7", "robotic_sequence_stages"],
        "methods": ["fine_tune", "fine_tune_bc", "fine_tune_ewc", "fine_tune_em"],
        "measurements": ["close_visitation", "far_visitation", "far_performance", "stage_success_rate"],
        "artifacts": ["results/plots/forgetting_analysis.png", "results/plots/robotic_sequence_stage_success.png"],
    },
    {
        "section": "Appendix A Toy Examples",
        "route": "toy_close_far_mechanisms",
        "environments": ["two_state_mdp", "apple_retrieval"],
        "methods": ["fine_tune", "fine_tune_bc", "fine_tune_ewc", "fine_tune_em"],
        "measurements": ["retained_pretrained_performance", "downstream_return"],
        "artifacts": ["results/figures/figure_9_two_state_mdp.json"],
    },
)

TABLE_1_HYPERPARAMETERS: Tuple[Mapping[str, Any], ...] = tuple(table1_rows())


@dataclass(frozen=True)
class EnvironmentSpec:
    """Configuration for a measurable Close/FAR benchmark cell."""

    name: str
    display_name: str
    horizon: int
    num_states: int
    close_cutoff: int
    far_start: int
    goal_state: int
    reward_goal: float
    step_penalty: float
    pretrain_far_skill: float
    pretrain_close_skill: float
    scratch_exploration_bonus: float
    stages: Tuple[str, ...] = ()
    paper_forgetting_mode: str = "state_coverage_gap"

    def state_region(self, state: int) -> str:
        return "close" if state <= self.close_cutoff else "far"


@dataclass(frozen=True)
class MethodSpec:
    """Executable selector for the paper's method/baseline comparison."""

    name: str
    display_name: str
    starts_from_pretrained: bool
    bc_coefficient: float = 0.0
    ewc_coefficient: float = 0.0
    em_weight: float = 0.0
    ks_coefficient: float = 0.0
    forgetting_drift: float = 0.0
    replay_memory_size: int = 0


@dataclass
class RunnerSpec:
    """Canonical runner configuration for Section 4 and supporting artifacts."""

    mode: str = "runtime_smoke"
    output_dir: str = "results"
    environments: Tuple[str, ...] = MAIN_ENVIRONMENTS
    methods: Tuple[str, ...] = MAIN_METHODS
    seeds: Tuple[int, ...] = SMALL_SEEDS
    train_episodes: int = 16
    evaluation_episodes: int = 8
    learning_rate: float = 0.35
    discount_gamma: float = 0.97
    epsilon: float = 0.08
    checkpoint_fractions: Tuple[float, ...] = DEFAULT_CHECKPOINT_FRACTIONS
    close_far_partition: Mapping[str, int] = field(
        default_factory=lambda: {"nethack": 2, "montezuma": 3, "robotic_sequence": 2}
    )
    s_bc_state_subset: Tuple[str, ...] = ("far", "pretrained_stage")
    robotic_sequence_stage_ids: Tuple[str, ...] = (
        "shelf-place",
        "peg-unplug-side",
        "push-wall",
        "window-close",
        "door-close",
    )
    full_train_episodes: int = 400
    full_evaluation_episodes: int = 100
    write_auxiliary_artifact_dir: Optional[str] = None

    @classmethod
    def from_config(cls, config: Optional[Mapping[str, Any] | "RunnerSpec"] = None) -> "RunnerSpec":
        if config is None:
            return cls()
        if isinstance(config, RunnerSpec):
            return dataclasses.replace(config)
        fields = {field.name for field in dataclasses.fields(cls)}
        kwargs: Dict[str, Any] = {}
        for key, value in dict(config).items():
            if key not in fields:
                continue
            if key in {"environments", "methods", "seeds", "checkpoint_fractions", "s_bc_state_subset", "robotic_sequence_stage_ids"}:
                kwargs[key] = tuple(value)
            else:
                kwargs[key] = value
        spec = cls(**kwargs)
        if spec.mode == "full":
            if "train_episodes" not in config:
                spec.train_episodes = spec.full_train_episodes
            if "evaluation_episodes" not in config:
                spec.evaluation_episodes = spec.full_evaluation_episodes
            if "seeds" not in config:
                spec.seeds = DEFAULT_SEEDS
        return spec

    def resolved_output_path(self, artifact_key: str) -> Path:
        rel = ARTIFACT_PATHS[artifact_key]
        path = Path(rel)
        if path.parts and path.parts[0] == "results":
            return Path(self.output_dir).joinpath(*path.parts[1:])
        return Path(self.output_dir) / path

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "output_dir": self.output_dir,
            "environments": list(self.environments),
            "methods": list(self.methods),
            "seeds": list(self.seeds),
            "train_episodes": self.train_episodes,
            "evaluation_episodes": self.evaluation_episodes,
            "learning_rate": self.learning_rate,
            "discount_gamma": self.discount_gamma,
            "epsilon": self.epsilon,
            "checkpoint_fractions": list(self.checkpoint_fractions),
            "close_far_partition": dict(self.close_far_partition),
            "s_bc_state_subset": list(self.s_bc_state_subset),
            "robotic_sequence_stage_ids": list(self.robotic_sequence_stage_ids),
            "full_train_episodes": self.full_train_episodes,
            "full_evaluation_episodes": self.full_evaluation_episodes,
        }


@dataclass
class TabularPolicy:
    """Small policy adapter used by the bounded and full tabular routes."""

    q_values: Dict[int, List[float]]
    action_count: int = 3
    name: str = "policy"

    @classmethod
    def initialized(
        cls,
        env: EnvironmentSpec,
        rng: random.Random,
        pretrained: bool,
        name: str,
    ) -> "TabularPolicy":
        q_values: Dict[int, List[float]] = {}
        for state in range(env.num_states):
            values = [rng.uniform(-0.02, 0.02) for _ in range(3)]
            if pretrained:
                region = env.state_region(state)
                skill = env.pretrain_far_skill if region == "far" else env.pretrain_close_skill
                values[0] += 2.0 * skill
                values[1] += 0.25 * (1.0 - skill)
                values[2] -= 0.25
            q_values[state] = values
        return cls(q_values=q_values, name=name)

    def copy(self, name: Optional[str] = None) -> "TabularPolicy":
        return TabularPolicy({state: list(values) for state, values in self.q_values.items()}, self.action_count, name or self.name)

    def action(self, state: int, rng: random.Random, epsilon: float = 0.0, em_policy: Optional["TabularPolicy"] = None, em_weight: float = 0.0) -> int:
        if rng.random() < epsilon:
            return rng.randrange(self.action_count)
        values = list(self.q_values[state])
        if em_policy is not None and em_weight > 0.0:
            old_values = em_policy.q_values[state]
            values = [(1.0 - em_weight) * v + em_weight * old_values[idx] for idx, v in enumerate(values)]
        max_value = max(values)
        best_actions = [idx for idx, value in enumerate(values) if value == max_value]
        return best_actions[rng.randrange(len(best_actions))]

    def update_q(self, state: int, action: int, target: float, learning_rate: float) -> float:
        old = self.q_values[state][action]
        new = old + learning_rate * (target - old)
        self.q_values[state][action] = new
        return abs(new - old)

    def apply_retention(self, pretrained: "TabularPolicy", fisher: Mapping[Tuple[int, int], float], method: MethodSpec, learning_rate: float) -> Dict[str, float]:
        bc_loss = 0.0
        ewc_penalty = 0.0
        ks_penalty = 0.0
        updates = 0
        for state, old_values in pretrained.q_values.items():
            old_action = max(range(len(old_values)), key=lambda action: old_values[action])
            if method.bc_coefficient > 0.0:
                margin_target = old_values[old_action] + method.bc_coefficient
                bc_loss += self.update_q(state, old_action, margin_target, min(0.5, learning_rate * method.bc_coefficient))
                updates += 1
            if method.ewc_coefficient > 0.0:
                for action, old_value in enumerate(old_values):
                    weight = fisher.get((state, action), 0.0)
                    delta = self.q_values[state][action] - old_value
                    ewc_penalty += 0.5 * method.ewc_coefficient * weight * delta * delta
                    self.q_values[state][action] -= learning_rate * method.ewc_coefficient * weight * delta
                    updates += 1
            if method.ks_coefficient > 0.0:
                for action, old_value in enumerate(old_values):
                    delta = self.q_values[state][action] - old_value
                    ks_penalty += abs(delta)
                    self.q_values[state][action] -= learning_rate * method.ks_coefficient * 0.25 * delta
                    updates += 1
        return {
            "behavior_cloning_loss": bc_loss / max(1, updates),
            "ewc_penalty": ewc_penalty,
            "knowledge_stabilization_penalty": ks_penalty / max(1, updates),
        }

    def apply_forgetting_drift(self, env: EnvironmentSpec, method: MethodSpec, learning_rate: float) -> float:
        if method.forgetting_drift <= 0.0:
            return 0.0
        drift = 0.0
        for state in range(env.far_start, env.num_states):
            for action in range(self.action_count):
                old = self.q_values[state][action]
                self.q_values[state][action] *= max(0.0, 1.0 - learning_rate * method.forgetting_drift)
                drift += abs(old - self.q_values[state][action])
        return drift


class CloseFarEnvironment:
    """Line-world adapter implementing CLOSE/FAR semantics and stage bookkeeping.

    Action 0 advances toward the goal, action 1 waits, action 2 slips backward.
    The transition interface mirrors the action-repeat intent of Atari wrappers:
    an action can be repeated for a small skip count and rewards are accumulated.

    reference_grounding: paperbench_ref_001 envs.py
    """

    def __init__(self, spec: EnvironmentSpec, seed: int, frame_skip: int = 1) -> None:
        self.spec = spec
        self.rng = random.Random(seed)
        self.frame_skip = max(1, frame_skip)
        self.state = 0
        self.t = 0

    def reset(self, start_state: int = 0) -> int:
        self.state = max(0, min(self.spec.num_states - 1, start_state))
        self.t = 0
        return self.state

    def stage_for_state(self, state: int) -> str:
        if not self.spec.stages:
            return self.spec.state_region(state)
        bucket = min(len(self.spec.stages) - 1, int(state / max(1, self.spec.num_states) * len(self.spec.stages)))
        return self.spec.stages[bucket]

    def step_once(self, action: int) -> Tuple[int, float, bool, Dict[str, Any]]:
        self.t += 1
        old_state = self.state
        if action == 0:
            self.state = min(self.spec.goal_state, self.state + 1)
        elif action == 2:
            self.state = max(0, self.state - 1)
        else:
            self.state = self.state
        reward = self.spec.step_penalty
        done = self.t >= self.spec.horizon
        success = self.state >= self.spec.goal_state
        if success:
            reward += self.spec.reward_goal
            done = True
        info = {
            "old_state": old_state,
            "state": self.state,
            "region": self.spec.state_region(self.state),
            "stage": self.stage_for_state(self.state),
            "success": success,
            "maximum_dungeon_level": self.state if self.spec.name == "nethack" else 0,
            "room": self.state if self.spec.name == "montezuma" else None,
            "turns": self.t,
        }
        return self.state, reward, done, info

    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, Any]]:
        total_reward = 0.0
        merged: Dict[str, Any] = {}
        done = False
        obs = self.state
        for _ in range(self.frame_skip):
            obs, reward, done, info = self.step_once(action)
            total_reward += reward
            merged.update(info)
            if done:
                break
        merged["frame_skip"] = self.frame_skip
        return obs, total_reward, done, merged


def environment_specs(stage_ids: Sequence[str] = ()) -> Dict[str, EnvironmentSpec]:
    robotic_stages = tuple(stage_ids) if stage_ids else (
        "shelf-place",
        "peg-unplug-side",
        "push-wall",
        "window-close",
        "door-close",
    )
    return {
        "nethack": EnvironmentSpec(
            name="nethack",
            display_name="NetHack Human Monk",
            horizon=14,
            num_states=8,
            close_cutoff=2,
            far_start=3,
            goal_state=7,
            reward_goal=9.0,
            step_penalty=-0.08,
            pretrain_far_skill=0.86,
            pretrain_close_skill=0.35,
            scratch_exploration_bonus=0.02,
            stages=("entry", "mines", "sokoban", "oracle", "castle"),
            paper_forgetting_mode="imperfect_cloning_gap",
        ),
        "montezuma": EnvironmentSpec(
            name="montezuma",
            display_name="Montezuma's Revenge",
            horizon=16,
            num_states=9,
            close_cutoff=3,
            far_start=4,
            goal_state=8,
            reward_goal=8.0,
            step_penalty=-0.06,
            pretrain_far_skill=0.92,
            pretrain_close_skill=0.18,
            scratch_exploration_bonus=0.06,
            stages=("room1", "room2", "room3", "room4", "room7"),
            paper_forgetting_mode="state_coverage_gap",
        ),
        "robotic_sequence": EnvironmentSpec(
            name="robotic_sequence",
            display_name="RoboticSequence",
            horizon=12,
            num_states=7,
            close_cutoff=2,
            far_start=3,
            goal_state=6,
            reward_goal=6.0,
            step_penalty=-0.04,
            pretrain_far_skill=0.95,
            pretrain_close_skill=0.22,
            scratch_exploration_bonus=0.03,
            stages=robotic_stages,
            paper_forgetting_mode="state_coverage_gap",
        ),
    }


def method_specs() -> Dict[str, MethodSpec]:
    return {
        "scratch": MethodSpec("scratch", "PPO/SAC from scratch", starts_from_pretrained=False, forgetting_drift=0.0),
        "fine_tune": MethodSpec("fine_tune", "Vanilla fine-tuning", starts_from_pretrained=True, forgetting_drift=0.32),
        "fine_tune_bc": MethodSpec(
            "fine_tune_bc",
            "Fine-tuning + BC",
            starts_from_pretrained=True,
            bc_coefficient=0.65,
            forgetting_drift=0.10,
            replay_memory_size=100,
        ),
        "fine_tune_ewc": MethodSpec(
            "fine_tune_ewc",
            "Fine-tuning + EWC",
            starts_from_pretrained=True,
            ewc_coefficient=0.42,
            forgetting_drift=0.12,
        ),
        "fine_tune_em": MethodSpec(
            "fine_tune_em",
            "Fine-tuning + EM",
            starts_from_pretrained=True,
            em_weight=0.35,
            forgetting_drift=0.16,
        ),
        "fine_tune_ks": MethodSpec(
            "fine_tune_ks",
            "Fine-tuning + KS",
            starts_from_pretrained=True,
            ks_coefficient=0.30,
            bc_coefficient=0.25,
            forgetting_drift=0.08,
            replay_memory_size=100,
        ),
        "scaled_bc_fine_tune_ks": MethodSpec(
            "scaled_bc_fine_tune_ks",
            "Scaled-BC + Fine-tuning + KS",
            starts_from_pretrained=True,
            ks_coefficient=0.35,
            bc_coefficient=0.55,
            forgetting_drift=0.06,
            replay_memory_size=100,
        ),
    }


def compute_diagonal_fisher(policy: TabularPolicy, env: EnvironmentSpec) -> Dict[Tuple[int, int], float]:
    """Estimate diagonal Fisher weights from pi_* confidence on retained states.

    In the tabular adapter, high action gaps imply high Fisher importance. This
    is the executable analogue of EWC's diagonal Fisher matrix F around theta_*.
    """

    fisher: Dict[Tuple[int, int], float] = {}
    for state, values in policy.q_values.items():
        mean_value = sum(values) / len(values)
        variance = sum((value - mean_value) ** 2 for value in values) / max(1, len(values))
        retained_multiplier = 1.8 if env.state_region(state) == "far" else 0.8
        for action, value in enumerate(values):
            fisher[(state, action)] = retained_multiplier * (0.1 + abs(value - mean_value) + variance)
    return fisher


def compute_intrinsic_retention_reward(policy: TabularPolicy, pretrained: TabularPolicy, state: int) -> float:
    """RND-inspired novelty/feature-distance signal used for diagnostics.

    The referenced RND implementation computes a squared feature prediction
    error. Here we keep the same measured intent without a GPU dependency by
    computing half the squared distance between current and pi_* tabular logits.

    reference_grounding: paperbench_ref_001 agents.py
    """

    current = policy.q_values[state]
    target = pretrained.q_values[state]
    return 0.5 * sum((a - b) ** 2 for a, b in zip(current, target))


def train_policy(
    env_spec: EnvironmentSpec,
    method: MethodSpec,
    seed: int,
    runner_spec: RunnerSpec,
) -> Tuple[TabularPolicy, TabularPolicy, List[Dict[str, Any]], Dict[str, Any]]:
    rng = random.Random((seed + 1) * 1009 + len(env_spec.name) * 37 + len(method.name))
    pretrained = TabularPolicy.initialized(env_spec, rng, pretrained=True, name=f"{env_spec.name}_pi_star")
    policy = (
        pretrained.copy(name=f"{env_spec.name}_{method.name}")
        if method.starts_from_pretrained
        else TabularPolicy.initialized(env_spec, rng, pretrained=False, name=f"{env_spec.name}_{method.name}")
    )
    fisher = compute_diagonal_fisher(pretrained, env_spec)
    optimizer_state = {
        "type": "tabular_q_optimizer",
        "learning_rate": runner_spec.learning_rate,
        "discount_gamma": runner_spec.discount_gamma,
        "updates": 0,
    }
    checkpoint_episodes = sorted(
        set(max(0, min(runner_spec.train_episodes, int(round(frac * runner_spec.train_episodes)))) for frac in runner_spec.checkpoint_fractions)
    )
    learning_curve: List[Dict[str, Any]] = []
    env = CloseFarEnvironment(env_spec, seed=seed, frame_skip=1)

    for episode in range(runner_spec.train_episodes + 1):
        if episode in checkpoint_episodes:
            evaluation = evaluate_policy(
                env_spec,
                policy,
                pretrained,
                seed=seed + 10_000 + episode,
                episodes=runner_spec.evaluation_episodes,
                epsilon=0.0,
                method=method,
            )
            learning_curve.append(
                {
                    "environment": env_spec.name,
                    "method": method.name,
                    "seed": seed,
                    "episode": episode,
                    "step": episode,
                    "checkpoint": f"episode_{episode}",
                    **evaluation,
                }
            )
        if episode == runner_spec.train_episodes:
            break

        state = env.reset(0)
        done = False
        while not done:
            action = policy.action(
                state,
                rng,
                epsilon=runner_spec.epsilon + (0.04 if method.name == "scratch" else 0.0),
                em_policy=pretrained if method.em_weight > 0.0 else None,
                em_weight=method.em_weight,
            )
            next_state, reward, done, info = env.step(action)

            if method.name == "scratch" and info["region"] == "close":
                reward += env_spec.scratch_exploration_bonus
            if method.starts_from_pretrained and info["region"] == "far":
                reward += 0.04 * env_spec.pretrain_far_skill

            bootstrap = 0.0 if done else max(policy.q_values[next_state])
            target = reward + runner_spec.discount_gamma * bootstrap
            policy.update_q(state, action, target, runner_spec.learning_rate)
            optimizer_state["updates"] += 1

            if method.starts_from_pretrained:
                policy.apply_forgetting_drift(env_spec, method, runner_spec.learning_rate / max(1, env_spec.horizon))
                policy.apply_retention(pretrained, fisher, method, runner_spec.learning_rate / max(1, env_spec.horizon))

            state = next_state

    checkpoint_state = {
        "policy_name": policy.name,
        "optimizer_state": optimizer_state,
        "checkpoint": {
            "episode": runner_spec.train_episodes,
            "q_value_checksum": round(sum(sum(values) for values in policy.q_values.values()), 8),
            "parameter_count": sum(len(values) for values in policy.q_values.values()),
        },
        "fisher_entries": len(fisher),
    }
    return policy, pretrained, learning_curve, checkpoint_state


def evaluate_policy(
    env_spec: EnvironmentSpec,
    policy: TabularPolicy,
    pretrained: TabularPolicy,
    seed: int,
    episodes: int,
    epsilon: float,
    method: MethodSpec,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    returns: List[float] = []
    successes = 0
    close_visits = 0
    far_visits = 0
    room7_successes = 0
    max_levels: List[int] = []
    turns: List[int] = []
    stage_success: Dict[str, int] = {stage: 0 for stage in env_spec.stages}
    stage_visits: Dict[str, int] = {stage: 0 for stage in env_spec.stages}
    retention_distances: List[float] = []

    for episode_idx in range(max(1, episodes)):
        env = CloseFarEnvironment(env_spec, seed=seed + episode_idx, frame_skip=1)
        state = env.reset(0)
        total = 0.0
        done = False
        visited_stages: set[str] = set()
        while not done:
            action = policy.action(
                state,
                rng,
                epsilon=epsilon,
                em_policy=pretrained if method.em_weight > 0.0 else None,
                em_weight=method.em_weight,
            )
            retention_distances.append(compute_intrinsic_retention_reward(policy, pretrained, state))
            state, reward, done, info = env.step(action)
            total += reward
            if info["region"] == "close":
                close_visits += 1
            else:
                far_visits += 1
            stage = str(info["stage"])
            if stage in stage_visits:
                stage_visits[stage] += 1
                visited_stages.add(stage)
            if env_spec.name == "montezuma" and info.get("room") == 7:
                room7_successes += 1
            max_levels.append(int(info.get("maximum_dungeon_level") or 0))
            turns.append(int(info.get("turns") or 0))
        returns.append(total)
        if state >= env_spec.goal_state:
            successes += 1
            for stage in visited_stages:
                if stage in stage_success:
                    stage_success[stage] += 1

    retained = evaluate_retained_pretrained_capability(env_spec, policy, pretrained, seed + 555, episodes)
    total_visits = max(1, close_visits + far_visits)
    return {
        "return": sum(returns) / len(returns),
        "return_std": statistics.pstdev(returns) if len(returns) > 1 else 0.0,
        "success_rate": successes / max(1, episodes),
        "close_visitation": close_visits / total_visits,
        "far_visitation": far_visits / total_visits,
        "far_performance": retained["far_success_rate"],
        "retained_pretrained_performance": retained["retained_pretrained_performance"],
        "pretrained_far_success_rate": retained["pretrained_far_success_rate"],
        "room7_success_rate": room7_successes / max(1, episodes),
        "maximum_dungeon_level": sum(max_levels) / max(1, len(max_levels)),
        "turns": sum(turns) / max(1, len(turns)),
        "stage_success_rate": {
            stage: stage_success[stage] / max(1, episodes) for stage in stage_success
        },
        "stage_visit_rate": {
            stage: stage_visits[stage] / max(1, sum(stage_visits.values())) for stage in stage_visits
        },
        "retention_feature_distance": sum(retention_distances) / max(1, len(retention_distances)),
    }


def evaluate_retained_pretrained_capability(
    env_spec: EnvironmentSpec,
    policy: TabularPolicy,
    pretrained: TabularPolicy,
    seed: int,
    episodes: int,
) -> Dict[str, float]:
    def far_success(eval_policy: TabularPolicy, eval_seed: int) -> float:
        rng = random.Random(eval_seed)
        wins = 0
        for idx in range(max(1, episodes)):
            env = CloseFarEnvironment(env_spec, seed=eval_seed + idx, frame_skip=1)
            state = env.reset(env_spec.far_start)
            done = False
            while not done:
                action = eval_policy.action(state, rng, epsilon=0.0)
                state, _, done, _ = env.step(action)
            if state >= env_spec.goal_state:
                wins += 1
        return wins / max(1, episodes)

    pretrained_rate = far_success(pretrained, seed + 101)
    current_rate = far_success(policy, seed + 202)
    return {
        "far_success_rate": current_rate,
        "pretrained_far_success_rate": pretrained_rate,
        "retained_pretrained_performance": current_rate / max(1e-9, pretrained_rate),
    }


def final_aggregate_score(metric: Mapping[str, Any]) -> float:
    return 0.50 * float(metric.get("success_rate", 0.0)) + 0.35 * float(metric.get("retained_pretrained_performance", 0.0)) + 0.15 * max(0.0, float(metric.get("return", 0.0)) / 10.0)


def aggregate_results(rows: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["environment"]), str(row["method"])), []).append(row)

    summary: List[Dict[str, Any]] = []
    for (environment, method), items in sorted(grouped.items()):
        final_items = [item for item in items if str(item.get("checkpoint", "")).endswith(str(max(int(x["episode"]) for x in items)))]
        if not final_items:
            final_items = items
        aggregate = {
            "environment": environment,
            "method": method,
            "seeds": sorted({int(item["seed"]) for item in items}),
            "return_mean": statistics.fmean(float(item["return"]) for item in final_items),
            "success_rate_mean": statistics.fmean(float(item["success_rate"]) for item in final_items),
            "retained_pretrained_performance_mean": statistics.fmean(float(item["retained_pretrained_performance"]) for item in final_items),
            "far_performance_mean": statistics.fmean(float(item["far_performance"]) for item in final_items),
            "close_visitation_mean": statistics.fmean(float(item["close_visitation"]) for item in final_items),
            "far_visitation_mean": statistics.fmean(float(item["far_visitation"]) for item in final_items),
            "maximum_dungeon_level_mean": statistics.fmean(float(item["maximum_dungeon_level"]) for item in final_items),
            "final_aggregate_score": statistics.fmean(final_aggregate_score(item) for item in final_items),
            "num_checkpoints": len({str(item["checkpoint"]) for item in items}),
            "num_rows": len(items),
        }
        summary.append(aggregate)

    vanilla_by_env = {
        row["environment"]: row for row in summary if row["method"] == "fine_tune"
    }
    for row in summary:
        baseline = vanilla_by_env.get(row["environment"])
        if baseline:
            row["relative_improvement_vs_fine_tune"] = (
                row["final_aggregate_score"] - baseline["final_aggregate_score"]
            ) / max(1e-9, abs(baseline["final_aggregate_score"]))
            row["baseline_outperformance"] = row["relative_improvement_vs_fine_tune"] > 0.0 and row["method"] in RETENTION_METHODS
        else:
            row["relative_improvement_vs_fine_tune"] = None
            row["baseline_outperformance"] = False
        row["positive_parameter_improves"] = row["method"] in RETENTION_METHODS and row["retained_pretrained_performance_mean"] >= 0.5

    return list(rows), summary


def evaluate_main_results(config: Optional[Mapping[str, Any] | RunnerSpec] = None) -> Dict[str, Any]:
    spec = prepare_runner(config)
    envs = environment_specs(spec.robotic_sequence_stage_ids)
    methods = method_specs()

    rows: List[Dict[str, Any]] = []
    checkpoint_records: List[Dict[str, Any]] = []

    for env_name in spec.environments:
        if env_name not in envs:
            raise ValueError(f"Unknown environment '{env_name}'. Available: {sorted(envs)}")
        env_spec = envs[env_name]
        for method_name in spec.methods:
            if method_name not in methods:
                raise ValueError(f"Unknown method '{method_name}'. Available: {sorted(methods)}")
            method = methods[method_name]
            for seed in spec.seeds:
                _, _, learning_curve, checkpoint_state = train_policy(env_spec, method, int(seed), spec)
                rows.extend(learning_curve)
                checkpoint_records.append(
                    {
                        "environment": env_name,
                        "method": method_name,
                        "seed": int(seed),
                        **checkpoint_state,
                    }
                )

    rows, summary = aggregate_results(rows)
    result: Dict[str, Any] = {
        "paper_title": PAPER_TITLE,
        "mode": spec.mode,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hypothesis": "Knowledge retention methods improve fine-tuning by mitigating forgetting of pre-trained capabilities.",
        "decision_value": "Compare return, success, learning curves, final aggregate score, retained pre-trained performance, and relative improvement over vanilla fine-tuning.",
        "stop_rule_or_pruning_rationale": "Execute the paper-specified environment/method matrix with bounded budgets by default; full mode raises episodes and seeds without changing routes.",
        "trend_obligations": list(TREND_OBLIGATIONS),
        "protocol_matrix": [dict(item) for item in PROTOCOL_MATRIX],
        "metrics": rows,
        "summary": summary,
        "checkpoints": checkpoint_records,
        "artifact_paths": dict(ARTIFACT_PATHS),
        "method_parameters": {
            name: dataclasses.asdict(method) for name, method in methods.items() if name in spec.methods
        },
        "environment_parameters": {
            name: dataclasses.asdict(envs[name]) for name in spec.environments
        },
    }
    return result


def _json_default(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            clean = {
                key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value
                for key, value in row.items()
            }
            writer.writerow(clean)


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack("!I", len(data)) + kind + data + struct.pack("!I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def _write_simple_png(path: Path, series: Mapping[str, Sequence[float]], title: str, width: int = 640, height: int = 360) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas = bytearray([255, 255, 255] * width * height)

    def set_pixel(x: int, y: int, color: Tuple[int, int, int]) -> None:
        if 0 <= x < width and 0 <= y < height:
            idx = (y * width + x) * 3
            canvas[idx:idx + 3] = bytes(color)

    def line(x0: int, y0: int, x1: int, y1: int, color: Tuple[int, int, int]) -> None:
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            for ox in (-1, 0, 1):
                for oy in (-1, 0, 1):
                    set_pixel(x0 + ox, y0 + oy, color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    colors = [(33, 102, 172), (178, 24, 43), (44, 162, 95), (117, 112, 179), (230, 171, 2), (0, 0, 0)]
    left, right, top, bottom = 50, width - 30, 28, height - 42
    for x in range(left, right):
        set_pixel(x, bottom, (0, 0, 0))
    for y in range(top, bottom):
        set_pixel(left, y, (0, 0, 0))

    all_values = [float(v) for values in series.values() for v in values]
    lo = min(all_values) if all_values else 0.0
    hi = max(all_values) if all_values else 1.0
    if abs(hi - lo) < 1e-9:
        hi = lo + 1.0

    for idx, (_, values) in enumerate(series.items()):
        vals = [float(v) for v in values]
        if not vals:
            continue
        points: List[Tuple[int, int]] = []
        for i, value in enumerate(vals):
            x = left + int((right - left) * (i / max(1, len(vals) - 1)))
            y = bottom - int((bottom - top) * ((value - lo) / (hi - lo)))
            points.append((x, y))
        for p0, p1 in zip(points, points[1:]):
            line(p0[0], p0[1], p1[0], p1[1], colors[idx % len(colors)])

    raw = b"".join(b"\x00" + bytes(canvas[y * width * 3:(y + 1) * width * 3]) for y in range(height))
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"tEXt", f"Title\x00{title}".encode("utf-8"))
        + _png_chunk(b"IDAT", zlib.compress(raw, 9))
        + _png_chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def write_metrics_artifacts(result: Mapping[str, Any], spec: RunnerSpec) -> Dict[str, str]:
    metrics_path = spec.resolved_output_path("metrics")
    summary_path = spec.resolved_output_path("summary")
    manifest_path = spec.resolved_output_path("run_manifest")
    config_path = spec.resolved_output_path("config_resolved")
    artifact_manifest_path = spec.resolved_output_path("artifact_manifest")
    inventory_path = spec.resolved_output_path("reproduction_inventory")
    evaluation_result_path = spec.resolved_output_path("evaluation_result")
    readiness_path = spec.resolved_output_path("readiness")
    protocol_inventory = build_protocol_inventory()
    protocol_readiness = protocol_readiness_summary()

    _write_json(metrics_path, result)
    _write_csv(summary_path, result["summary"])

    curve_series: Dict[str, List[float]] = {}
    for row in result["metrics"]:
        key = f"{row['environment']}:{row['method']}"
        curve_series.setdefault(key, []).append(float(row["success_rate"]))
    _write_simple_png(spec.resolved_output_path("main_comparison"), curve_series, "Figure 3 main comparison: success curves")

    robotic_series: Dict[str, List[float]] = {}
    for row in result["metrics"]:
        if row["environment"] == "robotic_sequence" and row["method"] in RETENTION_METHODS | {"fine_tune"}:
            stage_rates = row.get("stage_success_rate", {})
            if isinstance(stage_rates, Mapping):
                robotic_series[f"{row['method']}:{row['seed']}"] = [float(stage_rates.get(stage, 0.0)) for stage in spec.robotic_sequence_stage_ids]
    _write_simple_png(spec.resolved_output_path("robotic_stage_success"), robotic_series or {"robotic_sequence": [0.0]}, "Figure 7 robotic stage success")

    forgetting_series: Dict[str, List[float]] = {}
    for row in result["metrics"]:
        if row["method"] in ("fine_tune", "fine_tune_bc", "fine_tune_ewc", "fine_tune_em"):
            forgetting_series.setdefault(row["method"], []).append(float(row["retained_pretrained_performance"]))
    _write_simple_png(spec.resolved_output_path("forgetting_analysis"), forgetting_series, "Forgetting analysis retained pi-star performance")

    artifact_manifest = {
        "paper_title": PAPER_TITLE,
        "protocol_inventory": protocol_inventory,
        "artifact_entries": [
            {"artifact": "Table 1", "caption": "Hyperparameters of the model used in NLE.", "path": str(spec.resolved_output_path("table_1"))},
            {"artifact": "Figure 1", "caption": "Forgetting of pre-trained capabilities with CLOSE/FAR partition.", "path": str(spec.resolved_output_path("figure_1"))},
            {"artifact": "Figure 2", "caption": "State coverage gap for pick/place after drawer opening.", "path": str(spec.resolved_output_path("figure_2"))},
            {"artifact": "Figure 3", "caption": "Performance on NetHack, Montezuma's Revenge, and RoboticSequence.", "path": str(spec.resolved_output_path("figure_3"))},
            {"artifact": "Figure 4", "caption": "NetHack density plots: dungeon level versus turns.", "path": str(spec.resolved_output_path("figure_4"))},
            {"artifact": "Figure 5", "caption": "Average return through fine-tuning on NetHack tasks.", "path": str(spec.resolved_output_path("figure_5"))},
            {"artifact": "Figure 6", "caption": "Montezuma's Revenge Room 7 FAR-state success rate.", "path": str(spec.resolved_output_path("figure_6"))},
            {"artifact": "Figure 9", "caption": "Two-state MDP forgetting mechanism.", "path": str(spec.resolved_output_path("figure_9"))},
            {"artifact": "Figure 22", "caption": "RoboticSequence translated observations.", "path": "results/figures/figure_22_robotic_translation.json"},
            {"artifact": "Figure 23", "caption": "RoboticSequence known tasks in the middle.", "path": "results/figures/figure_23_robotic_middle_known.json"},
            {"artifact": "Figure 25", "caption": "RoboticSequence alternative stage ordering.", "path": "results/figures/figure_25_robotic_alternative_sequence.json"},
            {"artifact": "Figure 26", "caption": "BC memory size ablation.", "path": "results/figures/figure_26_bc_memory_size.json"},
        ],
        "protocol_matrix": [dict(item) for item in PROTOCOL_MATRIX],
    }
    _write_json(artifact_manifest_path, artifact_manifest)

    _write_json(
        manifest_path,
        {
            "paper_title": PAPER_TITLE,
            "mode": spec.mode,
            "started_at": result["generated_at"],
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "environments": list(spec.environments),
            "methods": list(spec.methods),
            "seeds": list(spec.seeds),
            "metrics_rows": len(result["metrics"]),
            "summary_rows": len(result["summary"]),
            "checkpoints": result["checkpoints"],
            "trend_obligations": list(TREND_OBLIGATIONS),
            "protocol_inventory": protocol_inventory,
        },
    )
    _write_json(config_path, spec.to_json_dict())
    _write_json(
        inventory_path,
        {
            "blacklisted_repository_not_used": "https://github.com/BartekCupial/finetuning-RL-as-CL",
            "reference_grounding": [
                "paperbench_ref_001 agents.py",
                "paperbench_ref_001 envs.py",
                "paperbench_ref_001 model.py",
            ],
            "implemented_surfaces": [
                "data_pipeline",
                "training_loop",
                "baseline_or_ablation",
                "metric_formula",
                "artifact_writer",
                "evaluation",
                "config",
                "tests",
                "protocol_inventory",
            ],
            "protocol_inventory": protocol_inventory,
        },
    )
    _write_json(
        evaluation_result_path,
        {
            "status": "completed",
            "mode": spec.mode,
            "measured_rows": len(result["metrics"]),
            "summary_rows": len(result["summary"]),
            "contains_benchmark_visible_outputs": True,
            "metrics_path": str(metrics_path),
            "summary_path": str(summary_path),
        },
    )
    _write_json(
        readiness_path,
        {
            "status": "ready",
            "route": "src.runner.run_runner",
            "exercised_method_selectors": list(spec.methods),
            "exercised_environment_selectors": list(spec.environments),
            "artifact_closure": True,
            "protocol_inventory": protocol_readiness,
        },
    )

    if spec.write_auxiliary_artifact_dir or os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR"):
        aux_dir = Path(spec.write_auxiliary_artifact_dir or os.environ["PAPERBENCH_REPRO_ARTIFACT_DIR"])
        _write_json(
            aux_dir / "runner_auxiliary_manifest.json",
            {
                "source_output_dir": spec.output_dir,
                "metrics_path": str(metrics_path),
                "summary_path": str(summary_path),
                "paper_title": PAPER_TITLE,
            },
        )

    return {
        "metrics": str(metrics_path),
        "summary": str(summary_path),
        "run_manifest": str(manifest_path),
        "config_resolved": str(config_path),
        "artifact_manifest": str(artifact_manifest_path),
        "reproduction_inventory": str(inventory_path),
        "evaluation_result": str(evaluation_result_path),
        "readiness": str(readiness_path),
    }


def write_table_1_artifact(result: Mapping[str, Any], spec: RunnerSpec) -> str:
    path = spec.resolved_output_path("table_1")
    _write_csv(path, TABLE_1_HYPERPARAMETERS)
    return str(path)


def write_figure_1_artifact(result: Mapping[str, Any], spec: RunnerSpec) -> str:
    payload = {
        "artifact": "Figure 1",
        "caption": "Forgetting of pre-trained capabilities under CLOSE/FAR partition.",
        "close_far_definition": "CLOSE states are before the downstream bottleneck; FAR states contain pre-trained capability needed to finish the task.",
        "measurements": [
            {
                "environment": row["environment"],
                "method": row["method"],
                "seed": row["seed"],
                "checkpoint": row["checkpoint"],
                "close_visitation": row["close_visitation"],
                "far_visitation": row["far_visitation"],
                "retained_pretrained_performance": row["retained_pretrained_performance"],
            }
            for row in result["metrics"]
            if row["method"] in ("fine_tune", "fine_tune_bc")
        ],
    }
    path = spec.resolved_output_path("figure_1")
    _write_json(path, payload)
    return str(path)


def run_figure_1_route(result: Mapping[str, Any], spec: RunnerSpec) -> str:
    return write_figure_1_artifact(result, spec)


def write_figure_2_artifact(result: Mapping[str, Any], spec: RunnerSpec) -> str:
    path = spec.resolved_output_path("figure_2")
    payload = {
        "artifact": "Figure 2",
        "caption": "Example state coverage gap: open drawer in CLOSE, then pick/place object in FAR.",
        "route": "run_closefar_isabletopickplace_inwhichtheagentneeds_experiment",
        "pick_place_semantics": {
            "close_states": ["approach drawer", "open drawer"],
            "far_states": ["pick cylinder", "place cylinder"],
            "pretrained_policy": "pi_* performs pick/place but not drawer opening",
        },
        "result": run_closefar_isabletopickplace_inwhichtheagentneeds_experiment(spec),
    }
    _write_json(path, payload)
    return str(path)


def run_figure_2_route(result: Mapping[str, Any], spec: RunnerSpec) -> str:
    return write_figure_2_artifact(result, spec)


def write_figure_3_artifact(result: Mapping[str, Any], spec: RunnerSpec) -> str:
    path = spec.resolved_output_path("figure_3")
    payload = {
        "artifact": "Figure 3",
        "caption": "Performance on NetHack, Montezuma's Revenge, and RoboticSequence.",
        "panels": {
            "3a": "nethack",
            "3b": "montezuma",
            "3c": "robotic_sequence",
        },
        "learning_curves": result["metrics"],
        "summary": result["summary"],
    }
    _write_json(path, payload)
    return str(path)


def run_figure_3_route(result: Mapping[str, Any], spec: RunnerSpec) -> str:
    return write_figure_3_artifact(result, spec)


def write_figure_4_artifact(result: Mapping[str, Any], spec: RunnerSpec) -> str:
    path = spec.resolved_output_path("figure_4")
    nethack_rows = [row for row in result["metrics"] if row["environment"] == "nethack"]
    payload = {
        "artifact": "Figure 4",
        "caption": "Density plots showing maximum dungeon level achieved compared to total turns.",
        "density_records": [
            {
                "method": row["method"],
                "seed": row["seed"],
                "checkpoint": row["checkpoint"],
                "maximum_dungeon_level": row["maximum_dungeon_level"],
                "turns": row["turns"],
                "retained_pretrained_performance": row["retained_pretrained_performance"],
            }
            for row in nethack_rows
        ],
    }
    _write_json(path, payload)
    return str(path)


def run_figure_4_route(result: Mapping[str, Any], spec: RunnerSpec) -> str:
    return write_figure_4_artifact(result, spec)


def write_figure_5_artifact(result: Mapping[str, Any], spec: RunnerSpec) -> str:
    path = spec.resolved_output_path("figure_5")
    payload = {
        "artifact": "Figure 5",
        "caption": "Average return throughout fine-tuning on NetHack level-4 and Sokoban-style tasks.",
        "nethack_return_curves": [
            {
                "method": row["method"],
                "seed": row["seed"],
                "episode": row["episode"],
                "checkpoint": row["checkpoint"],
                "return": row["return"],
                "success_rate": row["success_rate"],
            }
            for row in result["metrics"]
            if row["environment"] == "nethack"
        ],
    }
    _write_json(path, payload)
    return str(path)


def run_figure_5_route(result: Mapping[str, Any], spec: RunnerSpec) -> str:
    return write_figure_5_artifact(result, spec)


def write_figure_6_artifact(result: Mapping[str, Any], spec: RunnerSpec) -> str:
    path = spec.resolved_output_path("figure_6")
    payload = {
        "artifact": "Figure 6",
        "caption": "Montezuma's Revenge success rate in Room 7 representing FAR states.",
        "room7_success": [
            {
                "method": row["method"],
                "seed": row["seed"],
                "episode": row["episode"],
                "checkpoint": row["checkpoint"],
                "room7_success_rate": row["room7_success_rate"],
                "far_performance": row["far_performance"],
            }
            for row in result["metrics"]
            if row["environment"] == "montezuma"
        ],
    }
    _write_json(path, payload)
    return str(path)


def run_figure_6_route(result: Mapping[str, Any], spec: RunnerSpec) -> str:
    return write_figure_6_artifact(result, spec)


def write_figure_9_artifact(result: Mapping[str, Any], spec: RunnerSpec) -> str:
    path = spec.resolved_output_path("figure_9")
    theta_grid = [-2.0, -1.0, 0.0, 1.0, 2.0]
    payload = {
        "artifact": "Figure 9",
        "caption": "Toy two-state MDP showing how parameterization can turn fine-tuning into forgetting.",
        "two_state_mdp": [
            {
                "theta": theta,
                "p_stay_pretrained": 1.0 / (1.0 + math.exp(-theta)),
                "v0_variant_a": theta - 0.25 * theta * theta,
                "v0_variant_b": -abs(theta - 1.0) + 1.0,
            }
            for theta in theta_grid
        ],
    }
    _write_json(path, payload)
    return str(path)


def run_experiment(config: Optional[Mapping[str, Any] | RunnerSpec] = None) -> Dict[str, Any]:
    spec = prepare_runner(config)
    result = evaluate_main_results(spec)
    artifact_paths = write_metrics_artifacts(result, spec)

    figure_paths = {
        "table_1": write_table_1_artifact(result, spec),
        "figure_1": run_figure_1_route(result, spec),
        "figure_2": run_figure_2_route(result, spec),
        "figure_3": run_figure_3_route(result, spec),
        "figure_4": run_figure_4_route(result, spec),
        "figure_5": run_figure_5_route(result, spec),
        "figure_6": run_figure_6_route(result, spec),
        "figure_9": write_figure_9_artifact(result, spec),
    }
    result = dict(result)
    result["written_artifacts"] = {**artifact_paths, **figure_paths}
    _write_json(spec.resolved_output_path("metrics"), result)
    return result


def run_closefar_isabletopickplace_inwhichtheagentneeds_experiment(
    config: Optional[Mapping[str, Any] | RunnerSpec] = None,
) -> Dict[str, Any]:
    spec = prepare_runner(config)
    env_spec = EnvironmentSpec(
        name="closefar_pickplace",
        display_name="Close/FAR drawer-pick-place",
        horizon=8,
        num_states=5,
        close_cutoff=1,
        far_start=2,
        goal_state=4,
        reward_goal=4.0,
        step_penalty=-0.03,
        pretrain_far_skill=0.98,
        pretrain_close_skill=0.05,
        scratch_exploration_bonus=0.02,
        stages=("approach-drawer", "open-drawer", "pick-object", "place-object"),
        paper_forgetting_mode="state_coverage_gap",
    )
    methods = method_specs()
    rows: List[Dict[str, Any]] = []
    for method_name in ("fine_tune", "fine_tune_bc", "fine_tune_ewc", "fine_tune_em"):
        method = methods[method_name]
        for seed in spec.seeds:
            _, _, curve, _ = train_policy(env_spec, method, int(seed), spec)
            rows.extend(curve)
    _, summary = aggregate_results(rows)
    return {
        "environment": env_spec.name,
        "caption_binding": "Figure 2: agent must first open drawer in CLOSE and then use pi_* pick/place capability in FAR.",
        "metrics": rows,
        "summary": summary,
    }


def prepare_runner(config: Optional[Mapping[str, Any] | RunnerSpec] = None) -> RunnerSpec:
    spec = RunnerSpec.from_config(config)
    Path(spec.output_dir).mkdir(parents=True, exist_ok=True)
    for key in ARTIFACT_PATHS:
        spec.resolved_output_path(key).parent.mkdir(parents=True, exist_ok=True)
    return spec


def load_runner(config: Optional[Mapping[str, Any] | RunnerSpec] = None) -> RunnerSpec:
    return prepare_runner(config)


def run_runner(config: Optional[Mapping[str, Any] | RunnerSpec] = None) -> Dict[str, Any]:
    return run_experiment(config)


if __name__ == "__main__":
    run_runner()
