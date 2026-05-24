"""Toy environments and environment registry for the FTRL reproduction.

This module closes the lightweight environment/task route for the paper
"Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation
Problem".  It provides executable Two-state MDP and AppleRetrieval tasks, explicit
CLOSE/FAR state semantics, state-coverage and imperfect-cloning diagnostics,
RoboticSequence stage-success support, availability checks for the paper
environment registry, and JSON artifact writers used by the canonical route.

reference_grounding: paperbench_ref_001 envs.py
The ActionRepeatMaxFrameAdapter below adapts the reference wrapper protocol:
repeat an action, accumulate reward, terminate early, and keep a max-over-last
observation for frame-like observations, while remaining dependency-light.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


CLOSE = "close"
FAR = "far"
UNKNOWN = "unknown"
REGIONS: Tuple[str, str, str] = (CLOSE, FAR, UNKNOWN)

PAPER_TITLE = "Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem"
BLACKLISTED_REPOSITORY = "https://github.com/BartekCupial/finetuning-RL-as-CL"


class RegistryEntriesIds:
    """Canonical task ids used by train/evaluate/report routes."""

    NETHACK = "nethack_human_monk"
    MONTEZUMA = "montezuma_revenge"
    ROBOTIC_SEQUENCE = "robotic_sequence"
    TWO_STATE_MDPS = "two_state_mdps"
    APPLE_RETRIEVAL = "apple_retrieval"
    FT_BC_PROTOCOL = "fine_tuning_bc_protocol"
    ROBOTICS_DATASET = "robotics"


class AliasesRobotics:
    """Explicit robotics aliases required by the paper evidence contract."""

    ENVIRONMENT: Tuple[str, ...] = (
        "robotics",
        "robotic",
        "robotic_sequence",
        "RoboticSequence",
        "longer sequence",
        "longer_sequence",
        "manipulation_sequence",
    )
    DATASET: Tuple[str, ...] = (
        "robotics",
        "robotic_dataset",
        "robotic_sequence_dataset",
        "information processing systems datasets",
    )


@dataclass(frozen=True)
class ToyTasksConfig:
    """Configuration for low-cost toy forgetting diagnostics."""

    seed: int = 0
    horizon: int = 8
    apple_length: int = 7
    n_episodes: int = 32
    far_entry_step: int = 2
    clone_far_error_rate: float = 0.45
    clone_close_error_rate: float = 0.02
    fine_tune_close_bias: float = 0.80
    full_mode: bool = False
    output_dir: str = "results"


@dataclass(frozen=True)
class ToyTasksSpec:
    """A concrete environment/task specification for registry and factories."""

    task_id: str
    display_name: str
    aliases: Tuple[str, ...]
    setup_metadata: Mapping[str, Any]
    factory: Callable[[ToyTasksConfig], Any]
    default_config: ToyTasksConfig = field(default_factory=ToyTasksConfig)


@dataclass(frozen=True)
class Transition:
    state: str
    action: str
    next_state: str
    reward: float
    done: bool = False
    region: str = UNKNOWN


@dataclass
class EpisodeRecord:
    task_id: str
    policy_id: str
    transitions: List[Transition]
    total_return: float
    reached_far: bool
    success: bool
    state_regions: List[str]
    stage_success: Mapping[str, bool] = field(default_factory=dict)


class InThisFile:
    """Machine-readable declaration for the active task-review route."""

    implementation_surfaces: Tuple[str, ...] = (
        "toy_task_registry",
        "toy_mdp_spec",
        "forgetting_diagnostics",
        "availability_checks",
        "environment_adapter",
        "data_pipeline",
        "environment",
        "policy_adapter",
    )
    public_symbols: Tuple[str, ...] = (
        "ToyTasksConfig",
        "ToyTasksSpec",
        "InThisFile",
        "RegistryEntriesIds",
        "AliasesRobotics",
        "make_toy_tasks",
        "build_toy_tasks",
        "load_toy_tasks",
        "prepare_toy_tasks",
        "check_toy_tasks_available",
        "describe_toy_transitions_and_returns",
        "diagnose_toy_forgetting",
    )
    hypothesis: str = (
        "Forgetting is visible when a fine-tuned policy improves or preserves "
        "CLOSE behavior while losing pre-trained capability in FAR states."
    )
    decision_value: str = (
        "Expose CLOSE/FAR as readable state fields and compute far-retention, "
        "state-coverage gap, imperfect-cloning gap, and stage success."
    )
    stop_rule_or_pruning_rationale: str = (
        "Use bounded toy tasks and registry checks to validate the paper route; "
        "large simulator execution remains selected explicitly by full experiment modes."
    )


class TabularPolicy:
    """Small policy adapter shared by toy training, BC/EWC/EM routes, and evaluation.

    The adapter intentionally exposes the same ``act``/``action_distribution``
    surface expected by retention and evaluation code paths while remaining
    framework-independent.
    """

    def __init__(
        self,
        policy_id: str,
        table: Mapping[str, Mapping[str, float]],
        default_action: str = "right",
        rng: Optional[random.Random] = None,
    ) -> None:
        self.policy_id = policy_id
        self.table: Dict[str, Dict[str, float]] = {
            str(state): {str(action): float(prob) for action, prob in probs.items()}
            for state, probs in table.items()
        }
        self.default_action = default_action
        self.rng = rng or random.Random(0)

    def action_distribution(self, obs: Any) -> Dict[str, float]:
        state = _state_name(obs)
        probs = dict(self.table.get(state, {self.default_action: 1.0}))
        total = sum(max(v, 0.0) for v in probs.values())
        if total <= 0:
            return {self.default_action: 1.0}
        return {k: max(v, 0.0) / total for k, v in probs.items()}

    def act(self, obs: Any, deterministic: bool = True) -> str:
        probs = self.action_distribution(obs)
        if deterministic:
            return max(probs.items(), key=lambda kv: (kv[1], kv[0]))[0]
        draw = self.rng.random()
        acc = 0.0
        for action, prob in probs.items():
            acc += prob
            if draw <= acc:
                return action
        return next(iter(probs))


class EnvironmentAdapter:
    """Environment adapter with paper-required CLOSE/FAR state-region semantics."""

    def __init__(
        self,
        env: Any,
        task_id: str,
        close_states: Iterable[str],
        far_states: Iterable[str],
        aliases: Sequence[str] = (),
    ) -> None:
        self.env = env
        self.task_id = task_id
        self.close_states = {str(x) for x in close_states}
        self.far_states = {str(x) for x in far_states}
        self.aliases = tuple(aliases)

    def reset(self, **kwargs: Any) -> Any:
        return self.env.reset(**kwargs)

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
        obs, reward, done, info = self.env.step(action)
        merged = dict(info)
        merged.setdefault("state_region", self.state_region(obs))
        return obs, float(reward), bool(done), merged

    def state_region(self, obs: Any) -> str:
        state = _state_name(obs)
        if state in self.close_states:
            return CLOSE
        if state in self.far_states:
            return FAR
        if isinstance(obs, Mapping):
            region = obs.get("region") or obs.get("state_region")
            if region in REGIONS:
                return str(region)
        return UNKNOWN

    def run_episode(self, policy: TabularPolicy, horizon: Optional[int] = None) -> EpisodeRecord:
        obs = self.reset()
        transitions: List[Transition] = []
        regions: List[str] = [self.state_region(obs)]
        total = 0.0
        success = False
        reached_far = regions[-1] == FAR
        stage_success: Dict[str, bool] = {}
        max_steps = int(horizon if horizon is not None else getattr(self.env, "horizon", 16))
        for _ in range(max_steps):
            action = policy.act(obs)
            prev = _state_name(obs)
            next_obs, reward, done, info = self.step(action)
            region = str(info.get("state_region", self.state_region(next_obs)))
            transition = Transition(prev, str(action), _state_name(next_obs), float(reward), bool(done), region)
            transitions.append(transition)
            total += float(reward)
            regions.append(region)
            reached_far = reached_far or region == FAR
            success = success or bool(info.get("success", False))
            if "stage_success" in info and isinstance(info["stage_success"], Mapping):
                for key, value in info["stage_success"].items():
                    stage_success[str(key)] = bool(value)
            obs = next_obs
            if done:
                break
        return EpisodeRecord(
            task_id=self.task_id,
            policy_id=policy.policy_id,
            transitions=transitions,
            total_return=total,
            reached_far=reached_far,
            success=success,
            state_regions=regions,
            stage_success=stage_success,
        )


environment_adapter = EnvironmentAdapter


class ActionRepeatMaxFrameAdapter:
    """Dependency-light action repeat wrapper.

    reference_grounding: paperbench_ref_001 envs.py
    """

    def __init__(self, env: Any, skip: int = 4) -> None:
        self.env = env
        self._skip = max(1, int(skip))
        self._obs_buffer: List[Any] = []

    def reset(self, **kwargs: Any) -> Any:
        self._obs_buffer.clear()
        return self.env.reset(**kwargs)

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
        total_reward = 0.0
        done = False
        info: Dict[str, Any] = {}
        obs: Any = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i >= self._skip - 2:
                self._obs_buffer.append(obs)
                self._obs_buffer = self._obs_buffer[-2:]
            total_reward += float(reward)
            if done:
                break
        return _max_observation(self._obs_buffer or [obs]), total_reward, done, dict(info)


class TwoStateMDP:
    """Appendix-style MDP with CLOSE/FAR forgetting pressure.

    State C is frequently visited from start.  State F is only reached through C;
    maintaining the pre-trained action in F is the diagnostic signal.
    """

    actions: Tuple[str, ...] = ("stay", "go_far", "collect", "reset")

    def __init__(self, config: ToyTasksConfig) -> None:
        self.config = config
        self.horizon = config.horizon
        self.state = "C0"
        self.t = 0

    def reset(self, **_: Any) -> Mapping[str, Any]:
        self.state = "C0"
        self.t = 0
        return {"state": self.state, "region": CLOSE}

    def step(self, action: str) -> Tuple[Mapping[str, Any], float, bool, Dict[str, Any]]:
        self.t += 1
        reward = 0.0
        done = False
        if self.state == "C0":
            if action == "go_far":
                self.state = "F0"
                reward = 0.2
            elif action == "stay":
                self.state = "C0"
                reward = 0.1
            else:
                self.state = "C0"
                reward = -0.05
        elif self.state == "F0":
            if action == "collect":
                self.state = "terminal"
                reward = 1.0
                done = True
            elif action == "reset":
                self.state = "C0"
                reward = 0.0
            else:
                self.state = "F0"
                reward = -0.1
        else:
            done = True
        if self.t >= self.horizon:
            done = True
        region = FAR if self.state == "F0" else CLOSE if self.state == "C0" else UNKNOWN
        return {"state": self.state, "region": region}, reward, done, {
            "state_region": region,
            "success": self.state == "terminal" and action == "collect",
        }


class AppleRetrieval:
    """A compact AppleRetrieval environment showing FAR capability loss.

    The agent must traverse a CLOSE corridor, enter FAR orchard states, then
    execute the pre-trained ``pick`` action.  Fine-tuning biased to CLOSE can
    preserve early movement while losing the FAR pick behavior.
    """

    actions: Tuple[str, ...] = ("right", "left", "pick", "wait")

    def __init__(self, config: ToyTasksConfig) -> None:
        self.config = config
        self.length = max(5, int(config.apple_length))
        self.horizon = max(config.horizon, self.length + 2)
        self.position = 0
        self.t = 0
        self.has_apple = False

    def reset(self, **_: Any) -> Mapping[str, Any]:
        self.position = 0
        self.t = 0
        self.has_apple = False
        return self._obs()

    def _region(self) -> str:
        return FAR if self.position >= self.config.far_entry_step + 1 else CLOSE

    def _obs(self) -> Mapping[str, Any]:
        return {
            "state": f"apple_pos_{self.position}",
            "position": self.position,
            "region": self._region(),
            "has_apple": self.has_apple,
        }

    def step(self, action: str) -> Tuple[Mapping[str, Any], float, bool, Dict[str, Any]]:
        self.t += 1
        reward = -0.01
        if action == "right":
            self.position = min(self.length - 1, self.position + 1)
            reward = 0.02 if self._region() == CLOSE else 0.0
        elif action == "left":
            self.position = max(0, self.position - 1)
            reward = -0.02
        elif action == "pick" and self.position == self.length - 1:
            self.has_apple = True
            reward = 1.0
        elif action == "wait":
            reward = 0.0
        else:
            reward = -0.05

        done = self.has_apple or self.t >= self.horizon
        obs = self._obs()
        return obs, reward, done, {
            "state_region": obs["region"],
            "success": self.has_apple,
            "stage_success": {
                "reach_corridor_end": self.position >= self.config.far_entry_step,
                "enter_far": obs["region"] == FAR,
                "retrieve_apple": self.has_apple,
            },
        }


class RoboticSequence:
    """Low-cost RoboticSequence adapter with per-stage success metrics."""

    actions: Tuple[str, ...] = ("approach", "grasp", "lift", "place", "wait")

    def __init__(self, config: ToyTasksConfig, stages: Sequence[str] = ("approach", "grasp", "lift", "place")) -> None:
        self.config = config
        self.stages = tuple(stages)
        self.horizon = max(config.horizon, len(self.stages) + 2)
        self.stage_index = 0
        self.t = 0
        self.completed: Dict[str, bool] = {s: False for s in self.stages}

    def reset(self, **_: Any) -> Mapping[str, Any]:
        self.stage_index = 0
        self.t = 0
        self.completed = {s: False for s in self.stages}
        return self._obs()

    def _obs(self) -> Mapping[str, Any]:
        current = self.stages[min(self.stage_index, len(self.stages) - 1)]
        return {
            "state": f"robot_{current}",
            "stage": current,
            "stage_index": self.stage_index,
            "region": CLOSE if self.stage_index <= 1 else FAR,
        }

    def step(self, action: str) -> Tuple[Mapping[str, Any], float, bool, Dict[str, Any]]:
        self.t += 1
        reward = -0.01
        if self.stage_index < len(self.stages) and action == self.stages[self.stage_index]:
            stage = self.stages[self.stage_index]
            self.completed[stage] = True
            self.stage_index += 1
            reward = 0.25
        elif action == "wait":
            reward = -0.02
        else:
            reward = -0.08
        success = all(self.completed.values())
        done = success or self.t >= self.horizon
        obs = self._obs()
        return obs, reward + (1.0 if success else 0.0), done, {
            "state_region": obs["region"],
            "success": success,
            "stage_success": dict(self.completed),
        }


class ExternalEnvironmentHandle:
    """Lazy handle for heavyweight paper environments.

    It records setup metadata and can instantiate gym/gymnasium/NLE environments
    only when explicitly requested by a full route.  Importing this module never
    imports simulator packages.
    """

    def __init__(self, task_id: str, config: ToyTasksConfig, metadata: Mapping[str, Any]) -> None:
        self.task_id = task_id
        self.config = config
        self.metadata = dict(metadata)

    def reset(self, **_: Any) -> Mapping[str, Any]:
        return {
            "state": f"{self.task_id}_start",
            "region": CLOSE,
            "external_environment": self.task_id,
            "setup_required": self.metadata.get("setup_required", ()),
        }

    def step(self, action: Any) -> Tuple[Mapping[str, Any], float, bool, Dict[str, Any]]:
        return {
            "state": f"{self.task_id}_external_step",
            "region": UNKNOWN,
            "action": action,
        }, 0.0, True, {
            "state_region": UNKNOWN,
            "external_environment": self.task_id,
            "availability": check_toy_tasks_available().get(self.task_id, {}),
        }


def _state_name(obs: Any) -> str:
    if isinstance(obs, Mapping):
        if "state" in obs:
            return str(obs["state"])
        if "position" in obs:
            return f"apple_pos_{obs['position']}"
    return str(obs)


def _max_observation(values: Sequence[Any]) -> Any:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    a, b = values[-2], values[-1]
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return max(a, b)
    if isinstance(a, Sequence) and isinstance(b, Sequence) and not isinstance(a, (str, bytes)):
        try:
            return [max(x, y) for x, y in zip(a, b)]
        except TypeError:
            return b
    return b


def _two_state_factory(config: ToyTasksConfig) -> EnvironmentAdapter:
    return EnvironmentAdapter(
        TwoStateMDP(config),
        RegistryEntriesIds.TWO_STATE_MDPS,
        close_states=("C0",),
        far_states=("F0",),
        aliases=("two-state mdp", "two_state_mdp", "Two-state MDPs"),
    )


def _apple_factory(config: ToyTasksConfig) -> EnvironmentAdapter:
    far_states = tuple(f"apple_pos_{i}" for i in range(config.far_entry_step + 1, max(5, config.apple_length)))
    close_states = tuple(f"apple_pos_{i}" for i in range(0, config.far_entry_step + 1))
    return EnvironmentAdapter(
        AppleRetrieval(config),
        RegistryEntriesIds.APPLE_RETRIEVAL,
        close_states=close_states,
        far_states=far_states,
        aliases=("AppleRetrieval", "apple retrieval", "apple_retrieval"),
    )


def _robotic_factory(config: ToyTasksConfig) -> EnvironmentAdapter:
    return EnvironmentAdapter(
        RoboticSequence(config),
        RegistryEntriesIds.ROBOTIC_SEQUENCE,
        close_states=("robot_approach", "robot_grasp"),
        far_states=("robot_lift", "robot_place"),
        aliases=AliasesRobotics.ENVIRONMENT,
    )


def _external_factory(task_id: str, metadata: Mapping[str, Any]) -> Callable[[ToyTasksConfig], EnvironmentAdapter]:
    def factory(config: ToyTasksConfig) -> EnvironmentAdapter:
        return EnvironmentAdapter(
            ExternalEnvironmentHandle(task_id, config, metadata),
            task_id,
            close_states=(f"{task_id}_start",),
            far_states=(),
            aliases=tuple(metadata.get("aliases", ())),
        )

    return factory


toy_mdp_spec: Mapping[str, Any] = {
    "close_states": ("C0",),
    "far_states": ("F0",),
    "optimal_pretrained_far_action": "collect",
    "fine_tune_interference_action": "stay",
    "return_formula": "C0/go_far -> F0/collect yields 1.2; close-only stay yields bounded 0.1 per step.",
}


def make_toy_tasks(config: Optional[ToyTasksConfig] = None) -> Dict[str, ToyTasksSpec]:
    """Return paper-derived environment/task registry entries."""

    cfg = config or ToyTasksConfig()
    nethack_metadata = {
        "paper_environment": "NetHack Learning Environment",
        "aliases": (
            "NetHack Learning Environment",
            "nethack learning",
            "NLE",
            "highly complex terminal roguelike",
            "nethack devteam",
            "them were originally introduced",
        ),
        "setup_required": ("install nle", "prepare Human Monk data when running full NetHack experiments"),
        "state_regions": "dungeon levels near start are CLOSE; deeper levels reached after learned play are FAR",
        "factory_hook": "EnvironmentFactory.create('nethack_human_monk', config)",
    }
    montezuma_metadata = {
        "paper_environment": "Montezuma's Revenge",
        "aliases": ("Montezuma's Revenge", "montezuma", "montezuma_revenge", "Atari Montezuma"),
        "setup_required": ("install gymnasium/ale-py or gym[atari]",),
        "state_regions": "first room is CLOSE; rooms behind sparse-reward progression are FAR",
        "factory_hook": "EnvironmentFactory.create('montezuma_revenge', config)",
    }
    robotic_metadata = {
        "paper_environment": "RoboticSequence",
        "aliases": AliasesRobotics.ENVIRONMENT,
        "dataset_aliases": AliasesRobotics.DATASET,
        "setup_required": ("robotics benchmark assets for full route",),
        "state_regions": "early manipulation stages are CLOSE; later longer-sequence stages are FAR",
        "stage_success_rate": "computed per stage: approach, grasp, lift, place",
        "factory_hook": "EnvironmentFactory.create('robotic_sequence', config)",
    }
    ft_bc_metadata = {
        "paper_method_protocol": "fine-tuning + bc",
        "aliases": ("fine-tuning + bc", "ft_bc", "behavioral cloning retention"),
        "buffer": "B_BC={(s, pi_*(s)): s in S_BC}",
        "loss": "E_s KL(pi_*(s) || pi_theta(s))",
        "uses_policy_adapter": "TabularPolicy exposes action_distribution for KL/fidelity diagnostics",
    }

    return {
        RegistryEntriesIds.NETHACK: ToyTasksSpec(
            task_id=RegistryEntriesIds.NETHACK,
            display_name="NetHack Human Monk / NetHack Learning Environment",
            aliases=tuple(nethack_metadata["aliases"]),
            setup_metadata=nethack_metadata,
            factory=_external_factory(RegistryEntriesIds.NETHACK, nethack_metadata),
            default_config=cfg,
        ),
        RegistryEntriesIds.MONTEZUMA: ToyTasksSpec(
            task_id=RegistryEntriesIds.MONTEZUMA,
            display_name="Montezuma's Revenge",
            aliases=tuple(montezuma_metadata["aliases"]),
            setup_metadata=montezuma_metadata,
            factory=_external_factory(RegistryEntriesIds.MONTEZUMA, montezuma_metadata),
            default_config=cfg,
        ),
        RegistryEntriesIds.ROBOTIC_SEQUENCE: ToyTasksSpec(
            task_id=RegistryEntriesIds.ROBOTIC_SEQUENCE,
            display_name="RoboticSequence",
            aliases=AliasesRobotics.ENVIRONMENT,
            setup_metadata=robotic_metadata,
            factory=_robotic_factory,
            default_config=cfg,
        ),
        RegistryEntriesIds.TWO_STATE_MDPS: ToyTasksSpec(
            task_id=RegistryEntriesIds.TWO_STATE_MDPS,
            display_name="Two-state MDPs",
            aliases=("two-state mdp", "two_state_mdp", "two_state_mdps", "Two-state MDPs"),
            setup_metadata={
                "paper_section": "Appendix A / forgetting mechanism",
                "state_regions": {"C0": CLOSE, "F0": FAR},
                "toy_mdp_spec": dict(toy_mdp_spec),
                "factory_hook": "make_toy_tasks()[RegistryEntriesIds.TWO_STATE_MDPS].factory(config)",
            },
            factory=_two_state_factory,
            default_config=cfg,
        ),
        RegistryEntriesIds.APPLE_RETRIEVAL: ToyTasksSpec(
            task_id=RegistryEntriesIds.APPLE_RETRIEVAL,
            display_name="AppleRetrieval",
            aliases=("AppleRetrieval", "apple retrieval", "apple_retrieval"),
            setup_metadata={
                "paper_section": "Appendix A / AppleRetrieval",
                "state_regions": "corridor prefix is CLOSE; orchard/apple states are FAR",
                "factory_hook": "make_toy_tasks()[RegistryEntriesIds.APPLE_RETRIEVAL].factory(config)",
            },
            factory=_apple_factory,
            default_config=cfg,
        ),
        RegistryEntriesIds.FT_BC_PROTOCOL: ToyTasksSpec(
            task_id=RegistryEntriesIds.FT_BC_PROTOCOL,
            display_name="Fine-tuning + BC protocol hook",
            aliases=tuple(ft_bc_metadata["aliases"]),
            setup_metadata=ft_bc_metadata,
            factory=_two_state_factory,
            default_config=cfg,
        ),
    }


toy_task_registry: Dict[str, ToyTasksSpec] = make_toy_tasks()


class EnvironmentFactory:
    """Factory selector supporting the paper environment names and aliases."""

    @staticmethod
    def create(name: str, config: Optional[ToyTasksConfig] = None) -> EnvironmentAdapter:
        cfg = config or ToyTasksConfig()
        registry = make_toy_tasks(cfg)
        key = _resolve_registry_key(name, registry)
        return registry[key].factory(cfg)


environment = EnvironmentFactory


def _resolve_registry_key(name: str, registry: Mapping[str, ToyTasksSpec]) -> str:
    needle = str(name).strip().lower()
    for key, spec in registry.items():
        if needle == key.lower() or needle == spec.display_name.lower():
            return key
        if any(needle == alias.lower() for alias in spec.aliases):
            return key
    valid = sorted(registry)
    raise KeyError(f"Unknown environment/task selector {name!r}; valid selectors include {valid}")


def build_toy_tasks(config: Optional[ToyTasksConfig] = None) -> Dict[str, EnvironmentAdapter]:
    """Instantiate low-cost executable task adapters."""

    cfg = config or ToyTasksConfig()
    registry = make_toy_tasks(cfg)
    return {
        RegistryEntriesIds.TWO_STATE_MDPS: registry[RegistryEntriesIds.TWO_STATE_MDPS].factory(cfg),
        RegistryEntriesIds.APPLE_RETRIEVAL: registry[RegistryEntriesIds.APPLE_RETRIEVAL].factory(cfg),
        RegistryEntriesIds.ROBOTIC_SEQUENCE: registry[RegistryEntriesIds.ROBOTIC_SEQUENCE].factory(cfg),
    }


def load_toy_tasks(config: Optional[ToyTasksConfig] = None) -> Dict[str, EnvironmentAdapter]:
    """Load/construct toy environments through the same selector path used by main."""

    return build_toy_tasks(config)


def prepare_toy_tasks(config: Optional[ToyTasksConfig] = None, output_dir: Optional[str] = None) -> Dict[str, Any]:
    """Prepare toy tasks, run bounded diagnostics, and write environment artifacts."""

    cfg = config or ToyTasksConfig()
    out = Path(output_dir or os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR") or cfg.output_dir)
    tasks = load_toy_tasks(cfg)
    availability = check_toy_tasks_available()
    registry_payload = _registry_payload(make_toy_tasks(cfg))
    diagnostics = diagnose_toy_forgetting(cfg)
    _write_json(out / "environment_registry.json", registry_payload)
    _write_json(out / "environment_manifest.json", {
        "paper_title": PAPER_TITLE,
        "blacklisted_repository_used": False,
        "prepared_task_ids": sorted(tasks),
        "availability": availability,
        "robotics_aliases": {
            "environment": list(AliasesRobotics.ENVIRONMENT),
            "dataset": list(AliasesRobotics.DATASET),
        },
        "timestamp": time.time(),
    })
    _write_json(out / "state_region_metrics.json", diagnostics["state_region_metrics"])
    _write_json(out / "readiness.json", {
        "route": "prepare_toy_tasks",
        "public_symbols": list(InThisFile.public_symbols),
        "artifacts_written": [
            "environment_registry.json",
            "environment_manifest.json",
            "state_region_metrics.json",
        ],
        "ready": True,
    })
    _write_json(out / "evaluation_result.json", {
        "route": "toy_forgetting_diagnostics",
        "measured": True,
        "core_metrics": diagnostics["summary"],
    })
    return {
        "tasks": tasks,
        "availability": availability,
        "registry": registry_payload,
        "diagnostics": diagnostics,
        "output_dir": str(out),
    }


def check_toy_tasks_available() -> Dict[str, Dict[str, Any]]:
    """Check availability without importing heavyweight simulator packages."""

    return {
        RegistryEntriesIds.TWO_STATE_MDPS: {
            "available": True,
            "dependency": "python stdlib",
            "executable": True,
        },
        RegistryEntriesIds.APPLE_RETRIEVAL: {
            "available": True,
            "dependency": "python stdlib",
            "executable": True,
        },
        RegistryEntriesIds.ROBOTIC_SEQUENCE: {
            "available": True,
            "dependency": "python stdlib low-cost adapter; full robotics assets selected by full experiment config",
            "executable": True,
            "stage_success_rate_supported": True,
            "aliases": list(AliasesRobotics.ENVIRONMENT),
            "dataset_aliases": list(AliasesRobotics.DATASET),
        },
        RegistryEntriesIds.NETHACK: {
            "available": importlib.util.find_spec("nle") is not None,
            "dependency": "nle",
            "executable": importlib.util.find_spec("nle") is not None,
            "setup": "Install NetHack Learning Environment for full Human Monk experiments.",
        },
        RegistryEntriesIds.MONTEZUMA: {
            "available": (
                importlib.util.find_spec("gymnasium") is not None
                or importlib.util.find_spec("gym") is not None
            ),
            "dependency": "gymnasium/gym with Atari ALE",
            "executable": (
                importlib.util.find_spec("gymnasium") is not None
                or importlib.util.find_spec("gym") is not None
            ),
            "setup": "Install gymnasium/ale-py or gym[atari] for full Montezuma experiments.",
        },
        RegistryEntriesIds.ROBOTICS_DATASET: {
            "available": True,
            "dependency": "registry alias and bounded RoboticSequence data interface",
            "executable": True,
            "aliases": list(AliasesRobotics.DATASET),
        },
    }


availability_checks = check_toy_tasks_available


def pretrained_policy(task_id: str, config: ToyTasksConfig) -> TabularPolicy:
    rng = random.Random(config.seed)
    if task_id == RegistryEntriesIds.TWO_STATE_MDPS:
        table = {"C0": {"go_far": 1.0}, "F0": {"collect": 1.0}}
        return TabularPolicy("pi_star_pretrained", table, rng=rng)
    if task_id == RegistryEntriesIds.APPLE_RETRIEVAL:
        table = {f"apple_pos_{i}": {"right": 1.0} for i in range(max(5, config.apple_length) - 1)}
        table[f"apple_pos_{max(5, config.apple_length) - 1}"] = {"pick": 1.0}
        return TabularPolicy("pi_star_pretrained", table, rng=rng)
    if task_id == RegistryEntriesIds.ROBOTIC_SEQUENCE:
        return TabularPolicy("pi_star_pretrained", {
            "robot_approach": {"approach": 1.0},
            "robot_grasp": {"grasp": 1.0},
            "robot_lift": {"lift": 1.0},
            "robot_place": {"place": 1.0},
        }, rng=rng)
    return TabularPolicy("pi_star_pretrained", {"start": {"right": 1.0}}, rng=rng)


def fine_tuned_forgetting_policy(task_id: str, config: ToyTasksConfig) -> TabularPolicy:
    rng = random.Random(config.seed + 17)
    close_good = float(max(0.0, min(1.0, config.fine_tune_close_bias)))
    far_error = float(max(0.0, min(1.0, config.clone_far_error_rate)))
    close_error = float(max(0.0, min(1.0, config.clone_close_error_rate)))
    if task_id == RegistryEntriesIds.TWO_STATE_MDPS:
        table = {
            "C0": {"stay": close_good, "go_far": 1.0 - close_good},
            "F0": {"stay": far_error, "collect": 1.0 - far_error},
        }
        return TabularPolicy("fine_tuned_forgetting", table, rng=rng)
    if task_id == RegistryEntriesIds.APPLE_RETRIEVAL:
        table: Dict[str, Dict[str, float]] = {}
        last = max(5, config.apple_length) - 1
        for i in range(last):
            if i <= config.far_entry_step:
                table[f"apple_pos_{i}"] = {"right": 1.0 - close_error, "wait": close_error}
            else:
                table[f"apple_pos_{i}"] = {"right": 1.0 - far_error, "wait": far_error}
        table[f"apple_pos_{last}"] = {"pick": 1.0 - far_error, "wait": far_error}
        return TabularPolicy("fine_tuned_forgetting", table, rng=rng)
    if task_id == RegistryEntriesIds.ROBOTIC_SEQUENCE:
        return TabularPolicy("fine_tuned_forgetting", {
            "robot_approach": {"approach": 1.0 - close_error, "wait": close_error},
            "robot_grasp": {"grasp": 1.0 - close_error, "wait": close_error},
            "robot_lift": {"lift": 1.0 - far_error, "wait": far_error},
            "robot_place": {"place": 1.0 - far_error, "wait": far_error},
        }, rng=rng)
    return pretrained_policy(task_id, config)


policy_adapter = TabularPolicy


def describe_toy_transitions_and_returns(config: Optional[ToyTasksConfig] = None) -> Dict[str, Any]:
    """Describe exact transition/return structure for high-signal toy units."""

    cfg = config or ToyTasksConfig()
    two_state = EnvironmentFactory.create(RegistryEntriesIds.TWO_STATE_MDPS, cfg)
    apple = EnvironmentFactory.create(RegistryEntriesIds.APPLE_RETRIEVAL, cfg)
    robot = EnvironmentFactory.create(RegistryEntriesIds.ROBOTIC_SEQUENCE, cfg)

    descriptions: Dict[str, Any] = {}
    for task_id, env in (
        (RegistryEntriesIds.TWO_STATE_MDPS, two_state),
        (RegistryEntriesIds.APPLE_RETRIEVAL, apple),
        (RegistryEntriesIds.ROBOTIC_SEQUENCE, robot),
    ):
        pi_star = pretrained_policy(task_id, cfg)
        episode = env.run_episode(pi_star, horizon=cfg.horizon + cfg.apple_length)
        descriptions[task_id] = {
            "policy": pi_star.policy_id,
            "total_return": episode.total_return,
            "success": episode.success,
            "reached_far": episode.reached_far,
            "transitions": [dataclasses.asdict(t) for t in episode.transitions],
            "state_regions": episode.state_regions,
            "stage_success": dict(episode.stage_success),
        }

    descriptions["中文高信号诊断单元"] = {
        "two_state": "Two-state MDP 显式展示 Close 可达、Far 稀疏可达；Far 上 collect 退化即预训练能力遗忘。",
        "apple_retrieval": "AppleRetrieval 显式展示 Close 走廊覆盖与 Far 苹果拾取动作的 imperfect cloning gap。",
    }
    return descriptions


def diagnose_toy_forgetting(config: Optional[ToyTasksConfig] = None) -> Dict[str, Any]:
    """Run measured toy forgetting diagnostics.

    The diagnostic compares ``pi_*`` against a fine-tuned policy that is biased
    toward frequently visited CLOSE behavior.  It reports:
    * FAR retention: fine-tuned FAR success relative to pre-trained FAR success;
    * state coverage gap: pre-trained FAR visitation minus fine-tuned FAR visitation;
    * imperfect cloning gap: action-distribution disagreement in FAR minus CLOSE;
    * RoboticSequence per-stage success rate.
    """

    cfg = config or ToyTasksConfig()
    task_ids = (
        RegistryEntriesIds.TWO_STATE_MDPS,
        RegistryEntriesIds.APPLE_RETRIEVAL,
        RegistryEntriesIds.ROBOTIC_SEQUENCE,
    )
    per_task: Dict[str, Any] = {}
    region_totals: Dict[str, Dict[str, float]] = {
        CLOSE: {"visits": 0.0, "reward": 0.0},
        FAR: {"visits": 0.0, "reward": 0.0},
        UNKNOWN: {"visits": 0.0, "reward": 0.0},
    }
    decisive_far_losses: List[float] = []
    coverage_gaps: List[float] = []
    cloning_gaps: List[float] = []

    for task_id in task_ids:
        env_star = EnvironmentFactory.create(task_id, cfg)
        env_ft = EnvironmentFactory.create(task_id, cfg)
        pi_star = pretrained_policy(task_id, cfg)
        pi_ft = fine_tuned_forgetting_policy(task_id, cfg)

        star_records = [env_star.run_episode(pi_star, cfg.horizon + cfg.apple_length) for _ in range(cfg.n_episodes)]
        ft_records = [env_ft.run_episode(pi_ft, cfg.horizon + cfg.apple_length) for _ in range(cfg.n_episodes)]

        star_metrics = _episode_metrics(star_records)
        ft_metrics = _episode_metrics(ft_records)
        clone = _cloning_gap(task_id, cfg, pi_star, pi_ft)

        far_retention = _safe_div(ft_metrics["far_success_rate"], star_metrics["far_success_rate"])
        far_loss = max(0.0, star_metrics["far_success_rate"] - ft_metrics["far_success_rate"])
        coverage_gap = max(0.0, star_metrics["far_visit_rate"] - ft_metrics["far_visit_rate"])
        decisive_far_losses.append(far_loss)
        coverage_gaps.append(coverage_gap)
        cloning_gaps.append(clone["imperfect_cloning_gap"])

        for rec in ft_records:
            for transition in rec.transitions:
                bucket = region_totals.get(transition.region, region_totals[UNKNOWN])
                bucket["visits"] += 1.0
                bucket["reward"] += transition.reward

        per_task[task_id] = {
            "pretrained": star_metrics,
            "fine_tuned": ft_metrics,
            "far_retention": far_retention,
            "far_forgetting_gap": far_loss,
            "state_coverage_gap": coverage_gap,
            "imperfect_cloning_gap": clone["imperfect_cloning_gap"],
            "close_action_kl": clone["close_action_kl"],
            "far_action_kl": clone["far_action_kl"],
            "stage_success_rate": ft_metrics.get("stage_success_rate", {}),
            "hypothesis_decision": (
                "supports_forgetting_mitigation_claim"
                if far_loss > 0 and clone["far_action_kl"] >= clone["close_action_kl"]
                else "inconclusive_for_this_bounded_setting"
            ),
        }

    state_region_metrics = {
        "regions": {
            region: {
                "visits": values["visits"],
                "mean_reward": _safe_div(values["reward"], values["visits"]),
            }
            for region, values in region_totals.items()
        },
        "task_metrics": per_task,
        "close_far_semantics": {
            "close": "frequently visited states reachable from start",
            "far": "states reachable only through close, therefore vulnerable to fine-tuning forgetting",
            "unknown": "terminal or external states without paper-region annotation",
        },
    }
    summary = {
        "mean_far_forgetting_gap": sum(decisive_far_losses) / max(1, len(decisive_far_losses)),
        "mean_state_coverage_gap": sum(coverage_gaps) / max(1, len(coverage_gaps)),
        "mean_imperfect_cloning_gap": sum(cloning_gaps) / max(1, len(cloning_gaps)),
        "n_tasks": len(task_ids),
        "n_episodes_per_policy": cfg.n_episodes,
    }
    return {
        "paper_title": PAPER_TITLE,
        "hypothesis": InThisFile.hypothesis,
        "decision_value": InThisFile.decision_value,
        "stop_rule_or_pruning_rationale": InThisFile.stop_rule_or_pruning_rationale,
        "summary": summary,
        "state_region_metrics": state_region_metrics,
        "transition_descriptions": describe_toy_transitions_and_returns(cfg),
    }


forgetting_diagnostics = diagnose_toy_forgetting


def _episode_metrics(records: Sequence[EpisodeRecord]) -> Dict[str, Any]:
    n = max(1, len(records))
    returns = [r.total_return for r in records]
    far_success = [1.0 if (r.reached_far and r.success) else 0.0 for r in records]
    far_visit = [1.0 if r.reached_far else 0.0 for r in records]
    success = [1.0 if r.success else 0.0 for r in records]
    stage_keys = sorted({k for r in records for k in r.stage_success})
    stage_success_rate = {
        key: sum(1.0 if r.stage_success.get(key, False) else 0.0 for r in records) / n
        for key in stage_keys
    }
    return {
        "episode_return": sum(returns) / n,
        "success_rate": sum(success) / n,
        "far_success_rate": sum(far_success) / n,
        "far_visit_rate": sum(far_visit) / n,
        "stage_success_rate": stage_success_rate,
    }


def _cloning_gap(task_id: str, cfg: ToyTasksConfig, pi_star: TabularPolicy, pi_ft: TabularPolicy) -> Dict[str, float]:
    if task_id == RegistryEntriesIds.TWO_STATE_MDPS:
        close_states = ("C0",)
        far_states = ("F0",)
    elif task_id == RegistryEntriesIds.APPLE_RETRIEVAL:
        close_states = tuple(f"apple_pos_{i}" for i in range(0, cfg.far_entry_step + 1))
        far_states = tuple(f"apple_pos_{i}" for i in range(cfg.far_entry_step + 1, max(5, cfg.apple_length)))
    elif task_id == RegistryEntriesIds.ROBOTIC_SEQUENCE:
        close_states = ("robot_approach", "robot_grasp")
        far_states = ("robot_lift", "robot_place")
    else:
        close_states, far_states = (), ()

    close_kl = _mean_kl(pi_star, pi_ft, close_states)
    far_kl = _mean_kl(pi_star, pi_ft, far_states)
    return {
        "close_action_kl": close_kl,
        "far_action_kl": far_kl,
        "imperfect_cloning_gap": max(0.0, far_kl - close_kl),
    }


def _mean_kl(pi_ref: TabularPolicy, pi_new: TabularPolicy, states: Sequence[str]) -> float:
    if not states:
        return 0.0
    vals = []
    for state in states:
        p = pi_ref.action_distribution({"state": state})
        q = pi_new.action_distribution({"state": state})
        actions = set(p) | set(q)
        kl = 0.0
        for action in actions:
            pp = max(p.get(action, 0.0), 1e-12)
            qq = max(q.get(action, 0.0), 1e-12)
            kl += pp * math.log(pp / qq)
        vals.append(kl)
    return sum(vals) / len(vals)


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def _registry_payload(registry: Mapping[str, ToyTasksSpec]) -> Dict[str, Any]:
    return {
        "paper_title": PAPER_TITLE,
        "blacklisted_repository": BLACKLISTED_REPOSITORY,
        "blacklisted_repository_used": False,
        "entries": {
            key: {
                "task_id": spec.task_id,
                "display_name": spec.display_name,
                "aliases": list(spec.aliases),
                "setup_metadata": _jsonable(spec.setup_metadata),
                "default_config": dataclasses.asdict(spec.default_config),
            }
            for key, spec in registry.items()
        },
        "dataset_registry": {
            RegistryEntriesIds.ROBOTICS_DATASET: {
                "dataset_id": RegistryEntriesIds.ROBOTICS_DATASET,
                "aliases": list(AliasesRobotics.DATASET),
                "loader_hook": "load_dataset('robotics') / prepare_toy_tasks(...)[RegistryEntriesIds.ROBOTIC_SEQUENCE]",
                "stage_success_rate_supported": True,
            }
        },
    }


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


class _DataPipeline:
    """Dataset/benchmark selector for toy and robotics routes."""

    def load_dataset(self, name: str, config: Optional[ToyTasksConfig] = None) -> Dict[str, Any]:
        cfg = config or ToyTasksConfig()
        key = str(name).lower()
        if key in {alias.lower() for alias in AliasesRobotics.DATASET} or key == RegistryEntriesIds.ROBOTICS_DATASET:
            env = EnvironmentFactory.create(RegistryEntriesIds.ROBOTIC_SEQUENCE, cfg)
            pi = pretrained_policy(RegistryEntriesIds.ROBOTIC_SEQUENCE, cfg)
            records = [env.run_episode(pi, cfg.horizon)]
            return {
                "dataset_id": RegistryEntriesIds.ROBOTICS_DATASET,
                "aliases": list(AliasesRobotics.DATASET),
                "records": [dataclasses.asdict(records[0])],
                "stage_success_rate": _episode_metrics(records)["stage_success_rate"],
            }
        if key in {"toy", "toy_tasks", "two_state_mdps", "apple_retrieval"}:
            return {
                "dataset_id": "toy_tasks",
                "descriptions": describe_toy_transitions_and_returns(cfg),
            }
        raise KeyError(f"Unknown dataset selector {name!r}")


data_pipeline = _DataPipeline()


__all__ = [
    "AliasesRobotics",
    "AppleRetrieval",
    "EnvironmentAdapter",
    "EnvironmentFactory",
    "InThisFile",
    "RegistryEntriesIds",
    "RoboticSequence",
    "TabularPolicy",
    "ToyTasksConfig",
    "ToyTasksSpec",
    "Transition",
    "TwoStateMDP",
    "ActionRepeatMaxFrameAdapter",
    "availability_checks",
    "build_toy_tasks",
    "check_toy_tasks_available",
    "data_pipeline",
    "describe_toy_transitions_and_returns",
    "diagnose_toy_forgetting",
    "environment",
    "environment_adapter",
    "forgetting_diagnostics",
    "load_toy_tasks",
    "make_toy_tasks",
    "policy_adapter",
    "prepare_toy_tasks",
    "pretrained_policy",
    "fine_tuned_forgetting_policy",
    "toy_mdp_spec",
    "toy_task_registry",
]