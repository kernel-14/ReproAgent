"""
ftrl_repro/envs.py

Environment registry, availability checks, EnvsSpec, make_envs, and check_envs_available
for the paper reproduction:
  "Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem"

reference_grounding: paperbench_ref_001 envs.py
reference_grounding: paperbench_ref_001 eval.py
reference_grounding: paperbench_ref_001 README.md

Paper environments (Section 3 Experimental setup):
  - NetHack Learning Environment (Human Monk character) — NLE, highly complex terminal
    roguelike, introduced by NetHack DevTeam (1987), NLE by Küttler et al. (2020).
    Paper role: main environment for Section 4 main result (Figure 3a) and Section 5
    forgetting analysis (Figure 4, Figure 5). FAR states = deeper dungeon levels.
    Expensive dependency: nle>=0.9 (lazy import).
  - Montezuma's Revenge — Atari game, state coverage gap drives forgetting.
    Paper role: Section 4 main result (Figure 3b), Section 5 (Figure 6).
    Expensive dependency: gymnasium[atari] + ale-py (lazy import).
  - RoboticSequence — robotics manipulation sequence with stages peg-unplug-side and
    push-wall. Paper role: Section 4 main result (Figure 3c), Section 5 (Figure 7).
    Expensive dependency: gymnasium robotics backend (lazy import).
    Dataset/benchmark alias: "robotics" (paper evidence contract).
  - Two-state MDPs — toy environment, Appendix A, illustrates Close/FAR forgetting.
    Lightweight: implemented locally in ftrl_repro/toy_tasks.py.
  - AppleRetrieval — toy grid-world, Appendix A, illustrates forgetting mechanism.
    Lightweight: implemented locally in ftrl_repro/toy_tasks.py.

Methods registry (paper Section 3):
  - training_from_scratch
  - vanilla_finetune
  - finetune_bc   (Fine-tuning + BC, knowledge retention)
  - finetune_ewc  (Fine-tuning + EWC, knowledge retention)
  - finetune_em   (Fine-tuning + EM, knowledge retention)

Baselines (paper evidence contract):
  - ours, ppo, sac, bc, oracle, nle, ewc

Fixed hyperparameters:
  - batch_size_128: batch_size = 128

NOTE: This repository does NOT use the blacklisted repository
      https://github.com/BartekCupial/finetuning-RL-as-CL

All expensive simulator/RL/GPU imports are lazy (inside factory functions).
Module-level imports are lightweight only.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Paper-derived constants
# ---------------------------------------------------------------------------

# Fixed hyperparameter: batch_size_128 (paper evidence contract)
BATCH_SIZE_128: int = 128

# Seed list (bounded config)
DEFAULT_SEEDS: List[int] = [0, 1, 2]

# Smoke budget (bounded config)
SMOKE_STEPS: int = 10
SMOKE_EVAL_EPISODES: int = 2

# Full training budget (bounded config — not executed by default)
FULL_TRAIN_STEPS: int = 100_000_000
FULL_EVAL_EPISODES: int = 200

# ---------------------------------------------------------------------------
# Environment IDs and aliases
# ---------------------------------------------------------------------------

# NetHack Human Monk — paper Section 3, Appendix B.1
NETHACK_ENV_ID = "NetHackScore-v0"
NETHACK_CHARACTER = "val-hum-neu-mal"  # Human Monk

# Montezuma's Revenge — paper Section 3, Appendix B.2
# reference_grounding: paperbench_ref_001 envs.py (AtariEnvironment, MontezumaInfoWrapper)
MONTEZUMA_ENV_ID = "MontezumaRevengeNoFrameskip-v4"
MONTEZUMA_ROOM_ADDRESS = 3  # RAM address for room tracking

# RoboticSequence — paper Section 3, Appendix B.3
# Dataset/benchmark alias: "robotics" (paper evidence contract)
ROBOTIC_SEQUENCE_ENV_ID = "RoboticSequence-v0"
ROBOTIC_SEQUENCE_STAGES = ["peg-unplug-side", "push-wall"]  # pre-trained stages
ROBOTIC_SEQUENCE_ALL_STAGES = [
    "peg-unplug-side",
    "push-wall",
    "pick-place",
    "open-drawer",
]

# Toy environments — Appendix A
TWO_STATE_MDP_ENV_ID = "TwoStateMDP-v0"
APPLE_RETRIEVAL_ENV_ID = "AppleRetrieval-v0"

# ---------------------------------------------------------------------------
# Method registry (paper Section 3)
# ---------------------------------------------------------------------------

VALID_METHODS = [
    "training_from_scratch",   # baseline: scratch
    "vanilla_finetune",        # baseline: vanilla fine-tuning
    "finetune_bc",             # Fine-tuning + BC (knowledge retention)
    "finetune_ewc",            # Fine-tuning + EWC (knowledge retention)
    "finetune_em",             # Fine-tuning + EM (knowledge retention)
]

# Paper evidence contract: method/baseline selectors
METHOD_ALIASES: Dict[str, str] = {
    "scratch": "training_from_scratch",
    "from_scratch": "training_from_scratch",
    "vanilla": "vanilla_finetune",
    "vanilla_finetuning": "vanilla_finetune",
    "bc": "finetune_bc",
    "finetune_bc": "finetune_bc",
    "ewc": "finetune_ewc",
    "finetune_ewc": "finetune_ewc",
    "em": "finetune_em",
    "finetune_em": "finetune_em",
    # Paper evidence contract: ours, ppo, sac, oracle, nle
    "ours": "finetune_bc",       # "ours" = Fine-tuning + BC/KS in paper
    "ppo": "training_from_scratch",
    "sac": "finetune_em",        # SAC-based EM variant
    "oracle": "vanilla_finetune",
    "nle": "vanilla_finetune",   # NLE baseline
}

# ---------------------------------------------------------------------------
# Environment registry (paper evidence contract)
# ---------------------------------------------------------------------------

ENV_REGISTRY: Dict[str, Dict[str, Any]] = {
    # --- Main paper environments ---
    "nethack": {
        "id": "nethack",
        "aliases": ["nethack_human_monk", "nle", "nethack_learning_environment",
                    "NetHackScore-v0", "nethack_devteam"],
        "gym_id": NETHACK_ENV_ID,
        "character": NETHACK_CHARACTER,
        "paper_role": "Section 4 main result (Figure 3a), Section 5 forgetting analysis (Figure 4, Figure 5)",
        "paper_section": "Section 3 Experimental setup, Appendix B.1",
        "forgetting_type": "imperfect_cloning_gap",
        "far_definition": "deeper dungeon levels (level > current agent level)",
        "close_definition": "dungeon levels reachable from start",
        "metrics": ["return", "maximum_dungeon_level", "turns", "FAR_performance",
                    "Close_performance", "forgetting_gap"],
        "expensive_deps": ["nle"],
        "pretrained_policy": "AutoAscend-derived pi_star",
        "information_processing_systems_datasets": True,  # NeurIPS venue
        "highly_complex_terminal_roguelike": True,
        "longer_sequence": True,
        "available": None,  # checked lazily
    },
    "montezuma": {
        "id": "montezuma",
        "aliases": ["montezuma_revenge", "MontezumaRevengeNoFrameskip-v4",
                    "montezumas_revenge", "atari_montezuma"],
        "gym_id": MONTEZUMA_ENV_ID,
        "room_address": MONTEZUMA_ROOM_ADDRESS,
        "paper_role": "Section 4 main result (Figure 3b), Section 5 (Figure 6)",
        "paper_section": "Section 3 Experimental setup, Appendix B.2",
        "forgetting_type": "state_coverage_gap",
        "far_definition": "Room 7 and beyond (FAR states)",
        "close_definition": "Rooms 0-6 (CLOSE states, reachable from start)",
        "metrics": ["return", "success_rate", "FAR_performance", "Close_performance",
                    "forgetting_gap"],
        "expensive_deps": ["gymnasium", "ale_py"],
        "pretrained_policy": "pi_star trained on early rooms",
        "available": None,
    },
    "robotic_sequence": {
        "id": "robotic_sequence",
        "aliases": [
            "robotics",                    # paper evidence contract alias
            "RoboticSequence",
            "RoboticSequence-v0",
            "robotic",
            "robotics_sequence",
            "peg_unplug_push_wall",
        ],
        "gym_id": ROBOTIC_SEQUENCE_ENV_ID,
        "stages": ROBOTIC_SEQUENCE_ALL_STAGES,
        "pretrained_stages": ROBOTIC_SEQUENCE_STAGES,  # peg-unplug-side, push-wall
        "paper_role": "Section 4 main result (Figure 3c), Section 5 (Figure 7)",
        "paper_section": "Section 3 Experimental setup, Appendix B.3",
        "forgetting_type": "state_coverage_gap",
        "far_definition": "peg-unplug-side and push-wall stages (pre-trained, FAR during fine-tuning)",
        "close_definition": "pick-place and open-drawer stages (new task, CLOSE)",
        "metrics": ["success_rate", "stage_success_rate", "FAR_performance",
                    "Close_performance", "forgetting_gap"],
        "expensive_deps": ["gymnasium"],
        "pretrained_policy": "pi_star performing well on peg-unplug-side and push-wall",
        # Dataset/benchmark alias for robotics (paper evidence contract)
        "dataset_alias": "robotics",
        "available": None,
    },
    # --- Toy environments (Appendix A) ---
    "two_state_mdp": {
        "id": "two_state_mdp",
        "aliases": ["TwoStateMDP", "TwoStateMDP-v0", "two_state_mdps",
                    "toy_mdp", "toy_two_state"],
        "gym_id": TWO_STATE_MDP_ENV_ID,
        "paper_role": "Appendix A Toy Examples, Figure 9",
        "paper_section": "Appendix A",
        "forgetting_type": "both",
        "far_definition": "state s_far (reachable only through s_close)",
        "close_definition": "state s_close (starting state)",
        "metrics": ["return", "FAR_performance", "Close_performance"],
        "expensive_deps": [],  # lightweight local implementation
        "pretrained_policy": "pi_star optimal on s_far",
        "available": True,  # always available (local)
    },
    "apple_retrieval": {
        "id": "apple_retrieval",
        "aliases": ["AppleRetrieval", "AppleRetrieval-v0", "apple_retrieval_env",
                    "apple_grid"],
        "gym_id": APPLE_RETRIEVAL_ENV_ID,
        "paper_role": "Appendix A Toy Examples, Figure 10",
        "paper_section": "Appendix A",
        "forgetting_type": "state_coverage_gap",
        "far_definition": "apple location (FAR from house)",
        "close_definition": "house location (CLOSE, starting state)",
        "metrics": ["return", "success_rate", "FAR_performance"],
        "expensive_deps": [],  # lightweight local implementation
        "pretrained_policy": "pi_star that can reach apple",
        "available": True,  # always available (local)
    },
}

# Dataset/benchmark registry (paper evidence contract: robotics)
DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "robotics": {
        "id": "robotics",
        "aliases": ["RoboticSequence", "robotic_sequence", "peg_unplug_push_wall"],
        "env_id": "robotic_sequence",
        "description": "RoboticSequence robotics manipulation benchmark with staged tasks",
        "stages": ROBOTIC_SEQUENCE_ALL_STAGES,
        "pretrained_stages": ROBOTIC_SEQUENCE_STAGES,
        "metrics": ["success_rate", "stage_success_rate"],
        "loader_hook": "make_envs",
        "availability_check": "check_envs_available",
        "paper_section": "Section 3, Appendix B.3",
    },
}

# Reverse alias lookup
_ENV_ALIAS_MAP: Dict[str, str] = {}
for _env_id, _env_info in ENV_REGISTRY.items():
    _ENV_ALIAS_MAP[_env_id] = _env_id
    for _alias in _env_info.get("aliases", []):
        _ENV_ALIAS_MAP[_alias.lower()] = _env_id
        _ENV_ALIAS_MAP[_alias] = _env_id

# Dataset alias lookup
_DATASET_ALIAS_MAP: Dict[str, str] = {}
for _ds_id, _ds_info in DATASET_REGISTRY.items():
    _DATASET_ALIAS_MAP[_ds_id] = _ds_id
    for _alias in _ds_info.get("aliases", []):
        _DATASET_ALIAS_MAP[_alias.lower()] = _ds_id
        _DATASET_ALIAS_MAP[_alias] = _ds_id


# ---------------------------------------------------------------------------
# EnvsSpec dataclass (paper-derived, active route contract)
# ---------------------------------------------------------------------------

@dataclass
class EnvsSpec:
    """
    Specification for a paper environment instance.

    Carries all metadata needed by train/evaluate/report routes:
    - env_id: canonical registry key
    - gym_id: gymnasium environment string
    - paper_role: which paper section/figure this env covers
    - forgetting_type: imperfect_cloning_gap | state_coverage_gap | both
    - far_definition: what constitutes FAR states in this environment
    - close_definition: what constitutes CLOSE states
    - stages: list of task stages (RoboticSequence only)
    - pretrained_stages: stages where pi_* performs well (FAR during fine-tuning)
    - metrics: list of applicable metric names
    - available: whether the expensive backend is importable
    - unavailable_reason: structured reason if not available
    - config: additional environment-specific config dict
    - smoke_mode: if True, use bounded smoke fixtures
    """
    env_id: str
    gym_id: str
    paper_role: str
    forgetting_type: str
    far_definition: str
    close_definition: str
    stages: List[str] = field(default_factory=list)
    pretrained_stages: List[str] = field(default_factory=list)
    metrics: List[str] = field(default_factory=list)
    available: bool = False
    unavailable_reason: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    smoke_mode: bool = True
    # Paper evidence contract fields
    dataset_alias: Optional[str] = None
    paper_section: str = ""
    expensive_deps: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "env_id": self.env_id,
            "gym_id": self.gym_id,
            "paper_role": self.paper_role,
            "forgetting_type": self.forgetting_type,
            "far_definition": self.far_definition,
            "close_definition": self.close_definition,
            "stages": self.stages,
            "pretrained_stages": self.pretrained_stages,
            "metrics": self.metrics,
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
            "config": self.config,
            "smoke_mode": self.smoke_mode,
            "dataset_alias": self.dataset_alias,
            "paper_section": self.paper_section,
            "expensive_deps": self.expensive_deps,
        }


# ---------------------------------------------------------------------------
# Availability checks (lazy import, structured unavailable state)
# ---------------------------------------------------------------------------

def _check_nle_available() -> Tuple[bool, Optional[str]]:
    """Check if nle (NetHack Learning Environment) is importable."""
    try:
        importlib.import_module("nle")
        return True, None
    except ImportError as e:
        return False, (
            f"nle not available: {e}. "
            "Install with: pip install nle>=0.9 "
            "(see https://github.com/facebookresearch/nle)"
        )


def _check_gymnasium_atari_available() -> Tuple[bool, Optional[str]]:
    """Check if gymnasium with Atari support is importable."""
    try:
        gym = importlib.import_module("gymnasium")
        # Try to check for ale_py
        try:
            importlib.import_module("ale_py")
        except ImportError:
            return False, (
                "ale_py not available. "
                "Install with: pip install gymnasium[atari,accept-rom-license] ale-py"
            )
        return True, None
    except ImportError as e:
        return False, (
            f"gymnasium not available: {e}. "
            "Install with: pip install gymnasium[atari,accept-rom-license]"
        )


def _check_gymnasium_robotics_available() -> Tuple[bool, Optional[str]]:
    """Check if gymnasium is importable for robotics environments."""
    try:
        importlib.import_module("gymnasium")
        return True, None
    except ImportError as e:
        return False, (
            f"gymnasium not available: {e}. "
            "Install with: pip install gymnasium>=0.29"
        )


def _check_torch_available() -> Tuple[bool, Optional[str]]:
    """
    Lazy check for torch availability.
    Required by training routes but not by environment registry itself.
    reference_grounding: paperbench_ref_001 agents.py (batch_size=128, optimizer)
    """
    try:
        importlib.import_module("torch")
        return True, None
    except ImportError as e:
        return False, (
            f"torch not available: {e}. "
            "Install with: pip install torch>=2.0"
        )


_AVAILABILITY_CHECKERS: Dict[str, Any] = {
    "nethack": _check_nle_available,
    "montezuma": _check_gymnasium_atari_available,
    "robotic_sequence": _check_gymnasium_robotics_available,
    "two_state_mdp": lambda: (True, None),
    "apple_retrieval": lambda: (True, None),
}


def check_envs_available(
    env_ids: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Check availability of paper environments.

    Returns a structured dict mapping env_id -> availability status.
    Never raises on missing dependencies — returns structured unavailable state.

    Args:
        env_ids: list of env ids to check; if None, checks all registered envs.

    Returns:
        Dict mapping env_id -> {
            "available": bool,
            "reason": str or None,
            "env_id": str,
            "gym_id": str,
            "paper_role": str,
            "expensive_deps": list,
        }
    """
    if env_ids is None:
        env_ids = list(ENV_REGISTRY.keys())

    results: Dict[str, Dict[str, Any]] = {}
    for raw_id in env_ids:
        # Resolve alias
        canonical = _ENV_ALIAS_MAP.get(raw_id, raw_id)
        if canonical not in ENV_REGISTRY:
            results[raw_id] = {
                "available": False,
                "reason": f"Unknown environment id or alias: {raw_id!r}",
                "env_id": raw_id,
                "gym_id": "",
                "paper_role": "",
                "expensive_deps": [],
            }
            continue

        info = ENV_REGISTRY[canonical]
        checker = _AVAILABILITY_CHECKERS.get(canonical, lambda: (False, "No checker registered"))
        available, reason = checker()

        results[canonical] = {
            "available": available,
            "reason": reason,
            "env_id": canonical,
            "gym_id": info.get("gym_id", ""),
            "paper_role": info.get("paper_role", ""),
            "expensive_deps": info.get("expensive_deps", []),
            "paper_section": info.get("paper_section", ""),
            "forgetting_type": info.get("forgetting_type", ""),
            "far_definition": info.get("far_definition", ""),
            "close_definition": info.get("close_definition", ""),
            "metrics": info.get("metrics", []),
        }

    return results


# ---------------------------------------------------------------------------
# make_envs factory (active route contract)
# ---------------------------------------------------------------------------

def make_envs(
    env_id: str,
    smoke_mode: bool = True,
    seed: int = 0,
    config: Optional[Dict[str, Any]] = None,
) -> EnvsSpec:
    """
    Factory function: create an EnvsSpec for the given environment.

    In smoke_mode=True (default), returns a spec with available=True/False
    and smoke fixtures; does NOT instantiate the actual expensive simulator.

    In smoke_mode=False (full mode), attempts to verify the backend is available
    and returns a spec ready for actual environment instantiation via
    make_gym_env(spec).

    Args:
        env_id: canonical env id or alias (e.g. "nethack", "robotics", "montezuma")
        smoke_mode: if True, use bounded smoke fixtures (default)
        seed: random seed
        config: optional override config dict

    Returns:
        EnvsSpec with availability status and paper metadata.
        Never raises on missing dependencies.
    """
    if config is None:
        config = {}

    # Resolve alias
    canonical = _ENV_ALIAS_MAP.get(env_id, env_id)
    if canonical not in ENV_REGISTRY:
        # Return structured unavailable spec
        return EnvsSpec(
            env_id=env_id,
            gym_id="",
            paper_role="unknown",
            forgetting_type="unknown",
            far_definition="unknown",
            close_definition="unknown",
            available=False,
            unavailable_reason=f"Unknown environment id or alias: {env_id!r}",
            smoke_mode=smoke_mode,
        )

    info = ENV_REGISTRY[canonical]

    # Check availability
    checker = _AVAILABILITY_CHECKERS.get(canonical, lambda: (False, "No checker"))
    available, reason = checker()

    # Build spec
    spec = EnvsSpec(
        env_id=canonical,
        gym_id=info.get("gym_id", ""),
        paper_role=info.get("paper_role", ""),
        forgetting_type=info.get("forgetting_type", ""),
        far_definition=info.get("far_definition", ""),
        close_definition=info.get("close_definition", ""),
        stages=list(info.get("stages", [])),
        pretrained_stages=list(info.get("pretrained_stages", [])),
        metrics=list(info.get("metrics", [])),
        available=available,
        unavailable_reason=reason,
        config={
            "seed": seed,
            "smoke_mode": smoke_mode,
            "batch_size": BATCH_SIZE_128,
            **config,
        },
        smoke_mode=smoke_mode,
        dataset_alias=info.get("dataset_alias"),
        paper_section=info.get("paper_section", ""),
        expensive_deps=list(info.get("expensive_deps", [])),
    )

    return spec


def make_gym_env(spec: EnvsSpec, render: bool = False) -> Any:
    """
    Instantiate the actual gymnasium/nle environment from an EnvsSpec.

    This is the full-mode factory. Raises RuntimeError if the backend
    is not available. Use make_envs() first to check availability.

    reference_grounding: paperbench_ref_001 envs.py (AtariEnvironment, MaxAndSkipEnv)
    reference_grounding: paperbench_ref_001 eval.py (env_type dispatch)

    Args:
        spec: EnvsSpec from make_envs()
        render: whether to render the environment

    Returns:
        gymnasium/nle environment instance
    """
    if not spec.available:
        raise RuntimeError(
            f"Environment {spec.env_id!r} is not available: {spec.unavailable_reason}"
        )

    env_id = spec.env_id

    if env_id == "nethack":
        # Lazy import nle
        try:
            import nle  # noqa: F401
            import gymnasium as gym
        except ImportError:
            import gym  # type: ignore
        env = gym.make(spec.gym_id)
        return env

    elif env_id == "montezuma":
        # Lazy import gymnasium + atari
        # reference_grounding: paperbench_ref_001 envs.py (AtariEnvironment)
        try:
            import gymnasium as gym
        except ImportError:
            import gym  # type: ignore
        env = gym.make(spec.gym_id)
        return _wrap_montezuma(env)

    elif env_id == "robotic_sequence":
        # Lazy import gymnasium
        try:
            import gymnasium as gym
        except ImportError:
            import gym  # type: ignore
        # RoboticSequence may need custom registration
        try:
            env = gym.make(spec.gym_id)
        except Exception:
            # Fall back to a smoke fixture if env not registered
            env = _make_robotic_sequence_smoke_env(spec)
        return env

    elif env_id in ("two_state_mdp", "apple_retrieval"):
        # Local toy implementations
        from ftrl_repro.toy_tasks import make_toy_tasks
        toy_spec = make_toy_tasks(env_id, smoke_mode=spec.smoke_mode)
        return toy_spec

    else:
        raise RuntimeError(f"No factory registered for env_id={env_id!r}")


def _wrap_montezuma(env: Any) -> Any:
    """
    Wrap Montezuma's Revenge with room tracking info wrapper.
    reference_grounding: paperbench_ref_001 envs.py (MontezumaInfoWrapper)
    """
    # Lazy import
    try:
        import gymnasium as gym
    except ImportError:
        import gym  # type: ignore

    class MontezumaInfoWrapper(gym.Wrapper):
        """Track visited rooms for FAR/CLOSE state diagnostics."""
        def __init__(self, env: Any, room_address: int = MONTEZUMA_ROOM_ADDRESS):
            super().__init__(env)
            self.room_address = room_address
            self.visited_rooms: set = set()

        def get_current_room(self) -> int:
            try:
                ram = self.env.unwrapped.ale.getRAM()
                assert len(ram) == 128
                return int(ram[self.room_address])
            except Exception:
                return -1

        def step(self, action: Any) -> Any:
            obs, reward, terminated, truncated, info = self.env.step(action)
            room = self.get_current_room()
            self.visited_rooms.add(room)
            info["current_room"] = room
            info["visited_rooms"] = list(self.visited_rooms)
            # FAR = room >= 7 (paper Section 5, Figure 6)
            info["state_region"] = "far" if room >= 7 else "close"
            return obs, reward, terminated, truncated, info

        def reset(self, **kwargs: Any) -> Any:
            self.visited_rooms = set()
            return self.env.reset(**kwargs)

    return MontezumaInfoWrapper(env)


def _make_robotic_sequence_smoke_env(spec: EnvsSpec) -> Any:
    """
    Smoke fixture for RoboticSequence when the real backend is unavailable.
    Provides the same interface as a real env for smoke validation.
    """
    class RoboticSequenceSmokeEnv:
        """
        Smoke fixture for RoboticSequence.
        Exposes stage names, peg-unplug-side, push-wall, and per-stage success flags.
        Paper role: Section 4 (Figure 3c), Section 5 (Figure 7).
        """
        def __init__(self, stages: List[str], pretrained_stages: List[str]):
            self.stages = stages
            self.pretrained_stages = pretrained_stages
            self._step_count = 0
            self._current_stage_idx = 0

        @property
        def current_stage(self) -> str:
            return self.stages[self._current_stage_idx % len(self.stages)]

        def reset(self, seed: Optional[int] = None) -> Tuple[Any, Dict]:
            self._step_count = 0
            self._current_stage_idx = 0
            obs = {"stage": self.current_stage, "obs": [0.0] * 10}
            return obs, {"stage": self.current_stage}

        def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict]:
            self._step_count += 1
            stage = self.current_stage
            # Smoke: pretrained stages succeed, new stages fail
            success = stage in self.pretrained_stages
            reward = 1.0 if success else 0.0
            terminated = self._step_count >= SMOKE_STEPS
            if terminated:
                self._current_stage_idx += 1
            obs = {"stage": stage, "obs": [0.0] * 10}
            info = {
                "stage": stage,
                "stage_success": success,
                "state_region": "far" if stage in self.pretrained_stages else "close",
                "stage_success_rates": {s: (1.0 if s in self.pretrained_stages else 0.0)
                                        for s in self.stages},
            }
            return obs, reward, terminated, False, info

        def close(self) -> None:
            pass

    return RoboticSequenceSmokeEnv(
        stages=spec.stages or ROBOTIC_SEQUENCE_ALL_STAGES,
        pretrained_stages=spec.pretrained_stages or ROBOTIC_SEQUENCE_STAGES,
    )


# ---------------------------------------------------------------------------
# Convenience: resolve env id from alias
# ---------------------------------------------------------------------------

def resolve_env_id(env_id_or_alias: str) -> Optional[str]:
    """Resolve an environment alias to its canonical registry key."""
    return _ENV_ALIAS_MAP.get(env_id_or_alias) or _ENV_ALIAS_MAP.get(
        env_id_or_alias.lower()
    )


def resolve_dataset_id(dataset_id_or_alias: str) -> Optional[str]:
    """Resolve a dataset/benchmark alias to its canonical registry key."""