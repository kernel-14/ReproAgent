"""NetHack Section 4/5 evaluation protocols."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

from .nethack_appo import NetHackSaveLoadWrapper, nethack_evaluation_stop


@dataclass(frozen=True)
class NetHackEvaluationConfig:
    no_progress_stop_steps: int = 150
    max_rollout_steps: int = 100_000
    level4_saves: int = 200
    sokoban_saves: int = 200
    eval_interval_steps: int = 25_000_000
    full_eval_episodes: int = 1000


def average_return_over_trajectory_steps(rewards: Sequence[float]) -> float:
    """Section 4 metric: average return over all steps, not episode-sum only."""

    return float(sum(rewards) / max(1, len(rewards)))


def rollout_nethack_until_stop(
    env: NetHackSaveLoadWrapper,
    policy: Callable[[Any], Any],
    config: NetHackEvaluationConfig = NetHackEvaluationConfig(),
) -> Dict[str, Any]:
    rewards: List[float] = []
    max_dungeon_level = 1
    obs = env.reset()
    no_progress = 0
    best_progress = -1
    for step in range(config.max_rollout_steps):
        action = policy(obs)
        obs, reward, done, info = env.step(action)
        rewards.append(float(reward))
        progress = int(info.get("dungeon_level", info.get("maximum_dungeon_level", 0)))
        max_dungeon_level = max(max_dungeon_level, progress)
        if progress > best_progress:
            best_progress = progress
            no_progress = 0
        else:
            no_progress += 1
        if nethack_evaluation_stop(done, no_progress, step + 1):
            break
    return {
        "average_return_over_steps": average_return_over_trajectory_steps(rewards),
        "maximum_dungeon_level": max_dungeon_level,
        "num_steps": len(rewards),
    }


def evaluate_level4_from_200_autoascend_saves(
    env: NetHackSaveLoadWrapper,
    policy: Callable[[Any], Any],
    save_paths: Sequence[str | Path],
) -> Dict[str, Any]:
    """Load each of 200 AutoAscend Level-4 saves and compute score delta."""

    deltas: List[float] = []
    for path in list(save_paths)[:200]:
        env.load_game(path)
        result = rollout_nethack_until_stop(env, policy)
        deltas.append(float(result["average_return_over_steps"]))
    return {"num_saves": len(deltas), "average_level4_return": sum(deltas) / max(1, len(deltas))}


def evaluate_sokoban_from_200_autoascend_saves(
    env: NetHackSaveLoadWrapper,
    policy: Callable[[Any], Any],
    save_paths: Sequence[str | Path],
) -> Dict[str, Any]:
    """Load each of 200 AutoAscend Sokoban saves and count filled pits."""

    filled_pits: List[float] = []
    for path in list(save_paths)[:200]:
        env.load_game(path)
        result = rollout_nethack_until_stop(env, policy)
        filled_pits.append(float(result.get("filled_pits", result.get("maximum_dungeon_level", 0))))
    return {"num_saves": len(filled_pits), "average_sokoban_filled_pits": sum(filled_pits) / max(1, len(filled_pits))}


def every_25m_training_steps(total_steps: int, config: NetHackEvaluationConfig = NetHackEvaluationConfig()) -> bool:
    return int(total_steps) > 0 and int(total_steps) % config.eval_interval_steps == 0


def nethack_eval_protocol() -> Dict[str, Any]:
    return asdict(NetHackEvaluationConfig())


__all__ = [
    "NetHackEvaluationConfig",
    "average_return_over_trajectory_steps",
    "rollout_nethack_until_stop",
    "evaluate_level4_from_200_autoascend_saves",
    "evaluate_sokoban_from_200_autoascend_saves",
    "every_25m_training_steps",
    "nethack_eval_protocol",
]
