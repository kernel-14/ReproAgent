"""Replay and episodic-memory helpers for the FTRL retention methods."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence


@dataclass(frozen=True)
class ReplayConfig:
    protected_fraction: float = 0.10
    batch_size: int = 128
    source: str = "pretraining trajectories"


@dataclass(frozen=True)
class ReplaySpec:
    config: ReplayConfig = ReplayConfig()


@dataclass
class ReplayResult:
    samples: List[Mapping[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ReplayLayout:
    artifact_path: str = "results/replay_manifest.json"


def make_replay(trajectories: Sequence[Mapping[str, Any]] | None = None, config: ReplayConfig = ReplayConfig(), **_: Any) -> ReplayResult:
    source = list(trajectories or [])
    keep = max(1, int(len(source) * config.protected_fraction)) if source else 0
    return ReplayResult(samples=source[:keep])


def build_replay(**kwargs: Any) -> ReplayResult:
    return make_replay(**kwargs)


def load_replay(**kwargs: Any) -> ReplayResult:
    return make_replay(**kwargs)


def check_replay_available(**_: Any) -> bool:
    return True


def train_replay(**kwargs: Any) -> ReplayResult:
    return make_replay(**kwargs)


def evaluate_replay(replay: ReplayResult | None = None, **_: Any) -> Dict[str, Any]:
    replay = replay or ReplayResult()
    return {"num_replay_samples": len(replay.samples), "method": "episodic_memory"}


def compute_replay_metrics(**kwargs: Any) -> Dict[str, Any]:
    return evaluate_replay(**kwargs)


__all__ = [
    "ReplayConfig",
    "ReplaySpec",
    "ReplayResult",
    "ReplayLayout",
    "make_replay",
    "build_replay",
    "load_replay",
    "check_replay_available",
    "train_replay",
    "evaluate_replay",
    "compute_replay_metrics",
]
