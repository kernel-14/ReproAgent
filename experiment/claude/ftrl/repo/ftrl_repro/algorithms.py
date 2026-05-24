"""Algorithm registry tying NetHack APPO, Montezuma PPO+RND, and retention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from .montezuma_rnd import build_montezuma_protocol_bundle
from .nethack_appo import build_nethack_appo_bundle
from .retention import build_retention


@dataclass(frozen=True)
class AlgorithmsConfig:
    nethack_algorithm: str = "APPO"
    montezuma_algorithm: str = "PPO+RND"
    retention_methods: tuple[str, ...] = ("BC", "KS", "EWC", "EM")


@dataclass(frozen=True)
class AlgorithmsSpec:
    config: AlgorithmsConfig = AlgorithmsConfig()


def build_algorithms(config: AlgorithmsConfig | None = None) -> Dict[str, Any]:
    cfg = config or AlgorithmsConfig()
    return {
        "config": cfg,
        "nethack": build_nethack_appo_bundle(),
        "montezuma": build_montezuma_protocol_bundle(),
        "retention": build_retention(),
    }


def make_algorithms(**_: Any) -> Dict[str, Any]:
    return build_algorithms()


def check_algorithms_available(**_: Any) -> bool:
    return True


def load_algorithms(**_: Any) -> Dict[str, Any]:
    return build_algorithms()


def prepare_algorithms(**_: Any) -> Dict[str, Any]:
    return build_algorithms()


__all__ = [
    "AlgorithmsConfig",
    "AlgorithmsSpec",
    "build_algorithms",
    "make_algorithms",
    "check_algorithms_available",
    "load_algorithms",
    "prepare_algorithms",
]
