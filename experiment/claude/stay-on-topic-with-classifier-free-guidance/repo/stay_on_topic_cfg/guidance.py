from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class CFGConfig:
    """Runtime controls for classifier-free guidance over next-token logits."""

    gamma: float = 1.5
    unconditional_from_last_prompt_token: bool = True
    negative_prompt_ids: tuple[int, ...] | None = None
    top_p: float = 0.9

    def __post_init__(self) -> None:
        if self.gamma < 0:
            raise ValueError("gamma must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")


def _as_array(values: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("logits/probability arrays must be non-empty")
    return arr


def combine_cfg_logits(
    conditional_logits: Sequence[float] | np.ndarray,
    unconditional_logits: Sequence[float] | np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Apply Equation 7 from the paper to next-token logits.

    Equation 7 is equivalent in logits space to:
    gamma * conditional_logits - (gamma - 1) * unconditional_logits.
    When gamma is 1 this is exactly vanilla conditional generation.
    """

    cond = _as_array(conditional_logits)
    uncond = _as_array(unconditional_logits)
    if cond.shape != uncond.shape:
        raise ValueError(f"conditional and unconditional logits must share shape: {cond.shape} != {uncond.shape}")
    return gamma * cond - (gamma - 1.0) * uncond


def combine_negative_prompt_logits(
    conditional_logits: Sequence[float] | np.ndarray,
    negative_logits: Sequence[float] | np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Generalized CFG with a negative prompt as the baseline distribution."""

    return combine_cfg_logits(conditional_logits, negative_logits, gamma)


def prepare_unconditional_ids(
    prompt_ids: Sequence[int],
    generated_ids: Sequence[int] | None = None,
    config: CFGConfig | None = None,
) -> list[int]:
    """Build the unconditional context used by the paper's Section 3.1 protocol.

    The unconditional prompt starts at the last token of the initial prompt, then
    appends tokens generated so far. This preserves the autoregressive local
    state while dropping the condition being upweighted.
    """

    if not prompt_ids:
        raise ValueError("prompt_ids cannot be empty")
    generated = list(generated_ids or [])
    cfg = config or CFGConfig()
    if cfg.negative_prompt_ids is not None:
        return list(cfg.negative_prompt_ids) + generated
    if cfg.unconditional_from_last_prompt_token:
        return [int(prompt_ids[-1]), *map(int, generated)]
    return list(map(int, generated))


def softmax(logits: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = _as_array(logits)
    shifted = arr - np.max(arr)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def entropy(probabilities: Sequence[float] | np.ndarray) -> float:
    probs = _as_array(probabilities)
    total = probs.sum()
    if total <= 0:
        raise ValueError("probabilities must have positive mass")
    probs = probs / total
    nz = probs[probs > 0]
    return float(-np.sum(nz * np.log(nz)))


def top_p_token_count(probabilities: Sequence[float] | np.ndarray, top_p: float = 0.9) -> int:
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    probs = _as_array(probabilities)
    probs = probs / probs.sum()
    sorted_probs = np.sort(probs)[::-1]
    cumulative = np.cumsum(sorted_probs)
    return int(np.searchsorted(cumulative, top_p, side="left") + 1)


def rank_delta_trace(
    vocabulary: Sequence[str],
    conditional_logits: Sequence[float] | np.ndarray,
    unconditional_logits: Sequence[float] | np.ndarray,
    gamma: float,
    k: int = 5,
) -> list[dict[str, float | int | str]]:
    guided = combine_cfg_logits(conditional_logits, unconditional_logits, gamma)
    cond = _as_array(conditional_logits)
    uncond = _as_array(unconditional_logits)
    if len(vocabulary) != len(guided):
        raise ValueError("vocabulary length must match logits")
    order = np.argsort(guided)[::-1][:k]
    rows: list[dict[str, float | int | str]] = []
    for rank, idx in enumerate(order, start=1):
        rows.append(
            {
                "rank": rank,
                "token": vocabulary[int(idx)],
                "guided_logit": float(guided[int(idx)]),
                "conditional_logit": float(cond[int(idx)]),
                "unconditional_logit": float(uncond[int(idx)]),
                "cfg_boost": float(cond[int(idx)] - uncond[int(idx)]),
            }
        )
    return rows


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return float(sum(vals) / len(vals)) if vals else 0.0

