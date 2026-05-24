from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, Sequence


def exact_match(prediction: str, target: str) -> bool:
    return prediction.strip().lower() == target.strip().lower()


def accuracy(predictions: Sequence[str], targets: Sequence[str]) -> float:
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same length")
    return sum(exact_match(p, t) for p, t in zip(predictions, targets)) / max(1, len(targets))


def extract_final_answer(text: str) -> str | None:
    patterns = [r"answer is\s+([-+]?\d+(?:\.\d+)?)", r"####\s*([-+]?\d+(?:\.\d+)?)", r"\b([A-E])\b"]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def invalid_answer_rate(completions: Iterable[str]) -> float:
    rows = list(completions)
    invalid = sum(extract_final_answer(row) is None for row in rows)
    return invalid / max(1, len(rows))


def self_consistency_vote(completions: Iterable[str]) -> str | None:
    answers = [answer for row in completions if (answer := extract_final_answer(row)) is not None]
    if not answers:
        return None
    return Counter(answers).most_common(1)[0][0]


def estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """HumanEval pass@k estimator from Chen et al. 2021."""

    if num_samples <= 0:
        return 0.0
    if num_correct <= 0:
        return 0.0
    if num_samples - num_correct < k:
        return 1.0
    return 1.0 - math.prod(1.0 - k / i for i in range(num_samples - num_correct + 1, num_samples + 1))


def classifier_likelihood_delta(baseline_prob: float, guided_prob: float) -> float:
    if baseline_prob <= 0:
        raise ValueError("baseline probability must be positive")
    return 100.0 * (guided_prob - baseline_prob) / baseline_prob

