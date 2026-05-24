from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class TopicClassifier:
    """Lightweight classifier used for smoke scoring and topic-adherence checks."""

    labels: tuple[str, ...]
    keyword_weights: Mapping[str, Mapping[str, float]]

    def predict_proba(self, text: str) -> dict[str, float]:
        tokens = text.lower().replace(",", " ").replace(".", " ").split()
        scores = []
        for label in self.labels:
            weights = self.keyword_weights.get(label, {})
            score = 0.0
            for token in tokens:
                score += weights.get(token, 0.0)
            scores.append(score)
        arr = np.asarray(scores, dtype=float)
        arr = arr - arr.max()
        probs = np.exp(arr) / np.exp(arr).sum()
        return {label: float(prob) for label, prob in zip(self.labels, probs)}


def default_topic_classifier() -> TopicClassifier:
    return TopicClassifier(
        labels=("dragon_paris", "off_topic"),
        keyword_weights={
            "dragon_paris": {"dragon": 1.0, "flew": 1.0, "paris": 3.0, "france": 2.0},
            "off_topic": {"recipe": 1.5, "football": 1.5, "unrelated": 1.0},
        },
    )


def guidance_score(
    prompt: str,
    baseline_completion: str,
    guided_completion: str,
    classifier: TopicClassifier | None = None,
    target_label: str = "dragon_paris",
) -> dict[str, float | str]:
    clf = classifier or default_topic_classifier()
    baseline_text = f"{prompt} {baseline_completion}"
    guided_text = f"{prompt} {guided_completion}"
    baseline_prob = clf.predict_proba(baseline_text)[target_label]
    guided_prob = clf.predict_proba(guided_text)[target_label]
    return {
        "target_label": target_label,
        "baseline_probability": baseline_prob,
        "guided_probability": guided_prob,
        "absolute_gain": guided_prob - baseline_prob,
        "relative_gain_percent": 100.0 * (guided_prob - baseline_prob) / max(baseline_prob, 1e-12),
    }


def load_classifier(config: Mapping[str, object] | None = None) -> TopicClassifier:
    """Classifier factory matching the paper's classifier-guidance surfaces."""

    if not config:
        return default_topic_classifier()
    labels = tuple(str(x) for x in config.get("labels", ("dragon_paris", "off_topic")))  # type: ignore[union-attr]
    weights = config.get("keyword_weights", default_topic_classifier().keyword_weights)  # type: ignore[union-attr]
    return TopicClassifier(labels=labels, keyword_weights=weights)  # type: ignore[arg-type]


def finetune_classifier(config: Mapping[str, object] | None = None) -> TopicClassifier:
    """Deterministic stand-in for smoke; full runs can swap in a trained classifier.

    The repository exposes this callable because Section 6 compares classifier
    guidance/FUDGE-style scoring with CFG. The smoke implementation is active
    and deterministic; a full experiment can replace keyword weights with a
    model trained on IMDB or Jigsaw toxicity labels.
    """

    return load_classifier(config)
