"""Metric compatibility exports."""

from .full_protocol import average_em_drop_ratio_percent, exact_match_score, f1_score_binary


def exact_match(prediction: str, target: str) -> float:
    return exact_match_score(prediction, target)


def classification_metrics(y_true, y_pred):
    tp = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 0)
    tn = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 0)
    n = len(y_true)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "accuracy": (tp + tn) / n if n else 0.0,
        "precision": precision,
        "recall": recall,
        "F1": f1_score_binary(y_true, y_pred),
    }

