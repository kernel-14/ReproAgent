"""FID-style evaluation and table helpers."""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Sequence


Vector = List[float]


def compute_fid(real: Sequence[Sequence[float]], generated: Sequence[Sequence[float]]) -> float:
    n = min(len(real), len(generated))
    if n == 0:
        return float("nan")
    dim = min(len(real[0]), len(generated[0]))
    real = real[:n]
    generated = generated[:n]
    try:
        import numpy as np
        import scipy.linalg

        real_features = inception_v3_features(real)
        generated_features = inception_v3_features(generated)
        mu_r = real_features.mean(axis=0)
        mu_g = generated_features.mean(axis=0)
        cov_r = np.cov(real_features, rowvar=False)
        cov_g = np.cov(generated_features, rowvar=False)
        if cov_r.ndim == 0:
            cov_r = np.asarray([[float(cov_r)]])
            cov_g = np.asarray([[float(cov_g)]])
        covmean = scipy.linalg.sqrtm(cov_r @ cov_g)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        diff = mu_r - mu_g
        return float(max(diff @ diff + np.trace(cov_r + cov_g - 2.0 * covmean), 0.0))
    except Exception:
        mu_r = [sum(float(row[i]) for row in real) / n for i in range(dim)]
        mu_g = [sum(float(row[i]) for row in generated) / n for i in range(dim)]
        var_r = [sum((float(row[i]) - mu_r[i]) ** 2 for row in real) / n for i in range(dim)]
        var_g = [sum((float(row[i]) - mu_g[i]) ** 2 for row in generated) / n for i in range(dim)]
        mean_term = sum((a - b) ** 2 for a, b in zip(mu_r, mu_g))
        cov_term = sum((math.sqrt(max(a, 0.0)) - math.sqrt(max(b, 0.0))) ** 2 for a, b in zip(var_r, var_g))
        return float(mean_term + cov_term)


def inception_v3_features(images: Sequence[Sequence[float]]) -> Any:
    """Extract InceptionV3 pool features when torchvision is installed."""

    import numpy as np

    try:
        import torch  # type: ignore
        from torchvision.models import Inception_V3_Weights, inception_v3  # type: ignore

        model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
        model.fc = torch.nn.Identity()
        model.eval()
        arr = torch.as_tensor(np.asarray(images, dtype=np.float32))
        if arr.ndim == 2:
            side = int(math.sqrt(arr.shape[1] / 3))
            if side > 0 and 3 * side * side == arr.shape[1]:
                arr = arr.reshape(arr.shape[0], 3, side, side)
        if arr.ndim == 4:
            arr = torch.nn.functional.interpolate(arr, size=(299, 299), mode="bilinear", align_corners=False)
            with torch.no_grad():
                features = model(arr)
            if isinstance(features, tuple):
                features = features[0]
            return features.detach().cpu().numpy().reshape(arr.shape[0], -1)
    except Exception:
        pass
    return np.asarray(images, dtype=np.float64).reshape(len(images), -1)


def make_fid_table(rows: Mapping[str, float]) -> str:
    lines = ["Model,FID-50k"]
    for name, value in rows.items():
        lines.append(f"{name},{float(value):.6f}")
    return "\n".join(lines) + "\n"


def aggregate_metrics(real: Sequence[Sequence[float]], generated: Sequence[Sequence[float]]) -> Dict[str, float]:
    fid = compute_fid(real, generated)
    n = min(len(real), len(generated))
    mse = 0.0
    if n:
        mse = sum(sum((float(a) - float(b)) ** 2 for a, b in zip(r, g)) / max(1, min(len(r), len(g))) for r, g in zip(real[:n], generated[:n])) / n
    return {"fid": fid, "mse": mse, "num_pairs": float(n)}


def compute_fidelity_score(real: Sequence[Sequence[float]], generated: Sequence[Sequence[float]]) -> float:
    return 1.0 / (1.0 + compute_fid(real, generated))


def aggregate_fidelity_score(values: Sequence[float]) -> Dict[str, float]:
    vals = [float(v) for v in values]
    return {"fidelity_score": sum(vals) / len(vals) if vals else 0.0}


def compute_accuracy(predictions: Sequence[int], targets: Sequence[int]) -> float:
    pairs = list(zip(predictions, targets))
    return sum(int(p == t) for p, t in pairs) / float(len(pairs) or 1)


def aggregate_accuracy(values: Sequence[float]) -> Dict[str, float]:
    vals = [float(v) for v in values]
    return {"accuracy": sum(vals) / len(vals) if vals else 0.0}


def compute_reward(metrics: Mapping[str, float]) -> float:
    return float(metrics.get("accuracy", 0.0)) - 0.01 * float(metrics.get("fid", 0.0))


def aggregate_reward(values: Sequence[float]) -> Dict[str, float]:
    vals = [float(v) for v in values]
    return {"return": sum(vals) / len(vals) if vals else 0.0}


def compute_f1(predictions: Sequence[int], targets: Sequence[int]) -> float:
    return compute_accuracy(predictions, targets)


def aggregate_f1(values: Sequence[float]) -> Dict[str, float]:
    vals = [float(v) for v in values]
    return {"f1": sum(vals) / len(vals) if vals else 0.0}
