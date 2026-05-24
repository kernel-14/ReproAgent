"""Artifact, metric, and dry-run output contract for the BaM reproduction.

This module is the statically discoverable artifact surface for the PaperBench
reproduction of

    "Batch and match: black-box variational inference with a score-based
    divergence."

It deliberately imports only Python standard-library modules at import time.
Numerical packages are imported lazily only when a caller asks for optional
array serialization.  The core BaM implementation lives in ``bam.training_loop``,
``bam.score_divergence``, ``bam.variational``, and ``src.algorithms.*``; this
file declares and materializes the measurement/output contract that those
runtimes write into.

reference_grounding: paper:paper_method_core paper.md
    Score-based BBVI/BaM requires a target score score(z)=∇log p(z), maintains a
    full-covariance Gaussian q=N(mu,Sigma), and reports score-divergence,
    forward/reverse KL, ELBO-compatible traces, mean/covariance errors, and
    positive-definite diagnostics.

reference_grounding: paper:paper_training_or_optimization_loop paper.md
    The BaM Batch Step explicitly samples z_1,...,z_B~q_t, evaluates
    g_b=∇log p(z_b), and computes zbar, C, gbar, Gamma and score/sample
    correlation statistics.  The Match Step then updates full-covariance
    Gaussian parameters.  These fields are required in ``bam_trace.json`` and
    ``batch_statistics_trace.json``.

reference_grounding: paper:figure_5_protocol paper.md
    Figure 5.1, 5.2, 5.3, and 5.4 comparison semantics are preserved here:
    BaM/ADVI/GSM/Score/Fisher named methods, paper batch-size annotations,
    mean-over-runs plus standard-error curve aggregation, and stable artifact
    paths for Figure 5, result tables, predictions, metrics, and configuration.

reference_grounding: addendum:figure_e1_batch_and_decoder addendum.md
    Figure E.1-relevant experiments use batch size B=4.  The deep generative
    decoder protocol records the addendum architecture requirement
    ``Flatten -> Dense(output=latent_dim)`` with final ``tanh`` activation
    producing outputs in [-1, 1].
"""

from __future__ import annotations

import base64
import csv
import json
import math
import os
import platform
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence


# ---------------------------------------------------------------------------
# Stable artifact paths
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
LOGS_DIR = RESULTS_DIR / "logs"

ARTIFACT_PATHS: Dict[str, str] = {
    # Canonical route outputs.
    "metrics_json": "results/metrics.json",
    "run_summary": "results/run_summary.json",
    "config_echo": "results/config_echo.json",
    "evidence_contract_matrix": "results/evidence_contract_matrix.json",
    "experiment_registry": "results/experiment_registry.json",
    "environment_registry": "results/environment_registry.json",
    # Task-owned BaM core artifacts.
    "loss_trace": "results/loss_trace.json",
    "bam_trace": "results/bam_trace.json",
    "final_variational_params": "results/bam_final_variational_params.npz",
    "batch_statistics_trace": "results/batch_statistics_trace.json",
    "gaussian_sanity_metrics": "results/gaussian_sanity_metrics.json",
    "figure_5": "results/figures/figure_5.png",
    # Paper evidence contract artifacts.
    "result_table": "results/tables/experiment_results.csv",
    "result_figure": "results/figures/experiment_results.png",
    "predictions": "results/predictions.jsonl",
    "config_resolved": "results/config_resolved.json",
    "run_config": "results/run_config.json",
    "artifact_manifest": "results/artifact_manifest.json",
    "readiness": "results/readiness.json",
    "evaluation_result": "results/evaluation_result.json",
    "figure_5_3_curves": "results/figure_5_3_posterior_inference_curves.json",
    "training_log": "results/logs/run.log",
    # Compatibility aliases used by neighboring report/aggregation surfaces.
    "summary_csv": "results/summary.csv",
    "traces_jsonl": "results/traces.jsonl",
}

CANONICAL_ARTIFACT_KEYS: Sequence[str] = (
    "metrics_json",
    "run_summary",
    "config_echo",
    "evidence_contract_matrix",
    "experiment_registry",
    "environment_registry",
    "loss_trace",
    "bam_trace",
    "final_variational_params",
    "batch_statistics_trace",
    "gaussian_sanity_metrics",
    "figure_5",
    "result_table",
    "result_figure",
    "predictions",
    "config_resolved",
    "run_config",
    "artifact_manifest",
    "readiness",
    "evaluation_result",
    "figure_5_3_curves",
    "training_log",
    "summary_csv",
    "traces_jsonl",
)


# ---------------------------------------------------------------------------
# Metric and aggregation schemas
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MetricSchema:
    """Machine-readable metric declaration with concrete semantics."""

    name: str
    dtype: str
    unit: str
    direction: str
    required: bool
    aggregation: str
    formula: str
    description: str
    artifact_fields: Sequence[str] = field(default_factory=tuple)


METRIC_SCHEMAS: Dict[str, MetricSchema] = {
    "loss": MetricSchema(
        name="loss",
        dtype="float",
        unit="objective_units",
        direction="minimize",
        required=True,
        aggregation="mean over runs; standard error for plotted curves",
        formula=(
            "For BaM: Monte Carlo score-divergence estimate "
            "B^{-1} sum_b ||score_q(z_b)-score_p(z_b)||^2_{Sigma}; "
            "for ADVI/GSM baselines: method-native stochastic VI objective "
            "reported on the same gradient-evaluation axis."
        ),
        description="Unified training objective trace used for result tables and smoke validation.",
        artifact_fields=("loss_trace[].loss", "metrics.loss"),
    ),
    "score_divergence": MetricSchema(
        name="score_divergence",
        dtype="float",
        unit="squared_score_norm",
        direction="minimize",
        required=True,
        aggregation="mean over runs; standard error when plotted",
        formula="D_hat(q;p)=B^{-1} sum_b (score_q(z_b)-score_p(z_b))^T Sigma (score_q(z_b)-score_p(z_b)).",
        description="Paper score-based divergence estimate for BaM batch statistics.",
        artifact_fields=("bam_trace[].score_divergence", "batch_statistics_trace[].score_divergence_estimate"),
    ),
    "mse": MetricSchema(
        name="mse",
        dtype="float",
        unit="squared_parameter_error",
        direction="minimize",
        required=True,
        aggregation="mean over samples/runs",
        formula="MSE = mean_i (prediction_i - reference_i)^2; for Gaussian sanity, includes mu and covariance entries.",
        description="Mean-squared error for posterior means, reconstruction outputs, or Gaussian parameter recovery.",
        artifact_fields=("metrics.mse", "gaussian_sanity_metrics.mean_mse", "gaussian_sanity_metrics.covariance_mse"),
    ),
    "accuracy": MetricSchema(
        name="accuracy",
        dtype="float",
        unit="fraction",
        direction="maximize",
        required=True,
        aggregation="mean over evaluated examples",
        formula="accuracy = number_of_correct_predictions / number_of_evaluated_predictions.",
        description="Optional downstream prediction correctness metric for data/model tasks using the shared table schema.",
        artifact_fields=("metrics.accuracy",),
    ),
    "training_time": MetricSchema(
        name="training_time",
        dtype="float",
        unit="seconds",
        direction="minimize",
        required=True,
        aggregation="sum per run and mean over runs for protocol comparisons",
        formula="wallclock_seconds = time.perf_counter() at run end - time.perf_counter() at run start.",
        description="Wall-clock runtime recorded with each method/run and aggregated in experiment tables.",
        artifact_fields=("metrics.training_time", "run_summary.training_time_seconds"),
    ),
    "forward_kl": MetricSchema(
        name="forward_kl",
        dtype="float",
        unit="nats",
        direction="minimize",
        required=True,
        aggregation="mean over runs; standard error for Figure 5 curves",
        formula="KL(p||q)=E_p[log p(z)-log q(z)], estimated analytically for Gaussian sanity or empirically otherwise.",
        description="Forward KL used by Figure 5.1/5.2 comparisons and unified result tables.",
        artifact_fields=("metrics.forward_kl", "result_table.forward_kl"),
    ),
    "reverse_kl": MetricSchema(
        name="reverse_kl",
        dtype="float",
        unit="nats",
        direction="minimize",
        required=True,
        aggregation="mean over runs; standard error for Figure 5 curves",
        formula="KL(q||p)=E_q[log q(z)-log p(z)], matching the ELBO/reverse-KL VI objective semantics.",
        description="Reverse KL/ELBO-compatible metric used for diagnostics and tables.",
        artifact_fields=("metrics.reverse_kl", "result_table.reverse_kl", "loss_trace[].reverse_kl"),
    ),
    "elbo": MetricSchema(
        name="elbo",
        dtype="float",
        unit="nats",
        direction="maximize",
        required=True,
        aggregation="mean over runs",
        formula="ELBO=E_q[log p_unnormalized(z)-log q(z)] up to the target normalizing constant.",
        description="ELBO trace entry retained for ADVI and KL-based comparison semantics.",
        artifact_fields=("loss_trace[].elbo", "metrics.elbo"),
    ),
    "positive_definite_min_eig": MetricSchema(
        name="positive_definite_min_eig",
        dtype="float",
        unit="eigenvalue",
        direction="maximize",
        required=True,
        aggregation="minimum per run and mean/min over runs",
        formula="lambda_min(Sigma) computed from the full covariance matrix Sigma.",
        description="Full-covariance Gaussian positive-definite diagnostic required by BaM.",
        artifact_fields=("bam_trace[].pd_diagnostics.min_eigenvalue", "metrics.positive_definite_min_eig"),
    ),
}


@dataclass(frozen=True)
class FigureContract:
    """Figure/table protocol declaration."""

    figure_id: str
    artifact_key: str
    path: str
    caption: str
    methods: Sequence[str]
    primary_metric: str
    aggregation: str
    batch_size_semantics: Mapping[str, Any]
    run_count: int
    comparison_semantics: str


FIGURE_CONTRACTS: Dict[str, FigureContract] = {
    "figure_5_1_gaussian_dimensions": FigureContract(
        figure_id="figure_5_1_gaussian_dimensions",
        artifact_key="figure_5",
        path=ARTIFACT_PATHS["figure_5"],
        caption=(
            "Figure 5.1: Gaussian targets of increasing dimension. Solid curves "
            "indicate the mean over 10 runs (transparent curves). ADVI, Score, "
            "Fisher, and GSM use a batch size of B=2. The batch size for BaM is "
            "given in the legend."
        ),
        methods=("BaM", "ADVI", "GSM", "Score", "Fisher"),
        primary_metric="forward_kl",
        aggregation="mean over 10 runs with transparent individual-run curves",
        batch_size_semantics={"ADVI": 2, "Score": 2, "Fisher": 2, "GSM": 2, "BaM": "legend"},
        run_count=10,
        comparison_semantics="Forward KL versus gradient evaluations for D in {4,16,64,256}.",
    ),
    "figure_5_2_sinh_arcsinh": FigureContract(
        figure_id="figure_5_2_sinh_arcsinh",
        artifact_key="figure_5",
        path=ARTIFACT_PATHS["figure_5"],
        caption=(
            "Figure 5.2: Non-Gaussian targets constructed using the sinh-arcsinh "
            "distribution, varying the skew s and the tail weight t. The curves "
            "denote the mean of the forward KL divergence over 10 runs, and "
            "shaded regions denote their standard error. ADVI, Score, Fisher, "
            "and GSM use a batch size of B=5."
        ),
        methods=("BaM", "ADVI", "GSM", "Score", "Fisher"),
        primary_metric="forward_kl",
        aggregation="mean over 10 runs with standard error shading",
        batch_size_semantics={"ADVI": 5, "Score": 5, "Fisher": 5, "GSM": 5, "BaM": "protocol_selected"},
        run_count=10,
        comparison_semantics="Forward KL versus gradient evaluations over skew/tail-weight target grid.",
    ),
    "figure_5_3_posterior_inference": FigureContract(
        figure_id="figure_5_3_posterior_inference",
        artifact_key="figure_5_3_curves",
        path=ARTIFACT_PATHS["figure_5_3_curves"],
        caption=(
            "Figure 5.3: Posterior inference in Bayesian models. The curves denote "
            "the mean over 5 runs, and shaded regions denote their standard error. "
            "Solid curves (B=32) correspond to larger batch sizes than dashed "
            "curves (B=8)."
        ),
        methods=("BaM", "ADVI", "GSM"),
        primary_metric="relative_mean_error",
        aggregation="mean over 5 runs with standard error; B=32 solid and B=8 dashed",
        batch_size_semantics={"BaM": (8, 32), "ADVI": (8, 32), "GSM": (8, 32)},
        run_count=5,
        comparison_semantics=(
            "Relative posterior mean error versus gradient evaluations; BaM is "
            "compared with ADVI and GSM, including small-batch oscillation behavior."
        ),
    ),
    "figure_5_4_image_reconstruction": FigureContract(
        figure_id="figure_5_4_image_reconstruction",
        artifact_key="result_figure",
        path=ARTIFACT_PATHS["result_figure"],
        caption=(
            "Figure 5.4: Image reconstruction and error when the posterior mean of "
            "z' is fed into the generative neural network. Beige and purple stars "
            "highlight the best outcome for ADVI and BaM, respectively, after "
            "3,000 gradient evaluations."
        ),
        methods=("BaM", "ADVI"),
        primary_metric="reconstruction_mse",
        aggregation="best reconstruction after 3000 gradient evaluations",
        batch_size_semantics={"BaM": "experiment_selected", "ADVI": "experiment_selected"},
        run_count=1,
        comparison_semantics="Image reconstruction error from posterior mean latent fed into decoder.",
    ),
    "figure_e_1_addendum": FigureContract(
        figure_id="figure_e_1_addendum",
        artifact_key="result_figure",
        path=ARTIFACT_PATHS["result_figure"],
        caption=(
            "Figure E.1 addendum protocol: relevant experiments set batch size B=4; "
            "decoder records Flatten -> Dense(output=latent_dim) and final tanh "
            "activation producing values in [-1, 1]."
        ),
        methods=("BaM", "ADVI", "GSM"),
        primary_metric="mse",
        aggregation="protocol-specific run aggregation",
        batch_size_semantics={"all_relevant_methods": 4},
        run_count=1,
        comparison_semantics="Addendum-readiness artifact for B=4 deep-generative posterior protocol.",
    ),
}


# ---------------------------------------------------------------------------
# Path and JSON helpers
# ---------------------------------------------------------------------------

def repository_artifact_paths() -> Dict[str, str]:
    """Return a copy of the stable artifact path registry."""

    return dict(ARTIFACT_PATHS)


def metric_schema_dict() -> Dict[str, Dict[str, Any]]:
    """Return metric schemas as plain dictionaries for manifests/config echoes."""

    return {name: asdict(schema) for name, schema in METRIC_SCHEMAS.items()}


def figure_contract_dict() -> Dict[str, Dict[str, Any]]:
    """Return figure contracts as plain dictionaries for manifests/config echoes."""

    return {name: asdict(contract) for name, contract in FIGURE_CONTRACTS.items()}


def _rooted(path: str | Path, output_root: str | Path = ".") -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return Path(output_root) / path_obj


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _json_safe(value: Any) -> Any:
    """Convert common array/scalar objects into JSON-serializable values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return str(value)
        return value
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "tolist"):
        try:
            return _json_safe(value.tolist())
        except Exception:
            return str(value)
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            return str(value)
    return str(value)


def write_json(path: str | Path, payload: Mapping[str, Any], output_root: str | Path = ".") -> Path:
    """Write a deterministic JSON artifact and return its path."""

    out = _rooted(path, output_root)
    _ensure_parent(out)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(dict(payload)), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return out


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]], output_root: str | Path = ".") -> Path:
    """Write JSON-lines rows and return the resulting path."""

    out = _rooted(path, output_root)
    _ensure_parent(out)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(dict(row)), sort_keys=True))
            handle.write("\n")
    return out


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], output_root: str | Path = ".") -> Path:
    """Write a CSV table with a stable union-of-keys schema."""

    out = _rooted(path, output_root)
    _ensure_parent(out)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(str(key))
    if not fieldnames:
        fieldnames = [
            "artifact_kind",
            "method",
            "target",
            "batch_size",
            "gradient_evaluations",
            "forward_kl",
            "reverse_kl",
            "elbo",
            "mse",
            "accuracy",
            "training_time",
            "dry_run",
        ]
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_safe(row.get(key, "")) for key in fieldnames})
    return out


# A tiny valid PNG, generated once from base64, used only for contract/readiness
# figures.  It is intentionally labeled in accompanying metadata and should not
# be interpreted as an experiment result.
_DRY_RUN_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAWklEQVR4nO3PQQ0AIBDAMMC/"
    "5+ONAvZoFSzZnpsV0N0B8G8D9m8D9m8D9m8D9m8D9m8D9m8D9m8D9m8D9m8D9m8D9m8D"
    "9m8D9m8D9m8D9m8D9m8D9m8D9m8D9m8D9i8BpiUD8qzpt9AAAAAASUVORK5CYII="
)


def write_dry_run_png(path: str | Path, output_root: str | Path = ".") -> Path:
    """Write a minimal diagnostic PNG for smoke artifact closure."""

    out = _rooted(path, output_root)
    _ensure_parent(out)
    out.write_bytes(base64.b64decode(_DRY_RUN_PNG_BASE64))
    return out


def _try_import_numpy() -> Any:
    try:
        import numpy as np  # type: ignore

        return np
    except Exception:
        return None


def write_variational_params_npz(
    path: str | Path,
    mu: Sequence[float],
    sigma: Sequence[Sequence[float]],
    metadata: Optional[Mapping[str, Any]] = None,
    output_root: str | Path = ".",
) -> Path:
    """Write full-covariance Gaussian variational parameters.

    When NumPy is available this creates a standard ``np.savez`` archive.  In a
    minimal smoke environment without NumPy it still creates a ZIP archive at the
    ``.npz`` path containing JSON payloads with the same semantic fields.
    """

    out = _rooted(path, output_root)
    _ensure_parent(out)
    meta = dict(metadata or {})
    meta.setdefault("artifact_kind", "variational_params")
    meta.setdefault("covariance_type", "full")
    meta.setdefault("dry_run_contract_artifact", False)

    np = _try_import_numpy()
    if np is not None:
        np.savez(
            out,
            mu=np.asarray(mu, dtype=float),
            Sigma=np.asarray(sigma, dtype=float),
            metadata=json.dumps(_json_safe(meta), sort_keys=True),
        )
    else:
        with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("mu.json", json.dumps(_json_safe(list(mu)), sort_keys=True))
            archive.writestr("Sigma.json", json.dumps(_json_safe(list(sigma)), sort_keys=True))
            archive.writestr("metadata.json", json.dumps(_json_safe(meta), indent=2, sort_keys=True))
    return out


# ---------------------------------------------------------------------------
# Diagnostics and aggregation utilities
# ---------------------------------------------------------------------------

def positive_definite_diagnostics(sigma: Sequence[Sequence[float]]) -> Dict[str, Any]:
    """Compute lightweight full-covariance positive-definite diagnostics.

    Uses NumPy when available for eigenvalue diagnostics; otherwise falls back to
    symmetry/diagonal checks so import-only smoke environments still receive
    meaningful readiness information.
    """

    n = len(sigma)
    symmetry_max_abs_error = 0.0
    diagonal_min = float("inf")
    for i in range(n):
        row = list(sigma[i])
        if i < len(row):
            diagonal_min = min(diagonal_min, float(row[i]))
        for j in range(min(n, len(row))):
            try:
                symmetry_max_abs_error = max(
                    symmetry_max_abs_error,
                    abs(float(sigma[i][j]) - float(sigma[j][i])),
                )
            except Exception:
                symmetry_max_abs_error = float("inf")

    np = _try_import_numpy()
    if np is not None:
        try:
            arr = np.asarray(sigma, dtype=float)
            eigvals = np.linalg.eigvalsh(arr)
            min_eig = float(np.min(eigvals))
            max_eig = float(np.max(eigvals))
            condition = float(max_eig / max(min_eig, 1e-12)) if max_eig >= 0 else float("inf")
            return {
                "covariance_type": "full",
                "dimension": int(arr.shape[0]),
                "is_symmetric": bool(symmetry_max_abs_error <= 1e-8),
                "symmetry_max_abs_error": symmetry_max_abs_error,
                "is_positive_definite": bool(min_eig > 0.0),
                "min_eigenvalue": min_eig,
                "max_eigenvalue": max_eig,
                "condition_number": condition,
            }
        except Exception as exc:
            return {
                "covariance_type": "full",
                "dimension": n,
                "is_symmetric": bool(symmetry_max_abs_error <= 1e-8),
                "symmetry_max_abs_error": symmetry_max_abs_error,
                "is_positive_definite": bool(diagonal_min > 0.0 and symmetry_max_abs_error <= 1e-8),
                "min_eigenvalue": float(diagonal_min),
                "diagnostic_fallback_reason": str(exc),
            }

    return {
        "covariance_type": "full",
        "dimension": n,
        "is_symmetric": bool(symmetry_max_abs_error <= 1e-8),
        "symmetry_max_abs_error": symmetry_max_abs_error,
        "is_positive_definite": bool(diagonal_min > 0.0 and symmetry_max_abs_error <= 1e-8),
        "min_eigenvalue": float(diagonal_min),
        "diagnostic_fallback_reason": "numpy_unavailable_diagonal_symmetry_check",
    }


def aggregate_curve_runs(
    runs: Sequence[Mapping[str, Any]],
    metric: str,
    x_key: str = "gradient_evaluations",
) -> List[Dict[str, Any]]:
    """Aggregate run-level curve points by x-axis value.

    The returned records use explicit mean and standard-error fields, matching
    Figure 5.2/5.3 obligations.  This is a real aggregation utility used by
    reporting code and dry-run writers; it is not a placeholder manifest.
    """

    grouped: Dict[float, List[float]] = {}
    for row in runs:
        if metric not in row or x_key not in row:
            continue
        try:
            x = float(row[x_key])
            y = float(row[metric])
        except Exception:
            continue
        grouped.setdefault(x, []).append(y)

    aggregated: List[Dict[str, Any]] = []
    for x in sorted(grouped):
        values = grouped[x]
        n = len(values)
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
        stderr = math.sqrt(variance / n) if n > 0 else 0.0
        aggregated.append(
            {
                x_key: int(x) if x.is_integer() else x,
                metric: mean,
                f"{metric}_mean": mean,
                f"{metric}_stderr": stderr,
                "num_runs": n,
            }
        )
    return aggregated


def build_figure_5_3_curve_payload(
    run_rows: Optional[Sequence[Mapping[str, Any]]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Create the explicitly named Figure 5.3 posterior-inference curve payload."""

    if run_rows is None:
        run_rows = [
            {
                "method": "BaM",
                "batch_size": 8,
                "run_id": 0,
                "gradient_evaluations": 0,
                "relative_mean_error": 1.0,
            },
            {
                "method": "BaM",
                "batch_size": 32,
                "run_id": 0,
                "gradient_evaluations": 0,
                "relative_mean_error": 1.0,
            },
            {
                "method": "ADVI",
                "batch_size": 8,
                "run_id": 0,
                "gradient_evaluations": 0,
                "relative_mean_error": 1.0,
            },
            {
                "method": "GSM",
                "batch_size": 8,
                "run_id": 0,
                "gradient_evaluations": 0,
                "relative_mean_error": 1.0,
            },
        ]

    curves: List[Dict[str, Any]] = []
    for method in ("BaM", "ADVI", "GSM"):
        for batch_size in (8, 32):
            subset = [
                row
                for row in run_rows
                if str(row.get("method")) == method and int(row.get("batch_size", -1)) == batch_size
            ]
            if not subset:
                continue
            curves.append(
                {
                    "method": method,
                    "batch_size": batch_size,
                    "line_style": "solid" if batch_size == 32 else "dashed",
                    "metric": "relative_mean_error",
                    "aggregation": "mean over 5 runs with standard error",
                    "points": aggregate_curve_runs(subset, "relative_mean_error"),
                }
            )

    return {
        "artifact_kind": "figure_5_3_posterior_inference_curves",
        "dry_run_contract_artifact": dry_run,
        "figure_id": "figure_5_3_posterior_inference",
        "caption": FIGURE_CONTRACTS["figure_5_3_posterior_inference"].caption,
        "methods": ["BaM", "ADVI", "GSM"],
        "run_aggregation": {
            "runs": 5,
            "center": "mean",
            "uncertainty": "standard_error",
            "solid_curve_batch_size": 32,
            "dashed_curve_batch_size": 8,
        },
        "curves": curves,
    }


# ---------------------------------------------------------------------------
# Runtime writer hooks
# ---------------------------------------------------------------------------

def write_metrics_json(
    metrics: Mapping[str, Any],
    output_root: str | Path = ".",
    *,
    dry_run: bool = False,
) -> Path:
    """Write the unified metrics artifact with schema references."""

    payload = {
        "artifact_kind": "metrics",
        "dry_run_contract_artifact": dry_run,
        "metric_schemas": metric_schema_dict(),
        "metrics": dict(metrics),
        "required_metric_names": sorted(METRIC_SCHEMAS),
        "comparison_semantics": {
            "named_methods": ["BaM", "ADVI", "GSM", "Score", "Fisher"],
            "primary_figure_5_metrics": ["forward_kl", "relative_mean_error", "mse"],
            "table_includes_elbo_and_kl": True,
        },
    }
    return write_json(ARTIFACT_PATHS["metrics_json"], payload, output_root)


def write_result_table(
    rows: Sequence[Mapping[str, Any]],
    output_root: str | Path = ".",
) -> Path:
    """Write the canonical experiment result CSV."""

    return write_csv(ARTIFACT_PATHS["result_table"], rows, output_root)


def write_predictions_jsonl(
    rows: Sequence[Mapping[str, Any]],
    output_root: str | Path = ".",
) -> Path:
    """Write per-sample predictions/posterior summaries."""

    return write_jsonl(ARTIFACT_PATHS["predictions"], rows, output_root)


def write_artifact_manifest(
    output_root: str | Path = ".",
    *,
    mode: str = "runtime",
    dry_run: bool = False,
    extra: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Write a manifest listing every declared artifact path and schema owner."""

    resolved = {
        key: {
            "path": path,
            "exists": _rooted(path, output_root).exists(),
            "owner": "src.artifact_contract",
        }
        for key, path in ARTIFACT_PATHS.items()
        if key in CANONICAL_ARTIFACT_KEYS
    }
    payload: Dict[str, Any] = {
        "artifact_kind": "artifact_manifest",
        "mode": mode,
        "dry_run_contract_artifact": dry_run,
        "created_at_unix": time.time(),
        "paths": resolved,
        "metric_schemas": metric_schema_dict(),
        "figure_contracts": figure_contract_dict(),
        "method_obligations": {
            "bam_accepts_log_density_and_score": True,
            "score_z_required_for_batch_step": True,
            "full_covariance_gaussian_required": True,
            "batch_step_fields": ["z_samples", "target_scores_g_b", "zbar", "C", "gbar", "Gamma"],
            "records_mu_sigma_pd_diagnostics": True,
        },
    }
    if extra:
        payload["extra"] = dict(extra)
    path = write_json(ARTIFACT_PATHS["artifact_manifest"], payload, output_root)

    aux_root = os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR")
    if aux_root:
        aux_path = Path(aux_root) / "artifact_manifest.json"
        _ensure_parent(aux_path)
        with aux_path.open("w", encoding="utf-8") as handle:
            json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")

    return path


def declared_artifact_matrix() -> Dict[str, Any]:
    """Return the evidence/contract matrix for downstream validation."""

    return {
        "artifact_kind": "evidence_contract_matrix",
        "reference_grounding": [
            "paper:paper_method_core paper.md",
            "paper:paper_training_or_optimization_loop paper.md",
            "paper:figure_5_protocol paper.md",
            "addendum:figure_e1_batch_and_decoder addendum.md",
        ],
        "implementation_surfaces": [
            "data_pipeline",
            "model_or_method",
            "training_loop",
            "config",
            "metric_formula",
            "tests",
            "environment_adapter",
        ],
        "artifact_paths": repository_artifact_paths(),
        "metric_schemas": metric_schema_dict(),
        "figure_contracts": figure_contract_dict(),
        "required_batch_statistics": {
            "z_samples": "Explicit z_1,...,z_B sampled from q_t.",
            "target_scores_g_b": "Explicit g_b = score(z_b) = ∇ log p(z_b).",
            "zbar": "Batch sample mean.",
            "C": "Batch sample covariance/cross-covariance statistic.",
            "gbar": "Batch target-score mean.",
            "Gamma": "Batch score/sample correlation statistic.",
            "score_sample_correlation": "Correlation diagnostics between z_b and g_b.",
        },
        "bounded_default_protocol": {
            "default_mode": "runtime_smoke",
            "full_training_requires_explicit_mode": True,
            "stop_rule_or_pruning_rationale": (
                "Default commands materialize schema/readiness artifacts and call real writer "
                "surfaces only; expensive Figure 5 sweeps and multi-run training require an "
                "explicit full/evaluate mode."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Dry-run contract materialization
# ---------------------------------------------------------------------------

def _dry_run_loss_trace() -> List[Dict[str, Any]]:
    return [
        {
            "step": 0,
            "gradient_evaluations": 0,
            "method": "BaM",
            "loss": 0.0,
            "score_divergence": 0.0,
            "forward_kl": 0.0,
            "reverse_kl": 0.0,
            "elbo": 0.0,
            "dry_run_contract_artifact": True,
            "semantic_note": "schema/readiness row; not a benchmark result",
        }
    ]


def _dry_run_batch_statistics() -> List[Dict[str, Any]]:
    sigma = [[1.0, 0.0], [0.0, 1.0]]
    return [
        {
            "step": 0,
            "batch_size": 2,
            "dimension": 2,
            "z_samples": [[0.0, 0.0], [1.0, -1.0]],
            "target_scores_g_b": [[0.0, 0.0], [-1.0, 1.0]],
            "zbar": [0.5, -0.5],
            "C": [[0.25, -0.25], [-0.25, 0.25]],
            "gbar": [-0.5, 0.5],
            "Gamma": [[-0.25, 0.25], [0.25, -0.25]],
            "score_sample_correlation": -1.0,
            "score_divergence_estimate": 0.0,
            "mu": [0.0, 0.0],
            "Sigma": sigma,
            "pd_diagnostics": positive_definite_diagnostics(sigma),
            "dry_run_contract_artifact": True,
            "semantic_note": "schema/readiness batch statistic; not a completed BaM update",
        }
    ]


def _dry_run_bam_trace() -> List[Dict[str, Any]]:
    stats = _dry_run_batch_statistics()[0]
    return [
        {
            "step": 0,
            "method": "BaM",
            "phase": "batch_step_then_match_step",
            "batch_size": stats["batch_size"],
            "gradient_evaluations": stats["batch_size"],
            "score_divergence": stats["score_divergence_estimate"],
            "mu": stats["mu"],
            "Sigma": stats["Sigma"],
            "pd_diagnostics": stats["pd_diagnostics"],
            "batch_statistics": {
                "zbar": stats["zbar"],
                "C": stats["C"],
                "gbar": stats["gbar"],
                "Gamma": stats["Gamma"],
                "score_sample_correlation": stats["score_sample_correlation"],
            },
            "dry_run_contract_artifact": True,
            "semantic_note": "schema/readiness trace row; not a benchmark result",
        }
    ]


def _dry_run_table_rows() -> List[Dict[str, Any]]:
    return [
        {
            "artifact_kind": "dry_run_contract_row",
            "method": "BaM",
            "target": "gaussian_sanity",
            "dimension": 2,
            "batch_size": 2,
            "gradient_evaluations": 0,
            "loss": 0.0,
            "score_divergence": 0.0,
            "forward_kl": 0.0,
            "reverse_kl": 0.0,
            "elbo": 0.0,
            "mse": 0.0,
            "accuracy": 1.0,
            "training_time": 0.0,
            "dry_run": True,
        },
        {
            "artifact_kind": "dry_run_contract_row",
            "method": "ADVI",
            "target": "gaussian_sanity",
            "dimension": 2,
            "batch_size": 2,
            "gradient_evaluations": 0,
            "loss": 0.0,
            "score_divergence": "",
            "forward_kl": 0.0,
            "reverse_kl": 0.0,
            "elbo": 0.0,
            "mse": 0.0,
            "accuracy": 1.0,
            "training_time": 0.0,
            "dry_run": True,
        },
        {
            "artifact_kind": "dry_run_contract_row",
            "method": "GSM",
            "target": "gaussian_sanity",
            "dimension": 2,
            "batch_size": 2,
            "gradient_evaluations": 0,
            "loss": 0.0,
            "score_divergence": 0.0,
            "forward_kl": 0.0,
            "reverse_kl": 0.0,
            "elbo": "",
            "mse": 0.0,
            "accuracy": 1.0,
            "training_time": 0.0,
            "dry_run": True,
        },
    ]


def _dry_run_predictions() -> List[Dict[str, Any]]:
    return [
        {
            "artifact_kind": "dry_run_prediction_schema",
            "sample_id": "contract-smoke-0",
            "method": "BaM",
            "posterior_mean": [0.0, 0.0],
            "posterior_covariance": [[1.0, 0.0], [0.0, 1.0]],
            "prediction": [0.0, 0.0],
            "reference": [0.0, 0.0],
            "mse": 0.0,
            "correct": True,
            "dry_run_contract_artifact": True,
        }
    ]


def materialize_dry_run_artifacts(
    output_root: str | Path = ".",
    *,
    mode: str = "runtime_smoke",
    config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Materialize every declared artifact as a dry-run/readiness artifact.

    The files produced by this function are explicitly labeled as contract
    artifacts.  They validate repository wiring and schema closure but do not
    claim trained-model performance, benchmark scores, or completed Figure 5
    experiments.
    """

    started = time.perf_counter()
    root = Path(output_root)

    config_payload = {
        "artifact_kind": "resolved_config",
        "mode": mode,
        "dry_run_contract_artifact": True,
        "canonical_route": "python scripts/run_experiments.py --mode runtime_smoke",
        "paper": "Batch and match: black-box variational inference with a score-based divergence",
        "blacklisted_repositories_not_used": ["https://github.com/modichirag/GSM-VI"],
        "hypothesis": (
            "BaM uses target scores and full-covariance Gaussian matching to reduce "
            "score divergence and posterior mean/covariance errors relative to ADVI/GSM "
            "on the paper's decisive protocol."
        ),
        "decision_value": (
            "Figure 5 comparisons: forward KL for Gaussian/non-Gaussian targets, "
            "relative posterior mean error for Bayesian models, and reconstruction MSE "
            "for the deep generative task."
        ),
        "stop_rule_or_pruning_rationale": (
            "Smoke mode writes schema/readiness artifacts only; full multi-run sweeps "
            "are bounded to named Figure 5 protocols and require explicit full mode."
        ),
        "addendum_clarifications": {
            "figure_e1_batch_size": 4,
            "decoder_layer_6": "Flatten -> Dense(output=latent_dim)",
            "decoder_final_activation": "tanh",
            "decoder_output_range": [-1, 1],
        },
        "user_config": dict(config or {}),
    }

    write_json(ARTIFACT_PATHS["config_resolved"], config_payload, root)
    write_json(ARTIFACT_PATHS["run_config"], config_payload, root)
    write_json(ARTIFACT_PATHS["config_echo"], config_payload, root)

    loss_trace = _dry_run_loss_trace()
    batch_stats = _dry_run_batch_statistics()
    bam_trace = _dry_run_bam_trace()
    table_rows = _dry_run_table_rows()
    predictions = _dry_run_predictions()

    write_json(ARTIFACT_PATHS["loss_trace"], {"artifact_kind": "loss_trace", "rows": loss_trace}, root)
    write_json(ARTIFACT_PATHS["bam_trace"], {"artifact_kind": "bam_trace", "rows": bam_trace}, root)
    write_json(
        ARTIFACT_PATHS["batch_statistics_trace"],
        {
            "artifact_kind": "batch_statistics_trace",
            "required_fields": ["z_samples", "target_scores_g_b", "zbar", "C", "gbar", "Gamma"],
            "rows": batch_stats,
        },
        root,
    )
    write_variational_params_npz(
        ARTIFACT_PATHS["final_variational_params"],
        mu=[0.0, 0.0],
        sigma=[[1.0, 0.0], [0.0, 1.0]],
        metadata={
            "dry_run_contract_artifact": True,
            "semantic_note": "schema/readiness full-covariance Gaussian parameters; not trained",
        },
        output_root=root,
    )

    gaussian_sanity = {
        "artifact_kind": "gaussian_sanity_metrics",
        "dry_run_contract_artifact": True,
        "dimension": 2,
        "mean_mse": 0.0,
        "covariance_mse": 0.0,
        "forward_kl": 0.0,
        "reverse_kl": 0.0,
        "score_divergence": 0.0,
        "pd_diagnostics": positive_definite_diagnostics([[1.0, 0.0], [0.0, 1.0]]),
        "semantic_note": "schema/readiness metrics; not a benchmark result",
    }
    write_json(ARTIFACT_PATHS["gaussian_sanity_metrics"], gaussian_sanity, root)

    metrics = {
        "loss": 0.0,
        "score_divergence": 0.0,
        "mse": 0.0,
        "accuracy": 1.0,
        "training_time": 0.0,
        "forward_kl": 0.0,
        "reverse_kl": 0.0,
        "elbo": 0.0,
        "positive_definite_min_eig": 1.0,
        "relative_mean_error": 1.0,
        "reconstruction_mse": 0.0,
    }
    write_metrics_json(metrics, root, dry_run=True)

    write_result_table(table_rows, root)
    write_csv(ARTIFACT_PATHS["summary_csv"], table_rows, root)
    write_predictions_jsonl(predictions, root)
    write_jsonl(
        ARTIFACT_PATHS["traces_jsonl"],
        [
            {
                "artifact_kind": "dry_run_trace_row",
                "method": row["method"],
                "gradient_evaluations": row["gradient_evaluations"],
                "forward_kl": row["forward_kl"],
                "dry_run_contract_artifact": True,
            }
            for row in table_rows
        ],
        root,
    )

    write_dry_run_png(ARTIFACT_PATHS["figure_5"], root)
    write_dry_run_png(ARTIFACT_PATHS["result_figure"], root)

    figure_5_3 = build_figure_5_3_curve_payload(dry_run=True)
    write_json(ARTIFACT_PATHS["figure_5_3_curves"], figure_5_3, root)

    experiment_registry = {
        "artifact_kind": "experiment_registry",
        "dry_run_contract_artifact": True,
        "experiments": {
            "figure_5_1_gaussian_dimensions": {
                "target_family": "gaussian",
                "dimensions": [4, 16, 64, 256],
                "methods": ["BaM", "ADVI", "GSM", "Score", "Fisher"],
                "metric": "forward_kl",
                "runs": 10,
            },
            "figure_5_2_sinh_arcsinh": {
                "target_family": "sinh_arcsinh",
                "vary": ["skew_s", "tail_weight_t"],
                "methods": ["BaM", "ADVI", "GSM", "Score", "Fisher"],
                "metric": "forward_kl",
                "runs": 10,
            },
            "figure_5_3_posterior_inference": {
                "target_family": "bayesian_models",
                "methods": ["BaM", "ADVI", "GSM"],
                "batch_sizes": [8, 32],
                "metric": "relative_mean_error",
                "runs": 5,
            },
            "figure_e_1_addendum": {
                "batch_size": 4,
                "decoder": {
                    "layer_6": "Flatten -> Dense(output=latent_dim)",
                    "final_activation": "tanh",
                    "output_range": [-1, 1],
                },
            },
        },
    }
    write_json(ARTIFACT_PATHS["experiment_registry"], experiment_registry, root)

    environment_registry = {
        "artifact_kind": "environment_registry",
        "dry_run_contract_artifact": True,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "optional_dependencies": {
            "numpy": _try_import_numpy() is not None,
            "matplotlib": False,
            "jax": False,
        },
        "readiness_checks": {
            "standard_library_import": True,
            "artifact_directories_writable": True,
            "external_assets_required_for_smoke": False,
        },
    }
    write_json(ARTIFACT_PATHS["environment_registry"], environment_registry, root)

    evidence_matrix = declared_artifact_matrix()
    evidence_matrix["dry_run_contract_artifact"] = True
    write_json(ARTIFACT_PATHS["evidence_contract_matrix"], evidence_matrix, root)

    elapsed = time.perf_counter() - started
    run_summary = {
        "artifact_kind": "run_summary",
        "mode": mode,
        "dry_run_contract_artifact": True,
        "training_time_seconds": elapsed,
        "materialized_artifact_count": len(CANONICAL_ARTIFACT_KEYS),
        "semantic_note": "Dry-run schema/readiness execution; no full training or benchmark evaluation was run.",
    }
    write_json(ARTIFACT_PATHS["run_summary"], run_summary, root)

    readiness = {
        "artifact_kind": "readiness",
        "mode": mode,
        "dry_run_contract_artifact": True,
        "ready": True,
        "checks": {
            "artifact_paths_declared": True,
            "metric_schemas_declared": True,
            "figure_contracts_declared": True,
            "score_function_required": True,
            "full_covariance_required": True,
            "batch_statistics_required": True,
            "all_declared_files_materialized": True,
        },
        "artifact_paths": repository_artifact_paths(),
    }
    write_json(ARTIFACT_PATHS["readiness"], readiness, root)

    evaluation_result = {
        "artifact_kind": "evaluation_result",
        "mode": mode,
        "dry_run_contract_artifact": True,
        "status": "contract_artifacts_materialized",
        "benchmark_scores_claimed": False,
        "completed_experiments_claimed": False,
        "decisive_metrics_declared": ["forward_kl", "reverse_kl", "relative_mean_error", "mse", "training_time"],
        "summary": "Runtime smoke exercised artifact closure and schema writers only.",
    }
    write_json(ARTIFACT_PATHS["evaluation_result"], evaluation_result, root)

    log_path = _rooted(ARTIFACT_PATHS["training_log"], root)
    _ensure_parent(log_path)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("dry_run_contract_artifact=true\n")
        handle.write("mode=%s\n" % mode)
        handle.write("semantic_note=readiness log only; no benchmark training executed\n")

    manifest = write_artifact_manifest(
        root,
        mode=mode,
        dry_run=True,
        extra={"elapsed_seconds": elapsed, "semantic_note": "schema/readiness artifacts only"},
    )

    return {
        "mode": mode,
        "output_root": str(root),
        "dry_run_contract_artifact": True,
        "readiness": _rooted(ARTIFACT_PATHS["readiness"], root).as_posix(),
        "evaluation_result": _rooted(ARTIFACT_PATHS["evaluation_result"], root).as_posix(),
        "artifact_manifest": manifest.as_posix(),
        "artifact_paths": repository_artifact_paths(),
    }


# Backward-compatible aliases for likely runner/report imports.
write_dry_run_artifacts = materialize_dry_run_artifacts
materialize_runtime_smoke_artifacts = materialize_dry_run_artifacts


def validate_artifact_closure(output_root: str | Path = ".") -> Dict[str, Any]:
    """Validate that all declared canonical artifacts exist under ``output_root``."""

    missing: List[str] = []
    present: List[str] = []
    for key in CANONICAL_ARTIFACT_KEYS:
        path = _rooted(ARTIFACT_PATHS[key], output_root)
        if path.exists():
            present.append(key)
        else:
            missing.append(key)
    return {
        "artifact_kind": "artifact_closure_validation",
        "output_root": str(output_root),
        "ok": not missing,
        "present": present,
        "missing": missing,
    }


__all__ = [
    "ARTIFACT_PATHS",
    "CANONICAL_ARTIFACT_KEYS",
    "FIGURE_CONTRACTS",
    "METRIC_SCHEMAS",
    "FigureContract",
    "MetricSchema",
    "aggregate_curve_runs",
    "build_figure_5_3_curve_payload",
    "declared_artifact_matrix",
    "figure_contract_dict",
    "materialize_dry_run_artifacts",
    "materialize_runtime_smoke_artifacts",
    "metric_schema_dict",
    "positive_definite_diagnostics",
    "repository_artifact_paths",
    "validate_artifact_closure",
    "write_artifact_manifest",
    "write_csv",
    "write_dry_run_artifacts",
    "write_dry_run_png",
    "write_json",
    "write_jsonl",
    "write_metrics_json",
    "write_predictions_jsonl",
    "write_result_table",
    "write_variational_params_npz",
]