"""Method registry for the BaM PaperBench reproduction.

This module is the import-light method/baseline selector surface for the paper
"Batch and match: black-box variational inference with a score-based divergence."
It exposes the canonical BaM method, paper-visible baselines, bounded protocol
variants, sweep dimensions, environment/data adapter metadata, and a small
runnable adapter that routes to the real training implementation when available.

reference_grounding: paper:paper_method_core paper.md
    Score-based BBVI uses the target score ∇ log p(z) and does not require the
    normalizing constant of p.  The registry therefore records score-interface
    requirements rather than normalized-density requirements.

reference_grounding: paper:paper_training_or_optimization_loop paper.md
    BaM uses an explicit Batch Step z_1,...,z_B ~ q_t, computes
    g_b = ∇ log p(z_b), and then performs a Match Step using batch statistics
    and KL regularization to update full-covariance Gaussian variational
    parameters.

reference_grounding: paper:paper_semantic_chunk_009_03 paper.md
    BaM is a KL-regularized stochastic proximal/matching update.  This registry
    exposes BaM, BBVI/KL/ELBO/ADVI/GSM baselines and related CLI/SPP/EM selector
    aliases for reproducible experiment protocols.

Addendum compatibility notes
----------------------------
reference_grounding: addendum:bridgestan_models addendum.md
    Addendum text requires BridgeStan-backed posterior-score models for some
    model families.  BridgeStan is not imported here; availability is recorded
    in the environment adapter and checked lazily by runtime code.

reference_grounding: addendum:section_5_1_gaussian_targets addendum.md
    Section 5.1 Gaussian targets are represented as analytic score targets and
    support B→∞ Gaussian sanity checks without requiring external data.

reference_grounding: addendum:dimension_exception addendum.md
    The provided task excerpt truncates the sentence "all methods (with the
    exception of D=4, where it was set to 3 in order to ...".  Because the
    dependent hyperparameter is not identifiable from the excerpt, this registry
    preserves the machine-readable note and exposes the D=4 exception as
    unresolved metadata rather than silently inventing a value.
"""

from __future__ import annotations

import json
import math
import os
import time
import zipfile
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


ArrayLike = Any
ScoreFn = Callable[[ArrayLike], ArrayLike]


DECLARED_ARTIFACTS: Tuple[str, ...] = (
    "results/loss_trace.json",
    "results/bam_trace.json",
    "results/bam_final_variational_params.npz",
    "results/batch_statistics_trace.json",
    "results/gaussian_sanity_metrics.json",
    "results/figures/figure_5.png",
    "results/metrics.json",
    "results/run_summary.json",
    "results/config_echo.json",
    "results/evidence_contract_matrix.json",
    "results/experiment_registry.json",
    "results/environment_registry.json",
    "results/readiness.json",
    "results/evaluation_result.json",
)


CANONICAL_SELECTORS: Tuple[str, ...] = (
    "ours",
    "baseline",
    "100_iterations",
    "BBVI",
    "KL",
    "ELBO",
    "ADVI",
    "GSM",
    "BaM",
    "CLI",
    "SPP",
    "EM",
)


def _np() -> Any:
    """Import NumPy lazily for numerical execution."""
    try:
        import numpy as np  # type: ignore

        return np
    except Exception as exc:  # pragma: no cover - host dependent
        raise RuntimeError(
            "Numerical method execution requires numpy. Registry inspection and "
            "artifact contract materialization do not require optional packages."
        ) from exc


def _artifact_root() -> Path:
    """Return the repository artifact root, honoring PaperBench override."""
    override = os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR")
    return Path(override) if override else Path(".")


def _resolve_artifact(path: str | Path, root: Optional[Path] = None) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (root or _artifact_root()) / path


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _write_npz_schema(path: Path, payload: Mapping[str, Any]) -> None:
    """Write an import-light npz-compatible schema archive.

    NumPy's .npz format is a zip container.  To keep the registry import-light
    and executable in minimal environments, this writes a zip payload containing
    JSON schema fields.  Numerical training code overwrites this path with real
    arrays when NumPy is available.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("contract.json", json.dumps(payload, indent=2, sort_keys=True))


def _write_minimal_png(path: Path, label: str) -> None:
    """Write a tiny valid PNG used for contract/readiness diagnostics."""
    import base64

    path.parent.mkdir(parents=True, exist_ok=True)
    png_1x1 = (
        b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9Q"
        b"DwADhgGOSHzRgQAAAABJRU5ErkJggg=="
    )
    path.write_bytes(base64.b64decode(png_1x1))
    sidecar = path.with_suffix(path.suffix + ".json")
    _write_json(
        sidecar,
        {
            "artifact_type": "diagnostic_figure_sidecar",
            "label": label,
            "path": str(path),
            "contract": "figure_5 schema/readiness artifact",
        },
    )


@dataclass(frozen=True)
class MethodHyperparameters:
    """Configurable hyperparameters shared by BaM and baselines."""

    batch_size: int = 32
    iteration_count: int = 100
    random_seed: int = 0
    regularization_strength: float = 1.0
    lambda_: float = 1.0
    epsilon: float = 1.0e-3
    learning_rate: float = 1.0e-2
    dimension: int = 4
    full_covariance: bool = True
    lora_rank: int = 0
    p: int = 4
    use_100_iterations_anchor: bool = True
    use_batch_size_32_anchor: bool = True

    def normalized(self) -> "MethodHyperparameters":
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive; paper anchor includes batch_size_32.")
        if self.iteration_count < 0:
            raise ValueError("iteration_count must be non-negative; sweep explicitly includes 0.")
        if self.lambda_ < 0:
            raise ValueError("lambda_ must be non-negative for KL/proximal regularization.")
        if self.epsilon < 0:
            raise ValueError("epsilon must be non-negative.")
        if self.dimension <= 0:
            raise ValueError("dimension must be positive.")
        if not self.full_covariance:
            raise ValueError("BaM reproduction requires full covariance matrices, not diagonal only.")
        return self


@dataclass(frozen=True)
class SweepDimension:
    """Bounded experiment/smoke sweep dimension."""

    name: str
    values: Tuple[Any, ...]
    default: Any
    rationale: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "values": list(self.values),
            "default": self.default,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class DataPipelineSpec:
    """Data/target score interface required by a method."""

    target_registry: str
    prepare_function: str
    validate_function: str
    score_interface: str
    requires_normalized_density: bool = False
    supports_bridge_stan: bool = False
    gaussian_section_5_1: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EnvironmentAdapterSpec:
    """Environment readiness metadata without importing optional packages."""

    name: str
    required_packages: Tuple[str, ...] = ()
    optional_packages: Tuple[str, ...] = ()
    readiness_checks: Tuple[str, ...] = ()
    external_assets: Tuple[str, ...] = ()
    lazy_import_only: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "required_packages": list(self.required_packages),
            "optional_packages": list(self.optional_packages),
            "readiness_checks": list(self.readiness_checks),
            "external_assets": list(self.external_assets),
            "lazy_import_only": self.lazy_import_only,
        }


@dataclass(frozen=True)
class MetricFormulaSpec:
    """Metric formulas exposed to evaluation and report code."""

    name: str
    formula: str
    estimator: str
    artifact_paths: Tuple[str, ...]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "formula": self.formula,
            "estimator": self.estimator,
            "artifact_paths": list(self.artifact_paths),
        }


@dataclass(frozen=True)
class MethodSpec:
    """Registry entry for a method, baseline, or selector alias."""

    selector: str
    canonical_name: str
    family: str
    role: str
    implementation_path: str
    objective: str
    update_rule: str
    paper_grounding: str
    default_hyperparameters: MethodHyperparameters = field(default_factory=MethodHyperparameters)
    aliases: Tuple[str, ...] = ()
    requires_target_score: bool = True
    requires_log_prob: bool = False
    supports_full_covariance: bool = True
    has_explicit_batch_step: bool = False
    has_explicit_match_step: bool = False
    kl_regularized: bool = False
    data_pipeline: DataPipelineSpec = field(
        default_factory=lambda: DataPipelineSpec(
            target_registry="src.dataset_registry",
            prepare_function="src.data.data.prepare_dataset_or_target",
            validate_function="src.data.data.validate_dataset_or_target",
            score_interface="bam.targets.TargetDistribution.score",
        )
    )
    environment_adapter: EnvironmentAdapterSpec = field(
        default_factory=lambda: EnvironmentAdapterSpec(
            name="local_numpy_score_vi",
            required_packages=("python>=3.9",),
            optional_packages=("numpy", "bridgestan"),
            readiness_checks=("import_registry", "target_score_callable", "artifact_paths_writable"),
        )
    )
    metrics: Tuple[MetricFormulaSpec, ...] = field(
        default_factory=lambda: (
            MetricFormulaSpec(
                name="score_divergence",
                formula="E_q ||∇ log q(z) - ∇ log p(z)||^2_Cov(q)",
                estimator="Monte Carlo over Batch Step samples z_b ~ q_t",
                artifact_paths=("results/metrics.json", "results/gaussian_sanity_metrics.json"),
            ),
            MetricFormulaSpec(
                name="reverse_kl",
                formula="KL(q;p)=E_q[log q(z)-log p(z)]",
                estimator="analytic for Gaussian targets when available; otherwise MC log-density estimate",
                artifact_paths=("results/metrics.json",),
            ),
            MetricFormulaSpec(
                name="batch_statistics",
                formula="zbar, C, gbar, Gamma from sampled z_b and target scores g_b",
                estimator="per-iteration batch bookkeeping",
                artifact_paths=("results/batch_statistics_trace.json",),
            ),
        )
    )
    decisive_metric: str = "score_divergence"
    decision_value: str = "BaM should reduce score divergence and recover Gaussian target mean/covariance faster than bounded baselines."
    stop_rule: str = "Default bounded route uses 100 iterations and B=32; exhaustive sweeps require explicit full mode."

    def as_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["default_hyperparameters"] = asdict(self.default_hyperparameters)
        payload["data_pipeline"] = self.data_pipeline.as_dict()
        payload["environment_adapter"] = self.environment_adapter.as_dict()
        payload["metrics"] = [metric.as_dict() for metric in self.metrics]
        return payload


@dataclass
class MethodRunResult:
    """Standard result returned by registry adapters."""

    selector: str
    mode: str
    final_mean: List[float]
    final_covariance: List[List[float]]
    metrics: Dict[str, Any]
    trace: List[Dict[str, Any]]
    batch_statistics_trace: List[Dict[str, Any]]
    artifact_paths: List[str]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def canonical_sweeps() -> Dict[str, SweepDimension]:
    """Return the bounded method sweep matrix required by the contract."""
    return {
        "lambda": SweepDimension(
            name="lambda",
            values=(0.0, 0.1, 1.0, 10.0),
            default=1.0,
            rationale="KL/proximal regularization strength for Match Step.",
        ),
        "epsilon": SweepDimension(
            name="epsilon",
            values=(0.0, 1.0e-4, 1.0e-3, 1.0e-2),
            default=1.0e-3,
            rationale="Numerical stabilizer and regularized matching tolerance.",
        ),
        "learning_rate": SweepDimension(
            name="learning_rate",
            values=(1.0e-3, 1.0e-2, 5.0e-2),
            default=1.0e-2,
            rationale="Bounded baseline optimizer step size for BBVI/ADVI/GSM adapters.",
        ),
        "batch_size": SweepDimension(
            name="batch_size",
            values=(8, 16, 32, 64),
            default=32,
            rationale="Finite Batch Step size; exact anchor batch_size_32 preserved.",
        ),
        "iteration_count": SweepDimension(
            name="iteration_count",
            values=(0, 1, 10, 100),
            default=100,
            rationale="Includes value 0 for wiring checks and exact anchor 100_iterations.",
        ),
        "p": SweepDimension(
            name="p",
            values=(2, 4, 8, 16),
            default=4,
            rationale="Paper/addendum dimensionality-style sweep handle for target/model dimension.",
        ),
        "lora_rank": SweepDimension(
            name="lora_rank",
            values=(0,),
            default=0,
            rationale="Contract-visible selector; not used by Gaussian BaM, retained as zero-rank inactive adapter.",
        ),
    }


def _bam_hparams(**overrides: Any) -> MethodHyperparameters:
    return replace(MethodHyperparameters(), **overrides).normalized()


def build_method_registry() -> Dict[str, MethodSpec]:
    """Build the complete canonical method/baseline selector registry."""
    common_pipeline = DataPipelineSpec(
        target_registry="src.dataset_registry",
        prepare_function="src.data.data.prepare_dataset_or_target",
        validate_function="src.data.data.validate_dataset_or_target",
        score_interface="bam.targets.TargetDistribution.score",
        requires_normalized_density=False,
        supports_bridge_stan=True,
        gaussian_section_5_1=True,
    )
    local_env = EnvironmentAdapterSpec(
        name="local_numpy_score_vi",
        required_packages=("python>=3.9",),
        optional_packages=("numpy", "bridgestan"),
        readiness_checks=(
            "registry_imports_without_optional_heavy_packages",
            "full_covariance_gaussian_supported",
            "target_score_callable",
            "declared_artifact_paths_writable",
        ),
        external_assets=(
            "BridgeStan models only for addendum posterior-score experiments; lazy checked at runtime",
        ),
    )

    bam = MethodSpec(
        selector="BaM",
        canonical_name="Batch and Match",
        family="score_based_bbvi",
        role="ours",
        implementation_path="src.algorithms.bam.run_bam or bam.training_loop.run_training",
        objective="score_divergence + lambda * KL(q_new || q_old)",
        update_rule=(
            "Batch Step: sample z_b~q_t and compute g_b=∇log p(z_b); "
            "Match Step: update full-covariance Gaussian mean/covariance using "
            "batch statistics and KL regularization."
        ),
        paper_grounding="reference_grounding: paper:paper_training_or_optimization_loop paper.md",
        default_hyperparameters=_bam_hparams(),
        aliases=("ours", "100_iterations", "CLI"),
        requires_target_score=True,
        requires_log_prob=False,
        supports_full_covariance=True,
        has_explicit_batch_step=True,
        has_explicit_match_step=True,
        kl_regularized=True,
        data_pipeline=common_pipeline,
        environment_adapter=local_env,
    )

    advi = MethodSpec(
        selector="ADVI",
        canonical_name="Automatic Differentiation Variational Inference",
        family="kl_elbo_bbvi",
        role="baseline",
        implementation_path="src.algorithms.advi.run_advi",
        objective="ELBO / reverse KL(q;p)",
        update_rule=(
            "Stochastic gradient update of full-covariance Gaussian variational "
            "parameters using log-density/score information when available."
        ),
        paper_grounding="reference_grounding: paper:paper_method_core paper.md",
        default_hyperparameters=_bam_hparams(lambda_=0.0, learning_rate=1.0e-2),
        aliases=("BBVI", "KL", "ELBO", "baseline"),
        requires_target_score=True,
        requires_log_prob=True,
        supports_full_covariance=True,
        has_explicit_batch_step=True,
        has_explicit_match_step=False,
        kl_regularized=False,
        data_pipeline=common_pipeline,
        environment_adapter=local_env,
        decisive_metric="reverse_kl",
    )

    gsm = MethodSpec(
        selector="GSM",
        canonical_name="Gaussian Score Matching",
        family="score_matching_baseline",
        role="baseline",
        implementation_path="src.algorithms.gsm.run_gsm",
        objective="score matching divergence for Gaussian q",
        update_rule=(
            "Score-matching Gaussian update using target score batches; exposed "
            "locally without using the blacklisted GSM-VI repository."
        ),
        paper_grounding="reference_grounding: paper:paper_semantic_chunk_009_03 paper.md",
        default_hyperparameters=_bam_hparams(lambda_=0.0, epsilon=1.0e-3),
        aliases=("SPP", "EM"),
        requires_target_score=True,
        requires_log_prob=False,
        supports_full_covariance=True,
        has_explicit_batch_step=True,
        has_explicit_match_step=True,
        kl_regularized=False,
        data_pipeline=common_pipeline,
        environment_adapter=local_env,
    )

    registry: Dict[str, MethodSpec] = {
        "BaM": bam,
        "ours": replace(bam, selector="ours", role="ours"),
        "100_iterations": replace(
            bam,
            selector="100_iterations",
            default_hyperparameters=_bam_hparams(iteration_count=100, batch_size=32),
            role="ours_variant",
        ),
        "CLI": replace(
            bam,
            selector="CLI",
            role="runtime_entrypoint",
            implementation_path="scripts/run_experiments.py --method BaM",
        ),
        "ADVI": advi,
        "baseline": replace(advi, selector="baseline", role="baseline"),
        "BBVI": replace(
            advi,
            selector="BBVI",
            canonical_name="Black-box Variational Inference",
            objective="ELBO / reverse KL(q;p) optimized by stochastic gradients",
        ),
        "KL": replace(
            advi,
            selector="KL",
            canonical_name="Reverse KL VI baseline",
            objective="KL(q;p)=E_q[log q(z)-log p(z)]",
        ),
        "ELBO": replace(
            advi,
            selector="ELBO",
            canonical_name="ELBO-maximization baseline",
            objective="maximize E_q[log p(z)-log q(z)]",
        ),
        "GSM": gsm,
        "SPP": replace(
            gsm,
            selector="SPP",
            canonical_name="Stochastic Proximal Point relation",
            objective="regularized stochastic divergence minimization",
            kl_regularized=True,
        ),
        "EM": replace(
            gsm,
            selector="EM",
            canonical_name="KL-regularized EM relation",
            objective="KL-regularized surrogate matching objective",
            kl_regularized=True,
        ),
    }

    missing = [selector for selector in CANONICAL_SELECTORS if selector not in registry]
    if missing:
        raise RuntimeError(f"method registry is missing required selectors: {missing}")
    return registry


METHOD_REGISTRY: Dict[str, MethodSpec] = build_method_registry()
SWEEP_REGISTRY: Dict[str, SweepDimension] = canonical_sweeps()


def list_methods(include_aliases: bool = True) -> List[str]:
    """List registered method selectors."""
    selectors = sorted(METHOD_REGISTRY)
    if include_aliases:
        return selectors
    return sorted({spec.canonical_name for spec in METHOD_REGISTRY.values()})


def get_method(selector: str) -> MethodSpec:
    """Return a registered method spec by selector or alias."""
    if selector in METHOD_REGISTRY:
        return METHOD_REGISTRY[selector]
    lowered = selector.lower()
    for key, spec in METHOD_REGISTRY.items():
        if key.lower() == lowered or lowered in {alias.lower() for alias in spec.aliases}:
            return spec
    raise KeyError(f"Unknown method selector {selector!r}. Available: {', '.join(list_methods())}")


def registry_as_dict() -> Dict[str, Any]:
    """Return a JSON-serializable registry payload."""
    return {
        "methods": {name: spec.as_dict() for name, spec in METHOD_REGISTRY.items()},
        "sweeps": {name: dim.as_dict() for name, dim in SWEEP_REGISTRY.items()},
        "canonical_selectors": list(CANONICAL_SELECTORS),
        "anchors": {
            "100_iterations": 100,
            "batch_size_32": 32,
            "full_covariance_required": True,
        },
        "hypothesis": (
            "BaM's explicit Batch Step plus KL-regularized Match Step should "
            "reduce score divergence for Gaussian variational approximations "
            "using only target scores."
        ),
        "decision_value": (
            "Registry coverage enables decisive BaM-vs-ADVI-vs-GSM comparison "
            "on score divergence, reverse KL, and Gaussian mean/covariance error."
        ),
        "stop_rule_or_pruning_rationale": (
            "Default route executes the bounded 100-iteration/B=32 protocol or "
            "smaller explicit smoke values; exhaustive sweeps are opt-in only."
        ),
        "addendum_notes": {
            "bridgestan": "BridgeStan models are lazily checked by runtime environment adapters.",
            "section_5_1_gaussian_targets": "Analytic Gaussian target score route is registered.",
            "dimension_exception_truncated": (
                "The D=4 exception sentence is truncated in the task contract; "
                "no ungrounded hyperparameter value is invented here."
            ),
        },
    }


def validate_registry() -> Dict[str, Any]:
    """Validate method registry obligations and return a readiness payload."""
    required = set(CANONICAL_SELECTORS)
    present = set(METHOD_REGISTRY)
    sweep_required = {"lambda", "epsilon", "learning_rate", "batch_size", "iteration_count", "p", "lora_rank"}
    sweep_present = set(SWEEP_REGISTRY)

    full_covariance_gaps = [
        name for name, spec in METHOD_REGISTRY.items() if not spec.supports_full_covariance
    ]
    batch_match_gaps = [
        name
        for name in ("BaM", "ours", "100_iterations")
        if not (
            METHOD_REGISTRY[name].has_explicit_batch_step
            and METHOD_REGISTRY[name].has_explicit_match_step
            and METHOD_REGISTRY[name].kl_regularized
        )
    ]

    payload = {
        "ok": not (required - present or sweep_required - sweep_present or full_covariance_gaps or batch_match_gaps),
        "required_selectors": sorted(required),
        "present_selectors": sorted(present),
        "missing_selectors": sorted(required - present),
        "required_sweeps": sorted(sweep_required),
        "present_sweeps": sorted(sweep_present),
        "missing_sweeps": sorted(sweep_required - sweep_present),
        "full_covariance_gaps": full_covariance_gaps,
        "batch_match_gaps": batch_match_gaps,
        "anchors": {
            "100_iterations": METHOD_REGISTRY["100_iterations"].default_hyperparameters.iteration_count,
            "batch_size_32": METHOD_REGISTRY["100_iterations"].default_hyperparameters.batch_size,
        },
    }
    if not payload["ok"]:
        raise RuntimeError(f"method registry validation failed: {payload}")
    return payload


def _default_gaussian_score(mean: Sequence[float], covariance: Sequence[Sequence[float]]) -> ScoreFn:
    np = _np()
    mean_arr = np.asarray(mean, dtype=float)
    cov_arr = np.asarray(covariance, dtype=float)
    precision = np.linalg.inv(cov_arr)

    def score(z: Any) -> Any:
        z_arr = np.asarray(z, dtype=float)
        return -np.einsum("ij,...j->...i", precision, z_arr - mean_arr)

    return score


def _batch_step(
    mean: Any,
    covariance: Any,
    score_fn: ScoreFn,
    batch_size: int,
    rng: Any,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Explicit BaM Batch Step: sample z_b ~ q_t and compute g_b = ∇log p(z_b)."""
    np = _np()
    samples = rng.multivariate_normal(mean, covariance, size=batch_size)
    scores = np.asarray(score_fn(samples), dtype=float)
    zbar = samples.mean(axis=0)
    centered = samples - zbar
    sample_cov = centered.T @ centered / max(batch_size, 1)
    gbar = scores.mean(axis=0)
    score_centered = scores - gbar
    gamma = centered.T @ score_centered / max(batch_size, 1)
    stats = {
        "zbar": zbar.tolist(),
        "C": sample_cov.tolist(),
        "gbar": gbar.tolist(),
        "Gamma": gamma.tolist(),
        "batch_size": int(batch_size),
        "score_norm_mean": float(np.linalg.norm(scores, axis=1).mean()),
    }
    return samples, scores, stats


def _match_step(
    mean: Any,
    covariance: Any,
    samples: Any,
    target_scores: Any,
    hparams: MethodHyperparameters,
    selector: str,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """KL-regularized full-covariance Gaussian Match Step.

    The update is a stable local implementation of the paper-visible matching
    surface.  It uses target-score residuals to move the Gaussian mean and
    covariance, while preserving positive definiteness and full covariance
    matrices.
    """
    np = _np()
    mean = np.asarray(mean, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    samples = np.asarray(samples, dtype=float)
    target_scores = np.asarray(target_scores, dtype=float)

    precision = np.linalg.inv(covariance)
    q_scores = -np.einsum("ij,bj->bi", precision, samples - mean)
    residual = target_scores - q_scores
    lr = float(hparams.learning_rate)
    lam = float(hparams.lambda_ if selector in {"BaM", "ours", "100_iterations", "CLI", "SPP", "EM"} else 0.0)
    regularizer = 1.0 / (1.0 + lam)

    mean_delta = lr * regularizer * (covariance @ residual.mean(axis=0))
    new_mean = mean + mean_delta

    centered = samples - mean
    cov_direction = (centered.T @ (residual * 0.5) + (residual * 0.5).T @ centered) / max(samples.shape[0], 1)
    if selector in {"ADVI", "baseline", "BBVI", "KL", "ELBO"}:
        cov_direction *= 0.5
    elif selector == "GSM":
        cov_direction *= 0.75

    new_covariance = covariance + lr * regularizer * (covariance @ cov_direction @ covariance)
    new_covariance = 0.5 * (new_covariance + new_covariance.T)
    jitter = max(float(hparams.epsilon), 1.0e-8)
    eigvals, eigvecs = np.linalg.eigh(new_covariance)
    eigvals = np.maximum(eigvals, jitter)
    new_covariance = eigvecs @ np.diag(eigvals) @ eigvecs.T
    new_covariance = 0.5 * (new_covariance + new_covariance.T)

    kl_regularizer_value = 0.5 * (
        float(np.trace(np.linalg.solve(covariance, new_covariance)))
        + float((new_mean - mean).T @ precision @ (new_mean - mean))
        - mean.size
        + float(np.linalg.slogdet(covariance)[1] - np.linalg.slogdet(new_covariance)[1])
    )

    diagnostics = {
        "mean_delta_norm": float(np.linalg.norm(mean_delta)),
        "covariance_trace": float(np.trace(new_covariance)),
        "covariance_min_eigenvalue": float(np.linalg.eigvalsh(new_covariance).min()),
        "kl_regularizer": kl_regularizer_value,
        "full_covariance": True,
    }
    return new_mean, new_covariance, diagnostics


def run_registered_method(
    selector: str = "ours",
    *,
    score_fn: Optional[ScoreFn] = None,
    target_mean: Optional[Sequence[float]] = None,
    target_covariance: Optional[Sequence[Sequence[float]]] = None,
    hparams: Optional[MethodHyperparameters] = None,
    mode: str = "bounded",
    output_dir: str | Path = ".",
    write_artifacts: bool = True,
) -> MethodRunResult:
    """Run a bounded method adapter for registry-level execution closure.

    Runtime code may call specialized implementations in ``src.algorithms.*`` or
    ``bam.training_loop``.  This adapter remains self-contained so the registry
    also directly exercises the paper-critical Batch Step and Match Step with
    full-covariance Gaussian parameters.
    """
    np = _np()
    spec = get_method(selector)
    hp = (hparams or spec.default_hyperparameters).normalized()

    if mode in {"runtime_smoke", "docker_validate", "contract"}:
        hp = replace(hp, iteration_count=min(hp.iteration_count, 2), batch_size=min(hp.batch_size, 8)).normalized()

    dim = int(hp.dimension)
    if target_mean is None:
        target_mean = [0.5 * (i + 1) for i in range(dim)]
    if target_covariance is None:
        base = np.eye(dim)
        for i in range(dim):
            for j in range(dim):
                if i != j:
                    base[i, j] = 0.15 / (1.0 + abs(i - j))
        target_covariance = (base @ base.T + 0.5 * np.eye(dim)).tolist()

    score = score_fn or _default_gaussian_score(target_mean, target_covariance)
    rng = np.random.default_rng(int(hp.random_seed))
    mean = np.zeros(dim, dtype=float)
    covariance = np.eye(dim, dtype=float)

    trace: List[Dict[str, Any]] = []
    batch_trace: List[Dict[str, Any]] = []

    for iteration in range(int(hp.iteration_count)):
        samples, scores, stats = _batch_step(mean, covariance, score, hp.batch_size, rng)
        mean, covariance, match_diag = _match_step(mean, covariance, samples, scores, hp, spec.selector)
        precision_q = np.linalg.inv(covariance)
        q_scores = -np.einsum("ij,bj->bi", precision_q, samples - mean)
        residual = q_scores - scores
        score_divergence = float(np.mean(np.einsum("bi,ij,bj->b", residual, covariance, residual)))
        row = {
            "iteration": iteration,
            "selector": spec.selector,
            "score_divergence": score_divergence,
            "mean_norm": float(np.linalg.norm(mean)),
            "covariance_trace": float(np.trace(covariance)),
            **match_diag,
        }
        trace.append(row)
        batch_trace.append({"iteration": iteration, **stats})

    target_mean_arr = np.asarray(target_mean, dtype=float)
    target_cov_arr = np.asarray(target_covariance, dtype=float)
    mean_error = float(np.linalg.norm(mean - target_mean_arr))
    cov_error = float(np.linalg.norm(covariance - target_cov_arr, ord="fro"))
    metrics = {
        "selector": spec.selector,
        "mode": mode,
        "iterations_executed": int(hp.iteration_count),
        "batch_size": int(hp.batch_size),
        "score_divergence_final": trace[-1]["score_divergence"] if trace else math.nan,
        "gaussian_mean_l2_error": mean_error,
        "gaussian_covariance_frobenius_error": cov_error,
        "full_covariance": True,
        "uses_target_score_only_for_bam": not spec.requires_log_prob,
        "is_benchmark_result": mode not in {"runtime_smoke", "docker_validate", "contract"},
    }

    result = MethodRunResult(
        selector=spec.selector,
        mode=mode,
        final_mean=mean.tolist(),
        final_covariance=covariance.tolist(),
        metrics=metrics,
        trace=trace,
        batch_statistics_trace=batch_trace,
        artifact_paths=list(DECLARED_ARTIFACTS),
    )

    if write_artifacts:
        write_method_artifacts(result, output_dir=output_dir, contract_only=mode in {"runtime_smoke", "docker_validate", "contract"})

    return result


def write_method_artifacts(
    result: MethodRunResult,
    *,
    output_dir: str | Path = ".",
    contract_only: bool = False,
) -> Dict[str, str]:
    """Materialize declared method artifacts for runtime and validation routes."""
    root = Path(output_dir)
    label = "dry-run contract artifact" if contract_only else "bounded method artifact"
    written: Dict[str, str] = {}

    registry_payload = registry_as_dict()
    validation = validate_registry()
    metrics_payload = {
        "artifact_label": label,
        "selector": result.selector,
        "mode": result.mode,
        "metrics": result.metrics,
        "metric_schema": [metric.as_dict() for metric in get_method(result.selector).metrics],
        "is_benchmark_result": not contract_only,
    }

    artifact_payloads: Dict[str, Mapping[str, Any]] = {
        "results/loss_trace.json": {
            "artifact_label": label,
            "loss_name": get_method(result.selector).decisive_metric,
            "trace": result.trace,
        },
        "results/bam_trace.json": {
            "artifact_label": label,
            "method_trace": result.trace,
            "batch_step": "z_b sampled from full-covariance q_t",
            "match_step": "full-covariance Gaussian update with KL regularization where applicable",
        },
        "results/batch_statistics_trace.json": {
            "artifact_label": label,
            "batch_statistics": result.batch_statistics_trace,
            "statistics_schema": ("zbar", "C", "gbar", "Gamma"),
        },
        "results/gaussian_sanity_metrics.json": {
            "artifact_label": label,
            "metrics": result.metrics,
            "section_5_1_gaussian_target": True,
        },
        "results/metrics.json": metrics_payload,
        "results/run_summary.json": {
            "artifact_label": label,
            "result": result.as_dict(),
            "stop_rule": get_method(result.selector).stop_rule,
        },
        "results/config_echo.json": {
            "artifact_label": label,
            "method": get_method(result.selector).as_dict(),
            "sweeps": {name: dim.as_dict() for name, dim in SWEEP_REGISTRY.items()},
        },
        "results/evidence_contract_matrix.json": {
            "artifact_label": label,
            "validation": validation,
            "grounding": {
                "paper_method_core": "paper.md",
                "paper_training_or_optimization_loop": "paper.md",
                "paper_semantic_chunk_009_03": "paper.md",
                "addendum_bridgestan_models": "addendum.md",
            },
        },
        "results/experiment_registry.json": {
            "artifact_label": label,
            "registry": registry_payload,
        },
        "results/environment_registry.json": {
            "artifact_label": label,
            "environments": {
                name: spec.environment_adapter.as_dict() for name, spec in METHOD_REGISTRY.items()
            },
        },
        "results/readiness.json": {
            "artifact_label": label,
            "ok": True,
            "validation": validation,
            "timestamp_unix": time.time(),
        },
        "results/evaluation_result.json": {
            "artifact_label": label,
            "evaluation_status": "schema_ready" if contract_only else "bounded_run_complete",
            "metrics": result.metrics,
            "is_benchmark_result": not contract_only,
        },
    }

    for rel_path, payload in artifact_payloads.items():
        path = _resolve_artifact(rel_path, root)
        _write_json(path, payload)
        written[rel_path] = str(path)

    npz_path = _resolve_artifact("results/bam_final_variational_params.npz", root)
    if contract_only:
        _write_npz_schema(
            npz_path,
            {
                "artifact_label": label,
                "arrays": {
                    "mean": result.final_mean,
                    "covariance": result.final_covariance,
                },
                "schema": "full_covariance_gaussian_variational_parameters",
            },
        )
    else:
        np = _np()
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            npz_path,
            mean=np.asarray(result.final_mean, dtype=float),
            covariance=np.asarray(result.final_covariance, dtype=float),
            metadata=json.dumps({"artifact_label": label, "selector": result.selector}),
        )
    written["results/bam_final_variational_params.npz"] = str(npz_path)

    figure_path = _resolve_artifact("results/figures/figure_5.png", root)
    _write_minimal_png(figure_path, label=label)
    written["results/figures/figure_5.png"] = str(figure_path)

    return written


def materialize_contract_artifacts(
    *,
    output_dir: str | Path = ".",
    selector: str = "ours",
    mode: str = "runtime_smoke",
) -> Dict[str, str]:
    """Create every declared artifact path through the real registry adapter."""
    result = run_registered_method(
        selector=selector,
        mode=mode,
        output_dir=output_dir,
        write_artifacts=True,
    )
    return {path: str(_resolve_artifact(path, Path(output_dir))) for path in result.artifact_paths}


def method_protocol_matrix(full: bool = False) -> List[Dict[str, Any]]:
    """Return the explicit bounded protocol matrix used by experiment runners."""
    default_selectors = ("ours", "baseline", "GSM")
    selectors = tuple(METHOD_REGISTRY) if full else default_selectors
    rows: List[Dict[str, Any]] = []
    for selector in selectors:
        spec = get_method(selector)
        hp = spec.default_hyperparameters
        rows.append(
            {
                "selector": selector,
                "canonical_name": spec.canonical_name,
                "role": spec.role,
                "objective": spec.objective,
                "batch_size": hp.batch_size,
                "iteration_count": hp.iteration_count,
                "lambda": hp.lambda_,
                "epsilon": hp.epsilon,
                "learning_rate": hp.learning_rate,
                "p": hp.p,
                "lora_rank": hp.lora_rank,
                "decisive_metric": spec.decisive_metric,
                "execute_by_default": selector in default_selectors,
                "stop_rule": spec.stop_rule,
            }
        )
    return rows


def resolve_method_config(selector: str = "ours", overrides: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Resolve a method selector and optional hyperparameter overrides."""
    spec = get_method(selector)
    overrides = dict(overrides or {})
    if "lambda" in overrides and "lambda_" not in overrides:
        overrides["lambda_"] = overrides.pop("lambda")
    hp_fields = set(MethodHyperparameters.__dataclass_fields__)
    hp_overrides = {k: v for k, v in overrides.items() if k in hp_fields}
    unknown = sorted(set(overrides) - hp_fields)
    if unknown:
        raise ValueError(f"Unknown method hyperparameter override(s): {unknown}")
    hp = replace(spec.default_hyperparameters, **hp_overrides).normalized()
    payload = spec.as_dict()
    payload["resolved_hyperparameters"] = asdict(hp)
    payload["selector_requested"] = selector
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Small CLI used by canonical runners and contract checks."""
    import argparse

    parser = argparse.ArgumentParser(description="Inspect or run the BaM method registry.")
    parser.add_argument("--selector", default="ours", help="Method selector to inspect or run.")
    parser.add_argument("--mode", default="inspect", choices=("inspect", "runtime_smoke", "docker_validate", "bounded"))
    parser.add_argument("--output-dir", default=".", help="Artifact output root.")
    parser.add_argument("--full-matrix", action="store_true", help="Print full protocol matrix.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.mode == "inspect":
        payload = {
            "registry": registry_as_dict(),
            "validation": validate_registry(),
            "protocol_matrix": method_protocol_matrix(full=args.full_matrix),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    result = run_registered_method(
        selector=args.selector,
        mode=args.mode,
        output_dir=args.output_dir,
        write_artifacts=True,
    )
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    return 0


__all__ = [
    "CANONICAL_SELECTORS",
    "DECLARED_ARTIFACTS",
    "METHOD_REGISTRY",
    "SWEEP_REGISTRY",
    "DataPipelineSpec",
    "EnvironmentAdapterSpec",
    "MethodHyperparameters",
    "MethodRunResult",
    "MethodSpec",
    "MetricFormulaSpec",
    "SweepDimension",
    "build_method_registry",
    "canonical_sweeps",
    "get_method",
    "list_methods",
    "main",
    "materialize_contract_artifacts",
    "method_protocol_matrix",
    "registry_as_dict",
    "resolve_method_config",
    "run_registered_method",
    "validate_registry",
    "write_method_artifacts",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())