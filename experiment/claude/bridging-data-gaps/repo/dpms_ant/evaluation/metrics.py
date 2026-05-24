# dpms_ant/evaluation/metrics.py
# =============================================================================
# DPMs-ANT – Metrics Module
# Paper: "Bridging Data Gaps in Diffusion Models with Adversarial Noise-Based
#         Transfer Learning"
#
# reference_grounding: paper_method_core dpms_ant/evaluation/metrics.py
# reference_grounding: paper_semantic_chunk_012 10-shot image generation
# reference_grounding: paper_semantic_chunk_014_01 DDPM + LDM framework evaluation
#
# Implements metric formulas, aggregation schemas, and artifact writers for:
#   Table 1  – Intra-LPIPS results (DDPM+GAN baselines)
#   Table 2  – FID results FFHQ→Babies and Sunglasses
#   Table 3  – FID/Intra-LPIPS with different classifiers
#   Table 4  – Intra-LPIPS (DDPM-based + GAN baselines)
#   Table 5  – Effects of γ (similarity guidance scale)
#   Table 6  – Effects of ω (adversarial perturbation budget)
#   Table 7  – Effects of training iterations
#   Table 8  – GPU memory comparison
#   Table 9  – User study
#   Figures 1-6 artifact declarations
#
# Trend assertions (semantic contract):
#   baseline_outperformance: ANT FID < DDPM-PA on all target domains
#   ANT outperforms all GAN-based baselines (TGAN/ADA/EWC/CDC/DCL)
#   Ablation-SimGuide: removing similarity-guided training → FID increases
#   Ablation-AdvNoise: removing adversarial noise selection → FID increases
#   Paper anchors: Babies ANT=46.70 < PA=48.92 (~4.5% improvement)
#                  Sunglasses ANT=20.06 < PA=34.75 (~42.3% improvement)
#
# Protocol matrix linking named experiments to environments/tasks, methods,
# measurements, and artifact paths – fully materialised in EXPERIMENT_PROTOCOL.
# =============================================================================

from __future__ import annotations

import json
import os
import math
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Artifact path registry (statically discoverable)
# ---------------------------------------------------------------------------

ARTIFACT_DIR = Path(os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR", "results"))

ARTIFACT_PATHS: Dict[str, Path] = {
    # Core metrics output
    "metrics":               ARTIFACT_DIR / "metrics.json",
    # Registries (schema artefacts written by dry-run)
    "dataset_registry":      ARTIFACT_DIR / "dataset_registry.json",
    "data_manifest":         ARTIFACT_DIR / "data_manifest.json",
    "environment_registry":  ARTIFACT_DIR / "environment_registry.json",
    "scope_report":          ARTIFACT_DIR / "scope_report.json",
    "experiment_registry":   ARTIFACT_DIR / "experiment_registry.json",
    # Table reproduction artifacts
    "table1":   ARTIFACT_DIR / "table1_intra_lpips.json",
    "table2":   ARTIFACT_DIR / "table2_fid_babies_sunglasses.json",
    "table3":   ARTIFACT_DIR / "table3_fid_lpips_classifiers.json",
    "table4":   ARTIFACT_DIR / "table4_intra_lpips_full.json",
    "table5":   ARTIFACT_DIR / "table5_gamma_sensitivity.json",
    "table6":   ARTIFACT_DIR / "table6_omega_sensitivity.json",
    "table7":   ARTIFACT_DIR / "table7_iteration_sensitivity.json",
    "table8":   ARTIFACT_DIR / "table8_gpu_memory.json",
    "table9":   ARTIFACT_DIR / "table9_user_study.json",
    # Figure reproduction artifacts
    "figure1":  ARTIFACT_DIR / "figure1_lpips_finetuning_stages.json",
    "figure2":  ARTIFACT_DIR / "figure2_gradient_heatmaps.json",
    "figure3":  ARTIFACT_DIR / "figure3_qualitative_lsun_ffhq.json",
    "figure4":  ARTIFACT_DIR / "figure4_ablation_fid.json",
    "figure5":  ARTIFACT_DIR / "figure5_qualitative_sunglasses_babies.json",
    "figure6":  ARTIFACT_DIR / "figure6_ablation_iterations.json",
    # Readiness / smoke
    "readiness":           ARTIFACT_DIR / "readiness.json",
    "evaluation_result":   ARTIFACT_DIR / "evaluation_result.json",
}


def artifact_path(key: str) -> Path:
    """Return the canonical output path for a named artifact."""
    if key not in ARTIFACT_PATHS:
        raise KeyError(f"Unknown artifact key '{key}'. Known keys: {sorted(ARTIFACT_PATHS)}")
    return ARTIFACT_PATHS[key]


# ---------------------------------------------------------------------------
# Named baseline registry (paper Table 2 / Table 1 / Table 4)
# ---------------------------------------------------------------------------

BASELINES: List[str] = [
    "tgan",          # Transfer-GAN
    "ada",           # ADA
    "ewc",           # EWC
    "cdc",           # CDC
    "dcl",           # DCL
    "ddpm_pa",       # DDPM Pairwise (current DPM-based SOTA)
    "ddpm_adaptor",  # Only fine-tune adaptor (ablation)
    "ddpm_ant_wo_an",# DPMs-ANT without adversarial noise (ablation)
    "ddpm_ant",      # Proposed – DDPM framework
    "ldm_ant",       # Proposed – LDM framework
]

# Paper-reported reference values (Table 2): FID↓ on FFHQ→{Babies, Sunglasses}
REFERENCE_FID_TABLE2: Dict[str, Dict[str, float]] = {
    "ddpm_pa": {
        "ffhq_babies":     48.92,
        "ffhq_sunglasses": 34.75,
    },
    "ddpm_ant": {
        "ffhq_babies":     46.70,   # ANT < PA: +4.5% improvement
        "ffhq_sunglasses": 20.06,   # ANT < PA: +42.3% improvement
    },
    "ddpm_adaptor": {
        "ffhq_sunglasses": 38.65,
        "ffhq_babies":     None,
    },
    "direct_finetune": {
        "ffhq_sunglasses": 41.88,
        "ffhq_babies":     None,
    },
}

# ---------------------------------------------------------------------------
# Protocol matrix
# ---------------------------------------------------------------------------

EXPERIMENT_PROTOCOL: List[Dict[str, Any]] = [
    {
        "experiment_did":   "Experiment-TableMain",
        "description":      "FFHQ→Babies and Sunglasses, all methods, Table 2 reproduction",
        "source_domain":    "ffhq",
        "target_domains":   ["ffhq_babies", "ffhq_sunglasses"],
        "frameworks":       ["ddpm"],
        "methods":          BASELINES,
        "shot_count":       10,
        "iterations":       5000,
        "measurements":     ["fid"],
        "artifact_path":    str(ARTIFACT_PATHS["table2"]),
        "trend_assertions": [
            "ANT FID < DDPM-PA: Babies 46.70 < 48.92",
            "ANT FID < DDPM-PA: Sunglasses 20.06 < 34.75",
            "ANT outperforms TGAN/ADA/EWC/CDC/DCL on FID",
        ],
    },
    {
        "experiment_did":   "Experiment-FullDomain",
        "description":      "All 7 target domains, DDPM framework, FID, Table 1 / Table 4",
        "source_domain":    "ffhq+lsun_church",
        "target_domains":   [
            "ffhq_babies", "ffhq_sunglasses", "ffhq_raphael_peale",
            "ffhq_sketches", "ffhq_modigliani",
            "lsun_haunted_houses", "lsun_landscape_drawings",
        ],
        "frameworks":       ["ddpm"],
        "methods":          ["ddpm_ant", "ddpm_pa", "tgan", "ada", "ewc", "cdc", "dcl"],
        "shot_count":       10,
        "iterations":       5000,
        "measurements":     ["fid", "intra_lpips"],
        "artifact_path":    str(ARTIFACT_PATHS["table1"]),
        "trend_assertions": [
            "ANT Intra-LPIPS ↑ vs all GAN-based baselines on most target domains",
            "DDPM-ANT > DDPM-PA on Intra-LPIPS across most tasks",
        ],
    },
    {
        "experiment_did":   "Experiment-LDM",
        "description":      "LDM framework, FFHQ source, FID+Intra-LPIPS, Table 4",
        "source_domain":    "ffhq",
        "target_domains":   ["ffhq_babies", "ffhq_sunglasses"],
        "frameworks":       ["ldm"],
        "methods":          ["ldm_ant", "tgan", "ada", "ewc", "cdc", "dcl"],
        "shot_count":       10,
        "iterations":       5000,
        "measurements":     ["fid", "intra_lpips"],
        "artifact_path":    str(ARTIFACT_PATHS["table4"]),
        "trend_assertions": [
            "LDM-ANT exceeds state-of-the-art GAN-based approaches on Intra-LPIPS",
        ],
    },
    {
        "experiment_did":   "Ablation-SimGuide",
        "description":      "Remove similarity-guided training (use_sim_guide=False), FID↑",
        "source_domain":    "ffhq",
        "target_domains":   ["ffhq_sunglasses"],
        "frameworks":       ["ddpm"],
        "methods":          ["ddpm_ant_wo_simguide", "ddpm_ant"],
        "shot_count":       10,
        "iterations":       300,
        "measurements":     ["fid"],
        "artifact_path":    str(ARTIFACT_PATHS["figure4"]),
        "trend_assertions": [
            "Removing similarity-guided training causes FID to increase",
        ],
    },
    {
        "experiment_did":   "Ablation-AdvNoise",
        "description":      "Remove adversarial noise selection (use_adv_noise=False), FID↑",
        "source_domain":    "ffhq",
        "target_domains":   ["ffhq_sunglasses"],
        "frameworks":       ["ddpm"],
        "methods":          ["ddpm_ant_wo_an", "ddpm_ant"],
        "shot_count":       10,
        "iterations":       300,
        "measurements":     ["fid"],
        "artifact_path":    str(ARTIFACT_PATHS["figure4"]),
        "trend_assertions": [
            "Removing adversarial noise selection causes FID to increase",
        ],
    },
    {
        "experiment_did":   "Ablation-AdaptorHyper",
        "description":      "Different c/d configurations for Shift Adaptor",
        "source_domain":    "ffhq",
        "target_domains":   ["ffhq_sunglasses"],
        "frameworks":       ["ddpm"],
        "methods":          ["ddpm_ant"],
        "adaptor_variants": [
            {"c": 4, "d": 8},  # paper default
            {"c": 2, "d": 4},
            {"c": 8, "d": 8},
        ],
        "shot_count":       10,
        "iterations":       5000,
        "measurements":     ["fid", "intra_lpips", "parameter_rate"],
        "artifact_path":    str(ARTIFACT_PATHS["table1"]),
        "trend_assertions": [
            "c=4, d=8 yields best FID/Intra-LPIPS trade-off on DDPM framework",
        ],
    },
    {
        "experiment_did":   "SensitivityAnalysis-Alpha",
        "description":      "Effects of γ (similarity guidance scale), Table 5",
        "source_domain":    "ffhq",
        "target_domains":   ["ffhq_sunglasses"],
        "frameworks":       ["ddpm"],
        "methods":          ["ddpm_ant"],
        "gamma_values":     [1, 2, 5, 10, 20],
        "shot_count":       10,
        "iterations":       5000,
        "measurements":     ["fid", "intra_lpips"],
        "artifact_path":    str(ARTIFACT_PATHS["table5"]),
        "trend_assertions": [
            "γ=5 is optimal (default) for FFHQ→Sunglasses",
        ],
    },
    {
        "experiment_did":   "SensitivityAnalysis-Omega",
        "description":      "Effects of ω (adversarial budget), Table 6",
        "source_domain":    "ffhq",
        "target_domains":   ["ffhq_sunglasses"],
        "frameworks":       ["ddpm"],
        "methods":          ["ddpm_ant"],
        "omega_values":     [0.005, 0.01, 0.02, 0.05, 0.1],
        "shot_count":       10,
        "iterations":       5000,
        "measurements":     ["fid", "intra_lpips"],
        "artifact_path":    str(ARTIFACT_PATHS["table6"]),
        "trend_assertions": [
            "ω=0.02 is optimal (addendum anchor) for FFHQ→Sunglasses",
        ],
    },
    {
        "experiment_did":   "SensitivityAnalysis-Iterations",
        "description":      "Effects of training iteration count, Table 7 / Figure 6",
        "source_domain":    "ffhq",
        "target_domains":   ["ffhq_sunglasses"],
        "frameworks":       ["ddpm"],
        "methods":          ["ddpm_ant", "ddpm_ant_wo_an", "direct_finetune"],
        "iteration_values": [100, 200, 300, 500, 1000, 5000],
        "shot_count":       10,
        "measurements":     ["fid", "intra_lpips"],
        "artifact_path":    str(ARTIFACT_PATHS["table7"]),
        "trend_assertions": [
            "DPMs-ANT converges faster than baseline due to adversarial noise",
        ],
    },
    {
        "experiment_did":   "Table3-ClassifierVariants",
        "description":      "FID/Intra-LPIPS with classifiers trained on 10 vs 100 images",
        "source_domain":    "ffhq",
        "target_domains":   ["ffhq_sunglasses"],
        "frameworks":       ["ddpm"],
        "methods":          ["ddpm_ant"],
        "classifier_shots": [10, 100],
        "shot_count":       10,
        "iterations":       5000,
        "measurements":     ["fid", "intra_lpips"],
        "artifact_path":    str(ARTIFACT_PATHS["table3"]),
        "trend_assertions": [
            "10-image classifier achieves competitive FID to 100-image classifier",
        ],
    },
]


# ---------------------------------------------------------------------------
# Metric schemas
# ---------------------------------------------------------------------------

METRIC_SCHEMA: Dict[str, Dict[str, Any]] = {
    "fid": {
        "description": (
            "Fréchet Inception Distance (FID↓). Computed between 50k generated images "
            "and the real target-domain distribution (or 10-shot proxy), using InceptionV3 "
            "features (pool3 layer). Lower is better."
        ),
        "direction":   "lower_is_better",
        "unit":        "dimensionless",
        "reference":   "Heusel et al. 2017",
        "paper_anchors": {
            "ffhq_babies_ant":        46.70,
            "ffhq_babies_ddpm_pa":    48.92,
            "ffhq_sunglasses_ant":    20.06,
            "ffhq_sunglasses_ddpm_pa": 34.75,
            "ffhq_sunglasses_adaptor_only": 38.65,
            "ffhq_sunglasses_direct_finetune": 41.88,
        },
    },
    "intra_lpips": {
        "description": (
            "Intra-LPIPS (↑): perceptual diversity metric computed as the mean pairwise "
            "LPIPS distance among generated images in a batch. Higher = more diverse."
        ),
        "direction":   "higher_is_better",
        "unit":        "dimensionless",
        "reference":   "Zhang et al. 2018",
        "paper_anchors": {},
    },
    "fidelity_score": {
        "description": (
            "Fidelity score: perceptual similarity (LPIPS) between a generated image and "
            "the corresponding target (10-shot) image computed on noisy intermediate "
            "representations (t-step noisy image) during fine-tuning. Used in Figure 1."
        ),
        "direction":   "lower_is_better",
        "unit":        "LPIPS",
        "reference":   "Figure 1 paper / Zhang et al. 2018",
        "paper_anchors": {},
    },
    "accuracy": {
        "description": (
            "Domain classifier accuracy: proportion of generated images correctly "
            "classified as target-domain by the fine-tuned MobileNet classifier."
        ),
        "direction":   "higher_is_better",
        "unit":        "fraction [0,1]",
        "reference":   "dpms_ant/classifier/domain_classifier.py",
        "paper_anchors": {},
    },
    "loss": {
        "description": "Training loss (simplified DDPM denoising objective + KL guidance).",
        "direction":   "lower_is_better",
        "unit":        "nats or MSE",
        "reference":   "Algorithm 1 in paper",
        "paper_anchors": {},
    },
    "training_time": {
        "description": "Wall-clock training time in seconds.",
        "direction":   "lower_is_better",
        "unit":        "seconds",
        "reference":   "Table 8 context",
        "paper_anchors": {},
    },
    "memory_usage": {
        "description": "Peak CPU memory usage in MB.",
        "direction":   "lower_is_better",
        "unit":        "MB",
        "reference":   "Table 8",
        "paper_anchors": {},
    },
    "gpu_memory": {
        "description": (
            "GPU memory consumption (MB) per module. Table 8 shows only a slight "
            "increase when using Shift Adaptor (batch_size=1)."
        ),
        "direction":   "lower_is_better",
        "unit":        "MB",
        "reference":   "Table 8",
        "paper_anchors": {
            "without_adaptor": None,
            "with_adaptor":    None,
        },
    },
    "parameter_rate": {
        "description": (
            "Parameter rate: proportion of parameters fine-tuned vs. pretrained model total. "
            "Used in Table 1 to compare efficiency."
        ),
        "direction":   "lower_is_better",
        "unit":        "fraction [0,1]",
        "reference":   "Table 1",
        "paper_anchors": {},
    },
}

# ---------------------------------------------------------------------------
# Trend assertions (semantic contract for review)
# ---------------------------------------------------------------------------

TREND_ASSERTIONS: List[Dict[str, Any]] = [
    {
        "assertion_id":  "baseline_outperformance_babies",
        "description":   "ANT FID < DDPM-PA on FFHQ→Babies",
        "metric":        "fid",
        "domain":        "ffhq_babies",
        "method_a":      "ddpm_ant",
        "method_b":      "ddpm_pa",
        "expected":      "method_a < method_b",
        "reference_a":   46.70,
        "reference_b":   48.92,
        "improvement_pct": 4.5,
    },
    {
        "assertion_id":  "baseline_outperformance_sunglasses",
        "description":   "ANT FID < DDPM-PA on FFHQ→Sunglasses",
        "metric":        "fid",
        "domain":        "ffhq_sunglasses",
        "method_a":      "ddpm_ant",
        "method_b":      "ddpm_pa",
        "expected":      "method_a < method_b",
        "reference_a":   20.06,
        "reference_b":   34.75,
        "improvement_pct": 42.3,
    },
    {
        "assertion_id":  "gan_baseline_outperformance",
        "description":   "ANT outperforms all GAN-based baselines (TGAN/ADA/EWC/CDC/DCL) on FID",
        "metric":        "fid",
        "domain":        "multiple",
        "method_a":      "ddpm_ant",
        "method_b":      "tgan,ada,ewc,cdc,dcl",
        "expected":      "method_a < all(method_b)",
    },
    {
        "assertion_id":  "ablation_sim_guide_fid_increase",
        "description":   "Removing similarity-guided training causes FID to increase",
        "metric":        "fid",
        "domain":        "ffhq_sunglasses",
        "method_a":      "ddpm_ant_wo_simguide",
        "method_b":      "ddpm_ant",
        "expected":      "method_a > method_b",
    },
    {
        "assertion_id":  "ablation_adv_noise_fid_increase",
        "description":   "Removing adversarial noise selection causes FID to increase",
        "metric":        "fid",
        "domain":        "ffhq_sunglasses",
        "method_a":      "ddpm_ant_wo_an",
        "method_b":      "ddpm_ant",
        "expected":      "method_a > method_b",
    },
    {
        "assertion_id":  "ldm_ant_gan_outperformance",
        "description":   "LDM-ANT exceeds state-of-the-art GAN-based approaches on Intra-LPIPS",
        "metric":        "intra_lpips",
        "domain":        "multiple",
        "method_a":      "ldm_ant",
        "method_b":      "tgan,ada,ewc,cdc,dcl",
        "expected":      "method_a > all(method_b)",
    },
]


# ---------------------------------------------------------------------------
# Figure/table artifact declaration hooks
# ---------------------------------------------------------------------------

FIGURE_CAPTIONS: Dict[str, str] = {
    "figure1": (
        "Figure 1. Two sets of images generated from corresponding fixed noise inputs "
        "at different stages of fine-tuning DDPM from FFHQ to 10-shot Sunglasses. "
        "The perceptual distance, LPIPS (Zhang et al., 2018), between the generated "
        "image and the target image is shown on each generated image."
    ),
    "figure2": (
        "Figure 2. Visualizations of gradient changes and heat maps. Figure (a) shows "
        "gradient directions with various settings: the cyan line denotes the gradient "
        "computed on 10,000 samples in one step; the blue, red, and orange lines are "
        "gradients of baseline method (i.e., traditional DDPM), DDPM-ANT w/o AN, "
        "and DPMs-ANT respectively."
    ),
    "figure3": (
        "Figure 3. The 10-shot image generation samples on LSUN Church → Landscape "
        "drawings (top) and FFHQ → Raphael's paintings (bottom). When compared with "
        "other GAN-based and DDPM-based methods, our method, ANT, yields high-quality "
        "results that more closely resemble images of the target domain style."
    ),
    "figure4": (
        "Figure 4. Ablation study, all models trained for 300 iterations on a 10-shot "
        "sunglasses dataset, measured with FID↓: first line - baseline (direct "
        "fine-tuning), second line - Adaptor only, third line - DPMs-ANT w/o AN, "
        "fourth line - DPMs-ANT (ours)."
    ),
    "figure5": (
        "Figure 5. The 10-shot image generation samples on FFHQ → Sunglasses and "
        "FFHQ → Babies."
    ),
    "figure6": (
        "Figure 6. Ablation study with all models trained at different iterations on a "
        "10-shot sunglasses dataset: first line - baseline, second line - DPMs-ANT w/o "
        "AN (only similarity-guided training), third line - DPMs-ANT (our method)."
    ),
}

TABLE_CAPTIONS: Dict[str, str] = {
    "table1": (
        "Table 1. Intra-LPIPS (↑) results for both DDPM and GAN-based baselines for "
        "10-shot image generation tasks (FFHQ and LSUN Church source domains). "
        "'Parameter Rate' = proportion of parameters fine-tuned vs. pre-trained model."
    ),
    "table2": (
        "Table 2. FID (↓) results of each method on 10-shot FFHQ → Babies and "
        "Sunglasses. Best results in bold."
    ),
    "table3": (
        "Table 3. FID and Intra-LPIPS results of DPM-ANT from FFHQ → Sunglasses with "
        "different classifiers (trained on 10 and 100 images)."
    ),
    "table4": (
        "Table 4. Intra-LPIPS (↑) results for both DDPM-based strategies and GAN-based "
        "baselines for 10-shot image generation tasks. Best results in bold."
    ),
    "table5": (
        "Table 5. Effects of γ (similarity guidance scale) in FFHQ → Sunglasses case "
        "in terms of FID and Intra-LPIPS."
    ),
    "table6": (
        "Table 6. Effects of ω (adversarial perturbation budget) in FFHQ → Sunglasses "
        "case in terms of FID and Intra-LPIPS."
    ),
    "table7": (
        "Table 7. Effects of training iteration count in FFHQ → Sunglasses case in "
        "terms of FID and Intra-LPIPS."
    ),
    "table8": (
        "Table 8. GPU memory consumption (MB) for each module, comparing scenarios "
        "with and without the use of the adaptor (batch_size=1)."
    ),
    "table9": (
        "Table 9. Anonymous user study to assess the qualitative performance of our "
        "method (ANT) in comparison to DDPM-PA."
    ),
}


# ---------------------------------------------------------------------------
# Core metric computation functions
# ---------------------------------------------------------------------------

def compute_fid(
    real_features: "np.ndarray",  # type: ignore[name-defined]
    fake_features: "np.ndarray",  # type: ignore[name-defined]
) -> float:
    """
    Compute Fréchet Inception Distance between real and fake InceptionV3 features.

    FID = ||μ_r - μ_f||² + Tr(Σ_r + Σ_f - 2·(Σ_r Σ_f)^{1/2})

    reference_grounding: paper_method_core dpms_ant/evaluation/fid.py
    """
    import numpy as np
    from scipy import linalg  # type: ignore[import]

    mu_r = np.mean(real_features, axis=0)
    mu_f = np.mean(fake_features, axis=0)
    sigma_r = np.cov(real_features, rowvar=False)
    sigma_f = np.cov(fake_features, rowvar=False)

    diff = mu_r - mu_f
    mean_sq = float(np.dot(diff, diff))

    # Numerical stable sqrt of product of covariances
    covmean, _ = linalg.sqrtm(sigma_r @ sigma_f, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    trace_term = float(np.trace(sigma_r + sigma_f - 2.0 * covmean))
    fid_value = mean_sq + trace_term
    return float(fid_value)


def compute_intra_lpips(
    images: List[Any],
    device: str = "cpu",
    subsample: int = 256,
) -> float:
    """
    Compute Intra-LPIPS: mean pairwise LPIPS distance over a set of generated images.
    Higher = more diverse (↑).

    For efficiency, subsample `subsample` unique random pairs if len(images) > sqrt(subsample).

    reference_grounding: paper_method_core dpms_ant/evaluation/metrics.py
    """
    try:
        import lpips as lpips_lib  # type: ignore[import]
        import torch                # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "lpips and torch are required for Intra-LPIPS computation. "
            f"Original error: {exc}"
        ) from exc

    import random
    loss_fn = lpips_lib.LPIPS(net="alex").to(device)
    loss_fn.eval()

    n = len(images)
    if n < 2:
        return 0.0

    # Build random pairs (without replacement)
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if len(all_pairs) > subsample:
        all_pairs = random.sample(all_pairs, subsample)

    distances = []
    with torch.no_grad():
        for i, j in all_pairs:
            img_i = _to_lpips_tensor(images[i], device)
            img_j = _to_lpips_tensor(images[j], device)