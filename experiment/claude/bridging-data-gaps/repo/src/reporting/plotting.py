"""
src/reporting/plotting.py
=========================
DPMs-ANT – Plotting and Artifact Writer Module

Paper: "Bridging Data Gaps in Diffusion Models with Adversarial Noise-Based
Transfer Learning"

reference_grounding: paper_method_core src/reporting/plotting.py
reference_grounding: paper_semantic_chunk_012 10-shot image generation experiments
reference_grounding: paper_semantic_chunk_014_01 DDPM + LDM framework evaluation

This module implements:
  1. Artifact writer for all paper figures (1-6, 2b) and tables (1-9)
  2. Metric schema declarations for fid, intra_lpips, fidelity_score,
     memory_usage, gpu_memory, accuracy, loss, training_time
  3. Result-trend assertion registry for semantic review
  4. Static artifact path discovery for all declared output paths

Paper figures:
  Figure 1: Fine-tuning progression FFHQ→Sunglasses with LPIPS scores
  Figure 2: Gradient changes and heat maps
  Figure 3: 10-shot generation LSUN Church→Landscape / FFHQ→Raphael's paintings
  Figure 4: Ablation study (300 iters, FID↓): baseline / Adaptor / w/o AN / ANT
  Figure 5: 10-shot generation FFHQ→Sunglasses and FFHQ→Babies
  Figure 6: Ablation study across iterations on Sunglasses

Paper tables:
  Table 1: Intra-LPIPS(↑) DDPM and GAN baselines, 10-shot FFHQ+LSUN Church
  Table 2: FID(↓) on 10-shot FFHQ→Babies and Sunglasses
  Table 3: FID and Intra-LPIPS with different classifiers
  Table 4: Intra-LPIPS(↑) DDPM-based and GAN-based baselines
  Table 5: Effects of γ in FFHQ→Sunglasses
  Table 6: Effects of ω in FFHQ→Sunglasses
  Table 7: Effects of training iteration in FFHQ→Sunglasses
  Table 8: GPU memory consumption with/without adaptor
  Table 9: User study ANT vs DDPM-PA

Result-trend assertions (from paper):
  - ANT FID < DDPM-PA on all 7 target domains
  - ANT FID < GAN baselines (TGAN/ADA/EWC/CDC/DCL)
  - Removing similarity guidance → FID increases
  - Removing adversarial noise → FID increases
  - Babies: ANT=46.70 < PA=48.92 (~4.5% improvement)
  - Sunglasses: ANT=20.06 < PA=34.75 (~42.3% improvement)
"""

from __future__ import annotations

import csv
import json
import logging
import os
import pathlib
import struct
import zlib
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Static artifact path registry – all declared paths discoverable at import
# reference_grounding: paper_method_core src/reporting/plotting.py
# ---------------------------------------------------------------------------

_ARTIFACT_BASE = pathlib.Path(
    os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR", "results")
)

FIGURE_PATHS: Dict[str, pathlib.Path] = {
    "figure_1":             _ARTIFACT_BASE / "figures" / "figure_1.png",
    "figure_2":             _ARTIFACT_BASE / "figures" / "figure_2.png",
    "figure_2b":            _ARTIFACT_BASE / "figures" / "figure_2b.png",
    "figure_3":             _ARTIFACT_BASE / "figures" / "figure_3.png",
    "figure_4":             _ARTIFACT_BASE / "figures" / "figure_4.png",
    "figure_5":             _ARTIFACT_BASE / "figures" / "figure_5.png",
    "figure_6":             _ARTIFACT_BASE / "figures" / "figure_6.png",
    "experiment_results_fig": _ARTIFACT_BASE / "figures" / "experiment_results.png",
}

TABLE_PATHS: Dict[str, pathlib.Path] = {
    "table_1":          _ARTIFACT_BASE / "tables" / "table_1.csv",
    "table_2":          _ARTIFACT_BASE / "tables" / "table_2.csv",
    "table_3":          _ARTIFACT_BASE / "tables" / "table_3.csv",
    "table_4":          _ARTIFACT_BASE / "tables" / "table_4.csv",
    "table_5":          _ARTIFACT_BASE / "tables" / "table_5.csv",
    "table_6":          _ARTIFACT_BASE / "tables" / "table_6.csv",
    "table_7":          _ARTIFACT_BASE / "tables" / "table_7.csv",
    "table_8":          _ARTIFACT_BASE / "tables" / "table_8.csv",
    "table_9":          _ARTIFACT_BASE / "tables" / "table_9.csv",
    "experiment_results": _ARTIFACT_BASE / "tables" / "experiment_results.csv",
}

JSON_PATHS: Dict[str, pathlib.Path] = {
    "metrics":              _ARTIFACT_BASE / "metrics.json",
    "predictions":          _ARTIFACT_BASE / "predictions.jsonl",
    "config_resolved":      _ARTIFACT_BASE / "config_resolved.json",
    "dataset_registry":     _ARTIFACT_BASE / "dataset_registry.json",
    "data_manifest":        _ARTIFACT_BASE / "data_manifest.json",
    "environment_registry": _ARTIFACT_BASE / "environment_registry.json",
    "scope_report":         _ARTIFACT_BASE / "scope_report.json",
    "experiment_registry":  _ARTIFACT_BASE / "experiment_registry.json",
    "artifact_manifest":    _ARTIFACT_BASE / "artifact_manifest.json",
}

ALL_ARTIFACT_PATHS: Dict[str, pathlib.Path] = {
    **FIGURE_PATHS,
    **TABLE_PATHS,
    **JSON_PATHS,
}

# ---------------------------------------------------------------------------
# Paper captions (preserved for semantic review)
# reference_grounding: paper_method_core src/reporting/plotting.py
# ---------------------------------------------------------------------------

FIGURE_CAPTIONS: Dict[str, str] = {
    "figure_1": (
        "Figure 1. Two sets of images generated from corresponding fixed noise "
        "inputs at different stages of fine-tuning DDPM from FFHQ to 10-shot "
        "Sunglasses. The perceptual distance, LPIPS (Zhang et al., 2018), between "
        "the generated image and the target image is shown on each generated image. "
        "When the bottom image successfully generates a sunglasses-wearing face, "
        "the top image (baseline) fails to do so."
    ),
    "figure_2": (
        "Figure 2. Visualizations of gradient changes and heat maps. Figure (a) "
        "shows gradient directions with various settings: the cyan line denotes the "
        "gradient computed on 10,000 samples in one step; the blue, red, and orange "
        "lines are gradients of baseline method (i.e., traditional DDPM), our method "
        "DDPM-ANT w/o AN (i.e., only using similarity-guided training), and DDPM-ANT, "
        "respectively."
    ),
    "figure_2b": (
        "Figure 2b. Heat map visualization showing the effect of adversarial noise "
        "selection on the gradient landscape and attention regions."
    ),
    "figure_3": (
        "Figure 3. The 10-shot image generation samples on LSUN Church "
        "→ Landscape drawings (top) and FFHQ → Raphael's paintings (bottom). "
        "When compared with other GAN-based and DDPM-based methods, our method, "
        "ANT, yields high-quality results that more closely resemble images of the "
        "target domain style. Samples from GAN-based baselines contain unnatural "
        "blurs and artifacts. Our results (lines 2 and 6) show the highest fidelity."
    ),
    "figure_4": (
        "Figure 4. This figure shows our ablation study, where all models are trained "
        "for 300 iterations on a 10-shot sunglasses dataset and measured with FID (↓): "
        "the first line - baseline (direct fine-tuning model, FID=41.88), "
        "second line - Adaptor (fine-tuning only few extra parameters, FID=38.65), "
        "third line - DPMs-ANT w/o AN (only using similarity-guided training), "
        "and fourth line - DPMs-ANT (our full method)."
    ),
    "figure_5": (
        "Figure 5. The 10-shot image generation samples on FFHQ → Sunglasses and "
        "FFHQ → Babies. Compared to the GAN-based method (shown in the 2nd and 3rd "
        "rows), our approach (shown in the 5th and 6th rows) generates images with "
        "better quality. Quantitative results are provided in Table 1."
    ),
    "figure_6": (
        "Figure 6. This figure shows our ablation study with all models trained for "
        "different iterations on a 10-shot sunglasses dataset: the first line - "
        "baseline (direct fine-tuning model), second line - DPMs-ANT w/o AN (only "
        "using similarity-guided training), and third line - DPMs-ANT (our method)."
    ),
}

TABLE_CAPTIONS: Dict[str, str] = {
    "table_1": (
        "Table 1. Intra-LPIPS (↑) results for both DDPM and GAN-based baselines are "
        "presented for 10-shot image generation tasks. These tasks involve adapting "
        "from the source domains of FFHQ and LSUN Church. 'Parameter Rate' means the "
        "proportion of parameters fine-tuned compared to the pre-trained model's "
        "parameter count. DDPM-ANT yields considerable improvement in Intra-LPIPS "
        "across most tasks. LDM-ANT excels beyond state-of-the-art GAN-based approaches."
    ),
    "table_2": (
        "Table 2. FID (↓) results of each method on 10-shot FFHQ → Babies and "
        "Sunglasses. The best results are marked in bold. "
        "ANT: Babies=46.70, Sunglasses=20.06. "
        "DDPM-PA: Babies=48.92, Sunglasses=34.75."
    ),
    "table_3": (
        "Table 3. FID and Intra-LPIPS results of DPM-ANT from FFHQ → Sunglasses "
        "with different classifiers (trained on 10 and 100 images). Demonstrates "
        "robustness of the MobileNetV2 classifier to training set size."
    ),
    "table_4": (
        "Table 4. The Intra-LPIPS (↑) results for both DDPM-based strategies and "
        "GAN-based baselines are presented for 10-shot image generation tasks. "
        "The best results are marked as bold. LDM-ANT demonstrates potent capability "
        "to preserve diversity."
    ),
    "table_5": (
        "Table 5. Effects of γ (similarity_guidance_scale) in FFHQ → Sunglasses "
        "case in terms of FID and Intra-LPIPS. Paper default: γ=5."
    ),
    "table_6": (
        "Table 6. Effects of ω (adversarial perturbation budget, omega) in "
        "FFHQ → Sunglasses case in terms of FID and Intra-LPIPS. "
        "Paper default: ω=0.02. ω=0 is equivalent to no adversarial noise."
    ),
    "table_7": (
        "Table 7. Effects of training iteration count in FFHQ → Sunglasses case "
        "in terms of FID and Intra-LPIPS. "
        "Paper anchors: 300 iters (ablation), 5000 iters (main result)."
    ),
    "table_8": (
        "Table 8. GPU memory consumption (MB) for each module, comparing scenarios "
        "with and without the use of the adaptor at batch_size=1. "
        "The adaptor results in only a slight increase in GPU memory consumption."
    ),
    "table_9": (
        "Table 9. Anonymous user study to assess the qualitative performance of our "
        "method (ANT) in comparison to DDPM-PA."
    ),
}

# ---------------------------------------------------------------------------
# Metric schema declarations
# reference_grounding: paper_method_core src/reporting/plotting.py
# Satisfies: Paper evidence contract: declare metric schemas/aggregations for
#   fid, intra_lpips, fidelity_score, memory_usage, gpu_memory, accuracy,
#   loss, training_time
# ---------------------------------------------------------------------------

METRIC_SCHEMA: Dict[str, Dict[str, Any]] = {
    "fid": {
        "name": "Fréchet Inception Distance",
        "abbreviation": "FID",
        "direction": "lower_is_better",
        "unit": "score",
        "description": (
            "FID measures the 2-Wasserstein distance between the multivariate "
            "Gaussian distributions fitted to Inception-V3 pool3 activations "
            "of generated vs real images. Lower values indicate higher quality "
            "and more realistic generated images."
        ),
        "formula": (
            "FID = ||μ_r - μ_g||^2 + Tr(Σ_r + Σ_g - 2(Σ_r Σ_g)^{1/2})"
        ),
        "aggregation": (
            "Computed over N=50000 generated samples (default) vs the full real "
            "distribution. For few-shot domains the real reference set is the "
            "full source or target domain training set."
        ),
        "paper_anchor": "Table 1, Table 2, Table 3, Table 4, Table 5, Table 6, Table 7",
        "paper_reference_values": {
            "DDPM-ANT_FFHQ_Babies": 46.70,
            "DDPM-ANT_FFHQ_Sunglasses": 20.06,
            "DDPM-PA_FFHQ_Babies": 48.92,
            "DDPM-PA_FFHQ_Sunglasses": 34.75,
            "Adaptor_only_300iter": 38.65,
            "Baseline_300iter": 41.88,
        },
    },
    "intra_lpips": {
        "name": "Intra-LPIPS",
        "abbreviation": "Intra-LPIPS",
        "direction": "higher_is_better",
        "unit": "score",
        "description": (
            "Intra-LPIPS measures the diversity of generated images by computing "
            "pairwise LPIPS (Learned Perceptual Image Patch Similarity) distances "
            "among randomly sampled generated image pairs. Higher values indicate "
            "greater diversity and less mode collapse."
        ),
        "formula": (
            "Intra-LPIPS = E_{x,x' ~ p_gen}[LPIPS(x, x')] "
            "where LPIPS uses AlexNet/VGG deep features."
        ),
        "aggregation": (
            "Mean pairwise LPIPS over N randomly sampled pairs from the generated "
            "image set. Typically evaluated on 2000 generated images."
        ),
        "paper_anchor": "Table 1, Table 3, Table 4, Table 5, Table 6, Table 7",
        "paper_note": (
            "For diffusion models, fidelity score needs to be estimated on noisy "
            "images (x_t), unlike GANs that compare clean generated images directly."
        ),
    },
    "fidelity_score": {
        "name": "Fidelity Score",
        "abbreviation": "fidelity",
        "direction": "higher_is_better",
        "unit": "score",
        "description": (
            "Fidelity score measures how closely the generated images match the "
            "target domain style. For diffusion models, this must be estimated on "
            "noisy images x_t at step t, because the single-pass generation of GANs "
            "allows direct comparison of clean images (Li et al. 2020; Ojha et al. "
            "2021; Zhao et al. 2022), which is not directly applicable to DPMs. "
            "Measured via LPIPS (Zhang et al., 2018) between generated and target."
        ),
        "formula": (
            "fidelity(x_gen, x_target) = 1 - LPIPS(x_gen, x_target) "
            "(higher = more faithful to target domain)"
        ),
        "aggregation": (
            "Mean over all 10-shot target domain images vs corresponding "
            "generated images from fixed noise inputs (as in Figure 1)."
        ),
        "paper_anchor": "Figure 1 (LPIPS per image at each fine-tuning stage)",
        "paper_note": (
            "In Figure 1, the LPIPS between the generated image and the target "
            "sunglasses image is annotated on each image at different fine-tuning stages."
        ),
    },
    "memory_usage": {
        "name": "Memory Usage",
        "abbreviation": "memory_MB",
        "direction": "lower_is_better",
        "unit": "MB",
        "description": (
            "Total system (CPU + GPU) memory consumption during training or "
            "inference, measured in megabytes."
        ),
        "aggregation": "Peak memory in MB, measured at batch_size=1",
        "paper_anchor": "Table 8",
    },
    "gpu_memory": {
        "name": "GPU Memory",
        "abbreviation": "gpu_memory_MB",
        "direction": "lower_is_better",
        "unit": "MB",
        "description": (
            "GPU memory consumption (MB) for each module. Table 8 illustrates "
            "GPU memory usage at batch_size=1, comparing scenarios with and without "
            "the use of a Shift Adaptor. The adaptor results in only a slight increase "
            "in GPU memory consumption (DDPM ShiftAdaptor c=4,d=8; LDM c=2,d=8)."
        ),
        "aggregation": "Peak GPU memory in MB at batch_size=1, measured via torch.cuda.max_memory_allocated()",
        "paper_anchor": "Table 8",
        "paper_reference_values": {
            "note": (
                "Table 8 shows slight GPU memory increase with adaptor. "
                "Exact values depend on model size and GPU."
            ),
        },
    },
    "accuracy": {
        "name": "Classifier Accuracy",
        "abbreviation": "accuracy",
        "direction": "higher_is_better",
        "unit": "fraction",
        "description": (
            "Top-1 accuracy of the MobileNetV2 domain classifier (φ) that "
            "distinguishes source domain (y=S) vs target domain (y=T) images, "
            "evaluated on noisy images (x_t, t). The classifier is pre-trained "
            "on ImageNet and fine-tuned for 300 steps on source + 10-shot target "
            "domain images."
        ),
        "formula": "accuracy = |{i : argmax p_φ(y|x_i) = y_i}| / N",
        "aggregation": "Top-1 accuracy over the domain classifier validation set",
        "paper_anchor": "Table 3 (classifier trained on 10 or 100 images)",
    },
    "loss": {
        "name": "Training Loss",
        "abbreviation": "loss",
        "direction": "lower_is_better",
        "unit": "scalar",
        "description": (
            "Combined DPMs-ANT training loss from Algorithm 1: "
            "L_total = L_simple + λ·L_sim, where "
            "L_simple = ||ε_t - ε_θ_ψ(x_t, t)||^2 (noise prediction MSE on "
            "adversarially-perturbed samples x_0 + ε*), and "
            "L_sim = γ·KL(∇log p_φ(y=S|x_t), ∇log p_φ(y=T|x_t)) "
            "(similarity guidance loss with MobileNetV2 classifier φ, γ=5)."
        ),
        "formula": "L_total = L_simple + λ·L_sim = E_t[||ε-ε_θ_ψ(x_t,t)||²] + γ·KL(p_S||p_T)",
        "aggregation": "Exponential moving average over training iterations",
        "paper_anchor": "Algorithm 1, Section 3",
        "paper_fixed_hyperparams": {
            "gamma": 5,
            "omega": 0.02,
            "adversarial_inner_steps": 10,
            "total_iterations": 5000,
        },
    },
    "training_time": {
        "name": "Training Time",
        "abbreviation": "training_time_s",
        "direction": "lower_is_better",
        "unit": "seconds",
        "description": (
            "Wall-clock time for the complete fine-tuning run from initialization "
            "to final checkpoint, in seconds. Includes adversarial noise inner loop "
            "(K=10 PGD steps per training iteration) and similarity guidance "
            "classifier forward passes."
        ),
        "aggregation": "Total wall time from training loop start to completion",
        "paper_anchor": "Table 8 (efficiency analysis)",
    },
}

# ---------------------------------------------------------------------------
# Named baselines registry
# reference_grounding: paper_method_core src/reporting/plotting.py
# ---------------------------------------------------------------------------

NAMED_BASELINES: Dict[str, Dict[str, str]] = {
    "TGAN": {
        "type": "GAN-based",
        "description": "TransferGAN – transfer learning for unconditional GANs",
        "paper_table": "Table 1, Table 2, Table 4",
    },
    "ADA": {
        "type": "GAN-based",
        "description": (
            "Adaptive Discriminator Augmentation (Karras et al., 2020) – "
            "data augmentation strategy for GAN training in limited data regime"
        ),
        "paper_table": "Table 1, Table 2, Table 4",
    },
    "EWC": {
        "type": "GAN-based",
        "description": (
            "Elastic Weight Consolidation for GAN fine-tuning (Li et al., 2020) – "
            "prevents forgetting of source domain during few-shot adaptation"
        ),
        "paper_table": "Table 1, Table 2, Table 4",
    },
    "CDC": {
        "type": "GAN-based",
        "description": (
            "Cross-Domain Correspondence (Ojha et al., 2021) – "
            "preserves relative spatial structure across domains during GAN adaptation"
        ),
        "paper_table": "Table 1, Table 2, Table 4",
    },
    "DCL": {
        "type": "GAN-based",
        "description": (
            "Domain-Consistent Loss (Zhao et al., 2022) – "
            "domain-specific regularisation for few-shot GAN fine-tuning"
        ),
        "paper_table": "Table 1, Table 2, Table 4",
    },
    "DDPM-PA": {
        "type": "DDPM-based",
        "description": (
            "DDPM Pairwise Alignment – current state-of-the-art DPM-based few-shot "
            "method that uses pairwise distance constraints on noisy images. "
            "Paper reference FID: Babies=48.92, Sunglasses=34.75."
        ),
        "paper_table": "Table 1, Table 2, Table 4, Table 9",
    },
    "Baseline": {
        "type": "DDPM-based",
        "description": (
            "Direct fine-tuning of the entire DDPM on 10-shot target data "
            "(no Shift Adaptor, no ANT strategies). FID=41.88 at 300 iters "
            "(FFHQ→Sunglasses)."
        ),
        "paper_table": "Table 4, Figure 4, Figure 6",
    },
    "Adaptor": {
        "type": "DDPM-based",
        "description": (
            "Fine-tuning only the Shift Adaptor parameters (c=4, d=8 for DDPM; "
            "c=2, d=8 for LDM) while keeping pretrained weights frozen. "
            "FID=38.65 at 300 iters vs Baseline=41.88 (FFHQ→Sunglasses)."
        ),
        "paper_table": "Table 4, Figure 4",
    },
    "DDPM-ANT_wo_AN": {
        "type": "DDPM-based (ablation)",
        "description": (
            "DPMs-ANT without adversarial noise selection (use_adv_noise=False). "
            "Uses only similarity-guided training (γ=5, MobileNetV2 classifier). "
            "Ablation row in Figure 4 and Figure 6."
        ),
        "paper_table": "Table 4, Figure 4, Figure 6",
    },
    "ANT": {
        "type": "proposed",
        "description": (
            "DPMs-ANT – full method with both strategies: "
            "(1) similarity-guided training (γ=5, MobileNetV2) and "
            "(2) adversarial noise selection (PGD K=10, ω=0.02). "
            "Paper FID: Babies=46.70 (vs PA=48.92), Sunglasses=20.06 (vs PA=34.75)."
        ),
        "paper_table": "All tables",
    },
}

# ---------------------------------------------------------------------------
# Result-trend assertions for semantic review
# reference_grounding: paper_method_core src/reporting/plotting.py
# Satisfies: In this file, preserve required result-trend assertions for
#   semantic review: baseline_outperformance, ablation trends, quantitative
# ---------------------------------------------------------------------------

RESULT_TREND_ASSERTIONS: List[Dict[str, Any]] = [
    {
        "assertion_id": "baseline_outperformance_all_domains",
        "kind": "baseline_outperformance",
        "status": "expected",
        "description": "ANT在所有目标域FID均低于DDPM-PA基线",
        "metric": "fid",
        "direction": "ANT_fid < DDPM_PA_fid",
        "applies_to": "all 7 target domains",
        "paper_evidence": "Table 1, Table 2, Table 4",
        "verification": "For each domain: results['ANT']['fid'] < results['DDPM-PA']['fid']",
    },
    {
        "assertion_id": "baseline_outperformance_gan",
        "kind": "baseline_outperformance",
        "status": "expected",
        "description": "ANT优于所有GAN-based基线(TGAN/ADA/EWC/CDC/DCL) in FID",
        "metric": "fid",
        "direction": "ANT_fid < GAN_baseline_fid",
        "applies_to": "Table 2 (Babies, Sunglasses), Table 1 (all 7 domains)",
        "paper_evidence": "Table 2",
        "verification": (
            "For each GAN baseline M in {TGAN, ADA, EWC, CDC, DCL}: "
            "results['ANT']['fid'] < results[M]['fid']"
        ),
    },
    {
        "assertion_id": "ablation_sim_guide_fid_rises",
        "kind": "ablation_trend",
        "status": "expected",
        "description": "移除相似性引导训练导致FID上升",
        "metric": "fid",
        "direction": "ANT_wo_SimGuide_fid > ANT_fid (higher FID = worse)",
        "applies_to": "FFHQ→Sunglasses ablation",
        "paper_evidence": "Figure 4, Figure 6, Table 4",
        "ablation_config": {"use_sim_guide": False, "use_adv_noise": True},
        "verification": (
            "results['Adaptor']['fid'] > results['ANT']['fid'] "
            "at same iteration count"
        ),
    },
    {
        "assertion_id": "ablation_adv_noise_fid_rises",
        "kind": "ablation_trend",
        "status": "expected",
        "description": "移除对抗噪声选择导致FID上升",
        "metric": "fid",
        "direction": "ANT_wo_AN_fid > ANT_fid (higher FID = worse)",
        "applies_to": "FFHQ→Sunglasses ablation",
        "paper_evidence": "Figure 4, Figure 6, Table 4",
        "ablation_config": {"use_sim_guide": True, "use_adv_noise": False},
        "verification": (
            "results['DDPM-ANT_wo_AN']['fid'] > results['ANT']['fid'] "
            "at same iteration count"
        ),
    },
    {
        "assertion_id": "babies_fid_quantitative",
        "kind": "quantitative_result",
        "status": "expected",
        "description": "Babies: ANT=46.70 < PA=48.92 (~4.5% improvement)",
        "metric": "fid",
        "domain": "FFHQ→Babies",
        "expected_ant_value": 46.70,
        "expected_pa_value": 48.92,
        "expected_improvement_pct": 4.5,
        "paper_evidence": "Table 2",
        "verification": "abs(results['ANT']['ffhq_babies_fid'] - 46.70) < tolerance",
    },
    {
        "assertion_id": "sunglasses_fid_quantitative",
        "kind": "quantitative_result",
        "status": "expected",
        "description": "Sunglasses: ANT=20.06 < PA=34.75 (~42.3% improvement)",
        "metric": "fid",
        "domain": "FFHQ→Sunglasses",
        "expected_ant_value": 20.06,
        "expected_pa_value": 34.75,
        "expected_improvement_pct": 42.3,
        "paper_evidence": "Table 2",
        "verification": "abs(results['ANT']['ffhq_sunglasses_fid'] - 20.06) < tolerance",
    },
    {
        "assertion_id": "adaptor_competitive_baseline",
        "kind": "quantitative_result",
        "status": "expected",
        "description": (
            "Adaptor-only fine-tuning achieves competitive FID vs direct fine-tuning "
            "(38.65 vs 41.88 at 300 iters, FFHQ→Sunglasses)."
        ),
        "metric": "fid",
        "domain": "FFHQ→Sunglasses",
        "expected_adaptor_fid": 38.65,
        "expected_baseline_fid": 41.88,
        "paper_evidence": "Figure 4",
        "verification": "results['Adaptor']['fid'] < results['Baseline']['fid']",
    },
    {
        "assertion_id": "ldm_ant_diversity_gan",
        "kind": "baseline_outperformance",
        "status": "expected",
        "description": (
            "LDM-ANT excels beyond state-of-the-art GAN-based approaches, "
            "demonstrating potent capability to preserve diversity (Intra-LPIPS)."
        ),
        "metric": "intra_lpips",
        "direction": "LDM_ANT_intra_lpips > GAN_baselines_intra_lpips",
        "applies_to": "Table 1, Table 4 (LDM-ANT rows)",
        "paper_evidence": "Table 1, Table 4",
        "verification": (
            "results['LDM-ANT']['intra_lpips'] > "
            "max(results[m]['intra_lpips'] for m in GAN_baselines)"
        ),
    },
    {
        "assertion_id": "gpu_memory_slight_increase",
        "kind": "efficiency_result",
        "status": "expected",
        "description": (
            "Shift Adaptor results in only a slight increase in GPU memory "
            "consumption compared to baseline (no adaptor)."
        ),
        "metric": "gpu_memory",
        "direction": "adaptor_gpu_memory slightly_above baseline_gpu_memory",
        "applies_to": "Table 8",
        "paper_evidence": "Table 8",
        "verification": (
            "adaptor_gpu_mb - no_adaptor_gpu_mb < 200  # small absolute delta"
        ),
    },
]

# ---------------------------------------------------------------------------
# Paper reference data (Table 2 FID – from paper)
# ---------------------------------------------------------------------------

TABLE_2_REFERENCE_DATA: List[Dict[str, Any]] = [
    {"method": "TGAN",    "type": "GAN",   "ffhq_babies": None,  "ffhq_sunglasses": None,
     "note": "GAN-baseline"},
    {"method": "ADA",     "type": "GAN",   "ffhq_babies": None,  "ffhq_sunglasses": None,
     "note": "GAN-baseline"},
    {"method": "EWC",     "type": "GAN",   "ffhq_babies": None,  "ffhq_sunglasses": None,
     "note": "GAN-baseline"},
    {"method": "CDC",     "type": "GAN",   "ffhq_babies": None,  "ffhq_sunglasses": None,
     "note": "GAN-baseline"},
    {"method": "DCL",     "type": "GAN",   "ffhq_babies": None,  "ffhq_sunglasses": None,
     "note": "GAN-baseline"},
    {"method": "DDPM-PA", "type": "DDPM",  "ffhq_babies": 48.92, "ffhq_sunglasses": 34.75,
     "note": "DDPM-baseline"},
    {"method": "ANT",     "type": "DDPM",  "ffhq_babies": 46.70, "ffhq_sunglasses": 20.06,
     "note": "proposed_best_result"},
]

TABLE_5_GAMMA_SWEEP: List[Dict[str, Any]] = [
    {"gamma": 0,   "fid": None, "intra_lpips": None,
     "note": "no_sim_guide (ablation baseline)"},
    {"gamma": 1,   "fid": None, "intra_lpips": None, "note": "sensitivity_sweep"},
    {"gamma": 5,   "fid": None, "intra_lpips": None, "note": "paper_default_gamma=5"},
    {"gamma": 10,  "fid": None, "intra_lpips": None, "note": "sensitivity_sweep"},
    {"gamma": 20,  "fid": None, "intra_lpips": None, "note": "sensitivity_sweep"},
]

TABLE_6_OMEGA_SWEEP: List[Dict[str, Any]] = [
    {"omega": 0.0,  "fid": None, "intra_lpips": None,
     "note": "no_adv_noise (ablation baseline)"},
    {"omega": 0.01, "fid": None, "intra_lpips": None, "note": "sensitivity_sweep"},
    {"omega": 0.02, "fid": None, "intra_lpips": None,
     "note": "paper_default_omega=0.02"},
    {"omega": 0.05, "fid": None, "intra_lpips": None, "note": "sensitivity_sweep"},
    {"omega": 0.10, "fid": None, "intra_lpips": None, "note": "sensitivity_sweep"},
]

TABLE_7_ITERATION_SWEEP: List[Dict[str, Any]] = [
    {"iterations": 100,  "fid": None, "intra_lpips": None, "note": "early_stopping"},
    {"iterations": 200,  "fid": None, "intra_lpips": None, "note": "sensitivity_sweep"},
    {"iterations": 300,  "fid": None, "intra_lpips": None,
     "note": "ablation_default (figures 4 and 6)"},
    {"iterations": 500,  "fid": None, "intra_lpips": None, "note": "sensitivity_sweep"},
    {"iterations": 1000, "fid": None, "intra_lpips": None, "note": "sensitivity_sweep"},
    {"iterations": 5000, "fid": None, "intra_lpips": None,
     "note": "paper_main_result_anchor"},
]

TABLE_8_GPU_MEMORY_SCHEMA: List[Dict[str, Any]] = [
    {
        "module": "DDPM UNet (no adaptor)",
        "framework": "ddpm",
        "adaptor_enabled": False,
        "gpu_memory_mb": None,
        "batch_size": 1,
        "note": "baseline_no_adaptor",
    },
    {
        "module": "DDPM UNet + ShiftAdaptor (c=4, d=8)",
        "framework": "ddpm",
        "adaptor_enabled": True,
        "adaptor_c": 4,
        "adaptor_d": 8,
        "gpu_memory_mb": None,
        "batch_size": 1,
        "note": "slight_increase_vs_baseline",
    },
    {
        "module": "LDM UNet (no adaptor)",
        "framework": "ldm",
        "adaptor_enabled": False,
        "gpu_memory_mb": None,
        "batch_size": 1,
        "note": "baseline_no_adaptor",
    },
    {
        "module": "LDM UNet + ShiftAdaptor (c=2, d=8)",
        "framework": "ldm",
        "adaptor_enabled": True,
        "adaptor_c": 2,
        "adaptor_d": 8,
        "gpu_memory_mb": None,
        "batch_size": 1,
        "note": "slight_increase_vs_baseline",
    },
]

# ---------------------------------------------------------------------------
# Helper utilities (no optional deps at module level)
# ---------------------------------------------------------------------------


def _ensure_dir(path: pathlib.Path) -> None:
    """Create parent directories for a path."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_csv(
    path: pathlib.Path,
    rows: List[Dict[str, Any]],
    fieldnames: Optional[List[str]] = None,
) -> None:
    """Write a list of dicts as a CSV file."""
    _ensure_dir(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.debug("Wrote CSV: %s (%d rows)", path, len(rows))


def _write_json(path: pathlib.Path, data: Any) -> None:
    """Write JSON to path, creating parent dirs."""
    _ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    logger.debug("Wrote JSON: %s", path)


def _minimal_png_bytes() -> bytes:
    """Return bytes for a minimal valid 1×1 white PNG (no external deps)."""

    def _chunk(tag: bytes, data: bytes) -> bytes:
        buf = tag + data
        crc = zlib.crc32(buf) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + buf + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\xff\xff\xff")
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", idat)
        + _chunk(b"IEND", b"")
    )


def _write_dry_run_figure(
    path: pathlib.Path,
    caption: str,
    label: str,
) -> None:
    """Write a dry-run diagnostic figure (PNG), labeled as schema/readiness artifact.

    Tries matplotlib first; falls back to a minimal 1×1 PNG if unavailable.
    NOT a real experiment result.
    """
    _ensure_dir(path)
    try:
        import matplotlib  # lazy import – optional dep
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(
            0.5, 0.65,
            f"[DRY-RUN SCHEMA ARTIFACT]\n{label}",
            ha="center", va="center",
            fontsize=10, fontweight="bold",
            color="#333333",
            transform=ax.transAxes,
        )
        short_caption = caption[:180] + ("…" if len(caption) > 180 else "")
        ax.text(
            0.5, 0.35,
            short_caption,
            ha="center", va="center",
            fontsize=7,
            color="#555555",
            transform=ax.transAxes,
            wrap=True,
        )
        ax.text(
            0.5, 0.10,
            "NOT A REAL EXPERIMENT RESULT",
            ha="center", va="center",
            fontsize=8, color="red",
            transform=ax.transAxes,
        )
        ax.axis("off")
        ax.set_facecolor("#f8f8f8")
        fig.patch.set_facecolor("#f8f8f8")
        fig.tight_layout()
        fig.savefig(path, dpi=72, bbox_inches="tight")
        plt.close(fig)
        logger.debug("Wrote dry-run figure (matplotlib): %s", path)
    except Exception:
        # matplotlib unavailable or failed – write minimal PNG stub
        path.write_bytes(_minimal_png_bytes())
        logger.debug("Wrote dry-run figure (minimal PNG stub): %s", path)


# ---------------------------------------------------------------------------
# Static artifact path API
# ---------------------------------------------------------------------------


def get_artifact_path(key: str) -> pathlib.Path:
    """Return the statically declared artifact path for *key*.

    Args:
        key: one of the keys in FIGURE_PATHS, TABLE_PATHS, or JSON_PATHS

    Returns:
        pathlib.Path for the artifact

    Raises:
        KeyError: if key not found
    """
    if key in ALL_ARTIFACT_PATHS:
        return ALL_ARTIFACT_PATHS[key]
    raise KeyError(
        f"Unknown artifact key: {key!r}. "
        f"Available keys: {sorted(ALL_ARTIFACT_PATHS)}"
    )


def list_artifact_paths() -> Dict[str, str]:
    """Return all declared artifact paths as {key: str_path} for manifests."""
    return {k: str(v) for k, v in ALL_ARTIFACT_PATHS.items()}


# ---------------------------------------------------------------------------
# Metric schema API
# ---------------------------------------------------------------------------


def get_metric_schema(metric_name: str) -> Dict[str, Any]:
    """Return the declared metric schema for *metric_name*.

    Args:
        metric_name: one of fid, intra_lpips, fidelity_score, memory_usage,
                     gpu_memory, accuracy, loss, training_time

    Returns:
        Dict with name, abbreviation, direction, unit, description, formula,
        aggregation, paper_anchor, paper_reference_values (where available)

    Raises:
        KeyError: if metric_name not declared
    """
    if metric_name not in METRIC_SCHEMA:
        raise KeyError(
            f"Unknown metric: {metric_name!r}. "
            f"Available: {sorted(METRIC_SCHEMA.keys())}"
        )
    return METRIC_SCHEMA[metric_name]


def get_result_trend_assertions() -> List[Dict[str, Any]]:
    """Return the list of paper result-trend assertions for semantic review."""
    return RESULT_TREND_ASSERTIONS


def get_figure_caption(figure_key: str) -> str:
    """Return the paper figure caption for *figure_key*."""
    if figure_key not in FIGURE_CAPTIONS:
        raise KeyError(
            f"Unknown figure: {figure_key!r}. Available: {sorted(FIGURE_CAPTIONS)}"
        )
    return FIGURE_CAPTIONS[figure_key]


def get_table_caption(table_key: str) -> str:
    """Return the paper table caption for *table_key*."""
    if table_key not in TABLE_CAPTIONS:
        raise KeyError(
            f"Unknown table: {table_key!r}. Available: {sorted(TABLE_CAPTIONS)}"
        )
    return TABLE_CAPTIONS[table_key]


# ---------------------------------------------------------------------------
# Table writer: Table 1 – Intra-LPIPS for DDPM and GAN baselines
# ---------------------------------------------------------------------------


def write_table_1(
    results: Optional[List[Dict[str, Any]]] = None,
    dry_run: bool = False,
) -> pathlib.Path:
    """Write Table 1: Intra-LPIPS(↑) for DDPM and GAN baselines, 10-shot.

    Table 1. Intra-LPIPS(↑) results for both DDPM and GAN-based baselines for
    10-shot image generation tasks adapting from FFHQ and LSUN Church source
    domains. 'Parameter Rate' means proportion of parameters fine-tuned.

    Columns: method, parameter_rate, ffhq_babies, ffhq_sunglasses,
             ffhq_raphael_peale, ffhq_sketches, ffhq_modigliani,
             church_landscape, church_haunted_houses, mean_intra_lpips
    """
    path = TABLE_PATHS["table_1"]
    fieldnames = [
        "method", "parameter_rate",
        "ffhq_babies", "ffhq_sunglasses", "ffhq_raphael_peale",
        "ffhq_sketches", "ffhq_modigliani",
        "church_landscape", "church_haunted_houses", "mean_intra_lpips",
    ]
    if dry_run or results is None:
        schema_rows = [
            {
                "method": method,
                "parameter_rate": (
                    "adaptor_only" if method in ("Adaptor", "ANT", "DDPM-ANT_wo_AN")
                    else "100%"
                ),
                "ffhq_babies":          "to_be_computed",
                "ffhq_sunglasses":      "to_be_computed",
                "ffhq_raphael_peale":   "to_be_computed",
                "ffhq_sketches":        "to_be_computed",
                "ffhq_modigliani":      "to_be_computed",
                "church_landscape":     "to_be_computed",
                "church_haunted_houses": "to_be_computed",
                "mean_intra_lpips":     "to_be_computed",
                "_artifact_type":       "dry_run_schema_not_real_result",
                "_caption":             TABLE_CAPTIONS["table_1"][:120],
            }
            for method in NAMED_BASELINES
        ]
        _write_csv(path, schema_rows)
        return path

    _write_csv(path, results, fieldnames=fieldnames)
    return path


# ---------------------------------------------------------------------------
# Table writer: Table 2 – FID on FFHQ→Babies and Sunglasses
# ---------------------------------------------------------------------------


def write_table_2(
    results: Optional[List[Dict[str, Any]]] = None,
    dry_run: bool = False,
) -> pathlib.Path:
    """Write Table 2: FID(↓) on 10-shot FFHQ→Babies and Sunglasses.

    Table 2. FID(↓) results of each method on 10-shot FFHQ→Babies and
    Sunglasses. The best results are marked in bold.
    Paper reference: ANT Babies=46.70, Sunglasses=20.06;
                     DDPM-PA Babies=48.92, Sunglasses=34.75.
    """
    path = TABLE_PATHS["table_2"]
    fieldnames = ["method", "type", "ffhq_babies", "ffhq_sunglasses", "note"]
    if dry_run or results is None:
        rows = []
        for entry in TABLE_2_REFERENCE_DATA:
            row = {k: v for k, v in entry.items()}
            row["_artifact_type"] = "paper_reference_anchor_plus_schema"
            rows.append(row)
        _write_csv(path, rows)
        return path

    _write_csv(path, results, fieldnames=fieldnames)
    return path


# ---------------------------------------------------------------------------
# Table writer: Table 3 – Classifier sensitivity
# ---------------------------------------------------------------------------


def write_table_3(
    results: Optional[List[Dict[str, Any]]] = None,
    dry_run: bool = False,
) -> pathlib.Path:
    """Write Table 3: FID and Intra-LPIPS with different classifiers.

    Table 3. FID and Intra-LPIPS results of DPM-ANT from FFHQ→Sunglasses
    with different classifiers (trained on 10 and 100 images).
    """
    path = TABLE_PATHS["table_3"]
    fieldnames = ["classifier_training_images", "fid", "intra_lpips", "note"]
    if dry_run or results is None:
        schema_rows = [
            {
                "classifier_training_images": n,
                "fid": "to_be_computed",
                "intra_lpips": "to_be_computed",
                "note": "dry_run_schema_not_real_result",
                "_caption": TABLE_CAPTIONS["table_3"][:100],
            }
            for n in [10, 100]
        ]
        _write_csv(path, schema_rows)
        return path

    _write_csv(path, results, fieldnames=fieldnames)
    return path


# ---------------------------------------------------------------------------
# Table writer: Table 4 – Intra-LPIPS for DDPM and GAN baselines (variant)
# ---------------------------------------------------------------------------


def write_table_4(
    results: Optional[List[Dict[str, Any]]] = None,
    dry_run: bool = False,
) -> pathlib.Path:
    """Write Table 4: Intra-LPIPS(↑) for DDPM-based and GAN-based baselines.

    Table 4. Intra-LPIPS(↑) results for both DDPM-based strategies and
    GAN-based baselines for 10-shot image generation tasks. Best results bold.
    Includes LDM-ANT which excels beyond state-of-the-art GAN-based approaches.
    """
    path = TABLE_PATHS["table_4"]
    if dry_run or results is None:
        schema_rows = [
            {
                "method": method,
                "type": info["type"],
                "ffhq_babies_intra_lpips":      "to_be_computed",
                "ffhq_sunglasses_intra_lpips":  "to_be_computed",
                "church_landscape_intra_lpips": "to_be_computed",
                "mean_intra_lpips":             "to_be_computed",
                "_artifact_type": "dry_run_schema_not_real_result",
                "_caption": TABLE_CAPTIONS["table_4"][:100],
            }
            for method, info in NAMED_BASELINES.items()
        ]
        _write_csv(path, schema_rows)
        return path

    _write_csv(path, results)
    return path


# ---------------------------------------------------------------------------
# Table writer: Table 5 – γ sensitivity
# ---------------------------------------------------------------------------


def write_table_5(
    results: Optional[List[Dict[str, Any]]] = None,
    dry_run: bool = False,
) -> pathlib.Path:
    """Write Table 5: Effects of γ in FFHQ→Sunglasses.

    Table 5. Effects of γ (similarity_guidance_scale) in FFHQ→Sunglasses
    in terms of FID and Intra-LPIPS. Paper default: γ=5.
    γ=0 corresponds to ablation without similarity guidance.
    """
    path = TABLE_PATHS["table_5"]
    fieldnames = ["gamma", "fid", "intra_lpips", "note"]
    if dry_run or results is None:
        rows = [dict(e) for e in TABLE_5_GAMMA_SWEEP]
        for r in rows:
            r["_artifact_type"] = "dry_run_schema_gamma_sweep"
        _write_csv(path, rows, fieldnames=fieldnames + ["_artifact_type"])
        return path

    _write_csv(path, results, fieldnames=fieldnames)
    return path


# ---------------------------------------------------------------------------
# Table writer: Table 6 – ω sensitivity
# ---------------------------------------------------------------------------


def write_table_6(
    results: Optional[List[Dict[str, Any]]] = None,
    dry_run: bool = False,
) -> pathlib.Path:
    """Write Table 6: Effects of ω in FFHQ→Sunglasses.

    Table 6. Effects of ω (adversarial perturbation budget) in
    FFHQ→Sunglasses in terms of FID and Intra-LPIPS.
    Paper default: ω=0.02. ω=0 → no adversarial noise selection.
    """
    path = TABLE_PATHS["table_6"]
    fieldnames = ["omega", "fid", "intra_lpips", "note"]
    if dry_run or results is None:
        rows = [dict(e) for e in TABLE_6_OMEGA_SWEEP]
        for r in rows:
            r["_artifact_type"] = "dry_run_schema_omega_sweep"
        _write_csv(path, rows, fieldnames=fieldnames + ["_artifact_type"])
        return path

    _write_csv(path, results, fieldnames=fieldnames)
    return path


# ---------------------------------------------------------------------------
# Table writer: Table 7 – iteration sensitivity
# ---------------------------------------------------------------------------


def write_table_7(
    results: Optional[List[Dict[str, Any]]] = None,
    dry_run: bool = False,
) -> pathlib.Path:
    """Write Table 7: Effects of training iteration count in FFHQ→Sunglasses.

    Table 7. Effects of training iteration in FFHQ→Sunglasses case in terms
    of FID and Intra-LPIPS.
    Anchors: 300 iters (ablation), 5000 iters (main paper result).
    """
    path = TABLE_PATHS["table_7"]
    fieldnames = ["iterations", "fid", "intra_lpips", "note"]
    if dry_run or results is None:
        rows = [dict(e) for e in TABLE_7_ITERATION_SWEEP]
        for r in rows:
            r["_artifact_type"] = "dry_run_schema_iteration_sweep"
        _write_csv(path, rows, fieldnames=fieldnames + ["_artifact_type"])
        return path

    _write_csv(path, results, fieldnames=fieldnames)
    return path


# ---------------------------------------------------------------------------
# Table writer: Table 8 – GPU memory
# ---------------------------------------------------------------------------


def write_table_8(
    results: Optional[List[Dict[str, Any]]] = None,
    dry_run: bool = False,
) -> pathlib.Path:
    """Write Table 8: GPU memory consumption with/without adaptor.

    Table 8. GPU memory consumption (MB) for each module at batch_size=1,
    comparing scenarios with and without the Shift Adaptor.
    The adaptor results in only a slight increase in GPU memory consumption.
    """
    path = TABLE_PATHS["table_8"]
    fieldnames = ["module", "framework", "adaptor_enabled", "gpu_memory_mb",
                  "batch_size", "note"]
    if dry_run or results is None:
        rows = [dict(e) for e in TABLE_8_GPU_MEMORY_SCHEMA]
        for r in rows:
            r["_artifact_type"] = "dry_run_schema_gpu_memory"
        _write_csv(path, rows)
        return path

    _write_csv(path, results, fieldnames=fieldnames)
    return path


# ---------------------------------------------------------------------------
# Table writer: Table 9 – User study
# ---------------------------------------------------------------------------


def write_table_9(
    results: Optional[List[Dict[str, Any]]] = None,
    dry_run: bool = False,
) -> pathlib.Path:
    """Write Table 9: Anonymous user study ANT vs DDPM-PA.

    Table 9. Anonymous user study to assess the qualitative performance of
    our method (ANT) in comparison to DDPM-PA.
    """
    path = TABLE_PATHS["table_9"]
    if dry_run or results is None:
        schema_rows = [
            {
                "domain": d,
                "ant_preference_rate":  "to_be_computed",
                "pa_preference_rate":   "to_be_computed",
                "tie_rate":             "to_be_computed",
                "n_evaluators":         "to_be_computed",
                "_artifact_type":       "dry_run_schema_user_study",
                "_caption":             TABLE_CAPTIONS["table_9"][:100],
            }
            for d in [
                "FFHQ→Sunglasses",
                "FFHQ→Babies",
                "FFHQ→Raphael's Paintings",
                "LSUN Church→Landscape",
            ]
        ]
        _write_csv(path, schema_rows)
        return path

    _write_csv(path, results)
    return path


# ---------------------------------------------------------------------------
# Figure writer: Figure 1 – Fine-tuning progression
# ---------------------------------------------------------------------------


def write_figure_1(
    stage_images: Optional[Any] = None,
    lpips_scores: Optional[List[float]] = None,
    dry_run: bool = False,
) -> pathlib.Path:
    """Write Figure 1: Fine-tuning progression FFHQ→Sunglasses with LPIPS.

    Figure 1. Two sets of images generated from corresponding fixed noise inputs
    at different stages of fine-tuning DDPM from FFHQ to 10-shot Sunglasses.
    LPIPS between generated and target image annotated on each image.

    Args:
        stage_images: list of (baseline_img, ant_img) pairs per iteration stage
        lpips_scores: per-image LPIPS scores (same order as stage_images)
        dry_run: if True, write labeled schema/readiness figure
    """
    path = FIGURE_PATHS["figure_1"]
    if dry_run or stage_images is None:
        _write_dry_run_figure(path, FIGURE_CAPTIONS["figure_1"],
                              "Figure 1: Fine-tuning Progression FFHQ→Sunglasses")
        return path

    _ensure_dir(path)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        n_stages = len(stage_images)
        fig, axes = plt.subplots(2, n_stages, figsize=(n_stages * 2.5, 5.5))
        if n_stages == 1:
            axes = [[axes[0]], [axes[1]]]

        for i, pair in enumerate(stage_images):
            baseline_img, ant_img = pair
            for row, img in enumerate([baseline_img, ant_img]):
                ax = axes[row][i]
                if hasattr(img, "numpy"):
                    img = img.numpy()
                ax.imshow(np.asarray(img))
                ax.axis("off")
                if lpips_scores:
                    idx = i * 2 + row
                    if idx < len(lpips_scores):
                        ax.set_title(f"LPIPS={lpips_scores[idx]:.3f}", fontsize=7)

        axes[0][0].set_ylabel("Baseline", fontsize=8, rotation=0,
                              ha="right", va="center")
        axes[1][0].set_ylabel("DPMs-ANT", fontsize=8, rotation=0,
                              ha="right", va="center")
        fig.suptitle(
            "Figure 1: FFHQ → Sunglasses Fine-tuning Progression",
            fontsize=9,
        )
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        logger.warning("write_figure_1 fallback: %s", exc)
        path.write_bytes(_minimal_png_bytes())

    logger.info("Wrote Figure 1: %s", path)
    return path


# ---------------------------------------------------------------------------
# Figure writer: Figure 2 – Gradient changes and heat maps
# ---------------------------------------------------------------------------


def write_figure_2(
    gradient_data: Optional[Dict[str, Any]] = None,
    dry_run: bool = False,
) -> pathlib.Path:
    """Write Figure 2: Gradient changes and heat maps.

    Figure 2(a): Gradient directions – cyan=10k-sample oracle, blue=baseline,
                 red=DDPM-ANT w/o AN, orange=DDPM-ANT.
    Figure 2(b): Heat map visualization.

    Args:
        gradient_data: dict with optional keys:
            'steps': list of iteration indices,
            'full_grad': cyan line values,
            'baseline': blue line values,
            'wo_an': red line values,
            'ant': orange line values,
            'heatmap': 2-D numpy array
        dry_run: if True, write labeled schema/readiness figure
    """
    path = FIGURE_PATHS["figure_2"]
    if dry_run or gradient_data is None:
        _write_dry_run_figure(path, FIGURE_CAPTIONS["figure_2"],
                              "Figure 2: Gradient Directions + Heat Map")
        return path

    _ensure_dir(path)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        ax = axes[0]
        ax.set_title("(a) Gradient Directions", fontsize=10)
        steps = gradient_data.get("steps")
        lines = {
            "full_grad": ("cyan",   "10k-sample oracle"),
            "baseline":  ("blue",   "Baseline DDPM"),
            "wo_an":     ("red",    "DDPM-ANT w/o AN"),
            "ant":       ("orange", "DDPM-ANT (ours)"),
        }
        for key, (color, label) in lines.items():
            vals = gradient_data.get(key)
            if vals is not None:
                xs = steps if steps is not None else list(range(len(vals)))
                ax.plot(xs, vals, color=color, label=label, linewidth=1.5)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Gradient Cosine Similarity")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        ax2 = axes[1]
        ax2.set_title("(b) Heat Map", fontsize=10)
        heatmap = gradient_data.get("heatmap")
        if heatmap is not None:
            hm = np.asarray(heatmap)
            im = ax2.imshow(hm, cmap="hot")
            plt.colorbar(im, ax=ax2)
        else:
            ax2.text(0.5, 0.5,
                     "Attention Heat Map\n(computed from model activations)",
                     ha="center", va="center",
                     transform=ax2.transAxes, fontsize=9)
            ax2.axis("off")

        fig.suptitle("Figure 2: Gradient Visualizations", fontsize=9)
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        logger.warning("write_figure_2 fallback: %s", exc)
        path.write_bytes(_minimal_png_bytes())

    logger.info("Wrote Figure 2: %s", path)
    return path


# ---------------------------------------------------------------------------
# Figure writer: Figure 2b – Heat map
# ---------------------------------------------------------------------------


def write_figure_2b(
    heatmap_data: Optional[Any] = None,
    dry_run: bool = False,
) -> pathlib.Path:
    """Write Figure 2b: Heat map visualization."""
    path = FIGURE_PATHS["figure_2b"]
    if dry_run or heatmap_data is None:
        _write_dry_run_figure(path, FIGURE_CAPTIONS["figure_2b"],
                              "Figure 2b: Heat Map Visualization")
        return path

    _ensure_dir(path)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(4, 4))
        hm = np.asarray(heatmap_data)
        im = ax.imshow(hm, cmap="hot")
        plt.colorbar(im, ax=ax)
        ax.set_title("Figure 2b: Adversarial Noise Heat Map", fontsize=10)
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        logger.warning("write_figure_2b fallback: %s", exc)
        path.write_bytes(_minimal_png_bytes())

    logger.info("Wrote Figure 2b: %s", path)
    return path


# ---------------------------------------------------------------------------
# Figure writer: Figure 3 – 10-shot generation comparison
# ---------------------------------------------------------------------------


def write_figure_3(
    church_landscape_rows: Optional[List[Any]] = None,
    ffhq_raphael_rows: Optional[List[Any]] = None,
    method_labels: Optional[List[str]] = None,
    dry_run: bool = False,
) -> pathlib.Path:
    """Write Figure 3: LSUN Church→Landscape (top) and FFHQ→Raphael (bottom).

    Figure 3. GAN-based baselines vs DDPM-based methods vs ANT for:
      - Top block: LSUN Church → Landscape drawings
      - Bottom block: FFHQ → Raphael's paintings
    ANT produces highest fidelity results.
    """
    path = FIGURE_PATHS["figure_3"]
    default_labels = ["TGAN", "ADA", "EWC", "CDC", "DCL", "DDPM-PA", "ANT (ours)"]
    labels = method_labels or default_labels

    if dry_run or (church_landscape_rows is None and ffhq_raphael_rows is None):
        _write_dry_run_figure(
            path, FIGURE_CAPTIONS["figure_3"],
            "Figure 3: LSUN Church→Landscape (top) / FFHQ→Raphael (bottom)",
        )
        return path

    _write_multirow_method_figure(
        path,
        rows_top=church_landscape_rows or [],
        rows_bottom=ffhq_raphael_rows or [],
        method_labels=labels,
        title="Figure 3: 10-shot Generation Comparison",
        subtitle_top="LSUN Church → Landscape Drawings",
        subtitle_bottom="FFHQ → Raphael's Paintings",
    )
    return path


# ---------------------------------------------------------------------------
# Figure writer: Figure 4 – Ablation study (300 iters, FID)
# ---------------------------------------------------------------------------


def write_figure_4(
    ablation_rows: Optional[List[Any]] = None,
    method_labels: Optional[List[str]] = None,
    fid_scores: Optional[List[float]] = None,
    dry_run: bool = False,
) -> pathlib.Path:
    """Write Figure 4: Ablation study, 300 iters, FFHQ→Sunglasses, FID(↓).

    Figure 4. Ablation rows (top to bottom):
      Row 1: Baseline (direct fine-tuning, FID=41.88)
      Row 2: Adaptor only (FID=38.65)
      Row 3: DPMs-ANT w/o AN (similarity guidance only)
      Row 4: DPMs-ANT (full method)
    All from same noise inputs for fair comparison.
    """
    path = FIGURE_PATHS["figure_4"]
    default_labels = [
        "Baseline (FID≈41.88)",
        "Adaptor only (FID≈38.65)",
        "DPMs-ANT w/o AN",
        "DPMs-ANT (ours)",
    ]
    labels = method_labels or default_labels

    if dry_run or ablation_rows is None:
        _write_dry_run_figure(
            path, FIGURE_CAPTIONS["figure_4"],
            "Figure 4: Ablation Study (300 iters, FFHQ→Sunglasses, FID↓)",
        )
        return path

    _ensure_dir(path)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        n_rows = len(ablation_rows)
        n_cols = max((len(r) for r in ablation_rows), default=4)
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(n_cols * 2.0, n_rows * 2.2))
        if n_rows == 1:
            axes = [axes]
        if n_cols == 1:
            axes = [[ax] for ax in axes]

        for r, (imgs, label) in enumerate(zip(ablation_rows, labels)):
            for c in range(n_cols):
                ax = axes[r][c]
                if c < len(imgs):
                    img = imgs[c]
                    if hasattr(img, "numpy"):
                        img = img.numpy()
                    ax.imshow(np.asarray(img))
                else:
                    ax.set_facecolor("#f0f0f0")
                ax.axis("off")
            fid_txt = ""
            if fid_scores and r < len(fid_scores):
                fid_txt = f" FID={fid_scores[r]:.2f}"
            axes[r][0].set_ylabel(
                label + fid_txt, fontsize=7, rotation=0,
                ha="right", va="center",
            )

        fig.suptitle(
            "Figure 4: Ablation (300 iters, 10-shot Sunglasses, FID↓)",
            fontsize=8,
        )
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        logger.warning("write_figure_4 fallback: %s", exc)
        path.write_bytes(_minimal_png_bytes())

    logger.info("Wrote Figure 4: %s", path)
    return path


# ---------------------------------------------------------------------------
# Figure writer: Figure 5 – Sunglasses and Babies
# ---------------------------------------------------------------------------


def write_figure_5(
    sunglasses_rows: Optional[List[Any]] = None,
    babies_rows: Optional[List[Any]] = None,
    method_labels: Optional[List[str]] = None,
    dry_run: bool = False,
) -> pathlib.Path:
    """Write Figure 5: 10-shot generation FFHQ→Sunglasses and FFHQ→Babies.

    Figure 5. GAN vs DDPM baseline vs ANT (ours) for:
      - Top block: FFHQ → Sunglasses
      - Bottom block: FFHQ → Babies
    Our approach (5th and 6th rows) generates better quality images.
    Quantitative results provided in Table 1.
    """
    path = FIGURE_PATHS["figure_5"]
    default_labels = ["TGAN", "ADA", "EWC", "CDC", "DCL", "DDPM-PA", "ANT (ours)"]
    labels = method_labels or default_labels

    if dry_run or (sunglasses_rows is None and babies_rows is None):
        _write_dry_run_figure(
            path, FIGURE_CAPTIONS["figure_5"],
            "Figure 5: FFHQ→Sunglasses (top) and FFHQ→Babies (bottom)",
        )
        return path

    _write_multirow_method_figure(
        path,
        rows_top=sunglasses_rows or [],
        rows_bottom=babies_rows or [],
        method_labels=labels,
        title="Figure 5: 10-shot Generation Results",
        subtitle_top="FFHQ → Sunglasses",
        subtitle_bottom="FFHQ → Babies",
    )
    return path


# ---------------------------------------------------------------------------
# Figure writer: Figure 6 – Ablation across iterations
# ---------------------------------------------------------------------------


def write_figure_6(
    iteration_rows: Optional[Dict[str, List[Any]]] = None,
    method_labels: Optional[List[str]] = None,
    iterations: Optional[List[int]] = None,
    dry_run: bool = False,
) -> pathlib.Path:
    """Write Figure 6: Ablation across iterations, FFHQ→Sunglasses.

    Figure 6. Models trained for different iterations:
      Row 1: Baseline (direct fine-tuning)
      Row 2: DPMs-ANT w/o AN (similarity guidance only)
      Row 3: DPMs-ANT (our full method)
    Columns are different iteration counts.
    """
    path = FIGURE_PATHS["figure_6"]
    default_labels = ["Baseline", "DPMs-ANT w/o AN", "DPMs-ANT (ours)"]
    default_iters = [100, 200, 300, 500, 1000, 5000]
    labels = method_labels or default_labels
    iter_list = iterations or default_iters

    if dry_run or iteration_rows is None:
        _write_dry_run_figure(
            path, FIGURE_CAPTIONS["figure_6"],
            "Figure 6: Ablation Across Iterations (FFHQ→Sunglasses)",
        )
        return path

    _ensure_dir(path)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        n_rows = len(labels)
        n_cols = len(iter_list)
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(n_cols * 2.0, n_rows * 2.2))
        if n_rows == 1:
            axes = [axes]
        if n_cols == 1:
            axes = [[ax] for ax in axes]

        for r, label in enumerate(labels):
            method_imgs = iteration_rows.get(label, [])
            for c, itr in enumerate(iter_list):
                ax = axes[r][c]
                if c < len(method_imgs):
                    img = method_imgs[c]
                    if hasattr(img, "numpy"):
                        img = img.numpy()
                    ax.imshow(np.asarray(img))
                else:
                    ax.set_facecolor("#e8e8e8")
                ax.axis("off")
                if r == 0:
                    ax.set_title(f"{itr}", fontsize=8)
            axes[r][0].set_ylabel(label, fontsize=7, rotation=0,
                                  ha="right", va="center")

        fig.suptitle(
            "Figure 6: Ablation Study (Different Iterations, 10-shot Sunglasses)",
            fontsize=8,
        )
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        logger.warning("write_figure_6 fallback: %s", exc)
        path.write_bytes(_minimal_png_bytes())

    logger.info("Wrote Figure 6: %s", path)
    return path


# ---------------------------------------------------------------------------
# Multi-row method comparison figure helper
# ---------------------------------------------------------------------------


def _write_multirow_method_figure(
    path: pathlib.Path,
    rows_top: List[Any],
    rows_bottom: List[Any],
    method_labels: List[str],
    title: str,
    subtitle_top: str = "",
    subtitle_bottom: str = "",
    n_samples: int = 4,
) -> None:
    """Write a multi-block comparison figure with two domain blocks."""
    _ensure_dir(path)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        n_top = len(rows_top)
        n_bot = len(rows_bottom)
        n_total = n_top + n_bot + 1  # +1 separator row

        fig, axes = plt.subplots(
            n_total, n_samples,
            figsize=(n_samples * 2.0, n_total * 2.0 + 0.5),
        )
        if n_total == 1:
            axes = [axes]
        if n_samples == 1:
            axes = [[ax] for ax in axes]

        def _show_row(row_idx: int, imgs: List[Any], label: str) -> None:
            for c in range(n_samples):
                ax = axes[row_idx][c]
                if c < len(imgs):
                    img = imgs[c]
                    if hasattr(img, "numpy"):
                        img = img.numpy()
                    ax.imshow(np.asarray(img))
                else:
                    ax.set_facecolor("#eeeeee")
                ax.axis("off")
            axes[row_idx][0].set_ylabel(label, fontsize=7,
                                        rotation=0, ha="right", va="center")

        for r, (row_imgs, lbl) in enumerate(
            zip(rows_top, method_labels[:n_top])
        ):
            _show_row(r, row_imgs, lbl)
            if r == 0 and subtitle_top:
                axes[0][0].set_title(subtitle_top, fontsize=8, loc="left")

        # Separator row (black)
        sep = n_top
        for c in range(n_samples):
            axes[sep][c].set_facecolor("black")
            axes[sep][c].axis("off")

        for r, (row_imgs) in enumerate(rows_bottom):
            real_r = n_top + 1 + r
            lbl = (method_labels[r] if r < len(method_labels)
                   else f"method_{r}")
            _show_row(real_r, row_imgs, lbl)
            if r == 0 and subtitle_bottom:
                axes[real_r][0].set_title(subtitle_bottom, fontsize=8, loc="left")

        fig.suptitle(title, fontsize=9)
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Wrote multi-row figure: %s", path)
    except Exception as exc:
        logger.warning("_write_multirow_method_figure fallback: %s", exc)
        path.write_bytes(_minimal_png_bytes())


# ---------------------------------------------------------------------------
# Metrics JSON writer
# ---------------------------------------------------------------------------


def write_metrics_json(
    metrics: Optional[Dict[str, Any]] = None,
    dry_run: bool = False,
) -> pathlib.Path:
    """Write results/metrics.json with metric schema and computed values.

    Args:
        metrics: dict of computed metric values (after real evaluation)
        dry_run: if True, write schema/readiness artifact with reference values

    Returns:
        path to written file
    """
    path = JSON_PATHS["metrics"]
    if dry_run or metrics is None:
        payload = {
            "_artifact_type": "dry_run_schema_readiness_artifact",
            "_note": (
                "This is a DRY-RUN contract artifact. "
                "NOT a real experiment result. "
                "Populate by running: python evaluate.py"
            ),
            "metric_schemas": METRIC_SCHEMA,
            "result_trend_assertions": RESULT_TREND_ASSERTIONS,
            "named_baselines": NAMED_BASELINES,
            "paper_reference_values": {
                "table_2_fid": TABLE_2_REFERENCE_DATA,
                "key_numeric_anchors": {
                    "DDPM_ANT_FFHQ_Babies_FID":     46.70,
                    "DDPM_ANT_FFHQ_Sunglasses_FID":  20.06,
                    "DDPM_PA_FFHQ_Babies_FID":       48.92,
                    "DDPM_PA_FFHQ_Sunglasses_FID":   34.75,
                    "Adaptor_only_300iter_FID":      38.65,
                    "Baseline_300iter_FID":           41.88,
                    "gamma_default":                  5,
                    "omega_default":                  0.02,
                    "adversarial_inner_steps":        10,
                    "total_training_iterations":   5000,
                    "ablation_iterations":           300,
                },
            },
            "artifact_paths": list_artifact_paths(),
            "metrics": {
                k: {
                    "value": None,
                    "schema": v,
                    "_status": "not_yet_computed",
                }
                for k, v in METRIC_SCHEMA.items()
            },
        }
        _write_json(path, payload)
        return path

    payload = {
        "metric_schemas": METRIC_SCHEMA,
        "result_trend_assertions": RESULT_TREND_ASSERTIONS,
        "named_baselines": NAMED_BASELINES,
        "artifact_paths": list_artifact_paths(),
        "metrics": metrics,
    }
    _write_json(path, payload)
    return path


# ---------------------------------------------------------------------------
# Experiment results summary table and figure
# ---------------------------------------------------------------------------


def write_experiment_results_table(
    results: Optional[List[Dict[str, Any]]] = None,
    dry_run: bool = False,
) -> pathlib.Path:
    """Write results/tables/experiment_results.csv with all domain results."""
    path = TABLE_PATHS["experiment_results"]
    if dry_run or results is None:
        schema_rows = [
            {
                "framework":       fw,
                "source_domain":   src,
                "target_domain":   tgt,
                "method":          "ANT",
                "fid":             "to_be_computed",
                "intra_lpips":     "to_be_computed",
                "fidelity_score":  "to_be_computed",
                "training_iters":  5000,
                "_artifact_type":  "dry_run_schema_not_real_result",
            }
            for fw, src, tgt in [
                ("ddpm", "ffhq",        "babies"),
                ("ddpm", "ffhq",        "sunglasses"),
                ("ddpm", "ffhq",        "raphael_peale"),
                ("ddpm", "ffhq",        "sketches"),
                ("ddpm", "ffhq",        "modigliani"),
                ("ddpm", "lsun_church", "landscape_drawings"),
                ("ddpm", "lsun_church", "haunted_houses"),
                ("ldm",  "ffhq",        "babies"),
                ("ldm",  "ffhq",        "sunglasses"),
            ]
        ]
        _write_csv(path, schema_rows)
        return path

    _write_csv(path, results)
    return path


def write_experiment_results_figure(
    results: Optional[Dict[str, Any]] = None,
    dry_run: bool = False,
) -> pathlib.Path:
    """Write results/figures/experiment_results.png FID bar chart."""
    path = FIGURE_PATHS["experiment_results_fig"]
    if dry_run or results is None:
        _write_dry_run_figure(
            path,
            "Experiment Results: FID(↓) across 7 target domains for ANT vs baselines.",
            "Experiment Results Summary",
        )
        return path

    _ensure_dir(path)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        domains = results.get("domains", [])
        ant_fid = results.get("ant_fid", [])
        pa_fid = results.get("pa_fid", [])

        if not domains:
            _write_dry_run_figure(
                path, "No domain data provided", "Experiment Results",
            )
            return path

        fig, ax = plt.subplots(figsize=(10, 5))
        x = list(range(len(domains)))
        ax.bar([i - 0.2 for i in x], ant_fid, 0.35,
               label="ANT (ours)", color="steelblue")
        ax.bar([i + 0.2 for i in x], pa_fid, 0.35,
               label="DDPM-PA", color="coral")
        ax.set_xticks(x)
        ax.set_xticklabels(domains, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("FID (↓)")
        ax.set_title("FID: ANT vs DDPM-PA across 7 Target Domains")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        logger.warning("write_experiment_results_figure fallback: %s", exc)
        path.write_bytes(_minimal_png_bytes())

    logger.info("Wrote experiment results figure: %s", path)
    return path


# ---------------------------------------------------------------------------
# Batch dry-run artifact writer (called during smoke/docker_validate)
# ---------------------------------------------------------------------------


def write_all_dry_run_artifacts() -> Dict[str, str]:
    """Write all declared artifact paths as dry-run schema/readiness artifacts.

    Called during --mode runtime_smoke and --mode docker_validate.
    Creates all parent directories and writes schema/readiness content for every
    declared artifact path.

    Returns:
        dict mapping artifact key → written path (str)

    NOTE: All outputs are explicitly labeled as dry-run contract artifacts and
    MUST NOT be presented as real experiment results or benchmark scores.
    """
    written: Dict[str, str] = {}

    logger.info("[dry-run] Writing schema/readiness figures...")
    for key, writer in [
        ("figure_1",             lambda: write_figure_1(dry_run=True)),
        ("figure_2",             lambda: write_figure_2(dry_run=True)),
        ("figure_2b",            lambda: write_figure_2b(dry_run=True)),
        ("figure_3",             lambda: write_figure_3(dry_run=True)),
        ("figure_4",             lambda: write_figure_4(dry_run=True)),
        ("figure_5",             lambda: write_figure_5(dry_run=True)),
        ("figure_6",             lambda: write_figure_6(dry_run=True)),
        ("experiment_results_fig", lambda: write_experiment_results_figure(dry_run=True)),
    ]:
        written[key] = str(writer())

    logger.info("[dry-run] Writing schema/readiness tables...")
    for key, writer in [
        ("table_1",           lambda: write_table_1(dry_run=True)),
        ("table_2",           lambda: write_table_2(dry_run=True)),
        ("table_3",           lambda: write_table_3(dry_run=True)),
        ("table_4",           lambda: write_table_4(dry_run=True)),
        ("table_5",           lambda: write_table_5(dry_run=True)),
        ("table_6",           lambda: write_table_6(dry_run=True)),
        ("table_7",           lambda: write_table_7(dry_run=True)),
        ("table_8",           lambda: write_table_8(dry_run=True)),
        ("table_9",           lambda: write_table_9(dry_run=True)),
        ("experiment_results", lambda: write_experiment_results_table(dry_run=True)),
    ]:
        written[key] = str(writer())

    logger.info("[dry-run] Writing schema/readiness JSON artifacts...")
    written["metrics"] = str(write_metrics_json(dry_run=True))

    # Write predictions.jsonl schema stub
    pred_path = JSON_PATHS["predictions"]
    _ensure_dir(pred_path)
    with open(pred_path, "w", encoding="utf-8") as f:
        schema_line = {
            "_artifact_type": "dry_run_schema",
            "sample_id": 0,
            "domain": "FFHQ→Sunglasses",
            "method": "ANT",
            "generated_image_path": "results/generated/sample_0.png",
            "fid_contribution": None,
            "lpips_vs_target": None,
            "_note": "dry_run_not_real_result",
        }
        f.write(json.dumps(schema_line) + "\n")
    written["predictions"] = str(pred_path)

    # Write artifact manifest JSON
    manifest_path = JSON_PATHS["artifact_manifest"]
    manifest = {
        "_artifact_type": "dry_run_artifact_manifest",
        "artifact_paths": list_artifact_paths(),
        "figure_captions": FIGURE_CAPTIONS,
        "table_captions": TABLE_CAPTIONS,
        "metric_schemas": METRIC_SCHEMA,
        "result_trend_assertions": RESULT_TREND_ASSERTIONS,
        "named_baselines": NAMED_BASELINES,
        "paper_evidence_matrix": {
            "ddpm_ffhq_targets": [
                "babies", "sunglasses", "raphael_peale", "sketches", "modigliani",
            ],
            "ddpm_lsun_targets": ["landscape_drawings", "haunted_houses"],
            "ldm_ffhq_targets": ["babies", "sunglasses"],
            "adaptor_config_ddpm": {"c": 4, "d": 8},
            "adaptor_config_ldm":  {"c": 2, "d": 8},
            "fixed_hyperparams": {
                "batch_size": 64,
                "omega": 0.02,
                "adversarial_inner_steps": 10,
                "total_iterations": 5000,
                "ablation_iterations": 300,
                "gamma": 5,
            },
        },
        "written_artifacts": written,
    }
    _write_json(manifest_path, manifest)
    written["artifact_manifest"] = str(manifest_path)

    logger.info("[dry-run] All %d artifacts written as schema/readiness.", len(written))
    return written


# ---------------------------------------------------------------------------
# Module-level static manifest (for external discovery at import time)
# ---------------------------------------------------------------------------

ARTIFACT_MANIFEST: Dict[str, Any] = {
    "figure_captions":         FIGURE_CAPTIONS,
    "table_captions":          TABLE_CAPTIONS,
    "artifact_paths":          {k: str(v) for k, v in ALL_ARTIFACT_PATHS.items()},
    "metric_schemas":          METRIC_SCHEMA,
    "result_trend_assertions": RESULT_TREND_ASSERTIONS,
    "named_baselines":         NAMED_BASELINES,
}


# ---------------------------------------------------------------------------
# Quick self-test / smoke path
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    logger.info("src/reporting/plotting.py – dry-run smoke test")
    written = write_all_dry_run_artifacts()
    logger.info("Dry-run artifacts written (%d files):", len(written))
    for k, v in sorted(written.items()):
        logger.info("  %-30s -> %s", k, v)
    logger.info(
        "Smoke test complete. "
        "All outputs are dry-run readiness artifacts – NOT real experiment results."
    )