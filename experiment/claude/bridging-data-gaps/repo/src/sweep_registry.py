"""
src/sweep_registry.py
─────────────────────────────────────────────────────────────────────────────
DPMs-ANT: Bounded parameter sweep registry and model configuration store.

All sweep grid values, fixed anchors, and model/classifier registry entries
are derived directly from the paper evidence contract and the addendum.
This module is *configuration-only*: it exposes Python dicts/dataclasses that
training and evaluation callers import; it does not execute training loops.

Paper Figure / Table → sweep parameter mapping:
  Figure 1  – LPIPS progression during DDPM FFHQ→Sunglasses fine-tuning
              (fixed noise, 5 snapshot stages)
  Figure 2  – gradient-direction and heat-map visualisations across diffusion
              time-steps; x-axis=timestep, y-axis=sampled values
  Figure 3  – 10-shot LSUN-Church→Landscape / FFHQ→Raphael qualitative
  Table 1   – Intra-LPIPS(↑) baselines: DDPM & GAN methods, parameter rate
  Table 2   – FID(↓) FFHQ→Babies and FFHQ→Sunglasses ← primary result table
  Figure 4  – ablation @ 300 iter: baseline | adaptor | ANT w/o AN | ANT (FID↓)
  Table 3   – FID/Intra-LPIPS, FFHQ→Sunglasses, classifier shot-count 10 vs 100
  Figure 5  – qualitative FFHQ→Sunglasses and FFHQ→Babies
  Figure 6  – ablation across iteration checkpoints (0..350), 3 method lines
  Table 4   – Intra-LPIPS(↑) DDPM-based + GAN baselines (LDM framework)
  Table 5   – γ (similarity_guidance_scale) sensitivity, FFHQ→Sunglasses
  Table 6   – ω (adversarial_noise_scale) sensitivity, FFHQ→Sunglasses
  Table 7   – training-iteration sensitivity, FFHQ→Sunglasses
  Table 8   – GPU memory (MB) with/without adaptor
  Table 9   – anonymous user study, ANT vs DDPM-PA

reference_grounding: paper_semantic_chunk_012 fixed_hyperparameters addendum_section_5_2
reference_grounding: paper_semantic_chunk_014_01 DDPM_LDM_evaluation_framework
reference_grounding: paper_semantic_chunk_005 adversarial_noise_selection_pgd
reference_grounding: paper_semantic_chunk_008 shift_adaptor_bottleneck_W_down_W_up
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# =============================================================================
# §1  Fixed anchor hyperparameters
#     Every value is bound to a specific paper/addendum evidence obligation.
# reference_grounding: paper_semantic_chunk_012 fixed_hyperparameters
# =============================================================================

FIXED_HYPERPARAMETERS: Dict[str, Any] = {
    # ── Training budget ───────────────────────────────────────────────────────
    # anchor: 5000_iterations (total fine-tuning budget, Algorithm 1)
    "total_iterations": 5000,
    # anchor: 300_training_iterations (ablation iteration cap; classifier training)
    "ablation_iterations": 300,
    # anchor: 300_training_iterations (classifier fine-tuning, addendum §5.2)
    "classifier_training_iterations": 300,

    # ── Data ──────────────────────────────────────────────────────────────────
    # anchor: shot_count=10  (few-shot setting throughout paper)
    "shot_count": 10,
    # anchor: batch_size=64  (addendum §5.2 and training config)
    "batch_size": 64,

    # ── Adversarial noise selection (PGD inner loop) ─────────────────────────
    # anchor: omega=0.02  (PGD perturbation step size, Table 6 default)
    "omega": 0.02,
    # anchor: adversarial_inner_steps=10
    "adversarial_inner_steps": 10,

    # ── Similarity guidance ───────────────────────────────────────────────────
    # anchor: gamma=5  (similarity_guidance_scale, Table 5 default)
    "similarity_guidance_scale": 5,

    # ── DDPM Shift Adaptor bottleneck dimensions ──────────────────────────────
    # anchor: c=4  (compression ratio, DDPM framework)
    "ddpm_adaptor_c": 4,
    # anchor: d=8  (bottleneck width, DDPM framework)
    "ddpm_adaptor_d": 8,

    # ── LDM Shift Adaptor bottleneck dimensions ───────────────────────────────
    # anchor: c=2  (compression ratio, LDM framework)
    "ldm_adaptor_c": 2,
    # anchor: d=8  (bottleneck width, LDM framework)
    "ldm_adaptor_d": 8,

    # ── Adaptor parameter initialisation and freezing ─────────────────────────
    # anchor: adaptor_init_zero (W_down and W_up initialised to zero)
    "adaptor_init_zero": True,
    # anchor: freeze_pretrained (non-adaptor parameters have requires_grad=False)
    "freeze_pretrained": True,

    # ── Classifier fine-tuning (addendum §5.2) ────────────────────────────────
    "classifier_lr": 1e-4,
    "classifier_optimizer": "adam",
    # Last layer replaced → binary classifier (source vs target)
    "classifier_num_classes": 2,
}

# =============================================================================
# §2  Classifier pretrained model configuration  (addendum §5.2)
#
#   "The target classifiers used pre-trained models.
#    For DDPM, the pretrained model used is
#      https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_classifier.pt
#    while for LDM the pretrained model used is
#      https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_classifier.pt
#    These pre-trained models were fine-tuned by modifying the last layer to
#    output two classes to classify whether images were coming from the source
#    or the target dataset.
#    To fine-tune the model the authors used Adam as the optimizer with a
#    learning rate of 1e-4, a batch size of 64, and trained for 300 iterations."
#
# reference_grounding: paper_addendum_section_5_2 classifier_pretrained_models
# =============================================================================

CLASSIFIER_PRETRAINED_URLS: Dict[str, str] = {
    "ddpm": (
        "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/"
        "256x256_classifier.pt"
    ),
    "ldm": (
        "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/"
        "64x64_classifier.pt"
    ),
}

CLASSIFIER_CONFIG: Dict[str, Dict[str, Any]] = {
    "ddpm": {
        "pretrained_url": CLASSIFIER_PRETRAINED_URLS["ddpm"],
        "image_size": 256,
        "original_num_classes": 1000,        # ImageNet classifier
        "finetuned_num_classes": 2,           # binary: source(0) vs target(1)
        "last_layer_modification": "replace_with_linear_2class",
        "optimizer": "adam",
        "learning_rate": 1e-4,
        "batch_size": 64,
        "training_iterations": 300,
        "loss": "cross_entropy",
        "input_domain": "noisy_images",       # classifier receives x_t
    },
    "ldm": {
        "pretrained_url": CLASSIFIER_PRETRAINED_URLS["ldm"],
        "image_size": 64,
        "original_num_classes": 1000,
        "finetuned_num_classes": 2,
        "last_layer_modification": "replace_with_linear_2class",
        "optimizer": "adam",
        "learning_rate": 1e-4,
        "batch_size": 64,
        "training_iterations": 300,
        "loss": "cross_entropy",
        "input_domain": "noisy_latents",
    },
}

# =============================================================================
# §3  Shift Adaptor configuration per framework
#     ψ^l(x) = f(x · W_down) · W_up
#     Inserted residually into every UNet residual block.
#
# reference_grounding: paper_semantic_chunk_008 shift_adaptor_W_down_W_up
# =============================================================================

ADAPTOR_CONFIGS: Dict[str, Dict[str, Any]] = {
    "ddpm": {
        "c": 4,                          # spatial compression ratio  R^{w×h×r} → R^{w/c × h/c × d}
        "d": 8,                          # bottleneck channel width
        "init_zero": True,               # W_down, W_up initialised to 0
        "residual": True,                # output added to layer output
        "position": "all_res_blocks",    # inserted at every residual block
        "formula": "psi_l(x) = f(x @ W_down) @ W_up",
        "trainable_only": True,          # only adaptor params have requires_grad=True
    },
    "ldm": {
        "c": 2,
        "d": 8,
        "init_zero": True,
        "residual": True,
        "position": "all_res_blocks",
        "formula": "psi_l(x) = f(x @ W_down) @ W_up",
        "trainable_only": True,
    },
}

# =============================================================================
# §4  Sensitivity sweeps – bounded config grid values from paper evidence
# =============================================================================

# ─── Table 5: γ (similarity_guidance_scale) sensitivity ──────────────────────
# reference_grounding: paper_semantic_chunk_012 table_5_gamma_sensitivity
SWEEP_GAMMA: Dict[str, Any] = {
    "parameter": "similarity_guidance_scale",
    "symbol": "γ",
    # Paper Table 5: reported values tested for γ
    "values": [1, 2, 3, 5, 7, 9, 10],
    "default": 5,
    "paper_table": "Table 5",
    "paper_figure": None,
    "task": "FFHQ→Sunglasses",
    "shot_count": 10,
    "metrics": ["FID", "Intra-LPIPS"],
    "description": (
        "Effects of γ (similarity guidance scale) in FFHQ→Sunglasses. "
        "Reported in Table 5. Default γ=5 used throughout."
    ),
}

# ─── Table 6: ω (adversarial_noise_scale / PGD step size) sensitivity ────────
# reference_grounding: paper_semantic_chunk_012 table_6_omega_sensitivity
SWEEP_OMEGA: Dict[str, Any] = {
    "parameter": "adversarial_noise_scale",
    "symbol": "ω",
    # Paper Table 6: ω values tested
    "values": [0.01, 0.02, 0.03, 0.04, 0.05],
    "default": 0.02,
    "paper_table": "Table 6",
    "paper_figure": None,
    "task": "FFHQ→Sunglasses",
    "shot_count": 10,
    "metrics": ["FID", "Intra-LPIPS"],
    "description": (
        "Effects of ω (PGD adversarial noise step size) in FFHQ→Sunglasses. "
        "Reported in Table 6. Default ω=0.02 used throughout."
    ),
}

# ─── Table 7 / Figure 6: training-iteration sensitivity ──────────────────────
# reference_grounding: paper_semantic_chunk_012 table_7_iteration_sensitivity
SWEEP_ITERATIONS: Dict[str, Any] = {
    "parameter": "training_iteration_count",
    # Figure 6 x-axis: 0, 50, 100, 150, 200, 250, 300, 350
    "values": [0, 50, 100, 150, 200, 250, 300, 350],
    "default": 300,
    "paper_table": "Table 7",
    "paper_figure": "Figure 6",
    "task": "FFHQ→Sunglasses",
    "shot_count": 10,
    # Figure 6: three method lines
    "methods": ["baseline", "DPMs-ANT_w/o_AN", "DPMs-ANT"],
    "metrics": ["FID", "Intra-LPIPS"],
    "description": (
        "Training-iteration count sensitivity for FFHQ→Sunglasses. "
        "Figure 6 shows three method lines across iteration checkpoints [0..350]. "
        "Table 7 reports FID and Intra-LPIPS."
    ),
}

# ─── Table 3: classifier shot-count sensitivity ───────────────────────────────
# reference_grounding: paper_semantic_chunk_012 table_3_shot_count_classifier
SWEEP_SHOT_COUNT: Dict[str, Any] = {
    "parameter": "shot_count",
    # Table 3: classifiers trained on 10 images vs 100 images
    "values": [10, 100],
    "default": 10,
    "paper_table": "Table 3",
    "paper_figure": None,
    "task": "FFHQ→Sunglasses",
    "metrics": ["FID", "Intra-LPIPS"],
    "description": (
        "FID and Intra-LPIPS for DPMs-ANT (FFHQ→Sunglasses) with classifiers "
        "trained on 10 vs 100 images. Reported in Table 3."
    ),
}

# =============================================================================
# §5  Master sweep registry – all sensitivity axes indexed by name
# =============================================================================

SWEEP_REGISTRY: Dict[str, Dict[str, Any]] = {
    "gamma":              SWEEP_GAMMA,
    "omega":              SWEEP_OMEGA,
    "training_iterations": SWEEP_ITERATIONS,
    "shot_count":         SWEEP_SHOT_COUNT,
}

# =============================================================================
# §6  Experiment task matrix
#     Covers all 7 source→target domain pairs evaluated in Tables 1-4.
# reference_grounding: paper_semantic_chunk_012 experiment_tasks_7_domains
# =============================================================================

EXPERIMENT_TASKS: List[Dict[str, Any]] = [
    # ── FFHQ source domain (5 targets) ───────────────────────────────────────
    {
        "id": "ffhq_babies",
        "source": "ffhq",
        "target": "babies",
        "framework": "ddpm",
        "paper_tables": ["Table 1", "Table 2"],
        "paper_figures": ["Figure 5"],
        "shot_count": 10,
        "primary_metric": "FID",
    },
    {
        "id": "ffhq_sunglasses",
        "source": "ffhq",
        "target": "sunglasses",
        "framework": "ddpm",
        "paper_tables": ["Table 1", "Table 2", "Table 3", "Table 5", "Table 6", "Table 7"],
        "paper_figures": ["Figure 1", "Figure 4", "Figure 5", "Figure 6"],
        "shot_count": 10,
        "primary_metric": "FID",
        "is_ablation_task": True,
    },
    {
        "id": "ffhq_raphael",
        "source": "ffhq",
        "target": "raphael_paintings",
        "framework": "ddpm",
        "paper_tables": ["Table 1"],
        "paper_figures": ["Figure 3"],
        "shot_count": 10,
        "primary_metric": "Intra-LPIPS",
    },
    {
        "id": "ffhq_sketches",
        "source": "ffhq",
        "target": "sketches",
        "framework": "ddpm",
        "paper_tables": ["Table 1"],
        "paper_figures": [],
        "shot_count": 10,
        "primary_metric": "Intra-LPIPS",
    },
    {
        "id": "ffhq_modigliani",
        "source": "ffhq",
        "target": "modigliani",
        "framework": "ddpm",
        "paper_tables": ["Table 1"],
        "paper_figures": [],
        "shot_count": 10,
        "primary_metric": "Intra-LPIPS",
    },
    # ── LSUN-Church source domain (2 targets) ─────────────────────────────────
    {
        "id": "church_haunted",
        "source": "lsun_church",
        "target": "haunted_houses",
        "framework": "ddpm",
        "paper_tables": ["Table 1"],
        "paper_figures": [],
        "shot_count": 10,
        "primary_metric": "Intra-LPIPS",
    },
    {
        "id": "church_landscape",
        "source": "lsun_church",
        "target": "landscape_drawings",
        "framework": "ddpm",
        "paper_tables": ["Table 1", "Table 4"],
        "paper_figures": ["Figure 3"],
        "shot_count": 10,
        "primary_metric": "Intra-LPIPS",
    },
]

# =============================================================================
# §7  Baseline method registry
#     Covers all baselines reported in Tables 1-4.
# reference_grounding: paper_semantic_chunk_012 named_baselines
# =============================================================================

BASELINE_REGISTRY: Dict[str, Dict[str, Any]] = {
    # ── GAN-based baselines ───────────────────────────────────────────────────
    "tgan": {
        "name": "TGAN",
        "category": "gan",
        "paper_tables": ["Table 1", "Table 2", "Table 4"],
        "paper_figures": ["Figure 3", "Figure 5"],
    },
    "ada": {
        "name": "ADA",
        "category": "gan",
        "paper_tables": ["Table 1", "Table 2", "Table 4"],
        "paper_figures": ["Figure 3", "Figure 5"],
    },
    "ewc": {
        "name": "EWC",
        "category": "gan",
        "paper_tables": ["Table 1", "Table 2", "Table 4"],
        "paper_figures": [],
    },
    "cdc": {
        "name": "CDC",
        "category": "gan",
        "paper_tables": ["Table 1", "Table 2", "Table 4"],
        "paper_figures": ["Figure 3", "Figure 5"],
    },
    "dcl": {
        "name": "DCL",
        "category": "gan",
        "paper_tables": ["Table 1", "Table 4"],
        "paper_figures": [],
    },
    # ── DDPM-based baselines ───────────────────────────────────────────────────
    "ddpm_finetune": {
        "name": "DDPM (fine-tune all)",
        "category": "ddpm",
        "paper_tables": ["Table 1", "Table 2"],
        "paper_figures": ["Figure 4"],
        "description": "Direct fine-tuning of entire DDPM (baseline)",
    },
    "ddpm_adaptor_only": {
        "name": "DDPM + Adaptor only",
        "category": "ddpm",
        "paper_tables": [],
        "paper_figures": ["Figure 4"],
        "description": "Fine-tuning only the adaptor layer (no ANT)",
    },
    "ddpm_ant_wo_an": {
        "name": "DPMs-ANT w/o AN",
        "category": "ddpm",
        "paper_tables": ["Table 1", "Table 2"],
        "paper_figures": ["Figure 2", "Figure 4", "Figure 6"],
        "description": (
            "DPMs-ANT without adversarial noise selection – "
            "only similarity-guided training active"
        ),
    },
    "ddpm_pa": {
        "name": "DDPM-PA",
        "category": "ddpm",
        "paper_tables": ["Table 9"],
        "paper_figures": [],
        "description": "DDPM pairwise alignment baseline",
    },
    # ── Ours ──────────────────────────────────────────────────────────────────
    "dpms_ant": {
        "name": "DPMs-ANT (ours)",
        "category": "ours",
        "paper_tables": [
            "Table 1", "Table 2", "Table 3", "Table 4",
            "Table 5", "Table 6", "Table 7", "Table 9",
        ],
        "paper_figures": [
            "Figure 1", "Figure 3", "Figure 4", "Figure 5", "Figure 6",
        ],
        "description": (
            "Full DPMs-ANT: Shift Adaptor + Similarity-Guided Training + "
            "Adversarial Noise Selection (Algorithm 1)"
        ),
    },
    "ldm_ant": {
        "name": "LDM-ANT (ours)",
        "category": "ours",
        "paper_tables": ["Table 4"],
        "paper_figures": [],
        "description": "DPMs-ANT applied in LDM framework",
    },
}

# =============================================================================
# §8  Ablation experiment matrix
#     Maps ablation rows in paper Figure 4 and Figure 6 to method ids.
# reference_grounding: paper_semantic_chunk_012 ablation_figure4_figure6
# =============================================================================

ABLATION_MATRIX: List[Dict[str, Any]] = [
    {
        "row": 1,
        "label": "Baseline (direct fine-tuning)",
        "method_id": "ddpm_finetune",
        "adaptor": False,
        "similarity_guidance": False,
        "adversarial_noise": False,
        "paper_context": "Figure 4 row 1; Figure 6 line 1",
    },
    {
        "row": 2,
        "label": "Adaptor only",
        "method_id": "ddpm_adaptor_only",
        "adaptor": True,
        "similarity_guidance": False,
        "adversarial_noise": False,
        "paper_context": "Figure 4 row 2",
        "note": "Fine-tuning only adaptor parameters, no guidance",
    },
    {
        "row": 3,
        "label": "DPMs-ANT w/o AN",
        "method_id": "ddpm_ant_wo_an",
        "adaptor": True,
        "similarity_guidance": True,
        "adversarial_noise": False,
        "paper_context": "Figure 4 row 3; Figure 6 line 2",
        "note": "Only similarity-guided training, no PGD adversarial noise",
    },
    {
        "row": 4,
        "label": "DPMs-ANT (full)",
        "method_id": "dpms_ant",
        "adaptor": True,
        "similarity_guidance": True,
        "adversarial_noise": True,
        "paper_context": "Figure 4 row 4; Figure 6 line 3",
        "note": "Full method: adaptor + similarity guidance + adversarial noise selection",
    },
]

# =============================================================================
# §9  Pretrained diffusion model URLs
#     Used for model initialisation before few-shot fine-tuning.
# reference_grounding: paper_addendum_section_5_2 pretrained_diffusion_urls
# =============================================================================

DIFFUSION_PRETRAINED_URLS: Dict[str, str] = {
    "ddpm_ffhq_256": (
        "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/"
        "ffhq_10m.pt"
    ),
    "ddpm_lsun_church_256": (
        "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/"
        "lsun_church_uncond_100M_1200K_fp16.pt"
    ),
    "ldm_ffhq_64": (
        "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/"
        "ldm_ffhq_256_ldm.pt"
    ),
}

# =============================================================================
# §10  Helper accessors
# =============================================================================


def get_fixed(key: str) -> Any:
    """Return a fixed anchor hyperparameter value by name."""
    if key not in FIXED_HYPERPARAMETERS:
        raise KeyError(
            f"Unknown fixed hyperparameter '{key}'. "
            f"Valid keys: {sorted(FIXED_HYPERPARAMETERS)}"
        )
    return FIXED_HYPERPARAMETERS[key]


def get_adaptor_config(framework: str) -> Dict[str, Any]:
    """Return Shift Adaptor config for the given framework ('ddpm' or 'ldm')."""
    fw = framework.lower()
    if fw not in ADAPTOR_CONFIGS:
        raise ValueError(f"Unknown framework '{framework}'. Choose 'ddpm' or 'ldm'.")
    return ADAPTOR_CONFIGS[fw]


def get_classifier_config(framework: str) -> Dict[str, Any]:
    """Return classifier fine-tuning config for the given framework."""
    fw = framework.lower()
    if fw not in CLASSIFIER_CONFIG:
        raise ValueError(f"Unknown framework '{framework}'. Choose 'ddpm' or 'ldm'.")
    return CLASSIFIER_CONFIG[fw]


def get_sweep(sweep_name: str) -> Dict[str, Any]:
    """Return the sweep configuration dict for the given sensitivity axis."""
    if sweep_name not in SWEEP_REGISTRY:
        raise KeyError(
            f"Unknown sweep '{sweep_name}'. "
            f"Valid sweeps: {sorted(SWEEP_REGISTRY)}"
        )
    return SWEEP_REGISTRY[sweep_name]


def get_experiment_tasks(
    source: Optional[str] = None,
    framework: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Filter experiment tasks by source domain and/or framework."""
    tasks = EXPERIMENT_TASKS
    if source is not None:
        tasks = [t for t in tasks if t["source"] == source]
    if framework is not None:
        tasks = [t for t in tasks if t["framework"] == framework]
    return tasks


def get_baseline(method_id: str) -> Dict[str, Any]:
    """Look up a baseline method by its registry id."""
    if method_id not in BASELINE_REGISTRY:
        raise KeyError(
            f"Unknown baseline '{method_id}'. "
            f"Valid ids: {sorted(BASELINE_REGISTRY)}"
        )
    return BASELINE_REGISTRY[method_id]


# =============================================================================
# §11  Artifact writer functions
#      Writes results/adversarial_trace.json and results/model_registry.json
# =============================================================================


def _resolve_output_dir() -> Path:
    """Return the canonical results directory, honouring PAPERBENCH_REPRO_ARTIFACT_DIR."""
    env_dir = os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR")
    if env_dir:
        base = Path(env_dir)
    else:
        base = Path("results")
    base.mkdir(parents=True, exist_ok=True)
    return base


def build_adversarial_trace_payload() -> Dict[str, Any]:
    """
    Construct the adversarial_trace artifact payload.

    Contains:
      - PGD / adversarial noise sweep configuration (Table 6)
      - Fixed adversarial hyperparameters anchored in the paper
      - Gamma sweep cross-reference (Table 5)
      - Iteration sweep cross-reference (Table 7 / Figure 6)
    """
    return {
        "artifact_type": "adversarial_trace",
        "paper": "DPMs-ANT: Bridging Data Gaps in Diffusion Models with "
                 "Adversarial Noise-Based Transfer Learning",
        "description": (
            "Adversarial noise selection sweep registry. "
            "Entries correspond to paper Tables 5, 6, 7 sensitivity experiments."
        ),
        "fixed_hyperparameters": {
            "omega": FIXED_HYPERPARAMETERS["omega"],
            "adversarial_inner_steps": FIXED_HYPERPARAMETERS["adversarial_inner_steps"],
            "similarity_guidance_scale": FIXED_HYPERPARAMETERS["similarity_guidance_scale"],
            "total_iterations": FIXED_HYPERPARAMETERS["total_iterations"],
            "batch_size": FIXED_HYPERPARAMETERS["batch_size"],
            "shot_count": FIXED_HYPERPARAMETERS["shot_count"],
        },
        "sweeps": {
            "omega": {
                "paper_table": SWEEP_OMEGA["paper_table"],
                "symbol": SWEEP_OMEGA["symbol"],
                "values": SWEEP_OMEGA["values"],
                "default": SWEEP_OMEGA["default"],
                "task": SWEEP_OMEGA["task"],
                "metrics": SWEEP_OMEGA["metrics"],
            },
            "gamma": {
                "paper_table": SWEEP_GAMMA["paper_table"],
                "symbol": SWEEP_GAMMA["symbol"],
                "values": SWEEP_GAMMA["values"],
                "default": SWEEP_GAMMA["default"],
                "task": SWEEP_GAMMA["task"],
                "metrics": SWEEP_GAMMA["metrics"],
            },
            "training_iterations": {
                "paper_table": SWEEP_ITERATIONS["paper_table"],
                "paper_figure": SWEEP_ITERATIONS["paper_figure"],
                "values": SWEEP_ITERATIONS["values"],
                "default": SWEEP_ITERATIONS["default"],
                "methods": SWEEP_ITERATIONS["methods"],
                "task": SWEEP_ITERATIONS["task"],
                "metrics": SWEEP_ITERATIONS["metrics"],
            },
            "shot_count": {
                "paper_table": SWEEP_SHOT_COUNT["paper_table"],
                "values": SWEEP_SHOT_COUNT["values"],
                "default": SWEEP_SHOT_COUNT["default"],
                "task": SWEEP_SHOT_COUNT["task"],
                "metrics": SWEEP_SHOT_COUNT["metrics"],
            },
        },
        "pgd_algorithm": {
            "description": (
                "PGD targeted adversarial noise selection. "
                "Outer loop: Algorithm 1; inner loop: PGD with step ω and "
                "adversarial_inner_steps iterations."
            ),
            "inner_steps": FIXED_HYPERPARAMETERS["adversarial_inner_steps"],
            "step_size": FIXED_HYPERPARAMETERS["omega"],
            "reference": "paper_semantic_chunk_005 adversarial_noise_selection_pgd",
        },
        "ablation_matrix": ABLATION_MATRIX,
    }


def build_model_registry_payload() -> Dict[str, Any]:
    """
    Construct the model_registry artifact payload.

    Contains:
      - Diffusion backbone pretrained URLs
      - Classifier pretrained URLs and fine-tuning config (addendum §5.2)
      - Shift Adaptor configuration per framework
      - Experiment task matrix (7 domain pairs)
      - Baseline method registry
    """
    return {
        "artifact_type": "model_registry",
        "paper": "DPMs-ANT: Bridging Data Gaps in Diffusion Models with "
                 "Adversarial Noise-Based Transfer Learning",
        "description": (
            "Model configuration registry: pretrained URLs, adaptor configs, "
            "classifier fine-tuning protocol (addendum §5.2), and experiment matrix."
        ),
        "diffusion_pretrained": DIFFUSION_PRETRAINED_URLS,
        "classifier": {
            "description": (
                "Domain classifiers used for similarity-guided training. "
                "Addendum §5.2: pre-trained models fine-tuned with last layer "
                "replaced to output 2 classes (source vs target)."
            ),
            "pretrained_urls": CLASSIFIER_PRETRAINED_URLS,
            "config": CLASSIFIER_CONFIG,
        },
        "shift_adaptor": {
            "description": (
                "Shift Adaptor: ψ^l(x) = f(x·W_down)·W_up, "
                "inserted residually at every UNet residual block. "
                "All W_down, W_up parameters initialised to zero."
            ),
            "configs": ADAPTOR_CONFIGS,
        },
        "experiment_tasks": EXPERIMENT_TASKS,
        "baselines": BASELINE_REGISTRY,
        "fixed_hyperparameters": FIXED_HYPERPARAMETERS,
        "sweep_registry": {k: v["values"] for k, v in SWEEP_REGISTRY.items()},
    }


def write_artifacts(output_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Materialise both declared artifact paths:
      results/adversarial_trace.json
      results/model_registry.json

    Returns a dict mapping artifact name → written file path string.
    """
    base = Path(output_dir) if output_dir else _resolve_output_dir()
    base.mkdir(parents=True, exist_ok=True)

    adversarial_path = base / "adversarial_trace.json"
    model_registry_path = base / "model_registry.json"

    adversarial_payload = build_adversarial_trace_payload()
    model_registry_payload = build_model_registry_payload()

    with open(adversarial_path, "w", encoding="utf-8") as fh:
        json.dump(adversarial_payload, fh, indent=2, default=str)

    with open(model_registry_path, "w", encoding="utf-8") as fh:
        json.dump(model_registry_payload, fh, indent=2, default=str)

    return {
        "adversarial_trace": str(adversarial_path),
        "model_registry": str(model_registry_path),
    }


# =============================================================================
# §12  Quick self-test / registry validation
# =============================================================================


def validate_registry() -> Dict[str, Any]:
    """
    Validate sweep registry consistency and return a summary dict.
    Raises AssertionError on any invariant violation.
    """
    errors: List[str] = []

    # Fixed hyperparameter type checks
    assert isinstance(FIXED_HYPERPARAMETERS["batch_size"], int), "batch_size must be int"
    assert FIXED_HYPERPARAMETERS["batch_size"] == 64, "batch_size anchor=64"
    assert FIXED_HYPERPARAMETERS["omega"] == 0.02, "omega anchor=0.02"
    assert FIXED_HYPERPARAMETERS["adversarial_inner_steps"] == 10, "inner_steps anchor=10"
    assert FIXED_HYPERPARAMETERS["total_iterations"] == 5000, "total_iter anchor=5000"
    assert FIXED_HYPERPARAMETERS["classifier_training_iterations"] == 300, "clf_iter anchor=300"
    assert FIXED_HYPERPARAMETERS["shot_count"] == 10, "shot_count anchor=10"
    assert FIXED_HYPERPARAMETERS["ddpm_adaptor_c"] == 4, "DDPM c anchor=4"
    assert FIXED_HYPERPARAMETERS["ddpm_adaptor_d"] == 8, "DDPM d anchor=8"
    assert FIXED_HYPERPARAMETERS["ldm_adaptor_c"] == 2, "LDM c anchor=2"
    assert FIXED_HYPERPARAMETERS["ldm_adaptor_d"] == 8, "LDM d anchor=8"
    assert FIXED_HYPERPARAMETERS["adaptor_init_zero"] is True, "init_zero anchor=True"
    assert FIXED_HYPERPARAMETERS["freeze_pretrained"] is True, "freeze_pretrained anchor=True"
    assert FIXED_HYPERPARAMETERS["similarity_guidance_scale"] == 5, "gamma anchor=5"

    # Sweep value completeness
    assert set(SWEEP_OMEGA["values"]) == {0.01, 0.02, 0.03, 0.04, 0.05}, \
        "omega sweep must include paper Table 6 values"
    assert set(SWEEP_GAMMA["values"]) == {1, 2, 3, 5, 7, 9, 10}, \
        "gamma sweep must include paper Table 5 values"
    assert set(SWEEP_ITERATIONS["values"]) == {0, 50, 100, 150, 200, 250, 300, 350}, \
        "iteration sweep must include Figure 6 values"
    assert set(SWEEP_SHOT_COUNT["values"]) == {10, 100}, \
        "shot_count sweep must include Table 3 values"

    # Experiment task count
    assert len(EXPERIMENT_TASKS) == 7, "Registry must include exactly 7 experiment tasks"

    # Classifier URLs
    assert "256x256_classifier.pt" in CLASSIFIER_PRETRAINED_URLS["ddpm"]
    assert "64x64_classifier.pt" in CLASSIFIER_PRETRAINED_URLS["ldm"]
    assert CLASSIFIER_CONFIG["ddpm"]["finetuned_num_classes"] == 2
    assert CLASSIFIER_CONFIG["ldm"]["finetuned_num_classes"] == 2

    return {
        "status": "valid",
        "fixed_hyperparameters_count": len(FIXED_HYPERPARAMETERS),
        "sweep_axes": list(SWEEP_REGISTRY.keys()),
        "experiment_tasks": len(EXPERIMENT_TASKS),
        "baselines": len(BASELINE_REGISTRY),
        "ablation_rows": len(ABLATION_MATRIX),
        "errors": errors,
    }


# =============================================================================
# §13  CLI entry-point / artifact materialisation
# =============================================================================

if __name__ == "__main__":
    import sys

    validation = validate_registry()
    print(json.dumps(validation, indent=2))

    written = write_artifacts()
    for name, path in written.items():
        print(f"[sweep_registry] Wrote {name} → {path}")

    sys.exit(0 if validation["status"] == "valid" else 1)