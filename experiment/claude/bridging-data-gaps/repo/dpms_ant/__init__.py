"""
dpms_ant/__init__.py
====================
DPMs-ANT Package – Bridging Data Gaps in Diffusion Models with Adversarial
Noise-Based Transfer Learning.

Paper: "Bridging Data Gaps in Diffusion Models with Adversarial Noise-Based
Transfer Learning"

This package implements the two core strategies of DPMs-ANT (method_id=ours):

  1. Similarity-Guided Training  (use_sim_guide=True)
     MobileNetV2 classifier φ fine-tuned 300 steps on source+target images,
     providing domain logits for noisy images (x_t, t).
     Loss: L_sim = γ · KL(∇log p_φ(y=S|x_t), ∇log p_φ(y=T|x_t))  γ=5

  2. Adversarial Noise Selection  (use_adv_noise=True)
     PGD optimisation over additive perturbation ε ∈ [-δ,δ]:
       ε* = argmax_{ε} L_simple(x_0 + ε)
     inner_steps=10, step-size omega=0.02, compatible with DDPM noise schedule.

  Algorithm 1  (implemented in dpms_ant/trainer/ant_trainer.py):
    For each iteration:
      Step 1: Sample x_0 ~ D_T  (10-shot target domain)
      Step 2: If use_adv_noise → PGD inner loop K=10 to find ε* (omega=0.02)
      Step 3: Forward diffuse x_t = √ᾱ_t · (x_0+ε*) + √(1-ᾱ_t) · ε_t
      Step 4: L_simple = ||ε_t - ε_θ_ψ(x_t,t)||²   (adaptor ψ active)
      Step 5: If use_sim_guide → L_sim = γ·KL(∇log p_φ(y=S|x_t),∇log p_φ(y=T|x_t))
      Step 6: L_total = L_simple + λ·L_sim
      Step 7: Update adaptor ψ  (only adaptor parameters are trained)

  Ablation switches (ablation_switches):
    use_sim_guide=True/False   → w/ or w/o similarity-guided training
    use_adv_noise=True/False   → w/ or w/o adversarial noise selection

reference_grounding: paper_method_core dpms_ant/__init__.py
reference_grounding: paper_semantic_chunk_003_02 classifier_loader_finetuning_introduction_figure_two_sets
reference_grounding: paper_semantic_chunk_010_classifier_loader_finetuning_adversarial_noise_selection_subsection_adversarial_noise
"""

from __future__ import annotations

import json
import logging
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Package metadata
# ─────────────────────────────────────────────────────────────────────────────

__version__ = "1.0.0"
__author__ = "DPMs-ANT (reproduction)"
__paper__ = (
    "Bridging Data Gaps in Diffusion Models with Adversarial Noise-Based Transfer Learning"
)

# ─────────────────────────────────────────────────────────────────────────────
# Method Registry
# Paper-canonical identification: method_id = "ours" → DPMs-ANT
# reference_grounding: paper_method_core method_registry
# ─────────────────────────────────────────────────────────────────────────────

METHOD_REGISTRY: Dict[str, Dict[str, Any]] = {
    # ── Canonical method (full DPMs-ANT) ────────────────────────────────────
    "ours": {
        "name": "DPMs-ANT",
        "full_name": "Diffusion Probabilistic Models – Adversarial Noise-based Transfer",
        "description": (
            "Transfer learning for DPMs using Shift Adaptor + Similarity-Guided "
            "Training (MobileNetV2 + KL loss) + Adversarial Noise Selection (PGD)."
        ),
        "paper": __paper__,
        "algorithm": "Algorithm 1",
        # ── Core strategies
        "strategies": ["similarity_guided_training", "adversarial_noise_selection"],
        "classifier": {
            "architecture": "MobileNetV2",
            "pretrained": "ImageNet",
            "finetune_steps": 300,
            "task": "binary source/target domain classification",
            "input": "(x_t, t)  — noisy image at diffusion timestep t",
            "output": "logits [p(y=S|x_t), p(y=T|x_t)]",
        },
        # ── Paper-fixed hyperparameters (addendum constraints)
        "hyperparameters": {
            "gamma": 5,                    # similarity guidance scale  (γ=5)
            "omega": 0.02,                 # PGD step size              (ω=0.02)
            "adversarial_inner_steps": 10,  # PGD inner iterations       (K=10)
            "total_iterations": 5000,       # main fine-tuning budget
            "ablation_iterations": 300,     # ablation/sensitivity budget
            "shot_count": 10,              # few-shot target images
            "batch_size": 64,              # training batch size
            "lambda_sim": 1.0,             # coefficient for L_sim in L_total
        },
        # ── Ablation configuration (both enabled = full method)
        "ablation_switches": {
            "use_sim_guide": True,
            "use_adv_noise": True,
        },
        # ── Shift Adaptor per-framework config
        # reference_grounding: paper_method_core dpms_ant/adaptor/shift_adaptor.py
        "shift_adaptor": {
            "ddpm": {"c": 4, "d": 8},  # DDPM: compression c=4, insertion d=8
            "ldm":  {"c": 2, "d": 8},  # LDM:  compression c=2, insertion d=8
        },
        # ── Loss terms
        "loss_terms": {
            "L_simple": "||ε_t - ε_θ_ψ(x_t,t)||²  (standard DDPM MSE loss)",
            "L_sim": (
                "γ · KL(∇log p_φ(y=S|x_t), ∇log p_φ(y=T|x_t))  "
                "similarity-guided loss using classifier gradients"
            ),
            "L_total": "L_simple + λ·L_sim",
        },
    },

    # ── Ablation: w/o Similarity-Guided Training ────────────────────────────
    "dpms_ant_no_sim": {
        "name": "DPMs-ANT (w/o Sim-Guide)",
        "description": "Ablation: DPMs-ANT without similarity-guided training (L_sim=0)",
        "ablation_switches": {"use_sim_guide": False, "use_adv_noise": True},
    },

    # ── Ablation: w/o Adversarial Noise Selection ───────────────────────────
    "dpms_ant_no_adv": {
        "name": "DPMs-ANT (w/o Adv-Noise)",
        "description": "Ablation: DPMs-ANT without adversarial noise selection (ε=0)",
        "ablation_switches": {"use_sim_guide": True, "use_adv_noise": False},
    },

    # ── Ablation: vanilla (neither strategy) ────────────────────────────────
    "dpms_ant_vanilla": {
        "name": "DPMs-ANT (vanilla, no strategies)",
        "description": "Ablation: Shift Adaptor only, no similarity guidance, no adversarial noise",
        "ablation_switches": {"use_sim_guide": False, "use_adv_noise": False},
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Baseline Registry (Table 2 comparison baselines)
# ─────────────────────────────────────────────────────────────────────────────

BASELINE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "freezeD":  {"name": "FreezeD",  "type": "gan_transfer",        "framework": "gan"},
    "minegan":  {"name": "MineGAN",  "type": "gan_transfer",        "framework": "gan"},
    "ada":      {"name": "ADA",      "type": "gan_augmentation",    "framework": "gan"},
    "fastgan":  {"name": "FastGAN",  "type": "lightweight_gan",     "framework": "gan"},
    "ewc":      {"name": "EWC",      "type": "regularization",      "framework": "diffusion"},
    "cdc":      {"name": "CDC",      "type": "diffusion_transfer",  "framework": "diffusion"},
    "ddpm_pa":  {"name": "DDPM-PA",  "type": "partial_adaptation",  "framework": "ddpm"},
    "ldm_pa":   {"name": "LDM-PA",   "type": "partial_adaptation",  "framework": "ldm"},
}

# ─────────────────────────────────────────────────────────────────────────────
# Dataset Registry – 7 target domains (Table 2)
# reference_grounding: paper_semantic_chunk_012 10-shot image generation experiments
# ─────────────────────────────────────────────────────────────────────────────

DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "babies": {
        "name": "Babies",
        "source_domain": "ffhq",
        "target_domain": "babies",
        "shot_count": 10,
        "framework": "ddpm",
        "image_size": 256,
        "table2_fid_reference": 46.70,    # paper Table 2 DPMs-ANT result
    },
    "sunglasses": {
        "name": "Sunglasses",
        "source_domain": "ffhq",
        "target_domain": "sunglasses",
        "shot_count": 10,
        "framework": "ddpm",
        "image_size": 256,
        "table2_fid_reference": 20.06,
    },
    "raphael_peale": {
        "name": "Raphael Peale",
        "source_domain": "ffhq",
        "target_domain": "raphael_peale",
        "shot_count": 10,
        "framework": "ddpm",
        "image_size": 256,
        "table2_fid_reference": None,
    },
    "sketches": {
        "name": "Sketches",
        "source_domain": "ffhq",
        "target_domain": "sketches",
        "shot_count": 10,
        "framework": "ddpm",
        "image_size": 256,
        "table2_fid_reference": None,
    },
    "modigliani": {
        "name": "Modigliani",
        "source_domain": "ffhq",
        "target_domain": "modigliani",
        "shot_count": 10,
        "framework": "ddpm",
        "image_size": 256,
        "table2_fid_reference": None,
    },
    "haunted_houses": {
        "name": "Haunted Houses",
        "source_domain": "lsun_church",
        "target_domain": "haunted_houses",
        "shot_count": 10,
        "framework": "ddpm",
        "image_size": 256,
        "table2_fid_reference": None,
    },
    "landscape": {
        "name": "Landscape",
        "source_domain": "lsun_church",
        "target_domain": "landscape",
        "shot_count": 10,
        "framework": "ddpm",
        "image_size": 256,
        "table2_fid_reference": None,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Experiment Registry – cross product of methods × datasets
# ─────────────────────────────────────────────────────────────────────────────

EXPERIMENT_REGISTRY: Dict[str, Dict[str, Any]] = {}

for _did, _dinfo in DATASET_REGISTRY.items():
    for _mid in ["ours", "dpms_ant_no_sim", "dpms_ant_no_adv", "dpms_ant_vanilla"]:
        _exp_id = f"{_mid}__{_did}"
        _switches = METHOD_REGISTRY[_mid].get(
            "ablation_switches", {"use_sim_guide": True, "use_adv_noise": True}
        )
        EXPERIMENT_REGISTRY[_exp_id] = {
            "experiment_id": _exp_id,
            "method_id": _mid,
            "dataset_id": _did,
            "source_domain": _dinfo["source_domain"],
            "target_domain": _dinfo["target_domain"],
            "framework": _dinfo["framework"],
            "shot_count": _dinfo["shot_count"],
            "ablation_switches": _switches,
            "metrics": ["fid", "intra_lpips", "fidelity_score", "accuracy"],
            "result_path": f"results/{_exp_id}_metrics.json",
        }

# ─────────────────────────────────────────────────────────────────────────────
# Algorithm 1 – documented reference implementation
# reference_grounding: paper_method_core dpms_ant/trainer/ant_trainer.py
# ─────────────────────────────────────────────────────────────────────────────

ALGORITHM_1_SPEC: Dict[str, Any] = {
    "name": "DPMs-ANT Training Algorithm (Algorithm 1)",
    "paper_section": "Section 3.3 / Algorithm 1",
    "inputs": {
        "pretrained_model": "p_θ — source-domain DPM (DDPM or LDM)",
        "target_dataset": "D_T — few-shot target images (default 10)",
        "classifier": "φ — MobileNetV2 domain classifier (finetune_steps=300)",
        "adaptor": "ψ — Shift Adaptor (trainable; source model frozen)",
    },
    "hyperparameters": {
        "gamma":                   5,
        "omega":                   0.02,
        "adversarial_inner_steps": 10,
        "total_iterations":        5000,
        "lambda_sim":              1.0,
        "shot_count":              10,
        "batch_size":              64,
    },
    "steps": [
        "1. Sample x_0 ~ D_T  (target domain image)",
        "2. [use_adv_noise] PGD inner loop K=adversarial_inner_steps:",
        "     ε ← 0",
        "     for k in 1..K:",
        "       Compute L_simple(x_0 + ε)",
        "       ε ← clip(ε + ω·sign(∇_ε L_simple), −δ, δ)  [PGD ascent step]",
        "     x_0_adv ← x_0 + ε*  (adversarially perturbed image)",
        "3. Sample t ~ Uniform{1..T},  ε_t ~ N(0,I)",
        "4. Forward diffuse: x_t = √ᾱ_t · x_0_adv + √(1−ᾱ_t) · ε_t",
        "5. L_simple = ||ε_t − ε_{θ,ψ}(x_t, t)||²  (adaptor ψ active)",
        "6. [use_sim_guide] Compute similarity guidance:",
        "     g_S ← ∇_{x_t} log p_φ(y=S|x_t)  (source-domain classifier gradient)",
        "     g_T ← ∇_{x_t} log p_φ(y=T|x_t)  (target-domain classifier gradient)",
        "     L_sim ← γ · KL(g_S, g_T)",
        "7. L_total = L_simple + λ·L_sim   (or L_simple if not use_sim_guide)",
        "8. Backpropagate L_total; update adaptor ψ only (source model θ frozen)",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# Lazy import helpers
# All heavy ML packages (torch, torchvision, etc.) are imported lazily inside
# functions/classes so that `import dpms_ant` succeeds in a minimal environment.
# ─────────────────────────────────────────────────────────────────────────────


def _check_pkg(pkg_name: str) -> bool:
    """Return True if *pkg_name* can be imported, without actually importing it."""
    import importlib.util
    spec = importlib.util.find_spec(pkg_name)
    return spec is not None


def _lazy_import(module_path: str):
    """Lazily import a module; returns None if unavailable."""
    import importlib
    try:
        return importlib.import_module(module_path)
    except ImportError as exc:
        logger.debug("Optional module %s not available: %s", module_path, exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Factory functions
# ─────────────────────────────────────────────────────────────────────────────


def get_domain_classifier(
    pretrained: bool = True,
    finetune_steps: int = 300,
    num_classes: int = 2,
):
    """
    Return a MobileNetV2-based domain classifier.

    MobileNet分类器，从ImageNet预训练权重微调300步，支持noisy image的source vs target二分类
    reference_grounding: paper_semantic_chunk_003_02 classifier_loader_finetuning

    Classifier interface:
      forward(x_t: Tensor[B,C,H,W], t: Tensor[B]) → logits: Tensor[B, 2]
        logits[:, 0] = log p_φ(y=S|x_t)   (source domain)
        logits[:, 1] = log p_φ(y=T|x_t)   (target domain)

    Args:
        pretrained:      Load ImageNet weights (default True)
        finetune_steps:  Fine-tuning iterations on source+target images (paper=300)
        num_classes:     Output classes (2 = binary source/target)

    Returns:
        DomainClassifier instance
    """
    from dpms_ant.classifier.domain_classifier import DomainClassifier  # lazy path
    return DomainClassifier(
        pretrained=pretrained,
        finetune_steps=finetune_steps,
        num_classes=num_classes,
    )


def get_shift_adaptor(framework: str = "ddpm", **kwargs):
    """
    Return a Shift Adaptor module for the specified framework.

    ψ^l(x^{l−1}) = f(x^{l−1}·W_down)·W_up
      W_down: R^{w×h×r} → R^{(w/c)×(h/c)×d}   (compression)
      W_up:   R^{(w/c)×(h/c)×d} → R^{w×h×r}   (reconstruction)
      f = GELU non-linearity
    reference_grounding: paper_method_core dpms_ant/adaptor/shift_adaptor.py

    Args:
        framework:  "ddpm" (c=4, d=8) or "ldm" (c=2, d=8)
        **kwargs:   Passed to ShiftAdaptor constructor

    Returns:
        ShiftAdaptor instance initialised with zero weights
    """
    from dpms_ant.adaptor.shift_adaptor import ShiftAdaptor  # lazy path
    cfg = METHOD_REGISTRY["ours"]["shift_adaptor"]
    params = cfg.get(framework, cfg["ddpm"])
    return ShiftAdaptor(c=params["c"], d=params["d"], **kwargs)


def get_ant_trainer(
    model,
    adaptor,
    classifier,
    config: Optional[Dict[str, Any]] = None,
):
    """
    Return an ANTTrainer implementing Algorithm 1.

    Algorithm 1完整训练循环，集成两策略及消融开关(use_sim_guide/use_adv_noise)
    reference_grounding: paper_method_core dpms_ant/trainer/ant_trainer.py

    Args:
        model:      DDPM or LDM diffusion model (source weights, frozen)
        adaptor:    ShiftAdaptor instance (only trainable parameters)
        classifier: DomainClassifier instance (fixed after pre-training 300 steps)
        config:     Dict containing:
                      use_sim_guide       (default True)
                      use_adv_noise       (default True)
                      gamma               (default 5)
                      omega               (default 0.02)
                      adversarial_inner_steps (default 10)
                      total_iterations    (default 5000)
                      lambda_sim          (default 1.0)

    Returns:
        ANTTrainer instance
    """
    from dpms_ant.trainer.ant_trainer import ANTTrainer  # lazy path
    default_cfg: Dict[str, Any] = {
        "use_sim_guide":           True,
        "use_adv_noise":           True,
        "gamma":                   5.0,
        "omega":                   0.02,
        "adversarial_inner_steps": 10,
        "total_iterations":        5000,
        "ablation_iterations":     300,
        "lambda_sim":              1.0,
        "batch_size":              64,
        "shot_count":              10,
    }
    if config:
        default_cfg.update(config)
    return ANTTrainer(
        model=model, adaptor=adaptor, classifier=classifier, config=default_cfg
    )


def get_similarity_guidance(classifier, gamma: float = 5.0):
    """
    Return the similarity-guidance loss module.

    L_sim = γ · KL(∇log p_φ(y=S|x_t), ∇log p_φ(y=T|x_t))
    reference_grounding: paper_semantic_chunk_010 similarity_guided_training

    Gradient computation interface:
        grad_source = ∇_{x_t} log p_φ(y=S|x_t)
        grad_target = ∇_{x_t} log p_φ(y=T|x_t)
        L_sim = γ · KL(softmax(grad_source), softmax(grad_target))

    Args:
        classifier: DomainClassifier providing domain logits
        gamma:      Guidance scale (paper default: 5)

    Returns:
        SimilarityGuidance instance
    """
    from dpms_ant.trainer.similarity_guidance import SimilarityGuidance  # lazy
    return SimilarityGuidance(classifier=classifier, gamma=gamma)


def get_adversarial_noise(
    model,
    inner_steps: int = 10,
    omega: float = 0.02,
):
    """
    Return the adversarial noise selector.

    PGD optimisation:
      ε_{k+1} = clip(ε_k + ω·sign(∇_ε L_simple(x_0+ε_k)), −δ, δ)
      ε* = ε_K  after K=inner_steps iterations
    reference_grounding: paper_semantic_chunk_010 adversarial_noise_selection

    Compatibility: perturbation budget δ is derived from the DDPM noise schedule
    (αt-based scaling) so that the adversarial ε does not exceed meaningful noise
    relative to the current diffusion timestep.

    Args:
        model:       DDPM/LDM model providing noise schedule (alphas_cumprod)
        inner_steps: PGD iterations K (paper default: 10)
        omega:       PGD step size ω (paper default: 0.02)

    Returns:
        AdversarialNoise instance
    """
    from dpms_ant.trainer.adversarial_noise import AdversarialNoise  # lazy
    return AdversarialNoise(model=model, inner_steps=inner_steps, omega=omega)


# ─────────────────────────────────────────────────────────────────────────────
# Metric interfaces
# reference_grounding: paper_semantic_chunk_012 evaluation metrics
# ─────────────────────────────────────────────────────────────────────────────


def compute_fid(real_features, generated_features) -> float:
    """
    FID = ||μ_r − μ_g||² + Tr(Σ_r + Σ_g − 2·sqrtm(Σ_r·Σ_g))

    Args:
        real_features:      numpy array [N, D] – Inception-v3 pool3 features
        generated_features: numpy array [M, D] – Inception-v3 pool3 features

    Returns:
        Scalar FID value (lower is better)
    """
    from dpms_ant.evaluation.fid import compute_fid as _fid
    return _fid(real_features, generated_features)


def compute_intra_lpips(generated_images) -> float:
    """
    Intra-LPIPS diversity = mean pairwise LPIPS between generated images.
    Higher score → more diverse generated images.

    Args:
        generated_images: Tensor or list of images [N, C, H, W] in [-1, 1]

    Returns:
        Scalar Intra-LPIPS value (higher is better)
    """
    from dpms_ant.evaluation.metrics import compute_intra_lpips as _intra
    return _intra(generated_images)


def compute_fidelity_score(generated_images, target_images) -> float:
    """
    Fidelity score = mean minimum LPIPS distance from each generated image
    to the nearest target-domain image.
    Lower → more faithful to target domain.

    Args:
        generated_images: Tensor [N, C, H, W] in [-1, 1]
        target_images:    Tensor [M, C, H, W] in [-1, 1]  (10-shot target images)

    Returns:
        Scalar fidelity score (lower is better)
    """
    from dpms_ant.evaluation.metrics import compute_fidelity_score as _fid_score
    return _fid_score(generated_images, target_images)


def compute_domain_accuracy(classifier, images, true_labels) -> float:
    """
    Domain classifier accuracy: fraction of images correctly classified as
    source or target domain.

    Args:
        classifier:  DomainClassifier instance
        images:      Tensor [N, C, H, W]
        true_labels: Tensor [N] — 0=source, 1=target

    Returns:
        Accuracy in [0, 1]
    """
    from dpms_ant.evaluation.metrics import compute_domain_accuracy as _acc
    return _acc(classifier, images, true_labels)


# ─────────────────────────────────────────────────────────────────────────────
# Artifact writers
# Populate results/ directory with registry and schema artifacts.
# reference_grounding: paper_method_core artifact_writers
# ─────────────────────────────────────────────────────────────────────────────


def _artifact_dir(override: Optional[Path] = None) -> Path:
    """Resolve canonical artifact output directory."""
    if override is not None:
        d = Path(override)
    else:
        env_dir = os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR", "")
        d = Path(env_dir) if env_dir else Path("results")
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_method_registry(artifact_dir: Optional[Path] = None) -> Path:
    """
    Write results/method_registry.json.
    Contains canonical DPMs-ANT method registration (method_id=ours) and
    all ablation variants plus baseline identifiers.

    Artifact: results/method_registry.json
    """
    out_dir = _artifact_dir(artifact_dir)
    out_path = out_dir / "method_registry.json"
    payload: Dict[str, Any] = {
        "schema_version": "1.0",
        "paper": __paper__,
        "canonical_method_id": "ours",
        "canonical_method_name": "DPMs-ANT",
        "algorithm": "Algorithm 1",
        "methods": METHOD_REGISTRY,
        "baselines": BASELINE_REGISTRY,
        "ablation_switch_semantics": {
            "use_sim_guide": (
                "Enable similarity-guided training: "
                "L_sim = γ·KL(∇log p_φ(y=S|x_t), ∇log p_φ(y=T|x_t)), γ=5"
            ),
            "use_adv_noise": (
                "Enable adversarial noise selection: "
                "PGD inner loop K=10, ω=0.02, ε∈[-δ,δ]"
            ),
        },
        "loss_terms": ALGORITHM_1_SPEC["inputs"],
    }
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote method registry → %s", out_path)
    return out_path


def write_dataset_registry(artifact_dir: Optional[Path] = None) -> Path:
    """
    Write results/dataset_registry.json.
    Contains all 7 target domains from the paper (Table 2).

    Artifact: results/dataset_registry.json
    """
    out_dir = _artifact_dir(artifact_dir)
    out_path = out_dir / "dataset_registry.json"
    payload: Dict[str, Any] = {
        "schema_version": "1.0",
        "paper": __paper__,
        "description": "7 target domains evaluated in Table 2 of the paper (10-shot)",
        "default_shot_count": 10,
        "source_domains": ["ffhq", "lsun_church"],
        "datasets": DATASET_REGISTRY,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote dataset registry → %s", out_path)
    return out_path


def write_experiment_registry(artifact_dir: Optional[Path] = None) -> Path:
    """
    Write results/experiment_registry.json.
    Contains the full experiment matrix (methods × target domains).

    Artifact: results/experiment_registry.json
    """
    out_dir = _artifact_dir(artifact_dir)
    out_path = out_dir / "experiment_registry.json"
    payload: Dict[str, Any] = {
        "schema_version": "1.0",
        "paper": __paper__,
        "algorithm": "Algorithm 1",
        "hyperparameters": METHOD_REGISTRY["ours"]["hyperparameters"],
        "ablation_switch_semantics": {
            "use_sim_guide": "similarity-guided training (γ=5, KL classifier gradient loss)",
            "use_adv_noise": "adversarial noise selection (PGD, K=10, ω=0.02)",
        },
        "experiment_count": len(EXPERIMENT_REGISTRY),
        "experiments": EXPERIMENT_REGISTRY,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote experiment registry → %s", out_path)
    return out_path


def write_environment_registry(artifact_dir: Optional[Path] = None) -> Path:
    """
    Write results/environment_registry.json.
    Contains system/platform info and package availability.

    Artifact: results/environment_registry.json
    """
    out_dir = _artifact_dir(artifact_dir)
    out_path = out_dir / "environment_registry.json"

    def _pkg_version(pkg: str) -> str:
        try:
            import importlib.metadata
            return importlib.metadata.version(pkg)
        except Exception:
            return "not_available"

    payload: Dict[str, Any] = {
        "schema_version": "1.0",
        "python_version": sys.version,
        "platform": platform.platform(),
        "packages": {
            "torch":         _pkg_version("torch"),
            "torchvision":   _pkg_version("torchvision"),
            "numpy":         _pkg_version("numpy"),
            "Pillow":        _pkg_version("Pillow"),
            "lpips":         _pkg_version("lpips"),
            "pytorch-fid":   _pkg_version("pytorch-fid"),
            "timm":          _pkg_version("timm"),
            "scipy":         _pkg_version("scipy"),
        },
        "optional_available": {
            "torch":       _check_pkg("torch"),
            "torchvision": _check_pkg("torchvision"),
            "lpips":       _check_pkg("lpips"),
            "timm":        _check_pkg("timm"),
        },
    }
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote environment registry → %s", out_path)
    return out_path


def write_artifact_manifest(artifact_dir: Optional[Path] = None) -> Path:
    """
    Write results/artifact_manifest.json listing all expected output artifacts.

    Artifact: results/artifact_manifest.json
    """
    out_dir = _artifact_dir(artifact_dir)
    out_path = out_dir / "artifact_manifest.json"
    payload: Dict[str, Any] = {
        "schema_version": "1.0",
        "paper": __paper__,
        "registry_artifacts": [
            "results/metrics.json",
            "results/dataset_registry.json",
            "results/environment_registry.json",
            "results/experiment_registry.json",
            "results/artifact_manifest.json",
            "results/method_registry.json",
            "results/data_manifest.json",
            "results/scope_report.json",
        ],
        "table_artifacts": [
            "results/table2_fid.json",
            "results/table1_intra_lpips.json",
            "results/table3_domain_accuracy.json",
            "results/table4_ldm_comparison.json",
            "results/table5_ablation.json",
            "results/table6_sensitivity.json",
            "results/table7_fidelity.json",
            "results/table8_memory.json",
        ],
        "per_experiment_artifacts": [
            f"results/{eid}_metrics.json"
            for eid in list(EXPERIMENT_REGISTRY.keys())[:5]
        ],
        "status": "declared – run evaluate.py to populate with real results",
    }
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote artifact manifest → %s", out_path)
    return out_path


def write_metrics_schema(artifact_dir: Optional[Path] = None) -> Path:
    """
    Write results/metrics.json with metric schema.
    Populated with actual values by the evaluation pipeline (evaluate.py).

    Artifact: results/metrics.json
    """
    out_dir = _artifact_dir(artifact_dir)
    out_path = out_dir / "metrics.json"
    if out_path.exists():
        # Don't overwrite real evaluation results
        return out_path
    payload: Dict[str, Any] = {
        "schema_version": "1.0",
        "paper": __paper__,
        "metric_definitions": {
            "fid": {
                "description": "Fréchet Inception Distance",
                "formula": (
                    "||μ_r − μ_g||² + Tr(Σ_r + Σ_g − 2·sqrtm(Σ_r·Σ_g))"
                ),
                "lower_is_better": True,
                "table": "Table 2",
            },
            "intra_lpips": {
                "description": "Intra-LPIPS diversity (mean pairwise LPIPS)",
                "lower_is_better": False,
                "table": "Table 1 / Table 4",
            },
            "fidelity_score": {
                "description": "Fidelity score (mean min LPIPS to target images)",
                "lower_is_better": True,
                "table": "Table 7",
            },
            "accuracy": {
                "description": "Domain classifier accuracy (source vs target)",
                "lower_is_better": False,
                "table": "Table 3",
            },
        },
        "results": {},
        "status": "schema_only – run evaluate.py to populate",
    }
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote metrics schema → %s", out_path)
    return out_path


def write_all_registries(artifact_dir: Optional[Path] = None) -> Dict[str, Path]:
    """
    Write all registry artifacts in one call.
    Called by smoke/dry-run validation to satisfy artifact closure.

    Returns:
        Dict mapping artifact key → written Path
    """
    out_dir = _artifact_dir(artifact_dir)
    return {
        "method_registry":      write_method_registry(out_dir),
        "dataset_registry":     write_dataset_registry(out_dir),
        "experiment_registry":  write_experiment_registry(out_dir),
        "environment_registry": write_environment_registry(out_dir),
        "artifact_manifest":    write_artifact_manifest(out_dir),
        "metrics":              write_metrics_schema(out_dir),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Classifier gradient helpers
# reference_grounding: paper_semantic_chunk_010 classifier gradient computation
# ─────────────────────────────────────────────────────────────────────────────


def compute_classifier_gradients(
    classifier,
    x_t,
    t,
    source_class: int = 0,
    target_class: int = 1,
):
    """
    Compute ∇_{x_t} log p_φ(y=S|x_t) and ∇_{x_t} log p_φ(y=T|x_t).

    Used by similarity guidance:
        g_S = ∇_{x_t} log p_φ(y=S|x_t)
        g_T = ∇_{x_t} log p_φ(y=T|x_t)
        L_sim = γ · KL(softmax(g_S), softmax(g_T))

    reference_grounding: paper_semantic_chunk_010 adversarial_noise_selection

    Args:
        classifier:    DomainClassifier with forward(x_t, t) → logits [B, 2]
        x_t:           Tensor [B, C, H, W] — noisy image at timestep t
        t:             Tensor [B] — diffusion timestep
        source_class:  Class index for source domain (default 0)
        target_class:  Class index for target domain (default 1)

    Returns:
        grad_source: Tensor [B, C, H, W] — ∇ log p_φ(y=S|x_t)
        grad_target: Tensor [B, C, H, W] — ∇ log p_φ(y=T|x_t)
    """
    # Heavy torch import is done lazily here
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError(
            "torch is required for compute_classifier_gradients"
        ) from exc

    x_t_s = x_t.detach().requires_grad_(True)

    logits = classifier(x_t_s, t)               # [B, 2]
    log_probs = F.log_softmax(logits, dim=-1)   # [B, 2]

    # ∇_{x_t} log p_φ(y=S|x_t)
    log_p_source = log_probs[:, source_class].sum()
    grad_source = torch.autograd.grad(
        log_p_source, x_t_s, create_graph=False, retain_graph=True
    )[0]

    # ∇_{x_t} log p_φ(y=T|x_t)
    log_p_target = log_probs[:, target_class].sum()
    grad_target = torch.autograd.grad(
        log_p_target, x_t_s, create_graph=False, retain_graph=False
    )[0]

    return grad_source.detach(), grad_target.detach()


def pgd_adversarial_noise(
    model,
    x_0,
    t,
    alpha_bar_t,
    delta: float,
    inner_steps: int = 10,
    omega: float = 0.02,
):
    """
    PGD adversarial noise selection (inner optimisation loop).

    ε_{k+1} = clip(ε_k + ω · sign(∇_ε L_simple(x_0+ε_k)), −δ, δ)
    ε* = ε_K  (result after K inner steps)

    Compatible with DDPM noise schedule: perturbation budget δ is typically
    set as a fraction of √(1−ᾱ_t) to respect the diffusion noise level.

    reference_grounding: paper_semantic_chunk_010 adversarial_noise_selection

    Args:
        model:       Diffusion model with loss_simple(x_noisy, t, noise) method
        x_0:         Tensor [B, C, H, W] — clean target image
        t:           Tensor [B] — randomly sampled diffusion timestep
        alpha_bar_t: Tensor [B, 1, 1, 1] — ᾱ_t from noise schedule
        delta:       Perturbation budget (clip bound)
        inner_steps: K (paper default: 10)
        omega:       PGD step size ω (paper default: 0.02)

    Returns:
        eps_star: Tensor [B, C, H, W] — optimal adversarial perturbation
    """
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError(
            "torch is required for pgd_adversarial_noise"
        ) from exc

    eps = torch.zeros_like(x_0)

    for _ in range(inner_steps):
        eps = eps.detach().requires_grad_(True)
        x_adv = x_0 + eps

        # Sample fresh Gaussian noise for L_simple computation
        noise = torch.randn_like(x_0)
        sqrt_alpha_bar = alpha_bar_t.sqrt()
        sqrt_one_minus = (1.0 - alpha_bar_t).sqrt()
        x_t = sqrt_alpha_bar * x_adv + sqrt_one_minus * noise

        # Compute L_simple = ||noise - ε_θ(x_t, t)||²
        l_simple = model.compute_loss_simple(x_t, t, noise)

        # PGD ascent step (maximise L_simple)
        grad = torch.autograd.grad(l_simple.mean(), eps)[0]
        eps = (eps + omega * grad.sign()).clamp(-delta, delta)

    return eps.detach()


def compute_l_sim(
    classifier,
    x_t,
    t,
    gamma: float = 5.0,
):
    """
    Compute similarity guidance loss.

    L_sim = γ · KL(P_S || P_T)
    where P_S = softmax(∇_{x_t} log p_φ(y=S|x_t))
          P_T = softmax(∇_{x_t} log p_φ(y=T|x_t))

    reference_grounding: paper_semantic_chunk_010 similarity_guided_training

    Args:
        classifier: DomainClassifier
        x_t:        Tensor [B, C, H, W] noisy image
        t:          Tensor [B] timestep
        gamma:      Guidance scale (paper default: 5)

    Returns:
        l_sim: scalar Tensor — similarity guidance loss
    """
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError("torch is required for compute_l_sim") from exc

    g_s, g_t = compute_classifier_gradients(classifier, x_t, t)

    # Flatten spatial dims for softmax
    g_s_flat = g_s.view(g_s.shape[0], -1)   # [B, C*H*W]
    g_t_flat = g_t.view(g_t.shape[0], -1)   # [B, C*H*W]

    p_s = F.softmax(g_s_flat, dim=-1)        # [B, C*H*W]
    p_t = F.softmax(g_t_flat, dim=-1)        # [B, C*H*W]

    # KL(P_S || P_T)
    kl = F.kl_div(
        p_t.log().clamp(min=-100.0),
        p_s,
        reduction="batchmean",
        log_target=False,
    )
    return gamma * kl


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

__all__: List[str] = [
    # Metadata
    "__version__",
    "__paper__",
    # Registries
    "METHOD_REGISTRY",
    "BASELINE_REGISTRY",
    "DATASET_REGISTRY",
    "EXPERIMENT_REGISTRY",
    "ALGORITHM_1_SPEC",
    # Factory functions
    "get_domain_classifier",
    "get_shift_adaptor",
    "get_ant_trainer",
    "get_similarity_guidance",
    "get_adversarial_noise",
    # Core computation helpers
    "compute_classifier_gradients",
    "pgd_adversarial_noise",
    "compute_l_sim",
    # Metric interfaces
    "compute_fid",
    "compute_intra_lpips",
    "compute_fidelity_score",
    "compute_domain_accuracy",
    # Artifact writers
    "write_method_registry",
    "write_dataset_registry",
    "write_experiment_registry",
    "write_environment_registry",
    "write_artifact_manifest",
    "write_metrics_schema",
    "write_all_registries",
]