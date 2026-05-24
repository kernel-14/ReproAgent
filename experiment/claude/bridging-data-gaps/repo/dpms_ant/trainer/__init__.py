# dpms_ant/trainer/__init__.py
# =============================================================================
# DPMs-ANT Trainer Package Initializer
# Paper: "Bridging Data Gaps in Diffusion Models with Adversarial Noise-Based
#         Transfer Learning"
#
# reference_grounding: paper_method_core dpms_ant/trainer/__init__.py
# reference_grounding: paper_semantic_chunk_003_02 classifier_loader_finetuning
# reference_grounding: paper_semantic_chunk_010 adversarial_noise_selection
#
# Exposes:
#   - ANTTrainer: Algorithm 1 full training loop
#   - SimilarityGuidance: KL-divergence similarity guidance loss (γ=5)
#   - AdversarialNoiseSelector: PGD inner loop (steps=10, ω=0.02)
#   - METHOD_REGISTRY: selectable method/baseline adapters
#   - SWEEP_REGISTRY: bounded parameter sweeps
#   - FIXED_HYPERPARAMETERS: paper-anchored constants
# =============================================================================

from __future__ import annotations

# ---------------------------------------------------------------------------
# Paper-anchored fixed hyperparameters
# reference_grounding: paper_semantic_chunk_012 fixed_hyperparameters
# ---------------------------------------------------------------------------
FIXED_HYPERPARAMETERS: dict = {
    # anchor: 5000_iterations – total fine-tuning budget
    "total_iterations": 5000,
    # anchor: 300_training_iterations – classifier training steps
    "classifier_training_iterations": 300,
    # anchor: 10_shot_setting
    "shot_count": 10,
    # anchor: gamma_5 – similarity guidance weight
    "gamma": 5,
    # anchor: omega_0.02 – PGD step size
    "omega": 0.02,
    # anchor: adversarial_inner_steps_10
    "adversarial_inner_steps": 10,
    # anchor: batch_size_64
    "batch_size": 64,
    # DDPM Shift Adaptor bottleneck dims
    "ddpm_adaptor_c": 4,
    "ddpm_adaptor_d": 8,
    # LDM Shift Adaptor bottleneck dims
    "ldm_adaptor_c": 2,
    "ldm_adaptor_d": 8,
    # Adaptor initialization: all weights = 0
    "adaptor_init_zero": True,
    # Non-adaptor parameters: fully frozen
    "freeze_non_adaptor": True,
}

# ---------------------------------------------------------------------------
# Method / baseline registry
# reference_grounding: paper_method_core method_registry
# Covers: Ours | GAN | DDPM | FFHQ | LPIPS | TGAN | ADA | EWC | CDC | DCL |
#         DDPM-PA | DDPM-ANT
# ---------------------------------------------------------------------------
METHOD_REGISTRY: dict = {
    # ── DPMs-ANT variants ──────────────────────────────────────────────────
    "ours": {
        "id": "dpms_ant",
        "description": "DPMs-ANT: Similarity-Guided Training + Adversarial Noise Selection",
        "use_sim_guide": True,
        "use_adv_noise": True,
        "adaptor": True,
    },
    "dpms_ant": {
        "id": "dpms_ant",
        "description": "DPMs-ANT full method (alias for ours)",
        "use_sim_guide": True,
        "use_adv_noise": True,
        "adaptor": True,
    },
    "similarity_guided_training": {
        "id": "similarity_guided_training",
        "description": "Ablation: similarity guidance only, no adversarial noise",
        "use_sim_guide": True,
        "use_adv_noise": False,
        "adaptor": True,
    },
    "adversarial_noise_selection": {
        "id": "adversarial_noise_selection",
        "description": "Ablation: adversarial noise only, no similarity guidance",
        "use_sim_guide": False,
        "use_adv_noise": True,
        "adaptor": True,
    },
    # ── Diffusion model baselines ──────────────────────────────────────────
    "diffusion_model": {
        "id": "diffusion_model",
        "description": "Vanilla diffusion model fine-tuning (no adaptor, no guidance)",
        "use_sim_guide": False,
        "use_adv_noise": False,
        "adaptor": False,
    },
    "ddpm": {
        "id": "ddpm",
        "description": "DDPM fine-tuning baseline",
        "use_sim_guide": False,
        "use_adv_noise": False,
        "adaptor": False,
        "framework": "ddpm",
    },
    "ldm": {
        "id": "ldm",
        "description": "LDM fine-tuning baseline",
        "use_sim_guide": False,
        "use_adv_noise": False,
        "adaptor": False,
        "framework": "ldm",
    },
    "ddim": {
        "id": "ddim",
        "description": "DDIM sampler (inference-time baseline)",
        "use_sim_guide": False,
        "use_adv_noise": False,
        "adaptor": False,
        "sampler": "ddim",
    },
    "pgd": {
        "id": "pgd",
        "description": "PGD adversarial noise (inner loop component, standalone)",
        "use_sim_guide": False,
        "use_adv_noise": True,
        "adaptor": False,
    },
    # ── GAN / transfer baselines ───────────────────────────────────────────
    "tgan": {
        "id": "tgan",
        "description": "TransferGAN baseline",
        "use_sim_guide": False,
        "use_adv_noise": False,
        "adaptor": False,
        "framework": "gan",
    },
    "ada": {
        "id": "ada",
        "description": "ADA (Adaptive Data Augmentation) baseline",
        "use_sim_guide": False,
        "use_adv_noise": False,
        "adaptor": False,
        "framework": "gan",
    },
    "ewc": {
        "id": "ewc",
        "description": "EWC (Elastic Weight Consolidation) baseline",
        "use_sim_guide": False,
        "use_adv_noise": False,
        "adaptor": False,
        "framework": "diffusion",
    },
    "cdc": {
        "id": "cdc",
        "description": "CDC (Cross-Domain Correspondence) baseline",
        "use_sim_guide": False,
        "use_adv_noise": False,
        "adaptor": False,
        "framework": "gan",
    },
    "dcl": {
        "id": "dcl",
        "description": "DCL (Domain-Consistent Loss) baseline",
        "use_sim_guide": False,
        "use_adv_noise": False,
        "adaptor": False,
        "framework": "diffusion",
    },
    "ddpm_pa": {
        "id": "ddpm_pa",
        "description": "DDPM-PA (DDPM with Patch Attention) baseline",
        "use_sim_guide": False,
        "use_adv_noise": False,
        "adaptor": True,
        "framework": "ddpm",
    },
    # ── Metric / domain tags (used in result tables) ───────────────────────
    "ffhq": {
        "id": "ffhq",
        "description": "FFHQ source domain tag",
        "role": "source_domain",
    },
    "lpips": {
        "id": "lpips",
        "description": "LPIPS perceptual diversity metric",
        "role": "metric",
    },
    "gan": {
        "id": "gan",
        "description": "Generic GAN baseline family",
        "role": "baseline_family",
    },
}

# ---------------------------------------------------------------------------
# Bounded parameter sweep registry
# reference_grounding: paper_semantic_chunk_012 ablation_sensitivity
# These are config/registry values; execution is bounded to smoke/default subset.
# ---------------------------------------------------------------------------
SWEEP_REGISTRY: dict = {
    # ── Shot count sweep ───────────────────────────────────────────────────
    "shot_count": {
        "values": [10, 100],
        "default": 10,
        "paper_anchor": "10_shot_setting",
        "description": "Number of target-domain training images",
    },
    # ── Classifier training iterations ────────────────────────────────────
    "training_iteration_count": {
        "values": [0, 50, 100, 150, 200, 250, 300, 350],
        "default": 300,
        "paper_anchor": "300_training_iterations",
        "description": "Classifier fine-tuning steps (ablation)",
    },
    # ── Similarity guidance scale γ ───────────────────────────────────────
    "similarity_guidance_scale": {
        "values": [1, 2, 3, 5, 7, 9, 10],
        "default": 5,
        "paper_anchor": "gamma_5",
        "description": "Weight γ for KL similarity guidance loss",
    },
    # ── Adversarial noise scale ω (PGD step size) ─────────────────────────
    "adversarial_noise_scale": {
        "values": [0.01, 0.02, 0.03, 0.04, 0.05],
        "default": 0.02,
        "paper_anchor": "omega_0.02",
        "description": "PGD step size ω for adversarial noise selection",
    },
    # ── Perturbation budget α ─────────────────────────────────────────────
    "alpha": {
        "values": [0.01, 0.02, 0.05, 0.1],
        "default": 0.05,
        "description": "PGD perturbation budget α (clamp bound ±δ)",
    },
    # ── Epsilon (noise budget alias) ──────────────────────────────────────
    "epsilon": {
        "values": [0.01, 0.02, 0.03, 0.04, 0.05],
        "default": 0.02,
        "description": "Adversarial noise epsilon budget (alias for alpha in some ablations)",
    },
    # ── Total fine-tuning iterations ──────────────────────────────────────
    "iteration_count": {
        "values": [1000, 2000, 3000, 5000],
        "default": 5000,
        "paper_anchor": "5000_iterations",
        "description": "Total fine-tuning iterations",
    },
    # ── Batch size ────────────────────────────────────────────────────────
    "batch_size": {
        "values": [8, 16, 32, 64],
        "default": 64,
        "paper_anchor": "batch_size_64",
        "description": "Training batch size",
    },
    # ── Adaptor bottleneck dims (DDPM) ────────────────────────────────────
    "ddpm_adaptor_dims": {
        "values": [{"c": 4, "d": 8}],
        "default": {"c": 4, "d": 8},
        "description": "DDPM Shift Adaptor bottleneck (c=4, d=8)",
    },
    # ── Adaptor bottleneck dims (LDM) ─────────────────────────────────────
    "ldm_adaptor_dims": {
        "values": [{"c": 2, "d": 8}],
        "default": {"c": 2, "d": 8},
        "description": "LDM Shift Adaptor bottleneck (c=2, d=8)",
    },
}

# ---------------------------------------------------------------------------
# Lazy imports – expose trainer classes without requiring heavy deps at import
# ---------------------------------------------------------------------------

def _get_ant_trainer():
    """Lazy import of ANTTrainer to avoid heavy dep failures at module load."""
    from dpms_ant.trainer.ant_trainer import ANTTrainer  # noqa: PLC0415
    return ANTTrainer


def _get_similarity_guidance():
    """Lazy import of SimilarityGuidance."""
    from dpms_ant.trainer.similarity_guidance import SimilarityGuidance  # noqa: PLC0415
    return SimilarityGuidance


def _get_adversarial_noise_selector():
    """Lazy import of AdversarialNoiseSelector."""
    from dpms_ant.trainer.adversarial_noise import AdversarialNoiseSelector  # noqa: PLC0415
    return AdversarialNoiseSelector


# ---------------------------------------------------------------------------
# Public convenience accessors (safe for static import)
# ---------------------------------------------------------------------------

def get_trainer_class(method: str = "dpms_ant"):
    """
    Return the trainer class for the given method key.

    Parameters
    ----------
    method : str
        One of the keys in METHOD_REGISTRY.

    Returns
    -------
    type
        ANTTrainer configured for the selected method.
    """
    if method not in METHOD_REGISTRY:
        raise ValueError(
            f"Unknown method '{method}'. "
            f"Available: {sorted(METHOD_REGISTRY.keys())}"
        )
    return _get_ant_trainer()


def build_trainer(
    diffusion_model,
    adaptor,
    classifier,
    target_dataset,
    method: str = "dpms_ant",
    device: str = "cpu",
    **kwargs,
):
    """
    Instantiate an ANTTrainer with the paper-derived defaults.

    Parameters
    ----------
    diffusion_model : nn.Module
        Pre-trained DDPM or LDM backbone (non-adaptor params frozen).
    adaptor : ShiftAdaptor
        Shift Adaptor module (all params initialised to 0).
    classifier : DomainClassifier
        MobileNet-based domain classifier fine-tuned for 300 steps.
    target_dataset : Dataset
        Few-shot target domain dataset (default 10 images).
    method : str
        Method key from METHOD_REGISTRY.
    device : str
        Torch device string.
    **kwargs
        Override any FIXED_HYPERPARAMETERS value.

    Returns
    -------
    ANTTrainer
    """
    cfg = METHOD_REGISTRY.get(method, METHOD_REGISTRY["dpms_ant"])
    hp = {**FIXED_HYPERPARAMETERS, **kwargs}

    ANTTrainer = _get_ant_trainer()
    return ANTTrainer(
        diffusion_model=diffusion_model,
        adaptor=adaptor,
        classifier=classifier,
        target_dataset=target_dataset,
        use_sim_guide=cfg.get("use_sim_guide", True),
        use_adv_noise=cfg.get("use_adv_noise", True),
        total_iterations=hp["total_iterations"],
        batch_size=hp["batch_size"],
        gamma=hp["gamma"],
        omega=hp["omega"],
        adversarial_inner_steps=hp["adversarial_inner_steps"],
        device=device,
    )


def get_sweep_values(sweep_key: str):
    """
    Return the bounded sweep values for a given sweep key.

    Parameters
    ----------
    sweep_key : str
        One of the keys in SWEEP_REGISTRY.

    Returns
    -------
    list
        Bounded list of sweep values (not exhaustive execution).
    """
    if sweep_key not in SWEEP_REGISTRY:
        raise KeyError(
            f"Unknown sweep '{sweep_key}'. "
            f"Available: {sorted(SWEEP_REGISTRY.keys())}"
        )
    return SWEEP_REGISTRY[sweep_key]["values"]


def get_default_hyperparameters() -> dict:
    """Return a copy of the paper-anchored fixed hyperparameters."""
    return dict(FIXED_HYPERPARAMETERS)


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------
__all__ = [
    # Classes (lazy-loaded)
    "ANTTrainer",
    "SimilarityGuidance",
    "AdversarialNoiseSelector",
    # Registries
    "METHOD_REGISTRY",
    "SWEEP_REGISTRY",
    "FIXED_HYPERPARAMETERS",
    # Factory helpers
    "get_trainer_class",
    "build_trainer",
    "get_sweep_values",
    "get_default_hyperparameters",
]


# Provide attribute-style access for lazy classes without triggering imports
def __getattr__(name: str):
    if name == "ANTTrainer":
        return _get_ant_trainer()
    if name == "SimilarityGuidance":
        return _get_similarity_guidance()
    if name == "AdversarialNoiseSelector":
        return _get_adversarial_noise_selector()
    raise AttributeError(f"module 'dpms_ant.trainer' has no attribute '{name}'")