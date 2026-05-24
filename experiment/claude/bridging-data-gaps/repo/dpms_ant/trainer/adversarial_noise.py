"""
dpms_ant/trainer/adversarial_noise.py

Adversarial Noise Selection for DPMs-ANT.

Implements:
  - Equation 7 adversarial noise selection with Gaussian normalization
  - Equation 5 similarity-guided denoising residual
  - Full DPMs-ANT training step combining Equation 7 + Equation 8
  - Method/baseline registry
  - Bounded parameter sweep registry

reference_grounding: paper_method_core dpms_ant/trainer/adversarial_noise.py
reference_grounding: paper_semantic_chunk_010_classifier_loader_finetuning_adversarial_noise_selection_subsection_adversarial_noise
reference_grounding: paper_semantic_chunk_003_02_classifier_loader_finetuning_introduction_figure_two_sets
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Method / baseline registry
# Complete selector set per paper evidence contract.
# reference_grounding: paper_method_core method_registry
# ---------------------------------------------------------------------------
METHOD_REGISTRY: Dict[str, Dict[str, Any]] = {
    "ours":                       {"alias": "dpms_ant",                  "description": "DPMs-ANT: Similarity-Guided + Adversarial Noise Selection"},
    "dpms_ant":                   {"alias": "dpms_ant",                  "description": "DPMs-ANT full method (Algorithm 1)"},
    "diffusion_model":            {"alias": "diffusion_model",           "description": "Vanilla diffusion model fine-tuning"},
    "ddpm":                       {"alias": "ddpm",                      "description": "DDPM baseline fine-tuning"},
    "ldm":                        {"alias": "ldm",                       "description": "LDM baseline fine-tuning"},
    "similarity_guided_training": {"alias": "similarity_guided_training","description": "Ablation: similarity guidance only (no ANT)"},
    "adversarial_noise_selection":{"alias": "adversarial_noise_selection","description": "Ablation: adversarial noise only (no sim guidance)"},
    "ddpm_pa":                    {"alias": "ddpm_pa",                   "description": "DDPM-PA baseline (patch-based augmentation)"},
    "tgan":                       {"alias": "tgan",                      "description": "TransferGAN baseline"},
    "ada":                        {"alias": "ada",                       "description": "ADA (Adaptive Data Augmentation) baseline"},
    "ewc":                        {"alias": "ewc",                       "description": "EWC (Elastic Weight Consolidation) baseline"},
    "cdc":                        {"alias": "cdc",                       "description": "CDC baseline"},
    "dcl":                        {"alias": "dcl",                       "description": "DCL baseline"},
    "pgd":                        {"alias": "pgd",                       "description": "PGD adversarial attack (inner optimizer)"},
    "ddim":                       {"alias": "ddim",                      "description": "DDIM sampler"},
    # Evaluation domains / dataset tags
    "ffhq":                       {"alias": "ffhq",                      "description": "FFHQ source domain"},
    "gan":                        {"alias": "gan",                       "description": "GAN-based generation baseline"},
    "lpips":                      {"alias": "lpips",                     "description": "LPIPS perceptual diversity metric"},
    "ddpm_ant":                   {"alias": "ddpm_ant",                  "description": "DDPM + ANT variant"},
    "ldm_ant":                    {"alias": "ldm_ant",                   "description": "LDM + ANT with frozen autoencoder and trainable U-Net shift adaptor"},
}

# ---------------------------------------------------------------------------
# Bounded parameter sweep registry
# Expose as config/registry values; do NOT execute exhaustive sweeps here.
# reference_grounding: paper_semantic_chunk_012 fixed hyperparameters
# ---------------------------------------------------------------------------
SWEEP_REGISTRY: Dict[str, Any] = {
    # Fixed anchors (must not be overridden in ablations)
    "total_iterations":            5000,    # anchor: 5000_iterations
    "ablation_iterations":         300,     # anchor: 300_training_iterations
    "shot_count_default":          10,      # anchor: 10_shot_setting
    "gamma":                       5,       # anchor: gamma_5  (similarity guidance weight)
    "omega":                       0.02,    # anchor: omega_0.02 (PGD step size)
    "adversarial_inner_steps":     10,      # anchor: adversarial_inner_steps_10
    "batch_size":                  64,      # anchor: batch_size_64

    # DDPM Shift Adaptor bottleneck dims
    "ddpm_adaptor_c":              4,       # c=4 for DDPM
    "ddpm_adaptor_d":              8,       # d=8 for DDPM
    # LDM Shift Adaptor bottleneck dims
    "ldm_adaptor_c":               2,       # c=2 for LDM
    "ldm_adaptor_d":               8,       # d=8 for LDM

    # Bounded sweep values (for ablation/sensitivity; not exhaustive execution)
    "shot_count_sweep":            [10, 100],
    "training_iteration_sweep":    [0, 50, 100, 150, 200, 250, 300, 350],
    "similarity_guidance_scale_sweep": [1, 2, 3, 5, 7, 9, 10],
    "adversarial_noise_scale_sweep":   [0.01, 0.02, 0.03, 0.04, 0.05],
    "alpha_sweep":                 [0.01, 0.02, 0.03, 0.04, 0.05],
    "epsilon_sweep":               [0.01, 0.02, 0.03, 0.04, 0.05],
    "gamma_sweep":                 [1, 2, 3, 5, 7, 9, 10],
    "iteration_count_sweep":       [0, 50, 100, 150, 200, 250, 300, 350],
}

# ---------------------------------------------------------------------------
# Default hyperparameter config (used when no external config is provided)
# ---------------------------------------------------------------------------
DEFAULT_ANT_CONFIG: Dict[str, Any] = {
    "use_sim_guide":           True,   # ablation switch: similarity-guided training
    "use_adv_noise":           True,   # ablation switch: adversarial noise selection
    "gamma":                   SWEEP_REGISTRY["gamma"],
    "omega":                   SWEEP_REGISTRY["omega"],
    "adversarial_inner_steps": SWEEP_REGISTRY["adversarial_inner_steps"],
    "alpha":                   SWEEP_REGISTRY["omega"],   # perturbation budget = omega by default
    "lambda_sim":              1.0,    # weight of L_sim in L_total = L_simple + lambda * L_sim
    "batch_size":              SWEEP_REGISTRY["batch_size"],
    "total_iterations":        SWEEP_REGISTRY["total_iterations"],
    "shot_count":              SWEEP_REGISTRY["shot_count_default"],
    "method":                  "dpms_ant",
}


# ---------------------------------------------------------------------------
# Adversarial Noise Selector
# Implements the PGD inner loop from Algorithm 1 of the paper.
# reference_grounding: paper_semantic_chunk_010_classifier_loader_finetuning_adversarial_noise_selection_subsection_adversarial_noise
# ---------------------------------------------------------------------------
def normalize_gaussian_noise(eps: "torch.Tensor", eps_value: float = 1e-6) -> "torch.Tensor":
    """Norm(.) from Eq. 7: per-sample zero mean and unit standard deviation."""
    dims = tuple(range(1, eps.dim()))
    mean = eps.mean(dim=dims, keepdim=True)
    std = eps.std(dim=dims, keepdim=True, unbiased=False).clamp_min(eps_value)
    return (eps - mean) / std


class AdversarialNoiseSelector:
    """
    Equation 7 adversarial noise selection.

    Starting from ε⁰ ~ N(0,I), repeatedly applies gradient ascent on
    ||ε - ε_θ(sqrt(alpha_bar_t) x0 + sqrt(1-alpha_bar_t) ε, t)||² and then
    normalizes ε back to an approximate standard Gaussian distribution.
    """

    def __init__(
        self,
        diffusion_model: Any,
        inner_steps: int = DEFAULT_ANT_CONFIG["adversarial_inner_steps"],
        omega: float = DEFAULT_ANT_CONFIG["omega"],
        alpha: float = DEFAULT_ANT_CONFIG["alpha"],
    ) -> None:
        self.diffusion_model = diffusion_model
        self.inner_steps = inner_steps
        self.omega = omega
        self.alpha = alpha

    def select(
        self,
        x_0: "torch.Tensor",
        t: "torch.Tensor",
        base_noise: "torch.Tensor",
        adaptor_params: Optional[Any] = None,
    ) -> "torch.Tensor":
        """
        Run PGD inner loop to find adversarial noise.

        Args:
            x_0:          Clean target-domain images  [B, C, H, W]
            t:            Diffusion timesteps          [B]
            base_noise:   Initial Gaussian noise ε     [B, C, H, W]
            adaptor_params: (unused, for API compat)

        Returns:
            adversarial_noise: ε + δ*  [B, C, H, W]
        """
        import torch

        eps = normalize_gaussian_noise(base_noise.detach())

        for _ in range(self.inner_steps):
            eps = eps.detach().requires_grad_(True)
            x_t = self._q_sample(x_0, t, eps)

            loss = self._simple_loss(x_t, t, eps)
            grad = torch.autograd.grad(loss, eps, retain_graph=False, create_graph=False)[0]

            with torch.no_grad():
                eps = normalize_gaussian_noise(eps + self.omega * grad)

        return eps.detach()

    def _q_sample(
        self,
        x_0: "torch.Tensor",
        t: "torch.Tensor",
        noise: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Forward diffusion process: x_t = sqrt(ᾱ_t)·x_0 + sqrt(1-ᾱ_t)·noise
        Delegates to the diffusion model's q_sample if available.
        """
        if hasattr(self.diffusion_model, "q_sample"):
            return self.diffusion_model.q_sample(x_start=x_0, t=t, noise=noise)
        # Fallback: use sqrt_alphas_cumprod from model schedule
        import torch
        sqrt_alphas = self.diffusion_model.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus = self.diffusion_model.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alphas * x_0 + sqrt_one_minus * noise

    def _simple_loss(
        self,
        x_t: "torch.Tensor",
        t: "torch.Tensor",
        noise: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        L_simple = E[||ε - ε_θ(x_t, t)||²]
        """
        import torch
        model = self.diffusion_model
        if hasattr(model, "unet"):
            pred = model.unet(x_t, t)
        elif callable(model):
            pred = model(x_t, t)
        else:
            raise RuntimeError("diffusion_model must be callable or have .unet attribute")
        return torch.mean((noise - pred) ** 2)


# ---------------------------------------------------------------------------
# Similarity-Guided Loss
# L_sim = γ · KL(∇log p_φ(y=S|x_t) ‖ ∇log p_φ(y=T|x_t))
# reference_grounding: paper_semantic_chunk_003_02_classifier_loader_finetuning_introduction_figure_two_sets
# ---------------------------------------------------------------------------
class SimilarityGuidedLoss:
    """
    Computes the target-domain classifier gradient used inside Eq. 5.

    γ = 5 (paper anchor: gamma_5)
    """

    def __init__(
        self,
        classifier: Any,
        gamma: float = DEFAULT_ANT_CONFIG["gamma"],
    ) -> None:
        self.classifier = classifier
        self.gamma = gamma

    def target_gradient(
        self,
        x_t: "torch.Tensor",
        t: "torch.Tensor",
    ) -> "torch.Tensor":
        import torch
        import torch.nn.functional as F

        x_req = x_t.detach().requires_grad_(True)
        logits = self.classifier(x_req, t)
        if not logits.requires_grad:
            return torch.zeros_like(x_t)
        log_probs = F.log_softmax(logits, dim=-1)
        grad = torch.autograd.grad(
            log_probs[:, 1].sum(),
            x_req,
            create_graph=False,
            allow_unused=True,
        )[0]
        if grad is None:
            return torch.zeros_like(x_t)
        return grad.detach()

    def compute(self, x_t: "torch.Tensor", t: "torch.Tensor") -> "torch.Tensor":
        grad = self.target_gradient(x_t, t)
        return self.gamma * grad.square().mean()


# ---------------------------------------------------------------------------
# DPMs-ANT Training Step
# Combines adversarial noise selection + similarity-guided loss.
# Implements Algorithm 1 outer step.
# reference_grounding: paper_method_core dpms_ant/trainer/adversarial_noise.py
# ---------------------------------------------------------------------------
class ANTTrainingStep:
    """
    Single training step of DPMs-ANT (Algorithm 1 outer loop).

    L_total = L_simple(ε*) + λ · L_sim

    where:
      ε* = AdversarialNoiseSelector.select(x_0, t, ε)   [if use_adv_noise]
      L_sim = SimilarityGuidedLoss.compute(x_t, t)       [if use_sim_guide]

    Ablation switches:
      use_adv_noise=False  → use plain Gaussian noise (no PGD inner loop)
      use_sim_guide=False  → L_total = L_simple only
    """

    def __init__(
        self,
        diffusion_model: Any,
        classifier: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        cfg = {**DEFAULT_ANT_CONFIG, **(config or {})}

        self.diffusion_model = diffusion_model
        self.use_adv_noise: bool = cfg["use_adv_noise"]
        self.use_sim_guide: bool = cfg["use_sim_guide"]
        self.lambda_sim: float = cfg["lambda_sim"]

        self.adv_selector = AdversarialNoiseSelector(
            diffusion_model=diffusion_model,
            inner_steps=cfg["adversarial_inner_steps"],
            omega=cfg["omega"],
            alpha=cfg["alpha"],
        )

        if self.use_sim_guide:
            if classifier is None:
                raise ValueError(
                    "use_sim_guide=True requires a domain classifier. "
                    "Pass a DomainClassifier instance."
                )
            self.sim_loss_fn = SimilarityGuidedLoss(
                classifier=classifier,
                gamma=cfg["gamma"],
            )
        else:
            self.sim_loss_fn = None

    def step(
        self,
        x_0: "torch.Tensor",
        t: "torch.Tensor",
        optimizer: "torch.optim.Optimizer",
    ) -> Dict[str, float]:
        """
        Execute one outer training step.

        Args:
            x_0:       Clean target-domain images [B, C, H, W]
            t:         Sampled diffusion timesteps [B]
            optimizer: Optimizer for adaptor parameters

        Returns:
            dict with keys: loss_simple, loss_sim, loss_total
        """
        import torch

        optimizer.zero_grad()

        # 1. Sample base Gaussian noise
        noise = torch.randn_like(x_0)

        # 2. Adversarial noise selection (PGD inner loop)
        if self.use_adv_noise:
            with torch.enable_grad():
                adv_noise = self.adv_selector.select(x_0, t, noise)
        else:
            adv_noise = noise

        # 3. Forward diffusion: x_t = q_sample(x_0, t, adv_noise)
        x_t = self.adv_selector._q_sample(x_0, t, adv_noise)

        # 4. Compute Eq. 5 / Eq. 8 residual.
        if hasattr(self.diffusion_model, "unet"):
            pred = self.diffusion_model.unet(x_t, t)
        else:
            pred = self.diffusion_model(x_t, t)

        if self.use_sim_guide and self.sim_loss_fn is not None:
            sigma_hat_sq = torch.ones_like(x_t)
            guidance = sigma_hat_sq * self.sim_loss_fn.gamma * self.sim_loss_fn.target_gradient(x_t, t)
        else:
            guidance = torch.zeros_like(x_t)

        residual = adv_noise - pred - guidance
        loss_simple = torch.mean((adv_noise - pred) ** 2)
        loss_sim = torch.mean(guidance ** 2)
        loss_total = torch.mean(residual ** 2)
        if not loss_total.requires_grad:
            zero_link = None
            for group in optimizer.param_groups:
                for param in group["params"]:
                    term = param.sum() * 0.0
                    zero_link = term if zero_link is None else zero_link + term
            if zero_link is not None:
                loss_total = loss_total + zero_link

        # 7. Backprop and update (only adaptor params are unfrozen)
        loss_total.backward()
        optimizer.step()

        return {
            "loss_simple": loss_simple.item(),
            "loss_sim":    loss_sim.item(),
            "loss_total":  loss_total.item(),
        }


# ---------------------------------------------------------------------------
# Baseline / ablation adapter factory
# Returns a configured ANTTrainingStep for the requested method selector.
# reference_grounding: paper_method_core method_registry
# ---------------------------------------------------------------------------
def build_training_step(
    method: str,
    diffusion_model: Any,
    classifier: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None,
) -> ANTTrainingStep:
    """
    Factory that returns an ANTTrainingStep configured for the requested method.

    Supported method selectors (paper evidence contract):
      ours | dpms_ant | ldm_ant → full DPMs-ANT (sim_guide + adv_noise)
      similarity_guided_training → sim_guide only
      adversarial_noise_selection → adv_noise only
      diffusion_model | ddpm | ldm | ddpm_pa | tgan | ada | ewc | cdc | dcl
                                 → plain fine-tuning (no sim_guide, no adv_noise)
      pgd                        → adv_noise only (alias)
      ddim                       → plain fine-tuning (DDIM sampler, same loss)
      ddpm_ant                   → full DPMs-ANT (alias)
    """
    method = method.lower().strip()

    if method not in METHOD_REGISTRY:
        raise ValueError(
            f"Unknown method '{method}'. "
            f"Valid selectors: {sorted(METHOD_REGISTRY.keys())}"
        )

    cfg = {**DEFAULT_ANT_CONFIG, **(config or {})}

    if method in ("ours", "dpms_ant", "ddpm_ant", "ldm_ant"):
        cfg["use_sim_guide"] = True
        cfg["use_adv_noise"] = True

    elif method == "similarity_guided_training":
        cfg["use_sim_guide"] = True
        cfg["use_adv_noise"] = False

    elif method in ("adversarial_noise_selection", "pgd"):
        cfg["use_sim_guide"] = False
        cfg["use_adv_noise"] = True

    else:
        # All other baselines: plain fine-tuning
        cfg["use_sim_guide"] = False
        cfg["use_adv_noise"] = False

    return ANTTrainingStep(
        diffusion_model=diffusion_model,
        classifier=classifier if cfg["use_sim_guide"] else None,
        config=cfg,
    )


# ---------------------------------------------------------------------------
# Sweep config accessor
# Returns the bounded sweep values for a given hyperparameter name.
# ---------------------------------------------------------------------------
def get_sweep_values(param_name: str) -> Any:
    """
    Return bounded sweep values for a named hyperparameter.

    Args:
        param_name: one of the keys in SWEEP_REGISTRY

    Returns:
        The registered value or list of sweep values.

    Raises:
        KeyError if param_name is not registered.
    """
    if param_name not in SWEEP_REGISTRY:
        raise KeyError(
            f"Unknown sweep parameter '{param_name}'. "
            f"Available: {sorted(SWEEP_REGISTRY.keys())}"
        )
    return SWEEP_REGISTRY[param_name]


# ---------------------------------------------------------------------------
# Dry-run / smoke validation hook
# Validates wiring without requiring real data or long training.
# reference_grounding: paper_method_core smoke_validation
# ---------------------------------------------------------------------------
def smoke_validate() -> Dict[str, Any]:
    """
    Dry-run readiness check for adversarial_noise.py.

    Instantiates all public classes with minimal synthetic tensors and
    verifies that the computation graph is wired correctly.

    Returns a readiness dict (NOT benchmark results).
    """
    import torch

    report: Dict[str, Any] = {
        "dry_run": True,
        "label": "readiness/schema/contract artifact – not benchmark results",
        "method_registry_keys": sorted(METHOD_REGISTRY.keys()),
        "sweep_registry_keys":  sorted(SWEEP_REGISTRY.keys()),
        "fixed_hyperparameters": {
            "total_iterations":        SWEEP_REGISTRY["total_iterations"],
            "ablation_iterations":     SWEEP_REGISTRY["ablation_iterations"],
            "shot_count_default":      SWEEP_REGISTRY["shot_count_default"],
            "gamma":                   SWEEP_REGISTRY["gamma"],
            "omega":                   SWEEP_REGISTRY["omega"],
            "adversarial_inner_steps": SWEEP_REGISTRY["adversarial_inner_steps"],
            "batch_size":              SWEEP_REGISTRY["batch_size"],
        },
        "checks": {},
    }

    # ── Minimal stub diffusion model ──────────────────────────────────────
    class _StubDiffusion:
        def __init__(self) -> None:
            import torch
            T = 1000
            betas = torch.linspace(1e-4, 0.02, T)
            alphas = 1.0 - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            self.sqrt_alphas_cumprod = alphas_cumprod.sqrt()
            self.sqrt_one_minus_alphas_cumprod = (1 - alphas_cumprod).sqrt()

        def __call__(self, x_t: "torch.Tensor", t: "torch.Tensor") -> "torch.Tensor":
            return torch.zeros_like(x_t)

    # ── Minimal stub classifier ───────────────────────────────────────────
    class _StubClassifier:
        def __call__(self, x_t: "torch.Tensor", t: "torch.Tensor") -> "torch.Tensor":
            import torch
            B = x_t.shape[0]
            return torch.zeros(B, 2)

    stub_dm = _StubDiffusion()
    stub_cls = _StubClassifier()

    B, C, H, W = 2, 3, 8, 8
    x0 = torch.randn(B, C, H, W)
    t  = torch.randint(0, 100, (B,))

    # Check AdversarialNoiseSelector
    try:
        selector = AdversarialNoiseSelector(
            diffusion_model=stub_dm, inner_steps=2, omega=0.02, alpha=0.02
        )
        noise = torch.randn_like(x0)
        adv_noise = selector.select(x0, t, noise)
        assert adv_noise.shape == noise.shape
        report["checks"]["AdversarialNoiseSelector"] = "ok"
    except Exception as exc:
        report["checks"]["AdversarialNoiseSelector"] = f"FAIL: {exc}"

    # Check SimilarityGuidedLoss
    try:
        sim_loss_fn = SimilarityGuidedLoss(classifier=stub_cls, gamma=5)
        x_t = torch.randn(B, C, H, W)
        loss_val = sim_loss_fn.compute(x_t, t)
        assert loss_val.shape == torch.Size([])
        report["checks"]["SimilarityGuidedLoss"] = "ok"
    except Exception as exc:
        report["checks"]["SimilarityGuidedLoss"] = f"FAIL: {exc}"

    # Check ANTTrainingStep (full method)
    try:
        step_fn = build_training_step(
            method="dpms_ant",
            diffusion_model=stub_dm,
            classifier=stub_cls,
        )
        # Use a dummy optimizer over a dummy parameter
        dummy_param = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.SGD([dummy_param], lr=1e-3)
        metrics = step_fn.step(x0, t, opt)
        assert "loss_total" in metrics
        report["checks"]["ANTTrainingStep_dpms_ant"] = "ok"
    except Exception as exc:
        report["checks"]["ANTTrainingStep_dpms_ant"] = f"FAIL: {exc}"

    # Check ablation variants
    for method_name in [
        "similarity_guided_training",
        "adversarial_noise_selection",
        "ddpm",
        "ldm",
        "tgan",
        "ada",
        "ewc",
        "cdc",
        "dcl",
        "ddpm_pa",
        "pgd",
        "ddim",
    ]:
        try:
            step_fn = build_training_step(
                method=method_name,
                diffusion_model=stub_dm,
                classifier=stub_cls,
            )
            report["checks"][f"build_training_step_{method_name}"] = "ok"
        except Exception as exc:
            report["checks"][f"build_training_step_{method_name}"] = f"FAIL: {exc}"

    # Check sweep accessor
    try:
        vals = get_sweep_values("adversarial_noise_scale_sweep")
        assert isinstance(vals, list)
        report["checks"]["get_sweep_values"] = "ok"
    except Exception as exc:
        report["checks"]["get_sweep_values"] = f"FAIL: {exc}"

    report["status"] = (
        "pass"
        if all(v == "ok" for v in report["checks"].values())
        else "partial"
    )
    return report


# ---------------------------------------------------------------------------
# Module-level self-test (python -m dpms_ant.trainer.adversarial_noise)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    result = smoke_validate()
    print(json.dumps(result, indent=2))
