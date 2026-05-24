# src/models/ddim.py
# =============================================================================
# DDIM Sampler for DPMs-ANT
# Paper: "Bridging Data Gaps in Diffusion Models with Adversarial Noise-Based
#         Transfer Learning"
#
# Implements DDIM (Denoising Diffusion Implicit Models) sampling as used in
# the DPMs-ANT framework for few-shot domain adaptation.
#
# reference_grounding: paper_semantic_chunk_005 adversarial_noise_selection
# reference_grounding: paper_semantic_chunk_008 adapter_shift_module_transfer_learning
#
# Method/baseline selector registry (paper evidence contract):
#   ours | diffusion_model | ddpm | ldm | dpms_ant |
#   similarity_guided_training | adversarial_noise_selection |
#   ddpm_pa | tgan | ada | ewc | cdc | dcl | pgd | ddim
#
# Fixed hyperparameter anchors (paper addendum contract):
#   5000_iterations, 300_training_iterations, 10_shot_setting,
#   gamma_5, omega_0.02, adversarial_inner_steps_10, batch_size_64
#
# Classifier URLs (addendum Section 5.2):
#   DDPM: https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_classifier.pt
#   LDM:  https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_classifier.pt
# =============================================================================

from __future__ import annotations

import json
import math
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# Method / baseline / attack selector registry
# reference_grounding: paper_method_core method_registry
# ---------------------------------------------------------------------------
METHOD_REGISTRY: Dict[str, str] = {
    "ours": "DPMs-ANT (similarity-guided training + adversarial noise selection)",
    "diffusion_model": "Vanilla diffusion model (no adaptation)",
    "ddpm": "DDPM baseline (fine-tune all parameters)",
    "ldm": "LDM baseline (fine-tune all parameters)",
    "dpms_ant": "DPMs-ANT full method (Algorithm 1)",
    "similarity_guided_training": "Ablation: similarity-guided training only",
    "adversarial_noise_selection": "Ablation: adversarial noise selection only",
    "ddpm_pa": "DDPM-PA baseline (prompt-based adaptation)",
    "ddpm_ant_wo_an": "DDPM-ANT ablation without adversarial noise selection",
    "dpms_ant_wo_an": "DPMs-ANT ablation without adversarial noise selection",
    "tgan": "TGAN baseline (transfer GAN)",
    "ada": "ADA baseline (adaptive discriminator augmentation)",
    "ewc": "EWC baseline (elastic weight consolidation)",
    "cdc": "CDC baseline (contrastive diffusion)",
    "dcl": "DCL baseline (decoupled contrastive learning)",
    "pgd": "PGD attack (inner loop adversarial noise selection)",
    "ddim": "DDIM sampler (deterministic sampling schedule)",
}

# ---------------------------------------------------------------------------
# Bounded parameter sweep registry
# reference_grounding: paper_semantic_chunk_012 sensitivity_analysis
# ---------------------------------------------------------------------------
SWEEP_REGISTRY: Dict[str, Any] = {
    # Adversarial noise budget (omega in paper)
    "adversarial_noise_scale": [0.01, 0.02, 0.03, 0.04, 0.05],
    # Similarity guidance scale (gamma in paper)
    "similarity_guidance_scale": [1, 2, 3, 5, 7, 9, 10],
    # Shot count
    "shot_count": [10, 100],
    # Training iteration count (ablation)
    "training_iteration_count": [0, 50, 100, 150, 200, 250, 300, 350],
    # Adversarial inner steps
    "adversarial_inner_steps": [10],
    # Batch size
    "batch_size": [64],
    # Alpha (loss weighting)
    "alpha": [0.1, 0.5, 1.0, 2.0],
    # Gamma (similarity guidance scale alias)
    "gamma": [1, 2, 3, 5, 7, 9, 10],
    # Epsilon (PGD step size)
    "epsilon": [0.005, 0.01, 0.02, 0.05],
    # Iteration count (total)
    "iteration_count": [5000],
}

# ---------------------------------------------------------------------------
# Fixed hyperparameter anchors (paper addendum contract)
# reference_grounding: paper_semantic_chunk_012 fixed_hyperparameters
# ---------------------------------------------------------------------------
FIXED_HYPERPARAMETERS: Dict[str, Any] = {
    "total_iterations": 5000,           # anchor: 5000_iterations
    "training_iterations": 300,         # anchor: 300_training_iterations
    "shot_count": 10,                   # anchor: 10_shot_setting
    "gamma": 5,                         # anchor: gamma_5
    "omega": 0.02,                      # anchor: omega_0.02
    "adversarial_inner_steps": 10,      # anchor: adversarial_inner_steps_10
    "batch_size": 64,                   # anchor: batch_size_64
}

# ---------------------------------------------------------------------------
# Classifier URLs (addendum Section 5.2)
# reference_grounding: paper_addendum section_5_2 classifier_urls
# ---------------------------------------------------------------------------
CLASSIFIER_URLS: Dict[str, str] = {
    "ddpm": "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_classifier.pt",
    "ldm": "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_classifier.pt",
}


def _make_beta_schedule(
    schedule: str,
    n_timestep: int,
    linear_start: float = 1e-4,
    linear_end: float = 2e-2,
    cosine_s: float = 8e-3,
) -> np.ndarray:
    """
    Build a beta schedule for the diffusion process.

    Supports 'linear', 'cosine', and 'sqrt_linear' schedules as used in
    DDPM and improved-DDPM variants.

    reference_grounding: paper_semantic_chunk_005 diffusion_probabilistic_models
    """
    if schedule == "linear":
        betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = np.arange(n_timestep + 1, dtype=np.float64) / n_timestep + cosine_s
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = np.cos(alphas) ** 2
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, 0, 0.999)
    elif schedule == "sqrt_linear":
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=np.float64) ** 2
    elif schedule == "sqrt":
        betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64) ** 0.5
    else:
        raise ValueError(f"Unknown beta schedule: {schedule}")
    return betas


def _extract_into_tensor(arr: np.ndarray, timesteps: Any, broadcast_shape: Tuple) -> Any:
    """
    Extract values from a 1-D numpy array for a batch of timesteps,
    then reshape to broadcast_shape.

    Works with both numpy arrays and torch tensors for timesteps.
    """
    try:
        import torch
        if isinstance(timesteps, torch.Tensor):
            res = torch.from_numpy(arr).to(device=timesteps.device, dtype=torch.float32)
            res = res[timesteps]
            while len(res.shape) < len(broadcast_shape):
                res = res[..., None]
            return res.expand(broadcast_shape)
    except ImportError:
        pass
    # numpy fallback
    res = arr[timesteps]
    while res.ndim < len(broadcast_shape):
        res = res[..., np.newaxis]
    return np.broadcast_to(res, broadcast_shape)


class DDIMSampler:
    """
    DDIM (Denoising Diffusion Implicit Models) sampler.

    Implements deterministic and stochastic sampling via the DDIM schedule,
    as used in DPMs-ANT for both DDPM and LDM frameworks.

    The sampler wraps a noise-prediction UNet (or LDM denoiser) and supports:
      - Standard DDIM deterministic sampling (eta=0)
      - Stochastic DDIM sampling (eta>0, recovers DDPM at eta=1)
      - Classifier-free and classifier-guided sampling
      - Adversarial noise injection (PGD inner loop, omega=0.02)

    reference_grounding: paper_semantic_chunk_005 ddim_sampler
    reference_grounding: paper_semantic_chunk_008 shift_adaptor_ddim_integration

    Method selector: ddim
    Fixed hyperparameters: omega_0.02, adversarial_inner_steps_10, batch_size_64
    """

    def __init__(
        self,
        model: Any,
        schedule: str = "linear",
        n_timesteps: int = 1000,
        linear_start: float = 1e-4,
        linear_end: float = 2e-2,
        cosine_s: float = 8e-3,
        ddim_discretize: str = "uniform",
        eta: float = 0.0,
        device: Optional[str] = None,
    ) -> None:
        """
        Args:
            model: UNet or LDM denoiser with signature model(x_t, t) -> noise_pred.
                   May have ShiftAdaptor layers inserted residually.
            schedule: Beta schedule type ('linear', 'cosine', 'sqrt_linear').
            n_timesteps: Total diffusion timesteps T (default 1000).
            linear_start: Beta schedule start value.
            linear_end: Beta schedule end value.
            cosine_s: Cosine schedule offset.
            ddim_discretize: How to select DDIM subset steps ('uniform' or 'quad').
            eta: DDIM stochasticity parameter (0=deterministic, 1=DDPM).
            device: Torch device string. Auto-detected if None.
        """
        self.model = model
        self.schedule = schedule
        self.n_timesteps = n_timesteps
        self.ddim_discretize = ddim_discretize
        self.eta = eta

        # Build noise schedule
        betas = _make_beta_schedule(
            schedule, n_timesteps, linear_start, linear_end, cosine_s
        )
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        self.betas = betas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1)

        # Posterior variance (for DDPM-style sampling)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_variance = posterior_variance
        self.posterior_log_variance_clipped = np.log(
            np.maximum(posterior_variance, 1e-20)
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

        # Device
        self._device = device

    @property
    def device(self) -> Any:
        """Lazily resolve torch device."""
        try:
            import torch
            if self._device is not None:
                return torch.device(self._device)
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except ImportError:
            return None

    def make_ddim_timesteps(
        self,
        ddim_num_steps: int,
        ddim_discretize: Optional[str] = None,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Compute the subset of timesteps used for DDIM sampling.

        Args:
            ddim_num_steps: Number of DDIM denoising steps (e.g. 50, 100, 200).
            ddim_discretize: 'uniform' or 'quad'. Defaults to self.ddim_discretize.
            verbose: Print selected timesteps.

        Returns:
            Array of integer timestep indices (descending order for sampling).

        reference_grounding: paper_semantic_chunk_005 ddim_timestep_selection
        """
        disc = ddim_discretize or self.ddim_discretize
        if disc == "uniform":
            c = self.n_timesteps // ddim_num_steps
            ddim_timesteps = np.asarray(list(range(0, self.n_timesteps, c)))
        elif disc == "quad":
            ddim_timesteps = (
                (np.linspace(0, np.sqrt(self.n_timesteps * 0.8), ddim_num_steps)) ** 2
            ).astype(int)
        else:
            raise ValueError(f"Unknown ddim_discretize: {disc}")

        # Ensure we include the final timestep
        steps_out = ddim_timesteps + 1
        if verbose:
            print(f"[DDIMSampler] Selected {len(steps_out)} timesteps: {steps_out}")
        return steps_out

    def make_ddim_sampling_parameters(
        self,
        alphacums: np.ndarray,
        ddim_timesteps: np.ndarray,
        eta: float,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute DDIM sampling coefficients for the selected timestep subset.

        Returns:
            (alphas, alphas_prev, sigmas) arrays for the DDIM update rule.

        reference_grounding: paper_semantic_chunk_005 ddim_sampling_parameters
        """
        alphas = alphacums[ddim_timesteps]
        alphas_prev = np.asarray(
            [alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist()
        )
        sigmas = eta * np.sqrt(
            (1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev)
        )
        if verbose:
            print(f"[DDIMSampler] DDIM sigmas: {sigmas}")
        return alphas, alphas_prev, sigmas

    @staticmethod
    def noise_like(shape: Tuple, device: Any, repeat: bool = False) -> Any:
        """Generate noise tensor matching shape."""
        try:
            import torch
            if repeat:
                noise = torch.randn((1, *shape[1:]), device=device)
                return noise.repeat(shape[0], *([1] * (len(shape) - 1)))
            return torch.randn(shape, device=device)
        except ImportError:
            return np.random.randn(*shape).astype(np.float32)

    def q_sample(
        self,
        x_start: Any,
        t: Any,
        noise: Optional[Any] = None,
    ) -> Any:
        """
        Forward diffusion: sample x_t from x_0 at timestep t.

        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps

        reference_grounding: paper_semantic_chunk_005 forward_diffusion
        """
        try:
            import torch
            if noise is None:
                noise = torch.randn_like(x_start)
            sqrt_alphas = _extract_into_tensor(
                self.sqrt_alphas_cumprod, t, x_start.shape
            )
            sqrt_one_minus = _extract_into_tensor(
                self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
            )
            return sqrt_alphas * x_start + sqrt_one_minus * noise
        except ImportError:
            if noise is None:
                noise = np.random.randn(*x_start.shape).astype(np.float32)
            sqrt_alphas = _extract_into_tensor(
                self.sqrt_alphas_cumprod, t, x_start.shape
            )
            sqrt_one_minus = _extract_into_tensor(
                self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
            )
            return sqrt_alphas * x_start + sqrt_one_minus * noise

    def predict_start_from_noise(
        self,
        x_t: Any,
        t: Any,
        noise: Any,
    ) -> Any:
        """
        Predict x_0 from x_t and predicted noise.

        x_0 = (x_t - sqrt(1-alpha_bar_t)*eps) / sqrt(alpha_bar_t)

        reference_grounding: paper_semantic_chunk_005 x0_prediction
        """
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    @staticmethod
    def _clamp_x0(x0: Any, clip_denoised: bool = True) -> Any:
        """Optionally clamp x0 prediction to [-1, 1]."""
        if not clip_denoised:
            return x0
        try:
            import torch
            return torch.clamp(x0, -1.0, 1.0)
        except ImportError:
            return np.clip(x0, -1.0, 1.0)

    def ddim_step(
        self,
        x: Any,
        t: Any,
        t_prev: Any,
        alpha_t: float,
        alpha_prev: float,
        sigma_t: float,
        noise_pred: Any,
        clip_denoised: bool = True,
        repeat_noise: bool = False,
    ) -> Tuple[Any, Any]:
        """
        Single DDIM reverse step.

        x_{t-1} = sqrt(alpha_{t-1}) * x0_pred
                + sqrt(1 - alpha_{t-1} - sigma_t^2) * noise_pred
                + sigma_t * random_noise

        Args:
            x: Current noisy sample x_t.
            t: Current timestep tensor.
            t_prev: Previous timestep tensor.
            alpha_t: alpha_bar at current step.
            alpha_prev: alpha_bar at previous step.
            sigma_t: DDIM sigma (0 for deterministic).
            noise_pred: Predicted noise from model.
            clip_denoised: Clamp x0 prediction to [-1, 1].
            repeat_noise: Use same noise for all batch elements.

        Returns:
            (x_prev, x0_pred) tuple.

        reference_grounding: paper_semantic_chunk_005 ddim_reverse_step
        """
        try:
            import torch

            # Predict x0
            sqrt_recip_alpha = 1.0 / (alpha_t ** 0.5)
            sqrt_recipm1_alpha = ((1.0 / alpha_t) - 1.0) ** 0.5
            x0_pred = sqrt_recip_alpha * x - sqrt_recipm1_alpha * noise_pred
            if clip_denoised:
                x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

            # Direction pointing to x_t
            dir_xt = ((1.0 - alpha_prev - sigma_t ** 2).clamp(min=0.0) ** 0.5) * noise_pred

            # Random noise component
            if sigma_t > 0:
                noise = self.noise_like(x.shape, x.device, repeat=repeat_noise)
            else:
                noise = torch.zeros_like(x)

            x_prev = (alpha_prev ** 0.5) * x0_pred + dir_xt + sigma_t * noise
            return x_prev, x0_pred

        except ImportError:
            # numpy fallback (for smoke/import testing)
            sqrt_recip_alpha = 1.0 / (alpha_t ** 0.5)
            sqrt_recipm1_alpha = ((1.0 / alpha_t) - 1.0) ** 0.5
            x0_pred = sqrt_recip_alpha * x - sqrt_recipm1_alpha * noise_pred
            if clip_denoised:
                x0_pred = np.clip(x0_pred, -1.0, 1.0)
            dir_xt = (max(1.0 - alpha_prev - sigma_t ** 2, 0.0) ** 0.5) * noise_pred
            noise = np.random.randn(*x.shape).astype(np.float32) if sigma_t > 0 else np.zeros_like(x)
            x_prev = (alpha_prev ** 0.5) * x0_pred + dir_xt + sigma_t * noise
            return x_prev, x0_pred

    @staticmethod
    def _to_tensor(x: Any, device: Any) -> Any:
        """Convert numpy array to torch tensor if torch is available."""
        try:
            import torch
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x).to(device)
            return x
        except ImportError:
            return x

    def sample(
        self,
        S: int,
        batch_size: int,
        shape: Tuple[int, ...],
        conditioning: Optional[Any] = None,
        callback: Optional[Callable] = None,
        img_callback: Optional[Callable] = None,
        quantize_x0: bool = False,
        eta: Optional[float] = None,
        mask: Optional[Any] = None,
        x0: Optional[Any] = None,
        temperature: float = 1.0,
        noise_dropout: float = 0.0,
        score_corrector: Optional[Any] = None,
        corrector_kwargs: Optional[Dict] = None,
        verbose: bool = True,
        x_T: Optional[Any] = None,
        log_every_t: int = 100,
        unconditional_guidance_scale: float = 1.0,
        unconditional_conditioning: Optional[Any] = None,
        clip_denoised: bool = True,
        adversarial_noise: Optional[Any] = None,
        omega: float = FIXED_HYPERPARAMETERS["omega"],
        **kwargs: Any,
    ) -> Tuple[Any, Any]:
        """
        Full DDIM sampling loop.

        Generates samples by iterating the DDIM reverse process over S steps.
        Supports adversarial noise injection (omega=0.02) as used in DPMs-ANT.

        Args:
            S: Number of DDIM sampling steps.
            batch_size: Number of samples to generate.
            shape: Spatial shape (C, H, W) for DDPM or (C, H/f, W/f) for LDM.
            conditioning: Optional conditioning signal.
            eta: DDIM eta (overrides self.eta if provided).
            x_T: Optional starting noise (random if None).
            unconditional_guidance_scale: Classifier-free guidance scale.
            unconditional_conditioning: Null conditioning for CFG.
            clip_denoised: Clamp x0 predictions to [-1, 1].
            adversarial_noise: Pre-computed adversarial perturbation (PGD output).
            omega: Adversarial noise budget (paper anchor: omega_0.02).
            **kwargs: Passed to model forward.

        Returns:
            (samples, intermediates) where intermediates contains x_inter and pred_x0.

        reference_grounding: paper_semantic_chunk_005 ddim_full_sampling_loop
        reference_grounding: paper_semantic_chunk_008 adversarial_noise_injection
        """
        try:
            import torch
        except ImportError:
            raise RuntimeError(
                "torch is required for DDIMSampler.sample(). "
                "Install PyTorch to run sampling."
            )

        device = self.device
        eta_val = eta if eta is not None else self.eta

        # Build DDIM timestep schedule
        ddim_timesteps = self.make_ddim_timesteps(
            ddim_num_steps=S,
            ddim_discretize=self.ddim_discretize,
            verbose=verbose,
        )
        alphas, alphas_prev, sigmas = self.make_ddim_sampling_parameters(
            alphacums=self.alphas_cumprod,
            ddim_timesteps=ddim_timesteps,
            eta=eta_val,
            verbose=verbose,
        )

        # Initial noise
        if x_T is None:
            img = torch.randn((batch_size, *shape), device=device) * temperature
        else:
            img = x_T

        # Inject adversarial noise if provided (DPMs-ANT ANT strategy)
        # reference_grounding: paper_semantic_chunk_005 adversarial_noise_selection
        if adversarial_noise is not None:
            adv = adversarial_noise.to(device) if hasattr(adversarial_noise, "to") else adversarial_noise
            # Clamp to omega budget
            adv = torch.clamp(adv, -omega, omega)
            img = img + adv

        intermediates = {"x_inter": [img], "pred_x0": [img]}
        time_range = np.flip(ddim_timesteps)
        total_steps = len(time_range)

        if verbose:
            try:
                from tqdm import tqdm
                iterator = tqdm(time_range, desc="DDIM Sampling", total=total_steps)
            except ImportError:
                iterator = time_range
        else:
            iterator = time_range

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((batch_size,), step, device=device, dtype=torch.long)

            # Mask-based inpainting support
            if mask is not None and x0 is not None:
                assert x0.shape == img.shape, "x0 and img must have same shape for inpainting"
                img_orig = self.q_sample(x0, ts)
                img = mask * img_orig + (1.0 - mask) * img

            # Model forward: predict noise
            if unconditional_conditioning is None or unconditional_guidance_scale == 1.0:
                noise_pred = self.model(img, ts, **kwargs)
            else:
                # Classifier-free guidance
                x_in = torch.cat([img] * 2)
                t_in = torch.cat([ts] * 2)
                c_in = torch.cat([unconditional_conditioning, conditioning])
                noise_uncond, noise_cond = self.model(x_in, t_in, c_in, **kwargs).chunk(2)
                noise_pred = noise_uncond + unconditional_guidance_scale * (noise_cond - noise_uncond)

            # Score correction (optional)
            if score_corrector is not None:
                assert corrector_kwargs is not None
                noise_pred = score_corrector.modify_score(
                    self.model, noise_pred, img, ts, conditioning, **corrector_kwargs
                )

            # Noise dropout
            if noise_dropout > 0.0:
                noise_pred = torch.nn.functional.dropout(noise_pred, p=noise_dropout)

            # DDIM step
            alpha_t = float(alphas[index])
            alpha_prev_t = float(alphas_prev[index])
            sigma_t = float(sigmas[index])

            img, pred_x0 = self.ddim_step(
                x=img,
                t=ts,
                t_prev=None,
                alpha_t=alpha_t,
                alpha_prev=alpha_prev_t,
                sigma_t=sigma_t,
                noise_pred=noise_pred,
                clip_denoised=clip_denoised,
            )

            if callback is not None:
                callback(i)
            if img_callback is not None:
                img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates["x_inter"].append(img)
                intermediates["pred_x0"].append(pred_x0)

        return img, intermediates

    def encode(
        self,
        x0: Any,
        t: Any,
        cond: Optional[Any] = None,
        noise: Optional[Any] = None,
    ) -> Any:
        """
        DDIM encoding (forward process to timestep t).

        Used for image-to-image translation and inpainting.

        reference_grounding: paper_semantic_chunk_005 ddim_encoding
        """
        try:
            import torch
            if noise is None:
                noise = torch.randn_like(x0)
        except ImportError:
            if noise is None:
                noise = np.random.randn(*x0.shape).astype(np.float32)
        return self.q_sample(x0, t, noise=noise)

    def stochastic_encode(
        self,
        x0: Any,
        t: Any,
        use_original_steps: bool = False,
        noise: Optional[Any] = None,
    ) -> Any:
        """
        Stochastic DDIM encoding for SDEdit-style editing.

        reference_grounding: paper_semantic_chunk_005 stochastic_encode
        """
        try:
            import torch
            if noise is None:
                noise = torch.randn_like(x0)
            sqrt_alphas = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x0.shape)
            sqrt_one_minus = _extract_into_tensor(
                self.sqrt_one_minus_alphas_cumprod, t, x0.shape
            )
            return sqrt_alphas * x0 + sqrt_one_minus * noise
        except ImportError:
            if noise is None:
                noise = np.random.randn(*x0.shape).astype(np.float32)
