"""
dpms_ant/utils/ema.py
=====================
Exponential Moving Average (EMA) utility for DPMs-ANT training.

Maintains a shadow copy of model parameters updated as:
    θ_ema ← decay · θ_ema + (1 - decay) · θ

Used during DPMs-ANT training (Algorithm 1) to maintain a stable EMA of the
UNet + ShiftAdaptor parameters, improving generation quality at inference time.

reference_grounding: paper_method_core dpms_ant/utils/ema.py
reference_grounding: paper_semantic_chunk_012 fixed hyperparameters / training loop
"""

from __future__ import annotations

import copy
import logging
from contextlib import contextmanager
from typing import Dict, Iterable, Iterator, List, Optional, Union

logger = logging.getLogger(__name__)


class EMAModel:
    """
    Exponential Moving Average over model parameters.

    Typical usage in DPMs-ANT training (Algorithm 1):

        ema = EMAModel(model.parameters(), decay=0.9999)

        for step in training_loop:
            optimizer.step()
            ema.update(model.parameters())

        # At inference / checkpoint save:
        with ema.average_parameters(model.parameters()):
            generate_samples(model)

    Parameters
    ----------
    parameters : iterable of nn.Parameter
        Initial parameters to track.  Typically ``model.parameters()``.
    decay : float
        EMA decay coefficient (between 0 and 1). Higher values give more
        weight to the historical average.  Paper default: 0.9999.
    update_after_step : int
        Number of optimiser steps before EMA updates begin.  Allows the
        model to warm up before shadow weights are accumulated.
    update_every : int
        EMA is recomputed every this many optimiser steps.
    use_ema_warmup : bool
        If True, use the "min(decay, (1+n)/(10+n))" warm-up schedule
        (matches the improved-diffusion implementation).
    inv_gamma : float
        Inverse gamma used by warm-up schedule (default 1.0).
    power : float
        Power used by warm-up schedule (default 2/3).
    """

    def __init__(
        self,
        parameters: Optional[Iterable] = None,
        decay: float = 0.9999,
        update_after_step: int = 0,
        update_every: int = 1,
        use_ema_warmup: bool = False,
        inv_gamma: float = 1.0,
        power: float = 2.0 / 3.0,
    ) -> None:
        self.decay = decay
        self.update_after_step = update_after_step
        self.update_every = update_every
        self.use_ema_warmup = use_ema_warmup
        self.inv_gamma = inv_gamma
        self.power = power

        self._step_count: int = 0
        self._shadow_params: List = []
        self._collected_params: List = []

        if parameters is not None:
            self.register(parameters)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, parameters: Iterable) -> None:
        """Register (or re-register) a set of parameters."""
        params = list(parameters)
        self._shadow_params = [p.clone().detach() for p in params]
        for s in self._shadow_params:
            s.requires_grad_(False)
        self._collected_params = []

    # ------------------------------------------------------------------
    # Decay schedule
    # ------------------------------------------------------------------

    def get_current_decay(self) -> float:
        """Return the effective decay for the current step."""
        step = max(0, self._step_count - self.update_after_step - 1)
        if self.use_ema_warmup:
            value = 1.0 - (1.0 + step / self.inv_gamma) ** (-self.power)
        else:
            value = self.decay
        return min(self.decay, value)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, parameters: Iterable) -> None:
        """
        Step the EMA forward by one optimiser update.

        Parameters
        ----------
        parameters : iterable of nn.Parameter
            Current (live) model parameters in the same order as those
            passed to ``register`` / the constructor.
        """
        self._step_count += 1

        # Only update on the configured cadence.
        if self._step_count % self.update_every != 0:
            return

        # Warm-up: don't accumulate until after update_after_step.
        if self._step_count <= self.update_after_step:
            self._copy_to_shadow(parameters)
            return

        decay = self.get_current_decay()
        params = list(parameters)

        if len(params) != len(self._shadow_params):
            raise ValueError(
                f"EMAModel.update: parameter count mismatch – "
                f"expected {len(self._shadow_params)}, got {len(params)}"
            )

        for shadow, param in zip(self._shadow_params, params):
            if not param.requires_grad:
                # Non-trainable parameters (e.g. batch-norm statistics):
                # copy directly.
                shadow.copy_(param.detach())
            else:
                shadow.sub_((1.0 - decay) * (shadow - param.detach()))

    def _copy_to_shadow(self, parameters: Iterable) -> None:
        """Direct copy of live parameters into shadow (used during warm-up)."""
        for shadow, param in zip(self._shadow_params, parameters):
            shadow.copy_(param.detach())

    # ------------------------------------------------------------------
    # Apply / restore helpers
    # ------------------------------------------------------------------

    def copy_to(self, parameters: Iterable) -> None:
        """Overwrite *parameters* with the current EMA shadow values."""
        params = list(parameters)
        if len(params) != len(self._shadow_params):
            raise ValueError(
                f"EMAModel.copy_to: parameter count mismatch – "
                f"expected {len(self._shadow_params)}, got {len(params)}"
            )
        for shadow, param in zip(self._shadow_params, params):
            param.data.copy_(shadow.data)

    def store(self, parameters: Iterable) -> None:
        """
        Save the current live parameters so they can be restored later.

        Typical pattern:
            ema.store(model.parameters())
            ema.copy_to(model.parameters())
            # … run inference …
            ema.restore(model.parameters())
        """
        self._collected_params = [p.clone() for p in parameters]

    def restore(self, parameters: Iterable) -> None:
        """Restore parameters previously saved with ``store``."""
        if not self._collected_params:
            raise RuntimeError(
                "EMAModel.restore called before EMAModel.store."
            )
        for collected, param in zip(self._collected_params, parameters):
            param.data.copy_(collected.data)
        self._collected_params = []

    @contextmanager
    def average_parameters(self, parameters: Iterable) -> Iterator[None]:
        """
        Context manager that temporarily replaces *parameters* with their
        EMA shadow values and restores the original values on exit.

        Example::

            with ema.average_parameters(model.parameters()):
                samples = model.sample(noise)
        """
        params = list(parameters)
        self.store(params)
        self.copy_to(params)
        try:
            yield
        finally:
            self.restore(params)

    # ------------------------------------------------------------------
    # State dict serialisation
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict:
        """Return a serialisable state dictionary."""
        return {
            "decay": self.decay,
            "update_after_step": self.update_after_step,
            "update_every": self.update_every,
            "use_ema_warmup": self.use_ema_warmup,
            "inv_gamma": self.inv_gamma,
            "power": self.power,
            "step_count": self._step_count,
            "shadow_params": [s.clone() for s in self._shadow_params],
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        """Restore EMA state from a dictionary (e.g. loaded from checkpoint)."""
        self.decay = state_dict["decay"]
        self.update_after_step = state_dict.get("update_after_step", 0)
        self.update_every = state_dict.get("update_every", 1)
        self.use_ema_warmup = state_dict.get("use_ema_warmup", False)
        self.inv_gamma = state_dict.get("inv_gamma", 1.0)
        self.power = state_dict.get("power", 2.0 / 3.0)
        self._step_count = state_dict.get("step_count", 0)
        self._shadow_params = [s.clone() for s in state_dict["shadow_params"]]

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def num_parameters(self) -> int:
        """Total number of tracked scalar values."""
        return sum(s.numel() for s in self._shadow_params)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"EMAModel("
            f"decay={self.decay}, "
            f"step={self._step_count}, "
            f"params={self.num_parameters:,}"
            f")"
        )


# ---------------------------------------------------------------------------
# Functional helpers
# ---------------------------------------------------------------------------


def build_ema(
    model,  # nn.Module – not imported at module level to keep import smoke safe
    decay: float = 0.9999,
    use_ema_warmup: bool = True,
    update_after_step: int = 100,
) -> EMAModel:
    """
    Convenience factory: construct an EMAModel from an ``nn.Module``.

    Parameters
    ----------
    model : torch.nn.Module
        The model whose parameters should be tracked.
    decay : float
        EMA decay coefficient.
    use_ema_warmup : bool
        Whether to apply the warm-up decay schedule.
    update_after_step : int
        How many steps to wait before starting EMA accumulation.

    Returns
    -------
    EMAModel
    """
    return EMAModel(
        parameters=model.parameters(),
        decay=decay,
        use_ema_warmup=use_ema_warmup,
        update_after_step=update_after_step,
    )


def update_ema_params(
    ema_params: List,
    model_params: List,
    decay: float = 0.9999,
) -> None:
    """
    Low-level in-place EMA update without wrapping in EMAModel.

    Useful for lightweight integration in existing training loops where
    the caller already manages lists of parameter tensors.

    ema_param ← decay · ema_param + (1 - decay) · model_param
    """
    for ema_p, model_p in zip(ema_params, model_params):
        ema_p.detach().mul_(decay).add_(model_p.detach(), alpha=1.0 - decay)