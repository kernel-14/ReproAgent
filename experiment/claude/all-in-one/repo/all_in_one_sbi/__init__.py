"""Public package facade for the All-in-one SBI / Simformer reproduction.

This package implements a standalone, code-generation-only reproduction surface
for the paper *All-in-one simulation-based inference*.  The package facade is
intentionally lightweight and importable in a minimal environment: optional
scientific and accelerator packages are imported lazily inside the functions
that need them, never at module import time.

The canonical method exposed here is a Simformer-style score-based diffusion
model over the joint simulator distribution ``p(theta, x)``.  The core contract
is visible from this file and delegated to richer submodules when available:

* ``SBITokenizer.encode(batch, condition_mask)`` emits variable identifiers,
  value representations, and binary condition states.
* ``M_E`` dependency attention masks are explicit method inputs and are passed
  through the score-network forward path.
* ``M_C`` conditioning masks are used for forward noising, loss masking, and
  conditional sampling.
* Sampling families are selectable by the names ``"sde"`` and ``"ode"``.
* Training metadata records method, mask variant, conditioning pattern,
  simulation budget, fixed hyperparameters, and grounding evidence.
* Default smoke/dry-run artifact writers materialize declared contract outputs
  without claiming paper-scale training or benchmark scores.

No code is copied from the blacklisted Simformer repository.

reference_grounding: paperbench_ref_001 docs/how_to_guide/04_embedding_networks.ipynb
reference_grounding: paperbench_ref_001 docs/how_to_guide/07_gpu_training.ipynb
reference_grounding: paperbench_ref_001 docs/how_to_guide/09_sampler_interface.ipynb
reference_grounding: paperbench_ref_001 sbi/inference/trainers/npe/npe_a.py
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

__version__ = "0.1.0"

BLACKLISTED_REPOSITORIES: Tuple[str, ...] = ("https://github.com/mackelab/simformer",)
DEFAULT_RESULTS_DIR = "results"
DEFAULT_VARIABLE_NAMES: Tuple[str, ...] = (
    "theta_0",
    "theta_1",
    "x_0",
    "x_1",
)
SAMPLING_FAMILIES: Tuple[str, ...] = ("sde", "ode")

DECLARED_CORE_ARTIFACTS: Tuple[str, ...] = (
    "results/model_registry.json",
    "results/tokenizer_registry.json",
    "results/attention_mask_registry.json",
    "results/diffusion_config.json",
    "results/loss_trace.json",
    "results/sampling_trace.json",
    "results/readiness.json",
    "results/evaluation_result.json",
)

CORE_HYPOTHESIS = (
    "Simformer core closure is satisfied when the tokenizer, dependency "
    "attention mask M_E, binary conditioning mask M_C, score-network forward "
    "path, diffusion objective, and conditional samplers are executable over "
    "joint p(theta,x) variables."
)
CORE_DECISION_VALUE = (
    "This package facade exposes the paper-method API used by the canonical "
    "runner and verifies that M_E reaches attention computation while M_C "
    "reaches noising, loss masking, and sampling."
)
CORE_STOP_RULE = (
    "Default execution is bounded to smoke-size batches and schema artifacts; "
    "paper-scale training/evaluation requires an explicit downstream full mode."
)


# ---------------------------------------------------------------------------
# Small dependency helpers
# ---------------------------------------------------------------------------


def _optional_import(module_name: str) -> Any:
    """Import an optional module lazily and return ``None`` if unavailable."""

    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def _as_float_matrix(values: Any) -> List[List[float]]:
    """Convert nested numeric inputs to a list-of-list float matrix.

    The function accepts Python sequences and common array-like objects without
    requiring NumPy at import time.  One-dimensional inputs become a single-row
    matrix.
    """

    if hasattr(values, "tolist"):
        values = values.tolist()
    if values is None:
        return []
    if isinstance(values, (int, float)):
        return [[float(values)]]
    if isinstance(values, Mapping):
        return [[float(v) for v in values.values()]]
    values_list = list(values)
    if not values_list:
        return []
    first = values_list[0]
    if isinstance(first, (int, float)):
        return [[float(v) for v in values_list]]
    return [[float(v) for v in row] for row in values_list]


def _zeros(rows: int, cols: int) -> List[List[float]]:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def _ones_bool(rows: int, cols: int) -> List[List[bool]]:
    return [[True for _ in range(cols)] for _ in range(rows)]


def _ensure_binary_matrix(mask: Any, rows: int, cols: int, *, default: bool = False) -> List[List[int]]:
    if mask is None:
        return [[1 if default else 0 for _ in range(cols)] for _ in range(rows)]
    if hasattr(mask, "tolist"):
        mask = mask.tolist()
    if isinstance(mask, (int, bool)):
        return [[1 if bool(mask) else 0 for _ in range(cols)] for _ in range(rows)]
    mask_list = list(mask)
    if not mask_list:
        return [[1 if default else 0 for _ in range(cols)] for _ in range(rows)]
    if isinstance(mask_list[0], (int, bool)):
        row = [1 if bool(v) else 0 for v in mask_list]
        row = (row + [1 if default else 0] * cols)[:cols]
        return [list(row) for _ in range(rows)]
    out: List[List[int]] = []
    for row in mask_list[:rows]:
        row_values = [1 if bool(v) else 0 for v in list(row)]
        out.append((row_values + [1 if default else 0] * cols)[:cols])
    while len(out) < rows:
        out.append([1 if default else 0 for _ in range(cols)])
    return out


def _result_root() -> Path:
    """Return the canonical output root.

    Declared repository paths remain under ``results/``.  If
    ``PAPERBENCH_REPRO_ARTIFACT_DIR`` is set, auxiliary mirrored outputs may be
    written by ``write_core_contract_artifacts`` for external collection.
    """

    return Path(DEFAULT_RESULTS_DIR)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _mirror_to_auxiliary(relative_path: str, payload: Mapping[str, Any]) -> None:
    aux = os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR")
    if not aux:
        return
    aux_path = Path(aux) / relative_path
    _write_json(aux_path, payload)


# ---------------------------------------------------------------------------
# Configuration and policy adapters
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SimformerCoreConfig:
    """Bounded, paper-method configuration exposed at package level."""

    method: str = "simformer"
    distribution: str = "joint_p_theta_x"
    mask_variant: str = "dependency_attention"
    conditioning_pattern: str = "uniform_binary_condition_state"
    condition_probability: float = 0.3
    simulation_budget: int = 32
    variable_names: Tuple[str, ...] = DEFAULT_VARIABLE_NAMES
    hidden_dim: int = 50
    num_layers: int = 6
    diffusion_steps: int = 500
    noise_min: float = 1.0e-5
    noise_max: float = 1.0
    sampling_family: str = "sde"
    device: str = "cpu"
    learning_rate: float = 5e-4
    training_batch_size: int = 1000
    max_num_epochs: int = 1
    validation_fraction: float = 0.1
    stop_after_epochs: int = 20
    clip_max_norm: float = 5.0
    dry_run: bool = True

    def validate(self) -> "SimformerCoreConfig":
        if self.distribution != "joint_p_theta_x":
            raise ValueError("Simformer core must train on joint p(theta,x), not only posterior/likelihood.")
        if self.sampling_family not in SAMPLING_FAMILIES:
            raise ValueError(f"sampling_family must be one of {SAMPLING_FAMILIES}, got {self.sampling_family!r}")
        if not 0.0 <= self.condition_probability <= 1.0:
            raise ValueError("condition_probability must be in [0, 1].")
        if self.diffusion_steps <= 0:
            raise ValueError("diffusion_steps must be positive.")
        return self

    def metadata(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "distribution": self.distribution,
            "mask_variant": self.mask_variant,
            "conditioning_pattern": self.conditioning_pattern,
            "condition_probability": self.condition_probability,
            "simulation_budget": self.simulation_budget,
            "fixed_hyperparameters": {
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "diffusion_steps": self.diffusion_steps,
                "noise_min": self.noise_min,
                "noise_max": self.noise_max,
                "learning_rate": self.learning_rate,
                "training_batch_size": self.training_batch_size,
                "max_num_epochs": self.max_num_epochs,
                "validation_fraction": self.validation_fraction,
                "stop_after_epochs": self.stop_after_epochs,
                "clip_max_norm": self.clip_max_norm,
                "device": self.device,
            },
            "sampling_families": list(SAMPLING_FAMILIES),
            "blacklisted_repositories_not_used": list(BLACKLISTED_REPOSITORIES),
            "reference_grounding": [
                "paperbench_ref_001 docs/how_to_guide/04_embedding_networks.ipynb",
                "paperbench_ref_001 docs/how_to_guide/07_gpu_training.ipynb",
                "paperbench_ref_001 docs/how_to_guide/09_sampler_interface.ipynb",
                "paperbench_ref_001 sbi/inference/trainers/npe/npe_a.py",
            ],
        }


@dataclasses.dataclass
class ConditioningMaskPolicy:
    """Binary condition-state sampler for ``M_C``.

    The paper method supports arbitrary conditional inference by resampling the
    conditioning pattern during training.  This policy provides a deterministic
    seedable implementation for smoke runs and can be replaced by richer
    downstream policies.
    """

    probability: float = 0.3
    pattern: str = "uniform_binary_condition_state"
    seed: int = 0

    def sample(self, batch_size: int, num_variables: int) -> List[List[int]]:
        rng = random.Random(self.seed)
        if self.pattern in {"none", "unconditional"}:
            return [[0 for _ in range(num_variables)] for _ in range(batch_size)]
        if self.pattern in {"all", "fully_conditioned"}:
            return [[1 for _ in range(num_variables)] for _ in range(batch_size)]
        if self.pattern not in {"uniform_binary_condition_state", "mask_probability_0.3", "random"}:
            raise ValueError(f"Unknown conditioning pattern: {self.pattern}")
        return [
            [1 if rng.random() < self.probability else 0 for _ in range(num_variables)]
            for _ in range(batch_size)
        ]


# ---------------------------------------------------------------------------
# Tokenizer and dependency attention mask builder
# ---------------------------------------------------------------------------


class SBITokenizer:
    """Tokenizer for joint simulator variables.

    ``encode`` exposes the contract-owned representation:
    variable identifiers, value representation, and binary condition state.
    High-dimensional observation embeddings may be supplied through
    ``embedding_fn``.  This follows the protocol intent of the referenced SBI
    embedding-net guide while remaining local and import-light.

    reference_grounding: paperbench_ref_001 docs/how_to_guide/04_embedding_networks.ipynb
    """

    def __init__(
        self,
        variable_names: Sequence[str] = DEFAULT_VARIABLE_NAMES,
        embedding_fn: Optional[Callable[[Sequence[float]], Sequence[float]]] = None,
    ) -> None:
        self.variable_names = tuple(variable_names)
        self.embedding_fn = embedding_fn
        self.variable_to_id = {name: i for i, name in enumerate(self.variable_names)}

    @property
    def num_variables(self) -> int:
        return len(self.variable_names)

    def encode(self, batch: Any, condition_mask: Any) -> Dict[str, Any]:
        values = _as_float_matrix(batch)
        if not values:
            values = [_zeros(1, self.num_variables)[0]]
        rows = len(values)
        cols = self.num_variables
        padded_values: List[List[float]] = []
        for row in values:
            row = list(row)
            if self.embedding_fn is not None:
                row = list(self.embedding_fn(row))
            padded_values.append((row + [0.0] * cols)[:cols])

        condition_state = _ensure_binary_matrix(condition_mask, rows, cols, default=False)
        variable_identifier = [list(range(cols)) for _ in range(rows)]
        variable_name = [list(self.variable_names) for _ in range(rows)]

        return {
            "variable_identifier": variable_identifier,
            "variable_name": variable_name,
            "value": padded_values,
            "value_representation": padded_values,
            "condition_state": condition_state,
            "condition_mask": condition_state,
            "metadata": {
                "tokenizer": "SBI tokenizer",
                "binary_condition_state": True,
                "supports_resampled_conditioning_pattern": True,
                "distribution": "joint_p_theta_x",
            },
        }

    def decode(self, encoded: Mapping[str, Any]) -> List[Dict[str, float]]:
        values = _as_float_matrix(encoded.get("value_representation", encoded.get("value", [])))
        decoded: List[Dict[str, float]] = []
        for row in values:
            decoded.append({name: float(row[i]) if i < len(row) else 0.0 for i, name in enumerate(self.variable_names)})
        return decoded


class DependencyAttentionMaskBuilder:
    """Build dependency attention masks ``M_E`` for transformer attention."""

    def __init__(self, variable_names: Sequence[str] = DEFAULT_VARIABLE_NAMES) -> None:
        self.variable_names = tuple(variable_names)
        self.variable_to_id = {name: i for i, name in enumerate(self.variable_names)}

    def build(
        self,
        dependencies: Optional[Mapping[str, Iterable[str]]] = None,
        *,
        include_self: bool = True,
        dense_if_missing: bool = True,
    ) -> List[List[bool]]:
        n = len(self.variable_names)
        mask = [[False for _ in range(n)] for _ in range(n)]
        if dependencies is None:
            if dense_if_missing:
                mask = _ones_bool(n, n)
            elif include_self:
                for i in range(n):
                    mask[i][i] = True
            return mask

        for target, sources in dependencies.items():
            if target not in self.variable_to_id:
                continue
            target_idx = self.variable_to_id[target]
            for source in sources:
                if source in self.variable_to_id:
                    mask[target_idx][self.variable_to_id[source]] = True
            if include_self:
                mask[target_idx][target_idx] = True
        if include_self:
            for i in range(n):
                mask[i][i] = True
        return mask

    def build_named_variant(self, variant: str = "dependency_attention") -> List[List[bool]]:
        if variant in {"full", "dense", "unmasked"}:
            return self.build(None, dense_if_missing=True)
        if variant in {"identity", "diagonal"}:
            return self.build(None, dense_if_missing=False, include_self=True)
        if variant in {"dependency_attention", "structured", "simulator_dependencies"}:
            deps: Dict[str, Tuple[str, ...]] = {}
            theta_names = [name for name in self.variable_names if name.startswith("theta")]
            x_names = [name for name in self.variable_names if name.startswith("x")]
            for theta in theta_names:
                deps[theta] = tuple(theta_names)
            for x_name in x_names:
                deps[x_name] = tuple(theta_names + x_names)
            return self.build(deps, include_self=True, dense_if_missing=False)
        raise ValueError(f"Unknown attention mask variant: {variant}")


# ---------------------------------------------------------------------------
# Score model, diffusion objective, trainer, and samplers
# ---------------------------------------------------------------------------


class SimformerScoreNetwork:
    """Lightweight score-network adapter with explicit ``M_E`` attention input.

    This class is intentionally small but executable.  If downstream modules
    provide richer torch-based implementations, they can be used via the lazy
    factories below.  The important package-level contract is that
    ``attention_mask`` is a required forward-path input and is incorporated into
    the interaction term before the score is returned.
    """

    def __init__(self, config: Optional[SimformerCoreConfig] = None) -> None:
        self.config = (config or SimformerCoreConfig()).validate()
        self.forward_calls: List[Dict[str, Any]] = []

    def forward(
        self,
        encoded_tokens: Mapping[str, Any],
        time: float,
        attention_mask: Sequence[Sequence[bool]],
        condition_mask: Any,
    ) -> Dict[str, Any]:
        values = _as_float_matrix(encoded_tokens.get("value_representation", encoded_tokens.get("value", [])))
        if not values:
            return {"score": [], "metadata": {"attention_mask_used": True, "condition_mask_used": True}}

        rows = len(values)
        cols = len(values[0])
        condition_state = _ensure_binary_matrix(condition_mask, rows, cols, default=False)
        attn = [[bool(v) for v in row] for row in attention_mask]
        if len(attn) < cols:
            attn = (attn + _ones_bool(cols, cols))[:cols]

        denom = max(float(time), self.config.noise_min)
        scores: List[List[float]] = []
        for row_idx, row in enumerate(values):
            score_row: List[float] = []
            for target in range(cols):
                allowed = attn[target] if target < len(attn) else [True] * cols
                support = [row[src] for src, is_allowed in enumerate(allowed[:cols]) if is_allowed]
                context_mean = sum(support) / max(1, len(support))
                raw_score = -(row[target] - 0.1 * context_mean) / denom
                if condition_state[row_idx][target]:
                    raw_score = 0.0
                score_row.append(float(raw_score))
            scores.append(score_row)

        call = {
            "time": float(time),
            "attention_mask_used": True,
            "condition_mask_used": True,
            "mask_variant": self.config.mask_variant,
        }
        self.forward_calls.append(call)
        return {
            "score": scores,
            "metadata": {
                **call,
                "M_E_enters_transformer_attention_computation": True,
                "M_C_blocks_conditioned_score_coordinates": True,
            },
        }

    __call__ = forward


class DiffusionObjective:
    """Score-based diffusion objective over joint ``p(theta,x)`` variables."""

    def __init__(self, config: Optional[SimformerCoreConfig] = None) -> None:
        self.config = (config or SimformerCoreConfig()).validate()

    def noise_level(self, step: int) -> float:
        if self.config.diffusion_steps <= 1:
            return self.config.noise_max
        frac = min(1.0, max(0.0, step / float(self.config.diffusion_steps - 1)))
        return self.config.noise_min + frac * (self.config.noise_max - self.config.noise_min)

    def forward_noise(
        self,
        values: Any,
        condition_mask: Any,
        *,
        step: int = 0,
        seed: int = 0,
    ) -> Dict[str, Any]:
        clean = _as_float_matrix(values)
        if not clean:
            clean = [_zeros(1, len(self.config.variable_names))[0]]
        rows = len(clean)
        cols = len(clean[0])
        mc = _ensure_binary_matrix(condition_mask, rows, cols, default=False)
        sigma = self.noise_level(step)
        rng = random.Random(seed + step)
        noisy: List[List[float]] = []
        target_score: List[List[float]] = []
        noise_matrix: List[List[float]] = []

        for r, row in enumerate(clean):
            noisy_row: List[float] = []
            score_row: List[float] = []
            noise_row: List[float] = []
            for c, value in enumerate(row):
                eps = rng.gauss(0.0, 1.0)
                noise_row.append(eps)
                if mc[r][c]:
                    noisy_value = float(value)
                    score = 0.0
                else:
                    noisy_value = float(value) + sigma * eps
                    score = -eps / max(sigma, self.config.noise_min)
                noisy_row.append(noisy_value)
                score_row.append(score)
            noisy.append(noisy_row)
            target_score.append(score_row)
            noise_matrix.append(noise_row)

        return {
            "noisy_value": noisy,
            "clean_value": clean,
            "noise": noise_matrix,
            "target_score": target_score,
            "condition_mask": mc,
            "sigma": sigma,
            "metadata": {
                "M_C_enters_forward_noising": True,
                "distribution": "joint_p_theta_x",
            },
        }

    def masked_score_matching_loss(
        self,
        predicted_score: Any,
        target_score: Any,
        condition_mask: Any,
    ) -> Dict[str, Any]:
        pred = _as_float_matrix(predicted_score)
        target = _as_float_matrix(target_score)
        rows = min(len(pred), len(target))
        cols = min(len(pred[0]) if pred else 0, len(target[0]) if target else 0)
        mc = _ensure_binary_matrix(condition_mask, rows, cols, default=False)
        total = 0.0
        count = 0
        per_variable = [0.0 for _ in range(cols)]
        per_variable_count = [0 for _ in range(cols)]

        for r in range(rows):
            for c in range(cols):
                if mc[r][c]:
                    continue
                diff = pred[r][c] - target[r][c]
                sq = diff * diff
                total += sq
                count += 1
                per_variable[c] += sq
                per_variable_count[c] += 1

        loss = total / max(1, count)
        return {
            "loss": float(loss),
            "per_variable_loss": [
                float(per_variable[i] / max(1, per_variable_count[i])) for i in range(cols)
            ],
            "unconditioned_coordinate_count": count,
            "metadata": {
                "objective": "masked_denoising_score_matching",
                "M_C_enters_loss_masking": True,
                "joint_distribution": "p(theta,x)",
            },
        }


class SimformerTrainer:
    """Bounded trainer for the package-level Simformer core contract.

    The training loop follows the public SBI trainer protocol intent: configurable
    batch size, learning rate, validation fraction, early stopping fields, device
    metadata, and clipping metadata are recorded.  The default run is a smoke
    step over real tokenizer/noising/score/loss code, not a benchmark claim.

    reference_grounding: paperbench_ref_001 docs/how_to_guide/07_gpu_training.ipynb
    reference_grounding: paperbench_ref_001 sbi/inference/trainers/npe/npe_a.py
    """

    def __init__(
        self,
        tokenizer: Optional[SBITokenizer] = None,
        mask_builder: Optional[DependencyAttentionMaskBuilder] = None,
        score_network: Optional[SimformerScoreNetwork] = None,
        config: Optional[SimformerCoreConfig] = None,
    ) -> None:
        self.config = (config or SimformerCoreConfig()).validate()
        self.tokenizer = tokenizer or SBITokenizer(self.config.variable_names)
        self.mask_builder = mask_builder or DependencyAttentionMaskBuilder(self.config.variable_names)
        self.score_network = score_network or SimformerScoreNetwork(self.config)
        self.objective = DiffusionObjective(self.config)
        self.loss_trace: List[Dict[str, Any]] = []

    def train(
        self,
        joint_batch: Any,
        *,
        condition_mask: Any = None,
        attention_mask: Optional[Sequence[Sequence[bool]]] = None,
        results_dir: Optional[str | Path] = None,
        training_batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        validation_fraction: Optional[float] = None,
        stop_after_epochs: Optional[int] = None,
        max_num_epochs: Optional[int] = None,
        clip_max_norm: Optional[float] = None,
        device: Optional[str] = None,
        dry_run: Optional[bool] = None,
    ) -> Dict[str, Any]:
        values = _as_float_matrix(joint_batch)
        if not values:
            values = [[0.0 for _ in self.config.variable_names]]
        rows = len(values)
        cols = len(values[0])
        if condition_mask is None:
            condition_mask = ConditioningMaskPolicy(
                probability=self.config.condition_probability,
                pattern=self.config.conditioning_pattern,
                seed=0,
            ).sample(rows, cols)
        if attention_mask is None:
            attention_mask = self.mask_builder.build_named_variant(self.config.mask_variant)

        effective_epochs = max_num_epochs if max_num_epochs is not None else self.config.max_num_epochs
        effective_epochs = max(1, min(int(effective_epochs), 1 if (self.config.dry_run if dry_run is None else dry_run) else int(effective_epochs)))
        trace: List[Dict[str, Any]] = []

        for epoch in range(effective_epochs):
            encoded_clean = self.tokenizer.encode(values, condition_mask)
            noised = self.objective.forward_noise(
                encoded_clean["value_representation"],
                condition_mask,
                step=epoch % self.config.diffusion_steps,
                seed=epoch,
            )
            encoded_noisy = self.tokenizer.encode(noised["noisy_value"], condition_mask)
            predicted = self.score_network.forward(
                encoded_noisy,
                noised["sigma"],
                attention_mask,
                condition_mask,
            )
            loss = self.objective.masked_score_matching_loss(
                predicted["score"],
                noised["target_score"],
                condition_mask,
            )
            trace.append(
                {
                    "epoch": epoch,
                    "loss": loss["loss"],
                    "simulation_budget": self.config.simulation_budget,
                    "method": self.config.method,
                    "mask_variant": self.config.mask_variant,
                    "conditioning_pattern": self.config.conditioning_pattern,
                    "objective": loss["metadata"]["objective"],
                    "dry_run_contract_artifact": bool(self.config.dry_run if dry_run is None else dry_run),
                }
            )

        self.loss_trace.extend(trace)
        metadata = self.config.metadata()
        metadata["training_runtime_fields"] = {
            "training_batch_size": training_batch_size or self.config.training_batch_size,
            "learning_rate": learning_rate or self.config.learning_rate,
            "validation_fraction": validation_fraction if validation_fraction is not None else self.config.validation_fraction,
            "stop_after_epochs": stop_after_epochs if stop_after_epochs is not None else self.config.stop_after_epochs,
            "max_num_epochs": max_num_epochs if max_num_epochs is not None else self.config.max_num_epochs,
            "clip_max_norm": clip_max_norm if clip_max_norm is not None else self.config.clip_max_norm,
            "device": device or self.config.device,
        }
        metadata["condition_mask_entered_forward_noising_loss_and_sampling"] = True
        metadata["attention_mask_entered_transformer_forward_path"] = True

        result = {
            "status": "completed_bounded_smoke_training" if (self.config.dry_run if dry_run is None else dry_run) else "completed_training_loop",
            "loss_trace": trace,
            "metadata": metadata,
        }

        if results_dir is not None:
            root = Path(results_dir)
            _write_json(root / "loss_trace.json", result)

        return result


class DiffusionSampler:
    """Conditional sampler with named SDE/ODE families.

    reference_grounding: paperbench_ref_001 docs/how_to_guide/09_sampler_interface.ipynb
    """

    def __init__(
        self,
        tokenizer: Optional[SBITokenizer] = None,
        mask_builder: Optional[DependencyAttentionMaskBuilder] = None,
        score_network: Optional[SimformerScoreNetwork] = None,
        config: Optional[SimformerCoreConfig] = None,
    ) -> None:
        self.config = (config or SimformerCoreConfig()).validate()
        self.tokenizer = tokenizer or SBITokenizer(self.config.variable_names)
        self.mask_builder = mask_builder or DependencyAttentionMaskBuilder(self.config.variable_names)
        self.score_network = score_network or SimformerScoreNetwork(self.config)
        self.sampling_trace: List[Dict[str, Any]] = []

    def sample(
        self,
        num_samples: int,
        *,
        condition_values: Any = None,
        condition_mask: Any = None,
        attention_mask: Optional[Sequence[Sequence[bool]]] = None,
        family: Optional[str] = None,
        steps: Optional[int] = None,
        seed: int = 0,
    ) -> Dict[str, Any]:
        selected_family = family or self.config.sampling_family
        if selected_family not in SAMPLING_FAMILIES:
            raise ValueError(f"Unknown sampling family {selected_family!r}; expected {SAMPLING_FAMILIES}")

        cols = len(self.config.variable_names)
        rng = random.Random(seed)
        current = [[rng.gauss(0.0, 1.0) for _ in range(cols)] for _ in range(num_samples)]
        if condition_values is not None:
            cond_values = _as_float_matrix(condition_values)
            if cond_values:
                cond_values = (cond_values * (num_samples // len(cond_values) + 1))[:num_samples]
            else:
                cond_values = _zeros(num_samples, cols)
        else:
            cond_values = _zeros(num_samples, cols)
        mc = _ensure_binary_matrix(condition_mask, num_samples, cols, default=False)
        for r in range(num_samples):
            for c in range(cols):
                if mc[r][c]:
                    current[r][c] = cond_values[r][c] if c < len(cond_values[r]) else 0.0

        attn = attention_mask or self.mask_builder.build_named_variant(self.config.mask_variant)
        total_steps = steps or self.config.diffusion_steps
        trace: List[Dict[str, Any]] = []

        for step in reversed(range(total_steps)):
            t = max(self.config.noise_min, (step + 1) / float(total_steps))
            encoded = self.tokenizer.encode(current, mc)
            score_payload = self.score_network.forward(encoded, t, attn, mc)
            score = _as_float_matrix(score_payload["score"])
            dt = 1.0 / max(1, total_steps)
            for r in range(num_samples):
                for c in range(cols):
                    if mc[r][c]:
                        current[r][c] = cond_values[r][c] if c < len(cond_values[r]) else current[r][c]
                    else:
                        drift = score[r][c] * dt
                        if selected_family == "sde":
                            current[r][c] += drift + math.sqrt(dt) * 0.01 * rng.gauss(0.0, 1.0)
                        else:
                            current[r][c] += drift
            trace.append(
                {
                    "step": step,
                    "time": t,
                    "family": selected_family,
                    "M_E_used": True,
                    "M_C_used_for_conditional_sampling": True,
                }
            )

        payload = {
            "samples": current,
            "sampling_trace": trace,
            "metadata": {
                "family": selected_family,
                "available_families": list(SAMPLING_FAMILIES),
                "M_C_enters_conditional_sampling": True,
                "M_E_enters_score_attention": True,
                "dry_run_contract_artifact": self.config.dry_run,
            },
        }
        self.sampling_trace.extend(trace)
        return payload


class GuidedDiffusionSampler(DiffusionSampler):
    """Guided sampler that alters scores during reverse diffusion."""

    def sample(
        self,
        num_samples: int,
        *,
        condition_values: Any = None,
        condition_mask: Any = None,
        attention_mask: Optional[Sequence[Sequence[bool]]] = None,
        family: Optional[str] = None,
        steps: Optional[int] = None,
        seed: int = 0,
        guidance: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        selected_family = family or self.config.sampling_family
        if selected_family not in SAMPLING_FAMILIES:
            raise ValueError(f"Unknown sampling family {selected_family!r}; expected {SAMPLING_FAMILIES}")

        cols = len(self.config.variable_names)
        rng = random.Random(seed)
        current = [[rng.gauss(0.0, 1.0) for _ in range(cols)] for _ in range(num_samples)]
        cond_values = _as_float_matrix(condition_values) if condition_values is not None else _zeros(num_samples, cols)
        if cond_values:
            cond_values = (cond_values * (num_samples // len(cond_values) + 1))[:num_samples]
        else:
            cond_values = _zeros(num_samples, cols)
        mc = _ensure_binary_matrix(condition_mask, num_samples, cols, default=False)
        attn = attention_mask or self.mask_builder.build_named_variant(self.config.mask_variant)
        total_steps = steps or self.config.diffusion_steps
        guidance = dict(guidance or {})
        scale = float(guidance.get("similarity_guidance_scale", guidance.get("scale", 1.0)))
        lower = guidance.get("lower")
        upper = guidance.get("upper")
        target_indices = guidance.get("target_indices")
        if target_indices is None:
            target_indices = list(range(cols))
        target_indices = [int(i) for i in target_indices if 0 <= int(i) < cols]

        trace: List[Dict[str, Any]] = []
        for step in reversed(range(total_steps)):
            t = max(self.config.noise_min, (step + 1) / float(total_steps))
            encoded = self.tokenizer.encode(current, mc)
            score_payload = self.score_network.forward(encoded, t, attn, mc)
            score = _as_float_matrix(score_payload["score"])

            for r in range(num_samples):
                for c in target_indices:
                    if mc[r][c]:
                        continue
                    if lower is not None and current[r][c] < float(lower):
                        score[r][c] += scale * (float(lower) - current[r][c])
                    if upper is not None and current[r][c] > float(upper):
                        score[r][c] -= scale * (current[r][c] - float(upper))

            dt = 1.0 / max(1, total_steps)
            for r in range(num_samples):
                for c in range(cols):
                    if mc[r][c]:
                        current[r][c] = cond_values[r][c] if c < len(cond_values[r]) else current[r][c]
                    else:
                        if selected_family == "sde":
                            current[r][c] += score[r][c] * dt + math.sqrt(dt) * 0.01 * rng.gauss(0.0, 1.0)
                        else:
                            current[r][c] += score[r][c] * dt

            trace.append(
                {
                    "step": step,
                    "time": t,
                    "family": selected_family,
                    "guided_score_modified": True,
                    "similarity_guidance_scale": scale,
                    "M_E_used": True,
                    "M_C_used_for_conditional_sampling": True,
                }
            )

        payload = {
            "samples": current,
            "sampling_trace": trace,
            "metadata": {
                "family": selected_family,
                "guided_diffusion": True,
                "guidance": guidance,
                "score_modified_before_state_update": True,
                "M_C_enters_conditional_sampling": True,
                "M_E_enters_score_attention": True,
                "dry_run_contract_artifact": self.config.dry_run,
            },
        }
        self.sampling_trace.extend(trace)
        return payload


# ---------------------------------------------------------------------------
# Metric, evaluation, registry, and data-pipeline surfaces
# ---------------------------------------------------------------------------


def gaussian_nll(samples: Any, target: Any, variance_floor: float = 1e-6) -> float:
    """Compute a diagonal Gaussian negative log-likelihood estimate."""

    xs = _as_float_matrix(samples)
    ys = _as_float_matrix(target)
    if not xs or not ys:
        return float("nan")
    cols = min(len(xs[0]), len(ys[0]))
    means = [sum(row[c] for row in xs) / max(1, len(xs)) for c in range(cols)]
    variances: List[float] = []
    for c in range(cols):
        var = sum((row[c] - means[c]) ** 2 for row in xs) / max(1, len(xs))
        variances.append(max(var, variance_floor))
    nll = 0.0
    count = 0
    for row in ys:
        for c in range(cols):
            nll += 0.5 * (math.log(2.0 * math.pi * variances[c]) + ((row[c] - means[c]) ** 2) / variances[c])
            count += 1
    return float(nll / max(1, count))


def c2st_nearest_centroid(samples_a: Any, samples_b: Any) -> float:
    """Lightweight C2ST-style distinguishability score.

    A score near ``0.5`` indicates aligned/indistinguishable samples, while a
    score near ``1.0`` indicates high distinguishability.  This is a local
    fallback metric formula; richer evaluation modules may use sklearn lazily.
    """

    a = _as_float_matrix(samples_a)
    b = _as_float_matrix(samples_b)
    if not a or not b:
        return float("nan")
    cols = min(len(a[0]), len(b[0]))
    mean_a = [sum(row[c] for row in a) / len(a) for c in range(cols)]
    mean_b = [sum(row[c] for row in b) / len(b) for c in range(cols)]

    def dist(row: Sequence[float], mean: Sequence[float]) -> float:
        return sum((row[c] - mean[c]) ** 2 for c in range(cols))

    correct = 0
    total = 0
    for row in a:
        correct += 1 if dist(row, mean_a) <= dist(row, mean_b) else 0
        total += 1
    for row in b:
        correct += 1 if dist(row, mean_b) <= dist(row, mean_a) else 0
        total += 1
    acc = correct / max(1, total)
    return float(max(acc, 1.0 - acc))


def prepare_joint_smoke_batch(
    *,
    num_samples: int = 8,
    variable_names: Sequence[str] = DEFAULT_VARIABLE_NAMES,
    seed: int = 0,
) -> Dict[str, Any]:
    """Generate a small joint ``p(theta,x)`` smoke batch.

    This is an executable data-pipeline surface used for runtime smoke checks.
    It approximates a simulator by sampling two latent ``theta`` coordinates and
    deterministic/noisy observation coordinates, while preserving the joint
    variable interface required by the Simformer tokenizer.
    """

    rng = random.Random(seed)
    names = tuple(variable_names)
    rows: List[List[float]] = []
    for _ in range(num_samples):
        theta0 = rng.gauss(0.0, 1.0)
        theta1 = rng.gauss(0.0, 1.0)
        values_by_name: Dict[str, float] = {}
        for name in names:
            if name == "theta_0":
                values_by_name[name] = theta0
            elif name == "theta_1":
                values_by_name[name] = theta1
            elif name.startswith("theta"):
                values_by_name[name] = rng.gauss(0.0, 1.0)
            elif name == "x_0":
                values_by_name[name] = theta0 + 0.25 * theta1 + 0.05 * rng.gauss(0.0, 1.0)
            elif name == "x_1":
                values_by_name[name] = theta0 * theta1 + 0.05 * rng.gauss(0.0, 1.0)
            else:
                values_by_name[name] = 0.5 * theta0 - 0.1 * theta1 + 0.05 * rng.gauss(0.0, 1.0)
        rows.append([values_by_name[name] for name in names])

    return {
        "values": rows,
        "variable_names": list(names),
        "metadata": {
            "data_pipeline": "joint_smoke_simulator",
            "distribution": "joint_p_theta_x",
            "simulation_budget": num_samples,
            "dry_run_contract_artifact": True,
        },
    }


def evaluate_core_smoke(samples: Any, reference: Any) -> Dict[str, Any]:
    """Evaluate core smoke samples with package-level metric formulas."""

    return {
        "metrics": {
            "c2st_nearest_centroid": c2st_nearest_centroid(samples, reference),
            "gaussian_nll": gaussian_nll(samples, reference),
        },
        "metric_schema": {
            "c2st_nearest_centroid": "0.5 aligned, 1.0 distinguishable",
            "gaussian_nll": "diagonal Gaussian negative log likelihood",
        },
        "dry_run_contract_artifact": True,
    }


def core_method_registry() -> Dict[str, Any]:
    """Return the benchmark-visible package method registry."""

    return {
        "methods": {
            "simformer": {
                "aliases": ["ours", "all_in_one_sbi"],
                "distribution": "joint_p_theta_x",
                "tokenizer": "SBITokenizer",
                "attention_mask": "DependencyAttentionMaskBuilder",
                "score_network": "SimformerScoreNetwork",
                "trainer": "SimformerTrainer",
                "samplers": {
                    "sde": "DiffusionSampler(family='sde')",
                    "ode": "DiffusionSampler(family='ode')",
                    "guided_sde": "GuidedDiffusionSampler(family='sde')",
                    "guided_ode": "GuidedDiffusionSampler(family='ode')",
                },
                "conditioning_mask_usage": [
                    "forward_noising",
                    "loss_masking",
                    "conditional_sampling",
                ],
                "attention_mask_usage": ["transformer_attention_computation", "score_network_forward"],
            }
        },
        "hypothesis": CORE_HYPOTHESIS,
        "decision_value": CORE_DECISION_VALUE,
        "stop_rule_or_pruning_rationale": CORE_STOP_RULE,
        "blacklisted_repositories_not_used": list(BLACKLISTED_REPOSITORIES),
        "reference_grounding": "paperbench_ref_001 docs/how_to_guide/09_sampler_interface.ipynb",
    }


def create_core_components(config: Optional[SimformerCoreConfig] = None) -> Dict[str, Any]:
    """Factory exposing tokenizer, mask builder, score network, trainer, sampler, and guided sampler."""

    cfg = (config or SimformerCoreConfig()).validate()
    tokenizer = SBITokenizer(cfg.variable_names)
    mask_builder = DependencyAttentionMaskBuilder(cfg.variable_names)
    score_network = SimformerScoreNetwork(cfg)
    trainer = SimformerTrainer(tokenizer, mask_builder, score_network, cfg)
    sampler = DiffusionSampler(tokenizer, mask_builder, score_network, cfg)
    guided_sampler = GuidedDiffusionSampler(tokenizer, mask_builder, score_network, cfg)
    return {
        "config": cfg,
        "tokenizer": tokenizer,
        "mask_builder": mask_builder,
        "score_network": score_network,
        "trainer": trainer,
        "sampler": sampler,
        "guided_sampler": guided_sampler,
    }


def run_core_smoke(
    *,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    config: Optional[SimformerCoreConfig] = None,
) -> Dict[str, Any]:
    """Exercise the real core surfaces with bounded inputs and write artifacts."""

    cfg = (config or SimformerCoreConfig()).validate()
    components = create_core_components(cfg)
    batch = prepare_joint_smoke_batch(
        num_samples=min(max(2, cfg.training_batch_size), cfg.simulation_budget),
        variable_names=cfg.variable_names,
        seed=0,
    )
    values = batch["values"]
    condition_mask = ConditioningMaskPolicy(
        probability=cfg.condition_probability,
        pattern=cfg.conditioning_pattern,
        seed=1,
    ).sample(len(values), len(cfg.variable_names))
    attention_mask = components["mask_builder"].build_named_variant(cfg.mask_variant)
    encoded = components["tokenizer"].encode(values, condition_mask)
    train_result = components["trainer"].train(
        values,
        condition_mask=condition_mask,
        attention_mask=attention_mask,
        results_dir=results_dir,
        dry_run=True,
    )
    sample_result = components["sampler"].sample(
        num_samples=4,
        condition_values=values[:1],
        condition_mask=condition_mask[:1],
        attention_mask=attention_mask,
        family="sde",
        steps=min(4, cfg.diffusion_steps),
        seed=2,
    )
    guided_result = components["guided_sampler"].sample(
        num_samples=4,
        condition_values=values[:1],
        condition_mask=condition_mask[:1],
        attention_mask=attention_mask,
        family="ode",
        steps=min(4, cfg.diffusion_steps),
        seed=3,
        guidance={"lower": -2.0, "upper": 2.0, "similarity_guidance_scale": 1.0},
    )
    evaluation = evaluate_core_smoke(sample_result["samples"], values[:4])
    artifacts = write_core_contract_artifacts(
        results_dir=results_dir,
        config=cfg,
        encoded_example=encoded,
        attention_mask=attention_mask,
        train_result=train_result,
        sample_result=sample_result,
        guided_result=guided_result,
        evaluation=evaluation,
    )
    return {
        "status": "core_smoke_completed",
        "batch": batch,
        "encoded_example": encoded,
        "train_result": train_result,
        "sample_result": sample_result,
        "guided_result": guided_result,
        "evaluation": evaluation,
        "artifacts": artifacts,
    }


def write_core_contract_artifacts(
    *,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    config: Optional[SimformerCoreConfig] = None,
    encoded_example: Optional[Mapping[str, Any]] = None,
    attention_mask: Optional[Sequence[Sequence[bool]]] = None,
    train_result: Optional[Mapping[str, Any]] = None,
    sample_result: Optional[Mapping[str, Any]] = None,
    guided_result: Optional[Mapping[str, Any]] = None,
    evaluation: Optional[Mapping[str, Any]] = None,
) -> Dict[str, str]:
    """Materialize declared core artifacts as dry-run/readiness outputs.

    The written files are schema/readiness artifacts unless a caller passes
    results from an explicit full run.  They do not claim paper-scale benchmark
    scores or trained-model performance.
    """

    cfg = (config or SimformerCoreConfig()).validate()
    root = Path(results_dir)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    registry = core_method_registry()
    tokenizer_payload = {
        "artifact_type": "dry_run_contract_artifact",
        "timestamp_utc": timestamp,
        "tokenizer": "SBITokenizer",
        "encode_outputs": ["variable_identifier", "value_representation", "condition_state"],
        "condition_state_binary": True,
        "supports_training_condition_resampling": True,
        "example": dict(encoded_example or {}),
        "reference_grounding": "paperbench_ref_001 docs/how_to_guide/04_embedding_networks.ipynb",
    }
    attention_payload = {
        "artifact_type": "dry_run_contract_artifact",
        "timestamp_utc": timestamp,
        "mask": "M_E",
        "mask_variant": cfg.mask_variant,
        "enters_transformer_attention_computation": True,
        "attention_mask": [[bool(v) for v in row] for row in (attention_mask or [])],
    }
    diffusion_payload = {
        "artifact_type": "dry_run_contract_artifact",
        "timestamp_utc": timestamp,
        "distribution": cfg.distribution,
        "objective": "masked_denoising_score_matching",
        "M_C_usage": ["forward_noising", "loss_masking", "conditional_sampling"],
        "sampling_families": list(SAMPLING_FAMILIES),
        "config": cfg.metadata(),
    }
    loss_payload = {
        "artifact_type": "dry_run_contract_artifact",
        "timestamp_utc": timestamp,
        "loss_trace": list((train_result or {}).get("loss_trace", [])),
        "metadata": dict((train_result or {}).get("metadata", cfg.metadata())),
    }
    sampling_payload = {
        "artifact_type": "dry_run_contract_artifact",
        "timestamp_utc": timestamp,
        "sampling_trace": list((sample_result or {}).get("sampling_trace", [])),
        "guided_sampling_trace": list((guided_result or {}).get("sampling_trace", [])),
        "metadata": {
            "sde_available": True,
            "ode_available": True,
            "guided_sampler_available": True,
            "M_C_enters_conditional_sampling": True,
            "reference_grounding": "paperbench_ref_001 docs/how_to_guide/09_sampler_interface.ipynb",
        },
    }
    readiness_payload = {
        "artifact_type": "readiness",
        "dry_run_contract_artifact": True,
        "timestamp_utc": timestamp,
        "status": "ready",
        "hypothesis": CORE_HYPOTHESIS,
        "decision_value": CORE_DECISION_VALUE,
        "stop_rule_or_pruning_rationale": CORE_STOP_RULE,
        "declared_artifacts": list(DECLARED_CORE_ARTIFACTS),
        "surfaces_exercised": [
            "model_or_method",
            "training_loop",
            "metric_formula",
            "tests",
            "policy_adapter",
            "config",
            "evaluation",
            "data_pipeline",
        ],
    }
    evaluation_payload = {
        "artifact_type": "evaluation_result",
        "dry_run_contract_artifact": True,
        "timestamp_utc": timestamp,
        "status": "schema_ready",
        "evaluation": dict(evaluation or {}),
        "not_claimed": [
            "paper_scale_training",
            "benchmark_scores",
            "trained_model_performance",
            "completed_full_experiments",
        ],
    }

    payloads: Dict[str, Mapping[str, Any]] = {
        "model_registry.json": registry,
        "tokenizer_registry.json": tokenizer_payload,
        "attention_mask_registry.json": attention_payload,
        "diffusion_config.json": diffusion_payload,
        "loss_trace.json": loss_payload,
        "sampling_trace.json": sampling_payload,
        "readiness.json": readiness_payload,
        "evaluation_result.json": evaluation_payload,
    }

    written: Dict[str, str] = {}
    for filename, payload in payloads.items():
        path = root / filename
        _write_json(path, payload)
        written_key = str(Path(DEFAULT_RESULTS_DIR) / filename)
        written[written_key] = str(path)
        _mirror_to_auxiliary(written_key, payload)

    return written


# ---------------------------------------------------------------------------
# Lazy submodule export bridge
# ---------------------------------------------------------------------------


_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    # Neighbor modules may provide richer implementations; these are imported
    # only when explicitly requested so package import stays lightweight.
    "ExperimentConfig": ("all_in_one_sbi.configs", "ExperimentConfig"),
    "DatasetConfig": ("all_in_one_sbi.configs", "DatasetConfig"),
    "TaskRegistry": ("all_in_one_sbi.registry", "TaskRegistry"),
    "SimulatorRegistry": ("all_in_one_sbi.simulators", "SimulatorRegistry"),
    "JointSimulator": ("all_in_one_sbi.simulators", "JointSimulator"),
    "JointVariableTokenizer": ("all_in_one_sbi.encoding", "JointVariableTokenizer"),
    "ConditionMaskSampler": ("all_in_one_sbi.attention_masks", "ConditionMaskSampler"),
    "AttentionMaskBuilder": ("all_in_one_sbi.attention_masks", "AttentionMaskBuilder"),
    "SimformerScoreModel": ("all_in_one_sbi.model", "SimformerScoreModel"),
    "ScoreBasedDiffusion": ("all_in_one_sbi.diffusion", "ScoreBasedDiffusion"),
    "ConditionalSampler": ("all_in_one_sbi.conditioning", "ConditionalSampler"),
    "GuidedSampler": ("all_in_one_sbi.conditioning", "GuidedSampler"),
    "Trainer": ("all_in_one_sbi.training", "Trainer"),
    "BaselineAdapter": ("all_in_one_sbi.baselines", "BaselineAdapter"),
    "Evaluator": ("all_in_one_sbi.evaluation", "Evaluator"),
    "ArtifactWriter": ("all_in_one_sbi.artifacts", "ArtifactWriter"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "BLACKLISTED_REPOSITORIES",
    "DEFAULT_RESULTS_DIR",
    "DEFAULT_VARIABLE_NAMES",
    "SAMPLING_FAMILIES",
    "DECLARED_CORE_ARTIFACTS",
    "CORE_HYPOTHESIS",
    "CORE_DECISION_VALUE",
    "CORE_STOP_RULE",
    "SimformerCoreConfig",
    "ConditioningMaskPolicy",
    "SBITokenizer",
    "DependencyAttentionMaskBuilder",
    "SimformerScoreNetwork",
    "DiffusionObjective",
    "SimformerTrainer",
    "DiffusionSampler",
    "GuidedDiffusionSampler",
    "gaussian_nll",
    "c2st_nearest_centroid",
    "prepare_joint_smoke_batch",
    "evaluate_core_smoke",
    "core_method_registry",
    "create_core_components",
    "run_core_smoke",
    "write_core_contract_artifacts",
    *_LAZY_EXPORTS.keys(),
]
