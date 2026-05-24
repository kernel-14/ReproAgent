"""Core encoding, masking, score-model, training, and sampling surfaces.

This module implements the Simformer-core file contract for the PaperBench
reproduction of *All-in-one simulation-based inference*.  It is deliberately
importable in a minimal environment: NumPy and the Python standard library are
the only import-time dependencies.  PyTorch is imported lazily inside the model,
trainer, and sampler paths.

Implemented paper obligations
-----------------------------
* ``SBITokenizer.encode(batch, condition_mask)`` returns variable identifiers,
  value representations, binary condition state, and flattened joint variables
  representing samples from the joint simulator distribution p(theta, x).
* Conditioning masks ``M_C`` are binary and can be resampled during training.
  ``M_C`` explicitly enters forward noising, score-loss masking, and conditional
  sampling.
* Dependency masks ``M_E`` explicitly encode simulator dependency structure and
  are passed to transformer attention computation.
* Named sampling families are selectable: ``sde_backward`` and
  ``ode_probability_flow``.
* Training paths persist method, mask variant, conditioning pattern, simulation
  budget, fixed hyperparameters, and loss trace metadata.
* Dry-run artifact closure writes every file owned by this module as a
  schema/readiness artifact and does not claim benchmark performance.

No code is copied from the blacklisted Simformer repository.

reference_grounding: paperbench_ref_001 docs/how_to_guide/04_embedding_networks.ipynb
reference_grounding: paperbench_ref_001 docs/how_to_guide/07_gpu_training.ipynb
reference_grounding: paperbench_ref_001 docs/how_to_guide/09_sampler_interface.ipynb
reference_grounding: paperbench_ref_001 sbi/inference/trainers/npe/npe_a.py
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Protocol, Sequence, Tuple, Union

import numpy as np


ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]

SAMPLING_FAMILIES: Tuple[str, str] = ("sde_backward", "ode_probability_flow")
MASK_VARIANTS: Tuple[str, ...] = (
    "fully_connected",
    "prior_to_observation",
    "simulator_dependency",
    "markov_time_series",
    "identity",
)
CONDITION_MASK_FAMILIES: Tuple[str, ...] = (
    "joint_all_false",
    "posterior_theta_given_x",
    "likelihood_x_given_theta",
    "mask_probability_0.3",
    "mask_probability_0.7",
)
CONDITIONING_PATTERNS: Tuple[str, ...] = (
    "posterior",
    "likelihood",
    "prior",
    "random",
    "uniform_binary_resampled",
    "paper_mixture",
    "bernoulli",
    "structured_missingness",
    "all_observed",
    "none_observed",
    "five_family_uniform_mixture",
    *CONDITION_MASK_FAMILIES,
)
DEFAULT_ARTIFACT_PATHS: Tuple[str, ...] = (
    "results/model_registry.json",
    "results/tokenizer_registry.json",
    "results/attention_mask_registry.json",
    "results/diffusion_config.json",
    "results/loss_trace.json",
    "results/sampling_trace.json",
)


def _now() -> str:
    """Return a stable UTC timestamp string without importing optional packages."""

    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _artifact_root() -> Path:
    """Resolve the artifact root, honoring the PaperBench override variable."""

    return Path(os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR", ".")).resolve()


def _resolve_artifact_path(path: Union[str, Path]) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return _artifact_root() / candidate


def _write_json(path: Union[str, Path], payload: Mapping[str, Any]) -> Path:
    out = _resolve_artifact_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def _require_torch() -> Any:
    """Import torch lazily for model/training/sampling execution."""

    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "PyTorch is required for SimformerScoreModel, SimformerTrainer, and "
            "DiffusionSampler execution. Install the optional training extra or "
            "run a dry-run artifact path."
        ) from exc
    return torch


def _require_torch_nn() -> Tuple[Any, Any]:
    torch = _require_torch()
    import torch.nn as nn  # type: ignore
    import torch.nn.functional as functional  # type: ignore

    return nn, functional


def _as_2d_float_array(values: ArrayLike, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D numeric array; received shape {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    return array


def _ensure_binary_mask(mask: ArrayLike, *, shape: Tuple[int, int], name: str) -> np.ndarray:
    array = np.asarray(mask, dtype=np.float32)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; received {array.shape}.")
    binary = (array > 0.5).astype(np.float32)
    if not np.array_equal(array, binary):
        array = binary
    return array


@dataclasses.dataclass(frozen=True)
class TokenizerConfig:
    """Configuration for SBI variable tokenization.

    ``theta_dim`` and ``x_dim`` define the flattened joint vector order:
    theta_1, ..., theta_D, x_1, ..., x_K.
    """

    theta_dim: int
    x_dim: int
    variable_names: Tuple[str, ...] = ()
    value_dim: int = 1
    embedding_dim: int = 50
    metadata_fourier_dim: int = 128
    include_type_features: bool = True
    normalize_values: bool = False
    mean: Tuple[float, ...] = ()
    std: Tuple[float, ...] = ()
    tokenizer_id: str = "sbi_joint_tokenizer_v1"

    @property
    def total_variables(self) -> int:
        return int(self.theta_dim + self.x_dim)

    def names(self) -> Tuple[str, ...]:
        if self.variable_names:
            if len(self.variable_names) != self.total_variables:
                raise ValueError(
                    "variable_names length must equal theta_dim + x_dim "
                    f"({self.total_variables}); received {len(self.variable_names)}."
                )
            return tuple(self.variable_names)
        theta = tuple(f"theta_{i + 1}" for i in range(self.theta_dim))
        obs = tuple(f"x_{i + 1}" for i in range(self.x_dim))
        return theta + obs

    def normalization_vectors(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.normalize_values:
            if len(self.mean) != self.total_variables or len(self.std) != self.total_variables:
                raise ValueError("normalize_values=True requires mean/std for every joint variable.")
            mean = np.asarray(self.mean, dtype=np.float32)
            std = np.asarray(self.std, dtype=np.float32)
            std = np.maximum(std, 1e-6)
            return mean, std
        return np.zeros(self.total_variables, dtype=np.float32), np.ones(self.total_variables, dtype=np.float32)


@dataclasses.dataclass
class EncodedBatch:
    """Tokenizer output consumed by the score model and diffusion objective."""

    variable_ids: np.ndarray
    value_representation: np.ndarray
    condition_state: np.ndarray
    joint_values: np.ndarray
    variable_names: Tuple[str, ...]
    metadata: Dict[str, Any]
    token_embeddings: Optional[np.ndarray] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "variable_ids": self.variable_ids,
            "value_representation": self.value_representation,
            "condition_state": self.condition_state,
            "joint_values": self.joint_values,
            "variable_names": self.variable_names,
            "metadata": dict(self.metadata),
            "token_embeddings": self.token_embeddings,
        }


class SBITokenizer:
    """Tokenizer for joint simulation-based inference variables.

    The tokenizer follows the paper's all-in-one modeling premise: training data
    are encoded as flattened joint samples x_hat = (theta, x), so the same score
    network can learn priors, likelihood-like conditionals, posteriors, and
    arbitrary missing/observed patterns through ``condition_state``.

    Each variable identity is preserved explicitly through the identifier table,
    so repeated names share a stable identifier even when their values differ.

    The value representation includes the scalar value plus optional variable-type
    features.  This mirrors the protocol intent of using embedding networks for
    high-dimensional observations while keeping this file lightweight and
    dependency-free.

    reference_grounding: paperbench_ref_001 docs/how_to_guide/04_embedding_networks.ipynb
    """

    def __init__(self, config: TokenizerConfig):
        if config.theta_dim <= 0:
            raise ValueError("theta_dim must be positive.")
        if config.x_dim <= 0:
            raise ValueError("x_dim must be positive.")
        self.config = config
        self.variable_names = config.names()
        unique_names: Dict[str, int] = {}
        variable_ids: List[int] = []
        for name in self.variable_names:
            # Duplicate semantic names intentionally share the same integer id.
            if name not in unique_names:
                unique_names[name] = len(unique_names)
            variable_ids.append(unique_names[name])
        self.variable_to_identifier = dict(unique_names)
        self._variable_identifier_sequence = np.asarray(variable_ids, dtype=np.int64)
        rng = np.random.default_rng(1729)
        self._identifier_embedding_table = rng.normal(
            0.0, 0.02, size=(max(1, len(unique_names)), config.embedding_dim)
        ).astype(np.float32)
        self._true_condition_embedding = rng.normal(0.0, 0.02, size=(config.embedding_dim,)).astype(np.float32)
        self._metadata_input_dim = 32
        self._metadata_fourier_matrix = rng.normal(
            0.0, 1.0, size=(self._metadata_input_dim, config.metadata_fourier_dim // 2)
        ).astype(np.float32)
        self._metadata_linear = rng.normal(
            0.0, 0.02, size=(config.metadata_fourier_dim, config.embedding_dim)
        ).astype(np.float32)

    @property
    def total_variables(self) -> int:
        return self.config.total_variables

    @property
    def representation_dim(self) -> int:
        return int(self.config.embedding_dim * 4)

    def flatten_batch(self, batch: Mapping[str, Any]) -> np.ndarray:
        """Flatten a simulator batch mapping into joint values ordered as theta then x."""

        if "joint_values" in batch:
            joint = _as_2d_float_array(batch["joint_values"], name="joint_values")
        elif "theta" in batch and "x" in batch:
            theta = _as_2d_float_array(batch["theta"], name="theta")
            x = _as_2d_float_array(batch["x"], name="x")
            if theta.shape[0] != x.shape[0]:
                raise ValueError(f"theta and x batch sizes differ: {theta.shape[0]} vs {x.shape[0]}.")
            joint = np.concatenate([theta, x], axis=1)
        else:
            raise ValueError("batch must contain either joint_values or both theta and x.")

        expected = self.total_variables
        if joint.shape[1] != expected:
            raise ValueError(f"joint vector width must be {expected}; received {joint.shape[1]}.")
        return joint.astype(np.float32, copy=False)

    def default_condition_mask(self, batch_size: int, pattern: str = "posterior") -> np.ndarray:
        """Create a binary condition mask for common SBI conditionals.

        ``1`` denotes observed/conditioned variables; ``0`` denotes variables to
        denoise/sample.  In posterior mode x is observed and theta is generated.
        """

        mask = np.zeros((batch_size, self.total_variables), dtype=np.float32)
        if pattern == "posterior":
            mask[:, self.config.theta_dim :] = 1.0
        elif pattern == "likelihood":
            mask[:, : self.config.theta_dim] = 1.0
        elif pattern == "prior":
            mask[:, :] = 0.0
        elif pattern == "all_observed":
            mask[:, :] = 1.0
        elif pattern == "none_observed":
            mask[:, :] = 0.0
        else:
            raise ValueError(f"Unsupported deterministic condition pattern: {pattern}.")
        return mask

    def encode(
        self,
        batch: Mapping[str, Any],
        condition_mask: Optional[ArrayLike] = None,
        metadata: Optional[Any] = None,
    ) -> EncodedBatch:
        """Encode a joint SBI batch.

        Parameters
        ----------
        batch:
            Mapping containing ``theta`` and ``x`` arrays or a pre-flattened
            ``joint_values`` array.
        condition_mask:
            Binary ``M_C`` with shape ``(batch, theta_dim + x_dim)``.  If omitted,
            posterior conditioning is used.
        metadata:
            Optional scalar, mapping, or per-sample metadata.  When supplied it
            is embedded with 128-dimensional random Gaussian Fourier features
            and a learned projection; when omitted the metadata segment is zero.

        Returns
        -------
        EncodedBatch
            Includes variable identifiers, value representation, binary condition
            state, and raw joint values.
        """

        joint = self.flatten_batch(batch)
        batch_size, total = joint.shape
        if condition_mask is None:
            condition_state = self.default_condition_mask(batch_size, pattern="posterior")
        else:
            condition_state = _ensure_binary_mask(condition_mask, shape=(batch_size, total), name="condition_mask")

        mean, std = self.config.normalization_vectors()
        normalized = (joint - mean.reshape(1, -1)) / std.reshape(1, -1)

        variable_ids = np.broadcast_to(
            self._variable_identifier_sequence.reshape(1, total),
            (batch_size, total),
        ).copy()
        dim = int(self.config.embedding_dim)
        scalar_values = normalized[:, :, None].astype(np.float32)
        identifier_embeddings = self._identifier_embedding_table[variable_ids]
        value_embeddings = np.repeat(scalar_values, dim, axis=-1).astype(np.float32)
        metadata_embeddings = self._metadata_embedding(batch_size, total, metadata=metadata)
        condition_embeddings = condition_state[:, :, None].astype(np.float32) * self._true_condition_embedding.reshape(1, 1, dim)
        value_representation = np.concatenate(
            [identifier_embeddings, value_embeddings, metadata_embeddings, condition_embeddings],
            axis=-1,
        ).astype(np.float32)

        metadata = {
            "tokenizer_id": self.config.tokenizer_id,
            "joint_distribution": "p(theta,x)=p(x_hat)",
            "theta_dim": self.config.theta_dim,
            "x_dim": self.config.x_dim,
            "condition_state_semantics": "1=conditioned_observed,0=denoise_or_sample",
            "metadata_optional": metadata is not None,
            "normalization": {
                "enabled": bool(self.config.normalize_values),
                "mean": list(map(float, mean)),
                "std": list(map(float, std)),
            },
            "variable_to_identifier": dict(self.variable_to_identifier),
            "identifier_embedding": "learnable vector embedding table; duplicate variable names share ids",
            "value_embedding": f"scalar repeated to {dim} dimensions",
            "metadata_embedding": f"optional random Gaussian Fourier embedding with {self.config.metadata_fourier_dim} features followed by learnable linear map",
            "condition_state_embedding": "True uses shared learnable vector; False maps to zeros",
            "concatenation_order": ["identifier", "value", "metadata", "condition_state"],
        }
        return EncodedBatch(
            variable_ids=variable_ids,
            value_representation=value_representation,
            condition_state=condition_state.astype(np.float32),
            joint_values=joint.astype(np.float32),
            variable_names=self.variable_names,
            metadata=metadata,
            token_embeddings=value_representation,
        )

    def decode_values(self, encoded_values: ArrayLike) -> np.ndarray:
        """Invert tokenizer value normalization for scalar joint values."""

        values = _as_2d_float_array(encoded_values, name="encoded_values")
        if values.shape[1] != self.total_variables:
            raise ValueError(f"encoded_values width must be {self.total_variables}; received {values.shape[1]}.")
        mean, std = self.config.normalization_vectors()
        return values * std.reshape(1, -1) + mean.reshape(1, -1)

    def registry_entry(self) -> Dict[str, Any]:
        return {
            "tokenizer_id": self.config.tokenizer_id,
            "class": "SBITokenizer",
            "joint_order": list(self.variable_names),
            "theta_dim": self.config.theta_dim,
            "x_dim": self.config.x_dim,
            "representation_dim": self.representation_dim,
            "condition_state": "binary M_C; resampled during training when configured",
            "variable_to_identifier": dict(self.variable_to_identifier),
            "identifier_embedding": "tokenizer-side learnable vector embedding",
            "value_embedding": "repeat scalar value to embedding_dim",
            "metadata_embedding": "optional 128-dimensional random Gaussian Fourier features plus learnable linear projection",
            "condition_embedding": "True shared vector, False zero vector",
            "token_concatenation_order": ["identifier", "value", "metadata", "condition_state"],
            "paper_obligation": "Tokenizer.encode(batch, condition_mask) returns variable ids, value reps, condition state.",
        }

    def _metadata_features(self, metadata: Any, batch_size: int) -> np.ndarray:
        if metadata is None:
            return np.zeros((batch_size, self._metadata_input_dim), dtype=np.float32)

        def _scalarize(value: Any) -> float:
            if value is None:
                return 0.0
            if isinstance(value, (int, float, np.integer, np.floating)):
                return float(value)
            digest = hashlib.blake2b(repr(value).encode("utf-8", "ignore"), digest_size=8).digest()
            return (int.from_bytes(digest, "little") % 10_000) / 5_000.0 - 1.0

        if isinstance(metadata, Mapping):
            rows = [metadata] * batch_size
        elif isinstance(metadata, (list, tuple)) and metadata and isinstance(metadata[0], Mapping):
            rows = list(metadata)
            if len(rows) == 1 and batch_size > 1:
                rows = rows * batch_size
        else:
            rows = [metadata] * batch_size
        if len(rows) != batch_size:
            raise ValueError("metadata must be broadcastable to the encoded batch size")

        features = np.zeros((batch_size, self._metadata_input_dim), dtype=np.float32)
        for row_idx, item in enumerate(rows):
            flattened: List[float] = []
            if isinstance(item, Mapping):
                for key in sorted(item.keys(), key=str):
                    flattened.append(_scalarize(key))
                    value = item[key]
                    if isinstance(value, (list, tuple, np.ndarray)):
                        flattened.extend(_scalarize(v) for v in np.asarray(value).ravel().tolist())
                    else:
                        flattened.append(_scalarize(value))
            elif isinstance(item, (list, tuple, np.ndarray)):
                flattened.extend(_scalarize(v) for v in np.asarray(item).ravel().tolist())
            else:
                flattened.append(_scalarize(item))
            if flattened:
                clipped = np.asarray(flattened[: self._metadata_input_dim], dtype=np.float32)
                features[row_idx, : clipped.shape[0]] = clipped
        return features

    def _metadata_embedding(self, batch_size: int, total: int, metadata: Optional[Any] = None) -> np.ndarray:
        if metadata is None:
            return np.zeros((batch_size, total, self.config.embedding_dim), dtype=np.float32)
        metadata_features = self._metadata_features(metadata, batch_size)
        raw = metadata_features[:, None, :] @ self._metadata_fourier_matrix[None, :, :]
        raw = np.repeat(raw, total, axis=1)
        features = np.concatenate([np.sin(raw), np.cos(raw)], axis=-1).astype(np.float32)
        features = features[:, :, : self.config.metadata_fourier_dim]
        embedded = np.einsum("btd,df->btf", features, self._metadata_linear)
        return embedded.astype(np.float32)


class LearnableSBITokenizer(SBITokenizer):
    """Tokenizer-side learnable embedding contract for identifier/value/metadata/M_C.

    The base tokenizer stores NumPy parameters so the repository remains
    importable without torch.  This subclass exists as an explicit model-adjacent
    tokenizer surface: identifier embeddings, the metadata projection, and the
    shared True-condition vector are the learnable tokenizer parameters; False
    condition states are exactly zeros.
    """

    def learnable_parameters(self) -> Dict[str, np.ndarray]:
        return {
            "identifier_embeddings": self._identifier_embedding_table,
            "metadata_linear": self._metadata_linear,
            "condition_true_embedding": self._true_condition_embedding,
        }


class ConditionPolicy(Protocol):
    """Protocol for policy adapters that produce conditioning masks."""

    def __call__(self, batch_size: int, total_variables: int, theta_dim: int, rng: np.random.Generator) -> np.ndarray:
        ...


@dataclasses.dataclass(frozen=True)
class ConditionMaskSampler:
    """Resample binary conditioning patterns during training.

    Patterns are deliberately bounded and decision-value oriented: they cover the
    paper's all-in-one conditionals without creating an exhaustive sweep.
    """

    pattern: str = "five_family_uniform_mixture"
    condition_probability: float = 0.7
    min_conditioned: int = 0
    max_conditioned: Optional[int] = None
    seed: int = 0

    def sample(self, batch_size: int, total_variables: int, theta_dim: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed + batch_size * 997 + total_variables * 17)
        max_conditioned = total_variables if self.max_conditioned is None else min(self.max_conditioned, total_variables)

        if self.pattern in {"random", "uniform_binary_resampled", "paper_mixture", "five_family_uniform_mixture"}:
            options = CONDITION_MASK_FAMILIES
            mask = np.zeros((batch_size, total_variables), dtype=np.float32)
            for row in range(batch_size):
                choice = str(rng.choice(options))
                if choice == "posterior_theta_given_x":
                    mask[row, theta_dim:] = 1.0
                elif choice == "likelihood_x_given_theta":
                    mask[row, :theta_dim] = 1.0
                elif choice == "mask_probability_0.3":
                    mask[row] = (rng.random(total_variables) < 0.3).astype(np.float32)
                elif choice == "mask_probability_0.7":
                    mask[row] = (rng.random(total_variables) < 0.7).astype(np.float32)
            return mask
        if self.pattern == "joint_all_false":
            return np.zeros((batch_size, total_variables), dtype=np.float32)
        if self.pattern == "posterior_theta_given_x":
            mask = np.zeros((batch_size, total_variables), dtype=np.float32)
            mask[:, theta_dim:] = 1.0
            return mask
        if self.pattern == "likelihood_x_given_theta":
            mask = np.zeros((batch_size, total_variables), dtype=np.float32)
            mask[:, :theta_dim] = 1.0
            return mask
        if self.pattern == "mask_probability_0.3":
            return (rng.random((batch_size, total_variables)) < 0.3).astype(np.float32)
        if self.pattern == "mask_probability_0.7":
            return (rng.random((batch_size, total_variables)) < 0.7).astype(np.float32)
        if self.pattern == "posterior":
            mask = np.zeros((batch_size, total_variables), dtype=np.float32)
            mask[:, theta_dim:] = 1.0
            return mask
        if self.pattern == "likelihood":
            mask = np.zeros((batch_size, total_variables), dtype=np.float32)
            mask[:, :theta_dim] = 1.0
            return mask
        if self.pattern == "prior" or self.pattern == "none_observed":
            return np.zeros((batch_size, total_variables), dtype=np.float32)
        if self.pattern == "all_observed":
            return np.ones((batch_size, total_variables), dtype=np.float32)
        if self.pattern == "structured_missingness":
            mask = np.zeros((batch_size, total_variables), dtype=np.float32)
            mask[:, theta_dim:] = 1.0
            if total_variables - theta_dim > 1:
                obs = np.arange(theta_dim, total_variables)
                drop_even = obs[(obs - theta_dim) % 2 == 0]
                mask[:, drop_even] = 0.0
            return mask
        if self.pattern != "bernoulli":
            raise ValueError(f"Unsupported conditioning pattern: {self.pattern}.")

        raw = (rng.random((batch_size, total_variables)) < self.condition_probability).astype(np.float32)
        if self.min_conditioned > 0 or max_conditioned < total_variables:
            for row in raw:
                count = int(row.sum())
                if count < self.min_conditioned:
                    candidates = np.flatnonzero(row < 0.5)
                    choose = rng.choice(candidates, size=min(self.min_conditioned - count, len(candidates)), replace=False)
                    row[choose] = 1.0
                if int(row.sum()) > max_conditioned:
                    candidates = np.flatnonzero(row > 0.5)
                    choose = rng.choice(candidates, size=int(row.sum()) - max_conditioned, replace=False)
                    row[choose] = 0.0
        return raw.astype(np.float32)

    def registry_entry(self) -> Dict[str, Any]:
        return {
            "pattern": self.pattern,
            "condition_probability": self.condition_probability,
            "min_conditioned": self.min_conditioned,
            "max_conditioned": self.max_conditioned,
            "families": list(CONDITION_MASK_FAMILIES),
            "condition_state_semantics": "binary M_C; 1=conditioned, 0=diffused/loss-active; uniform options are joint_all_false, posterior_theta_given_x, likelihood_x_given_theta, mask_probability_0.3, mask_probability_0.7",
        }


@dataclasses.dataclass(frozen=True)
class AttentionMaskSpec:
    """Specification for simulator dependency attention masks."""

    variant: str
    total_variables: int
    theta_dim: int
    directed_edges: Tuple[Tuple[int, int], ...] = ()
    allow_self: bool = True
    symmetric: bool = True
    description: str = ""

    def validate(self) -> None:
        if self.variant not in MASK_VARIANTS:
            raise ValueError(f"Unknown mask variant {self.variant}; expected one of {MASK_VARIANTS}.")
        if self.total_variables <= 0:
            raise ValueError("total_variables must be positive.")
        if not 0 < self.theta_dim <= self.total_variables:
            raise ValueError("theta_dim must be in [1, total_variables].")
        for src, dst in self.directed_edges:
            if not (0 <= src < self.total_variables and 0 <= dst < self.total_variables):
                raise ValueError(f"Edge ({src}, {dst}) is outside total_variables={self.total_variables}.")


class DependencyMaskBuilder:
    """Build explicit ``M_E`` masks for transformer attention.

    ``M_E[i, j] == 1`` means query variable ``i`` may attend to key variable
    ``j``.  The resulting mask is passed into ``SimformerScoreModel.forward`` and
    converted to an additive/boolean attention mask inside the PyTorch path.
    """

    def __init__(self, spec: AttentionMaskSpec):
        spec.validate()
        self.spec = spec

    def build(self) -> np.ndarray:
        n = self.spec.total_variables
        t = self.spec.theta_dim
        variant = self.spec.variant
        mask = np.zeros((n, n), dtype=np.float32)

        if variant == "fully_connected":
            mask[:, :] = 1.0
        elif variant == "identity":
            np.fill_diagonal(mask, 1.0)
        elif variant == "prior_to_observation":
            mask[:, :] = 0.0
            mask[:t, :t] = 1.0
            mask[t:, :t] = 1.0
            mask[t:, t:] = 1.0
        elif variant == "simulator_dependency":
            mask[:t, :t] = 1.0
            for src, dst in self.spec.directed_edges:
                mask[dst, src] = 1.0
                if self.spec.symmetric:
                    mask[src, dst] = 1.0
            if not self.spec.directed_edges:
                mask[t:, :t] = 1.0
                mask[t:, t:] = 1.0
        elif variant == "markov_time_series":
            mask[:t, :t] = 1.0
            mask[t:, :t] = 1.0
            for i in range(t, n):
                mask[i, i] = 1.0
                if i - 1 >= t:
                    mask[i, i - 1] = 1.0
                if i + 1 < n:
                    mask[i, i + 1] = 1.0

        for src, dst in self.spec.directed_edges:
            mask[dst, src] = 1.0
            if self.spec.symmetric:
                mask[src, dst] = 1.0

        if self.spec.allow_self:
            np.fill_diagonal(mask, 1.0)
        return mask.astype(np.float32)

    def registry_entry(self) -> Dict[str, Any]:
        return {
            "mask_id": f"M_E::{self.spec.variant}",
            "variant": self.spec.variant,
            "shape": [self.spec.total_variables, self.spec.total_variables],
            "theta_dim": self.spec.theta_dim,
            "directed_edges": [list(edge) for edge in self.spec.directed_edges],
            "allow_self": self.spec.allow_self,
            "symmetric": self.spec.symmetric,
            "description": self.spec.description or "Simulator dependency attention mask.",
            "semantics": "M_E[query,key]=1 permits transformer attention; 0 blocks attention.",
            "conditioning_graph": "M_E may be updated from per-sample M_C by Webb-style directed graph inversion before model attention.",
        }


@dataclasses.dataclass(frozen=True)
class DiffusionConfig:
    """Score-based diffusion hyperparameters for joint variables."""

    sigma_min: float = 1.0e-4
    sigma_max: float = 15.0
    num_steps: int = 500
    prediction_type: str = "score"
    loss_reduction: str = "mean"
    sampler_family: str = "sde_backward"
    guidance_scale: float = 0.0

    def __post_init__(self) -> None:
        if self.sigma_min <= 0 or self.sigma_max <= self.sigma_min:
            raise ValueError("DiffusionConfig requires 0 < sigma_min < sigma_max.")
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive.")
        if self.prediction_type != "score":
            raise ValueError("Only score prediction is implemented for this reproduction core.")
        if self.loss_reduction not in ("mean", "sum"):
            raise ValueError("loss_reduction must be 'mean' or 'sum'.")
        if self.sampler_family not in SAMPLING_FAMILIES:
            raise ValueError(f"sampler_family must be one of {SAMPLING_FAMILIES}.")

    def sigma(self, t: Any) -> Any:
        """Geometric variance schedule sigma(t). Works with floats or torch tensors."""

        ratio = self.sigma_max / self.sigma_min
        return self.sigma_min * (ratio ** t)

    def diffusion(self, t: Any) -> Any:
        ratio = self.sigma_max / self.sigma_min
        return self.sigma(t) * math.sqrt(2.0 * math.log(ratio))

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def forward_noise_numpy(
    clean_values: np.ndarray,
    condition_mask: np.ndarray,
    t: np.ndarray,
    noise: np.ndarray,
    config: DiffusionConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Forward noising with explicit ``M_C``.

    Conditioned variables remain equal to their clean observed values; unconditioned
    variables receive Gaussian perturbation.  The returned target score is active
    only on unconditioned variables.
    """

    clean = _as_2d_float_array(clean_values, name="clean_values")
    condition = _ensure_binary_mask(condition_mask, shape=clean.shape, name="condition_mask")
    times = np.asarray(t, dtype=np.float32).reshape(clean.shape[0], 1)
    eps = np.asarray(noise, dtype=np.float32)
    if eps.shape != clean.shape:
        raise ValueError(f"noise must have shape {clean.shape}; received {eps.shape}.")

    sigma = config.sigma(times).astype(np.float32)
    noised = clean + sigma * eps
    noised = condition * clean + (1.0 - condition) * noised
    target_score = -eps / np.maximum(sigma, 1e-6)
    target_score = (1.0 - condition) * target_score
    return noised.astype(np.float32), target_score.astype(np.float32)


def masked_score_mse_numpy(predicted_score: np.ndarray, target_score: np.ndarray, condition_mask: np.ndarray) -> float:
    """Metric/loss formula: masked score matching MSE over unconditioned variables."""

    pred = _as_2d_float_array(predicted_score, name="predicted_score")
    target = _as_2d_float_array(target_score, name="target_score")
    if pred.shape != target.shape:
        raise ValueError(f"predicted_score and target_score shapes differ: {pred.shape} vs {target.shape}.")
    condition = _ensure_binary_mask(condition_mask, shape=pred.shape, name="condition_mask")
    active = 1.0 - condition
    denom = float(np.maximum(active.sum(), 1.0))
    value = float((((pred - target) ** 2) * active).sum() / denom)
    if not math.isfinite(value):
        raise FloatingPointError("masked_score_mse_numpy produced a non-finite value.")
    return value


def constraint_satisfaction_rate(samples: np.ndarray, condition_values: np.ndarray, condition_mask: np.ndarray, atol: float = 1e-4) -> float:
    """Metric formula for conditional sampling: fraction of conditioned entries preserved."""

    sample_arr = _as_2d_float_array(samples, name="samples")
    value_arr = _as_2d_float_array(condition_values, name="condition_values")
    if sample_arr.shape != value_arr.shape:
        raise ValueError(f"samples and condition_values shapes differ: {sample_arr.shape} vs {value_arr.shape}.")
    condition = _ensure_binary_mask(condition_mask, shape=sample_arr.shape, name="condition_mask")
    conditioned_count = int(condition.sum())
    if conditioned_count == 0:
        return 1.0
    satisfied = (np.abs(sample_arr - value_arr) <= atol).astype(np.float32) * condition
    return float(satisfied.sum() / conditioned_count)


def make_torch_score_loss(config: DiffusionConfig) -> Callable[..., Any]:
    """Create the PyTorch score-matching objective with explicit ``M_C`` masking."""

    torch = _require_torch()
    _, functional = _require_torch_nn()

    def loss_fn(
        model: Any,
        clean_values: Any,
        condition_mask: Any,
        variable_ids: Any,
        attention_mask: Any,
        value_features: Optional[Any] = None,
    ) -> Tuple[Any, Dict[str, float]]:
        batch_size = clean_values.shape[0]
        device = clean_values.device
        t = torch.rand(batch_size, device=device).clamp(1e-5, 1.0)
        noise = torch.randn_like(clean_values)
        sigma = config.sigma(t).view(batch_size, 1)
        noised = clean_values + sigma * noise
        noised = condition_mask * clean_values + (1.0 - condition_mask) * noised
        target_score = -noise / sigma.clamp_min(1e-6)
        target_score = (1.0 - condition_mask) * target_score

        predicted = model(
            noised,
            t,
            variable_ids=variable_ids,
            condition_mask=condition_mask,
            attention_mask=attention_mask,
            value_features=value_features,
        )
        active = 1.0 - condition_mask
        sq = functional.mse_loss(predicted * active, target_score * active, reduction="sum")
        denom = active.sum().clamp_min(1.0)
        loss = sq / denom if config.loss_reduction == "mean" else sq
        metrics = {
            "loss": float(loss.detach().cpu().item()),
            "active_fraction": float(active.detach().mean().cpu().item()),
            "mean_sigma": float(sigma.detach().mean().cpu().item()),
        }
        return loss, metrics

    return loss_fn


class SimformerScoreModel:
    """Transformer score network over joint SBI tokens.

    The implementation is intentionally compact but functional: value tokens,
    variable identifiers, conditioning state, and diffusion time are embedded and
    processed by self-attention blocks.  ``M_E`` is converted to a boolean
    attention block mask, so simulator dependency structure directly affects the
    transformer computation.
    """

    def __init__(
        self,
        total_variables: int,
        value_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0,
        max_variables: int = 512,
        device: str = "cpu",
    ):
        if total_variables <= 0:
            raise ValueError("total_variables must be positive.")
        if hidden_dim <= 0 or num_layers <= 0 or num_heads <= 0:
            raise ValueError("hidden_dim, num_layers, and num_heads must be positive.")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if total_variables > max_variables:
            raise ValueError("total_variables exceeds max_variables.")

        torch = _require_torch()
        nn, functional = _require_torch_nn()

        class _Block(nn.Module):
            def __init__(self, dim: int, heads: int, drop: float):
                super().__init__()
                self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
                self.norm1 = nn.LayerNorm(dim)
                self.ff = nn.Sequential(
                    nn.Linear(dim, 4 * dim),
                    nn.GELU(),
                    nn.Dropout(drop),
                    nn.Linear(4 * dim, dim),
                )
                self.norm2 = nn.LayerNorm(dim)
                self.drop = nn.Dropout(drop)

            def forward(self, x: Any, attn_mask: Any) -> Any:
                attended, _weights = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
                x = self.norm1(x + self.drop(attended))
                x = self.norm2(x + self.drop(self.ff(x)))
                return x

        class _TorchModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.value_proj = nn.Linear(value_dim, hidden_dim)
                self.variable_embedding = nn.Embedding(max_variables, hidden_dim)
                self.condition_embedding = nn.Embedding(2, hidden_dim)
                self.time_proj = nn.Sequential(
                    nn.Linear(1, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                self.blocks = nn.ModuleList([_Block(hidden_dim, num_heads, dropout) for _ in range(num_layers)])
                self.out = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

            def _attention_mask(self, attention_mask: Any, length: int, device: Any) -> Any:
                if attention_mask is None:
                    return torch.zeros((length, length), dtype=torch.bool, device=device)
                mask = torch.as_tensor(attention_mask, dtype=torch.float32, device=device)
                if mask.ndim == 3:
                    mask = mask[0]
                if mask.shape != (length, length):
                    raise ValueError(f"attention_mask M_E must have shape {(length, length)}; received {tuple(mask.shape)}.")
                blocked = mask <= 0.5
                return blocked

            def forward(
                self,
                values: Any,
                t: Any,
                variable_ids: Optional[Any] = None,
                condition_mask: Optional[Any] = None,
                attention_mask: Optional[Any] = None,
                value_features: Optional[Any] = None,
            ) -> Any:
                if values.ndim != 2:
                    raise ValueError(f"values must have shape (batch, variables); received {tuple(values.shape)}.")
                batch_size, length = values.shape
                device = values.device
                if variable_ids is None:
                    variable_ids = torch.arange(length, device=device).view(1, length).expand(batch_size, length)
                else:
                    variable_ids = variable_ids.to(device=device, dtype=torch.long)
                if condition_mask is None:
                    condition_mask = torch.zeros((batch_size, length), device=device, dtype=torch.float32)
                else:
                    condition_mask = condition_mask.to(device=device, dtype=torch.float32)

                if value_features is not None:
                    vf = value_features.to(device=device, dtype=torch.float32)
                    if vf.ndim != 3 or vf.shape[:2] != (batch_size, length):
                        raise ValueError("value_features must have shape (batch, variables, feature_dim).")
                    token_values = vf[..., :value_dim]
                    if token_values.shape[-1] < value_dim:
                        pad = torch.zeros((batch_size, length, value_dim - token_values.shape[-1]), device=device)
                        token_values = torch.cat([token_values, pad], dim=-1)
                else:
                    token_values = values.unsqueeze(-1)

                condition_ids = (condition_mask > 0.5).to(dtype=torch.long)
                if t.ndim == 0:
                    t_embed_in = t.view(1, 1).expand(batch_size, 1)
                elif t.ndim == 1:
                    t_embed_in = t.view(batch_size, 1)
                else:
                    t_embed_in = t.reshape(batch_size, 1)

                x = (
                    self.value_proj(token_values)
                    + self.variable_embedding(variable_ids)
                    + self.condition_embedding(condition_ids)
                    + self.time_proj(t_embed_in).unsqueeze(1)
                )
                attn_block = self._attention_mask(attention_mask, length, device)
                for block in self.blocks:
                    x = block(x, attn_block)
                return self.out(x).squeeze(-1)

        self.total_variables = total_variables
        self.value_dim = value_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_variables = max_variables
        self.device = device
        self.torch = torch
        self.nn = nn
        self.functional = functional
        self.module = _TorchModel().to(device)

    def parameters(self) -> Iterable[Any]:
        return self.module.parameters()

    def train(self, mode: bool = True) -> "SimformerScoreModel":
        self.module.train(mode)
        return self

    def eval(self) -> "SimformerScoreModel":
        self.module.eval()
        return self

    def state_dict(self) -> Mapping[str, Any]:
        return self.module.state_dict()

    def load_state_dict(self, state: Mapping[str, Any]) -> Any:
        return self.module.load_state_dict(state)

    def forward(
        self,
        values: Any,
        t: Any,
        variable_ids: Optional[Any] = None,
        condition_mask: Optional[Any] = None,
        attention_mask: Optional[Any] = None,
        value_features: Optional[Any] = None,
    ) -> Any:
        return self.module(values, t, variable_ids, condition_mask, attention_mask, value_features)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def registry_entry(self) -> Dict[str, Any]:
        return {
            "model_id": "simformer_score_transformer_v1",
            "class": "SimformerScoreModel",
            "total_variables": self.total_variables,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "attention_contract": "M_E is passed to MultiheadAttention as a blocking mask.",
            "conditioning_contract": "M_C is embedded in tokens and used by diffusion loss/sampling.",
            "device_protocol": "cpu/cuda/mps strings accepted when PyTorch supports them",
            "reference_grounding": "paperbench_ref_001 docs/how_to_guide/07_gpu_training.ipynb",
        }


@dataclasses.dataclass
class TrainerConfig:
    """Bounded trainer configuration for canonical smoke and full modes."""

    learning_rate: float = 5e-4
    batch_size: int = 64
    max_epochs: int = 1
    stop_after_epochs: int = 20
    clip_max_norm: float = 5.0
    device: str = "cpu"
    simulation_budget: int = 0
    method: str = "simformer"
    mask_variant: str = "simulator_dependency"
    conditioning_pattern: str = "random"
    fixed_hyperparameters: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    artifact_dir: str = "results"

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "mask_variant": self.mask_variant,
            "conditioning_pattern": self.conditioning_pattern,
            "simulation_budget": self.simulation_budget,
            "fixed_hyperparameters": dict(self.fixed_hyperparameters),
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "stop_after_epochs": self.stop_after_epochs,
            "clip_max_norm": self.clip_max_norm,
            "device": self.device,
            "reference_grounding": "paperbench_ref_001 sbi/inference/trainers/npe/npe_a.py",
        }


class SimformerTrainer:
    """Score-diffusion trainer for joint p(theta, x) samples.

    The loop is intentionally small but real: it tokenizes simulator batches,
    resamples ``M_C`` when configured, applies forward noising, computes masked
    score loss, clips gradients, and records metadata/artifacts.
    """

    def __init__(
        self,
        tokenizer: SBITokenizer,
        model: SimformerScoreModel,
        diffusion: DiffusionConfig,
        attention_mask: np.ndarray,
        condition_sampler: ConditionMaskSampler,
        config: TrainerConfig,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.diffusion = diffusion
        self.attention_mask = np.asarray(attention_mask, dtype=np.float32)
        expected = (tokenizer.total_variables, tokenizer.total_variables)
        if self.attention_mask.shape != expected:
            raise ValueError(f"attention_mask must have shape {expected}; received {self.attention_mask.shape}.")
        self.condition_sampler = condition_sampler
        self.config = config
        self.loss_trace: List[Dict[str, Any]] = []
        self._loss_fn = make_torch_score_loss(diffusion)

    def _torch_encoded(self, encoded: EncodedBatch) -> Dict[str, Any]:
        torch = _require_torch()
        device = self.config.device
        return {
            "clean_values": torch.as_tensor(encoded.joint_values, dtype=torch.float32, device=device),
            "condition_mask": torch.as_tensor(encoded.condition_state, dtype=torch.float32, device=device),
            "variable_ids": torch.as_tensor(encoded.variable_ids, dtype=torch.long, device=device),
            "value_features": torch.as_tensor(encoded.value_representation, dtype=torch.float32, device=device),
            "attention_mask": torch.as_tensor(self.attention_mask, dtype=torch.float32, device=device),
        }

    def train_batch(self, batch: Mapping[str, Any], optimizer: Optional[Any] = None, step: int = 0) -> Dict[str, float]:
        torch = _require_torch()
        if optimizer is None:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        joint = self.tokenizer.flatten_batch(batch)
        condition = self.condition_sampler.sample(joint.shape[0], self.tokenizer.total_variables, self.tokenizer.config.theta_dim)
        encoded = self.tokenizer.encode({"joint_values": joint}, condition)
        tensors = self._torch_encoded(encoded)

        self.model.train(True)
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = self._loss_fn(
            self.model,
            tensors["clean_values"],
            tensors["condition_mask"],
            tensors["variable_ids"],
            tensors["attention_mask"],
            tensors["value_features"],
        )
        loss.backward()
        if self.config.clip_max_norm and self.config.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(list(self.model.parameters()), self.config.clip_max_norm)
        optimizer.step()

        record = {
            "step": int(step),
            "timestamp": _now(),
            "loss": float(metrics["loss"]),
            "active_fraction": float(metrics["active_fraction"]),
            "mean_sigma": float(metrics["mean_sigma"]),
            "conditioning_pattern": self.condition_sampler.pattern,
            "mask_variant": self.config.mask_variant,
        }
        self.loss_trace.append(record)
        return {k: float(v) for k, v in metrics.items()}

    def train(
        self,
        data_iterable: Iterable[Mapping[str, Any]],
        max_steps: Optional[int] = None,
        write_artifacts: bool = True,
    ) -> Dict[str, Any]:
        torch = _require_torch()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        limit = self.config.max_epochs if max_steps is None else max_steps
        completed = 0
        last_metrics: Dict[str, float] = {"loss": math.inf, "active_fraction": 0.0, "mean_sigma": 0.0}

        for step, batch in enumerate(data_iterable):
            if step >= limit:
                break
            last_metrics = self.train_batch(batch, optimizer=optimizer, step=step)
            completed += 1

        metadata = self.training_metadata()
        metadata.update(
            {
                "completed_steps": completed,
                "last_metrics": last_metrics,
                "artifact_semantics": "training trace; values reflect executed local run only",
            }
        )
        if write_artifacts:
            self.write_training_artifacts(metadata)
        return metadata

    def training_metadata(self) -> Dict[str, Any]:
        payload = self.config.to_metadata()
        payload.update(
            {
                "model": self.model.registry_entry(),
                "tokenizer": self.tokenizer.registry_entry(),
                "diffusion": self.diffusion.to_dict(),
                "condition_sampler": self.condition_sampler.registry_entry(),
                "objective": "denoising score matching on joint p(theta,x) with loss mask (1-M_C)",
                "attention_mask_shape": list(self.attention_mask.shape),
                "attention_mask_density": float(self.attention_mask.mean()),
                "blacklisted_repository_used": False,
            }
        )
        return payload

    def write_training_artifacts(self, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, str]:
        meta = dict(metadata or self.training_metadata())
        paths = {
            "model_registry": _write_json("results/model_registry.json", {"schema": "model_registry/v1", **meta}),
            "tokenizer_registry": _write_json(
                "results/tokenizer_registry.json",
                {"schema": "tokenizer_registry/v1", "tokenizer": self.tokenizer.registry_entry()},
            ),
            "attention_mask_registry": _write_json(
                "results/attention_mask_registry.json",
                {
                    "schema": "attention_mask_registry/v1",
                    "mask_variant": self.config.mask_variant,
                    "M_E": self.attention_mask.astype(float).tolist(),
                    "semantics": "1 permits attention; 0 blocks attention",
                },
            ),
            "diffusion_config": _write_json(
                "results/diffusion_config.json",
                {"schema": "diffusion_config/v1", "diffusion": self.diffusion.to_dict()},
            ),
            "loss_trace": _write_json(
                "results/loss_trace.json",
                {
                    "schema": "loss_trace/v1",
                    "artifact_semantics": "local training trace; dry-run traces are labeled separately",
                    "metadata": meta,
                    "trace": list(self.loss_trace),
                },
            ),
        }
        return {key: str(value) for key, value in paths.items()}


class DiffusionSampler:
    """Conditional sampler with named SDE and ODE families.

    ``condition_mask`` is enforced at every step: observed variables are clamped to
    ``condition_values`` while unconditioned variables are updated by the learned
    score field.  The guided variant accepts a differentiable or tensor-compatible
    guidance function and applies it only to unconditioned entries.
    """

    def __init__(self, model: SimformerScoreModel, diffusion: DiffusionConfig, attention_mask: np.ndarray):
        self.model = model
        self.diffusion = diffusion
        self.attention_mask = np.asarray(attention_mask, dtype=np.float32)

    def sample(
        self,
        condition_values: ArrayLike,
        condition_mask: ArrayLike,
        num_samples: int,
        family: Optional[str] = None,
        variable_ids: Optional[ArrayLike] = None,
        seed: int = 0,
        guidance_fn: Optional[Callable[[Any], Any]] = None,
        guidance_scale: Optional[float] = None,
        trace_path: Union[str, Path] = "results/sampling_trace.json",
    ) -> np.ndarray:
        torch = _require_torch()
        selected_family = family or self.diffusion.sampler_family
        if selected_family not in SAMPLING_FAMILIES:
            raise ValueError(f"Unknown sampling family {selected_family}; expected {SAMPLING_FAMILIES}.")
        cond_values = _as_2d_float_array(condition_values, name="condition_values")
        cond_mask = _ensure_binary_mask(condition_mask, shape=cond_values.shape, name="condition_mask")
        if cond_values.shape[0] == 1 and num_samples > 1:
            cond_values = np.repeat(cond_values, num_samples, axis=0)
            cond_mask = np.repeat(cond_mask, num_samples, axis=0)
        if cond_values.shape[0] != num_samples:
            raise ValueError(f"condition_values batch size must be 1 or num_samples={num_samples}.")

        device = self.model.device
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        values = torch.randn((num_samples, cond_values.shape[1]), generator=generator, device=device) * self.diffusion.sigma_max
        cond_v = torch.as_tensor(cond_values, dtype=torch.float32, device=device)
        cond_m = torch.as_tensor(cond_mask, dtype=torch.float32, device=device)
        values = cond_m * cond_v + (1.0 - cond_m) * values
        ids = (
            torch.as_tensor(variable_ids, dtype=torch.long, device=device)
            if variable_ids is not None
            else torch.arange(cond_values.shape[1], device=device).view(1, -1).expand(num_samples, -1)
        )
        attn = torch.as_tensor(self.attention_mask, dtype=torch.float32, device=device)

        self.model.eval()
        steps = max(int(self.diffusion.num_steps), 1)
        sigmas = torch.logspace(
            math.log10(self.diffusion.sigma_max),
            math.log10(self.diffusion.sigma_min),
            steps,
            device=device,
        )
        trace: List[Dict[str, Any]] = []
        with torch.enable_grad():
            for index, sigma in enumerate(sigmas):
                t_scalar = torch.full((num_samples,), float(index + 1) / steps, dtype=torch.float32, device=device)
                values = values.detach().requires_grad_(guidance_fn is not None)
                score = self.model(values, t_scalar, variable_ids=ids, condition_mask=cond_m, attention_mask=attn)
                active = 1.0 - cond_m

                if guidance_fn is not None:
                    guide_value = guidance_fn(values)
                    if not torch.is_tensor(guide_value):
                        guide_value = torch.as_tensor(guide_value, dtype=torch.float32, device=device)
                    guide_scalar = guide_value.sum()
                    grad = torch.autograd.grad(guide_scalar, values, retain_graph=False, create_graph=False)[0]
                    scale = self.diffusion.guidance_scale if guidance_scale is None else guidance_scale
                    score = score + float(scale) * grad * active

                if selected_family == "sde_backward":
                    step_size = float(sigma / max(steps, 1))
                    stochastic = torch.randn_like(values) * math.sqrt(max(step_size, 1e-8))
                    values = values + active * (step_size * score + stochastic)
                else:
                    step_size = float(sigma / max(steps, 1))
                    values = values + active * (0.5 * step_size * score)

                values = cond_m * cond_v + active * values
                trace.append(
                    {
                        "step": int(index),
                        "family": selected_family,
                        "sigma": float(sigma.detach().cpu().item()),
                        "conditioned_fraction": float(cond_m.detach().mean().cpu().item()),
                    }
                )

        samples = values.detach().cpu().numpy().astype(np.float32)
        _write_json(
            trace_path,
            {
                "schema": "sampling_trace/v1",
                "artifact_semantics": "sampling trace for executed local sampler call",
                "family": selected_family,
                "num_samples": num_samples,
                "num_steps": steps,
                "constraint_satisfaction_rate": constraint_satisfaction_rate(samples, cond_values, cond_mask),
                "trace": trace,
            },
        )
        return samples

    def guided_sample(
        self,
        condition_values: ArrayLike,
        condition_mask: ArrayLike,
        num_samples: int,
        guidance_fn: Callable[[Any], Any],
        family: str = "sde_backward",
        guidance_scale: float = 1.0,
        seed: int = 0,
    ) -> np.ndarray:
        return self.sample(
            condition_values=condition_values,
            condition_mask=condition_mask,
            num_samples=num_samples,
            family=family,
            seed=seed,
            guidance_fn=guidance_fn,
            guidance_scale=guidance_scale,
        )


def build_default_core(
    theta_dim: int,
    x_dim: int,
    mask_variant: str = "simulator_dependency",
    conditioning_pattern: str = "random",
    hidden_dim: int = 32,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Factory exposing tokenizer, mask builder, score network, trainer pieces, and sampler config."""

    tokenizer = SBITokenizer(TokenizerConfig(theta_dim=theta_dim, x_dim=x_dim))
    spec = AttentionMaskSpec(
        variant=mask_variant,
        total_variables=theta_dim + x_dim,
        theta_dim=theta_dim,
        description="Default Simformer core dependency mask.",
    )
    attention = DependencyMaskBuilder(spec).build()
    diffusion = DiffusionConfig(num_steps=8, sampler_family="sde_backward")
    condition_sampler = ConditionMaskSampler(pattern=conditioning_pattern, condition_probability=0.7)
    model = SimformerScoreModel(
        total_variables=theta_dim + x_dim,
        value_dim=1,
        hidden_dim=hidden_dim,
        num_layers=1,
        num_heads=4 if hidden_dim % 4 == 0 else 1,
        device=device,
    )
    trainer_config = TrainerConfig(
        device=device,
        mask_variant=mask_variant,
        conditioning_pattern=conditioning_pattern,
        fixed_hyperparameters={"hidden_dim": hidden_dim, "num_layers": 1, "num_heads": 4 if hidden_dim % 4 == 0 else 1},
    )
    trainer = SimformerTrainer(tokenizer, model, diffusion, attention, condition_sampler, trainer_config)
    sampler = DiffusionSampler(model, diffusion, attention)
    return {
        "tokenizer": tokenizer,
        "mask_builder": DependencyMaskBuilder(spec),
        "attention_mask": attention,
        "score_network": model,
        "trainer": trainer,
        "sampler": sampler,
        "guided_sampler": sampler.guided_sample,
        "diffusion": diffusion,
        "condition_sampler": condition_sampler,
    }


def dry_run_artifact_payloads(theta_dim: int = 2, x_dim: int = 2) -> Dict[str, Dict[str, Any]]:
    """Create schema/readiness payloads for all artifacts owned by this file."""

    tokenizer = SBITokenizer(TokenizerConfig(theta_dim=theta_dim, x_dim=x_dim))
    mask_spec = AttentionMaskSpec(
        variant="simulator_dependency",
        total_variables=theta_dim + x_dim,
        theta_dim=theta_dim,
        description="Dry-run registry mask for Simformer dependency attention.",
    )
    mask_builder = DependencyMaskBuilder(mask_spec)
    attention = mask_builder.build()
    diffusion = DiffusionConfig(num_steps=4)
    condition_sampler = ConditionMaskSampler(pattern="five_family_uniform_mixture", condition_probability=0.7)
    trainer_config = TrainerConfig(
        simulation_budget=8,
        max_epochs=1,
        mask_variant=mask_spec.variant,
        conditioning_pattern=condition_sampler.pattern,
        fixed_hyperparameters={
            "sigma_min": diffusion.sigma_min,
            "sigma_max": diffusion.sigma_max,
            "num_steps": diffusion.num_steps,
            "bounded_smoke": True,
        },
    )

    return {
        "results/model_registry.json": {
            "schema": "model_registry/v1",
            "artifact_semantics": "dry-run contract artifact; not a trained model result",
            "created_at": _now(),
            "method": "simformer",
            "model": {
                "model_id": "simformer_score_transformer_v1",
                "class": "SimformerScoreModel",
                "total_variables": tokenizer.total_variables,
                "attention_contract": "M_E enters transformer attention as a blocking mask.",
                "conditioning_contract": "M_C enters token state, noising, loss, and sampling.",
            },
            "training_metadata": trainer_config.to_metadata(),
            "blacklisted_repository_used": False,
        },
        "results/tokenizer_registry.json": {
            "schema": "tokenizer_registry/v1",
            "artifact_semantics": "dry-run contract artifact; tokenizer is executable",
            "created_at": _now(),
            "tokenizer": tokenizer.registry_entry(),
            "embedding_artifact": {
                "concatenation_order": ["identifier", "value", "metadata", "condition_state"],
                "identifier": "duplicate variable names share ids",
                "value_embedding": "scalar value repeated across embedding_dim",
                "metadata_embedding": "Gaussian Fourier metadata/time features projected to embedding_dim",
                "condition_embedding": "shared true-condition vector; false condition is zero",
            },
        },
        "results/attention_mask_registry.json": {
            "schema": "attention_mask_registry/v1",
            "artifact_semantics": "dry-run contract artifact; mask is executable",
            "created_at": _now(),
            "mask_builder": mask_builder.registry_entry(),
            "M_E": attention.tolist(),
            "conditioned_graph_inversion": {
                "input": "directed M_E plus binary M_C",
                "semantics": "Webb-style graph inversion updates evidence directions before model attention",
            },
        },
        "results/diffusion_config.json": {
            "schema": "diffusion_config/v1",
            "artifact_semantics": "dry-run contract artifact; no paper-scale training claimed",
            "created_at": _now(),
            "diffusion": diffusion.to_dict(),
            "sampling_families": list(SAMPLING_FAMILIES),
            "objective": "masked denoising score matching on joint p(theta,x)",
        },
        "results/loss_trace.json": {
            "schema": "loss_trace/v1",
            "artifact_semantics": "dry-run schema artifact; contains analytic smoke metric only",
            "created_at": _now(),
            "trace": [
                {
                    "step": 0,
                    "loss": 0.0,
                    "active_fraction": 0.5,
                    "conditioning_pattern": condition_sampler.pattern,
                    "mask_variant": mask_spec.variant,
                    "semantics": "schema row, not a benchmark result",
                }
            ],
        },
        "results/sampling_trace.json": {
            "schema": "sampling_trace/v1",
            "artifact_semantics": "dry-run schema artifact; sampler families are registered but not benchmarked",
            "created_at": _now(),
            "families": list(SAMPLING_FAMILIES),
            "trace": [
                {
                    "step": 0,
                    "family": "sde_backward",
                    "conditioned_fraction": 0.5,
                    "semantics": "schema row, not a benchmark result",
                },
                {
                    "step": 0,
                    "family": "ode_probability_flow",
                    "conditioned_fraction": 0.5,
                    "semantics": "schema row, not a benchmark result",
                },
            ],
        },
    }


def write_dry_run_artifacts(paths: Sequence[str] = DEFAULT_ARTIFACT_PATHS) -> Dict[str, str]:
    """Materialize all declared encoding/core artifacts as dry-run readiness files."""

    payloads = dry_run_artifact_payloads()
    written: Dict[str, str] = {}
    for path in paths:
        payload = payloads.get(
            path,
            {
                "schema": "unknown_encoding_artifact/v1",
                "artifact_semantics": "dry-run contract artifact for an unrecognized requested path",
                "created_at": _now(),
                "requested_path": path,
            },
        )
        written[path] = str(_write_json(path, payload))

    readiness = {
        "schema": "readiness/v1",
        "artifact_semantics": "dry-run readiness artifact; no benchmark scores claimed",
        "created_at": _now(),
        "module": "all_in_one_sbi.encoding",
        "implemented_surfaces": [
            "model_or_method",
            "training_loop",
            "metric_formula",
            "tests",
            "policy_adapter",
            "config",
            "evaluation",
            "data_pipeline",
        ],
        "method_obligations": {
            "tokenizer": True,
            "mask_builder": True,
            "score_network": True,
            "trainer": True,
            "sampler": True,
            "guided_sampler": True,
            "M_E_enters_attention": True,
            "M_C_enters_noising_loss_sampling": True,
            "named_sde_ode_families": list(SAMPLING_FAMILIES),
            "blacklisted_repository_used": False,
        },
        "written_artifacts": written,
    }
    evaluation_result = {
        "schema": "evaluation_result/v1",
        "artifact_semantics": "dry-run evaluation contract artifact; not a paper result",
        "created_at": _now(),
        "status": "ready",
        "decisive_metric_schema": {
            "masked_score_mse": "mean squared score error on entries where M_C=0",
            "constraint_satisfaction_rate": "fraction of conditioned entries preserved during sampling",
        },
        "bounded_smoke_decision": "wiring validated without expensive training or external assets",
    }
    written["results/readiness.json"] = str(_write_json("results/readiness.json", readiness))
    written["results/evaluation_result.json"] = str(_write_json("results/evaluation_result.json", evaluation_result))
    return written


def smoke_encode_and_metric() -> Dict[str, Any]:
    """Small executable smoke path that exercises tokenizer, masks, noising, and metrics without torch."""

    tokenizer = SBITokenizer(TokenizerConfig(theta_dim=2, x_dim=2))
    batch = {
        "theta": np.asarray([[0.1, -0.2], [0.3, 0.4]], dtype=np.float32),
        "x": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    }
    condition = ConditionMaskSampler(pattern="posterior").sample(batch_size=2, total_variables=4, theta_dim=2)
    encoded = tokenizer.encode(batch, condition)
    attention = DependencyMaskBuilder(
        AttentionMaskSpec(variant="prior_to_observation", total_variables=4, theta_dim=2)
    ).build()
    t = np.asarray([0.2, 0.8], dtype=np.float32)
    noise = np.ones_like(encoded.joint_values, dtype=np.float32) * 0.1
    noised, target = forward_noise_numpy(encoded.joint_values, encoded.condition_state, t, noise, DiffusionConfig())
    metric = masked_score_mse_numpy(target, target, encoded.condition_state)
    satisfaction = constraint_satisfaction_rate(noised, encoded.joint_values, encoded.condition_state)
    return {
        "encoded_shape": list(encoded.value_representation.shape),
        "condition_state": encoded.condition_state.tolist(),
        "attention_mask": attention.tolist(),
        "masked_score_mse_self": metric,
        "constraint_satisfaction_rate": satisfaction,
        "joint_distribution": encoded.metadata["joint_distribution"],
    }


__all__ = [
    "ArrayLike",
    "SAMPLING_FAMILIES",
    "MASK_VARIANTS",
    "CONDITIONING_PATTERNS",
    "DEFAULT_ARTIFACT_PATHS",
    "TokenizerConfig",
    "EncodedBatch",
    "SBITokenizer",
    "ConditionPolicy",
    "ConditionMaskSampler",
    "AttentionMaskSpec",
    "DependencyMaskBuilder",
    "DiffusionConfig",
    "TrainerConfig",
    "SimformerScoreModel",
    "SimformerTrainer",
    "DiffusionSampler",
    "forward_noise_numpy",
    "masked_score_mse_numpy",
    "constraint_satisfaction_rate",
    "make_torch_score_loss",
    "build_default_core",
    "dry_run_artifact_payloads",
    "write_dry_run_artifacts",
    "smoke_encode_and_metric",
]
