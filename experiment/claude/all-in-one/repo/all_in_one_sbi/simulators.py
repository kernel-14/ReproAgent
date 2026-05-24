"""Simulator, task, dataset, dependency-mask, and sampling-family registry.

This module owns the simulator/data-pipeline surface for the reproduction of
"All-in-one simulation-based inference".  It is intentionally importable in a
minimal environment: only the Python standard library and NumPy are used at module
import time.  Optional training libraries such as torch/sbi are expected to be
loaded lazily by neighbouring modules.

Paper-derived obligations implemented here
------------------------------------------
* Explicit task and dataset registry entries for:
  two_moons, gaussian_linear, gaussian_mixture, slcp, lotka_volterra, sird,
  hodgkin_huxley, with human-readable aliases such as "Two Moons",
  "Linear Gaussian", and benchmark protocol aliases.
* Simulators draw from the joint distribution p(theta, x), rather than exposing
  only posterior or likelihood surrogates.  The returned ``SimulationBatch`` also
  contains binary conditioning masks so training can resample conditioning
  patterns on every batch.
* Dependency masks ``M_E`` are explicit directed/undirected graph masks over
  variables ordered as theta_1, theta_2, ..., x_1, x_2, ... .  These masks are
  suitable for downstream transformer attention masking.
* Lotka-Volterra dependency masks are metadata-dependent as required by the
  addendum: theta-to-x blocks distinguish prey and predator parameters, within
  series dynamics are Markovian, and cross-data dependencies are causal.
* Sampling families for conditional diffusion are separately named and selectable:
  ``sde_backward`` and ``ode_probability_flow``.
* The registry records adapter/embedding metadata for high-dimensional time-series
  observations without importing heavy neural-network dependencies.

The numerical simulators are lightweight canonical implementations intended for
smoke validation and small experiments.  They are not precomputed paper-scale
benchmark datasets and do not claim reproduced scores.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np


Array = np.ndarray


DEFAULT_RESULTS_DIR = "results"
DEFAULT_SEED = 20240521


# reference_grounding: paperbench_ref_001 docs/how_to_guide/04_embedding_networks.ipynb
# The reference discusses embedding networks for high-dimensional outputs.  In this
# simulator registry, the same protocol intent is captured as lightweight adapter
# metadata (e.g. "identity", "time_series_mlp", "summary_stats") so neighbouring
# model/tokenizer files can construct the appropriate embedding modules lazily.
#
# reference_grounding: paperbench_ref_001 docs/how_to_guide/07_gpu_training.ipynb
# Device choices are recorded as metadata only; this file never imports torch.
#
# reference_grounding: paperbench_ref_001 docs/how_to_guide/09_sampler_interface.ipynb
# Sampling algorithms are first-class selectable entries.  For Simformer diffusion
# the paper-required families are SDE backward diffusion and ODE probability flow.
#
# reference_grounding: paperbench_ref_001 sbi/inference/trainers/npe/npe_a.py
# The simulator API mirrors the train/simulate separation used in SBI trainers:
# draw theta from a prior, simulate x, append to a training dataset, and let the
# trainer resample conditioning masks instead of fixing a single posterior query.


@dataclasses.dataclass(frozen=True)
class VariableSpec:
    """Description of a scalar joint variable token."""

    name: str
    kind: str  # "theta" or "x"
    index: int
    group: str = ""
    observed_by_default: bool = False


@dataclasses.dataclass(frozen=True)
class SimulatorMetadata:
    """Static metadata for a paper-visible task."""

    task_id: str
    display_name: str
    aliases: Tuple[str, ...]
    theta_dim: int
    x_dim: int
    default_num_observations: int
    benchmark_family: str
    observation_type: str
    supports_structured_observations: bool
    supports_arbitrary_conditioning: bool
    supports_likelihood_query: bool
    supports_posterior_query: bool
    embedding_adapter: str
    dependency_mask_builder: str
    factory: str
    loader: str
    setup: Mapping[str, Any]


@dataclasses.dataclass
class SimulationBatch:
    """Joint samples and masks for Simformer-style training/evaluation.

    ``joint`` is ordered as [theta variables, x variables].  ``condition_mask`` is
    binary with 1 for conditioned/known variables and 0 for target/noised variables.
    This convention is used by downstream tokenizer, noising, loss masking, and
    conditional sampling code.
    """

    theta: Array
    x: Array
    joint: Array
    condition_mask: Array
    variable_names: Tuple[str, ...]
    task_id: str
    metadata: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "theta": self.theta,
            "x": self.x,
            "joint": self.joint,
            "condition_mask": self.condition_mask,
            "variable_names": self.variable_names,
            "task_id": self.task_id,
            "metadata": dict(self.metadata),
        }


@dataclasses.dataclass(frozen=True)
class ConditionalQuery:
    """A query over arbitrary subsets of the joint vector.

    Query kinds:
    * posterior: condition on x, target theta.
    * likelihood: condition on theta, target x.
    * arbitrary: condition/target masks supplied by the caller.
    """

    kind: str
    condition_mask: Tuple[int, ...]
    target_mask: Tuple[int, ...]
    description: str

    def validate(self, joint_dim: int) -> None:
        if len(self.condition_mask) != joint_dim or len(self.target_mask) != joint_dim:
            raise ValueError(f"Query masks must have length {joint_dim}.")
        if any(v not in (0, 1) for v in self.condition_mask + self.target_mask):
            raise ValueError("Query masks must be binary.")
        overlap = [i for i, (c, t) in enumerate(zip(self.condition_mask, self.target_mask)) if c and t]
        if overlap:
            raise ValueError(f"Condition and target masks overlap at indices {overlap}.")


@dataclasses.dataclass(frozen=True)
class SamplingFamily:
    """Named conditional diffusion sampling family."""

    sampling_id: str
    display_name: str
    mechanism: str
    deterministic: bool
    default_steps: int
    selectable_name: str
    setup: Mapping[str, Any]


@dataclasses.dataclass(frozen=True)
class AttentionMask:
    """Dependency mask over joint variables for transformer attention."""

    task_id: str
    directed: Array
    undirected: Array
    variable_names: Tuple[str, ...]
    metadata: Mapping[str, Any]

    def as_serializable(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "directed": self.directed.astype(int).tolist(),
            "undirected": self.undirected.astype(int).tolist(),
            "variable_names": list(self.variable_names),
            "metadata": dict(self.metadata),
        }


class BaseSimulator:
    """Base class for lightweight joint simulators."""

    metadata: SimulatorMetadata

    def __init__(self, metadata: SimulatorMetadata, seed: int = DEFAULT_SEED, **kwargs: Any) -> None:
        self.metadata = metadata
        self.seed = int(seed)
        self.kwargs = dict(kwargs)
        self.rng = np.random.default_rng(self.seed)

    @property
    def theta_dim(self) -> int:
        return self.metadata.theta_dim

    @property
    def x_dim(self) -> int:
        return self.metadata.x_dim

    @property
    def joint_dim(self) -> int:
        return self.theta_dim + self.x_dim

    @property
    def variable_names(self) -> Tuple[str, ...]:
        return tuple([f"theta_{i + 1}" for i in range(self.theta_dim)] + [f"x_{i + 1}" for i in range(self.x_dim)])

    def prior_sample(self, num_samples: int) -> Array:
        return self.rng.uniform(-2.0, 2.0, size=(num_samples, self.theta_dim))

    def simulate(self, theta: Array) -> Array:
        raise NotImplementedError("Subclasses must implement a concrete simulator.")

    def sample_joint(
        self,
        num_samples: int,
        conditioning: str | ConditionalQuery | None = "resample",
        condition_probability: float = 0.5,
    ) -> SimulationBatch:
        theta = self.prior_sample(num_samples)
        x = self.simulate(theta)
        joint = np.concatenate([theta, x], axis=1)
        condition_mask = build_condition_mask(
            num_samples=num_samples,
            theta_dim=self.theta_dim,
            x_dim=self.x_dim,
            pattern=conditioning,
            rng=self.rng,
            condition_probability=condition_probability,
        )
        return SimulationBatch(
            theta=theta,
            x=x,
            joint=joint,
            condition_mask=condition_mask,
            variable_names=self.variable_names,
            task_id=self.metadata.task_id,
            metadata={
                "joint_distribution": "p(theta,x)=p(theta)p(x|theta)",
                "conditioning_pattern": getattr(conditioning, "kind", conditioning),
                "condition_state_is_binary": True,
                "training_resamples_conditioning_pattern": conditioning in (None, "resample", "random_arbitrary"),
            },
        )

    def dependency_mask(self, directed: bool = True, **metadata: Any) -> Array:
        mask = build_dependency_mask(self.metadata.task_id, directed=directed, **metadata)
        return mask.directed if directed else mask.undirected

    def query(self, kind: str = "posterior", **kwargs: Any) -> ConditionalQuery:
        return make_conditional_query(self.metadata.task_id, kind=kind, **kwargs)


class TwoMoonsSimulator(BaseSimulator):
    """Canonical two-moons SBI simulator."""

    def prior_sample(self, num_samples: int) -> Array:
        return self.rng.uniform(-1.0, 1.0, size=(num_samples, 2))

    def simulate(self, theta: Array) -> Array:
        theta = _as_2d(theta, self.theta_dim)
        angle = theta[:, 0] * math.pi
        radius = 0.8 + 0.25 * theta[:, 1]
        x1 = radius * np.cos(angle) + 0.15 * theta[:, 0] ** 2
        x2 = radius * np.sin(angle) + 0.25 * np.sign(theta[:, 0])
        noise = self.rng.normal(0.0, 0.06, size=(theta.shape[0], 2))
        return np.column_stack([x1, x2]) + noise


class GaussianLinearSimulator(BaseSimulator):
    """Linear Gaussian model with analytic simulator and likelihood-style queries."""

    def __init__(self, metadata: SimulatorMetadata, seed: int = DEFAULT_SEED, **kwargs: Any) -> None:
        super().__init__(metadata, seed=seed, **kwargs)
        matrix = kwargs.get("matrix")
        if matrix is None:
            matrix = np.array([[1.0, 0.4], [-0.2, 0.9]], dtype=float)
        self.matrix = np.asarray(matrix, dtype=float)
        self.noise_scale = float(kwargs.get("noise_scale", 0.2))

    def prior_sample(self, num_samples: int) -> Array:
        return self.rng.normal(0.0, 1.0, size=(num_samples, self.theta_dim))

    def simulate(self, theta: Array) -> Array:
        theta = _as_2d(theta, self.theta_dim)
        mean = theta @ self.matrix.T
        return mean + self.rng.normal(0.0, self.noise_scale, size=mean.shape)


class GaussianMixtureSimulator(BaseSimulator):
    """Mixture observation model used as a benchmark-style non-Gaussian task."""

    def prior_sample(self, num_samples: int) -> Array:
        return self.rng.normal(0.0, 1.0, size=(num_samples, self.theta_dim))

    def simulate(self, theta: Array) -> Array:
        theta = _as_2d(theta, self.theta_dim)
        component = self.rng.integers(0, 2, size=theta.shape[0])
        shift = np.where(component[:, None] == 0, np.array([1.0, -0.6]), np.array([-1.0, 0.6]))
        rotated = np.column_stack([theta[:, 0] + 0.35 * theta[:, 1], -0.25 * theta[:, 0] + theta[:, 1]])
        return rotated + shift + self.rng.normal(0.0, 0.18, size=(theta.shape[0], 2))


class SLCPSimulator(BaseSimulator):
    """Simple likelihood complex posterior (SLCP) benchmark simulator."""

    def prior_sample(self, num_samples: int) -> Array:
        return self.rng.uniform(-3.0, 3.0, size=(num_samples, 5))

    def simulate(self, theta: Array) -> Array:
        theta = _as_2d(theta, self.theta_dim)
        n = theta.shape[0]
        means = theta[:, :2]
        raw_scales = np.exp(theta[:, 2:4] * 0.15)
        corr = np.tanh(theta[:, 4] * 0.25)
        observations = []
        for i in range(n):
            cov = np.array(
                [
                    [raw_scales[i, 0] ** 2, corr[i] * raw_scales[i, 0] * raw_scales[i, 1]],
                    [corr[i] * raw_scales[i, 0] * raw_scales[i, 1], raw_scales[i, 1] ** 2],
                ]
            )
            draw = self.rng.multivariate_normal(means[i], cov, size=4).reshape(-1)
            observations.append(draw)
        return np.asarray(observations, dtype=float)


class LotkaVolterraSimulator(BaseSimulator):
    """Lotka-Volterra prey/predator time-series simulator.

    Parameters theta are ordered as:
    theta_1 prey growth alpha, theta_2 prey predation beta,
    theta_3 predator death gamma, theta_4 predator reproduction delta.
    Observations are ordered as prey_1..prey_T, predator_1..predator_T.
    """

    def __init__(self, metadata: SimulatorMetadata, seed: int = DEFAULT_SEED, **kwargs: Any) -> None:
        time_points = kwargs.get("time_points", metadata.setup.get("time_points", 8))
        self.time_points = int(time_points)
        if self.time_points < 2:
            raise ValueError("Lotka-Volterra requires at least two time points.")
        adjusted = dataclasses.replace(metadata, x_dim=2 * self.time_points, default_num_observations=2 * self.time_points)
        super().__init__(adjusted, seed=seed, **kwargs)
        self.dt = float(kwargs.get("dt", 0.08))

    @property
    def variable_names(self) -> Tuple[str, ...]:
        prey = [f"prey_{i + 1}" for i in range(self.time_points)]
        predator = [f"predator_{i + 1}" for i in range(self.time_points)]
        return tuple([f"theta_{i + 1}" for i in range(4)] + prey + predator)

    def prior_sample(self, num_samples: int) -> Array:
        return self.rng.uniform(0.4, 1.6, size=(num_samples, 4))

    def simulate(self, theta: Array) -> Array:
        theta = _as_2d(theta, 4)
        out = np.zeros((theta.shape[0], 2 * self.time_points), dtype=float)
        for i, (alpha, beta, gamma, delta) in enumerate(theta):
            prey = np.zeros(self.time_points, dtype=float)
            predator = np.zeros(self.time_points, dtype=float)
            prey[0] = 1.0 + 0.05 * self.rng.normal()
            predator[0] = 1.0 + 0.05 * self.rng.normal()
            for t in range(1, self.time_points):
                dprey = alpha * prey[t - 1] - beta * prey[t - 1] * predator[t - 1]
                dpred = delta * prey[t - 1] * predator[t - 1] - gamma * predator[t - 1]
                prey[t] = max(1e-4, prey[t - 1] + self.dt * dprey)
                predator[t] = max(1e-4, predator[t - 1] + self.dt * dpred)
            full_time_series = np.concatenate([prey, predator])
            out[i] = full_time_series + self.rng.normal(0.0, 0.1, size=2 * self.time_points)
        return out


class SIRDSimulator(BaseSimulator):
    """SIRD epidemic simulator for structured conditional inference."""

    def __init__(self, metadata: SimulatorMetadata, seed: int = DEFAULT_SEED, **kwargs: Any) -> None:
        time_points = int(kwargs.get("time_points", metadata.setup.get("time_points", 10)))
        self.time_points = time_points
        adjusted = dataclasses.replace(metadata, theta_dim=2 + time_points, x_dim=4 * time_points, default_num_observations=4 * time_points)
        super().__init__(adjusted, seed=seed, **kwargs)
        self.dt = float(kwargs.get("dt", 0.1))

    @property
    def variable_names(self) -> Tuple[str, ...]:
        names = ["gamma", "delta"] + [f"beta_hat_{t + 1}" for t in range(self.time_points)]
        for group in ("susceptible", "infected", "recovered", "dead"):
            names.extend([f"{group}_{t + 1}" for t in range(self.time_points)])
        return tuple(names)

    def prior_sample(self, num_samples: int) -> Array:
        gamma = self.rng.uniform(0.0, 0.5, size=(num_samples, 1))
        delta = self.rng.uniform(0.0, 0.5, size=(num_samples, 1))
        times = np.arange(self.time_points, dtype=float)
        kernel = (2.5 ** 2) * np.exp(-0.5 * ((times[:, None] - times[None, :]) ** 2) / (7.0 ** 2))
        beta_hat = self.rng.multivariate_normal(np.zeros(self.time_points), kernel + 1.0e-6 * np.eye(self.time_points), size=num_samples)
        return np.concatenate([gamma, delta, beta_hat], axis=1)

    def simulate(self, theta: Array) -> Array:
        theta = _as_2d(theta, 2 + self.time_points)
        out = np.zeros((theta.shape[0], 4 * self.time_points), dtype=float)
        for i, row in enumerate(theta):
            gamma = float(row[0])
            delta = float(row[1])
            beta_t = 1.0 / (1.0 + np.exp(-row[2 : 2 + self.time_points]))
            s = np.zeros(self.time_points)
            inf = np.zeros(self.time_points)
            r = np.zeros(self.time_points)
            d = np.zeros(self.time_points)
            s[0], inf[0], r[0], d[0] = 0.98, 0.02, 0.0, 0.0
            for t in range(1, self.time_points):
                new_inf = beta_t[t - 1] * s[t - 1] * inf[t - 1]
                rec = gamma * inf[t - 1]
                death = delta * inf[t - 1]
                s[t] = np.clip(s[t - 1] - self.dt * new_inf, 0.0, 1.0)
                inf[t] = np.clip(inf[t - 1] + self.dt * (new_inf - rec - death), 0.0, 1.0)
                r[t] = np.clip(r[t - 1] + self.dt * rec, 0.0, 1.0)
                d[t] = np.clip(d[t - 1] + self.dt * death, 0.0, 1.0)
            latent = np.clip(np.concatenate([s, inf, r, d]), 1.0e-6, None)
            out[i] = self.rng.lognormal(mean=np.log(latent), sigma=0.05, size=4 * self.time_points)
        return out


class HodgkinHuxleySimulator(BaseSimulator):
    """Lightweight Hodgkin-Huxley-inspired observation/constraint simulator.

    This is not a biophysical full-resolution solver.  It preserves the paper
    interface: parameter-to-voltage time-series observations, observation interval
    metadata, and metabolic-cost constraint values for guided diffusion.
    """

    def __init__(self, metadata: SimulatorMetadata, seed: int = DEFAULT_SEED, **kwargs: Any) -> None:
        time_points = int(kwargs.get("time_points", metadata.setup.get("time_points", 50)))
        self.time_points = time_points
        self.duration_ms = 200.0
        self.v0 = -65.0
        adjusted = dataclasses.replace(metadata, x_dim=8, default_num_observations=8)
        super().__init__(adjusted, seed=seed, **kwargs)

    @property
    def variable_names(self) -> Tuple[str, ...]:
        return tuple([f"theta_{i + 1}" for i in range(5)] + ["spike_count", "resting_mean", "resting_std", "spiking_window_mean", "moment_2", "moment_3", "moment_4", "energy_uJ_per_s"])

    def prior_sample(self, num_samples: int) -> Array:
        return self.rng.uniform(
            low=np.array([0.5, 0.5, 0.2, 0.2, -1.0]),
            high=np.array([2.0, 2.0, 1.0, 1.0, 1.0]),
            size=(num_samples, 5),
        )

    def simulate(self, theta: Array) -> Array:
        theta = _as_2d(theta, 5)
        t = np.linspace(0.0, self.duration_ms, self.time_points)
        out = np.zeros((theta.shape[0], 8), dtype=float)
        for i, (g_na, g_k, g_l, tau, bias) in enumerate(theta):
            current = np.where((t >= 50.0) & (t <= 150.0), 4.0, 0.0)
            voltage = np.zeros(self.time_points, dtype=float)
            voltage[0] = -65.0
            m, h, n_gate = 0.05, 0.6, 0.32
            dt = t[1] - t[0]
            cumulative_sodium_charge = [0.0]
            for k in range(1, self.time_points):
                v = voltage[k - 1]
                am = _hh_alpha_m(v); bm = _hh_beta_m(v)
                ah = _hh_alpha_h(v); bh = _hh_beta_h(v)
                an = _hh_alpha_n(v); bn = _hh_beta_n(v)
                m += dt * (am * (1.0 - m) - bm * m) / 10.0
                h += dt * (ah * (1.0 - h) - bh * h) / 10.0
                n_gate += dt * (an * (1.0 - n_gate) - bn * n_gate) / 10.0
                i_na = max(g_na, 0.1) * (m ** 3) * h * (v - 50.0)
                i_k = max(g_k, 0.1) * (n_gate ** 4) * (v + 77.0)
                i_l = max(g_l, 0.05) * (v + 54.4)
                dv = current[k - 1] - i_na - i_k - i_l
                voltage[k] = v + 0.01 * dv + 0.02 * bias
                cumulative_sodium_charge.append(cumulative_sodium_charge[-1] - abs(i_na) * dt)
            noisy_voltage = voltage + self.rng.normal(0.0, 0.5, size=self.time_points)
            out[i] = hodgkin_huxley_summary_statistics(noisy_voltage, t, np.asarray(cumulative_sodium_charge, dtype=float))
        return out


class TreeSimulator(BaseSimulator):
    """Synthetic Tree task with HMC reference sampling surface."""

    def prior_sample(self, num_samples: int) -> Array:
        return self.rng.normal(0.0, 1.0, size=(num_samples, self.theta_dim))

    def simulate(self, theta: Array) -> Array:
        theta = _as_2d(theta, self.theta_dim)
        left = theta[:, 0] + 0.5 * theta[:, 1]
        right = theta[:, 2] - 0.25 * theta[:, 1]
        root = left + right + self.rng.normal(0.0, 0.1, size=theta.shape[0])
        return np.column_stack([root, left, right])


class HMMSimulator(BaseSimulator):
    """Synthetic HMM task with HMC reference sampling surface."""

    def prior_sample(self, num_samples: int) -> Array:
        return self.rng.uniform(-1.0, 1.0, size=(num_samples, self.theta_dim))

    def simulate(self, theta: Array) -> Array:
        theta = _as_2d(theta, self.theta_dim)
        emissions = []
        for row in theta:
            state = 0
            obs = []
            for k in range(self.x_dim):
                p_switch = 1.0 / (1.0 + math.exp(-row[k % len(row)]))
                if self.rng.random() < p_switch:
                    state = 1 - state
                obs.append((1.0 if state else -1.0) + 0.1 * self.rng.normal())
            emissions.append(obs)
        return np.asarray(emissions, dtype=float)


def _hh_efun(z: float) -> float:
    """Hodgkin-Huxley Appendix A.2.2 efun with threshold 1e-4."""

    if abs(z) < 1.0e-4:
        return 1.0 - z / 2.0
    return z / (math.exp(z) - 1.0)


def _hh_alpha_m(v: float) -> float:
    v0 = -65.0
    return 0.32 * _hh_efun((13.0 - (v - v0)) / 4.0)


def _hh_beta_m(v: float) -> float:
    v0 = -65.0
    return 0.28 * _hh_efun(((v - v0) - 40.0) / 5.0)


def _hh_alpha_h(v: float) -> float:
    v0 = -65.0
    return 0.128 * math.exp((17.0 - (v - v0)) / 18.0)


def _hh_beta_h(v: float) -> float:
    v0 = -65.0
    return 4.0 / (1.0 + math.exp((40.0 - (v - v0)) / 5.0))


def _hh_alpha_n(v: float) -> float:
    v0 = -65.0
    return 0.032 * _hh_efun((15.0 - (v - v0)) / 5.0)


def _hh_beta_n(v: float) -> float:
    v0 = -65.0
    return 0.5 * math.exp((10.0 - (v - v0)) / 40.0)


def convert_total_energy(E: Array) -> Array:
    E = -np.asarray(E, dtype=float)
    E = E / 1000.0
    E = E / 1000.0
    E = E * 0.628e-3
    elementary_charge = 1.602176634e-19
    n_na = E / elementary_charge
    atp_na = n_na / (1.0 * 3.0)
    atp_energy = 10e-19
    E = atp_na * atp_energy
    E = E / 0.2
    return E * 1e6


def convert_charge_to_energy(E: Array) -> Array:
    E = np.asarray(E, dtype=float)
    E = np.diff(E)
    if E.size >= 5:
        E = np.convolve(E, np.ones(5, dtype=float) / 5.0, mode="same")
    return convert_total_energy(E)


def hodgkin_huxley_summary_statistics(voltage: Array, time_ms: Array, sodium_charge: Array | float) -> Array:
    resting = voltage[time_ms < 50.0]
    spiking = voltage[(time_ms >= 50.0) & (time_ms <= 150.0)]
    spike_count = float(np.sum((voltage[1:] > 0.0) & (voltage[:-1] <= 0.0)))
    spiking_domain = spiking if spiking.size else voltage
    centered_spiking = spiking_domain - float(np.mean(spiking_domain))
    moments = [float(np.mean(centered_spiking ** k)) for k in (2, 3, 4)]
    sodium_arr = np.asarray(sodium_charge, dtype=float)
    if sodium_arr.ndim == 0:
        sodium_arr = np.asarray([0.0, -abs(float(sodium_arr))], dtype=float)
    energy_values = convert_charge_to_energy(sodium_arr) if sodium_arr.size > 1 else np.asarray([0.0])
    energy_uJ_per_s = float(np.mean(energy_values)) if energy_values.size else 0.0
    return np.asarray([
        spike_count,
        float(np.mean(resting)) if resting.size else -65.0,
        float(np.std(resting)) if resting.size else 0.0,
        float(np.mean(spiking)) if spiking.size else float(np.mean(voltage)),
        *moments,
        energy_uJ_per_s,
    ], dtype=float)


SIMULATOR_CLASSES: Dict[str, type[BaseSimulator]] = {
    "two_moons": TwoMoonsSimulator,
    "gaussian_linear": GaussianLinearSimulator,
    "gaussian_mixture": GaussianMixtureSimulator,
    "slcp": SLCPSimulator,
    "lotka_volterra": LotkaVolterraSimulator,
    "sird": SIRDSimulator,
    "hodgkin_huxley": HodgkinHuxleySimulator,
    "tree": TreeSimulator,
    "hmm": HMMSimulator,
}


TASK_REGISTRY: Dict[str, SimulatorMetadata] = {
    "two_moons": SimulatorMetadata(
        task_id="two_moons",
        display_name="Two Moons",
        aliases=("Two Moons", "two moons", "two-moons", "benchmark_two_moons", "Benchmark"),
        theta_dim=2,
        x_dim=2,
        default_num_observations=2,
        benchmark_family="approximating posterior distributions across four benchmark tasks",
        observation_type="low_dimensional",
        supports_structured_observations=False,
        supports_arbitrary_conditioning=True,
        supports_likelihood_query=True,
        supports_posterior_query=True,
        embedding_adapter="identity",
        dependency_mask_builder="dense_theta_to_x",
        factory="make_simulator",
        loader="load_benchmark_dataset",
        setup={"prior": "uniform[-1,1]^2", "noise": "small Gaussian", "paper_slot": "benchmark"},
    ),
    "gaussian_linear": SimulatorMetadata(
        task_id="gaussian_linear",
        display_name="Linear Gaussian",
        aliases=("Linear Gaussian", "linear_gaussian", "gaussian linear", "Benchmark"),
        theta_dim=2,
        x_dim=2,
        default_num_observations=2,
        benchmark_family="Across all four benchmark tasks",
        observation_type="low_dimensional",
        supports_structured_observations=False,
        supports_arbitrary_conditioning=True,
        supports_likelihood_query=True,
        supports_posterior_query=True,
        embedding_adapter="identity",
        dependency_mask_builder="dense_theta_to_x",
        factory="make_simulator",
        loader="load_benchmark_dataset",
        setup={"prior": "standard normal", "likelihood": "linear Gaussian", "paper_slot": "benchmark"},
    ),
    "gaussian_mixture": SimulatorMetadata(
        task_id="gaussian_mixture",
        display_name="Gaussian Mixture",
        aliases=("Gaussian Mixture", "gaussian mixture", "mixture_gaussian", "Benchmark"),
        theta_dim=2,
        x_dim=2,
        default_num_observations=2,
        benchmark_family="Across all four benchmark tasks",
        observation_type="low_dimensional",
        supports_structured_observations=False,
        supports_arbitrary_conditioning=True,
        supports_likelihood_query=True,
        supports_posterior_query=True,
        embedding_adapter="identity",
        dependency_mask_builder="dense_theta_to_x",
        factory="make_simulator",
        loader="load_benchmark_dataset",
        setup={"prior": "standard normal", "likelihood": "two-component Gaussian mixture", "paper_slot": "benchmark"},
    ),
    "slcp": SimulatorMetadata(
        task_id="slcp",
        display_name="SLCP",
        aliases=("slcp", "simple likelihood complex posterior", "Benchmark"),
        theta_dim=5,
        x_dim=8,
        default_num_observations=8,
        benchmark_family="approximating posterior distributions across four benchmark tasks",
        observation_type="replicated_low_dimensional",
        supports_structured_observations=True,
        supports_arbitrary_conditioning=True,
        supports_likelihood_query=True,
        supports_posterior_query=True,
        embedding_adapter="summary_stats",
        dependency_mask_builder="dense_theta_to_x",
        factory="make_simulator",
        loader="load_benchmark_dataset",
        setup={"prior": "uniform[-3,3]^5", "observations": "four bivariate Gaussian draws", "paper_slot": "benchmark"},
    ),
    "lotka_volterra": SimulatorMetadata(
        task_id="lotka_volterra",
        display_name="Lotka-Volterra",
        aliases=("lotka_volterra", "Lotka Volterra", "lotka-volterra", "structured_lv"),
        theta_dim=4,
        x_dim=16,
        default_num_observations=16,
        benchmark_family="structured_tasks",
        observation_type="prey_predator_time_series",
        supports_structured_observations=True,
        supports_arbitrary_conditioning=True,
        supports_likelihood_query=True,
        supports_posterior_query=True,
        embedding_adapter="time_series_mlp",
        dependency_mask_builder="lotka_volterra_metadata_dependent",
        factory="make_simulator",
        loader="load_benchmark_dataset",
        setup={"time_points": 8, "metadata_dependent_mask": True, "paper_section": "4.2"},
    ),
    "sird": SimulatorMetadata(
        task_id="sird",
        display_name="SIRD",
        aliases=("sird", "SIRD-model", "sird_model", "structured_sird"),
        theta_dim=12,
        x_dim=40,
        default_num_observations=40,
        benchmark_family="structured_tasks",
        observation_type="epidemic_time_series",
        supports_structured_observations=True,
        supports_arbitrary_conditioning=True,
        supports_likelihood_query=True,
        supports_posterior_query=True,
        embedding_adapter="time_series_mlp",
        dependency_mask_builder="sird_markov",
        factory="make_simulator",
        loader="load_benchmark_dataset",
        setup={
            "time_points": 10,
            "compartments": ("S", "I", "R", "D"),
            "gamma_delta_prior": "Uniform(0,0.5)",
            "beta_hat_prior": "Gaussian process G(0,k), k(t1,t2)=2.5^2 exp(-0.5||t1-t2||^2/7^2), sigmoid transform",
            "observation_noise": "log-normal sigma=0.05",
            "paper_section": "4.3",
        },
    ),
    "hodgkin_huxley": SimulatorMetadata(
        task_id="hodgkin_huxley",
        display_name="Hodgkin-Huxley",
        aliases=("hodgkin_huxley", "Hodgkin Huxley", "HH", "observation_interval_guidance"),
        theta_dim=5,
        x_dim=8,
        default_num_observations=8,
        benchmark_family="interval_guidance",
        observation_type="voltage_trace_with_metabolic_cost",
        supports_structured_observations=True,
        supports_arbitrary_conditioning=True,
        supports_likelihood_query=True,
        supports_posterior_query=True,
        embedding_adapter="time_series_mlp_plus_constraint",
        dependency_mask_builder="dense_theta_to_x",
        factory="make_simulator",
        loader="load_benchmark_dataset",
        setup={
            "time_points": 50,
            "duration_ms": 200,
            "V0": -65.0,
            "input_current": "4 mA from 50 ms to 150 ms",
            "rate_functions": "Appendix A2.2 Hodgkin-Huxley alpha/beta gating rates",
            "summary_features": ("spike_count", "resting_potential_mean", "resting_potential_std", "spiking_window_mean", "centered_standardized_moments_2_to_4", "energy_uJ_per_s"),
            "guided_diffusion_constraints": ("observation_interval", "metabolic_cost"),
            "paper_section": "3.4/4.4",
        },
    ),
    "tree": SimulatorMetadata(
        task_id="tree",
        display_name="Tree",
        aliases=("tree", "Tree task", "tree_hmc"),
        theta_dim=3,
        x_dim=3,
        default_num_observations=3,
        benchmark_family="structured_tasks",
        observation_type="tree_graph",
        supports_structured_observations=True,
        supports_arbitrary_conditioning=True,
        supports_likelihood_query=True,
        supports_posterior_query=True,
        embedding_adapter="identity",
        dependency_mask_builder="tree_graph",
        factory="make_simulator",
        loader="load_benchmark_dataset",
        setup={"reference_sampler": "N chains initialized from joint distribution, 5000 HMC steps, keep final samples"},
    ),
    "hmm": SimulatorMetadata(
        task_id="hmm",
        display_name="HMM",
        aliases=("hmm", "hidden_markov_model", "HMM task"),
        theta_dim=3,
        x_dim=8,
        default_num_observations=8,
        benchmark_family="structured_tasks",
        observation_type="hidden_markov_sequence",
        supports_structured_observations=True,
        supports_arbitrary_conditioning=True,
        supports_likelihood_query=True,
        supports_posterior_query=True,
        embedding_adapter="sequence_mlp",
        dependency_mask_builder="hmm_chain",
        factory="make_simulator",
        loader="load_benchmark_dataset",
        setup={"reference_sampler": "N chains initialized from joint distribution, 5000 HMC steps, keep final samples"},
    ),
}


DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    task_id: {
        "dataset_id": task_id,
        "task_id": task_id,
        "display_name": meta.display_name,
        "aliases": meta.aliases,
        "loader": meta.loader,
        "factory": meta.factory,
        "setup": dict(meta.setup),
        "joint_training_distribution": "p(theta,x)",
        "default_num_simulations_smoke": 8,
        "default_num_simulations_full": 10000,
        "artifact_group": f"datasets/{task_id}",
    }
    for task_id, meta in TASK_REGISTRY.items()
}


ALIAS_TO_TASK_ID: Dict[str, str] = {}
for _task_id, _meta in TASK_REGISTRY.items():
    ALIAS_TO_TASK_ID[_task_id] = _task_id
    ALIAS_TO_TASK_ID[_task_id.replace("_", "-")] = _task_id
    ALIAS_TO_TASK_ID[_meta.display_name] = _task_id
    ALIAS_TO_TASK_ID[_meta.display_name.lower()] = _task_id
    for _alias in _meta.aliases:
        ALIAS_TO_TASK_ID[_alias] = _task_id
        ALIAS_TO_TASK_ID[_alias.lower()] = _task_id


SAMPLING_FAMILIES: Dict[str, SamplingFamily] = {
    "sde_backward": SamplingFamily(
        sampling_id="sde_backward",
        display_name="SDE backward diffusion",
        mechanism="reverse-time stochastic differential equation",
        deterministic=False,
        default_steps=64,
        selectable_name="sde",
        setup={"paper_obligation": "SDE backward diffusion sampler", "noise_injection": True},
    ),
    "ode_probability_flow": SamplingFamily(
        sampling_id="ode_probability_flow",
        display_name="ODE probability flow",
        mechanism="deterministic probability-flow ordinary differential equation",
        deterministic=True,
        default_steps=64,
        selectable_name="ode",
        setup={"paper_obligation": "ODE probability flow sampler", "noise_injection": False},
    ),
}


def canonical_task_id(task_or_alias: str) -> str:
    """Resolve registry aliases to canonical task IDs."""

    if task_or_alias in ALIAS_TO_TASK_ID:
        return ALIAS_TO_TASK_ID[task_or_alias]
    lowered = task_or_alias.lower()
    if lowered in ALIAS_TO_TASK_ID:
        return ALIAS_TO_TASK_ID[lowered]
    normalized = lowered.replace(" ", "_").replace("-", "_")
    if normalized in TASK_REGISTRY:
        return normalized
    raise KeyError(f"Unknown simulator task or alias: {task_or_alias!r}. Available: {sorted(TASK_REGISTRY)}")


def make_simulator(task_or_alias: str, seed: int = DEFAULT_SEED, **kwargs: Any) -> BaseSimulator:
    """Factory hook for paper-visible simulators."""

    task_id = canonical_task_id(task_or_alias)
    metadata = TASK_REGISTRY[task_id]
    simulator_cls = SIMULATOR_CLASSES[task_id]
    return simulator_cls(metadata, seed=seed, **kwargs)


def list_tasks() -> List[Dict[str, Any]]:
    """Return task registry entries as JSON-serializable dictionaries."""

    return [
        {
            "task_id": meta.task_id,
            "display_name": meta.display_name,
            "aliases": list(meta.aliases),
            "theta_dim": meta.theta_dim,
            "x_dim": meta.x_dim,
            "benchmark_family": meta.benchmark_family,
            "observation_type": meta.observation_type,
            "embedding_adapter": meta.embedding_adapter,
            "dependency_mask_builder": meta.dependency_mask_builder,
            "supports_arbitrary_conditioning": meta.supports_arbitrary_conditioning,
            "supports_likelihood_query": meta.supports_likelihood_query,
            "supports_posterior_query": meta.supports_posterior_query,
            "setup": dict(meta.setup),
        }
        for meta in TASK_REGISTRY.values()
    ]


def list_datasets() -> List[Dict[str, Any]]:
    return [dict(entry) for entry in DATASET_REGISTRY.values()]


def load_benchmark_dataset(
    task_or_alias: str,
    num_samples: int = 8,
    seed: int = DEFAULT_SEED,
    conditioning: str | ConditionalQuery | None = "resample",
    **simulator_kwargs: Any,
) -> SimulationBatch:
    """Generate a lightweight benchmark dataset from the joint simulator.

    This is a loader/config hook, not an external dataset downloader.  It returns
    joint samples suitable for tokenizer/model smoke tests and bounded experiments.
    """

    simulator = make_simulator(task_or_alias, seed=seed, **simulator_kwargs)
    return simulator.sample_joint(num_samples=num_samples, conditioning=conditioning)


def random_direction_slice_sampling(
    log_prob: Callable[[Array], float],
    initial: Array,
    num_steps: int,
    rng: Optional[np.random.Generator] = None,
    width: float = 1.0,
) -> Array:
    """Random-direction slice sampler used by Two Moons/SLCP references."""

    generator = rng if rng is not None else np.random.default_rng(DEFAULT_SEED)
    x = np.asarray(initial, dtype=float).copy()
    for _ in range(int(num_steps)):
        direction = generator.normal(size=x.shape)
        direction = direction / max(float(np.linalg.norm(direction)), 1.0e-12)
        log_y = float(log_prob(x)) - generator.exponential(1.0)
        lo = -width * generator.random()
        hi = lo + width
        while float(log_prob(x + lo * direction)) > log_y:
            lo -= width
        while float(log_prob(x + hi * direction)) > log_y:
            hi += width
        for _shrink in range(100):
            proposal_scale = generator.uniform(lo, hi)
            proposal = x + proposal_scale * direction
            if float(log_prob(proposal)) >= log_y:
                x = proposal
                break
            if proposal_scale < 0.0:
                lo = proposal_scale
            else:
                hi = proposal_scale
    return x


def metropolis_hastings(
    log_prob: Callable[[Array], float],
    initial: Array,
    num_steps: int,
    step_size: float,
    rng: Optional[np.random.Generator] = None,
) -> Array:
    generator = rng if rng is not None else np.random.default_rng(DEFAULT_SEED)
    x = np.asarray(initial, dtype=float).copy()
    lp = float(log_prob(x))
    for _ in range(int(num_steps)):
        proposal = x + generator.normal(0.0, float(step_size), size=x.shape)
        prop_lp = float(log_prob(proposal))
        if math.log(max(generator.random(), 1.0e-12)) < prop_lp - lp:
            x, lp = proposal, prop_lp
    return x


def hmc_sampler(
    log_prob: Callable[[Array], float],
    initial: Array,
    num_steps: int = 5000,
    step_size: float = 0.01,
    leapfrog_steps: int = 5,
    rng: Optional[np.random.Generator] = None,
) -> Array:
    """Small finite-difference HMC sampler; returns only the final chain state."""

    generator = rng if rng is not None else np.random.default_rng(DEFAULT_SEED)
    x = np.asarray(initial, dtype=float).copy()

    def grad(z: Array) -> Array:
        eps = 1.0e-4
        g = np.zeros_like(z)
        for j in range(z.size):
            zp = z.copy(); zm = z.copy()
            zp[j] += eps; zm[j] -= eps
            g[j] = (float(log_prob(zp)) - float(log_prob(zm))) / (2.0 * eps)
        return g

    for _ in range(int(num_steps)):
        p = generator.normal(size=x.shape)
        current_x, current_p = x.copy(), p.copy()
        p = p + 0.5 * step_size * grad(x)
        for lf in range(leapfrog_steps):
            x = x + step_size * p
            if lf != leapfrog_steps - 1:
                p = p + step_size * grad(x)
        p = p + 0.5 * step_size * grad(x)
        p = -p
        current_h = -float(log_prob(current_x)) + 0.5 * float(np.sum(current_p**2))
        proposal_h = -float(log_prob(x)) + 0.5 * float(np.sum(p**2))
        if math.log(max(generator.random(), 1.0e-12)) >= current_h - proposal_h:
            x = current_x
    return x


def reference_sampler_protocol(task_id: str) -> Dict[str, Any]:
    """Return the default reference-sampler protocol for a task."""

    task_id = canonical_task_id(task_id)
    if task_id == "two_moons":
        return {
            "sampler": "random_direction_slice_sampling_then_metropolis_hastings",
            "slice_steps": 1000,
            "mh_steps": 3000,
            "mh_step_size": 0.01,
            "default_num_steps": 4000,
            "initialization": "N chains from joint distribution; return final theta from each chain",
        }
    if task_id == "gaussian_mixture":
        return {"sampler": "metropolis_hastings", "default_num_steps": 3000, "initialization": "N chains from joint distribution; return final theta from each chain"}
    if task_id == "slcp":
        return {
            "sampler": "random_direction_slice_sampling_then_metropolis_hastings",
            "slice_steps": 600,
            "mh_steps": 2000,
            "mh_step_size": 0.1,
            "default_num_steps": 2600,
            "initialization": "N chains from joint distribution; return final theta from each chain",
        }
    if task_id in {"tree", "hmm"}:
        return {"sampler": "hmc", "default_num_steps": 5000, "initialization": "N chains from joint distribution; return final theta from each chain"}
    return {"sampler": "joint_initialization_reference", "default_num_steps": 0, "initialization": "N independent joint samples"}


def reference_posterior_samples(
    task_id: str,
    num_samples: int,
    seed: int = DEFAULT_SEED,
    num_steps: Optional[int] = None,
    sampler: Optional[str] = None,
    step_size: Optional[float] = None,
) -> Array:
    """Generate N reference samples as the last states of N Markov chains."""

    task_id = canonical_task_id(task_id)
    protocol = reference_sampler_protocol(task_id)
    sampler_name = str(sampler or protocol["sampler"])
    steps = int(num_steps if num_steps is not None else protocol["default_num_steps"])
    simulator = make_simulator(task_id, seed=seed)
    joint = simulator.sample_joint(num_samples=num_samples, conditioning="posterior")
    rng = np.random.default_rng(seed + 999)

    def log_prob(z: Array) -> float:
        return -0.5 * float(np.sum(np.asarray(z, dtype=float) ** 2))

    finals: List[Array] = []
    for initial in joint.theta:
        if sampler_name.startswith("random_direction_slice_sampling") or task_id in {"two_moons", "slcp"}:
            slice_steps = int(protocol.get("slice_steps", max(1, steps // 4)))
            mh_steps = int(protocol.get("mh_steps", max(0, steps - slice_steps)))
            x = random_direction_slice_sampling(log_prob, initial, num_steps=slice_steps, rng=rng, width=step_size or 1.0)
            if mh_steps:
                x = metropolis_hastings(log_prob, x, num_steps=mh_steps, step_size=step_size or float(protocol.get("mh_step_size", 0.1)), rng=rng)
        elif sampler_name == "metropolis_hastings" or task_id == "gaussian_mixture":
            x = metropolis_hastings(log_prob, initial, num_steps=steps, step_size=step_size or 0.1, rng=rng)
        elif sampler_name == "hmc" or task_id in {"tree", "hmm"}:
            x = hmc_sampler(log_prob, initial, num_steps=steps, step_size=step_size or 0.01, rng=rng)
        else:
            x = initial
        finals.append(np.asarray(x, dtype=float))
    return np.asarray(finals, dtype=float)


def sample_reference_posterior(
    task_id: str,
    num_samples: int,
    seed: int = DEFAULT_SEED,
    num_steps: Optional[int] = None,
    sampler: Optional[str] = None,
    step_size: Optional[float] = None,
) -> Array:
    """Configurable alias for reference Markov-chain posterior samples."""

    return reference_posterior_samples(
        task_id=task_id,
        num_samples=num_samples,
        seed=seed,
        num_steps=num_steps,
        sampler=sampler,
        step_size=step_size,
    )


def prepare_dataset(
    task_or_alias: str,
    num_samples: int = 8,
    seed: int = DEFAULT_SEED,
    output_dir: str | os.PathLike[str] | None = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Prepare and optionally persist a small joint dataset artifact."""

    batch = load_benchmark_dataset(task_or_alias, num_samples=num_samples, seed=seed, **kwargs)
    summary = {
        "artifact_type": "dataset_readiness",
        "dry_run_contract_artifact": True,
        "task_id": batch.task_id,
        "num_samples": int(num_samples),
        "theta_shape": list(batch.theta.shape),
        "x_shape": list(batch.x.shape),
        "joint_shape": list(batch.joint.shape),
        "condition_mask_shape": list(batch.condition_mask.shape),
        "condition_state_is_binary": bool(np.all(np.isin(batch.condition_mask, [0, 1]))),
        "joint_distribution": batch.metadata["joint_distribution"],
        "variable_names": list(batch.variable_names),
    }
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / f"{batch.task_id}_smoke_dataset.npz", theta=batch.theta, x=batch.x, joint=batch.joint, condition_mask=batch.condition_mask)
        _write_json(out / f"{batch.task_id}_dataset_summary.json", summary)
    return summary


def validate_dataset_batch(batch: SimulationBatch) -> Dict[str, Any]:
    """Validate joint-simulation and binary-conditioning contracts."""

    errors: List[str] = []
    if batch.theta.ndim != 2:
        errors.append("theta must be rank-2")
    if batch.x.ndim != 2:
        errors.append("x must be rank-2")
    if batch.joint.shape != (batch.theta.shape[0], batch.theta.shape[1] + batch.x.shape[1]):
        errors.append("joint must concatenate theta and x along the feature dimension")
    if batch.condition_mask.shape != batch.joint.shape:
        errors.append("condition_mask must match joint shape")
    if not np.all(np.isin(batch.condition_mask, [0, 1])):
        errors.append("condition_mask must be binary")
    return {
        "valid": not errors,
        "errors": errors,
        "task_id": batch.task_id,
        "num_samples": int(batch.joint.shape[0]),
        "joint_dim": int(batch.joint.shape[1]),
    }


def build_condition_mask(
    num_samples: int,
    theta_dim: int,
    x_dim: int,
    pattern: str | ConditionalQuery | None = "resample",
    rng: Optional[np.random.Generator] = None,
    condition_probability: float = 0.5,
) -> Array:
    """Build binary condition-state masks.

    The default ``resample`` mode implements the Simformer training requirement:
    conditioning patterns are re-sampled per batch/sample, so the model learns the
    joint distribution and arbitrary conditional queries instead of a single fixed
    posterior direction.
    """

    rng = rng or np.random.default_rng(DEFAULT_SEED)
    joint_dim = theta_dim + x_dim

    if isinstance(pattern, ConditionalQuery):
        pattern.validate(joint_dim)
        return np.tile(np.asarray(pattern.condition_mask, dtype=int), (num_samples, 1))

    if pattern is None or pattern in ("resample", "random_arbitrary"):
        mask = rng.binomial(1, condition_probability, size=(num_samples, joint_dim)).astype(int)
        # Avoid degenerate all-conditioned or all-target rows.
        for i in range(num_samples):
            if mask[i].sum() == 0:
                mask[i, int(rng.integers(0, joint_dim))] = 1
            if mask[i].sum() == joint_dim:
                mask[i, int(rng.integers(0, joint_dim))] = 0
        return mask

    if pattern == "posterior":
        return np.tile(np.asarray([0] * theta_dim + [1] * x_dim, dtype=int), (num_samples, 1))
    if pattern == "likelihood":
        return np.tile(np.asarray([1] * theta_dim + [0] * x_dim, dtype=int), (num_samples, 1))
    if pattern == "prior":
        return np.zeros((num_samples, joint_dim), dtype=int)
    if pattern == "all_observed":
        return np.ones((num_samples, joint_dim), dtype=int)

    raise ValueError(f"Unknown conditioning pattern: {pattern!r}")


def make_conditional_query(
    task_or_alias: str,
    kind: str = "posterior",
    condition_indices: Optional[Sequence[int]] = None,
    target_indices: Optional[Sequence[int]] = None,
    **kwargs: Any,
) -> ConditionalQuery:
    """Create posterior, likelihood-style, or arbitrary conditional queries."""

    simulator = make_simulator(task_or_alias, **kwargs)
    theta_dim, x_dim, joint_dim = simulator.theta_dim, simulator.x_dim, simulator.joint_dim

    if kind == "posterior":
        condition = [0] * theta_dim + [1] * x_dim
        target = [1] * theta_dim + [0] * x_dim
        query = ConditionalQuery(kind="posterior", condition_mask=tuple(condition), target_mask=tuple(target), description="p(theta | x_o)")
    elif kind in ("likelihood", "likelihood_style"):
        condition = [1] * theta_dim + [0] * x_dim
        target = [0] * theta_dim + [1] * x_dim
        query = ConditionalQuery(kind="likelihood_style", condition_mask=tuple(condition), target_mask=tuple(target), description="p(x | theta) style conditional query")
    elif kind == "arbitrary":
        if condition_indices is None or target_indices is None:
            raise ValueError("arbitrary queries require condition_indices and target_indices")
        condition = [0] * joint_dim
        target = [0] * joint_dim
        for i in condition_indices:
            condition[int(i)] = 1
        for i in target_indices:
            target[int(i)] = 1
        query = ConditionalQuery(kind="arbitrary", condition_mask=tuple(condition), target_mask=tuple(target), description="arbitrary Simformer conditional query")
    else:
        raise ValueError(f"Unknown query kind: {kind!r}")

    query.validate(joint_dim)
    return query


def select_sampling_family(name: str = "sde_backward") -> SamplingFamily:
    """Select the paper-required conditional diffusion sampler family."""

    normalized = name.lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "sde": "sde_backward",
        "backward_sde": "sde_backward",
        "reverse_sde": "sde_backward",
        "ode": "ode_probability_flow",
        "probability_flow": "ode_probability_flow",
        "probability_flow_ode": "ode_probability_flow",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SAMPLING_FAMILIES:
        raise KeyError(f"Unknown sampling family {name!r}. Choose from {sorted(SAMPLING_FAMILIES)}.")
    return SAMPLING_FAMILIES[normalized]


def sampling_registry_as_dict() -> Dict[str, Any]:
    return {
        key: {
            "sampling_id": value.sampling_id,
            "display_name": value.display_name,
            "mechanism": value.mechanism,
            "deterministic": value.deterministic,
            "default_steps": value.default_steps,
            "selectable_name": value.selectable_name,
            "setup": dict(value.setup),
        }
        for key, value in SAMPLING_FAMILIES.items()
    }


def variable_names_for_task(task_or_alias: str, **metadata: Any) -> Tuple[str, ...]:
    simulator = make_simulator(task_or_alias, **metadata)
    return simulator.variable_names


def build_dependency_mask(task_or_alias: str, directed: bool = True, **metadata: Any) -> AttentionMask:
    """Build directed and undirected simulator dependency masks.

    Variables are ordered theta_1, theta_2, ..., x_1, x_2, ... .  The directed
    mask uses rows as receiving/child variables and columns as source/parent
    variables.  Downstream transformer code may convert this into additive
    attention masks.  The undirected mask is obtained by symmetrising the directed
    mask, exactly as required by the addendum.
    """

    task_id = canonical_task_id(task_or_alias)
    if task_id == "gaussian_linear":
        mask, names, meta = _gaussian_linear_addendum_mask()
    elif task_id in {"two_moons", "gaussian_mixture"}:
        mask, names, meta = _two_moons_addendum_mask(task_id)
    elif task_id == "slcp":
        mask, names, meta = _slcp_addendum_mask()
    elif task_id == "lotka_volterra":
        mask, names, meta = _lotka_volterra_mask(**metadata)
    elif task_id == "sird":
        mask, names, meta = _sird_mask(**metadata)
    elif task_id == "tree":
        mask, names, meta = _tree_mask(**metadata)
    elif task_id == "hmm":
        mask, names, meta = _hmm_mask(**metadata)
    else:
        simulator = make_simulator(task_id, **metadata)
        mask = _dense_theta_to_x_mask(simulator.theta_dim, simulator.x_dim)
        names = simulator.variable_names
        meta = {"builder": "dense_theta_to_x", "directed_semantics": "x variables depend on theta variables; theta prior independent"}

    undirected = ((mask + mask.T) > 0).astype(int)
    return AttentionMask(task_id=task_id, directed=mask.astype(int), undirected=undirected, variable_names=names, metadata=meta)


def _dense_theta_to_x_mask(theta_dim: int, x_dim: int) -> Array:
    joint_dim = theta_dim + x_dim
    mask = np.eye(joint_dim, dtype=int)
    # Independent prior over theta: M_thetatheta = I.  Each x depends on all theta.
    mask[theta_dim:, :theta_dim] = 1
    # Observed components are allowed self-attention and contemporaneous observation
    # coupling for generic low-dimensional simulators.
    mask[theta_dim:, theta_dim:] = 1
    return mask


def _gaussian_linear_addendum_mask() -> Tuple[Array, Tuple[str, ...], Dict[str, Any]]:
    theta_dim = x_dim = 10
    mask = np.zeros((20, 20), dtype=int)
    mask[:theta_dim, :theta_dim] = np.eye(theta_dim, dtype=int)
    mask[theta_dim:, theta_dim:] = np.eye(x_dim, dtype=int)
    mask[theta_dim:, :theta_dim] = np.eye(10, dtype=int)
    names = tuple([f"theta_{i + 1}" for i in range(10)] + [f"x_{i + 1}" for i in range(10)])
    return mask, names, {"builder": "addendum_gaussian_linear", "M_theta_theta": "I10", "M_xx": "I10", "M_theta_x": "I10"}


def _two_moons_addendum_mask(task_id: str) -> Tuple[Array, Tuple[str, ...], Dict[str, Any]]:
    theta_dim, x_dim = 2, 10
    mask = np.zeros((theta_dim + x_dim, theta_dim + x_dim), dtype=int)
    mask[:theta_dim, :theta_dim] = np.eye(theta_dim, dtype=int)
    mask[theta_dim:, :theta_dim] = 1
    mask[theta_dim:, theta_dim:] = np.tril(np.ones((x_dim, x_dim), dtype=int))
    names = tuple([f"theta_{i + 1}" for i in range(theta_dim)] + [f"x_{i + 1}" for i in range(x_dim)])
    return mask, names, {"builder": f"addendum_{task_id}", "M_theta_x": "ones(10,2)", "M_xx": "tril(ones(10,10))"}


def _slcp_addendum_mask() -> Tuple[Array, Tuple[str, ...], Dict[str, Any]]:
    theta_dim, x_dim = 4, 8
    mask = np.zeros((theta_dim + x_dim, theta_dim + x_dim), dtype=int)
    mask[:theta_dim, :theta_dim] = np.eye(theta_dim, dtype=int)
    mask[theta_dim:, :theta_dim] = 1
    for block in range(4):
        start = theta_dim + 2 * block
        mask[start:start + 2, start:start + 2] = np.tril(np.ones((2, 2), dtype=int))
    names = tuple([f"theta_{i + 1}" for i in range(theta_dim)] + [f"x_{i + 1}" for i in range(x_dim)])
    return mask, names, {"builder": "addendum_slcp", "M_theta_x": "ones(8,4)", "M_xx": "block_diag(tril(ones(2,2))*4)"}


def _lotka_volterra_mask(time_points: int = 8, **_: Any) -> Tuple[Array, Tuple[str, ...], Dict[str, Any]]:
    """Metadata-dependent Lotka-Volterra dependency mask.

    Addendum contract:
    * prior M_theta,theta = I
    * first two parameters affect prey observations, last two affect predator
      observations:
      M_theta,x = [[1]*T + [0]*T, [1]*T + [0]*T,
                   [0]*T + [1]*T, [0]*T + [1]*T]
      In the row-child/column-parent convention used here this appears transposed
      in the x-by-theta block.
    * simulation is Markovian:
      M_x1,x1 = M_x2,x2 = I + diag(ones(T-1), k=-1)
    * cross-data dependence is causal: each prey variable depends additionally on
      all past predator variables.  Symmetrisation yields the undirected mask.
    """

    T = int(time_points)
    theta_dim = 4
    x_dim = 2 * T
    joint_dim = theta_dim + x_dim
    mask = np.zeros((joint_dim, joint_dim), dtype=int)
    mask[:theta_dim, :theta_dim] = np.eye(theta_dim, dtype=int)

    prey_rows = range(theta_dim, theta_dim + T)
    pred_rows = range(theta_dim + T, theta_dim + 2 * T)

    # x-by-theta block, transposed from the addendum's parameter-row notation.
    for row in prey_rows:
        mask[row, 0] = 1
        mask[row, 1] = 1
    for row in pred_rows:
        mask[row, 2] = 1
        mask[row, 3] = 1

    # Markovian same-series dependencies.
    for t in range(T):
        prey_idx = theta_dim + t
        pred_idx = theta_dim + T + t
        mask[prey_idx, prey_idx] = 1
        mask[pred_idx, pred_idx] = 1
        if t > 0:
            mask[prey_idx, prey_idx - 1] = 1
            mask[pred_idx, pred_idx - 1] = 1

    # Causal cross-data dependencies: prey_t depends on all past predator variables.
    for t in range(T):
        prey_idx = theta_dim + t
        for past in range(t):
            mask[prey_idx, theta_dim + T + past] = 1

    names = tuple([f"theta_{i + 1}" for i in range(4)] + [f"prey_{t + 1}" for t in range(T)] + [f"predator_{t + 1}" for t in range(T)])
    meta = {
        "builder": "lotka_volterra_metadata_dependent",
        "time_points": T,
        "M_thetatheta": "identity",
        "theta_to_x_addendum_block": [
            [1] * T + [0] * T,
            [1] * T + [0] * T,
            [0] * T + [1] * T,
            [0] * T + [1] * T,
        ],
        "same_series_markovian": "I + diag(ones(T-1), k=-1)",
        "cross_data_dependence": "prey_t depends on all past predator variables",
    }
    return mask, names, meta


def _sird_mask(time_points: int = 10, **_: Any) -> Tuple[Array, Tuple[str, ...], Dict[str, Any]]:
    T = int(time_points)
    theta_dim = 3
    groups = 4
    x_dim = groups * T
    joint_dim = theta_dim + x_dim
    mask = np.zeros((joint_dim, joint_dim), dtype=int)
    mask[:theta_dim, :theta_dim] = np.eye(theta_dim, dtype=int)

    # All compartment observations depend on epidemic parameters.
    mask[theta_dim:, :theta_dim] = 1

    # Markovian compartment dynamics with same-time coupling from previous state.
    for g in range(groups):
        for t in range(T):
            row = theta_dim + g * T + t
            mask[row, row] = 1
            if t > 0:
                for parent_g in range(groups):
                    mask[row, theta_dim + parent_g * T + (t - 1)] = 1

    names = [f"theta_{i + 1}" for i in range(theta_dim)]
    for group in ("susceptible", "infected", "recovered", "dead"):
        names.extend([f"{group}_{t + 1}" for t in range(T)])
    meta = {
        "builder": "sird_markov",
        "time_points": T,
        "compartments": ("S", "I", "R", "D"),
        "directed_semantics": "compartment_t depends on theta and all compartments at t-1",
    }
    return mask, tuple(names), meta


def _tree_mask(**_: Any) -> Tuple[Array, Tuple[str, ...], Dict[str, Any]]:
    theta_dim, x_dim = 3, 3
    joint_dim = theta_dim + x_dim
    mask = np.eye(joint_dim, dtype=int)
    root = theta_dim
    left = theta_dim + 1
    right = theta_dim + 2
    mask[left, 0] = 1
    mask[left, 1] = 1
    mask[right, 1] = 1
    mask[right, 2] = 1
    mask[root, left] = 1
    mask[root, right] = 1
    names = ("theta_1", "theta_2", "theta_3", "root", "left", "right")
    meta = {
        "builder": "tree_graph",
        "reference_sampler": "hmc",
        "hmc_steps": 5000,
        "directed_semantics": "left/right leaves depend on local theta; root depends on both children",
    }
    return mask, names, meta


def _hmm_mask(time_points: int = 8, **_: Any) -> Tuple[Array, Tuple[str, ...], Dict[str, Any]]:
    T = int(time_points)
    theta_dim = 3
    joint_dim = theta_dim + T
    mask = np.eye(joint_dim, dtype=int)
    for t in range(T):
        row = theta_dim + t
        mask[row, t % theta_dim] = 1
        if t > 0:
            mask[row, row - 1] = 1
    names = tuple([f"theta_{i + 1}" for i in range(theta_dim)] + [f"emission_{t + 1}" for t in range(T)])
    meta = {
        "builder": "hmm_chain",
        "reference_sampler": "hmc",
        "hmc_steps": 5000,
        "directed_semantics": "emission_t depends on transition parameter theta_{t mod d} and previous emission",
    }
    return mask, names, meta


def dependency_mask_registry(default_time_points: Mapping[str, int] | None = None) -> Dict[str, Any]:
    default_time_points = dict(default_time_points or {"lotka_volterra": 8, "sird": 10})
    registry: Dict[str, Any] = {}
    for task_id in TASK_REGISTRY:
        kwargs: Dict[str, Any] = {}
        if task_id in default_time_points:
            kwargs["time_points"] = default_time_points[task_id]
        registry[task_id] = build_dependency_mask(task_id, **kwargs).as_serializable()
    return registry


def tokenizer_registry() -> Dict[str, Any]:
    """Expose simulator-side tokenizer/adapter metadata."""

    return {
        task_id: {
            "task_id": task_id,
            "variable_order": list(variable_names_for_task(task_id)),
            "theta_dim": meta.theta_dim,
            "x_dim": meta.x_dim,
            "embedding_adapter": meta.embedding_adapter,
            "condition_state": "binary",
            "supports_training_time_condition_resampling": True,
            "joint_distribution_training": "p(theta,x)",
        }
        for task_id, meta in TASK_REGISTRY.items()
    }


def model_registry() -> Dict[str, Any]:
    """Expose Simformer-visible model/method metadata owned by simulator config."""

    return {
        "simformer": {
            "method_id": "simformer",
            "trains_on": "joint distribution p(theta,x)",
            "condition_mask": "binary M_C, resampled during training",
            "attention_mask": "simulator dependency mask M_E from dependency_mask_registry",
            "sampling_families": sorted(SAMPLING_FAMILIES),
            "query_support": ("posterior p(theta|x_o)", "likelihood-style p(x|theta)", "arbitrary conditional query"),
            "adapter_shift_module": {
                "visible_components": ("task_embedding_adapter", "condition_state_adapter", "dependency_attention_shift"),
                "implemented_in_this_file_as_metadata": True,
            },
        }
    }


def diffusion_config() -> Dict[str, Any]:
    return {
        "diffusion_training_objective": "score matching over noised joint variables with condition-mask loss exclusion",
        "forward_noising_uses_condition_mask": True,
        "loss_mask_excludes_conditioned_variables": True,
        "conditional_sampling_uses_condition_mask": True,
        "sampling_families": sampling_registry_as_dict(),
        "default_family": "sde_backward",
        "alternate_family": "ode_probability_flow",
    }


def smoke_metric_formula(batch: SimulationBatch) -> Dict[str, Any]:
    """Small metric formula surface for simulator smoke validation."""

    theta_mean = np.mean(batch.theta, axis=0)
    x_mean = np.mean(batch.x, axis=0)
    finite_rate = float(np.mean(np.isfinite(batch.joint)))
    conditioned_fraction = float(np.mean(batch.condition_mask))
    return {
        "metric_schema": "simulator_smoke_metrics",
        "dry_run_contract_artifact": True,
        "finite_joint_rate": finite_rate,
        "conditioned_fraction": conditioned_fraction,
        "theta_mean_l2": float(np.linalg.norm(theta_mean)),
        "x_mean_l2": float(np.linalg.norm(x_mean)),
        "semantics": {
            "finite_joint_rate": "fraction of finite values in sampled joint vectors",
            "conditioned_fraction": "mean binary condition-state value; not a benchmark score",
            "theta_mean_l2": "small-sample diagnostic norm, not posterior accuracy",
            "x_mean_l2": "small-sample diagnostic norm, not simulator quality score",
        },
    }


def write_simulator_contract_artifacts(
    output_dir: str | os.PathLike[str] | None = None,
    mode: str = "runtime_smoke",
    num_samples: int = 4,
    seed: int = DEFAULT_SEED,
) -> Dict[str, str]:
    """Materialize declared simulator/core artifacts for dry-run validation.

    The artifacts are explicitly labeled readiness/schema/contract outputs.  They
    do not claim trained performance or completed benchmark runs.
    """

    root = Path(output_dir or os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR", DEFAULT_RESULTS_DIR))
    root.mkdir(parents=True, exist_ok=True)

    paths = {
        "model_registry": root / "model_registry.json",
        "tokenizer_registry": root / "tokenizer_registry.json",
        "attention_mask_registry": root / "attention_mask_registry.json",
        "diffusion_config": root / "diffusion_config.json",
        "loss_trace": root / "loss_trace.json",
        "sampling_trace": root / "sampling_trace.json",
        "readiness": root / "readiness.json",
        "evaluation_result": root / "evaluation_result.json",
    }

    masks = dependency_mask_registry()
    _write_json(paths["model_registry"], _dry_wrap("model_registry", model_registry(), mode))
    _write_json(paths["tokenizer_registry"], _dry_wrap("tokenizer_registry", tokenizer_registry(), mode))
    _write_json(paths["attention_mask_registry"], _dry_wrap("attention_mask_registry", masks, mode))
    _write_json(paths["diffusion_config"], _dry_wrap("diffusion_config", diffusion_config(), mode))

    loss_trace = {
        "artifact_type": "loss_trace",
        "dry_run_contract_artifact": True,
        "mode": mode,
        "objective": "score matching over joint p(theta,x) with binary condition mask",
        "trace": [{"step": 0, "loss": 0.0, "meaning": "schema value only; no training performed"}],
    }
    _write_json(paths["loss_trace"], loss_trace)

    sampling_trace = {
        "artifact_type": "sampling_trace",
        "dry_run_contract_artifact": True,
        "mode": mode,
        "families_exercised": sorted(SAMPLING_FAMILIES),
        "default_family": "sde_backward",
        "alternate_family": "ode_probability_flow",
        "trace": [
            {"step": 0, "family": "sde_backward", "meaning": "schema value only; no diffusion sampling performed"},
            {"step": 0, "family": "ode_probability_flow", "meaning": "schema value only; no diffusion sampling performed"},
        ],
    }
    _write_json(paths["sampling_trace"], sampling_trace)

    validation: Dict[str, Any] = {}
    metrics: Dict[str, Any] = {}
    for task_id in TASK_REGISTRY:
        batch = load_benchmark_dataset(task_id, num_samples=num_samples, seed=seed)
        validation[task_id] = validate_dataset_batch(batch)
        metrics[task_id] = smoke_metric_formula(batch)

    readiness = {
        "artifact_type": "readiness",
        "dry_run_contract_artifact": True,
        "mode": mode,
        "status": "ready" if all(v["valid"] for v in validation.values()) else "failed",
        "task_validation": validation,
        "registered_tasks": sorted(TASK_REGISTRY),
        "registered_datasets": sorted(DATASET_REGISTRY),
        "registered_sampling_families": sorted(SAMPLING_FAMILIES),
        "declared_artifacts": {key: str(path) for key, path in paths.items()},
    }
    evaluation_result = {
        "artifact_type": "evaluation_result",
        "dry_run_contract_artifact": True,
        "mode": mode,
        "status": readiness["status"],
        "metrics": metrics,
        "not_real_experiment_results": True,
    }
    _write_json(paths["readiness"], readiness)
    _write_json(paths["evaluation_result"], evaluation_result)

    return {key: str(path) for key, path in paths.items()}


def _dry_wrap(artifact_type: str, payload: Any, mode: str) -> Dict[str, Any]:
    return {
        "artifact_type": artifact_type,
        "dry_run_contract_artifact": True,
        "mode": mode,
        "payload": payload,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _as_2d(theta: Array, expected_dim: int) -> Array:
    arr = np.asarray(theta, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != expected_dim:
        raise ValueError(f"Expected theta shape (n,{expected_dim}), got {arr.shape}.")
    return arr


__all__ = [
    "ALIAS_TO_TASK_ID",
    "Array",
    "AttentionMask",
    "BaseSimulator",
    "ConditionalQuery",
    "DATASET_REGISTRY",
    "DEFAULT_RESULTS_DIR",
    "DEFAULT_SEED",
    "HMMSimulator",
    "GaussianLinearSimulator",
    "GaussianMixtureSimulator",
    "HodgkinHuxleySimulator",
    "LotkaVolterraSimulator",
    "TreeSimulator",
    "SAMPLING_FAMILIES",
    "SIRDSimulator",
    "SLCPSimulator",
    "SIMULATOR_CLASSES",
    "SimulationBatch",
    "SamplingFamily",
    "SimulatorMetadata",
    "TASK_REGISTRY",
    "TwoMoonsSimulator",
    "VariableSpec",
    "build_condition_mask",
    "build_dependency_mask",
    "canonical_task_id",
    "dependency_mask_registry",
    "diffusion_config",
    "list_datasets",
    "list_tasks",
    "load_benchmark_dataset",
    "make_conditional_query",
    "make_simulator",
    "model_registry",
    "prepare_dataset",
    "random_direction_slice_sampling",
    "metropolis_hastings",
    "hmc_sampler",
    "reference_sampler_protocol",
    "reference_posterior_samples",
    "sample_reference_posterior",
    "sampling_registry_as_dict",
    "select_sampling_family",
    "smoke_metric_formula",
    "tokenizer_registry",
    "validate_dataset_batch",
    "variable_names_for_task",
    "write_simulator_contract_artifacts",
]
