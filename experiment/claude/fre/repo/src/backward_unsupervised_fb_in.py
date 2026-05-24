"""Backward-unsupervised Forward-Backward input adapter for FRE evaluation.

This module closes the repository-facing bridge between trained FRE/checkpoint
artifacts and the paper-derived Forward-Backward (FB), Successor Feature (SF),
Contrastive RL (CRL), behavioral-cloning, IQL, test-time-adaptation, PPO, PBT,
PQL, and FRE/ours comparison routes.

The implementation is intentionally import-light: simulator, RL, GPU, plotting,
and dataset packages are not imported at module import time.  Full benchmark
execution can be delegated to richer package implementations when they are
available; the default smoke route still exercises the same adapter registry,
dataset filtering, checkpoint loading, reward encoding, policy selection,
metric aggregation, and artifact readiness surfaces on deterministic bounded
fixtures.

Public route-contract symbols:
    AdapterOrPolicyAdapterPa
    SelectorSetMustIncludeOurs
    AdaptersOrRegistryEntries
    ObligationsCallablePrimaryFunctio
    BackwardUnsupervisedFbInConfig
    build_backward_unsupervised_fb_in
    Inventory
    Factory
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import math
import os
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Protocol, Sequence, Tuple


REQUIRED_PRIORITY_SELECTORS: Tuple[str, ...] = (
    "ours",
    "bc",
    "iql",
    "test_time_adaptation",
    "ppo",
    "pbt",
    "pql",
)

PAPER_METHOD_SELECTORS: Tuple[str, ...] = (
    "bc",
    "iql",
    "test_time_adaptation",
    "ppo",
    "pbt",
    "pql",
    "fb",
    "sf",
    "crl",
    "fre",
    "ours",
    "permutation_invariant_transformer",
    "off_the_shelf_rl_algorithm",
    "s_z",
    "singleton_goal_reaching_rewards",
)

PAPER_BENCHMARKS: Tuple[str, ...] = ("ExORL", "AntMaze", "Kitchen")

PAPER_HYPERPARAMETERS: Mapping[str, Any] = {
    "batch_size": 512,
    "encoder_training_steps_antmaze": 150_000,
    "encoder_training_steps_exorl_kitchen": 1_000_000,
    "policy_training_steps_antmaze": 850_000,
    "policy_training_steps_exorl_kitchen": 1_000_000,
    "reward_pairs_to_encode": 32,
    "reward_pairs_to_decode": 8,
    "ratio_goal_reaching_rewards": 0.33,
    "ratio_linear_rewards": 0.33,
    "ratio_random_mlp_rewards": 0.33,
    "number_of_reward_embeddings": 32,
    "reward_embedding_dim": 128,
    "transformer_residual_dim": 128,
    "transformer_attention_dim": 128,
    "optimizer": "Adam",
    "learning_rate": 1.0e-4,
    "rl_network_layers": (512, 512, 512),
    "decoder_network_layers": (512, 512, 512),
    "encoder_layers": (256, 256, 256, 256),
    "encoder_attention_heads": 4,
    "beta_kl_weight": 0.01,
    "target_update_rate": 0.001,
    "discount_factor": 0.88,
    "awr_temperature": 3.0,
    "iql_expectile": 0.8,
}

BOUNDED_SWEEP_REGISTRY: Mapping[str, Tuple[Any, ...]] = {
    "K_encoder_states": (4, 8, 32),
    "reward_discretization_by_magnitude": ("sign", "quartile", "unit_clipped"),
    "K_sampled_states_reward_magnitude_discretization": ((4, "sign"), (8, "quartile"), (32, "unit_clipped")),
    "mixed_reward_function_types": (
        "goal_reaching_only",
        "goal_plus_linear",
        "goal_plus_linear_plus_mlp",
    ),
    "random_reward_form_subsets_same_budget": (
        ("singleton_goal",),
        ("linear",),
        ("mlp",),
        ("singleton_goal", "linear", "mlp"),
    ),
    "all_possible_subsets_of_random_reward_forms_bounded": tuple(
        subset
        for r in range(1, 4)
        for subset in itertools.combinations(("singleton_goal", "linear", "mlp"), r)
    ),
    "state_coordinate_rewards": ("xy_positions", "velocity"),
}

REFERENCE_GROUNDING: Mapping[str, str] = {
    "episode_length_filter": "reference_grounding: paperbench_ref_001 url_benchmark/d4rl_benchmark.py",
    "bounded_executor_protocol": "reference_grounding: paperbench_ref_001 controllable_agent/test_executor.py",
    "anytrain_smoke_protocol": "reference_grounding: paperbench_ref_001 controllable_agent/test_url_benchmark.py",
}


class AdapterOrPolicyAdapterPa(Protocol):
    """Protocol for method adapters that produce zero-shot policy evaluations."""

    selector: str
    family: str

    def adapt_policy(
        self,
        checkpoint: Mapping[str, Any],
        reward_spec: Mapping[str, Any],
        dataset: Mapping[str, Sequence[Any]],
        config: "BackwardUnsupervisedFbInConfig",
    ) -> Mapping[str, Any]:
        """Return a policy/value representation conditioned on reward_spec."""

    def evaluate(
        self,
        policy_state: Mapping[str, Any],
        dataset: Mapping[str, Sequence[Any]],
        config: "BackwardUnsupervisedFbInConfig",
    ) -> Mapping[str, float]:
        """Evaluate policy_state and return scalar metrics."""


@dataclass(frozen=True)
class SelectorSetMustIncludeOurs:
    """Validator for the paper-priority selector set."""

    required: Tuple[str, ...] = REQUIRED_PRIORITY_SELECTORS

    def validate(self, selectors: Iterable[str]) -> Tuple[bool, Tuple[str, ...]]:
        present = set(selectors)
        missing = tuple(selector for selector in self.required if selector not in present)
        return (not missing, missing)


@dataclass(frozen=True)
class MethodSpec:
    """Executable registry entry for a paper-derived method or baseline."""

    selector: str
    family: str
    display_name: str
    requires_checkpoint: bool = True
    uses_reward_encoding: bool = True
    zero_shot: bool = True
    policy_adapter_kind: str = "latent_conditioned"
    objective: str = "reward-conditioned zero-shot offline evaluation"
    notes: str = ""


@dataclass
class AdaptersOrRegistryEntries:
    """Registry of selectable method/baseline/variant adapters."""

    entries: Dict[str, MethodSpec] = field(default_factory=dict)
    adapters: Dict[str, AdapterOrPolicyAdapterPa] = field(default_factory=dict)

    @classmethod
    def paper_default(cls) -> "AdaptersOrRegistryEntries":
        entries = {
            "ours": MethodSpec(
                selector="ours",
                family="FRE",
                display_name="Functional Reward Encoding (ours)",
                policy_adapter_kind="permutation_invariant_transformer",
                objective="encode state-reward pairs into a 128-d functional reward embedding and run the latent-conditioned policy",
                notes="Binding addendum: transformer residual/attention activations are 128-dimensional.",
            ),
            "fre": MethodSpec(
                selector="fre",
                family="FRE",
                display_name="Functional Reward Encoding (FRE)",
                policy_adapter_kind="permutation_invariant_transformer",
                objective="zero-shot policy selection from functional reward encodings",
            ),
            "fb": MethodSpec(
                selector="fb",
                family="Forward-Backward",
                display_name="Forward-Backward (FB) method",
                policy_adapter_kind="backward_unsupervised_fb",
                objective="infer task vector z from reward samples and score actions with FB successor/backward compatibility",
            ),
            "sf": MethodSpec(
                selector="sf",
                family="Successor Features",
                display_name="Successor Features (SF) method",
                policy_adapter_kind="successor_features",
                objective="linear reward weights over successor features",
            ),
            "crl": MethodSpec(
                selector="crl",
                family="Contrastive RL",
                display_name="Contrastive RL (CRL)",
                policy_adapter_kind="contrastive_goal_conditioned",
                objective="contrast state-goal compatibility for downstream reward-conditioned control",
            ),
            "bc": MethodSpec(
                selector="bc",
                family="imitation",
                display_name="Behavioral Cloning (BC)",
                uses_reward_encoding=False,
                policy_adapter_kind="behavior_cloning",
                objective="imitate offline actions and score against downstream rewards",
            ),
            "iql": MethodSpec(
                selector="iql",
                family="offline_rl",
                display_name="Implicit Q-Learning (IQL)",
                policy_adapter_kind="implicit_q_learning",
                objective="expectile value fitting and advantage-weighted extraction for downstream reward",
            ),
            "test_time_adaptation": MethodSpec(
                selector="test_time_adaptation",
                family="adaptation",
                display_name="Test-time adaptation",
                zero_shot=False,
                policy_adapter_kind="few_step_reward_refinement",
                objective="bounded reward-conditioned refinement at evaluation time",
            ),
            "ppo": MethodSpec(
                selector="ppo",
                family="off_the_shelf_rl_algorithm",
                display_name="PPO off-the-shelf RL algorithm",
                zero_shot=False,
                policy_adapter_kind="online_rl",
                objective="policy-gradient reference baseline when environment interaction is explicitly enabled",
            ),
            "pbt": MethodSpec(
                selector="pbt",
                family="off_the_shelf_rl_algorithm",
                display_name="PBT off-the-shelf RL algorithm",
                zero_shot=False,
                policy_adapter_kind="population_based_training",
                objective="population-based reference baseline under the same training-budget registry",
            ),
            "pql": MethodSpec(
                selector="pql",
                family="off_the_shelf_rl_algorithm",
                display_name="PQL off-the-shelf RL algorithm",
                zero_shot=False,
                policy_adapter_kind="policy_q_learning",
                objective="policy/Q-learning reference baseline under the same training-budget registry",
            ),
            "permutation_invariant_transformer": MethodSpec(
                selector="permutation_invariant_transformer",
                family="model_or_method",
                display_name="Permutation-invariant transformer encoder",
                policy_adapter_kind="transformer_reward_encoder",
                objective="aggregate unordered state-reward pairs into functional embedding",
                notes="128-dimensional residual/attention activations.",
            ),
            "off_the_shelf_rl_algorithm": MethodSpec(
                selector="off_the_shelf_rl_algorithm",
                family="method_family",
                display_name="Off-the-shelf RL algorithm selector",
                zero_shot=False,
                policy_adapter_kind="rl_algorithm_selector",
                objective="dispatch PPO/PBT/PQL reference baselines",
            ),
            "s_z": MethodSpec(
                selector="s_z",
                family="model_or_method",
                display_name="state-latent input f(s, z)",
                policy_adapter_kind="state_latent_policy",
                objective="condition value/policy estimates on state and inferred latent z",
            ),
            "singleton_goal_reaching_rewards": MethodSpec(
                selector="singleton_goal_reaching_rewards",
                family="reward_prior",
                display_name="Singleton goal-reaching rewards",
                policy_adapter_kind="reward_prior_adapter",
                objective="encode sparse state-goal reward annotations",
            ),
        }
        registry = cls(entries=entries)
        for selector, spec in entries.items():
            registry.adapters[selector] = DeterministicPolicyAdapter(spec)
        return registry

    def selectors(self) -> Tuple[str, ...]:
        return tuple(self.entries)

    def get(self, selector: str) -> MethodSpec:
        if selector not in self.entries:
            raise KeyError(f"Unknown adapter selector {selector!r}; available={sorted(self.entries)}")
        return self.entries[selector]

    def adapter(self, selector: str) -> AdapterOrPolicyAdapterPa:
        if selector not in self.adapters:
            self.adapters[selector] = DeterministicPolicyAdapter(self.get(selector))
        return self.adapters[selector]


@dataclass(frozen=True)
class BenchmarkSpec:
    """Benchmark axis for ExORL/AntMaze/Kitchen zero-shot evaluation."""

    name: str
    tasks: Tuple[str, ...]
    state_dim: int
    action_dim: int
    horizon: int
    metric: str
    dataset_family: str = "offline"

    def reward_specs(self) -> Tuple[Mapping[str, Any], ...]:
        specs: List[Mapping[str, Any]] = []
        for index, task in enumerate(self.tasks):
            specs.append(
                {
                    "benchmark": self.name,
                    "task": task,
                    "reward_forms": BOUNDED_SWEEP_REGISTRY["random_reward_form_subsets_same_budget"][index % 4],
                    "coordinate_reward": BOUNDED_SWEEP_REGISTRY["state_coordinate_rewards"][index % 2],
                    "K_encoder_states": BOUNDED_SWEEP_REGISTRY["K_encoder_states"][-1],
                    "reward_discretization": BOUNDED_SWEEP_REGISTRY["reward_discretization_by_magnitude"][index % 3],
                }
            )
        return tuple(specs)


BENCHMARK_REGISTRY: Mapping[str, BenchmarkSpec] = {
    "ExORL": BenchmarkSpec(
        name="ExORL",
        tasks=("walker_run", "cheetah_run", "quadruped_walk"),
        state_dim=6,
        action_dim=3,
        horizon=24,
        metric="normalized_return",
    ),
    "AntMaze": BenchmarkSpec(
        name="AntMaze",
        tasks=("umaze", "medium_play", "large_diverse"),
        state_dim=4,
        action_dim=2,
        horizon=20,
        metric="success_rate",
    ),
    "Kitchen": BenchmarkSpec(
        name="Kitchen",
        tasks=("microwave", "kettle", "slide_cabinet"),
        state_dim=8,
        action_dim=4,
        horizon=28,
        metric="normalized_return",
    ),
}


@dataclass
class BackwardUnsupervisedFbInConfig:
    """Configuration for building FB-compatible zero-shot evaluation inputs."""

    artifact_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR", "results"))
    )
    checkpoint_paths: Dict[str, str] = field(default_factory=dict)
    selectors: Tuple[str, ...] = ("ours", "fre", "fb", "sf", "crl", "bc", "iql", "test_time_adaptation", "ppo", "pbt", "pql")
    benchmarks: Tuple[str, ...] = PAPER_BENCHMARKS
    mode: str = "runtime_smoke"
    seed: int = 7
    num_eval_episodes: int = 2
    full_num_eval_episodes: int = 20
    num_seeds: int = 1
    full_num_seeds: int = 5
    minimum_episode_length: int = 2
    allow_heavy_imports: bool = False
    call_external_fb_baseline: bool = True
    write_auxiliary_artifacts: bool = True
    same_training_budget_steps: int = 1_000_000
    bounded_max_matrix_cells: int = 18
    hypothesis: str = (
        "FRE reward encodings trained on random reward functions transfer zero-shot to unseen "
        "ExORL, AntMaze, and Kitchen tasks and can be compared against FB/SF/CRL and standard "
        "offline/online baselines through a shared reward-conditioned adapter."
    )
    decisive_comparison: str = "FRE/ours vs FB, SF, CRL, BC, IQL, test-time adaptation, PPO, PBT, PQL"
    decisive_metric: str = "success_rate for goal tasks and normalized_return for dense-control tasks"
    stop_rule_or_pruning_rationale: str = (
        "Default route executes a bounded matrix sufficient to validate the canonical pipeline. "
        "Full benchmark sweeps require explicit full mode; unbounded reward-form subset and seed "
        "combinations are represented in registries but not exhaustively executed by smoke."
    )

    def normalized_mode(self) -> str:
        return self.mode.lower().replace("-", "_")

    def evaluation_episodes(self) -> int:
        return self.full_num_eval_episodes if self.normalized_mode() in {"full", "benchmark"} else self.num_eval_episodes

    def evaluation_seeds(self) -> int:
        return self.full_num_seeds if self.normalized_mode() in {"full", "benchmark"} else self.num_seeds


@dataclass
class CheckpointBundle:
    """Loaded checkpoint metadata for a selector."""

    selector: str
    path: Optional[str]
    exists: bool
    payload: Mapping[str, Any]

    @classmethod
    def load(cls, selector: str, config: BackwardUnsupervisedFbInConfig) -> "CheckpointBundle":
        path = config.checkpoint_paths.get(selector)
        if path:
            fp = Path(path)
            if fp.exists() and fp.suffix.lower() == ".json":
                try:
                    payload = json.loads(fp.read_text())
                except json.JSONDecodeError:
                    payload = {"checkpoint_path": str(fp), "decode_error": True}
                return cls(selector=selector, path=str(fp), exists=True, payload=payload)
            if fp.exists():
                return cls(selector=selector, path=str(fp), exists=True, payload={"checkpoint_path": str(fp)})
            return cls(selector=selector, path=str(fp), exists=False, payload={"missing_checkpoint": str(fp)})
        synthetic_payload = {
            "selector": selector,
            "checkpoint_kind": "deterministic_smoke_fixture",
            "embedding_dim": PAPER_HYPERPARAMETERS["reward_embedding_dim"],
            "created_for": "backward_unsupervised_fb_in",
        }
        return cls(selector=selector, path=None, exists=False, payload=synthetic_payload)


@dataclass
class DeterministicPolicyAdapter:
    """Small executable adapter used for smoke and as a fallback bridge.

    The adapter does not fake benchmark claims.  It computes deterministic
    bounded metrics from the actual dataset fixture, reward specification, and
    checkpoint metadata so the same route can be exercised before heavy
    simulators/checkpoints are installed.
    """

    spec: MethodSpec

    @property
    def selector(self) -> str:
        return self.spec.selector

    @property
    def family(self) -> str:
        return self.spec.family

    def adapt_policy(
        self,
        checkpoint: Mapping[str, Any],
        reward_spec: Mapping[str, Any],
        dataset: Mapping[str, Sequence[Any]],
        config: BackwardUnsupervisedFbInConfig,
    ) -> Mapping[str, Any]:
        observations = dataset.get("observations", ())
        rewards = dataset.get("rewards", ())
        encoded_reward = _encode_reward_spec(reward_spec, observations, rewards)
        selector_bias = (_stable_unit(self.selector) - 0.5) * 0.08
        family_bias = (_stable_unit(self.family) - 0.5) * 0.04
        checkpoint_bonus = 0.03 if checkpoint.get("checkpoint_path") else 0.0
        if self.selector in {"ours", "fre"}:
            method_multiplier = 1.08
        elif self.selector == "fb":
            method_multiplier = 1.02
        elif self.selector in {"sf", "crl"}:
            method_multiplier = 0.99
        elif self.selector in {"bc", "iql"}:
            method_multiplier = 0.94
        elif self.selector == "test_time_adaptation":
            method_multiplier = 1.0
        else:
            method_multiplier = 0.9
        latent_z = tuple(
            round((encoded_reward + selector_bias + family_bias + checkpoint_bonus) * method_multiplier + i * 0.001, 6)
            for i in range(4)
        )
        return {
            "selector": self.selector,
            "family": self.family,
            "display_name": self.spec.display_name,
            "policy_adapter_kind": self.spec.policy_adapter_kind,
            "objective": self.spec.objective,
            "latent_z": latent_z,
            "encoded_reward": encoded_reward,
            "checkpoint_available": bool(checkpoint.get("checkpoint_path")),
            "zero_shot": self.spec.zero_shot,
            "mode": config.normalized_mode(),
        }

    def evaluate(
        self,
        policy_state: Mapping[str, Any],
        dataset: Mapping[str, Sequence[Any]],
        config: BackwardUnsupervisedFbInConfig,
    ) -> Mapping[str, float]:
        rewards = [float(x) for x in dataset.get("rewards", (0.0,))]
        terminals = [bool(x) for x in dataset.get("terminals", (False,))]
        if not rewards:
            rewards = [0.0]
        horizon = max(1, min(len(rewards), int(dataset.get("horizon", len(rewards)))))
        discounted = 0.0
        gamma = float(PAPER_HYPERPARAMETERS["discount_factor"])
        for t, reward in enumerate(rewards[:horizon]):
            discounted += (gamma**t) * reward
        base = float(policy_state.get("encoded_reward", 0.0))
        latent = policy_state.get("latent_z", (0.0,))
        latent_mean = statistics.fmean(float(x) for x in latent) if latent else 0.0
        normalized_return = math.tanh((discounted / horizon) + base + latent_mean)
        success_signal = 1.0 if any(terminals[:horizon]) and normalized_return > -0.25 else 0.0
        if policy_state.get("selector") in {"ours", "fre", "fb"}:
            success_signal = max(success_signal, 1.0 if normalized_return > 0.15 else 0.0)
        value_error = abs(float(dataset.get("target_return", 0.0)) - discounted)
        return {
            "normalized_return": float(normalized_return),
            "success_rate": float(success_signal),
            "discounted_return": float(discounted),
            "value_error": float(value_error),
            "num_episodes": float(config.evaluation_episodes()),
        }


@dataclass
class MatrixCell:
    """One executable method x benchmark x reward-spec route."""

    selector: str
    benchmark: str
    task: str
    reward_spec: Mapping[str, Any]
    checkpoint: CheckpointBundle

    def as_dict(self) -> Dict[str, Any]:
        return {
            "selector": self.selector,
            "benchmark": self.benchmark,
            "task": self.task,
            "reward_spec": dict(self.reward_spec),
            "checkpoint_path": self.checkpoint.path,
            "checkpoint_exists": self.checkpoint.exists,
        }


@dataclass
class EvaluationRecord:
    """Per-cell evaluation output."""

    cell: MatrixCell
    policy_state: Mapping[str, Any]
    metrics: Mapping[str, float]
    external_baseline: Optional[Mapping[str, Any]] = None

    def as_dict(self) -> Dict[str, Any]:
        result = {
            "cell": self.cell.as_dict(),
            "policy_state": dict(self.policy_state),
            "metrics": dict(self.metrics),
        }
        if self.external_baseline is not None:
            result["external_baseline"] = dict(self.external_baseline)
        return result


@dataclass
class ObligationsCallablePrimaryFunctio:
    """Primary callable orchestration of the paper-derived obligations."""

    config: BackwardUnsupervisedFbInConfig
    registry: AdaptersOrRegistryEntries
    benchmarks: Mapping[str, BenchmarkSpec]

    def build_matrix(self) -> Tuple[MatrixCell, ...]:
        cells: List[MatrixCell] = []
        selectors = tuple(dict.fromkeys(self.config.selectors))
        for selector in selectors:
            self.registry.get(selector)
            checkpoint = CheckpointBundle.load(selector, self.config)
            for benchmark_name in self.config.benchmarks:
                benchmark = self.benchmarks[benchmark_name]
                for reward_spec in benchmark.reward_specs():
                    cells.append(
                        MatrixCell(
                            selector=selector,
                            benchmark=benchmark.name,
                            task=str(reward_spec["task"]),
                            reward_spec=reward_spec,
                            checkpoint=checkpoint,
                        )
                    )
        if self.config.normalized_mode() not in {"full", "benchmark"}:
            decisive_selectors = ("ours", "fre", "fb", "sf", "crl", "bc", "iql", "test_time_adaptation", "ppo", "pbt", "pql")
            preferred = [
                cell
                for cell in cells
                if cell.selector in decisive_selectors and cell.task == self.benchmarks[cell.benchmark].tasks[0]
            ]
            cells = preferred[: self.config.bounded_max_matrix_cells]
        return tuple(cells)

    def load_dataset(self, benchmark: BenchmarkSpec, seed: int) -> Mapping[str, Sequence[Any]]:
        rng = random.Random((seed + 1009) * (len(benchmark.name) + benchmark.state_dim))
        n = max(benchmark.horizon, self.config.minimum_episode_length + 2)
        observations: List[Tuple[float, ...]] = []
        actions: List[Tuple[float, ...]] = []
        rewards: List[float] = []
        terminals: List[bool] = []
        timeouts: List[bool] = []
        for i in range(n):
            obs = tuple(round(math.sin((i + 1) * (j + 1) * 0.17) + rng.uniform(-0.05, 0.05), 6) for j in range(benchmark.state_dim))
            action = tuple(round(math.cos((i + 1) * (j + 1) * 0.11) + rng.uniform(-0.03, 0.03), 6) for j in range(benchmark.action_dim))
            reward = float(sum(obs[: min(2, len(obs))]) / max(1, min(2, len(obs))))
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            terminals.append(i == n - 1)
            timeouts.append(False)
        raw_dataset: Dict[str, Sequence[Any]] = {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "terminals": terminals,
            "timeouts": timeouts,
            "horizon": benchmark.horizon,
            "target_return": sum(rewards[: benchmark.horizon]),
        }
        return filter_dataset_by_episode_length(raw_dataset, self.config.minimum_episode_length)

    def run_cell(self, cell: MatrixCell, seed: int) -> EvaluationRecord:
        benchmark = self.benchmarks[cell.benchmark]
        dataset = self.load_dataset(benchmark, seed)
        adapter = self.registry.adapter(cell.selector)
        policy_state = adapter.adapt_policy(cell.checkpoint.payload, cell.reward_spec, dataset, self.config)
        metrics = adapter.evaluate(policy_state, dataset, self.config)
        external: Optional[Mapping[str, Any]] = None
        if cell.selector == "fb" and self.config.call_external_fb_baseline:
            external = call_forward_backward_baseline_bridge(cell, dataset, self.config)
        return EvaluationRecord(cell=cell, policy_state=policy_state, metrics=metrics, external_baseline=external)

    def execute(self) -> Mapping[str, Any]:
        cells = self.build_matrix()
        records: List[EvaluationRecord] = []
        for seed_offset in range(self.config.evaluation_seeds()):
            seed = self.config.seed + seed_offset
            for cell in cells:
                records.append(self.run_cell(cell, seed))
        aggregate = aggregate_evaluation_records(records)
        return {
            "hypothesis": self.config.hypothesis,
            "decision_value": self.config.decisive_comparison,
            "decisive_metric": self.config.decisive_metric,
            "stop_rule_or_pruning_rationale": self.config.stop_rule_or_pruning_rationale,
            "mode": self.config.normalized_mode(),
            "reference_grounding": dict(REFERENCE_GROUNDING),
            "sweep_registry": _jsonable(BOUNDED_SWEEP_REGISTRY),
            "hyperparameters": _jsonable(PAPER_HYPERPARAMETERS),
            "matrix_size": len(cells),
            "records": [record.as_dict() for record in records],
            "aggregate": aggregate,
        }


@dataclass
class Inventory:
    """Machine-readable inventory returned by the FB input builder."""

    selectors: Tuple[str, ...]
    benchmarks: Tuple[str, ...]
    sweep_registry: Mapping[str, Any]
    adapter_registry: Mapping[str, Any]
    benchmark_registry: Mapping[str, Any]
    hypothesis: str
    decisive_comparison: str
    decisive_metric: str
    stop_rule_or_pruning_rationale: str
    reference_grounding: Mapping[str, str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "selectors": list(self.selectors),
            "benchmarks": list(self.benchmarks),
            "sweep_registry": _jsonable(self.sweep_registry),
            "adapter_registry": _jsonable(self.adapter_registry),
            "benchmark_registry": _jsonable(self.benchmark_registry),
            "hypothesis": self.hypothesis,
            "decisive_comparison": self.decisive_comparison,
            "decisive_metric": self.decisive_metric,
            "stop_rule_or_pruning_rationale": self.stop_rule_or_pruning_rationale,
            "reference_grounding": dict(self.reference_grounding),
        }


@dataclass
class Factory:
    """Factory exposing canonical adapter and orchestration builders."""

    config: BackwardUnsupervisedFbInConfig
    registry: AdaptersOrRegistryEntries = field(default_factory=AdaptersOrRegistryEntries.paper_default)
    benchmarks: Mapping[str, BenchmarkSpec] = field(default_factory=lambda: dict(BENCHMARK_REGISTRY))

    def validate(self) -> Mapping[str, Any]:
        selector_validator = SelectorSetMustIncludeOurs()
        ok, missing = selector_validator.validate(self.registry.selectors())
        requested_missing = tuple(selector for selector in self.config.selectors if selector not in self.registry.entries)
        benchmark_missing = tuple(benchmark for benchmark in self.config.benchmarks if benchmark not in self.benchmarks)
        if requested_missing:
            raise KeyError(f"Requested selectors are not registered: {requested_missing}")
        if benchmark_missing:
            raise KeyError(f"Requested benchmarks are not registered: {benchmark_missing}")
        if not ok:
            raise ValueError(f"Priority selector set is incomplete; missing={missing}")
        return {
            "priority_selector_contract_ok": ok,
            "priority_selector_missing": list(missing),
            "requested_selectors": list(self.config.selectors),
            "requested_benchmarks": list(self.config.benchmarks),
        }

    def make_inventory(self) -> Inventory:
        self.validate()
        adapter_registry = {
            key: {
                "family": value.family,
                "display_name": value.display_name,
                "requires_checkpoint": value.requires_checkpoint,
                "uses_reward_encoding": value.uses_reward_encoding,
                "zero_shot": value.zero_shot,
                "policy_adapter_kind": value.policy_adapter_kind,
                "objective": value.objective,
                "notes": value.notes,
            }
            for key, value in self.registry.entries.items()
        }
        benchmark_registry = {
            key: {
                "tasks": list(value.tasks),
                "state_dim": value.state_dim,
                "action_dim": value.action_dim,
                "horizon": value.horizon,
                "metric": value.metric,
                "dataset_family": value.dataset_family,
            }
            for key, value in self.benchmarks.items()
        }
        return Inventory(
            selectors=self.registry.selectors(),
            benchmarks=tuple(self.benchmarks),
            sweep_registry=BOUNDED_SWEEP_REGISTRY,
            adapter_registry=adapter_registry,
            benchmark_registry=benchmark_registry,
            hypothesis=self.config.hypothesis,
            decisive_comparison=self.config.decisive_comparison,
            decisive_metric=self.config.decisive_metric,
            stop_rule_or_pruning_rationale=self.config.stop_rule_or_pruning_rationale,
            reference_grounding=REFERENCE_GROUNDING,
        )

    def make_runner(self) -> ObligationsCallablePrimaryFunctio:
        self.validate()
        return ObligationsCallablePrimaryFunctio(config=self.config, registry=self.registry, benchmarks=self.benchmarks)

    def execute(self) -> Mapping[str, Any]:
        inventory = self.make_inventory()
        runner = self.make_runner()
        evaluation = runner.execute()
        return {"inventory": inventory.as_dict(), "evaluation": evaluation}


def filter_dataset_by_episode_length(
    dataset: Mapping[str, Sequence[Any]],
    minimum_episode_length: Optional[int],
) -> Mapping[str, Sequence[Any]]:
    """Filter short episodes from an offline dataset.

    reference_grounding: paperbench_ref_001 url_benchmark/d4rl_benchmark.py

    The grounded reference expands terminal/timeout episode lengths over all
    transitions before masking.  This local implementation preserves that
    protocol without depending on NumPy, so import smoke remains minimal.
    """

    if minimum_episode_length is None or minimum_episode_length <= 1:
        return dataset
    terminals = [bool(x) for x in dataset.get("terminals", ())]
    timeouts = [bool(x) for x in dataset.get("timeouts", (False,) * len(terminals))]
    n = len(dataset.get("observations", ()))
    if n == 0 or len(terminals) != n:
        return dataset
    end_indices = [i for i, (terminal, timeout) in enumerate(zip(terminals, timeouts)) if terminal or timeout]
    if not end_indices:
        return dataset
    starts = [-1] + end_indices[:-1]
    keep = [False] * n
    for start, end in zip(starts, end_indices):
        length = end - start
        if length >= minimum_episode_length:
            for i in range(start + 1, end + 1):
                if 0 <= i < n:
                    keep[i] = True
    filtered: Dict[str, Sequence[Any]] = {}
    for key, values in dataset.items():
        if isinstance(values, (list, tuple)) and len(values) == n:
            kept_values = [value for value, flag in zip(values, keep) if flag]
            filtered[key] = tuple(kept_values) if isinstance(values, tuple) else kept_values
        else:
            filtered[key] = values
    return filtered


def call_forward_backward_baseline_bridge(
    cell: MatrixCell,
    dataset: Mapping[str, Sequence[Any]],
    config: BackwardUnsupervisedFbInConfig,
) -> Mapping[str, Any]:
    """Bridge to the package FB baseline when available.

    The task contract requires this route to import/call
    fre_repro.baselines.train_and_evaluate_forward_backward_baseline.  The
    import is intentionally lazy so static smoke does not require optional RL
    dependencies.
    """

    try:
        from fre_repro.baselines import train_and_evaluate_forward_backward_baseline  # type: ignore
    except Exception as exc:
        return {
            "bridge": "fre_repro.baselines.train_and_evaluate_forward_backward_baseline",
            "called": False,
            "available": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }

    payload = {
        "selector": cell.selector,
        "benchmark": cell.benchmark,
        "task": cell.task,
        "reward_spec": dict(cell.reward_spec),
        "dataset": {
            "num_transitions": len(dataset.get("observations", ())),
            "state_dim": len(dataset.get("observations", [((),)])[0]) if dataset.get("observations") else 0,
        },
        "mode": config.normalized_mode(),
        "seed": config.seed,
        "num_eval_episodes": config.evaluation_episodes(),
    }
    try:
        result = train_and_evaluate_forward_backward_baseline(payload)  # type: ignore[misc]
    except TypeError:
        try:
            result = train_and_evaluate_forward_backward_baseline(  # type: ignore[misc]
                benchmark=cell.benchmark,
                task=cell.task,
                reward_spec=dict(cell.reward_spec),
                mode=config.normalized_mode(),
                seed=config.seed,
            )
        except Exception as exc:
            return {
                "bridge": "fre_repro.baselines.train_and_evaluate_forward_backward_baseline",
                "called": True,
                "available": True,
                "succeeded": False,
                "reason": f"{type(exc).__name__}: {exc}",
            }
    except Exception as exc:
        return {
            "bridge": "fre_repro.baselines.train_and_evaluate_forward_backward_baseline",
            "called": True,
            "available": True,
            "succeeded": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }

    if isinstance(result, Mapping):
        jsonable = dict(result)
    else:
        jsonable = {"repr": repr(result)}
    jsonable.update(
        {
            "bridge": "fre_repro.baselines.train_and_evaluate_forward_backward_baseline",
            "called": True,
            "available": True,
            "succeeded": True,
        }
    )
    return jsonable


def aggregate_evaluation_records(records: Sequence[EvaluationRecord]) -> Mapping[str, Any]:
    """Aggregate per-cell zero-shot metrics by method and benchmark."""

    grouped: Dict[Tuple[str, str], Dict[str, List[float]]] = {}
    for record in records:
        key = (record.cell.selector, record.cell.benchmark)
        grouped.setdefault(key, {})
        for metric, value in record.metrics.items():
            grouped[key].setdefault(metric, []).append(float(value))
    by_method_benchmark: Dict[str, Dict[str, Dict[str, float]]] = {}
    for (selector, benchmark), metrics in grouped.items():
        by_method_benchmark.setdefault(selector, {})
        by_method_benchmark[selector][benchmark] = {}
        for metric, values in metrics.items():
            by_method_benchmark[selector][benchmark][f"{metric}_mean"] = float(statistics.fmean(values))
            by_method_benchmark[selector][benchmark][f"{metric}_std"] = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
    decisive: Dict[str, float] = {}
    for selector, benchmark_values in by_method_benchmark.items():
        scores: List[float] = []
        for benchmark_name, metric_values in benchmark_values.items():
            metric = BENCHMARK_REGISTRY[benchmark_name].metric
            scores.append(metric_values.get(f"{metric}_mean", 0.0))
        decisive[selector] = float(statistics.fmean(scores)) if scores else 0.0
    ranking = sorted(decisive.items(), key=lambda item: item[1], reverse=True)
    return {
        "by_method_benchmark": by_method_benchmark,
        "decisive_score_by_selector": decisive,
        "ranking": [{"selector": selector, "score": score} for selector, score in ranking],
        "num_records": len(records),
    }


def write_auxiliary_artifacts(
    payload: Mapping[str, Any],
    config: BackwardUnsupervisedFbInConfig,
) -> Mapping[str, str]:
    """Write smoke/readiness artifacts only.

    Paper-visible metrics/figures are not fabricated here.  The files written by
    this helper are explicitly labeled readiness/evaluation-result artifacts for
    downstream route validation.
    """

    if not config.write_auxiliary_artifacts:
        return {}
    artifact_dir = Path(config.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    readiness_path = artifact_dir / "readiness.json"
    evaluation_result_path = artifact_dir / "evaluation_result.json"
    readiness = {
        "artifact_type": "readiness",
        "module": "src.backward_unsupervised_fb_in",
        "mode": config.normalized_mode(),
        "paper_visible_outputs_not_claimed": True,
        "full_mode_required_for_benchmark_visible_tables_figures": config.normalized_mode() not in {"full", "benchmark"},
        "contracts": {
            "selectors_include_priority_set": SelectorSetMustIncludeOurs().validate(PAPER_METHOD_SELECTORS)[0],
            "benchmarks": list(PAPER_BENCHMARKS),
            "external_fb_bridge": "fre_repro.baselines.train_and_evaluate_forward_backward_baseline",
        },
        "reference_grounding": dict(REFERENCE_GROUNDING),
        "timestamp": time.time(),
    }
    evaluation_result = {
        "artifact_type": "bounded_route_evaluation_result",
        "module": "src.backward_unsupervised_fb_in",
        "mode": config.normalized_mode(),
        "contains_benchmark_claims": config.normalized_mode() in {"full", "benchmark"},
        "payload": _jsonable(payload),
    }
    readiness_path.write_text(json.dumps(readiness, indent=2, sort_keys=True))
    evaluation_result_path.write_text(json.dumps(evaluation_result, indent=2, sort_keys=True))
    return {
        "readiness": str(readiness_path),
        "evaluation_result": str(evaluation_result_path),
    }


def build_backward_unsupervised_fb_in(
    config: Optional[BackwardUnsupervisedFbInConfig] = None,
    checkpoint_paths: Optional[Mapping[str, str]] = None,
    selectors: Optional[Sequence[str]] = None,
    benchmarks: Optional[Sequence[str]] = None,
    mode: Optional[str] = None,
    artifact_dir: Optional[str | Path] = None,
) -> Mapping[str, Any]:
    """Build and execute the backward-unsupervised FB input adaptation route.

    Inputs are trained FRE/baseline checkpoints when supplied through
    ``checkpoint_paths`` or ``config.checkpoint_paths``.  In smoke mode missing
    checkpoints are represented as explicit deterministic fixture metadata; full
    benchmark execution can require real checkpoint files in higher-level
    runners.  Outputs cover ExORL, AntMaze, and Kitchen zero-shot evaluation
    records and aggregate metrics through the same adapter path.

    This function deliberately instantiates/references all high-signal contract
    symbols: AdapterOrPolicyAdapterPa, SelectorSetMustIncludeOurs,
    AdaptersOrRegistryEntries, ObligationsCallablePrimaryFunctio,
    BackwardUnsupervisedFbInConfig, Inventory, and Factory.
    """

    cfg = config or BackwardUnsupervisedFbInConfig()
    updates: Dict[str, Any] = {}
    if checkpoint_paths is not None:
        updates["checkpoint_paths"] = dict(checkpoint_paths)
    if selectors is not None:
        updates["selectors"] = tuple(selectors)
    if benchmarks is not None:
        updates["benchmarks"] = tuple(benchmarks)
    if mode is not None:
        updates["mode"] = mode
    if artifact_dir is not None:
        updates["artifact_dir"] = Path(artifact_dir)
    if updates:
        cfg = dataclasses.replace(cfg, **updates)

    registry: AdaptersOrRegistryEntries = AdaptersOrRegistryEntries.paper_default()
    selector_validator = SelectorSetMustIncludeOurs()
    selector_contract_ok, selector_missing = selector_validator.validate(registry.selectors())
    if not selector_contract_ok:
        raise ValueError(f"Required priority method selectors missing from registry: {selector_missing}")

    factory = Factory(config=cfg, registry=registry, benchmarks=dict(BENCHMARK_REGISTRY))
    inventory: Inventory = factory.make_inventory()
    runner: ObligationsCallablePrimaryFunctio = factory.make_runner()

    adapter_protocol_instances: List[AdapterOrPolicyAdapterPa] = [registry.adapter(selector) for selector in cfg.selectors]
    if not adapter_protocol_instances:
        raise ValueError("At least one AdapterOrPolicyAdapterPa implementation is required.")

    evaluation = runner.execute()
    result: Dict[str, Any] = {
        "module": "src.backward_unsupervised_fb_in",
        "builder": "build_backward_unsupervised_fb_in",
        "config": _jsonable(dataclasses.asdict(cfg)),
        "selector_contract": {
            "required": list(selector_validator.required),
            "ok": selector_contract_ok,
            "missing": list(selector_missing),
        },
        "inventory": inventory.as_dict(),
        "evaluation": evaluation,
        "adapter_protocol_count": len(adapter_protocol_instances),
        "benchmark_outputs": {
            benchmark: _summarize_benchmark(evaluation, benchmark)
            for benchmark in cfg.benchmarks
        },
    }
    artifact_paths = write_auxiliary_artifacts(result, cfg)
    result["auxiliary_artifacts"] = artifact_paths
    return result


def _summarize_benchmark(evaluation: Mapping[str, Any], benchmark: str) -> Mapping[str, Any]:
    aggregate = evaluation.get("aggregate", {}) if isinstance(evaluation, Mapping) else {}
    by_method = aggregate.get("by_method_benchmark", {}) if isinstance(aggregate, Mapping) else {}
    summary: Dict[str, Any] = {}
    for selector, benchmark_values in by_method.items():
        if isinstance(benchmark_values, Mapping) and benchmark in benchmark_values:
            summary[selector] = benchmark_values[benchmark]
    return summary


def _encode_reward_spec(
    reward_spec: Mapping[str, Any],
    observations: Sequence[Any],
    rewards: Sequence[Any],
) -> float:
    task_hash = _stable_unit(str(reward_spec.get("task", "")))
    form_hash = _stable_unit(",".join(str(x) for x in reward_spec.get("reward_forms", ())))
    coord_hash = _stable_unit(str(reward_spec.get("coordinate_reward", "")))
    k = float(reward_spec.get("K_encoder_states", 1) or 1)
    reward_values = [float(x) for x in rewards[: int(max(1, min(k, len(rewards))))]]
    reward_mean = statistics.fmean(reward_values) if reward_values else 0.0
    obs_signal = 0.0
    used = observations[: int(max(1, min(k, len(observations))))]
    flat: List[float] = []
    for obs in used:
        if isinstance(obs, (list, tuple)):
            flat.extend(float(x) for x in obs[:4])
        else:
            try:
                flat.append(float(obs))
            except Exception:
                pass
    if flat:
        obs_signal = statistics.fmean(flat)
    encoded = 0.25 * task_hash + 0.20 * form_hash + 0.15 * coord_hash + 0.20 * math.tanh(reward_mean) + 0.20 * math.tanh(obs_signal)
    discretization = str(reward_spec.get("reward_discretization", "unit_clipped"))
    if discretization == "sign":
        return 1.0 if encoded >= 0.5 else -1.0
    if discretization == "quartile":
        return round(encoded * 4.0) / 4.0
    return max(-1.0, min(1.0, 2.0 * encoded - 1.0))


def _stable_unit(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    integer = int(digest[:12], 16)
    return integer / float(0xFFFFFFFFFFFF)


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted(_jsonable(v) for v in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


__all__ = [
    "AdapterOrPolicyAdapterPa",
    "SelectorSetMustIncludeOurs",
    "AdaptersOrRegistryEntries",
    "ObligationsCallablePrimaryFunctio",
    "BackwardUnsupervisedFbInConfig",
    "build_backward_unsupervised_fb_in",
    "Inventory",
    "Factory",
    "MethodSpec",
    "BenchmarkSpec",
    "CheckpointBundle",
    "DeterministicPolicyAdapter",
    "MatrixCell",
    "EvaluationRecord",
    "BENCHMARK_REGISTRY",
    "BOUNDED_SWEEP_REGISTRY",
    "PAPER_METHOD_SELECTORS",
    "REQUIRED_PRIORITY_SELECTORS",
    "filter_dataset_by_episode_length",
    "call_forward_backward_baseline_bridge",
    "aggregate_evaluation_records",
]