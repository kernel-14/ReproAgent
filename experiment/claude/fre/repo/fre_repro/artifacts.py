"""Artifact, registry, and result-writing surfaces for FRE reproduction.

This module owns the paper-visible artifact contract for
"Unsupervised Zero-Shot Reinforcement Learning via Functional Reward Encodings"
(FRE).  It is intentionally importable in a minimal Python environment: no
simulator, RL, GPU, plotting, dataframe, or dataset package is imported at module
import time.

The artifact route is conservative:

* paper-visible performance artifacts (Table 1, Table 4, metrics.json,
  predictions.jsonl, result figures) are written only from supplied measured
  evaluation records or from an explicitly executed bounded evaluator;
* default smoke/dry-run calls write readiness and evaluation_result manifests
  that label missing benchmark measurements as not-run;
* all artifact paths, schemas, captions, baseline semantics, and experiment
  matrices are statically discoverable through registries and manifest writers.

Reference grounding adapted into this implementation:
  reference_grounding: paperbench_ref_001 url_benchmark/d4rl_benchmark.py
  reference_grounding: paperbench_ref_001 controllable_agent/test_executor.py
  reference_grounding: paperbench_ref_001 controllable_agent/test_url_benchmark.py

The first grounding informs the episode/task bookkeeping and filtering helpers
for offline benchmark records.  The executor/test grounding informs the bounded
smoke route: the same public registries and writers are exercised without
submitting expensive jobs or claiming results.  The anytrain-grid reference
intent is preserved as a tiny default protocol that validates wiring with
bounded inputs, CPU-safe execution, and explicit "not benchmark" status.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import math
import os
import statistics
import struct
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple


Number = float
MetricRecord = Mapping[str, Any]


# ---------------------------------------------------------------------------
# Paper-derived artifact and protocol declarations.
# ---------------------------------------------------------------------------

PAPER_TITLE = "Unsupervised Zero-Shot Reinforcement Learning via Functional Reward Encodings"

NAMED_METHODS: Tuple[str, ...] = (
    "FRE",
    "FB",
    "SF",
    "GCRL",
    "GC-IQL",
    "GC-BC",
    "CRL",
    "OPAL",
)

MAIN_BENCHMARKS: Tuple[str, ...] = (
    "antmaze",
    "exorl",
    "kitchen",
)

RANDOM_REWARD_FAMILIES: Tuple[str, ...] = (
    "goal_reaching",
    "linear",
    "mlp",
)

METRIC_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "return": {
        "type": "number",
        "aggregation": ["mean", "std", "count", "stderr"],
        "higher_is_better": True,
        "paper_context": "policy return under encoded task / expected downstream return",
    },
    "normalized_return": {
        "type": "number",
        "aggregation": ["mean", "std", "count", "stderr"],
        "higher_is_better": True,
        "paper_context": "zero-shot benchmark performance on AntMaze, ExORL, and Kitchen",
    },
    "success_rate": {
        "type": "number",
        "aggregation": ["mean", "std", "count", "stderr"],
        "higher_is_better": True,
        "paper_context": "goal/task completion accuracy",
    },
    "decoded_reward_mse": {
        "type": "number",
        "aggregation": ["mean", "std", "count", "stderr"],
        "higher_is_better": False,
        "paper_context": "decoded reward should preserve functional similarity",
    },
    "decoded_reward_correlation": {
        "type": "number",
        "aggregation": ["mean", "std", "count", "stderr"],
        "higher_is_better": True,
        "paper_context": "reward reconstruction/functional similarity",
    },
    "value_loss": {
        "type": "number",
        "aggregation": ["mean", "std", "count", "stderr"],
        "higher_is_better": False,
        "paper_context": "estimated value function consistency",
    },
    "policy_return": {
        "type": "number",
        "aggregation": ["mean", "std", "count", "stderr"],
        "higher_is_better": True,
        "paper_context": "executed latent-conditioned policy performance",
    },
    "reward_loss": {
        "type": "number",
        "aggregation": ["mean", "std", "count", "stderr"],
        "higher_is_better": False,
        "paper_context": "reward decoder training/evaluation loss",
    },
    "accuracy": {
        "type": "number",
        "aggregation": ["mean", "std", "count", "stderr"],
        "higher_is_better": True,
        "paper_context": "generic task success/selection accuracy schema",
    },
    "loss": {
        "type": "number",
        "aggregation": ["mean", "std", "count", "stderr"],
        "higher_is_better": False,
        "paper_context": "generic optimization loss schema",
    },
}

FIGURE_AND_TABLE_SPECS: Dict[str, Dict[str, Any]] = {
    "figure_1": {
        "path": "results/figures/figure_1.png",
        "kind": "figure",
        "caption": (
            "Figure 1. FRE discovers latent representations over random "
            "unsupervised reward functions; at evaluation, user-given "
            "downstream objectives are encoded into the latent space for "
            "zero-shot policy execution."
        ),
        "requires_measurements": False,
    },
    "figure_2": {
        "path": "results/figures/figure_2.png",
        "kind": "figure",
        "caption": (
            "Figure 2. A reward function is encoded by evaluating it on a "
            "random set of offline data states; the (state, reward) pairs are "
            "processed by a permutation-invariant encoder."
        ),
        "requires_measurements": False,
    },
    "figure_3": {
        "path": "results/figures/figure_3.png",
        "aliases": ["results/fig3_zero_shot_transfer.png", "results/figures/figure3.png"],
        "kind": "figure",
        "caption": (
            "Figure 3. Zero-shot AntMaze examples: true reward, sampled "
            "encoder states, decoded reward, estimated value function, and "
            "executed policy trajectory."
        ),
        "requires_measurements": True,
    },
    "figure_4": {
        "path": "results/figures/figure_4.png",
        "kind": "figure",
        "caption": "Figure 4. Evaluation domains: AntMaze, ExORL, and Kitchen.",
        "requires_measurements": False,
    },
    "figure_5": {
        "path": "results/figures/figure_5.png",
        "kind": "figure",
        "caption": (
            "Figure 5. FRE capability scales with diversity of random reward "
            "families; FRE-all is compared against agents trained on subsets."
        ),
        "requires_measurements": True,
    },
    "figure_6": {
        "path": "results/figures/figure_6.png",
        "kind": "figure",
        "caption": (
            "Figure 6. Domain-specific reward distributions can be added to "
            "the random reward prior without algorithmic changes."
        ),
        "requires_measurements": True,
    },
    "figure_7": {
        "path": "results/figures/figure_7.png",
        "kind": "figure",
        "caption": (
            "Figure 7. Additional AntMaze examples: true reward, predicted "
            "reward, Q1, encoder states, policy trajectory, Q2."
        ),
        "requires_measurements": True,
    },
    "figure_8": {
        "path": "results/figures/figure_8.png",
        "kind": "figure",
        "caption": (
            "Figure 8. Additional AntMaze examples: true reward, predicted "
            "reward, Q1, encoder states, policy trajectory, Q2."
        ),
        "requires_measurements": True,
    },
    "figure_9": {
        "path": "results/figures/figure_9.png",
        "kind": "figure",
        "caption": (
            "Figure 9. Additional AntMaze examples: true reward, predicted "
            "reward, Q1, encoder states, policy trajectory, Q2."
        ),
        "requires_measurements": True,
    },
    "table_1": {
        "path": "results/tables/table_1.csv",
        "kind": "table",
        "caption": (
            "Table 1. Offline zero-shot RL comparisons on AntMaze, ExORL, and "
            "Kitchen for FRE and prior methods including FB, SF, GCRL, OPAL, "
            "and CRL-style baselines."
        ),
        "requires_measurements": True,
    },
    "table_2": {
        "path": "results/tables/table_2.csv",
        "kind": "table",
        "caption": (
            "Table 2. FRE unifies prior method capabilities: OPAL lacks "
            "zero-shot capability and uses BC; GCRL/SF restrict reward "
            "families; FB handles broad rewards with linearized value form."
        ),
        "requires_measurements": False,
    },
    "table_3": {
        "path": "results/tables/table_3.csv",
        "kind": "table",
        "caption": "Table 3. Hyperparameters used for FRE.",
        "requires_measurements": False,
    },
    "table_4": {
        "path": "results/tables/table_4.csv",
        "kind": "table",
        "caption": (
            "Table 4. Full AntMaze results comparing FRE agents trained on "
            "different subsets of random reward functions."
        ),
        "requires_measurements": True,
    },
    "result_figure": {
        "path": "results/figures/experiment_results.png",
        "kind": "figure",
        "caption": "Aggregate measured zero-shot comparison results.",
        "requires_measurements": True,
    },
    "predictions": {
        "path": "results/predictions.jsonl",
        "kind": "jsonl",
        "caption": "Per-task/per-episode decoded reward, value, and policy execution records.",
        "requires_measurements": True,
    },
    "metrics_json": {
        "path": "results/metrics.json",
        "kind": "json",
        "caption": "Metric summary and aggregation outputs.",
        "requires_measurements": True,
    },
    "result_table": {
        "path": "results/tables/experiment_results.csv",
        "kind": "table",
        "caption": "Long-form measured evaluation result table.",
        "requires_measurements": True,
    },
    "config": {
        "path": "results/config_resolved.json",
        "kind": "json",
        "caption": "Resolved experiment and artifact configuration.",
        "requires_measurements": False,
    },
    "log": {
        "path": "results/run_log.jsonl",
        "kind": "jsonl",
        "caption": "Execution log for artifact-producing routes.",
        "requires_measurements": False,
    },
}

TREND_ASSERTIONS: Dict[str, Dict[str, Any]] = {
    "functional_similarity": {
        "claim": "reward encoding should preserve functional similarity under latent compression",
        "evidence_metrics": ["decoded_reward_mse", "decoded_reward_correlation"],
        "decision_rule": "decoded_reward_mse decreases and correlation increases relative to unconditioned/random decoders",
    },
    "latent_policy_return": {
        "claim": "latent-conditioned policy should maximize expected return for tasks within the prior reward distribution",
        "evidence_metrics": ["return", "normalized_return", "policy_return"],
        "decision_rule": "FRE return is competitive with or exceeds explicit baselines on matched benchmarks",
    },
    "baseline_outperformance": {
        "claim": "FRE should be compared against explicit baselines and should match or outperform prior methods where reported",
        "evidence_metrics": ["normalized_return", "success_rate"],
        "baselines": ["FB", "SF", "GCRL", "GC-IQL", "GC-BC", "CRL", "OPAL"],
        "decision_rule": "compare FRE aggregate mean to best available baseline per benchmark/task",
    },
    "zero_shot_transfer": {
        "claim": "FRE should generalize from randomly annotated states to unseen test tasks",
        "evidence_metrics": ["decoded_reward_mse", "value_loss", "return"],
        "decision_rule": "evaluate tasks unseen during unsupervised reward-prior training without fine-tuning",
    },
    "reward_family_scaling": {
        "claim": "more reward families may improve generalization or hurt via capacity/forgetting",
        "evidence_metrics": ["normalized_return", "success_rate"],
        "decision_rule": "compare all-family FRE to bounded singleton/subset family ablations",
    },
    "domain_prior_specificity": {
        "claim": "more specific priors should increase encoding specificity on matching downstream tasks",
        "evidence_metrics": ["decoded_reward_mse", "return"],
        "decision_rule": "domain-augmented prior improves matched-task metrics without changing FRE algorithm",
    },
    "universal_multitask": {
        "claim": "FRE should remain universal enough to operate as a multi-task RL method",
        "evidence_metrics": ["normalized_return"],
        "decision_rule": "report aggregate across AntMaze, ExORL, and Kitchen when all measured records exist",
    },
}

PAPER_HYPERPARAMETERS: Dict[str, Any] = {
    "batch_size": 512,
    "encoder_training_steps": "150000 AntMaze; 1000000 ExORL/Kitchen",
    "policy_training_steps": "850000 AntMaze; 1000000 ExORL/Kitchen",
    "reward_pairs_to_encode": 32,
    "reward_pairs_to_decode": 8,
    "ratio_goal_reaching_rewards": 0.33,
    "ratio_linear_rewards": 0.33,
    "ratio_random_mlp_rewards": 0.33,
    "number_of_reward_embeddings": 32,
    "reward_embedding_dim": 128,
    "optimizer": "Adam",
    "learning_rate": 0.0001,
    "rl_network_layers": [512, 512, 512],
    "decoder_network_layers": [512, 512, 512],
    "encoder_layers": [256, 256, 256, 256],
    "encoder_attention_heads": 4,
    "beta_kl_weight": 0.01,
    "target_update_rate": 0.001,
    "discount_factor": 0.88,
    "awr_temperature": 3.0,
    "iql_expectile": 0.8,
}

EVIDENCE_OBLIGATION_MATRIX: Tuple[Dict[str, Any], ...] = (
    {
        "paper_section": "4.2 Random Functions as a Prior Reward Distribution",
        "method": "FRE",
        "environment": "offline unlabeled trajectories",
        "parameters": {"reward_families": list(RANDOM_REWARD_FAMILIES), "reward_pairs_to_encode": 32},
        "trend": "prior over reward functions eta from unlabeled trajectories supports functional compression",
        "artifacts": ["results/reward_prior_config.json", "results/config_resolved.json", "results/metrics.json"],
    },
    {
        "paper_section": "4.3 Offline RL with FRE",
        "method": "FRE",
        "environment": "AntMaze/ExORL/Kitchen offline datasets",
        "parameters": {"sample_eta": True, "encode_z": True, "policy": "pi(a|s,z)"},
        "trend": "latent-conditioned policy maximizes return under encoded task",
        "artifacts": ["results/checkpoints/fre_encoder.pt", "results/checkpoints/fre_policy.pt", "results/metrics.json"],
    },
    {
        "paper_section": "5.1 Zero-shot transfer",
        "method": "FRE",
        "environment": "AntMaze",
        "parameters": {"examples": "32 state-reward pairs", "fine_tuning": False},
        "trend": "decoded reward, value function, and executed policy generalize to unseen tasks",
        "artifacts": ["results/figures/figure_3.png", "results/figures/figure_7.png", "results/figures/figure_8.png", "results/figures/figure_9.png"],
    },
    {
        "paper_section": "5.2 Zero-shot offline RL benchmark comparison",
        "method": "FRE vs FB/SF/GCRL/GC-IQL/GC-BC/CRL/OPAL",
        "environment": "AntMaze, ExORL, Kitchen",
        "parameters": {"episodes": 20, "seeds": 5, "metrics": ["return", "normalized_return", "success_rate"]},
        "trend": "FRE should be competitive with prior unsupervised RL methods",
        "artifacts": ["results/tables/table_1.csv", "results/metrics.json", "results/eval_summary.json"],
    },
    {
        "paper_section": "5.3 Reward-prior scaling",
        "method": "FRE reward-family subset ablations",
        "environment": "AntMaze",
        "parameters": {"families": "all possible non-empty subsets of goal/linear/mlp, bounded smoke subset by default"},
        "trend": "reward-space diversity changes general capability via generalization/capacity tradeoff",
        "artifacts": ["results/figures/figure_5.png", "results/tables/table_4.csv", "results/trends.json"],
    },
    {
        "paper_section": "Domain-specific priors",
        "method": "FRE with augmented specific reward distributions",
        "environment": "AntMaze, ExORL, Kitchen",
        "parameters": {"algorithm_change": False, "prior_augmented": True},
        "trend": "specific priors increase matching downstream specificity",
        "artifacts": ["results/figures/figure_6.png", "results/trends.json"],
    },
)


# ---------------------------------------------------------------------------
# Dataclasses.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BenchmarkTask:
    """A zero-shot evaluation task sampled from a benchmark registry."""

    benchmark: str
    task_id: str
    reward_family: str
    objective: str
    max_episodes: int = 20
    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkSpec:
    """Benchmark protocol declaration for AntMaze, ExORL, or Kitchen."""

    benchmark: str
    display_name: str
    suite: str
    task_ids: Tuple[str, ...]
    metric_names: Tuple[str, ...]
    default_reward_families: Tuple[str, ...] = RANDOM_REWARD_FAMILIES
    notes: str = ""


@dataclass(frozen=True)
class BaselineSpec:
    """Baseline/method comparison semantics."""

    method: str
    family: str
    zero_shot: bool
    policy_learning: str
    reward_family_assumption: str
    comparison_role: str
    checkpoint_keys: Tuple[str, ...] = ()


@dataclass
class ArtifactsLayout:
    """Stable output layout for FRE artifact writers.

    Primary paper-visible paths live under ``root``.  Auxiliary readiness and
    smoke outputs additionally respect ``PAPERBENCH_REPRO_ARTIFACT_DIR`` when it
    is set, so benchmark harnesses can collect validation artifacts without
    moving the canonical result tree.
    """

    root: Path = Path("results")
    auxiliary_root: Optional[Path] = None

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        if self.auxiliary_root is None:
            env_dir = os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR")
            self.auxiliary_root = Path(env_dir) if env_dir else self.root
        else:
            self.auxiliary_root = Path(self.auxiliary_root)

    @property
    def figures_dir(self) -> Path:
        return self.root / "figures"

    @property
    def tables_dir(self) -> Path:
        return self.root / "tables"

    @property
    def checkpoints_dir(self) -> Path:
        return self.root / "checkpoints"

    @property
    def metrics_path(self) -> Path:
        return self.root / "metrics.json"

    @property
    def eval_summary_path(self) -> Path:
        return self.root / "eval_summary.json"

    @property
    def evaluation_result_path(self) -> Path:
        return self.root / "evaluation_result.json"

    @property
    def readiness_path(self) -> Path:
        return Path(self.auxiliary_root or self.root) / "readiness.json"

    @property
    def artifact_manifest_path(self) -> Path:
        return self.root / "artifact_manifest.json"

    @property
    def experiment_registry_path(self) -> Path:
        return self.root / "experiment_registry.json"

    @property
    def model_registry_path(self) -> Path:
        return self.root / "model_registry.json"

    @property
    def trends_path(self) -> Path:
        return self.root / "trends.json"

    @property
    def predictions_path(self) -> Path:
        return self.root / "predictions.jsonl"

    @property
    def resolved_config_path(self) -> Path:
        return self.root / "config_resolved.json"

    def path_for(self, relative_or_key: str) -> Path:
        spec = FIGURE_AND_TABLE_SPECS.get(relative_or_key)
        if spec is not None:
            return Path(spec["path"])
        path = Path(relative_or_key)
        if path.is_absolute():
            return path
        if str(path).startswith("results/"):
            return path
        return self.root / path

    def ensure_dirs(self) -> None:
        for directory in (
            self.root,
            self.figures_dir,
            self.tables_dir,
            self.checkpoints_dir,
            Path(self.auxiliary_root or self.root),
        ):
            directory.mkdir(parents=True, exist_ok=True)


@dataclass
class ArtifactsResult:
    """Return object for artifact-producing evaluation routes."""

    status: str
    layout: ArtifactsLayout
    metrics: Dict[str, Any] = field(default_factory=dict)
    aggregate_metrics: Dict[str, Any] = field(default_factory=dict)
    written_artifacts: List[str] = field(default_factory=list)
    manifest_path: Optional[str] = None
    readiness_path: Optional[str] = None
    evaluation_result_path: Optional[str] = None
    registry_paths: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "metrics": self.metrics,
            "aggregate_metrics": self.aggregate_metrics,
            "written_artifacts": list(self.written_artifacts),
            "manifest_path": self.manifest_path,
            "readiness_path": self.readiness_path,
            "evaluation_result_path": self.evaluation_result_path,
            "registry_paths": dict(self.registry_paths),
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# Registries and task sampling.
# ---------------------------------------------------------------------------

def build_benchmark_registry() -> Dict[str, BenchmarkSpec]:
    """Return statically discoverable ExORL/AntMaze/Kitchen protocols."""

    return {
        "antmaze": BenchmarkSpec(
            benchmark="antmaze",
            display_name="AntMaze",
            suite="D4RL AntMaze",
            task_ids=(
                "antmaze-large-diverse-v2-goal",
                "antmaze-large-diverse-v2-navigate",
                "antmaze-large-diverse-v2-path",
            ),
            metric_names=("success_rate", "normalized_return", "return", "decoded_reward_mse", "value_loss"),
            notes="Figure 3/7/8/9 diagnostics and Table 1/Table 4 comparisons.",
        ),
        "exorl": BenchmarkSpec(
            benchmark="exorl",
            display_name="ExORL",
            suite="ExORL locomotion",
            task_ids=(
                "walker-walk",
                "walker-run",
                "cheetah-run",
                "quadruped-walk",
            ),
            metric_names=("normalized_return", "return", "policy_return"),
            notes="Table 1 comparison against prior zero-shot/unsupervised RL methods.",
        ),
        "kitchen": BenchmarkSpec(
            benchmark="kitchen",
            display_name="Kitchen",
            suite="D4RL Kitchen",
            task_ids=(
                "kitchen-mixed-v0",
                "kitchen-partial-v0",
                "kitchen-complete-v0",
            ),
            metric_names=("success_rate", "normalized_return", "return"),
            notes="Structured manipulation downstream objectives for Table 1.",
        ),
    }


def build_baseline_registry() -> Dict[str, BaselineSpec]:
    """Return named method and baseline adapters used in comparisons."""

    return {
        "FRE": BaselineSpec(
            method="FRE",
            family="functional_reward_encoding",
            zero_shot=True,
            policy_learning="latent-conditioned offline RL / IQL-AWR style policy extraction",
            reward_family_assumption="random goal, sparse linear, and random MLP reward functions; domain priors optional",
            comparison_role="proposed method",
            checkpoint_keys=("fre_encoder", "fre_policy", "fre_decoder", "fre_value"),
        ),
        "FB": BaselineSpec(
            method="FB",
            family="forward_backward",
            zero_shot=True,
            policy_learning="forward-backward representation with task-conditioned policy",
            reward_family_assumption="broad rewards through linearized value function",
            comparison_role="state-of-the-art zero-shot baseline",
            checkpoint_keys=("fb_forward", "fb_backward", "fb_policy"),
        ),
        "SF": BaselineSpec(
            method="SF",
            family="successor_features",
            zero_shot=True,
            policy_learning="successor features with linear reward weights",
            reward_family_assumption="linear reward functions",
            comparison_role="linear reward baseline",
            checkpoint_keys=("sf_features", "sf_policy"),
        ),
        "GCRL": BaselineSpec(
            method="GCRL",
            family="goal_conditioned_rl",
            zero_shot=True,
            policy_learning="goal-conditioned offline RL",
            reward_family_assumption="goal-reaching rewards",
            comparison_role="restricted reward family baseline",
            checkpoint_keys=("gcrl_policy",),
        ),
        "GC-IQL": BaselineSpec(
            method="GC-IQL",
            family="goal_conditioned_iql",
            zero_shot=True,
            policy_learning="goal-conditioned IQL",
            reward_family_assumption="goal-reaching rewards",
            comparison_role="goal-conditioned offline RL baseline",
            checkpoint_keys=("gciql_policy",),
        ),
        "GC-BC": BaselineSpec(
            method="GC-BC",
            family="goal_conditioned_behavior_cloning",
            zero_shot=True,
            policy_learning="behavior cloning with geometric goal sampling",
            reward_family_assumption="goal-reaching rewards",
            comparison_role="behavior-cloning goal baseline",
            checkpoint_keys=("gcbc_policy",),
        ),
        "CRL": BaselineSpec(
            method="CRL",
            family="contrastive_rl",
            zero_shot=True,
            policy_learning="contrastive representation learning for control",
            reward_family_assumption="implicit goal/task embeddings",
            comparison_role="contrastive unsupervised RL baseline",
            checkpoint_keys=("crl_encoder", "crl_policy"),
        ),
        "OPAL": BaselineSpec(
            method="OPAL",
            family="offline_primitive_discovery",
            zero_shot=False,
            policy_learning="behavior cloning / primitive latent policy",
            reward_family_assumption="does not encode arbitrary downstream rewards zero-shot",
            comparison_role="non-zero-shot offline skill baseline",
            checkpoint_keys=("opal_policy",),
        ),
    }


def build_experiment_registry(full: bool = False) -> Dict[str, Any]:
    """Build the experiment registry with bounded defaults and full selectors."""

    benchmarks = build_benchmark_registry()
    baselines = build_baseline_registry()
    bounded_methods = ("FRE", "FB", "SF", "CRL")
    bounded_benchmarks = ("antmaze", "exorl", "kitchen")
    scaling_subsets = [
        ("goal_reaching",),
        ("linear",),
        ("mlp",),
        ("goal_reaching", "linear", "mlp"),
    ]
    if full:
        methods = tuple(baselines.keys())
        scaling_subsets = [
            tuple(fam for i, fam in enumerate(RANDOM_REWARD_FAMILIES) if mask & (1 << i))
            for mask in range(1, 1 << len(RANDOM_REWARD_FAMILIES))
        ]
    else:
        methods = bounded_methods

    return {
        "paper": PAPER_TITLE,
        "hypothesis": (
            "FRE encodes unseen reward functions from a small set of "
            "state-reward examples and executes the corresponding policy "
            "zero-shot on AntMaze, ExORL, and Kitchen."
        ),
        "decision_value": (
            "The decisive comparison is FRE versus explicit FB/SF/GCRL/CRL/OPAL "
            "families on normalized return/success plus decoded reward/value diagnostics."
        ),
        "stop_rule_or_pruning_rationale": (
            "Default routes execute bounded smoke/diagnostic subsets only. Full "
            "20-episode x 5-seed benchmark evaluation and exhaustive reward-family "
            "ablations require explicit full mode and real checkpoints/datasets."
        ),
        "benchmarks": {k: dataclasses.asdict(v) for k, v in benchmarks.items() if k in bounded_benchmarks},
        "methods": {k: dataclasses.asdict(v) for k, v in baselines.items() if k in methods},
        "protocols": {
            "main_table_1": {
                "benchmarks": list(bounded_benchmarks),
                "methods": list(methods),
                "metrics": ["normalized_return", "success_rate", "return"],
                "episodes": 20,
                "seeds": [0, 1, 2, 3, 4],
                "artifact": "results/tables/table_1.csv",
            },
            "figure_3_zero_shot_transfer": {
                "benchmarks": ["antmaze"],
                "methods": ["FRE"],
                "metrics": ["decoded_reward_mse", "decoded_reward_correlation", "value_loss", "policy_return"],
                "reward_pairs_to_encode": 32,
                "artifact": "results/figures/figure_3.png",
            },
            "figure_5_scaling": {
                "benchmarks": ["antmaze"],
                "methods": ["FRE"],
                "reward_family_subsets": [list(x) for x in scaling_subsets],
                "artifact": "results/figures/figure_5.png",
            },
            "figure_6_domain_priors": {
                "benchmarks": list(bounded_benchmarks),
                "methods": ["FRE"],
                "comparison": "generic random priors vs domain-augmented priors",
                "artifact": "results/figures/figure_6.png",
            },
        },
        "trend_assertions": TREND_ASSERTIONS,
        "metric_schemas": METRIC_SCHEMAS,
        "artifact_specs": FIGURE_AND_TABLE_SPECS,
        "evidence_obligation_matrix": list(EVIDENCE_OBLIGATION_MATRIX),
    }


def sample_tasks(
    benchmarks: Optional[Sequence[str]] = None,
    *,
    seed: int = 0,
    max_tasks_per_benchmark: int = 2,
    full: bool = False,
) -> List[BenchmarkTask]:
    """Sample benchmark tasks for bounded or full evaluation.

    The default is intentionally small, matching the repository smoke route.
    Full mode exposes all registered tasks but still does not run environments
    unless a downstream evaluator supplies checkpoints/datasets.
    """

    registry = build_benchmark_registry()
    selected = list(benchmarks or registry.keys())
    rng_state = hashlib.sha256(f"fre-task-sampler:{seed}".encode("utf-8")).digest()
    offset = rng_state[0]
    tasks: List[BenchmarkTask] = []
    for bench in selected:
        if bench not in registry:
            raise KeyError(f"Unknown benchmark {bench!r}; known benchmarks: {sorted(registry)}")
        spec = registry[bench]
        task_ids = list(spec.task_ids)
        if task_ids:
            rotation = offset % len(task_ids)
            task_ids = task_ids[rotation:] + task_ids[:rotation]
        if not full:
            task_ids = task_ids[:max(1, max_tasks_per_benchmark)]
        for idx, task_id in enumerate(task_ids):
            reward_family = spec.default_reward_families[(idx + seed) % len(spec.default_reward_families)]
            tasks.append(
                BenchmarkTask(
                    benchmark=bench,
                    task_id=task_id,
                    reward_family=reward_family,
                    objective=f"{bench}:{task_id}:{reward_family}",
                    max_episodes=20 if full else 1,
                    seeds=(0, 1, 2, 3, 4) if full else (seed,),
                    metadata={
                        "protocol": "full" if full else "bounded_smoke",
                        "requires_real_environment": full,
                    },
                )
            )
    return tasks


# ---------------------------------------------------------------------------
# Lightweight artifact IO.
# ---------------------------------------------------------------------------

def _json_default(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def write_json_artifact(path: Path | str, payload: Mapping[str, Any], *, indent: int = 2) -> str:
    """Write a JSON artifact with deterministic formatting."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=indent, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    return str(out)


def write_jsonl_artifact(path: Path | str, records: Iterable[Mapping[str, Any]]) -> str:
    """Write JSONL records."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, default=_json_default) + "\n")
    return str(out)


def write_csv_artifact(path: Path | str, rows: Sequence[Mapping[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> str:
    """Write a CSV artifact from row dictionaries."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        ordered: List[str] = []
        for row in rows:
            for key in row.keys():
                if key not in ordered:
                    ordered.append(str(key))
        fieldnames = ordered or ["status"]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return str(out)


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)


def _write_simple_png(
    path: Path | str,
    *,
    width: int = 640,
    height: int = 400,
    bars: Optional[Sequence[Tuple[str, float]]] = None,
    title: str = "",
) -> str:
    """Write a tiny valid PNG using only the standard library.

    This is not a plotting replacement; it is a deterministic artifact writer
    used when measured values are available but optional plotting packages are
    absent.  The title is encoded into metadata and simple colored bars encode
    values.  Figure captions live in the manifest.
    """

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    width = max(64, int(width))
    height = max(64, int(height))
    background = (255, 255, 255)
    axis = (40, 40, 40)
    palette = [
        (43, 108, 176),
        (221, 107, 32),
        (56, 161, 105),
        (128, 90, 213),
        (214, 48, 49),
        (49, 130, 206),
        (113, 128, 150),
    ]
    pixels = [[background for _ in range(width)] for _ in range(height)]

    def rect(x0: int, y0: int, x1: int, y1: int, color: Tuple[int, int, int]) -> None:
        for y in range(max(0, y0), min(height, y1)):
            row = pixels[y]
            for x in range(max(0, x0), min(width, x1)):
                row[x] = color

    margin_left = 60
    margin_bottom = 48
    margin_top = 30
    plot_h = height - margin_top - margin_bottom
    plot_w = width - margin_left - 24
    rect(margin_left, margin_top, margin_left + 2, margin_top + plot_h, axis)
    rect(margin_left, margin_top + plot_h, margin_left + plot_w, margin_top + plot_h + 2, axis)

    bars = list(bars or [])
    finite_values = [float(v) for _, v in bars if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if finite_values:
        lo = min(0.0, min(finite_values))
        hi = max(1.0, max(finite_values))
        span = hi - lo if hi > lo else 1.0
        bar_w = max(8, plot_w // max(1, len(bars) * 2))
        gap = max(4, (plot_w - len(bars) * bar_w) // max(1, len(bars) + 1))
        for i, (_label, value) in enumerate(bars):
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(v):
                continue
            scaled = (v - lo) / span
            bh = int(max(1, min(plot_h - 4, scaled * (plot_h - 8))))
            x0 = margin_left + gap + i * (bar_w + gap)
            y0 = margin_top + plot_h - bh
            rect(x0, y0, x0 + bar_w, margin_top + plot_h, palette[i % len(palette)])

    raw = bytearray()
    for row in pixels:
        raw.append(0)
        for r, g, b in row:
            raw.extend((r, g, b))

    png = b"\x89PNG\r\n\x1a\n"
    png += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    if title:
        png += _png_chunk(b"tEXt", b"Title\x00" + title.encode("utf-8", errors="replace")[:512])
    png += _png_chunk(b"IDAT", zlib.compress(bytes(raw), level=9))
    png += _png_chunk(b"IEND", b"")
    out.write_bytes(png)
    return str(out)


def _hash_payload(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, default=_json_default)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Benchmark dataset/episode bookkeeping helpers.
# ---------------------------------------------------------------------------

def filter_records_by_episode_length(
    records: Sequence[Mapping[str, Any]],
    minimum_episode_length: Optional[int],
) -> List[Mapping[str, Any]]:
    """Filter evaluation records by per-episode length.

    reference_grounding: paperbench_ref_001 url_benchmark/d4rl_benchmark.py

    The referenced benchmark helper expands terminal/timeout-delimited episode
    lengths across transitions before filtering.  Evaluation artifacts often
    arrive already as per-episode records, so this implementation preserves the
    same protocol intent by removing records whose explicit ``episode_length`` is
    below the threshold and leaving records without a length unchanged.
    """

    if minimum_episode_length is None or minimum_episode_length <= 1:
        return list(records)
    filtered: List[Mapping[str, Any]] = []
    for record in records:
        length = record.get("episode_length")
        if length is None:
            filtered.append(record)
            continue
        try:
            if int(length) >= minimum_episode_length:
                filtered.append(record)
        except (TypeError, ValueError):
            filtered.append(record)
    return filtered


def baseline_or_ablation(method: str, variant: Optional[str] = None) -> Dict[str, Any]:
    """Return registry-backed comparison metadata for a method or ablation."""

    registry = build_baseline_registry()
    if method in registry:
        data = dataclasses.asdict(registry[method])
    elif method.startswith("FRE-"):
        data = dataclasses.asdict(registry["FRE"])
        data["method"] = method
        data["comparison_role"] = "FRE reward-family/domain-prior ablation"
    else:
        data = {
            "method": method,
            "family": "external_or_user_supplied",
            "zero_shot": True,
            "policy_learning": "user supplied",
            "reward_family_assumption": "user supplied",
            "comparison_role": "external comparison",
            "checkpoint_keys": (),
        }
    if variant:
        data["variant"] = variant
    return data


def adapt_checkpoint_registry(checkpoints: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Normalize trained FRE/baseline checkpoint inputs for artifact metadata."""

    checkpoints = checkpoints or {}
    baseline_specs = build_baseline_registry()
    normalized: Dict[str, Any] = {}
    for method, spec in baseline_specs.items():
        method_payload = checkpoints.get(method, checkpoints.get(method.lower(), {}))
        if isinstance(method_payload, (str, Path)):
            paths = {"checkpoint": str(method_payload)}
        elif isinstance(method_payload, Mapping):
            paths = {str(k): str(v) for k, v in method_payload.items()}
        else:
            paths = {}
        normalized[method] = {
            "method": method,
            "expected_keys": list(spec.checkpoint_keys),
            "provided": bool(paths),
            "paths": paths,
            "adapter": dataclasses.asdict(spec),
        }
    for method, payload in checkpoints.items():
        key = str(method)
        if key in normalized or key.upper() in normalized:
            continue
        normalized[key] = {
            "method": key,
            "expected_keys": [],
            "provided": True,
            "paths": {"checkpoint": str(payload)} if not isinstance(payload, Mapping) else {str(k): str(v) for k, v in payload.items()},
            "adapter": baseline_or_ablation(key),
        }
    return normalized


# ---------------------------------------------------------------------------
# Metric computation and aggregation.
# ---------------------------------------------------------------------------

def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _mean(values: Sequence[float]) -> Optional[float]:
    return float(sum(values) / len(values)) if values else None


def _std(values: Sequence[float]) -> Optional[float]:
    if len(values) <= 1:
        return 0.0 if values else None
    return float(statistics.stdev(values))


def _stderr(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    std = _std(values)
    if std is None:
        return None
    return float(std / math.sqrt(len(values)))


def _metric_values(records: Sequence[MetricRecord], metric: str) -> List[float]:
    values: List[float] = []
    for record in records:
        value = record.get(metric)
        if value is None and metric == "success_rate" and "success" in record:
            value = record.get("success")
        if value is None and metric == "policy_return" and "return" in record:
            value = record.get("return")
        if _is_number(value):
            values.append(float(value))
    return values


def aggregate_metrics(
    records: Sequence[MetricRecord],
    *,
    group_by: Sequence[str] = ("benchmark", "task_id", "method"),
    metrics: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Aggregate task-level and method-level metrics.

    Returns a JSON-serializable dictionary with long-form group summaries and
    global comparison statistics.  Metric schemas include reward, accuracy,
    loss, and return as required by the paper evidence contract.
    """

    metrics = tuple(metrics or METRIC_SCHEMAS.keys())
    groups: Dict[Tuple[Any, ...], List[MetricRecord]] = {}
    for record in records:
        key = tuple(record.get(field, "unknown") for field in group_by)
        groups.setdefault(key, []).append(record)

    group_rows: List[Dict[str, Any]] = []
    for key, subset in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        row: Dict[str, Any] = {field: value for field, value in zip(group_by, key)}
        row["count_records"] = len(subset)
        for metric in metrics:
            values = _metric_values(subset, metric)
            if not values:
                continue
            row[f"{metric}_mean"] = _mean(values)
            row[f"{metric}_std"] = _std(values)
            row[f"{metric}_stderr"] = _stderr(values)
            row[f"{metric}_count"] = len(values)
        group_rows.append(row)

    method_groups: Dict[str, List[MetricRecord]] = {}
    benchmark_method_groups: Dict[Tuple[str, str], List[MetricRecord]] = {}
    for record in records:
        method = str(record.get("method", "unknown"))
        benchmark = str(record.get("benchmark", "unknown"))
        method_groups.setdefault(method, []).append(record)
        benchmark_method_groups.setdefault((benchmark, method), []).append(record)

    by_method: Dict[str, Dict[str, Any]] = {}
    for method, subset in sorted(method_groups.items()):
        by_method[method] = {"count_records": len(subset)}
        for metric in metrics:
            values = _metric_values(subset, metric)
            if values:
                by_method[method][metric] = {
                    "mean": _mean(values),
                    "std": _std(values),
                    "stderr": _stderr(values),
                    "count": len(values),
                    "higher_is_better": METRIC_SCHEMAS.get(metric, {}).get("higher_is_better"),
                }

    by_benchmark_method: Dict[str, Dict[str, Any]] = {}
    for (benchmark, method), subset in sorted(benchmark_method_groups.items()):
        key = f"{benchmark}/{method}"
        by_benchmark_method[key] = {"benchmark": benchmark, "method": method, "count_records": len(subset)}
        for metric in metrics:
            values = _metric_values(subset, metric)
            if values:
                by_benchmark_method[key][metric] = {
                    "mean": _mean(values),
                    "std": _std(values),
                    "stderr": _stderr(values),
                    "count": len(values),
                }

    comparisons: Dict[str, Any] = {}
    for benchmark in sorted({str(r.get("benchmark", "unknown")) for r in records}):
        fre_records = [r for r in records if str(r.get("benchmark", "unknown")) == benchmark and str(r.get("method", "")).upper() == "FRE"]
        if not fre_records:
            continue
        benchmark_records = [r for r in records if str(r.get("benchmark", "unknown")) == benchmark]
        comparison_metric = "normalized_return"
        if not _metric_values(benchmark_records, comparison_metric):
            comparison_metric = "success_rate" if _metric_values(benchmark_records, "success_rate") else "return"
        fre_mean = _mean(_metric_values(fre_records, comparison_metric))
        baseline_means: Dict[str, float] = {}
        for method in sorted({str(r.get("method", "unknown")) for r in benchmark_records if str(r.get("method", "")).upper() != "FRE"}):
            values = _metric_values([r for r in benchmark_records if str(r.get("method", "unknown")) == method], comparison_metric)
            if values:
                baseline_means[method] = float(_mean(values) or 0.0)
        best_baseline = None
        if baseline_means:
            best_baseline = max(baseline_means.items(), key=lambda item: item[1])
        comparisons[benchmark] = {
            "metric": comparison_metric,
            "fre_mean": fre_mean,
            "baseline_means": baseline_means,
            "best_baseline": {"method": best_baseline[0], "mean": best_baseline[1]} if best_baseline else None,
            "fre_minus_best_baseline": (fre_mean - best_baseline[1]) if (fre_mean is not None and best_baseline) else None,
            "assertion": TREND_ASSERTIONS["baseline_outperformance"]["claim"],
        }

    return {
        "schema_version": "1.0",
        "metric_schemas": METRIC_SCHEMAS,
        "group_by": list(group_by),
        "group_summaries": group_rows,
        "by_method": by_method,
        "by_benchmark_method": by_benchmark_method,
        "comparisons": comparisons,
        "count_records": len(records),
    }


def compute_artifacts_metrics(
    evaluation_records: Sequence[MetricRecord],
    *,
    minimum_episode_length: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute paper-required metric summaries from measured evaluation records."""

    filtered = filter_records_by_episode_length(evaluation_records, minimum_episode_length)
    task_aggregates = aggregate_metrics(
        filtered,
        group_by=("benchmark", "task_id", "method"),
        metrics=(
            "return",
            "normalized_return",
            "success_rate",
            "decoded_reward_mse",
            "decoded_reward_correlation",
            "value_loss",
            "policy_return",
            "reward_loss",
            "accuracy",
            "loss",
        ),
    )
    benchmark_aggregates = aggregate_metrics(
        filtered,
        group_by=("benchmark", "method"),
        metrics=("return", "normalized_return", "success_rate", "policy_return"),
    )
    global_aggregates = aggregate_metrics(
        filtered,
        group_by=("method",),
        metrics=("return", "normalized_return", "success_rate", "decoded_reward_mse", "value_loss", "policy_return"),
    )
    return {
        "schema_version": "1.0",
        "paper": PAPER_TITLE,
        "record_count": len(filtered),
        "input_record_count": len(evaluation_records),
        "minimum_episode_length": minimum_episode_length,
        "task_level": task_aggregates,
        "benchmark_level": benchmark_aggregates,
        "aggregate_level": global_aggregates,
        "trend_assertions": TREND_ASSERTIONS,
        "evidence_obligation_matrix": list(EVIDENCE_OBLIGATION_MATRIX),
    }


# ---------------------------------------------------------------------------
# Artifact specs and manifests.
# ---------------------------------------------------------------------------

def write_tables_and_figure_artifact_specs(
    layout: Optional[ArtifactsLayout] = None,
    *,
    include_dirs: bool = True,
) -> Dict[str, Any]:
    """Declare stable paths/captions for all paper-visible artifacts.

    This function creates parent directories for smoke validation but does not
    write paper-visible performance files.  It returns machine-readable specs
    that the manifest writer and canonical runner can consume.
    """

    layout = layout or ArtifactsLayout()
    if include_dirs:
        layout.ensure_dirs()
        for spec in FIGURE_AND_TABLE_SPECS.values():
            Path(spec["path"]).parent.mkdir(parents=True, exist_ok=True)
            for alias in spec.get("aliases", []):
                Path(alias).parent.mkdir(parents=True, exist_ok=True)

    specs = {
        key: {
            **value,
            "path": str(Path(value["path"])),
            "aliases": [str(Path(alias)) for alias in value.get("aliases", [])],
        }
        for key, value in FIGURE_AND_TABLE_SPECS.items()
    }
    return {
        "schema_version": "1.0",
        "paper": PAPER_TITLE,
        "artifact_specs": specs,
        "metric_schemas": METRIC_SCHEMAS,
        "captions_preserved": True,
        "performance_artifacts_require_measurements": [
            key for key, value in specs.items() if value.get("requires_measurements")
        ],
    }


def write_experiment_registry_artifact(layout: Optional[ArtifactsLayout] = None, *, full: bool = False) -> str:
    layout = layout or ArtifactsLayout()
    layout.ensure_dirs()
    return write_json_artifact(layout.experiment_registry_path, build_experiment_registry(full=full))


def write_model_registry_artifact(
    layout: Optional[ArtifactsLayout] = None,
    *,
    checkpoints: Optional[Mapping[str, Any]] = None,
) -> str:
    layout = layout or ArtifactsLayout()
    layout.ensure_dirs()
    registry = {
        "schema_version": "1.0",
        "paper": PAPER_TITLE,
        "models": adapt_checkpoint_registry(checkpoints),
        "required_fre_checkpoints": [
            "results/checkpoints/fre_encoder.pt",
            "results/checkpoints/fre_policy.pt",
        ],
        "baseline_adapters": {k: dataclasses.asdict(v) for k, v in build_baseline_registry().items()},
    }
    return write_json_artifact(layout.model_registry_path, registry)


def write_artifact_manifest(
    layout: Optional[ArtifactsLayout] = None,
    *,
    measured: bool = False,
    written_artifacts: Optional[Sequence[str]] = None,
    full: bool = False,
    extra: Optional[Mapping[str, Any]] = None,
) -> str:
    """Write the canonical artifact manifest."""

    layout = layout or ArtifactsLayout()
    layout.ensure_dirs()
    specs = write_tables_and_figure_artifact_specs(layout, include_dirs=True)["artifact_specs"]
    entries: List[Dict[str, Any]] = []
    written_set = {str(Path(p)) for p in (written_artifacts or [])}
    for key, spec in specs.items():
        path = str(Path(spec["path"]))
        requires_measurements = bool(spec.get("requires_measurements"))
        entries.append(
            {
                "artifact_id": key,
                "path": path,
                "kind": spec.get("kind"),
                "caption": spec.get("caption"),
                "aliases": spec.get("aliases", []),
                "requires_measurements": requires_measurements,
                "status": (
                    "written"
                    if path in written_set or any(str(Path(alias)) in written_set for alias in spec.get("aliases", []))
                    else ("declared_requires_measured_evaluation" if requires_measurements and not measured else "declared")
                ),
            }
        )

    payload: Dict[str, Any] = {
        "schema_version": "1.0",
        "paper": PAPER_TITLE,
        "created_at_unix": time.time(),
        "measured": measured,
        "full_mode": full,
        "entries": entries,
        "canonical_outputs": {
            "experiment_registry": str(layout.experiment_registry_path),
            "artifact_manifest": str(layout.artifact_manifest_path),
            "model_registry": str(layout.model_registry_path),
            "metrics": str(layout.metrics_path),
            "eval_summary": str(layout.eval_summary_path),
            "readiness": str(layout.readiness_path),
            "evaluation_result": str(layout.evaluation_result_path),
        },
        "grounding": [
            "reference_grounding: paperbench_ref_001 url_benchmark/d4rl_benchmark.py",
            "reference_grounding: paperbench_ref_001 controllable_agent/test_executor.py",
            "reference_grounding: paperbench_ref_001 controllable_agent/test_url_benchmark.py",
        ],
    }
    if extra:
        payload.update(dict(extra))
    return write_json_artifact(layout.artifact_manifest_path, payload)


def write_artifact_manifest_artifact(
    layout: Optional[ArtifactsLayout] = None,
    *,
    measured: bool = False,
    written_artifacts: Optional[Sequence[str]] = None,
    full: bool = False,
    extra: Optional[Mapping[str, Any]] = None,
) -> str:
    """Compatibility wrapper required by executable routes."""

    return write_artifact_manifest(
        layout=layout,
        measured=measured,
        written_artifacts=written_artifacts,
        full=full,
        extra=extra,
    )


def write_readiness_artifact(
    layout: Optional[ArtifactsLayout] = None,
    *,
    status: str,
    full: bool = False,
    records_available: int = 0,
    warnings: Optional[Sequence[str]] = None,
) -> str:
    layout = layout or ArtifactsLayout()
    layout.ensure_dirs()
    payload = {
        "schema_version": "1.0",
        "paper": PAPER_TITLE,
        "status": status,
        "full_mode": full,
        "records_available": records_available,
        "message": (
            "Artifact, benchmark registry, task sampler, baseline adapter, and "
            "metric aggregation surfaces are importable and wired. Paper-visible "
            "performance artifacts require measured evaluation records."
        ),
        "registries": {
            "benchmarks": sorted(build_benchmark_registry()),
            "baselines": sorted(build_baseline_registry()),
            "artifact_specs": sorted(FIGURE_AND_TABLE_SPECS),
        },
        "sampled_smoke_tasks": [dataclasses.asdict(task) for task in sample_tasks(seed=0, max_tasks_per_benchmark=1, full=False)],
        "warnings": list(warnings or []),
    }
    return write_json_artifact(layout.readiness_path, payload)


def write_evaluation_result_artifact(
    layout: Optional[ArtifactsLayout] = None,
    *,
    result: Mapping[str, Any],
) -> str:
    layout = layout or ArtifactsLayout()
    layout.ensure_dirs()
    return write_json_artifact(layout.evaluation_result_path, dict(result))


# ---------------------------------------------------------------------------
# Table writers.
# ---------------------------------------------------------------------------

def _flatten_group_rows(metrics: Mapping[str, Any], level: str = "benchmark_level") -> List[Dict[str, Any]]:
    node = metrics.get(level, {})
    rows = node.get("group_summaries", []) if isinstance(node, Mapping) else []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def write_metrics_artifact(
    metrics: Mapping[str, Any],
    layout: Optional[ArtifactsLayout] = None,
) -> str:
    layout = layout or ArtifactsLayout()
    layout.ensure_dirs()
    return write_json_artifact(layout.metrics_path, dict(metrics))


def write_eval_summary_artifact(
    metrics: Mapping[str, Any],
    layout: Optional[ArtifactsLayout] = None,
) -> str:
    layout = layout or ArtifactsLayout()
    layout.ensure_dirs()
    comparisons = (
        metrics.get("aggregate_level", {})
        .get("comparisons", {})
        if isinstance(metrics.get("aggregate_level", {}), Mapping)
        else {}
    )
    payload = {
        "schema_version": "1.0",
        "paper": PAPER_TITLE,
        "record_count": metrics.get("record_count", 0),
        "comparisons": comparisons,
        "trend_assertions": TREND_ASSERTIONS,
    }
    return write_json_artifact(layout.eval_summary_path, payload)


def write_result_table_artifact(
    metrics: Mapping[str, Any],
    layout: Optional[ArtifactsLayout] = None,
    *,
    path: Optional[Path | str] = None,
) -> str:
    layout = layout or ArtifactsLayout()
    layout.ensure_dirs()
    rows = _flatten_group_rows(metrics, "task_level")
    if not rows:
        rows = [{"status": "no_measured_records"}]
    out = Path(path) if path is not None else layout.path_for("result_table")
    return write_csv_artifact(out, rows)


def write_table_1_artifact(metrics: Mapping[str, Any], layout: Optional[ArtifactsLayout] = None) -> str:
    """Write Table 1 measured zero-shot benchmark comparison."""

    layout = layout or ArtifactsLayout()
    rows = _flatten_group_rows(metrics, "benchmark_level")
    table_rows: List[Dict[str, Any]] = []
    for row in rows:
        table_rows.append(
            {
                "benchmark": row.get("benchmark", ""),
                "method": row.get("method", ""),
                "normalized_return_mean": row.get("normalized_return_mean", ""),
                "normalized_return_std": row.get("normalized_return_std", ""),
                "success_rate_mean": row.get("success_rate_mean", ""),
                "success_rate_std": row.get("success_rate_std", ""),
                "return_mean": row.get("return_mean", ""),
                "return_std": row.get("return_std", ""),
                "count": row.get("count_records", ""),
            }
        )
    return write_csv_artifact(
        layout.path_for("table_1"),
        table_rows,
        fieldnames=[
            "benchmark",
            "method",
            "normalized_return_mean",
            "normalized_return_std",
            "success_rate_mean",
            "success_rate_std",
            "return_mean",
            "return_std",
            "count",
        ],
    )


def write_table_2_artifact(layout: Optional[ArtifactsLayout] = None) -> str:
    """Write capability comparison table from paper-derived method semantics."""

    layout = layout or ArtifactsLayout()
    rows = []
    for method, spec in build_baseline_registry().items():
        rows.append(
            {
                "method": method,
                "zero_shot": spec.zero_shot,
                "policy_learning": spec.policy_learning,
                "reward_family_assumption": spec.reward_family_assumption,
                "comparison_role": spec.comparison_role,
            }
        )
    return write_csv_artifact(
        layout.path_for("table_2"),
        rows,
        fieldnames=["method", "zero_shot", "policy_learning", "reward_family_assumption", "comparison_role"],
    )


def write_table_3_artifact(layout: Optional[ArtifactsLayout] = None) -> str:
    """Write Table 3 FRE hyperparameters."""

    layout = layout or ArtifactsLayout()
    rows = [{"hyperparameter": key, "value": json.dumps(value) if isinstance(value, (list, dict)) else value} for key, value in PAPER_HYPERPARAMETERS.items()]
    return write_csv_artifact(layout.path_for("table_3"), rows, fieldnames=["hyperparameter", "value"])


def write_table_4_artifact(metrics: Mapping[str, Any], layout: Optional[ArtifactsLayout] = None) -> str:
    """Write AntMaze reward-family ablation table from measured records."""

    layout = layout or ArtifactsLayout()
    rows = _flatten_group_rows(metrics, "task_level")
    table_rows: List[Dict[str, Any]] = []
    for row in rows:
        if str(row.get("benchmark", "")).lower() != "antmaze":
            continue
        method = str(row.get("method", ""))
        if method.upper() == "FRE" or method.startswith("FRE-"):
            table_rows.append(
                {
                    "task_id": row.get("task_id", ""),
                    "method_or_family_subset": method,
                    "normalized_return_mean": row.get("normalized_return_mean", ""),
                    "success_rate_mean": row.get("success_rate_mean", ""),
                    "return_mean": row.get("return_mean", ""),
                    "count": row.get("count_records", ""),
                }
            )
    return write_csv_artifact(
        layout.path_for("table_4"),
        table_rows,
        fieldnames=["task_id", "method_or_family_subset", "normalized_return_mean", "success_rate_mean", "return_mean", "count"],
    )


# ---------------------------------------------------------------------------
# Figure writers.
# ---------------------------------------------------------------------------

def _bars_from_metrics(metrics: Mapping[str, Any], metric_name: str = "normalized_return") -> List[Tuple[str, float]]:
    rows = _flatten_group_rows(metrics, "benchmark_level")
    bars: List[Tuple[str, float]] = []
    for row in rows:
        label = f"{row.get('benchmark', '')}/{row.get('method', '')}"
        value = row.get(f"{metric_name}_mean")
        if _is_number(value):
            bars.append((label, float(value)))
    if not bars and metric_name != "return":
        return _bars_from_metrics(metrics, "return")
    return bars


def write_figure_3_artifact(
    metrics_or_records: Mapping[str, Any] | Sequence[MetricRecord],
    layout: Optional[ArtifactsLayout] = None,
) -> str:
    """Write Figure 3 style zero-shot transfer visualization from measured data."""

    layout = layout or ArtifactsLayout()
    if isinstance(metrics_or_records, Mapping):
        metrics = metrics_or_records
    else:
        metrics = compute_artifacts_metrics(list(metrics_or_records))
    bars = []
    rows = _flatten_group_rows(metrics, "task_level")
    for row in rows:
        if str(row.get("benchmark", "")).lower() == "antmaze" and str(row.get("method", "")).upper() == "FRE":
            value = row.get("policy_return_mean", row.get("return_mean", row.get("normalized_return_mean")))
            if _is_number(value):
                bars.append((str(row.get("task_id", "antmaze")), float(value)))
    if not bars:
        bars = _bars_from_metrics(metrics)
    return _write_simple_png(
        layout.path_for("figure_3"),
        bars=bars,
        title=FIGURE_AND_TABLE_SPECS["figure_3"]["caption"],
    )


def write_figure3_artifact(
    metrics_or_records: Mapping[str, Any] | Sequence[MetricRecord],
    layout: Optional[ArtifactsLayout] = None,
) -> str:
    """Alias required by active route contracts."""

    return write_figure_3_artifact(metrics_or_records, layout)


def write_figure_5_artifact(metrics: Mapping[str, Any], layout: Optional[ArtifactsLayout] = None) -> str:
    layout = layout or ArtifactsLayout()
    rows = _flatten_group_rows(metrics, "task_level")
    bars: List[Tuple[str, float]] = []
    for row in rows:
        method = str(row.get("method", ""))
        if method.upper() == "FRE" or method.startswith("FRE-"):
            value = row.get("normalized_return_mean", row.get("return_mean"))
            if _is_number(value):
                bars.append((method, float(value)))
    return _write_simple_png(layout.path_for("figure_5"), bars=bars or _bars_from_metrics(metrics), title=FIGURE_AND_TABLE_SPECS["figure_5"]["caption"])


def write_figure_6_artifact(metrics: Mapping[str, Any], layout: Optional[ArtifactsLayout] = None) -> str:
    layout = layout or ArtifactsLayout()
    return _write_simple_png(layout.path_for("figure_6"), bars=_bars_from_metrics(metrics), title=FIGURE_AND_TABLE_SPECS["figure_6"]["caption"])


def write_figure_7_artifact(
    metrics_or_records: Mapping[str, Any] | Sequence[MetricRecord],
    layout: Optional[ArtifactsLayout] = None,
) -> str:
    layout = layout or ArtifactsLayout()
    metrics = metrics_or_records if isinstance(metrics_or_records, Mapping) else compute_artifacts_metrics(list(metrics_or_records))
    return _write_simple_png(layout.path_for("figure_7"), bars=_bars_from_metrics(metrics, "policy_return"), title=FIGURE_AND_TABLE_SPECS["figure_7"]["caption"])


def write_figure_8_artifact(
    metrics_or_records: Mapping[str, Any] | Sequence[MetricRecord],
    layout: Optional[ArtifactsLayout] = None,
) -> str:
    layout = layout or ArtifactsLayout()
    metrics = metrics_or_records if isinstance(metrics_or_records, Mapping) else compute_artifacts_metrics(list(metrics_or_records))
    return _write_simple_png(layout.path_for("figure_8"), bars=_bars_from_metrics(metrics, "decoded_reward_correlation"), title=FIGURE_AND_TABLE_SPECS["figure_8"]["caption"])


def write_figure_9_artifact(
    metrics_or_records: Mapping[str, Any] | Sequence[MetricRecord],
    layout: Optional[ArtifactsLayout] = None,
) -> str:
    layout = layout or ArtifactsLayout()
    metrics = metrics_or_records if isinstance(metrics_or_records, Mapping) else compute_artifacts_metrics(list(metrics_or_records))
    return _write_simple_png(layout.path_for("figure_9"), bars=_bars_from_metrics(metrics, "return"), title=FIGURE_AND_TABLE_SPECS["figure_9"]["caption"])


def write_result_figure_artifact(metrics: Mapping[str, Any], layout: Optional[ArtifactsLayout] = None) -> str:
    layout = layout or ArtifactsLayout()
    return _write_simple_png(layout.path_for("result_figure"), bars=_bars_from_metrics(metrics), title=FIGURE_AND_TABLE_SPECS["result_figure"]["caption"])


def write_trends_artifact(metrics: Mapping[str, Any], layout: Optional[ArtifactsLayout] = None) -> str:
    layout = layout or ArtifactsLayout()
    comparisons = (
        metrics.get("aggregate_level", {}).get("comparisons", {})
        if isinstance(metrics.get("aggregate_level", {}), Mapping)
        else {}
    )
    payload = {
        "schema_version": "1.0",
        "paper": PAPER_TITLE,
        "trend_assertions": TREND_ASSERTIONS,
        "comparisons": comparisons,
        "measured_record_count": metrics.get("record_count", 0),
        "interpretation_guardrail": (
            "Trend assertions are decision rules for measured outputs; this file "
            "does not claim paper benchmark reproduction unless measured records "
            "were supplied by evaluation."
        ),
    }
    return write_json_artifact(layout.trends_path, payload)


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------

def _records_from_evaluator(
    evaluator: Optional[Callable[..., Sequence[MetricRecord]]],
    *,
    tasks: Sequence[BenchmarkTask],
    checkpoints: Optional[Mapping[str, Any]],
    methods: Optional[Sequence[str]],
    full: bool,
) -> List[MetricRecord]:
    if evaluator is None:
        return []
    result = evaluator(tasks=tasks, checkpoints=checkpoints or {}, methods=list(methods or ("FRE",)), full=full)
    return [dict(record) for record in result]


def write_artifacts_artifact(
    *,
    layout: Optional[ArtifactsLayout] = None,
    evaluation_records: Optional[Sequence[MetricRecord]] = None,
    checkpoints: Optional[Mapping[str, Any]] = None,
    full: bool = False,
    write_static_tables: bool = True,
) -> ArtifactsResult:
    """Write FRE artifact registries and, when measured records exist, results.

    This function intentionally wires all active-route writer symbols:
    ``write_json_artifact``, ``write_artifact_manifest``,
    ``write_experiment_registry_artifact``, ``write_artifact_manifest_artifact``,
    ``write_model_registry_artifact``, ``write_figure_3_artifact``,
    ``write_figure_7_artifact``, ``write_figure_8_artifact``,
    ``write_figure_9_artifact``, ``write_metrics_artifact``,
    ``write_figure3_artifact``, and ``write_trends_artifact``.
    """

    layout = layout or ArtifactsLayout()
    layout.ensure_dirs()
    written: List[str] = []
    warnings: List[str] = []

    registry_path = write_experiment_registry_artifact(layout, full=full)
    model_registry_path = write_model_registry_artifact(layout, checkpoints=checkpoints)
    written.extend([registry_path, model_registry_path])

    spec_path = write_json_artifact(
        layout.resolved_config_path,
        {
            "schema_version": "1.0",
            "paper": PAPER_TITLE,
            "full_mode": full,
            "artifact_specs": write_tables_and_figure_artifact_specs(layout, include_dirs=True),
            "benchmark_registry": {k: dataclasses.asdict(v) for k, v in build_benchmark_registry().items()},
            "baseline_registry": {k: dataclasses.asdict(v) for k, v in build_baseline_registry().items()},
        },
    )
    written.append(spec_path)

    if write_static_tables:
        written.append(write_table_2_artifact(layout))
        written.append(write_table_3_artifact(layout))

    records = [dict(record) for record in (evaluation_records or [])]
    measured = len(records) > 0
    metrics: Dict[str, Any] = {}
    if measured:
        metrics = compute_artifacts_metrics(records)
        written.append(write_metrics_artifact(metrics, layout))
        written.append(write_eval_summary_artifact(metrics, layout))
        written.append(write_result_table_artifact(metrics, layout))
        written.append(write_table_1_artifact(metrics, layout))
        written.append(write_table_4_artifact(metrics, layout))
        written.append(write_jsonl_artifact(layout.predictions_path, records))
        written.append(write_figure_3_artifact(metrics, layout))
        written.append(write_figure3_artifact(metrics, layout))
        written.append(write_figure_5_artifact(metrics, layout))
        written.append(write_figure_6_artifact(metrics, layout))
        written.append(write_figure_7_artifact(metrics, layout))
        written.append(write_figure_8_artifact(metrics, layout))
        written.append(write_figure_9_artifact(metrics, layout))
        written.append(write_result_figure_artifact(metrics, layout))
        written.append(write_trends_artifact(metrics, layout))
        status = "measured_artifacts_written"
    else:
        warnings.append(
            "No measured evaluation records were supplied; paper-visible performance "
            "tables, metrics, prediction logs, and result figures were not written."
        )
        status = "readiness_only_no_measured_records"

    readiness_path = write_readiness_artifact(
        layout,
        status=status,
        full=full,
        records_available=len(records),
        warnings=warnings,
    )
    written.append(readiness_path)

    evaluation_result_payload = {
        "schema_version": "1.0",
        "paper": PAPER_TITLE,
        "status": status,
        "measured": measured,
        "record_count": len(records),
        "full_mode": full,
        "written_artifacts": written,
        "metrics_hash": _hash_payload(metrics) if metrics else None,
        "warnings": warnings,
    }
    evaluation_result_path = write_evaluation_result_artifact(layout, result=evaluation_result_payload)
    written.append(evaluation_result_path)

    manifest_path = write_artifact_manifest_artifact(
        layout,
        measured=measured,
        written_artifacts=written,
        full=full,
        extra={
            "evaluation_status": status,
            "warnings": warnings,
            "measured_record_count": len(records),
        },
    )
    written.append(manifest_path)

    return ArtifactsResult(
        status=status,
        layout=layout,
        metrics=metrics,
        aggregate_metrics=metrics.get("aggregate_level", {}) if metrics else {},
        written_artifacts=written,
        manifest_path=manifest_path,
        readiness_path=readiness_path,
        evaluation_result_path=evaluation_result_path,
        registry_paths={
            "experiment_registry": registry_path,
            "model_registry": model_registry_path,
            "artifact_manifest": manifest_path,
        },
        warnings=warnings,
    )


def evaluate_artifacts(
    *,
    output_dir: Path | str = "results",
    checkpoints: Optional[Mapping[str, Any]] = None,
    evaluation_records: Optional[Sequence[MetricRecord]] = None,
    evaluator: Optional[Callable[..., Sequence[MetricRecord]]] = None,
    benchmarks: Optional[Sequence[str]] = None,
    methods: Optional[Sequence[str]] = None,
    seed: int = 0,
    full: bool = False,
    mode: str = "runtime_smoke",
) -> ArtifactsResult:
    """Evaluate trained FRE/baseline checkpoints and write artifacts.

    Parameters
    ----------
    output_dir:
        Canonical repository result directory.
    checkpoints:
        Mapping from method names to trained checkpoint paths or checkpoint
        component dictionaries.  The function records these in the model
        registry and passes them to an optional evaluator.
    evaluation_records:
        Precomputed measured per-episode/per-task records.  When supplied, this
        function computes all task-level and aggregate metrics and writes
        paper-visible artifacts.
    evaluator:
        Optional callable that executes the actual zero-shot evaluation.  It is
        invoked only when explicit records are absent and ``mode`` is not a
        dry-run/readiness mode.
    benchmarks, methods:
        Benchmark and method selectors backed by the registries.
    seed, full, mode:
        Bounded default route uses one sampled task per benchmark and does not
        claim benchmark scores.  Full mode selects the complete registry but
        still requires real evaluator/checkpoints/datasets.
    """

    layout = ArtifactsLayout(root=Path(output_dir))
    layout.ensure_dirs()

    registry = build_benchmark_registry()
    selected_benchmarks = list(benchmarks or MAIN_BENCHMARKS)
    unknown_benchmarks = [b for b in selected_benchmarks if b not in registry]
    if unknown_benchmarks:
        raise KeyError(f"Unknown benchmark(s): {unknown_benchmarks}; known benchmarks: {sorted(registry)}")

    baseline_registry = build_baseline_registry()
    selected_methods = list(methods or ("FRE", "FB", "SF", "CRL"))
    unknown_methods = [m for m in selected_methods if m not in baseline_registry and not str(m).startswith("FRE-")]
    if unknown_methods:
        raise KeyError(f"Unknown method(s): {unknown_methods}; known methods: {sorted(baseline_registry)}")

    tasks = sample_tasks(selected_benchmarks, seed=seed, max_tasks_per_benchmark=1, full=full)
    records = [dict(record) for record in (evaluation_records or [])]

    dry_modes = {"dry_run", "runtime_smoke", "docker_validate", "readiness", "import"}
    if not records and evaluator is not None and mode not in dry_modes:
        records = _records_from_evaluator(
            evaluator,
            tasks=tasks,
            checkpoints=checkpoints,
            methods=selected_methods,
            full=full,
        )

    if not records:
        selected_protocol = {
            "schema_version": "1.0",
            "mode": mode,
            "full_mode": full,
            "selected_benchmarks": selected_benchmarks,
            "selected_methods": selected_methods,
            "sampled_tasks": [dataclasses.asdict(task) for task in tasks],
            "checkpoint_registry": adapt_checkpoint_registry(checkpoints),
            "status": "no_measured_records",
            "paper_visible_outputs_withheld": [
                spec["path"]
                for spec in FIGURE_AND_TABLE_SPECS.values()
                if spec.get("requires_measurements")
            ],
        }
        write_json_artifact(layout.root / "evaluation_protocol.json", selected_protocol)

    result = write_artifacts_artifact(
        layout=layout,
        evaluation_records=records,
        checkpoints=checkpoints,
        full=full,
        write_static_tables=True,
    )

    if not records:
        result.warnings.append(
            "evaluate_artifacts completed readiness/registry closure only. "
            "Provide evaluation_records or a non-smoke evaluator for measured zero-shot results."
        )
        write_evaluation_result_artifact(
            layout,
            result={
                **result.to_dict(),
                "status": result.status,
                "mode": mode,
                "selected_benchmarks": selected_benchmarks,
                "selected_methods": selected_methods,
                "sampled_tasks": [dataclasses.asdict(task) for task in tasks],
                "paper_visible_outputs_withheld": [
                    spec["path"]
                    for spec in FIGURE_AND_TABLE_SPECS.values()
                    if spec.get("requires_measurements")
                ],
            },
        )

    return result


__all__ = [
    "ArtifactsLayout",
    "ArtifactsResult",
    "BenchmarkSpec",
    "BenchmarkTask",
    "BaselineSpec",
    "METRIC_SCHEMAS",
    "FIGURE_AND_TABLE_SPECS",
    "TREND_ASSERTIONS",
    "aggregate_metrics",
    "adapt_checkpoint_registry",
    "baseline_or_ablation",
    "build_baseline_registry",
    "build_benchmark_registry",
    "build_experiment_registry",
    "compute_artifacts_metrics",
    "evaluate_artifacts",
    "filter_records_by_episode_length",
    "sample_tasks",
    "write_artifact_manifest",
    "write_artifact_manifest_artifact",
    "write_artifacts_artifact",
    "write_csv_artifact",
    "write_eval_summary_artifact",
    "write_evaluation_result_artifact",
    "write_experiment_registry_artifact",
    "write_figure3_artifact",
    "write_figure_3_artifact",
    "write_figure_5_artifact",
    "write_figure_6_artifact",
    "write_figure_7_artifact",
    "write_figure_8_artifact",
    "write_figure_9_artifact",
    "write_json_artifact",
    "write_jsonl_artifact",
    "write_metrics_artifact",
    "write_model_registry_artifact",
    "write_readiness_artifact",
    "write_result_table_artifact",
    "write_table_1_artifact",
    "write_table_2_artifact",
    "write_table_3_artifact",
    "write_table_4_artifact",
    "write_tables_and_figure_artifact_specs",
    "write_trends_artifact",
]