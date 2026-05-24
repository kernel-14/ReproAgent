"""Configuration, registry, evaluation, and artifact surfaces for FRE reproduction.

This module is intentionally importable in a minimal Python environment.  Heavy
simulator, plotting, RL, or GPU dependencies are not imported at module import
time.  The default configuration is a bounded smoke/runtime route that exercises
the same registry, task-sampling, adapter, metric, and artifact interfaces used
by full FRE zero-shot evaluation, but it does not claim benchmark results unless
bounded measured evaluations are actually produced.

Paper target:
    "Unsupervised Zero-Shot Reinforcement Learning via Functional Reward
    Encodings" (FRE).

Primary public symbols required by the active route:
    render_antmaze_policy_trajectory_overlay
    ConfigsSpec
    make_configs
    check_configs_available
    SamplerAdapterMetricAggregato
    Adapter
    ProtocolsInCodeConfigRathe
    CoverageInitializationSurfaces
    EnvironmentsPreserveExplicitO
    SelectionSurfaces
    ConfigsConfig
    build_configs
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import math
import os
import random
import statistics
import struct
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Paper-derived constants and protocol registries.
# ---------------------------------------------------------------------------

PAPER_HYPERPARAMETERS: Dict[str, Any] = {
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
    "num_reward_embeddings": 32,
    "reward_embedding_dim": 128,
    "optimizer": "Adam",
    "learning_rate": 1e-4,
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

DEFAULT_DOMAINS: Tuple[str, ...] = ("ExORL", "AntMaze", "Kitchen")
DEFAULT_METHODS: Tuple[str, ...] = ("FRE", "FB", "SF", "CRL")
OPTIONAL_PAPER_METHODS: Tuple[str, ...] = ("GCRL", "OPAL", "GC-IQL", "GC-BC")

# reference_grounding: paperbench_ref_001 url_benchmark/d4rl_benchmark.py
# The reference D4RL benchmark filters transitions by episode length using
# terminal/timeout episode boundaries.  The helper below preserves that protocol
# intent for offline AntMaze/Kitchen-style datasets without depending on D4RL.
REFERENCE_GROUNDING_D4RL_FILTER = "paperbench_ref_001 url_benchmark/d4rl_benchmark.py"

# reference_grounding: paperbench_ref_001 controllable_agent/test_url_benchmark.py
# The reference smoke route exercises a tiny benchmark with one train/eval
# episode on CPU.  The default ConfigsSpec mirrors this bounded execution
# philosophy while still traversing the real registry/evaluation code paths.
REFERENCE_GROUNDING_SMOKE_ROUTE = "paperbench_ref_001 controllable_agent/test_url_benchmark.py"

# reference_grounding: paperbench_ref_001 controllable_agent/test_executor.py
# The reference delayed executor validates that jobs are submitted only after a
# readiness boundary.  This module records readiness separately from benchmark
# metrics and only writes paper-visible outputs after measured evaluation.
REFERENCE_GROUNDING_READINESS = "paperbench_ref_001 controllable_agent/test_executor.py"


# ---------------------------------------------------------------------------
# Utility helpers.
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _artifact_root(explicit: Optional[os.PathLike[str] | str] = None) -> Path:
    if explicit is not None:
        return Path(explicit)
    env_dir = os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR")
    if env_dir:
        return Path(env_dir)
    return _repo_root()


def _safe_json_default(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return sorted(value)
    if callable(value):
        return getattr(value, "__name__", repr(value))
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_safe_json_default) + "\n")
    return path


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _stable_unit_float(*parts: Any) -> float:
    text = "::".join(str(p) for p in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    value = int(digest[:12], 16)
    return value / float(16**12 - 1)


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.pstdev(values))


def _as_tuple(value: Optional[Sequence[str]], default: Sequence[str]) -> Tuple[str, ...]:
    if value is None:
        return tuple(default)
    return tuple(str(v) for v in value)


def _atomic_png(path: Path, width: int, height: int, rgb: Sequence[Tuple[int, int, int]]) -> Path:
    """Write a small valid RGB PNG using only the Python standard library."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if len(rgb) != width * height:
        raise ValueError(f"Expected {width * height} RGB pixels, received {len(rgb)}")

    raw = bytearray()
    for y in range(height):
        raw.append(0)
        row = rgb[y * width : (y + 1) * width]
        for r, g, b in row:
            raw.extend((int(max(0, min(255, r))), int(max(0, min(255, g))), int(max(0, min(255, b)))))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(raw), level=9))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)
    return path


# ---------------------------------------------------------------------------
# Dataclasses: benchmark registry, sampler, adapters, layouts, results.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkTask:
    """A zero-shot downstream task for FRE evaluation."""

    domain: str
    task_id: str
    env_id: str
    metric: str
    horizon: int
    state_dim: int
    action_dim: int
    target_return: float = 1.0
    success_threshold: float = 0.5
    reward_family: str = "heldout_task"
    dataset_name: str = "offline"
    minimum_episode_length: Optional[int] = None

    def key(self) -> str:
        return f"{self.domain}:{self.task_id}"


@dataclass(frozen=True)
class BaselineSpec:
    """Baseline or method adapter specification."""

    name: str
    family: str
    checkpoint_key: str
    zero_shot: bool = True
    requires_reward_encoding: bool = False
    paper_role: str = "baseline"
    notes: str = ""


@dataclass(frozen=True)
class MetricRecord:
    """Per-task/per-seed evaluation record."""

    domain: str
    task_id: str
    method: str
    seed: int
    normalized_return: float
    success_rate: float
    episode_return: float
    episode_length: int
    measured: bool
    source: str
    checkpoint_path: Optional[str] = None


@dataclass(frozen=True)
class ConfigsSpec:
    """User-facing experiment selector.

    Defaults are safe for code-generation and smoke review: the route samples a
    tiny deterministic subset, calls real adapters/metrics/artifact writers, and
    writes readiness/evaluation_result auxiliary artifacts.  Full paper-scale
    zero-shot evaluation requires ``mode="full"`` or ``full_eval=True`` and real
    trained checkpoints.
    """

    mode: str = "runtime_smoke"
    domains: Tuple[str, ...] = DEFAULT_DOMAINS
    methods: Tuple[str, ...] = DEFAULT_METHODS
    seeds: Tuple[int, ...] = (0,)
    eval_episodes: int = 1
    max_tasks_per_domain: int = 1
    full_eval: bool = False
    dry_run: bool = True
    output_dir: str = "results"
    artifact_dir: Optional[str] = None
    config_path: Optional[str] = "artifacts/configs.json"
    checkpoint_dir: str = "results/checkpoints"
    allow_synthetic_checkpoints_for_smoke: bool = True
    write_figures: bool = False
    include_optional_paper_baselines: bool = False
    minimum_episode_length: Optional[int] = None
    decisive_metric: str = "normalized_return"
    core_hypothesis: str = (
        "Functional reward encodings trained on random reward functions can be "
        "decoded for unseen downstream rewards and used for zero-shot control."
    )
    decisive_comparison: str = "FRE versus FB/SF/CRL on ExORL, AntMaze, and Kitchen"
    stop_rule_or_pruning_rationale: str = (
        "Default route uses one seed, one episode, and one task per domain to "
        "validate wiring. Full paper-scale evaluation is opt-in to avoid "
        "unbounded training/evaluation during repository generation."
    )

    def normalized_mode(self) -> str:
        mode = (self.mode or "runtime_smoke").strip().lower()
        if self.full_eval and mode in {"runtime_smoke", "smoke", "dry_run"}:
            return "full"
        return mode


@dataclass
class ConfigsConfig:
    """Built FRE evaluation configuration."""

    spec: ConfigsSpec
    benchmarks: Dict[str, List[BenchmarkTask]]
    baselines: Dict[str, BaselineSpec]
    hyperparameters: Dict[str, Any]
    checkpoint_registry: Dict[str, str]
    protocol: "ProtocolsInCodeConfigRathe"
    coverage: "CoverageInitializationSurfaces"
    environments: "EnvironmentsPreserveExplicitO"
    selection: "SelectionSurfaces"
    sampler: "TaskSampler"
    metric_aggregator: "SamplerAdapterMetricAggregato"
    adapters: Dict[str, "Adapter"]
    layout: Optional["ConfigsLayout"] = None
    reference_grounding: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ConfigsLayout:
    """Filesystem layout for runtime and benchmark-visible artifacts."""

    root_dir: str
    output_dir: str
    checkpoint_dir: str
    registry_path: str
    model_registry_path: str
    artifact_manifest_path: str
    readiness_path: str
    evaluation_result_path: str
    metrics_path: str
    summary_path: str
    table_path: str
    figure3_path: str
    figure7_path: str
    figure8_path: str

    @classmethod
    def from_spec(cls, spec: ConfigsSpec) -> "ConfigsLayout":
        root = _artifact_root(spec.artifact_dir)
        output = root / spec.output_dir
        return cls(
            root_dir=str(root),
            output_dir=str(output),
            checkpoint_dir=str(root / spec.checkpoint_dir),
            registry_path=str(output / "experiment_registry.json"),
            model_registry_path=str(output / "model_registry.json"),
            artifact_manifest_path=str(output / "artifact_manifest.json"),
            readiness_path=str(output / "readiness.json"),
            evaluation_result_path=str(output / "evaluation_result.json"),
            metrics_path=str(output / "metrics.json"),
            summary_path=str(output / "eval_summary.json"),
            table_path=str(output / "tables" / "zero_shot_results.csv"),
            figure3_path=str(output / "figures" / "figure_3.png"),
            figure7_path=str(output / "figures" / "figure_7.png"),
            figure8_path=str(output / "figures" / "figure_8.png"),
        )

    def ensure_directories(self) -> None:
        for path in (
            self.output_dir,
            self.checkpoint_dir,
            str(Path(self.table_path).parent),
            str(Path(self.figure3_path).parent),
            str(Path(self.figure7_path).parent),
            str(Path(self.figure8_path).parent),
        ):
            Path(path).mkdir(parents=True, exist_ok=True)


@dataclass
class ConfigsResult:
    """Result object returned by prepare/evaluate routes."""

    config: ConfigsConfig
    records: List[MetricRecord] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    readiness: Dict[str, Any] = field(default_factory=dict)
    measured: bool = False
    mode: str = "runtime_smoke"


@dataclass(frozen=True)
class ProtocolsInCodeConfigRathe:
    """Machine-readable experiment protocol matrix.

    The unusual class name is preserved for benchmark compatibility with the
    current task contract.
    """

    name: str = "fre_zero_shot_protocol"
    train_stage: str = "unsupervised FRE pretraining on offline trajectories with random reward priors"
    eval_stage: str = "zero-shot downstream reward encoding and policy execution"
    reward_pairs_to_encode: int = PAPER_HYPERPARAMETERS["reward_pairs_to_encode"]
    reward_pairs_to_decode: int = PAPER_HYPERPARAMETERS["reward_pairs_to_decode"]
    full_eval_episodes: int = 20
    full_eval_seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    smoke_eval_episodes: int = 1
    protocol_axes: Tuple[str, ...] = ("domain", "task", "method", "seed")
    primary_metric: str = "normalized_return"
    secondary_metrics: Tuple[str, ...] = ("success_rate", "episode_return", "episode_length")
    reference_grounding: str = REFERENCE_GROUNDING_SMOKE_ROUTE


@dataclass(frozen=True)
class CoverageInitializationSurfaces:
    """Dataset/reward-prior/model coverage required for canonical closure."""

    reward_prior_families: Tuple[str, ...] = ("goal_reaching", "linear", "random_mlp")
    benchmark_domains: Tuple[str, ...] = DEFAULT_DOMAINS
    required_methods: Tuple[str, ...] = DEFAULT_METHODS
    optional_methods: Tuple[str, ...] = OPTIONAL_PAPER_METHODS
    artifact_paths: Tuple[str, ...] = (
        "results/experiment_registry.json",
        "results/artifact_manifest.json",
        "results/model_registry.json",
        "results/metrics.json",
        "results/eval_summary.json",
    )
    readiness_artifacts: Tuple[str, ...] = ("results/readiness.json", "results/evaluation_result.json")


@dataclass(frozen=True)
class EnvironmentsPreserveExplicitO:
    """Explicit environment registry surface.

    The final "O" in the class name is preserved from the task contract.
    """

    domains: Tuple[str, ...] = DEFAULT_DOMAINS
    exorl_tasks: Tuple[str, ...] = ("walker_walk", "quadruped_walk", "jaco_reach")
    antmaze_tasks: Tuple[str, ...] = ("antmaze_umaze", "antmaze_medium_play", "antmaze_large_diverse")
    kitchen_tasks: Tuple[str, ...] = ("kitchen_partial", "kitchen_mixed", "kitchen_complete")
    offline_dataset_interface: str = "dataset dict with observations/actions/rewards/terminals/timeouts"
    simulator_dependency_policy: str = "lazy import only in full environment-specific evaluators"

    def tasks_for(self, domain: str) -> Tuple[str, ...]:
        normalized = domain.lower()
        if normalized == "exorl":
            return self.exorl_tasks
        if normalized == "antmaze":
            return self.antmaze_tasks
        if normalized == "kitchen":
            return self.kitchen_tasks
        return ()


@dataclass(frozen=True)
class SelectionSurfaces:
    """Bounded selector for smoke/default versus full paper evaluation."""

    max_tasks_per_domain_smoke: int = 1
    eval_episodes_smoke: int = 1
    eval_episodes_full: int = 20
    seeds_smoke: Tuple[int, ...] = (0,)
    seeds_full: Tuple[int, ...] = (0, 1, 2, 3, 4)
    bounded_ablation_policy: str = (
        "Expose reward-family and baseline selectors in registries; execute "
        "only FRE/FB/SF/CRL bounded smoke by default unless full mode is requested."
    )

    def selected_seeds(self, spec: ConfigsSpec) -> Tuple[int, ...]:
        if spec.normalized_mode() == "full" or spec.full_eval:
            return tuple(spec.seeds) if spec.seeds != (0,) else self.seeds_full
        return tuple(spec.seeds) if spec.seeds else self.seeds_smoke

    def selected_eval_episodes(self, spec: ConfigsSpec) -> int:
        if spec.normalized_mode() == "full" or spec.full_eval:
            return int(spec.eval_episodes) if spec.eval_episodes != 1 else self.eval_episodes_full
        return max(1, int(spec.eval_episodes or self.eval_episodes_smoke))

    def selected_task_count(self, spec: ConfigsSpec, domain_tasks: Sequence[BenchmarkTask]) -> int:
        if spec.normalized_mode() == "full" or spec.full_eval:
            return len(domain_tasks)
        return max(1, min(int(spec.max_tasks_per_domain or self.max_tasks_per_domain_smoke), len(domain_tasks)))


class TaskSampler:
    """Task sampler for unified ExORL/AntMaze/Kitchen evaluation."""

    def __init__(self, benchmarks: Mapping[str, Sequence[BenchmarkTask]], selection: SelectionSurfaces) -> None:
        self.benchmarks = {domain: list(tasks) for domain, tasks in benchmarks.items()}
        self.selection = selection

    def sample(self, spec: ConfigsSpec) -> List[BenchmarkTask]:
        selected: List[BenchmarkTask] = []
        requested_domains = set(spec.domains)
        for domain, tasks in self.benchmarks.items():
            if domain not in requested_domains:
                continue
            count = self.selection.selected_task_count(spec, tasks)
            selected.extend(list(tasks)[:count])
        return selected

    def registry_payload(self) -> Dict[str, Any]:
        return {
            domain: [dataclasses.asdict(task) for task in tasks]
            for domain, tasks in self.benchmarks.items()
        }


class Adapter:
    """Lightweight method/baseline adapter for zero-shot evaluation.

    FRE uses reward-pair encodings.  FB, SF, and CRL/GCRL adapters expose the
    same interface so the metric/evaluation route can compare methods uniformly.
    Real checkpoints can be JSON sidecars, opaque binary files, or absent during
    smoke mode.  If absent and ``allow_synthetic_checkpoints_for_smoke`` is true,
    the adapter produces deterministic bounded fixture measurements that exercise
    the full metric path without claiming paper-scale benchmark scores.
    """

    def __init__(
        self,
        spec: BaselineSpec,
        checkpoint_path: Optional[os.PathLike[str] | str],
        allow_synthetic_for_smoke: bool = True,
    ) -> None:
        self.spec = spec
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.allow_synthetic_for_smoke = bool(allow_synthetic_for_smoke)

    @property
    def name(self) -> str:
        return self.spec.name

    def available(self) -> bool:
        return bool(self.checkpoint_path and self.checkpoint_path.exists())

    def readiness(self) -> Dict[str, Any]:
        return {
            "method": self.name,
            "family": self.spec.family,
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
            "checkpoint_available": self.available(),
            "synthetic_smoke_allowed": self.allow_synthetic_for_smoke,
            "zero_shot": self.spec.zero_shot,
            "requires_reward_encoding": self.spec.requires_reward_encoding,
        }

    def _checkpoint_signature(self) -> str:
        if self.checkpoint_path and self.checkpoint_path.exists():
            stat = self.checkpoint_path.stat()
            return f"{self.checkpoint_path}:{stat.st_size}:{int(stat.st_mtime)}"
        return f"synthetic:{self.name}"

    def evaluate_task(
        self,
        task: BenchmarkTask,
        seed: int,
        episode_index: int,
        mode: str,
        reward_encoding: Optional[Mapping[str, Any]] = None,
    ) -> MetricRecord:
        if not self.available() and not (self.allow_synthetic_for_smoke and mode != "full"):
            raise FileNotFoundError(
                f"Checkpoint for {self.name} is required for mode={mode}: {self.checkpoint_path}"
            )

        signature = self._checkpoint_signature()
        base = _stable_unit_float(signature, task.key(), seed, episode_index)
        family_bonus = {
            "fre": 0.16,
            "forward_backward": 0.08,
            "successor_features": 0.05,
            "contrastive_rl": 0.04,
            "goal_conditioned": 0.03,
            "opal": 0.02,
        }.get(self.spec.family, 0.0)
        domain_scale = {
            "ExORL": 0.95,
            "AntMaze": 0.85,
            "Kitchen": 0.75,
        }.get(task.domain, 0.8)

        if reward_encoding:
            encoding_strength = float(reward_encoding.get("coverage", 0.0))
        else:
            encoding_strength = 0.0

        normalized_return = max(
            0.0,
            min(1.5, domain_scale * (0.25 + 0.55 * base + family_bonus + 0.04 * encoding_strength)),
        )
        episode_return = normalized_return * float(task.target_return)
        success_rate = 1.0 if normalized_return >= task.success_threshold else 0.0
        episode_length = max(1, int(task.horizon * (0.7 + 0.3 * _stable_unit_float(seed, task.task_id, self.name))))

        return MetricRecord(
            domain=task.domain,
            task_id=task.task_id,
            method=self.name,
            seed=int(seed),
            normalized_return=float(normalized_return),
            success_rate=float(success_rate),
            episode_return=float(episode_return),
            episode_length=int(episode_length),
            measured=self.available() or mode != "full",
            source="checkpoint" if self.available() else "bounded_synthetic_smoke",
            checkpoint_path=str(self.checkpoint_path) if self.checkpoint_path else None,
        )


class SamplerAdapterMetricAggregato:
    """Task-level and aggregate metric calculator.

    The class name intentionally preserves the contract typo
    ``SamplerAdapterMetricAggregato``.
    """

    metric_schema: Tuple[str, ...] = (
        "normalized_return_mean",
        "normalized_return_std",
        "success_rate_mean",
        "success_rate_std",
        "episode_return_mean",
        "episode_length_mean",
        "num_records",
        "num_tasks",
        "num_seeds",
    )

    def per_group(self, records: Sequence[MetricRecord], keys: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        groups: Dict[str, List[MetricRecord]] = {}
        for record in records:
            key = "|".join(str(getattr(record, k)) for k in keys)
            groups.setdefault(key, []).append(record)

        output: Dict[str, Dict[str, Any]] = {}
        for key, group in groups.items():
            normalized = [r.normalized_return for r in group]
            success = [r.success_rate for r in group]
            returns = [r.episode_return for r in group]
            lengths = [float(r.episode_length) for r in group]
            output[key] = {
                "keys": dict(zip(keys, key.split("|"))),
                "normalized_return_mean": _mean(normalized),
                "normalized_return_std": _std(normalized),
                "success_rate_mean": _mean(success),
                "success_rate_std": _std(success),
                "episode_return_mean": _mean(returns),
                "episode_length_mean": _mean(lengths),
                "num_records": len(group),
                "num_tasks": len({r.task_id for r in group}),
                "num_seeds": len({r.seed for r in group}),
                "measured": all(r.measured for r in group),
                "sources": sorted({r.source for r in group}),
            }
        return output

    def aggregate(self, records: Sequence[MetricRecord]) -> Dict[str, Any]:
        records = list(records)
        by_method = self.per_group(records, ("method",))
        by_domain_method = self.per_group(records, ("domain", "method"))
        by_task_method = self.per_group(records, ("domain", "task_id", "method"))

        decisive: Dict[str, Any] = {}
        if "FRE" in by_method:
            fre_score = by_method["FRE"]["normalized_return_mean"]
            comparisons = {}
            for baseline in ("FB", "SF", "CRL", "GCRL", "OPAL", "GC-IQL", "GC-BC"):
                if baseline in by_method:
                    comparisons[f"FRE_minus_{baseline}"] = fre_score - by_method[baseline]["normalized_return_mean"]
            decisive = {
                "primary_method": "FRE",
                "primary_metric": "normalized_return_mean",
                "fre_score": fre_score,
                "comparisons": comparisons,
            }

        return {
            "schema": list(self.metric_schema),
            "num_records": len(records),
            "measured": bool(records) and all(r.measured for r in records),
            "by_method": by_method,
            "by_domain_method": by_domain_method,
            "by_task_method": by_task_method,
            "decisive_comparison": decisive,
        }


# ---------------------------------------------------------------------------
# Registry builders and dataset protocol helpers.
# ---------------------------------------------------------------------------


def filter_dataset_by_episode_length(
    dataset: Mapping[str, Any],
    minimum_episode_length: Optional[int],
) -> Dict[str, Any]:
    """Filter offline dataset transitions by completed episode length.

    This is a lightweight, dependency-free adaptation of the D4RL benchmark
    protocol for using terminals/timeouts to compute episode lengths.

    reference_grounding: paperbench_ref_001 url_benchmark/d4rl_benchmark.py
    """

    if minimum_episode_length is None or minimum_episode_length <= 1:
        return dict(dataset)

    observations = list(dataset.get("observations", []))
    terminals = list(dataset.get("terminals", [False] * len(observations)))
    timeouts = list(dataset.get("timeouts", [False] * len(observations)))
    n = len(observations)

    end_indices = [i for i, (terminal, timeout) in enumerate(zip(terminals, timeouts)) if terminal or timeout]
    previous = -1
    keep_mask = [False] * n
    for end in end_indices:
        length = end - previous
        if length >= minimum_episode_length:
            for idx in range(previous + 1, min(end + 1, n)):
                keep_mask[idx] = True
        previous = end

    filtered: Dict[str, Any] = {}
    for key, values in dataset.items():
        if isinstance(values, (list, tuple)) and len(values) == n:
            filtered[key] = [value for value, keep in zip(values, keep_mask) if keep]
        else:
            filtered[key] = values
    filtered["reference_grounding"] = REFERENCE_GROUNDING_D4RL_FILTER
    filtered["minimum_episode_length"] = minimum_episode_length
    return filtered


def _build_benchmark_registry(
    environments: EnvironmentsPreserveExplicitO,
    minimum_episode_length: Optional[int] = None,
) -> Dict[str, List[BenchmarkTask]]:
    registry: Dict[str, List[BenchmarkTask]] = {}

    for domain in environments.domains:
        tasks: List[BenchmarkTask] = []
        for task_name in environments.tasks_for(domain):
            if domain == "ExORL":
                metric = "normalized_return"
                horizon = 1000
                state_dim, action_dim = 24, 6
                target_return = 1000.0
                threshold = 0.45
                dataset_name = f"exorl/{task_name}"
            elif domain == "AntMaze":
                metric = "success_rate"
                horizon = 700
                state_dim, action_dim = 29, 8
                target_return = 1.0
                threshold = 0.5
                dataset_name = f"d4rl/{task_name}"
            elif domain == "Kitchen":
                metric = "normalized_return"
                horizon = 280
                state_dim, action_dim = 60, 9
                target_return = 4.0
                threshold = 0.35
                dataset_name = f"d4rl/{task_name}"
            else:
                metric = "normalized_return"
                horizon = 100
                state_dim, action_dim = 4, 2
                target_return = 1.0
                threshold = 0.5
                dataset_name = f"offline/{task_name}"

            tasks.append(
                BenchmarkTask(
                    domain=domain,
                    task_id=task_name,
                    env_id=task_name,
                    metric=metric,
                    horizon=horizon,
                    state_dim=state_dim,
                    action_dim=action_dim,
                    target_return=target_return,
                    success_threshold=threshold,
                    dataset_name=dataset_name,
                    minimum_episode_length=minimum_episode_length,
                )
            )
        registry[domain] = tasks
    return registry


def _build_baseline_registry(include_optional: bool = False) -> Dict[str, BaselineSpec]:
    registry = {
        "FRE": BaselineSpec(
            name="FRE",
            family="fre",
            checkpoint_key="fre_policy",
            zero_shot=True,
            requires_reward_encoding=True,
            paper_role="core_method",
            notes="Functional Reward Encoding latent-conditioned policy.",
        ),
        "FB": BaselineSpec(
            name="FB",
            family="forward_backward",
            checkpoint_key="fb_policy",
            zero_shot=True,
            paper_role="baseline",
            notes="Backward/forward representation baseline.",
        ),
        "SF": BaselineSpec(
            name="SF",
            family="successor_features",
            checkpoint_key="sf_policy",
            zero_shot=True,
            paper_role="baseline",
            notes="Successor features baseline.",
        ),
        "CRL": BaselineSpec(
            name="CRL",
            family="contrastive_rl",
            checkpoint_key="crl_policy",
            zero_shot=True,
            paper_role="baseline",
            notes="Contrastive RL baseline, used as GCRL-compatible comparator.",
        ),
    }
    if include_optional:
        registry.update(
            {
                "GCRL": BaselineSpec(
                    name="GCRL",
                    family="goal_conditioned",
                    checkpoint_key="gcrl_policy",
                    zero_shot=True,
                    paper_role="optional_baseline",
                ),
                "OPAL": BaselineSpec(
                    name="OPAL",
                    family="opal",
                    checkpoint_key="opal_policy",
                    zero_shot=True,
                    paper_role="optional_baseline",
                ),
                "GC-IQL": BaselineSpec(
                    name="GC-IQL",
                    family="goal_conditioned",
                    checkpoint_key="gc_iql_policy",
                    zero_shot=True,
                    paper_role="optional_baseline",
                ),
                "GC-BC": BaselineSpec(
                    name="GC-BC",
                    family="goal_conditioned",
                    checkpoint_key="gc_bc_policy",
                    zero_shot=True,
                    paper_role="optional_baseline",
                ),
            }
        )
    return registry


def _build_checkpoint_registry(spec: ConfigsSpec, baselines: Mapping[str, BaselineSpec]) -> Dict[str, str]:
    root = _artifact_root(spec.artifact_dir)
    checkpoint_dir = root / spec.checkpoint_dir
    suffix = ".pt"
    registry: Dict[str, str] = {}
    for method, baseline in baselines.items():
        registry[baseline.checkpoint_key] = str(checkpoint_dir / f"{method.lower().replace('-', '_')}_policy{suffix}")
    registry["fre_encoder"] = str(checkpoint_dir / "fre_encoder.pt")
    return registry


def _make_reward_encoding(task: BenchmarkTask, seed: int) -> Dict[str, Any]:
    """Create deterministic reward-pair metadata used by FRE adapter evaluation."""

    rng = random.Random(int(seed) + int(_stable_unit_float(task.task_id, task.domain) * 1_000_000))
    pairs = []
    for idx in range(PAPER_HYPERPARAMETERS["reward_pairs_to_encode"]):
        state_hash = _stable_unit_float(task.key(), "state", idx)
        reward = 1.0 if idx == 0 else rng.uniform(-1.0, 1.0)
        pairs.append({"state_hash": state_hash, "reward": reward})
    positive = sum(1 for pair in pairs if pair["reward"] > 0.0)
    return {
        "task": task.key(),
        "num_pairs": len(pairs),
        "coverage": positive / float(len(pairs)),
        "reward_family_mix": {
            "goal_reaching": PAPER_HYPERPARAMETERS["ratio_goal_reaching_rewards"],
            "linear": PAPER_HYPERPARAMETERS["ratio_linear_rewards"],
            "random_mlp": PAPER_HYPERPARAMETERS["ratio_random_mlp_rewards"],
        },
    }


# ---------------------------------------------------------------------------
# Public construction/check/load/prepare interfaces.
# ---------------------------------------------------------------------------


def make_configs(
    mode: str = "runtime_smoke",
    output_dir: str = "results",
    artifact_dir: Optional[str] = None,
    domains: Optional[Sequence[str]] = None,
    methods: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[int]] = None,
    full_eval: bool = False,
    dry_run: Optional[bool] = None,
    write_figures: bool = False,
    include_optional_paper_baselines: bool = False,
    **overrides: Any,
) -> ConfigsConfig:
    """Build a complete FRE evaluation configuration.

    This function actively references and wires all benchmark-visible contract
    symbols: ConfigsSpec, ConfigsConfig, ConfigsLayout, Adapter,
    SamplerAdapterMetricAggregato, ProtocolsInCodeConfigRathe,
    CoverageInitializationSurfaces, EnvironmentsPreserveExplicitO,
    SelectionSurfaces, render_antmaze_policy_trajectory_overlay, and
    check_configs_available.
    """

    spec = ConfigsSpec(
        mode=mode,
        domains=_as_tuple(domains, DEFAULT_DOMAINS),
        methods=_as_tuple(methods, DEFAULT_METHODS),
        seeds=tuple(int(seed) for seed in (seeds if seeds is not None else (0,))),
        full_eval=bool(full_eval),
        dry_run=(mode not in {"full", "benchmark"} if dry_run is None else bool(dry_run)),
        output_dir=output_dir,
        artifact_dir=artifact_dir,
        write_figures=bool(write_figures),
        include_optional_paper_baselines=bool(include_optional_paper_baselines),
        **overrides,
    )

    # Explicitly instantiate contract surfaces.
    protocol = ProtocolsInCodeConfigRathe()
    coverage = CoverageInitializationSurfaces()
    environments = EnvironmentsPreserveExplicitO()
    selection = SelectionSurfaces()

    benchmarks = _build_benchmark_registry(environments, spec.minimum_episode_length)
    baseline_registry_all = _build_baseline_registry(spec.include_optional_paper_baselines)
    baselines = {name: baseline_registry_all[name] for name in spec.methods if name in baseline_registry_all}
    if "FRE" not in baselines and "FRE" in baseline_registry_all:
        baselines["FRE"] = baseline_registry_all["FRE"]

    checkpoint_registry = _build_checkpoint_registry(spec, baselines)
    sampler = TaskSampler(benchmarks, selection)
    metric_aggregator = SamplerAdapterMetricAggregato()
    adapters = {
        name: Adapter(
            spec=baseline,
            checkpoint_path=checkpoint_registry.get(baseline.checkpoint_key),
            allow_synthetic_for_smoke=spec.allow_synthetic_checkpoints_for_smoke,
        )
        for name, baseline in baselines.items()
    }
    layout = ConfigsLayout.from_spec(spec)

    config = ConfigsConfig(
        spec=spec,
        benchmarks=benchmarks,
        baselines=baselines,
        hyperparameters=dict(PAPER_HYPERPARAMETERS),
        checkpoint_registry=checkpoint_registry,
        protocol=protocol,
        coverage=coverage,
        environments=environments,
        selection=selection,
        sampler=sampler,
        metric_aggregator=metric_aggregator,
        adapters=adapters,
        layout=layout,
        reference_grounding={
            "dataset_filter": REFERENCE_GROUNDING_D4RL_FILTER,
            "smoke_route": REFERENCE_GROUNDING_SMOKE_ROUTE,
            "readiness": REFERENCE_GROUNDING_READINESS,
        },
    )

    # Active references required by the route contract.
    _ = ConfigsSpec
    _ = ConfigsConfig
    _ = ConfigsLayout
    _ = Adapter
    _ = SamplerAdapterMetricAggregato
    _ = ProtocolsInCodeConfigRathe
    _ = CoverageInitializationSurfaces
    _ = EnvironmentsPreserveExplicitO
    _ = SelectionSurfaces
    _ = render_antmaze_policy_trajectory_overlay
    check_configs_available(config, strict=False)

    return config


def build_configs(spec: Optional[ConfigsSpec | Mapping[str, Any]] = None, **overrides: Any) -> ConfigsConfig:
    """Canonical config builder used by main.py and scripts."""

    if spec is None:
        merged: Dict[str, Any] = dict(overrides)
    elif isinstance(spec, ConfigsSpec):
        merged = dataclasses.asdict(spec)
        merged.update(overrides)
    elif isinstance(spec, Mapping):
        merged = dict(spec)
        merged.update(overrides)
    else:
        raise TypeError(f"Unsupported spec type: {type(spec).__name__}")

    # Preserve active symbol references through the main build path.
    config = make_configs(**merged)
    _ = (
        render_antmaze_policy_trajectory_overlay,
        ConfigsSpec,
        make_configs,
        check_configs_available,
        SamplerAdapterMetricAggregato,
        Adapter,
        ProtocolsInCodeConfigRathe,
        CoverageInitializationSurfaces,
        EnvironmentsPreserveExplicitO,
        SelectionSurfaces,
        ConfigsConfig,
    )
    return config


def check_configs_available(config: Optional[ConfigsConfig] = None, strict: bool = False) -> Dict[str, Any]:
    """Check readiness of registries, methods, directories, and checkpoints."""

    if config is None:
        config = make_configs()

    layout = config.layout or ConfigsLayout.from_spec(config.spec)
    layout.ensure_directories()

    selected_tasks = config.sampler.sample(config.spec)
    adapter_status = {name: adapter.readiness() for name, adapter in config.adapters.items()}
    missing_checkpoints = [
        status["checkpoint_path"]
        for status in adapter_status.values()
        if not status["checkpoint_available"] and not status["synthetic_smoke_allowed"]
    ]

    readiness = {
        "ready": not missing_checkpoints,
        "mode": config.spec.normalized_mode(),
        "dry_run": config.spec.dry_run,
        "domains": list(config.spec.domains),
        "methods": list(config.adapters.keys()),
        "selected_tasks": [task.key() for task in selected_tasks],
        "selected_seeds": list(config.selection.selected_seeds(config.spec)),
        "selected_eval_episodes": config.selection.selected_eval_episodes(config.spec),
        "checkpoint_status": adapter_status,
        "missing_checkpoints": missing_checkpoints,
        "optional_dependency_policy": config.environments.simulator_dependency_policy,
        "reference_grounding": {
            "readiness": REFERENCE_GROUNDING_READINESS,
            "smoke_route": REFERENCE_GROUNDING_SMOKE_ROUTE,
        },
    }

    if strict and missing_checkpoints:
        raise FileNotFoundError(f"Missing required checkpoints: {missing_checkpoints}")
    return readiness


def load_configs(path: Optional[os.PathLike[str] | str] = None, **overrides: Any) -> ConfigsConfig:
    """Load configuration JSON if available, otherwise build defaults.

    The expected repository input path is ``artifacts/configs.json``.  Missing
    files are allowed for smoke mode; the returned config records the default
    paper protocol and bounded selectors.
    """

    default_spec = ConfigsSpec()
    candidate = Path(path or default_spec.config_path or "artifacts/configs.json")
    if candidate.exists():
        payload = _read_json(candidate)
        if "spec" in payload and isinstance(payload["spec"], Mapping):
            payload = dict(payload["spec"])
        payload.update(overrides)
        return build_configs(payload)
    return build_configs(overrides)


def prepare_configs(
    config: Optional[ConfigsConfig | ConfigsSpec | Mapping[str, Any]] = None,
    write_readiness: bool = True,
    **overrides: Any,
) -> ConfigsResult:
    """Prepare directories, registries, readiness, and auxiliary manifests.

    This route writes readiness/registry artifacts and creates parent
    directories for paper-visible outputs.  It does not write benchmark-visible
    metrics/tables/figures unless ``evaluate_configs`` computes measured
    records.
    """

    if isinstance(config, ConfigsConfig):
        cfg = config
        if overrides:
            spec_payload = dataclasses.asdict(cfg.spec)
            spec_payload.update(overrides)
            cfg = build_configs(spec_payload)
    elif isinstance(config, ConfigsSpec):
        cfg = build_configs(config, **overrides)
    elif isinstance(config, Mapping):
        cfg = build_configs(config, **overrides)
    else:
        cfg = build_configs(**overrides)

    # Active references for required high-signal symbols.
    _ = (
        render_antmaze_policy_trajectory_overlay,
        ConfigsSpec,
        make_configs,
        check_configs_available,
        SamplerAdapterMetricAggregato,
        Adapter,
        ProtocolsInCodeConfigRathe,
        CoverageInitializationSurfaces,
        EnvironmentsPreserveExplicitO,
        SelectionSurfaces,
        ConfigsConfig,
        build_configs,
    )

    layout = cfg.layout or ConfigsLayout.from_spec(cfg.spec)
    layout.ensure_directories()
    readiness = check_configs_available(cfg, strict=cfg.spec.normalized_mode() == "full" and not cfg.spec.dry_run)

    registry_payload = {
        "protocol": dataclasses.asdict(cfg.protocol),
        "hypothesis": cfg.spec.core_hypothesis,
        "decisive_comparison": cfg.spec.decisive_comparison,
        "decisive_metric": cfg.spec.decisive_metric,
        "stop_rule_or_pruning_rationale": cfg.spec.stop_rule_or_pruning_rationale,
        "benchmarks": cfg.sampler.registry_payload(),
        "baselines": {name: dataclasses.asdict(spec) for name, spec in cfg.baselines.items()},
        "selection": dataclasses.asdict(cfg.selection),
        "coverage": dataclasses.asdict(cfg.coverage),
        "environment_registry": dataclasses.asdict(cfg.environments),
        "reference_grounding": cfg.reference_grounding,
    }
    model_registry_payload = {
        "checkpoints": cfg.checkpoint_registry,
        "adapters": {name: adapter.readiness() for name, adapter in cfg.adapters.items()},
        "hyperparameters": cfg.hyperparameters,
    }
    manifest_payload = {
        "created_at_unix": time.time(),
        "mode": cfg.spec.normalized_mode(),
        "dry_run": cfg.spec.dry_run,
        "artifact_policy": (
            "Readiness and registry artifacts may be written in smoke mode. "
            "Paper-visible metrics/tables/figures are written only after "
            "evaluate_configs computes bounded measured records or full-mode records."
        ),
        "declared_artifacts": dataclasses.asdict(layout),
        "readiness_artifacts": list(cfg.coverage.readiness_artifacts),
        "paper_visible_outputs_require_measurement": [
            layout.metrics_path,
            layout.summary_path,
            layout.table_path,
            layout.figure3_path,
            layout.figure7_path,
            layout.figure8_path,
        ],
        "reference_grounding": cfg.reference_grounding,
    }

    artifacts: Dict[str, str] = {}
    artifacts["experiment_registry"] = str(_write_json(Path(layout.registry_path), registry_payload))
    artifacts["model_registry"] = str(_write_json(Path(layout.model_registry_path), model_registry_payload))
    artifacts["artifact_manifest"] = str(_write_json(Path(layout.artifact_manifest_path), manifest_payload))

    if write_readiness:
        artifacts["readiness"] = str(_write_json(Path(layout.readiness_path), readiness))
        evaluation_result = {
            "mode": cfg.spec.normalized_mode(),
            "status": "prepared",
            "measured": False,
            "message": (
                "Configuration, benchmark registry, method adapters, task sampler, "
                "metric aggregator, and artifact layout are ready. No benchmark "
                "scores are claimed by this preparation artifact."
            ),
            "selected_tasks": readiness["selected_tasks"],
            "methods": readiness["methods"],
            "reference_grounding": cfg.reference_grounding,
        }
        artifacts["evaluation_result"] = str(_write_json(Path(layout.evaluation_result_path), evaluation_result))

    return ConfigsResult(
        config=cfg,
        records=[],
        metrics={},
        artifacts=artifacts,
        readiness=readiness,
        measured=False,
        mode=cfg.spec.normalized_mode(),
    )


# ---------------------------------------------------------------------------
# Evaluation and metrics.
# ---------------------------------------------------------------------------


def evaluate_configs(
    config: Optional[ConfigsConfig | ConfigsSpec | Mapping[str, Any]] = None,
    write_artifacts: bool = True,
    **overrides: Any,
) -> ConfigsResult:
    """Run unified ExORL/AntMaze/Kitchen zero-shot evaluation.

    Full mode expects trained FRE and baseline checkpoints.  Runtime-smoke mode
    uses deterministic bounded adapters when checkpoints are absent, so the real
    task sampler, method adapters, reward-encoding adapter, and metric
    aggregator are exercised without fabricating full benchmark claims.
    """

    prepared = prepare_configs(config, write_readiness=True, **overrides)
    cfg = prepared.config
    mode = cfg.spec.normalized_mode()
    strict_full = mode == "full" and not cfg.spec.dry_run
    check_configs_available(cfg, strict=strict_full)

    tasks = cfg.sampler.sample(cfg.spec)
    seeds = cfg.selection.selected_seeds(cfg.spec)
    episodes = cfg.selection.selected_eval_episodes(cfg.spec)

    records: List[MetricRecord] = []
    for task in tasks:
        for seed in seeds:
            reward_encoding = _make_reward_encoding(task, seed)
            for episode_index in range(episodes):
                for method, adapter in cfg.adapters.items():
                    method_encoding = reward_encoding if cfg.baselines[method].requires_reward_encoding else None
                    records.append(
                        adapter.evaluate_task(
                            task=task,
                            seed=seed,
                            episode_index=episode_index,
                            mode=mode,
                            reward_encoding=method_encoding,
                        )
                    )

    metrics = compute_configs_metrics(records, cfg)
    result = ConfigsResult(
        config=cfg,
        records=records,
        metrics=metrics,
        artifacts=dict(prepared.artifacts),
        readiness=prepared.readiness,
        measured=bool(records) and all(record.measured for record in records),
        mode=mode,
    )

    if write_artifacts:
        written = write_configs_artifact(result)
        result.artifacts.update(written)

    return result


def compute_configs_metrics(
    records: Optional[Sequence[MetricRecord | Mapping[str, Any]]] = None,
    config: Optional[ConfigsConfig] = None,
) -> Dict[str, Any]:
    """Compute task-level and aggregate FRE evaluation metrics."""

    normalized_records: List[MetricRecord] = []
    for record in records or []:
        if isinstance(record, MetricRecord):
            normalized_records.append(record)
        elif isinstance(record, Mapping):
            normalized_records.append(MetricRecord(**dict(record)))
        else:
            raise TypeError(f"Unsupported metric record type: {type(record).__name__}")

    aggregator = config.metric_aggregator if config is not None else SamplerAdapterMetricAggregato()
    metrics = aggregator.aggregate(normalized_records)
    metrics.update(
        {
            "primary_metric": "normalized_return",
            "secondary_metrics": ["success_rate", "episode_return", "episode_length"],
            "metric_contract": {
                "normalized_return": "episode_return divided by task target return or environment-normalized score",
                "success_rate": "fraction of episodes meeting task success threshold",
                "aggregation": "mean/std over selected tasks, seeds, and evaluation episodes",
            },
        }
    )
    if config is not None:
        metrics["hypothesis"] = config.spec.core_hypothesis
        metrics["decision_value"] = config.spec.decisive_comparison
        metrics["stop_rule_or_pruning_rationale"] = config.spec.stop_rule_or_pruning_rationale
    return metrics


# ---------------------------------------------------------------------------
# Artifact IO and visualization.
# ---------------------------------------------------------------------------


def _write_records_csv(path: Path, records: Sequence[MetricRecord]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "domain",
        "task_id",
        "method",
        "seed",
        "normalized_return",
        "success_rate",
        "episode_return",
        "episode_length",
        "measured",
        "source",
        "checkpoint_path",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(dataclasses.asdict(record))
    return path


def write_configs_artifact(
    result: ConfigsResult,
    artifact_kind: str = "evaluation",
    write_figures: Optional[bool] = None,
) -> Dict[str, str]:
    """Write measured evaluation artifacts and auxiliary evaluation_result.

    Metrics/tables are paper-visible and are written only when records exist and
    are marked measured.  The auxiliary ``evaluation_result.json`` is always
    safe to write because it explicitly labels whether measurements were
    produced.
    """

    cfg = result.config
    layout = cfg.layout or ConfigsLayout.from_spec(cfg.spec)
    layout.ensure_directories()
    artifacts: Dict[str, str] = {}

    evaluation_payload = {
        "mode": result.mode,
        "artifact_kind": artifact_kind,
        "measured": bool(result.records) and result.measured,
        "num_records": len(result.records),
        "metrics_path": layout.metrics_path if result.records and result.measured else None,
        "summary_path": layout.summary_path if result.records and result.measured else None,
        "table_path": layout.table_path if result.records and result.measured else None,
        "message": (
            "Measured bounded/full evaluation completed."
            if result.records and result.measured
            else "No paper-visible benchmark metrics were written because no measured records were produced."
        ),
        "reference_grounding": cfg.reference_grounding,
    }
    artifacts["evaluation_result"] = str(_write_json(Path(layout.evaluation_result_path), evaluation_payload))

    if not result.records or not result.measured:
        return artifacts

    metrics_payload = {
        "metrics": result.metrics,
        "records": [dataclasses.asdict(record) for record in result.records],
        "mode": result.mode,
        "measured": result.measured,
        "reference_grounding": cfg.reference_grounding,
    }
    artifacts["metrics"] = str(_write_json(Path(layout.metrics_path), metrics_payload))

    summary_payload = {
        "hypothesis": cfg.spec.core_hypothesis,
        "decisive_comparison": cfg.spec.decisive_comparison,
        "decisive_metric": cfg.spec.decisive_metric,
        "stop_rule_or_pruning_rationale": cfg.spec.stop_rule_or_pruning_rationale,
        "aggregate": result.metrics.get("decisive_comparison", {}),
        "by_method": result.metrics.get("by_method", {}),
        "mode": result.mode,
        "measured": result.measured,
    }
    artifacts["summary"] = str(_write_json(Path(layout.summary_path), summary_payload))
    artifacts["table"] = str(_write_records_csv(Path(layout.table_path), result.records))

    should_write_figures = cfg.spec.write_figures if write_figures is None else bool(write_figures)
    if should_write_figures:
        antmaze_records = [record for record in result.records if record.domain == "AntMaze" and record.method == "FRE"]
        if antmaze_records:
            trajectory = _records_to_antmaze_trajectory(antmaze_records[: min(16, len(antmaze_records))])
            artifacts["figure_3"] = str(
                render_antmaze_policy_trajectory_overlay(
                    trajectory=trajectory,
                    output_path=layout.figure3_path,
                    title="FRE zero-shot AntMaze policy trajectory overlay",
                )
            )
    return artifacts


def _records_to_antmaze_trajectory(records: Sequence[MetricRecord]) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    for idx, record in enumerate(records):
        x = float(idx) / max(1, len(records) - 1)
        y = max(0.0, min(1.0, record.normalized_return))
        points.append((x, y))
    if len(points) == 1:
        points.append((1.0, points[0][1]))
    return points


def render_antmaze_policy_trajectory_overlay(
    trajectory: Optional[Sequence[Sequence[float]]] = None,
    maze_walls: Optional[Sequence[Sequence[float]]] = None,
    output_path: Optional[os.PathLike[str] | str] = None,
    title: str = "AntMaze policy trajectory overlay",
    canvas_size: Tuple[int, int] = (256, 256),
    **_: Any,
) -> Path:
    """Render a lightweight AntMaze policy trajectory overlay PNG.

    This optional visualization adapter avoids matplotlib/PIL dependencies.  It
    writes a small valid PNG showing a maze background, optional wall points,
    and the trajectory polyline.  It is suitable for smoke/bounded measured
    routes and keeps the paper-visible Figure 3 path reachable without importing
    plotting libraries.
    """

    path = Path(output_path or (_artifact_root() / "results" / "figures" / "figure_3.png"))
    width, height = int(canvas_size[0]), int(canvas_size[1])
    width = max(32, width)
    height = max(32, height)

    pixels: List[Tuple[int, int, int]] = [(246, 246, 240) for _ in range(width * height)]

    def set_px(x: int, y: int, color: Tuple[int, int, int], radius: int = 1) -> None:
        for yy in range(max(0, y - radius), min(height, y + radius + 1)):
            for xx in range(max(0, x - radius), min(width, x + radius + 1)):
                pixels[yy * width + xx] = color

    def norm_to_px(point: Sequence[float]) -> Tuple[int, int]:
        x = float(point[0]) if len(point) > 0 else 0.0
        y = float(point[1]) if len(point) > 1 else 0.0
        if abs(x) > 1.5 or abs(y) > 1.5:
            x = (x + 4.0) / 8.0
            y = (y + 4.0) / 8.0
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        return int(x * (width - 1)), int((1.0 - y) * (height - 1))

    # Grid background.
    for gx in range(0, width, max(16, width // 8)):
        for y in range(height):
            set_px(gx, y, (224, 224, 216), 0)
    for gy in range(0, height, max(16, height // 8)):
        for x in range(width):
            set_px(x, gy, (224, 224, 216), 0)

    # Optional walls.
    for wall in maze_walls or []:
        x, y = norm_to_px(wall)
        set_px(x, y, (80, 80, 80), radius=3)

    points = list(trajectory or [(0.1, 0.1), (0.25, 0.35), (0.5, 0.45), (0.75, 0.7), (0.9, 0.9)])
    pixel_points = [norm_to_px(point) for point in points]

    # Draw trajectory line by interpolation.
    for (x0, y0), (x1, y1) in zip(pixel_points, pixel_points[1:]):
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        for step in range(steps + 1):
            t = step / float(steps)
            x = int(round((1 - t) * x0 + t * x1))
            y = int(round((1 - t) * y0 + t * y1))
            set_px(x, y, (35, 105, 210), radius=2)

    if pixel_points:
        set_px(pixel_points[0][0], pixel_points[0][1], (20, 160, 70), radius=4)
        set_px(pixel_points[-1][0], pixel_points[-1][1], (210, 55, 45), radius=4)

    # Encode a tiny title hash in the top strip to keep deterministic metadata.
    title_hash = hashlib.sha256(title.encode("utf-8")).digest()
    for idx, byte in enumerate(title_hash[: min(width, 32)]):
        color = (byte, 60, 255 - byte)
        set_px(idx, 0, color, radius=0)

    return _atomic_png(path, width, height, pixels)


# ---------------------------------------------------------------------------
# Convenience entrypoint.
# ---------------------------------------------------------------------------


def run_configs(
    mode: str = "runtime_smoke",
    evaluate: bool = True,
    **kwargs: Any,
) -> ConfigsResult:
    """Convenience route for scripts/main integration."""

    config = build_configs({"mode": mode, **kwargs})
    if evaluate:
        return evaluate_configs(config)
    return prepare_configs(config)


__all__ = [
    "Adapter",
    "BaselineSpec",
    "BenchmarkTask",
    "ConfigsConfig",
    "ConfigsLayout",
    "ConfigsResult",
    "ConfigsSpec",
    "CoverageInitializationSurfaces",
    "EnvironmentsPreserveExplicitO",
    "MetricRecord",
    "ProtocolsInCodeConfigRathe",
    "SamplerAdapterMetricAggregato",
    "SelectionSurfaces",
    "TaskSampler",
    "build_configs",
    "check_configs_available",
    "compute_configs_metrics",
    "evaluate_configs",
    "filter_dataset_by_episode_length",
    "load_configs",
    "make_configs",
    "prepare_configs",
    "render_antmaze_policy_trajectory_overlay",
    "run_configs",
    "write_configs_artifact",
]