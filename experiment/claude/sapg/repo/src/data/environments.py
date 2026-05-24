# src/data/environments.py
# reference_grounding: wp_004 src/data/environments.py
#
# Paper-derived environment/task registry for SAPG: Split and Aggregate Policy Gradients.
# Exposes named task descriptors, benchmark registry, metric formulas, factory hooks,
# and smoke/readiness artifact writers.
#
# All heavy simulator imports (isaacgym, gym, torch) are lazy-guarded.
# This module is importable in a minimal code-only environment.

from __future__ import annotations

import importlib
import json
import os
import pathlib
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Availability helpers
# ---------------------------------------------------------------------------

def _pkg_available(name: str) -> bool:
    """Return True if *name* can be imported without actually importing it."""
    return importlib.util.find_spec(name) is not None


ISAACGYM_AVAILABLE = _pkg_available("isaacgym")
GYM_AVAILABLE = _pkg_available("gym") or _pkg_available("gymnasium")
TORCH_AVAILABLE = _pkg_available("torch")


# ---------------------------------------------------------------------------
# Task / environment descriptor
# ---------------------------------------------------------------------------

@dataclass
class TaskDescriptor:
    """Lightweight, import-free descriptor for a paper-derived task.

    Fields mirror the information needed to instantiate an IsaacGym task via
    the isaacgymenvs task registry, plus SAPG-specific metadata.
    """

    task_id: str                          # canonical id used in configs
    aliases: List[str]                    # alternative names used in the paper
    display_name: str                     # human-readable label
    robot: str                            # robot platform
    difficulty: str                       # easy | medium | hard
    obs_dim: int                          # observation space dimension
    act_dim: int                          # action space dimension
    num_envs_default: int                 # default parallel env count (paper: 24576)
    max_episode_length: int               # episode horizon
    success_threshold: float             # reward threshold for "solved"
    paper_table: str                      # which paper table/figure reports this task
    isaacgym_task_name: str              # name passed to isaacgymenvs
    config_file: str                      # relative path to task yaml
    extra_cfg: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Paper-derived task registry
# reference_grounding: wp_004 src/data/environments.py (Table 1, Figure 5)
# ---------------------------------------------------------------------------

_TASK_REGISTRY: Dict[str, TaskDescriptor] = {}


def _register(td: TaskDescriptor) -> TaskDescriptor:
    _TASK_REGISTRY[td.task_id] = td
    for alias in td.aliases:
        _TASK_REGISTRY[alias] = td
    return td


# --- Shadow Hand tasks (Table 1, Figure 5) ---

_register(TaskDescriptor(
    task_id="ShadowHandOver",
    aliases=["shadow_hand_over", "ShadowHand", "Shadow Hand", "shadow_hand"],
    display_name="Shadow Hand Over",
    robot="shadow_hand",
    difficulty="hard",
    obs_dim=211,
    act_dim=24,
    num_envs_default=24576,
    max_episode_length=200,
    success_threshold=0.9,
    paper_table="Table 1",
    isaacgym_task_name="ShadowHandOver",
    config_file="configs/tasks/shadow_hand_over.yaml",
    extra_cfg={"reset_on_success": False},
))

_register(TaskDescriptor(
    task_id="ShadowHandCatchUnderarm",
    aliases=["shadow_hand_catch_underarm", "ShadowHandCatch"],
    display_name="Shadow Hand Catch Underarm",
    robot="shadow_hand",
    difficulty="hard",
    obs_dim=211,
    act_dim=24,
    num_envs_default=24576,
    max_episode_length=200,
    success_threshold=0.9,
    paper_table="Table 1",
    isaacgym_task_name="ShadowHandCatchUnderarm",
    config_file="configs/tasks/shadow_hand_catch_underarm.yaml",
    extra_cfg={"reset_on_success": False},
))

_register(TaskDescriptor(
    task_id="ShadowHandCatchAbreast",
    aliases=["shadow_hand_catch_abreast"],
    display_name="Shadow Hand Catch Abreast",
    robot="shadow_hand",
    difficulty="hard",
    obs_dim=211,
    act_dim=24,
    num_envs_default=24576,
    max_episode_length=200,
    success_threshold=0.9,
    paper_table="Table 1",
    isaacgym_task_name="ShadowHandCatchAbreast",
    config_file="configs/tasks/shadow_hand_catch_abreast.yaml",
    extra_cfg={"reset_on_success": False},
))

_register(TaskDescriptor(
    task_id="ShadowHandReOrientation",
    aliases=[
        "shadow_hand_reorientation",
        "Reorientation",
        "reorientation",
        "Shadow Hand Reorientation",
    ],
    display_name="Shadow Hand Re-Orientation",
    robot="shadow_hand",
    difficulty="hard",
    obs_dim=211,
    act_dim=24,
    num_envs_default=24576,
    max_episode_length=200,
    success_threshold=0.9,
    paper_table="Table 1",
    isaacgym_task_name="ShadowHandReOrientation",
    config_file="configs/tasks/shadow_hand_reorientation.yaml",
    extra_cfg={"reset_on_success": False},
))

# --- Allegro Hand / AllegroKuka tasks (Table 1, Figure 5) ---

_register(TaskDescriptor(
    task_id="AllegroHandReOrientation",
    aliases=[
        "allegro_hand_reorientation",
        "Allegro Kuka Reorientation",
        "allegro_kuka_reorientation",
        "AllegroKukaReorientation",
    ],
    display_name="Allegro Hand Re-Orientation",
    robot="allegro_hand",
    difficulty="hard",
    obs_dim=88,
    act_dim=16,
    num_envs_default=24576,
    max_episode_length=200,
    success_threshold=0.9,
    paper_table="Table 1",
    isaacgym_task_name="AllegroHandReOrientation",
    config_file="configs/tasks/allegro_hand_reorientation.yaml",
    extra_cfg={"reset_on_success": False},
))

_register(TaskDescriptor(
    task_id="AllegroKuka",
    aliases=["allegro_kuka", "AllegroKuka", "Allegro Kuka"],
    display_name="AllegroKuka",
    robot="allegro_kuka",
    difficulty="hard",
    obs_dim=88,
    act_dim=23,
    num_envs_default=24576,
    max_episode_length=200,
    success_threshold=0.9,
    paper_table="Table 1 / Figure 5",
    isaacgym_task_name="AllegroKuka",
    config_file="configs/tasks/manipulation_hard.yaml",
    extra_cfg={"reset_on_success": False},
))

_register(TaskDescriptor(
    task_id="harder_AllegroKuka",
    aliases=[
        "harder_allegro_kuka",
        "harder AllegroKuka",
        "AllegroKukaHard",
        "allegro_kuka_hard",
    ],
    display_name="Harder AllegroKuka",
    robot="allegro_kuka",
    difficulty="hard",
    obs_dim=88,
    act_dim=23,
    num_envs_default=24576,
    max_episode_length=200,
    success_threshold=0.9,
    paper_table="Table 1 / Figure 5",
    isaacgym_task_name="AllegroKukaHard",
    config_file="configs/tasks/manipulation_hard.yaml",
    extra_cfg={"reset_on_success": False, "harder_variant": True},
))

# --- Sub-task variants referenced in the paper ---

_register(TaskDescriptor(
    task_id="Throw",
    aliases=["throw", "AllegroKukaThrow", "allegro_kuka_throw"],
    display_name="AllegroKuka Throw",
    robot="allegro_kuka",
    difficulty="hard",
    obs_dim=88,
    act_dim=23,
    num_envs_default=24576,
    max_episode_length=200,
    success_threshold=0.9,
    paper_table="Table 1",
    isaacgym_task_name="AllegroKukaThrow",
    config_file="configs/tasks/manipulation_hard.yaml",
    extra_cfg={"subtask": "throw"},
))

_register(TaskDescriptor(
    task_id="Regrasping",
    aliases=["regrasping", "AllegroKukaRegrasping", "allegro_kuka_regrasping"],
    display_name="AllegroKuka Regrasping",
    robot="allegro_kuka",
    difficulty="hard",
    obs_dim=88,
    act_dim=23,
    num_envs_default=24576,
    max_episode_length=200,
    success_threshold=0.9,
    paper_table="Table 1",
    isaacgym_task_name="AllegroKukaRegrasping",
    config_file="configs/tasks/manipulation_hard.yaml",
    extra_cfg={"subtask": "regrasping"},
))

# ---------------------------------------------------------------------------
# Benchmark registry
# reference_grounding: wp_004 src/data/environments.py (Table 1)
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkEntry:
    """A named benchmark grouping tasks and specifying evaluation protocol."""

    benchmark_id: str
    description: str
    task_ids: List[str]
    num_samples: float          # total env-steps used in paper (e.g. 2e10)
    metric: str                 # primary metric name
    higher_is_better: bool
    paper_reference: str
    methods_compared: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_BENCHMARK_REGISTRY: Dict[str, BenchmarkEntry] = {}


def _register_benchmark(be: BenchmarkEntry) -> BenchmarkEntry:
    _BENCHMARK_REGISTRY[be.benchmark_id] = be
    return be


_register_benchmark(BenchmarkEntry(
    benchmark_id="sapg_main",
    description=(
        "Main SAPG benchmark: Table 1 results after 2e10 samples across "
        "Shadow Hand and AllegroKuka tasks."
    ),
    task_ids=[
        "ShadowHandOver",
        "ShadowHandCatchUnderarm",
        "ShadowHandCatchAbreast",
        "ShadowHandReOrientation",
        "AllegroHandReOrientation",
        "AllegroKuka",
        "harder_AllegroKuka",
    ],
    num_samples=2e10,
    metric="mean_episode_reward",
    higher_is_better=True,
    paper_reference="Table 1",
    methods_compared=["ours", "sapg", "ppo", "pbt", "pql"],
))

_register_benchmark(BenchmarkEntry(
    benchmark_id="sapg_figure5",
    description=(
        "Figure 5 learning curves: SAPG vs PPO, PBT, PQL on AllegroKuka "
        "and Shadow Hand tasks."
    ),
    task_ids=["AllegroKuka", "harder_AllegroKuka", "ShadowHandOver"],
    num_samples=2e10,
    metric="mean_episode_reward",
    higher_is_better=True,
    paper_reference="Figure 5",
    methods_compared=["ours", "sapg", "ppo", "pbt", "pql"],
))

_register_benchmark(BenchmarkEntry(
    benchmark_id="sapg_ablation",
    description=(
        "Figure 6 ablation: symmetric aggregation, no off-policy, "
        "entropy coefficient variants."
    ),
    task_ids=["AllegroKuka"],
    num_samples=2e10,
    metric="mean_episode_reward",
    higher_is_better=True,
    paper_reference="Figure 6",
    methods_compared=[
        "ours",
        "symmetric_aggregation",
        "no_off_policy",
        "coef_0",
        "coef_0.005",
        "coef_0.01",
    ],
))

_register_benchmark(BenchmarkEntry(
    benchmark_id="sapg_coverage_pca",
    description=(
        "Figure 7: State space coverage analysis using PCA reconstruction "
        "error across policies."
    ),
    task_ids=["AllegroKuka"],
    num_samples=2e10,
    metric="pca_reconstruction_error",
    higher_is_better=False,
    paper_reference="Figure 7",
    methods_compared=["ours", "ppo"],
))

_register_benchmark(BenchmarkEntry(
    benchmark_id="sapg_coverage_mlp",
    description=(
        "Figure 8: State space coverage analysis using MLP reconstruction "
        "error. Two-layer ReLU network, Adam optimizer (PyTorch defaults)."
    ),
    task_ids=["AllegroKuka"],
    num_samples=2e10,
    metric="mlp_reconstruction_error",
    higher_is_better=False,
    paper_reference="Figure 8",
    methods_compared=["ours", "ppo"],
))

_register_benchmark(BenchmarkEntry(
    benchmark_id="sapg_subtasks",
    description=(
        "AllegroKuka sub-task breakdown: Throw, Regrasping, Reorientation."
    ),
    task_ids=["Throw", "Regrasping", "Reorientation"],
    num_samples=2e10,
    metric="mean_episode_reward",
    higher_is_better=True,
    paper_reference="Table 1",
    methods_compared=["ours", "sapg", "ppo"],
))


# ---------------------------------------------------------------------------
# Metric registry and formulas
# reference_grounding: wp_004 src/data/environments.py
# ---------------------------------------------------------------------------

@dataclass
class MetricSpec:
    metric_id: str
    description: str
    formula_str: str            # human-readable formula
    higher_is_better: bool
    unit: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_METRIC_REGISTRY: Dict[str, MetricSpec] = {
    "mean_episode_reward": MetricSpec(
        metric_id="mean_episode_reward",
        description=(
            "Mean undiscounted episode reward averaged over all parallel "
            "environments. Primary metric in Table 1 and Figure 5."
        ),
        formula_str="mean(sum(r_t for t in episode) for episode in episodes)",
        higher_is_better=True,
        unit="reward",
    ),
    "success_rate": MetricSpec(
        metric_id="success_rate",
        description="Fraction of episodes where the task success condition is met.",
        formula_str="mean(success_flag for episode in episodes)",
        higher_is_better=True,
        unit="fraction [0,1]",
    ),
    "pca_reconstruction_error": MetricSpec(
        metric_id="pca_reconstruction_error",
        description=(
            "PCA reconstruction error of state observations. Lower error "
            "indicates less state diversity (Figure 7)."
        ),
        formula_str="mean(||x - PCA_reconstruct(x)||^2 for x in states)",
        higher_is_better=False,
        unit="MSE",
    ),
    "mlp_reconstruction_error": MetricSpec(
        metric_id="mlp_reconstruction_error",
        description=(
            "MLP reconstruction error of state observations. Two-layer ReLU "
            "network trained with Adam (PyTorch defaults). Figure 8."
        ),
        formula_str="mean(||x - MLP_reconstruct(x)||^2 for x in states)",
        higher_is_better=False,
        unit="MSE",
    ),
    "importance_sampling_ratio": MetricSpec(
        metric_id="importance_sampling_ratio",
        description=(
            "Per-sample importance sampling ratio pi_leader(a|s) / pi_follower_i(a|s) "
            "used in SAPG aggregation step."
        ),
        formula_str="pi_leader(a|s) / pi_follower_i(a|s)",
        higher_is_better=False,
        unit="ratio",
    ),
}


def compute_mean_episode_reward(rewards: List[float]) -> float:
    """Compute mean episode reward from a flat list of per-step rewards.

    In IsaacGym the environment resets automatically; rewards here are
    already episode-summed scalars (one per completed episode).
    """
    if not rewards:
        return 0.0
    return sum(rewards) / len(rewards)


def compute_success_rate(success_flags: List[bool]) -> float:
    """Fraction of episodes where the task success condition was met."""
    if not success_flags:
        return 0.0
    return sum(bool(f) for f in success_flags) / len(success_flags)


def compute_pca_reconstruction_error(
    states: "Any",  # numpy array [N, D]
    n_components: int = 32,
) -> float:
    """PCA reconstruction MSE for state-space coverage analysis (Figure 7).

    Lazy-imports numpy and sklearn; raises ImportError with a clear message
    if unavailable.
    """
    try:
        import numpy as np
        from sklearn.decomposition import PCA
    except ImportError as exc:
        raise ImportError(
            "numpy and scikit-learn are required for PCA coverage analysis. "
            "Install them with: pip install numpy scikit-learn"
        ) from exc

    states_np = np.asarray(states)
    pca = PCA(n_components=min(n_components, states_np.shape[1]))
    projected = pca.fit_transform(states_np)
    reconstructed = pca.inverse_transform(projected)
    return float(np.mean((states_np - reconstructed) ** 2))


def compute_mlp_reconstruction_error(
    states: "Any",  # numpy array [N, D]
    hidden_size: int = 256,
    epochs: int = 50,
    lr: float = 1e-3,
) -> float:
    """MLP reconstruction MSE for state-space coverage analysis (Figure 8).

    Two-layer ReLU autoencoder trained with Adam (PyTorch defaults).
    Lazy-imports torch; raises ImportError with a clear message if unavailable.
    """
    try:
        import numpy as np
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError as exc:
        raise ImportError(
            "numpy and torch are required for MLP coverage analysis. "
            "Install them with: pip install numpy torch"
        ) from exc

    states_np = np.asarray(states, dtype=np.float32)
    D = states_np.shape[1]
    X = torch.from_numpy(states_np)

    # Two-layer ReLU autoencoder (Figure 8: hidden_size shown on x-axis)
    encoder = nn.Sequential(nn.Linear(D, hidden_size), nn.ReLU())
    decoder = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, D))
    model = nn.Sequential(encoder, decoder)

    optimizer = optim.Adam(model.parameters(), lr=lr)  # PyTorch defaults
    criterion = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, X)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        out = model(X)
        mse = criterion(out, X).item()
    return float(mse)


# ---------------------------------------------------------------------------
# Environment factory / readiness check
# ---------------------------------------------------------------------------

class EnvironmentUnavailableError(RuntimeError):
    """Raised when a required simulator is not installed."""


def check_environment_readiness(task_id: str) -> Dict[str, Any]:
    """Return a readiness dict for *task_id* without instantiating the env."""
    td = get_task(task_id)
    status: Dict[str, Any] = {
        "task_id": task_id,
        "display_name": td.display_name,
        "isaacgym_available": ISAACGYM_AVAILABLE,
        "torch_available": TORCH_AVAILABLE,
        "ready": ISAACGYM_AVAILABLE and TORCH_AVAILABLE,
        "missing": [],
    }
    if not ISAACGYM_AVAILABLE:
        status["missing"].append("isaacgym")
    if not TORCH_AVAILABLE:
        status["missing"].append("torch")
    return status


def make_env(
    task_id: str,
    num_envs: Optional[int] = None,
    device: str = "cuda:0",
    headless: bool = True,
    cfg_overrides: Optional[Dict[str, Any]] = None,
) -> Any:
    """Factory: create an IsaacGym environment for *task_id*.

    Raises EnvironmentUnavailableError if isaacgym is not installed.
    All heavy imports are deferred to this function.
    """
    if not ISAACGYM_AVAILABLE:
        raise EnvironmentUnavailableError(
            "isaacgym is not installed. Follow the IsaacGym installation guide "
            "at https://developer.nvidia.com/isaac-gym and install "
            "isaacgymenvs: pip install isaacgymenvs"
        )
    if not TORCH_AVAILABLE:
        raise EnvironmentUnavailableError(
            "torch is required to run IsaacGym environments. "
            "Install it with: pip install torch"
        )

    import isaacgymenvs  # type: ignore
    import torch

    td = get_task(task_id)
    n = num_envs if num_envs is not None else td.num_envs_default
    cfg = {
        "task": td.isaacgym_task_name,
        "num_envs": n,
        "headless": headless,
        "device": device,
    }
    if cfg_overrides:
        cfg.update(cfg_overrides)
    if td.extra_cfg:
        for k, v in td.extra_cfg.items():
            cfg.setdefault(k, v)

    env = isaacgymenvs.make(**cfg)
    return env


def make_smoke_env(task_id: str) -> Dict[str, Any]:
    """Return a minimal smoke fixture dict without instantiating the real env.

    Used during --mode runtime_smoke and --mode docker_validate.
    """
    td = get_task(task_id)
    return {
        "task_id": task_id,
        "display_name": td.display_name,
        "obs_dim": td.obs_dim,
        "act_dim": td.act_dim,
        "num_envs": 4,  # minimal for smoke
        "max_episode_length": td.max_episode_length,
        "smoke": True,
        "note": "dry-run smoke fixture — not a real environment instance",
    }


# ---------------------------------------------------------------------------
# Public accessors
# ---------------------------------------------------------------------------

def get_task(task_id: str) -> TaskDescriptor:
    """Look up a TaskDescriptor by id or alias. Raises KeyError if not found."""
    if task_id not in _TASK_REGISTRY:
        available = sorted({td.task_id for td in _TASK_REGISTRY.values()})
        raise KeyError(
            f"Unknown task_id {task_id!r}. Available tasks: {available}"
        )
    return _TASK_REGISTRY[task_id]


def get_benchmark(benchmark_id: str) -> BenchmarkEntry:
    """Look up a BenchmarkEntry by id. Raises KeyError if not found."""
    if benchmark_id not in _BENCHMARK_REGISTRY:
        raise KeyError(
            f"Unknown benchmark_id {benchmark_id!r}. "
            f"Available: {list(_BENCHMARK_REGISTRY.keys())}"
        )
    return _BENCHMARK_REGISTRY[benchmark_id]


def list_tasks() -> List[str]:
    """Return canonical task ids (no aliases)."""
    return sorted({td.task_id for td in _TASK_REGISTRY.values()})


def list_benchmarks() -> List[str]:
    return sorted(_BENCHMARK_REGISTRY.keys())


def list_metrics() -> List[str]:
    return sorted(_METRIC_REGISTRY.keys())


def get_metric(metric_id: str) -> MetricSpec:
    if metric_id not in _METRIC_REGISTRY:
        raise KeyError(f"Unknown metric {metric_id!r}. Available: {list_metrics()}")
    return _METRIC_REGISTRY[metric_id]


# ---------------------------------------------------------------------------
# Evaluation entry point
# ---------------------------------------------------------------------------

def evaluate_predictions(config: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate predictions against paper metrics.

    Parameters
    ----------
    config : dict
        Keys:
          - task_id (str)
          - metric (str)
          - predictions (list[float])  — episode rewards or success flags
          - ground_truth (list[float], optional)

    Returns
    -------
    dict with metric_id, value, task_id, timestamp
    """
    task_id = config.get("task_id", "AllegroKuka")
    metric_id = config.get("metric", "mean_episode_reward")
    predictions = config.get("predictions", [])

    metric_spec = get_metric(metric_id)

    if metric_id == "mean_episode_reward":
        value = compute_mean_episode_reward([float(p) for p in predictions])
    elif metric_id == "success_rate":
        value = compute_success_rate([bool(p) for p in predictions])
    else:
        # Generic mean for other registered metrics
        value = float(sum(predictions) / len(predictions)) if predictions else 0.0

    return {
        "task_id": task_id,
        "metric_id": metric_id,
        "value": value,
        "higher_is_better": metric_spec.higher_is_better,
        "unit": metric_spec.unit,
        "n_samples": len(predictions),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# ---------------------------------------------------------------------------
# Artifact writers
# reference_grounding: wp_004 src/data/environments.py
# Writes: results/dataset_registry.json, results/metrics.json,
#         results/data_manifest.json
# ---------------------------------------------------------------------------

def _artifact_dir() -> pathlib.Path:
    base = os.environ.get("PAPERBENCH_REPRO_ARTIFACT_DIR", "results")
    return pathlib.Path(base)


def write_dataset_registry(dry_run: bool = True) -> pathlib.Path:
    """Write results/dataset_registry.json.

    When dry_run=True the file is labeled as a readiness/schema artifact.
    """
    out_dir = _artifact_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dataset_registry.json"

    unique_tasks = {td.task_id: td for td in _TASK_REGISTRY.values()}
    payload = {
        "_dry_run": dry_run,
        "_note": (
            "dry-run contract artifact — not real benchmark scores"
            if dry_run
            else "dataset registry"
        ),
        "tasks": {tid: td.to_dict() for tid, td in unique_tasks.items()},
        "benchmarks": {
            bid: be.to_dict() for bid, be in _BENCHMARK_REGISTRY.items()
        },
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def write_metrics_registry(dry_run: bool = True) -> pathlib.Path:
    """Write results/metrics.json."""
    out_dir = _artifact_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metrics.json"

    payload = {
        "_dry_run": dry_run,
        "_note": (
            "dry-run contract artifact — not real benchmark scores"
            if dry_run
            else "metrics registry"
        ),
        "metrics": {mid: ms.to_dict() for mid, ms in _METRIC_REGISTRY.items()},
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def write_data_manifest(dry_run: bool = True) -> pathlib.Path:
    """Write results/data_manifest.json."""
    out_dir = _artifact_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "data_manifest.json"

    readiness = {
        tid: check_environment_readiness(td.task_id)
        for tid, td in {td.task_id: td for td in _TASK_REGISTRY.values()}.items()
    }

    payload = {
        "_dry_run": dry_run,
        "_note": (
            "dry-run contract artifact — not real benchmark scores"
            if dry_run
            else "data manifest"
        ),
        "environment_readiness": readiness,
        "isaacgym_available": ISAACGYM_AVAILABLE,
        "torch_available": TORCH_AVAILABLE,
        "task_count": len(list_tasks()),
        "benchmark_count": len(list_benchmarks()),
        "metric_count": len(list_metrics()),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
