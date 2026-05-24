"""
ftrl_repro/evaluation.py

Evaluation interfaces, metric formulas, Close/FAR diagnostics, environment adapters,
policy factory, and artifact writers for the paper reproduction:
  "Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem"

reference_grounding: paperbench_ref_001 envs.py
reference_grounding: paperbench_ref_001 eval.py
reference_grounding: paperbench_ref_001 agents.py

Paper Section 2 — Close/FAR state partition:
  States in CLOSE are easily reachable from the starting state; agent frequently visits them.
  States in FAR are reachable only by going through CLOSE; infrequently visited.
  Forgetting of pre-trained capabilities: model performing well on FAR loses this ability
  due to interference in the function approximator when training on CLOSE.

Paper Section 3 — Environments:
  - NetHack Learning Environment: dungeon depth, turns, pretrained capability diagnostics
  - Montezuma's Revenge: Atari-style obs/action, Room 7 = FAR states (Figure 6, Figure 12)
  - RoboticSequence: peg-unplug-side, push-wall, per-stage success rate (Figure 7)
  - Two-state MDPs: toy Close/FAR mechanism (Figure 9, Appendix A)
  - AppleRetrieval: toy grid-world forgetting demo (Figure 10, Appendix A)

Paper Section 4 — Main result:
  vanilla fine-tuning often fails to leverage pre-trained knowledge.
  knowledge retention methods (BC, EWC, EM) mitigate forgetting and unlock pre-trained model.

Paper Section 5 — Analysis:
  FAR/CLOSE visitation, NetHack density plots, RoboticSequence stage success rates.

Forward Transfer metric (Appendix F):
  FT = (AUC - AUC_b) / (1 - AUC_b)
  where AUC = (1/T) integral_0^T p(t) dt, AUC_b = scratch baseline AUC.

Result trend assertions (semantic metadata, not hard-coded scores):
  - vanilla fine-tuning often fails to leverage pre-trained knowledge
  - knowledge retention methods mitigate forgetting without hard-coding benchmark scores
  - state coverage gap can cause deterioration of prior knowledge
  - knowledge retention methods fix this problem
  - BC, EM, and EWC maintain or partly regain performance
  - knowledge retention methods unlock the potential of the pre-trained model
  - standard fine-tuning does not exhibit positive transfer of the knowledge of the last two stages
  - fine-tuning experiments start from pi_* that performs well on peg-unplug-side and push-wall

Artifact paths (statically discoverable):
  results/state_region_metrics.json
  results/environment_registry.json
  results/environment_manifest.json
  results/figures/figure_1.png  (mechanism via toy MDP config)
  results/figures/figure_2.png
  results/figures/figure_3.png
  results/figures/figure_3a.png
  results/figures/figure_3b.png
  results/figures/figure_3c.png
  results/figures/figure_4.png  (NOTE: not required per addendum)
  results/figures/figure_5.png
  results/figures/figure_6.png
  results/figures/figure_7.png
  results/figures/figure_8.png
  results/figures/figure_9.png
  results/figures/figure_12.png
  results/figures/figure_14.png
  results/figures/figure_15.png ... figure_27.png
  results/tables/table_1.csv
  results/tables/table_4.csv
  results/tables/table_5.csv
  results/tables/table_6.csv

NOTE: This repository does NOT use the blacklisted repository
      https://github.com/BartekCupial/finetuning-RL-as-CL

Lazy imports: torch, gym/gymnasium, nle are imported inside functions only.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union


# ---------------------------------------------------------------------------
# Paper-derived constants
# ---------------------------------------------------------------------------

# Fixed hyperparameter (paper evidence contract: batch_size_128)
BATCH_SIZE_128: int = 128

# RoboticSequence stage identifiers (Section 3, Figure 7)
ROBOTIC_SEQUENCE_STAGES: List[str] = [
    "peg-unplug-side",   # pre-trained stage 1
    "push-wall",         # pre-trained stage 2
    "pick-place",        # downstream stage
    "door-open",         # downstream stage
]

# Montezuma's Revenge FAR room (Figure 6, Figure 12)
MONTEZUMA_FAR_ROOM: int = 7
MONTEZUMA_CLOSE_ROOMS: List[int] = [0, 1, 2, 3]

# NetHack dungeon level thresholds for Close/FAR partition
NETHACK_CLOSE_MAX_LEVEL: int = 3   # levels 1-3 = CLOSE
NETHACK_FAR_MIN_LEVEL: int = 4     # levels 4+ = FAR

# Toy MDP state identifiers (Figure 9, Appendix A)
TOY_MDP_CLOSE_STATE: int = 0
TOY_MDP_FAR_STATE: int = 1

# AppleRetrieval phase identifiers (Figure 10, Appendix A)
APPLE_RETRIEVAL_CLOSE_PHASE: int = 1   # Phase 1: navigate to apple
APPLE_RETRIEVAL_FAR_PHASE: int = 2     # Phase 2: return to house

# Methods registry (paper evidence contract)
METHODS_REGISTRY: List[str] = [
    "training_from_scratch",
    "vanilla_finetune",
    "finetune_bc",       # Fine-tuning + BC (knowledge retention)
    "finetune_ewc",      # Fine-tuning + EWC (knowledge retention)
    "finetune_em",       # Fine-tuning + EM (knowledge retention)
    "finetune_ks",       # Fine-tuning + KS (knowledge retention, used in NetHack)
    "oracle",            # Oracle upper bound
    "ppo",               # PPO baseline
    "sac",               # SAC baseline (RoboticSequence)
    "bc",                # Behavioral cloning
    "nle",               # NLE baseline
    "ewc",               # EWC standalone
    "ours",              # Proposed method alias
    "scaled_bc_finetune_ks",  # Table 5 comparison
]

# Environments registry (paper evidence contract)
ENVIRONMENTS_REGISTRY: List[str] = [
    "nethack",
    "montezuma",
    "robotic_sequence",
    "robotics",          # alias for robotic_sequence (paper evidence contract)
    "two_state_mdp",
    "apple_retrieval",
]

# Artifact paths (statically discoverable)
ARTIFACT_PATHS: Dict[str, str] = {
    "state_region_metrics": "results/state_region_metrics.json",
    "environment_registry": "results/environment_registry.json",
    "environment_manifest": "results/environment_manifest.json",
    "metrics_json": "results/metrics.json",
    "summary_csv": "results/summary.csv",
    "figure_1": "results/figures/figure_1.png",
    "figure_2": "results/figures/figure_2.png",
    "figure_3": "results/figures/figure_3.png",
    "figure_3a": "results/figures/figure_3a.png",
    "figure_3b": "results/figures/figure_3b.png",
    "figure_3c": "results/figures/figure_3c.png",
    "figure_4": "results/figures/figure_4.png",
    "figure_5": "results/figures/figure_5.png",
    "figure_6": "results/figures/figure_6.png",
    "figure_7": "results/figures/figure_7.png",
    "figure_8": "results/figures/figure_8.png",
    "figure_9": "results/figures/figure_9.png",
    "figure_12": "results/figures/figure_12.png",
    "figure_14": "results/figures/figure_14.png",
    "figure_15": "results/figures/figure_15.png",
    "figure_16": "results/figures/figure_16.png",
    "figure_17": "results/figures/figure_17.png",
    "figure_18": "results/figures/figure_18.png",
    "figure_19": "results/figures/figure_19.png",
    "figure_20": "results/figures/figure_20.png",
    "figure_21": "results/figures/figure_21.png",
    "figure_22": "results/figures/figure_22.png",
    "figure_23": "results/figures/figure_23.png",
    "figure_24": "results/figures/figure_24.png",
    "figure_25": "results/figures/figure_25.png",
    "figure_26": "results/figures/figure_26.png",
    "figure_27": "results/figures/figure_27.png",
    "table_1": "results/tables/table_1.csv",
    "table_4": "results/tables/table_4.csv",
    "table_5": "results/tables/table_5.csv",
    "table_6": "results/tables/table_6.csv",
    "forgetting_analysis": "results/forgetting_analysis.json",
    "robotic_stage_success": "results/plots/robotic_sequence_stage_success.png",
    "forgetting_plot": "results/plots/forgetting_analysis.png",
    "main_comparison": "results/plots/main_comparison.png",
}

# Result trend assertions (semantic metadata for reporting — not hard-coded scores)
RESULT_TREND_ASSERTIONS: List[Dict[str, str]] = [
    {
        "id": "trend_vanilla_fails",
        "claim": "vanilla fine-tuning often fails to leverage pre-trained knowledge",
        "section": "Section 4",
        "environments": "NetHack, Montezuma's Revenge, RoboticSequence",
    },
    {
        "id": "trend_retention_fixes",
        "claim": "knowledge retention methods fix this problem",
        "section": "Section 4",
        "environments": "NetHack, Montezuma's Revenge, RoboticSequence",
    },
    {
        "id": "trend_state_coverage_gap",
        "claim": "state coverage gap can cause deterioration of prior knowledge",
        "section": "Section 2, Section 5",
        "environments": "Montezuma's Revenge, RoboticSequence",
    },
    {
        "id": "trend_bc_ewc_em_maintain",
        "claim": "BC, EM, and EWC maintain or partly regain performance",
        "section": "Section 4, Section 5",
        "environments": "NetHack, Montezuma's Revenge, RoboticSequence",
    },
    {
        "id": "trend_retention_unlocks",
        "claim": "knowledge retention methods unlock the potential of the pre-trained model",
        "section": "Section 4",
        "environments": "all",
    },
    {
        "id": "trend_no_positive_transfer_last_stages",
        "claim": "standard fine-tuning does not exhibit positive transfer of the knowledge of the last two stages",
        "section": "Section 5",
        "environments": "RoboticSequence",
    },
    {
        "id": "trend_pi_star_pretrained",
        "claim": "fine-tuning experiments start from pi_* that performs well on peg-unplug-side and push-wall",
        "section": "Section 3, Figure 7",
        "environments": "RoboticSequence",
    },
    {
        "id": "trend_baseline_outperformance",
        "claim": "baseline_outperformance: proposed method should be compared against explicit baselines",
        "section": "Section 4",
        "environments": "all",
    },
]

# Protocol matrix (Section 3 Experimental setup → artifact paths)
PROTOCOL_MATRIX: List[Dict[str, Any]] = [
    {
        "protocol": "Section 3 Experimental setup",
        "environments": ["nethack", "montezuma", "robotic_sequence"],
        "methods": METHODS_REGISTRY,
        "measurements": ["return", "success_rate", "stage_success_rate", "FAR_performance", "Close_performance"],
        "artifact_paths": [ARTIFACT_PATHS["environment_registry"], ARTIFACT_PATHS["environment_manifest"]],
    },
    {
        "protocol": "Section 4 Main result",
        "environments": ["nethack", "montezuma", "robotic_sequence"],
        "methods": ["training_from_scratch", "vanilla_finetune", "finetune_bc", "finetune_ewc", "finetune_em"],
        "measurements": ["return", "success_rate", "FAR_performance", "forgetting_gap", "final_performance"],
        "artifact_paths": [
            ARTIFACT_PATHS["metrics_json"],
            ARTIFACT_PATHS["figure_3a"],
            ARTIFACT_PATHS["figure_3b"],
            ARTIFACT_PATHS["figure_3c"],
            ARTIFACT_PATHS["table_4"],
            ARTIFACT_PATHS["table_5"],
        ],
    },
    {
        "protocol": "Section 5 Analysis",
        "environments": ["nethack", "montezuma", "robotic_sequence"],
        "methods": ["vanilla_finetune", "finetune_bc", "finetune_ewc", "finetune_em", "training_from_scratch"],
        "measurements": ["FAR_performance", "Close_performance", "maximum_dungeon_level", "turns", "stage_success_rate"],
        "artifact_paths": [
            ARTIFACT_PATHS["figure_4"],
            ARTIFACT_PATHS["figure_5"],
            ARTIFACT_PATHS["figure_6"],
            ARTIFACT_PATHS["figure_7"],
            ARTIFACT_PATHS["figure_8"],
            ARTIFACT_PATHS["state_region_metrics"],
        ],
    },
    {
        "protocol": "Appendix A Toy Examples",
        "environments": ["two_state_mdp", "apple_retrieval"],
        "methods": ["vanilla_finetune", "finetune_bc", "finetune_ewc", "finetune_em", "training_from_scratch"],
        "measurements": ["FAR_performance", "Close_performance", "forgetting_gap"],
        "artifact_paths": [
            ARTIFACT_PATHS["figure_9"],
            ARTIFACT_PATHS["figure_1"],
        ],
    },
    {
        "protocol": "environment setup",
        "environments": ENVIRONMENTS_REGISTRY,
        "methods": [],
        "measurements": [],
        "artifact_paths": [ARTIFACT_PATHS["environment_registry"], ARTIFACT_PATHS["environment_manifest"]],
    },
    {
        "protocol": "state coverage gap diagnostics",
        "environments": ["nethack", "montezuma", "robotic_sequence", "two_state_mdp", "apple_retrieval"],
        "methods": ["vanilla_finetune", "finetune_bc"],
        "measurements": ["FAR_performance", "Close_performance", "forgetting_gap"],
        "artifact_paths": [ARTIFACT_PATHS["state_region_metrics"], ARTIFACT_PATHS["figure_1"], ARTIFACT_PATHS["figure_2"]],
    },
    {
        "protocol": "baseline training",
        "environments": ["nethack", "montezuma", "robotic_sequence"],
        "methods": ["training_from_scratch", "vanilla_finetune"],
        "measurements": ["return", "success_rate"],
        "artifact_paths": [ARTIFACT_PATHS["metrics_json"]],
    },
    {
        "protocol": "fine-tuning",
        "environments": ["nethack", "montezuma", "robotic_sequence"],
        "methods": ["vanilla_finetune", "finetune_bc", "finetune_ewc", "finetune_em"],
        "measurements": ["return", "success_rate", "FAR_performance"],
        "artifact_paths": [ARTIFACT_PATHS["metrics_json"]],
    },
    {
        "protocol": "forgetting mitigation",
        "environments": ["nethack", "montezuma", "robotic_sequence"],
        "methods": ["finetune_bc", "finetune_ewc", "finetune_em", "finetune_ks"],
        "measurements": ["FAR_performance", "Close_performance", "forgetting_gap", "final_performance"],
        "artifact_paths": [ARTIFACT_PATHS["forgetting_analysis"], ARTIFACT_PATHS["state_region_metrics"]],
    },
    {
        "protocol": "main_results",
        "environments": ["nethack", "montezuma", "robotic_sequence"],
        "methods": METHODS_REGISTRY,
        "measurements": ["return", "success_rate", "FAR_performance", "forgetting_gap"],
        "artifact_paths": [ARTIFACT_PATHS["metrics_json"], ARTIFACT_PATHS["summary_csv"]],
    },
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CloseFar:
    """
    Close/FAR state partition for a given environment.
    Paper Section 2: states in CLOSE are easily reachable; FAR requires going through CLOSE.
    Not simplified to a generic metric — exposed as readable fields and computable diagnostics.
    """
    env_name: str
    close_states: List[Any] = field(default_factory=list)
    far_states: List[Any] = field(default_factory=list)
    close_label: str = "close"
    far_label: str = "far"
    partition_method: str = "distance_from_start"  # or "room_id", "dungeon_level", "stage_index"
    description: str = ""

    def classify(self, obs: Any) -> str:
        """Classify an observation as 'close', 'far', or 'unknown'."""
        raise NotImplementedError("Subclass or use environment-specific adapter")


@dataclass
class IsAbleToPickPlace:
    """
    Figure 2: Pre-trained model is able to pick and place objects (e.g., the cylinder).
    This is a FAR-state capability in RoboticSequence.
    """
    stage_name: str = "pick-place"
    is_pretrained_capability: bool = True
    region: str = "far"
    description: str = (
        "Pre-trained model can pick and place objects. "
        "In the new task, this is a FAR state reached only after opening the drawer (CLOSE)."
    )


@dataclass
class InWhichTheAgentNeeds:
    """
    Figure 2: New task in which the agent needs first to open the drawer (Close states)
    and then pick and place the object (FAR states).
    """
    close_task: str = "open-drawer"
    far_task: str = "pick-place"
    description: str = (
        "New downstream task: agent must first open drawer (CLOSE) "
        "then pick-and-place (FAR). State coverage gap causes forgetting of FAR capability."
    )


@dataclass
class ToFindSolutionsAHigh:
    """
    Figure 11: AppleRetrieval — encourages pre-trained model to find solutions
    with a high |b|/|w| ratio. Smaller c => greater forgetting.
    """
    c_value: float = 1.0
    m_distance: int = 30
    description: str = (
        "AppleRetrieval parameter c controls forgetting severity. "
        "Smaller c => greater forgetting of FAR capability."
    )


@dataclass
class UsedInNle:
    """
    NetHack Learning Environment (NLE) — used in Section 3, Figure 3a, Figure 4, Figure 5.
    Dungeon depth and turns are the key diagnostic metrics.
    """
    env_id: str = "NetHackScore-v0"
    character: str = "val-hum-neu-mal"  # Human Monk
    max_episode_steps: int = 10000
    close_max_level: int = NETHACK_CLOSE_MAX_LEVEL
    far_min_level: int = NETHACK_FAR_MIN_LEVEL
    description: str = (
        "NetHack Learning Environment. FAR = dungeon levels >= 4. "
        "FPC driven by imperfect cloning gap (Figure 4)."
    )


@dataclass
class UsedInMontezuma:
    """
    Montezuma's Revenge — used in Section 3, Figure 3b, Figure 6, Figure 12.
    Room 7 = FAR states. State coverage gap drives forgetting.
    """
    env_id: str = "MontezumaRevengeNoFrameskip-v4"
    far_room: int = MONTEZUMA_FAR_ROOM
    close_rooms: List[int] = field(default_factory=lambda: MONTEZUMA_CLOSE_ROOMS)
    room_address: int = 3  # RAM address for current room (from paperbench_ref_001 envs.py)
    description: str = (
        "Montezuma's Revenge. FAR = Room 7. "
        "FPC driven by state coverage gap (Figure 6, Figure 12)."
    )


@dataclass
class TheActivationsOfTheCurrent:
    """
    Figure 20: CKA values throughout vanilla fine-tuning, computed between activations
    of the pre-trained model and the current model. Higher values = more similar representations.
    Used in Section 5 analysis of internal representations.
    """
    metric_name: str = "CKA"
    description: str = (
        "Centered Kernel Alignment (CKA) between pre-trained and current model activations. "
        "Measures representational similarity throughout fine-tuning. "
        "Higher CKA = more similar to pre-trained representations."
    )
    artifact_path: str = ARTIFACT_PATHS["figure_20"]


@dataclass
class EvaluationSpec:
    """
    Specification for an evaluation run.
    Binds environment, method, metrics, and artifact paths.
    """
    env_name: str
    method: str
    seed: int = 0
    n_episodes: int = 200
    smoke_n_episodes: int = 2
    batch_size: int = BATCH_SIZE_128
    output_dir: str = "results"
    mode: str = "runtime_smoke"  # runtime_smoke | full
    close_far_partition: Optional[CloseFar] = None
    robotic_stages: List[str] = field(default_factory=lambda: ROBOTIC_SEQUENCE_STAGES)
    compute_far_performance: bool = True
    compute_close_performance: bool = True
    compute_stage_success: bool = True  # RoboticSequence per-stage success rate
    description: str = ""


@dataclass
class EvaluationResult:
    """
    Result of an evaluation run.
    Contains all paper-visible metrics as explicit readable fields.
    FAR and Close are NOT simplified to generic metrics — they are explicit fields.
    """
    env_name: str
    method: str
    seed: int
    n_episodes: int

    # Core metrics (paper evidence contract)
    episode_return: float = 0.0
    reward: float = 0.0
    loss: float = 0.0
    success_rate: float = 0.0
    accuracy: float = 0.0
    auc: float = 0.0
    fidelity_score: float = 0.0

    # Close/FAR metrics (Section 2, must be explicit fields — not generic)
    close_visitation_rate: float = 0.0   # fraction of steps in CLOSE states
    far_visitation_rate: float = 0.0     # fraction of steps in FAR states
    far_performance: float = 0.0         # success/return on FAR states
    close_performance: float = 0.0       # success/return on CLOSE states
    forgetting_gap: float = 0.0          # pretrained FAR perf - current FAR perf

    # NetHack-specific (Section 3, Figure 4, Figure 5)
    maximum_dungeon_level: float = 0.0
    turns: float = 0.0
    dungeon_depth: float = 0.0
    gold_score: float = 0.0
    eating_score: float = 0.0
    staircase_score: float = 0.0
    scout_score: float = 0.0
    experience_points: float = 0.0

    # RoboticSequence per-stage success rates (Section 3, Figure 7)
    # Must support per-stage computation, not just total return
    stage_success_rates: Dict[str, float] = field(default_factory=dict)
    peg_unplug_side_success: float = 0.0   # pre-trained stage
    push_wall_success: float = 0.0          # pre-trained stage
    pick_place_success: float = 0.0         # downstream stage
    door_open_success: float = 0.0          # downstream stage

    # Montezuma's Revenge (Figure 6, Figure 12)
    room_7_success_rate: float = 0.0        # FAR room success rate
    visited_rooms: List[int] = field(default_factory=list)

    # Forward transfer metric (Appendix F)
    forward_transfer: float = 0.0

    # Metadata
    retained_pretrained_performance: float = 0.0
    downstream_return: float = 0.0
    bc_loss: float = 0.0
    ewc_penalty: float = 0.0
    rl_loss: float = 0.0

    # Provenance
    is_smoke: bool = True
    timestamp: str = ""
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Close/FAR state region classifiers (environment-specific adapters)
# reference_grounding: paperbench_ref_001 envs.py
# ---------------------------------------------------------------------------

class EnvironmentAdapter:
    """
    Base adapter for environment-specific Close/FAR state classification.
    EnvironmentAdapter.state_region(obs) -> 'close' | 'far' | 'unknown'
    """

    def state_region(self, obs: Any) -> str:
        """Return 'close', 'far', or 'unknown' for the given observation."""
        return "unknown"

    def get_close_far_spec(self) -> CloseFar:
        raise NotImplementedError


class NetHackAdapter(EnvironmentAdapter):
    """
    NetHack Close/FAR adapter.
    CLOSE = dungeon levels 1-3 (frequently visited at start of fine-tuning).
    FAR = dungeon levels 4+ (reachable only after mastering CLOSE).
    Paper: imperfect cloning gap drives FPC in NetHack (Figure 4).
    """

    def __init__(self, close_max_level: int = NETHACK_CLOSE_MAX_LEVEL,
                 far_min_level: int = NETHACK_FAR_MIN_LEVEL):
        self.close_max_level = close_max_level
        self.far_min_level = far_min_level

    def state_region(self, obs: Any) -> str:
        """
        Classify NetHack observation by dungeon level.
        obs may be a dict with 'blstats' or an integer dungeon level.
        """
        level = self._extract_dungeon_level(obs)
        if level is None:
            return "unknown"
        if level <= self.close_max_level:
            return "close"
        if level >= self.far_min_level:
            return "far"
        return "unknown"

    def _extract_dungeon_level(self, obs: Any) -> Optional[int]:
        if isinstance(obs, int):
            return obs
        if isinstance(obs, dict):
            blstats = obs.get("blstats")
            if blstats is not None:
                try:
                    import numpy as np  # noqa: F401
                    # blstats[12] = dungeon_level in NLE
                    return int(blstats[12])
                except Exception:
                    pass
            level = obs.get("dungeon_level") or obs.get("dlevel")
            if level is not None:
                return int(level)
        return None

    def get_close_far_spec(self) -> CloseFar:
        return CloseFar(
            env_name="nethack",
            close_states=list(range(1, self.close_max_level + 1)),
            far_states=list(range(self.far_min_level, 30)),
            partition_method="dungeon_level",
            description=f"NetHack: CLOSE=levels 1-{self.close_max_level}, FAR=levels {self.far_min_level}+",
        )


class MontezumaAdapter(EnvironmentAdapter):
    """
    Montezuma's Revenge Close/FAR adapter.
    CLOSE = rooms 0-3 (easily reachable from start).
    FAR = Room 7 (requires mastering CLOSE rooms first).
    Paper: state coverage gap drives FPC (Figure 6, Figure 12).
    reference_grounding: paperbench_ref_001 envs.py (MontezumaInfoWrapper)
    """

    def __init__(self, far_room: int = MONTEZUMA_FAR_ROOM,
                 close_rooms: Optional[List[int]] = None):
        self.far_room = far_room
        self.close_rooms = close_rooms or MONTEZUMA_CLOSE_ROOMS

    def state_region(self, obs: Any) -> str:
        """
        Classify Montezuma observation by room ID.
        obs may be a dict with 'room_id' or an integer room number.
        """
        room = self._extract_room(obs)
        if room is None:
            return "unknown"
        if room == self.far_room:
            return "far"
        if room in self.close_rooms:
            return "close"
        return "unknown"

    def _extract_room(self, obs: Any) -> Optional[int]:
        if isinstance(obs, int):
            return obs
        if isinstance(obs, dict):
            room = obs.get("room_id") or obs.get("current_room")
            if room is not None:
                return int(room)
        return None

    def get_close_far_spec(self) -> CloseFar:
        return CloseFar(
            env_name="montezuma",
            close_states=self.close_rooms,
            far_states=[self.far_room],
            partition_method="room_id",
            description=f"Montezuma: CLOSE=rooms {self.close_rooms}, FAR=Room {self.far_room}",
        )


class RoboticSequenceAdapter(EnvironmentAdapter):
    """
    RoboticSequence Close/FAR adapter.
    CLOSE = stages the agent encounters first (e.g., pick-place, door-open).
    FAR = pre-trained stages (peg-unplug-side, push-wall) — reachable only after CLOSE.
    Paper: state coverage gap drives FPC (Figure 7).
    Must support per-stage success rate computation (not just total return).
    """

    # Pre-trained stages (pi_* performs well on these)
    PRETRAINED_STAGES: List[str] = ["peg-unplug-side", "push-wall"]
    # Downstream stages (new task)