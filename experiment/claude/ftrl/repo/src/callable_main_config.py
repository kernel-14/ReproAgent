"""
src/callable_main_config.py

Callable main configuration interface for the paper reproduction:
  "Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem"

reference_grounding: paperbench_ref_001 agents.py
reference_grounding: paperbench_ref_001 envs.py
reference_grounding: paperbench_ref_001 eval.py
reference_grounding: paperbench_ref_001 train.py

This module provides:
  - CallableMainConfigConfig / CallableMainConfigSpec: dataclasses for config contract
  - build_callable_main_config / make_callable_main_config: factory functions
  - check_callable_main_config_available: availability check
  - load_callable_main_config / prepare_callable_main_config: load/prepare hooks
  - SelectorSetMustIncludeOurs: method/baseline selector registry
  - AdaptersOrRegistryEntries: environment/dataset adapter registry
  - Inventory: paper-derived experiment inventory
  - Factory: model/method factory
  - ObligationsCallablePrimaryFunctio: wired callable obligations

Paper-derived method/baseline selector set (complete):
  ours, ppo, sac, bc, oracle, nle, ewc, pbt, pql,
  scaled_bc_finetune_ks, training_from_scratch, vanilla_finetune,
  finetune_bc, finetune_ewc, finetune_em

Fixed hyperparameters:
  batch_size_128 = 128  (paper evidence contract)

Parameter sweeps (bounded):
  seed_list, smoke_budget, full_training_budget, evaluation_episodes,
  close_far_partition, s_bc_subset, robotic_sequence_stages,
  bc_loss_coeff, ewc_reg_coeff, em_weight, fisher_diagonal_F

Environments: NetHack, Montezuma's Revenge, RoboticSequence, robotics
Datasets: robotics
Metrics: loss, reward, return, success_rate

NOTE: This repository does NOT use the blacklisted repository
      https://github.com/BartekCupial/finetuning-RL-as-CL
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Wire ConfigLayout from ftrl_repro/config.py (active route contract)
# ---------------------------------------------------------------------------
try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ftrl_repro.config import ConfigLayout, ConfigResult, load_config, prepare_config  # noqa: F401
    _CONFIG_LAYOUT_AVAILABLE = True
except ImportError:
    _CONFIG_LAYOUT_AVAILABLE = False
    ConfigLayout = None  # type: ignore[assignment,misc]
    ConfigResult = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Fixed hyperparameters (paper evidence contract: batch_size_128)
# ---------------------------------------------------------------------------

BATCH_SIZE_128: int = 128  # paper fixed hyperparameter anchor

# ---------------------------------------------------------------------------
# Method/baseline selector registry (complete set per paper evidence contract)
# ---------------------------------------------------------------------------

# reference_grounding: paperbench_ref_001 agents.py (batch_size=128, learning_rate=1e-4)
METHOD_REGISTRY: Dict[str, Dict[str, Any]] = {
    # Proposed method
    "ours": {
        "label": "Ours (scaled-bc + fine-tuning + KS)",
        "alias": ["scaled_bc_finetune_ks", "finetune_ks"],
        "category": "proposed",
        "description": "Fine-tuning with knowledge retention (BC + KS combination)",
        "paper_section": "Section 4",
        "trainer_key": "finetune_bc",
    },
    # On-policy baselines
    "ppo": {
        "label": "PPO",
        "alias": ["ppo_baseline"],
        "category": "on_policy_baseline",
        "description": "Proximal Policy Optimization (on-policy training from scratch or fine-tune)",
        "paper_section": "Section 3",
        "trainer_key": "ppo",
    },
    "nle": {
        "label": "NLE baseline",
        "alias": ["nle_baseline"],
        "category": "on_policy_baseline",
        "description": "NetHack Learning Environment state-of-the-art baseline (Tuyls et al., 2023)",
        "paper_section": "Section 4",
        "trainer_key": "nle",
    },
    # Off-policy baselines
    "sac": {
        "label": "SAC",
        "alias": ["sac_baseline"],
        "category": "off_policy_baseline",
        "description": "Soft Actor-Critic (off-policy, used for RoboticSequence)",
        "paper_section": "Section 3 / B.3",
        "trainer_key": "sac",
    },
    # Knowledge retention methods
    "bc": {
        "label": "Fine-tuning + BC",
        "alias": ["finetune_bc", "behavioral_cloning"],
        "category": "knowledge_retention",
        "description": "Behavioral cloning loss on S_BC states from pretrained policy pi_*",
        "paper_section": "Section 2 / Section 4",
        "trainer_key": "finetune_bc",
        "bc_loss_coeff": 1.0,
        "s_bc_buffer_size": 10000,
    },
    "ewc": {
        "label": "Fine-tuning + EWC",
        "alias": ["finetune_ewc", "elastic_weight_consolidation"],
        "category": "knowledge_retention",
        "description": "EWC penalty using diagonal Fisher matrix F and theta_* snapshot",
        "paper_section": "Section 2 / Section 4",
        "trainer_key": "finetune_ewc",
        "ewc_reg_coeff": 1.0,
        "fisher_sample_count": 10000,
    },
    # Oracle / upper bound
    "oracle": {
        "label": "Oracle",
        "alias": ["oracle_baseline"],
        "category": "oracle",
        "description": "Oracle upper bound with access to pretrained task data",
        "paper_section": "Section 4",
        "trainer_key": "oracle",
    },
    # Population / PBT-style
    "pbt": {
        "label": "PBT",
        "alias": ["population_based_training"],
        "category": "population_baseline",
        "description": "Population-based training variant",
        "paper_section": "Appendix",
        "trainer_key": "pbt",
    },
    # Off-policy / PQL-style
    "pql": {
        "label": "PQL",
        "alias": ["prioritized_q_learning"],
        "category": "off_policy_pql",
        "description": "Prioritized Q-learning style off-policy variant",
        "paper_section": "Appendix",
        "trainer_key": "pql",
    },
    # Baselines
    "training_from_scratch": {
        "label": "Training from scratch",
        "alias": ["scratch", "from_scratch"],
        "category": "baseline",
        "description": "Train from random initialization on target task",
        "paper_section": "Section 3 / Section 4",
        "trainer_key": "scratch",
    },
    "vanilla_finetune": {
        "label": "Vanilla fine-tuning",
        "alias": ["vanilla_finetuning", "finetune"],
        "category": "baseline",
        "description": "Fine-tune from pi_* with only RL loss, no knowledge retention",
        "paper_section": "Section 3 / Section 4",
        "trainer_key": "vanilla_finetune",
    },
    # EM (episodic memory)
    "finetune_em": {
        "label": "Fine-tuning + EM",
        "alias": ["em", "episodic_memory"],
        "category": "knowledge_retention",
        "description": "Episodic memory replay: protect 10% of SAC replay buffer as old samples",
        "paper_section": "Section 2 / Section 4",
        "trainer_key": "finetune_em",
        "em_weight": 0.1,
        "protected_fraction": 0.1,
    },
}

# Alias lookup
_METHOD_ALIAS_MAP: Dict[str, str] = {}
for _key, _entry in METHOD_REGISTRY.items():
    _METHOD_ALIAS_MAP[_key] = _key
    for _alias in _entry.get("alias", []):
        _METHOD_ALIAS_MAP[_alias] = _key


def resolve_method_key(method: str) -> str:
    """Resolve a method name or alias to its canonical registry key."""
    return _METHOD_ALIAS_MAP.get(method, method)


# ---------------------------------------------------------------------------
# Environment / dataset registry
# ---------------------------------------------------------------------------

ENVIRONMENT_REGISTRY: Dict[str, Dict[str, Any]] = {
    "nethack": {
        "label": "NetHack Learning Environment",
        "alias": ["nethack_human_monk", "nle", "nethack_learning_environment"],
        "paper_section": "Section 3 / B.1",
        "character": "val-hum-neu-mal",
        "action_space_size": 120,
        "observation_type": "multimodal",
        "far_states": "deeper_dungeon_levels",
        "close_states": "early_dungeon_levels",
        "pretrained_capability": "dungeon_navigation",
        "expensive_dependency": "nle>=0.9",
        "lazy_import": True,
        "figures": ["Figure 3a", "Figure 4", "Figure 5"],
        "metrics": ["return", "maximum_dungeon_level", "turns"],
    },
    "montezuma": {
        "label": "Montezuma's Revenge",
        "alias": ["montezuma_revenge", "MontezumaRevengeNoFrameskip-v4"],
        "paper_section": "Section 3 / B.2",
        "observation_type": "pixel",
        "far_states": "room_7_and_beyond",
        "close_states": "room_1_to_6",
        "pretrained_capability": "room_navigation",
        "expensive_dependency": "gymnasium[atari]>=0.29",
        "lazy_import": True,
        "figures": ["Figure 3b", "Figure 6"],
        "metrics": ["return", "success_rate"],
    },
    "robotic_sequence": {
        "label": "RoboticSequence",
        "alias": ["robotics", "meta_world", "robotic_manipulation"],
        "paper_section": "Section 3 / B.3",
        "algorithm": "SAC",
        "observation_type": "state_plus_stage_id",
        "policy_architecture": "MLP_4x256",
        "pretrained_stages": ["peg-unplug-side", "push-wall"],
        "pretrained_success_rate": 1.0,
        "stage_identifiers": [
            "reach", "push", "pick-place", "door-open",
            "drawer-open", "button-press", "peg-unplug-side", "push-wall"
        ],
        "num_seeds": 20,
        "confidence_interval": 0.90,
        "expensive_dependency": "gymnasium>=0.29",
        "lazy_import": True,
        "figures": ["Figure 3c", "Figure 7", "Figure 8"],
        "metrics": ["success_rate", "stage_success_rate"],
        "dataset_alias": "robotics",
    },
    "two_state_mdp": {
        "label": "Two-state MDPs",
        "alias": ["toy_mdp", "two_state"],
        "paper_section": "Appendix A",
        "observation_type": "discrete",
        "far_states": "state_1",
        "close_states": "state_0",
        "expensive_dependency": None,
        "lazy_import": False,
        "figures": ["Figure 9"],
        "metrics": ["return"],
    },
    "apple_retrieval": {
        "label": "AppleRetrieval",
        "alias": ["apple_retrieval_env"],
        "paper_section": "Appendix A",
        "observation_type": "grid",
        "expensive_dependency": None,
        "lazy_import": False,
        "figures": ["Figure 10", "Figure 12"],
        "metrics": ["return", "success_rate"],
    },
}

DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "robotics": {
        "label": "RoboticSequence / Meta-World dataset",
        "alias": ["meta_world", "robotic_sequence_data"],
        "environment": "robotic_sequence",
        "paper_section": "Section 3 / B.3",
        "lazy_download": True,
        "smoke_fixture": True,
        "metrics": ["success_rate", "stage_success_rate"],
        "artifact_paths": [
            "results/dataset_registry.json",
            "results/main_results.csv",
        ],
    },
}

# ---------------------------------------------------------------------------
# Bounded parameter sweeps (paper evidence contract)
# ---------------------------------------------------------------------------

SWEEP_DEFAULTS: Dict[str, Any] = {
    # Seeds
    "seed_list": [0, 1, 2],
    "seed_list_full": [0, 1, 2, 3, 4],
    # Budgets
    "smoke_budget_steps": 100,
    "smoke_budget_episodes": 2,
    "full_training_budget_steps": 1_000_000,
    "full_training_budget_episodes": 10000,
    # Evaluation
    "evaluation_episodes": 200,
    "evaluation_episodes_smoke": 2,
    # Close/FAR partition
    "close_far_partition": {
        "nethack": {"close_threshold_level": 4, "far_threshold_level": 5},
        "montezuma": {"close_rooms": list(range(1, 7)), "far_rooms": [7, 8, 9]},
        "robotic_sequence": {
            "close_stages": ["reach", "push", "pick-place", "door-open", "drawer-open", "button-press"],
            "far_stages": ["peg-unplug-side", "push-wall"],
        },
    },
    # S_BC state subset
    "s_bc_subset_size": 10000,
    "s_bc_subset_size_smoke": 100,
    # RoboticSequence stage identifiers
    "robotic_sequence_stage_identifiers": [
        "reach", "push", "pick-place", "door-open",
        "drawer-open", "button-press", "peg-unplug-side", "push-wall"
    ],
    # BC loss coefficient
    "bc_loss_coeff": 1.0,
    "bc_loss_coeff_sweep": [0.1, 0.5, 1.0, 2.0],
    # EWC regularization coefficient
    "ewc_reg_coeff": 1.0,
    "ewc_reg_coeff_sweep": [0.1, 1.0, 10.0],
    # EM weight (protected fraction of replay buffer)
    "em_weight": 0.1,
    "em_protected_fraction": 0.1,
    # Diagonal Fisher matrix F
    "fisher_diagonal_F": {
        "sample_count": 10000,
        "sample_count_smoke": 10,
        "damping": 1e-4,
        "source": "nld_aa_dataset_or_rollout",
        "addendum_note": "To compute the Fisher matrix 10000 batches should be sampled from the NLD-AA dataset.",
    },
    # Fixed hyperparameter anchor
    "batch_size_128": BATCH_SIZE_128,
    # Shot count sweep
    "shot_count": [1, 5, 10, 20],
    "shot_count_default": 1,
}

# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------

METRIC_REGISTRY: Dict[str, Dict[str, Any]] = {
    "loss": {
        "label": "Loss",
        "formula": "cross_entropy_or_mse",
        "aggregation": "mean",
        "environments": ["all"],
        "paper_section": "Section 2",
    },
    "reward": {
        "label": "Reward",
        "formula": "environment_reward_signal",
        "aggregation": "mean_per_episode",
        "environments": ["all"],
        "paper_section": "Section 3",
    },
    "return": {
        "label": "Episode Return",
        "formula": "sum_discounted_rewards",
        "aggregation": "mean_over_seeds_and_episodes",
        "environments": ["all"],
        "paper_section": "Section 4",
        "figures": ["Figure 3a", "Figure 3b", "Figure 5"],
    },
    "success_rate": {
        "label": "Success Rate",
        "formula": "fraction_of_successful_episodes",
        "aggregation": "mean_over_seeds",
        "environments": ["montezuma", "robotic_sequence"],
        "paper_section": "Section 4",
        "figures": ["Figure 3b", "Figure 3c", "Figure 6", "Figure 7"],
    },
    "stage_success_rate": {
        "label": "Stage Success Rate",
        "formula": "per_stage_fraction_of_successful_episodes",
        "aggregation": "mean_over_seeds",
        "environments": ["robotic_sequence"],
        "paper_section": "Section 5 / F",
        "figures": ["Figure 7"],
    },
    "maximum_dungeon_level": {
        "label": "Maximum Dungeon Level",
        "formula": "max_level_reached_in_episode",
        "aggregation": "density_plot",
        "environments": ["nethack"],
        "paper_section": "Section 5",
        "figures": ["Figure 4"],
    },
    "turns": {
        "label": "Total Turns",
        "formula": "total_in_game_turns",
        "aggregation": "density_plot",
        "environments": ["nethack"],
        "paper_section": "Section 5",
        "figures": ["Figure 4"],
    },
    "FAR_performance": {
        "label": "FAR State Performance",
        "formula": "mean_return_or_success_on_far_states",
        "aggregation": "mean_over_seeds",
        "environments": ["all"],
        "paper_section": "Section 2 / Section 5",
    },
    "Close_performance": {
        "label": "Close State Performance",
        "formula": "mean_return_or_success_on_close_states",
        "aggregation": "mean_over_seeds",
        "environments": ["all"],
        "paper_section": "Section 2 / Section 5",
    },
    "forgetting_gap": {
        "label": "Forgetting Gap",
        "formula": "pretrained_performance_minus_finetuned_performance_on_far",
        "aggregation": "mean_over_seeds",
        "environments": ["all"],
        "paper_section": "Section 5",
        "trend": "vanilla_finetune_shows_large_gap_retention_methods_reduce_it",
    },
    "final_performance": {
        "label": "Final Performance",
        "formula": "mean_return_or_success_at_final_checkpoint",
        "aggregation": "mean_over_seeds",
        "environments": ["all"],
        "paper_section": "Section 4",
        "figures": ["Figure 3a", "Figure 3b", "Figure 3c"],
    },
    "bc_loss": {
        "label": "BC Loss",
        "formula": "kl_divergence_or_cross_entropy_on_s_bc",
        "aggregation": "mean_per_update",
        "environments": ["all"],
        "paper_section": "Section 2",
    },
    "rl_loss": {
        "label": "RL Loss",
        "formula": "policy_gradient_or_actor_critic_loss",
        "aggregation": "mean_per_update",
        "environments": ["all"],
        "paper_section": "Section 2",
    },
}

# ---------------------------------------------------------------------------
# Artifact registry (Table 1, Figure 9, and all paper-visible artifacts)
# ---------------------------------------------------------------------------

ARTIFACT_REGISTRY: Dict[str, Dict[str, Any]] = {
    "table_1": {
        "label": "Table 1",
        "path": "results/tables/table_1.csv",
        "environment": ["nethack", "montezuma", "robotic_sequence"],
        "methods": list(METHOD_REGISTRY.keys()),
        "metrics": ["return", "success_rate", "final_performance"],
        "config_source": "callable_main_config",
        "paper_section": "Section 4",
        "trend": "baseline_outperformance",
    },
    "figure_9": {
        "label": "Figure 9",
        "path": "results/figures/figure_9.png",
        "environment": ["two_state_mdp"],
        "methods": ["ours", "vanilla_finetune", "training_from_scratch"],
        "metrics": ["return"],
        "config_source": "callable_main_config",
        "paper_section": "Appendix A",
        "description": "Toy two-state MDP policy and value function visualization",
    },
    "figure_1": {"label": "Figure 1", "path": "results/figures/figure_1.png",
                 "paper_section": "Section 2", "config_source": "callable_main_config"},
    "figure_2": {"label": "Figure 2", "path": "results/figures/figure_2.png",
                 "paper_section": "Section 2", "config_source": "callable_main_config"},
    "figure_3": {"label": "Figure 3", "path": "results/figures/figure_3.png",
                 "paper_section": "Section 4", "config_source": "callable_main_config"},
    "figure_3a": {"label": "Figure 3a", "path": "results/figures/figure_3a.png",
                  "environment": "nethack", "paper_section": "Section 4",
                  "config_source": "callable_main_config"},
    "figure_3b": {"label": "Figure 3b", "path": "results/figures/figure_3b.png",
                  "environment": "montezuma", "paper_section": "Section 4",
                  "config_source": "callable_main_config"},
    "figure_3c": {"label": "Figure 3c", "path": "results/figures/figure_3c.png",
                  "environment": "robotic_sequence", "paper_section": "Section 4",
                  "config_source": "callable_main_config"},
    "figure_4": {"label": "Figure 4", "path": "results/figures/figure_4.png",
                 "environment": "nethack", "paper_section": "Section 5",
                 "note": "Addendum: Figure 4 is not required to be reproduced",
                 "config_source": "callable_main_config"},
    "figure_5": {"label": "Figure 5", "path": "results/figures/figure_5.png",
                 "environment": "nethack", "paper_section": "Section 5",
                 "config_source": "callable_main_config"},
    "figure_6": {"label": "Figure 6", "path": "results/figures/figure_6.png",
                 "environment": "montezuma", "paper_section": "Section 5",
                 "config_source": "callable_main_config"},
    "figure_7": {"label": "Figure 7", "path": "results/figures/figure_7.png",
                 "environment": "robotic_sequence", "paper_section": "Section 5",
                 "config_source": "callable_main_config"},
    "figure_8": {"label": "Figure 8", "path": "results/figures/figure_8.png",
                 "environment": "robotic_sequence", "paper_section": "Section 5",
                 "config_source": "callable_main_config"},
    "figure_12": {"label": "Figure 12", "path": "results/figures/figure_12.png",
                  "paper_section": "Appendix", "config_source": "callable_main_config"},
    "figure_14": {"label": "Figure 14", "path": "results/figures/figure_14.png",
                  "paper_section": "Appendix", "config_source": "callable_main_config"},
    "figure_22": {"label": "Figure 22", "path": "results/figures/figure_22.png",
                  "paper_section": "Appendix F", "config_source": "callable_main_config"},
    "figure_23": {"label": "Figure 23", "path": "results/figures/figure_23.png",
                  "paper_section": "Appendix F", "config_source": "callable_main_config"},
    "figure_25": {"label": "Figure 25", "path": "results/figures/figure_25.png",
                  "paper_section": "Appendix F", "config_source": "callable_main_config"},
    "figure_26": {"label": "Figure 26", "path": "results/figures/figure_26.png",
                  "paper_section": "Appendix F", "config_source": "callable_main_config"},
    "table_4": {"label": "Table 4", "path": "results/tables/table_4.csv",
                "paper_section": "Appendix", "config_source": "callable_main_config"},
    "table_5": {"label": "Table 5", "path": "results/tables/table_5.csv",
                "paper_section": "Appendix", "config_source": "callable_main_config"},
    "table_6": {"label": "Table 6", "path": "results/tables/table_6.csv",
                "paper_section": "Appendix F", "config_source": "callable_main_config"},
    "metrics_json": {"label": "metrics.json", "path": "results/metrics.json",
                     "config_source": "callable_main_config"},
    "main_results_csv": {"label": "main_results.csv", "path": "results/main_results.csv",
                         "config_source": "callable_main_config"},
    "bc_buffer_manifest": {"label": "bc_buffer_manifest.json",
                           "path": "artifacts/bc_buffer_manifest.json",
                           "config_source": "callable_main_config"},
}

# ---------------------------------------------------------------------------
# Trend obligations (semantic metadata, not hardcoded benchmark scores)
# ---------------------------------------------------------------------------

TREND_OBLIGATIONS: Dict[str, str] = {
    "baseline_outperformance": (
        "Knowledge retention methods (BC, EWC, EM) outperform vanilla fine-tuning "
        "and training from scratch across NetHack, Montezuma's Revenge, and RoboticSequence. "
        "Vanilla fine-tuning often fails to leverage pre-trained knowledge. "
        "NOTE: dry-run does not claim paper-level numeric results."
    ),
    "vanilla_finetune_fails": (
        "Vanilla fine-tuning often fails to leverage pre-trained knowledge; "
        "the agent's performance deteriorates, losing pre-trained capabilities."
    ),
    "retention_methods_unlock_potential": (
        "Knowledge retention methods unlock the potential of the pre-trained model "
        "and lead to significant improvements."
    ),
    "bc_ewc_em_maintain_performance": (
        "BC, EM, and EWC maintain or to a certain degree regain performance on "
        "pre-trained stages (peg-unplug-side, push-wall) during fine-tuning."
    ),
    "state_coverage_gap": (
        "State coverage gap causes deterioration of prior knowledge: "
        "pre-trained states (FAR) are not visited during early fine-tuning."
    ),
}

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CallableMainConfigConfig:
    """
    Configuration for the callable main config interface.
    Shared between CLI and programmatic invocation.
    """
    # Execution mode
    mode: str = "runtime_smoke"  # runtime_smoke | dry_run | train | eval | report | full

    # Experiment selection
    experiment: str = "section_4_main_result"
    environment: str = "robotic_sequence"
    method: str = "bc"
    seed: int = 0
    seed_list: List[int] = field(default_factory=lambda: [0, 1, 2])

    # Budgets
    smoke_steps: int = SWEEP_DEFAULTS["smoke_budget_steps"]
    full_training_steps: int = SWEEP_DEFAULTS["full_training_budget_steps"]
    evaluation_episodes: int = SWEEP_DEFAULTS["evaluation_episodes"]

    # Fixed hyperparameters (paper evidence contract)
    batch_size: int = BATCH_SIZE_128  # batch_size_128

    # BC parameters
    bc_loss_coeff: float = SWEEP_DEFAULTS["bc_loss_coeff"]
    s_bc_subset_size: int = SWEEP_DEFAULTS["s_bc_subset_size"]

    # EWC parameters
    ewc_reg_coeff: float = SWEEP_DEFAULTS["ewc_reg_coeff"]
    fisher_sample_count: int = SWEEP_DEFAULTS["fisher_diagonal_F"]["sample_count"]

    # EM parameters
    em_weight: float = SWEEP_DEFAULTS["em_weight"]

    # Close/FAR partition
    close_far_partition: Dict[str, Any] = field(
        default_factory=lambda: SWEEP_DEFAULTS["close_far_partition"]
    )

    # RoboticSequence stage identifiers
    robotic_sequence_stages: List[str] = field(
        default_factory=lambda: SWEEP_DEFAULTS["robotic_sequence_stage_identifiers"]
    )

    # S_BC state subset
    s_bc_state_subset: Optional[str] = None  # path or "sample_from_rollout"

    # Output
    output_dir: str = "results"
    pretrained_checkpoint: Optional[str] = None

    # Sweep
    shot_count: int = SWEEP_DEFAULTS["shot_count_default"]

    # Dry-run flag
    dry_run: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CallableMainConfigSpec:
    """
    Specification for the callable main config, binding environments, methods,
    metrics, and artifact paths to the paper's evidence obligations.
    """
    config: CallableMainConfigConfig = field(default_factory=CallableMainConfigConfig)

    # Paper-derived registries (read-only references)
    method_registry: Dict[str, Any] = field(default_factory=lambda: METHOD_REGISTRY)
    environment_registry: Dict[str, Any] = field(default_factory=lambda: ENVIRONMENT_REGISTRY)
    dataset_registry: Dict[str, Any] = field(default_factory=lambda: DATASET_REGISTRY)
    metric_registry: Dict[str, Any] = field(default_factory=lambda: METRIC_REGISTRY)
    artifact_registry: Dict[str, Any] = field(default_factory=lambda: ARTIFACT_REGISTRY)
    sweep_defaults: Dict[str, Any] = field(default_factory=lambda: SWEEP_DEFAULTS)
    trend_obligations: Dict[str, str] = field(default_factory=lambda: TREND_OBLIGATIONS)

    # Availability status
    available: bool = True