"""
src/experiment_section_analysis.py

Section 4 & 5 experiment analysis for the paper reproduction:
  "Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem"

reference_grounding: paperbench_ref_001 agents.py
reference_grounding: paperbench_ref_001 envs.py
reference_grounding: paperbench_ref_001 model.py
reference_grounding: paperbench_ref_001 utils.py

Paper Section 4 Main result:
  - Compares all named methods on NetHack, Montezuma's Revenge, RoboticSequence
  - vanilla fine-tuning often fails to leverage pre-trained knowledge
  - knowledge retention methods (BC, EWC, EM) mitigate forgetting and unlock pre-trained model
  - Fine-tuning + KS surpasses prior SOTA on NetHack by 2x (10K vs 5K points)

Paper Section 5 Analysis:
  - FAR/CLOSE state partition diagnostics
  - NetHack density plots (max dungeon level vs turns)
  - RoboticSequence per-stage success rates
  - State coverage gap causes deterioration of prior knowledge

Result trend assertions (semantic metadata, NOT hard-coded scores):
  - vanilla_finetune_fails: vanilla fine-tuning often fails to leverage pre-trained knowledge
  - retention_methods_fix: knowledge retention methods fix this problem
  - bc_em_ewc_maintain: BC, EM, and EWC maintain or partly regain performance
  - retention_unlocks_potential: knowledge retention methods unlock the potential of the pre-trained model
  - state_coverage_gap_causes_deterioration: state coverage gap can cause deterioration of prior knowledge
  - standard_finetune_no_positive_transfer: standard fine-tuning does not exhibit positive transfer for last stages
  - baseline_outperformance: proposed method should be compared against explicit baselines

Artifact paths (Section 4 main results):
  results/metrics.json
  results/summary.csv
  results/main_results.csv
  results/main_results_table.json
  results/main_results_summary.md
  results/experiment_registry.json
  results/bc_metrics.json
  results/dataset_registry.json
  results/tables/table_1.csv
  results/plots/robotic_sequence_stage_success.png
  results/plots/forgetting_analysis.png
  results/plots/main_comparison.png
  results/run_manifest.json
  results/config_resolved.json
  results/reproduction_inventory.json
  results/environment_manifest.json
  results/loss_trace.json
  artifacts/bc_buffer_manifest.json

Fixed hyperparameters (paper evidence contract):
  batch_size_128 = 128

NOTE: This repository does NOT use the blacklisted repository
      https://github.com/BartekCupial/finetuning-RL-as-CL

Lazy imports: torch, gym/gymnasium, nle, metaworld are imported inside functions only.
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Fixed hyperparameters (paper evidence contract: batch_size_128)
# ---------------------------------------------------------------------------

BATCH_SIZE_128: int = 128  # paper fixed hyperparameter anchor

# ---------------------------------------------------------------------------
# Bounded config constants (paper-derived, not exhaustive sweeps)
# ---------------------------------------------------------------------------

SEED_LIST: List[int] = [0, 1, 2]
SMOKE_BUDGET_STEPS: int = 10
FULL_TRAINING_BUDGET_STEPS: int = 1_000_000
EVALUATION_EPISODES: int = 200  # paper: 200 episodes for NetHack evaluation

# Close/FAR state partition (paper Section 2)
CLOSE_FAR_PARTITION = {
    "nethack": {
        "close": "dungeon_level_1",
        "far": "dungeon_level_gt_1",
        "description": "CLOSE=level 1 (frequently visited), FAR=deeper levels (infrequently visited)",
    },
    "montezuma": {
        "close": "rooms_0_to_6",
        "far": "room_7_and_beyond",
        "description": "CLOSE=early rooms, FAR=Room 7+ (Figure 6, Figure 12)",
    },
    "robotic_sequence": {
        "close": "open_drawer",
        "far": "pick_and_place",
        "description": "CLOSE=open drawer (new task), FAR=pick-and-place (pre-trained capability)",
    },
}

# S_BC state subset (paper Section 2, BC method)
S_BC_BUFFER_SIZE: int = 10_000  # states visited by pre-trained policy pi_*
S_BC_SAMPLE_BATCH: int = BATCH_SIZE_128

# RoboticSequence stage identifiers (paper Section 3, Figure 7)
ROBOTIC_SEQUENCE_STAGES: List[str] = [
    "peg-unplug-side",   # pre-trained capability stage
    "push-wall",         # pre-trained capability stage
    "new_task_stage",    # downstream fine-tuning target
]

# Diagonal Fisher matrix F (EWC, paper Section 2)
EWC_FISHER_SAMPLE_COUNT: int = 10_000  # addendum: 10000 batches from NLD-AA
EWC_REGULARIZATION_COEFFICIENT: float = 1.0  # lambda in EWC penalty

# BC loss coefficient (paper Section 2)
BC_LOSS_COEFFICIENT: float = 1.0  # alpha weighting BC loss vs RL loss

# EM weight (paper Section 2, episodic memory)
EM_WEIGHT: float = 0.1  # fraction of replay buffer protected as old samples

# ---------------------------------------------------------------------------
# Result trend assertions (semantic metadata for reporting)
# ---------------------------------------------------------------------------

RESULT_TREND_ASSERTIONS: Dict[str, str] = {
    "vanilla_finetune_fails": (
        "vanilla fine-tuning often fails to leverage pre-trained knowledge"
    ),
    "retention_methods_fix": (
        "knowledge retention methods fix this problem"
    ),
    "bc_em_ewc_maintain": (
        "BC, EM, and EWC maintain or partly regain performance"
    ),
    "retention_unlocks_potential": (
        "knowledge retention methods unlock the potential of the pre-trained model"
    ),
    "retention_leads_to_improvements": (
        "knowledge retention methods lead to significant improvements"
    ),
    "state_coverage_gap_causes_deterioration": (
        "state coverage gap can cause deterioration of prior knowledge"
    ),
    "standard_finetune_no_positive_transfer": (
        "standard fine-tuning does not exhibit positive transfer of the knowledge of the last two stages"
    ),
    "baseline_outperformance": (
        "proposed method should be compared against explicit baselines; "
        "knowledge retention methods outperform vanilla fine-tuning"
    ),
    "positive_parameter_improves": (
        "increasing BC loss coefficient or EWC regularization improves retention of pre-trained capabilities"
    ),
    "finetune_starts_from_pi_star": (
        "fine-tuning experiments start from pi_* that performs well on peg-unplug-side and push-wall"
    ),
}

# ---------------------------------------------------------------------------
# Protocol matrix (paper-derived experiment matrix)
# ---------------------------------------------------------------------------

PROTOCOL_MATRIX: List[Dict[str, Any]] = [
    {
        "section": "Section 3 Experimental setup",
        "experiment": "environment_setup",
        "environments": ["nethack", "montezuma", "robotic_sequence"],
        "methods": ["training_from_scratch", "vanilla_finetune"],
        "measurements": ["return", "success_rate"],
        "artifact_paths": ["results/metrics.json", "results/environment_manifest.json"],
    },
    {
        "section": "Section 4 Main result",
        "experiment": "main_results",
        "environments": ["nethack", "montezuma", "robotic_sequence"],
        "methods": [
            "training_from_scratch",
            "vanilla_finetune",
            "finetune_bc",
            "finetune_ewc",
            "finetune_em",
        ],
        "measurements": [
            "return",
            "success_rate",
            "learning_curve",
            "final_aggregate_score",
            "retained_pretrained_performance",
            "improvement_over_vanilla",
        ],
        "artifact_paths": [
            "results/metrics.json",
            "results/main_results.csv",
            "results/main_results_table.json",
            "results/tables/table_1.csv",
            "results/plots/main_comparison.png",
        ],
        "figure_artifacts": ["Figure 3", "Figure 3a", "Figure 3b", "Figure 3c"],
        "table_artifacts": ["Table 4", "Table 5"],
        "trend_obligations": ["baseline_outperformance", "vanilla_finetune_fails", "retention_methods_fix"],
    },
    {
        "section": "Section 5 Analysis",
        "experiment": "forgetting_analysis",
        "environments": ["nethack", "montezuma", "robotic_sequence"],
        "methods": [
            "vanilla_finetune",
            "finetune_bc",
            "finetune_ewc",
            "finetune_em",
            "training_from_scratch",
        ],
        "measurements": [
            "FAR_performance",
            "Close_performance",
            "close_far_visitation",
            "maximum_dungeon_level",
            "turns",
            "stage_success_rate",
            "forgetting_gap",
        ],
        "artifact_paths": [
            "results/plots/forgetting_analysis.png",
            "results/plots/robotic_sequence_stage_success.png",
        ],
        "figure_artifacts": ["Figure 4", "Figure 5", "Figure 6", "Figure 7", "Figure 8", "Figure 12"],
        "trend_obligations": [
            "state_coverage_gap_causes_deterioration",
            "bc_em_ewc_maintain",
            "standard_finetune_no_positive_transfer",
        ],
    },
    {
        "section": "Appendix A Toy Examples",
        "experiment": "toy_diagnostics",
        "environments": ["two_state_mdp", "apple_retrieval"],
        "methods": [
            "training_from_scratch",
            "vanilla_finetune",
            "finetune_bc",
            "finetune_ewc",
            "finetune_em",
        ],
        "measurements": ["return", "FAR_performance", "forgetting_gap"],
        "artifact_paths": ["results/toy_diagnostics.json"],
        "figure_artifacts": ["Figure 9", "Figure 10", "Figure 11"],
        "trend_obligations": ["state_coverage_gap_causes_deterioration"],
    },
    {
        "section": "Global artifact contract",
        "experiment": "global_artifacts",
        "artifact_manifest_rows": [
            {"artifact": "Figure 9", "path": "results/figures/figure_9.png", "method": "toy_mdp", "env": "two_state_mdp"},
            {"artifact": "Figure 22", "path": "results/figures/figure_22.png", "method": "finetune_bc", "env": "robotic_sequence"},
            {"artifact": "Figure 23", "path": "results/figures/figure_23.png", "method": "finetune_bc", "env": "robotic_sequence"},
            {"artifact": "Figure 25", "path": "results/figures/figure_25.png", "method": "finetune_bc", "env": "robotic_sequence"},
            {"artifact": "Figure 26", "path": "results/figures/figure_26.png", "method": "finetune_bc", "env": "robotic_sequence"},
            {"artifact": "Table 1", "path": "results/tables/table_1.csv", "method": "all", "env": "nethack"},
        ],
    },
]

# ---------------------------------------------------------------------------
# Dataset registry (paper evidence contract: robotics)
# ---------------------------------------------------------------------------

DATASET_REGISTRY: Dict[str, Any] = {
    "robotics": {
        "id": "robotics",
        "alias": "RoboticSequence",
        "environment": "robotic_sequence",
        "backend": "metaworld",
        "lazy_load": True,
        "smoke_fixture": True,
        "description": "Meta-World robotic manipulation tasks: peg-unplug-side, push-wall, etc.",
        "paper_section": "Section 3",
        "stages": ROBOTIC_SEQUENCE_STAGES,
        "metrics": ["success_rate", "stage_success_rate", "return"],
        "artifact_paths": [
            "results/plots/robotic_sequence_stage_success.png",
            "results/tables/table_1.csv",
        ],
    },
    "nethack_nld_aa": {
        "id": "nethack_nld_aa",
        "alias": "NLD-AA",
        "environment": "nethack",
        "backend": "nle",
        "lazy_load": True,
        "smoke_fixture": True,
        "description": "NetHack Learning Dataset - AutoAscend trajectories",
        "download_url": "https://github.com/dungeonsdatasubmission/dungeonsdata-neurips2022",
        "paper_section": "Section 3, Appendix B.1",
        "fisher_sample_count": EWC_FISHER_SAMPLE_COUNT,
        "metrics": ["return", "maximum_dungeon_level", "turns", "FAR_performance"],
    },
    "montezuma": {
        "id": "montezuma",
        "alias": "MontezumaRevenge",
        "environment": "montezuma",
        "backend": "gymnasium_atari",
        "lazy_load": True,
        "smoke_fixture": True,
        "description": "Montezuma's Revenge Atari game",
        "paper_section": "Section 3",
        "far_states": "room_7_and_beyond",
        "metrics": ["return", "success_rate", "FAR_performance"],
    },
}

# ---------------------------------------------------------------------------
# Metric registry (paper evidence contract)
# ---------------------------------------------------------------------------

METRIC_REGISTRY: Dict[str, Any] = {
    "loss": {
        "id": "loss",
        "description": "Training loss (BC loss + RL loss)",
        "formula": "L_total = L_RL + alpha * L_BC",
        "components": ["rl_loss", "bc_loss"],
    },
    "reward": {
        "id": "reward",
        "description": "Per-step reward from environment",
        "aggregation": "mean over episodes",
    },
    "return": {
        "id": "return",
        "description": "Discounted cumulative reward per episode",
        "formula": "G_t = sum_{k=0}^{T} gamma^k * r_{t+k}",
        "aggregation": "mean over evaluation episodes",
    },
    "success_rate": {
        "id": "success_rate",
        "description": "Fraction of episodes where task is completed successfully",
        "formula": "success_rate = num_success / num_episodes",
        "aggregation": "mean over seeds",
    },
    "stage_success_rate": {
        "id": "stage_success_rate",
        "description": "Per-stage success rate for RoboticSequence",
        "stages": ROBOTIC_SEQUENCE_STAGES,
        "formula": "stage_success_rate[s] = num_success_stage_s / num_episodes",
    },
    "FAR_performance": {
        "id": "FAR_performance",
        "description": "Policy performance on FAR states (pre-trained capability retention)",
        "formula": "FAR_perf = mean(reward | state in FAR)",
        "trend": "vanilla fine-tuning causes FAR performance to drop",
    },
    "Close_performance": {
        "id": "Close_performance",
        "description": "Policy performance on CLOSE states",
        "formula": "Close_perf = mean(reward | state in CLOSE)",
    },
    "close_far_visitation": {
        "id": "close_far_visitation",
        "description": "Fraction of time steps spent in CLOSE vs FAR states",
        "formula": "visitation_ratio = count(FAR) / count(CLOSE + FAR)",
    },
    "maximum_dungeon_level": {
        "id": "maximum_dungeon_level",
        "description": "Maximum dungeon level achieved in NetHack episode",
        "aggregation": "histogram / density plot vs turns",
    },
    "turns": {
        "id": "turns",
        "description": "Total number of in-game turns (NetHack)",
        "aggregation": "per episode",
    },
    "forgetting_gap": {
        "id": "forgetting_gap",
        "description": "Performance drop on pre-trained tasks after fine-tuning",
        "formula": "forgetting_gap = perf_pretrained - perf_finetuned_on_pretrained_tasks",
    },
    "bc_loss": {
        "id": "bc_loss",
        "description": "Behavioral cloning loss on S_BC buffer",
        "formula": "L_BC = -E_{s in S_BC}[log pi_theta(a* | s)]  or KL(pi_* || pi_theta)",
    },
    "forward_transfer": {
        "id": "forward_transfer",
        "description": "Forward transfer metric (Appendix F)",
        "formula": "FT = (AUC - AUC_b) / (1 - AUC_b)",
        "components": ["AUC", "AUC_b"],
    },
    "fidelity_score": {
        "id": "fidelity_score",
        "description": "Log-likelihood of pi_* trajectories under fine-tuned policy",
        "formula": "fidelity = E_{(s,a*) ~ pi_*}[log pi_theta(a* | s)]",
    },
    "accuracy": {
        "id": "accuracy",
        "description": "Action prediction accuracy (BC evaluation)",
        "formula": "accuracy = mean(argmax(pi_theta(s)) == a*)",
    },
}

# ---------------------------------------------------------------------------
# Experiment registry (paper-derived, Section 4 + Section 5)
# ---------------------------------------------------------------------------

EXPERIMENT_REGISTRY: Dict[str, Any] = {
    "section_4_main_result": {
        "id": "section_4_main_result",
        "section": "Section 4",
        "title": "Main result: knowledge retention mitigates forgetting of pre-trained capabilities",
        "environments": ["nethack", "montezuma", "robotic_sequence"],
        "methods": [
            "training_from_scratch",
            "vanilla_finetune",
            "finetune_bc",
            "finetune_ewc",
            "finetune_em",
        ],
        "baselines": ["training_from_scratch", "vanilla_finetune"],
        "proposed_methods": ["finetune_bc", "finetune_ewc", "finetune_em"],
        "metrics": ["return", "success_rate", "learning_curve", "retained_pretrained_performance"],
        "seeds": SEED_LIST,
        "evaluation_episodes": EVALUATION_EPISODES,
        "figure_artifacts": ["Figure 3", "Figure 3a", "Figure 3b", "Figure 3c"],
        "table_artifacts": ["Table 4", "Table 5"],
        "artifact_paths": [
            "results/main_results.csv",
            "results/main_results_table.json",
            "results/tables/table_1.csv",
            "results/plots/main_comparison.png",
        ],
        "trend_obligations": RESULT_TREND_ASSERTIONS,
        "config_source": "configs/setup.yaml",
        "hyperparameters": {
            "batch_size": BATCH_SIZE_128,
            "bc_loss_coefficient": BC_LOSS_COEFFICIENT,
            "ewc_regularization_coefficient": EWC_REGULARIZATION_COEFFICIENT,
            "em_weight": EM_WEIGHT,
            "s_bc_buffer_size": S_BC_BUFFER_SIZE,
        },
    },
    "section_5_analysis": {
        "id": "section_5_analysis",
        "section": "Section 5",
        "title": "Analysis: forgetting of pre-trained capabilities hinders RL fine-tuning",
        "environments": ["nethack", "montezuma", "robotic_sequence"],
        "methods": [
            "vanilla_finetune",
            "finetune_bc",
            "finetune_ewc",
            "finetune_em",
            "training_from_scratch",
        ],
        "measurements": [
            "FAR_performance",
            "Close_performance",
            "close_far_visitation",
            "maximum_dungeon_level",
            "turns",
            "stage_success_rate",
            "forgetting_gap",
        ],
        "figure_artifacts": ["Figure 4", "Figure 5", "Figure 6", "Figure 7", "Figure 8", "Figure 12"],
        "artifact_paths": [
            "results/plots/forgetting_analysis.png",
            "results/plots/robotic_sequence_stage_success.png",
        ],
        "trend_obligations": {
            "state_coverage_gap": RESULT_TREND_ASSERTIONS["state_coverage_gap_causes_deterioration"],
            "bc_em_ewc_maintain": RESULT_TREND_ASSERTIONS["bc_em_ewc_maintain"],
            "standard_finetune_no_transfer": RESULT_TREND_ASSERTIONS["standard_finetune_no_positive_transfer"],
        },
    },
    "appendix_a_toy_examples": {
        "id": "appendix_a_toy_examples",
        "section": "Appendix A",
        "title": "Toy Examples: Two-state MDPs and AppleRetrieval",
        "environments": ["two_state_mdp", "apple_retrieval"],
        "methods": [
            "training_from_scratch",
            "vanilla_finetune",
            "finetune_bc",
            "finetune_ewc",
            "finetune_em",
        ],
        "measurements": ["return", "FAR_performance", "forgetting_gap"],
        "figure_artifacts": ["Figure 9", "Figure 10", "Figure 11"],
        "artifact_paths": ["results/toy_diagnostics.json"],
    },
}

# ---------------------------------------------------------------------------
# Baseline registry (paper evidence contract)
# ---------------------------------------------------------------------------

BASELINE_REGISTRY: Dict[str, Any] = {
    "training_from_scratch": {
        "id": "training_from_scratch",
        "alias": "scratch",
        "description": "Train policy from random initialization on target task",
        "paper_role": "baseline",
        "algorithm": "ppo_or_sac",
        "init": "random",
    },
    "vanilla_finetune": {
        "id": "vanilla_finetune",
        "alias": "finetune",
        "description": "Initialize from pi_* and fine-tune with RL loss only",
        "paper_role": "baseline",
        "algorithm": "ppo_or_sac",
        "init": "pretrained_pi_star",
    },
    "finetune_bc": {
        "id": "finetune_bc",
        "alias": "Fine-tuning + BC",
        "description": "Fine-tune with RL loss + BC loss on S_BC buffer",
        "paper_role": "proposed_knowledge_retention",
        "algorithm": "ppo_or_sac_with_bc",
        "init": "pretrained_pi_star",
        "bc_loss_coefficient": BC_LOSS_COEFFICIENT,
        "s_bc_buffer_size": S_BC_BUFFER_SIZE,
    },
    "finetune_ewc": {
        "id": "finetune_ewc",
        "alias": "Fine-tuning + EWC",
        "description": "Fine-tune with RL loss + EWC penalty (diagonal Fisher)",
        "paper_role": "proposed_knowledge_retention",
        "algorithm": "ppo_or_sac_with_ewc",
        "init": "pretrained_pi_star",
        "ewc_coefficient": EWC_REGULARIZATION_COEFFICIENT,
        "fisher_sample_count": EWC_FISHER_SAMPLE_COUNT,
    },
    "finetune_em": {
        "id": "finetune_em",
        "alias": "Fine-tuning + EM",
        "description": "Fine-tune with episodic memory replay (10% protected old samples)",
        "paper_role": "proposed_knowledge_retention",
        "algorithm": "sac_with_em",
        "init": "pretrained_pi_star",
        "em_weight": EM_WEIGHT,
    },
    "ours": {
        "id": "ours",
        "alias": "scaled-bc + fine-tuning + ks",
        "description": "Best performing knowledge retention method (Scaled-BC + Fine-tuning + KS)",
        "paper_role": "proposed_best",
    },
    "ppo": {
        "id": "ppo",
        "alias": "PPO",
        "description": "Proximal Policy Optimization (on-policy baseline)",
        "paper_role": "algorithm_backbone",
    },
    "sac": {
        "id": "sac",
        "alias": "SAC",
        "description": "Soft Actor-Critic (off-policy baseline for robotics)",
        "paper_role": "algorithm_backbone",
    },
    "oracle": {
        "id": "oracle",
        "alias": "oracle",
        "description": "Oracle policy with access to all tasks simultaneously",
        "paper_role": "upper_bound_baseline",
    },
    "nle": {
        "id": "nle",
        "alias": "NLE baseline",
        "description": "Prior state-of-the-art on NetHack (Tuyls et al., 2023, ~5K points)",
        "paper_role": "prior_sota_baseline",
    },
    "ewc": {
        "id": "ewc",
        "alias": "EWC",
        "description": "Elastic Weight Consolidation knowledge retention method",
        "paper_role": "proposed_knowledge_retention",
    },
    "bc": {
        "id": "bc",
        "alias": "BC",
        "description": "Behavioral Cloning knowledge retention method",
        "paper_role": "proposed_knowledge_retention",
    },
}

# ---------------------------------------------------------------------------
# Artifact path registry (statically discoverable)
# ---------------------------------------------------------------------------

ARTIFACT_PATHS: Dict[str, str] = {
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
    "figure_22": "results/figures/figure_22.png",
    "figure_23": "results/figures/figure_23.png",
    "figure_25": "results/figures/figure_25.png",
    "figure_26": "results/figures/figure_26.png",
    "table_1": "results/tables/table_1.csv",
    "table_4": "results/tables/table_4.csv",
    "table_5": "results/tables/table_5.csv",
    "table_6": "results/tables/table_6.csv",
    "metrics_json": "results/metrics.json",
    "summary_csv": "results/summary.csv",
    "main_results_csv": "results/main_results.csv",
    "main_results_table_json": "results/main_results_table.json",
    "main_results_summary_md": "results/main_results_summary.md",
    "experiment_registry_json": "results/experiment_registry.json",
    "bc_metrics_json": "results/bc_metrics.json",
    "dataset_registry_json": "results/dataset_registry.json",
    "run_manifest_json": "results/run_manifest.json",
    "config_resolved_json": "results/config_resolved.json",
    "reproduction_inventory_json": "results/reproduction_inventory.json",
    "environment_manifest_json": "results/environment_manifest.json",
    "loss_trace_json": "results/loss_trace.json",
    "bc_buffer_manifest_json": "artifacts/bc_buffer_manifest.json",
    "robotic_stage_success_png": "results/plots/robotic_sequence_stage_success.png",
    "forgetting_analysis_png": "results/plots/forgetting_analysis.png",
    "main_comparison_png": "results/plots/main_comparison.png",
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ExperimentSectionAnalysisSpec:
    """
    Specification for Section 4/5 experiment analysis.
    Binds environments, methods, metrics, seeds, and artifact paths.
    """
    experiment_id: str = "section_4_main_result"
    environments: List[str] = field(default_factory=lambda: ["nethack", "montezuma", "robotic_sequence"])
    methods: List[str] = field(default_factory=lambda: [
        "training_from_scratch", "vanilla_finetune", "finetune_bc", "finetune_ewc", "finetune_em"
    ])
    seeds: List[int] = field(default_factory=lambda: SEED_LIST)
    evaluation_episodes: int = EVALUATION_EPISODES
    batch_size: int = BATCH_SIZE_128
    smoke_budget_steps: int = SMOKE_BUDGET_STEPS
    full_training_budget_steps: int = FULL_TRAINING_BUDGET_STEPS
    bc_loss_coefficient: float = BC_LOSS_COEFFICIENT
    ewc_regularization_coefficient: float = EWC_REGULARIZATION_COEFFICIENT
    em_weight: float = EM_WEIGHT
    s_bc_buffer_size: int = S_BC_BUFFER_SIZE
    output_dir: str = "results"
    mode: str = "smoke"  # smoke | train | eval | report
    config_source: str = "configs/setup.yaml"
    trend_assertions: Dict[str, str] = field(default_factory=lambda: RESULT_TREND_ASSERTIONS)
    dataset_registry: Dict[str, Any] = field(default_factory=lambda: DATASET_REGISTRY)
    metric_registry: Dict[str, Any] = field(default_factory=lambda: METRIC_REGISTRY)
    baseline_registry: Dict[str, Any] = field(default_factory=lambda: BASELINE_REGISTRY)
    experiment_registry: Dict[str, Any] = field(default_factory=lambda: EXPERIMENT_REGISTRY)
    artifact_paths: Dict[str, str] = field(default_factory=lambda: ARTIFACT_PATHS)


@dataclass
class EpisodeResult:
    """Per-episode evaluation result."""