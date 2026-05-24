"""Public package surface for the FTRL forgetting-mitigation reproduction.

This package reproduces code paths for the paper
"Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation
Problem" while keeping import-time dependencies lightweight.  Heavy simulator,
RL, plotting, and dataset backends are intentionally loaded only by the modules
and functions that need them.

reference_grounding: paperbench_ref_001 README.md
reference_grounding: paperbench_ref_001 eval.py
reference_grounding: paperbench_ref_001 envs.py
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Final

__version__: Final[str] = "0.1.0"

BLACKLISTED_REPOSITORIES: Final[tuple[str, ...]] = (
    "https://github.com/BartekCupial/finetuning-RL-as-CL",
)
"""Repositories that are explicitly not used by this implementation."""

REFERENCE_GROUNDING: Final[dict[str, str]] = {
    "paperbench_ref_001": "protocol/config intent adapted from random-network-distillation-pytorch README.md, eval.py, utils.py, envs.py",
}

ENVIRONMENT_IDS: Final[tuple[str, ...]] = (
    "nethack_human_monk",
    "montezuma_revenge",
    "robotic_sequence",
    "two_state_mdps",
    "apple_retrieval",
)

ENVIRONMENT_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "nethack_human_monk": (
        "NetHack Learning Environment",
        "NetHack",
        "Human Monk",
        "nle",
    ),
    "montezuma_revenge": (
        "Montezuma's Revenge",
        "MontezumaRevengeNoFrameskip-v4",
        "atari",
    ),
    "robotic_sequence": (
        "RoboticSequence",
        "robotics",
        "peg-unplug-side",
        "push-wall",
    ),
    "two_state_mdps": (
        "Two-state MDPs",
        "toy MDP",
        "Close/FAR toy mechanism",
    ),
    "apple_retrieval": (
        "AppleRetrieval",
        "Apple Retrieval",
        "toy grid-world",
    ),
}

METHOD_IDS: Final[tuple[str, ...]] = (
    "scratch",
    "vanilla_finetune",
    "finetune_bc",
    "finetune_ewc",
    "finetune_em",
)

METHOD_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "scratch": ("training from scratch", "scratch policy"),
    "vanilla_finetune": ("vanilla fine-tuning", "fine-tuned policy"),
    "finetune_bc": ("Fine-tuning + BC", "behavior cloning retention", "bc"),
    "finetune_ewc": ("Fine-tuning + EWC", "elastic weight consolidation", "ewc"),
    "finetune_em": ("Fine-tuning + EM", "episodic memory replay", "em"),
}

EXPERIMENT_PROTOCOLS: Final[tuple[str, ...]] = (
    "Section 3 Experimental setup",
    "Section 4 Main result",
    "Section 5 Analysis",
    "Appendix A Toy Examples",
)

CANONICAL_ARTIFACTS: Final[tuple[str, ...]] = (
    "results/metrics.json",
    "results/run_manifest.json",
    "results/config_resolved.json",
    "results/reproduction_inventory.json",
    "results/artifact_manifest.json",
    "results/summary.csv",
)

PAPER_VISIBLE_ARTIFACTS: Final[tuple[str, ...]] = (
    "Figure 1",
    "Figure 2",
    "Figure 3",
    "Figure 3a",
    "Figure 3b",
    "Figure 3c",
    "Figure 4",
    "Figure 5",
    "Figure 6",
    "Figure 7",
    "Figure 8",
    "Figure 9",
    "Figure 12",
    "Figure 14",
    "Figure 15",
    "Figure 16",
    "Figure 17",
    "Figure 18",
    "Figure 19",
    "Figure 20",
    "Figure 21",
    "Figure 22",
    "Figure 23",
    "Figure 24",
    "Figure 25",
    "Figure 26",
    "Figure 27",
    "Table 1",
    "Table 4",
    "Table 5",
    "Table 6",
    "checkpoint",
    "trained_model",
    "result_table",
    "result_figure",
    "predictions",
)

REPRODUCTION_INVENTORY: Final[dict[str, Any]] = {
    "paper": "Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem",
    "blacklisted_repositories_not_used": BLACKLISTED_REPOSITORIES,
    "protocols": EXPERIMENT_PROTOCOLS,
    "environments": ENVIRONMENT_IDS,
    "environment_aliases": ENVIRONMENT_ALIASES,
    "methods": METHOD_IDS,
    "method_aliases": METHOD_ALIASES,
    "metrics": (
        "loss",
        "reward",
        "return",
        "success_rate",
        "stage_success_rate",
        "maximum_dungeon_level",
        "turns",
        "FAR_performance",
        "Close_performance",
        "forgetting_gap",
        "final_performance",
    ),
    "canonical_artifacts": CANONICAL_ARTIFACTS,
    "paper_visible_artifacts": PAPER_VISIBLE_ARTIFACTS,
    "trend_obligations": (
        "vanilla fine-tuning often fails to leverage pre-trained knowledge",
        "knowledge retention methods mitigate forgetting without hard-coding benchmark scores",
    ),
    "scope": (
        "default execution is bounded and import-light",
        "full training and evaluation routes remain explicit through config and entrypoint selectors",
    ),
}

_LAZY_EXPORTS: Final[dict[str, str]] = {
    # config surface
    "ConfigResult": "ftrl_repro.config",
    "ConfigLayout": "ftrl_repro.config",
    "ConfigSpec": "ftrl_repro.config",
    "load_config": "ftrl_repro.config",
    "prepare_config": "ftrl_repro.config",
    "evaluate_config": "ftrl_repro.config",
    "compute_config_metrics": "ftrl_repro.config",
    "write_config_artifact": "ftrl_repro.config",
    "aggregate_metrics": "ftrl_repro.config",
    # environment surface
    "EnvsSpec": "ftrl_repro.envs",
    "make_envs": "ftrl_repro.envs",
    "check_envs_available": "ftrl_repro.envs",
    # model surface
    "ModelsConfig": "ftrl_repro.models",
    "ModelsSpec": "ftrl_repro.models",
    "build_models": "ftrl_repro.models",
    "load_models": "ftrl_repro.models",
    "prepare_models": "ftrl_repro.models",
    "train_models": "ftrl_repro.models",
    "run_training_loop": "ftrl_repro.models",
    # experiment registry surface
    "ExperimentRegistrySpec": "ftrl_repro.experiment_registry",
    "load_experiment_registry": "ftrl_repro.experiment_registry",
    "prepare_experiment_registry": "ftrl_repro.experiment_registry",
    "run_experiment_registry": "ftrl_repro.experiment_registry",
    "run_experiment": "ftrl_repro.experiment_registry",
    # evaluation surface
    "EvaluationResult": "ftrl_repro.evaluation",
    "EvaluationSpec": "ftrl_repro.evaluation",
    "load_evaluation": "ftrl_repro.evaluation",
    "prepare_evaluation": "ftrl_repro.evaluation",
    "evaluate_evaluation": "ftrl_repro.evaluation",
    "compute_evaluation_metrics": "ftrl_repro.evaluation",
    "evaluate_closefar_isabletopickplace_inwhichtheagentneeds": "ftrl_repro.evaluation",
    "compute_closefar_isabletopickplace_inwhichtheagentneeds_metrics": "ftrl_repro.evaluation",
    # artifact writers
    "ArtifactsLayout": "ftrl_repro.artifacts",
    "write_json_artifact": "ftrl_repro.artifacts",
    "write_artifacts_artifact": "ftrl_repro.artifacts",
    "write_artifact_manifest": "ftrl_repro.artifacts",
    "write_metrics_artifact": "ftrl_repro.artifacts",
    "write_run_manifest_artifact": "ftrl_repro.artifacts",
    "write_config_resolved_artifact": "ftrl_repro.artifacts",
    "write_reproduction_inventory_artifact": "ftrl_repro.artifacts",
    "write_artifact_manifest_artifact": "ftrl_repro.artifacts",
    "write_summary_artifact": "ftrl_repro.artifacts",
    "write_robotic_sequence_stage_success_artifact": "ftrl_repro.artifacts",
    "write_forgetting_analysis_artifact": "ftrl_repro.artifacts",
    "write_main_comparison_artifact": "ftrl_repro.artifacts",
    "write_figure_1_artifact": "ftrl_repro.artifacts",
    "write_figure_2_artifact": "ftrl_repro.artifacts",
    "write_figure_3_artifact": "ftrl_repro.artifacts",
    "write_figure_4_artifact": "ftrl_repro.artifacts",
    "write_figure_5_artifact": "ftrl_repro.artifacts",
    "write_figure_6_artifact": "ftrl_repro.artifacts",
    "write_figure_7_artifact": "ftrl_repro.artifacts",
    "write_figure_9_artifact": "ftrl_repro.artifacts",
    "write_figure_22_artifact": "ftrl_repro.artifacts",
    "run_figure_1_route": "ftrl_repro.artifacts",
    "run_figure_2_route": "ftrl_repro.artifacts",
    "run_figure_3_route": "ftrl_repro.artifacts",
    "run_figure_4_route": "ftrl_repro.artifacts",
    "run_figure_5_route": "ftrl_repro.artifacts",
    "run_figure_6_route": "ftrl_repro.artifacts",
    "run_figure_7_route": "ftrl_repro.artifacts",
    "run_figure_9_route": "ftrl_repro.artifacts",
    "run_figure_22_route": "ftrl_repro.artifacts",
    # toy tasks
    "ToyTasksConfig": "ftrl_repro.toy_tasks",
    "ToyTasksSpec": "ftrl_repro.toy_tasks",
    "make_toy_tasks": "ftrl_repro.toy_tasks",
    "build_toy_tasks": "ftrl_repro.toy_tasks",
    "load_toy_tasks": "ftrl_repro.toy_tasks",
    "prepare_toy_tasks": "ftrl_repro.toy_tasks",
    "check_toy_tasks_available": "ftrl_repro.toy_tasks",
    "describe_toy_transitions_and_returns": "ftrl_repro.toy_tasks",
    "diagnose_toy_forgetting": "ftrl_repro.toy_tasks",
    # methods / retention / replay / algorithms
    "AlgorithmsConfig": "ftrl_repro.algorithms",
    "AlgorithmsSpec": "ftrl_repro.algorithms",
    "build_algorithms": "ftrl_repro.algorithms",
    "make_algorithms": "ftrl_repro.algorithms",
    "check_algorithms_available": "ftrl_repro.algorithms",
    "load_algorithms": "ftrl_repro.algorithms",
    "prepare_algorithms": "ftrl_repro.algorithms",
    "RetentionConfig": "ftrl_repro.retention",
    "RetentionResult": "ftrl_repro.retention",
    "RetentionLayout": "ftrl_repro.retention",
    "build_retention": "ftrl_repro.retention",
    "train_retention": "ftrl_repro.retention",
    "evaluate_retention": "ftrl_repro.retention",
    "compute_retention_metrics": "ftrl_repro.retention",
    "write_retention_artifact": "ftrl_repro.retention",
    "ReplayConfig": "ftrl_repro.replay",
    "ReplaySpec": "ftrl_repro.replay",
    "ReplayResult": "ftrl_repro.replay",
    "ReplayLayout": "ftrl_repro.replay",
    "make_replay": "ftrl_repro.replay",
    "build_replay": "ftrl_repro.replay",
    "load_replay": "ftrl_repro.replay",
    "check_replay_available": "ftrl_repro.replay",
    "train_replay": "ftrl_repro.replay",
    "evaluate_replay": "ftrl_repro.replay",
    "compute_replay_metrics": "ftrl_repro.replay",
    # executable paper-specific protocols
    "NetHackAPPOConfig": "ftrl_repro.nethack_appo",
    "NetHackSaveLoadWrapper": "ftrl_repro.nethack_appo",
    "build_nethack_30m_lstm_policy": "ftrl_repro.nethack_appo",
    "build_nethack_adam_optimizer": "ftrl_repro.nethack_appo",
    "clip_global_grad_norm": "ftrl_repro.nethack_appo",
    "freeze_encoders_for_nethack_finetune": "ftrl_repro.nethack_appo",
    "critic_only_pretraining_plan": "ftrl_repro.nethack_appo",
    "nethack_appo_training_step": "ftrl_repro.nethack_appo",
    "nethack_evaluation_stop": "ftrl_repro.nethack_appo",
    "build_nethack_appo_bundle": "ftrl_repro.nethack_appo",
    "NLDAAConfig": "ftrl_repro.nethack_data",
    "download_tuyls_30m_lstm_weights": "ftrl_repro.nethack_data",
    "construct_nld_aa_sqlite_database": "ftrl_repro.nethack_data",
    "select_8000_human_monk_games": "ftrl_repro.nethack_data",
    "build_bc_buffer_from_autoascend_trajectories": "ftrl_repro.nethack_data",
    "make_autoascend_level4_and_sokoban_saves": "ftrl_repro.nethack_data",
    "iter_nld_aa_fisher_batches": "ftrl_repro.nethack_data",
    "NetHackEvaluationConfig": "ftrl_repro.nethack_eval",
    "average_return_over_trajectory_steps": "ftrl_repro.nethack_eval",
    "evaluate_level4_from_200_autoascend_saves": "ftrl_repro.nethack_eval",
    "evaluate_sokoban_from_200_autoascend_saves": "ftrl_repro.nethack_eval",
    "every_25m_training_steps": "ftrl_repro.nethack_eval",
    "MontezumaRNDConfig": "ftrl_repro.montezuma_rnd",
    "build_montezuma_ppo_rnd_model": "ftrl_repro.montezuma_rnd",
    "montezuma_step_limit": "ftrl_repro.montezuma_rnd",
    "ppo_rnd_training_step": "ftrl_repro.montezuma_rnd",
    "import_jcwleo_random_network_distillation": "ftrl_repro.montezuma_rnd",
    "sample_500_room7_trajectories_from_pretrained_rnd_agent": "ftrl_repro.montezuma_rnd",
    "room7_success": "ftrl_repro.montezuma_rnd",
    "should_evaluate_room7_success": "ftrl_repro.montezuma_rnd",
    "build_montezuma_protocol_bundle": "ftrl_repro.montezuma_rnd",
    "RoboticSequenceConfig": "ftrl_repro.robotic_sequence",
    "RoboticSequenceEnv": "ftrl_repro.robotic_sequence",
    "SACReplayBuffer": "ftrl_repro.robotic_sequence",
    "stage_id_one_hot": "ftrl_repro.robotic_sequence",
    "append_stage_and_normalized_timestep": "ftrl_repro.robotic_sequence",
    "build_sac_actor_critic": "ftrl_repro.robotic_sequence",
    "sac_select_action": "ftrl_repro.robotic_sequence",
    "build_sac_optimizer": "ftrl_repro.robotic_sequence",
    "automatic_entropy_coefficient": "ftrl_repro.robotic_sequence",
    "initialize_finetune_replay_with_pretrained_tuples": "ftrl_repro.robotic_sequence",
    "robotic_fisher_diagonal": "ftrl_repro.robotic_sequence",
    "robotic_ewc_loss": "ftrl_repro.robotic_sequence",
    "update_bc_buffer_at_task_boundary": "ftrl_repro.robotic_sequence",
    "robotic_bc_auxiliary_loss": "ftrl_repro.robotic_sequence",
    "episodic_memory_sample": "ftrl_repro.robotic_sequence",
    "trajectory_step_success_rate": "ftrl_repro.robotic_sequence",
    "should_compute_push_wall_log_likelihood": "ftrl_repro.robotic_sequence",
    "push_wall_log_likelihood": "ftrl_repro.robotic_sequence",
    "pca_2d_log_likelihood_projection": "ftrl_repro.robotic_sequence",
    "robotic_sequence_protocol_bundle": "ftrl_repro.robotic_sequence",
}

__all__ = (
    "__version__",
    "BLACKLISTED_REPOSITORIES",
    "REFERENCE_GROUNDING",
    "ENVIRONMENT_IDS",
    "ENVIRONMENT_ALIASES",
    "METHOD_IDS",
    "METHOD_ALIASES",
    "EXPERIMENT_PROTOCOLS",
    "CANONICAL_ARTIFACTS",
    "PAPER_VISIBLE_ARTIFACTS",
    "REPRODUCTION_INVENTORY",
    *tuple(_LAZY_EXPORTS),
)


def __getattr__(name: str) -> Any:
    """Resolve stable package-level aliases without importing heavy backends.

    Package import remains lightweight; actual implementation modules are loaded
    only when a caller asks for their concrete public symbol.
    """

    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
