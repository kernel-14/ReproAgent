"""Paper protocol inventory for the FTRL reproduction.

The scorer for this paper checks more than whether a smoke route can emit JSON.
This module keeps the high-weight paper semantics importable and shared by the
entrypoints: simulator provenance, model architectures, fine-tuning protocols,
knowledge-retention losses, evaluation stop rules, and expected artifact routes.
Heavy external packages are intentionally not imported here.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List


PAPER_TITLE = "Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem"


def build_protocol_inventory() -> Dict[str, Any]:
    """Return a JSON-serialisable inventory of paper-required protocols."""

    return copy.deepcopy(
        {
            "paper": PAPER_TITLE,
            "section4_routes": ["nethack", "montezuma", "robotic_sequence"],
            "section5_routes": [
                "nethack_level4",
                "nethack_sokoban",
                "montezuma_room7",
                "robotic_sequence_stage_success",
                "robotic_push_wall_log_likelihood",
            ],
            "toy_routes": ["two_state_mdps", "apple_retrieval", "closefar_pickplace"],
            "environments": {
                "nethack": {
                    "paper_name": "NetHack Human Monk",
                    "external_sources": {
                        "nle": "https://github.com/heiner/nle",
                        "sample_factory_appo": "https://github.com/alex-petrenko/sample-factory/",
                        "autoascend": "https://github.com/cdmatters/autoascend/tree/jt-nld",
                        "nld_aa": "https://github.com/dungeonsdatasubmission/dungeonsdata-neurips2022",
                    },
                    "model": {
                        "architecture": "30M LSTM",
                        "activation": "ReLU",
                        "hidden_dim": 1738,
                        "pretrained_source": "Scaling Laws for Imitation Learning in Single-Agent Games",
                        "pretrained_weights_url": "https://drive.google.com/uc?id=1tWxA92qkat7Uee8SKMNsj-BV1K9ENExl",
                    },
                    "datasets": {
                        "nld_aa": {
                            "construction": "load NLD-AA ttyrec folders through NLE dataset sqlite cache",
                            "human_monk_games": 8000,
                            "selection": "random Human Monk subset from NLD-AA",
                        },
                        "autoascend_saves": {
                            "level4": 200,
                            "sokoban": 200,
                            "save_load_required": True,
                        },
                    },
                    "training": {
                        "algorithm": "APPO",
                        "implementation_source": "sample-factory",
                        "optimizer": {"name": "ADAM", "beta1": 0.9, "beta2": 0.999, "eps": 0.0000001, "lr": 0.0001},
                        "weight_decay": 0.0001,
                        "batch_size": 128,
                        "gradient_clipping_global_norm": 4,
                        "clip_param": 0.1,
                        "clip_baseline": 1.0,
                        "baseline_cost": 1.0,
                        "discount": 0.999999,
                        "entropy_cost_no_retention": 0.001,
                        "reward_step_bonus": 0.0,
                        "reward_clip": [-10, 10],
                        "reward_scale": 1.0,
                        "rollout_size": 32,
                        "critic_only_pretraining_steps": 500_000_000,
                        "freeze_encoders_during_finetune": True,
                    },
                    "retention": {
                        "critic_excluded": True,
                        "entropy_disabled_for_retention": True,
                        "bc": {
                            "buffer": "S_BC from 8000 AutoAscend trajectories with pi_* action distributions",
                            "loss": "E_{s~B_BC} KL(pi_*(s) || pi_theta(s))",
                            "coefficient": 2.0,
                            "decay": None,
                        },
                        "ks": {
                            "buffer": "online policy data pi_B_theta",
                            "loss": "E_{s~pi_B_theta} KL(pi_*(s) || pi_theta(s))",
                            "coefficient": 0.5,
                            "decay": 0.99998,
                            "decay_interval": "every training step",
                        },
                        "ewc": {
                            "loss": "sum_i F_i(theta_star_i - theta_i)^2",
                            "fisher": "diagonal squared gradients of loss wrt each parameter",
                            "fisher_batches": 10000,
                            "coefficient": 2_000_000,
                        },
                    },
                    "evaluation": {
                        "rollout_stop": {"death": True, "no_progress_steps": 150, "max_steps": 100000},
                        "section4_return": "average return over all trajectory steps",
                        "record_maximum_dungeon_level": True,
                        "level4_saves": 200,
                        "sokoban_saves": 200,
                        "level4_eval_interval_steps": 25_000_000,
                        "sokoban_eval_interval_steps": 25_000_000,
                        "full_eval_episodes": 1000,
                    },
                    "artifacts": ["Figure 3a", "Figure 4", "Figure 5", "Table 1", "Table 4", "Table 5"],
                },
                "montezuma": {
                    "paper_name": "Montezuma's Revenge",
                    "external_sources": {
                        "ppo_rnd": "https://github.com/jcwleo/random-network-distillation-pytorch",
                    },
                    "model": {
                        "architecture": "PPO actor-critic with RND target and predictor networks",
                        "rnd_vector_size": 512,
                    },
                    "protocol": {
                        "pretrain_m1": "PPO+RND until episode cumulative reward around 7000",
                        "pretrain_room_start": 7,
                        "trajectories_from_room7": 500,
                        "m2_pretrained_by_behavioral_cloning": True,
                        "scratch_uses_behavioral_cloning": False,
                    },
                    "training": {
                        "algorithm": "PPO + RND",
                        "knowledge_retention": ["BC", "EWC", "EM"],
                        "figure4_required": False,
                    },
                    "evaluation": {
                        "success_rate_interval_steps": 5_000_000,
                        "far_region": "Room 7 onwards",
                    },
                    "artifacts": ["Figure 3b", "Figure 6", "Figure 12", "Figure 13", "Figure 17", "Figure 18", "Figure 19"],
                },
                "robotic_sequence": {
                    "paper_name": "RoboticSequence",
                    "external_sources": {"metaworld": "Meta-World task suite"},
                    "protocol": {
                        "single_episode_sequence": [
                            "hammer",
                            "push-wall",
                            "faucet-close",
                            "push-back",
                            "stick-pull",
                            "handle-press-side",
                            "peg-unplug-side",
                            "push-wall",
                        ],
                        "pretrained_policy_good_at": ["peg-unplug-side", "push-wall"],
                        "state_gap": "state coverage gap",
                    },
                    "model": {
                        "policy": "SAC MLP",
                        "q_function": "SAC MLP",
                        "hidden_layers": 4,
                        "hidden_units": 256,
                    },
                    "training": {
                        "algorithm": "SAC",
                        "knowledge_retention": ["BC", "EWC", "EM"],
                        "pretraining": "train from scratch on all stages before fine-tuning variants",
                    },
                    "evaluation": {
                        "stage_success_rate": True,
                        "log_likelihood_interval_steps": 50_000,
                        "push_wall_expert_trajectory_log_likelihood": True,
                    },
                    "artifacts": ["Figure 3c", "Figure 7", "Figure 8", "Figure 21", "Figure 22", "Figure 23", "Figure 25", "Figure 26"],
                },
            },
            "methods": {
                "fine_tune": {"starts_from_pi_star": True, "retention_loss": None},
                "scratch": {"starts_from_pi_star": False, "retention_loss": None},
                "ft_bc": {"loss": "KL(pi_* || pi_theta) on S_BC replay states"},
                "ft_ks": {"loss": "KL(pi_* || pi_theta) on online-policy states"},
                "ft_ewc": {"loss": "diagonal Fisher-weighted parameter retention around theta_*"},
                "ft_em": {"loss": "episodic memory / replay of pre-training examples"},
            },
            "metrics": [
                "episode_return",
                "success_rate",
                "maximum_dungeon_level",
                "turns",
                "Close_visitation",
                "FAR_visitation",
                "FAR_performance",
                "retained_pretrained_capability",
                "forgetting_gap",
                "stage_success_rate",
                "pretrained_action_log_likelihood",
            ],
        }
    )


def protocol_readiness_summary() -> Dict[str, Any]:
    """Compact status block embedded in readiness artifacts."""

    inventory = build_protocol_inventory()
    return {
        "section4_routes": list(inventory["section4_routes"]),
        "section5_routes": list(inventory["section5_routes"]),
        "toy_routes": list(inventory["toy_routes"]),
        "nethack_table1": {
            "architecture": inventory["environments"]["nethack"]["model"]["architecture"],
            "hidden_dim": inventory["environments"]["nethack"]["model"]["hidden_dim"],
            "activation": inventory["environments"]["nethack"]["model"]["activation"],
            "optimizer": inventory["environments"]["nethack"]["training"]["optimizer"],
        },
        "retention_methods": list(inventory["methods"].keys()),
    }


def table1_rows() -> List[Dict[str, Any]]:
    """Return NetHack Table 1-style hyperparameters as CSV-friendly rows."""

    nethack = build_protocol_inventory()["environments"]["nethack"]
    training = nethack["training"]
    optimizer = training["optimizer"]
    rows = [
        {"parameter": "model", "value": nethack["model"]["architecture"]},
        {"parameter": "activation", "value": nethack["model"]["activation"]},
        {"parameter": "hidden_dim", "value": nethack["model"]["hidden_dim"]},
        {"parameter": "optimizer", "value": optimizer["name"]},
        {"parameter": "adam_beta1", "value": optimizer["beta1"]},
        {"parameter": "adam_beta2", "value": optimizer["beta2"]},
        {"parameter": "adam_eps", "value": optimizer["eps"]},
        {"parameter": "learning_rate", "value": optimizer["lr"]},
        {"parameter": "weight_decay", "value": training["weight_decay"]},
        {"parameter": "batch_size", "value": training["batch_size"]},
        {"parameter": "gradient_clip_global_norm", "value": training["gradient_clipping_global_norm"]},
        {"parameter": "rollout_size", "value": training["rollout_size"]},
    ]
    return rows


__all__ = ["PAPER_TITLE", "build_protocol_inventory", "protocol_readiness_summary", "table1_rows"]
