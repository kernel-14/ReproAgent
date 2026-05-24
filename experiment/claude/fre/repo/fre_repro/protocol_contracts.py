"""Paper protocol contracts for the FRE reproduction runtime.

These helpers make the high-risk reproduction obligations explicit as active
JSON artifacts.  They are consumed by the runtime smoke route and can also be
read by full training/evaluation orchestration without importing simulator or
deep learning dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping


APPENDIX_A_HYPERPARAMETERS: Dict[str, Any] = {
    "batch_size": 512,
    "num_reward_bins": 32,
    "state_embedding_dim": 64,
    "reward_embedding_dim": 64,
    "encoder_token_dim": 128,
    "latent_z_dim": 128,
    "encoder_transformer_mlp_dims": [256, 256, 256, 256],
    "encoder_attention_heads": 4,
    "decoder_hidden_layers": [512, 512, 512],
    "rl_hidden_layers": [512, 512, 512],
    "optimizer": "Adam",
    "learning_rate": 1e-4,
    "discount": 0.88,
    "iql_expectile": 0.8,
    "awr_temperature": 3.0,
    "target_update_rate": 0.001,
}


def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(path)


def fre_architecture_contract() -> Dict[str, Any]:
    return {
        "artifact_type": "fre_architecture_contract",
        "fre_encoder": {
            "input": "set_of_state_reward_pairs",
            "reward_discretization": {
                "num_bins": 32,
                "rescale": "[0,1]",
                "operation": "floor(rescaled_reward * 32)",
            },
            "reward_embedding": "learned_embedding_table",
            "state_embedding": "learned_linear_projection",
            "token_construction": "concat(state_embedding_64, reward_embedding_64)",
            "transformer": {
                "permutation_invariant": True,
                "causal_mask": False,
                "positional_embeddings": False,
                "pooling": "mean_final_layer_tokens",
            },
            "variational_outputs": ["z_mean", "z_log_std"],
            "prior": "unit_gaussian",
        },
        "fre_decoder": {
            "input": "raw_state_concatenated_with_shared_z",
            "output": "predicted_scalar_reward",
            "loss": "mean_squared_error_on_decode_states",
            "decode_states_separate_from_encode_states": True,
        },
        "fre_conditioned_iql": {
            "conditioning": "z_concatenated_to_observation",
            "actor": {
                "distribution": "Gaussian",
                "network": "3x512_layernorm_relu",
                "outputs": ["mean", "log_std"],
                "log_std_clamp_min": -5.0,
            },
            "critic": {"input": "concat(observation, action, z)", "network": "3x512_layernorm_relu"},
            "value": {"input": "concat(observation, z)", "network": "3x512_layernorm_relu"},
            "target_critic": {"soft_update_tau": APPENDIX_A_HYPERPARAMETERS["target_update_rate"]},
        },
        "appendix_a_hyperparameters": APPENDIX_A_HYPERPARAMETERS,
    }


def benchmark_protocol_contract() -> Dict[str, Any]:
    return {
        "artifact_type": "benchmark_protocol_contract",
        "datasets_envs": {
            "antmaze": {
                "dataset": "antmaze-large-diverse-v2",
                "source": "D4RL",
                "online_eval": True,
                "max_trajectory_length": 2000,
                "xy_discretization_bins": 32,
                "tasks": [
                    "goal-bottom",
                    "goal-left",
                    "goal-top",
                    "goal-center",
                    "goal-right",
                    "vel_left",
                    "vel_up",
                    "vel_down",
                    "vel_right",
                    "random_simplex_seed_1_to_5",
                    "path_center",
                    "path_loop",
                    "path_edges",
                ],
            },
            "exorl_walker": {
                "dataset": "RND",
                "source": "ExORL",
                "eval_env": "custom DeepMind Control Suite",
                "append_physics_info": True,
                "max_trajectory_length": 1000,
                "tasks": ["walker_velocity_thresholds", "walker_dataset_goals"],
            },
            "exorl_cheetah": {
                "dataset": "RND",
                "source": "ExORL",
                "eval_env": "custom DeepMind Control Suite",
                "append_physics_info": True,
                "max_trajectory_length": 1000,
                "tasks": ["cheetah_run", "cheetah_run_backwards", "cheetah_walk", "cheetah_walk_backwards", "cheetah_dataset_goals"],
            },
            "kitchen": {
                "dataset": "kitchen-complete-v0",
                "source": "D4RL",
                "online_eval": True,
                "tasks": [
                    "microwave",
                    "kettle",
                    "light_switch",
                    "slide_cabinet",
                    "hinge_cabinet",
                    "bottom_burner",
                    "top_burner",
                ],
            },
        },
        "baselines": {
            "FB": {"implementation": "facebookresearch/controllable_agent", "reward_samples_eval": 5120},
            "SF": {"implementation": "facebookresearch/controllable_agent", "features": "ICM"},
            "GC-IQL": {
                "network": "3x512_layernorm_relu_gaussian",
                "goal_sampling": {"random_goal": 0.3, "geometric_future_goal": 0.5, "current_goal": 0.2},
            },
            "GC-BC": {
                "network": "3x512_layernorm_relu_gaussian",
                "log_std_clamp_min": -5.0,
                "goal_sampling": "geometric_future_goal_only",
            },
            "OPAL": {
                "encoder": "q_phi(z|tau) transformer over trajectory c",
                "decoder": "pi(a|s,z)",
                "comparison": "privileged_skill_selection_with_online_rollouts",
            },
        },
    }


def training_objective_contract() -> Dict[str, Any]:
    return {
        "artifact_type": "training_objective_contract",
        "fre_objective": {
            "name": "Equation 6 variational lower bound",
            "terms": [
                "sum_log_q_theta_eta_decode_state_given_state_z",
                "minus_beta_kl_p_theta_z_given_encode_pairs_to_unit_gaussian",
            ],
            "encode_decode_state_samples_are_separate": True,
            "random_reward_prior_labels_states": True,
        },
        "strided_training": [
            "train_encoder_decoder_eq6",
            "freeze_encoder",
            "train_fre_conditioned_iql_policy",
        ],
        "algorithm_1_route": [
            "sample_eta_from_random_reward_prior",
            "sample_K_encoder_states",
            "encode_state_reward_pairs_to_z",
            "train_actor_critic_value_with_reward_eta",
        ],
        "iql_losses": {
            "critic": "Bellman target r + discount * mask * next_value",
            "value": "expectile regression between Q and V",
            "actor": "advantage_weighted_regression from exp((Q-V)/temperature)",
            "target_update": "soft target update",
        },
        "policy_conditioning": "z concatenated to observations for actor, critic, value, and target critic",
    }


def tables_figures_contract() -> Dict[str, Any]:
    return {
        "artifact_type": "tables_figures_contract",
        "tables": {
            "table_1": "AntMaze/ExORL/Kitchen zero-shot normalized return and success comparisons",
            "table_2": "method capability comparison",
            "table_3": "Appendix A hyperparameters",
            "table_4": "AntMaze reward-prior subset ablations",
        },
        "figures": {
            "figure_1": "method overview route",
            "figure_2": "encoder-decoder reward reconstruction route",
            "figure_3": "AntMaze zero-shot qualitative route",
            "figure_4": "domain registry route",
            "figure_5": "reward-prior diversity ablation route",
            "figure_6": "domain-specific prior augmentation route",
            "figures_7_to_9": "additional AntMaze qualitative routes",
        },
        "paper_visible_artifacts_require_measured_records": True,
    }


def write_protocol_contract_artifacts(output_dir: Path) -> Dict[str, str]:
    payloads = {
        "fre_architecture_contract.json": fre_architecture_contract(),
        "benchmark_protocol_contract.json": benchmark_protocol_contract(),
        "training_objective_contract.json": training_objective_contract(),
        "tables_figures_contract.json": tables_figures_contract(),
    }
    return {name: _write_json(output_dir / name, payload) for name, payload in payloads.items()}

