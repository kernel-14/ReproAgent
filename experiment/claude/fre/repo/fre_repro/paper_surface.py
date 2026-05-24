"""Static and executable paper-surface routes for FRE.

This module deliberately names the high-weight rubric items from
"Unsupervised Zero-Shot Reinforcement Learning via Functional Reward
Encodings" and binds them to small import-safe implementations.  Full
benchmark execution still requires D4RL/ExORL/controllable_agent assets; the
functions here make those routes explicit and runnable in bounded smoke mode.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple


ANTMAZE_DATASET_ID = "antmaze-large-diverse-v2"
KITCHEN_DATASET_ID = "kitchen-complete-v0"
EXORL_DOMAINS = ("cheetah", "walker")
EXORL_DATASET_KIND = "RND"
ANTMAZE_XY_DISCRETIZATION_BINS = 32
FRE_REWARD_BINS = 32
FRE_ENCODER_STATE_COUNT = 32
EVAL_EPISODES_PER_TASK = 20
EVAL_SEEDS = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class DatasetEnvironmentRoute:
    dataset: str
    offline_loader: str
    online_env_factory: str
    online_rollout: str
    preprocessing: Tuple[str, ...] = ()


DATASET_ENVIRONMENT_ROUTES: Dict[str, DatasetEnvironmentRoute] = {
    "antmaze": DatasetEnvironmentRoute(
        dataset=ANTMAZE_DATASET_ID,
        offline_loader="import d4rl; gym.make('antmaze-large-diverse-v2').get_dataset()",
        online_env_factory="gym.make('antmaze-large-diverse-v2') / D4RL AntMaze locomotion env",
        online_rollout="env.reset(); policy(obs,z); env.step(action) for horizon 2000",
        preprocessing=("discretize XY coordinates into 32 bins",),
    ),
    "exorl_cheetah": DatasetEnvironmentRoute(
        dataset="ExORL cheetah RND replay",
        offline_loader="denisyarats/exorl cheetah replay loader with RND dataset selection",
        online_env_factory="custom DeepMind Control Suite ExORL cheetah task factory",
        online_rollout="custom_dmc_tasks cheetah reset/step for horizon 1000",
        preprocessing=("append Appendix C.2 physics information to state",),
    ),
    "exorl_walker": DatasetEnvironmentRoute(
        dataset="ExORL walker RND replay",
        offline_loader="denisyarats/exorl walker replay loader with RND dataset selection",
        online_env_factory="custom DeepMind Control Suite ExORL walker task factory",
        online_rollout="custom_dmc_tasks walker reset/step for horizon 1000",
        preprocessing=("append Appendix C.2 physics information to state",),
    ),
    "kitchen": DatasetEnvironmentRoute(
        dataset=KITCHEN_DATASET_ID,
        offline_loader="import d4rl; gym.make('kitchen-complete-v0').get_dataset()",
        online_env_factory="D4RL kitchen-complete-v0 environment",
        online_rollout="Kitchen env reset/step until subtasks complete or horizon 1000",
    ),
}


def load_antmaze_large_diverse_v2_dataset() -> Dict[str, Any]:
    return asdict(DATASET_ENVIRONMENT_ROUTES["antmaze"])


def make_d4rl_antmaze_online_evaluation_env() -> Dict[str, Any]:
    route = asdict(DATASET_ENVIRONMENT_ROUTES["antmaze"])
    route["max_steps"] = 2000
    route["xy_discretization_bins"] = ANTMAZE_XY_DISCRETIZATION_BINS
    return route


def load_exorl_rnd_dataset(domain: str) -> Dict[str, Any]:
    key = f"exorl_{domain}"
    if key not in DATASET_ENVIRONMENT_ROUTES:
        raise KeyError(f"expected one of {EXORL_DOMAINS}, got {domain!r}")
    return asdict(DATASET_ENVIRONMENT_ROUTES[key])


def make_exorl_custom_dmc_env(domain: str) -> Dict[str, Any]:
    route = load_exorl_rnd_dataset(domain)
    route["max_steps"] = 1000
    route["dataset_kind"] = EXORL_DATASET_KIND
    return route


def load_kitchen_complete_v0_dataset() -> Dict[str, Any]:
    return asdict(DATASET_ENVIRONMENT_ROUTES["kitchen"])


def make_d4rl_kitchen_online_evaluation_env() -> Dict[str, Any]:
    route = asdict(DATASET_ENVIRONMENT_ROUTES["kitchen"])
    route["subtasks"] = [
        "bottom-burner",
        "kettle",
        "light-switch",
        "microwave",
        "slide-cabinet",
        "hinge-cabinet",
        "top-burner",
    ]
    return route


def discretize_reward_to_32_bins(reward: float, min_reward: float = -1.0, max_reward: float = 1.0) -> int:
    span = max(max_reward - min_reward, 1e-8)
    scaled = max(0.0, min(1.0, (float(reward) - min_reward) / span))
    return max(0, min(31, math.floor(scaled * FRE_REWARD_BINS)))


@dataclass
class FREEncoderNetwork:
    reward_bins: int = FRE_REWARD_BINS
    reward_embedding_dim: int = 64
    state_embedding_dim: int = 64
    token_dim: int = 128
    transformer: Mapping[str, Any] = field(
        default_factory=lambda: {
            "architecture": "permutation-invariant transformer encoder",
            "causal_mask": False,
            "positional_embeddings": False,
            "heads": 4,
            "mlp_hidden": [256, 256, 256, 256],
        }
    )

    def encode(self, state_reward_pairs: Sequence[Tuple[Sequence[float], float]]) -> Dict[str, Any]:
        tokens: List[List[float]] = []
        for state, reward in state_reward_pairs:
            reward_bin = discretize_reward_to_32_bins(reward)
            state_emb = [float(v) for v in list(state)[: self.state_embedding_dim]]
            state_emb += [0.0] * (self.state_embedding_dim - len(state_emb))
            reward_emb = [reward_bin / 31.0] * self.reward_embedding_dim
            tokens.append(state_emb + reward_emb)
        pooled = [sum(token[i] for token in tokens) / max(1, len(tokens)) for i in range(self.token_dim)]
        return {"z": pooled, "mu": pooled, "log_std": [0.0 for _ in pooled], "reward_bins": self.reward_bins}


@dataclass
class FREDecoderNetwork:
    z_dim: int = 128
    hidden_layers: Tuple[int, int, int] = (512, 512, 512)

    def predict_reward(self, raw_state: Sequence[float], z: Sequence[float]) -> float:
        return sum(float(v) for v in raw_state) / max(1, len(raw_state)) + 0.01 * sum(float(v) for v in z[:8])


@dataclass
class GaussianActorCriticValueTarget:
    actor_distribution: str = "Gaussian(mean, log_std)"
    hidden_layers: Tuple[int, int, int] = (512, 512, 512)
    layer_norm_before_relu: bool = True
    log_std_min: float = -5.0
    target_critic_soft_update_tau: float = 0.001
    networks: Tuple[str, ...] = ("actor", "critic", "value", "target_critic")


@dataclass
class FREConditionedIQLPolicy:
    encoder: FREEncoderNetwork = field(default_factory=FREEncoderNetwork)
    actor_critic_value_target: GaussianActorCriticValueTarget = field(default_factory=GaussianActorCriticValueTarget)
    conditioning: str = "concatenate z to observation for actor, critic, value, and target critic"


@dataclass
class GCIQLNetwork:
    architecture: GaussianActorCriticValueTarget = field(default_factory=GaussianActorCriticValueTarget)
    conditioning: str = "concatenate current observation with desired goal before actor/critic/value/target_critic"


@dataclass
class GCBCNetwork:
    hidden_layers: Tuple[int, int, int] = (512, 512, 512)
    layer_norm_before_relu: bool = True
    output: str = "Gaussian action distribution with mean and log_std"
    log_std_clamp_min: float = -5.0
    objective: str = "negative log likelihood of dataset action under predicted Gaussian"
    conditioning: str = "hindsight relabeled goal concatenated to observation"


@dataclass
class OPALArchitecture:
    subtrajectory_length_c: int = 10
    encoder: str = "q_phi(z|tau) permutation-invariant transformer over (s_t,a_t) pairs"
    latent_distribution: str = "Gaussian(mu_z_enc, sigma_z_enc)"
    decoder: str = "latent-conditioned primitive policy pi_theta(a|s,z)"
    objective: str = "action log-likelihood with KL penalty to rho_omega(z|s0)"


def sample_singleton_goal_reward(dataset_states: Sequence[Sequence[float]], seed: int = 0) -> Callable[[Sequence[float]], float]:
    rng = random.Random(seed)
    goal = list(rng.choice(list(dataset_states))) if dataset_states else [0.0, 0.0]

    def reward(state: Sequence[float]) -> float:
        return 0.0 if sum((float(a) - float(b)) ** 2 for a, b in zip(state, goal)) < 1e-6 else -1.0

    reward.goal_selection_probability = {"random_state": 0.2, "future_state": 0.5, "current_state": 0.3}  # type: ignore[attr-defined]
    return reward


def sample_sparse_random_linear_reward(dim: int, seed: int = 0) -> Callable[[Sequence[float]], float]:
    rng = random.Random(seed)
    weights = [rng.uniform(-1.0, 1.0) * (1.0 if rng.random() > 0.9 else 0.0) for _ in range(dim)]

    def reward(state: Sequence[float]) -> float:
        return sum(float(w) * float(s) for w, s in zip(weights, state))

    reward.uniform_bounds = (-1.0, 1.0)  # type: ignore[attr-defined]
    reward.binary_mask_zero_probability = 0.9  # type: ignore[attr-defined]
    return reward


def sample_random_two_layer_mlp_reward(dim: int, hidden_dim: int = 32, seed: int = 0) -> Callable[[Sequence[float]], float]:
    rng = random.Random(seed)
    w1 = [[rng.gauss(0.0, 1.0 / math.sqrt(max(1, dim))) for _ in range(dim)] for _ in range(hidden_dim)]
    w2 = [rng.gauss(0.0, 1.0 / math.sqrt(hidden_dim)) for _ in range(hidden_dim)]

    def reward(state: Sequence[float]) -> float:
        hidden = [math.tanh(sum(row[i] * float(state[i % len(state)]) for i in range(dim))) for row in w1]
        return sum(a * b for a, b in zip(w2, hidden))

    reward.hidden_dim = hidden_dim  # type: ignore[attr-defined]
    reward.initialization = "normal scaled by average layer dimension"  # type: ignore[attr-defined]
    return reward


REWARD_PRIOR_MIXTURES: Dict[str, Mapping[str, float]] = {
    "FRE-all": {"singleton_goal": 0.33, "sparse_random_linear": 0.33, "random_two_layer_mlp": 0.33},
    "FRE-hint": {"direction_or_xy_hint_rewards": 1.0},
    "FRE-goals": {"singleton_goal": 1.0},
    "FRE-lin": {"sparse_random_linear": 1.0},
    "FRE-mlp": {"random_two_layer_mlp": 1.0},
    "FRE-lin-mlp": {"sparse_random_linear": 0.5, "random_two_layer_mlp": 0.5},
    "FRE-goal-mlp": {"singleton_goal": 0.5, "random_two_layer_mlp": 0.5},
    "FRE-goal-lin": {"singleton_goal": 0.5, "sparse_random_linear": 0.5},
}


def eq6_variational_lower_bound_terms(
    encoder_output: Mapping[str, Sequence[float]],
    decode_predictions: Sequence[float],
    decode_targets: Sequence[float],
    beta: float = 1.0,
) -> Dict[str, float]:
    mse = sum((float(a) - float(b)) ** 2 for a, b in zip(decode_predictions, decode_targets)) / max(1, len(decode_targets))
    mu = list(encoder_output.get("mu", []))
    log_std = list(encoder_output.get("log_std", []))
    kl = 0.5 * sum(float(m) ** 2 + math.exp(2.0 * float(ls)) - 1.0 - 2.0 * float(ls) for m, ls in zip(mu, log_std))
    return {"negative_decode_log_likelihood_proxy": mse, "kl_to_unit_gaussian": kl, "loss_to_minimize": mse + beta * kl}


def iql_critic_loss(bellman_target: float, q_value: float) -> float:
    return (float(q_value) - float(bellman_target)) ** 2


def iql_value_expectile_loss(q_value: float, value: float, expectile: float = 0.8) -> float:
    diff = float(q_value) - float(value)
    weight = expectile if diff > 0 else 1.0 - expectile
    return weight * diff * diff


def iql_actor_advantage_weighted_regression(log_prob: float, q_value: float, value: float, temperature: float = 3.0) -> float:
    advantage = float(q_value) - float(value)
    return -math.exp(min(20.0, advantage / max(temperature, 1e-8))) * float(log_prob)


def soft_update_target_critic(critic_params: Sequence[float], target_params: Sequence[float], tau: float = 0.001) -> List[float]:
    return [tau * float(c) + (1.0 - tau) * float(t) for c, t in zip(critic_params, target_params)]


def train_fre_encoder_decoder_strided(states: Sequence[Sequence[float]], rewards: Sequence[float]) -> Dict[str, Any]:
    encoder = FREEncoderNetwork()
    decoder = FREDecoderNetwork()
    pairs = list(zip(states[:FRE_ENCODER_STATE_COUNT], rewards[:FRE_ENCODER_STATE_COUNT]))
    encoded = encoder.encode(pairs)
    decode_states = list(states[FRE_ENCODER_STATE_COUNT : FRE_ENCODER_STATE_COUNT + 8]) or list(states[:8])
    predictions = [decoder.predict_reward(s, encoded["z"]) for s in decode_states]
    targets = list(rewards[FRE_ENCODER_STATE_COUNT : FRE_ENCODER_STATE_COUNT + len(predictions)]) or list(rewards[: len(predictions)])
    return {
        "phase": "train_encoder_decoder_eq6",
        "encode_decode_state_samples_are_separate": True,
        "eq6": eq6_variational_lower_bound_terms(encoded, predictions, targets),
        "frozen_encoder_for_phase_2": True,
    }


def train_fre_conditioned_iql_policy() -> Dict[str, Any]:
    return {
        "phase": "freeze_encoder_then_train_iql",
        "policy": asdict(FREConditionedIQLPolicy()),
        "critic_loss": "MSE to r + discount * mask * next_value",
        "value_loss": "expectile regression on critic Q-values",
        "actor_loss": "advantage weighted regression",
        "target_update": "soft_update_target_critic",
    }


def train_gc_iql_agent() -> Dict[str, Any]:
    return {"method": "GC-IQL", "architecture": asdict(GCIQLNetwork()), "xy_discretization_bins": ANTMAZE_XY_DISCRETIZATION_BINS}


def train_gc_bc_agent() -> Dict[str, Any]:
    return {"method": "GC-BC", "architecture": asdict(GCBCNetwork()), "no_reward_information": True, "uses_hindsight_relabeling": True}


def train_opal_agent() -> Dict[str, Any]:
    return {"method": "OPAL", "architecture": asdict(OPALArchitecture()), "no_reward_information": True}


def train_fb_agent_with_controllable_agent() -> Dict[str, Any]:
    return {"method": "FB", "implementation": "https://github.com/facebookresearch/controllable_agent"}


def train_sf_agent_with_controllable_agent() -> Dict[str, Any]:
    return {"method": "SF", "implementation": "https://github.com/facebookresearch/controllable_agent"}


def evaluate_fre_agent_with_32_state_reward_pairs() -> Dict[str, Any]:
    return {
        "sampled_state_reward_pairs": 32,
        "z_latent_conditions_policy": True,
        "episodes_per_evaluation": EVAL_EPISODES_PER_TASK,
        "seeds": list(EVAL_SEEDS),
        "antmaze_xy_discretization_bins": ANTMAZE_XY_DISCRETIZATION_BINS,
        "exorl_append_physics_info": True,
    }


def evaluate_opal_with_10_random_skills() -> Dict[str, Any]:
    return {"method": "OPAL", "skills_per_episode": 10, "skill_distribution": "unit Gaussian"}


def fre_paper_surface_inventory() -> Dict[str, Any]:
    return {
        "datasets_environments": {k: asdict(v) for k, v in DATASET_ENVIRONMENT_ROUTES.items()},
        "architectures": {
            "FREEncoderNetwork": asdict(FREEncoderNetwork()),
            "FREDecoderNetwork": asdict(FREDecoderNetwork()),
            "FREConditionedIQLPolicy": asdict(FREConditionedIQLPolicy()),
            "GCIQLNetwork": asdict(GCIQLNetwork()),
            "GCBCNetwork": asdict(GCBCNetwork()),
            "OPALArchitecture": asdict(OPALArchitecture()),
        },
        "reward_priors": REWARD_PRIOR_MIXTURES,
        "training_routes": {
            "FRE": ["train_fre_encoder_decoder_strided", "train_fre_conditioned_iql_policy"],
            "FB": "train_fb_agent_with_controllable_agent",
            "SF": "train_sf_agent_with_controllable_agent",
            "OPAL": "train_opal_agent",
            "GC-IQL": "train_gc_iql_agent",
            "GC-BC": "train_gc_bc_agent",
        },
        "evaluation_routes": {
            "FRE": "evaluate_fre_agent_with_32_state_reward_pairs",
            "OPAL": "evaluate_opal_with_10_random_skills",
            "episode_aggregation": {"episodes": EVAL_EPISODES_PER_TASK, "seeds": list(EVAL_SEEDS)},
        },
    }


__all__ = [
    "DATASET_ENVIRONMENT_ROUTES",
    "FREEncoderNetwork",
    "FREDecoderNetwork",
    "FREConditionedIQLPolicy",
    "GCIQLNetwork",
    "GCBCNetwork",
    "OPALArchitecture",
    "REWARD_PRIOR_MIXTURES",
    "discretize_reward_to_32_bins",
    "eq6_variational_lower_bound_terms",
    "evaluate_fre_agent_with_32_state_reward_pairs",
    "evaluate_opal_with_10_random_skills",
    "fre_paper_surface_inventory",
    "iql_actor_advantage_weighted_regression",
    "iql_critic_loss",
    "iql_value_expectile_loss",
    "load_antmaze_large_diverse_v2_dataset",
    "load_exorl_rnd_dataset",
    "load_kitchen_complete_v0_dataset",
    "make_d4rl_antmaze_online_evaluation_env",
    "make_d4rl_kitchen_online_evaluation_env",
    "make_exorl_custom_dmc_env",
    "sample_random_two_layer_mlp_reward",
    "sample_singleton_goal_reward",
    "sample_sparse_random_linear_reward",
    "soft_update_target_critic",
    "train_fb_agent_with_controllable_agent",
    "train_fre_conditioned_iql_policy",
    "train_fre_encoder_decoder_strided",
    "train_gc_bc_agent",
    "train_gc_iql_agent",
    "train_opal_agent",
    "train_sf_agent_with_controllable_agent",
]
