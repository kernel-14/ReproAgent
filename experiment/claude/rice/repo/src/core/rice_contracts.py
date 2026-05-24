"""Paper-facing RICE implementation contracts.

This module keeps the core paper obligations in executable source rather than
only in prose.  It is intentionally lightweight: the actual environment
factories, PPO code, explanation code, and refinement code live in the other
modules, while this file exposes the exact contracts that bind them together.

The contracts cover Appendix C.2 environments, Section 3.3 explanation and
refinement objectives, Section 4.1 method choices, and the Section 4
experiment matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


MUJOCO_NON_SPARSE = ("hopper", "walker2d", "reacher", "halfcheetah")
MUJOCO_SPARSE_MAIN = ("hopper_sparse", "halfcheetah_sparse")
REAL_WORLD_ENVIRONMENTS = ("selfish_mining", "network_defense", "autonomous_driving")
EXPLANATION_ENV_GROUPS = ("mujoco", "selfish_mining", "network_defense", "autonomous_driving")
TOP_K_RATIOS = (0.10, 0.20, 0.30, 0.40)
THREE_SEED_PROTOCOL = (0, 1, 2)


@dataclass(frozen=True)
class EnvironmentContract:
    """Static and factory-facing contract for one paper environment."""

    key: str
    version: str
    family: str
    aliases: tuple[str, ...]
    can_initialize_for_experiments: bool
    observation_normalized_when_training: bool = False
    powered_by: str = ""
    source: str = ""
    sparse_variant: bool = False
    base_environment: str = ""


@dataclass(frozen=True)
class PolicyPretrainingContract:
    """PPO-compatible policy/pretraining contract for one environment family."""

    environment_key: str
    algorithm: str = "PPO"
    hidden_layers: tuple[int, ...] = (64, 64)
    pretrained_checkpoint_pattern: str = "checkpoints/{environment_key}_ppo_pretrained.pth"
    train_function: str = "train_ppo"
    supports_pretraining: bool = True


@dataclass(frozen=True)
class ExplanationMethodContract:
    """Contract for StateMask, RICE/Ours, and Random explanation methods."""

    method_key: str
    display_name: str
    train_selector: str
    rollout_selector: str
    environment_groups: tuple[str, ...]
    mask_output_critical_step: int | None
    mask_output_noncritical_step: int | None
    objective: str
    optimizer: str
    extra_reward_when_mask_outputs_one: bool = False
    mutable_hyperparameters: tuple[str, ...] = ()
    critical_step_source: str = ""


@dataclass(frozen=True)
class RefinementMethodContract:
    """Contract for Section 3.3 and Section 4.1 refinement methods."""

    method_key: str
    display_name: str
    environment_groups: tuple[str, ...]
    implementation_surface: str
    uses_mixed_initial_state_distribution: bool = False
    uses_rnd_exploration_bonus: bool = False
    mutable_hyperparameters: tuple[str, ...] = ()
    description: str = ""


@dataclass(frozen=True)
class ExperimentProtocolContract:
    """Protocol row for the RICE paper experiments."""

    experiment_id: str
    title: str
    environments: tuple[str, ...]
    methods: tuple[str, ...]
    metrics: tuple[str, ...]
    artifacts: tuple[str, ...]
    top_k_ratios: tuple[float, ...] = ()
    seeds: tuple[int, ...] = ()
    trajectory_count: int | None = None
    decision_value: str = ""


ENVIRONMENT_CONTRACTS: dict[str, EnvironmentContract] = {
    "hopper": EnvironmentContract(
        key="hopper",
        version="Hopper-v3",
        family="mujoco",
        aliases=("hopper", "hopper-v3", "Hopper-v3"),
        can_initialize_for_experiments=True,
    ),
    "walker2d": EnvironmentContract(
        key="walker2d",
        version="Walker2d-v3",
        family="mujoco",
        aliases=("walker2d", "walker2d-v3", "Walker2d-v3"),
        can_initialize_for_experiments=True,
        observation_normalized_when_training=True,
    ),
    "reacher": EnvironmentContract(
        key="reacher",
        version="Reacher-v2",
        family="mujoco",
        aliases=("reacher", "reacher-v2", "Reacher-v2"),
        can_initialize_for_experiments=True,
    ),
    "halfcheetah": EnvironmentContract(
        key="halfcheetah",
        version="HalfCheetah-v3",
        family="mujoco",
        aliases=("halfcheetah", "halfcheetah-v3", "HalfCheetah-v3"),
        can_initialize_for_experiments=True,
        observation_normalized_when_training=True,
    ),
    "hopper_sparse": EnvironmentContract(
        key="hopper_sparse",
        version="Hopper-v3",
        family="mujoco_sparse",
        aliases=("hopper_sparse", "hopper-sparse", "Sparse Hopper"),
        can_initialize_for_experiments=True,
        sparse_variant=True,
        base_environment="Hopper-v3",
    ),
    "halfcheetah_sparse": EnvironmentContract(
        key="halfcheetah_sparse",
        version="HalfCheetah-v3",
        family="mujoco_sparse",
        aliases=("halfcheetah_sparse", "halfcheetah-sparse", "Sparse HalfCheetah"),
        can_initialize_for_experiments=True,
        observation_normalized_when_training=True,
        sparse_variant=True,
        base_environment="HalfCheetah-v3",
    ),
    "selfish_mining": EnvironmentContract(
        key="selfish_mining",
        version="SelfishMining-v0",
        family="real_world",
        aliases=("selfish_mining", "selfish-mining", "bitcoin selfish mining"),
        can_initialize_for_experiments=True,
        source="blockchain selfish mining simulator",
    ),
    "network_defense": EnvironmentContract(
        key="network_defense",
        version="Cage Challenge 2",
        family="real_world",
        aliases=("network_defense", "network-defense", "CAGE Challenge 2", "CybORG"),
        can_initialize_for_experiments=True,
        source="CAGE Challenge 2 network defence simulator",
    ),
    "autonomous_driving": EnvironmentContract(
        key="autonomous_driving",
        version="Macro-v1",
        family="real_world",
        aliases=("autonomous_driving", "autonomous-driving", "Macro-v1"),
        can_initialize_for_experiments=True,
        powered_by="MetaDrive",
        source="MetaDrive simulator",
    ),
}


POLICY_PRETRAINING_CONTRACTS: dict[str, PolicyPretrainingContract] = {
    "mujoco": PolicyPretrainingContract(
        environment_key="mujoco",
        hidden_layers=(64, 64),
        pretrained_checkpoint_pattern="checkpoints/{environment_key}_ppo_pretrained.pth",
    ),
    "selfish_mining": PolicyPretrainingContract(
        environment_key="selfish_mining",
        hidden_layers=(128, 128, 128, 128),
        pretrained_checkpoint_pattern="checkpoints/selfish_mining_ppo_pretrained.pth",
    ),
    "network_defense": PolicyPretrainingContract(
        environment_key="network_defense",
        hidden_layers=(128, 128, 128, 128),
        pretrained_checkpoint_pattern="checkpoints/network_defense_ppo_pretrained.pth",
    ),
    "autonomous_driving": PolicyPretrainingContract(
        environment_key="autonomous_driving",
        hidden_layers=(256, 256),
        pretrained_checkpoint_pattern="checkpoints/autonomous_driving_ppo_pretrained.pth",
    ),
}


EXPLANATION_METHOD_CONTRACTS: dict[str, ExplanationMethodContract] = {
    "statemask": ExplanationMethodContract(
        method_key="statemask",
        display_name="Original StateMask",
        train_selector="statemask",
        rollout_selector="statemask_rollout",
        environment_groups=EXPLANATION_ENV_GROUPS,
        mask_output_critical_step=0,
        mask_output_noncritical_step=1,
        objective="J(theta)=min |eta(pi)-eta(bar_pi)|",
        optimizer="primal-dual / prime-dual",
        critical_step_source="trained StateMask mask network",
    ),
    "ours": ExplanationMethodContract(
        method_key="ours",
        display_name='Optimised StateMask "Ours" / RICE',
        train_selector="ours",
        rollout_selector="ours_rollout",
        environment_groups=EXPLANATION_ENV_GROUPS,
        mask_output_critical_step=0,
        mask_output_noncritical_step=1,
        objective="J(theta)=max eta(bar_pi)",
        optimizer="PPO",
        extra_reward_when_mask_outputs_one=True,
        mutable_hyperparameters=("alpha",),
        critical_step_source="PPO-optimized mask network",
    ),
    "random": ExplanationMethodContract(
        method_key="random",
        display_name="Random",
        train_selector="random",
        rollout_selector="random_rollout",
        environment_groups=EXPLANATION_ENV_GROUPS,
        mask_output_critical_step=None,
        mask_output_noncritical_step=None,
        objective="uniformly select previously visited states",
        optimizer="none",
        critical_step_source="randomly selected previously visited states",
    ),
}


REFINEMENT_METHOD_CONTRACTS: dict[str, RefinementMethodContract] = {
    "statemask_r": RefinementMethodContract(
        method_key="statemask_r",
        display_name="StateMask-R",
        environment_groups=EXPLANATION_ENV_GROUPS,
        implementation_surface="reset to identified critical states and continue training",
        description="StateMask fine-tuning baseline from Cheng et al. 2023.",
    ),
    "ours": RefinementMethodContract(
        method_key="ours",
        display_name="RICE refining",
        environment_groups=EXPLANATION_ENV_GROUPS,
        implementation_surface="Algorithm 2 mixed initial states plus RND exploration bonus",
        uses_mixed_initial_state_distribution=True,
        uses_rnd_exploration_bonus=True,
        mutable_hyperparameters=("lambda", "p"),
        description="Combines default initial states and critical states with probability threshold p.",
    ),
    "ppo_fine_tuning": RefinementMethodContract(
        method_key="ppo_fine_tuning",
        display_name="PPO fine-tuning",
        environment_groups=("mujoco", "selfish_mining", "network_defense", "autonomous_driving"),
        implementation_surface="lower learning rate and continue PPO training",
        mutable_hyperparameters=("learning_rate",),
    ),
    "jsrl": RefinementMethodContract(
        method_key="jsrl",
        display_name="Jump-Start Reinforcement Learning",
        environment_groups=("mujoco", "selfish_mining", "network_defense", "autonomous_driving"),
        implementation_surface="initialize exploration policy pi_e equal to guided policy pi_g",
    ),
}


EXPERIMENT_PROTOCOL_CONTRACTS: dict[str, ExperimentProtocolContract] = {
    "experiment_i": ExperimentProtocolContract(
        experiment_id="experiment_i",
        title="Fidelity and efficiency comparison",
        environments=(*MUJOCO_NON_SPARSE, *REAL_WORLD_ENVIRONMENTS),
        methods=("statemask", "ours"),
        metrics=("fidelity_average_reward_change", "fidelity_max_reward_change", "training_time_seconds"),
        artifacts=("results/figures/figure_5.png", "results/table4_training_time.json"),
        top_k_ratios=TOP_K_RATIOS,
        seeds=THREE_SEED_PROTOCOL,
        trajectory_count=500,
        decision_value="Does PPO-optimized StateMask preserve fidelity while reducing explanation training time?",
    ),
    "experiment_ii": ExperimentProtocolContract(
        experiment_id="experiment_ii",
        title="Refining performance comparison",
        environments=(*MUJOCO_NON_SPARSE, *REAL_WORLD_ENVIRONMENTS),
        methods=("ours", "statemask_r", "random", "ppo_fine_tuning", "jsrl"),
        metrics=("mean_episode_reward", "reward_improvement", "sample_efficiency"),
        artifacts=("results/table1_refining.json", "results/metrics.json"),
        decision_value="Does explanation-guided refinement outperform random and fine-tuning baselines?",
    ),
    "experiment_iii": ExperimentProtocolContract(
        experiment_id="experiment_iii",
        title="Real-world applications",
        environments=REAL_WORLD_ENVIRONMENTS,
        methods=("ours", "statemask", "random"),
        metrics=("mean_episode_reward", "success_rate"),
        artifacts=("results/tables/table_3.json",),
        decision_value="Does the contribution transfer to selfish mining, network defence, and autonomous driving?",
    ),
    "experiment_iv": ExperimentProtocolContract(
        experiment_id="experiment_iv",
        title="Sparse MuJoCo environments",
        environments=MUJOCO_SPARSE_MAIN,
        methods=("ours", "statemask_r", "random", "ppo_fine_tuning"),
        metrics=("mean_episode_reward", "reward_improvement"),
        artifacts=("results/tables/table_4.json",),
        decision_value="Does refinement help in sparse reward Hopper and HalfCheetah?",
    ),
    "experiment_v": ExperimentProtocolContract(
        experiment_id="experiment_v",
        title="Ablation and sensitivity",
        environments=("hopper", "walker2d"),
        methods=("ours",),
        metrics=("endpoint_low", "sweep_insensitive", "baseline_outperformance", "positive_parameter_improves"),
        artifacts=("results/ablation_studies.json", "results/sensitivity_report.json"),
        decision_value="Which alpha/lambda/p settings matter enough to guide later runs?",
    ),
}


def get_environment_contract(env_key: str) -> EnvironmentContract:
    """Return an environment contract by key, alias, or version name."""

    normalized = env_key.lower().replace("-", "_")
    if normalized in ENVIRONMENT_CONTRACTS:
        return ENVIRONMENT_CONTRACTS[normalized]
    for contract in ENVIRONMENT_CONTRACTS.values():
        aliases = {alias.lower().replace("-", "_") for alias in contract.aliases}
        aliases.add(contract.version.lower().replace("-", "_"))
        if normalized in aliases:
            return contract
    raise KeyError(f"Unknown RICE environment contract: {env_key}")


def observation_normalization_required(env_key: str) -> bool:
    """Whether observations must be normalized while training DRL agents."""

    return get_environment_contract(env_key).observation_normalized_when_training


def statemask_binary_mask(importance_scores: list[float], critical_threshold: float) -> list[int]:
    """Return StateMask mask values: 0 for critical steps and 1 otherwise."""

    return [0 if float(score) >= critical_threshold else 1 for score in importance_scores]


def original_statemask_objective(eta_pi: float, eta_bar_pi: float) -> float:
    """Original StateMask objective value J(theta)=min |eta(pi)-eta(bar_pi)|."""

    return abs(float(eta_pi) - float(eta_bar_pi))


def rice_ours_objective(eta_bar_pi: float) -> float:
    """RICE/Ours optimizes J(theta)=max eta(bar_pi)."""

    return float(eta_bar_pi)


def reward_with_alpha_bonus(task_reward: float, mask_output: int, alpha: float) -> float:
    """Add the mutable alpha reward bonus when the mask net outputs 1."""

    bonus = float(alpha) if int(mask_output) == 1 else 0.0
    return float(task_reward) + bonus


def primal_dual_statemask_update(
    eta_pi: float,
    eta_bar_pi: float,
    dual_value: float,
    dual_lr: float,
) -> dict[str, float]:
    """One prime/primal-dual bookkeeping step for original StateMask."""

    constraint_gap = float(eta_pi) - float(eta_bar_pi)
    next_dual = max(0.0, float(dual_value) + float(dual_lr) * constraint_gap)
    return {
        "objective_to_minimize": original_statemask_objective(eta_pi, eta_bar_pi),
        "constraint_gap": constraint_gap,
        "dual_value": next_dual,
    }


def ppo_mask_policy_objective(
    probability_ratio: float,
    advantage: float,
    clip_range: float = 0.2,
    entropy_bonus: float = 0.0,
) -> float:
    """Clipped PPO objective used by the optimized Ours mask policy."""

    ratio = float(probability_ratio)
    adv = float(advantage)
    clipped_ratio = min(max(ratio, 1.0 - float(clip_range)), 1.0 + float(clip_range))
    return min(ratio * adv, clipped_ratio * adv) + float(entropy_bonus)


def explanation_selector(method_key: str, *, for_rollout: bool = False) -> str:
    """Return the registered train or rollout selector for an explanation method."""

    method = EXPLANATION_METHOD_CONTRACTS[method_key]
    return method.rollout_selector if for_rollout else method.train_selector


def mixed_initial_state_distribution(
    default_initial_states: list[Any],
    critical_states: list[Any],
    p: float,
) -> Callable[[int], Any]:
    """Build Algorithm 2's mixed initial state sampler with mutable p."""

    if not default_initial_states:
        raise ValueError("default_initial_states must be non-empty")
    threshold = float(p)

    def sample(index: int) -> Any:
        use_critical = critical_states and ((index % 1000) / 1000.0) < threshold
        source = critical_states if use_critical else default_initial_states
        return source[index % len(source)]

    return sample


def rnd_exploration_bonus(prediction_error: float, lambda_: float) -> float:
    """Random Network Distillation exploration bonus scaled by mutable lambda."""

    return float(lambda_) * max(0.0, float(prediction_error))


def jsrl_policy_initialization(guided_policy: Any) -> dict[str, Any]:
    """JSRL contract: initialize exploration policy pi_e equal to guided policy pi_g."""

    return {"pi_g": guided_policy, "pi_e": guided_policy, "initialized_equal": True}


def contract_summary() -> dict[str, Any]:
    """Compact source-level evidence summary used by tests and review."""

    return {
        "environments": {key: vars(value) for key, value in ENVIRONMENT_CONTRACTS.items()},
        "pretraining": {key: vars(value) for key, value in POLICY_PRETRAINING_CONTRACTS.items()},
        "explanations": {key: vars(value) for key, value in EXPLANATION_METHOD_CONTRACTS.items()},
        "refinement": {key: vars(value) for key, value in REFINEMENT_METHOD_CONTRACTS.items()},
        "experiments": {key: vars(value) for key, value in EXPERIMENT_PROTOCOL_CONTRACTS.items()},
    }
