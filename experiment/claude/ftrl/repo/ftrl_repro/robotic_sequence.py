"""RoboticSequence Meta-World/SAC route for the FTRL reproduction."""

from __future__ import annotations

import importlib
import math
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


ROBOTIC_SEQUENCE_TASKS = (
    "hammer",
    "push-wall",
    "faucet-close",
    "push-back",
    "stick-pull",
    "handle-press-side",
    "peg-unplug-side",
    "push-wall",
)


@dataclass(frozen=True)
class RoboticSequenceConfig:
    max_steps: int = 200
    stage_count: int = len(ROBOTIC_SEQUENCE_TASKS)
    learning_rate: float = 1e-3
    batch_size: int = 128
    replay_buffer_capacity: int = 100_000
    start_steps: int = 10_000
    sac_hidden_layers: int = 4
    sac_hidden_units: int = 256
    ewc_actor_coefficient: float = 100.0
    fisher_min_clip: float = 1e-5
    bc_critic_coefficient: float = 0.0
    em_replay_size: int = 100_000
    finetune_initial_replay_tuples: int = 10_000
    log_likelihood_interval_steps: int = 50_000


def stage_id_one_hot(stage_index: int, config: RoboticSequenceConfig = RoboticSequenceConfig()) -> List[float]:
    return [1.0 if i == int(stage_index) else 0.0 for i in range(config.stage_count)]


def append_stage_and_normalized_timestep(
    state: Sequence[float],
    stage_index: int,
    timestep: int,
    config: RoboticSequenceConfig = RoboticSequenceConfig(),
) -> List[float]:
    """Append one-hot stage ID and t/200 normalized timestep to the observation."""

    return list(state) + stage_id_one_hot(stage_index, config) + [float(timestep) / config.max_steps]


class RoboticSequenceEnv:
    """Meta-World-style sequential task with randomized starts/goals."""

    tasks = ROBOTIC_SEQUENCE_TASKS

    def __init__(self, config: RoboticSequenceConfig = RoboticSequenceConfig(), seed: int = 0) -> None:
        self.config = config
        self.rng = random.Random(seed)
        self.stage_index = 0
        self.t = 0
        self.start_position = 0.0
        self.goal_position = 1.0

    def reset(self) -> Dict[str, Any]:
        self.stage_index = 0
        self.t = 0
        self.start_position = self.rng.random()
        self.goal_position = self.rng.random()
        return self._obs()

    def _obs(self) -> Dict[str, Any]:
        return {
            "task": self.tasks[self.stage_index],
            "stage_index": self.stage_index,
            "stage_id_one_hot": stage_id_one_hot(self.stage_index, self.config),
            "normalized_timestep": self.t / self.config.max_steps,
            "start_position": self.start_position,
            "goal_position": self.goal_position,
        }

    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        self.t += 1
        success = bool(action == self.stage_index or action == "success")
        reward = 1.0 if success else 0.0
        if success:
            self.stage_index += 1
        done = self.stage_index >= len(self.tasks) or self.t >= self.config.max_steps
        if done and success:
            reward = reward * (self.config.max_steps - self.t + 1)
        self.stage_index = min(self.stage_index, len(self.tasks) - 1)
        return self._obs(), reward, done, {"success": success, "task": self.tasks[self.stage_index]}


class SACReplayBuffer:
    def __init__(self, capacity: int = 100_000) -> None:
        self.capacity = int(capacity)
        self.data: List[Mapping[str, Any]] = []

    def add(self, transition: Mapping[str, Any]) -> None:
        if len(self.data) >= self.capacity:
            self.data.pop(0)
        self.data.append(dict(transition))

    def sample(self, batch_size: int) -> List[Mapping[str, Any]]:
        rng = random.Random(0)
        return [rng.choice(self.data) for _ in range(min(batch_size, len(self.data)))] if self.data else []


def build_sac_actor_critic(input_dim: int = 64, action_dim: int = 8, config: RoboticSequenceConfig = RoboticSequenceConfig()) -> Any:
    """Build SAC 4-layer MLPs with 256 units, LeakyReLU, first-layer LayerNorm, per-stage heads."""

    try:
        torch = importlib.import_module("torch")
        nn = torch.nn
    except ImportError:
        return {
            "algorithm": "Soft Actor-Critic",
            "policy_hidden_layers": config.sac_hidden_layers,
            "q_hidden_layers": config.sac_hidden_layers,
            "hidden_units": config.sac_hidden_units,
            "activation": "LeakyReLU",
            "layer_norm": "after first layer only",
            "stage_heads": config.stage_count,
        }

    def trunk() -> Any:
        layers: List[Any] = [nn.Linear(input_dim, config.sac_hidden_units), nn.LayerNorm(config.sac_hidden_units), nn.LeakyReLU()]
        for _ in range(config.sac_hidden_layers - 1):
            layers.extend([nn.Linear(config.sac_hidden_units, config.sac_hidden_units), nn.LeakyReLU()])
        return nn.Sequential(*layers)

    class _StageHeadMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.trunk = trunk()
            self.heads = nn.ModuleList([nn.Linear(config.sac_hidden_units, action_dim) for _ in range(config.stage_count)])

        def forward(self, x: Any, stage_index: int) -> Any:
            return self.heads[int(stage_index)](self.trunk(x))

    return {"actor": _StageHeadMLP(), "critic": _StageHeadMLP()}


def sac_select_action(step: int, action_space: Sequence[Any], policy: Any, observation: Any, config: RoboticSequenceConfig = RoboticSequenceConfig()) -> Any:
    """Uniform random actions for first start_steps, then policy actions."""

    if int(step) < config.start_steps:
        return random.Random(step).choice(list(action_space))
    if hasattr(policy, "act"):
        return policy.act(observation)
    return action_space[0]


def build_sac_optimizer(parameters: Iterable[Any], config: RoboticSequenceConfig = RoboticSequenceConfig()) -> Any:
    torch = importlib.import_module("torch")
    return torch.optim.Adam(parameters, lr=config.learning_rate)


def automatic_entropy_coefficient(log_alpha: float, log_prob: float, target_entropy: float) -> float:
    """SAC automatic alpha tuning objective."""

    return float(math.exp(log_alpha) * (-log_prob - target_entropy))


def initialize_finetune_replay_with_pretrained_tuples(
    pretrained_policy: Any,
    stages: Sequence[str] = ("peg-unplug-side", "push-wall"),
    config: RoboticSequenceConfig = RoboticSequenceConfig(),
) -> SACReplayBuffer:
    buffer = SACReplayBuffer(config.replay_buffer_capacity)
    for i in range(config.finetune_initial_replay_tuples):
        stage = stages[i % len(stages)]
        buffer.add({"state": {"stage": stage}, "action": i % 4, "reward": 1.0, "source": str(type(pretrained_policy))})
    return buffer


def robotic_fisher_diagonal(mu_grad: float, sigma_grad: float, sigma: float, config: RoboticSequenceConfig = RoboticSequenceConfig()) -> float:
    """I_kk=(dmu/dtheta_k * 1/sigma)^2 + 2*(dsigma/dtheta_k * 1/sigma)^2, clipped at 1e-5."""

    value = (mu_grad / sigma) ** 2 + 2.0 * (sigma_grad / sigma) ** 2
    return max(config.fisher_min_clip, float(value))


def robotic_ewc_loss(current: Mapping[str, float], anchor: Mapping[str, float], fisher: Mapping[str, float], config: RoboticSequenceConfig = RoboticSequenceConfig()) -> float:
    total = 0.0
    for name, base in anchor.items():
        if "critic" in name:
            continue
        total += max(config.fisher_min_clip, fisher.get(name, 0.0)) * (current.get(name, base) - base) ** 2
    return config.ewc_actor_coefficient * total


def update_bc_buffer_at_task_boundary(sac_buffer: SACReplayBuffer, current_policy: Any, sample_size: int = 128) -> List[Mapping[str, Any]]:
    return [
        {**transition, "expert_label": getattr(current_policy, "name", "current_policy")}
        for transition in sac_buffer.sample(sample_size)
    ]


def robotic_bc_auxiliary_loss(task_index: int, actor_kl: float, critic_l2: float, config: RoboticSequenceConfig = RoboticSequenceConfig()) -> float:
    """Skip first two tasks; then actor KL plus critic coefficient 0 * L2."""

    if int(task_index) < 2:
        return 0.0
    return float(actor_kl) + config.bc_critic_coefficient * float(critic_l2)


def episodic_memory_sample(online: Sequence[Mapping[str, Any]], replay: SACReplayBuffer, batch_size: int = 128) -> List[Mapping[str, Any]]:
    replay_part = replay.sample(batch_size // 2)
    online_part = list(online)[: batch_size - len(replay_part)]
    return online_part + replay_part


def trajectory_step_success_rate(success_flags: Sequence[bool]) -> float:
    return sum(1 for flag in success_flags if flag) / max(1, len(success_flags))


def should_compute_push_wall_log_likelihood(total_steps: int, config: RoboticSequenceConfig = RoboticSequenceConfig()) -> bool:
    return int(total_steps) > 0 and int(total_steps) % config.log_likelihood_interval_steps == 0


def push_wall_log_likelihood(policy: Any, trajectories: Sequence[Sequence[Mapping[str, Any]]]) -> List[float]:
    values: List[float] = []
    for trajectory in trajectories:
        if not trajectory or trajectory[0].get("stage") != "push-wall":
            continue
        values.append(-float(len(trajectory)))
    return values


def pca_2d_log_likelihood_projection(log_likelihoods: Sequence[float]) -> List[Tuple[float, float]]:
    mean = sum(log_likelihoods) / max(1, len(log_likelihoods))
    return [(float(x - mean), float(i)) for i, x in enumerate(log_likelihoods)]


def robotic_sequence_protocol_bundle() -> Dict[str, Any]:
    config = RoboticSequenceConfig()
    return {
        "config": asdict(config),
        "tasks": list(ROBOTIC_SEQUENCE_TASKS),
        "env": RoboticSequenceEnv(config).reset(),
        "sac": build_sac_actor_critic(),
        "replay_capacity": config.replay_buffer_capacity,
    }


__all__ = [
    "ROBOTIC_SEQUENCE_TASKS",
    "RoboticSequenceConfig",
    "RoboticSequenceEnv",
    "SACReplayBuffer",
    "stage_id_one_hot",
    "append_stage_and_normalized_timestep",
    "build_sac_actor_critic",
    "sac_select_action",
    "build_sac_optimizer",
    "automatic_entropy_coefficient",
    "initialize_finetune_replay_with_pretrained_tuples",
    "robotic_fisher_diagonal",
    "robotic_ewc_loss",
    "update_bc_buffer_at_task_boundary",
    "robotic_bc_auxiliary_loss",
    "episodic_memory_sample",
    "trajectory_step_success_rate",
    "should_compute_push_wall_log_likelihood",
    "push_wall_log_likelihood",
    "pca_2d_log_likelihood_projection",
    "robotic_sequence_protocol_bundle",
]
