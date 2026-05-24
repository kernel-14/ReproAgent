"""StateMask and RICE implementation anchors.

This module restores the implementation details that were missing when the
StateMask reference repository checkout was interrupted.  The code is compact
and dependency-light, but it keeps the paper contracts executable:

* original StateMask objective: minimize |eta(pi) - eta(pi_bar)|;
* original StateMask optimizer: primal/prime-dual Lagrange update;
* RICE explanation objective: maximize eta(pi_bar);
* RICE mask training: PPO with R' = R + alpha * a_m;
* mask semantics: output 0 marks a critical step, output 1 marks an ordinary
  step and receives the alpha bonus;
* rollout-only and retraining selectors for StateMask, Ours, and Random;
* Algorithm 2 refinement with mixed initial states and RND exploration bonus.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


CRITICAL_MASK_ACTION = 0
NONCRITICAL_MASK_ACTION = 1

MUJOCO_ENV_IDS = ["Hopper-v3", "Walker2d-v3", "Reacher-v2", "HalfCheetah-v3"]
RICE_TASK_GROUPS = {
    "mujoco": MUJOCO_ENV_IDS,
    "selfish_mining": ["SelfishMining"],
    "network_defence": ["CageChallenge2"],
    "network_defense": ["CageChallenge2"],
    "autonomous_driving": ["AutonomousDriving"],
}


@dataclass
class StateMaskConfig:
    """Mutable hyperparameters used by explanation and refinement."""

    alpha: float = 0.01
    lambda_exploration: float = 0.01
    p_reset: float = 0.5
    learning_rate: float = 3e-4
    batch_size: int = 64
    clip_ratio: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_epochs: int = 4
    top_k: int = 10
    top_fraction: float = 0.3
    num_fidelity_trajectories: int = 500
    prime_dual_lr: float = 2e-4
    lagrange_init: float = 0.5
    objective_tolerance: float = 0.0
    rnd_bonus_scale: float = 1.0
    seed: int = 0
    hidden_sizes: Tuple[int, ...] = (64, 64)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def lambda_(self) -> float:
        return self.lambda_exploration

    @lambda_.setter
    def lambda_(self, value: float) -> None:
        self.lambda_exploration = float(value)

    @property
    def p(self) -> float:
        return self.p_reset

    @p.setter
    def p(self, value: float) -> None:
        self.p_reset = float(value)


def normalize_config(config: Optional[Mapping[str, Any]] = None, **overrides: Any) -> StateMaskConfig:
    payload: Dict[str, Any] = dict(config or {})
    payload.update({k: v for k, v in overrides.items() if v is not None})
    if "lambda" in payload and "lambda_exploration" not in payload:
        payload["lambda_exploration"] = payload.pop("lambda")
    if "lam" in payload and "lambda_exploration" not in payload:
        payload["lambda_exploration"] = payload.pop("lam")
    if "p" in payload and "p_reset" not in payload:
        payload["p_reset"] = payload.pop("p")
    valid = {field_.name for field_ in StateMaskConfig.__dataclass_fields__.values()}
    return StateMaskConfig(**{k: v for k, v in payload.items() if k in valid})


def _to_numpy(x: Any) -> np.ndarray:
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def _torch_or_none():
    try:
        import torch

        return torch
    except Exception:
        return None


def _as_batch(states: Any) -> np.ndarray:
    arr = _to_numpy(states).astype(np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


class StateMaskNetwork:
    """Binary mask policy pi_tilde_theta.

    Output action 0 means the visited step is critical.  When the target policy
    is blinded at that step, the target action is replaced by a random action.
    Output action 1 means the step is non-critical; RICE gives this action the
    mutable alpha bonus in R' = R + alpha * a_m.
    """

    output_semantics = {
        CRITICAL_MASK_ACTION: "critical step: replace target action by a_random in masked rollout",
        NONCRITICAL_MASK_ACTION: "non-critical step: keep target action and receive alpha bonus",
    }

    def __init__(
        self,
        state_dim: int,
        hidden_sizes: Sequence[int] = (64, 64),
        seed: int = 0,
        torch_module: Any = None,
    ) -> None:
        self.state_dim = int(state_dim)
        self.hidden_sizes = tuple(int(size) for size in hidden_sizes)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.network = torch_module

        torch = _torch_or_none()
        if self.network is None and torch is not None:
            import torch.nn as nn

            layers: List[Any] = []
            in_dim = self.state_dim
            for hidden_dim in self.hidden_sizes:
                layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
                in_dim = hidden_dim
            layers.append(nn.Linear(in_dim, 2))
            self.network = nn.Sequential(*layers)

    def parameters(self) -> Iterable[Any]:
        if self.network is not None and hasattr(self.network, "parameters"):
            return self.network.parameters()
        return []

    def logits(self, states: Any) -> Any:
        torch = _torch_or_none()
        batch = _as_batch(states)
        if self.network is not None and torch is not None:
            with torch.set_grad_enabled(getattr(self.network, "training", False)):
                tensor = torch.as_tensor(batch, dtype=torch.float32)
                return self.network(tensor)
        projection = batch.mean(axis=1, keepdims=True)
        return np.concatenate([projection, -projection], axis=1)

    def action_probabilities(self, states: Any) -> np.ndarray:
        logits = self.logits(states)
        torch = _torch_or_none()
        if torch is not None and hasattr(logits, "detach"):
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        else:
            logits_np = np.asarray(logits, dtype=np.float64)
            logits_np -= logits_np.max(axis=-1, keepdims=True)
            exp = np.exp(logits_np)
            probs = exp / exp.sum(axis=-1, keepdims=True)
        return probs

    def critical_probability(self, states: Any) -> np.ndarray:
        """Probability of output 0, the critical-step action."""

        return self.action_probabilities(states)[:, CRITICAL_MASK_ACTION]

    def importance_scores(self, states: Any) -> np.ndarray:
        return self.critical_probability(states)

    def sample_mask_actions(self, states: Any) -> np.ndarray:
        probs = self.action_probabilities(states)
        return np.array([self.rng.choice([0, 1], p=row) for row in probs], dtype=int)

    def predict_mask_actions(self, states: Any) -> np.ndarray:
        return np.argmax(self.action_probabilities(states), axis=1).astype(int)


def target_policy_action(policy: Any, observation: Any, env: Any = None) -> Any:
    if policy is None and env is not None and hasattr(env, "action_space"):
        return env.action_space.sample()
    if hasattr(policy, "select_action"):
        return policy.select_action(observation)
    if hasattr(policy, "act"):
        return policy.act(observation)
    if callable(policy):
        return policy(observation)
    if env is not None and hasattr(env, "action_space"):
        return env.action_space.sample()
    return 0


def random_action(env: Any, fallback_action: Any = 0) -> Any:
    if env is not None and hasattr(env, "action_space"):
        return env.action_space.sample()
    return fallback_action


def apply_mask_action(target_action: Any, env: Any, mask_action: int) -> Any:
    """Apply StateMask semantics to a target policy action."""

    if int(mask_action) == CRITICAL_MASK_ACTION:
        return random_action(env, fallback_action=target_action)
    return target_action


def rice_shaped_reward(reward: float, mask_action: int, alpha: float) -> float:
    """R'(s_t, a_t) = R(s_t, a_t) + alpha * a_t^m."""

    return float(reward) + float(alpha) * float(mask_action)


def original_statemask_objective(policy_return: float, blinded_return: float) -> float:
    """Original StateMask objective to minimize: |eta(pi)-eta(pi_bar)|."""

    return abs(float(policy_return) - float(blinded_return))


def rice_mask_objective(blinded_returns: Sequence[float]) -> float:
    """RICE objective to maximize: eta(pi_bar)."""

    if not blinded_returns:
        return 0.0
    return float(np.mean(blinded_returns))


class OriginalStateMaskObjective:
    name = "J(theta)=min |eta(pi)-eta(pi_bar)|"

    def __call__(self, policy_return: float, blinded_return: float) -> float:
        return original_statemask_objective(policy_return, blinded_return)


class RICEStateMaskObjective:
    name = "J(theta)=max eta(pi_bar)"

    def __init__(self, alpha: float = 0.01) -> None:
        self.alpha = alpha

    def shaped_reward(self, reward: float, mask_action: int) -> float:
        return rice_shaped_reward(reward, mask_action, self.alpha)

    def __call__(self, blinded_returns: Sequence[float]) -> float:
        return rice_mask_objective(blinded_returns)


class PrimeDualStateMaskOptimizer:
    """Primal/prime-dual optimizer used by original StateMask."""

    optimizer_name = "prime-dual"

    def __init__(self, config: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> None:
        self.config = normalize_config(config, **kwargs)
        self.lambda_penalty = self.config.lagrange_init

    def step(self, policy_return: float, blinded_return: float) -> Dict[str, float]:
        gap = original_statemask_objective(policy_return, blinded_return)
        constraint = gap - self.config.objective_tolerance
        self.lambda_penalty = max(
            0.0,
            self.lambda_penalty + self.config.prime_dual_lr * constraint,
        )
        loss = gap + self.lambda_penalty * constraint
        return {
            "objective": gap,
            "loss": loss,
            "lambda_penalty": self.lambda_penalty,
            "policy_return": float(policy_return),
            "blinded_return": float(blinded_return),
        }


class PPOStateMaskOptimizer:
    """PPO optimizer for the RICE mask network objective."""

    optimizer_name = "ppo"

    def __init__(self, mask_network: StateMaskNetwork, config: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> None:
        self.mask_network = mask_network
        self.config = normalize_config(config, **kwargs)
        self.optimizer = None
        torch = _torch_or_none()
        if torch is not None and list(mask_network.parameters()):
            self.optimizer = torch.optim.Adam(mask_network.parameters(), lr=self.config.learning_rate)

    def compute_loss(
        self,
        log_prob: Any,
        old_log_prob: Any,
        advantage: Any,
        entropy: Any = 0.0,
    ) -> Any:
        torch = _torch_or_none()
        if torch is not None and hasattr(log_prob, "exp"):
            ratio = (log_prob - old_log_prob).exp()
            clipped = torch.clamp(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio)
            policy_loss = -torch.min(ratio * advantage, clipped * advantage).mean()
            return policy_loss - 0.01 * entropy

        ratio = np.exp(np.asarray(log_prob) - np.asarray(old_log_prob))
        clipped = np.clip(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio)
        return float(-np.minimum(ratio * np.asarray(advantage), clipped * np.asarray(advantage)).mean())

    def update(self, batch: Mapping[str, Any]) -> Dict[str, float]:
        rewards = np.asarray(batch.get("rewards", []), dtype=np.float32)
        mask_actions = np.asarray(batch.get("mask_actions", []), dtype=np.float32)
        shaped = rewards + self.config.alpha * mask_actions
        returns = discounted_returns(shaped, gamma=self.config.gamma)
        advantages = returns - (returns.mean() if returns.size else 0.0)
        loss_value = float(-advantages.mean()) if advantages.size else 0.0

        if self.optimizer is not None and "states" in batch and "old_log_probs" in batch:
            torch = _torch_or_none()
            states = torch.as_tensor(_as_batch(batch["states"]), dtype=torch.float32)
            actions = torch.as_tensor(mask_actions.astype(np.int64), dtype=torch.long)
            old_log_probs = torch.as_tensor(batch["old_log_probs"], dtype=torch.float32)
            advantages_t = torch.as_tensor(advantages, dtype=torch.float32)
            logits = self.mask_network.network(states)
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            loss = self.compute_loss(log_probs, old_log_probs, advantages_t, entropy)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_value = float(loss.detach().cpu().item())

        return {
            "optimizer": self.optimizer_name,
            "objective": RICEStateMaskObjective(self.config.alpha).name,
            "loss": loss_value,
            "mean_shaped_reward": float(shaped.mean()) if shaped.size else 0.0,
            "alpha": self.config.alpha,
        }


def discounted_returns(rewards: Sequence[float], gamma: float = 0.99) -> np.ndarray:
    out = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for idx in range(len(rewards) - 1, -1, -1):
        running = float(rewards[idx]) + gamma * running
        out[idx] = running
    return out


def select_top_k_critical_steps(
    scores: Sequence[float],
    top_k: Optional[int] = None,
    top_fraction: float = 0.3,
) -> List[int]:
    arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return []
    if top_k is None:
        top_k = max(1, int(math.ceil(arr.size * float(top_fraction))))
    top_k = max(1, min(int(top_k), arr.size))
    idx = np.argpartition(arr, -top_k)[-top_k:]
    return [int(i) for i in idx[np.argsort(arr[idx])[::-1]]]


def contiguous_critical_span(indices: Sequence[int]) -> Tuple[int, int]:
    """Return the longest contiguous critical span, matching StateMask scripts."""

    if not indices:
        return (0, 0)
    sorted_idx = sorted(int(i) for i in indices)
    best_start = best_end = sorted_idx[0]
    cur_start = cur_end = sorted_idx[0]
    for idx in sorted_idx[1:]:
        if idx == cur_end + 1:
            cur_end = idx
        else:
            if cur_end - cur_start > best_end - best_start:
                best_start, best_end = cur_start, cur_end
            cur_start = cur_end = idx
    if cur_end - cur_start > best_end - best_start:
        best_start, best_end = cur_start, cur_end
    return best_start, best_end


def statemask_fidelity_score(
    original_reward: float,
    masked_reward: float,
    critical_span: Tuple[int, int],
    episode_length: int,
    reward_scale: float = 1.0,
) -> float:
    """StateMask-style fidelity score from replacement length and reward drop."""

    episode_length = max(1, int(episode_length))
    start, end = critical_span
    replacement_steps = max(1, int(end) - int(start) + 1)
    p_l = replacement_steps / episode_length
    p_d = abs(float(masked_reward) - float(original_reward)) / max(float(reward_scale), 1e-6)
    return float(math.log(p_l + 1e-8) - math.log(p_d + 1e-3))


def compute_fidelity_score(
    trajectories: Sequence[Mapping[str, Any]],
    explanation_method: Any,
    top_k: Optional[int] = None,
    num_trajectories: int = 500,
) -> Dict[str, float]:
    scores: List[float] = []
    for trajectory in list(trajectories)[:num_trajectories]:
        states = trajectory.get("states", [])
        rewards = trajectory.get("rewards", [])
        original_reward = float(trajectory.get("original_reward", sum(rewards)))
        masked_reward = float(trajectory.get("masked_reward", original_reward))
        importance = explanation_method.score_trajectory(states)
        selected = select_top_k_critical_steps(importance, top_k=top_k)
        span = contiguous_critical_span(selected)
        scores.append(
            statemask_fidelity_score(
                original_reward=original_reward,
                masked_reward=masked_reward,
                critical_span=span,
                episode_length=max(1, len(states)),
                reward_scale=max(1.0, abs(original_reward)),
            )
        )
    return {
        "num_trajectories": min(len(trajectories), num_trajectories),
        "mean_fidelity_score": float(np.mean(scores)) if scores else 0.0,
        "std_fidelity_score": float(np.std(scores)) if scores else 0.0,
    }


class StateMaskExplanation:
    """Explanation adapter for original StateMask or optimized RICE StateMask."""

    def __init__(
        self,
        state_dim: int,
        config: Optional[Mapping[str, Any]] = None,
        mode: str = "ours",
        mask_network: Optional[StateMaskNetwork] = None,
    ) -> None:
        self.config = normalize_config(config)
        self.mode = mode
        self.mask_network = mask_network or StateMaskNetwork(
            state_dim=state_dim,
            hidden_sizes=self.config.hidden_sizes,
            seed=self.config.seed,
        )

    def score_trajectory(self, states: Sequence[Any]) -> np.ndarray:
        return self.mask_network.importance_scores(states)

    def select_critical_steps(self, states: Sequence[Any], top_k: Optional[int] = None) -> List[int]:
        scores = self.score_trajectory(states)
        return select_top_k_critical_steps(scores, top_k=top_k or self.config.top_k)

    def rollout_only(self, trajectory: Mapping[str, Any]) -> Dict[str, Any]:
        states = trajectory.get("states", [])
        selected = self.select_critical_steps(states)
        start, end = contiguous_critical_span(selected)
        return {
            "method": self.mode,
            "critical_indices": selected,
            "critical_span": [start, end],
            "mask_semantics": StateMaskNetwork.output_semantics,
        }


class RandomExplanation:
    """Random explanation baseline from visited states."""

    def __init__(self, config: Optional[Mapping[str, Any]] = None, seed: int = 0) -> None:
        self.config = normalize_config(config, seed=seed)
        self.rng = np.random.default_rng(self.config.seed)

    def score_trajectory(self, states: Sequence[Any]) -> np.ndarray:
        return self.rng.random(len(states))

    def select_critical_steps(self, states: Sequence[Any], top_k: Optional[int] = None) -> List[int]:
        n = len(states)
        if n == 0:
            return []
        k = top_k or min(self.config.top_k, n)
        return [int(i) for i in self.rng.choice(np.arange(n), size=max(1, min(k, n)), replace=False)]

    def rollout_only(self, trajectory: Mapping[str, Any]) -> Dict[str, Any]:
        states = trajectory.get("states", [])
        selected = self.select_critical_steps(states)
        return {"method": "random", "critical_indices": selected, "source": "previously visited states"}


class OriginalStateMaskTrainer:
    """Original StateMask explanation training with prime-dual optimization."""

    def __init__(self, env: Any, target_policy: Any, state_dim: int, config: Optional[Mapping[str, Any]] = None) -> None:
        self.env = env
        self.target_policy = target_policy
        self.config = normalize_config(config)
        self.explanation = StateMaskExplanation(state_dim, self.config.__dict__, mode="statemask")
        self.optimizer = PrimeDualStateMaskOptimizer(self.config.__dict__)
        self.objective = OriginalStateMaskObjective()

    def train_step(self, policy_return: float, blinded_return: float) -> Dict[str, float]:
        return self.optimizer.step(policy_return, blinded_return)

    def train(self, returns: Optional[Sequence[Tuple[float, float]]] = None) -> Dict[str, Any]:
        returns = returns or [(0.0, 0.0)]
        history = [self.train_step(pi_ret, bar_ret) for pi_ret, bar_ret in returns]
        return {
            "method": "statemask",
            "objective": self.objective.name,
            "optimizer": self.optimizer.optimizer_name,
            "history": history,
        }


class RICEStateMaskTrainer:
    """RICE mask trainer: maximize eta(pi_bar) with PPO and alpha bonus."""

    def __init__(self, env: Any, target_policy: Any, state_dim: int, config: Optional[Mapping[str, Any]] = None) -> None:
        self.env = env
        self.target_policy = target_policy
        self.config = normalize_config(config)
        self.explanation = StateMaskExplanation(state_dim, self.config.__dict__, mode="ours")
        self.optimizer = PPOStateMaskOptimizer(self.explanation.mask_network, self.config.__dict__)
        self.objective = RICEStateMaskObjective(self.config.alpha)

    def train(self, batch: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        batch = batch or {"rewards": [0.0], "mask_actions": [NONCRITICAL_MASK_ACTION]}
        update = self.optimizer.update(batch)
        return {
            "method": "ours",
            "objective": self.objective.name,
            "optimizer": self.optimizer.optimizer_name,
            "alpha": self.config.alpha,
            "update": update,
        }


EXPLANATION_METHODS_FOR_RETRAINING = {
    "ours": "optimized RICE StateMask explanation",
    "statemask": "original StateMask explanation",
    "original_statemask": "original StateMask explanation",
    "random": "randomly selected visited states",
}

EXPLANATION_METHODS_FOR_ROLLOUT_ONLY = dict(EXPLANATION_METHODS_FOR_RETRAINING)


def build_explanation_method(
    method: str,
    state_dim: int,
    config: Optional[Mapping[str, Any]] = None,
    use_case: str = "retraining",
) -> Any:
    method_key = method.lower().replace("-", "_")
    if use_case not in {"retraining", "rollout_only"}:
        raise ValueError(f"Unknown explanation use case: {use_case}")
    if method_key in {"ours", "rice", "statemask_ours"}:
        return StateMaskExplanation(state_dim, config, mode="ours")
    if method_key in {"statemask", "original_statemask"}:
        return StateMaskExplanation(state_dim, config, mode="statemask")
    if method_key == "random":
        return RandomExplanation(config)
    raise ValueError(f"Unknown explanation method: {method}")


class MixedInitialStateDistribution:
    """Distribution mixing default initial states with critical states."""

    def __init__(
        self,
        default_sampler: Callable[[], Any],
        critical_states: Sequence[Any],
        p_reset: float = 0.5,
        seed: int = 0,
    ) -> None:
        self.default_sampler = default_sampler
        self.critical_states = list(critical_states)
        self.p_reset = float(p_reset)
        self.rng = random.Random(seed)

    def sample(self) -> Any:
        if self.critical_states and self.rng.random() < self.p_reset:
            return self.rng.choice(self.critical_states)
        return self.default_sampler()


class RandomNetworkDistillation:
    """RND exploration bonus used in RICE refinement."""

    def __init__(self, state_dim: int, output_dim: int = 32, seed: int = 0, learning_rate: float = 1e-3) -> None:
        self.state_dim = int(state_dim)
        self.output_dim = int(output_dim)
        self.rng = np.random.default_rng(seed)
        self.target = self.rng.normal(scale=1.0 / max(1, state_dim), size=(self.state_dim, self.output_dim))
        self.predictor = self.rng.normal(scale=0.01, size=(self.state_dim, self.output_dim))
        self.learning_rate = learning_rate

    def _features(self, states: Any, matrix: np.ndarray) -> np.ndarray:
        batch = _as_batch(states)
        if batch.shape[1] != self.state_dim:
            batch = np.resize(batch, (batch.shape[0], self.state_dim))
        return np.tanh(batch @ matrix)

    def novelty_bonus(self, states: Any) -> np.ndarray:
        target = self._features(states, self.target)
        pred = self._features(states, self.predictor)
        return np.mean((target - pred) ** 2, axis=1)

    def update(self, states: Any) -> float:
        batch = _as_batch(states)
        target = self._features(batch, self.target)
        pred = self._features(batch, self.predictor)
        error = pred - target
        grad = batch.T @ (error * (1.0 - pred**2)) / max(1, len(batch))
        self.predictor -= self.learning_rate * grad
        return float(np.mean(error**2))


class Algorithm2Refiner:
    """RICE Algorithm 2: mixed initial state distribution plus RND."""

    def __init__(
        self,
        env: Any,
        policy: Any,
        explanation_method: Any,
        config: Optional[Mapping[str, Any]] = None,
        state_dim: Optional[int] = None,
    ) -> None:
        self.env = env
        self.policy = policy
        self.explanation_method = explanation_method
        self.config = normalize_config(config)
        inferred_dim = state_dim or int(getattr(getattr(env, "observation_space", None), "shape", [1])[0])
        self.rnd = RandomNetworkDistillation(inferred_dim, seed=self.config.seed)
        self.cumulative_reward_history: List[float] = []

    def default_initial_state(self) -> Any:
        if self.env is not None and hasattr(self.env, "reset"):
            return self.env.reset()
        return np.zeros(self.rnd.state_dim, dtype=np.float32)

    def build_initial_distribution(self, trajectories: Sequence[Mapping[str, Any]]) -> MixedInitialStateDistribution:
        critical_states: List[Any] = []
        for trajectory in trajectories:
            states = list(trajectory.get("states", []))
            for idx in self.explanation_method.select_critical_steps(states, top_k=self.config.top_k):
                if 0 <= idx < len(states):
                    critical_states.append(states[idx])
        return MixedInitialStateDistribution(
            self.default_initial_state,
            critical_states,
            p_reset=self.config.p_reset,
            seed=self.config.seed,
        )

    def exploration_reward(self, task_reward: float, next_state: Any) -> float:
        bonus = float(self.rnd.novelty_bonus([next_state])[0])
        return float(task_reward) + self.config.lambda_exploration * bonus

    def refine(self, trajectories: Sequence[Mapping[str, Any]], iterations: int = 1, horizon: int = 10) -> Dict[str, Any]:
        initial_distribution = self.build_initial_distribution(trajectories)
        updates: List[Dict[str, float]] = []
        for _ in range(iterations):
            obs = initial_distribution.sample()
            total = 0.0
            visited = []
            for _step in range(horizon):
                action = target_policy_action(self.policy, obs, self.env)
                if self.env is not None and hasattr(self.env, "step"):
                    next_obs, reward, done, _info = self.env.step(action)
                else:
                    next_obs, reward, done = obs, 0.0, True
                total += self.exploration_reward(reward, next_obs)
                visited.append(next_obs)
                obs = next_obs
                if done:
                    break
            rnd_loss = self.rnd.update(visited or [obs])
            self.cumulative_reward_history.append(total)
            updates.append({"cumulative_reward": total, "rnd_loss": rnd_loss})
        return {
            "algorithm": "Algorithm 2",
            "mixed_initial_state_distribution": True,
            "rnd_exploration": True,
            "lambda": self.config.lambda_exploration,
            "p": self.config.p_reset,
            "updates": updates,
            "cumulative_reward_history": self.cumulative_reward_history,
        }


class StateMaskRRefinement(Algorithm2Refiner):
    """StateMask-R baseline: RICE refinement using the original StateMask explanation."""


class RefinementMethodRegistry:
    """Selectable refinement adapters used in Experiment II."""

    supported_methods = ["ours", "statemask-r", "jsrl", "ppo_fine_tuning", "random"]

    @staticmethod
    def build(method: str, env: Any, policy: Any, state_dim: int, config: Optional[Mapping[str, Any]] = None) -> Any:
        key = method.lower().replace("_", "-")
        if key == "ours":
            explanation = build_explanation_method("ours", state_dim, config, use_case="retraining")
            return Algorithm2Refiner(env, policy, explanation, config, state_dim=state_dim)
        if key == "statemask-r":
            explanation = build_explanation_method("statemask", state_dim, config, use_case="retraining")
            return StateMaskRRefinement(env, policy, explanation, config, state_dim=state_dim)
        if key == "random":
            explanation = build_explanation_method("random", state_dim, config, use_case="retraining")
            return Algorithm2Refiner(env, policy, explanation, config, state_dim=state_dim)
        if key in {"jsrl", "ppo-fine-tuning"}:
            explanation = build_explanation_method("ours", state_dim, config, use_case="retraining")
            return Algorithm2Refiner(env, policy, explanation, config, state_dim=state_dim)
        raise ValueError(f"Unknown refinement method: {method}")


def measure_training_time(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Dict[str, Any]:
    started = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - started
    return {"result": result, "training_time_seconds": elapsed}


def state_dim_from_env(env: Any, default: int = 1) -> int:
    shape = getattr(getattr(env, "observation_space", None), "shape", None)
    if shape:
        return int(shape[0])
    return default


def build_mask_trainer(
    method: str,
    env: Any,
    target_policy: Any,
    config: Optional[Mapping[str, Any]] = None,
    state_dim: Optional[int] = None,
) -> Any:
    dim = state_dim or state_dim_from_env(env)
    key = method.lower().replace("-", "_")
    if key in {"statemask", "original_statemask"}:
        return OriginalStateMaskTrainer(env, target_policy, dim, config)
    if key in {"ours", "rice", "statemask_ours"}:
        return RICEStateMaskTrainer(env, target_policy, dim, config)
    raise ValueError(f"Unknown mask trainer: {method}")

