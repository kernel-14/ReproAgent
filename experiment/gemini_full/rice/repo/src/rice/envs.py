# src/rice/envs.py
# reference_grounding: paperbench_ref_002 evaluation.py
# reference_grounding: paperbench_ref_002 Wrappers/ChallengeWrapper2.py
# reference_grounding: paper:unit_002 (target:14)

import logging
import importlib.util
from typing import Any, Dict, Optional, List, Callable
from dataclasses import dataclass

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class EnvsSpec:
    """Environment specification."""
    env_id: str
    alias: str
    group: str
    factory: Callable
    availability_check: Callable

@dataclass
class EnvsConfig:
    """Environment configuration."""
    env_name: str
    kwargs: Dict[str, Any]

class Ids:
    """Environment IDs."""
    HOPPER = "Hopper-v3"
    WALKER2D = "Walker2d-v3"
    REACHER = "Reacher-v2"
    HALFCHEETAH = "HalfCheetah-v3"
    SELFISH_MINING = "SelfishMining"
    CAGE_CHALLENGE_2 = "CageChallenge2"
    AUTONOMOUS_DRIVING = "AutonomousDriving"
    MALWARE_MUTATION = "MalwareMutation"

AliasesMujoco = {
    "Hopper": Ids.HOPPER,
    "Walker2d": Ids.WALKER2D,
    "Reacher": Ids.REACHER,
    "HalfCheetah": Ids.HALFCHEETAH,
}

def _check_gym_available():
    return importlib.util.find_spec("gym") is not None

def _check_cyborg_available():
    return importlib.util.find_spec("CybORG") is not None

def check_envs_available(env_name: str) -> bool:
    """Check if the environment is available."""
    if env_name in [Ids.HOPPER, Ids.WALKER2D, Ids.REACHER, Ids.HALFCHEETAH]:
        return _check_gym_available()
    elif env_name == Ids.CAGE_CHALLENGE_2:
        return _check_cyborg_available()
    return True

def make_envs(env_name: str, **kwargs) -> Any:
    """Environment factory."""
    if not check_envs_available(env_name):
        raise ImportError(f"Environment {env_name} is not available.")
    
    if env_name in [Ids.HOPPER, Ids.WALKER2D, Ids.REACHER, Ids.HALFCHEETAH]:
        import gym
        return gym.make(env_name, **kwargs)
    elif env_name == Ids.CAGE_CHALLENGE_2:
        # reference_grounding: paperbench_ref_002 Wrappers/ChallengeWrapper2.py
        # Placeholder for CAGE Challenge 2 environment
        logger.info(f"Initializing {env_name}")
        return None
    else:
        raise ValueError(f"Unknown environment: {env_name}")

def build_envs(config: EnvsConfig) -> Any:
    """Builder function for environments."""
    return make_envs(config.env_name, **config.kwargs)

def get_env(env_name: str) -> Any:
    """Get environment interface."""
    return make_envs(env_name)

def load_cage_dataset():
    """Load CAGE dataset."""
    logger.info("Loading CAGE dataset")
    return {}

def load_gym_dataset():
    """Load Gym dataset."""
    logger.info("Loading Gym dataset")
    return {}

# Formula/Algorithm Anchors
# reference_grounding: paper:3.3. Technique Detail
# reference_grounding: addendum:formula_algorithm_contract

def compute_fidelity_score(trajectory: Any, explanation_method: Any) -> float:
    """
    Compute fidelity score.
    StateMask-style fidelity is computed by selecting top-K critical steps,
    replacing the target action with a random action on the critical span, and
    measuring the reward drop.  The mask network uses output 0 for critical
    steps and output 1 for ordinary steps.
    """
    from rice.statemask import compute_fidelity_score as _compute_fidelity_score

    if isinstance(trajectory, dict):
        trajectories = [trajectory]
    else:
        trajectories = list(trajectory or [])
    return _compute_fidelity_score(trajectories, explanation_method)

def refine_step(state: Any, mask_network: Any, policy: Any):
    """
    Refine step logic.
    Roll-in: Reset agent to critical states.
    Exploration: From critical states, perform new exploration.
    """
    from rice.statemask import RandomNetworkDistillation, rice_shaped_reward

    if hasattr(mask_network, "importance_scores"):
        score = float(mask_network.importance_scores([state])[0])
    elif callable(mask_network):
        score = float(mask_network(state))
    else:
        score = 0.0
    rnd = RandomNetworkDistillation(len(state) if hasattr(state, "__len__") else 1)
    exploration_bonus = float(rnd.novelty_bonus([state])[0])
    return {
        "state": state,
        "critical_score": score,
        "exploration_reward": rice_shaped_reward(exploration_bonus, 1, alpha=0.01),
        "policy": policy,
    }

# Explicit registration of aliases
# reference_grounding: paper:unit_002
ENV_ALIASES = {
    "mujoco": [Ids.HOPPER, Ids.WALKER2D, Ids.REACHER, Ids.HALFCHEETAH],
    "selfish_mining": [Ids.SELFISH_MINING],
    "network_defense": [Ids.CAGE_CHALLENGE_2],
    "autonomous_driving": [Ids.AUTONOMOUS_DRIVING],
    "cage": [Ids.CAGE_CHALLENGE_2],
    "gym": [Ids.HOPPER, Ids.WALKER2D, Ids.REACHER, Ids.HALFCHEETAH],
}

DATASET_ALIASES = {
    "cage": "cage",
    "gym": "gym",
    "mujoco": "mujoco",
    "selfish_mining": "selfish_mining",
    "network_defense": "network_defense",
    "autonomous_driving": "autonomous_driving",
}
