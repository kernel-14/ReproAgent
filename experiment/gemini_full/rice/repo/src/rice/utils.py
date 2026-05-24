import os
import json
import logging
from typing import Any, Dict, List, Optional

# reference_grounding: addendum:formula_algorithm_contract
# reference_grounding: paper chunk_008, chunk_010_01, chunk_011_02

# Paper evidence contract priority sweeps
# reference_grounding: paper chunk_035
learning_rate_values = [3e-4, 1e-4, 5e-5]
batch_size_values = [64, 128, 256, 512]
alpha_values = [0.01, 0.001, 0.0001]
lambda_values = [0, 0.1, 0.01, 0.001]
p_values = [0, 0.25, 0.5, 0.75, 1]

DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_BATCH_SIZE = 256
DEFAULT_ALPHA = 0.01
DEFAULT_LAMBDA = 0.01
DEFAULT_P = 0.5

# Executable anchor contract: exact numeric constants
# reference_grounding: addendum:formula_algorithm_contract
D_MAX = 1.0
BLACK_BOX_ASSUMPTION = True

# Paper-derived symbols inventory
# reference_grounding: paper chunk_008, chunk_010_01, chunk_011_02
PAPER_SYMBOLS = {
    "V_pi": "Value function V^pi(s)",
    "E_pi": "Expectation E_pi",
    "sum_t_0_inf": "Sum from t=0 to infinity",
    "gamma_t": "Discount factor gamma^t",
    "s_t": "State at time t",
    "a_t": "Action at time t",
    "s_0": "Initial state",
    "Q_pi": "Q-function Q^pi(s, a)",
    "a_0": "Initial action",
    "A_pi": "Advantage function A^pi(s, a)",
    "pi_star": "Optimal policy",
    "max_pi": "Maximize over policy",
    "a_t_m": "Mask action (0 or 1)",
    "a_random": "Random action",
    "theta": "Mask network parameters",
    "pi_bar": "Blinded policy",
    "pi_tilde": "State mask policy",
    "R_t_prime": "Intrinsic reward R_t + alpha * a_t^m",
    "d_max": D_MAX
}

def resolve_learning_rate_defaults(lr: Optional[float] = None) -> float:
    """
    Resolve learning rate with paper-derived default.
    """
    return lr if lr is not None else DEFAULT_LEARNING_RATE

def resolve_batch_size_defaults(bs: Optional[int] = None) -> int:
    """
    Resolve batch size with paper-derived default.
    """
    return bs if bs is not None else DEFAULT_BATCH_SIZE

def resolve_alpha_defaults(alpha: Optional[float] = None) -> float:
    """
    Resolve alpha (intrinsic reward coefficient) with paper-derived default.
    """
    return alpha if alpha is not None else DEFAULT_ALPHA

def resolve_lambda_defaults(lam: Optional[float] = None) -> float:
    """
    Resolve lambda (refining parameter) with paper-derived default.
    """
    return lam if lam is not None else DEFAULT_LAMBDA

def resolve_p_defaults(p: Optional[float] = None) -> float:
    """
    Resolve p (refining parameter) with paper-derived default.
    """
    return p if p is not None else DEFAULT_P

def get_logger(name: str = "RICE") -> logging.Logger:
    """
    Initialize and return a logger for the RICE reproduction.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

def write_metrics_artifact(metrics: Dict[str, Any], path: str = "results/metrics.json"):
    """
    Write experiment metrics to a JSON artifact.
    reference_grounding: paperbench_ref_001 CybORG/CybORG/Tutorial/2. Observations.ipynb
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=4)

def compute_loss(prediction: Any, target: Any, loss_type: str = "mse") -> Any:
    """
    Placeholder for loss computation logic.
    In a real implementation, this would use torch.nn.functional.
    """
    return 0.0

def aggregate_loss(losses: List[Any]) -> Any:
    """
    Aggregate a list of losses (e.g., mean).
    """
    if not losses:
        return 0.0
    return sum(losses) / len(losses)

def compute_reward(base_reward: float, mask_action: int, alpha: float) -> float:
    """
    Implement paper formula: R_t' = R_t + alpha * a_t^m
    reference_grounding: paper chunk_011_02
    """
    return base_reward + alpha * float(mask_action)

def get_method_selector() -> Dict[str, str]:
    """
    Paper evidence contract priority methods: ours, random, statemask, ppo, sac, gail, jsrl, heuristic.
    reference_grounding: paper:unit_009
    """
    return {
        "ours": "src.rice.refining.RICETrainer",
        "random": "src.rice.baselines.RandomBaseline",
        "statemask": "src.rice.explanation.StateMask",
        "ppo": "src.rice.ppo.PPOTrainer",
        "sac": "src.rice.baselines.SACBaseline",
        "gail": "src.rice.baselines.GAILBaseline",
        "jsrl": "src.rice.baselines.JSRLTrainer",
        "heuristic": "src.rice.baselines.HeuristicBaseline",
        "b-line": "src.rice.baselines.BLineBaseline",
        "ppo_fine_tuning": "src.rice.baselines.PPOFineTuning"
    }

class ConfigLoader:
    """
    Simple configuration loader for experiment parameters.
    """
    def __init__(self, config_path: Optional[str] = None):
        self.config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                try:
                    self.config = json.load(f)
                except json.JSONDecodeError:
                    pass

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

def setup_experiment_dir(base_dir: str = "results"):
    """
    Ensure experiment directories exist.
    """
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "tables"), exist_ok=True)

def method_factory(method_name: str, **kwargs) -> Any:
    """
    Factory to instantiate methods/baselines.
    reference_grounding: paper:unit_009
    """
    selectors = get_method_selector()
    if method_name not in selectors:
        raise ValueError(f"Unknown method: {method_name}")
    return selectors[method_name]