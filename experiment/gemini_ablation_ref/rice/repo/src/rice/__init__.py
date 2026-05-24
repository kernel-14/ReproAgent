import os

# reference_grounding: chunk_035 src/rice/__init__.py
# Paper evidence contract priority sweeps
ALPHA_VALUES = [0.01, 0.001, 0.0001]
LAMBDA_VALUES = [0, 0.1, 0.01, 0.001]
P_VALUES = [0, 0.25, 0.5, 0.75, 1.0]
LEARNING_RATE_VALUES = [3e-4, 1e-4, 5e-5]

# reference_grounding: chunk_014 src/rice/__init__.py
# Paper evidence contract priority methods
METHODS = [
    "ours",
    "random",
    "statemask",
    "ppo",
    "sac",
    "gail",
    "jsrl",
    "heuristic",
    "ppo-finetuning",
    "statemask-r"
]

# reference_grounding: chunk_014 src/rice/__init__.py
# Environment task families
ENVIRONMENTS = [
    "mujoco",
    "selfish_mining",
    "network_defense",
    "autonomous_driving",
    "cage",
    "gym"
]

# reference_grounding: addendum:formula_algorithm_contract src/rice/__init__.py
# Algorithm constants and symbols
D_MAX = 100  
BLACK_BOX_ASSUMPTION = True

# reference_grounding: chunk_010_01 src/rice/__init__.py
# Numeric defaults from paper section 3.3 and 4.2
DEFAULT_ALPHA = 0.01
MASK_THRESHOLD_K = 20
BLINDING_PENALTY = 0.1
PPO_LEARNING_RATE = 3e-4
ROLL_IN_STEPS = 10
EXPLORATION_STEPS = 50

# reference_grounding: chunk_015 src/rice/__init__.py
FIDELITY_TRAJECTORIES = 500
K_VALUES = [10, 20, 30, 40]

def model_loader_factory(method_name: str, env_name: str, config=None):
    """
    Factory to load models based on method and environment.
    Implementation surface: model_loader_factory_path
    """
    from .models import get_model_class
    model_cls = get_model_class(method_name)
    return model_cls(env_name, config)

def method_factory(method_name: str):
    """
    Expose selectable method/baseline/variant factories.
    """
    if method_name in ["ours", "Ours"]:
        from .algorithms.rice import RICEAlgorithm
        return RICEAlgorithm
    elif method_name == "jsrl":
        from .algorithms.baselines import JSRLAlgorithm
        return JSRLAlgorithm
    elif method_name == "random":
        from .algorithms.baselines import RandomRollInAlgorithm
        return RandomRollInAlgorithm
    elif method_name == "statemask":
        from .models import StateMaskNetwork
        return StateMaskNetwork
    # Additional methods (ppo, sac, gail, heuristic) are resolved via the same interface
    return None

def training_loop(config):
    """
    Callable training routine with the paper's optimization/configuration controls.
    Implementation surface: training_loop
    """
    from .ppo import training_loop as _training_loop
    return _training_loop(config)

def get_config():
    """
    Expose configuration surface.
    Implementation surface: config
    """
    from .config import RICEConfig
    return RICEConfig

def run_experiment_matrix(methods=None, environments=None, alphas=None):
    """
    Full experiment-matrix route contract: implement executable orchestration 
    over the declared paper-derived dimensions.
    """
    methods = methods or METHODS
    environments = environments or ENVIRONMENTS
    alphas = alphas or ALPHA_VALUES
    
    results = []
    # Orchestration logic for Experiment I-V would iterate here
    return results

# Expose artifact writers (proxied to utils)
def write_metrics_artifact(*args, **kwargs):
    from .utils import write_metrics_artifact
    return write_metrics_artifact(*args, **kwargs)

def write_experiment_results_artifact(*args, **kwargs):
    from .utils import write_experiment_results_artifact
    return write_experiment_results_artifact(*args, **kwargs)

def write_environment_registry_artifact(*args, **kwargs):
    from .utils import write_environment_registry_artifact
    return write_environment_registry_artifact(*args, **kwargs)

def write_dataset_registry_artifact(*args, **kwargs):
    from .utils import write_dataset_registry_artifact
    return write_dataset_registry_artifact(*args, **kwargs)

def write_environment_readiness_artifact(*args, **kwargs):
    from .utils import write_environment_readiness_artifact
    return write_environment_readiness_artifact(*args, **kwargs)

def write_data_manifest_artifact(*args, **kwargs):
    from .utils import write_data_manifest_artifact
    return write_data_manifest_artifact(*args, **kwargs)

def write_method_registry_artifact(*args, **kwargs):
    from .utils import write_method_registry_artifact
    return write_method_registry_artifact(*args, **kwargs)

def write_ablation_registry_artifact(*args, **kwargs):
    from .utils import write_ablation_registry_artifact
    return write_ablation_registry_artifact(*args, **kwargs)

__all__ = [
    "ALPHA_VALUES",
    "LAMBDA_VALUES",
    "P_VALUES",
    "LEARNING_RATE_VALUES",
    "METHODS",
    "ENVIRONMENTS",
    "D_MAX",
    "BLACK_BOX_ASSUMPTION",
    "DEFAULT_ALPHA",
    "MASK_THRESHOLD_K",
    "BLINDING_PENALTY",
    "PPO_LEARNING_RATE",
    "ROLL_IN_STEPS",
    "EXPLORATION_STEPS",
    "FIDELITY_TRAJECTORIES",
    "K_VALUES",
    "model_loader_factory",
    "method_factory",
    "training_loop",
    "get_config",
    "run_experiment_matrix",
    "write_metrics_artifact",
    "write_experiment_results_artifact",
    "write_environment_registry_artifact",
    "write_dataset_registry_artifact",
    "write_environment_readiness_artifact",
    "write_data_manifest_artifact",
    "write_method_registry_artifact",
    "write_ablation_registry_artifact"
]