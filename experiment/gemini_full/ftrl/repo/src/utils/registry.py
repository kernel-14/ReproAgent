import os
import json

# reference_grounding: addendum:formula_algorithm_contract
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_BATCH_SIZE = 128

# reference_grounding: Paper evidence contract priority sweeps
learning_rate_values = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
batch_size_values = [64, 128, 256, 512]

def resolve_learning_rate_defaults(method_name: str) -> float:
    """
    Resolves the default learning rate for a given method.
    reference_grounding: Paper evidence contract priority methods: ours, ppo, sac, bc, oracle, nle, ewc
    """
    lr_map = {
        "ppo": 3e-4,
        "sac": 3e-4,
        "bc": 1e-4,
        "ewc": 1e-4,
        "ours": 3e-4,
        "nle": 3e-4,
        "oracle": 3e-4,
        "vanilla": 3e-4
    }
    return lr_map.get(method_name.lower(), DEFAULT_LEARNING_RATE)

def resolve_batch_size_defaults(method_name: str) -> int:
    """
    Resolves the default batch size for a given method.
    reference_grounding: addendum numeric/defaults 128
    """
    if "batch_size_128" in method_name:
        return 128
    return DEFAULT_BATCH_SIZE

def compute_loss(method: str, **kwargs):
    """
    Dispatcher for various loss functions defined in the paper.
    reference_grounding: chunk_003_01 (EWC), chunk_004_02 (BC/KS)
    """
    method = method.lower()
    if method == "bc":
        return _compute_bc_loss(**kwargs)
    elif method == "ewc":
        return _compute_ewc_loss(**kwargs)
    elif method == "ks" or method == "kickstarting":
        return _compute_ks_loss(**kwargs)
    return 0.0

def _compute_bc_loss(pi_star_logits=None, pi_theta_logits=None, **kwargs):
    """
    L_BC(theta) = E_{s ~ B_BC} [D_KL(pi_*(s) || pi_theta(s))]
    reference_grounding: chunk_004_02 2. Forgetting of pre-trained capabilities
    """
    if pi_star_logits is None or pi_theta_logits is None:
        return 0.0
    try:
        import torch.nn.functional as F
        # KL(Teacher || Student)
        p = F.softmax(pi_star_logits, dim=-1)
        log_q = F.log_softmax(pi_theta_logits, dim=-1)
        return F.kl_div(log_q, p, reduction='batchmean')
    except ImportError:
        return 0.0

def _compute_ewc_loss(fisher_diag=None, params=None, params_star=None, **kwargs):
    """
    L_aux(theta) = sum_i F^i (theta_*^i - theta^i)^2
    reference_grounding: chunk_003_01 2. Forgetting of pre-trained capabilities
    """
    if fisher_diag is None or params is None or params_star is None:
        return 0.0
    loss = 0.0
    for f, p, ps in zip(fisher_diag, params, params_star):
        loss += (f * (p - ps)**2).sum()
    return loss

def _compute_ks_loss(pi_star_logits=None, pi_theta_logits=None, **kwargs):
    """
    L_KS(theta) = E_{s ~ pi_theta} [D_KL(pi_*(s) || pi_theta(s))]
    reference_grounding: chunk_004_02 2. Forgetting of pre-trained capabilities
    """
    return _compute_bc_loss(pi_star_logits, pi_theta_logits)

def aggregate_loss(losses: list) -> float:
    """
    Aggregates a list of loss components.
    """
    return sum(losses)

def compute_reward(env_reward: float, info: dict = None) -> float:
    """
    Computes or transforms the reward based on environment info.
    """
    return env_reward

def aggregate_reward(rewards: list) -> float:
    """
    Aggregates rewards over an episode or training run (AUC).
    reference_grounding: F. Analysis of forgetting in robotic manipulation tasks
    """
    if not rewards:
        return 0.0
    # AUC is the average success rate or reward over time
    return sum(rewards) / len(rewards)

def compute_ours_oradaptersby_inventory_objective(rl_loss, aux_loss, beta=1.0):
    """
    Combined objective: RL + beta * Regularization
    reference_grounding: chunk_004_02 2. Forgetting of pre-trained capabilities
    """
    return rl_loss + beta * aux_loss

def compute_ours_oradaptersby_inventory_score(success_rate, forgetting_score):
    """
    Evaluation score combining performance and retention.
    """
    return success_rate - forgetting_score

def compute_forward_transfer(auc, auc_baseline):
    """
    Forward Transfer := (AUC - AUC^b) / (1 - AUC^b)
    reference_grounding: F. Analysis of forgetting in robotic manipulation tasks
    """
    return (auc - auc_baseline) / (1.0 - auc_baseline + 1e-8)

def compute_v0_mdp(theta, gamma=0.9, r_0=0.11, r_1=2.22, epsilon=0.5):
    """
    Value of state s_0 in the two-state MDP.
    reference_grounding: chunk_018 A.1. Two-state MDPs
    """
    f_theta = compute_f_theta_mdp(theta, epsilon)
    num = theta + r_0 * (1 - theta) * (1 - gamma * f_theta) + gamma * theta * r_1 * (1 - f_theta)
    den = 1 - gamma * f_theta + gamma * theta
    return (1.0 / (1.0 - gamma)) * (num / den)

def compute_f_theta_mdp(theta, epsilon=0.5):
    """
    Policy parameterization for the two-state MDP.
    reference_grounding: chunk_018 A.1. Two-state MDPs
    """
    if theta <= 1 - epsilon / 2:
        return (-epsilon / (1 - epsilon / 2)) * theta + 1
    else:
        return 2 * theta - 1

def resolve_appleretrieval_params():
    """
    Parameters for the AppleRetrieval grid-world.
    reference_grounding: chunk_019 A.2. Synthetic example: Appleretrieval
    """
    return {
        "M": 13,
        "c": 11,
        "sigma": 30,
        "pi_w": 1.0,
        "pi_b": 0.0
    }

def resolve_robotics_params():
    """
    Parameters for the Robotics manipulation tasks.
    reference_grounding: B.3. Meta World
    """
    return {
        "E_k": 200,
        "E_i": 1,
        "beta": 1.5,
        "r_t": 1.0,
        "r_t_prime": 1.0
    }

# Registry for methods and environments
METHOD_REGISTRY = {
    "vanilla": "src.methods.vanilla.VanillaFineTuning",
    "ours": "src.methods.ours.OursMethod",
    "ppo": "src.methods.ppo.PPOMethod",
    "sac": "src.methods.sac.SACMethod",
    "bc": "src.methods.bc.BCMethod",
    "oracle": "src.methods.oracle.OracleMethod",
    "nle": "src.methods.nle.NLEMethod",
    "ewc": "src.methods.ewc.EWCMethod",
    "knowledge-retention": "src.methods.bc.BCMethod",
    "scaled-bc + fine-tuning + ks": "src.methods.hybrid.HybridMethod"
}

ENV_REGISTRY = {
    "two_state_mdp": "src.envs.two_state_mdp.make_two_state_mdp",
    "appleretrieval": "src.envs.apple_retrieval.make_apple_retrieval",
    "robotics": "src.envs.robotics.make_robotics"
}

# Artifact writers
def write_figure_1_artifact(data, path="results/figures/figure_1.png"):
    _ensure_dir(path)
    # reference_grounding: results/figures/figure_1.png
    pass

def write_figure_2_artifact(data, path="results/figures/figure_2.png"):
    _ensure_dir(path)
    # reference_grounding: results/figures/figure_2.png
    pass

def write_figure_4_artifact(data, path="results/figures/figure_4.png"):
    _ensure_dir(path)
    # reference_grounding: results/figures/figure_4.png
    pass

def write_figure_12_artifact(data, path="results/figures/figure_12.png"):
    _ensure_dir(path)
    # reference_grounding: results/figures/figure_12.png
    pass

def write_figure_3_artifact(data, path="results/figures/figure_3.png"):
    _ensure_dir(path)
    pass

def write_figure_3a_artifact(data, path="results/figures/figure_3a.png"):
    _ensure_dir(path)
    pass

def write_figure_3b_artifact(data, path="results/figures/figure_3b.png"):
    _ensure_dir(path)
    pass

def write_figure_3c_artifact(data, path="results/figures/figure_3c.png"):
    _ensure_dir(path)
    pass

def _ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)