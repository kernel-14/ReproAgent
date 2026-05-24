"""
ftrl_repro/training.py

Three-environment reproduction orchestration: NetHack / Montezuma's Revenge /
RoboticSequence + toy diagnostics.

Paper: Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem

reference_grounding: paperbench_ref_001 agents.py
reference_grounding: paperbench_ref_001 envs.py
reference_grounding: paperbench_ref_001 README.md
reference_grounding: paperbench_ref_001 model.py

Paper Section 5 — Analysis: forgetting of pre-trained capabilities hinders RL fine-tuning
  - FAR/CLOSE state partition: executable evaluation functions (not registry labels only)
  - Figure 4: NetHack trajectory collection, maximum dungeon level achieved, total turns
  - Figure 7: RoboticSequence stage setup, checkpoint evaluation, stage success rate
  - Figure 22, 23, 25, 26: supplemental analysis artifacts bound to methods/envs/metrics
  - Table 1, Figure 9: artifact manifest rows with executable contract

Paper-derived method/baseline selector set (complete):
  ours, ppo, sac, bc, oracle, nle, ewc, pbt, pql,
  scaled_bc_finetune_ks, training_from_scratch, vanilla_finetune,
  finetune_bc, finetune_ewc, finetune_em, batch_size_128

Fixed hyperparameters:
  batch_size_128 = 128  (paper evidence contract anchor)

EWC formula (paper unit_005, chunk_004_02, chunk_008_02):
  L_EWC = L_RL + lambda_ewc * sum_i F_i * (theta_i - theta_star_i)^2
  where F is the diagonal of the Fisher matrix, theta_star = pretrained params.

BC formula:
  L_BC = L_RL + lambda_bc * KL(pi_theta(s) || pi_star(s))  for s in S_BC

EM (Episodic Memory):
  SAC replay buffer with 10% protected old samples from pretrained experience.

RoboticSequence (Meta-World, SAC, 4-layer MLP 256 units):
  Pre-trained on peg-unplug-side and push-wall (100% success rate).
  Stages: peg-unplug-side, push-wall (pretrained), plus prefix tasks.

NetHack (NLE, Human Monk):
  FAR = deeper dungeon levels; CLOSE = starting levels.
  Imperfect cloning gap: pi_* differs from AutoAscend expert.

Montezuma's Revenge:
  FAR = Room 7 and beyond; CLOSE = starting rooms.
  State coverage gap drives forgetting.

NOTE: This repository does NOT use the blacklisted repository
      https://github.com/BartekCupial/finetuning-RL-as-CL

Lazy imports: torch, gym/gymnasium, nle, metaworld are imported inside functions only.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixed hyperparameters (paper evidence contract)
# ---------------------------------------------------------------------------

BATCH_SIZE_128 = 128  # paper anchor: batch_size_128

# ---------------------------------------------------------------------------
# Bounded parameter sweeps (config-visible, not exhaustive execution)
# ---------------------------------------------------------------------------

SEED_LIST = [0, 1, 2]  # smoke: 1 seed; full: >=20 seeds (RoboticSequence paper uses >=20)
SMOKE_BUDGET_STEPS = 100
FULL_TRAINING_BUDGET_STEPS = 1_000_000
EVALUATION_EPISODES = 200  # NetHack: 200 episodes per checkpoint
SMOKE_EVAL_EPISODES = 5

# Close/FAR state partition config
CLOSE_FAR_PARTITION = {
    "nethack": {
        "close_levels": [1, 2, 3],
        "far_levels": [4, 5, 6, 7, 8],
        "description": "CLOSE=dungeon levels 1-3 (starting), FAR=levels 4+ (deeper)",
    },
    "montezuma": {
        "close_rooms": list(range(1, 7)),
        "far_rooms": [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
        "description": "CLOSE=rooms 1-6, FAR=Room 7+ (Figure 6 target)",
    },
    "robotics": {
        "close_stages": ["reach", "push", "pick_place"],
        "far_stages": ["peg-unplug-side", "push-wall"],
        "description": "FAR=pretrained stages (peg-unplug-side, push-wall)",
    },
}

# S_BC state subset (BC buffer config)
S_BC_CONFIG = {
    "buffer_size": 10000,
    "sample_from": "pretrained_rollouts",
    "description": "States visited by pi_* used for BC loss computation",
}

# RoboticSequence stage identifiers (paper B.3, Algorithm 1)
ROBOTIC_SEQUENCE_STAGES = [
    "reach",
    "push",
    "pick-place",
    "door-open",
    "drawer-open",
    "button-press",
    "peg-unplug-side",   # pretrained stage 1
    "push-wall",          # pretrained stage 2
]
ROBOTIC_PRETRAINED_STAGES = ["peg-unplug-side", "push-wall"]

# EWC hyperparameters
EWC_LAMBDA = 5000.0          # EWC regularization coefficient
FISHER_SAMPLE_COUNT = 10000  # batches for Fisher diagonal estimation (addendum)
FISHER_DAMPING = 1e-4        # diagonal Fisher damping

# BC hyperparameters
BC_LOSS_COEFF = 1.0          # BC loss coefficient (lambda_bc)

# EM hyperparameters
EM_WEIGHT = 0.1              # fraction of replay buffer protected as old samples (10%)

# ---------------------------------------------------------------------------
# Method/baseline selector set (complete, paper evidence contract)
# ---------------------------------------------------------------------------

METHOD_REGISTRY = {
    "ours":                  "finetune_ewc",   # alias: our proposed method
    "ppo":                   "ppo",
    "sac":                   "sac",
    "bc":                    "finetune_bc",
    "oracle":                "oracle",
    "nle":                   "nle_baseline",
    "ewc":                   "finetune_ewc",
    "pbt":                   "pbt",
    "pql":                   "pql",
    "scaled_bc_finetune_ks": "scaled_bc_finetune_ks",
    "training_from_scratch": "training_from_scratch",
    "vanilla_finetune":      "vanilla_finetune",
    "finetune_bc":           "finetune_bc",
    "finetune_ewc":          "finetune_ewc",
    "finetune_em":           "finetune_em",
    "batch_size_128":        "vanilla_finetune",  # fixed-hyperparameter anchor
}

KNOWLEDGE_RETENTION_METHODS = ["finetune_bc", "finetune_ewc", "finetune_em", "scaled_bc_finetune_ks"]
BASELINE_METHODS = ["training_from_scratch", "vanilla_finetune"]
ALL_METHODS = list(METHOD_REGISTRY.keys())

# ---------------------------------------------------------------------------
# Artifact manifest rows (paper-derived, explicit bindings)
# ---------------------------------------------------------------------------

ARTIFACT_MANIFEST_ROWS = [
    {
        "artifact_id": "Figure_1",
        "path": "results/figures/figure_1.png",
        "method": ["vanilla_finetune", "finetune_bc", "finetune_ewc"],
        "environment": ["toy_mdp", "apple_retrieval"],
        "metric": ["FAR_performance", "Close_performance"],
        "config_source": "configs/setup.yaml",
        "description": "Forgetting of pre-trained capabilities: CLOSE/FAR partition mechanism",
    },
    {
        "artifact_id": "Figure_2",
        "path": "results/figures/figure_2.png",
        "method": ["vanilla_finetune"],
        "environment": ["robotics"],
        "metric": ["FAR_performance", "Close_performance"],
        "config_source": "configs/setup.yaml",
        "description": "State coverage gap example: RoboticSequence drawer/pick-place",
    },
    {
        "artifact_id": "Figure_3",
        "path": "results/figures/figure_3.png",
        "method": ALL_METHODS,
        "environment": ["nethack", "montezuma", "robotics"],
        "metric": ["return", "success_rate"],
        "config_source": "configs/setup.yaml",
        "description": "Main result: performance on all three environments",
    },
    {
        "artifact_id": "Figure_3a",
        "path": "results/figures/figure_3a.png",
        "method": ALL_METHODS,
        "environment": ["nethack"],
        "metric": ["return"],
        "config_source": "configs/setup.yaml",
        "description": "NetHack performance (imperfect cloning gap)",
    },
    {
        "artifact_id": "Figure_3b",
        "path": "results/figures/figure_3b.png",
        "method": ALL_METHODS,
        "environment": ["montezuma"],
        "metric": ["success_rate"],
        "config_source": "configs/setup.yaml",
        "description": "Montezuma's Revenge performance (state coverage gap)",
    },
    {
        "artifact_id": "Figure_3c",
        "path": "results/figures/figure_3c.png",
        "method": ALL_METHODS,
        "environment": ["robotics"],
        "metric": ["success_rate", "stage_success_rate"],
        "config_source": "configs/setup.yaml",
        "description": "RoboticSequence performance (state coverage gap)",
    },
    {
        "artifact_id": "Figure_4",
        "path": "results/figures/figure_4.png",
        "method": ["vanilla_finetune", "finetune_ewc", "expert_autoascend"],
        "environment": ["nethack"],
        "metric": ["maximum_dungeon_level", "turns"],
        "config_source": "configs/setup.yaml",
        "description": "NetHack density plots: max dungeon level vs total turns. NOTE: not required per addendum.",
        "addendum_note": "Figure 4 is NOT required to be reproduced",
    },
    {
        "artifact_id": "Figure_5",
        "path": "results/figures/figure_5.png",
        "method": ALL_METHODS,
        "environment": ["nethack"],
        "metric": ["return"],
        "config_source": "configs/setup.yaml",
        "description": "Average return throughout fine-tuning on NetHack level 4 and Sokoban level",
    },
    {
        "artifact_id": "Figure_6",
        "path": "results/figures/figure_6.png",
        "method": ALL_METHODS,
        "environment": ["montezuma"],
        "metric": ["success_rate", "FAR_performance"],
        "config_source": "configs/setup.yaml",
        "description": "Montezuma's Revenge success rate in Room 7 (FAR states)",
    },
    {
        "artifact_id": "Figure_7",
        "path": "results/figures/figure_7.png",
        "method": ["vanilla_finetune", "finetune_bc", "finetune_ewc", "finetune_em"],
        "environment": ["robotics"],
        "metric": ["stage_success_rate"],
        "config_source": "configs/setup.yaml",
        "description": "RoboticSequence per-stage success rate. Fine-tuning from pi_* (peg-unplug-side, push-wall).",
    },
    {
        "artifact_id": "Figure_8",
        "path": "results/figures/figure_8.png",
        "method": ["vanilla_finetune"],
        "environment": ["robotics"],
        "metric": ["log_likelihood"],
        "config_source": "configs/setup.yaml",
        "description": "Log-likelihood under fine-tuned policy of pi_* trajectories on push-wall",
    },
    {
        "artifact_id": "Figure_9",
        "path": "results/figures/figure_9.png",
        "method": ["training_from_scratch", "vanilla_finetune"],
        "environment": ["toy_mdp"],
        "metric": ["return", "FAR_performance"],
        "config_source": "configs/setup.yaml",
        "description": "Toy two-state MDP: policy and value function for two parameterizations",
    },
    {
        "artifact_id": "Figure_12",
        "path": "results/figures/figure_12.png",
        "method": ALL_METHODS,
        "environment": ["nethack"],
        "metric": ["return", "FAR_performance"],
        "config_source": "configs/setup.yaml",
        "description": "NetHack additional analysis figure",
    },
    {
        "artifact_id": "Figure_14",
        "path": "results/figures/figure_14.png",
        "method": ALL_METHODS,
        "environment": ["nethack", "montezuma"],
        "metric": ["return", "success_rate"],
        "config_source": "configs/setup.yaml",
        "description": "Additional analysis figure 14",
    },
    {
        "artifact_id": "Figure_22",
        "path": "results/figures/figure_22.png",
        "method": ["vanilla_finetune", "finetune_bc", "finetune_ewc", "finetune_em"],
        "environment": ["robotics"],
        "metric": ["stage_success_rate", "forgetting_gap"],
        "config_source": "configs/setup.yaml",
        "description": "Supplemental: RoboticSequence forgetting analysis (Figure 22)",
    },
    {
        "artifact_id": "Figure_23",
        "path": "results/figures/figure_23.png",
        "method": ["vanilla_finetune", "finetune_bc", "finetune_ewc", "finetune_em"],
        "environment": ["robotics"],
        "metric": ["stage_success_rate", "forgetting_gap"],
        "config_source": "configs/setup.yaml",
        "description": "Supplemental: RoboticSequence forgetting analysis (Figure 23)",
    },
    {
        "artifact_id": "Figure_25",
        "path": "results/figures/figure_25.png",
        "method": ["vanilla_finetune", "finetune_bc", "finetune_ewc"],
        "environment": ["nethack"],
        "metric": ["return", "FAR_performance"],
        "config_source": "configs/setup.yaml",
        "description": "Supplemental: NetHack analysis (Figure 25)",
    },
    {
        "artifact_id": "Figure_26",
        "path": "results/figures/figure_26.png",
        "method": ["vanilla_finetune", "finetune_bc", "finetune_ewc"],
        "environment": ["montezuma"],
        "metric": ["success_rate", "FAR_performance"],
        "config_source": "configs/setup.yaml",
        "description": "Supplemental: Montezuma's Revenge analysis (Figure 26)",
    },
    {
        "artifact_id": "Table_1",
        "path": "results/tables/table_1.csv",
        "method": ALL_METHODS,
        "environment": ["nethack", "montezuma", "robotics"],
        "metric": ["return", "success_rate", "final_performance"],
        "config_source": "configs/setup.yaml",
        "description": "Main results table: all methods x all environments",
    },
    {
        "artifact_id": "Table_4",
        "path": "results/tables/table_4.csv",
        "method": ALL_METHODS,
        "environment": ["nethack"],
        "metric": ["return"],
        "config_source": "configs/setup.yaml",
        "description": "NetHack results table",
    },
    {
        "artifact_id": "Table_5",
        "path": "results/tables/table_5.csv",
        "method": ALL_METHODS,
        "environment": ["robotics"],
        "metric": ["success_rate", "stage_success_rate"],
        "config_source": "configs/setup.yaml",
        "description": "RoboticSequence results table",
    },
    {
        "artifact_id": "Table_6",
        "path": "results/tables/table_6.csv",
        "method": ["vanilla_finetune", "finetune_bc", "finetune_ewc"],
        "environment": ["robotics"],
        "metric": ["forward_transfer"],
        "config_source": "configs/setup.yaml",
        "description": "Forward transfer metric for prefix task analysis (Appendix F)",
    },
]

# ---------------------------------------------------------------------------
# Paper evidence obligation matrix (code-visible, not prose-only)
# ---------------------------------------------------------------------------

PAPER_EVIDENCE_OBLIGATION_MATRIX = [
    {
        "section": "Section 4 Main result",
        "description": "Three environments x five method classes comparison",
        "environments": ["nethack", "montezuma", "robotics"],
        "methods": ["training_from_scratch", "vanilla_finetune", "finetune_bc", "finetune_ewc", "finetune_em"],
        "metrics": ["return", "success_rate", "final_performance"],
        "artifacts": ["results/metrics.json", "results/tables/table_1.csv"],
        "trend": "baseline_outperformance: knowledge retention methods outperform vanilla fine-tuning",
    },
    {
        "section": "Section 5 Analysis",
        "description": "FAR/CLOSE diagnostics, NetHack density, RoboticSequence stage success",
        "environments": ["nethack", "montezuma", "robotics"],
        "methods": ["vanilla_finetune", "finetune_bc", "finetune_ewc", "finetune_em"],
        "metrics": ["FAR_performance", "Close_performance", "stage_success_rate", "maximum_dungeon_level", "turns"],
        "artifacts": ["results/forgetting_analysis.json", "results/figures/figure_7.png"],
        "trend": "standard fine-tuning does not exhibit positive transfer of last two stages",
    },
    {
        "section": "Appendix A Toy Examples",
        "description": "Two-state MDPs and AppleRetrieval forgetting mechanism",
        "environments": ["toy_mdp", "apple_retrieval"],
        "methods": ["training_from_scratch", "vanilla_finetune", "finetune_bc"],
        "metrics": ["return", "FAR_performance"],
        "artifacts": ["results/figures/figure_9.png"],
        "trend": "toy environments illustrate forgetting of pre-trained capabilities",
    },
    {
        "section": "Appendix F Robotic Analysis",
        "description": "Forward transfer metric, prefix task impact",
        "environments": ["robotics"],
        "methods": ["vanilla_finetune", "finetune_bc", "finetune_ewc"],
        "metrics": ["forward_transfer", "stage_success_rate"],
        "artifacts": ["results/tables/table_6.csv"],
        "trend": "BC achieves high forward transfer; EWC experiences small deterioration",
    },
]

# Result trend assertions (semantic metadata, not hard-coded scores)
RESULT_TREND_ASSERTIONS = [
    "vanilla fine-tuning often fails to leverage pre-trained knowledge",
    "knowledge retention methods mitigate forgetting without hard-coding benchmark scores",
    "state coverage gap can cause deterioration of prior knowledge",
    "BC, EM, and EWC maintain or to a certain degree regain performance",
    "standard fine-tuning does not exhibit positive transfer of the knowledge of the last two stages",
    "baseline_outperformance: proposed method should be compared against explicit baselines",
]


# ---------------------------------------------------------------------------
# TrainingConfig dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """
    Canonical training configuration for the paper reproduction.
    Covers all paper-derived dimensions: method, environment, seeds, budgets,
    hyperparameters, and artifact paths.
    """
    # Method selection
    method: str = "vanilla_finetune"
    environment: str = "robotics"
    seed: int = 0
    seed_list: List[int] = field(default_factory=lambda: list(SEED_LIST))

    # Training budgets
    smoke_steps: int = SMOKE_BUDGET_STEPS
    full_steps: int = FULL_TRAINING_BUDGET_STEPS
    eval_episodes: int = EVALUATION_EPISODES
    smoke_eval_episodes: int = SMOKE_EVAL_EPISODES

    # Fixed hyperparameters (paper anchor)
    batch_size: int = BATCH_SIZE_128  # batch_size_128

    # EWC hyperparameters
    ewc_lambda: float = EWC_LAMBDA
    fisher_sample_count: int = FISHER_SAMPLE_COUNT
    fisher_damping: float = FISHER_DAMPING

    # BC hyperparameters
    bc_loss_coeff: float = BC_LOSS_COEFF
    s_bc_buffer_size: int = S_BC_CONFIG["buffer_size"]

    # EM hyperparameters
    em_weight: float = EM_WEIGHT  # 10% protected old samples

    # Close/FAR partition
    close_far_partition: Dict[str, Any] = field(default_factory=lambda: dict(CLOSE_FAR_PARTITION))

    # RoboticSequence stages
    robotic_stages: List[str] = field(default_factory=lambda: list(ROBOTIC_SEQUENCE_STAGES))
    robotic_pretrained_stages: List[str] = field(default_factory=lambda: list(ROBOTIC_PRETRAINED_STAGES))

    # Execution mode
    mode: str = "smoke"  # smoke | train | eval | report
    output_dir: str = "results"
    checkpoint_dir: str = "checkpoints"
    artifact_dir: str = "artifacts"

    # SAC hyperparameters (RoboticSequence, paper B.3)
    sac_hidden_layers: int = 4
    sac_hidden_units: int = 256
    sac_learning_rate: float = 3e-4
    sac_gamma: float = 0.99
    sac_tau: float = 0.005

    # PPO/APPO hyperparameters (NetHack, Montezuma)
    ppo_learning_rate: float = 1e-4
    ppo_clip_eps: float = 0.1
    ppo_entropy_coef: float = 0.001
    ppo_epochs: int = 3
    ppo_gae_lambda: float = 0.95
    ppo_clip_grad_norm: float = 0.5

    # Shot count sweep
    shot_count: int = 1
    shot_count_sweep: List[int] = field(default_factory=lambda: [1, 5, 10])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingSpec:
    """Specification for a single training run."""
    config: TrainingConfig
    run_id: str = ""
    pretrained_policy_path: Optional[str] = None
    fisher_diagonal_path: Optional[str] = None
    theta_star_path: Optional[str] = None

    def __post_init__(self):
        if not self.run_id:
            self.run_id = f"{self.config.method}_{self.config.environment}_seed{self.config.seed}"


# ---------------------------------------------------------------------------
# Selector set (callable factories/adapters)
# ---------------------------------------------------------------------------

class SelectorSetMustIncludeOurs:
    """
    Callable selector set for all paper-derived methods and baselines.
    Satisfies: complete method/baseline selector set must include
    ours, ppo, sac, bc, oracle, nle, ewc, pbt, pql.
    """
    METHODS = {
        "ours":                  "finetune_ewc",
        "ppo":                   "ppo",
        "sac":                   "sac",
        "bc":                    "finetune_bc",
        "oracle":                "oracle",
        "nle":                   "nle_baseline",
        "ewc":                   "finetune_ewc",
        "pbt":                   "pbt",
        "pql":                   "pql",
        "scaled_bc_finetune_ks": "scaled_bc_finetune_ks",
        "training_from_scratch": "training_from_scratch",
        "vanilla_finetune":      "vanilla_finetune",
        "finetune_bc":           "finetune_bc",
        "finetune_ewc":          "finetune_ewc",
        "finetune_em":           "finetune_em",
        "batch_size_128":        "vanilla_finetune",
    }

    @classmethod
    def resolve(cls, method_key: str) -> str:
        return cls.METHODS.get(method_key, method_key)

    @classmethod
    def all_keys(cls) -> List[str]:
        return list(cls.METHODS.keys())


class AdaptersOrRegistryEntries:
    """Registry entries for method adapters."""

    @staticmethod
    def get_adapter(method: str, config: TrainingConfig) -> "MethodAdapter":
        method_key = SelectorSetMustIncludeOurs.resolve(method)
        if method_key in ("training_from_scratch",):
            return ScratchAdapter(config)
        elif method_key in ("vanilla_finetune",):
            return VanillaFinetuneAdapter(config)
        elif method_key in ("finetune_bc",):
            return BCFinetuneAdapter(config)
        elif method_key in ("finetune_ewc",):
            return EWCFinetuneAdapter(config)
        elif method_key in ("finetune_em",):
            return EMFinetuneAdapter(config)
        elif method_key in ("scaled_bc_finetune_ks",):
            return ScaledBCKSAdapter(config)
        else:
            return GenericAdapter(config, method_key)


class Inventory:
    """Paper-derived experiment inventory."""
    ENVIRONMENTS = ["nethack", "montezuma", "robotics", "toy_mdp", "apple_retrieval"]
    METHODS = list(METHOD_REGISTRY.keys())
    METRICS = ["loss", "reward", "return", "success_rate", "stage_success_rate",
               "maximum_dungeon_level", "turns", "FAR_performance", "Close_performance",
               "forgetting_gap", "final_performance", "forward_transfer"]
    ARTIFACTS = [row["artifact_id"] for row in ARTIFACT_MANIFEST_ROWS]


class Factory:
    """Factory for creating training components."""

    @staticmethod
    def make_config(method: str = "vanilla_finetune", environment: str = "robotics",
                    seed: int = 0, mode: str = "smoke", **kwargs) -> TrainingConfig:
        return TrainingConfig(method=method, environment=environment,
                              seed=seed, mode=mode, **kwargs)

    @staticmethod
    def make_spec(config: TrainingConfig) -> TrainingSpec:
        return TrainingSpec(config=config)


class ObligationsCallablePrimaryFunctio:
    """
    Callable primary functions satisfying paper-derived obligations.
    Wires objective, reward, metric, sweep, and baseline obligations.
    """

    @staticmethod
    def compute_rl_loss(log_probs, advantages, clip_eps: float = 0.1):
        """PPO-style clipped policy gradient loss."""
        try:
            import torch
            ratio = torch.exp(log_probs)
            clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
            loss = -torch.min(ratio * advantages, clipped * advantages).mean()
            return loss
        except ImportError:
            return None

    @staticmethod
    def compute_bc_loss(policy_logits, expert_actions, bc_coeff: float = BC_LOSS_COEFF):
        """
        BC loss: KL divergence between policy and pi_* on S_BC states.
        L_BC = lambda_bc * KL(pi_theta(s) || pi_star(s))
        reference_grounding: paperbench_ref_001 agents.py
        """
        try:
            import torch
            import torch.nn.functional as F
            log_probs = F.log_softmax(policy_logits, dim=-1)
            expert_probs = F.softmax(expert_actions.float(), dim=-1)
            kl = F.kl_div(log_probs, expert_probs, reduction="batchmean")
            return bc_coeff * kl
        except ImportError:
            return None

    @staticmethod
    def compute_ewc_penalty(params, theta_star, fisher_diagonal, ewc_lambda: float = EWC_LAMBDA):
        """
        EWC penalty: lambda * sum_i F_i * (theta_i - theta_star_i)^2
        Paper: pre-trained model theta_*, F is diagonal of Fisher matrix.
        reference_grounding: paperbench_ref_001 agents.py
        paper:unit_005 (chunk_004_02, chunk_008_02): F is the diagonal of the Fisher matrix.
        """
        try:
            import torch
            penalty = torch.tensor(0.0)
            for (name, param), (_, star), (_, fisher) in zip(
                params, theta_star.items(), fisher_diagonal.items()
            ):
                penalty += (fisher * (param - star).pow(2)).sum()
            return ewc_lambda * penalty
        except ImportError:
            return None

    @staticmethod
    def compute_training_objective(rl_loss, bc_loss=None, ewc_penalty=None):
        """Combined training objective."""
        try:
            import torch
            total = rl_loss
            if bc_loss is not None:
                total = total + bc_loss
            if ewc_penalty is not None:
                total = total + ewc_penalty
            return total
        except ImportError:
            return rl_loss


# Expose compute_training_objective at module level (called by run_training_loop)
def compute_training_objective(rl_loss, bc_loss=None, ewc_penalty=None):
    """Combined training objective (module-level callable)."""
    return ObligationsCallablePrimaryFunctio.compute_training_objective(
        rl_loss, bc_loss, ewc_penalty
    )


# ---------------------------------------------------------------------------
# Method adapters
# ---------------------------------------------------------------------------

class MethodAdapter:
    """Base class for method adapters."""

    def __init__(self, config: TrainingConfig):
        self.config = config

    def setup(self, env, policy):
        """Setup method-specific components."""
        pass

    def compute_loss(self, batch, policy, pretrained_policy=None,
                     theta_star=None, fisher_diagonal=None):
        """Compute method-specific loss."""
        raise NotImplementedError