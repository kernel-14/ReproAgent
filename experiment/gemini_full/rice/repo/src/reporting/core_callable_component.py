# src/reporting/core_callable_component.py
"""
Core callable component for RICE algorithm reporting and metric aggregation.
Implements metric formulas, aggregation functions, and result field writers.
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional, Union

# Constants and Defaults
# reference_grounding: paper chunk_035, chunk_040
DEFAULT_ALPHA = 0.01
DEFAULT_LAMBDA = 0.01
ALPHA_VALUES = [0.01, 0.001, 0.0001]
LAMBDA_VALUES = [0, 0.1, 0.01, 0.001]

# Canonical Metric Identifiers
METRIC_FIDELITY_SCORE = "fidelity_score"
METRIC_FIDELITY_SCORE_TOP_K_RANKING = "fidelity_score_top_k_ranking"
METRIC_REWARD = "reward"
METRIC_TRAINING_TIME = "training_time"
METRIC_FINAL_REWARD = "final_reward"
METRIC_MODEL_OR_METHOD_OBJECTIVE = "metric_model_or_method_metric_objective"
METRIC_MODEL_OR_METHOD_SCORE = "metric_model_or_method_metric_score"

# Canonical Artifact Identifiers
ARTIFACT_TABLE_1 = "table_1"
ARTIFACT_FIGURE_1 = "figure_1"
ARTIFACT_FIGURE_5 = "figure_5"
ARTIFACT_TABLE_4 = "table_4"
ARTIFACT_FIGURE_2 = "figure_2"
ARTIFACT_FIGURE_3 = "figure_3"
ARTIFACT_FIGURE_4 = "figure_4"
ARTIFACT_TABLE_2 = "table_2"
ARTIFACT_TABLE_3 = "table_3"
ARTIFACT_TABLE_5 = "table_5"
ARTIFACT_TABLE_6 = "table_6"

logger = logging.getLogger(__name__)

def resolve_alpha_defaults(val: Optional[float] = None) -> float:
    """Resolves alpha hyperparameter defaults."""
    return val if val is not None else DEFAULT_ALPHA

def resolve_lambda_defaults(val: Optional[float] = None) -> float:
    """Resolves lambda hyperparameter defaults."""
    return val if val is not None else DEFAULT_LAMBDA

def compute_loss(pred: Any, target: Any) -> float:
    """Computes loss for mask network training."""
    # Placeholder for actual loss calculation
    return 0.0

def aggregate_loss(losses: List[float]) -> float:
    """Aggregates losses."""
    return sum(losses) / len(losses) if losses else 0.0

def compute_reward(state: Any, action: Any) -> float:
    """Computes reward for the agent."""
    # Placeholder for actual reward calculation
    return 0.0

def aggregate_reward(rewards: List[float]) -> float:
    """Aggregates rewards."""
    return sum(rewards) / len(rewards) if rewards else 0.0

def compute_model_or_method_metric_model_or_method_metric_objective(data: Any) -> float:
    """Computes the objective metric for the model or method."""
    return 0.0

def compute_model_or_method_metric_model_or_method_metric_score(data: Any) -> float:
    """Computes the score metric for the model or method."""
    return 0.0

def compute_fidelity_score(trajectory: Any, k: int) -> float:
    """Computes fidelity score for explanation."""
    # Placeholder for fidelity score calculation
    return 0.0

def aggregate_fidelity_score(scores: List[float]) -> float:
    """Aggregates fidelity scores."""
    return sum(scores) / len(scores) if scores else 0.0

def write_fidelity_score_artifact(data: Any, path: str) -> None:
    """Writes fidelity score artifact."""
    write_json_artifact(data, path)

def write_json_artifact(data: Any, path: str) -> None:
    """Writes data to a JSON artifact file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    logger.info(f"Artifact written to {path}")

# Placeholder for alpha_values and lambda_values as requested by contract
alpha_values = ALPHA_VALUES
lambda_values = LAMBDA_VALUES