"""
Artifact contract and metric computation for Robust CLIP reproduction.

This module provides:
- Static artifact path declarations for all paper tables and figures
- Metric schemas and computation functions
- Result aggregation and artifact writers
- Adversarial attack evaluation interfaces

Paper evidence contract:
- Metric schemas for: accuracy, clean_accuracy, f1, precision, loss, cider,
  vqa_accuracy, success_rate, robust_accuracy, training_time, attack_success_rate
- Result artifact writers for: Figure 1-14, Table 1-14, and supplementary outputs
- Stable output paths under results/
"""

import os
import json
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# Artifact Path Registry (Paper Evidence Contract)
# ============================================================================

class ArtifactPaths:
    """Static registry of all paper artifact output paths."""
    
    # Base directories
    RESULTS_DIR = Path("results")
    FIGURES_DIR = RESULTS_DIR / "figures"
    TABLES_DIR = RESULTS_DIR / "tables"
    CHECKPOINTS_DIR = Path("checkpoints")
    
    # Figure outputs (Paper evidence contract)
    FIGURE_1 = FIGURES_DIR / "figure_1.png"
    FIGURE_1_PDF = FIGURES_DIR / "figure_1.pdf"
    FIGURE_2 = FIGURES_DIR / "figure_2.png"
    FIGURE_3 = FIGURES_DIR / "figure_3.png"
    FIGURE_4 = FIGURES_DIR / "figure_4.png"
    FIGURE_5 = FIGURES_DIR / "figure_5.png"
    EXPERIMENT_RESULTS_FIGURE = FIGURES_DIR / "experiment_results.png"
    
    # Table outputs (Paper evidence contract)
    TABLE_1 = TABLES_DIR / "table_1.csv"
    TABLE_2 = TABLES_DIR / "table_2.csv"
    TABLE_3 = TABLES_DIR / "table_3.csv"
    TABLE_4 = TABLES_DIR / "table_4.csv"
    TABLE_5 = TABLES_DIR / "table_5.csv"
    TABLE_6 = TABLES_DIR / "table_6.csv"
    TABLE_7 = TABLES_DIR / "table_7.csv"
    TABLE_8 = TABLES_DIR / "table_8.csv"
    TABLE_9 = TABLES_DIR / "table_9.csv"
    TABLE_10 = TABLES_DIR / "table_10.csv"
    TABLE_11 = TABLES_DIR / "table_11.csv"
    TABLE_12 = TABLES_DIR / "table_12.csv"
    TABLE_13 = TABLES_DIR / "table_13.csv"
    TABLE_14 = TABLES_DIR / "table_14.csv"
    EXPERIMENT_RESULTS_TABLE = TABLES_DIR / "experiment_results.csv"
    
    # Metric and prediction outputs
    METRICS_JSON = RESULTS_DIR / "metrics.json"
    PREDICTIONS_JSONL = RESULTS_DIR / "predictions.jsonl"
    CONFIG_RESOLVED = RESULTS_DIR / "config_resolved.json"
    ROBUSTNESS_TABLE = RESULTS_DIR / "robustness_table.csv"
    
    # Checkpoint outputs
    FARE_CHECKPOINT = CHECKPOINTS_DIR / "fare_clip.pth"
    FARE_CHECKPOINT_FINAL = CHECKPOINTS_DIR / "fare_clip_final.pth"
    TECOA_CHECKPOINT = CHECKPOINTS_DIR / "tecoa_clip.pth"
    
    # Runtime artifacts
    READINESS_JSON = RESULTS_DIR / "readiness.json"
    EVALUATION_RESULT_JSON = RESULTS_DIR / "evaluation_result.json"
    
    @classmethod
    def ensure_directories(cls):
        """Create all required artifact directories."""
        cls.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        cls.TABLES_DIR.mkdir(parents=True, exist_ok=True)
        cls.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def all_artifacts(cls) -> List[Path]:
        """Return list of all declared artifact paths."""
        return [
            cls.FIGURE_1, cls.FIGURE_1_PDF, cls.FIGURE_2, cls.FIGURE_3,
            cls.FIGURE_4, cls.FIGURE_5, cls.EXPERIMENT_RESULTS_FIGURE,
            cls.TABLE_1, cls.TABLE_2, cls.TABLE_3, cls.TABLE_4, cls.TABLE_5,
            cls.TABLE_6, cls.TABLE_7, cls.TABLE_8, cls.TABLE_9, cls.TABLE_10,
            cls.TABLE_11, cls.TABLE_12, cls.TABLE_13, cls.TABLE_14,
            cls.EXPERIMENT_RESULTS_TABLE, cls.METRICS_JSON, cls.PREDICTIONS_JSONL,
            cls.CONFIG_RESOLVED, cls.ROBUSTNESS_TABLE,
            cls.READINESS_JSON, cls.EVALUATION_RESULT_JSON
        ]


# ============================================================================
# Metric Schema Definitions (Paper Evidence Contract)
# ============================================================================

@dataclass
class MetricSchema:
    """Schema for a single metric type."""
    name: str
    description: str
    aggregation: str  # mean, sum, max, etc.
    range: Tuple[float, float]
    higher_is_better: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MetricRegistry:
    """Registry of all metric schemas used in the paper."""
    
    # Classification metrics
    ACCURACY = MetricSchema(
        name="accuracy",
        description="Classification accuracy (correct predictions / total)",
        aggregation="mean",
        range=(0.0, 1.0),
        higher_is_better=True
    )
    
    CLEAN_ACCURACY = MetricSchema(
        name="clean_accuracy",
        description="Accuracy on clean (non-adversarial) images",
        aggregation="mean",
        range=(0.0, 1.0),
        higher_is_better=True
    )
    
    ROBUST_ACCURACY = MetricSchema(
        name="robust_accuracy",
        description="Accuracy on adversarial images at specified epsilon",
        aggregation="mean",
        range=(0.0, 1.0),
        higher_is_better=True
    )
    
    # Attack metrics
    ATTACK_SUCCESS_RATE = MetricSchema(
        name="attack_success_rate",
        description="Percentage of successful adversarial attacks",
        aggregation="mean",
        range=(0.0, 1.0),
        higher_is_better=False
    )
    
    SUCCESS_RATE = MetricSchema(
        name="success_rate",
        description="Task success rate (generic)",
        aggregation="mean",
        range=(0.0, 1.0),
        higher_is_better=True
    )
    
    # Vision-language metrics
    F1 = MetricSchema(
        name="f1",
        description="F1 score (harmonic mean of precision and recall)",
        aggregation="mean",
        range=(0.0, 1.0),
        higher_is_better=True
    )
    
    PRECISION = MetricSchema(
        name="precision",
        description="Precision (true positives / predicted positives)",
        aggregation="mean",
        range=(0.0, 1.0),
        higher_is_better=True
    )
    
    CIDER = MetricSchema(
        name="cider",
        description="CIDEr score for image captioning",
        aggregation="mean",
        range=(0.0, 10.0),
        higher_is_better=True
    )
    
    VQA_ACCURACY = MetricSchema(
        name="vqa_accuracy",
        description="Visual question answering accuracy",
        aggregation="mean",
        range=(0.0, 1.0),
        higher_is_better=True
    )
    
    # Training metrics
    LOSS = MetricSchema(
        name="loss",
        description="Training or validation loss",
        aggregation="mean",
        range=(0.0, float('inf')),
        higher_is_better=False
    )
    
    TRAINING_TIME = MetricSchema(
        name="training_time",
        description="Training time in seconds",
        aggregation="sum",
        range=(0.0, float('inf')),
        higher_is_better=False
    )
    
    @classmethod
    def get_schema(cls, metric_name: str) -> Optional[MetricSchema]:
        """Get metric schema by name."""
        metric_map = {
            "accuracy": cls.ACCURACY,
            "clean_accuracy": cls.CLEAN_ACCURACY,
            "robust_accuracy": cls.ROBUST_ACCURACY,
            "attack_success_rate": cls.ATTACK_SUCCESS_RATE,
            "success_rate": cls.SUCCESS_RATE,
            "f1": cls.F1,
            "precision": cls.PRECISION,
            "cider": cls.CIDER,
            "vqa_accuracy": cls.VQA_ACCURACY,
            "loss": cls.LOSS,
            "training_time": cls.TRAINING_TIME
        }
        return metric_map.get(metric_name)
    
    @classmethod
    def all_schemas(cls) -> Dict[str, MetricSchema]:
        """Return all metric schemas."""
        return {
            "accuracy": cls.ACCURACY,
            "clean_accuracy": cls.CLEAN_ACCURACY,
            "robust_accuracy": cls.ROBUST_ACCURACY,
            "attack_success_rate": cls.ATTACK_SUCCESS_RATE,
            "success_rate": cls.SUCCESS_RATE,
            "f1": cls.F1,
            "precision": cls.PRECISION,
            "cider": cls.CIDER,
            "vqa_accuracy": cls.VQA_ACCURACY,
            "loss": cls.LOSS,
            "training_time": cls.TRAINING_TIME
        }


# ============================================================================
# Metric Computation Functions
# ============================================================================

def compute_accuracy(predictions: List[int], labels: List[int]) -> float:
    """
    Compute classification accuracy.
    
    Args:
        predictions: List of predicted class indices
        labels: List of ground truth class indices
    
    Returns:
        Accuracy as float in [0, 1]
    """
    if len(predictions) != len(labels) or len(predictions) == 0:
        return 0.0
    correct = sum(1 for pred, label in zip(predictions, labels) if pred == label)
    return correct / len(predictions)


def compute_robust_accuracy(
    model,
    images,
    labels: List[int],
    epsilon: float,
    attack_fn
) -> float:
    """
    Compute robust accuracy under adversarial attack.
    
    Args:
        model: Model to evaluate
        images: Input images (tensor or list)
        labels: Ground truth labels
        epsilon: Attack perturbation budget (e.g., 2/255, 4/255)
        attack_fn: Attack function that generates adversarial examples
    
    Returns:
        Robust accuracy as float in [0, 1]
    """
    try:
        adv_images = attack_fn(model, images, epsilon)
        adv_predictions = model(adv_images)
        pred_classes = adv_predictions.argmax(dim=1).cpu().tolist()
        return compute_accuracy(pred_classes, labels)
    except Exception:
        return 0.0


def compute_f1_score(predictions: List[int], labels: List[int]) -> float:
    """
    Compute F1 score.
    
    Args:
        predictions: List of predicted binary labels
        labels: List of ground truth binary labels
    
    Returns:
        F1 score as float in [0, 1]
    """
    if len(predictions) == 0 or len(labels) == 0:
        return 0.0
    
    true_positives = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
    false_positives = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
    false_negatives = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)
    
    if true_positives == 0:
        return 0.0
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)


def compute_precision(predictions: List[int], labels: List[int]) -> float:
    """
    Compute precision.
    
    Args:
        predictions: List of predicted binary labels
        labels: List of ground truth binary labels
    
    Returns:
        Precision as float in [0, 1]
    """
    if len(predictions) == 0:
        return 0.0
    
    true_positives = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
    false_positives = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
    
    if true_positives + false_positives == 0:
        return 0.0
    
    return true_positives / (true_positives + false_positives)


def compute_attack_success_rate(
    clean_predictions: List[int],
    adversarial_predictions: List[int],
    labels: List[int]
) -> float:
    """
    Compute attack success rate.
    
    An attack is successful if:
    - The clean prediction was correct
    - The adversarial prediction is incorrect
    
    Args:
        clean_predictions: Predictions on clean images
        adversarial_predictions: Predictions on adversarial images
        labels: Ground truth labels
    
    Returns:
        Attack success rate as float in [0, 1]
    """
    if len(clean_predictions) == 0:
        return 0.0
    
    successful_attacks = 0
    total_clean_correct = 0
    
    for clean_pred, adv_pred, label in zip(clean_predictions, adversarial_predictions, labels):
        if clean_pred == label:
            total_clean_correct += 1
            if adv_pred != label:
                successful_attacks += 1
    
    if total_clean_correct == 0:
        return 0.0
    
    return successful_attacks / total_clean_correct


def aggregate_metrics(metric_dict: Dict[str, List[float]], aggregation: str = "mean") -> Dict[str, float]:
    """
    Aggregate metrics across multiple samples.
    
    Args:
        metric_dict: Dictionary mapping metric names to lists of values
        aggregation: Aggregation method (mean, sum, max, min)
    
    Returns:
        Dictionary mapping metric names to aggregated values
    """
    import numpy as np
    
    aggregated = {}
    for metric_name, values in metric_dict.items():
        if len(values) == 0:
            aggregated[metric_name] = 0.0
            continue
        
        if aggregation == "mean":
            aggregated[metric_name] = float(np.mean(values))
        elif aggregation == "sum":
            aggregated[metric_name] = float(np.sum(values))
        elif aggregation == "max":
            aggregated[metric_name] = float(np.max(values))
        elif aggregation == "min":
            aggregated[metric_name] = float(np.min(values))
        else:
            aggregated[metric_name] = float(np.mean(values))
    
    return aggregated


# ============================================================================
# Result Artifact Writers
# ============================================================================

class ArtifactWriter:
    """Writer for result artifacts (tables, figures, metrics)."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or ArtifactPaths.RESULTS_DIR
        ArtifactPaths.ensure_directories()
    
    def write_table(
        self,
        data: List[Dict[str, Any]],
        output_path: Path,
        is_dry_run: bool = False
    ):
        """
        Write table data to CSV.
        
        Args:
            data: List of dictionaries representing table rows
            output_path: Output CSV path
            is_dry_run: If True, write schema artifact with label
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if is_dry_run:
            # Write schema artifact for dry-run
            if len(data) == 0:
                data = [{"column_1": "schema_value"}]
            schema_data = [{k: f"[dry-run schema] {type(v).__name__}" for k, v in data[0].items()}]
            data = schema_data + data[:min(2, len(data))]
        
        with open(output_path, 'w', newline='') as f:
            if len(data) > 0:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
    
    def write_metrics_json(
        self,
        metrics: Dict[str, Any],
        output_path: Optional[Path] = None,
        is_dry_run: bool = False
    ):
        """
        Write metrics to JSON.
        
        Args:
            metrics: Dictionary of metrics
            output_path: Output JSON path (defaults to METRICS_JSON)
            is_dry_run: If True, label as dry-run artifact
        """
        if output_path is None:
            output_path = ArtifactPaths.METRICS_JSON
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if is_dry_run:
            metrics["_artifact_type"] = "dry-run schema artifact"
            metrics["_note"] = "This is a contract validation artifact, not real experiment results"
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def write_predictions_jsonl(
        self,
        predictions: List[Dict[str, Any]],
        output_path: Optional[Path] = None,
        is_dry_run: bool = False
    ):
        """
        Write predictions to JSONL.
        
        Args:
            predictions: List of prediction dictionaries
            output_path: Output JSONL path (defaults to PREDICTIONS_JSONL)
            is_dry_run: If True, label as dry-run artifact
        """
        if output_path is None:
            output_path = ArtifactPaths.PREDICTIONS_JSONL
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if is_dry_run and len(predictions) > 0:
            predictions[0]["_artifact_type"] = "dry-run schema artifact"
        
        with open(output_path, 'w') as f:
            for pred in predictions:
                f.write(json.dumps(pred) + '\n')
    
    def write_figure(
        self,
        output_path: Path,
        is_dry_run: bool = False
    ):
        """
        Write figure artifact.
        
        Args:
            output_path: Output figure path
            is_dry_run: If True, create minimal diagnostic placeholder
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if is_dry_run:
            # Create minimal placeholder for dry-run
            try:
                from PIL import Image, ImageDraw, ImageFont
                img = Image.new('RGB', (800, 600), color='white')
                draw = ImageDraw.Draw(img)
                text = f"[Dry-run schema artifact]\n{output_path.name}"
                draw.text((50, 50), text, fill='black')
                img.save(output_path)
            except ImportError:
                # Fallback: write empty file
                output_path.touch()
    
    def write_readiness_manifest(self, config: Dict[str, Any]):
        """Write readiness.json for smoke validation."""
        manifest = {
            "artifact_type": "readiness manifest",
            "note": "Auxiliary smoke artifact - not real experiment results",
            "status": "smoke_validation_complete",
            "artifacts_declared": [str(p) for p in ArtifactPaths.all_artifacts()],
            "config": config
        }
        
        with open(ArtifactPaths.READINESS_JSON, 'w') as f:
            json.dump(manifest, f, indent=2)
    
    def write_evaluation_result(self, metrics: Dict[str, Any], is_dry_run: bool = False):
        """Write evaluation_result.json for smoke validation."""
        result = {
            "artifact_type": "evaluation result",
            "metrics": metrics
        }
        
        if is_dry_run:
            result["note"] = "Dry-run schema artifact - not real experiment results"
        
        with open(ArtifactPaths.EVALUATION_RESULT_JSON, 'w') as f:
            json.dump(result, f, indent=2)
    
    def write_all_smoke_artifacts(self, config: Dict[str, Any]):
        """
        Write all declared artifacts in dry-run/smoke mode.
        
        Creates schema/readiness artifacts for every declared path.
        """
        # Write all table schemas
        table_paths = [
            ArtifactPaths.TABLE_1, ArtifactPaths.TABLE_2, ArtifactPaths.TABLE_3,
            ArtifactPaths.TABLE_4, ArtifactPaths.TABLE_5, ArtifactPaths.TABLE_6,
            ArtifactPaths.TABLE_7, ArtifactPaths.TABLE_8, ArtifactPaths.TABLE_9,
            ArtifactPaths.TABLE_10, ArtifactPaths.TABLE_11, ArtifactPaths.TABLE_12,
            ArtifactPaths.TABLE_13, ArtifactPaths.TABLE_14,
            ArtifactPaths.EXPERIMENT_RESULTS_TABLE, ArtifactPaths.ROBUSTNESS_TABLE
        ]
        
        for table_path in table_paths:
            schema_data = [{"model": "schema", "metric": 0.0, "dataset": "schema"}]
            self.write_table(schema_data, table_path, is_dry_run=True)
        
        # Write all figure schemas
        figure_paths = [
            ArtifactPaths.FIGURE_1, ArtifactPaths.FIGURE_1_PDF, ArtifactPaths.FIGURE_2,
            ArtifactPaths.FIGURE_3, ArtifactPaths.FIGURE_4, ArtifactPaths.FIGURE_5,
            ArtifactPaths.EXPERIMENT_RESULTS_FIGURE
        ]
        
        for figure_path in figure_paths:
            self.write_figure(figure_path, is_dry_run=True)
        
        # Write metrics and predictions
        schema_metrics = {
            metric_name: 0.0 for metric_name in MetricRegistry.all_schemas().keys()
        }
        self.write_metrics_json(schema_metrics, is_dry_run=True)
        
        schema_predictions = [
            {"prediction": 0, "label": 0, "confidence": 0.0}
        ]
        self.write_predictions_jsonl(schema_predictions, is_dry_run=True)
        
        # Write config
        with open(ArtifactPaths.CONFIG_RESOLVED, 'w') as f:
            json.dump({
                "_artifact_type": "dry-run schema artifact",
                "config": config
            }, f, indent=2)
        
        # Write readiness and evaluation result
        self.write_readiness_manifest(config)
        self.write_evaluation_result(schema_metrics, is_dry_run=True)


# ============================================================================
# Paper-Specific Result Formatters
# ============================================================================

def format_table_4_results(results: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
    """
    Format results for Table 4: Clean and adversarial evaluation on image classification.
    
    Table 4 structure:
    - Rows: Model (CLIP, TeCoA^2, FARE^2, TeCoA^4, FARE^4)
    - Columns: Dataset, Clean Acc, Robust Acc (ε=2/255), Robust Acc (ε=4/255)
    
    Args:
        results: Nested dict {model: {dataset: {metric: value}}}
    
    Returns:
        List of row dictionaries for CSV writing
    """
    rows = []
    for model_name, model_results in results.items():
        for dataset_name, metrics in model_results.items():
            row = {
                "model": model_name,
                "dataset": dataset_name,
                "clean_accuracy": metrics.get("clean_accuracy", 0.0),
                "robust_accuracy_eps_2_255": metrics.get("robust_accuracy_2/255", 0.0),
                "robust_accuracy_eps_4_255": metrics.get("robust_accuracy_4/255", 0.0)
            }
            rows.append(row)
    return rows


def format_table_1_results(results: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
    """
    Format results for Table 1: LVLM robustness with different CLIP models.
    
    Table 1 structure:
    - Rows: Model (CLIP, TeCoA, FARE) × LVLM (LLaVA, OpenFlamingo)
    - Columns: Task, Clean Score, Robust Score (ε=2/255), Robust Score (ε=4/255)
    
    Args:
        results: Nested dict {lvlm: {model: {task: {metric: value}}}}
    
    Returns:
        List of row dictionaries for CSV writing
    """
    rows = []
    for lvlm_name, lvlm_results in results.items():
        for model_name, model_results in lvlm_results.items():
            for task_name, metrics in model_results.items():
                row = {
                    "lvlm": lvlm_name,
                    "model": model_name,
                    "task": task_name,
                    "clean_score": metrics.get("clean_score", 0.0),
                    "robust_score_eps_2_255": metrics.get("robust_score_2/255", 0.0),
                    "robust_score_eps_4_255": metrics.get("robust_score_4/255", 0.0)
                }
                rows.append(row)
    return rows


def format_table_7_results(results: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
    """
    Format results for Table 7: Jailbreaking attacks against LLaVA 1.5.
    
    Table 7 structure:
    - Rows: Model (CLIP, TeCoA, FARE)
    - Columns: Category, Attack Success Rate
    
    Args:
        results: Nested dict {model: {category: success_rate}}
    
    Returns:
        List of row dictionaries for CSV writing
    """
    rows = []
    for model_name, model_results in results.items():
        for category, success_rate in model_results.items():
            row = {
                "model": model_name,
                "category": category,
                "attack_success_rate": success_rate
            }
            rows.append(row)
    return rows


# ============================================================================
# Public Interface
# ============================================================================

def get_artifact_paths() -> ArtifactPaths:
    """Get artifact path registry."""
    return ArtifactPaths


def get_metric_registry() -> MetricRegistry:
    """Get metric schema registry."""
    return MetricRegistry


def create_artifact_writer(output_dir: Optional[Path] = None) -> ArtifactWriter:
    """Create artifact writer instance."""
    return ArtifactWriter(output_dir)


__all__ = [
    'ArtifactPaths',
    'MetricSchema',
    'MetricRegistry',
    'ArtifactWriter',
    'compute_accuracy',
    'compute_robust_accuracy',
    'compute_f1_score',
    'compute_precision',
    'compute_attack_success_rate',
    'aggregate_metrics',
    'format_table_4_results',
    'format_table_1_results',
    'format_table_7_results',
    'get_artifact_paths',
    'get_metric_registry',
    'create_artifact_writer'
]