#!/usr/bin/env python3
"""
LCA-on-the-Line Artifact Contract

Declares metric schemas, artifact paths, and writer interfaces for all paper figures and tables.
This file serves as the canonical artifact registry for reproducibility verification.

Paper: LCA-on-the-Line: Benchmarking Out-of-Distribution Generalization with Class Taxonomies

Metric schemas:
- LCA distance: Average Lowest Common Ancestor distance between prediction and ground truth
- Top-1/Top-5 accuracy: Standard classification accuracy metrics
- R²: Coefficient of determination from linear regression
- Spearman ρ: Spearman rank correlation coefficient
- MAE: Mean Absolute Error for prediction tasks

reference_grounding: paperbench_ref_005 eval_many_models.py
reference_grounding: paperbench_ref_006 eval_tiny_imagenet_truncate.ipynb
reference_grounding: paperbench_ref_001 test/test_models.py
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
import numpy as np


# =============================================================================
# Metric Schema Definitions
# =============================================================================

@dataclass
class MetricSchema:
    """Schema for a single metric."""
    name: str
    key: str
    description: str
    formula: str
    aggregation: str
    unit: str
    lower_is_better: bool
    paper_usage: str


METRIC_REGISTRY: Dict[str, MetricSchema] = {
    # Paper core metrics - LCA distance
    "lca_distance": MetricSchema(
        name="LCA Distance",
        key="lca_distance",
        description="Average Lowest Common Ancestor distance between model predictions and ground truth in WordNet hierarchy",
        formula="LCA(pred, true) = min_path_length(pred, true, hierarchy) averaged over all samples",
        aggregation="mean",
        unit="hops",
        lower_is_better=True,
        paper_usage="ID LCA distance serves as robust OOD error predictor (Table 1, 2, 3)"
    ),
    
    # Standard accuracy metrics
    "top1_accuracy": MetricSchema(
        name="Top-1 Accuracy",
        key="top1_acc",
        description="Fraction of samples where top prediction matches ground truth",
        formula="acc = (1/N) * sum(argmax(pred) == true)",
        aggregation="mean",
        unit="percentage",
        lower_is_better=False,
        paper_usage="Primary performance metric across all datasets (Table 1, 2, Figure 1, 5)"
    ),
    
    "top5_accuracy": MetricSchema(
        name="Top-5 Accuracy",
        key="top5_acc",
        description="Fraction of samples where ground truth is in top-5 predictions",
        formula="acc = (1/N) * sum(true in top5(pred))",
        aggregation="mean",
        unit="percentage",
        lower_is_better=False,
        paper_usage="Secondary performance metric (Table 2, Table 11)"
    ),
    
    # Correlation metrics
    "r_squared": MetricSchema(
        name="R² Coefficient",
        key="r2",
        description="Coefficient of determination from linear regression",
        formula="R² = 1 - SS_res / SS_tot where SS_res = sum((y - ŷ)²), SS_tot = sum((y - ȳ)²)",
        aggregation="scalar",
        unit="ratio",
        lower_is_better=False,
        paper_usage="Measures linearity of LCA-on-the-line phenomenon (Table 2, Table 11)"
    ),
    
    "spearman_rho": MetricSchema(
        name="Spearman ρ",
        key="spearman_rho",
        description="Spearman rank correlation coefficient",
        formula="ρ = 1 - (6 * sum(d²)) / (n * (n² - 1)) where d = rank difference",
        aggregation="scalar",
        unit="correlation",
        lower_is_better=False,
        paper_usage="Measures rank preservation between ID LCA and OOD accuracy (Table 2, Table 15)"
    ),
    
    "pearson_r": MetricSchema(
        name="Pearson r",
        key="pearson_r",
        description="Pearson correlation coefficient",
        formula="r = cov(X,Y) / (σ_X * σ_Y)",
        aggregation="scalar",
        unit="correlation",
        lower_is_better=False,
        paper_usage="Alternative correlation measure (Table 4)"
    ),
    
    # Error prediction metrics
    "mae": MetricSchema(
        name="Mean Absolute Error",
        key="mae",
        description="Mean absolute error for OOD accuracy prediction",
        formula="MAE = (1/N) * sum(|y_pred - y_true|)",
        aggregation="mean",
        unit="percentage_points",
        lower_is_better=True,
        paper_usage="Error predictor comparison metric (Table 3, Table 12)"
    ),
    
    # Generic training metrics
    "loss": MetricSchema(
        name="Loss",
        key="loss",
        description="Training loss (cross-entropy or combined)",
        formula="Loss = -sum(y * log(ŷ)) or CE + λ * LCA_soft_loss",
        aggregation="mean",
        unit="nats",
        lower_is_better=True,
        paper_usage="Training objective (Table 5, 6, 9)"
    ),
    
    "accuracy": MetricSchema(
        name="Generic Accuracy",
        key="accuracy",
        description="Generic classification accuracy",
        formula="acc = correct / total",
        aggregation="mean",
        unit="percentage",
        lower_is_better=False,
        paper_usage="General performance metric"
    ),
    
    # Specialized metrics
    "return": MetricSchema(
        name="Return",
        key="return",
        description="Cumulative return (for RL tasks, if applicable)",
        formula="R = sum(rewards)",
        aggregation="mean",
        unit="scalar",
        lower_is_better=False,
        paper_usage="Not directly used in paper (included for completeness)"
    ),
    
    "fidelity_score": MetricSchema(
        name="Fidelity Score",
        key="fidelity_score",
        description="Model agreement or distillation fidelity",
        formula="fidelity = agreement(model_A, model_B)",
        aggregation="mean",
        unit="percentage",
        lower_is_better=False,
        paper_usage="Soft label quality (Table 10, Figure 8)"
    ),
}


# =============================================================================
# Artifact Path Registry
# =============================================================================

@dataclass
class ArtifactSchema:
    """Schema for an output artifact."""
    artifact_id: str
    artifact_type: str  # figure, table, data, model
    output_path: str
    paper_reference: str
    description: str
    schema: Dict[str, Any]
    metrics: List[str]
    required: bool


ARTIFACT_REGISTRY: Dict[str, ArtifactSchema] = {
    # Main paper figures
    "figure_1": ArtifactSchema(
        artifact_id="figure_1",
        artifact_type="figure",
        output_path="results/figures/figure_1.png",
        paper_reference="Figure 1",
        description="Correlation between LCA distance and OOD performance (VMs and VLMs)",
        schema={
            "x_axis": "OOD Top-1 Accuracy (ObjectNet)",
            "y_axes": ["ID Top-1 Accuracy (ImageNet)", "ID LCA Distance (ImageNet)"],
            "panels": ["Vision Models", "Vision-Language Models"],
            "format": "png"
        },
        metrics=["top1_accuracy", "lca_distance"],
        required=True
    ),
    
    "figure_2": ArtifactSchema(
        artifact_id="figure_2",
        artifact_type="figure",
        output_path="results/figures/figure_2.png",
        paper_reference="Figure 2",
        description="Comparison with prior work (Accuracy-on-the-line, Agreement-on-the-line)",
        schema={
            "panels": ["Prior work settings", "Our setting"],
            "comparison": "Unified VM+VLM evaluation",
            "format": "png"
        },
        metrics=["top1_accuracy"],
        required=True
    ),
    
    "figure_3": ArtifactSchema(
        artifact_id="figure_3",
        artifact_type="figure",
        output_path="results/figures/figure_3.png",
        paper_reference="Figure 3",
        description="LCA distance visualization with hierarchy examples",
        schema={
            "visualization": "mistake severity ranking",
            "hierarchy": "WordNet taxonomy",
            "examples": ["class pairs", "LCA distances"],
            "format": "png"
        },
        metrics=["lca_distance"],
        required=True
    ),
    
    "figure_4": ArtifactSchema(
        artifact_id="figure_4",
        artifact_type="figure",
        output_path="results/figures/figure_4.png",
        paper_reference="Figure 4",
        description="Transferable features visualization (ImageNet-R)",
        schema={
            "features": ["shape", "texture", "color"],
            "dataset": "ImageNet-R",
            "examples": ["giraffe features", "elephant features"],
            "format": "png"
        },
        metrics=["top1_accuracy"],
        required=False
    ),
    
    "figure_5": ArtifactSchema(
        artifact_id="figure_5",
        artifact_type="figure",
        output_path="results/figures/figure_5.png",
        paper_reference="Figure 5",
        description="OOD Top-1/Top-5 accuracy correlation (75 models, 4 OOD datasets)",
        schema={
            "datasets": ["ImageNet-A", "ImageNet-R", "ImageNet-Sketch", "ObjectNet"],
            "metrics": ["Top-1 Accuracy", "Top-5 Accuracy"],
            "x_axis": "ID LCA Distance or ID Top-1 Accuracy",
            "y_axis": "OOD Accuracy",
            "models": 75,
            "format": "png"
        },
        metrics=["top1_accuracy", "top5_accuracy", "lca_distance", "r_squared", "spearman_rho"],
        required=True
    ),
    
    "figure_6": ArtifactSchema(
        artifact_id="figure_6",
        artifact_type="figure",
        output_path="results/figures/figure_6.png",
        paper_reference="Figure 6",
        description="K-means hierarchical clustering structure",
        schema={
            "method": "K-means clustering",
            "levels": ["K=1", "K=2", "K=4", "K=8", "...", "K=1000"],
            "visualization": "dendrogram",
            "format": "png"
        },
        metrics=["lca_distance"],
        required=True
    ),
    
    "figure_7": ArtifactSchema(
        artifact_id="figure_7",
        artifact_type="figure",
        output_path="results/figures/figure_7.png",
        paper_reference="Figure 7",
        description="Pairwise LCA distance matrix visualization",
        schema={
            "matrices": ["WordNet", "ResNet50", "CLIP-ViT-B/16"],
            "size": "1000x1000",
            "format": "png"
        },
        metrics=["lca_distance"],
        required=True
    ),
    
    "figure_8": ArtifactSchema(
        artifact_id="figure_8",
        artifact_type="figure",
        output_path="results/figures/figure_8.png",
        paper_reference="Figure 8",
        description="Source model generalization vs soft label quality",
        schema={
            "x_axis": "OOD Top-1 Accuracy (linear probe)",
            "y_axis": "ID LCA Distance (WordNet vs source hierarchy)",
            "models": 75,
            "format": "png"
        },
        metrics=["top1_accuracy", "lca_distance"],
        required=True
    ),
    
    "figure_9": ArtifactSchema(
        artifact_id="figure_9",
        artifact_type="figure",
        output_path="results/figures/figure_9.png",
        paper_reference="Figure 9",
        description="Same-dataset LCA prediction (75 models)",
        schema={
            "x_axis": "Dataset Top-1 Accuracy",
            "y_axis": "Dataset LCA Distance",
            "datasets": 6,
            "format": "png"
        },
        metrics=["top1_accuracy", "lca_distance"],
        required=False
    ),
    
    # Main paper tables
    "table_1": ArtifactSchema(
        artifact_id="table_1",
        artifact_type="table",
        output_path="results/tables/table_1.csv",
        paper_reference="Table 1",
        description="Model performance: LCA distance and Top-1 accuracy across datasets",
        schema={
            "columns": ["Model", "ImageNet_LCA", "ImageNet_Top1", "ImageNet-v2_LCA", "ImageNet-v2_Top1",
                       "ImageNet-A_LCA", "ImageNet-A_Top1", "ImageNet-R_LCA", "ImageNet-R_Top1",
                       "Sketch_LCA", "Sketch_Top1", "ObjectNet_LCA", "ObjectNet_Top1"],
            "rows": "Selected models from VMs and VLMs families",
            "format": "csv"
        },
        metrics=["lca_distance", "top1_accuracy"],
        required=True
    ),
    
    "table_2": ArtifactSchema(
        artifact_id="table_2",
        artifact_type="table",
        output_path="results/tables/table_2.csv",
        paper_reference="Table 2",
        description="Correlation measurement (R² and Spearman ρ) of ID LCA/Top1 with OOD Top1/Top5",
        schema={
            "columns": ["Predictor", "Dataset", "OOD_Metric", "R2", "Spearman_rho"],
            "predictors": ["ID LCA", "ID Top-1"],
            "datasets": ["ImageNet-v2", "ImageNet-A", "ImageNet-R", "Sketch", "ObjectNet"],
            "ood_metrics": ["Top-1", "Top-5"],
            "models": 75,
            "format": "csv"
        },
        metrics=["r_squared", "spearman_rho", "top1_accuracy", "top5_accuracy", "lca_distance"],
        required=True
    ),
    
    "table_3": ArtifactSchema(
        artifact_id="table_3",
        artifact_type="table",
        output_path="results/tables/table_3.csv",
        paper_reference="Table 3",
        description="Error prediction (MAE) of OOD datasets across 75 models",
        schema={
            "columns": ["Method", "ImageNet-v2_MAE", "ImageNet-A_MAE", "ImageNet-R_MAE",
                       "Sketch_MAE", "ObjectNet_MAE"],
            "methods": ["ID Top-1 (Ours baseline)", "ID LCA (Ours)", "Aline-S", "Aline-D"],
            "models": 75,
            "format": "csv"
        },
        metrics=["mae"],
        required=True
    ),
    
    "table_4": ArtifactSchema(
        artifact_id="table_4",
        artifact_type="table",
        output_path="results/tables/table_4.csv",
        paper_reference="Table 4",
        description="Correlation (Pearson) between ID LCA/Top1 and OOD Top1 across 75 latent hierarchies",
        schema={
            "columns": ["Source_Model", "Predictor", "ImageNet-A", "ImageNet-R", "Sketch", "ObjectNet"],
            "latent_hierarchies": 75,
            "format": "csv"
        },
        metrics=["pearson_r", "top1_accuracy", "lca_distance"],
        required=True
    ),
    
    "table_5": ArtifactSchema(
        artifact_id="table_5",
        artifact_type="table",
        output_path="results/tables/table_5.csv",
        paper_reference="Table 5",
        description="Soft labeling with WordNet for linear probing",
        schema={
            "columns": ["Base_Model", "Method", "ImageNet", "ImageNet-v2", "ImageNet-A",
                       "ImageNet-R", "Sketch", "ObjectNet"],
            "methods": ["Baseline (CE only)", "Ours (CE + LCA soft loss + interpolation)"],
            "base_models": ["ResNet-18", "ResNet-50", "ViT-B/16", "ConvNeXt-B", "Swin-B"],
            "format": "csv"
        },
        metrics=["top1_accuracy"],
        required=True
    ),
    
    "table_6": ArtifactSchema(
        artifact_id="table_6",
        artifact_type="table",
        output_path="results/tables/table_6.csv",
        paper_reference="Table 6",
        description="Soft labeling with latent hierarchies for linear probing (ResNet-18)",
        schema={
            "columns": ["Source_Model", "Method", "ImageNet", "ImageNet-v2", "ImageNet-A",
                       "ImageNet-R", "Sketch", "ObjectNet"],
            "source_models": ["ResNet-50", "ViT-B/16", "CLIP-ViT-B/16"],
            "methods": ["Baseline", "Latent Hierarchy"],
            "format": "csv"
        },
        metrics=["top1_accuracy"],
        required=True
    ),
    
    "table_7": ArtifactSchema(
        artifact_id="table_7",
        artifact_type="table",
        output_path="results/tables/table_7.csv",
        paper_reference="Table 7",
        description="Simulation study: ID vs OOD error for good/bad models",
        schema={
            "columns": ["Model_Type", "ID_Top1_Error", "ID_LCA_Distance", "OOD_Top1_Error"],
            "model_types": ["Good (generalizable)", "Bad (non-generalizable)"],
            "trials": 100,
            "format": "csv"
        },
        metrics=["top1_accuracy", "lca_distance"],
        required=False
    ),
    
    "table_9": ArtifactSchema(
        artifact_id="table_9",
        artifact_type="table",
        output_path="results/tables/table_9.csv",
        paper_reference="Table 9",
        description="Ablation study on soft loss labels for linear probing",
        schema={
            "columns": ["Base_Model", "CE_only", "Soft_Loss", "Interpolation", "No_ID_Acc_Drop",
                       "ImageNet", "ImageNet-v2", "ImageNet-A", "ImageNet-R", "Sketch", "ObjectNet"],
            "ablations": ["CE only", "+ Soft Loss", "+ Interpolation", "+ No ID Acc Drop"],
            "format": "csv"
        },
        metrics=["top1_accuracy"],
        required=True
    ),
    
    "table_10": ArtifactSchema(
        artifact_id="table_10",
        artifact_type="table",
        output_path="results/tables/table_10.csv",
        paper_reference="Table 10",
        description="Source model generalization vs soft label quality",
        schema={
            "columns": ["Source_Model", "Source_OOD_Performance", "Soft_Label_Quality",
                       "Target_OOD_Performance"],
            "source_models": 75,
            "format": "csv"
        },
        metrics=["top1_accuracy", "lca_distance"],
        required=True
    ),
    
    "table_11": ArtifactSchema(
        artifact_id="table_11",
        artifact_type="table",
        output_path="results/tables/table_11.csv",
        paper_reference="Table 11",
        description="Correlation by modality (VM/VLM) - extended Table 2",
        schema={
            "columns": ["Modality", "Predictor", "Dataset", "OOD_Metric", "R2", "Spearman_rho"],
            "modalities": ["VM only (36)", "VLM only (39)", "ALL (75)"],
            "format": "csv"
        },
        metrics=["r_squared", "spearman_rho"],
        required=True
    ),
    
    "table_12": ArtifactSchema(
        artifact_id="table_12",
        artifact_type="table",
        output_path="results/tables/table_12.csv",
        paper_reference="Table 12",
        description="Error prediction MAE - extended Table 3",
        schema={
            "columns": ["Method", "ImageNet-v2", "ImageNet-A", "ImageNet-R", "Sketch", "ObjectNet"],
            "methods": ["ID Top-1", "ID LCA", "Aline-S", "Aline-D"],
            "format": "csv"
        },
        metrics=["mae"],
        required=True
    ),
    
    "table_13": ArtifactSchema(
        artifact_id="table_13",
        artifact_type="table",
        output_path="results/tables/table_13.csv",
        paper_reference="Table 13",
        description="Same-dataset correlation between Top-1 and LCA",
        schema={
            "columns": ["Dataset", "Modality", "R2", "Spearman_rho"],
            "datasets": 6,
            "modalities": ["VM", "VLM", "ALL"],
            "format": "csv"
        },
        metrics=["r_squared", "spearman_rho"],
        required=False
    ),
    
    "table_14": ArtifactSchema(
        artifact_id="table_14",
        artifact_type="table",
        output_path="results/tables/table_14.csv",
        paper_reference="Table 14",
        description="VLM prompting with taxonomy hierarchy",
        schema={
            "columns": ["Prompt_Type", "ImageNet", "ImageNet-v2", "ImageNet-A", "ImageNet-R",
                       "Sketch", "ObjectNet"],
            "prompt_types": ["Baseline (<dalmatian>)", "Stack Parent", "Taxonomy Parent", "Shuffle Parent"],
            "format": "csv"
        },
        metrics=["top1_accuracy"],
        required=True
    ),
    
    "table_15": ArtifactSchema(
        artifact_id="table_15",
        artifact_type="table",
        output_path="results/tables/table_15.csv",
        paper_reference="Table 15",
        description="Ranking measurement - extended analysis",
        schema={
            "columns": ["Predictor", "Dataset", "Kendall_tau", "Spearman_rho"],
            "format": "csv"
        },
        metrics=["spearman_rho"],
        required=False
    ),
    
    # Core result artifacts
    "lca_ood_correlation": ArtifactSchema(
        artifact_id="lca_ood_correlation",
        artifact_type="data",
        output_path="results/lca_ood_correlation.json",
        paper_reference="Core result",
        description="ID LCA distance vs OOD accuracy correlations (all models, all datasets)",
        schema={
            "structure": {
                "models": "list of 75 model names",
                "datasets": ["imagenet", "imagenet-v2", "imagenet-a", "imagenet-r", "sketch", "objectnet"],
                "metrics_per_model_dataset": ["lca_distance", "top1_acc", "top5_acc"],
                "correlations": {
                    "dataset": {
                        "id_lca_vs_ood_top1": {"r2": float, "spearman_rho": float, "mae": float},
                        "id_top1_vs_ood_top1": {"r2": float, "spearman_rho": float, "mae": float}
                    }
                }
            }
        },
        metrics=["lca_distance", "top1_accuracy", "top5_accuracy", "r_squared", "spearman_rho", "mae"],
        required=True
    ),
    
    "correlations": ArtifactSchema(
        artifact_id="correlations",
        artifact_type="data",
        output_path="results/correlations.json",
        paper_reference="Analysis result",
        description="Comprehensive correlation analysis results",
        schema={
            "structure": {
                "linear_fit": {"slope": float, "intercept": float, "r2": float},
                "rank_correlation": {"spearman_rho": float, "p_value": float},
                "error_prediction": {"mae": float, "rmse": float}
            }
        },
        metrics=["r_squared", "spearman_rho", "mae"],
        required=True
    ),
    
    "latent_hierarchies": ArtifactSchema(
        artifact_id="latent_hierarchies",
        artifact_type="data",
        output_path="results/latent_hierarchies",
        paper_reference="Section 4.3",
        description="K-means constructed hierarchies from pretrained models",
        schema={
            "structure": "directory with 75 model hierarchy files",
            "file_format": "json or npz",
            "content": "LCA distance matrices (1000x1000)"
        },
        metrics=["lca_distance"],
        required=True
    ),
    
    # Auxiliary artifacts
    "metrics_json": ArtifactSchema(
        artifact_id="metrics_json",
        artifact_type="data",
        output_path="results/metrics.json",
        paper_reference="Evaluation output",
        description="Aggregated metrics across all models and datasets",
        schema={
            "structure": {
                "model_name": {
                    "dataset_name": {
                        "top1_acc": float,
                        "top5_acc": float,
                        "lca_distance": float
                    }
                }
            }
        },
        metrics=["top1_accuracy", "top5_accuracy", "lca_distance"],
        required=True
    ),
    
    "predictions": ArtifactSchema(
        artifact_id="predictions",
        artifact_type="data",
        output_path="results/predictions.jsonl",
        paper_reference="Evaluation output",
        description="Per-sample predictions and LCA distances",
        schema={
            "format": "jsonl (one JSON object per line)",
            "fields": ["model", "dataset", "sample_id", "prediction", "ground_truth", "lca_distance", "correct"]
        },
        metrics=["lca_distance", "accuracy"],
        required=False
    ),
    
    "config_resolved": ArtifactSchema(
        artifact_id="config_resolved",
        artifact_type="data",
        output_path="results/config_resolved.json",
        paper_reference="Reproducibility",
        description="Resolved configuration used for experiment",
        schema={
            "structure": {
                "models": "list",
                "datasets": "list",
                "hierarchy_source": "string",
                "evaluation_settings": "dict"
            }
        },
        metrics=[],
        required=False
    ),
}


# =============================================================================
# Artifact Writer Interface
# =============================================================================

class ArtifactWriter:
    """Interface for writing experiment artifacts."""
    
    def __init__(self, output_dir: str = "results"):
        """Initialize artifact writer.
        
        Args:
            output_dir: Base directory for all outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "figures").mkdir(exist_ok=True)
        (self.output_dir / "tables").mkdir(exist_ok=True)
        (self.output_dir / "latent_hierarchies").mkdir(exist_ok=True)
    
    def write_metric(self, metric_key: str, value: Union[float, int], 
                     context: Optional[Dict[str, Any]] = None) -> None:
        """Write a single metric value.
        
        Args:
            metric_key: Metric identifier from METRIC_REGISTRY
            value: Metric value
            context: Additional context (model, dataset, etc.)
        """