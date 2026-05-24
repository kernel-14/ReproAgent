#!/usr/bin/env python3
"""
Artifact contract module for Test-Time Model Adaptation with Only Forward Passes.

This module implements the evidence obligation matrix registry, experiment registry,
parameter sweep config, and artifact writer interfaces as required by the paper
reproduction contract.

Satisfies method obligations:
- Paper artifact context for all tables and figures
- Measurement schemas and aggregation outputs for all metrics
- Statically discoverable result artifact paths
- Writer/declaration hooks for all paper artifacts
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import datetime


# ==============================================================================
# Metric Schemas
# ==============================================================================

METRIC_SCHEMAS = {
    "accuracy": {
        "name": "accuracy",
        "unit": "percentage",
        "range": [0.0, 100.0],
        "higher_is_better": True,
        "aggregation": "mean",
        "precision": 2,
        "description": "Classification accuracy on test samples"
    },
    "precision": {
        "name": "precision",
        "unit": "percentage",
        "range": [0.0, 100.0],
        "higher_is_better": True,
        "aggregation": "mean",
        "precision": 2,
        "description": "Precision metric for classification"
    },
    "loss": {
        "name": "loss",
        "unit": "scalar",
        "range": [0.0, float("inf")],
        "higher_is_better": False,
        "aggregation": "mean",
        "precision": 4,
        "description": "Cross-entropy or other loss value"
    },
    "training_time": {
        "name": "training_time",
        "unit": "seconds",
        "range": [0.0, float("inf")],
        "higher_is_better": False,
        "aggregation": "sum",
        "precision": 2,
        "description": "Wall-clock training or adaptation time"
    },
    "ece": {
        "name": "ece",
        "unit": "percentage",
        "range": [0.0, 100.0],
        "higher_is_better": False,
        "aggregation": "mean",
        "precision": 2,
        "description": "Expected Calibration Error"
    },
    "memory_usage": {
        "name": "memory_usage",
        "unit": "megabytes",
        "range": [0.0, float("inf")],
        "higher_is_better": False,
        "aggregation": "max",
        "precision": 1,
        "description": "Peak GPU memory usage (runtime and cached by PyTorch allocator)",
        "note": "Higher values observed via nvidia-smi include cached unused memory"
    }
}


# ==============================================================================
# Artifact Path Registry
# ==============================================================================

ARTIFACT_PATHS = {
    # Tables
    "table_1": "results/tables/table_1.csv",
    "table_2": "results/tables/table_2.csv",
    "table_3": "results/tables/table_3.csv",
    "table_4": "results/tables/table_4.csv",
    "table_5": "results/tables/table_5.csv",
    "table_6": "results/tables/table_6.csv",
    "table_7": "results/tables/table_7.csv",
    "table_8": "results/tables/table_8.csv",
    "table_9": "results/tables/table_9.csv",
    "table_10": "results/tables/table_10.csv",
    "table_11": "results/tables/table_11.csv",
    "table_12": "results/tables/table_12.csv",
    "table_13": "results/tables/table_13.csv",
    "table_14": "results/tables/table_14.csv",
    "table_15": "results/tables/table_15.csv",
    "table_16": "results/tables/table_16.csv",
    "table_17": "results/tables/table_17.csv",
    
    # Figures
    "figure_1": "results/figures/figure_1.png",
    "figure_2": "results/figures/figure_2.png",
    "figure_3": "results/figures/figure_3.png",
    "figure_4": "results/figures/figure_4.png",
    
    # Generic outputs
    "experiment_results": "results/tables/experiment_results.csv",
    "experiment_figure": "results/figures/experiment_results.png",
    "predictions": "results/predictions.jsonl",
    "metrics_json": "results/metrics.json",
    "config_resolved": "results/config_resolved.json",
    
    # Contract artifacts
    "evidence_contract_matrix": "results/evidence_contract_matrix.json",
    "experiment_registry": "results/experiment_registry.json",
    "environment_registry": "results/environment_registry.json",
    "dataset_registry": "results/dataset_registry.json",
    "artifact_manifest": "results/artifact_manifest.json",
    "readiness": "results/readiness.json",
    "evaluation_result": "results/evaluation_result.json"
}


# ==============================================================================
# Paper Artifact Metadata
# ==============================================================================

PAPER_ARTIFACTS = {
    "table_1": {
        "caption": "Comparison w.r.t. prior gradient-based Test-Time Adaptation (TTA) vs. our Forward-Optimization Adaptation. Memory usage and accuracy measured via ViT-Base, batch size 64 on ImageNet-C (level 5). Memory of 8-bit ViT is ideal estimation by 0.25x memory of 32-bit ViT.",
        "baselines": ["NoAdapt", "TENT", "CoTTA", "SAR", "T3A", "LAME", "FOA (Ours)"],
        "metrics": ["accuracy", "memory_usage"],
        "comparison_semantics": "FOA achieves competitive accuracy with 50-75% lower memory vs gradient-based methods",
        "datasets": ["ImageNet-C"],
        "model": "ViT-Base",
        "batch_size": 64,
        "corruption_level": 5
    },
    "table_2": {
        "caption": "Comparisons with SOTA methods on ImageNet-C (severity level 5) with ViT regarding Accuracy (%). BP is short for backward propagation and the bold number indicates the best result. Average ECE (%) reported.",
        "baselines": ["NoAdapt", "LAME", "T3A", "TENT", "CoTTA", "SAR", "FOA (Ours)"],
        "metrics": ["accuracy", "ece"],
        "comparison_semantics": "FOA achieves best average accuracy and ECE over 15 corruption types",
        "datasets": ["ImageNet-C"],
        "model": "ViT-Base",
        "corruption_level": 5,
        "corruption_types": 15
    },
    "table_3": {
        "caption": "Comparisons with state-of-the-art methods on ImageNetR/V2/Sketch with ViT-Base. BP is short for backward propagation and the bold number indicates the best result.",
        "baselines": ["NoAdapt", "LAME", "T3A", "TENT", "CoTTA", "SAR", "FOA (Ours)"],
        "metrics": ["accuracy"],
        "comparison_semantics": "FOA generalizes to diverse distribution shifts beyond corruptions",
        "datasets": ["ImageNet-R", "ImageNet-V2", "ImageNet-Sketch"],
        "model": "ViT-Base"
    },
    "table_4": {
        "caption": "Effectiveness of our FOA on Quantized ViT models. Corruption Accuracy (%) and average ECE (%) on ImageNet-C (severity level 5). Bold number indicates best result.",
        "baselines": ["T3A", "FOA (Ours)"],
        "metrics": ["accuracy", "ece"],
        "comparison_semantics": "FOA outperforms T3A significantly on 8-bit and 6-bit quantized models",
        "datasets": ["ImageNet-C"],
        "model": "ViT-Base",
        "quantization": ["8-bit", "6-bit"],
        "corruption_level": 5
    },
    "table_5": {
        "caption": "Ablations of components in our FOA. Entropy and Activation Discrepancy are fitness function components. Act. Shifting is back-to-source method. Average results over 15 corruptions on ImageNet-C (level 5) with ViT-Base.",
        "baselines": ["FOA variants"],
        "metrics": ["accuracy", "ece"],
        "comparison_semantics": "Ablation study showing importance of each FOA component",
        "datasets": ["ImageNet-C"],
        "model": "ViT-Base",
        "corruption_level": 5,
        "ablation_dimensions": ["entropy", "activation_discrepancy", "activation_shifting"]
    },
    "table_6": {
        "caption": "Effectiveness of FOA with interval update strategy (different intervals I), termed FOA-I, for single sample adaptation. Results on ImageNet-C (Gaussian, level 5) with ViT-Base.",
        "baselines": ["FOA-I variants"],
        "metrics": ["accuracy"],
        "comparison_semantics": "Interval update enables single-sample adaptation",
        "datasets": ["ImageNet-C"],
        "model": "ViT-Base",
        "corruption": "Gaussian",
        "corruption_level": 5,
        "parameter_sweep": "interval_I"
    },
    "table_7": {
        "caption": "Comparison w.r.t. run-time memory (MB) usage. Results via ViT-Base (32/8-bit) on ImageNet-C (Gaussian, level 5). FOA-I V1/V2 denote storing features/images for interval update under batch size 1. Memory for 8-bit ViT is ideal estimation by 0.25x memory of 32-bit ViT.",
        "baselines": ["NoAdapt", "TENT", "T3A", "FOA", "FOA-I V1", "FOA-I V2"],
        "metrics": ["memory_usage"],
        "comparison_semantics": "FOA exhibits marginally higher memory than NoAdapt but much lower than TENT",
        "datasets": ["ImageNet-C"],
        "model": "ViT-Base",
        "quantization": ["32-bit", "8-bit"],
        "corruption": "Gaussian",
        "corruption_level": 5,
        "note": "V1 stores features, V2 stores images between updates"
    },
    "table_8": {
        "caption": "Comparisons w.r.t. computation complexity. FP/BP is short forward/backward propagation. #FP and #BP counted per single sample. Accuracy (%) and ECE (%) average on ImageNet-C (level 5) with ViT-Base. Wall-Clock Time (seconds) and Memory Usage (MB) measured.",
        "baselines": ["NoAdapt", "TENT", "T3A", "FOA (Ours)"],
        "metrics": ["accuracy", "ece", "training_time", "memory_usage"],
        "comparison_semantics": "FOA requires more forward passes but no backward passes, reducing memory and boosting efficiency",
        "datasets": ["ImageNet-C"],
        "model": "ViT-Base",
        "corruption_level": 5,
        "computational_metrics": ["forward_passes", "backward_passes", "wall_clock_time"]
    },
    "table_9": {
        "caption": "Empirical studies of design choices w.r.t. learnable parameters, optimizer and loss function. Average results over 15 corruptions on ImageNet-C (level 5) with ViT-Base.",
        "baselines": ["Design variants"],
        "metrics": ["accuracy", "ece"],
        "comparison_semantics": "CMA-ES with prompt learning outperforms SGD and other parameter choices",
        "datasets": ["ImageNet-C"],
        "model": "ViT-Base",
        "corruption_level": 5,
        "design_dimensions": ["learnable_parameters", "optimizer", "loss_function"]
    },
    "table_10": {
        "caption": "Effectiveness of FOA on ResNet and VisionMamba. Results on ImageNet-C (Gaussian noise, level 5). FOA† modified from FOA by replacing CMA with SGD and updating affine parameters of norm layers.",
        "baselines": ["NoAdapt", "TENT", "FOA", "FOA†"],
        "metrics": ["accuracy"],
        "comparison_semantics": "FOA generalizes to different architectures beyond ViT",
        "datasets": ["ImageNet-C"],
        "models": ["ResNet", "VisionMamba"],
        "corruption": "Gaussian",
        "corruption_level": 5
    },
    "table_11": {
        "caption": "Effectiveness of FOA under non-i.i.d. scenarios. Results on ViT and ImageNet-C (level 5). For mild (i.i.d.) and online imbalanced label shift scenarios, average over 15 corruptions. For mixed shifts, performance on single data stream of 15 mixed corruptions.",
        "baselines": ["NoAdapt", "TENT", "CoTTA", "SAR", "FOA (Ours)"],
        "metrics": ["accuracy"],
        "comparison_semantics": "FOA robust to non-i.i.d. scenarios and label shifts",
        "datasets": ["ImageNet-C"],
        "model": "ViT-Base",
        "corruption_level": 5,
        "scenarios": ["mild_iid", "online_imbalanced_label_shift", "mixed_shifts"]
    },
    "table_12": {
        "caption": "Comparison w.r.t. in-distribution performance, i.e., on clean/original ImageNet validation set, with ViT as base model.",
        "baselines": ["NoAdapt", "LAME", "T3A", "TENT", "CoTTA", "SAR", "FOA (Ours)"],
        "metrics": ["accuracy"],
        "comparison_semantics": "FOA maintains almost same in-distribution accuracy as NoAdapt, outperforming other methods",
        "datasets": ["ImageNet"],
        "model": "ViT-Base",
        "note": "Success from: 1) no parameter modification, 2) regularizing features back to source"
    },
    "table_13": {
        "caption": "Sensitivity analyses regarding trade-off parameter λ (see Eqn. 5) in FOA. Results on ImageNet-C (Gaussian noise, severity level 5) using ViT-Base with batch size 64.",
        "baselines": ["FOA variants"],
        "metrics": ["accuracy", "ece"],
        "comparison_semantics": "λ=0.4 provides good balance between entropy and activation discrepancy",
        "datasets": ["ImageNet-C"],
        "model": "ViT-Base",
        "corruption": "Gaussian",
        "corruption_level": 5,
        "batch_size": 64,
        "parameter_sweep": "lambda",
        "lambda_values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    },
    "table_14": {
        "caption": "Effects of exponential moving average (EMA) (Eqn. 9) in Back-to-Source Activation Shifting scheme. For Act. Shifting w/o EMA, directly utilize batch statistics μ_N(X_t) to calculate shifting direction d_t in Eqn. 8. Average results over 15 corruptions on ImageNet-C (level 5) with ViT-Base.",
        "baselines": ["FOA w/ EMA", "FOA w/o EMA"],
        "metrics": ["accuracy", "ece"],
        "comparison_semantics": "EMA stabilizes activation shifting",
        "datasets": ["ImageNet-C"],
        "model": "ViT-Base",
        "corruption_level": 5
    },
    "table_15": {
        "caption": "Effects of exponential moving average (EMA) in calculating Activation Discrepancy fitness in Eqn. 5. Replace μ_i(X_t) with β*μ_i(X_t)+(1-β)*μ_i(t-1) and σ_i similarly. Average results over 15 corruptions on ImageNet-C (level 5) with ViT-Base.",
        "baselines": ["FOA w/ EMA", "FOA w/o EMA"],
        "metrics": ["accuracy", "ece"],
        "comparison_semantics": "EMA in fitness calculation improves stability",
        "datasets": ["ImageNet-C"],
        "model": "ViT-Base",
        "corruption_level": 5
    },
    "table_16": {
        "caption": "Comparisons with state-of-the-art methods on ImageNet-C (severity level 5) with ViT-Base regarding ECE (%). BP is short for backward propagation and the bold number indicates the best result.",
        "baselines": ["NoAdapt", "LAME", "T3A", "TENT", "CoTTA", "SAR", "FOA (Ours)"],
        "metrics": ["ece"],
        "comparison_semantics": "Detailed ECE results showing FOA calibration performance",
        "datasets": ["ImageNet-C"],
        "model": "ViT-Base",
        "corruption_level": 5,
        "corruption_types": 15
    },
    "table_17": {
        "caption": "Effectiveness of our FOA on Quantized ViT-Base models. Corruption ECE (%) on ImageNet-C (severity level 5). Bold number indicates best result.",
        "baselines": ["T3A", "FOA (Ours)"],
        "metrics": ["ece"],
        "comparison_semantics": "FOA consistently outperforms on quantized models for calibration",
        "datasets": ["ImageNet-C"],
        "model": "ViT-Base",
        "quantization": ["8-bit", "6-bit"],
        "corruption_level": 5
    },
    "figure_1": {
        "caption": "Illustration of proposed FOA. For each batch of online incoming test samples, feed them alongside prompts p into TTA model, calculate fitness value as learning signal for CMA optimizer in learning prompts p.",
        "content": "Method illustration diagram",
        "type": "diagram"
    },
    "figure_2": {
        "caption": "Parameter sensitivity analyses of our FOA. Experiments on ImageNet-C (Gaussian Noise, level 5) with ViT-Base.",
        "content": "Sensitivity plots for population size K, prompt count, etc.",
        "type": "line_plot",
        "datasets": ["ImageNet-C"],
        "model": "ViT-Base",
        "corruption": "Gaussian",
        "corruption_level": 5,
        "parameters": ["population_size_K", "prompt_count", "other_hyperparameters"]
    },
    "figure_3": {
        "caption": "Visualizations of images in ImageNet and ImageNet-C/V2/R/Sketch, directly taken from original papers.",
        "content": "Dataset sample visualizations",
        "type": "image_grid"
    },
    "figure_4": {
        "caption": "Online accuracy comparison with MEMO on ViT and ImageNet-C (Gaussian noise, severity level 5).",
        "content": "Online adaptation accuracy over number of test samples",
        "type": "line_plot",
        "datasets": ["ImageNet-C"],
        "model": "ViT-Base",
        "corruption": "Gaussian",
        "corruption_level": 5,
        "baselines": ["MEMO", "FOA (Ours)"]
    }
}


# ==============================================================================
# Evidence Contract Matrix
# ==============================================================================

def get_evidence_contract_matrix() -> Dict[str, Any]:
    """
    Build evidence obligation matrix binding experiments to:
    - environments/datasets/tasks
    - methods/baselines
    - parameter sweep values
    - expected trends and decision claims
    - result artifacts
    """
    matrix = {
        "version": "1.0.0",
        "paper_title": "Test-Time Model Adaptation with Only Forward Passes",
        "timestamp": datetime.datetime.now().isoformat(),
        "experiments": []
    }
    
    # Main comparison experiment (Table 1, 2)
    matrix["experiments"].append({
        "experiment_id": "main_comparison",
        "name": "Main FOA vs Gradient-based TTA Comparison",
        "datasets": ["ImageNet-C"],
        "environments": ["imagenet_c_level_5"],
        "methods": ["NoAdapt", "TENT", "CoTTA", "SAR", "T3A", "LAME", "FOA"],
        "models": ["ViT-Base"],
        "metrics": ["accuracy", "ece", "memory_usage"],
        "parameters": {"batch_size": 64, "corruption_level": 5},
        "expected_trend": "FOA achieves competitive accuracy with 50-75% lower memory usage",
        "decision_claim": "Forward-only adaptation viable alternative to gradient-based TTA",
        "artifacts": ["table_1", "table_2"]
    })
    
    # Generalization experiment (Table 3)
    matrix["experiments"].append({
        "experiment_id": "generalization",
        "name": "Generalization to ImageNet-R/V2/Sketch",
        "datasets": ["ImageNet-R", "ImageNet-V2", "ImageNet-Sketch"],
        "environments": ["imagenet_variants"],
        "methods": ["NoAdapt", "LAME", "T3A", "TENT", "CoTTA", "SAR", "FOA"],
        "models": ["ViT-Base"],
        "metrics": ["accuracy"],
        "parameters": {},
        "expected_trend": "FOA maintains performance across diverse distribution shifts",
        "decision_claim": "FOA not limited to corruption robustness",
        "artifacts": ["table_3"]
    })
    
    # Quantization experiment (Table 4, 17)
    matrix["experiments"].append({
        "experiment_id": "quantization",
        "name": "FOA on Quantized Models",
        "datasets": ["ImageNet-C"],
        "environments": ["imagenet_c_level_5"],
        "methods": ["T3A", "FOA"],
        "models": ["ViT-Base-8bit", "ViT-Base-6bit"],
        "metrics": ["accuracy", "ece"],
        "parameters": {"corruption_level": 5, "quantization": ["8-bit", "6-bit"]},
        "expected_trend": "FOA significantly outperforms T3A on quantized models",
        "decision_claim": "FOA adaptable to quantized models where gradient methods fail",
        "artifacts": ["table_4", "table_17"]
    })
    
    # Ablation experiment (Table 5)
    matrix["experiments"].append({
        "experiment_id": "ablation",
        "name": "Component Ablation Study",
        "datasets": ["ImageNet-C"],
        "environments": ["imagenet_c_level_5"],
        "methods": ["FOA", "FOA-no-entropy", "FOA-no-act-discrepancy", "FOA-no-act-shifting"],
        "models": ["ViT-Base"],
        "metrics": ["accuracy", "ece"],
        "parameters": {"corruption_level": 5},
        "expected_trend": "All components contribute to performance",
        "decision_claim": "Both fitness components and activation shifting are necessary",
        "artifacts": ["table_5"]
    })
    
    # Interval update experiment (Table 6, 7)
    matrix["experiments"].append({
        "experiment_id": "interval_update",
        "name": "FOA-I for Single Sample Adaptation",
        "datasets": ["ImageNet-C"],
        "environments": ["imagenet_c_gaussian_level_5"],
        "methods": ["FOA-I"],
        "models": ["ViT-Base"],
        "metrics": ["accuracy", "memory_usage"],
        "parameters": {"corruption": "Gaussian", "corruption_level": 5, "batch_size": 1, "interval_sweep": [1, 2, 4, 8, 16]},
        "expected_trend": "Interval update enables single-sample adaptation with controlled memory",
        "decision_claim": "FOA-I V1/V2 strategies trade off memory for single-sample scenarios",
        "artifacts": ["table_6", "table_7"]
    })
    
    # Computational complexity experiment (Table 8)
    matrix["experiments"].append({
        "experiment_id": "computational_complexity",
        "name": "Computational Complexity Analysis",
        "datasets": ["ImageNet-C"],
        "environments": ["imagenet_c_level_5"],
        "methods": ["NoAdapt", "TENT", "T3A", "FOA"],
        "models": ["ViT-Base"],
        "metrics": ["accuracy", "ece", "training_time", "memory_usage"],
        "parameters": {"corruption_level": 5},
        "expected_trend": "FOA trades more forward passes for no backward passes, net efficiency gain",
        "decision_claim": "FOA reduces memory and wall-clock time despite more forward passes",
        "artifacts": ["table_8"]
    })
    
    # Design choices experiment (Table 9)
    matrix["experiments"].append({
        "experiment_id": "design_choices",
        "name": "Empirical Design Choice Study",
        "datasets": ["ImageNet-C"],
        "environments": ["imagenet_c_level_5"],
        "methods": ["FOA-variants"],
        "models": ["ViT-Base"],
        "metrics": ["accuracy", "ece"],
        "parameters": {"corruption_level": 5, "design_dimensions": ["learnable_parameters", "optimizer", "loss_function"]},
        "expected_trend": "CMA-ES with prompt learning is optimal design",
        "decision_claim": "Prompt learning with CMA-ES superior to alternatives",
        "artifacts": ["table_9"]
    })
    
    # Architecture generalization experiment (Table 10)
    matrix["experiments"].append({
        "experiment_id": "architecture_generalization",
        "name": "FOA on ResNet and VisionMamba",
        "datasets": ["ImageNet-C"],
        "environments": ["imagenet_c_gaussian_level_5"],
        "methods": ["NoAdapt", "TENT", "FOA", "FOA-dagger"],
        "models": ["ResNet", "VisionMamba"],
        "metrics": ["accuracy"],
        "parameters": {"corruption": "Gaussian", "corruption_level": 5},
        "expected_trend": "FOA generalizes to different architectures",
        "decision_claim": "FOA not limited to ViT architecture",
        "artifacts": ["table_10"]
    })
    
    # Non-i.i.d. scenarios experiment (Table 11)
    matrix["experiments"].append({
        "experiment_id": "non_iid_scenarios",
        "name": "FOA under Non-i.i.d. Scenarios",
        "datasets": ["ImageNet-C"],
        "environments": ["imagenet_c_level_5"],
        "methods": ["NoAdapt", "TENT", "CoTTA", "SAR", "FOA"],
        "models": ["ViT-Base"],
        "metrics": ["accuracy"],
        "parameters": {"corruption_level": 5, "scenarios": ["mild_iid", "online_imbalanced_label_shift", "mixed_shifts"]},
        "expected_trend": "FOA robust to non-i.i.d. distribution and label shifts",
        "decision_claim": "FOA handles realistic non-stationary test streams",
        "artifacts": ["table_11"]
    })
    
    # In-distribution performance experiment (Table 12)
    matrix["experiments"].append({
        "experiment_id": "in_distribution",
        "name": "In-Distribution Performance Preservation",
        "datasets": ["ImageNet"],
        "environments": ["imagenet_clean"],
        "methods": ["NoAdapt", "LAME", "T3A", "TENT", "CoTTA", "SAR", "FOA"],
        "models": ["ViT-Base"],
        "metrics": ["accuracy"],
        "parameters": {},
        "expected_trend": "FOA maintains clean accuracy like NoAdapt",
        "decision_claim": "FOA does not degrade in-distribution performance",
        "artifacts": ["table_12"]
    })
    
    # Lambda sensitivity experiment (Table 13)
    matrix["experiments"].append({
        "experiment_id": "lambda_sensitivity",
        "name": "Trade-off Parameter λ Sensitivity",
        "datasets": ["ImageNet-C"],
        "environments": ["imagenet_c_gaussian_level_5"],
        "methods": ["FOA"],
        "models": ["ViT-Base"],
        "metrics": ["accuracy", "ece"],
        "parameters": {
            "corruption": "Gaussian",
            "corruption_level": 5,
            "batch_size": 64,
            "lambda_sweep": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        },
        "expected_trend": "λ=0.4 provides good balance",
        "decision_claim": "FOA not highly sensitive to λ choice",
        "artifacts": ["table_13"]
    })
    
    # EMA in activation shifting experiment (Table 14)
    matrix["experiments"].append({
        "experiment_id": "ema_activation_shifting",
        "name": "EMA Effects in Activation Shifting",
        "datasets": ["ImageNet-C"],
        "environments": ["imagenet_c_level_5"],
        "methods": ["FOA", "FOA-no-ema-shifting"],
        "models": ["ViT-Base"],
        "metrics": ["accuracy", "ece"],
        "parameters": {"corruption_level": 5},
        "expected_trend": "EMA stabilizes activation shifting",
        "decision_claim": "EMA in activation shifting improves stability",
        "artifacts": ["table_14"]
    })
    
    # EMA in fitness calculation experiment (Table 15)