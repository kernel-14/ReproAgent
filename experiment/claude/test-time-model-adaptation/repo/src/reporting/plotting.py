#!/usr/bin/env python3
"""
Plotting and artifact generation for Test-Time Model Adaptation with Only Forward Passes.

Implements artifact writers, measurement schemas, aggregation outputs, and result-trend 
assertions as required by the paper evidence contract.

This file satisfies:
- Metric schemas for accuracy, precision, loss, training_time, ece, memory_usage
- Artifact writers for Tables 1-17 and Figures 1-4
- Result-trend assertions: sweep_insensitive, baseline_outperformance, endpoint_low
- Statically discoverable artifact paths
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np


# ==============================================================================
# Metric Schema Definitions
# ==============================================================================

METRIC_SCHEMAS = {
    "accuracy": {
        "name": "Accuracy",
        "unit": "%",
        "direction": "higher_better",
        "range": [0.0, 100.0],
        "aggregation": "mean",
        "decimal_places": 2
    },
    "precision": {
        "name": "Precision",
        "unit": "%",
        "direction": "higher_better",
        "range": [0.0, 100.0],
        "aggregation": "mean",
        "decimal_places": 2
    },
    "loss": {
        "name": "Loss",
        "unit": "",
        "direction": "lower_better",
        "range": [0.0, float('inf')],
        "aggregation": "mean",
        "decimal_places": 4
    },
    "training_time": {
        "name": "Training Time",
        "unit": "seconds",
        "direction": "lower_better",
        "range": [0.0, float('inf')],
        "aggregation": "sum",
        "decimal_places": 2
    },
    "ece": {
        "name": "Expected Calibration Error",
        "unit": "%",
        "direction": "lower_better",
        "range": [0.0, 100.0],
        "aggregation": "mean",
        "decimal_places": 2
    },
    "memory_usage": {
        "name": "Memory Usage",
        "unit": "MB",
        "direction": "lower_better",
        "range": [0.0, float('inf')],
        "aggregation": "max",
        "decimal_places": 1
    }
}


# ==============================================================================
# Artifact Path Registry
# ==============================================================================

ARTIFACT_PATHS = {
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
    "figure_1": "results/figures/figure_1.png",
    "figure_2": "results/figures/figure_2.png",
    "figure_3": "results/figures/figure_3.png",
    "figure_4": "results/figures/figure_4.png",
    "experiment_results_table": "results/tables/experiment_results.csv",
    "experiment_results_figure": "results/figures/experiment_results.png",
    "predictions": "results/predictions.jsonl",
    "metrics_json": "results/metrics.json",
    "config_resolved": "results/config_resolved.json"
}


# ==============================================================================
# Trend Assertion Definitions
# ==============================================================================

TREND_ASSERTIONS = {
    "sweep_insensitive": {
        "description": "Parameter sweep should preserve stable/insensitive/robust trend claim",
        "parameters": ["population_size", "lambda", "prompt_count"],
        "validation": "variance_below_threshold",
        "threshold": 0.05
    },
    "baseline_outperformance": {
        "description": "Proposed method should be compared against explicit baselines",
        "baselines": ["NoAdapt", "T3A", "LAME", "TENT", "CoTTA", "SAR"],
        "validation": "ours_better_than_baseline",
        "metric": "accuracy"
    },
    "endpoint_low": {
        "description": "p=0 and p=1 endpoint/boundary cases expected to be lowest/minimum/worst",
        "parameters": ["p"],
        "endpoints": [0.0, 1.0],
        "validation": "endpoints_minimal",
        "metric": "accuracy"
    }
}


# ==============================================================================
# Utility Functions
# ==============================================================================

def ensure_directory(filepath: str) -> Path:
    """Ensure parent directory exists for a file path."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def aggregate_metric(values: List[float], metric_name: str) -> float:
    """Aggregate metric values according to schema."""
    if not values:
        return 0.0
    schema = METRIC_SCHEMAS.get(metric_name, {})
    agg_type = schema.get("aggregation", "mean")
    
    if agg_type == "mean":
        return float(np.mean(values))
    elif agg_type == "sum":
        return float(np.sum(values))
    elif agg_type == "max":
        return float(np.max(values))
    elif agg_type == "min":
        return float(np.min(values))
    else:
        return float(np.mean(values))


def format_metric(value: float, metric_name: str) -> str:
    """Format metric value according to schema."""
    schema = METRIC_SCHEMAS.get(metric_name, {})
    decimal_places = schema.get("decimal_places", 2)
    return f"{value:.{decimal_places}f}"


def validate_trend_assertion(
    results: Dict[str, Any],
    assertion_name: str
) -> Tuple[bool, str]:
    """Validate a trend assertion against results."""
    assertion = TREND_ASSERTIONS.get(assertion_name)
    if not assertion:
        return False, f"Unknown assertion: {assertion_name}"
    
    validation_type = assertion.get("validation")
    
    if validation_type == "variance_below_threshold":
        # Check parameter sweep stability
        parameters = assertion.get("parameters", [])
        threshold = assertion.get("threshold", 0.05)
        for param in parameters:
            if param in results:
                values = results[param].get("accuracy", [])
                if len(values) > 1:
                    variance = float(np.var(values))
                    if variance > threshold:
                        return False, f"Parameter {param} variance {variance:.4f} exceeds threshold {threshold}"
        return True, "Parameter sweep stable"
    
    elif validation_type == "ours_better_than_baseline":
        # Check baseline outperformance
        baselines = assertion.get("baselines", [])
        metric = assertion.get("metric", "accuracy")
        ours_value = results.get("ours", {}).get(metric, 0.0)
        for baseline in baselines:
            baseline_value = results.get(baseline, {}).get(metric, 0.0)
            if ours_value <= baseline_value:
                return False, f"Our method ({ours_value}) not better than {baseline} ({baseline_value})"
        return True, "Outperforms all baselines"
    
    elif validation_type == "endpoints_minimal":
        # Check endpoint minimality
        parameters = assertion.get("parameters", [])
        endpoints = assertion.get("endpoints", [])
        metric = assertion.get("metric", "accuracy")
        for param in parameters:
            if param in results:
                param_results = results[param]
                for endpoint in endpoints:
                    endpoint_key = f"{param}={endpoint}"
                    if endpoint_key in param_results:
                        endpoint_value = param_results[endpoint_key].get(metric, float('inf'))
                        other_values = [
                            v.get(metric, 0.0) for k, v in param_results.items()
                            if k != endpoint_key and isinstance(v, dict)
                        ]
                        if other_values and endpoint_value > min(other_values):
                            return False, f"Endpoint {endpoint_key} not minimal"
        return True, "Endpoints minimal"
    
    return False, "Unknown validation type"


# ==============================================================================
# Table Writers
# ==============================================================================

def write_table_1(results: Dict[str, Any], dry_run: bool = False) -> str:
    """
    Write Table 1: Memory and accuracy comparison (FOA vs. gradient-based TTA).
    
    Table caption: Comparison w.r.t. prior gradient-based Test-Time Adaptation (TTA)
    vs. our Forward-Optimization Adaptation. The memory usage and accuracy are measured
    via ViT-Base and batch size 64 on ImageNet-C (level 5).
    """
    filepath = ensure_directory(ARTIFACT_PATHS["table_1"])
    
    if dry_run:
        content = "Method,Accuracy (%),Memory (MB),BP Required,Notes\n"
        content += "NoAdapt,56.4,1024.0,No,Dry-run schema artifact\n"
        content += "T3A,57.2,1050.0,No,Dry-run schema artifact\n"
        content += "TENT,62.5,2048.0,Yes,Dry-run schema artifact\n"
        content += "FOA (Ours),64.8,1080.0,No,Dry-run schema artifact\n"
    else:
        # Extract from results
        methods = results.get("methods", {})
        content = "Method,Accuracy (%),Memory (MB),BP Required,Notes\n"
        for method_name, method_data in methods.items():
            accuracy = method_data.get("accuracy", 0.0)
            memory = method_data.get("memory_usage", 0.0)
            bp_required = "Yes" if method_data.get("requires_gradients", False) else "No"
            content += f"{method_name},{accuracy:.2f},{memory:.1f},{bp_required},\n"
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    return str(filepath)


def write_table_2(results: Dict[str, Any], dry_run: bool = False) -> str:
    """
    Write Table 2: ImageNet-C comparisons with SOTA methods (ViT-Base, severity 5).
    
    Table caption: Comparisons with SOTA methods on ImageNet-C (severity level 5)
    with ViT regarding Accuracy (%). BP is short for backward propagation.
    """
    filepath = ensure_directory(ARTIFACT_PATHS["table_2"])
    
    corruptions = [
        "Gaussian", "Shot", "Impulse", "Defocus", "Glass", "Motion", "Zoom",
        "Snow", "Frost", "Fog", "Bright", "Contrast", "Elastic", "Pixel", "JPEG"
    ]
    
    if dry_run:
        content = "Method," + ",".join(corruptions) + ",Average,ECE (%)\n"
        content += "NoAdapt," + ",".join(["56.4"] * 15) + ",56.4,15.2\n"
        content += "T3A," + ",".join(["57.2"] * 15) + ",57.2,14.8\n"
        content += "LAME," + ",".join(["58.1"] * 15) + ",58.1,14.5\n"
        content += "TENT," + ",".join(["62.5"] * 15) + ",62.5,12.3\n"
        content += "CoTTA," + ",".join(["63.2"] * 15) + ",63.2,12.0\n"
        content += "SAR," + ",".join(["63.8"] * 15) + ",63.8,11.8\n"
        content += "FOA (Ours)," + ",".join(["64.8"] * 15) + ",64.8,11.2\n"
    else:
        methods = results.get("imagenet_c", {})
        content = "Method," + ",".join(corruptions) + ",Average,ECE (%)\n"
        for method_name, method_data in methods.items():
            corruption_accs = [format_metric(method_data.get(c, 0.0), "accuracy") for c in corruptions]
            avg_acc = format_metric(aggregate_metric([method_data.get(c, 0.0) for c in corruptions], "accuracy"), "accuracy")
            ece = format_metric(method_data.get("ece", 0.0), "ece")
            content += f"{method_name}," + ",".join(corruption_accs) + f",{avg_acc},{ece}\n"
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    return str(filepath)


def write_table_3(results: Dict[str, Any], dry_run: bool = False) -> str:
    """
    Write Table 3: ImageNet-R/V2/Sketch comparisons with ViT-Base.
    """
    filepath = ensure_directory(ARTIFACT_PATHS["table_3"])
    
    if dry_run:
        content = "Method,ImageNet-R,ImageNet-V2,ImageNet-Sketch\n"
        content += "NoAdapt,45.2,63.1,38.5\n"
        content += "T3A,46.8,64.2,39.7\n"
        content += "LAME,47.5,64.8,40.2\n"
        content += "TENT,51.2,68.5,43.8\n"
        content += "CoTTA,52.1,69.2,44.5\n"
        content += "SAR,52.8,69.8,45.1\n"
        content += "FOA (Ours),53.5,70.5,45.8\n"
    else:
        methods = results.get("robustness_benchmarks", {})
        content = "Method,ImageNet-R,ImageNet-V2,ImageNet-Sketch\n"
        for method_name, method_data in methods.items():
            r_acc = format_metric(method_data.get("imagenet_r", 0.0), "accuracy")
            v2_acc = format_metric(method_data.get("imagenet_v2", 0.0), "accuracy")
            sketch_acc = format_metric(method_data.get("imagenet_sketch", 0.0), "accuracy")
            content += f"{method_name},{r_acc},{v2_acc},{sketch_acc}\n"
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    return str(filepath)


def write_table_4(results: Dict[str, Any], dry_run: bool = False) -> str:
    """
    Write Table 4: Quantized ViT models effectiveness.
    """
    filepath = ensure_directory(ARTIFACT_PATHS["table_4"])
    
    if dry_run:
        content = "Method,32-bit Acc (%),32-bit ECE (%),8-bit Acc (%),8-bit ECE (%),6-bit Acc (%),6-bit ECE (%)\n"
        content += "NoAdapt,56.4,15.2,54.2,16.8,51.5,18.5\n"
        content += "T3A,57.2,14.8,55.1,16.2,52.3,17.9\n"
        content += "FOA (Ours),64.8,11.2,62.5,12.5,59.8,13.8\n"
    else:
        quantization = results.get("quantization", {})
        content = "Method,32-bit Acc (%),32-bit ECE (%),8-bit Acc (%),8-bit ECE (%),6-bit Acc (%),6-bit ECE (%)\n"
        for method_name, method_data in quantization.items():
            row = [method_name]
            for bits in [32, 8, 6]:
                acc = format_metric(method_data.get(f"{bits}bit_accuracy", 0.0), "accuracy")
                ece = format_metric(method_data.get(f"{bits}bit_ece", 0.0), "ece")
                row.extend([acc, ece])
            content += ",".join(row) + "\n"
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    return str(filepath)


def write_table_5(results: Dict[str, Any], dry_run: bool = False) -> str:
    """
    Write Table 5: Ablation of FOA components.
    """
    filepath = ensure_directory(ARTIFACT_PATHS["table_5"])
    
    if dry_run:
        content = "Configuration,Entropy,Act. Discrepancy,Act. Shifting,Accuracy (%),ECE (%)\n"
        content += "Full FOA,Yes,Yes,Yes,64.8,11.2\n"
        content += "w/o Act. Shifting,Yes,Yes,No,62.3,12.5\n"
        content += "w/o Act. Discrepancy,Yes,No,Yes,61.8,13.1\n"
        content += "w/o Entropy,No,Yes,Yes,60.5,13.8\n"
        content += "Entropy only,Yes,No,No,58.2,14.5\n"
    else:
        ablations = results.get("ablations", {})
        content = "Configuration,Entropy,Act. Discrepancy,Act. Shifting,Accuracy (%),ECE (%)\n"
        for config_name, config_data in ablations.items():
            entropy = "Yes" if config_data.get("use_entropy", False) else "No"
            act_disc = "Yes" if config_data.get("use_activation_discrepancy", False) else "No"
            act_shift = "Yes" if config_data.get("use_activation_shifting", False) else "No"
            acc = format_metric(config_data.get("accuracy", 0.0), "accuracy")
            ece = format_metric(config_data.get("ece", 0.0), "ece")
            content += f"{config_name},{entropy},{act_disc},{act_shift},{acc},{ece}\n"
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    return str(filepath)


def write_table_6(results: Dict[str, Any], dry_run: bool = False) -> str:
    """
    Write Table 6: FOA with interval update strategy (FOA-I).
    """
    filepath = ensure_directory(ARTIFACT_PATHS["table_6"])
    
    if dry_run:
        content = "Method,Interval I,Accuracy (%),Memory (MB)\n"
        content += "FOA,N/A,64.8,1080.0\n"
        content += "FOA-I,1,64.8,1095.0\n"
        content += "FOA-I,2,64.5,1088.0\n"
        content += "FOA-I,4,64.2,1082.0\n"
        content += "FOA-I,8,63.8,1078.0\n"
    else:
        interval_results = results.get("interval_update", {})
        content = "Method,Interval I,Accuracy (%),Memory (MB)\n"
        for method_name, method_data in interval_results.items():
            interval = method_data.get("interval", "N/A")
            acc = format_metric(method_data.get("accuracy", 0.0), "accuracy")
            mem = format_metric(method_data.get("memory_usage", 0.0), "memory_usage")
            content += f"{method_name},{interval},{acc},{mem}\n"
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    return str(filepath)


def write_table_7(results: Dict[str, Any], dry_run: bool = False) -> str:
    """
    Write Table 7: Memory usage comparison across batch sizes.
    """
    filepath = ensure_directory(ARTIFACT_PATHS["table_7"])
    
    if dry_run:
        content = "Method,BS=1,BS=4,BS=16,BS=64\n"
        content += "NoAdapt (32-bit),256,512,1024,2048\n"
        content += "NoAdapt (8-bit),64,128,256,512\n"
        content += "TENT (32-bit),512,1024,2048,4096\n"
        content += "FOA (32-bit),260,520,1040,2080\n"
        content += "FOA (8-bit),68,132,260,520\n"
    else:
        memory_results = results.get("memory_analysis", {})
        content = "Method,BS=1,BS=4,BS=16,BS=64\n"
        for method_name, method_data in memory_results.items():
            row = [method_name]
            for bs in [1, 4, 16, 64]:
                mem = format_metric(method_data.get(f"bs_{bs}", 0.0), "memory_usage")
                row.append(mem)
            content += ",".join(row) + "\n"
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    return str(filepath)


def write_table_8(results: Dict[str, Any], dry_run: bool = False) -> str:
    """
    Write Table 8: Computational complexity comparison.
    """
    filepath = ensure_directory(ARTIFACT_PATHS["table_8"])
    
    if dry_run:
        content = "Method,#FP,#BP,Accuracy (%),ECE (%),Wall-Clock (s),Memory (MB)\n"
        content += "NoAdapt,1,0,56.4,15.2,0.05,1024\n"
        content += "T3A,1,0,57.2,14.8,0.08,1050\n"
        content += "TENT,1,1,62.5,12.3,0.15,2048\n"
        content += "CoTTA,1,1,63.2,12.0,0.18,2100\n"
        content += "SAR,1,1,63.8,11.8,0.17,2080\n"
        content += "FOA (Ours),10,0,64.8,11.2,0.12,1080\n"
    else:
        complexity = results.get("complexity", {})
        content = "Method,#FP,#BP,Accuracy (%),ECE (%),Wall-Clock (s),Memory (MB)\n"
        for method_name, method_data in complexity.items():
            fp = method_data.get("forward_passes", 1)
            bp = method_data.get("backward_passes", 0)
            acc = format_metric(method_data.get("accuracy", 0.0), "accuracy")
            ece = format_metric(method_data.get("ece", 0.0), "ece")
            time = format_metric(method_data.get("wall_clock_time", 0.0), "training_time")
            mem = format_metric(method_data.get("memory_usage", 0.0), "memory_usage")
            content += f"{method_name},{fp},{bp},{acc},{ece},{time},{mem}\n"
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    return str(filepath)


def write_table_9(results: Dict[str, Any], dry_run: bool = False) -> str:
    """
    Write Table 9: Design choices (learnable parameters, optimizer, loss).
    """
    filepath = ensure_directory(ARTIFACT_PATHS["table_9"])
    
    if dry_run:
        content = "Configuration,Learnable Param,Optimizer,Loss Function,Accuracy (%),ECE (%)\n"
        content += "FOA (Default),Prompt,CMA-ES,Entropy+Act.,64.8,11.2\n"
        content += "Variant 1,Prompt,Adam,Entropy+Act.,62.3,12.5\n"
        content += "Variant 2,Norm,CMA-ES,Entropy+Act.,61.5,13.1\n"
        content += "Variant 3,Prompt,CMA-ES,Entropy only,60.8,13.8\n"
        content += "Variant 4,Prompt,SGD,Entropy+Act.,59.5,14.2\n"
    else:
        design_choices = results.get("design_choices", {})
        content = "Configuration,Learnable Param,Optimizer,Loss Function,Accuracy (%),ECE (%)\n"
        for config_name, config_data in design_choices.items():
            param = config_data.get("learnable_param", "")
            optimizer = config_data.get("optimizer", "")
            loss = config_data.get("loss_function", "")
            acc = format_metric(config_data.get("accuracy", 0.0), "accuracy")
            ece = format_metric(config_data.get("ece", 0.0), "ece")
            content += f"{config_name},{param},{optimizer},{loss},{acc},{ece}\n"
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    return str(filepath)


def write_table_10(results: Dict[str, Any], dry_run: bool = False) -> str:
    """
    Write Table 10: FOA on different architectures (ResNet, VisionMamba).
    """
    filepath = ensure_directory(ARTIFACT_PATHS["table_10"])
    
    if dry_run:
        content = "Method,Architecture,Accuracy (%),ECE (%)\n"
        content += "NoAdapt,ResNet-50,52.3,17.5\n"
        content += "FOA,ResNet-50,58.2,14.2\n"
        content += "NoAdapt,VisionMamba,54.1,16.8\n"
        content += "FOA,VisionMamba,60.5,13.5\n"
        content += "NoAdapt,ViT-Base,56.4,15.2\n"
        content += "FOA,ViT-Base,64.8,11.2\n"
    else:
        architecture_results = results.get("architectures", {})
        content = "Method,Architecture,Accuracy (%),ECE (%)\n"
        for method_name, method_data in architecture_results.items():
            arch = method_data.get("architecture", "")
            acc = format_metric(method_data.get("accuracy", 0.0), "accuracy")
            ece = format_metric(method_data.get("ece", 0.0), "ece")
            content += f"{method_name},{arch},{acc},{ece}\n"
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    return str(filepath)


def write_table_11(results: Dict[str, Any], dry_run: bool = False) -> str:
    """
    Write Table 11: Non-i.i.d. scenarios.
    """
    filepath = ensure_directory(ARTIFACT_PATHS["table_11"])
    
    if dry_run:
        content = "Method,i.i.d.,Imbalanced Label,Mixed Shifts\n"
        content += "NoAdapt,56.4,53.2,51.8\n"
        content += "T3A,57.2,54.1,52.5\n"
        content += "TENT,62.5,59.8,57.2\n"
        content += "FOA (Ours),64.8,62.5,60.8\n"
    else:
        non_iid = results.get("non_iid", {})
        content = "Method,i.i.d.,Imbalanced Label,Mixed Shifts\n"
        for method_name, method_data in non_iid.items():
            iid = format_metric(method_data.get("iid", 0.0), "accuracy")
            imbalanced = format_metric(method_data.get("imbalanced_label", 0.0), "accuracy")
            mixed = format_metric(method_data.get("mixed_shifts", 0.0), "accuracy")
            content += f"{method_name},{iid},{imbalanced},{mixed}\n"
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    return str(filepath)


def write_table_12(results: Dict[str, Any], dry_run: bool = False) -> str:
    """
    Write Table 12: In-distribution performance on clean ImageNet.
    """
    filepath = ensure_directory(ARTIFACT_PATHS["table_12"])
    
    if dry_run:
        content = "Method,Clean ImageNet Accuracy (%)\n"
        content += "NoAdapt,76.5\n"
        content += "T3A,75.8\n"