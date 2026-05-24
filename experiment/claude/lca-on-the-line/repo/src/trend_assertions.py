"""
LCA-on-the-Line: Trend Assertions and Validation

This module defines expected result-trend assertions for semantic review and validation
of experimental results. It verifies that computed metrics match the paper's reported
phenomena and trends.

Paper evidence contract: preserve expected result-trend assertions for:
- endpoint_low: p=0 and p=1 endpoint/boundary cases expected to be lowest/minimum/worst
- baseline_outperformance: explicit comparison showing improvement over baselines
- positive_parameter_improves: nonzero/positive parameter values preserve improvement trend

Binding addendum clarifications:
- WordNet dataset from https://github.com/jvlmdr/hiercls/blob/main/resources/hierarchy/imagenet_fiveai.csv
- Reproducing Table 3: if tree_prefix!='WordNet'

reference_grounding: paperbench_ref_001 test/test_models.py
reference_grounding: paperbench_ref_005 eval_many_models.py
reference_grounding: paperbench_ref_006 eval_tiny_imagenet_truncate.ipynb
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import warnings

logger = logging.getLogger(__name__)


# =============================================================================
# Paper Result Trend Assertions - Table 2, Figure 5, Table 3
# =============================================================================

EXPECTED_TRENDS = {
    # Table 2: Correlation measurement by R² and PEA of ID LCA/Top1 with OOD Top1/Top5
    # Strong correlation R² > 0.7 for most datasets (except ImageNet-v2)
    "correlation_r2_threshold": {
        "imagenet-v2": 0.3,  # Exception: ImageNet-v2 has weaker correlation
        "imagenet-a": 0.7,
        "imagenet-r": 0.7,
        "imagenet-sketch": 0.7,
        "objectnet": 0.7,
    },
    
    # Table 3: Error prediction MAE - ID LCA should outperform baselines
    # LCA方法在多数OOD数据集上MAE最低
    "prediction_mae_baseline_methods": [
        "ID_Top1",  # Baseline: ID Top-1 accuracy
        "Aline-S",  # Agreement-on-the-line (soft)
        "Aline-D",  # Agreement-on-the-line (dense)
    ],
    
    # Table 5: Soft labeling improvement - 0.5-2% accuracy gain on OOD
    # 软标签训练在多数OOD数据集上提升0.5-2%准确率
    "soft_label_improvement_range": (0.5, 2.0),
    
    # Table 14: Hierarchy-aware prompting for VLMs - 0.3-1.5% gain
    # 层次感知提示在VLMs上带来0.3-1.5%的OOD准确率提升
    "hierarchy_prompt_improvement_range": (0.3, 1.5),
    
    # Endpoint behavior: p=0 and p=1 boundary cases
    # endpoint_low: p=0 and p=1 must be represented as lowest/minimum boundary cases
    "endpoint_behavior": {
        "p_zero_is_minimum": True,  # p=0 corresponds to minimum performance
        "p_one_is_minimum": True,   # p=1 corresponds to minimum performance
    },
}


# =============================================================================
# Figure 1, 5: LCA-on-the-Line Correlation Assertions
# =============================================================================

def validate_lca_correlation_trend(
    id_lca: List[float],
    ood_top1: List[float],
    dataset_name: str,
    model_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Validate LCA-on-the-Line correlation trend for a given OOD dataset.
    
    Paper evidence:
    - Figure 1: Strong linear correlation between ID LCA and OOD Top-1
    - Table 2: R² > 0.7 for most datasets (except ImageNet-v2)
    - 强正相关，R²通常>0.7，ImageNet-V2相关性最强
    
    Args:
        id_lca: ID LCA distances for each model
        ood_top1: OOD Top-1 accuracies for each model
        dataset_name: Name of OOD dataset
        model_names: Optional list of model names for detailed reporting
        
    Returns:
        Validation result dictionary with assertion outcomes
        
    reference_grounding: paperbench_ref_005 eval_many_models.py
    """
    try:
        import numpy as np
        from scipy.stats import pearsonr
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
    except ImportError as e:
        logger.warning(f"Missing dependency for correlation validation: {e}")
        return {
            "valid": False,
            "error": f"Missing dependency: {e}",
            "dataset": dataset_name,
        }
    
    id_lca = np.array(id_lca)
    ood_top1 = np.array(ood_top1)
    
    if len(id_lca) != len(ood_top1):
        return {
            "valid": False,
            "error": "Mismatched array lengths",
            "dataset": dataset_name,
        }
    
    if len(id_lca) < 10:
        warnings.warn(f"Small sample size ({len(id_lca)}) may not be representative")
    
    # Compute correlation metrics
    pearson_r, p_value = pearsonr(id_lca, ood_top1)
    
    # Fit linear model
    X = id_lca.reshape(-1, 1)
    y = ood_top1
    reg = LinearRegression().fit(X, y)
    r2 = r2_score(y, reg.predict(X))
    
    # Expected threshold for this dataset
    expected_r2 = EXPECTED_TRENDS["correlation_r2_threshold"].get(
        dataset_name.lower(), 0.7
    )
    
    # Check if correlation meets expectation
    correlation_valid = r2 >= expected_r2
    
    # Check for negative correlation (ID LCA should negatively correlate with OOD accuracy)
    # Higher LCA distance = worse semantic errors = lower OOD accuracy
    negative_correlation = pearson_r < 0
    
    validation_result = {
        "valid": correlation_valid and negative_correlation,
        "dataset": dataset_name,
        "metrics": {
            "pearson_r": float(pearson_r),
            "pearson_p_value": float(p_value),
            "r2_score": float(r2),
            "slope": float(reg.coef_[0]),
            "intercept": float(reg.intercept_),
        },
        "thresholds": {
            "expected_r2": expected_r2,
            "actual_r2": float(r2),
            "r2_meets_threshold": correlation_valid,
        },
        "assertions": {
            "strong_correlation": r2 >= 0.7,
            "negative_correlation": negative_correlation,
            "statistically_significant": p_value < 0.05,
        },
        "n_models": len(id_lca),
    }
    
    # Add detailed model information if provided
    if model_names is not None and len(model_names) == len(id_lca):
        validation_result["model_details"] = {
            "models": model_names,
            "id_lca": id_lca.tolist(),
            "ood_top1": ood_top1.tolist(),
        }
    
    return validation_result


# =============================================================================
# Table 3: Error Prediction MAE Baseline Comparison
# =============================================================================

def validate_mae_baseline_outperformance(
    mae_results: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    """
    Validate that ID LCA outperforms baseline methods in error prediction (Table 3).
    
    Paper evidence:
    - Table 3: ID LCA has lowest MAE across most OOD datasets
    - LCA方法在多数OOD数据集上MAE最低
    - baseline_outperformance: proposed method should be compared against explicit baselines
    
    Args:
        mae_results: Dictionary mapping method_name -> {dataset: mae_value}
        
    Returns:
        Validation result with baseline comparison
        
    reference_grounding: paperbench_ref_005 eval_many_models.py
    """
    if "ID_LCA" not in mae_results:
        return {
            "valid": False,
            "error": "ID_LCA method not found in results",
        }
    
    baseline_methods = EXPECTED_TRENDS["prediction_mae_baseline_methods"]
    available_baselines = [m for m in baseline_methods if m in mae_results]
    
    if not available_baselines:
        return {
            "valid": False,
            "error": f"No baseline methods found. Expected: {baseline_methods}",
        }
    
    id_lca_results = mae_results["ID_LCA"]
    datasets = list(id_lca_results.keys())
    
    # Count wins for ID_LCA
    wins = {}
    comparisons = {}
    
    for dataset in datasets:
        if dataset not in id_lca_results:
            continue
            
        id_lca_mae = id_lca_results[dataset]
        dataset_wins = []
        dataset_comparisons = {"ID_LCA": id_lca_mae}
        
        for baseline in available_baselines:
            if dataset in mae_results[baseline]:
                baseline_mae = mae_results[baseline][dataset]
                dataset_comparisons[baseline] = baseline_mae
                
                # ID_LCA wins if it has lower MAE
                if id_lca_mae < baseline_mae:
                    dataset_wins.append(baseline)
        
        wins[dataset] = dataset_wins
        comparisons[dataset] = dataset_comparisons
    
    # Check if ID_LCA wins on majority of datasets against each baseline
    total_datasets = len(datasets)
    baseline_win_rates = {}
    
    for baseline in available_baselines:
        baseline_wins = sum(
            1 for dataset_wins in wins.values() if baseline in dataset_wins
        )
        win_rate = baseline_wins / total_datasets if total_datasets > 0 else 0
        baseline_win_rates[baseline] = {
            "wins": baseline_wins,
            "total": total_datasets,
            "win_rate": win_rate,
        }
    
    # ID_LCA should outperform baselines on majority (> 50%) of datasets
    majority_threshold = 0.5
    all_baselines_outperformed = all(
        stats["win_rate"] > majority_threshold
        for stats in baseline_win_rates.values()
    )
    
    validation_result = {
        "valid": all_baselines_outperformed,
        "datasets": datasets,
        "n_datasets": total_datasets,
        "comparisons": comparisons,
        "wins_by_dataset": wins,
        "baseline_win_rates": baseline_win_rates,
        "assertions": {
            "outperforms_majority": all_baselines_outperformed,
            "majority_threshold": majority_threshold,
        },
    }
    
    return validation_result


# =============================================================================
# Table 5: Soft Label Training Improvement
# =============================================================================

def validate_soft_label_improvement(
    baseline_results: Dict[str, float],
    soft_label_results: Dict[str, float],
) -> Dict[str, Any]:
    """
    Validate soft label training improvement on OOD datasets (Table 5).
    
    Paper evidence:
    - Table 5: Soft labeling improves OOD performance by 0.5-2%
    - 软标签训练在多数OOD数据集上提升0.5-2%准确率
    - positive_parameter_improves: nonzero parameter values preserve improvement trend
    
    Args:
        baseline_results: Baseline (CE-only) accuracies {dataset: accuracy}
        soft_label_results: Soft label training accuracies {dataset: accuracy}
        
    Returns:
        Validation result with improvement analysis
    """
    min_improvement, max_improvement = EXPECTED_TRENDS["soft_label_improvement_range"]
    
    datasets = set(baseline_results.keys()) & set(soft_label_results.keys())
    
    if not datasets:
        return {
            "valid": False,
            "error": "No common datasets between baseline and soft label results",
        }
    
    improvements = {}
    in_range_count = 0
    positive_improvement_count = 0
    
    for dataset in datasets:
        baseline_acc = baseline_results[dataset]
        soft_label_acc = soft_label_results[dataset]
        
        # Improvement in percentage points
        improvement = soft_label_acc - baseline_acc
        improvements[dataset] = {
            "baseline": baseline_acc,
            "soft_label": soft_label_acc,
            "improvement": improvement,
            "improvement_percent": (improvement / baseline_acc * 100) if baseline_acc > 0 else 0,
        }
        
        # Check if improvement is positive
        if improvement > 0:
            positive_improvement_count += 1
        
        # Check if improvement is in expected range
        if min_improvement <= improvement <= max_improvement:
            in_range_count += 1
    
    total_datasets = len(datasets)
    positive_rate = positive_improvement_count / total_datasets if total_datasets > 0 else 0
    in_range_rate = in_range_count / total_datasets if total_datasets > 0 else 0
    
    # Soft labels should improve performance on majority of datasets
    majority_threshold = 0.5
    improvements_valid = positive_rate > majority_threshold
    
    validation_result = {
        "valid": improvements_valid,
        "datasets": list(datasets),
        "n_datasets": total_datasets,
        "improvements": improvements,
        "summary": {
            "positive_improvements": positive_improvement_count,
            "in_range_improvements": in_range_count,
            "positive_rate": positive_rate,
            "in_range_rate": in_range_rate,
        },
        "expected_range": {
            "min": min_improvement,
            "max": max_improvement,
        },
        "assertions": {
            "majority_positive": positive_rate > majority_threshold,
            "majority_in_range": in_range_rate > majority_threshold,
        },
    }
    
    return validation_result


# =============================================================================
# Table 14: Hierarchy-Aware Prompting for VLMs
# =============================================================================

def validate_hierarchy_prompt_improvement(
    baseline_results: Dict[str, float],
    taxonomy_results: Dict[str, float],
) -> Dict[str, Any]:
    """
    Validate hierarchy-aware prompting improvement for VLMs (Table 14).
    
    Paper evidence:
    - Table 14: Taxonomy Parent prompting improves 0.3-1.5% on OOD
    - 层次感知提示在VLMs上带来0.3-1.5%的OOD准确率提升
    
    Args:
        baseline_results: Baseline prompting accuracies {dataset: accuracy}
        taxonomy_results: Taxonomy Parent prompting accuracies {dataset: accuracy}
        
    Returns:
        Validation result with prompting improvement analysis
    """
    min_improvement, max_improvement = EXPECTED_TRENDS["hierarchy_prompt_improvement_range"]
    
    datasets = set(baseline_results.keys()) & set(taxonomy_results.keys())
    
    if not datasets:
        return {
            "valid": False,
            "error": "No common datasets between baseline and taxonomy prompting results",
        }
    
    improvements = {}
    in_range_count = 0
    positive_improvement_count = 0
    
    for dataset in datasets:
        baseline_acc = baseline_results[dataset]
        taxonomy_acc = taxonomy_results[dataset]
        
        improvement = taxonomy_acc - baseline_acc
        improvements[dataset] = {
            "baseline": baseline_acc,
            "taxonomy": taxonomy_acc,
            "improvement": improvement,
            "improvement_percent": (improvement / baseline_acc * 100) if baseline_acc > 0 else 0,
        }
        
        if improvement > 0:
            positive_improvement_count += 1
        
        if min_improvement <= improvement <= max_improvement:
            in_range_count += 1
    
    total_datasets = len(datasets)
    positive_rate = positive_improvement_count / total_datasets if total_datasets > 0 else 0
    
    majority_threshold = 0.5
    improvements_valid = positive_rate > majority_threshold
    
    validation_result = {
        "valid": improvements_valid,
        "datasets": list(datasets),
        "n_datasets": total_datasets,
        "improvements": improvements,
        "summary": {
            "positive_improvements": positive_improvement_count,
            "in_range_improvements": in_range_count,
            "positive_rate": positive_rate,
        },
        "expected_range": {
            "min": min_improvement,
            "max": max_improvement,
        },
        "assertions": {
            "majority_positive": positive_rate > majority_threshold,
        },
    }
    
    return validation_result


# =============================================================================
# Endpoint Behavior Validation
# =============================================================================

def validate_endpoint_behavior(
    parameter_values: List[float],
    performance_values: List[float],
    parameter_name: str = "p",
) -> Dict[str, Any]:
    """
    Validate that endpoint cases (p=0, p=1) correspond to minimum performance.
    
    Paper evidence:
    - endpoint_low: p=0 and p=1 must be represented as lowest/minimum boundary cases
    
    Args:
        parameter_values: Parameter sweep values (e.g., interpolation weight p)
        performance_values: Corresponding performance values
        parameter_name: Name of the parameter being swept
        
    Returns:
        Validation result for endpoint behavior
    """
    try:
        import numpy as np
    except ImportError:
        return {
            "valid": False,
            "error": "NumPy required for endpoint validation",
        }
    
    parameter_values = np.array(parameter_values)
    performance_values = np.array(performance_values)
    
    if len(parameter_values) != len(performance_values):
        return {
            "valid": False,
            "error": "Mismatched array lengths",
        }
    
    # Find indices of endpoint values
    p_zero_idx = np.where(np.abs(parameter_values - 0.0) < 1e-6)[0]
    p_one_idx = np.where(np.abs(parameter_values - 1.0) < 1e-6)[0]
    
    # Find minimum performance value
    min_performance = np.min(performance_values)
    min_idx = np.argmin(performance_values)
    
    # Check if endpoints are among the lowest values
    # Allow small tolerance for numerical stability
    tolerance = 0.01 * (np.max(performance_values) - min_performance)
    
    p_zero_is_low = False
    p_one_is_low = False
    
    if len(p_zero_idx) > 0:
        p_zero_perf = performance_values[p_zero_idx[0]]
        p_zero_is_low = p_zero_perf <= min_performance + tolerance
    
    if len(p_one_idx) > 0:
        p_one_perf = performance_values[p_one_idx[0]]
        p_one_is_low = p_one_perf <= min_performance + tolerance
    
    expected_behavior = EXPECTED_TRENDS["endpoint_behavior"]
    
    validation_result = {
        "valid": (
            (not expected_behavior["p_zero_is_minimum"] or p_zero_is_low) and
            (not expected_behavior["p_one_is_minimum"] or p_one_is_low)
        ),
        "parameter": parameter_name,
        "n_points": len(parameter_values),
        "endpoints": {
            "p_zero": {
                "found": len(p_zero_idx) > 0,
                "performance": float(performance_values[p_zero_idx[0]]) if len(p_zero_idx) > 0 else None,
                "is_minimum": p_zero_is_low,
            },
            "p_one": {
                "found": len(p_one_idx) > 0,
                "performance": float(performance_values[p_one_idx[0]]) if len(p_one_idx) > 0 else None,
                "is_minimum": p_one_is_low,
            },
        },
        "min_performance": float(min_performance),
        "min_parameter": float(parameter_values[min_idx]),
        "assertions": expected_behavior,
    }
    
    return validation_result


# =============================================================================
# Comprehensive Validation Suite
# =============================================================================

def validate_all_trends(
    results: Dict[str, Any],
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run all trend validations on experimental results.
    
    Args:
        results: Complete experimental results dictionary
        output_path: Optional path to save validation report
        
    Returns:
        Comprehensive validation report
    """
    validation_report = {
        "overall_valid": True,
        "validations": {},
        "summary": {},
    }
    
    # Validate LCA correlation for each OOD dataset
    if "lca_correlation" in results:
        correlation_validations = {}
        for dataset, data in results["lca_correlation"].items():
            if "id_lca" in data and "ood_top1" in data:
                validation = validate_lca_correlation_trend(
                    id_lca=data["id_lca"],
                    ood_top1=data["ood_top1"],
                    dataset_name=dataset,
                    model_names=data.get("models"),
                )
                correlation_validations[dataset] = validation
                if not validation["valid"]:
                    validation_report["overall_valid"] = False
        
        validation_report["validations"]["lca_correlation"] = correlation_validations
    
    # Validate MAE baseline outperformance
    if "mae_prediction" in results:
        mae_validation = validate_mae_baseline_outperformance(
            mae_results=results["mae_prediction"]
        )
        validation_report["validations"]["mae_baseline"] = mae_validation
        if not mae_validation["valid"]:
            validation_report["overall_valid"] = False
    
    # Validate soft label improvements
    if "soft_label_training" in results:
        soft_validation = validate_soft_label_improvement(
            baseline_results=results["soft_label_training"].get("baseline", {}),
            soft_label_results=results["soft_label_training"].get("soft_label", {}),
        )
        validation_report["validations"]["soft_label"] = soft_validation
        if not soft_validation["valid"]:
            validation_report["overall_valid"] = False
    
    # Validate hierarchy-aware prompting
    if "hierarchy_prompting" in results:
        prompt_validation = validate_hierarchy_prompt_improvement(
            baseline_results=results["hierarchy_prompting"].get("baseline", {}),
            taxonomy_results=results["hierarchy_prompting"].get("taxonomy", {}),
        )
        validation_report["validations"]["hierarchy_prompting"] = prompt_validation
        if not prompt_validation["valid"]:
            validation_report["overall_valid"] = False
    
    # Count successful validations
    n_validations = len(validation_report["validations"])
    n_passed = sum(
        1 for v in validation_report["validations"].values()
        if isinstance(v, dict) and v.get("valid", False)
    )
    
    validation_report["summary"] = {
        "total_validations": n_validations,
        "passed_validations": n_passed,
        "failed_validations": n_validations - n_passed,
        "pass_rate": n_passed / n_validations if n_validations > 0 else 0,
    }
    
    # Write validation report if path provided
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(validation_report, f, indent=2)
        logger.info(f"Validation report written to {output_path}")
    
    return validation_report


# =============================================================================
# Smoke Test Artifact Writer
# =============================================================================

def write_trend_assertion_artifacts(
    output_dir: Path,
    mode: str = "runtime_smoke",
) -> None:
    """
    Write trend assertion artifacts for smoke testing.
    
    Args:
        output_dir: Output directory for artifacts
        mode: Execution mode (runtime_smoke or full)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if mode == "runtime_smoke":
        # Dry-run contract artifacts for smoke testing
        smoke_validation = {
            "mode": "runtime_smoke",
            "artifact_type": "trend_assertion_schema",
            "note": "This is a dry-run contract artifact, not real experimental results",
            "expected_trends": EXPECTED_TRENDS,
            "validation_functions": [
                "validate_lca_correlation_trend",
                "validate_mae_baseline_outperformance",
                "validate_soft_label_improvement",
                "validate_hierarchy_prompt_improvement",
                "validate_endpoint_behavior",
            ],
            "paper_assertions": {
                "correlation": "强正相关，R²通常>0.7，ImageNet-V2相关性最强",
                "mae_baseline": "LCA方法在多数OOD数据集上MAE最低",
                "soft_label": "软标签训练在多数OOD数据集上提升0.5-2%准确率",
                "hierarchy_prompt": "层次感知提示在VLMs上带来0.3-1.5%的OOD准确率提升",
                "endpoint": "p=0 and p=1 must be represented as lowest/minimum boundary cases",
                "baseline_outperformance": "proposed method should be compared against explicit baselines",
            },
        }
        
        schema_path = output_dir / "trend_assertions_schema.json"
        with open(schema_path, 'w') as f:
            json.dump(smoke_validation, f, indent=2)
        logger.info(f"Written trend assertion schema to {schema_path}")
    
    else:
        logger.info("Full mode: trend assertions will validate actual results")


if __name__ == "__main__":
    # Smoke test
    import sys
    
    output_dir = Path("results")
    mode = "runtime_smoke" if "--mode" in sys.argv else "runtime_smoke"
    
    write_trend_assertion_artifacts(output_dir, mode=mode)
    
    print("Trend assertions module smoke test completed")
    print(f"Expected trends defined: {len(EXPECTED_TRENDS)}")
    print("Validation functions available:")
    print("  - validate_lca_correlation_trend")
    print("  - validate_mae_baseline_outperformance")
    print("  - validate_soft_label_improvement")
    print("  - validate_hierarchy_prompt_improvement")
    print("  - validate_endpoint_behavior")
    print("  - validate_all_trends")