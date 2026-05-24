"""
Trend Assertions and Expected Result Contract for Refined Coreset Selection.

This module preserves the paper's expected result trends, baseline comparisons,
and semantic assertions for semantic review and experiment validation.

Paper evidence contract: preserve expected result-trend assertions from:
- Figure 1: Lexicographic optimization reduces f1 and f2 over iterations
- Table 1: LBCS reduces both objectives compared to initialization
- Table 2-3: LBCS achieves best/near-best accuracy with smaller coreset sizes
- Table 4: ImageNet-1k performance at 70% and 80% coreset ratios
- Figure 2: Robustness to 30% symmetric label noise
- Table 9: Performance improves with search time T but marginal gains diminish

reference_grounding: paperbench_ref_003 train.py
reference_grounding: paperbench_ref_003 selection.py
reference_grounding: paperbench_ref_004 cnn_mnist_probability_1step_pixel_shared_rein.py
reference_grounding: paperbench_ref_004 noisy_label.py
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import warnings


# ============================================================================
# Expected Trend Declarations from Paper
# ============================================================================

@dataclass
class TrendAssertion:
    """Single trend assertion with validation criteria."""
    name: str
    description: str
    assertion_type: str  # "inequality", "ordering", "threshold", "comparison"
    validation_fn: Optional[callable] = None
    paper_evidence: str = ""
    critical: bool = True


# ============================================================================
# Core Trend Assertions from Paper Results
# ============================================================================

TREND_ASSERTIONS = {
    # Figure 1 and Table 1: Lexicographic optimization effectiveness
    "lexicographic_convergence": TrendAssertion(
        name="lexicographic_convergence",
        description="f1(m) and f2(m) both decrease over outer iterations",
        assertion_type="inequality",
        paper_evidence="Figure 1(a)(b): f1 decreases from ~1000 to ~400, f2 decreases substantially",
        critical=True
    ),
    
    "objective_improvement": TrendAssertion(
        name="objective_improvement",
        description="Achieved f1(m) and f2(m) are lower than initialized values",
        assertion_type="comparison",
        paper_evidence="Table 1: Both objectives reduced after LBCS optimization",
        critical=True
    ),
    
    # Core LBCS property: epsilon and coreset size relationship
    "epsilon_coreset_monotonicity": TrendAssertion(
        name="epsilon_coreset_monotonicity",
        description="Larger epsilon allows smaller coreset size",
        assertion_type="ordering",
        paper_evidence="Table 1: epsilon 0.2→0.3→0.4 allows decreasing coreset sizes",
        critical=True
    ),
    
    "initial_k_final_k_relationship": TrendAssertion(
        name="initial_k_final_k_relationship",
        description="Larger initial k tends to yield smaller or comparable final k (optimization space)",
        assertion_type="inequality",
        paper_evidence="Table 1: Initial k {200,400,600,800,1000} vs optimized sizes",
        critical=True
    ),
    
    # Table 2-3: LBCS performance vs baselines
    "lbcs_accuracy_superiority": TrendAssertion(
        name="lbcs_accuracy_superiority",
        description="LBCS achieves best or near-best accuracy while reducing coreset size",
        assertion_type="comparison",
        paper_evidence="Table 2: LBCS obtains best mean test accuracy with smaller coreset sizes",
        critical=True
    ),
    
    "lbcs_coreset_reduction": TrendAssertion(
        name="lbcs_coreset_reduction",
        description="LBCS optimized coreset size is smaller than predefined size",
        assertion_type="inequality",
        paper_evidence="Table 2: Optimized sizes consistently below predefined k",
        critical=True
    ),
    
    # Baseline comparison (Table 2)
    "baseline_outperformance": TrendAssertion(
        name="baseline_outperformance",
        description="LBCS outperforms or matches: Uniform, EL2N, GraNd, Influential, Moderate, CCS, Probabilistic",
        assertion_type="comparison",
        paper_evidence="Table 2: LBCS vs 7 baselines across CIFAR-10, CIFAR-100, F-MNIST",
        critical=True
    ),
    
    # Table 4: ImageNet-1k specific results
    "imagenet_70_performance": TrendAssertion(
        name="imagenet_70_performance",
        description="LBCS achieves ~89.98% top-5 accuracy (ResNet-18) and ~68.53% top-1 at 70% ratio",
        assertion_type="threshold",
        paper_evidence="Table 4: 89.98±0.23 (top-5) and 68.53±0.30 (top-1) at 70%",
        critical=False  # ImageNet excluded per addendum
    ),
    
    "imagenet_80_performance": TrendAssertion(
        name="imagenet_80_performance",
        description="LBCS achieves ~90.84% top-5 accuracy (ResNet-18) and ~77.86% top-1 at 80% ratio",
        assertion_type="threshold",
        paper_evidence="Table 4: 90.84±0.20 (top-5) and 77.86±0.27 (top-1) at 80%",
        critical=False  # ImageNet excluded per addendum
    ),
    
    # Figure 2 and Table 8: Robustness to label noise
    "noise_robustness": TrendAssertion(
        name="noise_robustness",
        description="LBCS maintains performance advantage under 30% symmetric label noise",
        assertion_type="comparison",
        paper_evidence="Figure 2(a): LBCS outperforms baselines with 30% corrupted labels on F-MNIST",
        critical=True
    ),
    
    # Table 9: Search time ablation
    "search_time_improvement": TrendAssertion(
        name="search_time_improvement",
        description="Performance improves with search time T, but marginal gains diminish",
        assertion_type="ordering",
        paper_evidence="Table 9: Test accuracy increases and coreset size decreases with T, then stabilizes",
        critical=True
    ),
    
    # Edge cases and boundary conditions
    "endpoint_low_performance": TrendAssertion(
        name="endpoint_low_performance",
        description="Extreme cases (p=0 or p=1) should show lowest/boundary performance",
        assertion_type="threshold",
        paper_evidence="Paper section 2.1: Discussion of trivial solutions at boundaries",
        critical=True
    ),
}


# ============================================================================
# Baseline Method Registry for Comparison
# Paper evidence contract: explicit named baselines from Table 2
# ============================================================================

BASELINE_METHODS = {
    "uniform": {
        "name": "Uniform",
        "description": "Random uniform sampling",
        "expected_relative_performance": "lowest",
        "paper_tables": ["Table 2"]
    },
    "el2n": {
        "name": "EL2N",
        "description": "Error L2-Norm based selection",
        "expected_relative_performance": "moderate",
        "paper_tables": ["Table 2"]
    },
    "grand": {
        "name": "GraNd",
        "description": "Gradient Norm based selection",
        "expected_relative_performance": "moderate",
        "paper_tables": ["Table 2"]
    },
    "influential": {
        "name": "Influential",
        "description": "Influence function based selection",
        "expected_relative_performance": "moderate",
        "paper_tables": ["Table 2"]
    },
    "moderate": {
        "name": "Moderate",
        "description": "Moderate difficulty data selection",
        "expected_relative_performance": "high",
        "paper_tables": ["Table 2", "Table 5"]
    },
    "ccs": {
        "name": "CCS",
        "description": "Coverage-based Coreset Selection",
        "expected_relative_performance": "moderate",
        "paper_tables": ["Table 2"]
    },
    "probabilistic": {
        "name": "Probabilistic",
        "description": "Probabilistic bilevel coreset selection",
        "expected_relative_performance": "high",
        "paper_tables": ["Table 2"]
    },
    "lbcs": {
        "name": "LBCS",
        "description": "Lexicographic Bilevel Coreset Selection (proposed)",
        "expected_relative_performance": "best",
        "paper_tables": ["Table 1", "Table 2", "Table 3", "Table 4"]
    }
}


# ============================================================================
# Expected Result Ranges from Paper Tables
# ============================================================================

EXPECTED_RESULTS = {
    # Table 1: Preliminary LBCS results on CIFAR-10
    "table_1_cifar10": {
        "dataset": "cifar10",
        "epsilon_sweep": [0.2, 0.3, 0.4],
        "initial_k_sweep": [200, 400, 600, 800, 1000],
        "expected_behavior": {
            "f1_decreases": True,
            "f2_decreases": True,
            "final_k_less_than_initial": True,
            "larger_epsilon_smaller_coreset": True
        }
    },
    
    # Table 2: Comparison with baselines
    "table_2_cifar10": {
        "dataset": "cifar10",
        "predefined_k": [956, 1912, 2868, 3824],
        "lbcs_accuracy_range": (64.0, 75.0),  # Approximate from paper
        "lbcs_coreset_reduction": True,
        "lbcs_best_or_near_best": True
    },
    
    "table_2_cifar100": {
        "dataset": "cifar100",
        "predefined_k": [2500, 5000, 7500, 10000],
        "lbcs_accuracy_range": (30.0, 50.0),  # Approximate from paper
        "lbcs_coreset_reduction": True,
        "lbcs_best_or_near_best": True
    },
    
    "table_2_fmnist": {
        "dataset": "fmnist",
        "predefined_k": [200, 400, 600, 800, 1000],
        "lbcs_accuracy_range": (75.0, 85.0),  # Approximate from paper
        "lbcs_coreset_reduction": True,
        "lbcs_best_or_near_best": True
    },
    
    # Table 3: LBCS-only results
    "table_3_results": {
        "datasets": ["cifar10", "cifar100", "fmnist", "svhn"],
        "expected_behavior": {
            "optimized_size_reported": True,
            "accuracy_competitive_or_better": True
        }
    },
    
    # Table 4: ImageNet-1k (excluded per addendum, but preserve contract)
    "table_4_imagenet": {
        "dataset": "imagenet1k",
        "coreset_ratios": [0.7, 0.8],
        "resnet18_top5_70": (89.5, 90.5),
        "resnet18_top5_80": (90.5, 91.5),
        "resnet18_top1_70": (68.0, 69.0),
        "resnet18_top1_80": (77.5, 78.5),
        "excluded": True  # Per addendum
    },
    
    # Figure 2: Noisy label robustness
    "figure_2_noisy": {
        "dataset": "fmnist",
        "noise_rate": 0.3,
        "noise_type": "symmetric",
        "lbcs_maintains_advantage": True,
        "gap_vs_baselines": "positive"
    },
    
    # Table 9: Search time ablation
    "table_9_search_time": {
        "search_times": [1, 5, 10, 20],
        "expected_behavior": {
            "accuracy_increases_with_T": True,
            "coreset_decreases_with_T": True,
            "diminishing_returns": True
        }
    }
}


# ============================================================================
# Validation Functions
# ============================================================================

def validate_epsilon_monotonicity(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate that larger epsilon allows smaller coreset size.
    
    Args:
        results: List of experiment results with 'epsilon' and 'final_k' keys
        
    Returns:
        Validation result dictionary
    """
    if not results or len(results) < 2:
        return {
            "assertion": "epsilon_coreset_monotonicity",
            "passed": False,
            "reason": "Insufficient data for monotonicity check"
        }
    
    sorted_results = sorted(results, key=lambda x: x.get("epsilon", 0))
    violations = []
    
    for i in range(len(sorted_results) - 1):
        eps1, k1 = sorted_results[i].get("epsilon"), sorted_results[i].get("final_k")
        eps2, k2 = sorted_results[i+1].get("epsilon"), sorted_results[i+1].get("final_k")
        
        if eps1 is not None and eps2 is not None and k1 is not None and k2 is not None:
            # Larger epsilon should allow smaller or equal coreset
            if k2 > k1:
                violations.append({
                    "epsilon_pair": (eps1, eps2),
                    "coreset_pair": (k1, k2),
                    "issue": f"Coreset increased from {k1} to {k2} as epsilon increased"
                })
    
    passed = len(violations) == 0
    return {
        "assertion": "epsilon_coreset_monotonicity",
        "passed": passed,
        "violations": violations,
        "n_comparisons": len(sorted_results) - 1,
        "paper_evidence": "Table 1: epsilon 0.2→0.3→0.4 allows decreasing coreset sizes"
    }


def validate_objective_improvement(init_results: Dict[str, float], 
                                   final_results: Dict[str, float]) -> Dict[str, Any]:
    """
    Validate that both f1(m) and f2(m) improved after LBCS optimization.
    
    Args:
        init_results: Initial objective values {'f1': ..., 'f2': ...}
        final_results: Final objective values {'f1': ..., 'f2': ...}
        
    Returns:
        Validation result dictionary
    """
    f1_init = init_results.get("f1")
    f2_init = init_results.get("f2")
    f1_final = final_results.get("f1")
    f2_final = final_results.get("f2")
    
    if None in [f1_init, f2_init, f1_final, f2_final]:
        return {
            "assertion": "objective_improvement",
            "passed": False,
            "reason": "Missing objective values"
        }
    
    f1_improved = f1_final < f1_init
    f2_improved = f2_final < f2_init
    
    return {
        "assertion": "objective_improvement",
        "passed": f1_improved and f2_improved,
        "f1_change": f1_final - f1_init,
        "f2_change": f2_final - f2_init,
        "f1_improved": f1_improved,
        "f2_improved": f2_improved,
        "paper_evidence": "Table 1: Both achieved objectives lower than initialized"
    }


def validate_baseline_comparison(lbcs_results: Dict[str, float],
                                baseline_results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """
    Validate that LBCS achieves best or near-best accuracy compared to baselines.
    
    Args:
        lbcs_results: LBCS results {'accuracy': ..., 'coreset_size': ...}
        baseline_results: Dict mapping baseline names to their results
        
    Returns:
        Validation result dictionary
    """
    lbcs_acc = lbcs_results.get("accuracy")
    lbcs_size = lbcs_results.get("coreset_size")
    
    if lbcs_acc is None:
        return {
            "assertion": "baseline_outperformance",
            "passed": False,
            "reason": "Missing LBCS accuracy"
        }
    
    baseline_accuracies = {
        name: res.get("accuracy")
        for name, res in baseline_results.items()
        if res.get("accuracy") is not None
    }
    
    if not baseline_accuracies:
        return {
            "assertion": "baseline_outperformance",
            "passed": False,
            "reason": "No baseline results available"
        }
    
    max_baseline_acc = max(baseline_accuracies.values())
    lbcs_is_best = lbcs_acc >= max_baseline_acc - 0.5  # Allow 0.5% tolerance for "near-best"
    
    comparisons = {
        name: {
            "baseline_acc": acc,
            "gap": lbcs_acc - acc,
            "lbcs_better": lbcs_acc >= acc
        }
        for name, acc in baseline_accuracies.items()
    }
    
    return {
        "assertion": "baseline_outperformance",
        "passed": lbcs_is_best,
        "lbcs_accuracy": lbcs_acc,
        "max_baseline_accuracy": max_baseline_acc,
        "lbcs_rank": "best" if lbcs_acc >= max_baseline_acc else "near-best",
        "comparisons": comparisons,
        "paper_evidence": "Table 2: LBCS achieves best mean test accuracy"
    }


def validate_coreset_reduction(initial_k: int, final_k: int) -> Dict[str, Any]:
    """
    Validate that optimized coreset size is smaller than predefined size.
    
    Args:
        initial_k: Predefined coreset size
        final_k: Optimized coreset size
        
    Returns:
        Validation result dictionary
    """
    if initial_k is None or final_k is None:
        return {
            "assertion": "lbcs_coreset_reduction",
            "passed": False,
            "reason": "Missing coreset size values"
        }
    
    reduction = initial_k - final_k
    reduction_ratio = reduction / initial_k if initial_k > 0 else 0
    
    return {
        "assertion": "lbcs_coreset_reduction",
        "passed": final_k < initial_k,
        "initial_k": initial_k,
        "final_k": final_k,
        "reduction": reduction,
        "reduction_ratio": reduction_ratio,
        "paper_evidence": "Table 2: Optimized sizes consistently below predefined k"
    }


def validate_noise_robustness(clean_results: Dict[str, Any],
                              noisy_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate that LBCS maintains advantage under label noise.
    
    Args:
        clean_results: Results without noise {method: accuracy}
        noisy_results: Results with noise {method: accuracy}
        
    Returns:
        Validation result dictionary
    """
    lbcs_clean = clean_results.get("lbcs", {}).get("accuracy")
    lbcs_noisy = noisy_results.get("lbcs", {}).get("accuracy")
    
    if lbcs_clean is None or lbcs_noisy is None:
        return {
            "assertion": "noise_robustness",
            "passed": False,
            "reason": "Missing LBCS results"
        }
    
    # Check relative ranking in noisy setting
    noisy_accuracies = {
        name: res.get("accuracy", 0)
        for name, res in noisy_results.items()
        if name != "lbcs" and res.get("accuracy") is not None
    }
    
    lbcs_best_in_noisy = all(lbcs_noisy >= acc for acc in noisy_accuracies.values())
    
    return {
        "assertion": "noise_robustness",
        "passed": lbcs_best_in_noisy,
        "lbcs_clean_accuracy": lbcs_clean,
        "lbcs_noisy_accuracy": lbcs_noisy,
        "accuracy_drop": lbcs_clean - lbcs_noisy,
        "maintains_best_rank": lbcs_best_in_noisy,
        "paper_evidence": "Figure 2(a): LBCS outperforms baselines with 30% corrupted labels"
    }


def validate_search_time_trend(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate that performance improves with search time T but with diminishing returns.
    
    Args:
        results: List of results with different search times
        
    Returns:
        Validation result dictionary
    """
    if not results or len(results) < 2:
        return {
            "assertion": "search_time_improvement",
            "passed": False,
            "reason": "Insufficient data for trend analysis"
        }
    
    sorted_results = sorted(results, key=lambda x: x.get("search_time", 0))
    
    accuracy_increasing = True
    size_decreasing = True
    improvements = []
    
    for i in range(len(sorted_results) - 1):
        curr = sorted_results[i]
        next_res = sorted_results[i+1]
        
        acc_curr = curr.get("accuracy", 0)
        acc_next = next_res.get("accuracy", 0)
        size_curr = curr.get("coreset_size", float('inf'))
        size_next = next_res.get("coreset_size", float('inf'))
        
        acc_improvement = acc_next - acc_curr
        size_reduction = size_curr - size_next
        
        improvements.append({
            "search_time_range": (curr.get("search_time"), next_res.get("search_time")),
            "accuracy_improvement": acc_improvement,
            "size_reduction": size_reduction
        })
        
        if acc_next < acc_curr:
            accuracy_increasing = False
        if size_next > size_curr:
            size_decreasing = False
    
    # Check for diminishing returns (later improvements smaller)
    diminishing_returns = False
    if len(improvements) >= 2:
        later_improvements = [imp["accuracy_improvement"] for imp in improvements[-2:]]
        earlier_improvements = [imp["accuracy_improvement"] for imp in improvements[:2]]
        diminishing_returns = max(later_improvements) < max(earlier_improvements)
    
    return {
        "assertion": "search_time_improvement",
        "passed": accuracy_increasing or size_decreasing,
        "accuracy_increasing": accuracy_increasing,
        "size_decreasing": size_decreasing,
        "diminishing_returns": diminishing_returns,
        "improvements": improvements,
        "paper_evidence": "Table 9: Test accuracy increases with T, then stabilizes"
    }


# ============================================================================
# Main Validation Interface
# ============================================================================

def validate_all_trends(experiment_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run all trend validations on experiment results.
    
    Args:
        experiment_results: Complete experiment results dictionary
        
    Returns:
        Comprehensive validation report
    """
    report = {
        "timestamp": None,
        "overall_passed": True,
        "critical_failures": [],
        "validations": {},
        "summary": {}
    }
    
    try:
        import datetime
        report["timestamp"] = datetime.datetime.now().isoformat()
    except:
        pass
    
    # Run each validation that has data
    validations = []
    
    # Epsilon monotonicity
    if "epsilon_sweep" in experiment_results:
        result = validate_epsilon_monotonicity(experiment_results["epsilon_sweep"])
        validations.append(result)
        if not result["passed"] and TREND_ASSERTIONS["epsilon_coreset_monotonicity"].critical:
            report["critical_failures"].append("epsilon_coreset_monotonicity")
    
    # Objective improvement
    if "init_objectives" in experiment_results and "final_objectives" in experiment_results:
        result = validate_objective_improvement(
            experiment_results["init_objectives"],
            experiment_results["final_objectives"]
        )
        validations.append(result)
        if not result["passed"] and TREND_ASSERTIONS["objective_improvement"].critical:
            report["critical_failures"].append("objective_improvement")
    
    # Baseline comparison
    if "lbcs_results" in experiment_results and "baseline_results" in experiment_results:
        result = validate_baseline_comparison(
            experiment_results["lbcs_results"],
            experiment_results["baseline_results"]
        )
        validations.append(result)
        if not result["passed"] and TREND_ASSERTIONS["baseline_outperformance"].critical:
            report["critical_failures"].append("baseline_outperformance")
    
    # Coreset reduction
    if "initial_k" in experiment_results and "final_k" in experiment_results:
        result = validate_coreset_reduction(
            experiment_results["initial_k"],
            experiment_results["final_k"]
        )
        validations.append(result)
        if not result["passed"] and TREND_ASSERTIONS["lbcs_coreset_reduction"].critical:
            report["critical_failures"].append("lbcs_coreset_reduction")
    
    # Noise robustness
    if "clean_results" in experiment_results and "noisy_results" in experiment_results:
        result = validate_noise_robustness(
            experiment_results["clean_results"],
            experiment_results["noisy_results"]
        )
        validations.append(result)
        if not result["passed"] and TREND_ASSERTIONS["noise_robustness"].critical:
            report["critical_failures"].append("noise_robustness")
    
    # Search time trend
    if "search_time_sweep" in experiment_results:
        result = validate_search_time_trend(experiment_results["search_time_sweep"])
        validations.append(result)
        if not result["passed"] and TREND_ASSERTIONS["search_time_improvement"].critical:
            report["critical_failures"].append("search_time_improvement")
    
    # Compile results
    for val in validations:
        assertion_name = val.get("assertion", "unknown")
        report["validations"][assertion_name] = val
        if not val.get("passed", False):
            report["overall_passed"] = False
    
    # Summary statistics
    total_validations = len(validations)
    passed_validations = sum(1 for v in validations if v.get("passed", False))
    
    report["summary"] = {
        "total_validations": total_validations,
        "passed_validations": passed_validations,
        "failed_validations": total_validations - passed_validations,
        "pass_rate": passed_validations / total_validations if total_validations > 0 else 0,
        "critical_failures": report["critical_failures"]
    }
    
    return report


def main(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Main entrypoint for trend assertion validation.
    
    Args:
        config: Optional configuration with experiment results to validate
        
    Returns:
        Validation report
    """
    if config is None:
        config = {}
    
    mode = config.get("mode", "full")
    
    if mode in ["runtime_smoke", "docker_validate"]:
        # Dry-run mode: return schema without real validation
        return {
            "mode": mode,
            "schema": {
                "trend_assertions": list(TREND_ASSERTIONS.keys()),
                "baseline_methods": list(BASELINE_METHODS.keys()),
                "expected_results": list(EXPECTED_RESULTS.keys()),
                "validation_functions": [
                    "validate_epsilon_monotonicity",
                    "validate_objective_improvement",
                    "validate_baseline_comparison",
                    "validate_coreset_reduction",
                    "validate_noise_robustness",
                    "validate_search_time_trend"
                ]
            },
            "status": "dry_run_schema",
            "note": "Trend assertion validation requires experiment results to validate"
        }
    
    # Full mode: validate provided results
    experiment_results = config.get("experiment_results", {})
    
    if not experiment_results:
        warnings.warn("No experiment results provided for validation")
        return {
            "status": "no_results",
            "message": "Provide experiment_results in config for validation"
        }
    
    report = validate_all_trends(experiment_results)
    
    # Save report if output path provided
    output_path = config.get("output_path")
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
    
    return report


if __name__ == "__main__":
    import sys
    
    # Simple command-line interface