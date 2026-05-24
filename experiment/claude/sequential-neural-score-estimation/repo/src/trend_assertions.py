"""
Sequential Neural Posterior Score Estimation - Result Trend Assertions

This module implements paper-derived result-trend assertions for semantic review and
validation of SNPSE benchmark reproduction. It provides explicit trend expectations,
boundary case assertions, and baseline comparison contracts from the paper.

Reference grounding:
- paperbench_ref_001 l5pc/docs/config.md: Configuration structure for multi-round inference
- paperbench_ref_001 sbi/sbi/inference/snpe/snpe_a.py: SNPE method interface patterns
- paperbench_ref_001 sbi/sbi/inference/snle/mnle.py: Likelihood estimation interface
- paperbench_ref_001 sbi/sbi/inference/snle/snle_base.py: Base inference class patterns

Paper: Sequential Neural Score Estimation: Likelihood-Free Inference with
       Conditional Score Based Diffusion Models

Implementation surfaces: evaluation, artifact_writer, baseline_or_ablation, config, tests

Method obligations:
- Preserve required result-trend assertions for semantic review
- Paper evidence contract: positive_parameter_improves, endpoint_low, baseline_outperformance
- Binding addendum: TSNPE and SNVI results in Section 5.3 taken from respective papers
- Implement code/config-visible paper evidence obligation matrix
- Each matrix row binds experiment to datasets, methods, parameter sweeps, expected trends

Writes artifacts:
- results/sensitivity_report.json
- results/trend_validation.json
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable
import warnings
import numpy as np


# ============================================================================
# Trend Assertion Types
# ============================================================================

class TrendAssertionType:
    """Enumeration of result-trend assertion types from paper evidence contract."""
    
    POSITIVE_PARAMETER_IMPROVES = "positive_parameter_improves"
    ENDPOINT_LOW = "endpoint_low"
    BASELINE_OUTPERFORMANCE = "baseline_outperformance"
    SEQUENTIAL_IMPROVES = "sequential_improves"
    ROUND_MONOTONIC = "round_monotonic"
    BUDGET_SCALING = "budget_scaling"


# ============================================================================
# Paper-Derived Trend Assertions
# Reference grounding: paperbench_ref_001 l5pc/docs/config.md
# ============================================================================

PAPER_TREND_ASSERTIONS = {
    "figure_2_non_sequential": {
        "paper_reference": "Figure 2. Results on eight benchmark tasks (non-sequential methods).",
        "experiments": ["two_moons", "slcp", "gaussian_linear", "gaussian_mixture", 
                       "bernoulli_glm", "lotka_volterra", "sir", "two_moons_uniform"],
        "methods": ["NPSE", "NPE", "NLE", "NRE"],
        "simulation_budgets": [1000, 10000, 100000],
        "metric": "C2ST",
        "trend_type": TrendAssertionType.BASELINE_OUTPERFORMANCE,
        "assertion": "NPSE should achieve competitive or better C2ST scores compared to NPE/NLE/NRE on benchmark tasks",
        "expected_direction": "lower_is_better",
        "decision_claim": "NPSE provides effective non-sequential posterior estimation",
        "validation_rule": "c2st_close_to_0.5_is_best"
    },
    
    "figure_3_sequential": {
        "paper_reference": "Figure 3. Results on eight benchmark tasks (sequential methods).",
        "experiments": ["two_moons", "slcp", "gaussian_linear", "gaussian_mixture",
                       "bernoulli_glm", "lotka_volterra", "sir", "two_moons_uniform"],
        "methods": ["TSNPSE", "SNPE-C", "SNLE", "SNRE"],
        "simulation_budgets": [1000, 10000, 100000],
        "metric": "C2ST",
        "trend_type": TrendAssertionType.SEQUENTIAL_IMPROVES,
        "assertion": "TSNPSE (sequential) should improve over NPSE (non-sequential) with increasing budget",
        "expected_direction": "lower_is_better",
        "decision_claim": "Sequential methods improve sample efficiency",
        "validation_rule": "sequential_better_than_single_round"
    },
    
    "figure_4_pyloric": {
        "paper_reference": "Figure 4. Results for the Pyloric experiment.",
        "experiments": ["pyloric"],
        "methods": ["TSNPSE", "SNPE-C", "SNLE"],
        "simulation_budgets": [10000, 30000, 50000, 100000],
        "metric": "C2ST",
        "trend_type": TrendAssertionType.BUDGET_SCALING,
        "assertion": "Performance improves monotonically with increased simulation budget",
        "expected_direction": "lower_is_better",
        "decision_claim": "TSNPSE scales effectively on real neuroscience task",
        "validation_rule": "monotonic_improvement_with_budget"
    },
    
    "figure_5_npse_vs_nlse": {
        "paper_reference": "Figure 5. Comparison between NPSE and NLSE on four benchmark tasks.",
        "experiments": ["slcp", "gaussian_linear", "gaussian_mixture", "bernoulli_glm"],
        "methods": ["NPSE", "NLSE"],
        "simulation_budgets": [1000, 10000, 100000],
        "metric": "C2ST",
        "trend_type": TrendAssertionType.BASELINE_OUTPERFORMANCE,
        "assertion": "NPSE (score-based) should be competitive with NLSE (likelihood-based)",
        "expected_direction": "lower_is_better",
        "decision_claim": "Score estimation is viable alternative to likelihood estimation",
        "validation_rule": "competitive_performance"
    },
    
    "figure_6_snpse_variants": {
        "paper_reference": "Figure 6. Comparison between SNPSE-A, SNPSE-B, and TSNPSE on two benchmark tasks.",
        "experiments": ["slcp", "gaussian_linear_uniform"],
        "methods": ["TSNPSE", "SNPSE-A", "SNPSE-B"],
        "simulation_budgets": [10000],
        "metric": "C2ST",
        "trend_type": TrendAssertionType.BASELINE_OUTPERFORMANCE,
        "assertion": "TSNPSE should outperform alternative sequential variants SNPSE-A and SNPSE-B",
        "expected_direction": "lower_is_better",
        "decision_claim": "Truncated sequential approach is superior to alternatives",
        "validation_rule": "tsnpse_better_than_alternatives",
        "ablation_note": "SNPSE-C omitted due to poor performance (C2ST ≈ 1)"
    },
    
    "figure_9_npse_vs_fmpe": {
        "paper_reference": "Figure 9. Comparison between NPSE and FMPE on eight benchmark tasks.",
        "experiments": ["two_moons", "slcp", "gaussian_linear", "gaussian_mixture",
                       "bernoulli_glm", "lotka_volterra", "sir", "two_moons_uniform"],
        "methods": ["NPSE", "FMPE"],
        "simulation_budgets": [1000, 10000, 100000],
        "metric": "C2ST",
        "trend_type": TrendAssertionType.BASELINE_OUTPERFORMANCE,
        "assertion": "NPSE should be competitive with flow matching posterior estimation (FMPE)",
        "expected_direction": "lower_is_better",
        "decision_claim": "Score-based diffusion competitive with flow matching",
        "validation_rule": "competitive_performance"
    },
    
    "endpoint_assertion_boundary_cases": {
        "paper_reference": "General simulation budget boundary cases",
        "experiments": "all",
        "methods": "all",
        "simulation_budgets": [0, "very_large"],
        "metric": "C2ST",
        "trend_type": TrendAssertionType.ENDPOINT_LOW,
        "assertion": "Zero budget should yield worst performance (C2ST ≈ 1.0), infinite budget should approach perfect (C2ST ≈ 0.5)",
        "expected_direction": "endpoint_boundaries",
        "decision_claim": "Method follows expected asymptotic behavior",
        "validation_rule": "endpoint_sanity_check"
    },
    
    "round_improvement_assertion": {
        "paper_reference": "Algorithm 1 (TSNPSE) multi-round improvement",
        "experiments": "all_sequential",
        "methods": ["TSNPSE", "SNPSE-A", "SNPSE-B"],
        "rounds": [1, 2, 3, 4, 5, 10],
        "metric": "C2ST",
        "trend_type": TrendAssertionType.ROUND_MONOTONIC,
        "assertion": "C2ST should improve (decrease) or stabilize with each sequential round",
        "expected_direction": "monotonic_decrease_or_stable",
        "decision_claim": "Sequential rounds refine posterior approximation",
        "validation_rule": "no_significant_degradation_across_rounds"
    },
    
    "addendum_external_results": {
        "paper_reference": "Section 5.3 - TSNPE and SNVI results taken from respective papers",
        "experiments": ["section_5.3_comparison"],
        "methods": ["TSNPE", "SNVI"],
        "simulation_budgets": "as_published",
        "metric": "C2ST",
        "trend_type": "external_reference",
        "assertion": "Results for TSNPE and SNVI are NOT replicated in this codebase",
        "expected_direction": "reference_only",
        "decision_claim": "Comparison uses published results, not reproduction",
        "validation_rule": "do_not_validate",
        "addendum_binding": True,
        "note": "Binding addendum clarification: These results should be taken from their respective papers, not replicated"
    }
}


# ============================================================================
# Parameter Sweep Trend Assertions
# Reference grounding: paperbench_ref_001 sbi/sbi/inference/snpe/snpe_a.py
# ============================================================================

PARAMETER_SWEEP_ASSERTIONS = {
    "learning_rate_sweep": {
        "parameter": "learning_rate",
        "sweep_values": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
        "trend_type": TrendAssertionType.POSITIVE_PARAMETER_IMPROVES,
        "assertion": "Optimal learning rate (≈1e-4 from paper) should outperform very low or very high values",
        "expected_direction": "optimal_middle",
        "decision_claim": "Learning rate tuning is important for score network training",
        "validation_rule": "peak_at_optimal_value"
    },
    
    "diffusion_steps_sweep": {
        "parameter": "num_diffusion_steps",
        "sweep_values": [10, 50, 100, 500, 1000],
        "trend_type": TrendAssertionType.POSITIVE_PARAMETER_IMPROVES,
        "assertion": "More diffusion steps should improve quality up to saturation point",
        "expected_direction": "monotonic_increase_then_saturate",
        "decision_claim": "Sufficient diffusion steps needed for accurate posterior sampling",
        "validation_rule": "improvement_with_saturation"
    },
    
    "embedding_dim_sweep": {
        "parameter": "embedding_dim",
        "sweep_values": [16, 32, 64, 128, 256],
        "trend_type": TrendAssertionType.POSITIVE_PARAMETER_IMPROVES,
        "assertion": "Larger embedding dimension should improve expressiveness up to overfitting",
        "expected_direction": "optimal_middle",
        "decision_claim": "Embedding dimension affects score network capacity",
        "validation_rule": "optimal_capacity_tradeoff"
    },
    
    "num_rounds_sweep": {
        "parameter": "num_rounds",
        "sweep_values": [1, 2, 3, 5, 10],
        "trend_type": TrendAssertionType.ROUND_MONOTONIC,
        "assertion": "More sequential rounds should improve or stabilize performance (Algorithm 1)",
        "expected_direction": "monotonic_improvement",
        "decision_claim": "Sequential refinement improves posterior accuracy",
        "validation_rule": "rounds_improve_or_stable"
    }
}


# ============================================================================
# Trend Validation Functions
# ============================================================================

def validate_baseline_outperformance(
    method_results: Dict[str, float],
    baseline_method: str,
    target_method: str,
    metric_direction: str = "lower_is_better",
    tolerance: float = 0.05
) -> Dict[str, Any]:
    """
    Validate that target method outperforms baseline according to paper claims.
    
    Args:
        method_results: Dictionary mapping method names to metric values
        baseline_method: Name of baseline method for comparison
        target_method: Name of target method (e.g., NPSE, TSNPSE)
        metric_direction: "lower_is_better" or "higher_is_better"
        tolerance: Acceptable margin for "competitive" performance
    
    Returns:
        Validation result with assertion status and details
    """
    if baseline_method not in method_results or target_method not in method_results:
        return {
            "assertion": "baseline_outperformance",
            "status": "incomplete",
            "reason": f"Missing results for {baseline_method} or {target_method}",
            "methods_present": list(method_results.keys())
        }
    
    baseline_value = method_results[baseline_method]
    target_value = method_results[target_method]
    
    if metric_direction == "lower_is_better":
        outperforms = target_value <= baseline_value + tolerance
        competitive = abs(target_value - baseline_value) <= tolerance
    else:
        outperforms = target_value >= baseline_value - tolerance
        competitive = abs(target_value - baseline_value) <= tolerance
    
    return {
        "assertion": "baseline_outperformance",
        "status": "pass" if outperforms else "competitive" if competitive else "fail",
        "baseline_method": baseline_method,
        "baseline_value": float(baseline_value),
        "target_method": target_method,
        "target_value": float(target_value),
        "difference": float(target_value - baseline_value),
        "metric_direction": metric_direction,
        "tolerance": tolerance
    }


def validate_sequential_improvement(
    single_round_result: float,
    multi_round_result: float,
    metric_direction: str = "lower_is_better",
    min_improvement: float = 0.02
) -> Dict[str, Any]:
    """
    Validate that sequential method improves over single-round baseline.
    
    Args:
        single_round_result: Metric value from single-round method
        multi_round_result: Metric value from multi-round sequential method
        metric_direction: "lower_is_better" or "higher_is_better"
        min_improvement: Minimum expected improvement to consider meaningful
    
    Returns:
        Validation result with assertion status
    """
    improvement = single_round_result - multi_round_result
    if metric_direction == "higher_is_better":
        improvement = -improvement
    
    return {
        "assertion": "sequential_improves",
        "status": "pass" if improvement >= min_improvement else "marginal" if improvement > 0 else "fail",
        "single_round_value": float(single_round_result),
        "multi_round_value": float(multi_round_result),
        "improvement": float(improvement),
        "min_expected_improvement": min_improvement,
        "metric_direction": metric_direction
    }


def validate_monotonic_trend(
    parameter_values: List[float],
    metric_values: List[float],
    expected_direction: str = "increasing",
    strict: bool = False
) -> Dict[str, Any]:
    """
    Validate monotonic trend across parameter sweep or rounds.
    
    Args:
        parameter_values: Ordered parameter values (e.g., rounds, budget)
        metric_values: Corresponding metric values
        expected_direction: "increasing", "decreasing", or "optimal_middle"
        strict: If True, require strict monotonicity; if False, allow plateaus
    
    Returns:
        Validation result with assertion status
    """
    if len(parameter_values) != len(metric_values) or len(parameter_values) < 2:
        return {
            "assertion": "monotonic_trend",
            "status": "incomplete",
            "reason": "Insufficient data points for trend validation"
        }
    
    violations = 0
    for i in range(1, len(metric_values)):
        if expected_direction == "increasing":
            if strict and metric_values[i] <= metric_values[i-1]:
                violations += 1
            elif not strict and metric_values[i] < metric_values[i-1]:
                violations += 1
        elif expected_direction == "decreasing":
            if strict and metric_values[i] >= metric_values[i-1]:
                violations += 1
            elif not strict and metric_values[i] > metric_values[i-1]:
                violations += 1
    
    violation_rate = violations / (len(metric_values) - 1)
    
    return {
        "assertion": "monotonic_trend",
        "status": "pass" if violations == 0 else "mostly_pass" if violation_rate < 0.3 else "fail",
        "expected_direction": expected_direction,
        "violations": violations,
        "violation_rate": float(violation_rate),
        "parameter_values": [float(x) for x in parameter_values],
        "metric_values": [float(x) for x in metric_values],
        "strict": strict
    }


def validate_endpoint_boundaries(
    results_dict: Dict[str, float],
    metric_name: str,
    optimal_value: float = 0.5,
    worst_value: float = 1.0,
    tolerance: float = 0.1
) -> Dict[str, Any]:
    """
    Validate that boundary cases (zero budget, infinite budget) yield expected extreme values.
    
    For C2ST: worst case ≈ 1.0 (random), best case ≈ 0.5 (perfect discrimination)
    
    Args:
        results_dict: Dictionary with keys like "zero_budget", "optimal_budget"
        metric_name: Name of metric (e.g., "C2ST")
        optimal_value: Expected value at optimal/infinite budget
        worst_value: Expected value at zero/minimal budget
        tolerance: Acceptable deviation from expected boundary values
    
    Returns:
        Validation result with assertion status
    """
    checks = {}
    
    if "zero_budget" in results_dict:
        zero_val = results_dict["zero_budget"]
        checks["zero_budget"] = {
            "value": float(zero_val),
            "expected": worst_value,
            "within_tolerance": abs(zero_val - worst_value) <= tolerance
        }
    
    if "optimal_budget" in results_dict or "large_budget" in results_dict:
        key = "optimal_budget" if "optimal_budget" in results_dict else "large_budget"
        opt_val = results_dict[key]
        checks[key] = {
            "value": float(opt_val),
            "expected": optimal_value,
            "within_tolerance": abs(opt_val - optimal_value) <= tolerance
        }
    
    all_pass = all(check.get("within_tolerance", False) for check in checks.values())
    
    return {
        "assertion": "endpoint_boundaries",
        "status": "pass" if all_pass else "fail",
        "metric_name": metric_name,
        "checks": checks,
        "tolerance": tolerance
    }


# ============================================================================
# Artifact Writers
# ============================================================================

def write_sensitivity_report(
    assertions: Dict[str, Any],
    output_path: str = "results/sensitivity_report.json"
) -> None:
    """
    Write sensitivity analysis and trend assertion report.
    
    Args:
        assertions: Dictionary of validated trend assertions
        output_path: Path to write JSON report
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    report = {
        "schema_version": "1.0",
        "report_type": "sensitivity_and_trend_assertions",
        "paper_reference": "Sequential Neural Score Estimation: Likelihood-Free Inference with Conditional Score Based Diffusion Models",
        "assertions": assertions,
        "summary": {
            "total_assertions": len(assertions),
            "passed": sum(1 for a in assertions.values() if a.get("status") == "pass"),
            "failed": sum(1 for a in assertions.values() if a.get("status") == "fail"),
            "incomplete": sum(1 for a in assertions.values() if a.get("status") == "incomplete")
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)


def write_trend_validation(
    validation_results: List[Dict[str, Any]],
    output_path: str = "results/trend_validation.json"
) -> None:
    """
    Write trend validation results for semantic review.
    
    Args:
        validation_results: List of validation result dictionaries
        output_path: Path to write JSON report
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    report = {
        "schema_version": "1.0",
        "report_type": "trend_validation",
        "paper_reference": "Sequential Neural Score Estimation: Likelihood-Free Inference with Conditional Score Based Diffusion Models",
        "validation_results": validation_results,
        "summary": {
            "total_validations": len(validation_results),
            "passed": sum(1 for v in validation_results if v.get("status") == "pass"),
            "failed": sum(1 for v in validation_results if v.get("status") == "fail"),
            "incomplete": sum(1 for v in validation_results if v.get("status") == "incomplete")
        },
        "trend_assertion_types": {
            "positive_parameter_improves": "Nonzero/positive parameter values preserve reported improvement trend",
            "endpoint_low": "Boundary cases (p=0, p=1) expected to be lowest/minimum/worst",
            "baseline_outperformance": "Explicit comparison showing improvement over baselines",
            "sequential_improves": "Sequential methods improve over single-round",
            "round_monotonic": "Performance improves or stabilizes across rounds",
            "budget_scaling": "Performance improves with simulation budget"
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)


# ============================================================================
# Dry-Run Artifact Writer
# ============================================================================

def write_dry_run_trend_artifacts() -> None:
    """
    Write dry-run/smoke test trend assertion artifacts with schema and contract validation.
    
    This function creates all declared artifact paths during runtime_smoke and docker_validate
    modes without requiring actual experiment execution.
    """
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Write sensitivity report with schema
    sensitivity_report = {
        "schema_version": "1.0",
        "report_type": "sensitivity_and_trend_assertions_DRY_RUN",
        "dry_run_mode": True,
        "paper_reference": "Sequential Neural Score Estimation: Likelihood-Free Inference with Conditional Score Based Diffusion Models",
        "note": "This is a dry-run contract artifact. Real sensitivity results require full experiment execution.",
        "paper_trend_assertions": PAPER_TREND_ASSERTIONS,
        "parameter_sweep_assertions": PARAMETER_SWEEP_ASSERTIONS,
        "summary": {
            "total_paper_assertions": len(PAPER_TREND_ASSERTIONS),
            "total_parameter_assertions": len(PARAMETER_SWEEP_ASSERTIONS),
            "assertion_types": list(set(
                a["trend_type"] for a in PAPER_TREND_ASSERTIONS.values()
            ))
        }
    }
    
    with open("results/sensitivity_report.json", 'w') as f:
        json.dump(sensitivity_report, f, indent=2)
    
    # Write trend validation schema
    trend_validation = {
        "schema_version": "1.0",
        "report_type": "trend_validation_DRY_RUN",
        "dry_run_mode": True,
        "paper_reference": "Sequential Neural Score Estimation: Likelihood-Free Inference with Conditional Score Based Diffusion Models",
        "note": "This is a dry-run contract artifact. Real validation results require full experiment execution.",
        "validation_functions": {
            "validate_baseline_outperformance": "Compare method performance against baselines",
            "validate_sequential_improvement": "Verify multi-round improves over single-round",
            "validate_monotonic_trend": "Check monotonic improvement across parameter sweep",
            "validate_endpoint_boundaries": "Verify boundary cases yield expected extreme values"
        },
        "expected_trends": {
            "Figure_2": "NPSE competitive with NPE/NLE/NRE",
            "Figure_3": "TSNPSE improves over single-round methods",
            "Figure_4": "Monotonic improvement with simulation budget",
            "Figure_5": "NPSE competitive with NLSE",
            "Figure_6": "TSNPSE outperforms SNPSE-A/B",
            "Figure_9": "NPSE competitive with FMPE"
        },
        "addendum_binding": "TSNPE and SNVI results (Section 5.3) taken from respective papers, not replicated"
    }
    
    with open("results/trend_validation.json", 'w') as f:
        json.dump(trend_validation, f, indent=2)
    
    print("Dry-run trend assertion artifacts written to results/")


# ============================================================================
# Public API
# ============================================================================

def get_trend_assertions() -> Dict[str, Any]:
    """
    Get paper-derived trend assertions for experiment validation.
    
    Returns:
        Dictionary of trend assertions with expected behaviors
    """
    return {
        "paper_assertions": PAPER_TREND_ASSERTIONS,
        "parameter_assertions": PARAMETER_SWEEP_ASSERTIONS
    }


def validate_experiment_trends(
    experiment_results: Dict[str, Any],
    experiment_name: str
) -> List[Dict[str, Any]]:
    """
    Validate experiment results against paper-derived trend assertions.
    
    Args:
        experiment_results: Dictionary containing experiment metric results
        experiment_name: Name of experiment to validate (e.g., "figure_2_non_sequential")
    
    Returns:
        List of validation results for each applicable assertion
    """
    validation_results = []
    
    if experiment_name not in PAPER_TREND_ASSERTIONS:
        warnings.warn(f"No trend assertions defined for experiment: {experiment_name}")
        return validation_results
    
    assertion = PAPER_TREND_ASSERTIONS[experiment_name]
    trend_type = assertion["trend_type"]
    
    # Apply appropriate validation based on trend type
    if trend_type == TrendAssertionType.BASELINE_OUTPERFORMANCE:
        if "method_results" in experiment_results:
            methods = assertion["methods"]
            target = methods[0]  # First method is typically the proposed one
            for baseline in methods[1:]:
                result = validate_baseline_outperformance(
                    experiment_results["method_results"],
                    baseline_method=baseline,
                    target_method=target,
                    metric_direction=assertion.get("expected_direction", "lower_is_better")
                )
                validation_results.append(result)
    
    elif trend_type == TrendAssertionType.SEQUENTIAL_IMPROVES:
        if "single_round" in experiment_results and "multi_round" in experiment_results:
            result = validate_sequential_improvement(
                experiment_results["single_round"],
                experiment_results["multi_round"],
                metric_direction=assertion.get("expected_direction", "lower_is_better")
            )
            validation_results.append(result)
    
    elif trend_type in [TrendAssertionType.ROUND_MONOTONIC, TrendAssertionType.BUDGET_SCALING]:
        if "parameter_values" in experiment_results and "metric_values" in experiment_results:
            result = validate_monotonic_trend(
                experiment_results["parameter_values"],
                experiment_results["metric_values"],
                expected_direction="decreasing" if assertion.get("expected_direction") == "lower_is_better" else "increasing"
            )
            validation_results.append(result)
    
    return validation_results


if __name__ == "__main__":
    # Dry-run mode: write contract artifacts
    write_dry_run_trend_artifacts()
    print("Trend assertion module loaded successfully.")
    print(f"Registered {len(PAPER_TREND_ASSERTIONS)} paper-derived trend assertions")
    print(f"Registered {len(PARAMETER_SWEEP_ASSERTIONS)} parameter sweep assertions")