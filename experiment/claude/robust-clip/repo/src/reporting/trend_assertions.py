"""
Trend assertions and result validation for Robust CLIP reproduction.

This module provides trend assertion functions that validate experimental results
against expected patterns from the paper. It implements semantic review checks
for key findings and baseline comparisons.

Paper evidence contract:
- Original CLIP completely broken by attack (accuracy → 0% at ε=4/255)
- FARE⁴ best at ε=2/255 on average across zero-shot datasets
- FARE maintains clean performance better than TeCoA
- Baseline outperformance: FARE consistently better than TeCoA
- Sweep insensitive: hyperparameters stable across weight decay values

Method obligations:
- Preserve result-trend assertions for semantic review
- Validate baseline_outperformance: proposed method vs explicit baselines
- Check sweep_insensitive: stable/robust parameter-sweep behavior
"""

import os
import json
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# Trend Assertion Classes
# ============================================================================

@dataclass
class TrendAssertion:
    """A single trend assertion from the paper."""
    name: str
    description: str
    assertion_type: str  # 'robustness_degradation', 'baseline_outperformance', 'sweep_insensitive'
    expected_pattern: Dict[str, Any]
    tolerance: float = 0.05
    critical: bool = True


@dataclass
class AssertionResult:
    """Result of a trend assertion check."""
    assertion_name: str
    passed: bool
    message: str
    actual_values: Dict[str, Any]
    expected_values: Dict[str, Any]
    deviation: Optional[float] = None


# ============================================================================
# Paper Trend Assertions (Evidence Contract)
# ============================================================================

PAPER_TREND_ASSERTIONS = {
    # Table 4: Original CLIP completely non-robust
    'clip_robustness_failure': TrendAssertion(
        name='clip_robustness_failure',
        description='Original CLIP completely broken by adversarial attack',
        assertion_type='robustness_degradation',
        expected_pattern={
            'model': 'clip',
            'clean_accuracy_min': 0.60,  # Strong clean performance
            'adv_accuracy_max_eps2': 0.05,  # ≈0% at ε=2/255
            'adv_accuracy_max_eps4': 0.01,  # ≈0% at ε=4/255
        },
        tolerance=0.05,
        critical=True
    ),
    
    # Table 4: FARE best robustness at ε=2/255
    'fare_best_robustness': TrendAssertion(
        name='fare_best_robustness',
        description='FARE achieves best average robustness at ε=2/255',
        assertion_type='baseline_outperformance',
        expected_pattern={
            'model': 'fare',
            'epsilon': '2/255',
            'outperforms': ['clip', 'tecoa'],
            'metric': 'robust_accuracy',
            'margin_min': 0.02,  # At least 2% better
        },
        tolerance=0.02,
        critical=True
    ),
    
    # Table 1: FARE maintains clean performance better than TeCoA
    'fare_clean_performance': TrendAssertion(
        name='fare_clean_performance',
        description='FARE maintains better clean performance than TeCoA',
        assertion_type='baseline_outperformance',
        expected_pattern={
            'model': 'fare',
            'metric': 'clean_accuracy',
            'outperforms': ['tecoa'],
            'margin_min': 0.01,  # At least 1% better on average
        },
        tolerance=0.01,
        critical=True
    ),
    
    # Table 8: Hyperparameter sweep insensitivity (weight decay)
    'hyperparameter_stability': TrendAssertion(
        name='hyperparameter_stability',
        description='FARE performance stable across weight decay values',
        assertion_type='sweep_insensitive',
        expected_pattern={
            'model': 'fare',
            'parameter': 'weight_decay',
            'values': [0.0, 1e-4, 1e-3, 1e-2],
            'max_std_deviation': 0.02,  # <2% std across sweep
            'metric': 'robust_accuracy',
        },
        tolerance=0.02,
        critical=False
    ),
    
    # Table 1: FARE² vs TeCoA² comparison
    'fare2_vs_tecoa2': TrendAssertion(
        name='fare2_vs_tecoa2',
        description='FARE² outperforms TeCoA² on LVLM tasks',
        assertion_type='baseline_outperformance',
        expected_pattern={
            'model': 'fare2',
            'epsilon': '2/255',
            'outperforms': ['tecoa2'],
            'metrics': ['cider', 'vqa_accuracy'],
            'margin_min': 0.01,
        },
        tolerance=0.01,
        critical=True
    ),
    
    # Table 5: POPE hallucination comparison
    'fare_hallucination_resistance': TrendAssertion(
        name='fare_hallucination_resistance',
        description='FARE has lower hallucination than TeCoA on POPE',
        assertion_type='baseline_outperformance',
        expected_pattern={
            'model': 'fare',
            'metric': 'pope_f1',
            'outperforms': ['tecoa'],
            'margin_min': 0.05,  # Significant improvement
        },
        tolerance=0.02,
        critical=True
    ),
    
    # Table 6: SQA-I reasoning accuracy
    'fare_reasoning_accuracy': TrendAssertion(
        name='fare_reasoning_accuracy',
        description='FARE better than TeCoA on SQA-I reasoning',
        assertion_type='baseline_outperformance',
        expected_pattern={
            'model': 'fare',
            'metric': 'sqai_accuracy',
            'outperforms': ['tecoa'],
            'margin_min': 0.024,  # 2.4% improvement
        },
        tolerance=0.01,
        critical=True
    ),
    
    # Table 3: Targeted attack resistance
    'fare4_targeted_robustness': TrendAssertion(
        name='fare4_targeted_robustness',
        description='FARE⁴ completely robust to targeted attacks',
        assertion_type='robustness_degradation',
        expected_pattern={
            'model': 'fare4',
            'epsilon': '4/255',
            'attack_type': 'targeted',
            'success_rate_max': 0.01,  # ≈0% success
        },
        tolerance=0.02,
        critical=True
    ),
    
    # Table 7: Jailbreaking robustness
    'robust_jailbreak_defense': TrendAssertion(
        name='robust_jailbreak_defense',
        description='FARE/TeCoA significantly more robust than CLIP to jailbreaking',
        assertion_type='baseline_outperformance',
        expected_pattern={
            'models': ['fare', 'tecoa'],
            'metric': 'jailbreak_success_rate',
            'baseline': 'clip',
            'reduction_min': 0.30,  # At least 30% reduction
        },
        tolerance=0.05,
        critical=True
    ),
}


# ============================================================================
# Assertion Validation Functions
# ============================================================================

def validate_robustness_degradation(
    results: Dict[str, Any],
    assertion: TrendAssertion
) -> AssertionResult:
    """
    Validate that a model shows expected robustness degradation pattern.
    
    Args:
        results: Experimental results dictionary
        assertion: Trend assertion to validate
        
    Returns:
        AssertionResult with validation outcome
    """
    pattern = assertion.expected_pattern
    model_name = pattern['model']
    
    # Extract model results
    if model_name not in results:
        return AssertionResult(
            assertion_name=assertion.name,
            passed=False,
            message=f"Model '{model_name}' not found in results",
            actual_values={},
            expected_values=pattern
        )
    
    model_results = results[model_name]
    actual_values = {}
    
    # Check clean accuracy requirement
    if 'clean_accuracy_min' in pattern:
        clean_acc = model_results.get('clean_accuracy', 0.0)
        actual_values['clean_accuracy'] = clean_acc
        if clean_acc < pattern['clean_accuracy_min']:
            return AssertionResult(
                assertion_name=assertion.name,
                passed=False,
                message=f"Clean accuracy {clean_acc:.3f} below minimum {pattern['clean_accuracy_min']:.3f}",
                actual_values=actual_values,
                expected_values=pattern
            )
    
    # Check adversarial accuracy at ε=2/255
    if 'adv_accuracy_max_eps2' in pattern:
        adv_acc_2 = model_results.get('adv_accuracy_eps2', 1.0)
        actual_values['adv_accuracy_eps2'] = adv_acc_2
        if adv_acc_2 > pattern['adv_accuracy_max_eps2']:
            return AssertionResult(
                assertion_name=assertion.name,
                passed=False,
                message=f"Adversarial accuracy at ε=2/255 ({adv_acc_2:.3f}) above maximum {pattern['adv_accuracy_max_eps2']:.3f}",
                actual_values=actual_values,
                expected_values=pattern
            )
    
    # Check adversarial accuracy at ε=4/255
    if 'adv_accuracy_max_eps4' in pattern:
        adv_acc_4 = model_results.get('adv_accuracy_eps4', 1.0)
        actual_values['adv_accuracy_eps4'] = adv_acc_4
        if adv_acc_4 > pattern['adv_accuracy_max_eps4']:
            return AssertionResult(
                assertion_name=assertion.name,
                passed=False,
                message=f"Adversarial accuracy at ε=4/255 ({adv_acc_4:.3f}) above maximum {pattern['adv_accuracy_max_eps4']:.3f}",
                actual_values=actual_values,
                expected_values=pattern
            )
    
    # Check targeted attack success rate
    if 'success_rate_max' in pattern:
        success_rate = model_results.get('targeted_attack_success_rate', 1.0)
        actual_values['targeted_attack_success_rate'] = success_rate
        if success_rate > pattern['success_rate_max']:
            return AssertionResult(
                assertion_name=assertion.name,
                passed=False,
                message=f"Targeted attack success rate {success_rate:.3f} above maximum {pattern['success_rate_max']:.3f}",
                actual_values=actual_values,
                expected_values=pattern
            )
    
    return AssertionResult(
        assertion_name=assertion.name,
        passed=True,
        message=f"Robustness degradation pattern validated for {model_name}",
        actual_values=actual_values,
        expected_values=pattern
    )


def validate_baseline_outperformance(
    results: Dict[str, Any],
    assertion: TrendAssertion
) -> AssertionResult:
    """
    Validate that a model outperforms specified baselines.
    
    Args:
        results: Experimental results dictionary
        assertion: Trend assertion to validate
        
    Returns:
        AssertionResult with validation outcome
    """
    pattern = assertion.expected_pattern
    
    # Handle single model vs multiple models
    if 'model' in pattern:
        model_names = [pattern['model']]
    elif 'models' in pattern:
        model_names = pattern['models']
    else:
        return AssertionResult(
            assertion_name=assertion.name,
            passed=False,
            message="No model specified in assertion pattern",
            actual_values={},
            expected_values=pattern
        )
    
    # Get baseline models to compare against
    baseline_models = pattern.get('outperforms', [])
    if 'baseline' in pattern:
        baseline_models = [pattern['baseline']]
    
    # Get metric(s) to compare
    metrics = pattern.get('metrics', [pattern.get('metric', 'accuracy')])
    if isinstance(metrics, str):
        metrics = [metrics]
    
    actual_values = {}
    all_passed = True
    messages = []
    
    for model_name in model_names:
        if model_name not in results:
            return AssertionResult(
                assertion_name=assertion.name,
                passed=False,
                message=f"Model '{model_name}' not found in results",
                actual_values=actual_values,
                expected_values=pattern
            )
        
        model_results = results[model_name]
        
        for metric in metrics:
            model_value = model_results.get(metric, 0.0)
            actual_values[f'{model_name}_{metric}'] = model_value
            
            # Compare against each baseline
            for baseline in baseline_models:
                if baseline not in results:
                    continue
                
                baseline_value = results[baseline].get(metric, 0.0)
                actual_values[f'{baseline}_{metric}'] = baseline_value
                
                margin = model_value - baseline_value
                min_margin = pattern.get('margin_min', 0.0)
                
                # For success rate metrics, lower is better
                if 'success_rate' in metric or 'jailbreak' in metric:
                    margin = baseline_value - model_value  # Flip comparison
                    if 'reduction_min' in pattern:
                        min_margin = pattern['reduction_min']
                
                if margin < min_margin - assertion.tolerance:
                    all_passed = False
                    messages.append(
                        f"{model_name} does not outperform {baseline} on {metric}: "
                        f"{model_value:.3f} vs {baseline_value:.3f} (margin: {margin:.3f}, required: {min_margin:.3f})"
                    )
                else:
                    messages.append(
                        f"{model_name} outperforms {baseline} on {metric}: "
                        f"{model_value:.3f} vs {baseline_value:.3f} (margin: {margin:.3f})"
                    )
    
    return AssertionResult(
        assertion_name=assertion.name,
        passed=all_passed,
        message='; '.join(messages),
        actual_values=actual_values,
        expected_values=pattern
    )


def validate_sweep_insensitive(
    results: Dict[str, Any],
    assertion: TrendAssertion
) -> AssertionResult:
    """
    Validate that a model is insensitive to parameter sweep.
    
    Args:
        results: Experimental results dictionary
        assertion: Trend assertion to validate
        
    Returns:
        AssertionResult with validation outcome
    """
    pattern = assertion.expected_pattern
    model_name = pattern['model']
    parameter = pattern['parameter']
    metric = pattern['metric']
    max_std = pattern['max_std_deviation']
    
    # Check if sweep results exist
    sweep_key = f'{model_name}_sweep_{parameter}'
    if sweep_key not in results:
        # Try alternate key format
        sweep_key = f'{model_name}_hyperparameter_sweep'
        if sweep_key not in results:
            return AssertionResult(
                assertion_name=assertion.name,
                passed=False,
                message=f"Sweep results not found for {model_name} parameter {parameter}",
                actual_values={},
                expected_values=pattern
            )
    
    sweep_results = results[sweep_key]
    values = []
    actual_values = {}
    
    # Extract metric values across sweep
    for param_value in pattern['values']:
        key = f'{parameter}_{param_value}'
        if key in sweep_results:
            value = sweep_results[key].get(metric, None)
            if value is not None:
                values.append(value)
                actual_values[key] = value
    
    if len(values) < 2:
        return AssertionResult(
            assertion_name=assertion.name,
            passed=False,
            message=f"Insufficient sweep data: only {len(values)} values found",
            actual_values=actual_values,
            expected_values=pattern
        )
    
    # Compute standard deviation
    import numpy as np
    mean_value = np.mean(values)
    std_value = np.std(values)
    
    actual_values['mean'] = float(mean_value)
    actual_values['std'] = float(std_value)
    actual_values['values'] = [float(v) for v in values]
    
    if std_value > max_std + assertion.tolerance:
        return AssertionResult(
            assertion_name=assertion.name,
            passed=False,
            message=f"Sweep std {std_value:.4f} exceeds maximum {max_std:.4f}",
            actual_values=actual_values,
            expected_values=pattern,
            deviation=std_value
        )
    
    return AssertionResult(
        assertion_name=assertion.name,
        passed=True,
        message=f"Sweep insensitive: std={std_value:.4f} within tolerance (max={max_std:.4f})",
        actual_values=actual_values,
        expected_values=pattern,
        deviation=std_value
    )


# ============================================================================
# Main Validation Interface
# ============================================================================

def validate_trend_assertion(
    results: Dict[str, Any],
    assertion: TrendAssertion
) -> AssertionResult:
    """
    Validate a single trend assertion against experimental results.
    
    Args:
        results: Experimental results dictionary
        assertion: Trend assertion to validate
        
    Returns:
        AssertionResult with validation outcome
    """
    if assertion.assertion_type == 'robustness_degradation':
        return validate_robustness_degradation(results, assertion)
    elif assertion.assertion_type == 'baseline_outperformance':
        return validate_baseline_outperformance(results, assertion)
    elif assertion.assertion_type == 'sweep_insensitive':
        return validate_sweep_insensitive(results, assertion)
    else:
        return AssertionResult(
            assertion_name=assertion.name,
            passed=False,
            message=f"Unknown assertion type: {assertion.assertion_type}",
            actual_values={},
            expected_values=assertion.expected_pattern
        )


def validate_all_assertions(
    results: Dict[str, Any],
    critical_only: bool = False
) -> Dict[str, AssertionResult]:
    """
    Validate all paper trend assertions.
    
    Args:
        results: Experimental results dictionary
        critical_only: If True, only validate critical assertions
        
    Returns:
        Dictionary mapping assertion names to validation results
    """
    validation_results = {}
    
    for assertion_name, assertion in PAPER_TREND_ASSERTIONS.items():
        if critical_only and not assertion.critical:
            continue
        
        result = validate_trend_assertion(results, assertion)
        validation_results[assertion_name] = result
    
    return validation_results


def generate_assertion_report(
    validation_results: Dict[str, AssertionResult],
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a report summarizing assertion validation results.
    
    Args:
        validation_results: Dictionary of validation results
        output_path: Optional path to write JSON report
        
    Returns:
        Report dictionary with summary statistics
    """
    total = len(validation_results)
    passed = sum(1 for r in validation_results.values() if r.passed)
    failed = total - passed
    
    report = {
        'summary': {
            'total_assertions': total,
            'passed': passed,
            'failed': failed,
            'pass_rate': passed / total if total > 0 else 0.0
        },
        'assertions': {}
    }
    
    for name, result in validation_results.items():
        report['assertions'][name] = {
            'passed': result.passed,
            'message': result.message,
            'actual_values': result.actual_values,
            'expected_values': result.expected_values,
            'deviation': result.deviation
        }
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
    
    return report


# ============================================================================
# Smoke Test / Dry-Run Support
# ============================================================================

def create_dry_run_results() -> Dict[str, Any]:
    """
    Create synthetic results for dry-run/smoke testing that satisfy assertions.
    
    Returns:
        Dictionary of synthetic results matching expected patterns
    """
    return {
        'clip': {
            'clean_accuracy': 0.685,
            'adv_accuracy_eps2': 0.002,
            'adv_accuracy_eps4': 0.000,
            'robust_accuracy': 0.002,
            'targeted_attack_success_rate': 1.000,
        },
        'tecoa': {
            'clean_accuracy': 0.648,
            'adv_accuracy_eps2': 0.412,
            'adv_accuracy_eps4': 0.298,
            'robust_accuracy': 0.412,
            'cider': 0.892,
            'vqa_accuracy': 0.512,
            'pope_f1': 0.781,
            'sqai_accuracy': 0.621,
            'jailbreak_success_rate': 0.245,
        },
        'tecoa2': {
            'clean_accuracy': 0.648,
            'adv_accuracy_eps2': 0.412,
            'cider': 0.892,
            'vqa_accuracy': 0.512,
            'pope_f1': 0.781,
            'sqai_accuracy': 0.621,
        },
        'fare': {
            'clean_accuracy': 0.663,
            'adv_accuracy_eps2': 0.435,
            'adv_accuracy_eps4': 0.321,
            'robust_accuracy': 0.435,
            'cider': 0.908,
            'vqa_accuracy': 0.528,
            'pope_f1': 0.834,
            'sqai_accuracy': 0.645,
            'jailbreak_success_rate': 0.238,
        },
        'fare2': {
            'clean_accuracy': 0.663,
            'adv_accuracy_eps2': 0.435,
            'cider': 0.908,
            'vqa_accuracy': 0.528,
            'pope_f1': 0.834,
            'sqai_accuracy': 0.645,
        },
        'fare4': {
            'clean_accuracy': 0.641,
            'adv_accuracy_eps4': 0.321,
            'targeted_attack_success_rate': 0.000,
        },
        'fare_sweep_weight_decay': {
            'weight_decay_0.0': {'robust_accuracy': 0.432},
            'weight_decay_0.0001': {'robust_accuracy': 0.435},
            'weight_decay_0.001': {'robust_accuracy': 0.433},
            'weight_decay_0.01': {'robust_accuracy': 0.434},
        },
    }


def run_smoke_validation() -> bool:
    """
    Run smoke test validation with synthetic data.
    
    Returns:
        True if all critical assertions pass, False otherwise
    """
    print("Running trend assertion smoke validation...")
    
    # Create synthetic results
    dry_run_results = create_dry_run_results()
    
    # Validate critical assertions
    validation_results = validate_all_assertions(dry_run_results, critical_only=True)
    
    # Generate report
    report = generate_assertion_report(
        validation_results,
        output_path='results/trend_assertions_smoke.json'
    )
    
    # Print summary
    print(f"\nSmoke Validation Results:")
    print(f"  Total assertions: {report['summary']['total_assertions']}")
    print(f"  Passed: {report['summary']['passed']}")
    print(f"  Failed: {report['summary']['failed']}")
    print(f"  Pass rate: {report['summary']['pass_rate']:.1%}")
    
    for name, result in validation_results.items():
        status = "✓" if result.passed else "✗"
        print(f"  {status} {name}: {result.message}")
    
    return report['summary']['failed'] == 0


# ============================================================================
# CLI Interface
# ============================================================================

if __name__ == '__main__':
    import sys
    
    if '--smoke' in sys.argv:
        success = run_smoke_validation()
        sys.exit(0 if success else 1)
    else:
        print("Trend Assertions Module")
        print("=" * 60)
        print(f"Loaded {len(PAPER_TREND_ASSERTIONS)} paper trend assertions")
        print("\nAssertions:")
        for name, assertion in PAPER_TREND_ASSERTIONS.items():
            critical_flag = "[CRITICAL]" if assertion.critical else ""
            print(f"  - {name}: {assertion.description} {critical_flag}")
        print("\nRun with --smoke to test with synthetic data")