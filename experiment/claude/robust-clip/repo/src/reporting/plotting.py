"""
Plotting and artifact generation for Robust CLIP reproduction.

This module implements visualization and artifact writing for all paper tables
and figures, including radar plots, robustness curves, and metric comparison tables.

Paper evidence contract:
- Figure 1: Radar plot of zero-shot task performance (CLIP, TeCoA, FARE)
- Tables 1-14: Robustness metrics, ablations, and downstream task evaluations
- Metric schemas: accuracy, clean_accuracy, f1, precision, loss, cider, vqa_accuracy,
  success_rate, robust_accuracy, training_time, attack_success_rate
- Trend assertions: CLIP broken by attack, FARE best at ε=2/255, baseline outperformance
"""

import os
import json
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import numpy as np
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# Metric Computation and Aggregation
# ============================================================================

def compute_accuracy(predictions: List[Any], targets: List[Any]) -> float:
    """Compute classification accuracy."""
    if len(predictions) == 0:
        return 0.0
    correct = sum(1 for pred, target in zip(predictions, targets) if pred == target)
    return correct / len(predictions)


def compute_f1_score(predictions: List[Any], targets: List[Any], 
                     positive_class: Optional[Any] = None) -> float:
    """Compute F1 score for binary or multiclass classification."""
    if len(predictions) == 0:
        return 0.0
    
    if positive_class is not None:
        # Binary F1
        tp = sum(1 for pred, target in zip(predictions, targets) 
                if pred == positive_class and target == positive_class)
        fp = sum(1 for pred, target in zip(predictions, targets) 
                if pred == positive_class and target != positive_class)
        fn = sum(1 for pred, target in zip(predictions, targets) 
                if pred != positive_class and target == positive_class)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return f1
    else:
        # Macro-averaged F1
        unique_classes = set(targets)
        f1_scores = []
        for cls in unique_classes:
            f1 = compute_f1_score(predictions, targets, positive_class=cls)
            f1_scores.append(f1)
        return np.mean(f1_scores) if f1_scores else 0.0


def compute_robust_accuracy(clean_predictions: List[Any], 
                           adversarial_predictions: List[Any],
                           targets: List[Any]) -> float:
    """Compute robust accuracy (correct on both clean and adversarial)."""
    if len(targets) == 0:
        return 0.0
    
    robust_correct = sum(
        1 for clean_pred, adv_pred, target in zip(clean_predictions, adversarial_predictions, targets)
        if clean_pred == target and adv_pred == target
    )
    return robust_correct / len(targets)


def compute_attack_success_rate(clean_predictions: List[Any],
                                adversarial_predictions: List[Any],
                                targets: List[Any]) -> float:
    """Compute attack success rate (clean correct, adversarial incorrect)."""
    if len(targets) == 0:
        return 0.0
    
    initially_correct = [(clean_pred == target, adv_pred, target) 
                         for clean_pred, adv_pred, target in zip(clean_predictions, adversarial_predictions, targets)
                         if clean_pred == target]
    
    if len(initially_correct) == 0:
        return 0.0
    
    successful_attacks = sum(1 for _, adv_pred, target in initially_correct if adv_pred != target)
    return successful_attacks / len(initially_correct)


def aggregate_metrics(results: Dict[str, Any]) -> Dict[str, float]:
    """Aggregate evaluation metrics across datasets."""
    aggregated = {}
    
    # Collect all metric keys
    metric_keys = set()
    for dataset_results in results.values():
        if isinstance(dataset_results, dict):
            metric_keys.update(dataset_results.keys())
    
    # Average each metric across datasets
    for metric_key in metric_keys:
        values = []
        for dataset_results in results.values():
            if isinstance(dataset_results, dict) and metric_key in dataset_results:
                val = dataset_results[metric_key]
                if isinstance(val, (int, float)) and not np.isnan(val):
                    values.append(val)
        
        if values:
            aggregated[f"{metric_key}_mean"] = np.mean(values)
            aggregated[f"{metric_key}_std"] = np.std(values)
    
    return aggregated


# ============================================================================
# Figure Generation (Lazy Import)
# ============================================================================

def generate_radar_plot(data: Dict[str, Dict[str, float]], 
                       output_path: str,
                       title: str = "Zero-Shot Task Performance",
                       dry_run: bool = False) -> None:
    """
    Generate radar plot for Figure 1: zero-shot task performance comparison.
    
    Args:
        data: Dict mapping model names to task performance dicts
        output_path: Path to save figure
        title: Plot title
        dry_run: If True, create minimal schema artifact
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        # Fallback: create minimal marker file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write("# Radar plot placeholder (matplotlib not available)\n")
        return
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if dry_run:
        # Create schema artifact
        fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(projection='polar'))
        ax.text(0.5, 0.5, "DRY-RUN SCHEMA ARTIFACT\nFigure 1: Radar Plot", 
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        return
    
    # Extract task names and normalize data
    tasks = list(next(iter(data.values())).keys())
    num_tasks = len(tasks)
    
    # Compute maximum value for normalization
    max_values = {}
    for task in tasks:
        max_val = max(model_data.get(task, 0.0) for model_data in data.values())
        max_values[task] = max_val if max_val > 0 else 1.0
    
    # Create radar plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='polar')
    
    angles = np.linspace(0, 2 * np.pi, num_tasks, endpoint=False).tolist()
    angles += angles[:1]  # Close the plot
    
    colors = {'CLIP': '#1f77b4', 'TeCoA²': '#ff7f0e', 'TeCoA⁴': '#d62728', 
              'FARE²': '#2ca02c', 'FARE⁴': '#9467bd'}
    
    for model_name, model_data in data.items():
        values = [model_data.get(task, 0.0) / max_values[task] for task in tasks]
        values += values[:1]  # Close the plot
        
        color = colors.get(model_name, '#000000')
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f"{task}\n(max={max_values[task]:.1f})" for task in tasks], 
                       fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax.grid(True)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    plt.title(title, fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_robustness_curve(data: Dict[str, Dict[str, float]],
                             output_path: str,
                             title: str = "Robustness vs Epsilon",
                             xlabel: str = "Epsilon",
                             ylabel: str = "Accuracy",
                             dry_run: bool = False) -> None:
    """Generate line plot for robustness curves."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write("# Robustness curve placeholder (matplotlib not available)\n")
        return
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if dry_run:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "DRY-RUN SCHEMA ARTIFACT\nRobustness Curve", 
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'CLIP': '#1f77b4', 'TeCoA²': '#ff7f0e', 'TeCoA⁴': '#d62728',
              'FARE²': '#2ca02c', 'FARE⁴': '#9467bd'}
    
    for model_name, model_data in data.items():
        epsilons = sorted(model_data.keys(), key=lambda x: float(x.split('/')[0]))
        accuracies = [model_data[eps] for eps in epsilons]
        
        color = colors.get(model_name, '#000000')
        ax.plot(epsilons, accuracies, 'o-', linewidth=2, label=model_name, 
                color=color, markersize=8)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_comparison_barplot(data: Dict[str, Dict[str, float]],
                               output_path: str,
                               title: str = "Model Comparison",
                               ylabel: str = "Metric",
                               dry_run: bool = False) -> None:
    """Generate grouped bar plot for model comparisons."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write("# Comparison barplot placeholder (matplotlib not available)\n")
        return
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if dry_run:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "DRY-RUN SCHEMA ARTIFACT\nComparison Barplot", 
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    categories = list(next(iter(data.values())).keys())
    models = list(data.keys())
    
    x = np.arange(len(categories))
    width = 0.15
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, model in enumerate(models):
        values = [data[model].get(cat, 0.0) for cat in categories]
        offset = (i - len(models) / 2) * width
        ax.bar(x + offset, values, width, label=model, color=colors[i % len(colors)])
    
    ax.set_xlabel('Task', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# Table Generation
# ============================================================================

def write_table_csv(data: Dict[str, Dict[str, Any]],
                   output_path: str,
                   column_order: Optional[List[str]] = None) -> None:
    """Write results table to CSV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if not data:
        # Write empty schema
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['model', 'metric', 'value'])
        return
    
    # Determine columns
    if column_order is None:
        all_keys = set()
        for row_data in data.values():
            all_keys.update(row_data.keys())
        column_order = ['model'] + sorted(all_keys)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=column_order)
        writer.writeheader()
        
        for model_name, metrics in data.items():
            row = {'model': model_name}
            row.update(metrics)
            writer.writerow(row)


def format_table_latex(data: Dict[str, Dict[str, Any]],
                      caption: str,
                      label: str) -> str:
    """Format results table as LaTeX."""
    if not data:
        return f"% Empty table: {caption}\n"
    
    columns = ['Model'] + list(next(iter(data.values())).keys())
    
    latex = "\\begin{table}[htbp]\n"
    latex += "\\centering\n"
    latex += f"\\caption{{{caption}}}\n"
    latex += f"\\label{{{label}}}\n"
    latex += "\\begin{tabular}{" + "l" + "c" * (len(columns) - 1) + "}\n"
    latex += "\\toprule\n"
    latex += " & ".join(columns) + " \\\\\n"
    latex += "\\midrule\n"
    
    for model_name, metrics in data.items():
        row = [model_name]
        for key in columns[1:]:
            val = metrics.get(key, 0.0)
            if isinstance(val, float):
                row.append(f"{val:.2f}")
            else:
                row.append(str(val))
        latex += " & ".join(row) + " \\\\\n"
    
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    
    return latex


# ============================================================================
# Paper-Specific Artifact Writers
# ============================================================================

def write_table_1(results: Dict[str, Any], output_dir: str = "results/tables") -> None:
    """Table 1: Robustness of large vision-language models with different CLIP-models."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Structure: Model -> {Task -> {Clean, Robust}}
    table_data = {}
    
    for model_name, model_results in results.items():
        if not isinstance(model_results, dict):
            continue
        
        row = {}
        for task_name, task_metrics in model_results.items():
            if isinstance(task_metrics, dict):
                row[f"{task_name}_clean"] = task_metrics.get('clean_accuracy', 0.0)
                row[f"{task_name}_robust"] = task_metrics.get('robust_accuracy', 0.0)
        
        if row:
            table_data[model_name] = row
    
    output_path = os.path.join(output_dir, "table_1.csv")
    write_table_csv(table_data, output_path)


def write_table_4(results: Dict[str, Any], output_dir: str = "results/tables") -> None:
    """Table 4: Clean and adversarial evaluation on image classification datasets."""
    os.makedirs(output_dir, exist_ok=True)
    
    table_data = {}
    
    for model_name, model_results in results.items():
        if not isinstance(model_results, dict):
            continue
        
        row = {
            'clean_accuracy': model_results.get('clean_accuracy', 0.0),
            'robust_eps2': model_results.get('robust_accuracy_eps2', 0.0),
            'robust_eps4': model_results.get('robust_accuracy_eps4', 0.0),
        }
        table_data[model_name] = row
    
    output_path = os.path.join(output_dir, "table_4.csv")
    write_table_csv(table_data, output_path)


def write_figure_1(results: Dict[str, Any], output_dir: str = "results/figures",
                  dry_run: bool = False) -> None:
    """Figure 1: Radar plot of zero-shot task performance."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract zero-shot task performance for each model
    radar_data = {}
    
    for model_name, model_results in results.items():
        if not isinstance(model_results, dict):
            continue
        
        task_scores = {}
        for task_name, metrics in model_results.items():
            if isinstance(metrics, dict) and 'accuracy' in metrics:
                task_scores[task_name] = metrics['accuracy']
        
        if task_scores:
            radar_data[model_name] = task_scores
    
    output_path = os.path.join(output_dir, "figure_1.png")
    generate_radar_plot(
        radar_data, 
        output_path,
        title="Figure 1: Zero-Shot Task Performance (CLIP, TeCoA, FARE)",
        dry_run=dry_run
    )


# ============================================================================
# Main Artifact Writer Orchestrator
# ============================================================================

class ArtifactWriter:
    """Main artifact writer for all paper figures and tables."""
    
    def __init__(self, output_dir: str = "results", dry_run: bool = False):
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / "figures"
        self.tables_dir = self.output_dir / "tables"
        self.dry_run = dry_run
        
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
    
    def write_all_artifacts(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Write all paper artifacts and return paths."""
        artifact_paths = {}
        
        # Figures
        artifact_paths['figure_1'] = self._write_figure_1(results)
        artifact_paths['figure_2'] = self._write_figure_2(results)
        artifact_paths['figure_3'] = self._write_figure_3(results)
        artifact_paths['figure_4'] = self._write_figure_4(results)
        artifact_paths['figure_5'] = self._write_figure_5(results)
        artifact_paths['experiment_results'] = self._write_experiment_results(results)
        
        # Tables
        artifact_paths['table_1'] = self._write_table_1(results)
        artifact_paths['table_2'] = self._write_table_2(results)
        artifact_paths['table_3'] = self._write_table_3(results)
        artifact_paths['table_4'] = self._write_table_4(results)
        artifact_paths['table_5'] = self._write_table_5(results)
        artifact_paths['table_6'] = self._write_table_6(results)
        artifact_paths['table_7'] = self._write_table_7(results)
        
        return artifact_paths
    
    def _write_figure_1(self, results: Dict[str, Any]) -> str:
        """Figure 1: Radar plot of zero-shot task performance."""
        output_path = str(self.figures_dir / "figure_1.png")
        write_figure_1(results, str(self.figures_dir), dry_run=self.dry_run)
        return output_path
    
    def _write_figure_2(self, results: Dict[str, Any]) -> str:
        """Figure 2: Robustness curves across epsilon values."""
        output_path = str(self.figures_dir / "figure_2.png")
        
        # Extract epsilon-accuracy data
        curve_data = {}
        for model_name, model_results in results.items():
            if isinstance(model_results, dict):
                eps_data = {
                    '0/255': model_results.get('clean_accuracy', 0.0),
                    '2/255': model_results.get('robust_accuracy_eps2', 0.0),
                    '4/255': model_results.get('robust_accuracy_eps4', 0.0),
                }
                curve_data[model_name] = eps_data
        
        generate_robustness_curve(
            curve_data, output_path,
            title="Figure 2: Robustness vs Perturbation Budget",
            dry_run=self.dry_run
        )
        return output_path
    
    def _write_figure_3(self, results: Dict[str, Any]) -> str:
        """Figure 3: Targeted attack visualization."""
        output_path = str(self.figures_dir / "figure_3.png")
        
        # Generate comparison plot
        comparison_data = {}
        for model_name, model_results in results.items():
            if isinstance(model_results, dict):
                comparison_data[model_name] = {
                    'Benign': model_results.get('clean_accuracy', 0.0),
                    'Attacked': model_results.get('targeted_attack_success', 0.0),
                }
        
        generate_comparison_barplot(
            comparison_data, output_path,
            title="Figure 3: Targeted Attack Success Rate",
            ylabel="Success Rate",
            dry_run=self.dry_run
        )
        return output_path
    
    def _write_figure_4(self, results: Dict[str, Any]) -> str:
        """Figure 4: POPE hallucination examples."""
        output_path = str(self.figures_dir / "figure_4.png")
        
        hallucination_data = {}
        for model_name, model_results in results.items():
            if isinstance(model_results, dict) and 'pope_f1' in model_results:
                hallucination_data[model_name] = {
                    'Random': model_results.get('pope_f1_random', 0.0),
                    'Popular': model_results.get('pope_f1_popular', 0.0),
                    'Adversarial': model_results.get('pope_f1_adversarial', 0.0),
                }
        
        generate_comparison_barplot(
            hallucination_data, output_path,
            title="Figure 4: POPE Hallucination Evaluation (F1 Score)",
            ylabel="F1 Score",
            dry_run=self.dry_run
        )
        return output_path
    
    def _write_figure_5(self, results: Dict[str, Any]) -> str:
        """Figure 5: Additional analysis plots."""
        output_path = str(self.figures_dir / "figure_5.png")
        
        # Generate supplementary comparison
        comparison_data = {}
        for model_name, model_results in results.items():
            if isinstance(model_results, dict):
                comparison_data[model_name] = {
                    'SQA-I': model_results.get('sqa_accuracy', 0.0),
                    'Jailbreak Defense': 1.0 - model_results.get('jailbreak_success_rate', 0.0),
                }
        
        generate_comparison_barplot(
            comparison_data, output_path,
            title="Figure 5: Downstream Task Performance",
            ylabel="Score",
            dry_run=self.dry_run
        )
        return output_path
    
    def _write_experiment_results(self, results: Dict[str, Any]) -> str:
        """Generate summary experiment results figure."""
        output_path = str(self.figures_dir / "experiment_results.png")
        
        # Aggregate all key metrics
        summary_data = {}
        for model_name, model_results in results.items():
            if isinstance(model_results, dict):
                summary_data[model_name] = {
                    'Clean': model_results.get('clean_accuracy', 0.0),
                    'Robust (ε=2/255)': model_results.get('robust_accuracy_eps2', 0.0),
                    'Robust (ε=4/255)': model_results.get('robust_accuracy_eps4', 0.0),
                    'POPE F1': model_results.get('pope_f1', 0.0),
                }
        
        generate_comparison_barplot(
            summary_data, output_path,
            title="Experiment Results Summary",
            ylabel="Score",
            dry_run=self.dry_run
        )
        return output_path
    
    def _write_table_1(self, results: Dict[str, Any]) -> str:
        """Table 1: LVLM robustness."""
        output_path = str(self.tables_dir / "table_1.csv")
        write_table_1(results, str(self.tables_dir))
        return output_path
    
    def _write_table_2(self, results: Dict[str, Any]) -> str:
        """Table 2: Transfer attack results."""
        output_path = str(self.tables_dir / "table_2.csv")
        
        table_data = {}
        for model_name, model_results in results.items():
            if isinstance(model_results, dict):
                table_data[model_name] = {
                    'transfer_attack_success': model_results.get('transfer_attack_success', 0.0),
                }
        
        write_table_csv(table_data, output_path)
        return output_path
    
    def _write_table_3(self, results: Dict[str, Any]) -> str:
        """Table 3: Targeted attack quantitative analysis."""
        output_path = str(self.tables_dir / "table_3.csv")
        
        table_data = {}
        for model_name, model_results in results.items():
            if isinstance(model_results, dict):
                table_data[model_name] = {
                    'targeted_success_rate': model_results.get('targeted_attack_success', 0.0),
                }
        
        write_table_csv(table_data, output_path)
        return output_path