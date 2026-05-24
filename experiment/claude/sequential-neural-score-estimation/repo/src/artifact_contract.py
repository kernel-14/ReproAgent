"""
Sequential Neural Posterior Score Estimation - Artifact Contract and Evidence Matrix

This module implements the paper-derived evidence obligation matrix, experiment registry,
metric schemas, and artifact writer interfaces for SNPSE benchmark reproduction.

Reference grounding:
- paperbench_ref_001 l5pc/docs/config.md: Configuration structure for multi-round inference
- paperbench_ref_001 sbi/sbi/inference/snpe/snpe_a.py: SNPE method interface patterns
- paperbench_ref_001 sbi/sbi/inference/snle/mnle.py: Likelihood estimation interface
- paperbench_ref_001 sbi/sbi/inference/snle/snle_base.py: Base inference class patterns

Paper: Sequential Neural Score Estimation: Likelihood-Free Inference with
       Conditional Score Based Diffusion Models

Implementation surfaces: artifact_writer, evaluation, baseline_or_ablation, config, tests

Method obligations:
- Preserve table/figure captions, named baselines, comparison semantics for all paper figures
- Declare measurement schemas and aggregation outputs for: loss, c2st, accuracy
- Make result artifact paths statically discoverable with writer/declaration hooks
- Paper evidence contract: declare metric schemas/aggregations for loss, c2st, accuracy
- Paper evidence contract: declare result artifact writers for all figures with stable output paths
- Binding addendum: TSNPE and SNVI results in Section 5.3 taken from respective papers

Writes artifacts:
- results/evidence_contract_matrix.json
- results/experiment_registry.json
- results/metrics.json
- results/dataset_registry.json
- results/artifact_manifest.json
- results/sensitivity_report.json
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Union
import warnings
import numpy as np


# ============================================================================
# Metric Schema Registry
# reference_grounding: paperbench_ref_001 sbi/sbi/inference/snle/mnle.py
# Paper evidence contract: declare metric schemas/aggregations for loss, c2st, accuracy
# ============================================================================

METRIC_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "loss": {
        "name": "Training Loss",
        "type": "scalar",
        "aggregation": "mean",
        "lower_is_better": True,
        "description": "Score matching loss for diffusion model training",
        "paper_section": "Section 4 (Training Objective)",
        "formula": "E[||s_theta(x_t, t) - nabla log p_t(x_t)||^2]",
        "unit": "nats",
        "range": [0.0, float("inf")],
        "decisive": False,
    },
    "c2st": {
        "name": "Classifier Two-Sample Test",
        "type": "scalar",
        "aggregation": "mean",
        "lower_is_better": False,  # 0.5 is ideal (indistinguishable)
        "description": "Binary classification accuracy between reference and approximate posterior samples",
        "paper_section": "Section 5 (Evaluation Metrics)",
        "formula": "Accuracy of classifier distinguishing p(theta|x) from q(theta|x)",
        "unit": "probability",
        "range": [0.0, 1.0],
        "optimal_value": 0.5,
        "decisive": True,
        "paper_note": "Uses sbibm library with default hyperparameters per addendum",
    },
    "accuracy": {
        "name": "Posterior Accuracy",
        "type": "scalar",
        "aggregation": "mean",
        "lower_is_better": False,
        "description": "Generic accuracy metric for posterior approximation quality",
        "paper_section": "Section 5",
        "unit": "probability",
        "range": [0.0, 1.0],
        "decisive": False,
    },
    "mmd": {
        "name": "Maximum Mean Discrepancy",
        "type": "scalar",
        "aggregation": "mean",
        "lower_is_better": True,
        "description": "Kernel-based distance between distributions",
        "paper_section": "Section 5",
        "unit": "distance",
        "range": [0.0, float("inf")],
        "decisive": False,
    },
    "coverage": {
        "name": "Credible Region Coverage",
        "type": "scalar",
        "aggregation": "mean",
        "lower_is_better": False,
        "description": "Fraction of true parameters covered by credible regions",
        "paper_section": "Section 5.3 (Pyloric Experiment)",
        "paper_figure": "Figure 8",
        "unit": "probability",
        "range": [0.0, 1.0],
        "optimal_value": 0.95,  # For 95% credible regions
        "decisive": False,
    },
}


# ============================================================================
# Artifact Path Registry
# Paper evidence contract: declare result artifact writers with stable output paths
# ============================================================================

ARTIFACT_PATHS: Dict[str, str] = {
    "figure_1": "results/figures/figure_1.png",
    "figure_2": "results/figures/figure_2.png",
    "figure_3": "results/figures/figure_3.png",
    "figure_4": "results/figures/figure_4.png",
    "figure_4a": "results/figures/figure_4a.png",
    "figure_4c": "results/figures/figure_4c.png",
    "figure_5": "results/figures/figure_5.png",
    "figure_6": "results/figures/figure_6.png",
    "figure_7": "results/figures/figure_7.png",
    "figure_8": "results/figures/figure_8.png",
    "figure_9": "results/figures/figure_9.png",
    "result_table": "results/tables/experiment_results.csv",
    "result_figure": "results/figures/experiment_results.png",
    "predictions": "results/predictions.jsonl",
    "metrics_json": "results/metrics.json",
    "config": "results/config_resolved.json",
    "log": "results/training.log",
    "checkpoint": "results/checkpoints/model_final.pt",
    "posterior_samples": "results/posterior_samples.npz",
    "benchmark_metrics": "results/benchmark_metrics.csv",
    "evidence_contract_matrix": "results/evidence_contract_matrix.json",
    "experiment_registry": "results/experiment_registry.json",
    "dataset_registry": "results/dataset_registry.json",
    "artifact_manifest": "results/artifact_manifest.json",
    "sensitivity_report": "results/sensitivity_report.json",
    "readiness": "results/readiness.json",
    "evaluation_result": "results/evaluation_result.json",
}


# ============================================================================
# Paper Evidence Obligation Matrix
# reference_grounding: paperbench_ref_001 l5pc/docs/config.md
# Method obligation: Implement code/config-visible paper evidence obligation matrix
# Each row binds paper experiment to datasets, methods, parameters, trends, artifacts
# ============================================================================

EVIDENCE_CONTRACT_MATRIX: List[Dict[str, Any]] = [
    {
        "experiment_id": "exp_two_moons_visualization",
        "paper_section": "Section 5.1",
        "paper_figure": "Figure 1",
        "name": "Two Moons Posterior Visualization",
        "description": "Visualisation of posterior inference using NPSE in Two Moons experiment. Forward process transforms samples from target posterior to tractable reference distribution. Backward process transports samples from reference to approximate posterior.",
        "datasets": ["two_moons"],
        "tasks": ["two_moons"],
        "methods": ["NPSE"],
        "baselines": [],
        "parameters": {
            "simulation_budget": 1000,
            "num_samples": 1000,
        },
        "expected_trend": "Visual verification that reverse diffusion process produces samples matching true posterior",
        "decision_claim": "NPSE can accurately approximate bimodal posteriors",
        "metrics": ["visual_inspection"],
        "artifacts": ["figure_1"],
        "decisive": True,
        "replicate": True,
    },
    {
        "experiment_id": "exp_benchmark_non_sequential",
        "paper_section": "Section 5.2",
        "paper_figure": "Figure 2",
        "name": "Eight Benchmark Tasks (Non-Sequential Methods)",
        "description": "Results on eight benchmark tasks comparing non-sequential methods: NPSE, NPE, NLE, NRE",
        "datasets": ["slcp", "gaussian_linear", "gaussian_mixture", "two_moons", "uniform", "bernoulli_glm", "lotka_volterra", "sir"],
        "tasks": ["slcp", "gaussian_linear", "gaussian_mixture", "two_moons", "uniform", "bernoulli_glm", "lotka_volterra", "sir"],
        "methods": ["NPSE", "NPE", "NLE", "NRE"],
        "baselines": ["NPE", "NLE", "NRE"],
        "parameters": {
            "simulation_budgets": [1000, 10000, 100000],
            "num_rounds": 1,
        },
        "expected_trend": "NPSE competitive or superior to NPE/NLE/NRE across simulation budgets",
        "decision_claim": "NPSE is a viable non-sequential method for SBI",
        "metrics": ["c2st"],
        "artifacts": ["figure_2", "result_table"],
        "decisive": True,
        "replicate": True,
    },
    {
        "experiment_id": "exp_benchmark_sequential",
        "paper_section": "Section 5.2",
        "paper_figure": "Figure 3",
        "name": "Eight Benchmark Tasks (Sequential Methods)",
        "description": "Results on eight benchmark tasks comparing sequential methods: TSNPSE, SNPE-A, SNPE-C, TSNPE (external), SNVI (external)",
        "datasets": ["slcp", "gaussian_linear", "gaussian_mixture", "two_moons", "uniform", "bernoulli_glm", "lotka_volterra", "sir"],
        "tasks": ["slcp", "gaussian_linear", "gaussian_mixture", "two_moons", "uniform", "bernoulli_glm", "lotka_volterra", "sir"],
        "methods": ["TSNPSE", "SNPE-A", "SNPE-C"],
        "baselines": ["SNPE-A", "SNPE-C", "TSNPE", "SNVI"],
        "parameters": {
            "simulation_budgets": [1000, 10000, 100000],
            "num_rounds": 10,
        },
        "expected_trend": "TSNPSE competitive with or superior to SNPE-A/C, comparable to TSNPE",
        "decision_claim": "Sequential NPSE variants improve sample efficiency",
        "metrics": ["c2st"],
        "artifacts": ["figure_3", "result_table"],
        "decisive": True,
        "replicate": True,
        "external_baseline_note": "TSNPE and SNVI results taken from respective papers (addendum)",
    },
    {
        "experiment_id": "exp_pyloric",
        "paper_section": "Section 5.3",
        "paper_figure": "Figure 4",
        "name": "Pyloric Neuron Experiment",
        "description": "Results for the Pyloric experiment comparing TSNPSE, SNPE-A, SNPE-C, TSNPE (external), SNVI (external)",
        "datasets": ["pyloric"],
        "tasks": ["pyloric"],
        "methods": ["TSNPSE", "SNPE-A", "SNPE-C"],
        "baselines": ["SNPE-A", "SNPE-C", "TSNPE", "SNVI"],
        "parameters": {
            "simulation_budget": 119600,
            "num_rounds": 10,
        },
        "expected_trend": "TSNPSE achieves comparable performance to TSNPE on realistic neuroscience task",
        "decision_claim": "TSNPSE scales to high-dimensional realistic inference problems",
        "metrics": ["c2st", "coverage"],
        "artifacts": ["figure_4", "figure_4a", "figure_4c"],
        "decisive": True,
        "replicate": True,
        "external_baseline_note": "TSNPE and SNVI results taken from respective papers (addendum)",
    },
    {
        "experiment_id": "exp_npse_vs_nlse",
        "paper_section": "Section E.2 (Appendix)",
        "paper_figure": "Figure 5",
        "name": "NPSE vs NLSE Comparison",
        "description": "Comparison between NPSE and NLSE on four benchmark tasks",
        "datasets": ["slcp", "gaussian_linear", "two_moons", "lotka_volterra"],
        "tasks": ["slcp", "gaussian_linear", "two_moons", "lotka_volterra"],
        "methods": ["NPSE", "NLSE"],
        "baselines": ["NLSE"],
        "parameters": {
            "simulation_budgets": [1000, 10000, 100000],
            "num_rounds": 1,
        },
        "expected_trend": "NPSE shows competitive performance to neural likelihood score estimation",
        "decision_claim": "Score estimation on posterior is effective alternative to likelihood score estimation",
        "metrics": ["c2st"],
        "artifacts": ["figure_5"],
        "decisive": False,
        "replicate": True,
    },
    {
        "experiment_id": "exp_sequential_variants",
        "paper_section": "Section E.3 (Appendix)",
        "paper_figure": "Figure 6",
        "name": "SNPSE Variant Comparison",
        "description": "Comparison between TSNPSE, SNPSE-A, and SNPSE-B on SLCP and GLU tasks. SNPSE-C omitted as it failed to provide meaningful results (C2ST ≈ 1).",
        "datasets": ["slcp", "glu"],
        "tasks": ["slcp", "glu"],
        "methods": ["TSNPSE", "SNPSE-A", "SNPSE-B"],
        "baselines": [],
        "parameters": {
            "simulation_budgets": [1000, 10000, 100000],
            "num_rounds": 10,
        },
        "expected_trend": "TSNPSE outperforms SNPSE-A and SNPSE-B across simulation budgets",
        "decision_claim": "Truncated sequential approach (Algorithm 1) is superior to alternative sequential strategies",
        "metrics": ["c2st"],
        "artifacts": ["figure_6"],
        "decisive": True,
        "replicate": True,
        "ablation_note": "SNPSE-C results omitted due to failure (C2ST ≈ 1)",
    },
    {
        "experiment_id": "exp_pyloric_marginals",
        "paper_section": "Section 5.3",
        "paper_figure": "Figure 7",
        "name": "Pyloric Pairwise Marginal Plot",
        "description": "Pairwise marginal plot for the posterior approximation obtained in the Pyloric experiment. The posterior mean is plotted in red.",
        "datasets": ["pyloric"],
        "tasks": ["pyloric"],
        "methods": ["TSNPSE"],
        "baselines": [],
        "parameters": {
            "simulation_budget": 119600,
            "num_rounds": 10,
        },
        "expected_trend": "Posterior marginals show reasonable correlation structure and concentration",
        "decision_claim": "TSNPSE produces interpretable posterior approximations for neuroscience applications",
        "metrics": ["visual_inspection"],
        "artifacts": ["figure_7"],
        "decisive": False,
        "replicate": True,
    },
    {
        "experiment_id": "exp_pyloric_coverage",
        "paper_section": "Section 5.3",
        "paper_figure": "Figure 8",
        "name": "Pyloric Coverage Plot",
        "description": "Coverage plot for the Pyloric experiment showing calibration of credible regions",
        "datasets": ["pyloric"],
        "tasks": ["pyloric"],
        "methods": ["TSNPSE"],
        "baselines": [],
        "parameters": {
            "simulation_budget": 119600,
            "num_rounds": 10,
        },
        "expected_trend": "Coverage matches nominal credible levels (well-calibrated)",
        "decision_claim": "TSNPSE produces well-calibrated uncertainty estimates",
        "metrics": ["coverage"],
        "artifacts": ["figure_8"],
        "decisive": False,
        "replicate": True,
    },
    {
        "experiment_id": "exp_npse_vs_fmpe",
        "paper_section": "Section E.4 (Appendix)",
        "paper_figure": "Figure 9",
        "name": "NPSE vs FMPE Comparison",
        "description": "Comparison between NPSE and flow matching posterior estimation (FMPE) on eight SBI benchmark experiments",
        "datasets": ["slcp", "gaussian_linear", "gaussian_mixture", "two_moons", "uniform", "bernoulli_glm", "lotka_volterra", "sir"],
        "tasks": ["slcp", "gaussian_linear", "gaussian_mixture", "two_moons", "uniform", "bernoulli_glm", "lotka_volterra", "sir"],
        "methods": ["NPSE", "FMPE"],
        "baselines": ["FMPE"],
        "parameters": {
            "simulation_budgets": [1000, 10000, 100000],
            "num_rounds": 1,
        },
        "expected_trend": "NPSE shows competitive or superior performance to flow matching methods",
        "decision_claim": "Score-based diffusion is competitive with flow matching for posterior estimation",
        "metrics": ["c2st"],
        "artifacts": ["figure_9"],
        "decisive": False,
        "replicate": True,
    },
]


# ============================================================================
# Experiment Registry
# reference_grounding: paperbench_ref_001 sbi/sbi/inference/snpe/snpe_a.py
# Paper evidence contract: explicit named experiment/result-protocol anchors
# ============================================================================

EXPERIMENT_REGISTRY: Dict[str, Dict[str, Any]] = {
    exp["experiment_id"]: {
        "id": exp["experiment_id"],
        "name": exp["name"],
        "paper_section": exp["paper_section"],
        "paper_figure": exp.get("paper_figure"),
        "description": exp["description"],
        "datasets": exp["datasets"],
        "methods": exp["methods"],
        "baselines": exp["baselines"],
        "parameters": exp["parameters"],
        "metrics": exp["metrics"],
        "artifacts": exp["artifacts"],
        "decisive": exp["decisive"],
        "replicate": exp["replicate"],
        "expected_trend": exp["expected_trend"],
        "decision_claim": exp["decision_claim"],
    }
    for exp in EVIDENCE_CONTRACT_MATRIX
}


# ============================================================================
# Artifact Writer Interface
# Implementation surface: artifact_writer
# reference_grounding: paperbench_ref_001 sbi/sbi/inference/snle/snle_base.py
# ============================================================================

class ArtifactWriter:
    """
    Artifact writer interface for SNPSE paper reproduction.
    
    Provides methods to write metrics, figures, tables, and manifests
    for both dry-run/smoke validation and full experiment execution.
    
    During smoke validation, writes schema/readiness artifacts labeled as dry-run.
    During full execution, writes actual experiment results.
    """
    
    def __init__(self, base_dir: str = "results", mode: str = "runtime_smoke"):
        """
        Initialize artifact writer.
        
        Args:
            base_dir: Base directory for all output artifacts
            mode: Execution mode (runtime_smoke, docker_validate, train, evaluate, full)
        """
        self.base_dir = Path(base_dir)
        self.mode = mode
        self.is_dry_run = mode in ["runtime_smoke", "docker_validate"]
        
        # Create all output directories
        self._create_directories()
    
    def _create_directories(self):
        """Create all required output directories."""
        directories = [
            self.base_dir,
            self.base_dir / "figures",
            self.base_dir / "tables",
            self.base_dir / "checkpoints",
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def write_metrics(self, metrics: Dict[str, Any], path: Optional[str] = None) -> str:
        """
        Write metrics JSON artifact.
        
        Args:
            metrics: Dictionary of metric values
            path: Optional custom path (defaults to ARTIFACT_PATHS["metrics_json"])
        
        Returns:
            Path to written artifact
        """
        if path is None:
            path = ARTIFACT_PATHS["metrics_json"]
        
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.is_dry_run:
            # Dry-run: write schema with explicit dry-run label
            schema_metrics = {
                "_dry_run_artifact": True,
                "_mode": self.mode,
                "_description": "Metrics schema for SNPSE paper reproduction",
                "metrics": {
                    metric_id: {
                        "value": 0.0,
                        "schema": METRIC_SCHEMAS.get(metric_id, {}),
                        "status": "bounded_smoke_schema_only",
                    }
                    for metric_id in METRIC_SCHEMAS.keys()
                },
            }
            with open(output_path, "w") as f:
                json.dump(schema_metrics, f, indent=2)
        else:
            # Full execution: write actual metrics
            with open(output_path, "w") as f:
                json.dump(metrics, f, indent=2)
        
        return str(output_path)
    
    def write_figure(self, figure_id: str, data: Optional[Any] = None) -> str:
        """
        Write figure artifact.
        
        Args:
            figure_id: Figure identifier (e.g., "figure_1")
            data: Optional figure data (for full execution)
        
        Returns:
            Path to written artifact
        """
        path = ARTIFACT_PATHS.get(figure_id)
        if path is None:
            raise ValueError(f"Unknown figure_id: {figure_id}")
        
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.is_dry_run:
            # Dry-run: create minimal bounded smoke image with dry-run label
            try:
                import matplotlib
                matplotlib.use('Agg')  # Non-interactive backend
                import matplotlib.pyplot as plt
            except ImportError:
                # If matplotlib not available, write JSON manifest instead
                with open(output_path.with_suffix('.json'), "w") as f:
                    json.dump({
                        "_dry_run_artifact": True,
                        "_mode": self.mode,
                        "figure_id": figure_id,
                        "path": str(output_path),
                        "status": "schema_only",
                    }, f, indent=2)
                return str(output_path.with_suffix('.json'))
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f"DRY RUN ARTIFACT\n{figure_id}\nMode: {self.mode}",
                   ha='center', va='center', fontsize=14, color='red')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
        else:
            # Full execution: write actual figure
            if data is not None:
                # Assume data is a matplotlib figure object
                data.savefig(output_path, dpi=300, bbox_inches='tight')
                import matplotlib.pyplot as plt
                plt.close(data)
        
        return str(output_path)
    
    def write_table(self, table_id: str, data: Optional[Any] = None) -> str:
        """
        Write table artifact (CSV format).
        
        Args:
            table_id: Table identifier
            data: Optional table data (pandas DataFrame or dict)
        
        Returns:
            Path to written artifact
        """
        path = ARTIFACT_PATHS.get(table_id)
        if path is None:
            raise ValueError(f"Unknown table_id: {table_id}")
        
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.is_dry_run:
            # Dry-run: write CSV header schema
            with open(output_path, "w") as f:
                f.write("# DRY RUN ARTIFACT\n")
                f.write(f"# Mode: {self.mode}\n")
                f.write("experiment,method,task,simulation_budget,metric,value\n")
                f.write("# Bounded smoke schema for experiment results\n")
        else:
            # Full execution: write actual table
            try:
                import pandas as pd
                if isinstance(data, pd.DataFrame):
                    data.to_csv(output_path, index=False)
                elif isinstance(data, dict):
                    pd.DataFrame(data).to_csv(output_path, index=False)
                else:
                    raise ValueError(f"Unsupported table data type: {type(data)}")
            except ImportError:
                # Pandas not available, write as JSON
                with open(output_path.with_suffix('.json'), "w") as f:
                    json.dump(data if isinstance(data, dict) else {"data": str(data)}, f, indent=2)
        
        return str(output_path)
    
    def write_evidence_contract_matrix(self, path: Optional[str] = None) -> str:
        """Write evidence contract matrix artifact."""
        if path is None:
            path = ARTIFACT_PATHS["evidence_contract_matrix"]
        
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump({
                "_description": "Paper-derived evidence obligation matrix for SNPSE reproduction",
                "_source": "paper.md Sections 5, E; addendum.md",
                "matrix": EVIDENCE_CONTRACT_MATRIX,
            }, f, indent=2)
        
        return str(output_path)
    
    def write_experiment_registry(self, path: Optional[str] = None) -> str:
        """Write experiment registry artifact."""
        if path is None:
            path = ARTIFACT_PATHS["experiment_registry"]
        
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump({
                "_description": "Experiment registry for SNPSE paper reproduction",
                "_source": "Derived from evidence_contract_matrix",
                "experiments": EXPERIMENT_REGISTRY,
            }, f, indent=2)
        
        return str(output_path)
    
    def write_artifact_manifest(self, path: Optional[str] = None) -> str:
        """Write artifact manifest declaring all output paths."""
        if path is None:
            path = ARTIFACT_PATHS["artifact_manifest"]
        
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        manifest = {
            "_description": "Artifact manifest for SNPSE paper reproduction",
            "_mode": self.mode,
            "_is_dry_run": self.is_dry_run,
            "artifact_paths": ARTIFACT_PATHS,
            "metric_schemas": METRIC_SCHEMAS,
            "experiments": list(EXPERIMENT_REGISTRY.keys()),
        }
        
        with open(output_path, "w") as f:
            json.dump(manifest, f, indent=2)
        
        return str(output_path)
    
    def write_sensitivity_report(self, path: Optional[str] = None) -> str:
        """Write sensitivity analysis report."""
        if path is None:
            path = ARTIFACT_PATHS["sensitivity_report"]
        
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.is_dry_run:
            report = {
                "_dry_run_artifact": True,
                "_mode": self.mode,
                "_description": "Sensitivity analysis report schema",
                "sensitivity_analyses": {
                    "learning_rate": {
                        "parameter": "learning_rate",
                        "values_tested": None,
                        "metric": "c2st",
                        "results": None,
                        "status": "schema_only",
                    },
                    "num_diffusion_steps": {
                        "parameter": "num_diffusion_steps",
                        "values_tested": None,
                        "metric": "c2st",
                        "results": None,
                        "status": "schema_only",
                    },
                },
            }
        else:
            report = {
                "_description": "Sensitivity analysis report for SNPSE",
                "sensitivity_analyses": {},
            }
        
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        
        return str(output_path)
    
    def write_readiness_manifest(self, path: Optional[str] = None) -> str:
        """Write readiness manifest for smoke validation."""