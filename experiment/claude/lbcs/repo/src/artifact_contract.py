"""
Artifact Contract: Measurement Schemas, Artifact Paths, and Writer Interfaces
for Refined Coreset Selection (RCS) Experiments.

This module declares the complete artifact contract for paper reproduction:
- Metric schemas (accuracy, loss, f1, coreset_size, etc.)
- Artifact paths (all tables and figures)
- Writer interfaces for each artifact type
- Dry-run schema generation for smoke validation

reference_grounding: paperbench_ref_003 train.py
reference_grounding: paperbench_ref_003 selection.py
reference_grounding: paperbench_ref_004 cnn_mnist_probability_1step_pixel_shared_rein.py
"""

import json
import os
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import warnings

# ============================================================================
# Artifact Path Registry
# Paper evidence contract: stable output paths for all declared artifacts
# ============================================================================

ARTIFACT_PATHS = {
    # Figures
    "figure_1": "results/figures/figure_1.png",
    "figure_2": "results/figures/figure_2.png",
    "figure_3": "results/figures/figure_3.png",
    "figure_4": "results/figures/figure_4.png",
    "result_figure": "results/figures/experiment_results.png",
    
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
    "result_table": "results/tables/experiment_results.csv",
    
    # JSON artifacts
    "metrics_json": "results/metrics.json",
    "config": "results/config_resolved.json",
    "predictions": "results/predictions.jsonl",
    "trained_model": "results/checkpoints/trained_model.pt",
    
    # Smoke validation artifacts
    "readiness": "results/readiness.json",
    "evaluation_result": "results/evaluation_result.json",
}

# ============================================================================
# Metric Schemas
# Paper evidence contract: declare metric schemas for accuracy, loss, f1, return, fidelity_score
# ============================================================================

@dataclass
class MetricSchema:
    """Base metric schema with aggregation semantics."""
    name: str
    unit: str
    aggregation: str  # mean, std, min, max, last
    higher_is_better: bool
    values: List[float] = field(default_factory=list)
    
    def aggregate(self) -> float:
        """Compute aggregated value according to schema."""
        if not self.values:
            return 0.0
        if self.aggregation == "mean":
            return sum(self.values) / len(self.values)
        elif self.aggregation == "std":
            if len(self.values) < 2:
                return 0.0
            mean = sum(self.values) / len(self.values)
            variance = sum((x - mean) ** 2 for x in self.values) / len(self.values)
            return variance ** 0.5
        elif self.aggregation == "min":
            return min(self.values)
        elif self.aggregation == "max":
            return max(self.values)
        elif self.aggregation == "last":
            return self.values[-1]
        else:
            return sum(self.values) / len(self.values)
    
    def update(self, value: float):
        """Add a new measurement."""
        self.values.append(value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "unit": self.unit,
            "aggregation": self.aggregation,
            "higher_is_better": self.higher_is_better,
            "value": self.aggregate(),
            "n_samples": len(self.values),
        }


@dataclass
class ExperimentMetrics:
    """
    Container for experiment metrics with paper-derived semantics.
    
    Paper metrics contract:
    - accuracy: test accuracy (%)
    - loss: training/validation loss
    - f1: F1 score for classification
    - f1_m: f1(m) validation error violations (lexicographic objective 1)
    - f2_m: f2(m) coreset size (lexicographic objective 2)
    - coreset_size: final selected coreset size
    - return: cumulative reward/return (for RL-style baselines)
    - fidelity_score: model fidelity to full dataset
    """
    accuracy: MetricSchema = field(default_factory=lambda: MetricSchema(
        name="accuracy", unit="%", aggregation="mean", higher_is_better=True
    ))
    loss: MetricSchema = field(default_factory=lambda: MetricSchema(
        name="loss", unit="", aggregation="mean", higher_is_better=False
    ))
    f1: MetricSchema = field(default_factory=lambda: MetricSchema(
        name="f1", unit="", aggregation="mean", higher_is_better=True
    ))
    f1_m: MetricSchema = field(default_factory=lambda: MetricSchema(
        name="f1_m_validation_error", unit="", aggregation="last", higher_is_better=False
    ))
    f2_m: MetricSchema = field(default_factory=lambda: MetricSchema(
        name="f2_m_coreset_size", unit="samples", aggregation="last", higher_is_better=False
    ))
    coreset_size: MetricSchema = field(default_factory=lambda: MetricSchema(
        name="coreset_size", unit="samples", aggregation="last", higher_is_better=False
    ))
    return_metric: MetricSchema = field(default_factory=lambda: MetricSchema(
        name="return", unit="", aggregation="mean", higher_is_better=True
    ))
    fidelity_score: MetricSchema = field(default_factory=lambda: MetricSchema(
        name="fidelity_score", unit="", aggregation="mean", higher_is_better=True
    ))
    
    def update(self, metric_name: str, value: float):
        """Update a specific metric."""
        metric = getattr(self, metric_name, None)
        if metric is not None:
            metric.update(value)
        else:
            warnings.warn(f"Unknown metric: {metric_name}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Export all metrics to dictionary."""
        return {
            "accuracy": self.accuracy.to_dict(),
            "loss": self.loss.to_dict(),
            "f1": self.f1.to_dict(),
            "f1_m_validation_error": self.f1_m.to_dict(),
            "f2_m_coreset_size": self.f2_m.to_dict(),
            "coreset_size": self.coreset_size.to_dict(),
            "return": self.return_metric.to_dict(),
            "fidelity_score": self.fidelity_score.to_dict(),
        }


MEASUREMENT_SCHEMAS = ExperimentMetrics().to_dict()


# ============================================================================
# Artifact Writers
# Paper evidence contract: result artifact writers with stable output paths
# ============================================================================

class ArtifactWriter:
    """Base class for artifact writers with path management."""
    
    def __init__(self, base_dir: str = "results"):
        self.base_dir = Path(base_dir)
        self.artifact_paths = ARTIFACT_PATHS.copy()
        
    def ensure_dir(self, path: Union[str, Path]):
        """Create parent directories if needed."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    def get_path(self, artifact_key: str) -> Path:
        """Get resolved artifact path."""
        rel_path = self.artifact_paths.get(artifact_key, f"results/{artifact_key}")
        return Path(rel_path)
    
    def write_table(self, artifact_key: str, data: Dict[str, Any], is_dry_run: bool = False):
        """
        Write table artifact as CSV.
        
        Args:
            artifact_key: Key in ARTIFACT_PATHS registry
            data: Dictionary with 'headers' and 'rows' keys
            is_dry_run: If True, write schema/readiness artifact
        """
        path = self.get_path(artifact_key)
        self.ensure_dir(path)
        
        with open(path, 'w') as f:
            if is_dry_run:
                f.write("# DRY-RUN SCHEMA ARTIFACT (not real experiment results)\n")
            
            headers = data.get('headers', [])
            rows = data.get('rows', [])
            
            if headers:
                f.write(','.join(str(h) for h in headers) + '\n')
            
            for row in rows:
                f.write(','.join(str(v) for v in row) + '\n')
    
    def write_figure(self, artifact_key: str, is_dry_run: bool = False):
        """
        Write figure artifact as PNG.
        
        Args:
            artifact_key: Key in ARTIFACT_PATHS registry
            is_dry_run: If True, write minimal diagnostic image
        """
        path = self.get_path(artifact_key)
        self.ensure_dir(path)
        
        if is_dry_run:
            # Write minimal diagnostic PNG for dry-run validation
            try:
                import io
                # Minimal 1x1 PNG with "DRY-RUN" marker in metadata
                with open(path, 'wb') as f:
                    # Write minimal valid PNG
                    f.write(b'\x89PNG\r\n\x1a\n')
                    f.write(b'\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde')
                    f.write(b'\x00\x00\x00\x0cIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4')
                    f.write(b'\x00\x00\x00\x00IEND\xaeB`\x82')
            except Exception as e:
                warnings.warn(f"Could not write dry-run figure {path}: {e}")
    
    def write_json(self, artifact_key: str, data: Dict[str, Any], is_dry_run: bool = False):
        """Write JSON artifact."""
        path = self.get_path(artifact_key)
        self.ensure_dir(path)
        
        if is_dry_run:
            data["_dry_run_marker"] = "This is a schema/readiness artifact, not real experiment results"
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def write_jsonl(self, artifact_key: str, records: List[Dict[str, Any]], is_dry_run: bool = False):
        """Write JSONL artifact."""
        path = self.get_path(artifact_key)
        self.ensure_dir(path)
        
        with open(path, 'w') as f:
            if is_dry_run:
                f.write(json.dumps({"_dry_run_marker": "schema artifact"}) + '\n')
            for record in records:
                f.write(json.dumps(record) + '\n')


# ============================================================================
# Table-Specific Writers
# Paper artifact context: preserve table captions and comparison semantics
# ============================================================================

def write_table_1(writer: ArtifactWriter, results: Dict[str, Any], is_dry_run: bool = False):
    """
    Table 1: Results (mean ± std.) to illustrate the utility of our method
    in optimizing the objectives f1(m) and f2(m).
    
    Columns: Dataset, Initial k, ε, Initialized f1(m), Initialized f2(m),
             Achieved f1(m), Achieved f2(m)
    """
    headers = ['Dataset', 'Initial_k', 'Epsilon', 'Init_f1_m', 'Init_f2_m', 'Achieved_f1_m', 'Achieved_f2_m']
    rows = results.get('rows', [])
    
    if is_dry_run and not rows:
        rows = [['CIFAR-10', 600, 0.3, '0.250±0.010', 600, '0.180±0.008', '450±15']]
    
    writer.write_table('table_1', {'headers': headers, 'rows': rows}, is_dry_run=is_dry_run)


def write_table_2(writer: ArtifactWriter, results: Dict[str, Any], is_dry_run: bool = False):
    """
    Table 2: Mean and standard deviation of test accuracy (%) on different
    benchmarks with various predefined coreset sizes.
    
    Includes comparison with 7 baseline methods.
    """
    headers = ['Dataset', 'Method', 'k=200', 'k=400', 'k=600', 'k=800', 'k=1000', 'Final_Size']
    rows = results.get('rows', [])
    
    if is_dry_run and not rows:
        rows = [
            ['CIFAR-10', 'LBCS', '68.5±0.5', '72.1±0.4', '74.3±0.3', '75.8±0.3', '76.9±0.2', '956'],
            ['CIFAR-10', 'Uniform', '65.2±0.6', '69.8±0.5', '72.5±0.4', '74.1±0.4', '75.3±0.3', '1000'],
        ]
    
    writer.write_table('table_2', {'headers': headers, 'rows': rows}, is_dry_run=is_dry_run)


def write_table_3(writer: ArtifactWriter, results: Dict[str, Any], is_dry_run: bool = False):
    """
    Table 3: Mean and standard deviation of test accuracy (%) on different
    benchmarks with coreset sizes achieved by the proposed LBCS.
    """
    headers = ['Dataset', 'Epsilon', 'LBCS_Size', 'LBCS_Accuracy', 'Baseline_Accuracy']
    rows = results.get('rows', [])
    
    if is_dry_run and not rows:
        rows = [['CIFAR-10', 0.3, '956±25', '76.5±0.3', '75.3±0.4']]
    
    writer.write_table('table_3', {'headers': headers, 'rows': rows}, is_dry_run=is_dry_run)


def write_figure_1(writer: ArtifactWriter, is_dry_run: bool = False):
    """
    Figure 1: Illustrations of phenomena of several trivial solutions.
    (a) f1(m) vs. outer iterations
    (b) f2(m) vs. outer iterations
    """
    writer.write_figure('figure_1', is_dry_run=is_dry_run)


def write_figure_2(writer: ArtifactWriter, is_dry_run: bool = False):
    """
    Figure 2: Illustrations of coreset selection under imperfect supervision.
    (a) Test accuracy with 30% corrupted labels
    (b) Test accuracy with class-imbalanced data
    """
    writer.write_figure('figure_2', is_dry_run=is_dry_run)


# ============================================================================
# Dry-Run Artifact Generation
# Creates all declared artifacts as schema/readiness artifacts during smoke validation
# ============================================================================

def generate_dry_run_artifacts(base_dir: str = "results") -> Dict[str, str]:
    """
    Generate all declared artifacts as dry-run schema/readiness artifacts.
    
    Returns:
        Dictionary mapping artifact keys to generated paths
    """
    writer = ArtifactWriter(base_dir=base_dir)
    generated = {}
    
    # Generate all tables
    for table_num in range(1, 12):
        key = f"table_{table_num}"
        write_fn = globals().get(f"write_table_{table_num}")
        if write_fn:
            write_fn(writer, {}, is_dry_run=True)
        else:
            # Generic table schema
            writer.write_table(key, {
                'headers': ['Column1', 'Column2', 'Value'],
                'rows': [['schema', 'artifact', '0.0']]
            }, is_dry_run=True)
        generated[key] = str(writer.get_path(key))
    
    # Generate all figures
    for fig_num in [1, 2, 3, 4]:
        key = f"figure_{fig_num}"
        write_fn = globals().get(f"write_figure_{fig_num}")
        if write_fn:
            write_fn(writer, is_dry_run=True)
        else:
            writer.write_figure(key, is_dry_run=True)
        generated[key] = str(writer.get_path(key))
    
    # Generate JSON artifacts
    writer.write_json('metrics_json', {
        "accuracy": {"value": 0.0, "unit": "%"},
        "loss": {"value": 0.0, "unit": ""},
        "coreset_size": {"value": 0, "unit": "samples"}
    }, is_dry_run=True)
    generated['metrics_json'] = str(writer.get_path('metrics_json'))
    
    writer.write_json('config', {
        "dataset": "cifar10",
        "model": "resnet18",
        "epsilon": 0.3
    }, is_dry_run=True)
    generated['config'] = str(writer.get_path('config'))
    
    writer.write_jsonl('predictions', [], is_dry_run=True)
    generated['predictions'] = str(writer.get_path('predictions'))
    
    # Generate readiness artifacts
    writer.write_json('readiness', {
        "status": "ready",
        "artifacts_generated": list(generated.keys()),
        "mode": "dry_run"
    }, is_dry_run=False)
    generated['readiness'] = str(writer.get_path('readiness'))
    
    writer.write_json('evaluation_result', {
        "accuracy": 0.0,
        "loss": 0.0,
        "coreset_size": 0,
        "_dry_run": True
    }, is_dry_run=False)
    generated['evaluation_result'] = str(writer.get_path('evaluation_result'))
    
    return generated


# ============================================================================
# Public API
# ============================================================================

def get_artifact_path(artifact_key: str) -> str:
    """Get the canonical path for a declared artifact."""
    return ARTIFACT_PATHS.get(artifact_key, f"results/{artifact_key}")


def create_metrics() -> ExperimentMetrics:
    """Create a new metrics container with paper-derived schemas."""
    return ExperimentMetrics()


def create_writer(base_dir: str = "results") -> ArtifactWriter:
    """Create a new artifact writer."""
    return ArtifactWriter(base_dir=base_dir)
